## [0] DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics

**Authors**: *Zhiao Huang, Feng Chen, Yewen Pu, Chunru Lin, Hao Su, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html)

**Abstract**:

Combining gradient-based trajectory optimization with differentiable physics simulation is an efficient technique for solving soft-body manipulation problems.Using a well-crafted optimization objective, the solver can quickly converge onto a valid trajectory.However, writing the appropriate objective functions requires expert knowledge, making it difficult to collect a large set of naturalistic problems from non-expert users.We introduce DiffVL, a method that enables non-expert users to communicate soft-body manipulation tasks -- a combination of vision and natural language, given in multiple stages -- that can be readily leveraged by a differential physics solver. We have developed GUI tools that enable non-expert users to specify 100 tasks inspired by real-life soft-body manipulations from online videos, which we'll make public.We leverage large language models to translate task descriptions into machine-interpretable optimization objectives. The optimization objectives can help differentiable physics solvers to solve these long-horizon multistage tasks that are challenging for previous baselines.

----

## [0] Label-efficient Segmentation via Affinity Propagation

**Authors**: *Wentong Li, Yuqian Yuan, Song Wang, Wenyu Liu, Dongqi Tang, Jian Liu, Jianke Zhu, Lei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f6fae52f3b62c3334e288e3bc58230d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f6fae52f3b62c3334e288e3bc58230d-Abstract-Conference.html)

**Abstract**:

Weakly-supervised segmentation with label-efficient sparse annotations has attracted increasing research attention to reduce the cost of laborious pixel-wise labeling process, while the pairwise affinity modeling techniques play an essential role in this task. Most of the existing approaches focus on using the local appearance kernel to model the neighboring pairwise potentials. However, such a local operation fails to capture the long-range dependencies and ignores the  topology of objects. In this work, we formulate the affinity modeling as an affinity propagation process, and propose a local and a global pairwise affinity terms to generate accurate soft pseudo labels. An efficient algorithm is also developed to reduce significantly the computational cost. The proposed approach can be conveniently plugged into existing segmentation networks. Experiments on three typical label-efficient segmentation tasks, i.e. box-supervised instance segmentation, point/scribble-supervised semantic segmentation and CLIP-guided semantic segmentation, demonstrate the superior performance of the proposed approach.

----

## [0] Segment Anything in High Quality

**Authors**: *Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html)

**Abstract**:

The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures.  We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced detaset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 10 diverse segmentation datasets across different downstream tasks, where 8 out of them are evaluated in a zero-shot transfer protocol. Our code and pretrained models are at https://github.com/SysCV/SAM-HQ.

----

## [0] Variational Inference with Gaussian Score Matching

**Authors**: *Chirag Modi, Robert M. Gower, Charles Margossian, Yuling Yao, David M. Blei, Lawrence K. Saul*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f9453c4848b89d4d8c5d6041f5fb9ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f9453c4848b89d4d8c5d6041f5fb9ec-Abstract-Conference.html)

**Abstract**:

Variational inference (VI) is a method to approximate the computationally intractable posterior distributions that arise in  Bayesian statistics. Typically, VI fits a simple parametric distribution to be close to the target posterior, optimizing an appropriate objective such as the evidence lower bound (ELBO).   In this work, we present a new approach to VI. Our method is based on the principle of score matching---namely, that if two distributions are equal then their score functions (i.e., gradients of the log density) are equal at every point on their support. With this principle, we develop score-matching VI, an iterative algorithm that seeks to match the scores between the variational approximation and the exact posterior. At each iteration, score-matching VI solves an inner optimization, one that minimally adjusts the current variational estimate to match the scores at a newly sampled value of the latent variables. We show that when the variational family is a  Gaussian, this inner optimization enjoys a closed-form solution, which we call Gaussian score matching VI (GSM-VI). GSM-VI is a ``black box'' variational algorithm in that it only requires a differentiable joint distribution, and as such it can be applied to a wide class of models. We compare GSM-VI to black box variational inference (BBVI), which has similar requirements but instead optimizes the ELBO. We first study how GSM-VI behaves as a function of the problem dimensionality, the condition number of the target covariance matrix (when the target is Gaussian), and the degree of mismatch between the approximating and exact posterior distribution. We then study GSM-VI on a collection of real-world Bayesian inference problems from the posteriorDB database of datasets and models. We find that GSM-VI is faster than BBVI and equally or more accurate. Specifically, over a wide range of target posteriors, GSM-VI requires 10-100x fewer gradient evaluations than BBVI to obtain a comparable quality of approximation.

----

## [0] Feature Adaptation for Sparse Linear Regression

**Authors**: *Jonathan A. Kelner, Frederic Koehler, Raghu Meka, Dhruv Rohatgi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f999632c48f87cffb214e575581e4a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f999632c48f87cffb214e575581e4a9-Abstract-Conference.html)

**Abstract**:

Sparse linear regression is a central problem in high-dimensional statistics. We study the correlated random design setting, where the covariates are drawn from a multivariate Gaussian $N(0,\Sigma)$, and we seek an estimator with small excess risk. If the true signal is $t$-sparse, information-theoretically, it is possible to achieve strong recovery guarantees with only $O(t\log n)$ samples. However, computationally efficient algorithms have sample complexity linear in (some variant of) the *condition number* of $\Sigma$. Classical algorithms such as the Lasso can require significantly more samples than necessary even if there is only a single sparse approximate dependency among the covariates.We provide a polynomial-time algorithm that, given $\Sigma$, automatically adapts the Lasso to tolerate a small number of approximate dependencies. In particular, we achieve near-optimal sample complexity for constant sparsity and if $\Sigma$ has few ``outlier'' eigenvalues.Our algorithm fits into a broader framework of *feature adaptation* for sparse linear regression with ill-conditioned covariates. With this framework, we additionally provide the first polynomial-factor improvement over brute-force search for constant sparsity $t$ and arbitrary covariance $\Sigma$.

----

## [0] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling

**Authors**: *Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html)

**Abstract**:

Time series analysis is widely used in extensive areas. Recently, to reduce labeling expenses and benefit various tasks, self-supervised pre-training has attracted immense interest. One mainstream paradigm is masked modeling, which successfully pre-trains deep models by learning to reconstruct the masked content based on the unmasked part. However, since the semantic information of time series is mainly contained in temporal variations, the standard way of randomly masking a portion of time points will seriously ruin vital temporal variations of time series, making the reconstruction task too difficult to guide representation learning. We thus present SimMTM, a Simple pre-training framework for Masked Time-series Modeling. By relating masked modeling to manifold learning, SimMTM proposes to recover masked time points by the weighted aggregation of multiple neighbors outside the manifold, which eases the reconstruction task by assembling ruined but complementary temporal variations from multiple masked series. SimMTM further learns to uncover the local structure of the manifold, which is helpful for masked modeling. Experimentally, SimMTM achieves state-of-the-art fine-tuning performance compared to the most advanced time series pre-training methods in two canonical time series analysis tasks: forecasting and classification, covering both in- and cross-domain settings.

----

## [0] CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs

**Authors**: *Guangyao Zhai, Evin Pinar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fba70900a84a8fb755c48ba99420c95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fba70900a84a8fb755c48ba99420c95-Abstract-Conference.html)

**Abstract**:

Controllable scene synthesis aims to create interactive environments for numerous industrial use cases. Scene graphs provide a highly suitable interface to facilitate these applications by abstracting the scene context in a compact manner. Existing methods, reliant on retrieval from extensive databases or pre-trained shape embeddings, often overlook scene-object and object-object relationships, leading to inconsistent results due to their limited generation capacity. To address this issue, we present CommonScenes, a fully generative model that converts scene graphs into corresponding controllable 3D scenes, which are semantically realistic and conform to commonsense. Our pipeline consists of two branches, one predicting the overall scene layout via a variational auto-encoder and the other generating compatible shapes via latent diffusion, capturing global scene-object and local inter-object relationships in the scene graph while preserving shape diversity. The generated scenes can be manipulated by editing the input scene graph and sampling the noise in the diffusion model. Due to the lack of a scene graph dataset offering high-quality object-level meshes with relations, we also construct SG-FRONT, enriching the off-the-shelf indoor dataset 3D-FRONT with additional scene graph labels. Extensive experiments are conducted on SG-FRONT, where CommonScenes shows clear advantages over other methods regarding generation consistency, quality, and diversity. Codes and the dataset are available on the website.

----

## [0] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback

**Authors**: *Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fc47800ee5b30b8777fdd30abcaaf3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fc47800ee5b30b8777fdd30abcaaf3b-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) such as ChatGPT have seen widespread adoption due to their ability to follow user instructions well.Developing these LLMs involves a complex yet poorly understood workflow requiring training with human feedback. Replicating and understanding this instruction-following process faces three major challenges: the high cost of data collection, the lack of trustworthy evaluation, and the absence of reference method implementations. We address these bottlenecks with AlpacaFarm, a simulator that enables research and development for learning from feedback at a low cost. First, we design LLM based simulator for human feedback that is 45x cheaper than crowdworkers and displays high agreement with humans. Second, we identify an evaluation dataset representative of real-world instructions and propose an automatic evaluation procedure. Third, we contribute reference implementations for several methods (PPO, best-of-n, expert iteration, among others) that learn from pairwise feedback. Finally, as an end-to-end validation of AlpacaFarm, we train and evaluate eleven models on 10k pairs of human feedback and show that rankings of models trained in AlpacaFarm match rankings of models trained on human data. As a demonstration of the research possible in AlpacaFarm, we find that methods that use a reward model can substantially improve over supervised fine-tuning and that our reference PPO implementation leads to a +10% win-rate improvement against Davinci003.

----

## [0] Beta Diffusion

**Authors**: *Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html)

**Abstract**:

We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, further supports the efficacy of KLUBs for optimization. Experimental results on both synthetic data and natural images demonstrate the unique capabilities of beta diffusion in generative modeling of range-bounded data and validate the effectiveness of KLUBs in optimizing diffusion models, thereby making them valuable additions to the family of diffusion-based generative models and the optimization techniques used to train them.

----

## [0] Minimax Optimal Rate for Parameter Estimation in Multivariate Deviated Models

**Authors**: *Dat Do, Huy Nguyen, Khai Nguyen, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fed79713d97df88f9912c8d886fccb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fed79713d97df88f9912c8d886fccb3-Abstract-Conference.html)

**Abstract**:

We study the maximum likelihood estimation (MLE) in the multivariate deviated model where the data are generated from the density function $(1-\lambda^{\ast})h_{0}(x)+\lambda^{\ast}f(x|\mu^{\ast}, \Sigma^{\ast})$ in which $h_{0}$ is a known function, $\lambda^{\ast} \in [0,1]$ and $(\mu^{\ast}, \Sigma^{\ast})$ are unknown parameters to estimate. The main challenges in deriving the convergence rate of the MLE mainly come from two issues: (1) The interaction between the function $h_{0}$ and the density function $f$; (2) The deviated proportion $\lambda^{\ast}$ can go to the extreme points of $[0,1]$ as the sample size tends to infinity. To address these challenges, we develop the \emph{distinguishability condition} to capture the linear independent relation between the function $h_{0}$ and the density function $f$. We then provide comprehensive convergence rates of the MLE via the vanishing rate of $\lambda^{\ast}$ to zero as well as the distinguishability of two functions $h_{0}$ and $f$.

----

## [0] Partial Matrix Completion

**Authors**: *Elad Hazan, Adam Tauman Kalai, Varun Kanade, Clara Mohri, Y. Jennifer Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ff7b1f30e0caf3cc0b2fbfd4d7ebdd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ff7b1f30e0caf3cc0b2fbfd4d7ebdd4-Abstract-Conference.html)

**Abstract**:

The matrix completion problem involves reconstructing a low-rank matrix by using a given set of revealed (and potentially noisy) entries. Although existing methods address the completion of the entire matrix, the accuracy of the completed entries can vary significantly across the matrix, due to differences in the sampling distribution. For instance, users may rate movies primarily from their country or favorite genres, leading to inaccurate predictions for the majority of completed entries.We propose a novel formulation of the problem as Partial Matrix Completion, where the objective is to complete a substantial subset of the entries with high confidence. Our algorithm efficiently handles the unknown and arbitrarily complex nature of the sampling distribution, ensuring high accuracy for all completed entries and sufficient coverage across the matrix. Additionally, we introduce an online version of the problem and present a low-regret efficient algorithm based on iterative gradient updates. Finally, we conduct a preliminary empirical evaluation of our methods.

----

## [0] BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

**Authors**: *Dongxu Li, Junnan Li, Steven C. H. Hoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/602e1a5de9c47df34cae39353a7f5bb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/602e1a5de9c47df34cae39353a7f5bb1-Abstract-Conference.html)

**Abstract**:

Subject-driven text-to-image generation models create novel renditions of an input subject based on text prompts. Existing models suffer from lengthy fine-tuning and difficulties preserving the subject fidelity. To overcome these limitations, we introduce BLIP-Diffusion, a new subject-driven image generation model that supports multimodal control which consumes inputs of subject images and text prompts. Unlike other subject-driven generation models, BLIP-Diffusion introduces a new multimodal encoder which is pre-trained to provide subject representation. We first pre-train the multimodal encoder following BLIP-2 to produce visual representation aligned with the text.Then we design a subject representation learning task which enables a diffusion model to leverage such visual representation and generates new subject renditions. Compared with previous methods such as DreamBooth, our model enables zero-shot subject-driven generation, and efficient fine-tuning for customized subject with up to 20x speedup. We also demonstrate that BLIP-Diffusion can be flexibly combined with existing techniques such as ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications. Implementations are available at: https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion.

----

## [0] Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data

**Authors**: *Yiwen Kou, Zixiang Chen, Quanquan Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/602f5c1b803c53b2aaf0b3864bf3383a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/602f5c1b803c53b2aaf0b3864bf3383a-Abstract-Conference.html)

**Abstract**:

The implicit bias towards solutions with favorable properties is believed to be a key reason why neural networks trained by gradient-based optimization can generalize well. While the implicit bias of gradient flow has been widely studied for homogeneous neural networks (including ReLU and leaky ReLU networks), the implicit bias of gradient descent is currently only understood for smooth neural networks. Therefore, implicit bias in non-smooth neural networks trained by gradient descent remains an open question. In this paper, we aim to answer this question by studying the implicit bias of gradient descent for training two-layer fully connected (leaky) ReLU neural networks. We showed that when the training data are nearly-orthogonal, for leaky ReLU activation function, gradient descent will find a network with a stable rank that converges to $1$, whereas for ReLU activation function, gradient descent will find a neural network with a stable rank that is upper bounded by a constant. Additionally, we show that gradient descent will find a neural network such that all the training data points have the same normalized margin asymptotically. Experiments on both synthetic and real data backup our theoretical findings.

----

## [0] SpecTr: Fast Speculative Decoding via Optimal Transport

**Authors**: *Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix X. Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html)

**Abstract**:

Autoregressive sampling from large language models has led to state-of-the-art results in several natural language tasks.However, autoregressive sampling generates tokens one at a time making it slow, and even prohibitive in certain tasks. One way to speed up sampling is *speculative decoding*: use a small model to sample a *draft* (block or sequence of tokens), and then score all tokens in the draft by the large language model in parallel. A subset of the tokens in the draft are accepted (and the rest rejected) based on a statistical method to guarantee that the final output follows the distribution of the large model. In this work, we provide a principled understanding of speculative decoding through the lens of optimal transport (OT) with *membership cost*. This framework can be viewed as an extension of the well-known *maximal-coupling* problem. This new formulation enables us to generalize the speculative decoding method to allow for a set of $k$ candidates at the token-level, which leads to an improved optimal membership cost. We show that the optimal draft selection algorithm (transport plan) can be computed via linear programming, whose best-known runtime is exponential in $k$. We then propose a valid draft selection algorithm whose acceptance probability is $(1-1/e)$-optimal multiplicatively. Moreover, it can be computed in time almost linear with size of domain of a single token.Using this new draft selection algorithm, we develop a new autoregressive sampling algorithm called *SpecTr*, which provides speedup in decoding while ensuring that there is no quality degradation in the decoded output.We experimentally demonstrate that for state-of-the-art large language models, the proposed approach achieves a wall clock speedup of 2.13X, a further 1.37X speedup over speculative decoding on standard benchmarks.

----

## [0] LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing

**Authors**: *Su Zheng, Haoyu Yang, Binwu Zhu, Bei Yu, Martin D. F. Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/604b9fa9e1c16284e6517d923cf9ff20-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/604b9fa9e1c16284e6517d923cf9ff20-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Computational lithography provides algorithmic and mathematical support for resolution enhancement in optical lithography, which is the critical step in semiconductor manufacturing. The time-consuming lithography simulation and mask optimization processes limit the practical application of inverse lithography technology (ILT), a promising solution to the challenges of advanced-node lithography. Although various machine learning methods for ILT have shown promise for reducing the computational burden, this field is in lack of a dataset that can train the models thoroughly and evaluate the performance comprehensively. To boost the development of AI-driven computational lithography, we present the LithoBench dataset, a collection of circuit layout tiles for deep-learning-based lithography simulation and mask optimization. LithoBench consists of more than 120k tiles that are cropped from real circuit designs or synthesized according to the layout topologies of famous ILT testcases. The ground truths are generated by a famous lithography model in academia and an advanced ILT method. Based on the data, we provide a framework to design and evaluate deep neural networks (DNNs) with the data. The framework is used to benchmark state-of-the-art models on lithography simulation and mask optimization. We hope LithoBench can promote the research and development of computational lithography. LithoBench is available at https://anonymous.4open.science/r/lithobench-APPL.

----

## [0] Proportional Response: Contextual Bandits for Simple and Cumulative Regret Minimization

**Authors**: *Sanath Kumar Krishnamurthy, Ruohan Zhan, Susan Athey, Emma Brunskill*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6058d0c628a03fd95dfe5c72cbdf9e64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6058d0c628a03fd95dfe5c72cbdf9e64-Abstract-Conference.html)

**Abstract**:

In many applications, e.g. in healthcare and e-commerce, the goal of a contextual bandit may be to learn an optimal treatment assignment policy at the end of the experiment. That is, to minimize simple regret. However, this objective remains understudied. We propose a new family of computationally efficient bandit algorithms for the stochastic contextual bandit setting, where a tuning parameter determines the weight placed on cumulative regret minimization (where we establish near-optimal minimax guarantees) versus simple regret minimization (where we establish state-of-the-art guarantees). Our algorithms work with any function class, are robust to model misspecification, and can be used in continuous arm settings. This flexibility comes from constructing and relying on â€œconformal arm sets" (CASs). CASs provide a set of arms for every context, encompassing the context-specific optimal arm with a certain probability across the context distribution. Our positive results on simple and cumulative regret guarantees are contrasted with a negative result, which shows that no algorithm can achieve instance-dependent simple regret guarantees while simultaneously achieving minimax optimal cumulative regret guarantees.

----

## [0] Higher-Order Uncoupled Dynamics Do Not Lead to Nash Equilibrium - Except When They Do

**Authors**: *Sarah Toonsi, Jeff S. Shamma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/605e02ae04cba1ebf6a08206299e76b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/605e02ae04cba1ebf6a08206299e76b9-Abstract-Conference.html)

**Abstract**:

The framework of multi-agent learning explores the dynamics of how an agent's strategies evolve in response to the evolving strategies of other agents. Of particular interest is whether or not agent strategies converge to well known solution concepts such as Nash Equilibrium (NE). In "higher order'' learning, agent dynamics include auxiliary states that can capture phenomena such as path dependencies. We introduce higher-order gradient play dynamics that resemble projected gradient ascent with auxiliary states. The dynamics are "payoff based'' and "uncoupled'' in that each agent's dynamics depend on its own evolving payoff and has no explicit dependence on the utilities of other agents. We first show that for any specific game with an isolated completely mixed-strategy NE, there exist higher-order gradient play dynamics that lead (locally) to that NE, both for the specific game and nearby games with perturbed utility functions. Conversely, we show that for any higher-order gradient play dynamics, there exists a game with a unique isolated completely mixed-strategy NE for which the dynamics do not lead to NE. Finally, we show that convergence to the mixed-strategy equilibrium in coordination games, comes at the expense of the dynamics being inherently internally unstable.

----

## [0] Subject-driven Text-to-Image Generation via Apprenticeship Learning

**Authors**: *Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, William W. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6091bf1542b118287db4088bc16be8d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6091bf1542b118287db4088bc16be8d9-Abstract-Conference.html)

**Abstract**:

Recent text-to-image generation models like DreamBooth have made remarkable progress in generating highly customized images of a target subject, by fine-tuning an ``expert model'' for a given subject from a few examples.However, this process is expensive, since a new expert model must be learned for each subject. In this paper, we present SuTI, a Subject-driven Text-to-Image generator that replaces subject-specific fine tuning with {in-context} learning.Given a few demonstrations of a new subject, SuTI can instantly generate novel renditions of the subject in different scenes, without any subject-specific optimization.SuTI is powered by {apprenticeship learning}, where a single apprentice model is learned from data generated by a massive number of subject-specific expert models. Specifically, we mine millions of image clusters from the Internet, each centered around a specific visual subject. We adopt these clusters to train a massive number of expert models, each specializing in a different subject. The apprentice model SuTI then learns to imitate the behavior of these fine-tuned experts. SuTI can generate high-quality and customized subject-specific images 20x faster than optimization-based SoTA methods. On the challenging DreamBench and DreamBench-v2, our human evaluation shows that SuTI significantly outperforms existing models like InstructPix2Pix, Textual Inversion, Imagic, Prompt2Prompt, Re-Imagen and DreamBooth.

----

## [0] GSLB: The Graph Structure Learning Benchmark

**Authors**: *Zhixun Li, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, Jeffrey Xu Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60bc87f3cf5257579435d92ec12c761b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/60bc87f3cf5257579435d92ec12c761b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Graph Structure Learning (GSL) has recently garnered considerable attention due to its ability to optimize both the parameters of Graph Neural Networks (GNNs) and the computation graph structure simultaneously. Despite the proliferation of GSL methods developed in recent years, there is no standard experimental setting or fair comparison for performance evaluation, which creates a great obstacle to understanding the progress in this field. To fill this gap, we systematically analyze the performance of GSL in different scenarios and develop a comprehensive Graph Structure Learning Benchmark (GSLB) curated from 20 diverse graph datasets and 16 distinct GSL algorithms. Specifically, GSLB systematically investigates the characteristics of GSL in terms of three dimensions: effectiveness, robustness, and complexity. We comprehensively evaluate state-of-the-art GSL algorithms in node- and graph-level tasks, and analyze their performance in robust learning and model complexity. Further, to facilitate reproducible research, we have developed an easy-to-use library for training, evaluating, and visualizing different GSL methods. Empirical results of our extensive experiments demonstrate the ability of GSL and reveal its potential benefits on various downstream tasks, offering insights and opportunities for future research. The code of GSLB is available at: https://github.com/GSL-Benchmark/GSLB.

----

## [0] Consensus and Subjectivity of Skin Tone Annotation for ML Fairness

**Authors**: *Candice Schumann, Femi Olanubi, Auriel Wright, Ellis Monk Jr., Courtney Heldreth, Susanna Ricco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60d25b3210c92f5ba2002a8e1f1adf1c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/60d25b3210c92f5ba2002a8e1f1adf1c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Understanding different human attributes and how they affect model behavior may become a standard need for all model creation and usage, from traditional computer vision tasks to the newest multimodal generative AI systems. In computer vision specifically, we have relied on datasets augmented with perceived attribute signals (eg, gender presentation, skin tone, and age) and benchmarks enabled by these datasets. Typically labels for these tasks come from human annotators. However, annotating attribute signals, especially skin tone, is a difficult and subjective task. Perceived skin tone is affected by technical factors, like lighting conditions, and social factors that shape an annotator's lived experience.This paper examines the subjectivity of skin tone annotation through a series of annotation experiments using the Monk Skin Tone (MST) scale~\cite{Monk2022Monk}, a small pool of professional photographers, and a much larger pool of trained crowdsourced annotators. Along with this study we release the Monk Skin Tone Examples (MST-E) dataset, containing 1515 images and 31 videos spread across the full MST scale. MST-E is designed to help train human annotators to annotate MST effectively.Our study shows that annotators can reliably annotate skin tone in a way that aligns with an expert in the MST scale, even under challenging environmental conditions. We also find evidence that annotators from different geographic regions rely on different mental models of MST categories resulting in annotations that systematically vary across regions. Given this, we advise practitioners to use a diverse set of annotators and a higher replication count for each image when annotating skin tone for fairness research.

----

## [0] Large Language Models can Implement Policy Iteration

**Authors**: *Ethan A. Brooks, Logan Walls, Richard L. Lewis, Satinder Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60dc7fa827f5f761ad481e2ad40b5573-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/60dc7fa827f5f761ad481e2ad40b5573-Abstract-Conference.html)

**Abstract**:

In this work, we demonstrate a method for implementing policy iteration using a large language model. While the application of foundation models to RL has received considerable attention, most approaches rely on either (1) the curation of expert demonstrations (either through manual design or task-specific pretraining) or (2) adaptation to the task of interest using gradient methods (either fine-tuning or training of adapter layers). Both of these techniques have drawbacks. Collecting demonstrations is labor-intensive, and algorithms that rely on them do not outperform the experts from which the demonstrations were derived. All gradient techniques are inherently slow, sacrificing the “few-shot” quality that makes in-context learning attractive to begin with. Our method demonstrates that a large language model can be used to implement policy iteration using the machinery of in-context learning, enabling it to learn to perform RL tasks without expert demonstrations or gradients. Our approach iteratively updates the contents of the prompt from which it derives its policy through trial-and-error interaction with an RL environment. In order to eliminate the role of in-weights learning (on which approaches like Decision Transformer rely heavily), we demonstrate our method using Codex (M. Chen et al. 2021b), a language model with no prior knowledge of the domains on which we evaluate it.

----

## [0] ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

**Authors**: *Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)

**Abstract**:

Auxiliary-Task Learning (ATL) aims to improve the performance of the target task by leveraging the knowledge obtained from related tasks. Occasionally, learning multiple tasks simultaneously results in lower accuracy than learning only the target task, which is known as negative transfer. This problem is often attributed to the gradient conflicts among tasks, and is frequently tackled by coordinating the task gradients in previous works. However, these optimization-based methods largely overlook the auxiliary-target generalization capability. To better understand the root cause of negative transfer, we experimentally investigate it from both optimization and generalization perspectives. Based on our findings, we introduce ForkMerge, a novel approach that periodically forks the model into multiple branches, automatically searches the varying task weights by minimizing target validation errors, and dynamically merges all branches to filter out detrimental task-parameter updates. On a series of auxiliary-task learning benchmarks, ForkMerge outperforms existing methods and effectively mitigates negative transfer.

----

## [0] Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness

**Authors**: *Xinli Yue, Ningping Mou, Qian Wang, Lingchen Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)

**Abstract**:

Adversarial Robustness Distillation (ARD) aims to transfer the robustness of large teacher models to small student models, facilitating the attainment of robust performance on resource-limited devices. However, existing research on ARD primarily focuses on the overall robustness of student models, overlooking the crucial aspect of $\textit{robust fairness}$. Specifically, these models may demonstrate strong robustness on some classes of data while exhibiting high vulnerability on other classes. Unfortunately, the "buckets effect" implies that the robustness of the deployed model depends on the classes with the lowest level of robustness. In this paper, we first investigate the inheritance of robust fairness during ARD and reveal that student models only partially inherit robust fairness from teacher models. We further validate this issue through fine-grained experiments with various model capacities and find that it may arise due to the gap in capacity between teacher and student models, as well as the existing methods treating each class equally during distillation. Based on these observations, we propose $\textbf{Fair}$ $\textbf{A}$dversarial $\textbf{R}$obustness $\textbf{D}$istillation (Fair-ARD), a novel framework for enhancing the robust fairness of student models by increasing the weights of difficult classes, and design a geometric perspective-based method to quantify the difficulty of different classes for determining the weights. Extensive experiments show that Fair-ARD surpasses both state-of-the-art ARD methods and existing robust fairness algorithms in terms of robust fairness (e.g., the worst-class robustness under AutoAttack is improved by at most 12.3\% and 5.3\% using ResNet18 on CIFAR10, respectively), while also slightly improving overall robustness. Our code is available at: [https://github.com/NISP-official/Fair-ARD](https://github.com/NISP-official/Fair-ARD).

----

## [0] Real3D-AD: A Dataset of Point Cloud Anomaly Detection

**Authors**: *Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, Feng Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

High-precision point cloud anomaly detection is the gold standard for identifying the defects of advancing machining and precision manufacturing. Despite some methodological advances in this area, the scarcity of datasets and the lack of a systematic benchmark hinder its development. We introduce Real3D-AD, a challenging high-precision point cloud anomaly detection dataset, addressing the limitations in the field. With 1,254 high-resolution 3D items (from forty thousand to millions of points for each item), Real3D-AD is the largest dataset for high-precision 3D industrial anomaly detection to date. Real3D-AD surpasses existing 3D anomaly detection datasets available in terms of point cloud resolution (0.0010mm-0.0015mm), $360^{\circ}$ degree coverage and perfect prototype. Additionally, we present a comprehensive benchmark for Real3D-AD, revealing the absence of baseline methods for high-precision point cloud anomaly detection. To address this, we propose Reg3D-AD, a registration-based 3D anomaly detection method incorporating a novel feature memory bank that preserves local and global representations. Extensive experiments on the Real3D-AD dataset highlight the effectiveness of Reg3D-AD. For reproducibility and accessibility, we provide the Real3D-AD dataset, benchmark source code, and Reg3D-AD on our website: https://github.com/M-3LAB/Real3D-AD.

----

## [0] Differentiable Sampling of Categorical Distributions Using the CatLog-Derivative Trick

**Authors**: *Lennert De Smet, Emanuele Sansone, Pedro Zuidberg Dos Martires*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61202bb341e7e0a6026ea134a5057abf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61202bb341e7e0a6026ea134a5057abf-Abstract-Conference.html)

**Abstract**:

Categorical random variables can faithfully represent the discrete and uncertain aspects of data as part of a discrete latent variable model. Learning in such models necessitates taking gradients with respect to the parameters of the categorical probability distributions, which is often intractable due to their combinatorial nature. A popular technique to estimate these otherwise intractable gradients is the Log-Derivative trick. This trick forms the basis of the well-known REINFORCE gradient estimator and its many extensions. While the Log-Derivative trick allows us to differentiate through samples drawn from categorical distributions, it does not take into account the discrete nature of the distribution itself. Our first contribution addresses this shortcoming by introducing the CatLog-Derivative trick -- a variation of the Log-Derivative trick tailored towards categorical distributions. Secondly, we use the CatLog-Derivative trick to introduce IndeCateR, a novel and unbiased gradient estimator for the important case of products of independent categorical distributions with provably lower variance than REINFORCE. Thirdly, we empirically show that IndeCateR can be efficiently implemented and that its gradient estimates have significantly lower bias and variance for the same number of samples compared to the state of the art.

----

## [0] Variational Imbalanced Regression: Fair Uncertainty Quantification via Probabilistic Smoothing

**Authors**: *Ziyan Wang, Hao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/612a56f193d031687683445cd0001083-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/612a56f193d031687683445cd0001083-Abstract-Conference.html)

**Abstract**:

Existing regression models tend to fall short in both accuracy and uncertainty estimation when the label distribution is imbalanced. In this paper, we propose a probabilistic deep learning model, dubbed variational imbalanced regression (VIR), which not only performs well in imbalanced regression but naturally produces reasonable uncertainty estimation as a byproduct. Different from typical variational autoencoders assuming I.I.D. representations (a data point's representation is not directly affected by other data points), our VIR borrows data with similar regression labels to compute the latent representation's variational distribution; furthermore, different from deterministic regression models producing point estimates, VIR predicts the entire normal-inverse-gamma distributions and modulates the associated conjugate distributions to impose probabilistic reweighting on the imbalanced data, thereby providing better uncertainty estimation. Experiments in several real-world datasets show that our VIR can outperform state-of-the-art imbalanced regression models in terms of both accuracy and uncertainty estimation. Code will soon be available at https://github.com/Wang-ML-Lab/variational-imbalanced-regression.

----

## [0] Towards A Richer 2D Understanding of Hands at Scale

**Authors**: *Tianyi Cheng, Dandan Shan, Ayda Hassen, Richard E. L. Higgins, David Fouhey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html)

**Abstract**:

As humans, we learn a lot about how to interact with the world by observing others interacting with their hands. To help AI systems obtain a better understanding of hand interactions, we introduce a new model that produces a rich understanding of hand interaction. Our system produces a richer output than past systems at a larger scale. Our outputs include boxes and segments for hands, in-contact objects, and second objects touched by tools as well as contact and grasp type. Supporting this method are annotations of 257K images, 401K hands, 288K objects, and 19K second objects spanning four datasets. We show that our method provides rich information and performs and generalizes well.

----

## [0] Effective Human-AI Teams via Learned Natural Language Rules and Onboarding

**Authors**: *Hussein Mozannar, Jimin J. Lee, Dennis Wei, Prasanna Sattigeri, Subhro Das, David A. Sontag*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61355b9c218505505d1bedede9da56b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61355b9c218505505d1bedede9da56b2-Abstract-Conference.html)

**Abstract**:

People are relying on AI agents to assist them with various tasks. The human must know when to rely on the agent, collaborate with the agent, or ignore its suggestions. In this work, we propose to learn rules grounded in data regions and described in natural language that illustrate how the human should collaborate with the AI. Our novel region discovery algorithm finds local regions in the data as neighborhoods in an embedding space that corrects the human prior. Each region is then described using an iterative and contrastive procedure where a large language model describes the region. We then teach these rules to the human via an onboarding stage. Through user studies on object detection and question-answering tasks, we show that our method can lead to more accurate human-AI teams. We also evaluate our region discovery and description algorithms separately.

----

## [0] On Certified Generalization in Structured Prediction

**Authors**: *Bastian Boll, Christoph Schnörr*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61674667d642ae52f6bb281bea90ee29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61674667d642ae52f6bb281bea90ee29-Abstract-Conference.html)

**Abstract**:

In structured prediction, target objects have rich internal structure which does not factorize into independent components and violates common i.i.d. assumptions. This challenge becomes apparent through the exponentially large output space in applications such as image segmentation or scene graph generation.We present a novel PAC-Bayesian risk bound for structured prediction wherein the rate of generalization scales not only with the number of structured examples but also with their size.The underlying assumption, conforming to ongoing research on generative models, is that data are generated by the Knothe-Rosenblatt rearrangement of a factorizing reference measure. This allows to explicitly distill the structure between random output variables into a Wasserstein dependency matrix. Our work makes a preliminary step towards leveraging powerful generative models to establish generalization bounds for discriminative downstream tasks in the challenging setting of structured prediction.

----

## [0] SutraNets: Sub-series Autoregressive Networks for Long-Sequence, Probabilistic Forecasting

**Authors**: *Shane Bergsma, Timothy Zeyl, Lei Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6171c9e600432a42688ad61a525951bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6171c9e600432a42688ad61a525951bf-Abstract-Conference.html)

**Abstract**:

We propose SutraNets, a novel method for neural probabilistic forecasting of long-sequence time series. SutraNets use an autoregressive generative model to factorize the likelihood of long sequences into products of conditional probabilities. When generating long sequences, most autoregressive approaches suffer from harmful error accumulation, as well as challenges in modeling long-distance dependencies. SutraNets treat long, univariate prediction as multivariate prediction over lower-frequency sub-series. Autoregression proceeds across time and across sub-series in order to ensure coherent multivariate (and, hence, high-frequency univariate) outputs. Since sub-series can be generated using fewer steps, SutraNets effectively reduce error accumulation and signal path distances. We find SutraNets to significantly improve forecasting accuracy over competitive alternatives on six real-world datasets, including when we vary the number of sub-series and scale up the depth and width of the underlying sequence models.

----

## [0] Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints

**Authors**: *Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, Yangqiu Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html)

**Abstract**:

Querying knowledge graphs (KGs) using deep learning approaches can naturally leverage the reasoning and generalization ability to learn to infer better answers. Traditional neural complex query answering (CQA) approaches mostly work on entity-centric KGs. However, in the real world, we also need to make logical inferences about events, states, and activities (i.e., eventualities or situations) to push learning systems from System I to System II, as proposed by Yoshua Bengio. Querying logically from an EVentuality-centric KG (EVKG) can naturally provide references to such kind of intuitive and logical inference. Thus, in this paper, we propose a new framework to leverage neural methods to answer complex logical queries based on an EVKG, which can satisfy not only traditional first-order logic constraints but also implicit logical constraints over eventualities concerning their occurrences and orders. For instance, if we know that Food is bad happens before PersonX adds soy sauce, then PersonX adds soy sauce is unlikely to be the cause of Food is bad due to implicit temporal constraint. To facilitate consistent reasoning on EVKGs, we propose Complex Eventuality Query Answering (CEQA), a more rigorous definition of CQA that considers the implicit logical constraints governing the temporal order and occurrence of eventualities. In this manner, we propose to leverage theorem provers for constructing benchmark datasets to ensure the answers satisfy implicit logical constraints. We also propose a Memory-Enhanced Query Encoding (MEQE) approach to significantly improve the performance of state-of-the-art neural query encoders on the CEQA task.

----

## [0] Fast Bellman Updates for Wasserstein Distributionally Robust MDPs

**Authors**: *Zhuodong Yu, Ling Dai, Shaohang Xu, Siyang Gao, Chin Pang Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61779e9b0c26a31c5f36bd3e8c180dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61779e9b0c26a31c5f36bd3e8c180dcf-Abstract-Conference.html)

**Abstract**:

Markov decision processes (MDPs) often suffer from the sensitivity issue under model ambiguity. In recent years, robust MDPs have emerged as an effective framework to overcome this challenge. Distributionally robust MDPs extend the robust MDP framework by incorporating distributional information of the uncertain model parameters to alleviate the conservative nature of robust MDPs. This paper proposes a computationally efficient solution framework for solving distributionally robust MDPs with Wasserstein ambiguity sets. By exploiting the specific problem structure, the proposed framework decomposes the optimization problems associated with distributionally robust Bellman updates into smaller subproblems, which can be solved efficiently. The overall complexity of the proposed algorithm is quasi-linear in both the numbers of states and actions when the distance metric of the Wasserstein distance is chosen to be $L_1$, $L_2$, or $L_{\infty}$ norm, and so the computational cost of distributional robustness is substantially reduced. Our numerical experiments demonstrate that the proposed algorithms outperform other state-of-the-art solution methods.

----

## [0] LoRA: A Logical Reasoning Augmented Dataset for Visual Question Answering

**Authors**: *Jingying Gao, Qi Wu, Alan Blair, Maurice Pagnucco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/617ff5271b2b41dfb217a3b0f1b3d1be-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/617ff5271b2b41dfb217a3b0f1b3d1be-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The capacity to reason logically is a hallmark of human cognition. Humans excel at integrating multimodal information for locigal reasoning, as exemplified by the Visual Question Answering (VQA) task, which is a challenging multimodal task. VQA tasks and large vision-and-language models aim to tackle reasoning problems, but the accuracy, consistency and fabrication of the generated answers is hard to evaluate in the absence of a VQA dataset that can offer formal, comprehensive and systematic complex logical reasoning questions. To address this gap, we present LoRA, a novel Logical Reasoning Augmented VQA dataset that requires formal and complex description logic reasoning based on a food-and-kitchen knowledge base. Our main objective in creating LoRA is to enhance the complex and formal logical reasoning capabilities of VQA models, which are not adequately measured by existing VQA datasets. We devise strong and flexible programs to automatically generate 200,000 diverse description logic reasoning questions based on the SROIQ Description Logic, along with realistic kitchen scenes and ground truth answers. We fine-tune the latest transformer VQA models and evaluate the zero-shot performance of the state-of-the-art large vision-and-language models on LoRA. The results reveal that LoRA presents a unique challenge in logical reasoning, setting a systematic and comprehensive evaluation standard.

----

## [0] Practical Contextual Bandits with Feedback Graphs

**Authors**: *Mengxiao Zhang, Yuheng Zhang, Olga Vrousgou, Haipeng Luo, Paul Mineiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html)

**Abstract**:

While contextual bandit has a mature theory, effectively leveraging different feedback patterns to enhance the pace of learning remains unclear. Bandits with feedback graphs, which interpolates between the full information and bandit regimes, provides a promising framework to mitigate the statistical complexity of learning. In this paper, we propose and analyze an approach to contextual bandits with feedback graphs based upon reduction to regression.  The resulting algorithms are computationally practical and achieve established minimax rates, thereby reducing the statistical complexity in real-world applications.

----

## [0] Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control

**Authors**: *Nate Rahn, Pierluca D'Oro, Harley Wiltzer, Pierre-Luc Bacon, Marc G. Bellemare*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6191ab7080c840f67eaf5dff7d5edfcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6191ab7080c840f67eaf5dff7d5edfcb-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning agents for continuous control are known to exhibit significant instability in their performance over time. In this work, we provide a fresh perspective on these behaviors by studying the return landscape: the mapping between a policy and a return. We find that popular algorithms traverse noisy neighborhoods of this landscape, in which a single update to the policy parameters leads to a wide range of returns. By taking a distributional view of these returns, we map the landscape, characterizing failure-prone regions of policy space and revealing a hidden dimension of policy quality. We show that the landscape exhibits surprising structure by finding simple paths in parameter space which improve the stability of a policy. To conclude, we develop a distribution-aware procedure which finds such paths, navigating away from noisy neighborhoods in order to improve the robustness of a policy. Taken together, our results provide new insight into the optimization, evaluation, and design of agents.

----

## [0] Real-World Image Variation by Aligning Diffusion Inversion Chain

**Authors**: *Yuechen Zhang, Jinbo Xing, Eric Lo, Jiaya Jia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html)

**Abstract**:

Recent diffusion model advancements have enabled high-fidelity images to be generated using text prompts. However, a domain gap exists between generated images and real-world images, which poses a challenge in generating high-quality variations of real-world images. Our investigation uncovers that this domain gap originates from a latents' distribution gap in different diffusion processes. To address this issue, we propose a novel inference pipeline called Real-world Image Variation by ALignment (RIVAL) that utilizes diffusion models to generate image variations from a single image exemplar. Our pipeline enhances the generation quality of image variations by aligning the image generation process to the source image's inversion chain. Specifically, we demonstrate that step-wise latent distribution alignment is essential for generating high-quality variations. To attain this, we design a cross-image self-attention injection for feature interaction and a step-wise distribution normalization to align the latent features. Incorporating these alignment processes into a diffusion model allows RIVAL to generate high-quality image variations without further parameter optimization. Our experimental results demonstrate that our proposed approach outperforms existing methods concerning semantic similarity and perceptual quality. This generalized inference pipeline can be easily applied to other diffusion-based generation tasks, such as image-conditioned text-to-image generation and stylization. Project page: https://rival-diff.github.io

----

## [0] Vocabulary-free Image Classification

**Authors**: *Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, Elisa Ricci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/619cbddb92b8c6fecaf2b86463153be9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/619cbddb92b8c6fecaf2b86463153be9-Abstract-Conference.html)

**Abstract**:

Recent advances in large vision-language models have revolutionized the image classification paradigm. Despite showing impressive zero-shot capabilities, a pre-defined set of categories, a.k.a. the vocabulary, is assumed at test time for composing the textual prompts. However, such assumption can be impractical when the semantic context is unknown and evolving. We thus formalize a novel task, termed as Vocabulary-free Image Classification (VIC), where we aim to assign to an input image a class that resides in an unconstrained language-induced semantic space, without the prerequisite of a known vocabulary. VIC is a challenging task as the semantic space is extremely large, containing millions of concepts, with hard-to-discriminate fine-grained categories. In this work, we first empirically verify that representing this semantic space by means of an external vision-language database is the most effective way to obtain semantically relevant content for classifying the image. We then propose Category Search from External Databases (CaSED), a method that exploits a pre-trained vision-language model and an external vision-language database to address VIC in a training-free manner. CaSED first extracts a set of candidate categories from captions retrieved from the database based on their semantic similarity to the image, and then assigns to the image the best matching candidate category according to the same vision-language model. Experiments on benchmark datasets validate that CaSED outperforms other complex vision-language frameworks, while being efficient with much fewer parameters, paving the way for future research in this direction.

----

## [0] A Novel Framework for Policy Mirror Descent with General Parameterization and Linear Convergence

**Authors**: *Carlo Alfano, Rui Yuan, Patrick Rebeschini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61a9278dfef5f871b5e472389f8d6fa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61a9278dfef5f871b5e472389f8d6fa1-Abstract-Conference.html)

**Abstract**:

Modern policy optimization methods in reinforcement learning, such as TRPO and PPO, owe their success to the use of parameterized policies. However, while theoretical guarantees have been established for this class of algorithms, especially in the tabular setting, the use of general parameterization schemes remains mostly unjustified. In this work, we introduce a novel framework for policy optimization based on mirror descent that naturally accommodates general parameterizations. The policy class induced by our scheme recovers known classes, e.g., softmax, and generates new ones depending on the choice of mirror map. Using our framework, we obtain the first result that guarantees linear convergence for a policy-gradient-based method involving general parameterization. To demonstrate the ability of our framework to accommodate general parameterization schemes, we provide its sample complexity when using shallow neural networks, show that it represents an improvement upon the previous best results, and empirically validate the effectiveness of our theoretical claims on classic control tasks.

----

## [0] Weakly-Supervised Concealed Object Segmentation with SAM-based Pseudo Labeling and Multi-scale Feature Grouping

**Authors**: *Chunming He, Kai Li, Yachao Zhang, Guoxia Xu, Longxiang Tang, Yulun Zhang, Zhenhua Guo, Xiu Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html)

**Abstract**:

Weakly-Supervised Concealed Object Segmentation (WSCOS) aims to  segment objects well blended with surrounding environments using sparsely-annotated data for model training. It remains a challenging task since (1) it is hard to distinguish concealed objects from the background due to the intrinsic similarity and (2) the sparsely-annotated training data only provide weak supervision for model learning. In this paper, we propose a new WSCOS method to address these two challenges. To tackle the intrinsic similarity challenge, we design a multi-scale feature grouping module that first groups features at different granularities and then aggregates these grouping results. By grouping similar features together, it encourages segmentation coherence, helping obtain complete segmentation results for both single and multiple-object images. For the weak supervision challenge, we utilize the recently-proposed vision foundation model, ``Segment Anything Model (SAM)'', and use the provided sparse annotations as prompts to generate segmentation masks, which are used to train the model. To alleviate the impact of low-quality segmentation masks, we further propose a series of strategies, including multi-augmentation result ensemble, entropy-based pixel-level weighting, and entropy-based image-level selection. These strategies help provide more reliable supervision to train the segmentation model. We verify the effectiveness of our method on various WSCOS tasks, and experiments demonstrate that our method achieves state-of-the-art performance on these tasks.

----

## [0] Ordering-based Conditions for Global Convergence of Policy Gradient Methods

**Authors**: *Jincheng Mei, Bo Dai, Alekh Agarwal, Mohammad Ghavamzadeh, Csaba Szepesvári, Dale Schuurmans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61c00c07e6d27285e4b952e96cc65666-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61c00c07e6d27285e4b952e96cc65666-Abstract-Conference.html)

**Abstract**:

We prove that, for finite-arm bandits with linear function approximation, the global convergence of policy gradient (PG) methods depends on inter-related properties between the policy update and the representation. textcolor{blue}{First}, we establish a few key observations that frame the study: \textbf{(i)} Global convergence can be achieved under linear function approximation without policy or reward realizability, both for the standard Softmax PG and natural policy gradient (NPG). \textbf{(ii)} Approximation error is not a key quantity for characterizing global convergence in either algorithm. \textbf{(iii)} The conditions on the representation that imply global convergence are different between these two algorithms. Overall, these observations call into question approximation error as an appropriate quantity for characterizing the global convergence of PG methods under linear function approximation. \textcolor{blue}{Second}, motivated by these observations, we establish new general results: \textbf{(i)} NPG with linear function approximation achieves global convergence \emph{if and only if} the projection of the reward  onto the representable space preserves the optimal action's rank, a quantity that is not strongly related to approximation error. \textbf{(ii)} The global convergence of Softmax PG occurs if the representation satisfies a non-domination condition and can preserve the ranking of rewards, which goes well beyond policy or reward realizability. We provide experimental results to support these theoretical findings.

----

## [0] Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning

**Authors**: *Berken Utku Demirel, Christian Holz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61c2c6338033da68885e0226881cbe71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61c2c6338033da68885e0226881cbe71-Abstract-Conference.html)

**Abstract**:

The success of contrastive learning is well known to be dependent on data augmentation.Although the degree of data augmentations has been well controlled by utilizing pre-defined techniques in some domains like vision, time-series data augmentation is less explored and remains a challenging problem due to the complexity of the data generation mechanism, such as the intricate mechanism involved in the cardiovascular system.Moreover, there is no widely recognized and general time-series augmentation method that can be applied across different tasks.In this paper, we propose a novel data augmentation method for time-series tasks that aims to connect intra-class samples together, and thereby find order in the latent space.Our method builds upon the well-known data augmentation technique of mixup by incorporating a novel approach that accounts for the non-stationary nature of time-series data.Also, by controlling the degree of chaos created by data augmentation, our method leads to improved feature representations and performance on downstream tasks.We evaluate our proposed method on three time-series tasks, including heart rate estimation, human activity recognition, and cardiovascular disease detection. Extensive experiments against the state-of-the-art methods show that the proposed method outperforms prior works on optimal data generation and known data augmentation techniques in three tasks, reflecting the effectiveness of the presented method. The source code is available at double-blind policy.

----

## [0] List and Certificate Complexities in Replicable Learning

**Authors**: *Peter Dixon, Aduri Pavan, Jason Vander Woude, N. V. Vinodchandran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61d0a96d4a73b626367310b3ad32579d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61d0a96d4a73b626367310b3ad32579d-Abstract-Conference.html)

**Abstract**:

We investigate replicable learning algorithms. Informally  a learning algorithm is replicable if the algorithm outputs the same canonical hypothesis over multiple runs with high probability, even when different runs observe a different set of samples from the unknown data distribution. In general, such a strong notion of replicability is not achievable. Thus we consider two feasible notions of replicability called {\em list replicability} and {\em certificate replicability}. Intuitively, these notions capture the degree of (non) replicability. The goal is to design learning algorithms with optimal list and certificate complexities while minimizing the sample complexity.  Our contributions are the following.1. We first study the learning task of estimating the biases of $d$ coins, up to an additive error of $\varepsilon$, by observing samples. For this task, we design a $(d+1)$-list replicable algorithm. To complement this result, we establish that the list complexity is optimal, i.e there are no learning algorithms with a list size smaller than $d+1$ for this task. We also design learning algorithms with certificate complexity $\tilde{O}(\log d)$.   The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\varepsilon^2})$ where $\varepsilon$ is the approximation error parameter (for a constant error probability).  2. In the PAC model, we show that any hypothesis class that is learnable with $d$-nonadaptive statistical queries can be learned via a $(d+1)$-list replicable algorithm and also via a $\tilde{O}(\log d)$-certificate replicable algorithm. The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\nu^2})$ where $\nu$ is the approximation error of the statistical query. We also show that for the concept class \dtep, the list complexity is exactly $d+1$ with respect to the uniform distribution.   To establish our upper bound results we use rounding schemes induced by geometric partitions with certain properties. We use Sperner/KKM Lemma to establish the lower bound results.

----

## [0] Flat Seeking Bayesian Neural Networks

**Authors**: *Van-Anh Nguyen, Tung-Long Vuong, Hoang Phan, Thanh-Toan Do, Dinh Q. Phung, Trung Le*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html)

**Abstract**:

Bayesian Neural Networks (BNNs) provide a probabilistic interpretation for deep learning models by imposing a prior distribution over model parameters and inferring a posterior distribution based on observed data. The model sampled from the posterior distribution can be used for providing ensemble predictions and quantifying prediction uncertainty. It is well-known that deep learning models with lower sharpness have better generalization ability. However, existing posterior inferences are not aware of sharpness/flatness in terms of formulation, possibly leading to high sharpness for the models sampled from them. In this paper, we develop theories, the Bayesian setting, and the variational inference approach for the sharpness-aware posterior. Specifically, the models sampled from our sharpness-aware posterior, and the optimal approximate posterior estimating this sharpness-aware posterior, have better flatness, hence possibly possessing higher generalization ability. We conduct experiments by leveraging the sharpness-aware posterior with state-of-the-art Bayesian Neural Networks, showing that the flat-seeking counterparts outperform their baselines in all metrics of interest.

----

## [0] Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns

**Authors**: *Xin Liu, Zheng Li, Yifan Gao, Jingfeng Yang, Tianyu Cao, Zhengyang Wang, Bing Yin, Yangqiu Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/621d0fd41c720ab252e178b77c200d90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/621d0fd41c720ab252e178b77c200d90-Abstract-Conference.html)

**Abstract**:

The goal of session-based recommendation in E-commerce is to predict the next item that an anonymous user will purchase based on the browsing and purchase history. However, constructing global or local transition graphs to supplement session data can lead to noisy correlations and user intent vanishing. In this work, we propose the Frequent Attribute Pattern Augmented Transformer (FAPAT) that characterizes user intents by building attribute transition graphs and matching attribute patterns. Specifically, the frequent and compact attribute patterns are served as memory to augment session representations, followed by a gate and a transformer block to fuse the whole session information. Through extensive experiments on two public benchmarks and 100 million industrial data in three domains, we demonstrate that FAPAT consistently outperforms state-of-the-art methods by an average of 4.5% across various evaluation metrics (Hits, NDCG, MRR). Besides evaluating the next-item prediction, we estimate the models' capabilities to capture user intents via predicting items' attributes and period-item recommendations.

----

## [0] Can Language Models Solve Graph Problems in Natural Language?

**Authors**: *Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) are increasingly adopted for a variety of tasks with implicit graphical structures, such as planning in robotics, multi-hop question answering or knowledge probing, structured commonsense reasoning, and more. While LLMs have advanced the state-of-the-art on these tasks with structure implications, whether LLMs could explicitly process textual descriptions of graphs and structures, map them to grounded conceptual spaces, and perform structured operations remains underexplored. To this end, we propose NLGraph (Natural Language Graph), a comprehensive benchmark of graph-based problem solving designed in natural language. NLGraph contains 29,370 problems, covering eight graph reasoning tasks with varying complexity from simple tasks such as connectivity and shortest path up to complex problems such as maximum flow and simulating graph neural networks. We evaluate LLMs (GPT-3/4) with various prompting approaches on the NLGraph benchmark and find that 1) language models do demonstrate preliminary graph reasoning abilities, 2) the benefit of advanced prompting and in-context learning diminishes on more complex graph problems, while 3) LLMs are also (un)surprisingly brittle in the face of spurious correlations in graph and problem settings. We then propose Build-a-Graph Prompting and Algorithmic Prompting, two instruction-based approaches to enhance LLMs in solving natural language graph problems. Build-a-Graph and Algorithmic prompting improve the performance of LLMs on NLGraph by 3.07% to 16.85% across multiple tasks and settings, while how to solve the most complicated graph reasoning tasks in our setup with language models remains an open research question.

----

## [0] Language-driven Scene Synthesis using Multi-conditional Diffusion Model

**Authors**: *Vuong Dinh An, Minh Nhat Vu, Toan Nguyen, Baoru Huang, Dzung Nguyen, Thieu Vo, Anh Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html)

**Abstract**:

Scene synthesis is a challenging problem with several industrial applications. Recently, substantial efforts have been directed to synthesize the scene using human motions, room layouts, or spatial graphs as the input. However, few studies have addressed this problem from multiple modalities, especially combining text prompts. In this paper, we propose a language-driven scene synthesis task, which is a new task that integrates text prompts, human motion, and existing objects for scene synthesis. Unlike other single-condition synthesis tasks, our problem involves multiple conditions and requires a strategy for processing and encoding them into a unified space. To address the challenge, we present a multi-conditional diffusion model, which differs from the implicit unification approach of other diffusion literature by explicitly predicting the guiding points for the original data distribution. We demonstrate that our approach is theoretically supportive. The intensive experiment results illustrate that our method outperforms state-of-the-art benchmarks and enables natural scene editing applications. The source code and dataset can be accessed at https://lang-scene-synth.github.io/.

----

## [0] Implicit Contrastive Representation Learning with Guided Stop-gradient

**Authors**: *Byeongchan Lee, Sehyun Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6274172f7d981a8d58bbfd52342a9d1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6274172f7d981a8d58bbfd52342a9d1f-Abstract-Conference.html)

**Abstract**:

In self-supervised representation learning, Siamese networks are a natural architecture for learning transformation-invariance by bringing representations of positive pairs closer together. But it is prone to collapse into a degenerate solution. To address the issue, in contrastive learning, a contrastive loss is used to prevent collapse by moving representations of negative pairs away from each other. But it is known that algorithms with negative sampling are not robust to a reduction in the number of negative samples. So, on the other hand, there are algorithms that do not use negative pairs. Many positive-only algorithms adopt asymmetric network architecture consisting of source and target encoders as a key factor in coping with collapse. By exploiting the asymmetric architecture, we introduce a methodology to implicitly incorporate the idea of contrastive learning. As its implementation, we present a novel method guided stop-gradient. We apply our method to benchmark algorithms SimSiam and BYOL and show that our method stabilizes training and boosts performance. We also show that the algorithms with our method work well with small batch sizes and do not collapse even when there is no predictor. The code is available in the supplementary material.

----

## [0] QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

**Authors**: *Simone Papicchio, Paolo Papotti, Luca Cagliero*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62a24b69b820d30e9e5ad4f15ff7bf72-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/62a24b69b820d30e9e5ad4f15ff7bf72-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Table Representation Learning (TRL) models are commonly pre-trained on large open-domain datasets comprising millions of tables and then used to address downstream tasks. Choosing the right TRL model to use on proprietary data can be challenging, as the best results depend on the content domain, schema, and data quality. Our purpose is to support end-users in testing TRL models on proprietary data in two established SQL-centric tasks, i.e., Question Answering (QA) and Semantic Parsing (SP). We present QATCH (Query-Aided TRL Checklist), a toolbox to highlight TRL models’ strengths and weaknesses on relational tables unseen at training time. For an input table, QATCH automatically generates a testing checklist tailored to QA and SP. Checklist generation is driven by a SQL query engine that crafts tests of different complexity. This design facilitates inherent portability, allowing the checks to be used by alternative models. We also introduce a set of cross-task performance metrics evaluating the TRL model’s performance over its output. Finally, we show how QATCH automatically generates tests for proprietary datasets to evaluate various state-of-the-art models including TAPAS, TAPEX, and CHATGPT.

----

## [0] Improved Best-of-Both-Worlds Guarantees for Multi-Armed Bandits: FTRL with General Regularizers and Multiple Optimal Arms

**Authors**: *Tiancheng Jin, Junyan Liu, Haipeng Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html)

**Abstract**:

We study the problem of designing adaptive multi-armed bandit algorithms that perform optimally in both the stochastic setting and the adversarial setting simultaneously (often known as a best-of-both-world guarantee). A line of recent works shows that when configured and analyzed properly, the Follow-the-Regularized-Leader (FTRL) algorithm, originally designed for the adversarial setting, can in fact optimally adapt to the stochastic setting as well. Such results, however, critically rely on an assumption that there exists one unique optimal arm. Recently, Ito [2021] took the first step to remove such an undesirable uniqueness assumption for one particular FTRL algorithm withthe 1/2-Tsallis entropy regularizer. In this work, we significantly improve and generalize this result, showing that uniqueness is unnecessary for FTRL with a broad family of regularizers and a new learning rate schedule. For some regularizers, our regret bounds also improve upon prior results even when uniqueness holds. We further provide an application of our results to the decoupled exploration and exploitation problem, demonstrating that our techniques are broadly applicable.

----

## [0] AiluRus: A Scalable ViT Framework for Dense Prediction

**Authors**: *Jin Li, Yaoming Wang, Xiaopeng Zhang, Bowen Shi, Dongsheng Jiang, Chenglin Li, Wenrui Dai, Hongkai Xiong, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html)

**Abstract**:

Vision transformers (ViTs) have emerged as a prevalent architecture for vision tasks owing to their impressive performance. However, their complexity dramatically increases when handling long token sequences, particularly for dense prediction tasks that require high-resolution input. Notably, dense prediction tasks, such as semantic segmentation or object detection, emphasize more on the contours or shapes of objects, while the texture inside objects is less informative. Motivated by this observation, we propose to apply adaptive resolution for different regions in the image according to their importance. Specifically, at the intermediate layer of the ViT, we select anchors from the token sequence using the proposed spatial-aware density-based clustering algorithm. Tokens that are adjacent to anchors are merged to form low-resolution regions, while others are preserved independently as high-resolution. This strategy could significantly reduce the number of tokens, and the following layers only handle the reduced token sequence for acceleration. At the output end, the resolution of the feature map is recovered by unfolding merged tokens for task prediction. Consequently, we can considerably accelerate ViTs for dense prediction tasks. The proposed method is evaluated across three different datasets and demonstrates promising performance. For instance, "Segmenter ViT-L" can be accelerated by 48\% FPS without fine-tuning, while maintaining the performance. Moreover, our method can also be applied to accelerate fine-tuning. Experiments indicate that we can save 52\% training time while accelerating 2.46$\times$ FPS with only a 0.09\% performance drop.

----

## [0] CORL: Research-oriented Deep Offline Reinforcement Learning Library

**Authors**: *Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, Sergey Kolesnikov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62d2cec62b7fd46dd35fa8f2d4aeb52d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/62d2cec62b7fd46dd35fa8f2d4aeb52d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

CORL is an open-source library that provides thoroughly benchmarked single-file implementations of both deep offline and offline-to-online reinforcement learning algorithms. It emphasizes a simple developing experience with a straightforward codebase and a modern analysis tracking tool. In CORL, we isolate methods implementation into separate single files, making performance-relevant details easier to recognize. Additionally, an experiment tracking feature is available to help log metrics, hyperparameters, dependencies, and more to the cloud. Finally, we have ensured the reliability of the implementations by benchmarking commonly employed D4RL datasets providing a transparent source of results that can be reused for robust evaluation tools such as performance profiles, probability of improvement, or expected online performance.

----

## [0] LightSpeed: Light and Fast Neural Light Fields on Mobile Devices

**Authors**: *Aarush Gupta, Junli Cao, Chaoyang Wang, Ju Hu, Sergey Tulyakov, Jian Ren, László A. Jeni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html)

**Abstract**:

Real-time novel-view image synthesis on mobile devices is prohibitive due to the limited computational power and storage. Using volumetric rendering methods, such as NeRF and its derivatives, on mobile devices is not suitable due to the high computational cost of volumetric rendering. On the other hand, recent advances in neural light field representations have shown promising real-time view synthesis results on mobile devices. Neural light field methods learn a direct mapping from a ray representation to the pixel color. The current choice of ray representation is either stratified ray sampling or Plücker coordinates, overlooking the classic light slab (two-plane) representation, the preferred representation to interpolate between light field views. In this work, we find that using the light slab representation is an efficient representation for learning a neural light field. More importantly, it is a lower-dimensional ray representation enabling us to learn the 4D ray space using feature grids which are significantly faster to train and render. Although mostly designed for frontal views, we show that the light-slab representation can be further extended to non-frontal scenes using a divide-and-conquer strategy. Our method provides better rendering quality than prior light field methods and a significantly better trade-off between rendering quality and speed than prior light field methods.

----

## [0] CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models

**Authors**: *Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631bb9434d718ea309af82566347d607-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631bb9434d718ea309af82566347d607-Abstract-Conference.html)

**Abstract**:

The ability to perform causal reasoning is widely considered a core feature of intelligence. In this work, we investigate whether large language models (LLMs) can coherently reason about causality. Much of the existing work in natural language processing (NLP) focuses on evaluating commonsense causal reasoning in LLMs, thus failing to assess whether a model can perform causal inference in accordance with a set of well-defined formal rules. To address this, we propose a new NLP task, causal inference in natural language, inspired by the "causal inference engine" postulated by Judea Pearl et al. We compose a large dataset, CLadder, with 10K samples: based on a collection of causal graphs and queries (associational, interventional, and counterfactual), we obtain symbolic questions and ground-truth answers, through an oracle causal inference engine. These are then translated into natural language. We evaluate multiple LLMs on our dataset, and we introduce and evaluate a bespoke chain-of-thought prompting strategy, CausalCoT. We show that our task is highly challenging for LLMs, and we conduct an in-depth analysis to gain deeper insight into the causal reasoning abilities of LLMs. Our data is open-sourced at https://huggingface.co/datasets/causalNLP/cladder, and our code can be found at https://github.com/causalNLP/cladder.

----

## [0] Riemannian Laplace approximations for Bayesian neural networks

**Authors**: *Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, Georgios Arvanitidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html)

**Abstract**:

Bayesian neural networks often approximate the weight-posterior with a Gaussian distribution. However, practical posteriors are often, even locally, highly non-Gaussian, and empirical performance deteriorates. We propose a simple parametric approximate posterior that adapts to the shape of the true posterior through a Riemannian metric that is determined by the log-posterior gradient. We develop a Riemannian Laplace approximation where samples naturally fall into weight-regions with low negative log-posterior. We show that these samples can be drawn by solving a system of ordinary differential equations, which can be done efficiently by leveraging the structure of the Riemannian metric and automatic differentiation. Empirically, we demonstrate that our approach consistently improves over the conventional Laplace approximation across tasks. We further show that, unlike the conventional Laplace approximation, our method is not overly sensitive to the choice of prior, which alleviates a practical pitfall of current approaches.

----

## [0] SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality

**Authors**: *Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the last year alone, a surge of new benchmarks to measure $\textit{compositional}$ understanding of vision-language models have permeated the machine learning ecosystem.Given an image, these benchmarks probe a model's ability to identify its associated caption amongst a set of compositional distractors.Surprisingly, we find significant biases in $\textit{all}$ these benchmarks rendering them hackable. This hackability is so dire that blind models with no access to the image outperform state-of-the-art vision-language models.To remedy this rampant vulnerability, we introduce $\textit{SugarCrepe}$, a new benchmark for vision-language compositionality evaluation.We employ large language models, instead of rule-based templates used in previous benchmarks, to generate fluent and sensical hard negatives, and utilize an adversarial refinement mechanism to maximally reduce biases. We re-evaluate state-of-the-art models and recently proposed compositionality inducing strategies, and find that their improvements were hugely overestimated, suggesting that more innovation is needed in this important direction.We release $\textit{SugarCrepe}$ and the code for evaluation at: https://github.com/RAIVNLab/sugar-crepe.

----

## [0] Contrastive Retrospection: honing in on critical steps for rapid learning and generalization in RL

**Authors**: *Chen Sun, Wannan Yang, Thomas Jiralerspong, Dane Malenfant, Benjamin Alsbury-Nealy, Yoshua Bengio, Blake A. Richards*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6357d6d068622c962391081d296bed69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6357d6d068622c962391081d296bed69-Abstract-Conference.html)

**Abstract**:

In real life, success is often contingent upon multiple critical steps that are distant in time from each other and from the final reward. These critical steps are challenging to identify with traditional reinforcement learning (RL) methods that rely on the Bellman equation for credit assignment. Here, we present a new RL algorithm that uses offline contrastive learning to hone in on these critical steps. This algorithm, which we call Contrastive Retrospection (ConSpec), can be added to any existing RL algorithm. ConSpec learns a set of prototypes for the critical steps in a task by a novel contrastive loss and delivers an intrinsic reward when the current state matches one of the prototypes. The prototypes in ConSpec provide two key benefits for credit assignment: (i) They enable rapid identification of all the critical steps. (ii) They do so in a readily interpretable manner, enabling out-of-distribution generalization when sensory features are altered. Distinct from other contemporary RL approaches to credit assignment, ConSpec takes advantage of the fact that it is easier to retrospectively identify the small set of steps that success is contingent upon (and ignoring other states) than it is to prospectively predict reward at every taken step. ConSpec greatly improves learning in a diverse set of RL tasks. The code is available at the link: https://github.com/sunchipsster1/ConSpec

----

## [0] A Hierarchical Training Paradigm for Antibody Structure-sequence Co-design

**Authors**: *Fang Wu, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/636d57c09a5baacd83722639265802f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/636d57c09a5baacd83722639265802f6-Abstract-Conference.html)

**Abstract**:

Therapeutic antibodies are an essential and rapidly flourishing drug modality. The binding specificity between antibodies and antigens is decided by complementarity-determining regions (CDRs) at the tips of these Y-shaped proteins. In this paper, we propose a \textbf{h}ierarchical \textbf{t}raining \textbf{p}aradigm (HTP) for the antibody sequence-structure co-design. HTP consists of four levels of training stages, each corresponding to a specific protein modality within a particular protein domain. Through carefully crafted tasks in different stages, HTP seamlessly and effectively integrates geometric graph neural networks (GNNs) with large-scale protein language models to excavate evolutionary information from not only geometric structures but also vast antibody and non-antibody sequence databases, which determines ligand binding pose and strength. Empirical experiments show HTP sets the new state-of-the-art performance in the co-design problem as well as the fix-backbone design. Our research offers a hopeful path to unleash the potential of deep generative architectures and seeks to illuminate the way forward for the antibody sequence and structure co-design challenge.

----

## [0] Differentiable Clustering with Perturbed Spanning Forests

**Authors**: *Lawrence Stewart, Francis R. Bach, Felipe Llinares-López, Quentin Berthet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html)

**Abstract**:

We introduce a differentiable clustering method based on stochastic perturbations of minimum-weight spanning forests. This allows us to include clustering in end-to-end trainable pipelines, with efficient gradients. We show that our method performs well even in difficult settings, such as data sets with high noise and challenging geometries. We also formulate an ad hoc loss to efficiently learn from partial clustering data using this operation. We demonstrate its performance on several data sets for supervised and semi-supervised tasks.

----

## [0] Logarithmic-Regret Quantum Learning Algorithms for Zero-Sum Games

**Authors**: *Minbo Gao, Zhengfeng Ji, Tongyang Li, Qisheng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/637df18481a6aa74238bd2cafff94cb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/637df18481a6aa74238bd2cafff94cb9-Abstract-Conference.html)

**Abstract**:

We propose the first online quantum algorithm for zero-sum games with $\widetilde O(1)$ regret under the game setting. Moreover, our quantum algorithm computes an $\varepsilon$-approximate Nash equilibrium of an $m \times n$ matrix zero-sum game in quantum time $\widetilde O(\sqrt{m+n}/\varepsilon^{2.5})$. Our algorithm uses standard quantum inputs and generates classical outputs with succinct descriptions, facilitating end-to-end applications. Technically, our online quantum algorithm "quantizes" classical algorithms based on the optimistic multiplicative weight update method. At the heart of our algorithm is a fast quantum multi-sampling procedure for the Gibbs sampling problem, which may be of independent interest.

----

## [0] Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network

**Authors**: *Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, Yoshua Bengio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)

**Abstract**:

Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized by neural networks. We show that our method, called JSP-GFN, offers an accurate approximation of the joint posterior, while comparing favorably against existing methods on both simulated and real data.

----

## [0] DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

**Authors**: *Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, Bo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Generative Pre-trained Transformer (GPT) models have exhibited exciting progress in capabilities, capturing the interest of practitioners and the public alike. Yet, while the literature on the trustworthiness of GPT models remains limited, practitioners have proposed employing capable GPT models for sensitive applications to healthcare and finance – where mistakes can be costly. To this end, this work proposes a comprehensive trustworthiness evaluation for large language models with a focus on GPT-4 and GPT-3.5, considering diverse perspectives – including toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, robustness on adversarial demonstrations, privacy, machine ethics, and fairness. Based on our evaluations, we discover previously unpublished vulnerabilities to trustworthiness threats. For instance, we find that GPT models can be easily misled to generate toxic and biased outputs and leak private information in both training data and conversation history. We also find that although GPT-4 is usually more trustworthy than GPT-3.5 on standard benchmarks, GPT-4 is more vulnerable given jailbreaking system or user prompts, potentially due to the reason that GPT-4 follows the (misleading) instructions more precisely. Our work illustrates a comprehensive trustworthiness evaluation of GPT models and sheds light on the trustworthiness gaps. Our benchmark is publicly available at https://decodingtrust.github.io/.

----

## [0] Three Towers: Flexible Contrastive Learning with Pretrained Image Models

**Authors**: *Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63d4316315900a62e610e5c17bab900a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63d4316315900a62e610e5c17bab900a-Abstract-Conference.html)

**Abstract**:

We introduce Three Towers (3T), a flexible method to improve the contrastive learning of vision-language models by incorporating pretrained image classifiers. While contrastive models are usually trained from scratch, LiT (Zhai et al., 2022) has recently shown performance gains from using pretrained classifier embeddings. However, LiT directly replaces the image tower with the frozen embeddings, excluding any potential benefits from training the image tower contrastively. With 3T, we propose a more flexible strategy that allows the image tower to benefit from both pretrained embeddings and contrastive training. To achieve this, we introduce a third tower that contains the frozen pretrained embeddings, and we encourage alignment between this third tower and the main image-text towers. Empirically, 3T consistently improves over LiT and the CLIP-style from-scratch baseline for retrieval tasks. For classification, 3T reliably improves over the from-scratch baseline, and while it underperforms relative to LiT for JFT-pretrained models, it outperforms LiT for ImageNet-21k and Places365 pretraining.

----

## [0] Practical and Asymptotically Exact Conditional Sampling in Diffusion Models

**Authors**: *Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, John P. Cunningham*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html)

**Abstract**:

Diffusion models have been successful on a range of conditional generation tasks including molecular design and text-to-image generation. However, these achievements have primarily depended on task-specific conditional training or error-prone heuristic approximations. Ideally, a conditional generation method should provide exact samples for a broad range of conditional distributions without requiring task-specific training. To this end, we introduce the Twisted Diffusion Sampler, or TDS. TDS is a sequential Monte Carlo (SMC) algorithm that targets the conditional distributions of diffusion models through simulating a set of weighted particles. The main idea is to use twisting, an SMC technique that enjoys good computational efficiency, to incorporate heuristic approximations without compromising asymptotic exactness. We first find in simulation and in conditional image generation tasks that TDS provides a computational statistical trade-off, yielding more accurate approximations with many particles but with empirical improvements over heuristics with as few as two particles. We then turn to motif-scaffolding, a core task in protein design, using a TDS extension to Riemannian diffusion models; on benchmark tasks, TDS allows flexible conditioning criteria and often outperforms the state-of-the-art, conditionally trained model. Code can be found in https://github.com/blt2114/twisteddiffusionsampler

----

## [0] Actively Testing Your Model While It Learns: Realizing Label-Efficient Learning in Practice

**Authors**: *Dayou Yu, Weishi Shi, Qi Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63ef323523f3be8b58ed9277cc747485-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63ef323523f3be8b58ed9277cc747485-Abstract-Conference.html)

**Abstract**:

In active learning (AL), we focus on reducing the data annotation cost from the model training perspective. However, "testing'', which often refers to the model evaluation process of using empirical risk to estimate the intractable true generalization risk, also requires data annotations. The annotation cost for "testing'' (model evaluation) is under-explored. Even in works that study active model evaluation or active testing (AT), the learning and testing ends are disconnected. In this paper, we propose a novel active testing while learning (ATL) framework that integrates active learning with active testing. ATL provides an unbiased sample-efficient estimation of the model risk during active learning. It leverages test samples annotated from different periods of a dynamic active learning process to achieve fair model evaluations based on a theoretically guaranteed optimal integration of different test samples. Periodic testing also enables effective early-stopping to further save the total annotation cost. ATL further integrates an "active feedback'' mechanism, which is inspired by human learning, where the teacher (active tester) provides immediate guidance given by the prior performance of the student (active learner). Our theoretical result reveals that active feedback maintains the label complexity of the integrated learning-testing objective, while improving the model's generalization capability. We study the realistic setting where we maximize the performance gain from choosing "testing'' samples for feedback without sacrificing the risk estimation accuracy. An agnostic-style analysis and empirical evaluations on real-world datasets demonstrate that the ATL framework can effectively improve the annotation efficiency of both active learning and evaluation tasks.

----

## [0] MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing

**Authors**: *Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, Yu Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Text-guided image editing is widely needed in daily life, ranging from personal use to professional applications such as Photoshop.However, existing methods are either zero-shot or trained on an automatically synthesized dataset, which contains a high volume of noise.Thus, they still require lots of manual tuning to produce desirable outcomes in practice.To address this issue, we introduce MagicBrush, the first large-scale, manually annotated dataset for instruction-guided real image editing that covers diverse scenarios: single-turn, multi-turn, mask-provided, and mask-free editing.MagicBrush comprises over 10K manually annotated triplets (source image, instruction, target image), which supports trainining large-scale text-guided image editing models.We fine-tune InstructPix2Pix on MagicBrush and show that the new model can produce much better images according to human evaluation.We further conduct extensive experiments to evaluate current image editing baselines from multiple dimensions including quantitative, qualitative, and human evaluations.The results reveal the challenging nature of our dataset and the gap between current baselines and real-world editing needs.

----

## [0] Causal Interpretation of Self-Attention in Pre-Trained Transformers

**Authors**: *Raanan Y. Rohekar, Yaniv Gurwicz, Shami Nisimov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/642a321fba8a0f03765318e629cb93ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/642a321fba8a0f03765318e629cb93ea-Abstract-Conference.html)

**Abstract**:

We propose a causal interpretation of self-attention in the Transformer neural network architecture. We interpret self-attention as a mechanism that estimates a structural equation model for a given input sequence of symbols (tokens). The structural equation model can be interpreted, in turn, as a causal structure over the input symbols under the specific context of the input sequence. Importantly, this interpretation remains valid in the presence of latent confounders. Following this interpretation, we estimate conditional independence relations between input symbols by calculating partial correlations between their corresponding representations in the deepest attention layer. This enables learning the causal structure over an input sequence using existing constraint-based algorithms. In this sense, existing pre-trained Transformers can be utilized for zero-shot causal-discovery. We demonstrate this method by providing causal explanations for the outcomes of Transformers in two tasks: sentiment classification (NLP) and recommendation.

----

## [0] Parsel🦆: Algorithmic Reasoning with Language Models by Composing Decompositions

**Authors**: *Eric Zelikman, Qian Huang, Gabriel Poesia, Noah D. Goodman, Nick Haber*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html)

**Abstract**:

Despite recent success in large language model (LLM) reasoning, LLMs struggle with hierarchical multi-step reasoning tasks like generating complex programs. For these tasks, humans often start with a high-level algorithmic design and implement each part gradually. We introduce Parsel, a framework enabling automatic implementation and validation of complex algorithms with code LLMs. With Parsel, we automatically decompose algorithmic tasks into hierarchical natural language function descriptions and then search over combinations of possible function implementations using tests. We show that Parsel can be used across domains requiring hierarchical reasoning, including program synthesis and robotic planning. We find that, using Parsel, LLMs solve more competition-level problems in the APPS dataset, resulting in pass rates over 75\% higher than prior results from directly sampling AlphaCode and Codex, while often using a smaller sample budget. Moreover, with automatically generated tests, we find that Parsel can improve the state-of-the-art pass@1 performance on HumanEval from 67\% to 85\%. We also find that LLM-generated robotic plans using Parsel are more than twice as likely to be considered accurate than directly generated plans. Lastly, we explore how Parsel addresses LLM limitations and discuss how Parsel may be useful for human programmers. We release our code at https://github.com/ezelikman/parsel.

----

## [0] Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation

**Authors**: *Yuan Wang, Naisong Luo, Tianzhu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6447714b83edcbed61dbe10371dd7ae5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6447714b83edcbed61dbe10371dd7ae5-Abstract-Conference.html)

**Abstract**:

Few-shot segmentation (FSS) aims to segment objects of new categories given only a handful of annotated samples. Previous works focus their efforts on exploring the support information while paying less attention to the mining of the critical query branch. In this paper, we rethink the importance of support information and propose a new query-centric FSS model Adversarial Mining Transformer (AMFormer), which achieves accurate query image segmentation with only rough support guidance or even weak support labels. The proposed AMFormer enjoys several merits. First, we design an object mining transformer (G) that can achieve the expansion of incomplete region activated by support clue, and a detail mining transformer (D) to discriminate the detailed local difference between the expanded mask and the ground truth. Second, we propose to train G and D via an adversarial process, where G is optimized to generate more accurate masks approaching ground truth to fool D. We conduct extensive experiments on commonly used Pascal-5i and COCO-20i benchmarks and achieve state-of-the-art results across all settings. In addition, the decent performance with weak support labels in our query-centric paradigm may inspire the development of more general FSS models.

----

## [0] Improving Language Plasticity via Pretraining with Active Forgetting

**Authors**: *Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Ifeoluwa Adelani, Pontus Lars Erik Saito Stenetorp, Sebastian Riedel, Mikel Artetxe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html)

**Abstract**:

Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a barrier to making their capabilities universally accessible. While prior work has shown it possible to address this issue by learning a new embedding layer for the new language, doing so is both data and compute inefficient. We propose to use an active forgetting mechanism during pretraining, as a simple way of creating PLMs that can quickly adapt to new languages. Concretely, by resetting the embedding layer every K updates during pretraining, we encourage the PLM to improve its ability of learning new embeddings within limited number of updates, similar to a meta-learning effect. Experiments with RoBERTa show that models pretrained with our forgetting mechanism not only demonstrate faster convergence during language adaptation, but also outperform standard ones in a low-data regime, particularly for languages that are distant from English. Code will be available at https://github.com/facebookresearch/language-model-plasticity.

----

## [0] Stability of Random Forests and Coverage of Random-Forest Prediction Intervals

**Authors**: *Yan Wang, Huaiqing Wu, Dan Nettleton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6452474601429509f3035dc81c233226-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6452474601429509f3035dc81c233226-Abstract-Conference.html)

**Abstract**:

We establish stability of random forests under the mild condition that the squared response ($Y^2$) does not have a heavy tail. In particular, our analysis holds for the practical version of random forests that is implemented in popular packages like \texttt{randomForest} in \texttt{R}. Empirical results show that stability may persist even beyond our assumption and hold for heavy-tailed $Y^2$. Using the stability property, we prove a non-asymptotic lower bound for the coverage probability of prediction intervals constructed from the out-of-bag error of random forests. With another mild condition that is typically satisfied when $Y$ is continuous, we also establish a complementary upper bound, which can be similarly established for the jackknife prediction interval constructed from an arbitrary stable algorithm. We also discuss the asymptotic coverage probability under assumptions weaker than those considered in previous literature. Our work implies that random forests, with its stability property, is an effective machine learning method that can provide not only satisfactory point prediction but also justified interval prediction at almost no extra computational cost.

----

## [0] Multi-Fidelity Multi-Armed Bandits Revisited

**Authors**: *Xuchuang Wang, Qingyun Wu, Wei Chen, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64602b87c31db70a3ef060f6c5d5b01d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64602b87c31db70a3ef060f6c5d5b01d-Abstract-Conference.html)

**Abstract**:

We study the multi-fidelity multi-armed bandit ($\texttt{MF-MAB}$), an extension of the canonical multi-armed bandit (MAB) problem.$\texttt{MF-MAB}$ allows each arm to be pulled with different costs (fidelities) and observation accuracy.We study both the best arm identification with fixed confidence ($\texttt{BAI}$) and the regret minimization objectives.For $\texttt{BAI}$, we present (a) a cost complexity lower bound, (b) an algorithmic framework with two alternative fidelity selection procedures,and (c) both procedures' cost complexity upper bounds.From both cost complexity bounds of $\texttt{MF-MAB}$,one can recover the standard sample complexity bounds of the classic (single-fidelity) MAB.For regret minimization of $\texttt{MF-MAB}$, we propose a new regret definition, prove its problem-independent regret lower bound $\Omega(K^{1/3}\Lambda^{2/3})$ and problem-dependent lower bound $\Omega(K\log \Lambda)$, where $K$ is the number of arms and $\Lambda$ is the decision budget in terms of cost, and devise an elimination-based algorithm whose worst-cost regret upper bound matches its corresponding lower bound up to some logarithmic terms and, whose problem-dependent bound matches its corresponding lower bound in terms of $\Lambda$.

----

## [0] Augmentation-Aware Self-Supervision for Data-Efficient GAN Training

**Authors**: *Liang Hou, Qi Cao, Yige Yuan, Songtao Zhao, Chongyang Ma, Siyuan Pan, Pengfei Wan, Zhongyuan Wang, Huawei Shen, Xueqi Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6464638c2472e4cae607f0c96a6fe774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6464638c2472e4cae607f0c96a6fe774-Abstract-Conference.html)

**Abstract**:

Training generative adversarial networks (GANs) with limited data is challenging because the discriminator is prone to overfitting. Previously proposed differentiable augmentation demonstrates improved data efficiency of training GANs. However, the augmentation implicitly introduces undesired invariance to augmentation for the discriminator since it ignores the change of semantics in the label space caused by data transformation, which may limit the representation learning ability of the discriminator and ultimately affect the generative modeling performance of the generator. To mitigate the negative impact of invariance while inheriting the benefits of data augmentation, we propose a novel augmentation-aware self-supervised discriminator that predicts the augmentation parameter of the augmented data. Particularly, the prediction targets of real data and generated data are required to be distinguished since they are different during training. We further encourage the generator to adversarially learn from the self-supervised discriminator by generating augmentation-predictable real and not fake data. This formulation connects the learning objective of the generator and the arithmetic $-$ harmonic mean divergence under certain assumptions. We compare our method with state-of-the-art (SOTA) methods using the class-conditional BigGAN and unconditional StyleGAN2 architectures on data-limited CIFAR-10, CIFAR-100, FFHQ, LSUN-Cat, and five low-shot datasets. Experimental results demonstrate significant improvements of our method over SOTA methods in training data-efficient GANs.

----

## [0] Template-free Articulated Neural Point Clouds for Reposable View Synthesis

**Authors**: *Lukas Uzolas, Elmar Eisemann, Petr Kellnhofer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html)

**Abstract**:

Dynamic Neural Radiance Fields (NeRFs) achieve remarkable visual quality when synthesizing novel views of time-evolving 3D scenes. However, the common reliance on backward deformation fields makes reanimation of the captured object poses challenging. Moreover, the state of the art dynamic models are often limited by low visual fidelity, long reconstruction time or specificity to narrow application domains.Â  In this paper, we present a novel method utilizing a point-based representation and Linear Blend Skinning (LBS) to jointly learn a Dynamic NeRF and an associated skeletal model from even sparse multi-view video. Our forward-warping approach achieves state-of-the-art visual fidelity when synthesizing novel views and poses while significantly reducing the necessary learning time when compared to existing work. We demonstrate the versatility of our representation on a variety of articulated objects from common datasets and obtain reposable 3D reconstructions without the need of object-specific skeletal templates.

----

## [0] Demystifying the Optimal Performance of Multi-Class Classification

**Authors**: *Minoh Jeong, Martina Cardone, Alex Dytso*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/647e122fc406573c51276692f20379b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/647e122fc406573c51276692f20379b5-Abstract-Conference.html)

**Abstract**:

Classification is a fundamental task in science and engineering on which machine learning methods have shown outstanding performances. However, it is challenging to determine whether such methods have achieved the Bayes error rate, that is, the lowest error rate attained by any classifier. This is mainly due to the fact that the Bayes error rate is not known in general and hence, effectively estimating it is paramount. Inspired by the work by Ishida et al. (2023), we propose an estimator for the Bayes error rate of supervised multi-class classification problems. We analyze several theoretical aspects of such estimator, including its consistency, unbiasedness, convergence rate, variance, and robustness. We also propose a denoising method that reduces the noise that potentially corrupts the data labels, and we improve the robustness of the proposed estimator to outliers by incorporating the median-of-means estimator. Our analysis demonstrates the consistency, asymptotic unbiasedness, convergence rate, and robustness of the proposed estimators. Finally, we validate the effectiveness of our theoretical results via experiments both on synthetic data under various noise settings and on real data.

----

## [0] HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models

**Authors**: *Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, Chng Eng Siong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Advancements in deep neural networks have allowed automatic speech recognition (ASR) systems to attain human parity on several publicly available clean speech datasets. However, even state-of-the-art ASR systems experience performance degradation when confronted with adverse conditions, as a well-trained acoustic model is sensitive to variations in the speech domain, e.g., background noise. Intuitively, humans address this issue by relying on their linguistic knowledge: the meaning of ambiguous spoken terms is usually inferred from contextual cues thereby reducing the dependency on the auditory system. Inspired by this observation, we introduce the first open-source benchmark to utilize external large language models (LLMs) for ASR error correction, where N-best decoding hypotheses provide informative elements for true transcription prediction. This approach is a paradigm shift from the traditional language model rescoring strategy that can only select one candidate hypothesis as output transcription. The proposed benchmark contains a novel dataset, "HyPoradise" (HP), encompassing more than 316,000 pairs of N-best hypotheses and corresponding accurate transcriptions across prevalent speech domains. Given this dataset, we examine three types of error correction techniques based on LLMs with varying amounts of labeled hypotheses-transcription pairs, which gains significant word error rate (WER) reduction. Experimental evidence demonstrates the proposed technique achieves a breakthrough by surpassing the upper bound of traditional re-ranking based methods. More surprisingly, LLM with reasonable prompt design can even correct those tokens that are missing in N-best list. We make our results publicly accessible for reproducible pipelines with released pre-trained models, thus providing a new paradigm for ASR error correction with LLMs.

----

## [0] Structured Voronoi Sampling

**Authors**: *Afra Amini, Li Du, Ryan Cotterell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64ae05e3f1a88ebac7f9263b69f4e702-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64ae05e3f1a88ebac7f9263b69f4e702-Abstract-Conference.html)

**Abstract**:

Gradient-based sampling algorithms have demonstrated their effectiveness in text generation, especially in the context of controlled text generation. However, there exists a lack of theoretically grounded and principled approaches for this task. In this paper, we take an important step toward building a principled approach for sampling from language models with gradient-based methods. We use discrete distributions given by language models to define densities and develop an algorithm based on Hamiltonian Monte Carlo to sample from them. We name our gradient-based technique Structured Voronoi Sampling (SVS). In an experimental setup where the reference distribution is known, we show that the empirical distribution of SVS samples is closer to the reference distribution compared to alternative sampling schemes. Furthermore, in a controlled generation task, SVS is able to generate fluent and diverse samples while following the control targets significantly better than other methods.

----

## [0] Stability and Generalization of the Decentralized Stochastic Gradient Descent Ascent Algorithm

**Authors**: *Miaoxi Zhu, Li Shen, Bo Du, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html)

**Abstract**:

The growing size of available data has attracted increasing interest in solving minimax problems in a decentralized manner for various machine learning tasks. Previous theoretical research has primarily focused on the convergence rate and communication complexity of decentralized minimax algorithms, with little attention given to their generalization. In this paper, we investigate the primal-dual generalization bound of the decentralized stochastic gradient descent ascent (D-SGDA) algorithm using the approach of algorithmic stability under both convex-concave and nonconvex-nonconcave settings. Our theory refines the algorithmic stability in a decentralized manner and demonstrates that the decentralized structure does not destroy the stability and generalization of D-SGDA, implying that it can generalize as well as the vanilla SGDA in certain situations. Our results analyze the impact of different topologies on the generalization bound of the D-SGDA algorithm beyond trivial factors such as sample sizes, learning rates, and iterations. We also evaluate the optimization error and balance it with the generalization gap to obtain the optimal population risk of D-SGDA in the convex-concave setting. Additionally, we perform several numerical experiments which validate our theoretical findings.

----

## [0] Hierarchical clustering with dot products recovers hidden tree structure

**Authors**: *Annie Gray, Alexander Modell, Patrick Rubin-Delanchy, Nick Whiteley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6521937507d78f327cd402401be73bf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6521937507d78f327cd402401be73bf2-Abstract-Conference.html)

**Abstract**:

In this paper we offer a new perspective on the well established agglomerative clustering algorithm, focusing on recovery of hierarchical structure. We recommend a simple variant of the standard algorithm, in which clusters are merged by maximum average dot product and not, for example, by minimum distance or within-cluster variance. We demonstrate that the tree output by this algorithm provides a bona fide estimate of generative hierarchical structure in data, under a generic probabilistic graphical model. The key technical innovations are to understand how hierarchical information in this model translates into tree geometry which can be recovered from data, and to characterise the benefits of simultaneously growing sample size and data dimension. We demonstrate superior tree recovery performance with real data over existing approaches such as UPGMA, Ward's method, and HDBSCAN.

----

## [0] Latent Field Discovery in Interacting Dynamical Systems with Neural Fields

**Authors**: *Miltiadis Kofinas, Erik J. Bekkers, Naveen Shankar Nagaraja, Efstratios Gavves*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6521bd47ebaa28228cd6c74cb85afb65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6521bd47ebaa28228cd6c74cb85afb65-Abstract-Conference.html)

**Abstract**:

Systems of interacting objects often evolve under the influence of underlying field effects that govern their dynamics, yet previous works have abstracted away from such effects, and assume that systems evolve in a vacuum. In this work, we focus on discovering these fields, and infer them from the observed dynamics alone, without directly observing them. We theorize the presence of latent force fields, and propose neural fields to learn them. Since the observed dynamics constitute the net effect of local object interactions and global field effects, recently popularized equivariant networks are inapplicable, as they fail to capture global information. To address this, we propose to disentangle local object interactions --which are SE(3) equivariant and depend on relative states-- from external global field effects --which depend on absolute states. We model the interactions with equivariant graph networks, and combine them with neural fields in a novel graph network that integrates field forces. Our experiments show that we can accurately discover the underlying fields in charged particles settings, traffic scenes, and gravitational n-body problems, and effectively use them to learn the system and forecast future trajectories.

----

## [0] Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration

**Authors**: *Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, Younggyo Seo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6530db249c161fe9254db2667453952c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6530db249c161fe9254db2667453952c-Abstract-Conference.html)

**Abstract**:

A promising technique for exploration is to maximize the entropy of visited state distribution, i.e., state entropy, by encouraging uniform coverage of visited state space. While it has been effective for an unsupervised setup, it tends to struggle in a supervised setup with a task reward, where an agent prefers to visit high-value states to exploit the task reward. Such a preference can cause an imbalance between the distributions of high-value states and low-value states, which biases exploration towards low-value state regions as a result of the state entropy increasing when the distribution becomes more uniform. This issue is exacerbated when high-value states are narrowly distributed within the state space, making it difficult for the agent to complete the tasks. In this paper, we present a novel exploration technique that maximizes the value-conditional state entropy, which separately estimates the state entropies that are conditioned on the value estimates of each state, then maximizes their average. By only considering the visited states with similar value estimates for computing the intrinsic bonus, our method prevents the distribution of low-value states from affecting exploration around high-value states, and vice versa. We demonstrate that the proposed alternative to the state entropy baseline significantly accelerates various reinforcement learning algorithms across a variety of tasks within MiniGrid, DeepMind Control Suite, and Meta-World benchmarks. Source code is available at https://sites.google.com/view/rl-vcse.

----

## [0] Learning World Models with Identifiable Factorization

**Authors**: *Yuren Liu, Biwei Huang, Zhengmao Zhu, Hong-Long Tian, Mingming Gong, Yang Yu, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65496a4902252d301cdf219339bfbf9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65496a4902252d301cdf219339bfbf9e-Abstract-Conference.html)

**Abstract**:

Extracting a stable and compact representation of the environment is crucial for efficient reinforcement learning in high-dimensional, noisy, and non-stationary environments.  Different categories of information coexist in such environments -- how to effectively extract and disentangle the information remains a challenging problem. In this paper, we propose IFactor, a general framework to model four distinct categories of latent state variables that capture various aspects of information within the RL system, based on their interactions with actions and rewards. Our analysis establishes block-wise identifiability of these latent variables, which not only provides a stable and compact representation but also discloses that all reward-relevant factors are significant for policy learning. We further present a practical approach to learning the world model with identifiable blocks, ensuring the removal of redundancies but retaining minimal and sufficient information for policy optimization. Experiments in synthetic worlds demonstrate that our method accurately identifies the ground-truth latent variables, substantiating our theoretical findings. Moreover, experiments in variants of the DeepMind Control Suite and RoboDesk showcase the superior performance of our approach over baselines.

----

## [0] Online Map Vectorization for Autonomous Driving: A Rasterization Perspective

**Authors**: *Gongjie Zhang, Jiahao Lin, Shuang Wu, Yilin Song, Zhipeng Luo, Yang Xue, Shijian Lu, Zuoguan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/654f61ecd998c9095d30d42c03b832aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/654f61ecd998c9095d30d42c03b832aa-Abstract-Conference.html)

**Abstract**:

High-definition (HD) vectorized map is essential for autonomous driving, providing detailed and precise environmental information for advanced perception and planning. However, current map vectorization methods often exhibit deviations, and the existing evaluation metric for map vectorization lacks sufficient sensitivity to detect these deviations. To address these limitations, we propose integrating the philosophy of rasterization into map vectorization. Specifically, we introduce a new rasterization-based evaluation metric, which has superior sensitivity and is better suited to real-world autonomous driving scenarios. Furthermore, we propose MapVR (Map Vectorization via Rasterization), a novel framework that applies differentiable rasterization to vectorized outputs and then performs precise and geometry-aware supervision on rasterized HD maps. Notably, MapVR designs tailored rasterization strategies for various geometric shapes, enabling effective adaptation to a wide range of map elements. Experiments show that incorporating rasterization into map vectorization greatly enhances performance with no extra computational cost during inference, leading to more accurate map perception and ultimately promoting safer autonomous driving. Codes are available at https://github.com/ZhangGongjie/MapVR. A standalone map vectorization evaluation toolkit is available at https://github.com/jiahaoLjh/MapVectorizationEvalToolkit.

----

## [0] NAP: Neural 3D Articulated Object Prior

**Authors**: *Jiahui Lei, Congyue Deng, William B. Shen, Leonidas J. Guibas, Kostas Daniilidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/655846cc914cb7ff977a1ada40866441-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/655846cc914cb7ff977a1ada40866441-Abstract-Conference.html)

**Abstract**:

We propose Neural 3D Articulated object Prior (NAP), the first 3D deep generative model to synthesize 3D articulated object models. Despite the extensive research on generating 3D static objects, compositions, or scenes, there are hardly any approaches on capturing the distribution of articulated objects, a common object category for human and robot interaction. To generate articulated objects, we first design a novel articulation tree/graph parameterization and then apply a diffusion-denoising probabilistic model over this representation where articulated objects can be generated via denoising from random complete graphs. In order to capture both the geometry and the motion structure whose distribution will affect each other, we design a graph denoising network for learning the reverse diffusion process. We propose a novel distance that adapts widely used 3D generation metrics to our novel task to evaluate generation quality. Experiments demonstrate our high performance in articulated object generation as well as its applications on conditioned generation, including Part2Motion, PartNet-Imagination, Motion2Part, and GAPart2Object.

----

## [0] Compressed Video Prompt Tuning

**Authors**: *Bing Li, Jiaxin Chen, Xiuguo Bao, Di Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/656678aa961a99a6a3d59bfbf88daf77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/656678aa961a99a6a3d59bfbf88daf77-Abstract-Conference.html)

**Abstract**:

Compressed videos offer a compelling alternative to raw videos, showing the possibility to significantly reduce the on-line computational and storage cost. However, current approaches to compressed video processing generally follow the resource-consuming pre-training and fine-tuning paradigm, which does not fully take advantage of such properties, making them not favorable enough for widespread applications. Inspired by recent successes of prompt tuning techniques in computer vision, this paper presents the first attempt to build a prompt based representation learning framework, which enables effective and efficient adaptation of pre-trained raw video models to compressed video understanding tasks. To this end, we propose a novel prompt tuning approach, namely Compressed Video Prompt Tuning (CVPT), emphatically dealing with the challenging issue caused by the inconsistency between pre-training and downstream data modalities. Specifically, CVPT replaces the learnable prompts with compressed modalities (\emph{e.g.} Motion Vectors and Residuals) by re-parameterizing them into conditional prompts followed by layer-wise refinement. The conditional prompts exhibit improved adaptability and generalizability to instances compared to conventional individual learnable ones, and the Residual prompts enhance the noisy motion cues in the Motion Vector prompts for further fusion with the visual cues from I-frames. Additionally, we design Selective Cross-modal Complementary Prompt (SCCP) blocks. After inserting them into the backbone, SCCP blocks leverage semantic relations across diverse levels and modalities to improve cross-modal interactions between prompts and input flows. Extensive evaluations on HMDB-51, UCF-101 and Something-Something v2 demonstrate that CVPT remarkably outperforms the state-of-the-art counterparts, delivering a much better balance between accuracy and efficiency.

----

## [0] Sampling from Structured Log-Concave Distributions via a Soft-Threshold Dikin Walk

**Authors**: *Oren Mangoubi, Nisheeth K. Vishnoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/656faa09eb6e82dd86de9a417111c3b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/656faa09eb6e82dd86de9a417111c3b0-Abstract-Conference.html)

**Abstract**:

Given a Lipschitz or smooth convex function $f:K \to \mathbb{R}^d$ for a bounded polytope $K:=${ $\theta \in \mathbb{R}^d: A\theta \leq b$}, where $A\in  \mathbb{R}^{m\times d}$ and $b \in \mathbb{R}^m$, we consider the problem of sampling from the log-concave distribution $\pi(\theta) \propto e^{-f(\theta)}$ constrained to $K$. Interest in this problem derives from its applications to  Bayesian inference and differential privacy. We present  a generalization of the Dikin walk  to this setting that requires at most $O((md + d L^2 R^2) \times md^{\omega-1} \log(\frac{w}{\delta}))$ arithmetic operations to sample from $\pi$ within error $\delta>0$ in the total variation distance from a $w$-warm start. Here $L$ is the Lipschitz constant of $f$, $K$ is contained in a ball of radius $R$ and contains a ball of smaller radius $r$, and $\omega \approx 2.37$ is the matrix-multiplication constant. This improves on the running time of prior works for a range of structured settings important for the aforementioned inference and privacy applications. Technically, we depart from previous Dikin walks by adding a soft-threshold regularizer derived from the Lipschitz or smoothness properties of $f$  to a barrier function for $K$ that allows our version of the Dikin walk to propose updates  that have a high Metropolis acceptance ratio for $f$, while at the same time remaining inside the polytope $K$.

----

## [0] Implicit Regularization in Over-Parameterized Support Vector Machine

**Authors**: *Yang Sui, Xin He, Yang Bai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/659e07806dc17bd69d0d9aed47f85e7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/659e07806dc17bd69d0d9aed47f85e7c-Abstract-Conference.html)

**Abstract**:

In this paper, we design a regularization-free algorithm for high-dimensional support vector machines (SVMs) by integrating over-parameterization with Nesterov's smoothing method, and provide theoretical guarantees for the induced implicit regularization phenomenon. In particular, we construct an over-parameterized hinge loss function and estimate the true parameters by leveraging regularization-free gradient descent on this loss function. The utilization of Nesterov's method enhances the computational efficiency of our algorithm, especially in terms of determining the stopping criterion and reducing computational complexity. With appropriate choices of initialization, step size, and smoothness parameter, we demonstrate that unregularized gradient descent achieves a near-oracle statistical convergence rate. Additionally, we verify our theoretical findings through a variety of numerical experiments and compare the proposed method with explicit regularization. Our results illustrate the advantages of employing implicit regularization via gradient descent in conjunction with over-parameterization in sparse SVMs.

----

## [0] Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

**Authors**: *Zirui Zhao, Wee Sun Lee, David Hsu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html)

**Abstract**:

Large-scale task planning is a major challenge.  Recent work exploits large  language models (LLMs) directly as a policy and shows surprisingly  interesting results.  This paper shows that LLMs provide a  commonsense model of the world in addition to a policy that acts on it.  The world model and the policy can be combined in a search  algorithm, such as Monte Carlo Tree Search (MCTS), to scale up task  planning.  In our new LLM-MCTS algorithm, the LLM-induced world model  provides a commonsense prior belief for MCTS to achieve effective reasoning;  the LLM-induced policy acts as a heuristic to guide the search, vastly  improving search efficiency. Experiments show that LLM-MCTS outperforms  both MCTS alone and policies induced by LLMs (GPT2 and GPT3.5) by a wide  margin, for complex, novel tasks.   Further experiments and analyses on multiple tasks --  multiplication, travel planning, object rearrangement --  suggest minimum description length (MDL)  as a general guiding principle: if the  description length of the world model is substantially smaller than that of  the  policy, using LLM as a world model for model-based planning is likely better  than using LLM solely as a policy.

----

## [0] Uncovering Meanings of Embeddings via Partial Orthogonality

**Authors**: *Yibo Jiang, Bryon Aragam, Victor Veitch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html)

**Abstract**:

Machine learning tools often rely on embedding text as vectors of real numbers.In this paper, we study how the semantic structure of language is encoded in the algebraic structure of such embeddings.Specifically, we look at a notion of  "semantic independence" capturing the idea that, e.g., "eggplant" and "tomato" are independent given "vegetable". Although such examples are intuitive, it is difficult to formalize such a notion of semantic independence. The key observation here is that any sensible formalization should obey a set of so-called independence axioms, and thus any algebraic encoding of this structure should also obey these axioms. This leads us naturally to use partial orthogonality as the relevant algebraic structure. We develop theory and methods that allow us to demonstrate that partial orthogonality does indeed capture semantic independence.Complementary to this, we also introduce the concept of independence preserving embeddings where embeddings preserve the conditional independence structures of a distribution, and we prove the existence of such embeddings and approximations to them.

----

## [0] Federated Learning with Bilateral Curation for Partially Class-Disjoint Data

**Authors**: *Ziqing Fan, Ruipeng Zhang, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html)

**Abstract**:

Partially class-disjoint data (PCDD), a common yet under-explored data formation where each client contributes a part of classes (instead of all classes) of samples, severely challenges the performance of federated algorithms. Without full classes, the local objective will contradict the global objective, yielding the angle collapse problem for locally missing classes and the space waste problem for locally existing classes. As far as we know, none of the existing methods can intrinsically mitigate PCDD challenges to achieve holistic improvement in the bilateral views (both global view and local view) of federated learning. To address this dilemma, we are inspired by the strong generalization of simplex Equiangular Tight Frame (ETF) on the imbalanced data, and propose a novel approach called FedGELA where the classifier is globally fixed as a simplex ETF while locally adapted to the personal distributions. Globally, FedGELA provides fair and equal discrimination for all classes and avoids inaccurate updates of the classifier, while locally it utilizes the space of locally missing classes for locally existing classes. We conduct extensive experiments on a range of datasets to demonstrate that our FedGELA achieves promising performance (averaged improvement of 3.9% to FedAvg and 1.5% to best baselines) and provide both local and global convergence guarantees.

----

## [0] Partial Counterfactual Identification of Continuous Outcomes with a Curvature Sensitivity Model

**Authors**: *Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html)

**Abstract**:

Counterfactual inference aims to answer retrospective "what if" questions and thus belongs to the most fine-grained type of inference in Pearl's causality ladder. Existing methods for counterfactual inference with continuous outcomes aim at point identification and thus make strong and unnatural assumptions about the underlying structural causal model. In this paper, we relax these assumptions and aim at partial counterfactual identification of continuous outcomes, i.e., when the counterfactual query resides in an ignorance interval with informative bounds. We prove that, in general, the ignorance interval of the counterfactual queries has non-informative bounds, already when functions of structural causal models are continuously differentiable. As a remedy, we propose a novel sensitivity model called Curvature Sensitivity Model. This allows us to obtain informative bounds by bounding the curvature of level sets of the functions. We further show that existing point counterfactual identification methods are special cases of our Curvature Sensitivity Model when the bound of the curvature is set to zero. We then propose an implementation of our Curvature Sensitivity Model in the form of a novel deep generative model, which we call Augmented Pseudo-Invertible Decoder. Our implementation employs (i) residual normalizing flows with (ii) variational augmentations. We empirically demonstrate the effectiveness of our Augmented Pseudo-Invertible Decoder. To the best of our knowledge, ours is the first partial identification model for Markovian structural causal models with continuous outcomes.

----

## [0] Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies

**Authors**: *Wayne Soo, Vishwa Goudar, Xiao-Jing Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html)

**Abstract**:

Training recurrent neural networks (RNNs) has become a go-to approach for generating and evaluating mechanistic neural hypotheses for cognition. The ease and efficiency of training RNNs with backpropagation through time and the availability of robustly supported deep learning libraries has made RNN modeling more approachable and accessible to neuroscience. Yet, a major technical hindrance remains. Cognitive processes such as working memory and decision making involve neural population dynamics over a long period of time within a behavioral trial and across trials. It is difficult to train RNNs to accomplish tasks where neural representations and dynamics have long temporal dependencies without gating mechanisms such as LSTMs or GRUs which currently lack experimental support and prohibit direct comparison between RNNs and biological neural circuits. We tackled this problem based on the idea of specialized skip-connections through time to support the emergence of task-relevant dynamics, and subsequently reinstitute biological plausibility by reverting to the original architecture. We show that this approach enables RNNs to successfully learn cognitive tasks that prove impractical if not impossible to learn using conventional methods. Over numerous tasks considered here, we achieve less training steps and shorter wall-clock times, particularly in tasks that require learning long-term dependencies via temporal integration over long timescales or maintaining a memory of past events in hidden-states. Our methods expand the range of experimental tasks that biologically plausible RNN models can learn, thereby supporting the development of theory for the emergent neural mechanisms of computations involving long-term dependencies.

----

## [0] Diffusion Probabilistic Models for Structured Node Classification

**Authors**: *Hyosoon Jang, Seonghyun Park, Sangwoo Mo, Sungsoo Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html)

**Abstract**:

This paper studies structured node classification on graphs, where the predictions should consider dependencies between the node labels. In particular, we focus on solving the problem for partially labeled graphs where it is essential to incorporate the information in the known label for predicting the unknown labels. To address this issue, we propose a novel framework leveraging the diffusion probabilistic model for structured node classification (DPM-SNC). At the heart of our framework is the extraordinary capability of DPM-SNC to (a) learn a joint distribution over the labels with an expressive reverse diffusion process and (b) make predictions conditioned on the known labels utilizing manifold-constrained sampling. Since the DPMs lack training algorithms for partially labeled data, we design a novel training algorithm to apply DPMs, maximizing a new variational lower bound. We also theoretically analyze how DPMs benefit node classification by enhancing the expressive power of GNNs based on proposing AGG-WL, which is strictly more powerful than the classic 1-WL test. We extensively verify the superiority of our DPM-SNC in diverse scenarios, which include not only the transductive setting on partially labeled graphs but also the inductive setting and unlabeled graphs.

----

## [0] Non-stationary Experimental Design under Linear Trends

**Authors**: *David Simchi-Levi, Chonghuan Wang, Zeyu Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65e837e76a5308df3d5544aab6196e21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65e837e76a5308df3d5544aab6196e21-Abstract-Conference.html)

**Abstract**:

Experimentation has been critical and increasingly popular  across various domains, such as clinical trials and online platforms, due to its widely recognized benefits. One of the primary objectives of classical experiments is to estimate the average treatment effect (ATE) to inform future decision-making. However, in healthcare and many other settings, treatment effects may be non-stationary, meaning that they can change over time, rendering the traditional experimental design inadequate and the classical static ATE uninformative. In this work, we address the problem of non-stationary experimental design under linear trends by considering two objectives: estimating the dynamic treatment effect and minimizing welfare loss within the experiment. We propose an efficient design that can be customized for optimal estimation error rate, optimal regret rate, or the Pareto optimal trade-off between the two objectives. We establish information-theoretical lower bounds that highlight the inherent challenge in estimating dynamic treatment effects and minimizing welfare loss, and also statistically reveal the fundamental trade-off between them.

----

## [0] EICIL: Joint Excitatory Inhibitory Cycle Iteration Learning for Deep Spiking Neural Networks

**Authors**: *Zihang Shao, Xuanye Fang, Yaxin Li, Chaoran Feng, Jiangrong Shen, Qi Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65e876f6a98c6799d0b3145966dd73e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65e876f6a98c6799d0b3145966dd73e2-Abstract-Conference.html)

**Abstract**:

Spiking neural networks (SNNs) have undergone continuous development and extensive study for decades, leading to increased biological plausibility and optimal energy efficiency. However, traditional training methods for deep SNNs have some limitations, as they rely on strategies such as pre-training and fine-tuning, indirect coding and reconstruction, and approximate gradients. These strategies lack a complete training model and require gradient approximation. To overcome these limitations, we propose a novel learning method named Joint Excitatory Inhibitory Cycle Iteration learning for Deep Spiking Neural Networks (EICIL) that integrates both excitatory and inhibitory behaviors inspired by the signal transmission of biological neurons.By organically embedding these two behavior patterns into one framework, the proposed EICIL significantly improves the bio-mimicry and adaptability of spiking neuron models, as well as expands the representation space of spiking neurons. Extensive experiments based on EICIL and traditional learning methods demonstrate that EICIL outperforms traditional methods on various datasets, such as CIFAR10 and CIFAR100, revealing the crucial role of the learning approach that integrates both behaviors during training.

----

## [0] Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency

**Authors**: *Owen Queen, Tom Hartvigsen, Teddy Koker, Huan He, Theodoros Tsiligkaridis, Marinka Zitnik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65ea878cb90b440e8b4cd34fe0959914-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65ea878cb90b440e8b4cd34fe0959914-Abstract-Conference.html)

**Abstract**:

Interpreting time series models is uniquely challenging because it requires identifying both the location of time series signals that drive model predictions and their matching to an interpretable temporal pattern. While explainers from other modalities can be applied to time series, their inductive biases do not transfer well to the inherently challenging interpretation of time series. We present TimeX, a time series consistency model for training explainers. TimeX trains an interpretable surrogate to mimic the behavior of a pretrained time series model. It addresses the issue of model faithfulness by introducing model behavior consistency, a novel formulation that preserves relations in the latent space induced by the pretrained model with relations in the latent space induced by TimeX. TimeX provides discrete attribution maps and, unlike existing interpretability methods, it learns a latent space of explanations that can be used in various ways, such as to provide landmarks to visually aggregate similar explanations and easily recognize temporal patterns. We evaluate TimeX on eight synthetic and real-world datasets and compare its performance against state-of-the-art interpretability methods. We also conduct case studies using physiological time series. Quantitative evaluations demonstrate that TimeX achieves the highest or second-highest performance in every metric compared to baselines across all datasets. Through case studies, we show that the novel components of TimeX show potential for training faithful, interpretable models that capture the behavior of pretrained time series models.

----

## [0] NeuroEvoBench: Benchmarking Evolutionary Optimizers for Deep Learning Applications

**Authors**: *Robert Tjarko Lange, Yujin Tang, Yingtao Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/660ba7851661638c559df47743c69e40-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/660ba7851661638c559df47743c69e40-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recently, the Deep Learning community has become interested in evolutionary optimization (EO) as a means to address hard optimization problems, e.g. meta-learning through long inner loop unrolls or optimizing non-differentiable operators. One core reason for this trend has been the recent innovation in hardware acceleration and compatible software -- making distributed population evaluations much easier than before. Unlike for gradient descent-based methods though, there is a lack of hyperparameter understanding and best practices for EO â€“ arguably due to severely less `graduate student descent' and benchmarking being performed for EO methods. Additionally, classical benchmarks from the evolutionary community provide few practical insights for Deep Learning applications. This poses challenges for newcomers to hardware-accelerated EO and hinders significant adoption. Hence, we establish a new benchmark of EO methods (NEB) tailored toward Deep Learning applications and exhaustively evaluate traditional and meta-learned EO. We investigate core scientific questions including resource allocation, fitness shaping, normalization, regularization & scalability of EO. The benchmark is open-sourced at https://github.com/neuroevobench/neuroevobench under Apache-2.0 license.

----

## [0] HyTrel: Hypergraph-enhanced Tabular Data Representation Learning

**Authors**: *Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, George Karypis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html)

**Abstract**:

Language models pretrained on large collections of tabular data have demonstrated their effectiveness in several downstream tasks.However, many of these models do not take into account the row/column permutation invariances, hierarchical structure, etc. that exist in tabular data. To alleviate these limitations, we propose HyTrel, a tabular language model, that captures the permutation invariances and three more structural properties of tabular data by using hypergraphs--where the table cells make up the nodes and the cells occurring jointly together in each row, column, and the entire table are used to form three different types of hyperedges. We show thatHyTrel is maximally invariant under certain conditions for tabular data, i.e., two tables obtain the same representations via HyTreliff the two tables are identical up to permutation. Our empirical results demonstrate that HyTrel consistently outperforms other competitive baselines on four downstream tasks with minimal pretraining, illustrating the advantages of incorporating inductive biases associated with tabular data into the representations. Finally, our qualitative analyses showcase that HyTrel can assimilate the table structure to generate robust representations for the cells, rows, columns, and the entire table.

----

## [0] PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance

**Authors**: *Peiqing Yang, Shangchen Zhou, Qingyi Tao, Chen Change Loy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661c37f3b098bdee53fd7d9c4ef6964a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661c37f3b098bdee53fd7d9c4ef6964a-Abstract-Conference.html)

**Abstract**:

Exploiting pre-trained diffusion models for restoration has recently become a favored alternative to the traditional task-specific training approach. Previous works have achieved noteworthy success by limiting the solution space using explicit degradation models. However, these methods often fall short when faced with complex degradations as they generally cannot be precisely modeled. In this paper, we introduce $\textit{partial guidance}$, a fresh perspective that is more adaptable to real-world degradations compared to existing works. Rather than specifically defining the degradation process, our approach models the desired properties, such as image structure and color statistics of high-quality images, and applies this guidance during the reverse diffusion process. These properties are readily available and make no assumptions about the degradation process. When combined with a diffusion prior, this partial guidance can deliver appealing results across a range of restoration tasks. Additionally, our method can be extended to handle composite tasks by consolidating multiple high-quality image properties, achieved by integrating the guidance from respective tasks. Experimental results demonstrate that our method not only outperforms existing diffusion-prior-based approaches but also competes favorably with task-specific models.

----

## [0] Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP

**Authors**: *Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html)

**Abstract**:

Open-vocabulary segmentation is a challenging task requiring segmenting and recognizing objects from an open set of categories in diverse environments. One way to address this challenge is to leverage multi-modal models, such as CLIP, to provide image and text features in a shared embedding space, which effectively bridges the gap between closed-vocabulary and open-vocabulary recognition.Hence, existing methods often adopt a two-stage framework to tackle the problem, where the inputs first go through a mask generator and then through the CLIP model along with the predicted masks. This process involves extracting features from raw images multiple times, which can be ineffective and inefficient. By contrast, we propose to build everything into a single-stage framework using a shared Frozen Convolutional CLIP backbone, which not only significantly simplifies the current two-stage pipeline, but also remarkably yields a better accuracy-cost trade-off. The resulting single-stage system, called FC-CLIP, benefits from the following observations: the frozen CLIP backbone maintains the ability of open-vocabulary classification and can also serve as a strong mask generator, and the convolutional CLIP generalizes well to a larger input resolution than the one used during contrastive image-text pretraining. Surprisingly, FC-CLIP advances state-of-the-art results on various benchmarks, while running practically fast. Specifically, when training on COCO panoptic data only and testing in a zero-shot manner, FC-CLIP achieve 26.8 PQ, 16.8 AP, and 34.1 mIoU on ADE20K, 18.2 PQ, 27.9 mIoU on Mapillary Vistas, 44.0 PQ, 26.8 AP, 56.2 mIoU on Cityscapes, outperforming the prior art under the same setting by +4.2 PQ, +2.4 AP, +4.2 mIoU on ADE20K, +4.0 PQ on Mapillary Vistas and +20.1 PQ on Cityscapes, respectively. Additionally, the training and testing time of FC-CLIP is 7.5x and 6.6x significantly faster than the same prior art, while using 5.9x fewer total model parameters. Meanwhile, FC-CLIP also sets a new state-of-the-art performance across various open-vocabulary semantic segmentation datasets. Code and models are available at https://github.com/bytedance/fc-clip

----

## [0] CLIP-OGD: An Experimental Design for Adaptive Neyman Allocation in Sequential Experiments

**Authors**: *Jessica Dai, Paula Gradu, Christopher Harshaw*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661d4fda173120a2f119e0319e6bcf97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661d4fda173120a2f119e0319e6bcf97-Abstract-Conference.html)

**Abstract**:

From clinical development of cancer therapies to investigations into partisan bias, adaptive sequential designs have become increasingly popular method for causal inference, as they offer the possibility of improved precision over their non-adaptive counterparts. However, even in simple settings (e.g. two treatments) the extent to which adaptive designs can improve precision is not sufficiently well understood. In this work, we study the problem of Adaptive Neyman Allocation in a design-based potential outcomes framework, where the experimenter seeks to construct an adaptive design which is nearly as efficient as the optimal (but infeasible) non-adaptive Neyman design, which has access to all potential outcomes. Motivated by connections to online optimization, we propose Neyman Ratio and Neyman Regret as two (equivalent) performance measures of adaptive designs for this problem. We present Clip-OGD, an adaptive design which achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ expected Neyman regret and thereby recovers the optimal Neyman variance in large samples. Finally, we construct a conservative variance estimator which facilitates the development of asymptotically valid confidence intervals. To complement our theoretical results, we conduct simulations using data from a microeconomic experiment.

----



[Go to the previous page](NIPS-2023-list1300.md)

[Go to the next page](NIPS-2023-list1302.md)

[Go to the catalog section](README.md)