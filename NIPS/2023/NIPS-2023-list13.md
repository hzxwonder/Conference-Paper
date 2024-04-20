## [0] D-CIPHER: Discovery of Closed-form Partial Differential Equations

**Authors**: *Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57c30b677add9aa78e1745f0643104d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57c30b677add9aa78e1745f0643104d0-Abstract-Conference.html)

**Abstract**:

Closed-form differential equations, including partial differential equations and higher-order ordinary differential equations, are one of the most important tools used by scientists to model and better understand natural phenomena. Discovering these equations directly from data is challenging because it requires modeling relationships between various derivatives that are not observed in the data (equation-data mismatch) and it involves searching across a huge space of possible equations. Current approaches make strong assumptions about the form of the equation and thus fail to discover many well-known phenomena. Moreover, many of them resolve the equation-data mismatch by estimating the derivatives, which makes them inadequate for noisy and infrequent observations. To this end, we propose D-CIPHER, which is robust to measurement artifacts and can uncover a new and very general class of differential equations. We further design a novel optimization procedure, CoLLie, to help D-CIPHER search through this class efficiently. Finally, we demonstrate empirically that it can discover many well-known equations that are beyond the capabilities of current methods.

----

## [0] Training neural operators to preserve invariant measures of chaotic attractors

**Authors**: *Ruoxi Jiang, Peter Y. Lu, Elena Orlova, Rebecca Willett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html)

**Abstract**:

Chaotic systems make long-horizon forecasts difficult because small perturbations in initial conditions cause trajectories to diverge at an exponential rate. In this setting, neural operators trained to minimize squared error losses, while capable of accurate short-term forecasts, often fail to reproduce statistical or structural properties of the dynamics over longer time horizons and can yield degenerate results. In this paper, we propose an alternative framework designed to preserve invariant measures of chaotic attractors that characterize the time-invariant statistical properties of the dynamics. Specifically, in the multi-environment setting (where each sample trajectory is governed by slightly different dynamics),  we consider two novel approaches to training with noisy data. First, we propose a loss based on the optimal transport distance between the observed dynamics and the neural operator outputs. This approach requires expert knowledge of the underlying physics to determine what statistical features should be included in the optimal transport loss. Second, we show that a  contrastive learning framework, which does not require any specialized prior knowledge, can preserve statistical properties of the dynamics nearly as well as the optimal transport approach. On a variety of chaotic systems, our method is shown empirically to preserve invariant measures of chaotic attractors.

----

## [0] Certification of Distributional Individual Fairness

**Authors**: *Matthew Wicker, Vihari Piratla, Adrian Weller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57d8ebf4c2f050a6485f370d47656a9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57d8ebf4c2f050a6485f370d47656a9e-Abstract-Conference.html)

**Abstract**:

Providing formal guarantees of algorithmic fairness is of paramount importance to socially responsible deployment of machine learning algorithms. In this work, we study formal guarantees, i.e., certificates, for individual fairness (IF) of neural networks. We start by introducing a novel convex approximation of IF constraints that exponentially decreases the computational cost of providing formal guarantees of local individual fairness. We highlight that prior methods are constrained by their focus on global IF certification and can therefore only scale to models with a few dozen hidden neurons, thus limiting their practical impact. We propose to certify \textit{distributional} individual fairness which ensures that for a given empirical distribution and all distributions within a $\gamma$-Wasserstein ball, the neural network has guaranteed individually fair predictions. Leveraging developments in quasi-convex optimization, we provide novel and efficient certified bounds on distributional individual fairness and show that our method allows us to certify and regularize neural networks that are several orders of magnitude larger than those considered by prior works. Moreover, we study real-world distribution shifts and find our bounds to be a scalable, practical, and sound source of IF guarantees.

----

## [0] Leveraging sparse and shared feature activations for disentangled representation learning

**Authors**: *Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, Francesco Locatello*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57fabaa549352c52d5d312171b16970e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57fabaa549352c52d5d312171b16970e-Abstract-Conference.html)

**Abstract**:

Recovering the latent factors of variation of high dimensional data has so far focused on simple synthetic settings. Mostly building on unsupervised and weakly-supervised objectives, prior work missed out on the positive implications for representation learning on real world data. In this work, we propose to leverage knowledge extracted from a diversified set of supervised tasks to learn a common disentangled representation. Assuming each supervised task only depends on an unknown subset of the factors of variation, we disentangle the feature space of a supervised multi-task model, with features activating sparsely across different tasks and information being shared as appropriate. Importantly, we never directly observe the factors of variations but establish that access to multiple tasks is sufficient for identifiability under sufficiency and minimality assumptions.We validate our approach on six real world distribution shift benchmarks, and different data modalities (images, text), demonstrating how disentangled representations can be transferred to real settings.

----

## [0] Mathematical Capabilities of ChatGPT

**Authors**: *Simon Frieder, Luca Pinchetti, Alexis Chevalier, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, Julius Berner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We investigate the mathematical capabilities of two iterations of ChatGPT (released 9-January-2023 and 30-January-2023) and of GPT-4 by testing them on publicly available datasets, as well as hand-crafted ones, using a novel methodology. In contrast to formal mathematics, where large databases of formal proofs are available (e.g., mathlib, the Lean Mathematical Library), current datasets of natural-language mathematics used to benchmark language models either cover only elementary mathematics or are very small. We address this by publicly releasing two new datasets: GHOSTS and miniGHOSTS. These are the first natural-language datasets curated by working researchers in mathematics that (1) aim to cover graduate-level mathematics, (2) provide a holistic overview of the mathematical capabilities of language models, and (3) distinguish multiple dimensions of mathematical reasoning. These datasets test on 1636 human expert evaluations whether ChatGPT and GPT-4 can be helpful assistants to professional mathematicians by emulating use cases that arise in the daily professional activities of mathematicians. We benchmark the models on a range of fine-grained performance metrics. For advanced mathematics, this is the most detailed evaluation effort to date. We find that ChatGPT and GPT-4 can be used most successfully as mathematical assistants for querying facts, acting as mathematical search engines and knowledge base interfaces. GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty. Contrary to many positive reports in the media about GPT-4 and ChatGPT's exam-solving abilities (a potential case of selection bias), their overall mathematical performance is well below the level of a graduate student. Hence, if you aim to use ChatGPT to pass a graduate-level math exam, you would be better off copying from your average peer!

----

## [0] A Unified Framework for U-Net Design and Analysis

**Authors**: *Christopher Williams, Fabian Falck, George Deligiannidis, Chris C. Holmes, Arnaud Doucet, Saifuddin Syed*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58575be50c9b47902359920a4d5d1ab4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58575be50c9b47902359920a4d5d1ab4-Abstract-Conference.html)

**Abstract**:

U-Nets are a go-to neural architecture across numerous tasks for continuous signals on a square such as images and Partial Differential Equations (PDE), however their design and architecture is understudied. In this paper, we provide a framework for designing and analysing general U-Net architectures. We present theoretical results which characterise the role of the encoder and decoder in a U-Net, their high-resolution scaling limits and their conjugacy to ResNets via preconditioning. We propose Multi-ResNets, U-Nets with a simplified, wavelet-based encoder without learnable parameters. Further, we show how to design novel U-Net architectures which encode function constraints, natural bases, or the geometry of the data. In diffusion models, our framework enables us to identify that high-frequency information is dominated by noise exponentially faster, and show how U-Nets with average pooling exploit this. In our experiments, we demonstrate how Multi-ResNets achieve competitive and often superior performance compared to classical U-Nets in image segmentation, PDE surrogate modelling, and generative modelling with diffusion models. Our U-Net framework paves the way to study the theoretical properties of U-Nets and design natural, scalable neural architectures for a multitude of problems beyond the square.

----

## [0] On the Importance of Feature Separability in Predicting Out-Of-Distribution Error

**Authors**: *Renchunzi Xie, Hongxin Wei, Lei Feng, Yuzhou Cao, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/585e9cf25585612ac27b535457116513-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/585e9cf25585612ac27b535457116513-Abstract-Conference.html)

**Abstract**:

Estimating the generalization performance is practically challenging on out-of-distribution (OOD) data without ground-truth labels. While previous methods emphasize the connection between distribution difference and OOD accuracy, we show that a large domain gap not necessarily leads to a low test accuracy. In this paper, we investigate this problem from the perspective of feature separability empirically and theoretically. Specifically, we propose a dataset-level score based upon feature dispersion to estimate the test accuracy under distribution shift. Our method is inspired by desirable properties of features in representation learning: high inter-class dispersion and high intra-class compactness. Our analysis shows that inter-class dispersion is strongly correlated with the model accuracy, while intra-class compactness does not reflect the generalization performance on OOD data. Extensive experiments demonstrate the superiority of our method in both prediction performance and computational efficiency.

----

## [0] The Transient Nature of Emergent In-Context Learning in Transformers

**Authors**: *Aaditya K. Singh, Stephanie C. Y. Chan, Ted Moskovitz, Erin Grant, Andrew M. Saxe, Felix Hill*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58692a1701314e09cbd7a5f5f3871cc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58692a1701314e09cbd7a5f5f3871cc9-Abstract-Conference.html)

**Abstract**:

Transformer neural networks can exhibit a surprising capacity for in-context learning (ICL) despite not being explicitly trained for it.  Prior work has provided a deeper understanding of how ICL emerges in transformers, e.g. through the lens of mechanistic interpretability, Bayesian inference, or by examining the distributional properties of training data. However, in each of these cases, ICL is treated largely as a persistent phenomenon; namely, once ICL emerges, it is assumed to persist asymptotically. Here, we show that the emergence of ICL during transformer training is, in fact, often transient. We train transformers on synthetic data designed so that both ICL and in-weights learning (IWL) strategies can lead to correct predictions. We find that ICL first emerges, then disappears and gives way to IWL, all while the training loss decreases, indicating an asymptotic preference for IWL. The transient nature of ICL is observed in transformers across a range of model sizes and datasets, raising the question of how much to ``overtrain'' transformers when seeking compact, cheaper-to-run models. We find that L2 regularization may offer a path to more persistent ICL that removes the need for early stopping based on ICL-style validation tasks. Finally, we present initial evidence that ICL transience may be caused by competition between ICL and IWL circuits.

----

## [0] When is Agnostic Reinforcement Learning Statistically Tractable?

**Authors**: *Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, Nati Srebro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58a799d16fb0c1f2014e98f4ba972b25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58a799d16fb0c1f2014e98f4ba972b25-Abstract-Conference.html)

**Abstract**:

We study the problem of agnostic PAC reinforcement learning (RL): given a policy class $\Pi$, how many rounds of interaction with an unknown MDP (with a potentially large state and action space) are required to learn an $\epsilon$-suboptimal policy with respect to \(\Pi\)? Towards that end, we introduce a new complexity measure, called the \emph{spanning capacity}, that depends solely on the set \(\Pi\) and is independent of the MDP dynamics. With a generative model, we show that the spanning capacity characterizes PAC learnability for every policy class $\Pi$. However, for online RL, the situation is more subtle. We show there exists a policy class $\Pi$ with a bounded spanning capacity that requires a superpolynomial number of samples to learn. This reveals a surprising separation for agnostic learnability between generative access and online access models (as well as between deterministic/stochastic MDPs under online access). On the positive side, we identify an additional \emph{sunflower} structure which in conjunction with bounded spanning capacity enables statistically efficient online RL via a new algorithm called POPLER, which takes inspiration from classical importance sampling methods as well as recent developments for reachable-state identification and policy evaluation in reward-free exploration.

----

## [0] Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models

**Authors**: *Yeongbin Kim, Gautam Singh, Junyeong Park, Çaglar Gülçehre, Sungjin Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58af908d6293810f1a29e69bf723dc48-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/58af908d6293810f1a29e69bf723dc48-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Systematic compositionality, or the ability to adapt to novel situations by creating a mental model of the world using reusable pieces of knowledge, remains a significant challenge in machine learning. While there has been considerable progress in the language domain, efforts towards systematic visual imagination, or envisioning the dynamical implications of a visual observation, are in their infancy. We introduce the Systematic Visual Imagination Benchmark (SVIB), the first benchmark designed to address this problem head-on. SVIB offers a novel framework for a minimal world modeling problem, where models are evaluated based on their ability to generate one-step image-to-image transformations under a latent world dynamics. The framework provides benefits such as the possibility to jointly optimize for systematic perception and imagination, a range of difficulty levels, and the ability to control the fraction of possible factor combinations used during training. We provide a comprehensive evaluation of various baseline models on SVIB, offering insight into the current state-of-the-art in systematic visual imagination. We hope that this benchmark will help advance visual systematic compositionality.

----

## [0] Convolutional Visual Prompt for Robust Visual Perception

**Authors**: *Yun-Yun Tsai, Chengzhi Mao, Junfeng Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html)

**Abstract**:

Vision models are often vulnerable to out-of-distribution (OOD) samples without adapting. While visual prompts offer a lightweight method of input-space adaptation for large-scale vision models, they rely on a high-dimensional additive vector and labeled data. This leads to overfitting when adapting models in a self-supervised test-time setting without labels. We introduce convolutional visual prompts (CVP) for label-free test-time adaptation for robust visual perception. The structured nature of CVP demands fewer trainable parameters, less than 1\% compared to standard visual prompts, combating overfitting. Extensive experiments and analysis on a wide variety of OOD visual perception tasks show that our approach is effective, improving robustness by up to 5.87\% over several large-scale models.

----

## [0] LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching

**Authors**: *Duy M. H. Nguyen, Hoang Nguyen, Nghiem Tuong Diep, Tan Ngoc Pham, Tri Cao, Binh T. Nguyen, Paul Swoboda, Nhat Ho, Shadi Albarqouni, Pengtao Xie, Daniel Sonntag, Mathias Niepert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58cc11cda2a2679e8af5c6317aed0af8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58cc11cda2a2679e8af5c6317aed0af8-Abstract-Conference.html)

**Abstract**:

Obtaining large pre-trained models that can be fine-tuned to new tasks with limited annotated samples has remained an open challenge for medical imaging data. While pre-trained networks on ImageNet and vision-language foundation models trained on web-scale data are the prevailing approaches, their effectiveness on medical tasks is limited due to the significant domain shift between natural and medical images. To bridge this gap, we introduce LVM-Med, the first family of deep networks trained on large-scale medical datasets. We have collected approximately 1.3 million medical images from 55 publicly available datasets, covering a large number of organs and modalities such as CT, MRI, X-ray, and Ultrasound. We benchmark several state-of-the-art self-supervised algorithms on this dataset and propose a novel self-supervised contrastive learning algorithm using a graph-matching formulation. The proposed approach makes three contributions: (i) it integrates prior pair-wise image similarity metrics based on local and global information; (ii) it captures the structural constraints of feature embeddings through a loss function constructed through a combinatorial graph-matching objective, and (iii) it can be trained efficiently end-to-end using modern gradient-estimation techniques for black-box solvers. We thoroughly evaluate the proposed LVM-Med on 15 downstream medical tasks ranging from segmentation and classification to object detection, and both for the in and out-of-distribution settings. LVM-Med empirically outperforms a number of state-of-the-art supervised, self-supervised, and foundation models. For challenging tasks such as Brain Tumor Classification or Diabetic Retinopathy Grading, LVM-Med improves previous vision-language models trained on 1 billion masks by 6-7%  while using only a ResNet-50.

----

## [0] Lending Interaction Wings to Recommender Systems with Conversational Agents

**Authors**: *Jiarui Jin, Xianyu Chen, Fanghua Ye, Mengyue Yang, Yue Feng, Weinan Zhang, Yong Yu, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html)

**Abstract**:

An intelligent conversational agent (a.k.a., chat-bot) could embrace conversational technologies to obtain user preferences online, to overcome inherent limitations of recommender systems trained over the offline historical user behaviors. In this paper, we propose CORE, a new offline-training and online-checking framework to plug a COnversational agent into REcommender systems. Unlike most prior conversational recommendation approaches that systemically combine conversational and recommender parts through a reinforcement learning framework, CORE bridges the conversational agent and recommender system through a unified uncertainty minimization framework, which can be easily applied to any existing recommendation approach. Concretely, CORE treats a recommender system as an offline estimator to produce an estimated relevance score for each item, while CORE regards a conversational agent as an online checker that checks these estimated scores in each online session. We define uncertainty as the sum of unchecked relevance scores. In this regard, the conversational agent acts to minimize uncertainty via querying either attributes or items. Towards uncertainty minimization, we derive the certainty gain of querying each attribute and item, and develop a novel online decision tree algorithm to decide what to query at each turn. Our theoretical analysis reveals the bound of the expected number of turns of CORE in a cold-start setting. Experimental results demonstrate that CORE can be seamlessly employed on a variety of recommendation approaches, and can consistently bring significant improvements in both hot-start and cold-start settings.

----

## [0] High-Fidelity Audio Compression with Improved RVQGAN

**Authors**: *Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract-Conference.html)

**Abstract**:

Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves ~90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling.

----

## [0] Comparing Apples to Oranges: Learning Similarity Functions for Data Produced by Different Distributions

**Authors**: *Leonidas Tsepenekas, Ivan Brugere, Freddy Lécué, Daniele Magazzeni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59056767478c7df64e6250eadfeb0a04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59056767478c7df64e6250eadfeb0a04-Abstract-Conference.html)

**Abstract**:

Similarity functions measure how comparable pairs of elements are, and play a key role in a wide variety of applications, e.g., notions of Individual Fairness abiding by the seminal paradigm of Dwork et al., as well as Clustering problems. However, access to an accurate similarity function should not always be considered guaranteed, and this point was even raised by Dwork et al. For instance, it is reasonable to assume that when the elements to be compared are produced by different distributions, or in other words belong to different ``demographic'' groups, knowledge of their true similarity might be very difficult to obtain. In this work, we present an efficient sampling framework that learns these across-groups similarity functions, using only a limited amount of experts' feedback. We show analytical results with rigorous theoretical bounds, and empirically validate our algorithms via a large suite of experiments.

----

## [0] Scalable Transformer for PDE Surrogate Modeling

**Authors**: *Zijie Li, Dule Shu, Amir Barati Farimani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/590daf74f99ee85df3d8c007df9c8187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/590daf74f99ee85df3d8c007df9c8187-Abstract-Conference.html)

**Abstract**:

Transformer has shown state-of-the-art performance on various applications and has recently emerged as a promising tool for surrogate modeling of partial differential equations (PDEs). Despite the introduction of linear-complexity attention, applying Transformer to problems with a large number of grid points can be numerically unstable and computationally expensive. In this work, we propose Factorized Transformer (FactFormer), which is based on an axial factorized kernel integral. Concretely, we introduce a learnable projection operator that decomposes the input function into multiple sub-functions with one-dimensional domain. These sub-functions are then evaluated and used to compute the instance-based kernel with an axial factorized scheme. We showcase that the proposed model is able to simulate 2D Kolmogorov flow on a $256\times 256$ grid and 3D smoke buoyancy on a $64\times64\times64$ grid with good accuracy and efficiency. The proposed factorized scheme can serve as a computationally efficient low-rank surrogate for the full attention scheme when dealing with multi-dimensional problems.

----

## [0] What is the Inductive Bias of Flatness Regularization? A Study of Deep Matrix Factorization Models

**Authors**: *Khashayar Gatmiry, Zhiyuan Li, Tengyu Ma, Sashank Reddi, Stefanie Jegelka, Ching-Yao Chuang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5927edd18c5dd83aa8936a4610c72029-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5927edd18c5dd83aa8936a4610c72029-Abstract-Conference.html)

**Abstract**:

Recent works on over-parameterized neural networks have shown that  the stochasticity in optimizers has the implicit regularization effect of minimizing the sharpness of the loss function (in particular, the trace of its Hessian) over the family zero-loss solutions. More explicit forms of flatness regularization also empirically improve the generalization performance. However, it remains unclear why and when flatness regularization leads to better generalization. This work takes the first step towards understanding the inductive bias of the minimum trace of the Hessian solutions in an important setting: learning deep linear networks from linear measurements, also known as \emph{deep matrix factorization}. We show that with the standard Restricted Isometry Property (RIP) on the measurements, minimizing the trace of Hessian is approximately equivalent to minimizing the Schatten 1-norm of the corresponding end-to-end matrix parameters (i.e., the product of all layer matrices), which in turn leads to better generalization.

----

## [0] Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation

**Authors**: *Shutong Ding, Tianyu Cui, Jingya Wang, Ye Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html)

**Abstract**:

Deep Equilibrium Models (DEQs) and Neural Ordinary Differential Equations (Neural ODEs) are two branches of implicit models that have achieved remarkable success owing to their superior performance and low memory consumption. While both are implicit models, DEQs and Neural ODEs are derived from different mathematical formulations. Inspired by homotopy continuation, we establish a connection between these two models and illustrate that they are actually two sides of the same coin. Homotopy continuation is a classical method of solving nonlinear equations based on a corresponding ODE. Given this connection, we proposed a new implicit model called HomoODE that inherits the property of high accuracy from DEQs and the property of stability from Neural ODEs. Unlike DEQs, which explicitly solve an equilibrium-point-finding problem via Newton's methods in the forward pass, HomoODE solves the equilibrium-point-finding problem implicitly using a modified Neural ODE via homotopy continuation. Further, we developed an acceleration method for HomoODE with a shared learnable initial point. It is worth noting that our model also provides a better understanding of why Augmented Neural ODEs work as long as the augmented part is regarded as the equilibrium point to find. Comprehensive experiments with several image classification tasks demonstrate that HomoODE surpasses existing implicit models in terms of both accuracy and memory consumption.

----

## [0] Emergent and Predictable Memorization in Large Language Models

**Authors**: *Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html)

**Abstract**:

Memorization, or the tendency of large language models (LLMs) to output entire sequences from their training data verbatim, is a key concern for deploying language models. In particular, it is vital to minimize a model's memorization of sensitive datapoints such as those containing personal identifiable information (PII). The prevalence of such undesirable memorization can pose issues for model trainers, and may even require discarding an otherwise functional model. We therefore seek to predict which sequences will be memorized before a large model's full train-time by extrapolating the memorization behavior of lower-compute trial runs. We measure memorization in the Pythia model suite and plot scaling laws for forecasting memorization, allowing us to provide equi-compute recommendations to maximize the reliability (recall) of such predictions. We additionally provide further novel discoveries on the distribution of memorization scores across models and data. We release all code and data necessary to reproduce the results in this paper at https://github.com/EleutherAI/pythia.

----

## [0] Mind2Web: Towards a Generalist Agent for the Web

**Authors**: *Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samual Stevens, Boshi Wang, Huan Sun, Yu Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further research on building a generalist agent for the web.

----

## [0] MultiMoDN - Multimodal, Multi-Task, Interpretable Modular Networks

**Authors**: *Vinitra Swamy, Malika Satayeva, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser, Mary-Anne Hartley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5951641ad71b0052cf776f9b71f18932-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5951641ad71b0052cf776f9b71f18932-Abstract-Conference.html)

**Abstract**:

Predicting multiple real-world tasks in a single model often requires a particularly diverse feature space. Multimodal (MM) models aim to extract the synergistic predictive potential of multiple data types to create a shared feature space with aligned semantic meaning across inputs of drastically varying sizes (i.e. images, text, sound). Most current MM architectures fuse these representations in parallel, which not only limits their interpretability but also creates a dependency on modality availability. We present MultiModN, a multimodal, modular network that fuses latent representations in a sequence of any number, combination, or type of modality while providing granular real-time predictive feedback on any number or combination of predictive tasks. MultiModN's composable pipeline is interpretable-by-design, as well as innately multi-task and robust to the fundamental issue of biased missingness. We perform four experiments on several benchmark MM datasets across 10 real-world tasks (predicting medical diagnoses, academic performance, and weather), and show that MultiModN's sequential MM fusion does not compromise performance compared with a baseline of parallel fusion. By simulating the challenging bias of missing not-at-random (MNAR), this work shows that, contrary to MultiModN, parallel fusion baselines erroneously learn MNAR and suffer catastrophic failure when faced with different patterns of MNAR at inference. To the best of our knowledge, this is the first inherently MNAR-resistant approach to MM modeling. In conclusion, MultiModN provides granular insights, robustness, and flexibility without compromising performance.

----

## [0] Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs

**Authors**: *Shengyuan Chen, Yunfeng Cai, Huang Fang, Xiao Huang, Mingming Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5965f3a748a8d41415db2bfa44635cc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5965f3a748a8d41415db2bfa44635cc3-Abstract-Conference.html)

**Abstract**:

Knowledge graph (KG) reasoning utilizes two primary techniques, i.e., rule-based and KG-embedding based. The former provides precise inferences, but inferring via concrete rules is not scalable. The latter enables efficient reasoning at the cost of ambiguous inference accuracy. Neuro-symbolic reasoning seeks to amalgamate the advantages of both techniques. The crux of this approach is replacing the predicted existence of all possible triples (i.e., truth scores inferred from rules) with a suitable approximation grounded in embedding representations. However, constructing an effective approximation of all possible triples' truth scores is a challenging task, because it needs to balance the tradeoff between accuracy and efficiency, while compatible with both the rule-based and KG-embedding models. To this end, we proposed a differentiable framework - DiffLogic. Instead of directly approximating all possible triples, we design a tailored filter to adaptively select essential triples based on the dynamic rules and weights. The truth scores assessed by KG-embedding are continuous, so we employ a continuous Markov logic network named probabilistic soft logic (PSL). It employs the truth scores of essential triples to assess the overall agreement among rules, weights, and observed triples. PSL enables end-to-end differentiable optimization, so we can alternately update embedding and weighted rules. On benchmark datasets, we empirically show that DiffLogic surpasses baselines in both effectiveness and efficiency.

----

## [0] Topological Parallax: A Geometric Specification for Deep Perception Models

**Authors**: *Abraham D. Smith, Michael J. Catanzaro, Gabrielle Angeloro, Nirav Patel, Paul Bendich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html)

**Abstract**:

For safety and robustness of AI systems, we introduce topological parallax as a theoretical and computational tool that compares a trained model to a reference dataset to determine whether they have similar multiscale geometric structure. Our proofs and examples show that this geometric similarity between dataset and model is essential to trustworthy interpolation and perturbation, and we conjecture that this new concept will add value to the current debate regarding the unclear relationship between "overfitting"' and "generalization'' in applications of deep-learning. In typical deep-learning applications, an explicit geometric description of the model isimpossible, but parallax can estimate topological features (components, cycles, voids, etc.)in the model by examining the effect on the Rips complex of geodesic distortions using the reference dataset.Thus, parallax indicates whether the model shares similar multiscale geometric features with the dataset.Parallax presents theoretically via topological data analysis [TDA] as a bi-filtered persistence module,and the key properties of this module are stable under perturbation of the reference dataset.

----

## [0] Rewiring Neurons in Non-Stationary Environments

**Authors**: *Zhicheng Sun, Yadong Mu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/599221d7ebf6b3403190f38a3f282a1c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/599221d7ebf6b3403190f38a3f282a1c-Abstract-Conference.html)

**Abstract**:

The human brain rewires itself for neuroplasticity in the presence of new tasks. We are inspired to harness this key process in continual reinforcement learning, prioritizing adaptation to non-stationary environments. In distinction to existing rewiring approaches that rely on pruning or dynamic routing, which may limit network capacity and plasticity, this work presents a novel rewiring scheme by permuting hidden neurons. Specifically, the neuron permutation is parameterized to be end-to-end learnable and can rearrange all available synapses to explore a large span of weight space, thereby promoting adaptivity. In addition, we introduce two main designs to steer the rewiring process in continual reinforcement learning: first, a multi-mode rewiring strategy is proposed which diversifies the policy and encourages exploration when encountering new environments. Secondly, to ensure stability on history tasks, the network is devised to cache each learned wiring while subtly updating its weights, allowing for retrospective recovery of any previous state appropriate for the task. Meanwhile, an alignment mechanism is curated to achieve better plasticity-stability tradeoff by jointly optimizing cached wirings and weights. Our proposed method is comprehensively evaluated on 18 continual reinforcement learning scenarios ranging from locomotion to manipulation, demonstrating its advantages over state-of-the-art competitors in performance-efficiency tradeoffs. Code is available at https://github.com/feifeiobama/RewireNeuron.

----

## [0] TransHP: Image Classification with Hierarchical Prompting

**Authors**: *Wenhao Wang, Yifan Sun, Wei Li, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59b7c1e1716c4feadefd6c70b1dd4630-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59b7c1e1716c4feadefd6c70b1dd4630-Abstract-Conference.html)

**Abstract**:

This paper explores a hierarchical prompting mechanism for the hierarchical image classification (HIC) task. Different from prior HIC methods, our hierarchical prompting is the first to explicitly inject ancestor-class information as a tokenized hint that benefits the descendant-class discrimination. We think it well imitates human visual recognition, i.e., humans may use the ancestor class as a prompt to draw focus on the subtle differences among descendant classes. We model this prompting mechanism into a Transformer with Hierarchical Prompting (TransHP). TransHP consists of three steps: 1) learning a set of prompt tokens to represent the coarse (ancestor) classes, 2) on-the-fly predicting the coarse class of the input image at an intermediate block, and 3) injecting the prompt token of the predicted coarse class into the intermediate feature. Though the parameters of TransHP maintain the same for all input images, the injected coarse-class prompt conditions (modifies) the subsequent feature extraction and encourages a dynamic focus on relatively subtle differences among the descendant classes. Extensive experiments show that TransHP improves image classification on accuracy (e.g., improving ViT-B/16 by +2.83% ImageNet classification accuracy), training data efficiency (e.g., +12.69% improvement under 10% ImageNet training data), and model explainability. Moreover, TransHP also performs favorably against prior HIC methods, showing that TransHP well exploits the hierarchical information. The code is available at: https://github.com/WangWenhao0716/TransHP.

----

## [0] Practical Differentially Private Hyperparameter Tuning with Subsampling

**Authors**: *Antti Koskela, Tejas D. Kulkarni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html)

**Abstract**:

Tuning the hyperparameters of differentially private (DP) machine learning (ML) algorithms often requires use of sensitive data and this may leak private information via hyperparameter values. Recently, Papernot and Steinke (2022) proposed a certain class of DP hyperparameter tuning algorithms, where the number of random search samples is randomized. Commonly, these algorithms still considerably increase the DP privacy parameter $\varepsilon$ over non-tuned DP ML model training and can be computationally heavy as evaluating each hyperparameter candidate requires a new training run. We focus on lowering both the DP bounds and the compute cost of these methods by using only a random subset of the sensitive data for the hyperparameter tuning and by appropriately extrapolating the optimal values to a larger dataset. We carry out a RÃ©nyi differential privacy analysis for the proposed method and experimentally show that it consistently leads to better privacy-utility trade-off than the baseline method by Papernot and Steinke.

----

## [0] Learning to Discover Skills through Guidance

**Authors**: *Hyunseung Kim, Byungkun Lee, Hojoon Lee, Dongyoon Hwang, Sejik Park, Kyushik Min, Jaegul Choo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59d4e18a60490b9ed9913f3be2b14839-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59d4e18a60490b9ed9913f3be2b14839-Abstract-Conference.html)

**Abstract**:

In the field of unsupervised skill discovery (USD), a major challenge is limited exploration, primarily due to substantial penalties when skills deviate from their initial trajectories. To enhance exploration, recent methodologies employ auxiliary rewards to maximize the epistemic uncertainty or entropy of states. However, we have identified that the effectiveness of these rewards declines as the environmental complexity rises. Therefore, we present a novel USD algorithm, skill discovery with guidance (DISCO-DANCE), which (1) selects the guide skill that possesses the highest potential to reach unexplored states, (2) guides other skills to follow guide skill, then (3) the guided skills are dispersed to maximize their discriminability in unexplored states. Empirical evaluation demonstrates that DISCO-DANCE outperforms other USD baselines in challenging environments, including two navigation benchmarks and a continuous control benchmark. Qualitative visualizations and code of DISCO-DANCE are available at https://mynsng.github.io/discodance/.

----

## [0] Polynomial-Time Linear-Swap Regret Minimization in Imperfect-Information Sequential Games

**Authors**: *Gabriele Farina, Charilaos Pipis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html)

**Abstract**:

No-regret learners seek to minimize the difference between the loss they cumulated through the actions they played, and the loss they would have cumulated in hindsight had they consistently modified their behavior according to some strategy transformation function. The size of the set of transformations considered by the learner determines a natural notion of rationality. As the set of transformations each learner considers grows, the strategies played by the learners recover more complex game-theoretic equilibria, including correlated equilibria in normal-form games and extensive-form correlated equilibria in extensive-form games. At the extreme, a no-swap-regret agent is one that minimizes regret against the set of all functions from the set of strategies to itself. While it is known that the no-swap-regret condition can be attained efficiently in nonsequential (normal-form) games, understanding what is the strongest notion of rationality that can be attained efficiently in the worst case in sequential (extensive-form) games is a longstanding open problem. In this paper we provide a positive result, by showing that it is possible, in any sequential game, to retain polynomial-time (in the game tree size) iterations while achieving sublinear regret with respect to all linear transformations of the mixed strategy space, a notion called no-linear-swap regret. This notion of hindsight rationality is as strong as no-swap-regret in nonsequential games, and stronger than no-trigger-regret in sequential gamesâ€”thereby proving the existence of a subset of extensive-form correlated equilibria robust to linear deviations, which we call linear-deviation correlated equilibria, that can be approached efficiently.

----

## [0] Counterfactually Comparing Abstaining Classifiers

**Authors**: *Yo Joong Choe, Aditya Gangrade, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59fe467d5e71ba6b8d41bb3928da6f4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59fe467d5e71ba6b8d41bb3928da6f4c-Abstract-Conference.html)

**Abstract**:

Abstaining classifiers have the option to abstain from making predictions on inputs that they are unsure about. These classifiers are becoming increasingly popular in high-stakes decision-making problems, as they can withhold uncertain predictions to improve their reliability and safety. When evaluating black-box abstaining classifier(s), however, we lack a principled approach that accounts for what the classifier would have predicted on its abstentions. These missing predictions matter when they can eventually be utilized, either directly or as a backup option in a failure mode. In this paper, we introduce a novel approach and perspective to the problem of evaluating and comparing abstaining classifiers by treating abstentions as missing data. Our evaluation approach is centered around defining the counterfactual score of an abstaining classifier, defined as the expected performance of the classifier had it not been allowed to abstain. We specify the conditions under which the counterfactual score is identifiable: if the abstentions are stochastic, and if the evaluation data is independent of the training data (ensuring that the predictions are missing at random), then the score is identifiable. Note that, if abstentions are deterministic, then the score is unidentifiable because the classifier can perform arbitrarily poorly on its abstentions. Leveraging tools from observational causal inference, we then develop nonparametric and doubly robust methods to efficiently estimate this quantity under identification. Our approach is examined in both simulated and real data experiments.

----

## [0] Video Timeline Modeling For News Story Understanding

**Authors**: *Meng Liu, Mingda Zhang, Jialu Liu, Hanjun Dai, Ming-Hsuan Yang, Shuiwang Ji, Zheyun Feng, Boqing Gong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a0f92efaa8f0cd67992caf6b2fa2bac-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a0f92efaa8f0cd67992caf6b2fa2bac-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In this paper, we present a novel problem, namely video timeline modeling. Our objective is to create a video-associated timeline from a set of videos related to a specific topic, thereby facilitating the content and structure understanding of the story being told. This problem has significant potential in various real-world applications, for instance, news story summarization. To bootstrap research in this area, we curate a realistic benchmark dataset, YouTube-News-Timeline, consisting of over $12$k timelines and $300$k YouTube news videos. Additionally, we propose a set of quantitative metrics to comprehensively evaluate and compare methodologies. With such a testbed, we further develop and benchmark several deep learning approaches to tackling this problem. We anticipate that this exploratory work will pave the way for further research in video timeline modeling. The assets are available via https://github.com/google-research/google-research/tree/master/video_timeline_modeling.

----

## [0] Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning

**Authors**: *Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam, Remi Tachet des Combes, Romain Laroche*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html)

**Abstract**:

While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks,  their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to understand why bisimulation methods succeed in online settings, but falter in offline tasks. Our analysis reveals that missing transitions in the dataset are particularly harmful to the bisimulation principle, leading to ineffective estimation. We also shed light on the critical role of reward scaling in bounding the scale of bisimulation measurements and of the value error they induce. Based on these findings, we propose to apply the expectile operator for representation learning to our offline RL setting, which helps to prevent overfitting to incomplete data. Meanwhile, by introducing an appropriate reward scaling strategy, we avoid the risk of feature collapse in representation space. We implement these recommendations on two state-of-the-art bisimulation-based algorithms, MICo and SimSR, and demonstrate performance gains on two benchmark suites: D4RL and Visual D4RL. Codes are provided at \url{https://github.com/zanghyu/Offline_Bisimulation}.

----

## [0] Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting

**Authors**: *Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally-trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (predict). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the predictions of base forecasters with reduced computational overhead over reverse diffusion (refine). Notably, the generative performance of the model remains intact â€” downstream forecasters trained on synthetic samples from TSDiff outperform forecasters that are trained on samples from other state-of-the-art generative time series models, occasionally even outperforming models trained on real data (synthesize).Our code is available at https://github.com/amazon-science/unconditional-time-series-diffusion

----

## [0] Learning Layer-wise Equivariances Automatically using Gradients

**Authors**: *Tycho F. A. van der Ouderaa, Alexander Immer, Mark van der Wilk*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a33645b5c5b7a9882652526d30d0acc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a33645b5c5b7a9882652526d30d0acc-Abstract-Conference.html)

**Abstract**:

Convolutions encode equivariance symmetries into neural networks leading to better generalisation performance. However, symmetries provide fixed hard constraints on the functions a network can represent, need to be specified in advance, and can not be adapted. Our goal is to allow flexible symmetry constraints that can automatically be learned from data using gradients. Learning symmetry and associated weight connectivity structures from scratch is difficult for two reasons. First, it requires efficient and flexible parameterisations of layer-wise equivariances. Secondly, symmetries act as constraints and are therefore not encouraged by training losses measuring data fit. To overcome these challenges, we improve parameterisations of soft equivariance and learn the amount of equivariance in layers by optimising the marginal likelihood, estimated using differentiable Laplace approximations. The objective balances data fit and model complexity enabling layer-wise symmetry discovery in deep networks. We demonstrate the ability to automatically learn layer-wise equivariances on image classification tasks, achieving equivalent or improved performance over baselines with hard-coded symmetry.

----

## [0] PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated Learning

**Authors**: *Mingjia Shi, Yuhao Zhou, Kai Wang, Huaizheng Zhang, Shudong Huang, Qing Ye, Jiancheng Lv*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html)

**Abstract**:

Classical federated learning (FL) enables training machine learning models without sharing data for privacy preservation, but heterogeneous data characteristic degrades the performance of the localized model. Personalized FL (PFL) addresses this by synthesizing personalized models from a global model via training on local data. Such a global model may overlook the specific information that the clients have been sampled. In this paper, we propose a novel scheme to inject personalized prior knowledge into the global model in each client, which attempts to mitigate the introduced incomplete information problem in PFL. At the heart of our proposed approach is a framework, the $\textit{PFL with Bregman Divergence}$ (pFedBreD), decoupling the personalized prior from the local objective function regularized by Bregman divergence for greater adaptability in personalized scenarios. We also relax the mirror descent (RMD) to extract the prior explicitly to provide optional strategies. Additionally, our pFedBreD is backed up by a convergence analysis. Sufficient experiments demonstrate that our method reaches the $\textit{state-of-the-art}$ performances on 5 datasets and outperforms other methods by up to 3.5% across 8 benchmarks. Extensive analyses verify the robustness and necessity of proposed designs. The code will be made public.

----

## [0] Byzantine-Tolerant Methods for Distributed Variational Inequalities

**Authors**: *Nazarii Tupitsa, Abdulla Jasem Almansoori, Yanlin Wu, Martin Takác, Karthik Nandakumar, Samuel Horváth, Eduard Gorbunov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html)

**Abstract**:

Robustness to Byzantine attacks is a necessity for various distributed training scenarios. When the training reduces to the process of solving a minimization problem, Byzantine robustness is relatively well-understood. However, other problem formulations, such as min-max problems or, more generally, variational inequalities, arise in many modern machine learning and, in particular, distributed learning tasks. These problems significantly differ from the standard minimization ones and, therefore, require separate consideration. Nevertheless, only one work [Abidi et al., 2022] addresses this important question in the context of Byzantine robustness. Our work makes a further step in this direction by providing several (provably) Byzantine-robust methods for distributed variational inequality, thoroughly studying their theoretical convergence, removing the limitations of the previous work, and providing numerical comparisons supporting the theoretical findings.

----

## [0] Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow

**Authors**: *Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, Ya Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html)

**Abstract**:

Collaborative perception can substantially boost each agent's perception ability by facilitating communication among multiple agents. However, temporal asynchrony among agents is inevitable in the real world due to communication delays, interruptions, and clock misalignments. This issue causes information mismatch during multi-agent fusion, seriously shaking the foundation of collaboration. To address this issue, we propose CoBEVFlow, an asynchrony-robust collaborative perception system based on bird's eye view (BEV) flow. The key intuition of CoBEVFlow is to compensate motions to align asynchronous collaboration messages sent by multiple agents. To model the motion in a scene, we propose BEV flow, which is a collection of the motion vector corresponding to each spatial location. Based on BEV flow, asynchronous perceptual features can be reassigned to appropriate positions, mitigating the impact of asynchrony. CoBEVFlow has two advantages: (i) CoBEVFlow can handle asynchronous collaboration messages sent at irregular, continuous time stamps without discretization; and (ii) with BEV flow, CoBEVFlow only transports the original perceptual features, instead of generating new perceptual features, avoiding additional noises. To validate CoBEVFlow's efficacy, we create IRregular V2V(IRV2V), the first synthetic collaborative perception dataset with various temporal asynchronies that simulate different real-world scenarios. Extensive experiments conducted on both IRV2V and the real-world dataset DAIR-V2X show that CoBEVFlow consistently outperforms other baselines and is robust in extremely asynchronous settings. The code is available at https://github.com/MediaBrain-SJTU/CoBEVFlow.

----

## [0] Train 'n Trade: Foundations of Parameter Markets

**Authors**: *Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html)

**Abstract**:

Organizations typically train large models individually. This is costly and time-consuming, particularly for large-scale foundation models. Such vertical production is known to be suboptimal. Inspired by this economic insight, we ask whether it is possible to leverage others' expertise by trading the constituent parts in models, i.e., sets of weights, as if they were market commodities. While recent advances in aligning and interpolating models suggest that doing so may be possible, a number of fundamental questions must be answered to create viable parameter markets. In this work, we address these basic questions, propose a framework containing the infrastructure necessary for market operations to take place, study strategies for exchanging parameters, and offer means for agents to monetize parameters. Excitingly, compared to agents who train siloed models from scratch, we show that it is possible to mutually gain by using the market, even in competitive settings. This suggests that the notion of parameter markets may be a useful paradigm for improving large-scale model training in the future.

----

## [0] Relax, it doesn't matter how you get there: A new self-supervised approach for multi-timescale behavior analysis

**Authors**: *Mehdi Azabou, Michael Mendelson, Nauman Ahad, Maks Sorokin, Shantanu Thakoor, Carolina Urzay, Eva L. Dyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aad86aa2a3c00b70c71e19bc4780319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aad86aa2a3c00b70c71e19bc4780319-Abstract-Conference.html)

**Abstract**:

Unconstrained and natural  behavior consists of dynamics that are complex and  unpredictable, especially when trying to predict what will happen  multiple steps into the future. While some success has been found in building representations of animal behavior under constrained or simplified task-based conditions, many of these models cannot be applied to free and naturalistic settings where behavior becomes increasingly hard to model. In this work, we develop a multi-task representation learning model for animal behavior that combines two novel components: (i) an action-prediction objective that aims to predict the  distribution of actions over future timesteps, and (ii) a multi-scale architecture that builds separate latent spaces to accommodate short- and long-term dynamics. After demonstrating the ability of the method to build representations of both local and global dynamics in robots in varying environments and terrains, we apply our method to the MABe 2022 Multi-Agent Behavior challenge, where our model ranks first overall on both mice and fly benchmarks. In all of these cases, we show that our model can build representations that capture the many different factors that drive behavior and solve a wide range of downstream tasks.

----

## [0] A Measure-Theoretic Axiomatisation of Causality

**Authors**: *Junhyung Park, Simon Buchholz, Bernhard Schölkopf, Krikamol Muandet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aadf1e309cc03cab3ec35afb7c9d0c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aadf1e309cc03cab3ec35afb7c9d0c8-Abstract-Conference.html)

**Abstract**:

Causality is a central concept in a wide range of research areas, yet there is still no universally agreed axiomatisation of causality. We view causality both as an extension of probability theory and as a study of what happens when one intervenes on a system, and argue in favour of taking Kolmogorov's measure-theoretic axiomatisation of probability as the starting point towards an axiomatisation of causality. To that end, we propose the notion of a causal space, consisting of a probability space along with a collection of transition probability kernels, called causal kernels, that encode the causal information of the space. Our proposed framework is not only rigorously grounded in measure theory, but it also sheds light on long-standing limitations of existing frameworks including, for example, cycles, latent variables and stochastic processes.

----

## [0] LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

**Authors**: *Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, Jianfeng Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Conversational generative AI has demonstrated remarkable promise for empowering biomedical practitioners, but current investigations focus on unimodal text. Multimodal conversational AI has seen rapid progress by leveraging billions of image-text pairs from the public web, but such general-domain vision-language models still lack sophistication in understanding and conversing about biomedical images. In this paper, we propose a cost-efficient approach for training a vision-language conversational assistant that can answer open-ended research questions of biomedical images. The key idea is to leverage a large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central, use GPT-4 to self-instruct open-ended instruction-following data from the captions, and then fine-tune a large general-domain vision-language model using a novel curriculum learning method. Specifically, the model first learns to align biomedical vocabulary using the figure-caption pairs as is, then learns to master open-ended conversational semantics using GPT-4 generated instruction-following data, broadly mimicking how a layperson gradually acquires biomedical knowledge. This enables us to train a Large Language and Vision Assistant for BioMedicine (LLaVA-Med) in less than 15 hours (with eight A100s). LLaVA-Med exhibits excellent multimodal conversational capability and can follow open-ended instruction to assist with inquiries about a biomedical image. On three standard biomedical visual question answering datasets, LLaVA-Med outperforms previous supervised state-of-the-art on certain metrics. To facilitate biomedical multimodal research, we will release our instruction-following data and the LLaVA-Med model.

----

## [0] Networks are Slacking Off: Understanding Generalization Problem in Image Deraining

**Authors**: *Jinjin Gu, Xianzheng Ma, Xiangtao Kong, Yu Qiao, Chao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aca18e0192b2c1300479e5b700c76a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aca18e0192b2c1300479e5b700c76a9-Abstract-Conference.html)

**Abstract**:

Deep deraining networks consistently encounter substantial generalization issues when deployed in real-world applications, although they are successful in laboratory benchmarks. A prevailing perspective in deep learning encourages using highly complex data for training, with the expectation that richer image background content will facilitate overcoming the generalization problem. However, through comprehensive and systematic experimentation, we discover that this strategy does not enhance the generalization capability of these networks. On the contrary, it exacerbates the tendency of networks to overfit specific degradations. Our experiments reveal that better generalization in a deraining network can be achieved by simplifying the complexity of the training background images. This is because that the networks are ``slacking off'' during training, that is, learning the least complex elements in the image background and degradation to minimize training loss. When the background images are less complex than the rain streaks, the network will prioritize the background reconstruction, thereby suppressing overfitting the rain patterns and leading to improved generalization performance. Our research offers a valuable perspective and methodology for better understanding the generalization problem in low-level vision tasks and displays promising potential for practical application.

----

## [0] Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning

**Authors**: *Xiaojun Guo, Yifei Wang, Zeming Wei, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5acf5a0ee5c17d372bfe7fdaeffd6e33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5acf5a0ee5c17d372bfe7fdaeffd6e33-Abstract-Conference.html)

**Abstract**:

With the prosperity of contrastive learning for visual representation learning (VCL), it is also adapted to the graph domain and yields promising performance. However, through a systematic study of various graph contrastive learning (GCL) methods, we observe that some common phenomena among existing GCL methods that are quite different from the original VCL methods, including 1) positive samples are not a must for GCL; 2) negative samples are not necessary for graph classification, neither for node classification when adopting specific normalization modules; 3) data augmentations have much less influence on GCL, as simple domain-agnostic augmentations (e.g., Gaussian noise) can also attain fairly good performance. By uncovering how the implicit inductive bias of GNNs works in contrastive learning, we theoretically provide insights into the above intriguing properties of GCL. Rather than directly porting existing VCL methods to GCL, we advocate for more attention toward the unique architecture of graph learning and consider its implicit influence when designing GCL methods. Code is available at https://github.com/PKU-ML/ArchitectureMattersGCL.

----

## [0] Text Promptable Surgical Instrument Segmentation with Vision-Language Models

**Authors**: *Zijian Zhou, Oluwatosin Alabi, Meng Wei, Tom Vercauteren, Miaojing Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5af741d487c5f0b08bfe56e11d1883e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5af741d487c5f0b08bfe56e11d1883e4-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a novel text promptable surgical instrument segmentation approach to overcome challenges associated with diversity and differentiation of surgical instruments in minimally invasive surgeries. We redefine the task as text promptable, thereby enabling a more nuanced comprehension of surgical instruments and adaptability to new instrument types. Inspired by recent advancements in vision-language models, we leverage pretrained image and text encoders as our model backbone and design a text promptable mask decoder consisting of attention- and convolution-based prompting schemes for surgical instrument segmentation prediction. Our model leverages multiple text prompts for each surgical instrument through a new mixture of prompts mechanism, resulting in enhanced segmentation performance. Additionally, we introduce a hard instrument area reinforcement module to improve image feature comprehension and segmentation precision. Extensive experiments on several surgical instrument segmentation datasets demonstrate our model's superior performance and promising generalization capability. To our knowledge, this is the first implementation of a promptable approach to surgical instrument segmentation, offering significant potential for practical application in the field of robotic-assisted surgery. Code is available at https://github.com/franciszzj/TP-SIS.

----

## [0] OpenDataVal: a Unified Benchmark for Data Valuation

**Authors**: *Kevin Fu Jiang, Weixin Liang, James Y. Zou, Yongchan Kwon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b047c7d862059a5df623c1ce2982fca-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b047c7d862059a5df623c1ce2982fca-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Assessing the quality and impact of individual data points is critical for improving model performance and mitigating undesirable biases within the training dataset. Several data valuation algorithms have been proposed to quantify data quality, however, there lacks a systemic and standardized benchmarking system for data valuation. In this paper, we introduce OpenDataVal, an easy-to-use and unified benchmark framework that empowers researchers and practitioners to apply and compare various data valuation algorithms. OpenDataVal provides an integrated environment that includes (i) a diverse collection of image, natural language, and tabular datasets, (ii) implementations of eleven different state-of-the-art data valuation algorithms, and (iii) a prediction model API that can import any models in scikit-learn. Furthermore, we propose four downstream machine learning tasks for evaluating the quality of data values. We perform benchmarking analysis using OpenDataVal, quantifying and comparing the efficacy of state-of-the-art data valuation approaches. We find that no single algorithm performs uniformly best across all tasks, and an appropriate algorithm should be employed for a user's downstream task. OpenDataVal is publicly available at https://opendataval.github.io with comprehensive documentation. Furthermore, we provide a leaderboard where researchers can evaluate the effectiveness of their own data valuation algorithms.

----

## [0] On the Consistency of Maximum Likelihood Estimation of Probabilistic Principal Component Analysis

**Authors**: *Arghya Datta, Sayak Chakrabarty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b0c0b2c2efdd736a53688ebfdc3bcdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b0c0b2c2efdd736a53688ebfdc3bcdb-Abstract-Conference.html)

**Abstract**:

Probabilistic principal component analysis (PPCA) is currently one of the most used statistical tools to reduce the ambient dimension of the data. From multidimensional scaling to the imputation of missing data, PPCA has a broad spectrum of applications ranging from science and engineering to quantitative finance.\Despite this wide applicability in various fields, hardly any theoretical guarantees exist to justify the soundness of the maximal likelihood (ML) solution for this model. In fact, it is well known that the maximum likelihood estimation (MLE) can only recover the true model parameters up to a rotation. The main obstruction is posed by the inherent identifiability nature of the PPCA model resulting from the rotational symmetry of the parameterization. To resolve this ambiguity, we propose a novel approach using quotient topological spaces and in particular, we show that the maximum likelihood solution is consistent in an appropriate quotient Euclidean space. Furthermore, our consistency results encompass a more general class of estimators beyond the MLE. Strong consistency of the ML estimate and consequently strong covariance estimation of the PPCA model have also been established under a compactness assumption.

----

## [0] Similarity, Compression and Local Steps: Three Pillars of Efficient Communications for Distributed Variational Inequalities

**Authors**: *Aleksandr Beznosikov, Martin Takác, Alexander V. Gasnikov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html)

**Abstract**:

Variational inequalities are a broad and flexible class of problems that includes minimization, saddle point, and fixed point problems as special cases. Therefore, variational inequalities are used in various applications ranging from equilibrium search to adversarial learning. With the increasing size of data and models, today's instances demand parallel and distributed computing for real-world machine learning problems, most of which can be represented as variational inequalities. Meanwhile, most distributed approaches have a significant bottleneck -- the cost of communications. The three main techniques to reduce the total number of communication rounds and the cost of one such round are the similarity of local functions, compression of transmitted information, and local updates. In this paper, we combine all these approaches. Such a triple synergy did not exist before for variational inequalities and saddle problems, nor even for minimization problems. The methods presented in this paper have the best theoretical guarantees of communication complexity and are significantly ahead of other methods for distributed variational inequalities. The theoretical results are confirmed by adversarial learning experiments on synthetic and real datasets.

----

## [0] Lookaround Optimizer: k steps around, 1 step average

**Authors**: *Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, Mingli Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b4b967d4222d87fa5b28b6ec7144058-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b4b967d4222d87fa5b28b6ec7144058-Abstract-Conference.html)

**Abstract**:

Weight Average (WA) is an active research topic due to its simplicity in ensembling deep networks and the effectiveness in promoting generalization. Existing weight average approaches, however, are often carried out along only one training trajectory in a post-hoc manner (i.e., the weights are averaged after the entire training process is finished), which significantly degrades the diversity between networks and thus impairs the effectiveness. In this paper, inspired by weight average, we propose Lookaround, a straightforward yet effective SGD-based optimizer leading to flatter minima with better generalization. Specifically, Lookaround iterates two steps during the whole training period: the around step and the average step. In each iteration, 1) the around step starts from a common point and trains multiple networks simultaneously, each on transformed data by a different data augmentation, and 2) the average step averages these trained networks to get the averaged network, which serves as the starting point for the next iteration. The around step improves the functionality diversity while the average step guarantees the weight locality of these networks during the whole training, which is essential for WA to work. We theoretically explain the superiority of Lookaround by convergence analysis, and make extensive experiments to evaluate Lookaround on popular benchmarks including CIFAR and ImageNet with both CNNs and ViTs, demonstrating clear superiority over state-of-the-arts. Our code is available at https://github.com/Ardcy/Lookaround.

----

## [0] The Quantization Model of Neural Scaling

**Authors**: *Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html)

**Abstract**:

We propose the Quantization Model of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale.  We derive this model from what we call the Quantization Hypothesis, where network knowledge and skills are "quantized" into discrete chunks (quanta). We show that when quanta are learned in order of decreasing use frequency, then a power law in use frequencies explains observed power law scaling of loss. We validate this prediction on toy datasets, then study how scaling curves decompose for large language models.  Using language model gradients, we automatically decompose model behavior into a diverse set of skills (quanta). We tentatively find that the frequency at which these quanta are used in the training distribution roughly follows a power law corresponding with the empirical scaling exponent for language models, a prediction of our theory.

----

## [0] ε-fractional core stability in Hedonic Games

**Authors**: *Simone Fioravanti, Michele Flammini, Bojana Kodric, Giovanna Varricchio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b755cf5598a4324d253025e1fbbba52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b755cf5598a4324d253025e1fbbba52-Abstract-Conference.html)

**Abstract**:

Hedonic Games (HGs) are a classical framework modeling coalition formation of strategic agents guided by their individual preferences. According to these preferences, it is desirable that a coalition structure (i.e. a partition of agents into coalitions) satisfies some form of stability. The most well-known and natural of such notions is arguably core-stability. Informally, a partition is core-stable if no subset of agents would like to deviate by regrouping in a so-called core-blocking coalition. Unfortunately, core-stable partitions seldom exist and even when they do, it is often computationally intractable to find one. To circumvent these problems, we propose the notion of $\varepsilon$-fractional core-stability, where at most an $\varepsilon$-fraction of all possible coalitions is allowed to core-block. It turns out that such a relaxation may guarantee both existence and polynomial-time computation. Specifically, we design efficient algorithms returning an $\varepsilon$-fractional core-stable partition, with $\varepsilon$ exponentially decreasing in the number of agents, for two fundamental classes of HGs: Simple Fractional and Anonymous. From a probabilistic point of view, being the definition of $\varepsilon$-fractional core equivalent to requiring that uniformly sampled coalitions core-block with probability lower than $\varepsilon$, we further extend the definition to handle more complex sampling distributions. Along this line, when valuations have to be learned from samples in a PAC-learning fashion, we give positive and negative results on which distributions allow the efficient computation of outcomes that are $\varepsilon$-fractional core-stable with arbitrarily high confidence.

----

## [0] Semi-Supervised Domain Generalization with Known and Unknown Classes

**Authors**: *Lei Zhang, Ji-Fu Li, Wei Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html)

**Abstract**:

Semi-Supervised Domain Generalization (SSDG) aims to learn a model that is generalizable to an unseen target domain with only a few labels, and most existing SSDG methods assume that unlabeled training and testing samples are all known classes. However, a more realistic scenario is that known classes may be mixed with some unknown classes in unlabeled training and testing data. To deal with such a scenario, we propose the Class-Wise Adaptive Exploration and Exploitation (CWAEE) method. In particular, we explore unlabeled training data by using one-vs-rest classifiers and class-wise adaptive thresholds to detect known and unknown classes, and exploit them by adopting consistency regularization on augmented samples based on Fourier Transformation to improve the unseen domain generalization. The experiments conducted on real-world datasets verify the effectiveness and superiority of our method.

----

## [0] When Do Graph Neural Networks Help with Node Classification? Investigating the Homophily Principle on Node Distinguishability

**Authors**: *Sitao Luan, Chenqing Hua, Minkai Xu, Qincheng Lu, Jiaqi Zhu, Xiao-Wen Chang, Jie Fu, Jure Leskovec, Doina Precup*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ba11de4c74548071899cf41dec078bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ba11de4c74548071899cf41dec078bf-Abstract-Conference.html)

**Abstract**:

Homophily principle, i.e., nodes with the same labels are more likely to be connected, has been believed to be the main reason for the performance superiority of Graph Neural Networks (GNNs) over Neural Networks on node classification tasks. Recent research suggests that, even in the absence of homophily, the advantage of GNNs still exists as long as nodes from the same class share similar neighborhood patterns. However, this argument only considers intra-class Node Distinguishability (ND) but neglects inter-class ND, which provides incomplete understanding of homophily on GNNs. In this paper, we first demonstrate such deficiency with examples and argue that an ideal situation for ND is to have smaller intra-class ND than inter-class ND. To formulate this idea and study ND deeply, we propose Contextual Stochastic Block Model for Homophily (CSBM-H) and define two metrics, Probabilistic Bayes Error (PBE) and negative generalized Jeffreys divergence, to quantify ND. With the metrics, we visualize and analyze how graph filters, node degree distributions and class variances influence ND, and investigate the combined effect of intra- and inter-class ND. Besides, we discovered the mid-homophily pitfall, which occurs widely in graph datasets. Furthermore, we verified that, in real-work tasks, the superiority of GNNs is indeed closely related to both intra- and inter-class ND regardless of homophily levels. Grounded in this observation, we propose a new hypothesis-testing based performance metric beyond homophily, which is non-linear, feature-based and can provide statistical threshold value for GNNs' the superiority. Experiments indicate that it is significantly more effective than the existing homophily metrics on revealing the advantage and disadvantage of graph-aware modes on both synthetic and benchmark real-world datasets.

----

## [0] (Almost) Provable Error Bounds Under Distribution Shift via Disagreement Discrepancy

**Authors**: *Elan Rosenfeld, Saurabh Garg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bacb12bf81e98e2ee0eed953a23c656-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bacb12bf81e98e2ee0eed953a23c656-Abstract-Conference.html)

**Abstract**:

We derive a new, (almost) guaranteed upper bound on the error of deep neural networks under distribution shift using unlabeled test data. Prior methods are either vacuous in practice or accurate on average but heavily underestimate error for a sizeable fraction of shifts. In particular, the latter only give guarantees based on complex continuous measures such as test calibration, which cannot be identified without labels, and are therefore unreliable. Instead, our bound requires a simple, intuitive condition which is well justified by prior empirical works and holds in practice effectively 100\% of the time. The bound is inspired by $\mathcal{H}\Delta\mathcal{H}$-divergence but is easier to evaluate and substantially tighter, consistently providing non-vacuous test error upper bounds. Estimating the bound requires optimizing one multiclass classifier to disagree with another, for which some prior works have used sub-optimal proxy losses; we devise a "disagreement loss" which is theoretically justified and performs better in practice. We expect this loss can serve as a drop-in replacement for future methods which require maximizing multiclass disagreement. Across a wide range of natural and synthetic distribution shift benchmarks, our method gives valid error bounds while achieving average accuracy comparable to—though not better than—competitive estimation baselines.

----

## [0] Schema-learning and rebinding as mechanisms of in-context learning and emergence

**Authors**: *Sivaramakrishnan Swaminathan, Antoine Dedieu, Rajkumar Vasudeva Raju, Murray Shanahan, Miguel Lázaro-Gredilla, Dileep George*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html)

**Abstract**:

In-context learning (ICL) is one of the most powerful and most unexpected capabilities to emerge in recent transformer-based large language models (LLMs). Yet the mechanisms that underlie it are poorly understood. In this paper, we demonstrate that comparable ICL capabilities can be acquired by an alternative sequence prediction learning method using clone-structured causal graphs (CSCGs). Moreover, a key property of CSCGs is that, unlike transformer-based LLMs, they are {\em interpretable}, which considerably simplifies the task of explaining how ICL works. Specifically, we show that it uses a combination of (a) learning template (schema) circuits for pattern completion, (b) retrieving relevant templates in a context-sensitive manner, and (c) rebinding of novel tokens to appropriate slots in the templates. We go on to marshall evidence for the hypothesis that similar mechanisms underlie ICL in LLMs. For example, we find that, with CSCGs as with LLMs, different capabilities emerge at different levels of overparameterization, suggesting that overparameterization helps in learning more complex template (schema) circuits. By showing how ICL can be achieved with small models and datasets, we open up a path to novel architectures, and take a vital step towards a more general understanding of the mechanics behind this important capability.

----

## [0] CAP: Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models

**Authors**: *Denis Kuznedelev, Eldar Kurtic, Elias Frantar, Dan Alistarh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html)

**Abstract**:

Driven by significant improvements in architectural design and training pipelines, computer visionhas recently experienced dramatic progress in terms of accuracy on classic benchmarks such as ImageNet. These highly-accurate models are challenging to deploy,  as they appear harder to compress using standard techniques such as pruning. We address this issue by introducing the Correlation Aware Pruner (CAP), a new unstructured pruning framework which significantly pushes the compressibility limits for state-of-the-art architectures.Our method is based on two technical advancements: a new theoretically-justified pruner, which can handle complex weight correlations accurately and efficiently during the pruning process itself,  and an efficient finetuning procedure for post-compression recovery. We validate our approach via extensive experiments on several modern vision models such as Vision Transformers (ViT), modern CNNs, and ViT-CNN hybrids, showing for the first time that these can be pruned to high sparsity levels (e.g. $\geq 75$%) with low impact on accuracy ($\leq 1$% relative drop). Our approach is also compatible with structured pruning and quantization, and can lead to practical speedups of 1.5 to 2.4x without accuracy loss. To further showcase CAP's accuracy and scalability, we use it to show for the first time that extremely-accurate large vision models, trained via self-supervised techniques,  can also be pruned to moderate sparsities, with negligible accuracy loss.

----

## [0] Your representations are in the network: composable and parallel adaptation for large scale models

**Authors**: *Yonatan Dukler, Alessandro Achille, Hao Yang, Varsha Vivek, Luca Zancato, Benjamin Bowman, Avinash Ravichandran, Charless C. Fowlkes, Ashwin Swaminathan, Stefano Soatto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5be3783ea9d43d7add5409c101d87d83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5be3783ea9d43d7add5409c101d87d83-Abstract-Conference.html)

**Abstract**:

We present a framework for transfer learning that efficiently adapts a large base-model by learning lightweight cross-attention modules attached to its intermediate activations.We name our approach InCA (Introspective-Cross-Attention) and show that it can efficiently survey a networkâ€™s representations and identify strong performing adapter models for a downstream task.During training, InCA enables training numerous adapters efficiently and in parallel, isolated from the frozen base model. On the ViT-L/16 architecture, our experiments show that a single adapter, 1.3% of the full model, is able to reach full fine-tuning accuracy on average across 11 challenging downstream classification tasks.Compared with other forms of parameter-efficient adaptation, the isolated nature of the InCA adaptation is computationally desirable for large-scale models. For instance, we adapt ViT-G/14 (1.8B+ parameters) quickly with 20+ adapters in parallel on a single V100 GPU (76% GPU memory reduction) and exhaustively identify its most useful representations.We further demonstrate how the adapters learned by InCA can be incrementally modified or combined for flexible learning scenarios and our approach achieves state of the art performance on the ImageNet-to-Sketch multi-task benchmark.

----

## [0] Learning Energy-based Model via Dual-MCMC Teaching

**Authors**: *Jiali Cui, Tian Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html)

**Abstract**:

This paper studies the fundamental learning problem of the energy-based model (EBM). Learning the EBM can be achieved using the maximum likelihood estimation (MLE), which typically involves the Markov Chain Monte Carlo (MCMC) sampling, such as the Langevin dynamics. However, the noise-initialized Langevin dynamics can be challenging in practice and hard to mix. This motivates the exploration of joint training with the generator model where the generator model serves as a complementary model to bypass MCMC sampling. However, such a method can be less accurate than the MCMC and result in biased EBM learning. While the generator can also serve as an initializer model for better MCMC sampling, its learning can be biased since it only matches the EBM and has no access to empirical training examples. Such biased generator learning may limit the potential of learning the EBM. To address this issue, we present a joint learning framework that interweaves the maximum likelihood learning algorithm for both the EBM and the complementary generator model. In particular, the generator model is learned by MLE to match both the EBM and the empirical data distribution, making it a more informative initializer for MCMC sampling of EBM. Learning generator with observed examples typically requires inference of the generator posterior. To ensure accurate and efficient inference, we adopt the MCMC posterior sampling and introduce a complementary inference model to initialize such latent MCMC sampling. We show that three separate models can be seamlessly integrated into our joint framework through two (dual-) MCMC teaching, enabling effective and efficient EBM learning.

----

## [0] Performance-optimized deep neural networks are evolving into worse models of inferotemporal visual cortex

**Authors**: *Drew Linsley, Ivan F. Rodriguez Rodriguez, Thomas Fel, Michael Arcaro, Saloni Sharma, Margaret S. Livingstone, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bf234ecf83cd77bc5b77a24ba9338b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bf234ecf83cd77bc5b77a24ba9338b0-Abstract-Conference.html)

**Abstract**:

One of the most impactful findings in computational neuroscience over the past decade is that the object recognition accuracy of deep neural networks (DNNs) correlates with their ability to predict neural responses to natural images in the inferotemporal (IT) cortex. This discovery supported the long-held theory that object recognition is a core objective of the visual cortex, and suggested that more accurate DNNs would serve as better models of IT neuron responses to images. Since then, deep learning has undergone a revolution of scale: billion parameter-scale DNNs trained on billions of images are rivaling or outperforming humans at visual tasks including object recognition. Have today's DNNs become more accurate at predicting IT neuron responses to images as they have grown more accurate at object recognition?Surprisingly, across three independent experiments, we find that this is not the case. DNNs have become progressively worse models of IT as their accuracy has increased on ImageNet. To understand why DNNs experience this trade-off and evaluate if they are still an appropriate paradigm for modeling the visual system, we turn to recordings of IT that capture spatially resolved maps of neuronal activity elicited by natural images. These neuronal activity maps reveal that DNNs trained on ImageNet learn to rely on different visual features than those encoded by IT and that this problem worsens as their accuracy increases. We successfully resolved this issue with the neural harmonizer, a plug-and-play training routine for DNNs that aligns their learned representations with humans. Our results suggest that harmonized DNNs break the trade-off between ImageNet accuracy and neural prediction accuracy that assails current DNNs and offer a path to more accurate models of biological vision. Our work indicates that the standard approach for modeling IT with task-optimized DNNs needs revision, and other biological constraints, including human psychophysics data, are needed to accurately reverse-engineer the visual cortex.

----

## [0] Private Federated Frequency Estimation: Adapting to the Hardness of the Instance

**Authors**: *Jingfeng Wu, Wennan Zhu, Peter Kairouz, Vladimir Braverman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bf40077b2bac53399676d33d564ef58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bf40077b2bac53399676d33d564ef58-Abstract-Conference.html)

**Abstract**:

In federated frequency estimation (FFE), multiple clients work together to estimate the frequency of their local data by communicating with a server, while maintaining the security constraint of $\mathtt{secsum}$ where the server can only access the sum of client-held vectors. For FFE with a single communication round, it is known that count sketch is nearly information-theoretically optimal [Chen et al., 2022]. However, when multiple communication rounds are allowed, we propose a new sketch algorithm that is provably more accurate than a naive adaptation of count sketch. Furthermore, we show that both our sketch algorithm and count sketch can achieve better accuracy when the problem instance is simpler. Therefore, we propose a two-phase approach to enable the use of a smaller sketch size for simpler problems. Finally, we provide mechanisms to make our proposed algorithm differentially private. We verify the performance of our methods through experiments conducted on real datasets.

----

## [0] On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

**Authors**: *Federico Errica*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html)

**Abstract**:

Researchers have used nearest neighbor graphs to transform classical machine learning problems on tabular data into node classification tasks to solve with graph representation learning methods. Such artificial structures often reflect the homophily assumption, believed to be a key factor in the performances of deep graph networks. In light of recent results demystifying these beliefs, we introduce a theoretical framework to understand the benefits of Nearest Neighbor (NN) graphs when a graph structure is missing. We formally analyze the Cross-Class Neighborhood Similarity (CCNS), used to empirically evaluate the usefulness of structures, in the context of nearest neighbor graphs. Moreover, we study the class separability induced by deep graph networks on a k-NN graph. Motivated by the theory, our quantitative experiments demonstrate that, under full supervision, employing a k-NN graph offers no benefits compared to a structure-agnostic baseline. Qualitative analyses suggest that our framework is good at estimating the CCNS and hint at k-NN graphs never being useful for such classification tasks under full supervision, thus advocating for the study of alternative graph construction techniques in combination with deep graph networks.

----

## [0] VRA: Variational Rectified Activation for Out-of-distribution Detection

**Authors**: *Mingyu Xu, Zheng Lian, Bin Liu, Jianhua Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c20c00504e0c049ec2370d0cceaf3c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c20c00504e0c049ec2370d0cceaf3c4-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection is critical to building reliable machine learning systems in the open world. Researchers have proposed various strategies to reduce model overconfidence on OOD data. Among them, ReAct is a typical and effective technique to deal with model overconfidence, which truncates high activations to increase the gap between in-distribution and OOD. Despite its promising results, is this technique the best choice? To answer this question, we leverage the variational method to find the optimal operation and verify the necessity of suppressing abnormally low and high activations and amplifying intermediate activations in OOD detection, rather than focusing only on high activations like ReAct. This motivates us to propose a novel technique called ``Variational Rectified Activation (VRA)'', which simulates these suppression and amplification operations using piecewise functions. Experimental results on multiple benchmark datasets demonstrate that our method outperforms existing post-hoc strategies. Meanwhile, VRA is compatible with different scoring functions and network architectures. Our code is available at https://github.com/zeroQiaoba/VRA.

----

## [0] Variational Gaussian processes for linear inverse problems

**Authors**: *Thibault Randrianarisoa, Botond Szabó*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c25c15b5b2fd386ab188a918e54c7d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c25c15b5b2fd386ab188a918e54c7d5-Abstract-Conference.html)

**Abstract**:

By now Bayesian methods are routinely used in practice for solving inverse problems. In inverse problems the parameter or signal of interest is observed only indirectly, as an image of a given map, and the observations are typically further corrupted with noise. Bayes offers a natural way to regularize these problems via the prior distribution and provides a probabilistic solution, quantifying the remaining uncertainty in the problem. However, the computational costs of standard, sampling based Bayesian approaches can be overly large in such complex models. Therefore, in practice variational Bayes is becoming increasingly popular. Nevertheless, the theoretical understanding of these methods is still relatively limited, especially in context of inverse problems.In our analysis we investigate variational Bayesian methods for Gaussian process priors to solve linear inverse problems. We consider both mildly and severely ill-posed inverse problems and work with the popular inducing variable variational Bayes approach proposed by Titsias [Titsias, 2009]. We derive posterior contraction rates for the variational posterior in general settings and show that the minimax estimation rate can be attained by correctly tunned procedures. As specific examples we consider a collection of inverse problems including the heat equation, Volterra operator and Radon transform and inducing variable methods based on population and empirical spectral features.

----

## [0] Self-Supervised Learning with Lie Symmetries for Partial Differential Equations

**Authors**: *Grégoire Mialon, Quentin Garrido, Hannah Lawrence, Danyal Rehman, Yann LeCun, Bobak T. Kiani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c46ae130105fa012da0446126c01d1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c46ae130105fa012da0446126c01d1d-Abstract-Conference.html)

**Abstract**:

Machine learning for differential equations paves the way for computationally efficient alternatives to numerical solvers, with potentially broad impacts in science and engineering. Though current algorithms typically require simulated training data tailored to a given setting, one may instead wish to learn useful information from heterogeneous sources, or from real dynamical systems observations that are messy or incomplete. In this work, we learn general-purpose representations of PDEs from heterogeneous data by implementing joint embedding methods for self-supervised learning (SSL), a framework for unsupervised representation learning that has had notable success in computer vision. Our representation outperforms baseline approaches to invariant tasks, such as regressing the coefficients of a PDE, while also improving the time-stepping performance of neural solvers. We hope that our proposed methodology will prove useful in the eventual development of general-purpose foundation models for PDEs.

----

## [0] TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

**Authors**: *Jialin Chen, Rex Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html)

**Abstract**:

Temporal graphs are widely used to model dynamic systems with time-varying interactions. In real-world scenarios, the underlying mechanisms of generating future interactions in dynamic systems are typically governed by a set of recurring substructures within the graph, known as temporal motifs. Despite the success and prevalence of current temporal graph neural networks (TGNN), it remains uncertain which temporal motifs are recognized as the significant indications that trigger a certain prediction from the model, which is a critical challenge for advancing the explainability and trustworthiness of current TGNNs. To address this challenge, we propose a novel approach, called Temporal Motifs Explainer (TempME), which uncovers the most pivotal temporal motifs guiding the prediction of TGNNs.  Derived from the information bottleneck principle, TempME extracts the most interaction-related motifs while minimizing the amount of contained information to preserve the sparsity and succinctness of the explanation. Events in the explanations generated by TempME are verified to be more spatiotemporally correlated than those of existing approaches, providing more understandable insights. Extensive experiments validate the superiority of TempME, with up to 8.21% increase in terms of explanation accuracy across six real-world datasets and up to 22.96% increase in boosting the prediction Average Precision of current TGNNs.

----

## [0] YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus

**Authors**: *David Uthus, Garrett Tanzer, Manfred Georg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning for sign languages is bottlenecked by data. In this paper, we present YouTube-ASL, a large-scale, open-domain corpus of American Sign Language (ASL) videos and accompanying English captions drawn from YouTube. With ~1000 hours of videos and >2500 unique signers, YouTube-ASL is ~3x as large and has ~10x as many unique signers as the largest prior ASL dataset. We train baseline models for ASL to English translation on YouTube-ASL and evaluate them on How2Sign, where we achieve a new fine-tuned state of the art of 12.397 BLEU and, for the first time, nontrivial zero-shot results.

----

## [0] Finite-Time Analysis of Whittle Index based Q-Learning for Restless Multi-Armed Bandits with Neural Network Function Approximation

**Authors**: *Guojun Xiong, Jian Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c7c66dfc9f93f0c738947f3b1c13832-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c7c66dfc9f93f0c738947f3b1c13832-Abstract-Conference.html)

**Abstract**:

Whittle index policy is a heuristic to the intractable restless multi-armed bandits (RMAB) problem. Although it is provably asymptotically optimal, finding Whittle indices remains difficult.  In this paper, we present Neural-Q-Whittle, a Whittle index based Q-learning algorithm for RMAB with neural network function approximation, which is an example of  nonlinear two-timescale stochastic approximation with Q-function values updated on a faster timescale and Whittle indices on a slower timescale.  Despite the empirical success of deep Q-learning, the non-asymptotic convergence rate of Neural-Q-Whittle, which couples neural networks with two-timescale Q-learning largely remains unclear.  This paper provides a finite-time analysis of Neural-Q-Whittle, where data are generated from a Markov chain, and Q-function is approximated by a ReLU neural network. Our analysis leverages a Lyapunov drift approach to capture the evolution of two coupled parameters, and the nonlinearity in value function approximation further requires us to characterize the approximation error.  Combing these provide Neural-Q-Whittle  with $\mathcal{O}(1/k^{2/3})$ convergence rate, where $k$ is the number of iterations.

----

## [0] Adapting to Continuous Covariate Shift via Online Density Ratio Estimation

**Authors**: *Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html)

**Abstract**:

Dealing with distribution shifts is one of the central challenges for modern machine learning. One fundamental situation is the covariate shift, where the input distributions of data change from the training to testing stages while the input-conditional output distribution remains unchanged. In this paper, we initiate the study of a more challenging scenario --- continuous covariate shift --- in which the test data appear sequentially, and their distributions can shift continuously. Our goal is to adaptively train the predictor such that its prediction risk accumulated over time can be minimized. Starting with the importance-weighted learning, we theoretically show the method works effectively if the time-varying density ratios of test and train inputs can be accurately estimated. However, existing density ratio estimation methods would fail due to data scarcity at each time step. To this end, we propose an online density ratio estimation method that can appropriately reuse historical information. Our method is proven to perform well by enjoying a dynamic regret bound, which finally leads to an excess risk guarantee for the predictor. Empirical results also validate the effectiveness.

----

## [0] Hierarchical Integration Diffusion Model for Realistic Image Deblurring

**Authors**: *Zheng Chen, Yulun Zhang, Ding Liu, Bin Xia, Jinjin Gu, Linghe Kong, Xin Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html)

**Abstract**:

Diffusion models (DMs) have recently been introduced in image deblurring and exhibited promising performance, particularly in terms of details reconstruction. However, the diffusion model requires a large number of inference iterations to recover the clean image from pure Gaussian noise, which consumes massive computational resources. Moreover, the distribution synthesized by the diffusion model is often misaligned with the target results, leading to restrictions in distortion-based metrics. To address the above issues, we propose the Hierarchical Integration Diffusion Model (HI-Diff), for realistic image deblurring. Specifically, we perform the DM in a highly compacted latent space to generate the prior feature for the deblurring process. The deblurring process is implemented by a regression-based method to obtain better distortion accuracy. Meanwhile, the highly compact latent space ensures the efficiency of the DM. Furthermore, we design the hierarchical integration module to fuse the prior into the regression-based model from multiple scales, enabling better generalization in complex blurry scenarios. Comprehensive experiments on synthetic and real-world blur datasets demonstrate that our HI-Diff outperforms state-of-the-art methods. Code and trained models are available at https://github.com/zhengchen1999/HI-Diff.

----

## [0] Efficient Beam Tree Recursion

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cf93940e37f7a7877cd57b6dba6b7ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cf93940e37f7a7877cd57b6dba6b7ab-Abstract-Conference.html)

**Abstract**:

Beam Tree Recursive Neural Network (BT-RvNN) was recently proposed as an extension of Gumbel Tree RvNN and it was shown to achieve state-of-the-art length generalization performance in ListOps while maintaining comparable performance on other tasks. However, although better than previous approaches in terms of memory usage, BT-RvNN can be still exorbitantly expensive. In this paper, we identify the main bottleneck in BT-RvNN's memory usage to be the entanglement of the scorer function and the recursive cell function. We propose strategies to remove this bottleneck and further simplify its memory usage. Overall, our strategies not only reduce the memory usage of BT-RvNN by $10-16$ times but also create a new state-of-the-art in ListOps while maintaining similar performance in other tasks. In addition, we also propose a strategy to utilize the induced latent-tree node representations produced by BT-RvNN to turn BT-RvNN from a sentence encoder of the form $f:\mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{d}$ into a token contextualizer of the form  $f:\mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}$. Thus, our proposals not only open up a path for further scalability of RvNNs but also standardize a way to use BT-RvNNs as another building block in the deep learning toolkit that can be easily stacked or interfaced with other popular models such as Transformers and Structured State Space models. Our code is available at the link: https://github.com/JRC1995/BeamRecursionFamily.

----

## [0] Holistic Transfer: Towards Non-Disruptive Fine-Tuning with Partial Target Data

**Authors**: *Cheng-Hao Tu, Hong-You Chen, Zheda Mai, Jike Zhong, Vardaan Pahuja, Tanya Y. Berger-Wolf, Song Gao, Charles V. Stewart, Yu Su, Wei-Lun Chao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html)

**Abstract**:

We propose a learning problem involving adapting a pre-trained source model to the target domain for classifying all classes that appeared in the source data, using target data that covers only a partial label space. This problem is practical, as it is unrealistic for the target end-users to collect data for all classes prior to adaptation. However, it has received limited attention in the literature. To shed light on this issue, we construct benchmark datasets and conduct extensive experiments to uncover the inherent challenges. We found a dilemma --- on the one hand, adapting to the new target domain is important to claim better performance; on the other hand, we observe that preserving the classification accuracy of classes missing in the target adaptation data is highly challenging, let alone improving them. To tackle this, we identify two key directions: 1) disentangling domain gradients from classification gradients, and 2) preserving class relationships. We present several effective solutions that maintain the accuracy of the missing classes and enhance the overall performance, establishing solid baselines for holistic transfer of pre-trained models with partial target data.

----

## [0] Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition

**Authors**: *Divin Yan, Gengchen Wei, Chen Yang, Shengzhong Zhang, Zengfeng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d1233f819202ade06023346df80a6d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d1233f819202ade06023346df80a6d2-Abstract-Conference.html)

**Abstract**:

This paper introduces a new approach to address the issue of class imbalance in graph neural networks (GNNs) for learning on graph-structured data. Our approach integrates imbalanced node classification and Bias-Variance Decomposition, establishing a theoretical framework that closely relates data imbalance to model variance. We also leverage graph augmentation technique to estimate the variance and design a regularization term to alleviate the impact of imbalance. Exhaustive tests are conducted on multiple benchmarks, including naturally imbalanced datasets and public-split class-imbalanced datasets, demonstrating that our approach outperforms state-of-the-art methods in various imbalanced scenarios. This work provides a novel theoretical perspective for addressing the problem of imbalanced node classification in GNNs.

----

## [0] Practical Equivariances via Relational Conditional Neural Processes

**Authors**: *Daolang Huang, Manuel Haussmann, Ulpu Remes, St John, Grégoire Clarté, Kevin Sebastian Luck, Samuel Kaski, Luigi Acerbi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d1a382162cb5ed326f1d3dbbfac4c82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d1a382162cb5ed326f1d3dbbfac4c82-Abstract-Conference.html)

**Abstract**:

Conditional Neural Processes (CNPs) are a class of metalearning models popular for combining the runtime efficiency of amortized inference with reliable uncertainty quantification. Many relevant machine learning tasks, such as in spatio-temporal modeling, Bayesian Optimization and continuous control, inherently contain equivariances – for example to translation – which the model can exploit for maximal performance. However, prior attempts to include equivariances in CNPs do not scale effectively beyond two input dimensions. In this work, we propose Relational Conditional Neural Processes (RCNPs), an effective approach to incorporate equivariances into any neural process model. Our proposed method extends the applicability and impact of equivariant neural processes to higher dimensions. We empirically demonstrate the competitive performance of RCNPs on a large array of tasks naturally containing equivariances.

----

## [0] FourierHandFlow: Neural 4D Hand Representation Using Fourier Query Flow

**Authors**: *Jihyun Lee, Junbong Jang, Donghwan Kim, Minhyuk Sung, Tae-Kyun Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)

**Abstract**:

Recent 4D shape representations model continuous temporal evolution of implicit shapes by (1) learning query flows without leveraging shape and articulation priors or (2) decoding shape occupancies separately for each time value. Thus, they do not effectively capture implicit correspondences between articulated shapes or regularize jittery temporal deformations. In this work, we present FourierHandFlow, which is a spatio-temporally continuous representation for human hands that combines a 3D occupancy field with articulation-aware query flows represented as Fourier series. Given an input RGB sequence, we aim to learn a fixed number of Fourier coefficients for each query flow to guarantee smooth and continuous temporal shape dynamics. To effectively model spatio-temporal deformations of articulated hands, we compose our 4D representation based on two types of Fourier query flow: (1) pose flow that models query dynamics influenced by hand articulation changes via implicit linear blend skinning and (2) shape flow that models query-wise displacement flow. In the experiments, our method achieves state-of-the-art results on video-based 4D reconstruction while being computationally more efficient than the existing 3D/4D implicit shape representations. We additionally show our results on motion inter- and extrapolation and texture transfer using the learned correspondences of implicit shapes. To the best of our knowledge, FourierHandFlow is the first neural 4D continuous hand representation learned from RGB videos. The code will be publicly accessible.

----

## [0] Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms

**Authors**: *Akifumi Wachi, Wataru Hashimoto, Xun Shen, Kazumune Hashimoto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d4cd12ef6efedbf26b69b410f1f7d67-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d4cd12ef6efedbf26b69b410f1f7d67-Abstract-Conference.html)

**Abstract**:

Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.

----

## [0] RevColV2: Exploring Disentangled Representations in Masked Image Modeling

**Authors**: *Qi Han, Yuxuan Cai, Xiangyu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d56e69c317429945785ede86c00b44e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d56e69c317429945785ede86c00b44e-Abstract-Conference.html)

**Abstract**:

Masked image modeling (MIM) has become a prevalent pre-training setup for vision foundation models and attains promising performance. Despite its success, existing MIM methods discard the decoder network during downstream applica- tions, resulting in inconsistent representations between pre-training and fine-tuning and can hamper downstream task performance. In this paper, we propose a new architecture, RevColV2, which tackles this issue by keeping the entire autoen- coder architecture during both pre-training and fine-tuning. The main body of RevColV2 contains bottom-up columns and top-down columns, between which information is reversibly propagated and gradually disentangled. Such design enables our architecture with the nice property: maintaining disentangled low-level and semantic information at the end of the network in MIM pre-training. Our experimental results suggest that a foundation model with decoupled features can achieve competitive performance across multiple downstream vision tasks such as image classification, semantic segmentation and object detection. For exam- ple, after intermediate fine-tuning on ImageNet-22K dataset, RevColV2-L attains 88.4\% top-1 accuracy on ImageNet-1K classification and 58.6 mIoU on ADE20K semantic segmentation. With extra teacher and large scale dataset, RevColv2-L achieves 62.1 APbox on COCO detection and 60.4 mIoU on ADE20K semantic segmentation.

----

## [0] Mass-Producing Failures of Multimodal Systems with Language Models

**Authors**: *Shengbang Tong, Erik Jones, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d570ed1708bbe19cb60f7a7aff60575-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d570ed1708bbe19cb60f7a7aff60575-Abstract-Conference.html)

**Abstract**:

Deployed multimodal models can fail in ways that evaluators did not anticipate. In order to find these failures before deployment, we introduce MultiMon, a system that automatically identifies systematic failures---generalizable, natural-language descriptions that describe categories of individual failures. To uncover systematic failures, MultiMon scrapes for examples of erroneous agreement: inputs that produce the same output, but should not. It then prompts a language model to identify common categories and describe them in natural language. We use MultiMon to find 14 systematic failures (e.g."ignores quantifiers'') of the CLIP text-encoder, each comprising hundreds of distinct inputs (e.g."a shelf with a few/many books''). Because CLIP is the backbone for most state-of-the-art multimodal models, these inputs produce failures in Midjourney 5.1, DALL-E, VideoFusion, and others. MultiMon can also steer towards failures relevant to specific use cases, such as self-driving cars. We see MultiMon as a step towards evaluation that autonomously explores the long-tail of potential system failures.

----

## [0] STXD: Structural and Temporal Cross-Modal Distillation for Multi-View 3D Object Detection

**Authors**: *Sujin Jang, Dae Ung Jo, Sung Ju Hwang, Dongwook Lee, Daehyun Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html)

**Abstract**:

3D object detection (3DOD) from multi-view images is an economically appealing alternative to expensive LiDAR-based detectors, but also an extremely challenging task due to the absence of precise spatial cues. Recent studies have leveraged the teacher-student paradigm for cross-modal distillation, where a strong LiDAR-modality teacher transfers useful knowledge to a multi-view-based image-modality student. However, prior approaches have only focused on minimizing global distances between cross-modal features, which may lead to suboptimal knowledge distillation results. Based on these insights, we propose a novel structural and temporal cross-modal knowledge distillation (STXD) framework for multi-view 3DOD. First, STXD reduces redundancy of the feature components of the student by regularizing the cross-correlation of cross-modal features, while maximizing their similarities. Second, to effectively transfer temporal knowledge, STXD encodes temporal relations of features across a sequence of frames via similarity maps. Lastly, STXD also adopts a response distillation method to further enhance the quality of knowledge distillation at the output-level. Our extensive experiments demonstrate that STXD significantly improves the NDS and mAP of the based student detectors by 2.8%~4.5% on the nuScenes testing dataset.

----

## [0] Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks

**Authors**: *Micah Goldblum, Hossein Souri, Renkun Ni, Manli Shu, Viraj Prabhu, Gowthami Somepalli, Prithvijit Chattopadhyay, Mark Ibrahim, Adrien Bardes, Judy Hoffman, Rama Chellappa, Andrew Gordon Wilson, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d9571470bb750f0e2325a030016f63f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d9571470bb750f0e2325a030016f63f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Neural network based computer vision systems are typically built on a backbone, a pretrained or randomly initialized feature extractor.  Several years ago, the default option was an ImageNet-trained convolutional neural network.  However, the recent past has seen the emergence of countless backbones pretrained using various algorithms and datasets. While this abundance of choice has led to performance increases for a range of systems, it is difficult for practitioners to make informed decisions about which backbone to choose.  Battle of the Backbones (BoB) makes this choice easier by benchmarking a diverse suite of pretrained models, including vision-language models, those trained via self-supervised learning, and the Stable Diffusion backbone, across a diverse set of computer vision tasks ranging from classification to object detection to OOD generalization and more.  Furthermore, BoB sheds light on promising directions for the research community to advance computer vision by illuminating strengths and weakness of existing approaches through a comprehensive analysis conducted on more than 1500 training runs.  While vision transformers (ViTs) and self-supervised learning (SSL) are increasingly popular, we find that convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks among the models we consider. Moreover, in apples-to-apples comparisons on the same architectures and similarly sized pretraining datasets, we find that SSL backbones are highly competitive, indicating that future works should perform SSL pretraining with advanced architectures and larger pretraining datasets.  We release the raw results of our experiments along with code that allows researchers to put their own backbones through the gauntlet here: https://github.com/hsouri/Battle-of-the-Backbones.

----

## [0] Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

**Authors**: *Florian Seligmann, Philipp Becker, Michael Volpp, Gerhard Neumann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html)

**Abstract**:

Bayesian deep learning (BDL) is a promising approach to achieve well-calibrated predictions on distribution-shifted data. Nevertheless, there exists no large-scale survey that evaluates recent SOTA methods on diverse, realistic, and challenging benchmark tasks in a systematic manner. To provide a clear picture of the current state of BDL research, we evaluate modern BDL algorithms on real-world datasets from the WILDS collection containing challenging classification and regression tasks, with a focus on generalization capability and calibration under distribution shift. We compare the algorithms on a wide range of large, convolutional and transformer-based neural network architectures. In particular, we investigate a signed version of the expected calibration error that reveals whether the methods are over- or underconfident, providing further insight into the behavior of the methods. Further, we provide the first systematic evaluation of BDL for fine-tuning large pre-trained models, where training from scratch is prohibitively expensive. Finally, given the recent success of Deep Ensembles, we extend popular single-mode posterior approximations to multiple modes by the use of ensembles.   While we find that ensembling single-mode approximations generally improves the generalization capability and calibration of the models by a significant margin, we also identify a failure mode of ensembles when finetuning large transformer-based language models.  In this setting, variational inference based approaches such as last-layer Bayes By Backprop outperform other methods in terms of accuracy by a large margin, while modern approximate inference algorithms such as SWAG achieve the best calibration.

----

## [0] (S)GD over Diagonal Linear Networks: Implicit bias, Large Stepsizes and Edge of Stability

**Authors**: *Mathieu Even, Scott Pesme, Suriya Gunasekar, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5da6ce80e97671b70c01a2e703b868b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5da6ce80e97671b70c01a2e703b868b3-Abstract-Conference.html)

**Abstract**:

In this paper, we investigate the impact of stochasticity and large stepsizes on the implicit regularisation of gradient descent (GD) and stochastic gradient descent (SGD) over $2$-layer diagonal linear networks. We prove the convergence of GD and SGD with macroscopic stepsizes in an overparametrised regression setting and characterise their solutions through an implicit regularisation problem. Our crisp characterisation leads to qualitative insights about the impact of stochasticity and stepsizes on the recovered solution. Specifically, we show that large stepsizes consistently benefit SGD for sparse regression problems, while they can hinder the recovery of sparse solutions for GD. These effects are magnified for stepsizes in a tight window just below the divergence threshold, in the ``edge of stability'' regime. Our findings are supported by experimental results.

----

## [0] Optimal Preconditioning and Fisher Adaptive Langevin Sampling

**Authors**: *Michalis K. Titsias*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5da6d5818a156791090c875abeca3cf8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5da6d5818a156791090c875abeca3cf8-Abstract-Conference.html)

**Abstract**:

We define  an optimal preconditioning for the Langevin diffusion by analytically optimizing the expected squared jumped distance. This yields as the optimal preconditioning an inverse Fisher information covariance matrix, where the covariance matrix is computed as the outer product of log target gradients averaged under the target.  We apply this result to the Metropolis adjusted Langevin algorithm (MALA)  and derive a computationally efficient adaptive MCMC scheme that learns the preconditioning from the history of gradients produced as the algorithm runs. We show in several experiments that the proposed algorithm is very robust in high dimensions and significantly outperforms other methods, including a closely related adaptive MALA scheme that learns the preconditioning with standard adaptive MCMC as well as the position-dependent  Riemannian manifold MALA sampler.

----

## [0] AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking

**Authors**: *Yang Yu, Qi Liu, Kai Zhang, Yuren Zhang, Chao Song, Min Hou, Yuqing Yuan, Zhihao Ye, Zaixi Zhang, Sanshi Lei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html)

**Abstract**:

User modeling, which aims to capture users' characteristics or interests, heavily relies on task-specific labeled data and suffers from the data sparsity issue. Several recent studies tackled this problem by pre-training the user model on massive user behavior sequences with a contrastive learning task. Generally, these methods assume different views of the same behavior sequence constructed via data augmentation are semantically consistent, i.e., reflecting similar characteristics or interests of the user, and thus maximizing their agreement in the feature space. However, due to the diverse interests and heavy noise in user behaviors, existing augmentation methods tend to lose certain characteristics of the user or introduce noisy behaviors. Thus, forcing the user model to directly maximize the similarity between the augmented views may result in a negative transfer. To this end, we propose to replace the contrastive learning task with a new pretext task: Augmentation-Adaptive SelfSupervised Ranking (AdaptSSR), which alleviates the requirement of semantic consistency between the augmented views while pre-training a discriminative user model. Specifically, we adopt a multiple pairwise ranking loss which trains the user model to capture the similarity orders between the implicitly augmented view, the explicitly augmented view, and views from other users. We further employ an in-batch hard negative sampling strategy to facilitate model training. Moreover, considering the distinct impacts of data augmentation on different behavior sequences, we design an augmentation-adaptive fusion mechanism to automatically adjust the similarity order constraint applied to each sample based on the estimated similarity between the augmented views. Extensive experiments on both public and industrial datasets with six downstream tasks verify the effectiveness of AdaptSSR.

----

## [0] Anytime Model Selection in Linear Bandits

**Authors**: *Parnian Kassraie, Nicolas Emmenegger, Andreas Krause, Aldo Pacchiano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e11d23b18261d1b76d14da7a285fd0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e11d23b18261d1b76d14da7a285fd0c-Abstract-Conference.html)

**Abstract**:

Model selection in the context of bandit optimization is a challenging problem, as it requires balancing exploration and exploitation not only for action selection, but also for model selection. One natural approach is to rely on online learning algorithms that treat different models as experts. Existing methods, however, scale poorly ($\mathrm{poly}M$) with the number of models $M$ in terms of their regret.Our key insight is that, for model selection in linear bandits, we can emulate full-information feedback to the online learner with a favorable bias-variance trade-off. This allows us to develop ALEXP, which has an exponentially improved ($\log M$) dependence on $M$ for its regret.ALEXP has anytime guarantees on its regret, and neither requires knowledge of the horizon $n$, nor relies on an initial purely exploratory stage.Our approach utilizes a  novel time-uniform analysis of the Lasso, establishing a new connection between online learning and high-dimensional statistics.

----

## [0] Towards Personalized Federated Learning via Heterogeneous Model Reassembly

**Authors**: *Jiaqi Wang, Xingyi Yang, Suhan Cui, Liwei Che, Lingjuan Lyu, Dongkuan Xu, Fenglong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html)

**Abstract**:

This paper focuses on addressing the practical yet challenging problem of model heterogeneity in federated learning, where clients possess models with different network structures. To track this problem, we propose a novel framework called pFedHR, which leverages heterogeneous model reassembly to achieve personalized federated learning. In particular, we approach the problem of heterogeneous model personalization as a model-matching optimization task on the server side. Moreover, pFedHR automatically and dynamically generates informative and diverse personalized candidates with minimal human intervention. Furthermore, our proposed heterogeneous model reassembly technique mitigates the adverse impact introduced by using public data with different distributions from the client data to a certain extent. Experimental results demonstrate that pFedHR outperforms baselines on three datasets under both IID and Non-IID settings. Additionally, pFedHR effectively reduces the adverse impact of using different public data and dynamically generates diverse personalized models in an automated manner.

----

## [0] Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning

**Authors**: *Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)

**Abstract**:

Large language models have shown astonishing performance on a wide range of reasoning tasks. In this paper, we investigate whether they could reason about real-world events and help improve the prediction performance of event sequence models. We design LAMP, a framework that integrates a large language model in event prediction. Particularly, the language model performs abductive reasoning to assist an event sequence model: the event model proposes predictions on future events given the past; instructed by a few expert-annotated demonstrations, the language model learns to suggest possible causes for each proposal; a search module finds out the previous events that match the causes; a scoring function learns to examine whether the retrieved events could actually cause the proposal. Through extensive experiments on several challenging real-world datasets, we demonstrate that our framework---thanks to the reasoning capabilities of large language models---could significantly outperform the state-of-the-art event sequence models.

----

## [0] Complexity Matters: Rethinking the Latent Space for Generative Modeling

**Authors**: *Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html)

**Abstract**:

In generative modeling, numerous successful approaches leverage a low-dimensional latent space, e.g., Stable Diffusion models the latent space induced by an encoder and generates images through a paired decoder. Although the selection of the latent space is empirically pivotal, determining the optimal choice and the process of identifying it remain unclear. In this study, we aim to shed light on this under-explored topic by rethinking the latent space from the perspective of model complexity. Our investigation starts with the classic generative adversarial networks (GANs). Inspired by the GAN training objective, we propose a novel "distance" between the latent and data distributions, whose minimization coincides with that of the generator complexity. The minimizer of this distance is characterized as the optimal data-dependent latent that most effectively capitalizes on the generator's capacity. Then, we consider parameterizing such a latent distribution by an encoder network and propose a two-stage training strategy called Decoupled Autoencoder (DAE), where the encoder is only updated in the first stage with an auxiliary decoder and then frozen in the second stage while the actual decoder is being trained. DAE can improve the latent distribution and as a result, improve the generative performance. Our theoretical analyses are corroborated by comprehensive experiments on various models such as VQGAN and Diffusion Transformer, where our modifications yield significant improvements in sample quality with decreased model complexity.

----

## [0] Riemannian stochastic optimization methods avoid strict saddle points

**Authors**: *Ya-Ping Hsieh, Mohammad Reza Karimi Jaghargh, Andreas Krause, Panayotis Mertikopoulos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e809ba53f34d9170386ebfc8b60300f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e809ba53f34d9170386ebfc8b60300f-Abstract-Conference.html)

**Abstract**:

Many modern machine learning applications - from online principal component analysis to covariance matrix identification and dictionary learning - can be formulated as minimization problems on Riemannian manifolds, typically solved with a Riemannian stochastic gradient method (or some variant thereof). However, in many cases of interest, the resulting minimization problem is _not_ geodesically convex, so the convergence of the chosen solver to a desirable solution - i.e., a local minimizer - is by no means guaranteed. In this paper, we study precisely this question, that is, whether stochastic Riemannian optimization algorithms are guaranteed to avoid saddle points with probability $1$. For generality, we study a family of retraction-based methods which, in addition to having a potentially much lower per-iteration cost relative to Riemannian gradient descent, include other widely used algorithms, such as natural policy gradient methods and mirror descent in ordinary convex spaces. In this general setting, we show that, under mild assumptions for the ambient manifold and the oracle providing gradient information, the policies under study avoid strict saddle points / submanifolds with probability $1$, from any initial condition. This result provides an important sanity check for the use of gradient methods on manifolds as it shows that, almost always, the end state of a stochastic Riemannian algorithm can only be a local minimizer.

----

## [0] Toward Better PAC-Bayes Bounds for Uniformly Stable Algorithms

**Authors**: *Sijia Zhou, Yunwen Lei, Ata Kabán*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e8309c9ca683e11672e3dbcd4b87776-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e8309c9ca683e11672e3dbcd4b87776-Abstract-Conference.html)

**Abstract**:

We give sharper bounds for uniformly stable randomized algorithms in a PAC-Bayesian framework, which improve the existing results by up to a factor of $\sqrt{n}$ (ignoring a log factor), where $n$ is the sample size. The key idea is to bound the moment generating function of the generalization gap using concentration of weakly dependent random variables due to Bousquet et al (2020). We introduce an assumption of sub-exponential stability parameter, which allows a general treatment that we instantiate in two applications: stochastic gradient descent and randomized coordinate descent. Our results eliminate the requirement of strong convexity from previous results, and hold for non-smooth convex problems.

----

## [0] Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models

**Authors**: *Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, Rongrong Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html)

**Abstract**:

Recently,  growing interest has been aroused in extending the multimodal capability of large language models (LLMs), e.g., vision-language (VL) learning, which is regarded as the next milestone of artificial general intelligence. However, existing solutions are prohibitively expensive, which not only need to optimize excessive parameters, but also require another large-scale pre-training before VL instruction tuning. In this paper, we propose a novel and  affordable  solution for the effective VL adaption of LLMs, called  Mixture-of-Modality Adaptation (MMA).  Instead of using large neural networks to connect the image encoder and LLM, MMA adopts lightweight modules, i.e., adapters, to  bridge the gap between LLMs and VL tasks, which also enables the joint optimization of the image and language models. Meanwhile, MMA is also equipped with a routing algorithm to help LLMs achieve an  automatic  shift between single- and multi-modal instructions without compromising their ability of natural language understanding.  To validate MMA, we apply it to a recent LLM called LLaMA and term this formed large vision-language instructed model as LaVIN.  To validate MMA and LaVIN, we conduct extensive experiments  under two setups, namely  multimodal science question answering and multimodal dialogue. The experimental results not only demonstrate the competitive performance and the superior training efficiency  of LaVIN  than existing multimodal LLMs, but also confirm  its  great potential   as a general-purpose chatbot. More importantly, the actual expenditure of LaVIN is extremely cheap, e.g., only 1.4 training hours with 3.8M trainable parameters, greatly confirming the effectiveness of MMA.   Our  code is anonymously released at:  https://anonymous.4open.science/r/LaVIN--1067.

----

## [0] GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

**Authors**: *Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

With a long history of traditional Graph Anomaly Detection (GAD) algorithms and recently popular Graph Neural Networks (GNNs), it is still not clear (1) how they perform under a standard comprehensive setting, (2) whether GNNs can outperform traditional algorithms such as tree ensembles, and (3) how about their efficiency on large-scale graphs. In response, we introduce GADBench---a benchmark tool dedicated to supervised anomalous node detection in static graphs. GADBench facilitates a detailed comparison across 29 distinct models on ten real-world GAD datasets, encompassing thousands to millions (~6M) nodes. Our main finding is that tree ensembles with simple neighborhood aggregation can outperform the latest GNNs tailored for the GAD task. We shed light on the current progress of GAD, setting a robust groundwork for subsequent investigations in this domain. GADBench is open-sourced at https://github.com/squareRoot3/GADBench.

----

## [0] Brain encoding models based on multimodal transformers can transfer across language and vision

**Authors**: *Jerry Tang, Meng Du, Vy A. Vo, Vasudev Lal, Alexander Huth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html)

**Abstract**:

Encoding models have been used to assess how the human brain represents concepts in language and vision. While language and vision rely on similar concept representations, current encoding models are typically trained and tested on brain responses to each modality in isolation. Recent advances in multimodal pretraining have produced transformers that can extract aligned representations of concepts in language and vision. In this work, we used representations from multimodal transformers to train encoding models that can transfer across fMRI responses to stories and movies. We found that encoding models trained on brain responses to one modality can successfully predict brain responses to the other modality, particularly in cortical regions that represent conceptual meaning. Further analysis of these encoding models revealed shared semantic dimensions that underlie concept representations in language and vision. Comparing encoding models trained using representations from multimodal and unimodal transformers, we found that multimodal transformers learn more aligned representations of concepts in language and vision. Our results demonstrate how multimodal transformers can provide insights into the brainâ€™s capacity for multimodal processing.

----

## [0] PointGPT: Auto-regressively Generative Pre-training from Point Clouds

**Authors**: *Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) based on the generative pre-training transformer (GPT) have demonstrated remarkable effectiveness across a diverse range of downstream tasks. Inspired by the advancements of the GPT, we present PointGPT, a novel approach that extends the concept of GPT to point clouds, addressing the challenges associated with disorder properties, low information density, and task gaps. Specifically, a point cloud auto-regressive generation task is proposed to pre-train transformer models. Our method partitions the input point cloud into multiple point patches and arranges them in an ordered sequence based on their spatial proximity. Then, an extractor-generator based transformer decode, with a dual masking strategy, learns latent representations conditioned on the preceding point patches, aiming to predict the next one in an auto-regressive manner. To explore scalability and enhance performance, a larger pre-training dataset is collected. Additionally, a subsequent post-pre-training stage is introduced, incorporating a labeled hybrid dataset. Our scalable approach allows for learning high-capacity models that generalize well, achieving state-of-the-art performance on various downstream tasks. In particular, our approach achieves classification accuracies of 94.9% on the ModelNet40 dataset and 93.4% on the ScanObjectNN dataset, outperforming all other transformer models. Furthermore, our method also attains new state-of-the-art accuracies on all four few-shot learning benchmarks. Codes are available at https://github.com/CGuangyan-BIT/PointGPT.

----

## [0] Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning

**Authors**: *Xiaoqian Wu, Yonglu Li, Jianhua Sun, Cewu Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)

**Abstract**:

Human reasoning can be understood as a cooperation between the intuitive, associative "System-1'' and the deliberative, logical "System-2''. For existing System-1-like methods in visual activity understanding, it is crucial to integrate System-2 processing to improve explainability, generalization, and data efficiency. One possible path of activity reasoning is building a symbolic system composed of symbols and rules, where one rule connects multiple symbols, implying human knowledge and reasoning abilities.Previous methods have made progress, but are defective with limited symbols from handcraft and limited rules from visual-based annotations, failing to cover the complex patterns of activities and lacking compositional generalization. To overcome the defects, we propose a new symbolic system with two ideal important properties: broad-coverage symbols and rational rules. Collecting massive human knowledge via manual annotations is expensive to instantiate this symbolic system. Instead, we leverage the recent advancement of LLMs (Large Language Models) as an approximation of the two ideal properties, i.e., Symbols from Large Language Models (Symbol-LLM). Then, given an image, visual contents from the images are extracted andchecked as symbols and activity semantics are reasoned out based on rules via fuzzy logic calculation.Our method shows superiority in extensive activity understanding tasks. Code and data are available at https://mvig-rhos.com/symbol_llm.

----

## [0] Non-Convex Bilevel Optimization with Time-Varying Objective Functions

**Authors**: *Sen Lin, Daouda Sow, Kaiyi Ji, Yingbin Liang, Ness B. Shroff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ee60ca5686bbcf756e56a6c75e66f32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ee60ca5686bbcf756e56a6c75e66f32-Abstract-Conference.html)

**Abstract**:

Bilevel optimization has become a powerful tool in a wide variety of machine learning problems. However, the current nonconvex bilevel optimization considers an offline dataset and static functions, which may not work well in emerging online applications with streaming data and time-varying functions. In this work, we study online bilevel optimization (OBO) where the functions can be time-varying and the agent continuously updates the decisions with online streaming data. To deal with the function variations and the unavailability of the true hypergradients in OBO, we propose a single-loop online bilevel optimizer with window averaging (SOBOW), which updates the outer-level decision based on a window average of the most recent hypergradient estimations stored in the memory. Compared to existing algorithms, SOBOW is computationally efficient and does not need to know previous functions. To handle the unique technical difficulties rooted in single-loop update and function variations for OBO, we develop a novel analytical technique that disentangles the complex couplings between decision variables, and carefully controls the hypergradient estimation error. We show that SOBOW can achieve a sublinear bilevel local regret under mild conditions. Extensive experiments across multiple domains corroborate the effectiveness of SOBOW.

----

## [0] Online Pricing for Multi-User Multi-Item Markets

**Authors**: *Yigit Efe Erginbas, Thomas A. Courtade, Kannan Ramchandran, Soham Phade*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html)

**Abstract**:

Online pricing has been the focus of extensive research in recent years, particularly in the context of selling an item to sequentially arriving users. However, what if a provider wants to maximize revenue by selling multiple items to multiple users in each round? This presents a complex problem, as the provider must intelligently offer the items to those users who value them the most without exceeding their highest acceptable prices. In this study, we tackle this challenge by designing online algorithms that can efficiently offer and price items while learning user valuations from accept/reject feedback. We focus on three user valuation models (fixed valuations, random experiences, and random valuations) and provide algorithms with nearly-optimal revenue regret guarantees. In particular, for any market setting with $N$ users, $M$ items, and load $L$ (which roughly corresponds to the maximum number of simultaneous allocations possible), our algorithms achieve regret of order $O(NM\log\log(LT))$ under fixed valuations model, $\widetilde{O}(\sqrt{NMLT})$ under random experiences model and $\widetilde{O}(\sqrt{NMLT})$ under random valuations model in $T$ rounds.

----

## [0] Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost

**Authors**: *Yu-Jie Zhang, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html)

**Abstract**:

This paper investigates the logistic bandit problem, a variant of the generalized linear bandit model that utilizes a logistic model to depict the feedback from an action. While most existing research focuses on the binary logistic bandit problem, the multinomial case, which considers more than two possible feedback values, offers increased practical relevance and adaptability for use in complex decision-making problems such as reinforcement learning. In this paper, we provide an algorithm that enjoys both statistical and computational efficiency for the logistic bandit problem. In the binary case, our method improves the state-of-the-art binary logistic bandit method by reducing the per-round computation cost from $\mathcal{O}(\log T)$ to $\mathcal{O}(1)$ with respect to the time horizon $T$, while still preserving the minimax optimal guarantee up to logarithmic factors. In the multinomial case, with $K+1$ potential feedback values, our algorithm achieves an $\tilde{\mathcal{O}}(K\sqrt{T})$ regret bound with $\mathcal{O}(1)$ computational cost per round. The result not only improves the $\tilde{\mathcal{O}}(K\sqrt{\kappa T})$ bound for the best-known tractable algorithm—where the large constant $\kappa$ increases exponentially with the diameter of the parameter domain—but also reduces the $\mathcal{O}(T)$ computational complexity demanded by the previous method.

----

## [0] Transfer learning for atomistic simulations using GNNs and kernel mean embeddings

**Authors**: *John Isak Texas Falk, Luigi Bonati, Pietro Novelli, Michele Parrinello, Massimiliano Pontil*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f02c76bc411a6f7c9a8bb2cbf981260-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f02c76bc411a6f7c9a8bb2cbf981260-Abstract-Conference.html)

**Abstract**:

Interatomic potentials learned using machine learning methods have been successfully applied to atomistic simulations. However, accurate models require large training datasets, while generating reference calculations is computationally demanding. To bypass this difficulty, we propose a transfer learning algorithm that leverages the ability of graph neural networks (GNNs) to represent chemical environments together with kernel mean embeddings. We extract a feature map from GNNs pre-trained on the OC20 dataset and use it to learn the potential energy surface from system-specific datasets of catalytic processes. Our method is further enhanced by incorporating into the kernel the chemical species information, resulting in improved performance and interpretability. We test our approach on a series of realistic datasets of increasing complexity, showing excellent generalization and transferability performance, and improving on methods that rely on GNNs or ridge regression alone, as well as similar fine-tuning approaches.

----

## [0] StressID: a Multimodal Dataset for Stress Identification

**Authors**: *Hava Chaptoukaev, Valeriya Strizhkova, Michele Panariello, Bianca Dalpaos, Aglind Reka, Valeria Manera, Susanne Thümmler, Esma Ismailova, Nicholas W. D. Evans, François Brémond, Massimiliano Todisco, Maria A. Zuluaga, Laura M. Ferrari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f09bfe6730e9627a9f800d01a8ad5cd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f09bfe6730e9627a9f800d01a8ad5cd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

StressID is a new dataset specifically designed for stress identification fromunimodal and multimodal data. It contains videos of facial expressions, audiorecordings, and physiological signals. The video and audio recordings are acquiredusing an RGB camera with an integrated microphone. The physiological datais composed of electrocardiography (ECG), electrodermal activity (EDA), andrespiration signals that are recorded and monitored using a wearable device. Thisexperimental setup ensures a synchronized and high-quality multimodal data col-lection. Different stress-inducing stimuli, such as emotional video clips, cognitivetasks including mathematical or comprehension exercises, and public speakingscenarios, are designed to trigger a diverse range of emotional responses. Thefinal dataset consists of recordings from 65 participants who performed 11 tasks,as well as their ratings of perceived relaxation, stress, arousal, and valence levels.StressID is one of the largest datasets for stress identification that features threedifferent sources of data and varied classes of stimuli, representing more than39 hours of annotated data in total. StressID offers baseline models for stressclassification including a cleaning, feature extraction, and classification phase foreach modality. Additionally, we provide multimodal predictive models combiningvideo, audio, and physiological inputs. The data and the code for the baselines areavailable at https://project.inria.fr/stressid/.

----

## [0] Statistical Knowledge Assessment for Large Language Models

**Authors**: *Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f0a4cd23e1c6eedd3edebba674ab877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f0a4cd23e1c6eedd3edebba674ab877-Abstract-Conference.html)

**Abstract**:

Given varying prompts regarding a factoid question, can a large language model (LLM) reliably generate factually correct answers? Existing LLMs may generate distinct responses for different prompts. In this paper, we study the problem of quantifying knowledge contained in an LLM regarding a given set of facts. We propose KaRR, a statistical approach to assess factual knowledge for LLMs. The main idea is to estimate the ratio of LLM generating text corresponding to the answer entity given diverse prompts of the subject and the querying relation, versus it generating by random chances. Our assessment suite contains a comprehensive set of 994,123 entities and 600 relations, with 1,395,905 text aliases. We use our method to evaluate 20 LLMs of various sizes, including LLaMA, Alpaca, OPT, etc. Experiments show that our results have a strong correlation (0.43 Kendall's $\tau$) with the results of human assessment on LLMs. Our results reveal that the knowledge in LLMs with the same backbone architecture adheres to the scaling law, while tuning on instruction-following data sometimes compromises the model's capability to generate factually correct text reliably.

----

## [0] Color Equivariant Convolutional Networks

**Authors**: *Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, Jan van Gemert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f173562e7662b14fb5c5695f225ea46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f173562e7662b14fb5c5695f225ea46-Abstract-Conference.html)

**Abstract**:

Color is a crucial visual cue readily exploited by Convolutional Neural Networks (CNNs) for object recognition. However, CNNs struggle if there is data imbalance between color variations introduced by accidental recording conditions. Color invariance addresses this issue but does so at the cost of removing all color information, which sacrifices discriminative power. In this paper, we propose Color Equivariant Convolutions (CEConvs), a novel deep learning building block that enables shape feature sharing across the color spectrum while retaining important color information. We extend the notion of equivariance from geometric to photometric transformations by incorporating parameter sharing over hue-shifts in a neural network. We demonstrate the benefits of CEConvs in terms of downstream performance to various tasks and improved robustness to color changes, including train-test distribution shifts. Our approach can be seamlessly integrated into existing architectures, such as ResNets, and offers a promising solution for addressing color-based domain shifts in CNNs.

----

## [0] Realistic Synthetic Financial Transactions for Anti-Money Laundering Models

**Authors**: *Erik Altman, Jovan Blanusa, Luc von Niederhäusern, Beni Egressy, Andreea Anghel, Kubilay Atasu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

With the widespread digitization of finance and the increasing popularity of cryptocurrencies, the sophistication of fraud schemes devised by cybercriminals is growing. Money laundering -- the movement of illicit funds to conceal their origins -- can cross bank and national boundaries, producing complex transaction patterns. The UN estimates 2-5\% of global GDP or \$0.8 - \$2.0 trillion dollars are laundered globally each year.  Unfortunately, real data to train machine learning models to detect laundering is generally not available, and previous synthetic data generators have had significant shortcomings. A realistic, standardized, publicly-available benchmark is needed for comparing models and for the advancement of the area.To this end, this paper contributes a synthetic financial transaction dataset generator and a set of synthetically generated AML (Anti-Money Laundering) datasets.  We have calibrated this agent-based generator to match real transactions as closely as possible and made the datasets public. We describe the generator in detail and demonstrate how the datasets generated can help compare different machine learning models in terms of their AML abilities. In a key way, using synthetic data in these comparisons can be even better than using real data: the ground truth labels are complete, whilst many laundering transactions in real data are never detected.

----



[Go to the previous page](NIPS-2023-list1200.md)

[Go to the next page](NIPS-2023-list1202.md)

[Go to the catalog section](README.md)