## [1200] D-CIPHER: Discovery of Closed-form Partial Differential Equations

        **Authors**: *Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57c30b677add9aa78e1745f0643104d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57c30b677add9aa78e1745f0643104d0-Abstract-Conference.html)

        **Abstract**:

        Closed-form differential equations, including partial differential equations and higher-order ordinary differential equations, are one of the most important tools used by scientists to model and better understand natural phenomena. Discovering these equations directly from data is challenging because it requires modeling relationships between various derivatives that are not observed in the data (equation-data mismatch) and it involves searching across a huge space of possible equations. Current approaches make strong assumptions about the form of the equation and thus fail to discover many well-known phenomena. Moreover, many of them resolve the equation-data mismatch by estimating the derivatives, which makes them inadequate for noisy and infrequent observations. To this end, we propose D-CIPHER, which is robust to measurement artifacts and can uncover a new and very general class of differential equations. We further design a novel optimization procedure, CoLLie, to help D-CIPHER search through this class efficiently. Finally, we demonstrate empirically that it can discover many well-known equations that are beyond the capabilities of current methods.

        ----

        ## [1201] Training neural operators to preserve invariant measures of chaotic attractors

        **Authors**: *Ruoxi Jiang, Peter Y. Lu, Elena Orlova, Rebecca Willett*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html)

        **Abstract**:

        Chaotic systems make long-horizon forecasts difficult because small perturbations in initial conditions cause trajectories to diverge at an exponential rate. In this setting, neural operators trained to minimize squared error losses, while capable of accurate short-term forecasts, often fail to reproduce statistical or structural properties of the dynamics over longer time horizons and can yield degenerate results. In this paper, we propose an alternative framework designed to preserve invariant measures of chaotic attractors that characterize the time-invariant statistical properties of the dynamics. Specifically, in the multi-environment setting (where each sample trajectory is governed by slightly different dynamics),  we consider two novel approaches to training with noisy data. First, we propose a loss based on the optimal transport distance between the observed dynamics and the neural operator outputs. This approach requires expert knowledge of the underlying physics to determine what statistical features should be included in the optimal transport loss. Second, we show that a  contrastive learning framework, which does not require any specialized prior knowledge, can preserve statistical properties of the dynamics nearly as well as the optimal transport approach. On a variety of chaotic systems, our method is shown empirically to preserve invariant measures of chaotic attractors.

        ----

        ## [1202] Certification of Distributional Individual Fairness

        **Authors**: *Matthew Wicker, Vihari Piratla, Adrian Weller*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57d8ebf4c2f050a6485f370d47656a9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57d8ebf4c2f050a6485f370d47656a9e-Abstract-Conference.html)

        **Abstract**:

        Providing formal guarantees of algorithmic fairness is of paramount importance to socially responsible deployment of machine learning algorithms. In this work, we study formal guarantees, i.e., certificates, for individual fairness (IF) of neural networks. We start by introducing a novel convex approximation of IF constraints that exponentially decreases the computational cost of providing formal guarantees of local individual fairness. We highlight that prior methods are constrained by their focus on global IF certification and can therefore only scale to models with a few dozen hidden neurons, thus limiting their practical impact. We propose to certify \textit{distributional} individual fairness which ensures that for a given empirical distribution and all distributions within a $\gamma$-Wasserstein ball, the neural network has guaranteed individually fair predictions. Leveraging developments in quasi-convex optimization, we provide novel and efficient certified bounds on distributional individual fairness and show that our method allows us to certify and regularize neural networks that are several orders of magnitude larger than those considered by prior works. Moreover, we study real-world distribution shifts and find our bounds to be a scalable, practical, and sound source of IF guarantees.

        ----

        ## [1203] Leveraging sparse and shared feature activations for disentangled representation learning

        **Authors**: *Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, Francesco Locatello*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57fabaa549352c52d5d312171b16970e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57fabaa549352c52d5d312171b16970e-Abstract-Conference.html)

        **Abstract**:

        Recovering the latent factors of variation of high dimensional data has so far focused on simple synthetic settings. Mostly building on unsupervised and weakly-supervised objectives, prior work missed out on the positive implications for representation learning on real world data. In this work, we propose to leverage knowledge extracted from a diversified set of supervised tasks to learn a common disentangled representation. Assuming each supervised task only depends on an unknown subset of the factors of variation, we disentangle the feature space of a supervised multi-task model, with features activating sparsely across different tasks and information being shared as appropriate. Importantly, we never directly observe the factors of variations but establish that access to multiple tasks is sufficient for identifiability under sufficiency and minimality assumptions.We validate our approach on six real world distribution shift benchmarks, and different data modalities (images, text), demonstrating how disentangled representations can be transferred to real settings.

        ----

        ## [1204] Mathematical Capabilities of ChatGPT

        **Authors**: *Simon Frieder, Luca Pinchetti, Alexis Chevalier, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, Julius Berner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We investigate the mathematical capabilities of two iterations of ChatGPT (released 9-January-2023 and 30-January-2023) and of GPT-4 by testing them on publicly available datasets, as well as hand-crafted ones, using a novel methodology. In contrast to formal mathematics, where large databases of formal proofs are available (e.g., mathlib, the Lean Mathematical Library), current datasets of natural-language mathematics used to benchmark language models either cover only elementary mathematics or are very small. We address this by publicly releasing two new datasets: GHOSTS and miniGHOSTS. These are the first natural-language datasets curated by working researchers in mathematics that (1) aim to cover graduate-level mathematics, (2) provide a holistic overview of the mathematical capabilities of language models, and (3) distinguish multiple dimensions of mathematical reasoning. These datasets test on 1636 human expert evaluations whether ChatGPT and GPT-4 can be helpful assistants to professional mathematicians by emulating use cases that arise in the daily professional activities of mathematicians. We benchmark the models on a range of fine-grained performance metrics. For advanced mathematics, this is the most detailed evaluation effort to date. We find that ChatGPT and GPT-4 can be used most successfully as mathematical assistants for querying facts, acting as mathematical search engines and knowledge base interfaces. GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty. Contrary to many positive reports in the media about GPT-4 and ChatGPT's exam-solving abilities (a potential case of selection bias), their overall mathematical performance is well below the level of a graduate student. Hence, if you aim to use ChatGPT to pass a graduate-level math exam, you would be better off copying from your average peer!

        ----

        ## [1205] A Unified Framework for U-Net Design and Analysis

        **Authors**: *Christopher Williams, Fabian Falck, George Deligiannidis, Chris C. Holmes, Arnaud Doucet, Saifuddin Syed*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58575be50c9b47902359920a4d5d1ab4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58575be50c9b47902359920a4d5d1ab4-Abstract-Conference.html)

        **Abstract**:

        U-Nets are a go-to neural architecture across numerous tasks for continuous signals on a square such as images and Partial Differential Equations (PDE), however their design and architecture is understudied. In this paper, we provide a framework for designing and analysing general U-Net architectures. We present theoretical results which characterise the role of the encoder and decoder in a U-Net, their high-resolution scaling limits and their conjugacy to ResNets via preconditioning. We propose Multi-ResNets, U-Nets with a simplified, wavelet-based encoder without learnable parameters. Further, we show how to design novel U-Net architectures which encode function constraints, natural bases, or the geometry of the data. In diffusion models, our framework enables us to identify that high-frequency information is dominated by noise exponentially faster, and show how U-Nets with average pooling exploit this. In our experiments, we demonstrate how Multi-ResNets achieve competitive and often superior performance compared to classical U-Nets in image segmentation, PDE surrogate modelling, and generative modelling with diffusion models. Our U-Net framework paves the way to study the theoretical properties of U-Nets and design natural, scalable neural architectures for a multitude of problems beyond the square.

        ----

        ## [1206] On the Importance of Feature Separability in Predicting Out-Of-Distribution Error

        **Authors**: *Renchunzi Xie, Hongxin Wei, Lei Feng, Yuzhou Cao, Bo An*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/585e9cf25585612ac27b535457116513-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/585e9cf25585612ac27b535457116513-Abstract-Conference.html)

        **Abstract**:

        Estimating the generalization performance is practically challenging on out-of-distribution (OOD) data without ground-truth labels. While previous methods emphasize the connection between distribution difference and OOD accuracy, we show that a large domain gap not necessarily leads to a low test accuracy. In this paper, we investigate this problem from the perspective of feature separability empirically and theoretically. Specifically, we propose a dataset-level score based upon feature dispersion to estimate the test accuracy under distribution shift. Our method is inspired by desirable properties of features in representation learning: high inter-class dispersion and high intra-class compactness. Our analysis shows that inter-class dispersion is strongly correlated with the model accuracy, while intra-class compactness does not reflect the generalization performance on OOD data. Extensive experiments demonstrate the superiority of our method in both prediction performance and computational efficiency.

        ----

        ## [1207] The Transient Nature of Emergent In-Context Learning in Transformers

        **Authors**: *Aaditya K. Singh, Stephanie C. Y. Chan, Ted Moskovitz, Erin Grant, Andrew M. Saxe, Felix Hill*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58692a1701314e09cbd7a5f5f3871cc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58692a1701314e09cbd7a5f5f3871cc9-Abstract-Conference.html)

        **Abstract**:

        Transformer neural networks can exhibit a surprising capacity for in-context learning (ICL) despite not being explicitly trained for it.  Prior work has provided a deeper understanding of how ICL emerges in transformers, e.g. through the lens of mechanistic interpretability, Bayesian inference, or by examining the distributional properties of training data. However, in each of these cases, ICL is treated largely as a persistent phenomenon; namely, once ICL emerges, it is assumed to persist asymptotically. Here, we show that the emergence of ICL during transformer training is, in fact, often transient. We train transformers on synthetic data designed so that both ICL and in-weights learning (IWL) strategies can lead to correct predictions. We find that ICL first emerges, then disappears and gives way to IWL, all while the training loss decreases, indicating an asymptotic preference for IWL. The transient nature of ICL is observed in transformers across a range of model sizes and datasets, raising the question of how much to ``overtrain'' transformers when seeking compact, cheaper-to-run models. We find that L2 regularization may offer a path to more persistent ICL that removes the need for early stopping based on ICL-style validation tasks. Finally, we present initial evidence that ICL transience may be caused by competition between ICL and IWL circuits.

        ----

        ## [1208] When is Agnostic Reinforcement Learning Statistically Tractable?

        **Authors**: *Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, Nati Srebro*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58a799d16fb0c1f2014e98f4ba972b25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58a799d16fb0c1f2014e98f4ba972b25-Abstract-Conference.html)

        **Abstract**:

        We study the problem of agnostic PAC reinforcement learning (RL): given a policy class $\Pi$, how many rounds of interaction with an unknown MDP (with a potentially large state and action space) are required to learn an $\epsilon$-suboptimal policy with respect to \(\Pi\)? Towards that end, we introduce a new complexity measure, called the \emph{spanning capacity}, that depends solely on the set \(\Pi\) and is independent of the MDP dynamics. With a generative model, we show that the spanning capacity characterizes PAC learnability for every policy class $\Pi$. However, for online RL, the situation is more subtle. We show there exists a policy class $\Pi$ with a bounded spanning capacity that requires a superpolynomial number of samples to learn. This reveals a surprising separation for agnostic learnability between generative access and online access models (as well as between deterministic/stochastic MDPs under online access). On the positive side, we identify an additional \emph{sunflower} structure which in conjunction with bounded spanning capacity enables statistically efficient online RL via a new algorithm called POPLER, which takes inspiration from classical importance sampling methods as well as recent developments for reachable-state identification and policy evaluation in reward-free exploration.

        ----

        ## [1209] Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models

        **Authors**: *Yeongbin Kim, Gautam Singh, Junyeong Park, Çaglar Gülçehre, Sungjin Ahn*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58af908d6293810f1a29e69bf723dc48-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/58af908d6293810f1a29e69bf723dc48-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Systematic compositionality, or the ability to adapt to novel situations by creating a mental model of the world using reusable pieces of knowledge, remains a significant challenge in machine learning. While there has been considerable progress in the language domain, efforts towards systematic visual imagination, or envisioning the dynamical implications of a visual observation, are in their infancy. We introduce the Systematic Visual Imagination Benchmark (SVIB), the first benchmark designed to address this problem head-on. SVIB offers a novel framework for a minimal world modeling problem, where models are evaluated based on their ability to generate one-step image-to-image transformations under a latent world dynamics. The framework provides benefits such as the possibility to jointly optimize for systematic perception and imagination, a range of difficulty levels, and the ability to control the fraction of possible factor combinations used during training. We provide a comprehensive evaluation of various baseline models on SVIB, offering insight into the current state-of-the-art in systematic visual imagination. We hope that this benchmark will help advance visual systematic compositionality.

        ----

        ## [1210] Convolutional Visual Prompt for Robust Visual Perception

        **Authors**: *Yun-Yun Tsai, Chengzhi Mao, Junfeng Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html)

        **Abstract**:

        Vision models are often vulnerable to out-of-distribution (OOD) samples without adapting. While visual prompts offer a lightweight method of input-space adaptation for large-scale vision models, they rely on a high-dimensional additive vector and labeled data. This leads to overfitting when adapting models in a self-supervised test-time setting without labels. We introduce convolutional visual prompts (CVP) for label-free test-time adaptation for robust visual perception. The structured nature of CVP demands fewer trainable parameters, less than 1\% compared to standard visual prompts, combating overfitting. Extensive experiments and analysis on a wide variety of OOD visual perception tasks show that our approach is effective, improving robustness by up to 5.87\% over several large-scale models.

        ----

        ## [1211] LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching

        **Authors**: *Duy M. H. Nguyen, Hoang Nguyen, Nghiem Tuong Diep, Tan Ngoc Pham, Tri Cao, Binh T. Nguyen, Paul Swoboda, Nhat Ho, Shadi Albarqouni, Pengtao Xie, Daniel Sonntag, Mathias Niepert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58cc11cda2a2679e8af5c6317aed0af8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58cc11cda2a2679e8af5c6317aed0af8-Abstract-Conference.html)

        **Abstract**:

        Obtaining large pre-trained models that can be fine-tuned to new tasks with limited annotated samples has remained an open challenge for medical imaging data. While pre-trained networks on ImageNet and vision-language foundation models trained on web-scale data are the prevailing approaches, their effectiveness on medical tasks is limited due to the significant domain shift between natural and medical images. To bridge this gap, we introduce LVM-Med, the first family of deep networks trained on large-scale medical datasets. We have collected approximately 1.3 million medical images from 55 publicly available datasets, covering a large number of organs and modalities such as CT, MRI, X-ray, and Ultrasound. We benchmark several state-of-the-art self-supervised algorithms on this dataset and propose a novel self-supervised contrastive learning algorithm using a graph-matching formulation. The proposed approach makes three contributions: (i) it integrates prior pair-wise image similarity metrics based on local and global information; (ii) it captures the structural constraints of feature embeddings through a loss function constructed through a combinatorial graph-matching objective, and (iii) it can be trained efficiently end-to-end using modern gradient-estimation techniques for black-box solvers. We thoroughly evaluate the proposed LVM-Med on 15 downstream medical tasks ranging from segmentation and classification to object detection, and both for the in and out-of-distribution settings. LVM-Med empirically outperforms a number of state-of-the-art supervised, self-supervised, and foundation models. For challenging tasks such as Brain Tumor Classification or Diabetic Retinopathy Grading, LVM-Med improves previous vision-language models trained on 1 billion masks by 6-7%  while using only a ResNet-50.

        ----

        ## [1212] Lending Interaction Wings to Recommender Systems with Conversational Agents

        **Authors**: *Jiarui Jin, Xianyu Chen, Fanghua Ye, Mengyue Yang, Yue Feng, Weinan Zhang, Yong Yu, Jun Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58cd3b02902d79aea4b3b603fb0d0941-Abstract-Conference.html)

        **Abstract**:

        An intelligent conversational agent (a.k.a., chat-bot) could embrace conversational technologies to obtain user preferences online, to overcome inherent limitations of recommender systems trained over the offline historical user behaviors. In this paper, we propose CORE, a new offline-training and online-checking framework to plug a COnversational agent into REcommender systems. Unlike most prior conversational recommendation approaches that systemically combine conversational and recommender parts through a reinforcement learning framework, CORE bridges the conversational agent and recommender system through a unified uncertainty minimization framework, which can be easily applied to any existing recommendation approach. Concretely, CORE treats a recommender system as an offline estimator to produce an estimated relevance score for each item, while CORE regards a conversational agent as an online checker that checks these estimated scores in each online session. We define uncertainty as the sum of unchecked relevance scores. In this regard, the conversational agent acts to minimize uncertainty via querying either attributes or items. Towards uncertainty minimization, we derive the certainty gain of querying each attribute and item, and develop a novel online decision tree algorithm to decide what to query at each turn. Our theoretical analysis reveals the bound of the expected number of turns of CORE in a cold-start setting. Experimental results demonstrate that CORE can be seamlessly employed on a variety of recommendation approaches, and can consistently bring significant improvements in both hot-start and cold-start settings.

        ----

        ## [1213] High-Fidelity Audio Compression with Improved RVQGAN

        **Authors**: *Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/58d0e78cf042af5876e12661087bea12-Abstract-Conference.html)

        **Abstract**:

        Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves ~90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling.

        ----

        ## [1214] Comparing Apples to Oranges: Learning Similarity Functions for Data Produced by Different Distributions

        **Authors**: *Leonidas Tsepenekas, Ivan Brugere, Freddy Lécué, Daniele Magazzeni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59056767478c7df64e6250eadfeb0a04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59056767478c7df64e6250eadfeb0a04-Abstract-Conference.html)

        **Abstract**:

        Similarity functions measure how comparable pairs of elements are, and play a key role in a wide variety of applications, e.g., notions of Individual Fairness abiding by the seminal paradigm of Dwork et al., as well as Clustering problems. However, access to an accurate similarity function should not always be considered guaranteed, and this point was even raised by Dwork et al. For instance, it is reasonable to assume that when the elements to be compared are produced by different distributions, or in other words belong to different ``demographic'' groups, knowledge of their true similarity might be very difficult to obtain. In this work, we present an efficient sampling framework that learns these across-groups similarity functions, using only a limited amount of experts' feedback. We show analytical results with rigorous theoretical bounds, and empirically validate our algorithms via a large suite of experiments.

        ----

        ## [1215] Scalable Transformer for PDE Surrogate Modeling

        **Authors**: *Zijie Li, Dule Shu, Amir Barati Farimani*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/590daf74f99ee85df3d8c007df9c8187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/590daf74f99ee85df3d8c007df9c8187-Abstract-Conference.html)

        **Abstract**:

        Transformer has shown state-of-the-art performance on various applications and has recently emerged as a promising tool for surrogate modeling of partial differential equations (PDEs). Despite the introduction of linear-complexity attention, applying Transformer to problems with a large number of grid points can be numerically unstable and computationally expensive. In this work, we propose Factorized Transformer (FactFormer), which is based on an axial factorized kernel integral. Concretely, we introduce a learnable projection operator that decomposes the input function into multiple sub-functions with one-dimensional domain. These sub-functions are then evaluated and used to compute the instance-based kernel with an axial factorized scheme. We showcase that the proposed model is able to simulate 2D Kolmogorov flow on a $256\times 256$ grid and 3D smoke buoyancy on a $64\times64\times64$ grid with good accuracy and efficiency. The proposed factorized scheme can serve as a computationally efficient low-rank surrogate for the full attention scheme when dealing with multi-dimensional problems.

        ----

        ## [1216] What is the Inductive Bias of Flatness Regularization? A Study of Deep Matrix Factorization Models

        **Authors**: *Khashayar Gatmiry, Zhiyuan Li, Tengyu Ma, Sashank Reddi, Stefanie Jegelka, Ching-Yao Chuang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5927edd18c5dd83aa8936a4610c72029-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5927edd18c5dd83aa8936a4610c72029-Abstract-Conference.html)

        **Abstract**:

        Recent works on over-parameterized neural networks have shown that  the stochasticity in optimizers has the implicit regularization effect of minimizing the sharpness of the loss function (in particular, the trace of its Hessian) over the family zero-loss solutions. More explicit forms of flatness regularization also empirically improve the generalization performance. However, it remains unclear why and when flatness regularization leads to better generalization. This work takes the first step towards understanding the inductive bias of the minimum trace of the Hessian solutions in an important setting: learning deep linear networks from linear measurements, also known as \emph{deep matrix factorization}. We show that with the standard Restricted Isometry Property (RIP) on the measurements, minimizing the trace of Hessian is approximately equivalent to minimizing the Schatten 1-norm of the corresponding end-to-end matrix parameters (i.e., the product of all layer matrices), which in turn leads to better generalization.

        ----

        ## [1217] Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation

        **Authors**: *Shutong Ding, Tianyu Cui, Jingya Wang, Ye Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html)

        **Abstract**:

        Deep Equilibrium Models (DEQs) and Neural Ordinary Differential Equations (Neural ODEs) are two branches of implicit models that have achieved remarkable success owing to their superior performance and low memory consumption. While both are implicit models, DEQs and Neural ODEs are derived from different mathematical formulations. Inspired by homotopy continuation, we establish a connection between these two models and illustrate that they are actually two sides of the same coin. Homotopy continuation is a classical method of solving nonlinear equations based on a corresponding ODE. Given this connection, we proposed a new implicit model called HomoODE that inherits the property of high accuracy from DEQs and the property of stability from Neural ODEs. Unlike DEQs, which explicitly solve an equilibrium-point-finding problem via Newton's methods in the forward pass, HomoODE solves the equilibrium-point-finding problem implicitly using a modified Neural ODE via homotopy continuation. Further, we developed an acceleration method for HomoODE with a shared learnable initial point. It is worth noting that our model also provides a better understanding of why Augmented Neural ODEs work as long as the augmented part is regarded as the equilibrium point to find. Comprehensive experiments with several image classification tasks demonstrate that HomoODE surpasses existing implicit models in terms of both accuracy and memory consumption.

        ----

        ## [1218] Emergent and Predictable Memorization in Large Language Models

        **Authors**: *Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html)

        **Abstract**:

        Memorization, or the tendency of large language models (LLMs) to output entire sequences from their training data verbatim, is a key concern for deploying language models. In particular, it is vital to minimize a model's memorization of sensitive datapoints such as those containing personal identifiable information (PII). The prevalence of such undesirable memorization can pose issues for model trainers, and may even require discarding an otherwise functional model. We therefore seek to predict which sequences will be memorized before a large model's full train-time by extrapolating the memorization behavior of lower-compute trial runs. We measure memorization in the Pythia model suite and plot scaling laws for forecasting memorization, allowing us to provide equi-compute recommendations to maximize the reliability (recall) of such predictions. We additionally provide further novel discoveries on the distribution of memorization scores across models and data. We release all code and data necessary to reproduce the results in this paper at https://github.com/EleutherAI/pythia.

        ----

        ## [1219] Mind2Web: Towards a Generalist Agent for the Web

        **Authors**: *Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samual Stevens, Boshi Wang, Huan Sun, Yu Su*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further research on building a generalist agent for the web.

        ----

        ## [1220] MultiMoDN - Multimodal, Multi-Task, Interpretable Modular Networks

        **Authors**: *Vinitra Swamy, Malika Satayeva, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser, Mary-Anne Hartley*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5951641ad71b0052cf776f9b71f18932-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5951641ad71b0052cf776f9b71f18932-Abstract-Conference.html)

        **Abstract**:

        Predicting multiple real-world tasks in a single model often requires a particularly diverse feature space. Multimodal (MM) models aim to extract the synergistic predictive potential of multiple data types to create a shared feature space with aligned semantic meaning across inputs of drastically varying sizes (i.e. images, text, sound). Most current MM architectures fuse these representations in parallel, which not only limits their interpretability but also creates a dependency on modality availability. We present MultiModN, a multimodal, modular network that fuses latent representations in a sequence of any number, combination, or type of modality while providing granular real-time predictive feedback on any number or combination of predictive tasks. MultiModN's composable pipeline is interpretable-by-design, as well as innately multi-task and robust to the fundamental issue of biased missingness. We perform four experiments on several benchmark MM datasets across 10 real-world tasks (predicting medical diagnoses, academic performance, and weather), and show that MultiModN's sequential MM fusion does not compromise performance compared with a baseline of parallel fusion. By simulating the challenging bias of missing not-at-random (MNAR), this work shows that, contrary to MultiModN, parallel fusion baselines erroneously learn MNAR and suffer catastrophic failure when faced with different patterns of MNAR at inference. To the best of our knowledge, this is the first inherently MNAR-resistant approach to MM modeling. In conclusion, MultiModN provides granular insights, robustness, and flexibility without compromising performance.

        ----

        ## [1221] Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs

        **Authors**: *Shengyuan Chen, Yunfeng Cai, Huang Fang, Xiao Huang, Mingming Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5965f3a748a8d41415db2bfa44635cc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5965f3a748a8d41415db2bfa44635cc3-Abstract-Conference.html)

        **Abstract**:

        Knowledge graph (KG) reasoning utilizes two primary techniques, i.e., rule-based and KG-embedding based. The former provides precise inferences, but inferring via concrete rules is not scalable. The latter enables efficient reasoning at the cost of ambiguous inference accuracy. Neuro-symbolic reasoning seeks to amalgamate the advantages of both techniques. The crux of this approach is replacing the predicted existence of all possible triples (i.e., truth scores inferred from rules) with a suitable approximation grounded in embedding representations. However, constructing an effective approximation of all possible triples' truth scores is a challenging task, because it needs to balance the tradeoff between accuracy and efficiency, while compatible with both the rule-based and KG-embedding models. To this end, we proposed a differentiable framework - DiffLogic. Instead of directly approximating all possible triples, we design a tailored filter to adaptively select essential triples based on the dynamic rules and weights. The truth scores assessed by KG-embedding are continuous, so we employ a continuous Markov logic network named probabilistic soft logic (PSL). It employs the truth scores of essential triples to assess the overall agreement among rules, weights, and observed triples. PSL enables end-to-end differentiable optimization, so we can alternately update embedding and weighted rules. On benchmark datasets, we empirically show that DiffLogic surpasses baselines in both effectiveness and efficiency.

        ----

        ## [1222] Topological Parallax: A Geometric Specification for Deep Perception Models

        **Authors**: *Abraham D. Smith, Michael J. Catanzaro, Gabrielle Angeloro, Nirav Patel, Paul Bendich*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html)

        **Abstract**:

        For safety and robustness of AI systems, we introduce topological parallax as a theoretical and computational tool that compares a trained model to a reference dataset to determine whether they have similar multiscale geometric structure. Our proofs and examples show that this geometric similarity between dataset and model is essential to trustworthy interpolation and perturbation, and we conjecture that this new concept will add value to the current debate regarding the unclear relationship between "overfitting"' and "generalization'' in applications of deep-learning. In typical deep-learning applications, an explicit geometric description of the model isimpossible, but parallax can estimate topological features (components, cycles, voids, etc.)in the model by examining the effect on the Rips complex of geodesic distortions using the reference dataset.Thus, parallax indicates whether the model shares similar multiscale geometric features with the dataset.Parallax presents theoretically via topological data analysis [TDA] as a bi-filtered persistence module,and the key properties of this module are stable under perturbation of the reference dataset.

        ----

        ## [1223] Rewiring Neurons in Non-Stationary Environments

        **Authors**: *Zhicheng Sun, Yadong Mu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/599221d7ebf6b3403190f38a3f282a1c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/599221d7ebf6b3403190f38a3f282a1c-Abstract-Conference.html)

        **Abstract**:

        The human brain rewires itself for neuroplasticity in the presence of new tasks. We are inspired to harness this key process in continual reinforcement learning, prioritizing adaptation to non-stationary environments. In distinction to existing rewiring approaches that rely on pruning or dynamic routing, which may limit network capacity and plasticity, this work presents a novel rewiring scheme by permuting hidden neurons. Specifically, the neuron permutation is parameterized to be end-to-end learnable and can rearrange all available synapses to explore a large span of weight space, thereby promoting adaptivity. In addition, we introduce two main designs to steer the rewiring process in continual reinforcement learning: first, a multi-mode rewiring strategy is proposed which diversifies the policy and encourages exploration when encountering new environments. Secondly, to ensure stability on history tasks, the network is devised to cache each learned wiring while subtly updating its weights, allowing for retrospective recovery of any previous state appropriate for the task. Meanwhile, an alignment mechanism is curated to achieve better plasticity-stability tradeoff by jointly optimizing cached wirings and weights. Our proposed method is comprehensively evaluated on 18 continual reinforcement learning scenarios ranging from locomotion to manipulation, demonstrating its advantages over state-of-the-art competitors in performance-efficiency tradeoffs. Code is available at https://github.com/feifeiobama/RewireNeuron.

        ----

        ## [1224] TransHP: Image Classification with Hierarchical Prompting

        **Authors**: *Wenhao Wang, Yifan Sun, Wei Li, Yi Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59b7c1e1716c4feadefd6c70b1dd4630-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59b7c1e1716c4feadefd6c70b1dd4630-Abstract-Conference.html)

        **Abstract**:

        This paper explores a hierarchical prompting mechanism for the hierarchical image classification (HIC) task. Different from prior HIC methods, our hierarchical prompting is the first to explicitly inject ancestor-class information as a tokenized hint that benefits the descendant-class discrimination. We think it well imitates human visual recognition, i.e., humans may use the ancestor class as a prompt to draw focus on the subtle differences among descendant classes. We model this prompting mechanism into a Transformer with Hierarchical Prompting (TransHP). TransHP consists of three steps: 1) learning a set of prompt tokens to represent the coarse (ancestor) classes, 2) on-the-fly predicting the coarse class of the input image at an intermediate block, and 3) injecting the prompt token of the predicted coarse class into the intermediate feature. Though the parameters of TransHP maintain the same for all input images, the injected coarse-class prompt conditions (modifies) the subsequent feature extraction and encourages a dynamic focus on relatively subtle differences among the descendant classes. Extensive experiments show that TransHP improves image classification on accuracy (e.g., improving ViT-B/16 by +2.83% ImageNet classification accuracy), training data efficiency (e.g., +12.69% improvement under 10% ImageNet training data), and model explainability. Moreover, TransHP also performs favorably against prior HIC methods, showing that TransHP well exploits the hierarchical information. The code is available at: https://github.com/WangWenhao0716/TransHP.

        ----

        ## [1225] Practical Differentially Private Hyperparameter Tuning with Subsampling

        **Authors**: *Antti Koskela, Tejas D. Kulkarni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html)

        **Abstract**:

        Tuning the hyperparameters of differentially private (DP) machine learning (ML) algorithms often requires use of sensitive data and this may leak private information via hyperparameter values. Recently, Papernot and Steinke (2022) proposed a certain class of DP hyperparameter tuning algorithms, where the number of random search samples is randomized. Commonly, these algorithms still considerably increase the DP privacy parameter $\varepsilon$ over non-tuned DP ML model training and can be computationally heavy as evaluating each hyperparameter candidate requires a new training run. We focus on lowering both the DP bounds and the compute cost of these methods by using only a random subset of the sensitive data for the hyperparameter tuning and by appropriately extrapolating the optimal values to a larger dataset. We carry out a RÃ©nyi differential privacy analysis for the proposed method and experimentally show that it consistently leads to better privacy-utility trade-off than the baseline method by Papernot and Steinke.

        ----

        ## [1226] Learning to Discover Skills through Guidance

        **Authors**: *Hyunseung Kim, Byungkun Lee, Hojoon Lee, Dongyoon Hwang, Sejik Park, Kyushik Min, Jaegul Choo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59d4e18a60490b9ed9913f3be2b14839-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59d4e18a60490b9ed9913f3be2b14839-Abstract-Conference.html)

        **Abstract**:

        In the field of unsupervised skill discovery (USD), a major challenge is limited exploration, primarily due to substantial penalties when skills deviate from their initial trajectories. To enhance exploration, recent methodologies employ auxiliary rewards to maximize the epistemic uncertainty or entropy of states. However, we have identified that the effectiveness of these rewards declines as the environmental complexity rises. Therefore, we present a novel USD algorithm, skill discovery with guidance (DISCO-DANCE), which (1) selects the guide skill that possesses the highest potential to reach unexplored states, (2) guides other skills to follow guide skill, then (3) the guided skills are dispersed to maximize their discriminability in unexplored states. Empirical evaluation demonstrates that DISCO-DANCE outperforms other USD baselines in challenging environments, including two navigation benchmarks and a continuous control benchmark. Qualitative visualizations and code of DISCO-DANCE are available at https://mynsng.github.io/discodance/.

        ----

        ## [1227] Polynomial-Time Linear-Swap Regret Minimization in Imperfect-Information Sequential Games

        **Authors**: *Gabriele Farina, Charilaos Pipis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html)

        **Abstract**:

        No-regret learners seek to minimize the difference between the loss they cumulated through the actions they played, and the loss they would have cumulated in hindsight had they consistently modified their behavior according to some strategy transformation function. The size of the set of transformations considered by the learner determines a natural notion of rationality. As the set of transformations each learner considers grows, the strategies played by the learners recover more complex game-theoretic equilibria, including correlated equilibria in normal-form games and extensive-form correlated equilibria in extensive-form games. At the extreme, a no-swap-regret agent is one that minimizes regret against the set of all functions from the set of strategies to itself. While it is known that the no-swap-regret condition can be attained efficiently in nonsequential (normal-form) games, understanding what is the strongest notion of rationality that can be attained efficiently in the worst case in sequential (extensive-form) games is a longstanding open problem. In this paper we provide a positive result, by showing that it is possible, in any sequential game, to retain polynomial-time (in the game tree size) iterations while achieving sublinear regret with respect to all linear transformations of the mixed strategy space, a notion called no-linear-swap regret. This notion of hindsight rationality is as strong as no-swap-regret in nonsequential games, and stronger than no-trigger-regret in sequential gamesâ€”thereby proving the existence of a subset of extensive-form correlated equilibria robust to linear deviations, which we call linear-deviation correlated equilibria, that can be approached efficiently.

        ----

        ## [1228] Counterfactually Comparing Abstaining Classifiers

        **Authors**: *Yo Joong Choe, Aditya Gangrade, Aaditya Ramdas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/59fe467d5e71ba6b8d41bb3928da6f4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/59fe467d5e71ba6b8d41bb3928da6f4c-Abstract-Conference.html)

        **Abstract**:

        Abstaining classifiers have the option to abstain from making predictions on inputs that they are unsure about. These classifiers are becoming increasingly popular in high-stakes decision-making problems, as they can withhold uncertain predictions to improve their reliability and safety. When evaluating black-box abstaining classifier(s), however, we lack a principled approach that accounts for what the classifier would have predicted on its abstentions. These missing predictions matter when they can eventually be utilized, either directly or as a backup option in a failure mode. In this paper, we introduce a novel approach and perspective to the problem of evaluating and comparing abstaining classifiers by treating abstentions as missing data. Our evaluation approach is centered around defining the counterfactual score of an abstaining classifier, defined as the expected performance of the classifier had it not been allowed to abstain. We specify the conditions under which the counterfactual score is identifiable: if the abstentions are stochastic, and if the evaluation data is independent of the training data (ensuring that the predictions are missing at random), then the score is identifiable. Note that, if abstentions are deterministic, then the score is unidentifiable because the classifier can perform arbitrarily poorly on its abstentions. Leveraging tools from observational causal inference, we then develop nonparametric and doubly robust methods to efficiently estimate this quantity under identification. Our approach is examined in both simulated and real data experiments.

        ----

        ## [1229] Video Timeline Modeling For News Story Understanding

        **Authors**: *Meng Liu, Mingda Zhang, Jialu Liu, Hanjun Dai, Ming-Hsuan Yang, Shuiwang Ji, Zheyun Feng, Boqing Gong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a0f92efaa8f0cd67992caf6b2fa2bac-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a0f92efaa8f0cd67992caf6b2fa2bac-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In this paper, we present a novel problem, namely video timeline modeling. Our objective is to create a video-associated timeline from a set of videos related to a specific topic, thereby facilitating the content and structure understanding of the story being told. This problem has significant potential in various real-world applications, for instance, news story summarization. To bootstrap research in this area, we curate a realistic benchmark dataset, YouTube-News-Timeline, consisting of over $12$k timelines and $300$k YouTube news videos. Additionally, we propose a set of quantitative metrics to comprehensively evaluate and compare methodologies. With such a testbed, we further develop and benchmark several deep learning approaches to tackling this problem. We anticipate that this exploratory work will pave the way for further research in video timeline modeling. The assets are available via https://github.com/google-research/google-research/tree/master/video_timeline_modeling.

        ----

        ## [1230] Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning

        **Authors**: *Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam, Remi Tachet des Combes, Romain Laroche*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html)

        **Abstract**:

        While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks,  their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to understand why bisimulation methods succeed in online settings, but falter in offline tasks. Our analysis reveals that missing transitions in the dataset are particularly harmful to the bisimulation principle, leading to ineffective estimation. We also shed light on the critical role of reward scaling in bounding the scale of bisimulation measurements and of the value error they induce. Based on these findings, we propose to apply the expectile operator for representation learning to our offline RL setting, which helps to prevent overfitting to incomplete data. Meanwhile, by introducing an appropriate reward scaling strategy, we avoid the risk of feature collapse in representation space. We implement these recommendations on two state-of-the-art bisimulation-based algorithms, MICo and SimSR, and demonstrate performance gains on two benchmark suites: D4RL and Visual D4RL. Codes are provided at \url{https://github.com/zanghyu/Offline_Bisimulation}.

        ----

        ## [1231] Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting

        **Authors**: *Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally-trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (predict). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the predictions of base forecasters with reduced computational overhead over reverse diffusion (refine). Notably, the generative performance of the model remains intact â€” downstream forecasters trained on synthetic samples from TSDiff outperform forecasters that are trained on samples from other state-of-the-art generative time series models, occasionally even outperforming models trained on real data (synthesize).Our code is available at https://github.com/amazon-science/unconditional-time-series-diffusion

        ----

        ## [1232] Learning Layer-wise Equivariances Automatically using Gradients

        **Authors**: *Tycho F. A. van der Ouderaa, Alexander Immer, Mark van der Wilk*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a33645b5c5b7a9882652526d30d0acc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a33645b5c5b7a9882652526d30d0acc-Abstract-Conference.html)

        **Abstract**:

        Convolutions encode equivariance symmetries into neural networks leading to better generalisation performance. However, symmetries provide fixed hard constraints on the functions a network can represent, need to be specified in advance, and can not be adapted. Our goal is to allow flexible symmetry constraints that can automatically be learned from data using gradients. Learning symmetry and associated weight connectivity structures from scratch is difficult for two reasons. First, it requires efficient and flexible parameterisations of layer-wise equivariances. Secondly, symmetries act as constraints and are therefore not encouraged by training losses measuring data fit. To overcome these challenges, we improve parameterisations of soft equivariance and learn the amount of equivariance in layers by optimising the marginal likelihood, estimated using differentiable Laplace approximations. The objective balances data fit and model complexity enabling layer-wise symmetry discovery in deep networks. We demonstrate the ability to automatically learn layer-wise equivariances on image classification tasks, achieving equivalent or improved performance over baselines with hard-coded symmetry.

        ----

        ## [1233] PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated Learning

        **Authors**: *Mingjia Shi, Yuhao Zhou, Kai Wang, Huaizheng Zhang, Shudong Huang, Qing Ye, Jiancheng Lv*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html)

        **Abstract**:

        Classical federated learning (FL) enables training machine learning models without sharing data for privacy preservation, but heterogeneous data characteristic degrades the performance of the localized model. Personalized FL (PFL) addresses this by synthesizing personalized models from a global model via training on local data. Such a global model may overlook the specific information that the clients have been sampled. In this paper, we propose a novel scheme to inject personalized prior knowledge into the global model in each client, which attempts to mitigate the introduced incomplete information problem in PFL. At the heart of our proposed approach is a framework, the $\textit{PFL with Bregman Divergence}$ (pFedBreD), decoupling the personalized prior from the local objective function regularized by Bregman divergence for greater adaptability in personalized scenarios. We also relax the mirror descent (RMD) to extract the prior explicitly to provide optional strategies. Additionally, our pFedBreD is backed up by a convergence analysis. Sufficient experiments demonstrate that our method reaches the $\textit{state-of-the-art}$ performances on 5 datasets and outperforms other methods by up to 3.5% across 8 benchmarks. Extensive analyses verify the robustness and necessity of proposed designs. The code will be made public.

        ----

        ## [1234] Byzantine-Tolerant Methods for Distributed Variational Inequalities

        **Authors**: *Nazarii Tupitsa, Abdulla Jasem Almansoori, Yanlin Wu, Martin Takác, Karthik Nandakumar, Samuel Horváth, Eduard Gorbunov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html)

        **Abstract**:

        Robustness to Byzantine attacks is a necessity for various distributed training scenarios. When the training reduces to the process of solving a minimization problem, Byzantine robustness is relatively well-understood. However, other problem formulations, such as min-max problems or, more generally, variational inequalities, arise in many modern machine learning and, in particular, distributed learning tasks. These problems significantly differ from the standard minimization ones and, therefore, require separate consideration. Nevertheless, only one work [Abidi et al., 2022] addresses this important question in the context of Byzantine robustness. Our work makes a further step in this direction by providing several (provably) Byzantine-robust methods for distributed variational inequality, thoroughly studying their theoretical convergence, removing the limitations of the previous work, and providing numerical comparisons supporting the theoretical findings.

        ----

        ## [1235] Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow

        **Authors**: *Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, Ya Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html)

        **Abstract**:

        Collaborative perception can substantially boost each agent's perception ability by facilitating communication among multiple agents. However, temporal asynchrony among agents is inevitable in the real world due to communication delays, interruptions, and clock misalignments. This issue causes information mismatch during multi-agent fusion, seriously shaking the foundation of collaboration. To address this issue, we propose CoBEVFlow, an asynchrony-robust collaborative perception system based on bird's eye view (BEV) flow. The key intuition of CoBEVFlow is to compensate motions to align asynchronous collaboration messages sent by multiple agents. To model the motion in a scene, we propose BEV flow, which is a collection of the motion vector corresponding to each spatial location. Based on BEV flow, asynchronous perceptual features can be reassigned to appropriate positions, mitigating the impact of asynchrony. CoBEVFlow has two advantages: (i) CoBEVFlow can handle asynchronous collaboration messages sent at irregular, continuous time stamps without discretization; and (ii) with BEV flow, CoBEVFlow only transports the original perceptual features, instead of generating new perceptual features, avoiding additional noises. To validate CoBEVFlow's efficacy, we create IRregular V2V(IRV2V), the first synthetic collaborative perception dataset with various temporal asynchronies that simulate different real-world scenarios. Extensive experiments conducted on both IRV2V and the real-world dataset DAIR-V2X show that CoBEVFlow consistently outperforms other baselines and is robust in extremely asynchronous settings. The code is available at https://github.com/MediaBrain-SJTU/CoBEVFlow.

        ----

        ## [1236] Train 'n Trade: Foundations of Parameter Markets

        **Authors**: *Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html)

        **Abstract**:

        Organizations typically train large models individually. This is costly and time-consuming, particularly for large-scale foundation models. Such vertical production is known to be suboptimal. Inspired by this economic insight, we ask whether it is possible to leverage others' expertise by trading the constituent parts in models, i.e., sets of weights, as if they were market commodities. While recent advances in aligning and interpolating models suggest that doing so may be possible, a number of fundamental questions must be answered to create viable parameter markets. In this work, we address these basic questions, propose a framework containing the infrastructure necessary for market operations to take place, study strategies for exchanging parameters, and offer means for agents to monetize parameters. Excitingly, compared to agents who train siloed models from scratch, we show that it is possible to mutually gain by using the market, even in competitive settings. This suggests that the notion of parameter markets may be a useful paradigm for improving large-scale model training in the future.

        ----

        ## [1237] Relax, it doesn't matter how you get there: A new self-supervised approach for multi-timescale behavior analysis

        **Authors**: *Mehdi Azabou, Michael Mendelson, Nauman Ahad, Maks Sorokin, Shantanu Thakoor, Carolina Urzay, Eva L. Dyer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aad86aa2a3c00b70c71e19bc4780319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aad86aa2a3c00b70c71e19bc4780319-Abstract-Conference.html)

        **Abstract**:

        Unconstrained and natural  behavior consists of dynamics that are complex and  unpredictable, especially when trying to predict what will happen  multiple steps into the future. While some success has been found in building representations of animal behavior under constrained or simplified task-based conditions, many of these models cannot be applied to free and naturalistic settings where behavior becomes increasingly hard to model. In this work, we develop a multi-task representation learning model for animal behavior that combines two novel components: (i) an action-prediction objective that aims to predict the  distribution of actions over future timesteps, and (ii) a multi-scale architecture that builds separate latent spaces to accommodate short- and long-term dynamics. After demonstrating the ability of the method to build representations of both local and global dynamics in robots in varying environments and terrains, we apply our method to the MABe 2022 Multi-Agent Behavior challenge, where our model ranks first overall on both mice and fly benchmarks. In all of these cases, we show that our model can build representations that capture the many different factors that drive behavior and solve a wide range of downstream tasks.

        ----

        ## [1238] A Measure-Theoretic Axiomatisation of Causality

        **Authors**: *Junhyung Park, Simon Buchholz, Bernhard Schölkopf, Krikamol Muandet*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aadf1e309cc03cab3ec35afb7c9d0c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aadf1e309cc03cab3ec35afb7c9d0c8-Abstract-Conference.html)

        **Abstract**:

        Causality is a central concept in a wide range of research areas, yet there is still no universally agreed axiomatisation of causality. We view causality both as an extension of probability theory and as a study of what happens when one intervenes on a system, and argue in favour of taking Kolmogorov's measure-theoretic axiomatisation of probability as the starting point towards an axiomatisation of causality. To that end, we propose the notion of a causal space, consisting of a probability space along with a collection of transition probability kernels, called causal kernels, that encode the causal information of the space. Our proposed framework is not only rigorously grounded in measure theory, but it also sheds light on long-standing limitations of existing frameworks including, for example, cycles, latent variables and stochastic processes.

        ----

        ## [1239] LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

        **Authors**: *Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, Jianfeng Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Conversational generative AI has demonstrated remarkable promise for empowering biomedical practitioners, but current investigations focus on unimodal text. Multimodal conversational AI has seen rapid progress by leveraging billions of image-text pairs from the public web, but such general-domain vision-language models still lack sophistication in understanding and conversing about biomedical images. In this paper, we propose a cost-efficient approach for training a vision-language conversational assistant that can answer open-ended research questions of biomedical images. The key idea is to leverage a large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central, use GPT-4 to self-instruct open-ended instruction-following data from the captions, and then fine-tune a large general-domain vision-language model using a novel curriculum learning method. Specifically, the model first learns to align biomedical vocabulary using the figure-caption pairs as is, then learns to master open-ended conversational semantics using GPT-4 generated instruction-following data, broadly mimicking how a layperson gradually acquires biomedical knowledge. This enables us to train a Large Language and Vision Assistant for BioMedicine (LLaVA-Med) in less than 15 hours (with eight A100s). LLaVA-Med exhibits excellent multimodal conversational capability and can follow open-ended instruction to assist with inquiries about a biomedical image. On three standard biomedical visual question answering datasets, LLaVA-Med outperforms previous supervised state-of-the-art on certain metrics. To facilitate biomedical multimodal research, we will release our instruction-following data and the LLaVA-Med model.

        ----

        ## [1240] Networks are Slacking Off: Understanding Generalization Problem in Image Deraining

        **Authors**: *Jinjin Gu, Xianzheng Ma, Xiangtao Kong, Yu Qiao, Chao Dong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5aca18e0192b2c1300479e5b700c76a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5aca18e0192b2c1300479e5b700c76a9-Abstract-Conference.html)

        **Abstract**:

        Deep deraining networks consistently encounter substantial generalization issues when deployed in real-world applications, although they are successful in laboratory benchmarks. A prevailing perspective in deep learning encourages using highly complex data for training, with the expectation that richer image background content will facilitate overcoming the generalization problem. However, through comprehensive and systematic experimentation, we discover that this strategy does not enhance the generalization capability of these networks. On the contrary, it exacerbates the tendency of networks to overfit specific degradations. Our experiments reveal that better generalization in a deraining network can be achieved by simplifying the complexity of the training background images. This is because that the networks are ``slacking off'' during training, that is, learning the least complex elements in the image background and degradation to minimize training loss. When the background images are less complex than the rain streaks, the network will prioritize the background reconstruction, thereby suppressing overfitting the rain patterns and leading to improved generalization performance. Our research offers a valuable perspective and methodology for better understanding the generalization problem in low-level vision tasks and displays promising potential for practical application.

        ----

        ## [1241] Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning

        **Authors**: *Xiaojun Guo, Yifei Wang, Zeming Wei, Yisen Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5acf5a0ee5c17d372bfe7fdaeffd6e33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5acf5a0ee5c17d372bfe7fdaeffd6e33-Abstract-Conference.html)

        **Abstract**:

        With the prosperity of contrastive learning for visual representation learning (VCL), it is also adapted to the graph domain and yields promising performance. However, through a systematic study of various graph contrastive learning (GCL) methods, we observe that some common phenomena among existing GCL methods that are quite different from the original VCL methods, including 1) positive samples are not a must for GCL; 2) negative samples are not necessary for graph classification, neither for node classification when adopting specific normalization modules; 3) data augmentations have much less influence on GCL, as simple domain-agnostic augmentations (e.g., Gaussian noise) can also attain fairly good performance. By uncovering how the implicit inductive bias of GNNs works in contrastive learning, we theoretically provide insights into the above intriguing properties of GCL. Rather than directly porting existing VCL methods to GCL, we advocate for more attention toward the unique architecture of graph learning and consider its implicit influence when designing GCL methods. Code is available at https://github.com/PKU-ML/ArchitectureMattersGCL.

        ----

        ## [1242] Text Promptable Surgical Instrument Segmentation with Vision-Language Models

        **Authors**: *Zijian Zhou, Oluwatosin Alabi, Meng Wei, Tom Vercauteren, Miaojing Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5af741d487c5f0b08bfe56e11d1883e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5af741d487c5f0b08bfe56e11d1883e4-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose a novel text promptable surgical instrument segmentation approach to overcome challenges associated with diversity and differentiation of surgical instruments in minimally invasive surgeries. We redefine the task as text promptable, thereby enabling a more nuanced comprehension of surgical instruments and adaptability to new instrument types. Inspired by recent advancements in vision-language models, we leverage pretrained image and text encoders as our model backbone and design a text promptable mask decoder consisting of attention- and convolution-based prompting schemes for surgical instrument segmentation prediction. Our model leverages multiple text prompts for each surgical instrument through a new mixture of prompts mechanism, resulting in enhanced segmentation performance. Additionally, we introduce a hard instrument area reinforcement module to improve image feature comprehension and segmentation precision. Extensive experiments on several surgical instrument segmentation datasets demonstrate our model's superior performance and promising generalization capability. To our knowledge, this is the first implementation of a promptable approach to surgical instrument segmentation, offering significant potential for practical application in the field of robotic-assisted surgery. Code is available at https://github.com/franciszzj/TP-SIS.

        ----

        ## [1243] OpenDataVal: a Unified Benchmark for Data Valuation

        **Authors**: *Kevin Fu Jiang, Weixin Liang, James Y. Zou, Yongchan Kwon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b047c7d862059a5df623c1ce2982fca-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b047c7d862059a5df623c1ce2982fca-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Assessing the quality and impact of individual data points is critical for improving model performance and mitigating undesirable biases within the training dataset. Several data valuation algorithms have been proposed to quantify data quality, however, there lacks a systemic and standardized benchmarking system for data valuation. In this paper, we introduce OpenDataVal, an easy-to-use and unified benchmark framework that empowers researchers and practitioners to apply and compare various data valuation algorithms. OpenDataVal provides an integrated environment that includes (i) a diverse collection of image, natural language, and tabular datasets, (ii) implementations of eleven different state-of-the-art data valuation algorithms, and (iii) a prediction model API that can import any models in scikit-learn. Furthermore, we propose four downstream machine learning tasks for evaluating the quality of data values. We perform benchmarking analysis using OpenDataVal, quantifying and comparing the efficacy of state-of-the-art data valuation approaches. We find that no single algorithm performs uniformly best across all tasks, and an appropriate algorithm should be employed for a user's downstream task. OpenDataVal is publicly available at https://opendataval.github.io with comprehensive documentation. Furthermore, we provide a leaderboard where researchers can evaluate the effectiveness of their own data valuation algorithms.

        ----

        ## [1244] On the Consistency of Maximum Likelihood Estimation of Probabilistic Principal Component Analysis

        **Authors**: *Arghya Datta, Sayak Chakrabarty*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b0c0b2c2efdd736a53688ebfdc3bcdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b0c0b2c2efdd736a53688ebfdc3bcdb-Abstract-Conference.html)

        **Abstract**:

        Probabilistic principal component analysis (PPCA) is currently one of the most used statistical tools to reduce the ambient dimension of the data. From multidimensional scaling to the imputation of missing data, PPCA has a broad spectrum of applications ranging from science and engineering to quantitative finance.\Despite this wide applicability in various fields, hardly any theoretical guarantees exist to justify the soundness of the maximal likelihood (ML) solution for this model. In fact, it is well known that the maximum likelihood estimation (MLE) can only recover the true model parameters up to a rotation. The main obstruction is posed by the inherent identifiability nature of the PPCA model resulting from the rotational symmetry of the parameterization. To resolve this ambiguity, we propose a novel approach using quotient topological spaces and in particular, we show that the maximum likelihood solution is consistent in an appropriate quotient Euclidean space. Furthermore, our consistency results encompass a more general class of estimators beyond the MLE. Strong consistency of the ML estimate and consequently strong covariance estimation of the PPCA model have also been established under a compactness assumption.

        ----

        ## [1245] Similarity, Compression and Local Steps: Three Pillars of Efficient Communications for Distributed Variational Inequalities

        **Authors**: *Aleksandr Beznosikov, Martin Takác, Alexander V. Gasnikov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html)

        **Abstract**:

        Variational inequalities are a broad and flexible class of problems that includes minimization, saddle point, and fixed point problems as special cases. Therefore, variational inequalities are used in various applications ranging from equilibrium search to adversarial learning. With the increasing size of data and models, today's instances demand parallel and distributed computing for real-world machine learning problems, most of which can be represented as variational inequalities. Meanwhile, most distributed approaches have a significant bottleneck -- the cost of communications. The three main techniques to reduce the total number of communication rounds and the cost of one such round are the similarity of local functions, compression of transmitted information, and local updates. In this paper, we combine all these approaches. Such a triple synergy did not exist before for variational inequalities and saddle problems, nor even for minimization problems. The methods presented in this paper have the best theoretical guarantees of communication complexity and are significantly ahead of other methods for distributed variational inequalities. The theoretical results are confirmed by adversarial learning experiments on synthetic and real datasets.

        ----

        ## [1246] Lookaround Optimizer: k steps around, 1 step average

        **Authors**: *Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, Mingli Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b4b967d4222d87fa5b28b6ec7144058-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b4b967d4222d87fa5b28b6ec7144058-Abstract-Conference.html)

        **Abstract**:

        Weight Average (WA) is an active research topic due to its simplicity in ensembling deep networks and the effectiveness in promoting generalization. Existing weight average approaches, however, are often carried out along only one training trajectory in a post-hoc manner (i.e., the weights are averaged after the entire training process is finished), which significantly degrades the diversity between networks and thus impairs the effectiveness. In this paper, inspired by weight average, we propose Lookaround, a straightforward yet effective SGD-based optimizer leading to flatter minima with better generalization. Specifically, Lookaround iterates two steps during the whole training period: the around step and the average step. In each iteration, 1) the around step starts from a common point and trains multiple networks simultaneously, each on transformed data by a different data augmentation, and 2) the average step averages these trained networks to get the averaged network, which serves as the starting point for the next iteration. The around step improves the functionality diversity while the average step guarantees the weight locality of these networks during the whole training, which is essential for WA to work. We theoretically explain the superiority of Lookaround by convergence analysis, and make extensive experiments to evaluate Lookaround on popular benchmarks including CIFAR and ImageNet with both CNNs and ViTs, demonstrating clear superiority over state-of-the-arts. Our code is available at https://github.com/Ardcy/Lookaround.

        ----

        ## [1247] The Quantization Model of Neural Scaling

        **Authors**: *Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html)

        **Abstract**:

        We propose the Quantization Model of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale.  We derive this model from what we call the Quantization Hypothesis, where network knowledge and skills are "quantized" into discrete chunks (quanta). We show that when quanta are learned in order of decreasing use frequency, then a power law in use frequencies explains observed power law scaling of loss. We validate this prediction on toy datasets, then study how scaling curves decompose for large language models.  Using language model gradients, we automatically decompose model behavior into a diverse set of skills (quanta). We tentatively find that the frequency at which these quanta are used in the training distribution roughly follows a power law corresponding with the empirical scaling exponent for language models, a prediction of our theory.

        ----

        ## [1248] ε-fractional core stability in Hedonic Games

        **Authors**: *Simone Fioravanti, Michele Flammini, Bojana Kodric, Giovanna Varricchio*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b755cf5598a4324d253025e1fbbba52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b755cf5598a4324d253025e1fbbba52-Abstract-Conference.html)

        **Abstract**:

        Hedonic Games (HGs) are a classical framework modeling coalition formation of strategic agents guided by their individual preferences. According to these preferences, it is desirable that a coalition structure (i.e. a partition of agents into coalitions) satisfies some form of stability. The most well-known and natural of such notions is arguably core-stability. Informally, a partition is core-stable if no subset of agents would like to deviate by regrouping in a so-called core-blocking coalition. Unfortunately, core-stable partitions seldom exist and even when they do, it is often computationally intractable to find one. To circumvent these problems, we propose the notion of $\varepsilon$-fractional core-stability, where at most an $\varepsilon$-fraction of all possible coalitions is allowed to core-block. It turns out that such a relaxation may guarantee both existence and polynomial-time computation. Specifically, we design efficient algorithms returning an $\varepsilon$-fractional core-stable partition, with $\varepsilon$ exponentially decreasing in the number of agents, for two fundamental classes of HGs: Simple Fractional and Anonymous. From a probabilistic point of view, being the definition of $\varepsilon$-fractional core equivalent to requiring that uniformly sampled coalitions core-block with probability lower than $\varepsilon$, we further extend the definition to handle more complex sampling distributions. Along this line, when valuations have to be learned from samples in a PAC-learning fashion, we give positive and negative results on which distributions allow the efficient computation of outcomes that are $\varepsilon$-fractional core-stable with arbitrarily high confidence.

        ----

        ## [1249] Semi-Supervised Domain Generalization with Known and Unknown Classes

        **Authors**: *Lei Zhang, Ji-Fu Li, Wei Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html)

        **Abstract**:

        Semi-Supervised Domain Generalization (SSDG) aims to learn a model that is generalizable to an unseen target domain with only a few labels, and most existing SSDG methods assume that unlabeled training and testing samples are all known classes. However, a more realistic scenario is that known classes may be mixed with some unknown classes in unlabeled training and testing data. To deal with such a scenario, we propose the Class-Wise Adaptive Exploration and Exploitation (CWAEE) method. In particular, we explore unlabeled training data by using one-vs-rest classifiers and class-wise adaptive thresholds to detect known and unknown classes, and exploit them by adopting consistency regularization on augmented samples based on Fourier Transformation to improve the unseen domain generalization. The experiments conducted on real-world datasets verify the effectiveness and superiority of our method.

        ----

        ## [1250] When Do Graph Neural Networks Help with Node Classification? Investigating the Homophily Principle on Node Distinguishability

        **Authors**: *Sitao Luan, Chenqing Hua, Minkai Xu, Qincheng Lu, Jiaqi Zhu, Xiao-Wen Chang, Jie Fu, Jure Leskovec, Doina Precup*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ba11de4c74548071899cf41dec078bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ba11de4c74548071899cf41dec078bf-Abstract-Conference.html)

        **Abstract**:

        Homophily principle, i.e., nodes with the same labels are more likely to be connected, has been believed to be the main reason for the performance superiority of Graph Neural Networks (GNNs) over Neural Networks on node classification tasks. Recent research suggests that, even in the absence of homophily, the advantage of GNNs still exists as long as nodes from the same class share similar neighborhood patterns. However, this argument only considers intra-class Node Distinguishability (ND) but neglects inter-class ND, which provides incomplete understanding of homophily on GNNs. In this paper, we first demonstrate such deficiency with examples and argue that an ideal situation for ND is to have smaller intra-class ND than inter-class ND. To formulate this idea and study ND deeply, we propose Contextual Stochastic Block Model for Homophily (CSBM-H) and define two metrics, Probabilistic Bayes Error (PBE) and negative generalized Jeffreys divergence, to quantify ND. With the metrics, we visualize and analyze how graph filters, node degree distributions and class variances influence ND, and investigate the combined effect of intra- and inter-class ND. Besides, we discovered the mid-homophily pitfall, which occurs widely in graph datasets. Furthermore, we verified that, in real-work tasks, the superiority of GNNs is indeed closely related to both intra- and inter-class ND regardless of homophily levels. Grounded in this observation, we propose a new hypothesis-testing based performance metric beyond homophily, which is non-linear, feature-based and can provide statistical threshold value for GNNs' the superiority. Experiments indicate that it is significantly more effective than the existing homophily metrics on revealing the advantage and disadvantage of graph-aware modes on both synthetic and benchmark real-world datasets.

        ----

        ## [1251] (Almost) Provable Error Bounds Under Distribution Shift via Disagreement Discrepancy

        **Authors**: *Elan Rosenfeld, Saurabh Garg*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bacb12bf81e98e2ee0eed953a23c656-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bacb12bf81e98e2ee0eed953a23c656-Abstract-Conference.html)

        **Abstract**:

        We derive a new, (almost) guaranteed upper bound on the error of deep neural networks under distribution shift using unlabeled test data. Prior methods are either vacuous in practice or accurate on average but heavily underestimate error for a sizeable fraction of shifts. In particular, the latter only give guarantees based on complex continuous measures such as test calibration, which cannot be identified without labels, and are therefore unreliable. Instead, our bound requires a simple, intuitive condition which is well justified by prior empirical works and holds in practice effectively 100\% of the time. The bound is inspired by $\mathcal{H}\Delta\mathcal{H}$-divergence but is easier to evaluate and substantially tighter, consistently providing non-vacuous test error upper bounds. Estimating the bound requires optimizing one multiclass classifier to disagree with another, for which some prior works have used sub-optimal proxy losses; we devise a "disagreement loss" which is theoretically justified and performs better in practice. We expect this loss can serve as a drop-in replacement for future methods which require maximizing multiclass disagreement. Across a wide range of natural and synthetic distribution shift benchmarks, our method gives valid error bounds while achieving average accuracy comparable to—though not better than—competitive estimation baselines.

        ----

        ## [1252] Schema-learning and rebinding as mechanisms of in-context learning and emergence

        **Authors**: *Sivaramakrishnan Swaminathan, Antoine Dedieu, Rajkumar Vasudeva Raju, Murray Shanahan, Miguel Lázaro-Gredilla, Dileep George*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html)

        **Abstract**:

        In-context learning (ICL) is one of the most powerful and most unexpected capabilities to emerge in recent transformer-based large language models (LLMs). Yet the mechanisms that underlie it are poorly understood. In this paper, we demonstrate that comparable ICL capabilities can be acquired by an alternative sequence prediction learning method using clone-structured causal graphs (CSCGs). Moreover, a key property of CSCGs is that, unlike transformer-based LLMs, they are {\em interpretable}, which considerably simplifies the task of explaining how ICL works. Specifically, we show that it uses a combination of (a) learning template (schema) circuits for pattern completion, (b) retrieving relevant templates in a context-sensitive manner, and (c) rebinding of novel tokens to appropriate slots in the templates. We go on to marshall evidence for the hypothesis that similar mechanisms underlie ICL in LLMs. For example, we find that, with CSCGs as with LLMs, different capabilities emerge at different levels of overparameterization, suggesting that overparameterization helps in learning more complex template (schema) circuits. By showing how ICL can be achieved with small models and datasets, we open up a path to novel architectures, and take a vital step towards a more general understanding of the mechanics behind this important capability.

        ----

        ## [1253] CAP: Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models

        **Authors**: *Denis Kuznedelev, Eldar Kurtic, Elias Frantar, Dan Alistarh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html)

        **Abstract**:

        Driven by significant improvements in architectural design and training pipelines, computer visionhas recently experienced dramatic progress in terms of accuracy on classic benchmarks such as ImageNet. These highly-accurate models are challenging to deploy,  as they appear harder to compress using standard techniques such as pruning. We address this issue by introducing the Correlation Aware Pruner (CAP), a new unstructured pruning framework which significantly pushes the compressibility limits for state-of-the-art architectures.Our method is based on two technical advancements: a new theoretically-justified pruner, which can handle complex weight correlations accurately and efficiently during the pruning process itself,  and an efficient finetuning procedure for post-compression recovery. We validate our approach via extensive experiments on several modern vision models such as Vision Transformers (ViT), modern CNNs, and ViT-CNN hybrids, showing for the first time that these can be pruned to high sparsity levels (e.g. $\geq 75$%) with low impact on accuracy ($\leq 1$% relative drop). Our approach is also compatible with structured pruning and quantization, and can lead to practical speedups of 1.5 to 2.4x without accuracy loss. To further showcase CAP's accuracy and scalability, we use it to show for the first time that extremely-accurate large vision models, trained via self-supervised techniques,  can also be pruned to moderate sparsities, with negligible accuracy loss.

        ----

        ## [1254] Your representations are in the network: composable and parallel adaptation for large scale models

        **Authors**: *Yonatan Dukler, Alessandro Achille, Hao Yang, Varsha Vivek, Luca Zancato, Benjamin Bowman, Avinash Ravichandran, Charless C. Fowlkes, Ashwin Swaminathan, Stefano Soatto*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5be3783ea9d43d7add5409c101d87d83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5be3783ea9d43d7add5409c101d87d83-Abstract-Conference.html)

        **Abstract**:

        We present a framework for transfer learning that efficiently adapts a large base-model by learning lightweight cross-attention modules attached to its intermediate activations.We name our approach InCA (Introspective-Cross-Attention) and show that it can efficiently survey a networkâ€™s representations and identify strong performing adapter models for a downstream task.During training, InCA enables training numerous adapters efficiently and in parallel, isolated from the frozen base model. On the ViT-L/16 architecture, our experiments show that a single adapter, 1.3% of the full model, is able to reach full fine-tuning accuracy on average across 11 challenging downstream classification tasks.Compared with other forms of parameter-efficient adaptation, the isolated nature of the InCA adaptation is computationally desirable for large-scale models. For instance, we adapt ViT-G/14 (1.8B+ parameters) quickly with 20+ adapters in parallel on a single V100 GPU (76% GPU memory reduction) and exhaustively identify its most useful representations.We further demonstrate how the adapters learned by InCA can be incrementally modified or combined for flexible learning scenarios and our approach achieves state of the art performance on the ImageNet-to-Sketch multi-task benchmark.

        ----

        ## [1255] Learning Energy-based Model via Dual-MCMC Teaching

        **Authors**: *Jiali Cui, Tian Han*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html)

        **Abstract**:

        This paper studies the fundamental learning problem of the energy-based model (EBM). Learning the EBM can be achieved using the maximum likelihood estimation (MLE), which typically involves the Markov Chain Monte Carlo (MCMC) sampling, such as the Langevin dynamics. However, the noise-initialized Langevin dynamics can be challenging in practice and hard to mix. This motivates the exploration of joint training with the generator model where the generator model serves as a complementary model to bypass MCMC sampling. However, such a method can be less accurate than the MCMC and result in biased EBM learning. While the generator can also serve as an initializer model for better MCMC sampling, its learning can be biased since it only matches the EBM and has no access to empirical training examples. Such biased generator learning may limit the potential of learning the EBM. To address this issue, we present a joint learning framework that interweaves the maximum likelihood learning algorithm for both the EBM and the complementary generator model. In particular, the generator model is learned by MLE to match both the EBM and the empirical data distribution, making it a more informative initializer for MCMC sampling of EBM. Learning generator with observed examples typically requires inference of the generator posterior. To ensure accurate and efficient inference, we adopt the MCMC posterior sampling and introduce a complementary inference model to initialize such latent MCMC sampling. We show that three separate models can be seamlessly integrated into our joint framework through two (dual-) MCMC teaching, enabling effective and efficient EBM learning.

        ----

        ## [1256] Performance-optimized deep neural networks are evolving into worse models of inferotemporal visual cortex

        **Authors**: *Drew Linsley, Ivan F. Rodriguez Rodriguez, Thomas Fel, Michael Arcaro, Saloni Sharma, Margaret S. Livingstone, Thomas Serre*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bf234ecf83cd77bc5b77a24ba9338b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bf234ecf83cd77bc5b77a24ba9338b0-Abstract-Conference.html)

        **Abstract**:

        One of the most impactful findings in computational neuroscience over the past decade is that the object recognition accuracy of deep neural networks (DNNs) correlates with their ability to predict neural responses to natural images in the inferotemporal (IT) cortex. This discovery supported the long-held theory that object recognition is a core objective of the visual cortex, and suggested that more accurate DNNs would serve as better models of IT neuron responses to images. Since then, deep learning has undergone a revolution of scale: billion parameter-scale DNNs trained on billions of images are rivaling or outperforming humans at visual tasks including object recognition. Have today's DNNs become more accurate at predicting IT neuron responses to images as they have grown more accurate at object recognition?Surprisingly, across three independent experiments, we find that this is not the case. DNNs have become progressively worse models of IT as their accuracy has increased on ImageNet. To understand why DNNs experience this trade-off and evaluate if they are still an appropriate paradigm for modeling the visual system, we turn to recordings of IT that capture spatially resolved maps of neuronal activity elicited by natural images. These neuronal activity maps reveal that DNNs trained on ImageNet learn to rely on different visual features than those encoded by IT and that this problem worsens as their accuracy increases. We successfully resolved this issue with the neural harmonizer, a plug-and-play training routine for DNNs that aligns their learned representations with humans. Our results suggest that harmonized DNNs break the trade-off between ImageNet accuracy and neural prediction accuracy that assails current DNNs and offer a path to more accurate models of biological vision. Our work indicates that the standard approach for modeling IT with task-optimized DNNs needs revision, and other biological constraints, including human psychophysics data, are needed to accurately reverse-engineer the visual cortex.

        ----

        ## [1257] Private Federated Frequency Estimation: Adapting to the Hardness of the Instance

        **Authors**: *Jingfeng Wu, Wennan Zhu, Peter Kairouz, Vladimir Braverman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5bf40077b2bac53399676d33d564ef58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5bf40077b2bac53399676d33d564ef58-Abstract-Conference.html)

        **Abstract**:

        In federated frequency estimation (FFE), multiple clients work together to estimate the frequency of their local data by communicating with a server, while maintaining the security constraint of $\mathtt{secsum}$ where the server can only access the sum of client-held vectors. For FFE with a single communication round, it is known that count sketch is nearly information-theoretically optimal [Chen et al., 2022]. However, when multiple communication rounds are allowed, we propose a new sketch algorithm that is provably more accurate than a naive adaptation of count sketch. Furthermore, we show that both our sketch algorithm and count sketch can achieve better accuracy when the problem instance is simpler. Therefore, we propose a two-phase approach to enable the use of a smaller sketch size for simpler problems. Finally, we provide mechanisms to make our proposed algorithm differentially private. We verify the performance of our methods through experiments conducted on real datasets.

        ----

        ## [1258] On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

        **Authors**: *Federico Errica*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html)

        **Abstract**:

        Researchers have used nearest neighbor graphs to transform classical machine learning problems on tabular data into node classification tasks to solve with graph representation learning methods. Such artificial structures often reflect the homophily assumption, believed to be a key factor in the performances of deep graph networks. In light of recent results demystifying these beliefs, we introduce a theoretical framework to understand the benefits of Nearest Neighbor (NN) graphs when a graph structure is missing. We formally analyze the Cross-Class Neighborhood Similarity (CCNS), used to empirically evaluate the usefulness of structures, in the context of nearest neighbor graphs. Moreover, we study the class separability induced by deep graph networks on a k-NN graph. Motivated by the theory, our quantitative experiments demonstrate that, under full supervision, employing a k-NN graph offers no benefits compared to a structure-agnostic baseline. Qualitative analyses suggest that our framework is good at estimating the CCNS and hint at k-NN graphs never being useful for such classification tasks under full supervision, thus advocating for the study of alternative graph construction techniques in combination with deep graph networks.

        ----

        ## [1259] VRA: Variational Rectified Activation for Out-of-distribution Detection

        **Authors**: *Mingyu Xu, Zheng Lian, Bin Liu, Jianhua Tao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c20c00504e0c049ec2370d0cceaf3c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c20c00504e0c049ec2370d0cceaf3c4-Abstract-Conference.html)

        **Abstract**:

        Out-of-distribution (OOD) detection is critical to building reliable machine learning systems in the open world. Researchers have proposed various strategies to reduce model overconfidence on OOD data. Among them, ReAct is a typical and effective technique to deal with model overconfidence, which truncates high activations to increase the gap between in-distribution and OOD. Despite its promising results, is this technique the best choice? To answer this question, we leverage the variational method to find the optimal operation and verify the necessity of suppressing abnormally low and high activations and amplifying intermediate activations in OOD detection, rather than focusing only on high activations like ReAct. This motivates us to propose a novel technique called ``Variational Rectified Activation (VRA)'', which simulates these suppression and amplification operations using piecewise functions. Experimental results on multiple benchmark datasets demonstrate that our method outperforms existing post-hoc strategies. Meanwhile, VRA is compatible with different scoring functions and network architectures. Our code is available at https://github.com/zeroQiaoba/VRA.

        ----

        ## [1260] Variational Gaussian processes for linear inverse problems

        **Authors**: *Thibault Randrianarisoa, Botond Szabó*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c25c15b5b2fd386ab188a918e54c7d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c25c15b5b2fd386ab188a918e54c7d5-Abstract-Conference.html)

        **Abstract**:

        By now Bayesian methods are routinely used in practice for solving inverse problems. In inverse problems the parameter or signal of interest is observed only indirectly, as an image of a given map, and the observations are typically further corrupted with noise. Bayes offers a natural way to regularize these problems via the prior distribution and provides a probabilistic solution, quantifying the remaining uncertainty in the problem. However, the computational costs of standard, sampling based Bayesian approaches can be overly large in such complex models. Therefore, in practice variational Bayes is becoming increasingly popular. Nevertheless, the theoretical understanding of these methods is still relatively limited, especially in context of inverse problems.In our analysis we investigate variational Bayesian methods for Gaussian process priors to solve linear inverse problems. We consider both mildly and severely ill-posed inverse problems and work with the popular inducing variable variational Bayes approach proposed by Titsias [Titsias, 2009]. We derive posterior contraction rates for the variational posterior in general settings and show that the minimax estimation rate can be attained by correctly tunned procedures. As specific examples we consider a collection of inverse problems including the heat equation, Volterra operator and Radon transform and inducing variable methods based on population and empirical spectral features.

        ----

        ## [1261] Self-Supervised Learning with Lie Symmetries for Partial Differential Equations

        **Authors**: *Grégoire Mialon, Quentin Garrido, Hannah Lawrence, Danyal Rehman, Yann LeCun, Bobak T. Kiani*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c46ae130105fa012da0446126c01d1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c46ae130105fa012da0446126c01d1d-Abstract-Conference.html)

        **Abstract**:

        Machine learning for differential equations paves the way for computationally efficient alternatives to numerical solvers, with potentially broad impacts in science and engineering. Though current algorithms typically require simulated training data tailored to a given setting, one may instead wish to learn useful information from heterogeneous sources, or from real dynamical systems observations that are messy or incomplete. In this work, we learn general-purpose representations of PDEs from heterogeneous data by implementing joint embedding methods for self-supervised learning (SSL), a framework for unsupervised representation learning that has had notable success in computer vision. Our representation outperforms baseline approaches to invariant tasks, such as regressing the coefficients of a PDE, while also improving the time-stepping performance of neural solvers. We hope that our proposed methodology will prove useful in the eventual development of general-purpose foundation models for PDEs.

        ----

        ## [1262] TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

        **Authors**: *Jialin Chen, Rex Ying*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html)

        **Abstract**:

        Temporal graphs are widely used to model dynamic systems with time-varying interactions. In real-world scenarios, the underlying mechanisms of generating future interactions in dynamic systems are typically governed by a set of recurring substructures within the graph, known as temporal motifs. Despite the success and prevalence of current temporal graph neural networks (TGNN), it remains uncertain which temporal motifs are recognized as the significant indications that trigger a certain prediction from the model, which is a critical challenge for advancing the explainability and trustworthiness of current TGNNs. To address this challenge, we propose a novel approach, called Temporal Motifs Explainer (TempME), which uncovers the most pivotal temporal motifs guiding the prediction of TGNNs.  Derived from the information bottleneck principle, TempME extracts the most interaction-related motifs while minimizing the amount of contained information to preserve the sparsity and succinctness of the explanation. Events in the explanations generated by TempME are verified to be more spatiotemporally correlated than those of existing approaches, providing more understandable insights. Extensive experiments validate the superiority of TempME, with up to 8.21% increase in terms of explanation accuracy across six real-world datasets and up to 22.96% increase in boosting the prediction Average Precision of current TGNNs.

        ----

        ## [1263] YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus

        **Authors**: *David Uthus, Garrett Tanzer, Manfred Georg*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c61452daca5f0c260e683b317d13a3f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Machine learning for sign languages is bottlenecked by data. In this paper, we present YouTube-ASL, a large-scale, open-domain corpus of American Sign Language (ASL) videos and accompanying English captions drawn from YouTube. With ~1000 hours of videos and >2500 unique signers, YouTube-ASL is ~3x as large and has ~10x as many unique signers as the largest prior ASL dataset. We train baseline models for ASL to English translation on YouTube-ASL and evaluate them on How2Sign, where we achieve a new fine-tuned state of the art of 12.397 BLEU and, for the first time, nontrivial zero-shot results.

        ----

        ## [1264] Finite-Time Analysis of Whittle Index based Q-Learning for Restless Multi-Armed Bandits with Neural Network Function Approximation

        **Authors**: *Guojun Xiong, Jian Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5c7c66dfc9f93f0c738947f3b1c13832-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5c7c66dfc9f93f0c738947f3b1c13832-Abstract-Conference.html)

        **Abstract**:

        Whittle index policy is a heuristic to the intractable restless multi-armed bandits (RMAB) problem. Although it is provably asymptotically optimal, finding Whittle indices remains difficult.  In this paper, we present Neural-Q-Whittle, a Whittle index based Q-learning algorithm for RMAB with neural network function approximation, which is an example of  nonlinear two-timescale stochastic approximation with Q-function values updated on a faster timescale and Whittle indices on a slower timescale.  Despite the empirical success of deep Q-learning, the non-asymptotic convergence rate of Neural-Q-Whittle, which couples neural networks with two-timescale Q-learning largely remains unclear.  This paper provides a finite-time analysis of Neural-Q-Whittle, where data are generated from a Markov chain, and Q-function is approximated by a ReLU neural network. Our analysis leverages a Lyapunov drift approach to capture the evolution of two coupled parameters, and the nonlinearity in value function approximation further requires us to characterize the approximation error.  Combing these provide Neural-Q-Whittle  with $\mathcal{O}(1/k^{2/3})$ convergence rate, where $k$ is the number of iterations.

        ----

        ## [1265] Adapting to Continuous Covariate Shift via Online Density Ratio Estimation

        **Authors**: *Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cad96c4433955a2e76749ee74a424f5-Abstract-Conference.html)

        **Abstract**:

        Dealing with distribution shifts is one of the central challenges for modern machine learning. One fundamental situation is the covariate shift, where the input distributions of data change from the training to testing stages while the input-conditional output distribution remains unchanged. In this paper, we initiate the study of a more challenging scenario --- continuous covariate shift --- in which the test data appear sequentially, and their distributions can shift continuously. Our goal is to adaptively train the predictor such that its prediction risk accumulated over time can be minimized. Starting with the importance-weighted learning, we theoretically show the method works effectively if the time-varying density ratios of test and train inputs can be accurately estimated. However, existing density ratio estimation methods would fail due to data scarcity at each time step. To this end, we propose an online density ratio estimation method that can appropriately reuse historical information. Our method is proven to perform well by enjoying a dynamic regret bound, which finally leads to an excess risk guarantee for the predictor. Empirical results also validate the effectiveness.

        ----

        ## [1266] Hierarchical Integration Diffusion Model for Realistic Image Deblurring

        **Authors**: *Zheng Chen, Yulun Zhang, Ding Liu, Bin Xia, Jinjin Gu, Linghe Kong, Xin Yuan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html)

        **Abstract**:

        Diffusion models (DMs) have recently been introduced in image deblurring and exhibited promising performance, particularly in terms of details reconstruction. However, the diffusion model requires a large number of inference iterations to recover the clean image from pure Gaussian noise, which consumes massive computational resources. Moreover, the distribution synthesized by the diffusion model is often misaligned with the target results, leading to restrictions in distortion-based metrics. To address the above issues, we propose the Hierarchical Integration Diffusion Model (HI-Diff), for realistic image deblurring. Specifically, we perform the DM in a highly compacted latent space to generate the prior feature for the deblurring process. The deblurring process is implemented by a regression-based method to obtain better distortion accuracy. Meanwhile, the highly compact latent space ensures the efficiency of the DM. Furthermore, we design the hierarchical integration module to fuse the prior into the regression-based model from multiple scales, enabling better generalization in complex blurry scenarios. Comprehensive experiments on synthetic and real-world blur datasets demonstrate that our HI-Diff outperforms state-of-the-art methods. Code and trained models are available at https://github.com/zhengchen1999/HI-Diff.

        ----

        ## [1267] Efficient Beam Tree Recursion

        **Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5cf93940e37f7a7877cd57b6dba6b7ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5cf93940e37f7a7877cd57b6dba6b7ab-Abstract-Conference.html)

        **Abstract**:

        Beam Tree Recursive Neural Network (BT-RvNN) was recently proposed as an extension of Gumbel Tree RvNN and it was shown to achieve state-of-the-art length generalization performance in ListOps while maintaining comparable performance on other tasks. However, although better than previous approaches in terms of memory usage, BT-RvNN can be still exorbitantly expensive. In this paper, we identify the main bottleneck in BT-RvNN's memory usage to be the entanglement of the scorer function and the recursive cell function. We propose strategies to remove this bottleneck and further simplify its memory usage. Overall, our strategies not only reduce the memory usage of BT-RvNN by $10-16$ times but also create a new state-of-the-art in ListOps while maintaining similar performance in other tasks. In addition, we also propose a strategy to utilize the induced latent-tree node representations produced by BT-RvNN to turn BT-RvNN from a sentence encoder of the form $f:\mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{d}$ into a token contextualizer of the form  $f:\mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}$. Thus, our proposals not only open up a path for further scalability of RvNNs but also standardize a way to use BT-RvNNs as another building block in the deep learning toolkit that can be easily stacked or interfaced with other popular models such as Transformers and Structured State Space models. Our code is available at the link: https://github.com/JRC1995/BeamRecursionFamily.

        ----

        ## [1268] Holistic Transfer: Towards Non-Disruptive Fine-Tuning with Partial Target Data

        **Authors**: *Cheng-Hao Tu, Hong-You Chen, Zheda Mai, Jike Zhong, Vardaan Pahuja, Tanya Y. Berger-Wolf, Song Gao, Charles V. Stewart, Yu Su, Wei-Lun Chao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html)

        **Abstract**:

        We propose a learning problem involving adapting a pre-trained source model to the target domain for classifying all classes that appeared in the source data, using target data that covers only a partial label space. This problem is practical, as it is unrealistic for the target end-users to collect data for all classes prior to adaptation. However, it has received limited attention in the literature. To shed light on this issue, we construct benchmark datasets and conduct extensive experiments to uncover the inherent challenges. We found a dilemma --- on the one hand, adapting to the new target domain is important to claim better performance; on the other hand, we observe that preserving the classification accuracy of classes missing in the target adaptation data is highly challenging, let alone improving them. To tackle this, we identify two key directions: 1) disentangling domain gradients from classification gradients, and 2) preserving class relationships. We present several effective solutions that maintain the accuracy of the missing classes and enhance the overall performance, establishing solid baselines for holistic transfer of pre-trained models with partial target data.

        ----

        ## [1269] Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition

        **Authors**: *Divin Yan, Gengchen Wei, Chen Yang, Shengzhong Zhang, Zengfeng Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d1233f819202ade06023346df80a6d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d1233f819202ade06023346df80a6d2-Abstract-Conference.html)

        **Abstract**:

        This paper introduces a new approach to address the issue of class imbalance in graph neural networks (GNNs) for learning on graph-structured data. Our approach integrates imbalanced node classification and Bias-Variance Decomposition, establishing a theoretical framework that closely relates data imbalance to model variance. We also leverage graph augmentation technique to estimate the variance and design a regularization term to alleviate the impact of imbalance. Exhaustive tests are conducted on multiple benchmarks, including naturally imbalanced datasets and public-split class-imbalanced datasets, demonstrating that our approach outperforms state-of-the-art methods in various imbalanced scenarios. This work provides a novel theoretical perspective for addressing the problem of imbalanced node classification in GNNs.

        ----

        ## [1270] Practical Equivariances via Relational Conditional Neural Processes

        **Authors**: *Daolang Huang, Manuel Haussmann, Ulpu Remes, St John, Grégoire Clarté, Kevin Sebastian Luck, Samuel Kaski, Luigi Acerbi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d1a382162cb5ed326f1d3dbbfac4c82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d1a382162cb5ed326f1d3dbbfac4c82-Abstract-Conference.html)

        **Abstract**:

        Conditional Neural Processes (CNPs) are a class of metalearning models popular for combining the runtime efficiency of amortized inference with reliable uncertainty quantification. Many relevant machine learning tasks, such as in spatio-temporal modeling, Bayesian Optimization and continuous control, inherently contain equivariances – for example to translation – which the model can exploit for maximal performance. However, prior attempts to include equivariances in CNPs do not scale effectively beyond two input dimensions. In this work, we propose Relational Conditional Neural Processes (RCNPs), an effective approach to incorporate equivariances into any neural process model. Our proposed method extends the applicability and impact of equivariant neural processes to higher dimensions. We empirically demonstrate the competitive performance of RCNPs on a large array of tasks naturally containing equivariances.

        ----

        ## [1271] FourierHandFlow: Neural 4D Hand Representation Using Fourier Query Flow

        **Authors**: *Jihyun Lee, Junbong Jang, Donghwan Kim, Minhyuk Sung, Tae-Kyun Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)

        **Abstract**:

        Recent 4D shape representations model continuous temporal evolution of implicit shapes by (1) learning query flows without leveraging shape and articulation priors or (2) decoding shape occupancies separately for each time value. Thus, they do not effectively capture implicit correspondences between articulated shapes or regularize jittery temporal deformations. In this work, we present FourierHandFlow, which is a spatio-temporally continuous representation for human hands that combines a 3D occupancy field with articulation-aware query flows represented as Fourier series. Given an input RGB sequence, we aim to learn a fixed number of Fourier coefficients for each query flow to guarantee smooth and continuous temporal shape dynamics. To effectively model spatio-temporal deformations of articulated hands, we compose our 4D representation based on two types of Fourier query flow: (1) pose flow that models query dynamics influenced by hand articulation changes via implicit linear blend skinning and (2) shape flow that models query-wise displacement flow. In the experiments, our method achieves state-of-the-art results on video-based 4D reconstruction while being computationally more efficient than the existing 3D/4D implicit shape representations. We additionally show our results on motion inter- and extrapolation and texture transfer using the learned correspondences of implicit shapes. To the best of our knowledge, FourierHandFlow is the first neural 4D continuous hand representation learned from RGB videos. The code will be publicly accessible.

        ----

        ## [1272] Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms

        **Authors**: *Akifumi Wachi, Wataru Hashimoto, Xun Shen, Kazumune Hashimoto*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d4cd12ef6efedbf26b69b410f1f7d67-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d4cd12ef6efedbf26b69b410f1f7d67-Abstract-Conference.html)

        **Abstract**:

        Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.

        ----

        ## [1273] RevColV2: Exploring Disentangled Representations in Masked Image Modeling

        **Authors**: *Qi Han, Yuxuan Cai, Xiangyu Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d56e69c317429945785ede86c00b44e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d56e69c317429945785ede86c00b44e-Abstract-Conference.html)

        **Abstract**:

        Masked image modeling (MIM) has become a prevalent pre-training setup for vision foundation models and attains promising performance. Despite its success, existing MIM methods discard the decoder network during downstream applica- tions, resulting in inconsistent representations between pre-training and fine-tuning and can hamper downstream task performance. In this paper, we propose a new architecture, RevColV2, which tackles this issue by keeping the entire autoen- coder architecture during both pre-training and fine-tuning. The main body of RevColV2 contains bottom-up columns and top-down columns, between which information is reversibly propagated and gradually disentangled. Such design enables our architecture with the nice property: maintaining disentangled low-level and semantic information at the end of the network in MIM pre-training. Our experimental results suggest that a foundation model with decoupled features can achieve competitive performance across multiple downstream vision tasks such as image classification, semantic segmentation and object detection. For exam- ple, after intermediate fine-tuning on ImageNet-22K dataset, RevColV2-L attains 88.4\% top-1 accuracy on ImageNet-1K classification and 58.6 mIoU on ADE20K semantic segmentation. With extra teacher and large scale dataset, RevColv2-L achieves 62.1 APbox on COCO detection and 60.4 mIoU on ADE20K semantic segmentation.

        ----

        ## [1274] Mass-Producing Failures of Multimodal Systems with Language Models

        **Authors**: *Shengbang Tong, Erik Jones, Jacob Steinhardt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d570ed1708bbe19cb60f7a7aff60575-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d570ed1708bbe19cb60f7a7aff60575-Abstract-Conference.html)

        **Abstract**:

        Deployed multimodal models can fail in ways that evaluators did not anticipate. In order to find these failures before deployment, we introduce MultiMon, a system that automatically identifies systematic failures---generalizable, natural-language descriptions that describe categories of individual failures. To uncover systematic failures, MultiMon scrapes for examples of erroneous agreement: inputs that produce the same output, but should not. It then prompts a language model to identify common categories and describe them in natural language. We use MultiMon to find 14 systematic failures (e.g."ignores quantifiers'') of the CLIP text-encoder, each comprising hundreds of distinct inputs (e.g."a shelf with a few/many books''). Because CLIP is the backbone for most state-of-the-art multimodal models, these inputs produce failures in Midjourney 5.1, DALL-E, VideoFusion, and others. MultiMon can also steer towards failures relevant to specific use cases, such as self-driving cars. We see MultiMon as a step towards evaluation that autonomously explores the long-tail of potential system failures.

        ----

        ## [1275] STXD: Structural and Temporal Cross-Modal Distillation for Multi-View 3D Object Detection

        **Authors**: *Sujin Jang, Dae Ung Jo, Sung Ju Hwang, Dongwook Lee, Daehyun Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html)

        **Abstract**:

        3D object detection (3DOD) from multi-view images is an economically appealing alternative to expensive LiDAR-based detectors, but also an extremely challenging task due to the absence of precise spatial cues. Recent studies have leveraged the teacher-student paradigm for cross-modal distillation, where a strong LiDAR-modality teacher transfers useful knowledge to a multi-view-based image-modality student. However, prior approaches have only focused on minimizing global distances between cross-modal features, which may lead to suboptimal knowledge distillation results. Based on these insights, we propose a novel structural and temporal cross-modal knowledge distillation (STXD) framework for multi-view 3DOD. First, STXD reduces redundancy of the feature components of the student by regularizing the cross-correlation of cross-modal features, while maximizing their similarities. Second, to effectively transfer temporal knowledge, STXD encodes temporal relations of features across a sequence of frames via similarity maps. Lastly, STXD also adopts a response distillation method to further enhance the quality of knowledge distillation at the output-level. Our extensive experiments demonstrate that STXD significantly improves the NDS and mAP of the based student detectors by 2.8%~4.5% on the nuScenes testing dataset.

        ----

        ## [1276] Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks

        **Authors**: *Micah Goldblum, Hossein Souri, Renkun Ni, Manli Shu, Viraj Prabhu, Gowthami Somepalli, Prithvijit Chattopadhyay, Mark Ibrahim, Adrien Bardes, Judy Hoffman, Rama Chellappa, Andrew Gordon Wilson, Tom Goldstein*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d9571470bb750f0e2325a030016f63f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d9571470bb750f0e2325a030016f63f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Neural network based computer vision systems are typically built on a backbone, a pretrained or randomly initialized feature extractor.  Several years ago, the default option was an ImageNet-trained convolutional neural network.  However, the recent past has seen the emergence of countless backbones pretrained using various algorithms and datasets. While this abundance of choice has led to performance increases for a range of systems, it is difficult for practitioners to make informed decisions about which backbone to choose.  Battle of the Backbones (BoB) makes this choice easier by benchmarking a diverse suite of pretrained models, including vision-language models, those trained via self-supervised learning, and the Stable Diffusion backbone, across a diverse set of computer vision tasks ranging from classification to object detection to OOD generalization and more.  Furthermore, BoB sheds light on promising directions for the research community to advance computer vision by illuminating strengths and weakness of existing approaches through a comprehensive analysis conducted on more than 1500 training runs.  While vision transformers (ViTs) and self-supervised learning (SSL) are increasingly popular, we find that convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks among the models we consider. Moreover, in apples-to-apples comparisons on the same architectures and similarly sized pretraining datasets, we find that SSL backbones are highly competitive, indicating that future works should perform SSL pretraining with advanced architectures and larger pretraining datasets.  We release the raw results of our experiments along with code that allows researchers to put their own backbones through the gauntlet here: https://github.com/hsouri/Battle-of-the-Backbones.

        ----

        ## [1277] Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

        **Authors**: *Florian Seligmann, Philipp Becker, Michael Volpp, Gerhard Neumann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5d97b7e62022c859347397f6c1e8d0f9-Abstract-Conference.html)

        **Abstract**:

        Bayesian deep learning (BDL) is a promising approach to achieve well-calibrated predictions on distribution-shifted data. Nevertheless, there exists no large-scale survey that evaluates recent SOTA methods on diverse, realistic, and challenging benchmark tasks in a systematic manner. To provide a clear picture of the current state of BDL research, we evaluate modern BDL algorithms on real-world datasets from the WILDS collection containing challenging classification and regression tasks, with a focus on generalization capability and calibration under distribution shift. We compare the algorithms on a wide range of large, convolutional and transformer-based neural network architectures. In particular, we investigate a signed version of the expected calibration error that reveals whether the methods are over- or underconfident, providing further insight into the behavior of the methods. Further, we provide the first systematic evaluation of BDL for fine-tuning large pre-trained models, where training from scratch is prohibitively expensive. Finally, given the recent success of Deep Ensembles, we extend popular single-mode posterior approximations to multiple modes by the use of ensembles.   While we find that ensembling single-mode approximations generally improves the generalization capability and calibration of the models by a significant margin, we also identify a failure mode of ensembles when finetuning large transformer-based language models.  In this setting, variational inference based approaches such as last-layer Bayes By Backprop outperform other methods in terms of accuracy by a large margin, while modern approximate inference algorithms such as SWAG achieve the best calibration.

        ----

        ## [1278] (S)GD over Diagonal Linear Networks: Implicit bias, Large Stepsizes and Edge of Stability

        **Authors**: *Mathieu Even, Scott Pesme, Suriya Gunasekar, Nicolas Flammarion*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5da6ce80e97671b70c01a2e703b868b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5da6ce80e97671b70c01a2e703b868b3-Abstract-Conference.html)

        **Abstract**:

        In this paper, we investigate the impact of stochasticity and large stepsizes on the implicit regularisation of gradient descent (GD) and stochastic gradient descent (SGD) over $2$-layer diagonal linear networks. We prove the convergence of GD and SGD with macroscopic stepsizes in an overparametrised regression setting and characterise their solutions through an implicit regularisation problem. Our crisp characterisation leads to qualitative insights about the impact of stochasticity and stepsizes on the recovered solution. Specifically, we show that large stepsizes consistently benefit SGD for sparse regression problems, while they can hinder the recovery of sparse solutions for GD. These effects are magnified for stepsizes in a tight window just below the divergence threshold, in the ``edge of stability'' regime. Our findings are supported by experimental results.

        ----

        ## [1279] Optimal Preconditioning and Fisher Adaptive Langevin Sampling

        **Authors**: *Michalis K. Titsias*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5da6d5818a156791090c875abeca3cf8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5da6d5818a156791090c875abeca3cf8-Abstract-Conference.html)

        **Abstract**:

        We define  an optimal preconditioning for the Langevin diffusion by analytically optimizing the expected squared jumped distance. This yields as the optimal preconditioning an inverse Fisher information covariance matrix, where the covariance matrix is computed as the outer product of log target gradients averaged under the target.  We apply this result to the Metropolis adjusted Langevin algorithm (MALA)  and derive a computationally efficient adaptive MCMC scheme that learns the preconditioning from the history of gradients produced as the algorithm runs. We show in several experiments that the proposed algorithm is very robust in high dimensions and significantly outperforms other methods, including a closely related adaptive MALA scheme that learns the preconditioning with standard adaptive MCMC as well as the position-dependent  Riemannian manifold MALA sampler.

        ----

        ## [1280] AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking

        **Authors**: *Yang Yu, Qi Liu, Kai Zhang, Yuren Zhang, Chao Song, Min Hou, Yuqing Yuan, Zhihao Ye, Zaixi Zhang, Sanshi Lei Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e0da5da69b71349ae0bd7ad716e4bc9-Abstract-Conference.html)

        **Abstract**:

        User modeling, which aims to capture users' characteristics or interests, heavily relies on task-specific labeled data and suffers from the data sparsity issue. Several recent studies tackled this problem by pre-training the user model on massive user behavior sequences with a contrastive learning task. Generally, these methods assume different views of the same behavior sequence constructed via data augmentation are semantically consistent, i.e., reflecting similar characteristics or interests of the user, and thus maximizing their agreement in the feature space. However, due to the diverse interests and heavy noise in user behaviors, existing augmentation methods tend to lose certain characteristics of the user or introduce noisy behaviors. Thus, forcing the user model to directly maximize the similarity between the augmented views may result in a negative transfer. To this end, we propose to replace the contrastive learning task with a new pretext task: Augmentation-Adaptive SelfSupervised Ranking (AdaptSSR), which alleviates the requirement of semantic consistency between the augmented views while pre-training a discriminative user model. Specifically, we adopt a multiple pairwise ranking loss which trains the user model to capture the similarity orders between the implicitly augmented view, the explicitly augmented view, and views from other users. We further employ an in-batch hard negative sampling strategy to facilitate model training. Moreover, considering the distinct impacts of data augmentation on different behavior sequences, we design an augmentation-adaptive fusion mechanism to automatically adjust the similarity order constraint applied to each sample based on the estimated similarity between the augmented views. Extensive experiments on both public and industrial datasets with six downstream tasks verify the effectiveness of AdaptSSR.

        ----

        ## [1281] Anytime Model Selection in Linear Bandits

        **Authors**: *Parnian Kassraie, Nicolas Emmenegger, Andreas Krause, Aldo Pacchiano*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e11d23b18261d1b76d14da7a285fd0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e11d23b18261d1b76d14da7a285fd0c-Abstract-Conference.html)

        **Abstract**:

        Model selection in the context of bandit optimization is a challenging problem, as it requires balancing exploration and exploitation not only for action selection, but also for model selection. One natural approach is to rely on online learning algorithms that treat different models as experts. Existing methods, however, scale poorly ($\mathrm{poly}M$) with the number of models $M$ in terms of their regret.Our key insight is that, for model selection in linear bandits, we can emulate full-information feedback to the online learner with a favorable bias-variance trade-off. This allows us to develop ALEXP, which has an exponentially improved ($\log M$) dependence on $M$ for its regret.ALEXP has anytime guarantees on its regret, and neither requires knowledge of the horizon $n$, nor relies on an initial purely exploratory stage.Our approach utilizes a  novel time-uniform analysis of the Lasso, establishing a new connection between online learning and high-dimensional statistics.

        ----

        ## [1282] Towards Personalized Federated Learning via Heterogeneous Model Reassembly

        **Authors**: *Jiaqi Wang, Xingyi Yang, Suhan Cui, Liwei Che, Lingjuan Lyu, Dongkuan Xu, Fenglong Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e2217482fa75556f1970be809acd3f8-Abstract-Conference.html)

        **Abstract**:

        This paper focuses on addressing the practical yet challenging problem of model heterogeneity in federated learning, where clients possess models with different network structures. To track this problem, we propose a novel framework called pFedHR, which leverages heterogeneous model reassembly to achieve personalized federated learning. In particular, we approach the problem of heterogeneous model personalization as a model-matching optimization task on the server side. Moreover, pFedHR automatically and dynamically generates informative and diverse personalized candidates with minimal human intervention. Furthermore, our proposed heterogeneous model reassembly technique mitigates the adverse impact introduced by using public data with different distributions from the client data to a certain extent. Experimental results demonstrate that pFedHR outperforms baselines on three datasets under both IID and Non-IID settings. Additionally, pFedHR effectively reduces the adverse impact of using different public data and dynamically generates diverse personalized models in an automated manner.

        ----

        ## [1283] Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning

        **Authors**: *Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html)

        **Abstract**:

        Large language models have shown astonishing performance on a wide range of reasoning tasks. In this paper, we investigate whether they could reason about real-world events and help improve the prediction performance of event sequence models. We design LAMP, a framework that integrates a large language model in event prediction. Particularly, the language model performs abductive reasoning to assist an event sequence model: the event model proposes predictions on future events given the past; instructed by a few expert-annotated demonstrations, the language model learns to suggest possible causes for each proposal; a search module finds out the previous events that match the causes; a scoring function learns to examine whether the retrieved events could actually cause the proposal. Through extensive experiments on several challenging real-world datasets, we demonstrate that our framework---thanks to the reasoning capabilities of large language models---could significantly outperform the state-of-the-art event sequence models.

        ----

        ## [1284] Complexity Matters: Rethinking the Latent Space for Generative Modeling

        **Authors**: *Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html)

        **Abstract**:

        In generative modeling, numerous successful approaches leverage a low-dimensional latent space, e.g., Stable Diffusion models the latent space induced by an encoder and generates images through a paired decoder. Although the selection of the latent space is empirically pivotal, determining the optimal choice and the process of identifying it remain unclear. In this study, we aim to shed light on this under-explored topic by rethinking the latent space from the perspective of model complexity. Our investigation starts with the classic generative adversarial networks (GANs). Inspired by the GAN training objective, we propose a novel "distance" between the latent and data distributions, whose minimization coincides with that of the generator complexity. The minimizer of this distance is characterized as the optimal data-dependent latent that most effectively capitalizes on the generator's capacity. Then, we consider parameterizing such a latent distribution by an encoder network and propose a two-stage training strategy called Decoupled Autoencoder (DAE), where the encoder is only updated in the first stage with an auxiliary decoder and then frozen in the second stage while the actual decoder is being trained. DAE can improve the latent distribution and as a result, improve the generative performance. Our theoretical analyses are corroborated by comprehensive experiments on various models such as VQGAN and Diffusion Transformer, where our modifications yield significant improvements in sample quality with decreased model complexity.

        ----

        ## [1285] Riemannian stochastic optimization methods avoid strict saddle points

        **Authors**: *Ya-Ping Hsieh, Mohammad Reza Karimi Jaghargh, Andreas Krause, Panayotis Mertikopoulos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e809ba53f34d9170386ebfc8b60300f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e809ba53f34d9170386ebfc8b60300f-Abstract-Conference.html)

        **Abstract**:

        Many modern machine learning applications - from online principal component analysis to covariance matrix identification and dictionary learning - can be formulated as minimization problems on Riemannian manifolds, typically solved with a Riemannian stochastic gradient method (or some variant thereof). However, in many cases of interest, the resulting minimization problem is _not_ geodesically convex, so the convergence of the chosen solver to a desirable solution - i.e., a local minimizer - is by no means guaranteed. In this paper, we study precisely this question, that is, whether stochastic Riemannian optimization algorithms are guaranteed to avoid saddle points with probability $1$. For generality, we study a family of retraction-based methods which, in addition to having a potentially much lower per-iteration cost relative to Riemannian gradient descent, include other widely used algorithms, such as natural policy gradient methods and mirror descent in ordinary convex spaces. In this general setting, we show that, under mild assumptions for the ambient manifold and the oracle providing gradient information, the policies under study avoid strict saddle points / submanifolds with probability $1$, from any initial condition. This result provides an important sanity check for the use of gradient methods on manifolds as it shows that, almost always, the end state of a stochastic Riemannian algorithm can only be a local minimizer.

        ----

        ## [1286] Toward Better PAC-Bayes Bounds for Uniformly Stable Algorithms

        **Authors**: *Sijia Zhou, Yunwen Lei, Ata Kabán*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e8309c9ca683e11672e3dbcd4b87776-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e8309c9ca683e11672e3dbcd4b87776-Abstract-Conference.html)

        **Abstract**:

        We give sharper bounds for uniformly stable randomized algorithms in a PAC-Bayesian framework, which improve the existing results by up to a factor of $\sqrt{n}$ (ignoring a log factor), where $n$ is the sample size. The key idea is to bound the moment generating function of the generalization gap using concentration of weakly dependent random variables due to Bousquet et al (2020). We introduce an assumption of sub-exponential stability parameter, which allows a general treatment that we instantiate in two applications: stochastic gradient descent and randomized coordinate descent. Our results eliminate the requirement of strong convexity from previous results, and hold for non-smooth convex problems.

        ----

        ## [1287] Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models

        **Authors**: *Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, Rongrong Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html)

        **Abstract**:

        Recently,  growing interest has been aroused in extending the multimodal capability of large language models (LLMs), e.g., vision-language (VL) learning, which is regarded as the next milestone of artificial general intelligence. However, existing solutions are prohibitively expensive, which not only need to optimize excessive parameters, but also require another large-scale pre-training before VL instruction tuning. In this paper, we propose a novel and  affordable  solution for the effective VL adaption of LLMs, called  Mixture-of-Modality Adaptation (MMA).  Instead of using large neural networks to connect the image encoder and LLM, MMA adopts lightweight modules, i.e., adapters, to  bridge the gap between LLMs and VL tasks, which also enables the joint optimization of the image and language models. Meanwhile, MMA is also equipped with a routing algorithm to help LLMs achieve an  automatic  shift between single- and multi-modal instructions without compromising their ability of natural language understanding.  To validate MMA, we apply it to a recent LLM called LLaMA and term this formed large vision-language instructed model as LaVIN.  To validate MMA and LaVIN, we conduct extensive experiments  under two setups, namely  multimodal science question answering and multimodal dialogue. The experimental results not only demonstrate the competitive performance and the superior training efficiency  of LaVIN  than existing multimodal LLMs, but also confirm  its  great potential   as a general-purpose chatbot. More importantly, the actual expenditure of LaVIN is extremely cheap, e.g., only 1.4 training hours with 3.8M trainable parameters, greatly confirming the effectiveness of MMA.   Our  code is anonymously released at:  https://anonymous.4open.science/r/LaVIN--1067.

        ----

        ## [1288] GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

        **Authors**: *Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        With a long history of traditional Graph Anomaly Detection (GAD) algorithms and recently popular Graph Neural Networks (GNNs), it is still not clear (1) how they perform under a standard comprehensive setting, (2) whether GNNs can outperform traditional algorithms such as tree ensembles, and (3) how about their efficiency on large-scale graphs. In response, we introduce GADBench---a benchmark tool dedicated to supervised anomalous node detection in static graphs. GADBench facilitates a detailed comparison across 29 distinct models on ten real-world GAD datasets, encompassing thousands to millions (~6M) nodes. Our main finding is that tree ensembles with simple neighborhood aggregation can outperform the latest GNNs tailored for the GAD task. We shed light on the current progress of GAD, setting a robust groundwork for subsequent investigations in this domain. GADBench is open-sourced at https://github.com/squareRoot3/GADBench.

        ----

        ## [1289] Brain encoding models based on multimodal transformers can transfer across language and vision

        **Authors**: *Jerry Tang, Meng Du, Vy A. Vo, Vasudev Lal, Alexander Huth*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html)

        **Abstract**:

        Encoding models have been used to assess how the human brain represents concepts in language and vision. While language and vision rely on similar concept representations, current encoding models are typically trained and tested on brain responses to each modality in isolation. Recent advances in multimodal pretraining have produced transformers that can extract aligned representations of concepts in language and vision. In this work, we used representations from multimodal transformers to train encoding models that can transfer across fMRI responses to stories and movies. We found that encoding models trained on brain responses to one modality can successfully predict brain responses to the other modality, particularly in cortical regions that represent conceptual meaning. Further analysis of these encoding models revealed shared semantic dimensions that underlie concept representations in language and vision. Comparing encoding models trained using representations from multimodal and unimodal transformers, we found that multimodal transformers learn more aligned representations of concepts in language and vision. Our results demonstrate how multimodal transformers can provide insights into the brainâ€™s capacity for multimodal processing.

        ----

        ## [1290] PointGPT: Auto-regressively Generative Pre-training from Point Clouds

        **Authors**: *Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) based on the generative pre-training transformer (GPT) have demonstrated remarkable effectiveness across a diverse range of downstream tasks. Inspired by the advancements of the GPT, we present PointGPT, a novel approach that extends the concept of GPT to point clouds, addressing the challenges associated with disorder properties, low information density, and task gaps. Specifically, a point cloud auto-regressive generation task is proposed to pre-train transformer models. Our method partitions the input point cloud into multiple point patches and arranges them in an ordered sequence based on their spatial proximity. Then, an extractor-generator based transformer decode, with a dual masking strategy, learns latent representations conditioned on the preceding point patches, aiming to predict the next one in an auto-regressive manner. To explore scalability and enhance performance, a larger pre-training dataset is collected. Additionally, a subsequent post-pre-training stage is introduced, incorporating a labeled hybrid dataset. Our scalable approach allows for learning high-capacity models that generalize well, achieving state-of-the-art performance on various downstream tasks. In particular, our approach achieves classification accuracies of 94.9% on the ModelNet40 dataset and 93.4% on the ScanObjectNN dataset, outperforming all other transformer models. Furthermore, our method also attains new state-of-the-art accuracies on all four few-shot learning benchmarks. Codes are available at https://github.com/CGuangyan-BIT/PointGPT.

        ----

        ## [1291] Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning

        **Authors**: *Xiaoqian Wu, Yonglu Li, Jianhua Sun, Cewu Lu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)

        **Abstract**:

        Human reasoning can be understood as a cooperation between the intuitive, associative "System-1'' and the deliberative, logical "System-2''. For existing System-1-like methods in visual activity understanding, it is crucial to integrate System-2 processing to improve explainability, generalization, and data efficiency. One possible path of activity reasoning is building a symbolic system composed of symbols and rules, where one rule connects multiple symbols, implying human knowledge and reasoning abilities.Previous methods have made progress, but are defective with limited symbols from handcraft and limited rules from visual-based annotations, failing to cover the complex patterns of activities and lacking compositional generalization. To overcome the defects, we propose a new symbolic system with two ideal important properties: broad-coverage symbols and rational rules. Collecting massive human knowledge via manual annotations is expensive to instantiate this symbolic system. Instead, we leverage the recent advancement of LLMs (Large Language Models) as an approximation of the two ideal properties, i.e., Symbols from Large Language Models (Symbol-LLM). Then, given an image, visual contents from the images are extracted andchecked as symbols and activity semantics are reasoned out based on rules via fuzzy logic calculation.Our method shows superiority in extensive activity understanding tasks. Code and data are available at https://mvig-rhos.com/symbol_llm.

        ----

        ## [1292] Non-Convex Bilevel Optimization with Time-Varying Objective Functions

        **Authors**: *Sen Lin, Daouda Sow, Kaiyi Ji, Yingbin Liang, Ness B. Shroff*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ee60ca5686bbcf756e56a6c75e66f32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ee60ca5686bbcf756e56a6c75e66f32-Abstract-Conference.html)

        **Abstract**:

        Bilevel optimization has become a powerful tool in a wide variety of machine learning problems. However, the current nonconvex bilevel optimization considers an offline dataset and static functions, which may not work well in emerging online applications with streaming data and time-varying functions. In this work, we study online bilevel optimization (OBO) where the functions can be time-varying and the agent continuously updates the decisions with online streaming data. To deal with the function variations and the unavailability of the true hypergradients in OBO, we propose a single-loop online bilevel optimizer with window averaging (SOBOW), which updates the outer-level decision based on a window average of the most recent hypergradient estimations stored in the memory. Compared to existing algorithms, SOBOW is computationally efficient and does not need to know previous functions. To handle the unique technical difficulties rooted in single-loop update and function variations for OBO, we develop a novel analytical technique that disentangles the complex couplings between decision variables, and carefully controls the hypergradient estimation error. We show that SOBOW can achieve a sublinear bilevel local regret under mild conditions. Extensive experiments across multiple domains corroborate the effectiveness of SOBOW.

        ----

        ## [1293] Online Pricing for Multi-User Multi-Item Markets

        **Authors**: *Yigit Efe Erginbas, Thomas A. Courtade, Kannan Ramchandran, Soham Phade*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5eee634cb9729b8bcc2ec9f2a46a74ae-Abstract-Conference.html)

        **Abstract**:

        Online pricing has been the focus of extensive research in recent years, particularly in the context of selling an item to sequentially arriving users. However, what if a provider wants to maximize revenue by selling multiple items to multiple users in each round? This presents a complex problem, as the provider must intelligently offer the items to those users who value them the most without exceeding their highest acceptable prices. In this study, we tackle this challenge by designing online algorithms that can efficiently offer and price items while learning user valuations from accept/reject feedback. We focus on three user valuation models (fixed valuations, random experiences, and random valuations) and provide algorithms with nearly-optimal revenue regret guarantees. In particular, for any market setting with $N$ users, $M$ items, and load $L$ (which roughly corresponds to the maximum number of simultaneous allocations possible), our algorithms achieve regret of order $O(NM\log\log(LT))$ under fixed valuations model, $\widetilde{O}(\sqrt{NMLT})$ under random experiences model and $\widetilde{O}(\sqrt{NMLT})$ under random valuations model in $T$ rounds.

        ----

        ## [1294] Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost

        **Authors**: *Yu-Jie Zhang, Masashi Sugiyama*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html)

        **Abstract**:

        This paper investigates the logistic bandit problem, a variant of the generalized linear bandit model that utilizes a logistic model to depict the feedback from an action. While most existing research focuses on the binary logistic bandit problem, the multinomial case, which considers more than two possible feedback values, offers increased practical relevance and adaptability for use in complex decision-making problems such as reinforcement learning. In this paper, we provide an algorithm that enjoys both statistical and computational efficiency for the logistic bandit problem. In the binary case, our method improves the state-of-the-art binary logistic bandit method by reducing the per-round computation cost from $\mathcal{O}(\log T)$ to $\mathcal{O}(1)$ with respect to the time horizon $T$, while still preserving the minimax optimal guarantee up to logarithmic factors. In the multinomial case, with $K+1$ potential feedback values, our algorithm achieves an $\tilde{\mathcal{O}}(K\sqrt{T})$ regret bound with $\mathcal{O}(1)$ computational cost per round. The result not only improves the $\tilde{\mathcal{O}}(K\sqrt{\kappa T})$ bound for the best-known tractable algorithm—where the large constant $\kappa$ increases exponentially with the diameter of the parameter domain—but also reduces the $\mathcal{O}(T)$ computational complexity demanded by the previous method.

        ----

        ## [1295] Transfer learning for atomistic simulations using GNNs and kernel mean embeddings

        **Authors**: *John Isak Texas Falk, Luigi Bonati, Pietro Novelli, Michele Parrinello, Massimiliano Pontil*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f02c76bc411a6f7c9a8bb2cbf981260-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f02c76bc411a6f7c9a8bb2cbf981260-Abstract-Conference.html)

        **Abstract**:

        Interatomic potentials learned using machine learning methods have been successfully applied to atomistic simulations. However, accurate models require large training datasets, while generating reference calculations is computationally demanding. To bypass this difficulty, we propose a transfer learning algorithm that leverages the ability of graph neural networks (GNNs) to represent chemical environments together with kernel mean embeddings. We extract a feature map from GNNs pre-trained on the OC20 dataset and use it to learn the potential energy surface from system-specific datasets of catalytic processes. Our method is further enhanced by incorporating into the kernel the chemical species information, resulting in improved performance and interpretability. We test our approach on a series of realistic datasets of increasing complexity, showing excellent generalization and transferability performance, and improving on methods that rely on GNNs or ridge regression alone, as well as similar fine-tuning approaches.

        ----

        ## [1296] StressID: a Multimodal Dataset for Stress Identification

        **Authors**: *Hava Chaptoukaev, Valeriya Strizhkova, Michele Panariello, Bianca Dalpaos, Aglind Reka, Valeria Manera, Susanne Thümmler, Esma Ismailova, Nicholas W. D. Evans, François Brémond, Massimiliano Todisco, Maria A. Zuluaga, Laura M. Ferrari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f09bfe6730e9627a9f800d01a8ad5cd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f09bfe6730e9627a9f800d01a8ad5cd-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        StressID is a new dataset specifically designed for stress identification fromunimodal and multimodal data. It contains videos of facial expressions, audiorecordings, and physiological signals. The video and audio recordings are acquiredusing an RGB camera with an integrated microphone. The physiological datais composed of electrocardiography (ECG), electrodermal activity (EDA), andrespiration signals that are recorded and monitored using a wearable device. Thisexperimental setup ensures a synchronized and high-quality multimodal data col-lection. Different stress-inducing stimuli, such as emotional video clips, cognitivetasks including mathematical or comprehension exercises, and public speakingscenarios, are designed to trigger a diverse range of emotional responses. Thefinal dataset consists of recordings from 65 participants who performed 11 tasks,as well as their ratings of perceived relaxation, stress, arousal, and valence levels.StressID is one of the largest datasets for stress identification that features threedifferent sources of data and varied classes of stimuli, representing more than39 hours of annotated data in total. StressID offers baseline models for stressclassification including a cleaning, feature extraction, and classification phase foreach modality. Additionally, we provide multimodal predictive models combiningvideo, audio, and physiological inputs. The data and the code for the baselines areavailable at https://project.inria.fr/stressid/.

        ----

        ## [1297] Statistical Knowledge Assessment for Large Language Models

        **Authors**: *Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f0a4cd23e1c6eedd3edebba674ab877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f0a4cd23e1c6eedd3edebba674ab877-Abstract-Conference.html)

        **Abstract**:

        Given varying prompts regarding a factoid question, can a large language model (LLM) reliably generate factually correct answers? Existing LLMs may generate distinct responses for different prompts. In this paper, we study the problem of quantifying knowledge contained in an LLM regarding a given set of facts. We propose KaRR, a statistical approach to assess factual knowledge for LLMs. The main idea is to estimate the ratio of LLM generating text corresponding to the answer entity given diverse prompts of the subject and the querying relation, versus it generating by random chances. Our assessment suite contains a comprehensive set of 994,123 entities and 600 relations, with 1,395,905 text aliases. We use our method to evaluate 20 LLMs of various sizes, including LLaMA, Alpaca, OPT, etc. Experiments show that our results have a strong correlation (0.43 Kendall's $\tau$) with the results of human assessment on LLMs. Our results reveal that the knowledge in LLMs with the same backbone architecture adheres to the scaling law, while tuning on instruction-following data sometimes compromises the model's capability to generate factually correct text reliably.

        ----

        ## [1298] Color Equivariant Convolutional Networks

        **Authors**: *Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, Jan van Gemert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f173562e7662b14fb5c5695f225ea46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f173562e7662b14fb5c5695f225ea46-Abstract-Conference.html)

        **Abstract**:

        Color is a crucial visual cue readily exploited by Convolutional Neural Networks (CNNs) for object recognition. However, CNNs struggle if there is data imbalance between color variations introduced by accidental recording conditions. Color invariance addresses this issue but does so at the cost of removing all color information, which sacrifices discriminative power. In this paper, we propose Color Equivariant Convolutions (CEConvs), a novel deep learning building block that enables shape feature sharing across the color spectrum while retaining important color information. We extend the notion of equivariance from geometric to photometric transformations by incorporating parameter sharing over hue-shifts in a neural network. We demonstrate the benefits of CEConvs in terms of downstream performance to various tasks and improved robustness to color changes, including train-test distribution shifts. Our approach can be seamlessly integrated into existing architectures, such as ResNets, and offers a promising solution for addressing color-based domain shifts in CNNs.

        ----

        ## [1299] Realistic Synthetic Financial Transactions for Anti-Money Laundering Models

        **Authors**: *Erik Altman, Jovan Blanusa, Luc von Niederhäusern, Beni Egressy, Andreea Anghel, Kubilay Atasu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        With the widespread digitization of finance and the increasing popularity of cryptocurrencies, the sophistication of fraud schemes devised by cybercriminals is growing. Money laundering -- the movement of illicit funds to conceal their origins -- can cross bank and national boundaries, producing complex transaction patterns. The UN estimates 2-5\% of global GDP or \$0.8 - \$2.0 trillion dollars are laundered globally each year.  Unfortunately, real data to train machine learning models to detect laundering is generally not available, and previous synthetic data generators have had significant shortcomings. A realistic, standardized, publicly-available benchmark is needed for comparing models and for the advancement of the area.To this end, this paper contributes a synthetic financial transaction dataset generator and a set of synthetically generated AML (Anti-Money Laundering) datasets.  We have calibrated this agent-based generator to match real transactions as closely as possible and made the datasets public. We describe the generator in detail and demonstrate how the datasets generated can help compare different machine learning models in terms of their AML abilities. In a key way, using synthetic data in these comparisons can be even better than using real data: the ground truth labels are complete, whilst many laundering transactions in real data are never detected.

        ----

        ## [1300] DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics

        **Authors**: *Zhiao Huang, Feng Chen, Yewen Pu, Chunru Lin, Hao Su, Chuang Gan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html)

        **Abstract**:

        Combining gradient-based trajectory optimization with differentiable physics simulation is an efficient technique for solving soft-body manipulation problems.Using a well-crafted optimization objective, the solver can quickly converge onto a valid trajectory.However, writing the appropriate objective functions requires expert knowledge, making it difficult to collect a large set of naturalistic problems from non-expert users.We introduce DiffVL, a method that enables non-expert users to communicate soft-body manipulation tasks -- a combination of vision and natural language, given in multiple stages -- that can be readily leveraged by a differential physics solver. We have developed GUI tools that enable non-expert users to specify 100 tasks inspired by real-life soft-body manipulations from online videos, which we'll make public.We leverage large language models to translate task descriptions into machine-interpretable optimization objectives. The optimization objectives can help differentiable physics solvers to solve these long-horizon multistage tasks that are challenging for previous baselines.

        ----

        ## [1301] Label-efficient Segmentation via Affinity Propagation

        **Authors**: *Wentong Li, Yuqian Yuan, Song Wang, Wenyu Liu, Dongqi Tang, Jian Liu, Jianke Zhu, Lei Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f6fae52f3b62c3334e288e3bc58230d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f6fae52f3b62c3334e288e3bc58230d-Abstract-Conference.html)

        **Abstract**:

        Weakly-supervised segmentation with label-efficient sparse annotations has attracted increasing research attention to reduce the cost of laborious pixel-wise labeling process, while the pairwise affinity modeling techniques play an essential role in this task. Most of the existing approaches focus on using the local appearance kernel to model the neighboring pairwise potentials. However, such a local operation fails to capture the long-range dependencies and ignores the  topology of objects. In this work, we formulate the affinity modeling as an affinity propagation process, and propose a local and a global pairwise affinity terms to generate accurate soft pseudo labels. An efficient algorithm is also developed to reduce significantly the computational cost. The proposed approach can be conveniently plugged into existing segmentation networks. Experiments on three typical label-efficient segmentation tasks, i.e. box-supervised instance segmentation, point/scribble-supervised semantic segmentation and CLIP-guided semantic segmentation, demonstrate the superior performance of the proposed approach.

        ----

        ## [1302] Segment Anything in High Quality

        **Authors**: *Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html)

        **Abstract**:

        The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures.  We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced detaset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 10 diverse segmentation datasets across different downstream tasks, where 8 out of them are evaluated in a zero-shot transfer protocol. Our code and pretrained models are at https://github.com/SysCV/SAM-HQ.

        ----

        ## [1303] Variational Inference with Gaussian Score Matching

        **Authors**: *Chirag Modi, Robert M. Gower, Charles Margossian, Yuling Yao, David M. Blei, Lawrence K. Saul*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f9453c4848b89d4d8c5d6041f5fb9ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f9453c4848b89d4d8c5d6041f5fb9ec-Abstract-Conference.html)

        **Abstract**:

        Variational inference (VI) is a method to approximate the computationally intractable posterior distributions that arise in  Bayesian statistics. Typically, VI fits a simple parametric distribution to be close to the target posterior, optimizing an appropriate objective such as the evidence lower bound (ELBO).   In this work, we present a new approach to VI. Our method is based on the principle of score matching---namely, that if two distributions are equal then their score functions (i.e., gradients of the log density) are equal at every point on their support. With this principle, we develop score-matching VI, an iterative algorithm that seeks to match the scores between the variational approximation and the exact posterior. At each iteration, score-matching VI solves an inner optimization, one that minimally adjusts the current variational estimate to match the scores at a newly sampled value of the latent variables. We show that when the variational family is a  Gaussian, this inner optimization enjoys a closed-form solution, which we call Gaussian score matching VI (GSM-VI). GSM-VI is a ``black box'' variational algorithm in that it only requires a differentiable joint distribution, and as such it can be applied to a wide class of models. We compare GSM-VI to black box variational inference (BBVI), which has similar requirements but instead optimizes the ELBO. We first study how GSM-VI behaves as a function of the problem dimensionality, the condition number of the target covariance matrix (when the target is Gaussian), and the degree of mismatch between the approximating and exact posterior distribution. We then study GSM-VI on a collection of real-world Bayesian inference problems from the posteriorDB database of datasets and models. We find that GSM-VI is faster than BBVI and equally or more accurate. Specifically, over a wide range of target posteriors, GSM-VI requires 10-100x fewer gradient evaluations than BBVI to obtain a comparable quality of approximation.

        ----

        ## [1304] Feature Adaptation for Sparse Linear Regression

        **Authors**: *Jonathan A. Kelner, Frederic Koehler, Raghu Meka, Dhruv Rohatgi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f999632c48f87cffb214e575581e4a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f999632c48f87cffb214e575581e4a9-Abstract-Conference.html)

        **Abstract**:

        Sparse linear regression is a central problem in high-dimensional statistics. We study the correlated random design setting, where the covariates are drawn from a multivariate Gaussian $N(0,\Sigma)$, and we seek an estimator with small excess risk. If the true signal is $t$-sparse, information-theoretically, it is possible to achieve strong recovery guarantees with only $O(t\log n)$ samples. However, computationally efficient algorithms have sample complexity linear in (some variant of) the *condition number* of $\Sigma$. Classical algorithms such as the Lasso can require significantly more samples than necessary even if there is only a single sparse approximate dependency among the covariates.We provide a polynomial-time algorithm that, given $\Sigma$, automatically adapts the Lasso to tolerate a small number of approximate dependencies. In particular, we achieve near-optimal sample complexity for constant sparsity and if $\Sigma$ has few ``outlier'' eigenvalues.Our algorithm fits into a broader framework of *feature adaptation* for sparse linear regression with ill-conditioned covariates. With this framework, we additionally provide the first polynomial-factor improvement over brute-force search for constant sparsity $t$ and arbitrary covariance $\Sigma$.

        ----

        ## [1305] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling

        **Authors**: *Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html)

        **Abstract**:

        Time series analysis is widely used in extensive areas. Recently, to reduce labeling expenses and benefit various tasks, self-supervised pre-training has attracted immense interest. One mainstream paradigm is masked modeling, which successfully pre-trains deep models by learning to reconstruct the masked content based on the unmasked part. However, since the semantic information of time series is mainly contained in temporal variations, the standard way of randomly masking a portion of time points will seriously ruin vital temporal variations of time series, making the reconstruction task too difficult to guide representation learning. We thus present SimMTM, a Simple pre-training framework for Masked Time-series Modeling. By relating masked modeling to manifold learning, SimMTM proposes to recover masked time points by the weighted aggregation of multiple neighbors outside the manifold, which eases the reconstruction task by assembling ruined but complementary temporal variations from multiple masked series. SimMTM further learns to uncover the local structure of the manifold, which is helpful for masked modeling. Experimentally, SimMTM achieves state-of-the-art fine-tuning performance compared to the most advanced time series pre-training methods in two canonical time series analysis tasks: forecasting and classification, covering both in- and cross-domain settings.

        ----

        ## [1306] CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs

        **Authors**: *Guangyao Zhai, Evin Pinar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fba70900a84a8fb755c48ba99420c95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fba70900a84a8fb755c48ba99420c95-Abstract-Conference.html)

        **Abstract**:

        Controllable scene synthesis aims to create interactive environments for numerous industrial use cases. Scene graphs provide a highly suitable interface to facilitate these applications by abstracting the scene context in a compact manner. Existing methods, reliant on retrieval from extensive databases or pre-trained shape embeddings, often overlook scene-object and object-object relationships, leading to inconsistent results due to their limited generation capacity. To address this issue, we present CommonScenes, a fully generative model that converts scene graphs into corresponding controllable 3D scenes, which are semantically realistic and conform to commonsense. Our pipeline consists of two branches, one predicting the overall scene layout via a variational auto-encoder and the other generating compatible shapes via latent diffusion, capturing global scene-object and local inter-object relationships in the scene graph while preserving shape diversity. The generated scenes can be manipulated by editing the input scene graph and sampling the noise in the diffusion model. Due to the lack of a scene graph dataset offering high-quality object-level meshes with relations, we also construct SG-FRONT, enriching the off-the-shelf indoor dataset 3D-FRONT with additional scene graph labels. Extensive experiments are conducted on SG-FRONT, where CommonScenes shows clear advantages over other methods regarding generation consistency, quality, and diversity. Codes and the dataset are available on the website.

        ----

        ## [1307] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback

        **Authors**: *Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fc47800ee5b30b8777fdd30abcaaf3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fc47800ee5b30b8777fdd30abcaaf3b-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) such as ChatGPT have seen widespread adoption due to their ability to follow user instructions well.Developing these LLMs involves a complex yet poorly understood workflow requiring training with human feedback. Replicating and understanding this instruction-following process faces three major challenges: the high cost of data collection, the lack of trustworthy evaluation, and the absence of reference method implementations. We address these bottlenecks with AlpacaFarm, a simulator that enables research and development for learning from feedback at a low cost. First, we design LLM based simulator for human feedback that is 45x cheaper than crowdworkers and displays high agreement with humans. Second, we identify an evaluation dataset representative of real-world instructions and propose an automatic evaluation procedure. Third, we contribute reference implementations for several methods (PPO, best-of-n, expert iteration, among others) that learn from pairwise feedback. Finally, as an end-to-end validation of AlpacaFarm, we train and evaluate eleven models on 10k pairs of human feedback and show that rankings of models trained in AlpacaFarm match rankings of models trained on human data. As a demonstration of the research possible in AlpacaFarm, we find that methods that use a reward model can substantially improve over supervised fine-tuning and that our reference PPO implementation leads to a +10% win-rate improvement against Davinci003.

        ----

        ## [1308] Beta Diffusion

        **Authors**: *Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fe1b43c882d746c187456eb4c8cdf52-Abstract-Conference.html)

        **Abstract**:

        We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, further supports the efficacy of KLUBs for optimization. Experimental results on both synthetic data and natural images demonstrate the unique capabilities of beta diffusion in generative modeling of range-bounded data and validate the effectiveness of KLUBs in optimizing diffusion models, thereby making them valuable additions to the family of diffusion-based generative models and the optimization techniques used to train them.

        ----

        ## [1309] Minimax Optimal Rate for Parameter Estimation in Multivariate Deviated Models

        **Authors**: *Dat Do, Huy Nguyen, Khai Nguyen, Nhat Ho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5fed79713d97df88f9912c8d886fccb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5fed79713d97df88f9912c8d886fccb3-Abstract-Conference.html)

        **Abstract**:

        We study the maximum likelihood estimation (MLE) in the multivariate deviated model where the data are generated from the density function $(1-\lambda^{\ast})h_{0}(x)+\lambda^{\ast}f(x|\mu^{\ast}, \Sigma^{\ast})$ in which $h_{0}$ is a known function, $\lambda^{\ast} \in [0,1]$ and $(\mu^{\ast}, \Sigma^{\ast})$ are unknown parameters to estimate. The main challenges in deriving the convergence rate of the MLE mainly come from two issues: (1) The interaction between the function $h_{0}$ and the density function $f$; (2) The deviated proportion $\lambda^{\ast}$ can go to the extreme points of $[0,1]$ as the sample size tends to infinity. To address these challenges, we develop the \emph{distinguishability condition} to capture the linear independent relation between the function $h_{0}$ and the density function $f$. We then provide comprehensive convergence rates of the MLE via the vanishing rate of $\lambda^{\ast}$ to zero as well as the distinguishability of two functions $h_{0}$ and $f$.

        ----

        ## [1310] Partial Matrix Completion

        **Authors**: *Elad Hazan, Adam Tauman Kalai, Varun Kanade, Clara Mohri, Y. Jennifer Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5ff7b1f30e0caf3cc0b2fbfd4d7ebdd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5ff7b1f30e0caf3cc0b2fbfd4d7ebdd4-Abstract-Conference.html)

        **Abstract**:

        The matrix completion problem involves reconstructing a low-rank matrix by using a given set of revealed (and potentially noisy) entries. Although existing methods address the completion of the entire matrix, the accuracy of the completed entries can vary significantly across the matrix, due to differences in the sampling distribution. For instance, users may rate movies primarily from their country or favorite genres, leading to inaccurate predictions for the majority of completed entries.We propose a novel formulation of the problem as Partial Matrix Completion, where the objective is to complete a substantial subset of the entries with high confidence. Our algorithm efficiently handles the unknown and arbitrarily complex nature of the sampling distribution, ensuring high accuracy for all completed entries and sufficient coverage across the matrix. Additionally, we introduce an online version of the problem and present a low-regret efficient algorithm based on iterative gradient updates. Finally, we conduct a preliminary empirical evaluation of our methods.

        ----

        ## [1311] BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

        **Authors**: *Dongxu Li, Junnan Li, Steven C. H. Hoi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/602e1a5de9c47df34cae39353a7f5bb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/602e1a5de9c47df34cae39353a7f5bb1-Abstract-Conference.html)

        **Abstract**:

        Subject-driven text-to-image generation models create novel renditions of an input subject based on text prompts. Existing models suffer from lengthy fine-tuning and difficulties preserving the subject fidelity. To overcome these limitations, we introduce BLIP-Diffusion, a new subject-driven image generation model that supports multimodal control which consumes inputs of subject images and text prompts. Unlike other subject-driven generation models, BLIP-Diffusion introduces a new multimodal encoder which is pre-trained to provide subject representation. We first pre-train the multimodal encoder following BLIP-2 to produce visual representation aligned with the text.Then we design a subject representation learning task which enables a diffusion model to leverage such visual representation and generates new subject renditions. Compared with previous methods such as DreamBooth, our model enables zero-shot subject-driven generation, and efficient fine-tuning for customized subject with up to 20x speedup. We also demonstrate that BLIP-Diffusion can be flexibly combined with existing techniques such as ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications. Implementations are available at: https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion.

        ----

        ## [1312] Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data

        **Authors**: *Yiwen Kou, Zixiang Chen, Quanquan Gu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/602f5c1b803c53b2aaf0b3864bf3383a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/602f5c1b803c53b2aaf0b3864bf3383a-Abstract-Conference.html)

        **Abstract**:

        The implicit bias towards solutions with favorable properties is believed to be a key reason why neural networks trained by gradient-based optimization can generalize well. While the implicit bias of gradient flow has been widely studied for homogeneous neural networks (including ReLU and leaky ReLU networks), the implicit bias of gradient descent is currently only understood for smooth neural networks. Therefore, implicit bias in non-smooth neural networks trained by gradient descent remains an open question. In this paper, we aim to answer this question by studying the implicit bias of gradient descent for training two-layer fully connected (leaky) ReLU neural networks. We showed that when the training data are nearly-orthogonal, for leaky ReLU activation function, gradient descent will find a network with a stable rank that converges to $1$, whereas for ReLU activation function, gradient descent will find a neural network with a stable rank that is upper bounded by a constant. Additionally, we show that gradient descent will find a neural network such that all the training data points have the same normalized margin asymptotically. Experiments on both synthetic and real data backup our theoretical findings.

        ----

        ## [1313] SpecTr: Fast Speculative Decoding via Optimal Transport

        **Authors**: *Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix X. Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html)

        **Abstract**:

        Autoregressive sampling from large language models has led to state-of-the-art results in several natural language tasks.However, autoregressive sampling generates tokens one at a time making it slow, and even prohibitive in certain tasks. One way to speed up sampling is *speculative decoding*: use a small model to sample a *draft* (block or sequence of tokens), and then score all tokens in the draft by the large language model in parallel. A subset of the tokens in the draft are accepted (and the rest rejected) based on a statistical method to guarantee that the final output follows the distribution of the large model. In this work, we provide a principled understanding of speculative decoding through the lens of optimal transport (OT) with *membership cost*. This framework can be viewed as an extension of the well-known *maximal-coupling* problem. This new formulation enables us to generalize the speculative decoding method to allow for a set of $k$ candidates at the token-level, which leads to an improved optimal membership cost. We show that the optimal draft selection algorithm (transport plan) can be computed via linear programming, whose best-known runtime is exponential in $k$. We then propose a valid draft selection algorithm whose acceptance probability is $(1-1/e)$-optimal multiplicatively. Moreover, it can be computed in time almost linear with size of domain of a single token.Using this new draft selection algorithm, we develop a new autoregressive sampling algorithm called *SpecTr*, which provides speedup in decoding while ensuring that there is no quality degradation in the decoded output.We experimentally demonstrate that for state-of-the-art large language models, the proposed approach achieves a wall clock speedup of 2.13X, a further 1.37X speedup over speculative decoding on standard benchmarks.

        ----

        ## [1314] LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing

        **Authors**: *Su Zheng, Haoyu Yang, Binwu Zhu, Bei Yu, Martin D. F. Wong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/604b9fa9e1c16284e6517d923cf9ff20-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/604b9fa9e1c16284e6517d923cf9ff20-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Computational lithography provides algorithmic and mathematical support for resolution enhancement in optical lithography, which is the critical step in semiconductor manufacturing. The time-consuming lithography simulation and mask optimization processes limit the practical application of inverse lithography technology (ILT), a promising solution to the challenges of advanced-node lithography. Although various machine learning methods for ILT have shown promise for reducing the computational burden, this field is in lack of a dataset that can train the models thoroughly and evaluate the performance comprehensively. To boost the development of AI-driven computational lithography, we present the LithoBench dataset, a collection of circuit layout tiles for deep-learning-based lithography simulation and mask optimization. LithoBench consists of more than 120k tiles that are cropped from real circuit designs or synthesized according to the layout topologies of famous ILT testcases. The ground truths are generated by a famous lithography model in academia and an advanced ILT method. Based on the data, we provide a framework to design and evaluate deep neural networks (DNNs) with the data. The framework is used to benchmark state-of-the-art models on lithography simulation and mask optimization. We hope LithoBench can promote the research and development of computational lithography. LithoBench is available at https://anonymous.4open.science/r/lithobench-APPL.

        ----

        ## [1315] Proportional Response: Contextual Bandits for Simple and Cumulative Regret Minimization

        **Authors**: *Sanath Kumar Krishnamurthy, Ruohan Zhan, Susan Athey, Emma Brunskill*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6058d0c628a03fd95dfe5c72cbdf9e64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6058d0c628a03fd95dfe5c72cbdf9e64-Abstract-Conference.html)

        **Abstract**:

        In many applications, e.g. in healthcare and e-commerce, the goal of a contextual bandit may be to learn an optimal treatment assignment policy at the end of the experiment. That is, to minimize simple regret. However, this objective remains understudied. We propose a new family of computationally efficient bandit algorithms for the stochastic contextual bandit setting, where a tuning parameter determines the weight placed on cumulative regret minimization (where we establish near-optimal minimax guarantees) versus simple regret minimization (where we establish state-of-the-art guarantees). Our algorithms work with any function class, are robust to model misspecification, and can be used in continuous arm settings. This flexibility comes from constructing and relying on â€œconformal arm sets" (CASs). CASs provide a set of arms for every context, encompassing the context-specific optimal arm with a certain probability across the context distribution. Our positive results on simple and cumulative regret guarantees are contrasted with a negative result, which shows that no algorithm can achieve instance-dependent simple regret guarantees while simultaneously achieving minimax optimal cumulative regret guarantees.

        ----

        ## [1316] Higher-Order Uncoupled Dynamics Do Not Lead to Nash Equilibrium - Except When They Do

        **Authors**: *Sarah Toonsi, Jeff S. Shamma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/605e02ae04cba1ebf6a08206299e76b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/605e02ae04cba1ebf6a08206299e76b9-Abstract-Conference.html)

        **Abstract**:

        The framework of multi-agent learning explores the dynamics of how an agent's strategies evolve in response to the evolving strategies of other agents. Of particular interest is whether or not agent strategies converge to well known solution concepts such as Nash Equilibrium (NE). In "higher order'' learning, agent dynamics include auxiliary states that can capture phenomena such as path dependencies. We introduce higher-order gradient play dynamics that resemble projected gradient ascent with auxiliary states. The dynamics are "payoff based'' and "uncoupled'' in that each agent's dynamics depend on its own evolving payoff and has no explicit dependence on the utilities of other agents. We first show that for any specific game with an isolated completely mixed-strategy NE, there exist higher-order gradient play dynamics that lead (locally) to that NE, both for the specific game and nearby games with perturbed utility functions. Conversely, we show that for any higher-order gradient play dynamics, there exists a game with a unique isolated completely mixed-strategy NE for which the dynamics do not lead to NE. Finally, we show that convergence to the mixed-strategy equilibrium in coordination games, comes at the expense of the dynamics being inherently internally unstable.

        ----

        ## [1317] Subject-driven Text-to-Image Generation via Apprenticeship Learning

        **Authors**: *Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, William W. Cohen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6091bf1542b118287db4088bc16be8d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6091bf1542b118287db4088bc16be8d9-Abstract-Conference.html)

        **Abstract**:

        Recent text-to-image generation models like DreamBooth have made remarkable progress in generating highly customized images of a target subject, by fine-tuning an ``expert model'' for a given subject from a few examples.However, this process is expensive, since a new expert model must be learned for each subject. In this paper, we present SuTI, a Subject-driven Text-to-Image generator that replaces subject-specific fine tuning with {in-context} learning.Given a few demonstrations of a new subject, SuTI can instantly generate novel renditions of the subject in different scenes, without any subject-specific optimization.SuTI is powered by {apprenticeship learning}, where a single apprentice model is learned from data generated by a massive number of subject-specific expert models. Specifically, we mine millions of image clusters from the Internet, each centered around a specific visual subject. We adopt these clusters to train a massive number of expert models, each specializing in a different subject. The apprentice model SuTI then learns to imitate the behavior of these fine-tuned experts. SuTI can generate high-quality and customized subject-specific images 20x faster than optimization-based SoTA methods. On the challenging DreamBench and DreamBench-v2, our human evaluation shows that SuTI significantly outperforms existing models like InstructPix2Pix, Textual Inversion, Imagic, Prompt2Prompt, Re-Imagen and DreamBooth.

        ----

        ## [1318] GSLB: The Graph Structure Learning Benchmark

        **Authors**: *Zhixun Li, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, Jeffrey Xu Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60bc87f3cf5257579435d92ec12c761b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/60bc87f3cf5257579435d92ec12c761b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Graph Structure Learning (GSL) has recently garnered considerable attention due to its ability to optimize both the parameters of Graph Neural Networks (GNNs) and the computation graph structure simultaneously. Despite the proliferation of GSL methods developed in recent years, there is no standard experimental setting or fair comparison for performance evaluation, which creates a great obstacle to understanding the progress in this field. To fill this gap, we systematically analyze the performance of GSL in different scenarios and develop a comprehensive Graph Structure Learning Benchmark (GSLB) curated from 20 diverse graph datasets and 16 distinct GSL algorithms. Specifically, GSLB systematically investigates the characteristics of GSL in terms of three dimensions: effectiveness, robustness, and complexity. We comprehensively evaluate state-of-the-art GSL algorithms in node- and graph-level tasks, and analyze their performance in robust learning and model complexity. Further, to facilitate reproducible research, we have developed an easy-to-use library for training, evaluating, and visualizing different GSL methods. Empirical results of our extensive experiments demonstrate the ability of GSL and reveal its potential benefits on various downstream tasks, offering insights and opportunities for future research. The code of GSLB is available at: https://github.com/GSL-Benchmark/GSLB.

        ----

        ## [1319] Consensus and Subjectivity of Skin Tone Annotation for ML Fairness

        **Authors**: *Candice Schumann, Femi Olanubi, Auriel Wright, Ellis Monk Jr., Courtney Heldreth, Susanna Ricco*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60d25b3210c92f5ba2002a8e1f1adf1c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/60d25b3210c92f5ba2002a8e1f1adf1c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Understanding different human attributes and how they affect model behavior may become a standard need for all model creation and usage, from traditional computer vision tasks to the newest multimodal generative AI systems. In computer vision specifically, we have relied on datasets augmented with perceived attribute signals (eg, gender presentation, skin tone, and age) and benchmarks enabled by these datasets. Typically labels for these tasks come from human annotators. However, annotating attribute signals, especially skin tone, is a difficult and subjective task. Perceived skin tone is affected by technical factors, like lighting conditions, and social factors that shape an annotator's lived experience.This paper examines the subjectivity of skin tone annotation through a series of annotation experiments using the Monk Skin Tone (MST) scale~\cite{Monk2022Monk}, a small pool of professional photographers, and a much larger pool of trained crowdsourced annotators. Along with this study we release the Monk Skin Tone Examples (MST-E) dataset, containing 1515 images and 31 videos spread across the full MST scale. MST-E is designed to help train human annotators to annotate MST effectively.Our study shows that annotators can reliably annotate skin tone in a way that aligns with an expert in the MST scale, even under challenging environmental conditions. We also find evidence that annotators from different geographic regions rely on different mental models of MST categories resulting in annotations that systematically vary across regions. Given this, we advise practitioners to use a diverse set of annotators and a higher replication count for each image when annotating skin tone for fairness research.

        ----

        ## [1320] Large Language Models can Implement Policy Iteration

        **Authors**: *Ethan A. Brooks, Logan Walls, Richard L. Lewis, Satinder Singh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60dc7fa827f5f761ad481e2ad40b5573-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/60dc7fa827f5f761ad481e2ad40b5573-Abstract-Conference.html)

        **Abstract**:

        In this work, we demonstrate a method for implementing policy iteration using a large language model. While the application of foundation models to RL has received considerable attention, most approaches rely on either (1) the curation of expert demonstrations (either through manual design or task-specific pretraining) or (2) adaptation to the task of interest using gradient methods (either fine-tuning or training of adapter layers). Both of these techniques have drawbacks. Collecting demonstrations is labor-intensive, and algorithms that rely on them do not outperform the experts from which the demonstrations were derived. All gradient techniques are inherently slow, sacrificing the “few-shot” quality that makes in-context learning attractive to begin with. Our method demonstrates that a large language model can be used to implement policy iteration using the machinery of in-context learning, enabling it to learn to perform RL tasks without expert demonstrations or gradients. Our approach iteratively updates the contents of the prompt from which it derives its policy through trial-and-error interaction with an RL environment. In order to eliminate the role of in-weights learning (on which approaches like Decision Transformer rely heavily), we demonstrate our method using Codex (M. Chen et al. 2021b), a language model with no prior knowledge of the domains on which we evaluate it.

        ----

        ## [1321] ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

        **Authors**: *Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html)

        **Abstract**:

        Auxiliary-Task Learning (ATL) aims to improve the performance of the target task by leveraging the knowledge obtained from related tasks. Occasionally, learning multiple tasks simultaneously results in lower accuracy than learning only the target task, which is known as negative transfer. This problem is often attributed to the gradient conflicts among tasks, and is frequently tackled by coordinating the task gradients in previous works. However, these optimization-based methods largely overlook the auxiliary-target generalization capability. To better understand the root cause of negative transfer, we experimentally investigate it from both optimization and generalization perspectives. Based on our findings, we introduce ForkMerge, a novel approach that periodically forks the model into multiple branches, automatically searches the varying task weights by minimizing target validation errors, and dynamically merges all branches to filter out detrimental task-parameter updates. On a series of auxiliary-task learning benchmarks, ForkMerge outperforms existing methods and effectively mitigates negative transfer.

        ----

        ## [1322] Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness

        **Authors**: *Xinli Yue, Ningping Mou, Qian Wang, Lingchen Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)

        **Abstract**:

        Adversarial Robustness Distillation (ARD) aims to transfer the robustness of large teacher models to small student models, facilitating the attainment of robust performance on resource-limited devices. However, existing research on ARD primarily focuses on the overall robustness of student models, overlooking the crucial aspect of $\textit{robust fairness}$. Specifically, these models may demonstrate strong robustness on some classes of data while exhibiting high vulnerability on other classes. Unfortunately, the "buckets effect" implies that the robustness of the deployed model depends on the classes with the lowest level of robustness. In this paper, we first investigate the inheritance of robust fairness during ARD and reveal that student models only partially inherit robust fairness from teacher models. We further validate this issue through fine-grained experiments with various model capacities and find that it may arise due to the gap in capacity between teacher and student models, as well as the existing methods treating each class equally during distillation. Based on these observations, we propose $\textbf{Fair}$ $\textbf{A}$dversarial $\textbf{R}$obustness $\textbf{D}$istillation (Fair-ARD), a novel framework for enhancing the robust fairness of student models by increasing the weights of difficult classes, and design a geometric perspective-based method to quantify the difficulty of different classes for determining the weights. Extensive experiments show that Fair-ARD surpasses both state-of-the-art ARD methods and existing robust fairness algorithms in terms of robust fairness (e.g., the worst-class robustness under AutoAttack is improved by at most 12.3\% and 5.3\% using ResNet18 on CIFAR10, respectively), while also slightly improving overall robustness. Our code is available at: [https://github.com/NISP-official/Fair-ARD](https://github.com/NISP-official/Fair-ARD).

        ----

        ## [1323] Real3D-AD: A Dataset of Point Cloud Anomaly Detection

        **Authors**: *Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, Feng Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        High-precision point cloud anomaly detection is the gold standard for identifying the defects of advancing machining and precision manufacturing. Despite some methodological advances in this area, the scarcity of datasets and the lack of a systematic benchmark hinder its development. We introduce Real3D-AD, a challenging high-precision point cloud anomaly detection dataset, addressing the limitations in the field. With 1,254 high-resolution 3D items (from forty thousand to millions of points for each item), Real3D-AD is the largest dataset for high-precision 3D industrial anomaly detection to date. Real3D-AD surpasses existing 3D anomaly detection datasets available in terms of point cloud resolution (0.0010mm-0.0015mm), $360^{\circ}$ degree coverage and perfect prototype. Additionally, we present a comprehensive benchmark for Real3D-AD, revealing the absence of baseline methods for high-precision point cloud anomaly detection. To address this, we propose Reg3D-AD, a registration-based 3D anomaly detection method incorporating a novel feature memory bank that preserves local and global representations. Extensive experiments on the Real3D-AD dataset highlight the effectiveness of Reg3D-AD. For reproducibility and accessibility, we provide the Real3D-AD dataset, benchmark source code, and Reg3D-AD on our website: https://github.com/M-3LAB/Real3D-AD.

        ----

        ## [1324] Differentiable Sampling of Categorical Distributions Using the CatLog-Derivative Trick

        **Authors**: *Lennert De Smet, Emanuele Sansone, Pedro Zuidberg Dos Martires*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61202bb341e7e0a6026ea134a5057abf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61202bb341e7e0a6026ea134a5057abf-Abstract-Conference.html)

        **Abstract**:

        Categorical random variables can faithfully represent the discrete and uncertain aspects of data as part of a discrete latent variable model. Learning in such models necessitates taking gradients with respect to the parameters of the categorical probability distributions, which is often intractable due to their combinatorial nature. A popular technique to estimate these otherwise intractable gradients is the Log-Derivative trick. This trick forms the basis of the well-known REINFORCE gradient estimator and its many extensions. While the Log-Derivative trick allows us to differentiate through samples drawn from categorical distributions, it does not take into account the discrete nature of the distribution itself. Our first contribution addresses this shortcoming by introducing the CatLog-Derivative trick -- a variation of the Log-Derivative trick tailored towards categorical distributions. Secondly, we use the CatLog-Derivative trick to introduce IndeCateR, a novel and unbiased gradient estimator for the important case of products of independent categorical distributions with provably lower variance than REINFORCE. Thirdly, we empirically show that IndeCateR can be efficiently implemented and that its gradient estimates have significantly lower bias and variance for the same number of samples compared to the state of the art.

        ----

        ## [1325] Variational Imbalanced Regression: Fair Uncertainty Quantification via Probabilistic Smoothing

        **Authors**: *Ziyan Wang, Hao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/612a56f193d031687683445cd0001083-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/612a56f193d031687683445cd0001083-Abstract-Conference.html)

        **Abstract**:

        Existing regression models tend to fall short in both accuracy and uncertainty estimation when the label distribution is imbalanced. In this paper, we propose a probabilistic deep learning model, dubbed variational imbalanced regression (VIR), which not only performs well in imbalanced regression but naturally produces reasonable uncertainty estimation as a byproduct. Different from typical variational autoencoders assuming I.I.D. representations (a data point's representation is not directly affected by other data points), our VIR borrows data with similar regression labels to compute the latent representation's variational distribution; furthermore, different from deterministic regression models producing point estimates, VIR predicts the entire normal-inverse-gamma distributions and modulates the associated conjugate distributions to impose probabilistic reweighting on the imbalanced data, thereby providing better uncertainty estimation. Experiments in several real-world datasets show that our VIR can outperform state-of-the-art imbalanced regression models in terms of both accuracy and uncertainty estimation. Code will soon be available at https://github.com/Wang-ML-Lab/variational-imbalanced-regression.

        ----

        ## [1326] Towards A Richer 2D Understanding of Hands at Scale

        **Authors**: *Tianyi Cheng, Dandan Shan, Ayda Hassen, Richard E. L. Higgins, David Fouhey*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html)

        **Abstract**:

        As humans, we learn a lot about how to interact with the world by observing others interacting with their hands. To help AI systems obtain a better understanding of hand interactions, we introduce a new model that produces a rich understanding of hand interaction. Our system produces a richer output than past systems at a larger scale. Our outputs include boxes and segments for hands, in-contact objects, and second objects touched by tools as well as contact and grasp type. Supporting this method are annotations of 257K images, 401K hands, 288K objects, and 19K second objects spanning four datasets. We show that our method provides rich information and performs and generalizes well.

        ----

        ## [1327] Effective Human-AI Teams via Learned Natural Language Rules and Onboarding

        **Authors**: *Hussein Mozannar, Jimin J. Lee, Dennis Wei, Prasanna Sattigeri, Subhro Das, David A. Sontag*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61355b9c218505505d1bedede9da56b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61355b9c218505505d1bedede9da56b2-Abstract-Conference.html)

        **Abstract**:

        People are relying on AI agents to assist them with various tasks. The human must know when to rely on the agent, collaborate with the agent, or ignore its suggestions. In this work, we propose to learn rules grounded in data regions and described in natural language that illustrate how the human should collaborate with the AI. Our novel region discovery algorithm finds local regions in the data as neighborhoods in an embedding space that corrects the human prior. Each region is then described using an iterative and contrastive procedure where a large language model describes the region. We then teach these rules to the human via an onboarding stage. Through user studies on object detection and question-answering tasks, we show that our method can lead to more accurate human-AI teams. We also evaluate our region discovery and description algorithms separately.

        ----

        ## [1328] On Certified Generalization in Structured Prediction

        **Authors**: *Bastian Boll, Christoph Schnörr*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61674667d642ae52f6bb281bea90ee29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61674667d642ae52f6bb281bea90ee29-Abstract-Conference.html)

        **Abstract**:

        In structured prediction, target objects have rich internal structure which does not factorize into independent components and violates common i.i.d. assumptions. This challenge becomes apparent through the exponentially large output space in applications such as image segmentation or scene graph generation.We present a novel PAC-Bayesian risk bound for structured prediction wherein the rate of generalization scales not only with the number of structured examples but also with their size.The underlying assumption, conforming to ongoing research on generative models, is that data are generated by the Knothe-Rosenblatt rearrangement of a factorizing reference measure. This allows to explicitly distill the structure between random output variables into a Wasserstein dependency matrix. Our work makes a preliminary step towards leveraging powerful generative models to establish generalization bounds for discriminative downstream tasks in the challenging setting of structured prediction.

        ----

        ## [1329] SutraNets: Sub-series Autoregressive Networks for Long-Sequence, Probabilistic Forecasting

        **Authors**: *Shane Bergsma, Timothy Zeyl, Lei Guo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6171c9e600432a42688ad61a525951bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6171c9e600432a42688ad61a525951bf-Abstract-Conference.html)

        **Abstract**:

        We propose SutraNets, a novel method for neural probabilistic forecasting of long-sequence time series. SutraNets use an autoregressive generative model to factorize the likelihood of long sequences into products of conditional probabilities. When generating long sequences, most autoregressive approaches suffer from harmful error accumulation, as well as challenges in modeling long-distance dependencies. SutraNets treat long, univariate prediction as multivariate prediction over lower-frequency sub-series. Autoregression proceeds across time and across sub-series in order to ensure coherent multivariate (and, hence, high-frequency univariate) outputs. Since sub-series can be generated using fewer steps, SutraNets effectively reduce error accumulation and signal path distances. We find SutraNets to significantly improve forecasting accuracy over competitive alternatives on six real-world datasets, including when we vary the number of sub-series and scale up the depth and width of the underlying sequence models.

        ----

        ## [1330] Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints

        **Authors**: *Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, Yangqiu Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html)

        **Abstract**:

        Querying knowledge graphs (KGs) using deep learning approaches can naturally leverage the reasoning and generalization ability to learn to infer better answers. Traditional neural complex query answering (CQA) approaches mostly work on entity-centric KGs. However, in the real world, we also need to make logical inferences about events, states, and activities (i.e., eventualities or situations) to push learning systems from System I to System II, as proposed by Yoshua Bengio. Querying logically from an EVentuality-centric KG (EVKG) can naturally provide references to such kind of intuitive and logical inference. Thus, in this paper, we propose a new framework to leverage neural methods to answer complex logical queries based on an EVKG, which can satisfy not only traditional first-order logic constraints but also implicit logical constraints over eventualities concerning their occurrences and orders. For instance, if we know that Food is bad happens before PersonX adds soy sauce, then PersonX adds soy sauce is unlikely to be the cause of Food is bad due to implicit temporal constraint. To facilitate consistent reasoning on EVKGs, we propose Complex Eventuality Query Answering (CEQA), a more rigorous definition of CQA that considers the implicit logical constraints governing the temporal order and occurrence of eventualities. In this manner, we propose to leverage theorem provers for constructing benchmark datasets to ensure the answers satisfy implicit logical constraints. We also propose a Memory-Enhanced Query Encoding (MEQE) approach to significantly improve the performance of state-of-the-art neural query encoders on the CEQA task.

        ----

        ## [1331] Fast Bellman Updates for Wasserstein Distributionally Robust MDPs

        **Authors**: *Zhuodong Yu, Ling Dai, Shaohang Xu, Siyang Gao, Chin Pang Ho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61779e9b0c26a31c5f36bd3e8c180dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61779e9b0c26a31c5f36bd3e8c180dcf-Abstract-Conference.html)

        **Abstract**:

        Markov decision processes (MDPs) often suffer from the sensitivity issue under model ambiguity. In recent years, robust MDPs have emerged as an effective framework to overcome this challenge. Distributionally robust MDPs extend the robust MDP framework by incorporating distributional information of the uncertain model parameters to alleviate the conservative nature of robust MDPs. This paper proposes a computationally efficient solution framework for solving distributionally robust MDPs with Wasserstein ambiguity sets. By exploiting the specific problem structure, the proposed framework decomposes the optimization problems associated with distributionally robust Bellman updates into smaller subproblems, which can be solved efficiently. The overall complexity of the proposed algorithm is quasi-linear in both the numbers of states and actions when the distance metric of the Wasserstein distance is chosen to be $L_1$, $L_2$, or $L_{\infty}$ norm, and so the computational cost of distributional robustness is substantially reduced. Our numerical experiments demonstrate that the proposed algorithms outperform other state-of-the-art solution methods.

        ----

        ## [1332] LoRA: A Logical Reasoning Augmented Dataset for Visual Question Answering

        **Authors**: *Jingying Gao, Qi Wu, Alan Blair, Maurice Pagnucco*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/617ff5271b2b41dfb217a3b0f1b3d1be-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/617ff5271b2b41dfb217a3b0f1b3d1be-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The capacity to reason logically is a hallmark of human cognition. Humans excel at integrating multimodal information for locigal reasoning, as exemplified by the Visual Question Answering (VQA) task, which is a challenging multimodal task. VQA tasks and large vision-and-language models aim to tackle reasoning problems, but the accuracy, consistency and fabrication of the generated answers is hard to evaluate in the absence of a VQA dataset that can offer formal, comprehensive and systematic complex logical reasoning questions. To address this gap, we present LoRA, a novel Logical Reasoning Augmented VQA dataset that requires formal and complex description logic reasoning based on a food-and-kitchen knowledge base. Our main objective in creating LoRA is to enhance the complex and formal logical reasoning capabilities of VQA models, which are not adequately measured by existing VQA datasets. We devise strong and flexible programs to automatically generate 200,000 diverse description logic reasoning questions based on the SROIQ Description Logic, along with realistic kitchen scenes and ground truth answers. We fine-tune the latest transformer VQA models and evaluate the zero-shot performance of the state-of-the-art large vision-and-language models on LoRA. The results reveal that LoRA presents a unique challenge in logical reasoning, setting a systematic and comprehensive evaluation standard.

        ----

        ## [1333] Practical Contextual Bandits with Feedback Graphs

        **Authors**: *Mengxiao Zhang, Yuheng Zhang, Olga Vrousgou, Haipeng Luo, Paul Mineiro*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html)

        **Abstract**:

        While contextual bandit has a mature theory, effectively leveraging different feedback patterns to enhance the pace of learning remains unclear. Bandits with feedback graphs, which interpolates between the full information and bandit regimes, provides a promising framework to mitigate the statistical complexity of learning. In this paper, we propose and analyze an approach to contextual bandits with feedback graphs based upon reduction to regression.  The resulting algorithms are computationally practical and achieve established minimax rates, thereby reducing the statistical complexity in real-world applications.

        ----

        ## [1334] Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control

        **Authors**: *Nate Rahn, Pierluca D'Oro, Harley Wiltzer, Pierre-Luc Bacon, Marc G. Bellemare*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6191ab7080c840f67eaf5dff7d5edfcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6191ab7080c840f67eaf5dff7d5edfcb-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning agents for continuous control are known to exhibit significant instability in their performance over time. In this work, we provide a fresh perspective on these behaviors by studying the return landscape: the mapping between a policy and a return. We find that popular algorithms traverse noisy neighborhoods of this landscape, in which a single update to the policy parameters leads to a wide range of returns. By taking a distributional view of these returns, we map the landscape, characterizing failure-prone regions of policy space and revealing a hidden dimension of policy quality. We show that the landscape exhibits surprising structure by finding simple paths in parameter space which improve the stability of a policy. To conclude, we develop a distribution-aware procedure which finds such paths, navigating away from noisy neighborhoods in order to improve the robustness of a policy. Taken together, our results provide new insight into the optimization, evaluation, and design of agents.

        ----

        ## [1335] Real-World Image Variation by Aligning Diffusion Inversion Chain

        **Authors**: *Yuechen Zhang, Jinbo Xing, Eric Lo, Jiaya Jia*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html)

        **Abstract**:

        Recent diffusion model advancements have enabled high-fidelity images to be generated using text prompts. However, a domain gap exists between generated images and real-world images, which poses a challenge in generating high-quality variations of real-world images. Our investigation uncovers that this domain gap originates from a latents' distribution gap in different diffusion processes. To address this issue, we propose a novel inference pipeline called Real-world Image Variation by ALignment (RIVAL) that utilizes diffusion models to generate image variations from a single image exemplar. Our pipeline enhances the generation quality of image variations by aligning the image generation process to the source image's inversion chain. Specifically, we demonstrate that step-wise latent distribution alignment is essential for generating high-quality variations. To attain this, we design a cross-image self-attention injection for feature interaction and a step-wise distribution normalization to align the latent features. Incorporating these alignment processes into a diffusion model allows RIVAL to generate high-quality image variations without further parameter optimization. Our experimental results demonstrate that our proposed approach outperforms existing methods concerning semantic similarity and perceptual quality. This generalized inference pipeline can be easily applied to other diffusion-based generation tasks, such as image-conditioned text-to-image generation and stylization. Project page: https://rival-diff.github.io

        ----

        ## [1336] Vocabulary-free Image Classification

        **Authors**: *Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, Elisa Ricci*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/619cbddb92b8c6fecaf2b86463153be9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/619cbddb92b8c6fecaf2b86463153be9-Abstract-Conference.html)

        **Abstract**:

        Recent advances in large vision-language models have revolutionized the image classification paradigm. Despite showing impressive zero-shot capabilities, a pre-defined set of categories, a.k.a. the vocabulary, is assumed at test time for composing the textual prompts. However, such assumption can be impractical when the semantic context is unknown and evolving. We thus formalize a novel task, termed as Vocabulary-free Image Classification (VIC), where we aim to assign to an input image a class that resides in an unconstrained language-induced semantic space, without the prerequisite of a known vocabulary. VIC is a challenging task as the semantic space is extremely large, containing millions of concepts, with hard-to-discriminate fine-grained categories. In this work, we first empirically verify that representing this semantic space by means of an external vision-language database is the most effective way to obtain semantically relevant content for classifying the image. We then propose Category Search from External Databases (CaSED), a method that exploits a pre-trained vision-language model and an external vision-language database to address VIC in a training-free manner. CaSED first extracts a set of candidate categories from captions retrieved from the database based on their semantic similarity to the image, and then assigns to the image the best matching candidate category according to the same vision-language model. Experiments on benchmark datasets validate that CaSED outperforms other complex vision-language frameworks, while being efficient with much fewer parameters, paving the way for future research in this direction.

        ----

        ## [1337] A Novel Framework for Policy Mirror Descent with General Parameterization and Linear Convergence

        **Authors**: *Carlo Alfano, Rui Yuan, Patrick Rebeschini*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61a9278dfef5f871b5e472389f8d6fa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61a9278dfef5f871b5e472389f8d6fa1-Abstract-Conference.html)

        **Abstract**:

        Modern policy optimization methods in reinforcement learning, such as TRPO and PPO, owe their success to the use of parameterized policies. However, while theoretical guarantees have been established for this class of algorithms, especially in the tabular setting, the use of general parameterization schemes remains mostly unjustified. In this work, we introduce a novel framework for policy optimization based on mirror descent that naturally accommodates general parameterizations. The policy class induced by our scheme recovers known classes, e.g., softmax, and generates new ones depending on the choice of mirror map. Using our framework, we obtain the first result that guarantees linear convergence for a policy-gradient-based method involving general parameterization. To demonstrate the ability of our framework to accommodate general parameterization schemes, we provide its sample complexity when using shallow neural networks, show that it represents an improvement upon the previous best results, and empirically validate the effectiveness of our theoretical claims on classic control tasks.

        ----

        ## [1338] Weakly-Supervised Concealed Object Segmentation with SAM-based Pseudo Labeling and Multi-scale Feature Grouping

        **Authors**: *Chunming He, Kai Li, Yachao Zhang, Guoxia Xu, Longxiang Tang, Yulun Zhang, Zhenhua Guo, Xiu Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html)

        **Abstract**:

        Weakly-Supervised Concealed Object Segmentation (WSCOS) aims to  segment objects well blended with surrounding environments using sparsely-annotated data for model training. It remains a challenging task since (1) it is hard to distinguish concealed objects from the background due to the intrinsic similarity and (2) the sparsely-annotated training data only provide weak supervision for model learning. In this paper, we propose a new WSCOS method to address these two challenges. To tackle the intrinsic similarity challenge, we design a multi-scale feature grouping module that first groups features at different granularities and then aggregates these grouping results. By grouping similar features together, it encourages segmentation coherence, helping obtain complete segmentation results for both single and multiple-object images. For the weak supervision challenge, we utilize the recently-proposed vision foundation model, ``Segment Anything Model (SAM)'', and use the provided sparse annotations as prompts to generate segmentation masks, which are used to train the model. To alleviate the impact of low-quality segmentation masks, we further propose a series of strategies, including multi-augmentation result ensemble, entropy-based pixel-level weighting, and entropy-based image-level selection. These strategies help provide more reliable supervision to train the segmentation model. We verify the effectiveness of our method on various WSCOS tasks, and experiments demonstrate that our method achieves state-of-the-art performance on these tasks.

        ----

        ## [1339] Ordering-based Conditions for Global Convergence of Policy Gradient Methods

        **Authors**: *Jincheng Mei, Bo Dai, Alekh Agarwal, Mohammad Ghavamzadeh, Csaba Szepesvári, Dale Schuurmans*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61c00c07e6d27285e4b952e96cc65666-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61c00c07e6d27285e4b952e96cc65666-Abstract-Conference.html)

        **Abstract**:

        We prove that, for finite-arm bandits with linear function approximation, the global convergence of policy gradient (PG) methods depends on inter-related properties between the policy update and the representation. textcolor{blue}{First}, we establish a few key observations that frame the study: \textbf{(i)} Global convergence can be achieved under linear function approximation without policy or reward realizability, both for the standard Softmax PG and natural policy gradient (NPG). \textbf{(ii)} Approximation error is not a key quantity for characterizing global convergence in either algorithm. \textbf{(iii)} The conditions on the representation that imply global convergence are different between these two algorithms. Overall, these observations call into question approximation error as an appropriate quantity for characterizing the global convergence of PG methods under linear function approximation. \textcolor{blue}{Second}, motivated by these observations, we establish new general results: \textbf{(i)} NPG with linear function approximation achieves global convergence \emph{if and only if} the projection of the reward  onto the representable space preserves the optimal action's rank, a quantity that is not strongly related to approximation error. \textbf{(ii)} The global convergence of Softmax PG occurs if the representation satisfies a non-domination condition and can preserve the ranking of rewards, which goes well beyond policy or reward realizability. We provide experimental results to support these theoretical findings.

        ----

        ## [1340] Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning

        **Authors**: *Berken Utku Demirel, Christian Holz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61c2c6338033da68885e0226881cbe71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61c2c6338033da68885e0226881cbe71-Abstract-Conference.html)

        **Abstract**:

        The success of contrastive learning is well known to be dependent on data augmentation.Although the degree of data augmentations has been well controlled by utilizing pre-defined techniques in some domains like vision, time-series data augmentation is less explored and remains a challenging problem due to the complexity of the data generation mechanism, such as the intricate mechanism involved in the cardiovascular system.Moreover, there is no widely recognized and general time-series augmentation method that can be applied across different tasks.In this paper, we propose a novel data augmentation method for time-series tasks that aims to connect intra-class samples together, and thereby find order in the latent space.Our method builds upon the well-known data augmentation technique of mixup by incorporating a novel approach that accounts for the non-stationary nature of time-series data.Also, by controlling the degree of chaos created by data augmentation, our method leads to improved feature representations and performance on downstream tasks.We evaluate our proposed method on three time-series tasks, including heart rate estimation, human activity recognition, and cardiovascular disease detection. Extensive experiments against the state-of-the-art methods show that the proposed method outperforms prior works on optimal data generation and known data augmentation techniques in three tasks, reflecting the effectiveness of the presented method. The source code is available at double-blind policy.

        ----

        ## [1341] List and Certificate Complexities in Replicable Learning

        **Authors**: *Peter Dixon, Aduri Pavan, Jason Vander Woude, N. V. Vinodchandran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61d0a96d4a73b626367310b3ad32579d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61d0a96d4a73b626367310b3ad32579d-Abstract-Conference.html)

        **Abstract**:

        We investigate replicable learning algorithms. Informally  a learning algorithm is replicable if the algorithm outputs the same canonical hypothesis over multiple runs with high probability, even when different runs observe a different set of samples from the unknown data distribution. In general, such a strong notion of replicability is not achievable. Thus we consider two feasible notions of replicability called {\em list replicability} and {\em certificate replicability}. Intuitively, these notions capture the degree of (non) replicability. The goal is to design learning algorithms with optimal list and certificate complexities while minimizing the sample complexity.  Our contributions are the following.1. We first study the learning task of estimating the biases of $d$ coins, up to an additive error of $\varepsilon$, by observing samples. For this task, we design a $(d+1)$-list replicable algorithm. To complement this result, we establish that the list complexity is optimal, i.e there are no learning algorithms with a list size smaller than $d+1$ for this task. We also design learning algorithms with certificate complexity $\tilde{O}(\log d)$.   The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\varepsilon^2})$ where $\varepsilon$ is the approximation error parameter (for a constant error probability).  2. In the PAC model, we show that any hypothesis class that is learnable with $d$-nonadaptive statistical queries can be learned via a $(d+1)$-list replicable algorithm and also via a $\tilde{O}(\log d)$-certificate replicable algorithm. The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\nu^2})$ where $\nu$ is the approximation error of the statistical query. We also show that for the concept class \dtep, the list complexity is exactly $d+1$ with respect to the uniform distribution.   To establish our upper bound results we use rounding schemes induced by geometric partitions with certain properties. We use Sperner/KKM Lemma to establish the lower bound results.

        ----

        ## [1342] Flat Seeking Bayesian Neural Networks

        **Authors**: *Van-Anh Nguyen, Tung-Long Vuong, Hoang Phan, Thanh-Toan Do, Dinh Q. Phung, Trung Le*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html)

        **Abstract**:

        Bayesian Neural Networks (BNNs) provide a probabilistic interpretation for deep learning models by imposing a prior distribution over model parameters and inferring a posterior distribution based on observed data. The model sampled from the posterior distribution can be used for providing ensemble predictions and quantifying prediction uncertainty. It is well-known that deep learning models with lower sharpness have better generalization ability. However, existing posterior inferences are not aware of sharpness/flatness in terms of formulation, possibly leading to high sharpness for the models sampled from them. In this paper, we develop theories, the Bayesian setting, and the variational inference approach for the sharpness-aware posterior. Specifically, the models sampled from our sharpness-aware posterior, and the optimal approximate posterior estimating this sharpness-aware posterior, have better flatness, hence possibly possessing higher generalization ability. We conduct experiments by leveraging the sharpness-aware posterior with state-of-the-art Bayesian Neural Networks, showing that the flat-seeking counterparts outperform their baselines in all metrics of interest.

        ----

        ## [1343] Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns

        **Authors**: *Xin Liu, Zheng Li, Yifan Gao, Jingfeng Yang, Tianyu Cao, Zhengyang Wang, Bing Yin, Yangqiu Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/621d0fd41c720ab252e178b77c200d90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/621d0fd41c720ab252e178b77c200d90-Abstract-Conference.html)

        **Abstract**:

        The goal of session-based recommendation in E-commerce is to predict the next item that an anonymous user will purchase based on the browsing and purchase history. However, constructing global or local transition graphs to supplement session data can lead to noisy correlations and user intent vanishing. In this work, we propose the Frequent Attribute Pattern Augmented Transformer (FAPAT) that characterizes user intents by building attribute transition graphs and matching attribute patterns. Specifically, the frequent and compact attribute patterns are served as memory to augment session representations, followed by a gate and a transformer block to fuse the whole session information. Through extensive experiments on two public benchmarks and 100 million industrial data in three domains, we demonstrate that FAPAT consistently outperforms state-of-the-art methods by an average of 4.5% across various evaluation metrics (Hits, NDCG, MRR). Besides evaluating the next-item prediction, we estimate the models' capabilities to capture user intents via predicting items' attributes and period-item recommendations.

        ----

        ## [1344] Can Language Models Solve Graph Problems in Natural Language?

        **Authors**: *Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) are increasingly adopted for a variety of tasks with implicit graphical structures, such as planning in robotics, multi-hop question answering or knowledge probing, structured commonsense reasoning, and more. While LLMs have advanced the state-of-the-art on these tasks with structure implications, whether LLMs could explicitly process textual descriptions of graphs and structures, map them to grounded conceptual spaces, and perform structured operations remains underexplored. To this end, we propose NLGraph (Natural Language Graph), a comprehensive benchmark of graph-based problem solving designed in natural language. NLGraph contains 29,370 problems, covering eight graph reasoning tasks with varying complexity from simple tasks such as connectivity and shortest path up to complex problems such as maximum flow and simulating graph neural networks. We evaluate LLMs (GPT-3/4) with various prompting approaches on the NLGraph benchmark and find that 1) language models do demonstrate preliminary graph reasoning abilities, 2) the benefit of advanced prompting and in-context learning diminishes on more complex graph problems, while 3) LLMs are also (un)surprisingly brittle in the face of spurious correlations in graph and problem settings. We then propose Build-a-Graph Prompting and Algorithmic Prompting, two instruction-based approaches to enhance LLMs in solving natural language graph problems. Build-a-Graph and Algorithmic prompting improve the performance of LLMs on NLGraph by 3.07% to 16.85% across multiple tasks and settings, while how to solve the most complicated graph reasoning tasks in our setup with language models remains an open research question.

        ----

        ## [1345] Language-driven Scene Synthesis using Multi-conditional Diffusion Model

        **Authors**: *Vuong Dinh An, Minh Nhat Vu, Toan Nguyen, Baoru Huang, Dzung Nguyen, Thieu Vo, Anh Nguyen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html)

        **Abstract**:

        Scene synthesis is a challenging problem with several industrial applications. Recently, substantial efforts have been directed to synthesize the scene using human motions, room layouts, or spatial graphs as the input. However, few studies have addressed this problem from multiple modalities, especially combining text prompts. In this paper, we propose a language-driven scene synthesis task, which is a new task that integrates text prompts, human motion, and existing objects for scene synthesis. Unlike other single-condition synthesis tasks, our problem involves multiple conditions and requires a strategy for processing and encoding them into a unified space. To address the challenge, we present a multi-conditional diffusion model, which differs from the implicit unification approach of other diffusion literature by explicitly predicting the guiding points for the original data distribution. We demonstrate that our approach is theoretically supportive. The intensive experiment results illustrate that our method outperforms state-of-the-art benchmarks and enables natural scene editing applications. The source code and dataset can be accessed at https://lang-scene-synth.github.io/.

        ----

        ## [1346] Implicit Contrastive Representation Learning with Guided Stop-gradient

        **Authors**: *Byeongchan Lee, Sehyun Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6274172f7d981a8d58bbfd52342a9d1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6274172f7d981a8d58bbfd52342a9d1f-Abstract-Conference.html)

        **Abstract**:

        In self-supervised representation learning, Siamese networks are a natural architecture for learning transformation-invariance by bringing representations of positive pairs closer together. But it is prone to collapse into a degenerate solution. To address the issue, in contrastive learning, a contrastive loss is used to prevent collapse by moving representations of negative pairs away from each other. But it is known that algorithms with negative sampling are not robust to a reduction in the number of negative samples. So, on the other hand, there are algorithms that do not use negative pairs. Many positive-only algorithms adopt asymmetric network architecture consisting of source and target encoders as a key factor in coping with collapse. By exploiting the asymmetric architecture, we introduce a methodology to implicitly incorporate the idea of contrastive learning. As its implementation, we present a novel method guided stop-gradient. We apply our method to benchmark algorithms SimSiam and BYOL and show that our method stabilizes training and boosts performance. We also show that the algorithms with our method work well with small batch sizes and do not collapse even when there is no predictor. The code is available in the supplementary material.

        ----

        ## [1347] QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

        **Authors**: *Simone Papicchio, Paolo Papotti, Luca Cagliero*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62a24b69b820d30e9e5ad4f15ff7bf72-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/62a24b69b820d30e9e5ad4f15ff7bf72-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Table Representation Learning (TRL) models are commonly pre-trained on large open-domain datasets comprising millions of tables and then used to address downstream tasks. Choosing the right TRL model to use on proprietary data can be challenging, as the best results depend on the content domain, schema, and data quality. Our purpose is to support end-users in testing TRL models on proprietary data in two established SQL-centric tasks, i.e., Question Answering (QA) and Semantic Parsing (SP). We present QATCH (Query-Aided TRL Checklist), a toolbox to highlight TRL models’ strengths and weaknesses on relational tables unseen at training time. For an input table, QATCH automatically generates a testing checklist tailored to QA and SP. Checklist generation is driven by a SQL query engine that crafts tests of different complexity. This design facilitates inherent portability, allowing the checks to be used by alternative models. We also introduce a set of cross-task performance metrics evaluating the TRL model’s performance over its output. Finally, we show how QATCH automatically generates tests for proprietary datasets to evaluate various state-of-the-art models including TAPAS, TAPEX, and CHATGPT.

        ----

        ## [1348] Improved Best-of-Both-Worlds Guarantees for Multi-Armed Bandits: FTRL with General Regularizers and Multiple Optimal Arms

        **Authors**: *Tiancheng Jin, Junyan Liu, Haipeng Luo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html)

        **Abstract**:

        We study the problem of designing adaptive multi-armed bandit algorithms that perform optimally in both the stochastic setting and the adversarial setting simultaneously (often known as a best-of-both-world guarantee). A line of recent works shows that when configured and analyzed properly, the Follow-the-Regularized-Leader (FTRL) algorithm, originally designed for the adversarial setting, can in fact optimally adapt to the stochastic setting as well. Such results, however, critically rely on an assumption that there exists one unique optimal arm. Recently, Ito [2021] took the first step to remove such an undesirable uniqueness assumption for one particular FTRL algorithm withthe 1/2-Tsallis entropy regularizer. In this work, we significantly improve and generalize this result, showing that uniqueness is unnecessary for FTRL with a broad family of regularizers and a new learning rate schedule. For some regularizers, our regret bounds also improve upon prior results even when uniqueness holds. We further provide an application of our results to the decoupled exploration and exploitation problem, demonstrating that our techniques are broadly applicable.

        ----

        ## [1349] AiluRus: A Scalable ViT Framework for Dense Prediction

        **Authors**: *Jin Li, Yaoming Wang, Xiaopeng Zhang, Bowen Shi, Dongsheng Jiang, Chenglin Li, Wenrui Dai, Hongkai Xiong, Qi Tian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html)

        **Abstract**:

        Vision transformers (ViTs) have emerged as a prevalent architecture for vision tasks owing to their impressive performance. However, their complexity dramatically increases when handling long token sequences, particularly for dense prediction tasks that require high-resolution input. Notably, dense prediction tasks, such as semantic segmentation or object detection, emphasize more on the contours or shapes of objects, while the texture inside objects is less informative. Motivated by this observation, we propose to apply adaptive resolution for different regions in the image according to their importance. Specifically, at the intermediate layer of the ViT, we select anchors from the token sequence using the proposed spatial-aware density-based clustering algorithm. Tokens that are adjacent to anchors are merged to form low-resolution regions, while others are preserved independently as high-resolution. This strategy could significantly reduce the number of tokens, and the following layers only handle the reduced token sequence for acceleration. At the output end, the resolution of the feature map is recovered by unfolding merged tokens for task prediction. Consequently, we can considerably accelerate ViTs for dense prediction tasks. The proposed method is evaluated across three different datasets and demonstrates promising performance. For instance, "Segmenter ViT-L" can be accelerated by 48\% FPS without fine-tuning, while maintaining the performance. Moreover, our method can also be applied to accelerate fine-tuning. Experiments indicate that we can save 52\% training time while accelerating 2.46$\times$ FPS with only a 0.09\% performance drop.

        ----

        ## [1350] CORL: Research-oriented Deep Offline Reinforcement Learning Library

        **Authors**: *Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, Sergey Kolesnikov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/62d2cec62b7fd46dd35fa8f2d4aeb52d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/62d2cec62b7fd46dd35fa8f2d4aeb52d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        CORL is an open-source library that provides thoroughly benchmarked single-file implementations of both deep offline and offline-to-online reinforcement learning algorithms. It emphasizes a simple developing experience with a straightforward codebase and a modern analysis tracking tool. In CORL, we isolate methods implementation into separate single files, making performance-relevant details easier to recognize. Additionally, an experiment tracking feature is available to help log metrics, hyperparameters, dependencies, and more to the cloud. Finally, we have ensured the reliability of the implementations by benchmarking commonly employed D4RL datasets providing a transparent source of results that can be reused for robust evaluation tools such as performance profiles, probability of improvement, or expected online performance.

        ----

        ## [1351] LightSpeed: Light and Fast Neural Light Fields on Mobile Devices

        **Authors**: *Aarush Gupta, Junli Cao, Chaoyang Wang, Ju Hu, Sergey Tulyakov, Jian Ren, László A. Jeni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html)

        **Abstract**:

        Real-time novel-view image synthesis on mobile devices is prohibitive due to the limited computational power and storage. Using volumetric rendering methods, such as NeRF and its derivatives, on mobile devices is not suitable due to the high computational cost of volumetric rendering. On the other hand, recent advances in neural light field representations have shown promising real-time view synthesis results on mobile devices. Neural light field methods learn a direct mapping from a ray representation to the pixel color. The current choice of ray representation is either stratified ray sampling or Plücker coordinates, overlooking the classic light slab (two-plane) representation, the preferred representation to interpolate between light field views. In this work, we find that using the light slab representation is an efficient representation for learning a neural light field. More importantly, it is a lower-dimensional ray representation enabling us to learn the 4D ray space using feature grids which are significantly faster to train and render. Although mostly designed for frontal views, we show that the light-slab representation can be further extended to non-frontal scenes using a divide-and-conquer strategy. Our method provides better rendering quality than prior light field methods and a significantly better trade-off between rendering quality and speed than prior light field methods.

        ----

        ## [1352] CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models

        **Authors**: *Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631bb9434d718ea309af82566347d607-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631bb9434d718ea309af82566347d607-Abstract-Conference.html)

        **Abstract**:

        The ability to perform causal reasoning is widely considered a core feature of intelligence. In this work, we investigate whether large language models (LLMs) can coherently reason about causality. Much of the existing work in natural language processing (NLP) focuses on evaluating commonsense causal reasoning in LLMs, thus failing to assess whether a model can perform causal inference in accordance with a set of well-defined formal rules. To address this, we propose a new NLP task, causal inference in natural language, inspired by the "causal inference engine" postulated by Judea Pearl et al. We compose a large dataset, CLadder, with 10K samples: based on a collection of causal graphs and queries (associational, interventional, and counterfactual), we obtain symbolic questions and ground-truth answers, through an oracle causal inference engine. These are then translated into natural language. We evaluate multiple LLMs on our dataset, and we introduce and evaluate a bespoke chain-of-thought prompting strategy, CausalCoT. We show that our task is highly challenging for LLMs, and we conduct an in-depth analysis to gain deeper insight into the causal reasoning abilities of LLMs. Our data is open-sourced at https://huggingface.co/datasets/causalNLP/cladder, and our code can be found at https://github.com/causalNLP/cladder.

        ----

        ## [1353] Riemannian Laplace approximations for Bayesian neural networks

        **Authors**: *Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, Georgios Arvanitidis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html)

        **Abstract**:

        Bayesian neural networks often approximate the weight-posterior with a Gaussian distribution. However, practical posteriors are often, even locally, highly non-Gaussian, and empirical performance deteriorates. We propose a simple parametric approximate posterior that adapts to the shape of the true posterior through a Riemannian metric that is determined by the log-posterior gradient. We develop a Riemannian Laplace approximation where samples naturally fall into weight-regions with low negative log-posterior. We show that these samples can be drawn by solving a system of ordinary differential equations, which can be done efficiently by leveraging the structure of the Riemannian metric and automatic differentiation. Empirically, we demonstrate that our approach consistently improves over the conventional Laplace approximation across tasks. We further show that, unlike the conventional Laplace approximation, our method is not overly sensitive to the choice of prior, which alleviates a practical pitfall of current approaches.

        ----

        ## [1354] SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality

        **Authors**: *Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/63461de0b4cb760fc498e85b18a7fe81-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In the last year alone, a surge of new benchmarks to measure $\textit{compositional}$ understanding of vision-language models have permeated the machine learning ecosystem.Given an image, these benchmarks probe a model's ability to identify its associated caption amongst a set of compositional distractors.Surprisingly, we find significant biases in $\textit{all}$ these benchmarks rendering them hackable. This hackability is so dire that blind models with no access to the image outperform state-of-the-art vision-language models.To remedy this rampant vulnerability, we introduce $\textit{SugarCrepe}$, a new benchmark for vision-language compositionality evaluation.We employ large language models, instead of rule-based templates used in previous benchmarks, to generate fluent and sensical hard negatives, and utilize an adversarial refinement mechanism to maximally reduce biases. We re-evaluate state-of-the-art models and recently proposed compositionality inducing strategies, and find that their improvements were hugely overestimated, suggesting that more innovation is needed in this important direction.We release $\textit{SugarCrepe}$ and the code for evaluation at: https://github.com/RAIVNLab/sugar-crepe.

        ----

        ## [1355] Contrastive Retrospection: honing in on critical steps for rapid learning and generalization in RL

        **Authors**: *Chen Sun, Wannan Yang, Thomas Jiralerspong, Dane Malenfant, Benjamin Alsbury-Nealy, Yoshua Bengio, Blake A. Richards*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6357d6d068622c962391081d296bed69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6357d6d068622c962391081d296bed69-Abstract-Conference.html)

        **Abstract**:

        In real life, success is often contingent upon multiple critical steps that are distant in time from each other and from the final reward. These critical steps are challenging to identify with traditional reinforcement learning (RL) methods that rely on the Bellman equation for credit assignment. Here, we present a new RL algorithm that uses offline contrastive learning to hone in on these critical steps. This algorithm, which we call Contrastive Retrospection (ConSpec), can be added to any existing RL algorithm. ConSpec learns a set of prototypes for the critical steps in a task by a novel contrastive loss and delivers an intrinsic reward when the current state matches one of the prototypes. The prototypes in ConSpec provide two key benefits for credit assignment: (i) They enable rapid identification of all the critical steps. (ii) They do so in a readily interpretable manner, enabling out-of-distribution generalization when sensory features are altered. Distinct from other contemporary RL approaches to credit assignment, ConSpec takes advantage of the fact that it is easier to retrospectively identify the small set of steps that success is contingent upon (and ignoring other states) than it is to prospectively predict reward at every taken step. ConSpec greatly improves learning in a diverse set of RL tasks. The code is available at the link: https://github.com/sunchipsster1/ConSpec

        ----

        ## [1356] A Hierarchical Training Paradigm for Antibody Structure-sequence Co-design

        **Authors**: *Fang Wu, Stan Z. Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/636d57c09a5baacd83722639265802f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/636d57c09a5baacd83722639265802f6-Abstract-Conference.html)

        **Abstract**:

        Therapeutic antibodies are an essential and rapidly flourishing drug modality. The binding specificity between antibodies and antigens is decided by complementarity-determining regions (CDRs) at the tips of these Y-shaped proteins. In this paper, we propose a \textbf{h}ierarchical \textbf{t}raining \textbf{p}aradigm (HTP) for the antibody sequence-structure co-design. HTP consists of four levels of training stages, each corresponding to a specific protein modality within a particular protein domain. Through carefully crafted tasks in different stages, HTP seamlessly and effectively integrates geometric graph neural networks (GNNs) with large-scale protein language models to excavate evolutionary information from not only geometric structures but also vast antibody and non-antibody sequence databases, which determines ligand binding pose and strength. Empirical experiments show HTP sets the new state-of-the-art performance in the co-design problem as well as the fix-backbone design. Our research offers a hopeful path to unleash the potential of deep generative architectures and seeks to illuminate the way forward for the antibody sequence and structure co-design challenge.

        ----

        ## [1357] Differentiable Clustering with Perturbed Spanning Forests

        **Authors**: *Lawrence Stewart, Francis R. Bach, Felipe Llinares-López, Quentin Berthet*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html)

        **Abstract**:

        We introduce a differentiable clustering method based on stochastic perturbations of minimum-weight spanning forests. This allows us to include clustering in end-to-end trainable pipelines, with efficient gradients. We show that our method performs well even in difficult settings, such as data sets with high noise and challenging geometries. We also formulate an ad hoc loss to efficiently learn from partial clustering data using this operation. We demonstrate its performance on several data sets for supervised and semi-supervised tasks.

        ----

        ## [1358] Logarithmic-Regret Quantum Learning Algorithms for Zero-Sum Games

        **Authors**: *Minbo Gao, Zhengfeng Ji, Tongyang Li, Qisheng Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/637df18481a6aa74238bd2cafff94cb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/637df18481a6aa74238bd2cafff94cb9-Abstract-Conference.html)

        **Abstract**:

        We propose the first online quantum algorithm for zero-sum games with $\widetilde O(1)$ regret under the game setting. Moreover, our quantum algorithm computes an $\varepsilon$-approximate Nash equilibrium of an $m \times n$ matrix zero-sum game in quantum time $\widetilde O(\sqrt{m+n}/\varepsilon^{2.5})$. Our algorithm uses standard quantum inputs and generates classical outputs with succinct descriptions, facilitating end-to-end applications. Technically, our online quantum algorithm "quantizes" classical algorithms based on the optimistic multiplicative weight update method. At the heart of our algorithm is a fast quantum multi-sampling procedure for the Gibbs sampling problem, which may be of independent interest.

        ----

        ## [1359] Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network

        **Authors**: *Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, Yoshua Bengio*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)

        **Abstract**:

        Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized by neural networks. We show that our method, called JSP-GFN, offers an accurate approximation of the joint posterior, while comparing favorably against existing methods on both simulated and real data.

        ----

        ## [1360] DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

        **Authors**: *Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, Bo Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Generative Pre-trained Transformer (GPT) models have exhibited exciting progress in capabilities, capturing the interest of practitioners and the public alike. Yet, while the literature on the trustworthiness of GPT models remains limited, practitioners have proposed employing capable GPT models for sensitive applications to healthcare and finance – where mistakes can be costly. To this end, this work proposes a comprehensive trustworthiness evaluation for large language models with a focus on GPT-4 and GPT-3.5, considering diverse perspectives – including toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, robustness on adversarial demonstrations, privacy, machine ethics, and fairness. Based on our evaluations, we discover previously unpublished vulnerabilities to trustworthiness threats. For instance, we find that GPT models can be easily misled to generate toxic and biased outputs and leak private information in both training data and conversation history. We also find that although GPT-4 is usually more trustworthy than GPT-3.5 on standard benchmarks, GPT-4 is more vulnerable given jailbreaking system or user prompts, potentially due to the reason that GPT-4 follows the (misleading) instructions more precisely. Our work illustrates a comprehensive trustworthiness evaluation of GPT models and sheds light on the trustworthiness gaps. Our benchmark is publicly available at https://decodingtrust.github.io/.

        ----

        ## [1361] Three Towers: Flexible Contrastive Learning with Pretrained Image Models

        **Authors**: *Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63d4316315900a62e610e5c17bab900a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63d4316315900a62e610e5c17bab900a-Abstract-Conference.html)

        **Abstract**:

        We introduce Three Towers (3T), a flexible method to improve the contrastive learning of vision-language models by incorporating pretrained image classifiers. While contrastive models are usually trained from scratch, LiT (Zhai et al., 2022) has recently shown performance gains from using pretrained classifier embeddings. However, LiT directly replaces the image tower with the frozen embeddings, excluding any potential benefits from training the image tower contrastively. With 3T, we propose a more flexible strategy that allows the image tower to benefit from both pretrained embeddings and contrastive training. To achieve this, we introduce a third tower that contains the frozen pretrained embeddings, and we encourage alignment between this third tower and the main image-text towers. Empirically, 3T consistently improves over LiT and the CLIP-style from-scratch baseline for retrieval tasks. For classification, 3T reliably improves over the from-scratch baseline, and while it underperforms relative to LiT for JFT-pretrained models, it outperforms LiT for ImageNet-21k and Places365 pretraining.

        ----

        ## [1362] Practical and Asymptotically Exact Conditional Sampling in Diffusion Models

        **Authors**: *Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, John P. Cunningham*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have been successful on a range of conditional generation tasks including molecular design and text-to-image generation. However, these achievements have primarily depended on task-specific conditional training or error-prone heuristic approximations. Ideally, a conditional generation method should provide exact samples for a broad range of conditional distributions without requiring task-specific training. To this end, we introduce the Twisted Diffusion Sampler, or TDS. TDS is a sequential Monte Carlo (SMC) algorithm that targets the conditional distributions of diffusion models through simulating a set of weighted particles. The main idea is to use twisting, an SMC technique that enjoys good computational efficiency, to incorporate heuristic approximations without compromising asymptotic exactness. We first find in simulation and in conditional image generation tasks that TDS provides a computational statistical trade-off, yielding more accurate approximations with many particles but with empirical improvements over heuristics with as few as two particles. We then turn to motif-scaffolding, a core task in protein design, using a TDS extension to Riemannian diffusion models; on benchmark tasks, TDS allows flexible conditioning criteria and often outperforms the state-of-the-art, conditionally trained model. Code can be found in https://github.com/blt2114/twisteddiffusionsampler

        ----

        ## [1363] Actively Testing Your Model While It Learns: Realizing Label-Efficient Learning in Practice

        **Authors**: *Dayou Yu, Weishi Shi, Qi Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/63ef323523f3be8b58ed9277cc747485-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/63ef323523f3be8b58ed9277cc747485-Abstract-Conference.html)

        **Abstract**:

        In active learning (AL), we focus on reducing the data annotation cost from the model training perspective. However, "testing'', which often refers to the model evaluation process of using empirical risk to estimate the intractable true generalization risk, also requires data annotations. The annotation cost for "testing'' (model evaluation) is under-explored. Even in works that study active model evaluation or active testing (AT), the learning and testing ends are disconnected. In this paper, we propose a novel active testing while learning (ATL) framework that integrates active learning with active testing. ATL provides an unbiased sample-efficient estimation of the model risk during active learning. It leverages test samples annotated from different periods of a dynamic active learning process to achieve fair model evaluations based on a theoretically guaranteed optimal integration of different test samples. Periodic testing also enables effective early-stopping to further save the total annotation cost. ATL further integrates an "active feedback'' mechanism, which is inspired by human learning, where the teacher (active tester) provides immediate guidance given by the prior performance of the student (active learner). Our theoretical result reveals that active feedback maintains the label complexity of the integrated learning-testing objective, while improving the model's generalization capability. We study the realistic setting where we maximize the performance gain from choosing "testing'' samples for feedback without sacrificing the risk estimation accuracy. An agnostic-style analysis and empirical evaluations on real-world datasets demonstrate that the ATL framework can effectively improve the annotation efficiency of both active learning and evaluation tasks.

        ----

        ## [1364] MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing

        **Authors**: *Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, Yu Su*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Text-guided image editing is widely needed in daily life, ranging from personal use to professional applications such as Photoshop.However, existing methods are either zero-shot or trained on an automatically synthesized dataset, which contains a high volume of noise.Thus, they still require lots of manual tuning to produce desirable outcomes in practice.To address this issue, we introduce MagicBrush, the first large-scale, manually annotated dataset for instruction-guided real image editing that covers diverse scenarios: single-turn, multi-turn, mask-provided, and mask-free editing.MagicBrush comprises over 10K manually annotated triplets (source image, instruction, target image), which supports trainining large-scale text-guided image editing models.We fine-tune InstructPix2Pix on MagicBrush and show that the new model can produce much better images according to human evaluation.We further conduct extensive experiments to evaluate current image editing baselines from multiple dimensions including quantitative, qualitative, and human evaluations.The results reveal the challenging nature of our dataset and the gap between current baselines and real-world editing needs.

        ----

        ## [1365] Causal Interpretation of Self-Attention in Pre-Trained Transformers

        **Authors**: *Raanan Y. Rohekar, Yaniv Gurwicz, Shami Nisimov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/642a321fba8a0f03765318e629cb93ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/642a321fba8a0f03765318e629cb93ea-Abstract-Conference.html)

        **Abstract**:

        We propose a causal interpretation of self-attention in the Transformer neural network architecture. We interpret self-attention as a mechanism that estimates a structural equation model for a given input sequence of symbols (tokens). The structural equation model can be interpreted, in turn, as a causal structure over the input symbols under the specific context of the input sequence. Importantly, this interpretation remains valid in the presence of latent confounders. Following this interpretation, we estimate conditional independence relations between input symbols by calculating partial correlations between their corresponding representations in the deepest attention layer. This enables learning the causal structure over an input sequence using existing constraint-based algorithms. In this sense, existing pre-trained Transformers can be utilized for zero-shot causal-discovery. We demonstrate this method by providing causal explanations for the outcomes of Transformers in two tasks: sentiment classification (NLP) and recommendation.

        ----

        ## [1366] Parsel🦆: Algorithmic Reasoning with Language Models by Composing Decompositions

        **Authors**: *Eric Zelikman, Qian Huang, Gabriel Poesia, Noah D. Goodman, Nick Haber*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html)

        **Abstract**:

        Despite recent success in large language model (LLM) reasoning, LLMs struggle with hierarchical multi-step reasoning tasks like generating complex programs. For these tasks, humans often start with a high-level algorithmic design and implement each part gradually. We introduce Parsel, a framework enabling automatic implementation and validation of complex algorithms with code LLMs. With Parsel, we automatically decompose algorithmic tasks into hierarchical natural language function descriptions and then search over combinations of possible function implementations using tests. We show that Parsel can be used across domains requiring hierarchical reasoning, including program synthesis and robotic planning. We find that, using Parsel, LLMs solve more competition-level problems in the APPS dataset, resulting in pass rates over 75\% higher than prior results from directly sampling AlphaCode and Codex, while often using a smaller sample budget. Moreover, with automatically generated tests, we find that Parsel can improve the state-of-the-art pass@1 performance on HumanEval from 67\% to 85\%. We also find that LLM-generated robotic plans using Parsel are more than twice as likely to be considered accurate than directly generated plans. Lastly, we explore how Parsel addresses LLM limitations and discuss how Parsel may be useful for human programmers. We release our code at https://github.com/ezelikman/parsel.

        ----

        ## [1367] Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation

        **Authors**: *Yuan Wang, Naisong Luo, Tianzhu Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6447714b83edcbed61dbe10371dd7ae5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6447714b83edcbed61dbe10371dd7ae5-Abstract-Conference.html)

        **Abstract**:

        Few-shot segmentation (FSS) aims to segment objects of new categories given only a handful of annotated samples. Previous works focus their efforts on exploring the support information while paying less attention to the mining of the critical query branch. In this paper, we rethink the importance of support information and propose a new query-centric FSS model Adversarial Mining Transformer (AMFormer), which achieves accurate query image segmentation with only rough support guidance or even weak support labels. The proposed AMFormer enjoys several merits. First, we design an object mining transformer (G) that can achieve the expansion of incomplete region activated by support clue, and a detail mining transformer (D) to discriminate the detailed local difference between the expanded mask and the ground truth. Second, we propose to train G and D via an adversarial process, where G is optimized to generate more accurate masks approaching ground truth to fool D. We conduct extensive experiments on commonly used Pascal-5i and COCO-20i benchmarks and achieve state-of-the-art results across all settings. In addition, the decent performance with weak support labels in our query-centric paradigm may inspire the development of more general FSS models.

        ----

        ## [1368] Improving Language Plasticity via Pretraining with Active Forgetting

        **Authors**: *Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Ifeoluwa Adelani, Pontus Lars Erik Saito Stenetorp, Sebastian Riedel, Mikel Artetxe*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html)

        **Abstract**:

        Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a barrier to making their capabilities universally accessible. While prior work has shown it possible to address this issue by learning a new embedding layer for the new language, doing so is both data and compute inefficient. We propose to use an active forgetting mechanism during pretraining, as a simple way of creating PLMs that can quickly adapt to new languages. Concretely, by resetting the embedding layer every K updates during pretraining, we encourage the PLM to improve its ability of learning new embeddings within limited number of updates, similar to a meta-learning effect. Experiments with RoBERTa show that models pretrained with our forgetting mechanism not only demonstrate faster convergence during language adaptation, but also outperform standard ones in a low-data regime, particularly for languages that are distant from English. Code will be available at https://github.com/facebookresearch/language-model-plasticity.

        ----

        ## [1369] Stability of Random Forests and Coverage of Random-Forest Prediction Intervals

        **Authors**: *Yan Wang, Huaiqing Wu, Dan Nettleton*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6452474601429509f3035dc81c233226-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6452474601429509f3035dc81c233226-Abstract-Conference.html)

        **Abstract**:

        We establish stability of random forests under the mild condition that the squared response ($Y^2$) does not have a heavy tail. In particular, our analysis holds for the practical version of random forests that is implemented in popular packages like \texttt{randomForest} in \texttt{R}. Empirical results show that stability may persist even beyond our assumption and hold for heavy-tailed $Y^2$. Using the stability property, we prove a non-asymptotic lower bound for the coverage probability of prediction intervals constructed from the out-of-bag error of random forests. With another mild condition that is typically satisfied when $Y$ is continuous, we also establish a complementary upper bound, which can be similarly established for the jackknife prediction interval constructed from an arbitrary stable algorithm. We also discuss the asymptotic coverage probability under assumptions weaker than those considered in previous literature. Our work implies that random forests, with its stability property, is an effective machine learning method that can provide not only satisfactory point prediction but also justified interval prediction at almost no extra computational cost.

        ----

        ## [1370] Multi-Fidelity Multi-Armed Bandits Revisited

        **Authors**: *Xuchuang Wang, Qingyun Wu, Wei Chen, John C. S. Lui*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64602b87c31db70a3ef060f6c5d5b01d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64602b87c31db70a3ef060f6c5d5b01d-Abstract-Conference.html)

        **Abstract**:

        We study the multi-fidelity multi-armed bandit ($\texttt{MF-MAB}$), an extension of the canonical multi-armed bandit (MAB) problem.$\texttt{MF-MAB}$ allows each arm to be pulled with different costs (fidelities) and observation accuracy.We study both the best arm identification with fixed confidence ($\texttt{BAI}$) and the regret minimization objectives.For $\texttt{BAI}$, we present (a) a cost complexity lower bound, (b) an algorithmic framework with two alternative fidelity selection procedures,and (c) both procedures' cost complexity upper bounds.From both cost complexity bounds of $\texttt{MF-MAB}$,one can recover the standard sample complexity bounds of the classic (single-fidelity) MAB.For regret minimization of $\texttt{MF-MAB}$, we propose a new regret definition, prove its problem-independent regret lower bound $\Omega(K^{1/3}\Lambda^{2/3})$ and problem-dependent lower bound $\Omega(K\log \Lambda)$, where $K$ is the number of arms and $\Lambda$ is the decision budget in terms of cost, and devise an elimination-based algorithm whose worst-cost regret upper bound matches its corresponding lower bound up to some logarithmic terms and, whose problem-dependent bound matches its corresponding lower bound in terms of $\Lambda$.

        ----

        ## [1371] Augmentation-Aware Self-Supervision for Data-Efficient GAN Training

        **Authors**: *Liang Hou, Qi Cao, Yige Yuan, Songtao Zhao, Chongyang Ma, Siyuan Pan, Pengfei Wan, Zhongyuan Wang, Huawei Shen, Xueqi Cheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6464638c2472e4cae607f0c96a6fe774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6464638c2472e4cae607f0c96a6fe774-Abstract-Conference.html)

        **Abstract**:

        Training generative adversarial networks (GANs) with limited data is challenging because the discriminator is prone to overfitting. Previously proposed differentiable augmentation demonstrates improved data efficiency of training GANs. However, the augmentation implicitly introduces undesired invariance to augmentation for the discriminator since it ignores the change of semantics in the label space caused by data transformation, which may limit the representation learning ability of the discriminator and ultimately affect the generative modeling performance of the generator. To mitigate the negative impact of invariance while inheriting the benefits of data augmentation, we propose a novel augmentation-aware self-supervised discriminator that predicts the augmentation parameter of the augmented data. Particularly, the prediction targets of real data and generated data are required to be distinguished since they are different during training. We further encourage the generator to adversarially learn from the self-supervised discriminator by generating augmentation-predictable real and not fake data. This formulation connects the learning objective of the generator and the arithmetic $-$ harmonic mean divergence under certain assumptions. We compare our method with state-of-the-art (SOTA) methods using the class-conditional BigGAN and unconditional StyleGAN2 architectures on data-limited CIFAR-10, CIFAR-100, FFHQ, LSUN-Cat, and five low-shot datasets. Experimental results demonstrate significant improvements of our method over SOTA methods in training data-efficient GANs.

        ----

        ## [1372] Template-free Articulated Neural Point Clouds for Reposable View Synthesis

        **Authors**: *Lukas Uzolas, Elmar Eisemann, Petr Kellnhofer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html)

        **Abstract**:

        Dynamic Neural Radiance Fields (NeRFs) achieve remarkable visual quality when synthesizing novel views of time-evolving 3D scenes. However, the common reliance on backward deformation fields makes reanimation of the captured object poses challenging. Moreover, the state of the art dynamic models are often limited by low visual fidelity, long reconstruction time or specificity to narrow application domains.Â  In this paper, we present a novel method utilizing a point-based representation and Linear Blend Skinning (LBS) to jointly learn a Dynamic NeRF and an associated skeletal model from even sparse multi-view video. Our forward-warping approach achieves state-of-the-art visual fidelity when synthesizing novel views and poses while significantly reducing the necessary learning time when compared to existing work. We demonstrate the versatility of our representation on a variety of articulated objects from common datasets and obtain reposable 3D reconstructions without the need of object-specific skeletal templates.

        ----

        ## [1373] Demystifying the Optimal Performance of Multi-Class Classification

        **Authors**: *Minoh Jeong, Martina Cardone, Alex Dytso*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/647e122fc406573c51276692f20379b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/647e122fc406573c51276692f20379b5-Abstract-Conference.html)

        **Abstract**:

        Classification is a fundamental task in science and engineering on which machine learning methods have shown outstanding performances. However, it is challenging to determine whether such methods have achieved the Bayes error rate, that is, the lowest error rate attained by any classifier. This is mainly due to the fact that the Bayes error rate is not known in general and hence, effectively estimating it is paramount. Inspired by the work by Ishida et al. (2023), we propose an estimator for the Bayes error rate of supervised multi-class classification problems. We analyze several theoretical aspects of such estimator, including its consistency, unbiasedness, convergence rate, variance, and robustness. We also propose a denoising method that reduces the noise that potentially corrupts the data labels, and we improve the robustness of the proposed estimator to outliers by incorporating the median-of-means estimator. Our analysis demonstrates the consistency, asymptotic unbiasedness, convergence rate, and robustness of the proposed estimators. Finally, we validate the effectiveness of our theoretical results via experiments both on synthetic data under various noise settings and on real data.

        ----

        ## [1374] HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models

        **Authors**: *Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, Chng Eng Siong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Advancements in deep neural networks have allowed automatic speech recognition (ASR) systems to attain human parity on several publicly available clean speech datasets. However, even state-of-the-art ASR systems experience performance degradation when confronted with adverse conditions, as a well-trained acoustic model is sensitive to variations in the speech domain, e.g., background noise. Intuitively, humans address this issue by relying on their linguistic knowledge: the meaning of ambiguous spoken terms is usually inferred from contextual cues thereby reducing the dependency on the auditory system. Inspired by this observation, we introduce the first open-source benchmark to utilize external large language models (LLMs) for ASR error correction, where N-best decoding hypotheses provide informative elements for true transcription prediction. This approach is a paradigm shift from the traditional language model rescoring strategy that can only select one candidate hypothesis as output transcription. The proposed benchmark contains a novel dataset, "HyPoradise" (HP), encompassing more than 316,000 pairs of N-best hypotheses and corresponding accurate transcriptions across prevalent speech domains. Given this dataset, we examine three types of error correction techniques based on LLMs with varying amounts of labeled hypotheses-transcription pairs, which gains significant word error rate (WER) reduction. Experimental evidence demonstrates the proposed technique achieves a breakthrough by surpassing the upper bound of traditional re-ranking based methods. More surprisingly, LLM with reasonable prompt design can even correct those tokens that are missing in N-best list. We make our results publicly accessible for reproducible pipelines with released pre-trained models, thus providing a new paradigm for ASR error correction with LLMs.

        ----

        ## [1375] Structured Voronoi Sampling

        **Authors**: *Afra Amini, Li Du, Ryan Cotterell*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64ae05e3f1a88ebac7f9263b69f4e702-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64ae05e3f1a88ebac7f9263b69f4e702-Abstract-Conference.html)

        **Abstract**:

        Gradient-based sampling algorithms have demonstrated their effectiveness in text generation, especially in the context of controlled text generation. However, there exists a lack of theoretically grounded and principled approaches for this task. In this paper, we take an important step toward building a principled approach for sampling from language models with gradient-based methods. We use discrete distributions given by language models to define densities and develop an algorithm based on Hamiltonian Monte Carlo to sample from them. We name our gradient-based technique Structured Voronoi Sampling (SVS). In an experimental setup where the reference distribution is known, we show that the empirical distribution of SVS samples is closer to the reference distribution compared to alternative sampling schemes. Furthermore, in a controlled generation task, SVS is able to generate fluent and diverse samples while following the control targets significantly better than other methods.

        ----

        ## [1376] Stability and Generalization of the Decentralized Stochastic Gradient Descent Ascent Algorithm

        **Authors**: *Miaoxi Zhu, Li Shen, Bo Du, Dacheng Tao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html)

        **Abstract**:

        The growing size of available data has attracted increasing interest in solving minimax problems in a decentralized manner for various machine learning tasks. Previous theoretical research has primarily focused on the convergence rate and communication complexity of decentralized minimax algorithms, with little attention given to their generalization. In this paper, we investigate the primal-dual generalization bound of the decentralized stochastic gradient descent ascent (D-SGDA) algorithm using the approach of algorithmic stability under both convex-concave and nonconvex-nonconcave settings. Our theory refines the algorithmic stability in a decentralized manner and demonstrates that the decentralized structure does not destroy the stability and generalization of D-SGDA, implying that it can generalize as well as the vanilla SGDA in certain situations. Our results analyze the impact of different topologies on the generalization bound of the D-SGDA algorithm beyond trivial factors such as sample sizes, learning rates, and iterations. We also evaluate the optimization error and balance it with the generalization gap to obtain the optimal population risk of D-SGDA in the convex-concave setting. Additionally, we perform several numerical experiments which validate our theoretical findings.

        ----

        ## [1377] Hierarchical clustering with dot products recovers hidden tree structure

        **Authors**: *Annie Gray, Alexander Modell, Patrick Rubin-Delanchy, Nick Whiteley*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6521937507d78f327cd402401be73bf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6521937507d78f327cd402401be73bf2-Abstract-Conference.html)

        **Abstract**:

        In this paper we offer a new perspective on the well established agglomerative clustering algorithm, focusing on recovery of hierarchical structure. We recommend a simple variant of the standard algorithm, in which clusters are merged by maximum average dot product and not, for example, by minimum distance or within-cluster variance. We demonstrate that the tree output by this algorithm provides a bona fide estimate of generative hierarchical structure in data, under a generic probabilistic graphical model. The key technical innovations are to understand how hierarchical information in this model translates into tree geometry which can be recovered from data, and to characterise the benefits of simultaneously growing sample size and data dimension. We demonstrate superior tree recovery performance with real data over existing approaches such as UPGMA, Ward's method, and HDBSCAN.

        ----

        ## [1378] Latent Field Discovery in Interacting Dynamical Systems with Neural Fields

        **Authors**: *Miltiadis Kofinas, Erik J. Bekkers, Naveen Shankar Nagaraja, Efstratios Gavves*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6521bd47ebaa28228cd6c74cb85afb65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6521bd47ebaa28228cd6c74cb85afb65-Abstract-Conference.html)

        **Abstract**:

        Systems of interacting objects often evolve under the influence of underlying field effects that govern their dynamics, yet previous works have abstracted away from such effects, and assume that systems evolve in a vacuum. In this work, we focus on discovering these fields, and infer them from the observed dynamics alone, without directly observing them. We theorize the presence of latent force fields, and propose neural fields to learn them. Since the observed dynamics constitute the net effect of local object interactions and global field effects, recently popularized equivariant networks are inapplicable, as they fail to capture global information. To address this, we propose to disentangle local object interactions --which are SE(3) equivariant and depend on relative states-- from external global field effects --which depend on absolute states. We model the interactions with equivariant graph networks, and combine them with neural fields in a novel graph network that integrates field forces. Our experiments show that we can accurately discover the underlying fields in charged particles settings, traffic scenes, and gravitational n-body problems, and effectively use them to learn the system and forecast future trajectories.

        ----

        ## [1379] Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration

        **Authors**: *Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, Younggyo Seo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6530db249c161fe9254db2667453952c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6530db249c161fe9254db2667453952c-Abstract-Conference.html)

        **Abstract**:

        A promising technique for exploration is to maximize the entropy of visited state distribution, i.e., state entropy, by encouraging uniform coverage of visited state space. While it has been effective for an unsupervised setup, it tends to struggle in a supervised setup with a task reward, where an agent prefers to visit high-value states to exploit the task reward. Such a preference can cause an imbalance between the distributions of high-value states and low-value states, which biases exploration towards low-value state regions as a result of the state entropy increasing when the distribution becomes more uniform. This issue is exacerbated when high-value states are narrowly distributed within the state space, making it difficult for the agent to complete the tasks. In this paper, we present a novel exploration technique that maximizes the value-conditional state entropy, which separately estimates the state entropies that are conditioned on the value estimates of each state, then maximizes their average. By only considering the visited states with similar value estimates for computing the intrinsic bonus, our method prevents the distribution of low-value states from affecting exploration around high-value states, and vice versa. We demonstrate that the proposed alternative to the state entropy baseline significantly accelerates various reinforcement learning algorithms across a variety of tasks within MiniGrid, DeepMind Control Suite, and Meta-World benchmarks. Source code is available at https://sites.google.com/view/rl-vcse.

        ----

        ## [1380] Learning World Models with Identifiable Factorization

        **Authors**: *Yuren Liu, Biwei Huang, Zhengmao Zhu, Hong-Long Tian, Mingming Gong, Yang Yu, Kun Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65496a4902252d301cdf219339bfbf9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65496a4902252d301cdf219339bfbf9e-Abstract-Conference.html)

        **Abstract**:

        Extracting a stable and compact representation of the environment is crucial for efficient reinforcement learning in high-dimensional, noisy, and non-stationary environments.  Different categories of information coexist in such environments -- how to effectively extract and disentangle the information remains a challenging problem. In this paper, we propose IFactor, a general framework to model four distinct categories of latent state variables that capture various aspects of information within the RL system, based on their interactions with actions and rewards. Our analysis establishes block-wise identifiability of these latent variables, which not only provides a stable and compact representation but also discloses that all reward-relevant factors are significant for policy learning. We further present a practical approach to learning the world model with identifiable blocks, ensuring the removal of redundancies but retaining minimal and sufficient information for policy optimization. Experiments in synthetic worlds demonstrate that our method accurately identifies the ground-truth latent variables, substantiating our theoretical findings. Moreover, experiments in variants of the DeepMind Control Suite and RoboDesk showcase the superior performance of our approach over baselines.

        ----

        ## [1381] Online Map Vectorization for Autonomous Driving: A Rasterization Perspective

        **Authors**: *Gongjie Zhang, Jiahao Lin, Shuang Wu, Yilin Song, Zhipeng Luo, Yang Xue, Shijian Lu, Zuoguan Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/654f61ecd998c9095d30d42c03b832aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/654f61ecd998c9095d30d42c03b832aa-Abstract-Conference.html)

        **Abstract**:

        High-definition (HD) vectorized map is essential for autonomous driving, providing detailed and precise environmental information for advanced perception and planning. However, current map vectorization methods often exhibit deviations, and the existing evaluation metric for map vectorization lacks sufficient sensitivity to detect these deviations. To address these limitations, we propose integrating the philosophy of rasterization into map vectorization. Specifically, we introduce a new rasterization-based evaluation metric, which has superior sensitivity and is better suited to real-world autonomous driving scenarios. Furthermore, we propose MapVR (Map Vectorization via Rasterization), a novel framework that applies differentiable rasterization to vectorized outputs and then performs precise and geometry-aware supervision on rasterized HD maps. Notably, MapVR designs tailored rasterization strategies for various geometric shapes, enabling effective adaptation to a wide range of map elements. Experiments show that incorporating rasterization into map vectorization greatly enhances performance with no extra computational cost during inference, leading to more accurate map perception and ultimately promoting safer autonomous driving. Codes are available at https://github.com/ZhangGongjie/MapVR. A standalone map vectorization evaluation toolkit is available at https://github.com/jiahaoLjh/MapVectorizationEvalToolkit.

        ----

        ## [1382] NAP: Neural 3D Articulated Object Prior

        **Authors**: *Jiahui Lei, Congyue Deng, William B. Shen, Leonidas J. Guibas, Kostas Daniilidis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/655846cc914cb7ff977a1ada40866441-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/655846cc914cb7ff977a1ada40866441-Abstract-Conference.html)

        **Abstract**:

        We propose Neural 3D Articulated object Prior (NAP), the first 3D deep generative model to synthesize 3D articulated object models. Despite the extensive research on generating 3D static objects, compositions, or scenes, there are hardly any approaches on capturing the distribution of articulated objects, a common object category for human and robot interaction. To generate articulated objects, we first design a novel articulation tree/graph parameterization and then apply a diffusion-denoising probabilistic model over this representation where articulated objects can be generated via denoising from random complete graphs. In order to capture both the geometry and the motion structure whose distribution will affect each other, we design a graph denoising network for learning the reverse diffusion process. We propose a novel distance that adapts widely used 3D generation metrics to our novel task to evaluate generation quality. Experiments demonstrate our high performance in articulated object generation as well as its applications on conditioned generation, including Part2Motion, PartNet-Imagination, Motion2Part, and GAPart2Object.

        ----

        ## [1383] Compressed Video Prompt Tuning

        **Authors**: *Bing Li, Jiaxin Chen, Xiuguo Bao, Di Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/656678aa961a99a6a3d59bfbf88daf77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/656678aa961a99a6a3d59bfbf88daf77-Abstract-Conference.html)

        **Abstract**:

        Compressed videos offer a compelling alternative to raw videos, showing the possibility to significantly reduce the on-line computational and storage cost. However, current approaches to compressed video processing generally follow the resource-consuming pre-training and fine-tuning paradigm, which does not fully take advantage of such properties, making them not favorable enough for widespread applications. Inspired by recent successes of prompt tuning techniques in computer vision, this paper presents the first attempt to build a prompt based representation learning framework, which enables effective and efficient adaptation of pre-trained raw video models to compressed video understanding tasks. To this end, we propose a novel prompt tuning approach, namely Compressed Video Prompt Tuning (CVPT), emphatically dealing with the challenging issue caused by the inconsistency between pre-training and downstream data modalities. Specifically, CVPT replaces the learnable prompts with compressed modalities (\emph{e.g.} Motion Vectors and Residuals) by re-parameterizing them into conditional prompts followed by layer-wise refinement. The conditional prompts exhibit improved adaptability and generalizability to instances compared to conventional individual learnable ones, and the Residual prompts enhance the noisy motion cues in the Motion Vector prompts for further fusion with the visual cues from I-frames. Additionally, we design Selective Cross-modal Complementary Prompt (SCCP) blocks. After inserting them into the backbone, SCCP blocks leverage semantic relations across diverse levels and modalities to improve cross-modal interactions between prompts and input flows. Extensive evaluations on HMDB-51, UCF-101 and Something-Something v2 demonstrate that CVPT remarkably outperforms the state-of-the-art counterparts, delivering a much better balance between accuracy and efficiency.

        ----

        ## [1384] Sampling from Structured Log-Concave Distributions via a Soft-Threshold Dikin Walk

        **Authors**: *Oren Mangoubi, Nisheeth K. Vishnoi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/656faa09eb6e82dd86de9a417111c3b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/656faa09eb6e82dd86de9a417111c3b0-Abstract-Conference.html)

        **Abstract**:

        Given a Lipschitz or smooth convex function $f:K \to \mathbb{R}^d$ for a bounded polytope $K:=${ $\theta \in \mathbb{R}^d: A\theta \leq b$}, where $A\in  \mathbb{R}^{m\times d}$ and $b \in \mathbb{R}^m$, we consider the problem of sampling from the log-concave distribution $\pi(\theta) \propto e^{-f(\theta)}$ constrained to $K$. Interest in this problem derives from its applications to  Bayesian inference and differential privacy. We present  a generalization of the Dikin walk  to this setting that requires at most $O((md + d L^2 R^2) \times md^{\omega-1} \log(\frac{w}{\delta}))$ arithmetic operations to sample from $\pi$ within error $\delta>0$ in the total variation distance from a $w$-warm start. Here $L$ is the Lipschitz constant of $f$, $K$ is contained in a ball of radius $R$ and contains a ball of smaller radius $r$, and $\omega \approx 2.37$ is the matrix-multiplication constant. This improves on the running time of prior works for a range of structured settings important for the aforementioned inference and privacy applications. Technically, we depart from previous Dikin walks by adding a soft-threshold regularizer derived from the Lipschitz or smoothness properties of $f$  to a barrier function for $K$ that allows our version of the Dikin walk to propose updates  that have a high Metropolis acceptance ratio for $f$, while at the same time remaining inside the polytope $K$.

        ----

        ## [1385] Implicit Regularization in Over-Parameterized Support Vector Machine

        **Authors**: *Yang Sui, Xin He, Yang Bai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/659e07806dc17bd69d0d9aed47f85e7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/659e07806dc17bd69d0d9aed47f85e7c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we design a regularization-free algorithm for high-dimensional support vector machines (SVMs) by integrating over-parameterization with Nesterov's smoothing method, and provide theoretical guarantees for the induced implicit regularization phenomenon. In particular, we construct an over-parameterized hinge loss function and estimate the true parameters by leveraging regularization-free gradient descent on this loss function. The utilization of Nesterov's method enhances the computational efficiency of our algorithm, especially in terms of determining the stopping criterion and reducing computational complexity. With appropriate choices of initialization, step size, and smoothness parameter, we demonstrate that unregularized gradient descent achieves a near-oracle statistical convergence rate. Additionally, we verify our theoretical findings through a variety of numerical experiments and compare the proposed method with explicit regularization. Our results illustrate the advantages of employing implicit regularization via gradient descent in conjunction with over-parameterization in sparse SVMs.

        ----

        ## [1386] Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

        **Authors**: *Zirui Zhao, Wee Sun Lee, David Hsu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html)

        **Abstract**:

        Large-scale task planning is a major challenge.  Recent work exploits large  language models (LLMs) directly as a policy and shows surprisingly  interesting results.  This paper shows that LLMs provide a  commonsense model of the world in addition to a policy that acts on it.  The world model and the policy can be combined in a search  algorithm, such as Monte Carlo Tree Search (MCTS), to scale up task  planning.  In our new LLM-MCTS algorithm, the LLM-induced world model  provides a commonsense prior belief for MCTS to achieve effective reasoning;  the LLM-induced policy acts as a heuristic to guide the search, vastly  improving search efficiency. Experiments show that LLM-MCTS outperforms  both MCTS alone and policies induced by LLMs (GPT2 and GPT3.5) by a wide  margin, for complex, novel tasks.   Further experiments and analyses on multiple tasks --  multiplication, travel planning, object rearrangement --  suggest minimum description length (MDL)  as a general guiding principle: if the  description length of the world model is substantially smaller than that of  the  policy, using LLM as a world model for model-based planning is likely better  than using LLM solely as a policy.

        ----

        ## [1387] Uncovering Meanings of Embeddings via Partial Orthogonality

        **Authors**: *Yibo Jiang, Bryon Aragam, Victor Veitch*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html)

        **Abstract**:

        Machine learning tools often rely on embedding text as vectors of real numbers.In this paper, we study how the semantic structure of language is encoded in the algebraic structure of such embeddings.Specifically, we look at a notion of  "semantic independence" capturing the idea that, e.g., "eggplant" and "tomato" are independent given "vegetable". Although such examples are intuitive, it is difficult to formalize such a notion of semantic independence. The key observation here is that any sensible formalization should obey a set of so-called independence axioms, and thus any algebraic encoding of this structure should also obey these axioms. This leads us naturally to use partial orthogonality as the relevant algebraic structure. We develop theory and methods that allow us to demonstrate that partial orthogonality does indeed capture semantic independence.Complementary to this, we also introduce the concept of independence preserving embeddings where embeddings preserve the conditional independence structures of a distribution, and we prove the existence of such embeddings and approximations to them.

        ----

        ## [1388] Federated Learning with Bilateral Curation for Partially Class-Disjoint Data

        **Authors**: *Ziqing Fan, Ruipeng Zhang, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html)

        **Abstract**:

        Partially class-disjoint data (PCDD), a common yet under-explored data formation where each client contributes a part of classes (instead of all classes) of samples, severely challenges the performance of federated algorithms. Without full classes, the local objective will contradict the global objective, yielding the angle collapse problem for locally missing classes and the space waste problem for locally existing classes. As far as we know, none of the existing methods can intrinsically mitigate PCDD challenges to achieve holistic improvement in the bilateral views (both global view and local view) of federated learning. To address this dilemma, we are inspired by the strong generalization of simplex Equiangular Tight Frame (ETF) on the imbalanced data, and propose a novel approach called FedGELA where the classifier is globally fixed as a simplex ETF while locally adapted to the personal distributions. Globally, FedGELA provides fair and equal discrimination for all classes and avoids inaccurate updates of the classifier, while locally it utilizes the space of locally missing classes for locally existing classes. We conduct extensive experiments on a range of datasets to demonstrate that our FedGELA achieves promising performance (averaged improvement of 3.9% to FedAvg and 1.5% to best baselines) and provide both local and global convergence guarantees.

        ----

        ## [1389] Partial Counterfactual Identification of Continuous Outcomes with a Curvature Sensitivity Model

        **Authors**: *Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html)

        **Abstract**:

        Counterfactual inference aims to answer retrospective "what if" questions and thus belongs to the most fine-grained type of inference in Pearl's causality ladder. Existing methods for counterfactual inference with continuous outcomes aim at point identification and thus make strong and unnatural assumptions about the underlying structural causal model. In this paper, we relax these assumptions and aim at partial counterfactual identification of continuous outcomes, i.e., when the counterfactual query resides in an ignorance interval with informative bounds. We prove that, in general, the ignorance interval of the counterfactual queries has non-informative bounds, already when functions of structural causal models are continuously differentiable. As a remedy, we propose a novel sensitivity model called Curvature Sensitivity Model. This allows us to obtain informative bounds by bounding the curvature of level sets of the functions. We further show that existing point counterfactual identification methods are special cases of our Curvature Sensitivity Model when the bound of the curvature is set to zero. We then propose an implementation of our Curvature Sensitivity Model in the form of a novel deep generative model, which we call Augmented Pseudo-Invertible Decoder. Our implementation employs (i) residual normalizing flows with (ii) variational augmentations. We empirically demonstrate the effectiveness of our Augmented Pseudo-Invertible Decoder. To the best of our knowledge, ours is the first partial identification model for Markovian structural causal models with continuous outcomes.

        ----

        ## [1390] Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies

        **Authors**: *Wayne Soo, Vishwa Goudar, Xiao-Jing Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html)

        **Abstract**:

        Training recurrent neural networks (RNNs) has become a go-to approach for generating and evaluating mechanistic neural hypotheses for cognition. The ease and efficiency of training RNNs with backpropagation through time and the availability of robustly supported deep learning libraries has made RNN modeling more approachable and accessible to neuroscience. Yet, a major technical hindrance remains. Cognitive processes such as working memory and decision making involve neural population dynamics over a long period of time within a behavioral trial and across trials. It is difficult to train RNNs to accomplish tasks where neural representations and dynamics have long temporal dependencies without gating mechanisms such as LSTMs or GRUs which currently lack experimental support and prohibit direct comparison between RNNs and biological neural circuits. We tackled this problem based on the idea of specialized skip-connections through time to support the emergence of task-relevant dynamics, and subsequently reinstitute biological plausibility by reverting to the original architecture. We show that this approach enables RNNs to successfully learn cognitive tasks that prove impractical if not impossible to learn using conventional methods. Over numerous tasks considered here, we achieve less training steps and shorter wall-clock times, particularly in tasks that require learning long-term dependencies via temporal integration over long timescales or maintaining a memory of past events in hidden-states. Our methods expand the range of experimental tasks that biologically plausible RNN models can learn, thereby supporting the development of theory for the emergent neural mechanisms of computations involving long-term dependencies.

        ----

        ## [1391] Diffusion Probabilistic Models for Structured Node Classification

        **Authors**: *Hyosoon Jang, Seonghyun Park, Sangwoo Mo, Sungsoo Ahn*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html)

        **Abstract**:

        This paper studies structured node classification on graphs, where the predictions should consider dependencies between the node labels. In particular, we focus on solving the problem for partially labeled graphs where it is essential to incorporate the information in the known label for predicting the unknown labels. To address this issue, we propose a novel framework leveraging the diffusion probabilistic model for structured node classification (DPM-SNC). At the heart of our framework is the extraordinary capability of DPM-SNC to (a) learn a joint distribution over the labels with an expressive reverse diffusion process and (b) make predictions conditioned on the known labels utilizing manifold-constrained sampling. Since the DPMs lack training algorithms for partially labeled data, we design a novel training algorithm to apply DPMs, maximizing a new variational lower bound. We also theoretically analyze how DPMs benefit node classification by enhancing the expressive power of GNNs based on proposing AGG-WL, which is strictly more powerful than the classic 1-WL test. We extensively verify the superiority of our DPM-SNC in diverse scenarios, which include not only the transductive setting on partially labeled graphs but also the inductive setting and unlabeled graphs.

        ----

        ## [1392] Non-stationary Experimental Design under Linear Trends

        **Authors**: *David Simchi-Levi, Chonghuan Wang, Zeyu Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65e837e76a5308df3d5544aab6196e21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65e837e76a5308df3d5544aab6196e21-Abstract-Conference.html)

        **Abstract**:

        Experimentation has been critical and increasingly popular  across various domains, such as clinical trials and online platforms, due to its widely recognized benefits. One of the primary objectives of classical experiments is to estimate the average treatment effect (ATE) to inform future decision-making. However, in healthcare and many other settings, treatment effects may be non-stationary, meaning that they can change over time, rendering the traditional experimental design inadequate and the classical static ATE uninformative. In this work, we address the problem of non-stationary experimental design under linear trends by considering two objectives: estimating the dynamic treatment effect and minimizing welfare loss within the experiment. We propose an efficient design that can be customized for optimal estimation error rate, optimal regret rate, or the Pareto optimal trade-off between the two objectives. We establish information-theoretical lower bounds that highlight the inherent challenge in estimating dynamic treatment effects and minimizing welfare loss, and also statistically reveal the fundamental trade-off between them.

        ----

        ## [1393] EICIL: Joint Excitatory Inhibitory Cycle Iteration Learning for Deep Spiking Neural Networks

        **Authors**: *Zihang Shao, Xuanye Fang, Yaxin Li, Chaoran Feng, Jiangrong Shen, Qi Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65e876f6a98c6799d0b3145966dd73e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65e876f6a98c6799d0b3145966dd73e2-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks (SNNs) have undergone continuous development and extensive study for decades, leading to increased biological plausibility and optimal energy efficiency. However, traditional training methods for deep SNNs have some limitations, as they rely on strategies such as pre-training and fine-tuning, indirect coding and reconstruction, and approximate gradients. These strategies lack a complete training model and require gradient approximation. To overcome these limitations, we propose a novel learning method named Joint Excitatory Inhibitory Cycle Iteration learning for Deep Spiking Neural Networks (EICIL) that integrates both excitatory and inhibitory behaviors inspired by the signal transmission of biological neurons.By organically embedding these two behavior patterns into one framework, the proposed EICIL significantly improves the bio-mimicry and adaptability of spiking neuron models, as well as expands the representation space of spiking neurons. Extensive experiments based on EICIL and traditional learning methods demonstrate that EICIL outperforms traditional methods on various datasets, such as CIFAR10 and CIFAR100, revealing the crucial role of the learning approach that integrates both behaviors during training.

        ----

        ## [1394] Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency

        **Authors**: *Owen Queen, Tom Hartvigsen, Teddy Koker, Huan He, Theodoros Tsiligkaridis, Marinka Zitnik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/65ea878cb90b440e8b4cd34fe0959914-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/65ea878cb90b440e8b4cd34fe0959914-Abstract-Conference.html)

        **Abstract**:

        Interpreting time series models is uniquely challenging because it requires identifying both the location of time series signals that drive model predictions and their matching to an interpretable temporal pattern. While explainers from other modalities can be applied to time series, their inductive biases do not transfer well to the inherently challenging interpretation of time series. We present TimeX, a time series consistency model for training explainers. TimeX trains an interpretable surrogate to mimic the behavior of a pretrained time series model. It addresses the issue of model faithfulness by introducing model behavior consistency, a novel formulation that preserves relations in the latent space induced by the pretrained model with relations in the latent space induced by TimeX. TimeX provides discrete attribution maps and, unlike existing interpretability methods, it learns a latent space of explanations that can be used in various ways, such as to provide landmarks to visually aggregate similar explanations and easily recognize temporal patterns. We evaluate TimeX on eight synthetic and real-world datasets and compare its performance against state-of-the-art interpretability methods. We also conduct case studies using physiological time series. Quantitative evaluations demonstrate that TimeX achieves the highest or second-highest performance in every metric compared to baselines across all datasets. Through case studies, we show that the novel components of TimeX show potential for training faithful, interpretable models that capture the behavior of pretrained time series models.

        ----

        ## [1395] NeuroEvoBench: Benchmarking Evolutionary Optimizers for Deep Learning Applications

        **Authors**: *Robert Tjarko Lange, Yujin Tang, Yingtao Tian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/660ba7851661638c559df47743c69e40-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/660ba7851661638c559df47743c69e40-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recently, the Deep Learning community has become interested in evolutionary optimization (EO) as a means to address hard optimization problems, e.g. meta-learning through long inner loop unrolls or optimizing non-differentiable operators. One core reason for this trend has been the recent innovation in hardware acceleration and compatible software -- making distributed population evaluations much easier than before. Unlike for gradient descent-based methods though, there is a lack of hyperparameter understanding and best practices for EO â€“ arguably due to severely less `graduate student descent' and benchmarking being performed for EO methods. Additionally, classical benchmarks from the evolutionary community provide few practical insights for Deep Learning applications. This poses challenges for newcomers to hardware-accelerated EO and hinders significant adoption. Hence, we establish a new benchmark of EO methods (NEB) tailored toward Deep Learning applications and exhaustively evaluate traditional and meta-learned EO. We investigate core scientific questions including resource allocation, fitness shaping, normalization, regularization & scalability of EO. The benchmark is open-sourced at https://github.com/neuroevobench/neuroevobench under Apache-2.0 license.

        ----

        ## [1396] HyTrel: Hypergraph-enhanced Tabular Data Representation Learning

        **Authors**: *Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, George Karypis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html)

        **Abstract**:

        Language models pretrained on large collections of tabular data have demonstrated their effectiveness in several downstream tasks.However, many of these models do not take into account the row/column permutation invariances, hierarchical structure, etc. that exist in tabular data. To alleviate these limitations, we propose HyTrel, a tabular language model, that captures the permutation invariances and three more structural properties of tabular data by using hypergraphs--where the table cells make up the nodes and the cells occurring jointly together in each row, column, and the entire table are used to form three different types of hyperedges. We show thatHyTrel is maximally invariant under certain conditions for tabular data, i.e., two tables obtain the same representations via HyTreliff the two tables are identical up to permutation. Our empirical results demonstrate that HyTrel consistently outperforms other competitive baselines on four downstream tasks with minimal pretraining, illustrating the advantages of incorporating inductive biases associated with tabular data into the representations. Finally, our qualitative analyses showcase that HyTrel can assimilate the table structure to generate robust representations for the cells, rows, columns, and the entire table.

        ----

        ## [1397] PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance

        **Authors**: *Peiqing Yang, Shangchen Zhou, Qingyi Tao, Chen Change Loy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661c37f3b098bdee53fd7d9c4ef6964a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661c37f3b098bdee53fd7d9c4ef6964a-Abstract-Conference.html)

        **Abstract**:

        Exploiting pre-trained diffusion models for restoration has recently become a favored alternative to the traditional task-specific training approach. Previous works have achieved noteworthy success by limiting the solution space using explicit degradation models. However, these methods often fall short when faced with complex degradations as they generally cannot be precisely modeled. In this paper, we introduce $\textit{partial guidance}$, a fresh perspective that is more adaptable to real-world degradations compared to existing works. Rather than specifically defining the degradation process, our approach models the desired properties, such as image structure and color statistics of high-quality images, and applies this guidance during the reverse diffusion process. These properties are readily available and make no assumptions about the degradation process. When combined with a diffusion prior, this partial guidance can deliver appealing results across a range of restoration tasks. Additionally, our method can be extended to handle composite tasks by consolidating multiple high-quality image properties, achieved by integrating the guidance from respective tasks. Experimental results demonstrate that our method not only outperforms existing diffusion-prior-based approaches but also competes favorably with task-specific models.

        ----

        ## [1398] Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP

        **Authors**: *Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html)

        **Abstract**:

        Open-vocabulary segmentation is a challenging task requiring segmenting and recognizing objects from an open set of categories in diverse environments. One way to address this challenge is to leverage multi-modal models, such as CLIP, to provide image and text features in a shared embedding space, which effectively bridges the gap between closed-vocabulary and open-vocabulary recognition.Hence, existing methods often adopt a two-stage framework to tackle the problem, where the inputs first go through a mask generator and then through the CLIP model along with the predicted masks. This process involves extracting features from raw images multiple times, which can be ineffective and inefficient. By contrast, we propose to build everything into a single-stage framework using a shared Frozen Convolutional CLIP backbone, which not only significantly simplifies the current two-stage pipeline, but also remarkably yields a better accuracy-cost trade-off. The resulting single-stage system, called FC-CLIP, benefits from the following observations: the frozen CLIP backbone maintains the ability of open-vocabulary classification and can also serve as a strong mask generator, and the convolutional CLIP generalizes well to a larger input resolution than the one used during contrastive image-text pretraining. Surprisingly, FC-CLIP advances state-of-the-art results on various benchmarks, while running practically fast. Specifically, when training on COCO panoptic data only and testing in a zero-shot manner, FC-CLIP achieve 26.8 PQ, 16.8 AP, and 34.1 mIoU on ADE20K, 18.2 PQ, 27.9 mIoU on Mapillary Vistas, 44.0 PQ, 26.8 AP, 56.2 mIoU on Cityscapes, outperforming the prior art under the same setting by +4.2 PQ, +2.4 AP, +4.2 mIoU on ADE20K, +4.0 PQ on Mapillary Vistas and +20.1 PQ on Cityscapes, respectively. Additionally, the training and testing time of FC-CLIP is 7.5x and 6.6x significantly faster than the same prior art, while using 5.9x fewer total model parameters. Meanwhile, FC-CLIP also sets a new state-of-the-art performance across various open-vocabulary semantic segmentation datasets. Code and models are available at https://github.com/bytedance/fc-clip

        ----

        ## [1399] CLIP-OGD: An Experimental Design for Adaptive Neyman Allocation in Sequential Experiments

        **Authors**: *Jessica Dai, Paula Gradu, Christopher Harshaw*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/661d4fda173120a2f119e0319e6bcf97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/661d4fda173120a2f119e0319e6bcf97-Abstract-Conference.html)

        **Abstract**:

        From clinical development of cancer therapies to investigations into partisan bias, adaptive sequential designs have become increasingly popular method for causal inference, as they offer the possibility of improved precision over their non-adaptive counterparts. However, even in simple settings (e.g. two treatments) the extent to which adaptive designs can improve precision is not sufficiently well understood. In this work, we study the problem of Adaptive Neyman Allocation in a design-based potential outcomes framework, where the experimenter seeks to construct an adaptive design which is nearly as efficient as the optimal (but infeasible) non-adaptive Neyman design, which has access to all potential outcomes. Motivated by connections to online optimization, we propose Neyman Ratio and Neyman Regret as two (equivalent) performance measures of adaptive designs for this problem. We present Clip-OGD, an adaptive design which achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ expected Neyman regret and thereby recovers the optimal Neyman variance in large samples. Finally, we construct a conservative variance estimator which facilitates the development of asymptotically valid confidence intervals. To complement our theoretical results, we conduct simulations using data from a microeconomic experiment.

        ----

        

[Go to the previous page](NIPS-2023-list06.md)

[Go to the next page](NIPS-2023-list08.md)

[Go to the catalog section](README.md)