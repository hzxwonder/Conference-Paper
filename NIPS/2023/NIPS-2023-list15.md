## [0] Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context

**Authors**: *Lakshya A Agrawal, Aditya Kanade, Navin Goyal, Shuvendu K. Lahiri, Sriram K. Rajamani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html)

**Abstract**:

Language models of code (LMs) work well when the surrounding code provides sufficient context. This is not true when it becomes necessary to use types, functionality or APIs defined elsewhere in the repository or a linked library, especially those not seen during training. LMs suffer from limited awareness of such global context and end up hallucinating.Integrated development environments (IDEs) assist developers in understanding repository context using static analysis.  We extend this assistance, enjoyed by developers, to LMs. We propose monitor-guided decoding (MGD) where a monitor uses static analysis to guide the decoding. We construct a repository-level dataset PragmaticCode for method-completion in Java and evaluate MGD on it. On models of varying parameter scale, by monitoring for type-consistent object dereferences, MGD consistently improves compilation rates and agreement with ground truth. Further, LMs with fewer parameters, when augmented with MGD, can outperform larger LMs. With MGD, SantaCoder-1.1B achieves better compilation rate and next-identifier match than the much larger text-davinci-003 model.We also conduct a generalizability study to evaluate the ability of MGD to generalize to multiple programming languages (Java, C# and Rust), coding scenarios (e.g., correct number of arguments to method calls), and to enforce richer semantic constraints (e.g., stateful API protocols). Our data and implementation are available at https://github.com/microsoft/monitors4codegen.

----

## [0] Towards Federated Foundation Models: Scalable Dataset Pipelines for Group-Structured Learning

**Authors**: *Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, Zachary Garrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/662bb9c4dcc96aeaac8e7cd3fc6a0add-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/662bb9c4dcc96aeaac8e7cd3fc6a0add-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Dataset Grouper, a library to create large-scale group-structured (e.g., federated) datasets, enabling federated learning simulation at the scale of foundation models. This library facilitates the creation of group-structured versions of existing datasets based on user-specified partitions, and directly leads to a variety of useful heterogeneous datasets that can be plugged into existing software frameworks. Dataset Grouper offers three key advantages. First, it scales to settings where even a single group's dataset is too large to fit in memory. Second, it provides flexibility, both in choosing the base (non-partitioned) dataset and in defining partitions. Finally, it is framework-agnostic. We empirically demonstrate that Dataset Grouper enables large-scale federated language modeling simulations on datasets that are orders of magnitude larger than in previous work, allowing for federated training of language models with hundreds of millions, and even billions, of parameters. Our experimental results show that algorithms like FedAvg operate more as meta-learning methods than as empirical risk minimization methods at this scale, suggesting their utility in downstream personalization and task-specific adaptation. Dataset Grouper is available at https://github.com/google-research/dataset_grouper.

----

## [0] Sharp Spectral Rates for Koopman Operator Learning

**Authors**: *Vladimir Kostic, Karim Lounici, Pietro Novelli, Massimiliano Pontil*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/663bce02a0050c4a11f1eb8a7f1429d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/663bce02a0050c4a11f1eb8a7f1429d3-Abstract-Conference.html)

**Abstract**:

Non-linear dynamical systems can be handily described by the associated Koopman operator, whose action evolves every observable of the system forward in time. Learning the Koopman operator and its spectral decomposition from data is enabled by a number of algorithms. In this work we present for the first time non-asymptotic learning bounds for the Koopman eigenvalues and eigenfunctions. We focus on time-reversal-invariant stochastic dynamical systems, including the important example of Langevin dynamics. We analyze two popular estimators: Extended Dynamic Mode Decomposition (EDMD) and Reduced Rank Regression (RRR). Our results critically hinge on novel {minimax} estimation bounds for the operator norm error, that may be of independent interest. Our spectral learning bounds are driven by the simultaneous control of the operator norm error and a novel metric distortion functional of the estimated eigenfunctions. The bounds indicates that both EDMD and RRR have similar variance, but EDMD suffers from a larger bias which might be detrimental to its learning rate. Our results shed new light on the emergence of spurious eigenvalues, an issue which is well known empirically. Numerical experiments illustrate the implications of the bounds in practice.

----

## [0] Continual Learning for Instruction Following from Realtime Feedback

**Authors**: *Alane Suhr, Yoav Artzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html)

**Abstract**:

We propose and deploy an approach to  continually train an instruction-following agent from feedback provided by users during collaborative interactions. During interaction, human users instruct an agent using natural language, and provide realtime binary feedback as they observe the agent following their instructions. We design a contextual bandit learning approach, converting  user feedback to immediate reward. We evaluate through thousands of human-agent interactions, demonstrating 15.4% absolute improvement in instruction execution accuracy over time. We also show our approach is robust to several design variations, and that the feedback signal is roughly equivalent to the learning signal of supervised demonstration data.

----

## [0] Embracing the chaos: analysis and diagnosis of numerical instability in variational flows

**Authors**: *Zuheng Xu, Trevor Campbell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html)

**Abstract**:

In this paper, we investigate the impact of numerical instability on the reliability of sampling, density evaluation, and evidence lower bound (ELBO) estimation in variational flows. We first empirically demonstrate that common flows can exhibit a catastrophic accumulation of error: the numerical flow map deviates significantly from the exact map---which affects sampling---and the numerical inverse flow map does not accurately recover the initial input---which affects density and ELBO computations. Surprisingly though, we find that results produced by flows are often accurate enough for applications despite the presence of serious numerical instability. In this work, we treat variational flows as chaotic dynamical systems, and leverage shadowing theory to elucidate this behavior via theoretical guarantees on the error of sampling, density evaluation, and ELBO estimation. Finally, we develop and empirically test a diagnostic procedure that can be used to validate results produced by numerically unstable flows in practice.

----

## [0] Masked Two-channel Decoupling Framework for Incomplete Multi-view Weak Multi-label Learning

**Authors**: *Chengliang Liu, Jie Wen, Yabo Liu, Chao Huang, Zhihao Wu, Xiaoling Luo, Yong Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66772e6aa61e54ae16443ae1d78a7319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66772e6aa61e54ae16443ae1d78a7319-Abstract-Conference.html)

**Abstract**:

Multi-view learning has become a popular research topic in recent years, but research on the cross-application of classic multi-label classification and multi-view learning is still in its early stages. In this paper, we focus on the complex yet highly realistic task of incomplete multi-view weak multi-label learning and propose a masked two-channel decoupling framework based on deep neural networks to solve this problem. The core innovation of our method lies in decoupling the single-channel view-level representation, which is common in deep multi-view learning methods, into a shared representation and a view-proprietary representation. We also design a cross-channel contrastive loss to enhance the semantic property of the two channels. Additionally, we exploit supervised information to design a label-guided graph regularization loss, helping the extracted embedding features preserve the geometric structure among samples. Inspired by the success of masking mechanisms in image and text analysis, we develop a random fragment masking strategy for vector features to improve the learning ability of encoders. Finally, it is important to emphasize that our model is fully adaptable to arbitrary view and label absences while also performing well on the ideal full data. We have conducted sufficient and convincing experiments to confirm the effectiveness and advancement of our model.

----

## [0] Exponential Lower Bounds for Fictitious Play in Potential Games

**Authors**: *Ioannis Panageas, Nikolas Patris, Stratis Skoulakis, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66820ab16b817d8a6b00d60b3d24b83a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66820ab16b817d8a6b00d60b3d24b83a-Abstract-Conference.html)

**Abstract**:

Fictitious Play (FP) is a simple and natural dynamic for repeated play with many applications in game theory and multi-agent reinforcement learning. It was introduced by Brown and its convergence properties for two-player zero-sum games was established later by Robinson. Potential games [Monderer and Shapley 1996] is another class of games which exhibit the FP property [Monderer and Shapley 1996], i.e., FP dynamics converges to a Nash equilibrium if all agents follows it. Nevertheless, except for two-player zero-sum games and for specific instances of payoff matrices [Abernethy et. al. 2021] or for adversarial tie-breaking rules [Daskalakis and Pan, 2014], the \textit{convergence rate} of FP is unknown. In this work, we focus on the rate of convergence of FP when applied to potential games and more specifically identical payoff games. We prove that FP can take exponential time (in the number of strategies) to reach a Nash equilibrium, even if the game is restricted to \textit{two agents}. To prove this, we recursively construct a two-player coordination game with a unique Nash equilibrium. Moreover, every approximate Nash equilibrium in the constructed game must be close to the pure Nash equilibrium in $\ell_1$-distance.

----

## [0] Cocktail: Mixing Multi-Modality Control for Text-Conditional Image Generation

**Authors**: *Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, Tat-Jen Cham*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html)

**Abstract**:

Text-conditional diffusion models are able to generate high-fidelity images with diverse contents.However, linguistic representations frequently exhibit ambiguous descriptions of the envisioned objective imagery, requiring the incorporation of additional control signals to bolster the efficacy of text-guided diffusion models. In this work, we propose Cocktail, a pipeline to mix various modalities into one embedding, amalgamated with a generalized ControlNet (gControlNet), a controllable normalisation (ControlNorm), and a spatial guidance sampling method, to actualize multi-modal and spatially-refined control for text-conditional diffusion models. Specifically, we introduce a hyper-network gControlNet, dedicated to the alignment and infusion of the control signals from disparate modalities into the pre-trained diffusion model. gControlNet is capable of accepting flexible modality signals, encompassing the simultaneous reception of any combination of modality signals, or the supplementary fusion of multiple modality signals. The control signals are then fused and injected into the backbone model according to our proposed ControlNorm.Furthermore, our advanced spatial guidance sampling methodology proficiently incorporates the control signal into the designated region, thereby circumventing the manifestation of undesired objects within the generated image.We demonstrate the results of our method in controlling various modalities, proving high-quality synthesis and fidelity to multiple external signals.

----

## [0] RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability

**Authors**: *Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6692e1b0e8a31e8de84bd90ad4d8d9e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6692e1b0e8a31e8de84bd90ad4d8d9e0-Abstract-Conference.html)

**Abstract**:

Visual model-based RL methods typically encode image observations into low-dimensional representations in a manner that does not eliminate redundant information. This leaves them susceptible to spurious variations -- changes in task-irrelevant components such as background distractors or lighting conditions. In this paper, we propose a visual model-based RL method that learns a latent representation resilient to such spurious variations. Our training objective encourages the representation to be maximally predictive of dynamics and reward, while constraining the information flow from the observation to the latent representation. We demonstrate that this objective significantly bolsters the resilience of visual model-based RL methods to visual distractors, allowing them to operate in dynamic environments. We then show that while the learned encoder is able to operate in dynamic environments, it is not invariant under significant distribution shift. To address this, we propose a simple reward-free alignment procedure that enables test time adaptation of the encoder. This allows for quick adaptation to widely differing environments without having to relearn the dynamics and policy. Our effort is a step towards making model-based RL a practical and useful tool for dynamic, diverse domains and we show its effectiveness in simulation tasks with significant spurious variations.

----

## [0] Circuit as Set of Points

**Authors**: *Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, Chang Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6697bb267dc517379bc8aa326e844f8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6697bb267dc517379bc8aa326e844f8d-Abstract-Conference.html)

**Abstract**:

As the size of circuit designs continues to grow rapidly, artificial intelligence technologies are being extensively used in Electronic Design Automation (EDA) to assist with circuit design.Placement and routing are the most time-consuming parts of the physical design process, and how to quickly evaluate the placement has become a hot research topic. Prior works either transformed circuit designs into images using hand-crafted methods and then used Convolutional Neural Networks (CNN) to extract features, which are limited by the quality of the hand-crafted methods and could not achieve end-to-end training, or treated the circuit design as a graph structure and used Graph Neural Networks (GNN) to extract features, which require time-consuming preprocessing.In our work, we propose a novel perspective for circuit design by treating circuit components as point clouds and using Transformer-based point cloud perception methods to extract features from the circuit. This approach enables direct feature extraction from raw data without any preprocessing, allows for end-to-end training, and results in high performance.Experimental results show that our method achieves state-of-the-art performance in congestion prediction tasks on both the CircuitNet and ISPD2015 datasets, as well as in design rule check (DRC) violation prediction tasks on the CircuitNet dataset.Our method establishes a bridge between the relatively mature point cloud perception methods and the fast-developing EDA algorithms, enabling us to leverage more collective intelligence to solve this task. To facilitate the research of open EDA design, source codes and pre-trained models are released at https://github.com/hustvl/circuitformer.

----

## [0] Causal Component Analysis

**Authors**: *Wendong Liang, Armin Kekic, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67089958e98b243d5cc1881ad60418b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67089958e98b243d5cc1881ad60418b8-Abstract-Conference.html)

**Abstract**:

Independent Component Analysis (ICA) aims to recover independent latent variables from observed mixtures thereof. Causal Representation Learning (CRL) aims instead to infer causally related (thus often statistically dependent) latent variables, together with the unknown graph encoding their causal relationships. We introduce an intermediate problem termed Causal Component Analysis (CauCA). CauCA can be viewed as a generalization of ICA, modelling the causal dependence among the latent components, and as a special case of CRL. In contrast to CRL, it presupposes knowledge of the causal graph, focusing solely on learning the unmixing function and the causal mechanisms. Any impossibility results regarding the recovery of the ground truth in CauCA also apply for CRL, while possibility results may serve as a stepping stone for extensions to CRL. We characterize CauCA identifiability from multiple datasets generated through different types of interventions on the latent causal variables. As a corollary, this interventional perspective also leads to new identifiability results for nonlinear ICA—a special case of CauCA with an empty graph—requiring strictly fewer datasets than previous results. We introduce a likelihood-based approach using normalizing flows to estimate both the unmixing function and the causal mechanisms, and demonstrate its effectiveness through extensive synthetic experiments in the CauCA and ICA setting.

----

## [0] Latent Graph Inference with Limited Supervision

**Authors**: *Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67101f97dc23fcc10346091181fff6cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67101f97dc23fcc10346091181fff6cb-Abstract-Conference.html)

**Abstract**:

Latent graph inference (LGI) aims to jointly learn the underlying graph structure and node representations from data features. However, existing LGI methods commonly suffer from the issue of supervision starvation, where massive edge weights are learned without semantic supervision and do not contribute to the training loss. Consequently, these supervision-starved weights, which determine the predictions of testing samples, cannot be semantically optimal, resulting in poor generalization. In this paper, we observe that this issue is actually caused by the graph sparsification operation, which severely destroys the important connections established between pivotal nodes and labeled ones. To address this, we propose to restore the corrupted affinities and replenish the missed supervision for better LGI. The key challenge then lies in identifying the critical nodes and recovering the corrupted affinities. We begin by defining the pivotal nodes as k-hop starved nodes, which can be identified based on a given adjacency matrix. Considering the high computational burden, we further present a more efficient alternative inspired by CUR matrix decomposition. Subsequently, we eliminate the starved nodes by reconstructing the destroyed connections. Extensive experiments on representative benchmarks demonstrate that reducing the starved nodes consistently improves the performance of state-of-the-art LGI methods, especially under extremely limited supervision (6.12% improvement on Pubmed with a labeling rate of only 0.3%).

----

## [0] Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows

**Authors**: *Alexandre Verine, Benjamin Négrevergne, Muni Sreenivas Pydi, Yann Chevaleyre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67159f1c0cab15dd34c76a5dd830a389-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67159f1c0cab15dd34c76a5dd830a389-Abstract-Conference.html)

**Abstract**:

Achieving a balance between image quality (precision) and diversity (recall) is a significant challenge in the domain of generative models. Current state-of-the-art models primarily rely on optimizing heuristics, such as the Fr\'echet Inception Distance. While recent developments have introduced principled methods for evaluating precision and recall, they have yet to be successfully integrated into the training of generative models. Our main contribution is a novel training method for generative models, such as Generative Adversarial Networks and Normalizing Flows, which explicitly optimizes a user-defined trade-off between precision and recall.  More precisely, we show that achieving a specified precision-recall trade-off corresponds to minimizing a unique $f$-divergence from a family we call the \mbox{\em PR-divergences}. Conversely, any $f$-divergence can be written as a linear combination of PR-divergences and  corresponds to a weighted precision-recall trade-off. Through comprehensive evaluations, we show that our approach improves the performance of existing state-of-the-art models like BigGAN in terms of either precision or recall when tested on datasets such as ImageNet.

----

## [0] Energy Guided Diffusion for Generating Neurally Exciting Images

**Authors**: *Pawel A. Pierzchlewicz, Konstantin Willeke, Arne Nix, Pavithra Elumalai, Kelli Restivo, Tori Shinn, Cate Nealley, Gabrielle Rodriguez, Saumil S. Patel, Katrin Franke, Andreas S. Tolias, Fabian H. Sinz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html)

**Abstract**:

In recent years, most exciting inputs (MEIs) synthesized from encoding models of neuronal activity have become an established method for studying tuning properties of biological and artificial visual systems.    However, as we move up the visual hierarchy, the complexity of neuronal computations increases.     Consequently, it becomes more challenging to model neuronal activity, requiring more complex models.    In this study, we introduce a novel readout architecture inspired by the mechanism of visual attention. This new architecture, which we call attention readout, together with a data-driven convolutional core outperforms previous task-driven models in predicting the activity  of neurons in macaque area V4.    However, as our predictive network becomes deeper and more complex, synthesizing MEIs via straightforward gradient ascent (GA) can struggle to produce qualitatively good results and overfit to idiosyncrasies of a more complex model, potentially decreasing the MEI's model-to-brain transferability.    To solve this problem, we propose a diffusion-based method for generating MEIs via Energy Guidance (EGG).    We show that for models of macaque V4, EGG generates single neuron MEIs that generalize better across varying model architectures than the state-of-the-art GA, while at the same time reducing computational costs by a factor of 4.7x, facilitating experimentally challenging closed-loop experiments.    Furthermore, EGG diffusion can be used to generate other neurally exciting images, like most exciting naturalistic images that are on par with a selection of highly activating natural images, or image reconstructions that generalize better across architectures.    Finally, EGG is simple to implement, requires no retraining of the diffusion model, and can easily be generalized to provide other characterizations of the visual system, such as invariances.    Thus, EGG provides a general and flexible framework to study the coding properties of the visual system in the context of natural images.

----

## [0] An active learning framework for multi-group mean estimation

**Authors**: *Abdellah Aznag, Rachel Cummings, Adam N. Elmachtoub*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67390075fe466276797f489115582cdc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67390075fe466276797f489115582cdc-Abstract-Conference.html)

**Abstract**:

We consider a fundamental problem where there are multiple groups whose data distributions are unknown, and an analyst would like to learn the mean of each group. We consider an active learning framework to sequentially collect $T$ samples with bandit, each period observing a sample from a chosen group. After observing a sample, the analyst may update their estimate of the mean and variance of that group and choose the next group accordingly.  The objective is to dynamically collect samples to minimize the $p$-norm of the vector of variances of our mean estimators after $T$ rounds. We propose an algorithm, Variance-UCB, that selects groups according to a an upper bound on the variance estimate adjusted to the $p$-norm chosen. We show that the regret of Variance-UCB is $O(T^{-2})$ for finite $p$, and prove that no algorithm can do better. When $p$ is infinite, we recover the $O(T^{-1.5})$ obtained in \cite{activelearning, carpentier2011upper} and provide a new lower bound showing that no algorithm can do better.

----

## [0] CAT-Walk: Inductive Hypergraph Learning via Set Walks

**Authors**: *Ali Behrouz, Farnoosh Hashemi, Sadaf Sadeghian, Margo I. Seltzer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html)

**Abstract**:

Temporal hypergraphs provide a powerful paradigm for modeling time-dependent, higher-order interactions in complex systems. Representation learning for hypergraphs is essential for extracting patterns of the higher-order interactions that are critically important in real-world problems in social network analysis, neuroscience, finance, etc. However, existing methods are typically designed only for specific tasks or static hypergraphs. We present  CAT-Walk, an inductive method that learns the underlying dynamic laws that govern the temporal and structural processes underlying a temporal hypergraph. CAT-Walk introduces a temporal, higher-order walk on hypergraphs, SetWalk, that extracts higher-order causal patterns. CAT-Walk uses a novel adaptive and permutation invariant pooling strategy, SetMixer, along with a set-based anonymization process that hides the identity of hyperedges. Finally, we present a simple yet effective neural network model to encode hyperedges. Our evaluation on 10 hypergraph benchmark datasets shows that CAT-Walk attains outstanding performance on temporal hyperedge prediction benchmarks in both inductive and transductive settings. It also shows competitive performance with state-of-the-art methods for node classification. (https://github.com/ubc-systopia/CATWalk)

----

## [0] Unbiased constrained sampling with Self-Concordant Barrier Hamiltonian Monte Carlo

**Authors**: *Maxence Noble, Valentin De Bortoli, Alain Durmus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6745cb9889cc213bda803535f2d3902e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6745cb9889cc213bda803535f2d3902e-Abstract-Conference.html)

**Abstract**:

In this paper, we propose Barrier Hamiltonian Monte Carlo (BHMC), a version of the  HMC algorithm which aims at sampling from a Gibbs distribution $\pi$ on a manifold  $\mathsf{M}$, endowed with a Hessian metric $\mathfrak{g}$ derived from a self-concordant  barrier. Our method relies on Hamiltonian  dynamics which comprises $\mathfrak{g}$. Therefore, it incorporates the constraints defining  $\mathsf{M}$ and is able to exploit its underlying geometry. However,   the corresponding Hamiltonian dynamics is defined via non separable Ordinary Differential Equations (ODEs) in contrast to the Euclidean case. It implies unavoidable bias in existing generalization of HMC to Riemannian manifolds. In this paper, we propose a new filter step, called ``involution checking step'', to address this problem. This step is implemented in two versions of BHMC, coined continuous BHMC (c-bHMC) and  numerical BHMC (n-BHMC) respectively.  Our main results establish that these two new algorithms  generate reversible Markov  chains with respect to $\pi$ and do not suffer from any bias in comparison to previous implementations. Our conclusions are supported by numerical experiments where  we consider target distributions defined on polytopes.

----

## [0] Directional diffusion models for graph representation learning

**Authors**: *Run Yang, Yuling Yang, Fan Zhou, Qiang Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6751ee6546b31ceb7d4ee12276b9f4d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6751ee6546b31ceb7d4ee12276b9f4d9-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved remarkable success in diverse domains such as image synthesis, super-resolution, and 3D molecule generation. Surprisingly, the application of diffusion models in graph learning has garnered little attention. In this paper, we aim to bridge this gap by exploring the use of diffusion models for unsupervised graph representation learning. Our investigation commences with the identification of anisotropic structures within graphs and the recognition of a crucial limitation in the vanilla forward diffusion process when dealing with these anisotropic structures. The original forward diffusion process continually adds  isotropic Gaussian noise to the data, which may excessively dilute anisotropic signals, leading to rapid signal-to-noise conversion. This rapid conversion poses challenges for training denoising neural networks and obstructs the acquisition of semantically meaningful representations during the reverse process. To overcome this challenge, we introduce a novel class of models termed {\it directional diffusion models}.  These models adopt data-dependent, anisotropic, and directional noises in the forward diffusion process. In order to assess the effectiveness of our proposed models, we conduct extensive experiments on 12 publicly available datasets, with a particular focus on two distinct graph representation learning tasks. The experimental results unequivocally establish the superiority of our models over state-of-the-art baselines, underscoring their effectiveness in capturing meaningful graph representations. Our research not only sheds light on the intricacies of the forward process in diffusion models but also underscores the vast potential of these models in addressing a wide spectrum of graph-related tasks. Our code is available at \url{https://github.com/statsle/DDM}.

----

## [0] UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition

**Authors**: *Qiufu Li, Xi Jia, Jiancan Zhou, Linlin Shen, Jinming Duan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6776737cd11cf4afa3af226898474418-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6776737cd11cf4afa3af226898474418-Abstract-Conference.html)

**Abstract**:

Sample-to-class-based face recognition models can not fully explore the cross-sample relationship among large amounts of facial images, while sample-to-sample-based models require sophisticated pairing processes for training. Furthermore, neither method satisfies the requirements of real-world face verification applications, which expect a unified threshold separating positive from negative facial pairs. In this paper, we propose a unified threshold integrated sample-to-sample based loss (USS loss), which features an explicit unified threshold for distinguishing positive from negative pairs. Inspired by our USS loss, we also derive the sample-to-sample based softmax and BCE losses, and discuss their relationship. Extensive evaluation on multiple benchmark datasets, including MFR, IJB-C, LFW, CFP-FP, AgeDB, and MegaFace, demonstrates that the proposed USS loss is highly efficient and can work seamlessly with sample-to-class-based losses. The embedded loss (USS and sample-to-class Softmax loss) overcomes the pitfalls of previous approaches and the trained facial model UniTSFace exhibits exceptional performance, outperforming state-of-the-art methods, such as CosFace, ArcFace, VPL, AnchorFace, and UNPG. Our code is available at https://github.com/CVI-SZU/UniTSFace.

----

## [0] Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks

**Authors**: *Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/677c8dc72c99482507323f313faf4738-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/677c8dc72c99482507323f313faf4738-Abstract-Conference.html)

**Abstract**:

Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP. The code of MDP is publicly available.

----

## [0] On the Power of SVD in the Stochastic Block Model

**Authors**: *Xinyu Mao, Jiapeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/678594bcff6f99f3b7a8ff459989b1a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/678594bcff6f99f3b7a8ff459989b1a3-Abstract-Conference.html)

**Abstract**:

A popular heuristic method for improving clustering results is to apply dimensionality reduction before running clustering algorithms.It has been observed that spectral-based dimensionality reduction tools, such as PCA or SVD, improve the performance of clustering algorithms in many applications. This phenomenon indicates that spectral method not only serves as a dimensionality reduction tool, but also contributes to the clustering procedure in some sense. It is an interesting question to understand the behavior of spectral steps in clustering problems.As an initial step in this direction, this paper studies the power of vanilla-SVD algorithm in the stochastic block model (SBM). We show that, in the symmetric setting, vanilla-SVD algorithm recovers all clusters correctly. This result answers an open question posed by Van Vu (Combinatorics Probability and Computing, 2018) in the symmetric setting.

----

## [0] Continuous-time Analysis of Anchor Acceleration

**Authors**: *Jaewook J. Suh, Jisun Park, Ernest K. Ryu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/678cffc05549fdabda971127602084c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/678cffc05549fdabda971127602084c6-Abstract-Conference.html)

**Abstract**:

Recently, the anchor acceleration, an acceleration mechanism distinct from Nesterov's, has been discovered for minimax optimization and fixed-point problems, but its mechanism is not understood well, much less so than Nesterov acceleration. In this work, we analyze continuous-time models of anchor acceleration. We provide tight, unified analyses for characterizing the convergence rate as a function of the anchor coefficient $\beta(t)$, thereby providing insight into the anchor acceleration mechanism and its accelerated $\mathcal{O}(1/k^2)$-convergence rate. Finally, we present an adaptive method inspired by the continuous-time analyses and establish its effectiveness through theoretical analyses and experiments.

----

## [0] BEDD: The MineRL BASALT Evaluation and Demonstrations Dataset for Training and Benchmarking Agents that Solve Fuzzy Tasks

**Authors**: *Stephanie Milani, Anssi Kanervisto, Karolis Ramanauskas, Sander Schulhoff, Brandon Houghton, Rohin Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67a6726dcd555b982cabb3446ffac01d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/67a6726dcd555b982cabb3446ffac01d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The MineRL BASALT competition has served to catalyze advances in learning from human feedback through four hard-to-specify tasks in Minecraft, such as create and photograph a waterfall. Given the completion of two years of BASALT competitions, we offer to the community a formalized benchmark through the BASALT Evaluation and Demonstrations Dataset (BEDD), which serves as a resource for algorithm development and performance assessment. BEDD consists of a collection of 26 million image-action pairs from nearly 14,000 videos of human players completing the BASALT tasks in Minecraft. It also includes over 3,000 dense pairwise human evaluations of human and algorithmic agents. These comparisons serve as a fixed, preliminary leaderboard for evaluating newly-developed algorithms.  To enable this comparison, we present a streamlined codebase for benchmarking new algorithms against the leaderboard. In addition to presenting these datasets, we conduct a detailed analysis of the data from both datasets to guide algorithm development and evaluation. The released code and data are available at https://github.com/minerllabs/basalt-benchmark.

----

## [0] Self-supervised Object-Centric Learning for Videos

**Authors**: *Görkay Aydemir, Weidi Xie, Fatma Güney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html)

**Abstract**:

Unsupervised multi-object segmentation has shown impressive results on images by utilizing powerful semantics learned from self-supervised pretraining. An additional modality such as depth or motion is often used to facilitate the segmentation in video sequences. However, the performance improvements observed in synthetic sequences, which rely on the robustness of an additional cue, do not translate to more challenging real-world scenarios. In this paper, we propose the first fully unsupervised method for segmenting multiple objects in real-world sequences. Our object-centric learning framework spatially binds objects to slots on each frame and then relates these slots across frames. From these temporally-aware slots, the training objective is to reconstruct the middle frame in a high-level semantic feature space. We propose a masking strategy by dropping a significant portion of tokens in the feature space for efficiency and regularization. Additionally, we address over-clustering by merging slots based on similarity. Our method can successfully segment multiple instances of complex and high-variety classes in YouTube videos.

----

## [0] Improving Adversarial Transferability via Intermediate-level Perturbation Decay

**Authors**: *Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67b2e2e895380fa6acd537c2894e490e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67b2e2e895380fa6acd537c2894e490e-Abstract-Conference.html)

**Abstract**:

Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

----

## [0] GUST: Combinatorial Generalization by Unsupervised Grouping with Neuronal Coherence

**Authors**: *Hao Zheng, Hui Lin, Rong Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67d5c7dd7930dfce2725defdb0552b6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67d5c7dd7930dfce2725defdb0552b6e-Abstract-Conference.html)

**Abstract**:

Dynamically grouping sensory information into structured entities is essential for understanding the world of combinatorial nature. However, the grouping ability and therefore combinatorial generalization are still challenging artificial neural networks. Inspired by the evidence that successful grouping is indicated by neuronal coherence in the human brain, we introduce GUST (Grouping Unsupervisely by Spike Timing network), an iterative network architecture with biological constraints to bias the network towards a dynamical state of neuronal coherence that softly reflects the grouping information in the temporal structure of its spiking activity. We evaluate and analyze the model on synthetic datasets. Interestingly, the segregation ability is directly learned from superimposed stimuli with a succinct unsupervised objective. Two learning stages are present, from coarsely perceiving global features to additionally capturing local features. Further, the learned symbol-like building blocks can be systematically composed to represent novel scenes in a bio-plausible manner.

----

## [0] State Regularized Policy Optimization on Data with Dynamics Shift

**Authors**: *Zhenghai Xue, Qingpeng Cai, Shuchang Liu, Dong Zheng, Peng Jiang, Kun Gai, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67dd6a41bf9539cffc0fc0165e4d0616-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67dd6a41bf9539cffc0fc0165e4d0616-Abstract-Conference.html)

**Abstract**:

In many real-world scenarios, Reinforcement Learning (RL) algorithms are trained on data with dynamics shift, i.e., with different underlying environment dynamics. A majority of current methods address such issue by training context encoders to identify environment parameters. Data with dynamics shift are separated according to their environment parameters to train the corresponding policy.However, these methods can be sample inefficient as data are used \textit{ad hoc}, and policies trained for one dynamics cannot benefit from data collected in all other environments with different dynamics. In this paper, we find that in many environments with similar structures and different dynamics, optimal policies have similar stationary state distributions. We exploit such property and learn the stationary state distribution from data with dynamics shift for efficient data reuse. Such distribution is used to regularize the policy trained in a new environment, leading to the SRPO (\textbf{S}tate \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization) algorithm. To conduct theoretical analyses, the intuition of similar environment structures is characterized by the notion of homomorphous MDPs. We then demonstrate a lower-bound performance guarantee on policies regularized by the stationary state distribution. In practice, SRPO can be an add-on module to context-based algorithms in both online and offline RL settings.Experimental results show that SRPO can make several context-based algorithms far more data efficient and significantly improve their overall performance.

----

## [0] Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

**Authors**: *Andy Zhou, Jindong Wang, Yu-Xiong Wang, Haohan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html)

**Abstract**:

We propose a conceptually simple and lightweight framework for improving the robustness of vision models through the combination of knowledge distillation and data augmentation. We address the conjecture that larger models do not make for better teachers by showing strong gains in out-of-distribution robustness when distilling from pretrained foundation models. Following this finding, we propose Discrete Adversarial Distillation (DAD), which leverages a robust teacher to generate adversarial examples and a VQGAN to discretize them, creating more informative samples than standard data augmentation techniques. We provide a theoretical framework for the use of a robust teacher in the knowledge distillation with data augmentation setting and demonstrate strong gains in out-of-distribution robustness and clean accuracy across different student architectures. Notably, our method adds minor computational overhead compared to similar techniques and can be easily combined with other data augmentations for further improvements.

----

## [0] XES3G5M: A Knowledge Tracing Benchmark Dataset with Auxiliary Information

**Authors**: *Zitao Liu, Qiongqiong Liu, Teng Guo, Jiahao Chen, Shuyan Huang, Xiangyu Zhao, Jiliang Tang, Weiqi Luo, Jian Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Knowledge tracing (KT) is a task that predicts students' future performance based on their historical learning interactions. With the rapid development of deep learning techniques, existing KT approaches follow a data-driven paradigm that uses massive problem-solving records to model students' learning processes. However, although the educational contexts contain various factors that may have an influence on student learning outcomes, existing public KT datasets mainly consist of anonymized ID-like features, which may hinder the research advances towards this field. Therefore, in this work, we present, \emph{XES3G5M}, a large-scale dataset with rich auxiliary information about questions and their associated knowledge components (KCs)\footnote{\label{ft:kc}A KC is a generalization of everyday terms like concept, principle, fact, or skill.}. The XES3G5M dataset is collected from a real-world online math learning platform, which contains 7,652 questions, and 865 KCs with 5,549,635 interactions from 18,066 students. To the best of our knowledge, the XES3G5M dataset not only has the largest number of KCs in math domain but contains the richest contextual information including tree structured KC relations, question types, textual contents and analysis and student response timestamps. Furthermore, we build a comprehensive benchmark on 19 state-of-the-art deep learning based knowledge tracing (DLKT) models. Extensive experiments demonstrate the effectiveness of leveraging the auxiliary information in our XES3G5M with DLKT models. We hope the proposed dataset can effectively facilitate the KT research work.

----

## [0] Factorized Contrastive Learning: Going Beyond Multi-view Redundancy

**Authors**: *Paul Pu Liang, Zihao Deng, Martin Q. Ma, James Y. Zou, Louis-Philippe Morency, Ruslan Salakhutdinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html)

**Abstract**:

In a wide range of multimodal tasks, contrastive learning has become a particularly appealing approach since it can successfully learn representations from abundant unlabeled data with only pairing information (e.g., image-caption or video-audio pairs). Underpinning these approaches is the assumption of multi-view redundancy - that shared information between modalities is necessary and sufficient for downstream tasks. However, in many real-world settings, task-relevant information is also contained in modality-unique regions: information that is only present in one modality but still relevant to the task. How can we learn self-supervised multimodal representations to capture both shared and unique information relevant to downstream tasks? This paper proposes FactorCL, a new multimodal representation learning method to go beyond multi-view redundancy. FactorCL is built from three new contributions: (1) factorizing task-relevant information into shared and unique representations, (2) capturing task-relevant information via maximizing MI lower bounds and removing task-irrelevant information via minimizing MI upper bounds, and (3) multimodal data augmentations to approximate task relevance without labels. On large-scale real-world datasets, FactorCL captures both shared and unique information and achieves state-of-the-art results on six benchmarks.

----

## [0] Semantic Image Synthesis with Unconditional Generator

**Authors**: *Jungwoo Chae, Hyunin Cho, Sooyeon Go, Kyungmook Choi, Youngjung Uh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/683464f40aa1a6b7c939c3e9cd64b1fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/683464f40aa1a6b7c939c3e9cd64b1fd-Abstract-Conference.html)

**Abstract**:

Semantic image synthesis (SIS) aims to generate realistic images according to semantic masks given by a user. Although recent methods produce high quality results with fine spatial control, SIS requires expensive pixel-level annotation of the training images. On the other hand, manipulating intermediate feature maps in a pretrained unconditional generator such as StyleGAN supports coarse spatial control without heavy annotation. In this paper, we introduce a new approach, for reflecting user's detailed guiding masks on a pretrained unconditional generator. Our method converts a user's guiding mask to a proxy mask through a semantic mapper. Then the proxy mask conditions the resulting image through a rearranging network based on cross-attention mechanism. The proxy mask is simple clustering of intermediate feature maps in the generator. The semantic mapper and the rearranging network are easy to train (less than half an hour). Our method is useful for many tasks: semantic image synthesis, spatially editing real images, and unaligned local transplantation. Last but not least, it is generally applicable to various datasets such as human faces, animal faces, and churches.

----

## [0] Learning Neural Implicit through Volume Rendering with Attentive Depth Fusion Priors

**Authors**: *Pengchong Hu, Zhizhong Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68637ee6b30276f900bc67320466b69f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68637ee6b30276f900bc67320466b69f-Abstract-Conference.html)

**Abstract**:

Learning neural implicit representations has achieved remarkable performance in 3D reconstruction from multi-view images. Current methods use volume rendering to render implicit representations into either RGB or depth images that are supervised by the multi-view ground truth. However, rendering a view each time suffers from incomplete depth at holes and unawareness of occluded structures from the depth supervision, which severely affects the accuracy of geometry inference via volume rendering. To resolve this issue, we propose to learn neural implicit representations from multi-view RGBD images through volume rendering with an attentive depth fusion prior. Our prior allows neural networks to sense coarse 3D structures from the Truncated Signed Distance Function (TSDF) fused from all available depth images for rendering. The TSDF enables accessing the missing depth at holes on one depth image and the occluded parts that are invisible from the current view. By introducing a novel attention mechanism, we allow neural networks to directly use the depth fusion prior with the inferred occupancy as the learned implicit function. Our attention mechanism works with either a one-time fused TSDF that represents a whole scene or an incrementally fused TSDF that represents a partial scene in the context of Simultaneous Localization and Mapping (SLAM). Our evaluations on widely used benchmarks including synthetic and real-world scans show our superiority over the latest neural implicit methods.

----

## [0] SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning

**Authors**: *Yifan Yang, Peiyao Xiao, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/686a3f32067838c8dbb68da6e9e3cf69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/686a3f32067838c8dbb68da6e9e3cf69-Abstract-Conference.html)

**Abstract**:

Federated bilevel optimization (FBO) has shown great potential recently in machine learning and edge computing due to the emerging nested optimization structure in meta-learning, fine-tuning, hyperparameter tuning, etc. However, existing FBO algorithms often involve complicated computations and require multiple sub-loops per iteration, each of which contains a number of communication rounds. In this paper, we propose a simple and flexible FBO framework named SimFBO, which is easy to implement without sub-loops, and includes a generalized server-side aggregation and update for improving communication efficiency. We further propose System-level heterogeneity robust FBO (ShroFBO) as a variant of SimFBO with stronger resilience to heterogeneous local computation. We show that SimFBO and ShroFBO provably achieve a linear convergence speedup with partial client participation and client sampling without replacement, as well as improved sample and communication complexities. Experiments demonstrate the effectiveness of the proposed methods over existing FBO algorithms.

----

## [0] PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification

**Authors**: *Qianli Shen, Wai Hoh Tang, Zhun Deng, Apostolos F. Psaros, Kenji Kawaguchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68730224bbf35ffac7a4fbf9b1ea4bfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68730224bbf35ffac7a4fbf9b1ea4bfe-Abstract-Conference.html)

**Abstract**:

Standard approaches for uncertainty quantification in deep learning and physics-informed learning have persistent limitations. Indicatively, strong assumptions regarding the data likelihood are required, the performance highly depends on the selection of priors, and the posterior can be sampled only approximately, which leads to poor approximations because of the associated computational cost.This paper introduces and studies confidence interval (CI) estimation for deterministic partial differential equations as a novel problem.That is, to propagate confidence, in the form of CIs, from data locations to the entire domain with probabilistic guarantees.We propose a method, termed Physics-Informed Confidence Propagation (PICProp), based on bi-level optimization to compute a valid CI without making heavy assumptions.We provide a theorem regarding the validity of our method, and computational experiments, where the focus is on physics-informed learning. Code is available at https://github.com/ShenQianli/PICProp.

----

## [0] Foundation Model is Efficient Multimodal Multitask Model Selector

**Authors**: *Fanqing Meng, Wenqi Shao, Zhanglin Peng, Chonghe Jiang, Kaipeng Zhang, Yu Qiao, Ping Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/687b7b2bdcc2ced577c0a989b44e7078-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/687b7b2bdcc2ced577c0a989b44e7078-Abstract-Conference.html)

**Abstract**:

This paper investigates an under-explored but important problem: given a collection of pre-trained neural networks, predicting their performance on each multi-modal task without fine-tuning them, such as image recognition, referring, captioning, visual question answering, and text question answering.A brute-force approach is to finetune all models on all target datasets, bringing high computational costs. Although recent-advanced approaches employed lightweight metrics to measure models’ transferability, they often depend heavily on the prior knowledge of a single task, making them inapplicable in a multi-modal multi-task scenario. To tackle this issue, we propose an efficient multi-task model selector (EMMS), which employs large-scale foundation models to transform diverse label formats such as categories, texts, and bounding boxes of different downstream tasks into a unified noisy label embedding. EMMS can estimate a model’s transferability through a simple weighted linear regression, which can be efficiently solved by an alternating minimization algorithm with a convergence guarantee. Extensive experiments on 5 downstream tasks with 24 datasets show that EMMS is fast, effective, and generic enough to assess the transferability of pre-trained models, making it the first model selection method in the multi-task scenario. For instance, compared with the state- of-the-art method LogME enhanced by our label embeddings, EMMS achieves 9.0%, 26.3%, 20.1%, 54.8%, 12.2% performance gain on image recognition, referring, captioning, visual question answering, and text question answering, while bringing 5.13×, 6.29×, 3.59×, 6.19×, and 5.66× speedup in wall-clock time, respectively. The code is available at https://github.com/OpenGVLab/Multitask-Model-Selector.

----

## [0] Feature Likelihood Score: Evaluating the Generalization of Generative Models Using Samples

**Authors**: *Marco Jiralerspong, Avishek Joey Bose, Ian Gemp, Chongli Qin, Yoram Bachrach, Gauthier Gidel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68b138608ef80b08d65b1bd9594d9559-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68b138608ef80b08d65b1bd9594d9559-Abstract-Conference.html)

**Abstract**:

The past few years have seen impressive progress in the development of deep generative models capable of producing high-dimensional, complex, and photo-realistic data. However, current methods for evaluating such models remain incomplete: standard likelihood-based metrics do not always apply and rarely correlate with perceptual fidelity, while sample-based metrics, such as FID, are insensitive to overfitting, i.e., inability to generalize beyond the training set. To address these limitations, we propose a new metric called the Feature Likelihood Divergence (FLD), a parametric sample-based score that uses density estimation to provide a comprehensive trichotomic evaluation accounting for novelty (i.e., different from the training samples), fidelity, and diversity of generated samples.  We empirically demonstrate the ability of FLD to identify specific overfitting problem cases, where previously proposed metrics fail. We also extensively evaluate FLD on various image datasets and model classes, demonstrating its ability to match intuitions of previous metrics like FID while offering a more comprehensive evaluation of generative models.

----

## [0] LOVM: Language-Only Vision Model Selection

**Authors**: *Orr Zohar, Shih-Cheng Huang, Kuan-Chieh Wang, Serena Yeung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68c33c4e6fc97f7b31c964dc83303a28-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/68c33c4e6fc97f7b31c964dc83303a28-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Pre-trained multi-modal vision-language models (VLMs) are becoming increasingly popular due to their exceptional performance on downstream vision applications, particularly in the few- and zero-shot settings. However, selecting the best-performing VLM for some downstream applications is non-trivial, as it is dataset and task-dependent. Meanwhile, the exhaustive evaluation of all available VLMs on a novel application is not only time and  computationally demanding but also necessitates the collection of a labeled dataset for evaluation. As the number of open-source VLM variants increases, there is a need for an efficient model selection strategy that does not require access to a curated evaluation dataset. This paper proposes a novel task and benchmark for efficiently evaluating VLMs' zero-shot performance on downstream applications without access to the downstream task dataset. Specifically, we introduce a new task LOVM: Language-Only  Vision  Model Selection , where methods are expected to perform both model selection and performance prediction based solely on a text description of the desired downstream application. We then introduced an extensive LOVM benchmark consisting of ground-truth evaluations of 35 pre-trained VLMs and 23 datasets, where methods are expected to rank the pre-trained VLMs and predict their zero-shot performance.

----

## [0] Statistical Analysis of Quantum State Learning Process in Quantum Neural Networks

**Authors**: *Hao-Kai Zhang, Chenghong Zhu, Mingrui Jing, Xin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68efc144ad3b41108f779b51b9fb1300-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68efc144ad3b41108f779b51b9fb1300-Abstract-Conference.html)

**Abstract**:

Quantum neural networks (QNNs) have been a promising framework in pursuing near-term quantum advantage in various fields, where many applications can be viewed as learning a quantum state that encodes useful data. As a quantum analog of probability distribution learning, quantum state learning is theoretically and practically essential in quantum machine learning. In this paper, we develop a no-go theorem for learning an unknown quantum state with QNNs even starting from a high-fidelity initial state. We prove that when the loss value is lower than a critical threshold, the probability of avoiding local minima vanishes exponentially with the qubit count, while only grows polynomially with the circuit depth. The curvature of local minima is concentrated to the quantum Fisher information times a loss-dependent constant, which characterizes the sensibility of the output state with respect to parameters in QNNs. These results hold for any circuit structures, initialization strategies, and work for both fixed ansatzes and adaptive methods. Extensive numerical simulations are performed to validate our theoretical results. Our findings place generic limits on good initial guesses and adaptive methods for improving the learnability and scalability of QNNs, and deepen the understanding of prior information's role in QNNs.

----

## [0] SOL: Sampling-based Optimal Linear bounding of arbitrary scalar functions

**Authors**: *Yuriy Biktairov, Jyotirmoy Deshmukh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690b93e9ab0cc3b1d88b32f6f473ce69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690b93e9ab0cc3b1d88b32f6f473ce69-Abstract-Conference.html)

**Abstract**:

Finding tight linear bounds for activation functions in neural networksis an essential part of several state of the art neural network robustness certification tools. An activation function is an arbitrary, nonlinear,scalar function $f: \mathbb{R}^d \rightarrow \mathbb{R}$. In the existing work on robustness certification, such bounds have been computed using human ingenuity for a handful of the most popular activation functions. While a number of heuristics have been proposed for bounding arbitrary functions,no analysis of the tightness optimality for general scalar functions has been offered yet, to the best of our knowledge. We fill this gap by formulating a concise optimality criterion for tightness of the approximation which allows us tobuild optimal bounds for any function convex in the region of interest $R$. Fora more general class of functions Lipshitz-continuous in $R$ we propose a sampling-based approach (SOL) which, given an instance of the bounding problem, efficiently computes the tightest linear bounds within a given $\varepsilon > 0$ threshold. We leverage an adaptive sampling technique to iteratively build a setof sample points suitable for representing the target activation function. While the theoretical worst case time complexity of our approach is$O(\varepsilon^{-2d})$,it typically only takes $O(\log^{\beta} \frac{1}{\varepsilon})$ time for some $\beta \ge 1$ and isthus sufficiently fast in practice. We provide empirical evidence of SOL's practicalityby incorporating it into a robustness certifier and observing that itproduces similar or higher certification rates while taking as low as quarter of the time compared to the other methods.

----

## [0] Opening the Vocabulary of Egocentric Actions

**Authors**: *Dibyadip Chatterjee, Fadime Sener, Shugao Ma, Angela Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690e82a09bcb3f101831962bf3cb54ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690e82a09bcb3f101831962bf3cb54ec-Abstract-Conference.html)

**Abstract**:

Human actions in egocentric videos often feature hand-object interactions composed of a verb (performed by the hand) applied to an object. Despite their extensive scaling up, egocentric datasets still face two limitations â€” sparsity of action compositions and a closed set of interacting objects. This paper proposes a novel open vocabulary action recognition task. Given a set of verbs and objects observed during training, the goal is to generalize the verbs to an open vocabulary of actions with seen and novel objects. To this end, we decouple the verb and object predictions via an object-agnostic verb encoder and a prompt-based object encoder. The prompting leverages CLIP representations to predict an open vocabulary of interacting objects. We create open vocabulary benchmarks on the EPIC-KITCHENS-100 and Assembly101 datasets; whereas closed-action methods fail to generalize, our proposed method is effective. In addition, our object encoder significantly outperforms existing open-vocabulary visual recognition methods in recognizing novel interacting objects.

----

## [0] On the Pareto Front of Multilingual Neural Machine Translation

**Authors**: *Liang Chen, Shuming Ma, Dongdong Zhang, Furu Wei, Baobao Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690eb240baf1180b69dac48fc905c918-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690eb240baf1180b69dac48fc905c918-Abstract-Conference.html)

**Abstract**:

In this work, we study how the performance of a given direction changes with its sampling ratio in Multilingual Neural Machine Translation (MNMT). By training over 200 multilingual models with various model sizes, data sizes, and language directions, we find it interesting that the performance of certain translation direction does not always improve with the increase of its weight in the multi-task optimization objective. Accordingly, scalarization method leads to a multitask trade-off front that deviates from the traditional Pareto front when there exists data imbalance in the training corpus, which poses a great challenge to improve the overall performance of all directions. Based on our observations, we propose the Double Power Law to predict the unique performance trade-off front in MNMT, which is robust across various languages, data adequacy, and the number of tasks. Finally, we formulate the sample ratio selection problem in MNMT as an optimization problem based on the Double Power Law. Extensive experiments show that it achieves better performance than temperature searching and gradient manipulation methods with only 1/5 to 1/2 of the total training budget. We release the code at https://github.com/pkunlp-icler/ParetoMNMT for reproduction.

----

## [0] Hierarchically Gated Recurrent Neural Network for Sequence Modeling

**Authors**: *Zhen Qin, Songlin Yang, Yiran Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html)

**Abstract**:

Transformers have surpassed RNNs in popularity due to their superior abilities in parallel training and long-term dependency modeling.Recently, there has been a renewed interest in using linear RNNs for efficient sequence modeling.These linear RNNs often employ gating mechanisms in the output of the linear recurrence layer while ignoring the significance of using forget gates within the recurrence. In this paper, we propose a gated linear RNN model dubbed Hierarchically Gated Recurrent Neural Network (HGRN), which includes forget gates that are lower bounded by a learnable value. The lower bound increases monotonically when moving up layers. This allows the upper layers to model long-term dependencies and the lower layers to model more local, short-term dependencies. Experiments on language modeling, image classification, and long-range arena benchmarks showcase the efficiency and effectiveness of our proposed model. The source code is available at https://github.com/OpenNLPLab/HGRN.

----

## [0] Why Did This Model Forecast This Future? Information-Theoretic Saliency for Counterfactual Explanations of Probabilistic Regression Models

**Authors**: *Chirag Raman, Alec Nonnemaker, Amelia Villegas-Morcillo, Hayley Hung, Marco Loog*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/694ec0018b9fd0ebe863ec29fa5a89b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/694ec0018b9fd0ebe863ec29fa5a89b9-Abstract-Conference.html)

**Abstract**:

We propose a post hoc saliency-based explanation framework for counterfactual reasoning in probabilistic multivariate time-series forecasting (regression) settings. Building upon Miller's framework of explanations derived from research in multiple social science disciplines, we establish a conceptual link between counterfactual reasoning and saliency-based explanation techniques. To address the lack of a principled notion of saliency, we leverage a unifying definition of information-theoretic saliency grounded in preattentive human visual cognition and extend it to forecasting settings. Specifically, we obtain a closed-form expression for commonly used density functions to identify which observed timesteps appear salient to an underlying model in making its probabilistic forecasts. We empirically validate our framework in a principled manner using synthetic data to establish ground-truth saliency that is unavailable for real-world data. Finally, using real-world data and forecasting models, we demonstrate how our framework can assist domain experts in forming new data-driven hypotheses about the causal relationships between features in the wild.

----

## [0] Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions

**Authors**: *Kai Liu, Zhihang Fu, Chao Chen, Sheng Jin, Ze Chen, Mingyuan Tao, Rongxin Jiang, Jieping Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/695b6f9490d27d852e439e35c56e73e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/695b6f9490d27d852e439e35c56e73e3-Abstract-Conference.html)

**Abstract**:

The key to OOD detection has two aspects: generalized feature representation and precise category description. Recently, vision-language models such as CLIP provide significant advances in both two issues, but constructing precise category descriptions is still in its infancy due to the absence of unseen categories. This work introduces two hierarchical contexts, namely perceptual context and spurious context, to carefully describe the precise category boundary through automatic prompt tuning. Specifically, perceptual contexts perceive the inter-category difference (e.g., cats vs apples) for current classification tasks, while spurious contexts further identify spurious (similar but exactly not) OOD samples for every single category (e.g., cats vs panthers, apples vs peaches). The two contexts hierarchically construct the precise description for a certain category, which is, first roughly classifying a sample to the predicted category and then delicately identifying whether it is truly an ID sample or actually OOD. Moreover, the precise descriptions for those categories within the vision-language framework present a novel application: CATegory-EXtensible OOD detection (CATEX). One can efficiently extend the set of recognizable categories by simply merging the hierarchical contexts learned under different sub-task settings. And extensive experiments are conducted to demonstrate CATEXâ€™s effectiveness, robustness, and category-extensibility. For instance, CATEX consistently surpasses the rivals by a large margin with several protocols on the challenging ImageNet-1K dataset. In addition, we offer new insights on how to efficiently scale up the prompt engineering in vision-language models to recognize thousands of object categories, as well as how to incorporate large language models (like GPT-3) to boost zero-shot applications.

----

## [0] Online Corrupted User Detection and Regret Minimization

**Authors**: *Zhiyong Wang, Jize Xie, Tong Yu, Shuai Li, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html)

**Abstract**:

In real-world online web systems, multiple users usually arrive sequentially into the system. For applications like click fraud and fake reviews, some users can maliciously perform corrupted (disrupted) behaviors to trick the system. Therefore, it is crucial to design efficient online learning algorithms to robustly learn from potentially corrupted user behaviors and accurately identify the corrupted users in an online manner.  Existing works propose bandit algorithms robust to adversarial corruption. However, these algorithms are designed for a single user, and cannot leverage the implicit social relations among multiple users for more efficient learning. Moreover, none of them consider how to detect corrupted users online in the multiple-user scenario. In this paper, we present an important online learning problem named LOCUD to learn and utilize unknown user relations from disrupted behaviors to speed up learning, and identify the corrupted users in an online setting. To robustly learn and utilize the unknown relations among potentially corrupted users, we propose a novel bandit algorithm RCLUB-WCU. To detect the corrupted users, we devise a novel online detection algorithm OCCUD based on RCLUB-WCU's inferred user relations. We prove a regret upper bound for RCLUB-WCU, which asymptotically matches the lower bound with respect to $T$ up to logarithmic factors, and matches the state-of-the-art results in degenerate cases. We also give a theoretical guarantee for the detection accuracy of OCCUD. With extensive experiments, our methods achieve superior performance over previous bandit algorithms and high corrupted user detection accuracy.

----

## [0] Nash Regret Guarantees for Linear Bandits

**Authors**: *Ayush Sawarni, Soumyabrata Pal, Siddharth Barman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69bf9fd8d3b7b792b6c8c19149024d22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69bf9fd8d3b7b792b6c8c19149024d22-Abstract-Conference.html)

**Abstract**:

We obtain essentially tight upper bounds for a strengthened notion of regret in the stochastic linear bandits framework. The strengthening---referred to as Nash regret---is defined as the difference between the (a priori unknown) optimum and the geometric mean of expected rewards accumulated by the linear bandit algorithm. Since the geometric mean corresponds to the well-studied Nash social welfare (NSW) function, this formulation quantifies the performance of a bandit algorithm as the collective welfare it generates across rounds. NSW is known to satisfy fairness axioms and, hence, an upper bound on Nash regret provides a principled fairness guarantee.    We consider the stochastic linear bandits problem over a horizon of $\mathsf{T}$ rounds and with a set of arms ${\cal X}$ in ambient dimension $d$. Furthermore, we focus on settings in which the stochastic reward---associated with each arm in ${\cal X}$---is a non-negative, sub-Poisson random variable. For this setting, we develop an algorithm that achieves a Nash regret of $O\left( \sqrt{\frac{d}{\mathsf{T}}} \log(\mathsf{T} |{\cal X}|)\right)$. In addition, addressing linear bandit instances in which the set of arms ${\cal X}$ is not necessarily finite, we obtain a Nash regret upper bound of $O\left( \frac{d^\frac{5}{4}}{\sqrt{\mathsf{T}}}  \log(\mathsf{T})\right)$. Since bounded random variables are sub-Poisson, these results hold for bounded, non-negative rewards. Our linear bandit algorithm is built upon the successive elimination method with novel technical insights, including tailored concentration bounds and the use of sampling via John ellipsoid in conjunction with the Kieferâ€“Wolfowitz optimal design.

----

## [0] ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

**Authors**: *Haoran You, Huihong Shi, Yipin Guo, Yingyan Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html)

**Abstract**:

Vision Transformers (ViTs) have shown impressive performance and have become a unified backbone for multiple vision tasks. However, both the attention mechanism and multi-layer perceptrons (MLPs) in ViTs are not sufficiently efficient due to dense multiplications, leading to costly training and inference. To this end, we propose to reparameterize pre-trained ViTs with a mixture of multiplication primitives, e.g., bitwise shifts and additions, towards a new type of multiplication-reduced model, dubbed $\textbf{ShiftAddViT}$, which aims to achieve end-to-end inference speedups on GPUs without requiring training from scratch. Specifically, all $\texttt{MatMuls}$ among queries, keys, and values are reparameterized using additive kernels, after mapping queries and keys to binary codes in Hamming space. The remaining MLPs or linear layers are then reparameterized with shift kernels. We utilize TVM to implement and optimize those customized kernels for practical hardware deployment on GPUs. We find that such a reparameterization on (quadratic or linear) attention maintains model accuracy, while inevitably leading to accuracy drops when being applied to MLPs. To marry the best of both worlds, we further propose a new mixture of experts (MoE) framework to reparameterize MLPs by taking multiplication or its primitives as experts, e.g., multiplication and shift, and designing a new latency-aware load-balancing loss. Such a loss helps to train a generic router for assigning a dynamic amount of input tokens to different experts according to their latency. In principle, the faster the experts run, the more input tokens they are assigned. Extensive experiments on various 2D/3D Transformer-based vision tasks consistently validate the effectiveness of our proposed ShiftAddViT, achieving up to $\textbf{5.18$\times$}$ latency reductions on GPUs and $\textbf{42.9}$% energy savings, while maintaining a comparable accuracy as original or efficient ViTs. Codes and models are available at https://github.com/GATECH-EIC/ShiftAddViT.

----

## [0] Optimal Extragradient-Based Algorithms for Stochastic Variational Inequalities with Separable Structure

**Authors**: *Angela Yuan, Chris Junchi Li, Gauthier Gidel, Michael I. Jordan, Quanquan Gu, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html)

**Abstract**:

We consider the problem of solving stochastic monotone variational inequalities with a separable structure using a stochastic first-order oracle. Building on standard extragradient for variational inequalities we propose a novel algorithm---stochastic \emph{accelerated gradient-extragradient} (AG-EG)---for strongly monotone variational inequalities (VIs). Our approach combines the strengths of extragradient and Nesterov acceleration. By showing that its iterates remain in a bounded domain and applying scheduled restarting, we prove that AG-EG has an optimal convergence rate for strongly monotone VIs. Furthermore, when specializing to the particular case of bilinearly coupled strongly-convex-strongly-concave saddle-point problems, including bilinear games, our algorithm achieves fine-grained convergence rates that match the respective lower bounds, with the stochasticity being characterized by an additive statistical error term that is optimal up to a constant prefactor.

----

## [0] Combinatorial Group Testing with Selfish Agents

**Authors**: *Georgios Chionas, Dariusz R. Kowalski, Piotr Krysta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69f98acf161316ed896047e45da3bc0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69f98acf161316ed896047e45da3bc0c-Abstract-Conference.html)

**Abstract**:

We study the Combinatorial Group Testing (CGT) problem in a novel game-theoretic framework, with a solution concept of Adversarial Equilibrium (AE). In this new framework, we have $n$ selfish agents corresponding to the elements of the universe $[n] =\{0,1,\ldots,n-1\}$ and a hidden set $K \subseteq [n]$ of active agents of size $|K| = k \ll n$. In each round of the game, each active agent decides if it is present in a query $Q \subseteq [n]$, and all agents receive feedback on $Q \cap K$. The goal of each active agent is to assure that its id could be learned from the feedback as early as possible. We present a comprehensive set of results in this new game, where we design and analyze adaptive algorithmic strategies of agents which are AE's. In particular, if $k$ is known to the agents, then we design adaptive AE strategies with provably near optimal learning time of $O(k \log(n/k))$. In the case of unknown $k$, we design an adaptive AE strategies with learning time of order $n^k$, and we prove a lower bound of $\Omega(n)$ on the learning time of any such algorithmic strategies. This shows a strong separations between the two models of known and unknown $k$, as well as between the classic CGT, i.e., without selfish agents, and our game theoretic CGT model.

----

## [0] A Hierarchical Spatial Transformer for Massive Point Samples in Continuous Space

**Authors**: *Wenchong He, Zhe Jiang, Tingsong Xiao, Zelin Xu, Shigang Chen, Ronald Fick, Miles Medina, Christine Angelini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a0480190bbe6b622c7f1d3aa9be9c0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a0480190bbe6b622c7f1d3aa9be9c0f-Abstract-Conference.html)

**Abstract**:

Transformers are widely used deep learning architectures. Existing transformers are mostly designed for sequences (texts or time series), images or videos, and graphs. This paper proposes a novel transformer model for massive (up to a million) point samples in continuous space. Such data are ubiquitous in environment sciences (e.g., sensor observations), numerical simulations (e.g., particle-laden flow, astrophysics), and location-based services (e.g., POIs and trajectories). However, designing a transformer for massive spatial points is non-trivial due to several challenges, including implicit long-range and multi-scale dependency on irregular points in continuous space, a non-uniform point distribution, the potential high computational costs of calculating all-pair attention across massive points, and the risks of over-confident predictions due to varying point density. To address these challenges, we propose a new hierarchical spatial transformer model, which includes multi-resolution representation learning within a quad-tree hierarchy and efficient spatial attention via coarse approximation. We also design an uncertainty quantification branch to estimate prediction confidence related to input feature noise and point sparsity. We provide a theoretical analysis of computational time complexity and memory costs. Extensive experiments on both real-world and synthetic datasets show that our method outperforms multiple baselines in prediction accuracy and our model can scale up to one million points on one NVIDIA A100 GPU. The code is available at https://github.com/spatialdatasciencegroup/HST

----

## [0] Rethinking Gauss-Newton for learning over-parameterized models

**Authors**: *Michael Arbel, Romain Menegaux, Pierre Wolinski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a14c7f9fb3f42645cfa6bd5aa446819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a14c7f9fb3f42645cfa6bd5aa446819-Abstract-Conference.html)

**Abstract**:

This work studies the global convergence and implicit bias of Gauss Newton's (GN) when optimizing over-parameterized one-hidden layer networks in the mean-field regime. We first establish a global convergence result for GN in the continuous-time limit exhibiting a faster convergence rate compared to GD due to improved conditioning. We then perform an empirical study on a synthetic regression task to investigate the implicit bias of GN's method.While GN is consistently faster than GD in finding a global optimum, the learned model generalizes well on test data when starting from random initial weights with a small variance and using a small step size to slow down convergence. Specifically, our study shows that such a setting results in a hidden learning phenomenon, where the dynamics are able to recover features with good generalization properties despite the model having sub-optimal training and test performances due to an under-optimized linear layer. This study exhibits a trade-off between the convergence speed of GN and the generalization ability of the learned solution.

----

## [0] Knowledge Distillation for High Dimensional Search Index

**Authors**: *Zepu Lu, Jin Chen, Defu Lian, Zaixi Zhang, Yong Ge, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a15378acabd1aef017ec79a3ed744d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a15378acabd1aef017ec79a3ed744d2-Abstract-Conference.html)

**Abstract**:

Lightweight compressed models are prevalent in Approximate Nearest Neighbor Search (ANNS) and Maximum Inner Product Search (MIPS) owing to their superiority of retrieval efficiency in large-scale datasets. However, results given by compressed methods are less accurate due to the curse of dimension and the limitations of optimization objectives (e.g., lacking interactions between queries and documents). Thus, we are encouraged to design a new learning algorithm for the compressed search index on high dimensions to improve retrieval performance. In this paper, we propose a novel KnowledgeDistillation for high dimensional search index framework (KDindex), with the aim of efficiently learning lightweight indexes by distilling knowledge from high-precision ANNS and MIPS models such as graph-based indexes. Specifically, the student is guided to keep the same ranking order of the top-k relevant results yielded by the teacher model, which acts as the additional supervision signals between queries and documents to learn the similarities between documents. Furthermore, to avoid the trivial solutions that all candidates are partitioned to the same centroid, the reconstruction loss that minimizes the compressed error, and the posting list balance strategy that equally allocates the candidates, are integrated into the learning objective. Experiment results demonstrate that KDindex outperforms existing learnable quantization-based indexes and is 40Ã— lighter than the state-of-the-art non-exhaustive methods while achieving comparable recall quality.

----

## [0] Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations

**Authors**: *Yu Cao, Jingrun Chen, Yixin Luo, Xiang Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a27ee6f66d13557f15f070274c51721-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a27ee6f66d13557f15f070274c51721-Abstract-Conference.html)

**Abstract**:

The diffusion model has shown remarkable success in computer vision, but it remains unclear whether the ODE-based probability flow or the SDE-based diffusion model is more superior and under what circumstances. Comparing the two is challenging due to dependencies on data distributions, score training, and other numerical issues. In this paper, we study the problem mathematically for two limiting scenarios: the zero diffusion (ODE) case and the large diffusion case. We first introduce a pulse-shape error to perturb the score function and analyze error accumulation of sampling quality, followed by a thorough analysis for generalization to arbitrary error. Our findings indicate that when the perturbation occurs at the end of the generative process, the ODE model outperforms the SDE model with a large diffusion coefficient. However, when the perturbation occurs earlier, the SDE model outperforms the ODE model, and we demonstrate that the error of sample generation due to such a pulse-shape perturbation is exponentially suppressed as the diffusion term's magnitude increases to infinity. Numerical validation of this phenomenon is provided using Gaussian, Gaussian mixture, and Swiss roll distribution, as well as realistic datasets like MNIST and CIFAR-10.

----

## [0] PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance

**Authors**: *Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Although large language models (LLMs) have shown great performance in natural language processing (NLP) in the financial domain, there are no publicly available financially tailored LLMs, instruction tuning datasets, and evaluation benchmarks, which is critical for continually pushing forward the open-source development of financial artificial intelligence (AI). This paper introduces PIXIU, a comprehensive framework including the first financial LLM based on fine-tuning LLaMA with instruction data, the first instruction data with 128K data samples to support the fine-tuning, and an evaluation benchmark with 8 tasks and 15 datasets. We first construct the large-scale multi-task instruction data considering a variety of financial tasks, financial document types, and financial data modalities. We then propose a financial LLM called FinMA by fine-tuning LLaMA with the constructed dataset to be able to follow instructions for various financial tasks. To support the evaluation of financial LLMs, we propose a standardized benchmark that covers a set of critical financial tasks, including six financial NLP tasks and two financial prediction tasks. With this benchmark, we conduct a detailed analysis of FinMA and several existing LLMs, uncovering their strengths and weaknesses in handling critical financial tasks. The model, datasets, benchmark, and experimental results are open-sourced to facilitate future research in financial AI.

----

## [0] EgoDistill: Egocentric Head Motion Distillation for Efficient Video Understanding

**Authors**: *Shuhan Tan, Tushar Nagarajan, Kristen Grauman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a412f0037b0df295a39a198666ea6a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a412f0037b0df295a39a198666ea6a6-Abstract-Conference.html)

**Abstract**:

Recent advances in egocentric video understanding models are promising, but their heavy computational expense is a barrier for many real-world applications. To address this challenge, we propose EgoDistill, a distillation-based approach that learns to reconstruct heavy ego-centric video clip features by combining the semantics from a sparse set of video frames with head motion from lightweight IMU readings. We further devise a novel IMU-based self-supervised pretraining strategy. Our method leads to significant improvements in efficiency, requiring 200Ã— fewer GFLOPs than equivalent video models. We demonstrate its effectiveness on the Ego4D and EPIC- Kitchens datasets, where our method outperforms state-of-the-art efficient video understanding methods.

----

## [0] Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework

**Authors**: *Weiduo Liao, Ying Wei, Mingchen Jiang, Qingfu Zhang, Hisao Ishibuchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a42b45af2b72e6e5b5e3a6fe695809f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a42b45af2b72e6e5b5e3a6fe695809f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Compositionality facilitates the comprehension of novel objects using acquired concepts and the maintenance of a knowledge pool. This is particularly crucial for continual learners to prevent catastrophic forgetting and enable compositionally forward transfer of knowledge. However, the existing state-of-the-art benchmarks inadequately evaluate the capability of compositional generalization, leaving an intriguing question unanswered. To comprehensively assess this capability, we introduce two vision benchmarks, namely Compositional GQA (CGQA) and Compositional OBJects365 (COBJ), along with a novel evaluation framework called Compositional Few-Shot Testing (CFST). These benchmarks evaluate the systematicity, productivity, and substitutivity aspects of compositional generalization. Experimental results on five baselines and two modularity-based methods demonstrate that current continual learning techniques do exhibit somewhat favorable compositionality in their learned feature extractors. Nonetheless, further efforts are required in developing modularity-based approaches to enhance compositional generalization. We anticipate that our proposed benchmarks and evaluation protocol will foster research on continual learning and compositionality.

----

## [0] Unsupervised Protein-Ligand Binding Energy Prediction via Neural Euler's Rotation Equation

**Authors**: *Wengong Jin, Siranush Sarkizova, Xun Chen, Nir Hacohen, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a45a1b0697ee086bd8bf494cacc6567-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a45a1b0697ee086bd8bf494cacc6567-Abstract-Conference.html)

**Abstract**:

Protein-ligand binding prediction is a fundamental problem in AI-driven drug discovery. Previous work focused on supervised learning methods for small molecules where binding affinity data is abundant, but it is hard to apply the same strategy to other ligand classes like antibodies where labelled data is limited. In this paper, we explore unsupervised approaches and reformulate binding energy prediction as a generative modeling task. Specifically, we train an energy-based model on a set of unlabelled protein-ligand complexes using SE(3) denoising score matching (DSM) and interpret its log-likelihood as binding affinity. Our key contribution is a new equivariant rotation prediction network called Neural Euler's Rotation Equations (NERE) for SE(3) DSM. It predicts a rotation by modeling the force and torque between protein and ligand atoms, where the force is defined as the gradient of an energy function with respect to atom coordinates. Using two protein-ligand and antibody-antigen binding affinity prediction benchmarks, we show that NERE outperforms all unsupervised baselines (physics-based potentials and protein language models) in both cases and surpasses supervised baselines in the antibody case.

----

## [0] ProteinNPT: Improving protein property prediction and design with non-parametric transformers

**Authors**: *Pascal Notin, Ruben Weitzman, Debora S. Marks, Yarin Gal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html)

**Abstract**:

Protein design holds immense potential for optimizing naturally occurring proteins, with broad applications in drug discovery, material design, and sustainability. However, computational methods for protein engineering are  confronted with significant challenges, such as an expansive design space, sparse functional regions, and a scarcity of available labels. These issues are further exacerbated in practice by the fact most real-life design scenarios necessitate the simultaneous optimization of multiple properties. In this work, we introduce ProteinNPT, a non-parametric transformer variant tailored to protein sequences and particularly suited to label-scarce and multi-task learning settings. We first focus on the supervised fitness prediction setting and develop several cross-validation schemes which support robust performance assessment. We subsequently reimplement prior top-performing baselines, introduce several extensions of these baselines by integrating diverse branches of the protein engineering literature, and demonstrate that ProteinNPT consistently outperforms all of them across a diverse set of protein property prediction tasks. Finally, we demonstrate the value of our approach for iterative protein design across extensive in silico Bayesian optimization and conditional sampling experiments.

----

## [0] Mitigating Source Bias for Fairer Weak Supervision

**Authors**: *Changho Shin, Sonia Cromp, Dyah Adila, Frederic Sala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html)

**Abstract**:

Weak supervision enables efficient development of training sets by reducing the need for ground truth labels. However, the techniques that make weak supervision attractive---such as integrating any source of signal to estimate unknown labels---also entail the danger that the produced pseudolabels are highly biased. Surprisingly, given everyday use and the potential for increased bias, weak supervision has not been studied from the point of view of fairness. We begin such a study, starting with the observation that even when a fair model can be built from a dataset with access to ground-truth labels, the corresponding dataset labeled via weak supervision can be arbitrarily unfair. To address this, we propose and empirically validate a model for source unfairness in weak supervision, then introduce a simple counterfactual fairness-based technique that can mitigate these biases. Theoretically, we show that it is possible for our approach to simultaneously improve both accuracy and fairness---in contrast to standard fairness approaches that suffer from tradeoffs. Empirically, we show that our technique improves accuracy on weak supervision baselines by as much as 32\% while reducing demographic parity gap by 82.5\%. A simple extension of our method aimed at maximizing performance produces state-of-the-art performance in five out of ten datasets in the WRENCH benchmark.

----

## [0] GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

**Authors**: *Xin Zheng, Miao Zhang, Chunyang Chen, Soheila Molaei, Chuan Zhou, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a55f024db3f771194bdadc8f3a35381-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a55f024db3f771194bdadc8f3a35381-Abstract-Conference.html)

**Abstract**:

Evaluating the performance of graph neural networks (GNNs) is an essential task for practical GNN model deployment and serving, as deployed GNNs face significant performance uncertainty when inferring on unseen and unlabeled test graphs, due to mismatched training-test graph distributions. In this paper, we study a new problem, GNN model evaluation, that aims to assess the performance of a specific GNN model trained on labeled and observed graphs, by precisely estimating its performance (e.g., node classification accuracy) on unseen graphs without labels. Concretely, we propose a two-stage GNN model evaluation framework, including (1) DiscGraph set construction and (2) GNNEvaluator training and inference. The DiscGraph set captures wide-range and diverse graph data distribution discrepancies through a discrepancy measurement function, which exploits the GNN outputs of latent node embeddings and node class predictions. Under the effective training supervision from the DiscGraph set, GNNEvaluator learns to precisely estimate node classification accuracy of the to-be-evaluated GNN model and makes an accurate inference for evaluating GNN model performance. Extensive experiments on real-world unseen and unlabeled test graphs demonstrate the effectiveness of our proposed method for GNN model evaluation.

----

## [0] Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures

**Authors**: *Runa Eschenhagen, Alexander Immer, Richard E. Turner, Frank Schneider, Philipp Hennig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6679e3d5b9f7d5f09cdb79a5fc3fd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6679e3d5b9f7d5f09cdb79a5fc3fd8-Abstract-Conference.html)

**Abstract**:

The core components of many modern neural network architectures, such as transformers, convolutional, or graph neural networks, can be expressed as linear layers with *weight-sharing*. Kronecker-Factored Approximate Curvature (K-FAC), a second-order optimisation method, has shown promise to speed up neural network training and thereby reduce computational costs. However, there is currently no framework to apply it to generic architectures, specifically ones with linear weight-sharing layers. In this work, we identify two different settings of linear weight-sharing layers which motivate two flavours of K-FAC -- *expand* and *reduce*. We show that they are exact for deep linear networks with weight-sharing in their respective setting. Notably, K-FAC-reduce is generally faster than K-FAC-expand, which we leverage to speed up automatic hyperparameter selection via optimising the marginal likelihood for a Wide ResNet. Finally, we observe little difference between these two K-FAC variations when using them to train both a graph neural network and a vision transformer. However, both variations are able to reach a fixed validation metric target in $50$-$75$\% of the number of steps of a first-order reference run, which translates into a comparable improvement in wall-clock time. This highlights the potential of applying K-FAC to modern neural network architectures.

----

## [0] Tanimoto Random Features for Scalable Molecular Machine Learning

**Authors**: *Austin Tripp, Sergio Bacallado, Sukriti Singh, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a69d44b3386e50c06f7107ef4f29302-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a69d44b3386e50c06f7107ef4f29302-Abstract-Conference.html)

**Abstract**:

The Tanimoto coefficient is commonly used to measure the similarity between molecules represented as discrete fingerprints,either as a distance metric or a positive definite kernel. While many kernel methods can be accelerated using random feature approximations, at present there is a lack of such approximations for the Tanimoto kernel. In this paper we propose two kinds of novel random features to allow this kernel to scale to large datasets, and in the process discover a novel extension of the kernel to real-valued vectors. We theoretically characterize these random features, and provide error bounds on the spectral norm of the Gram matrix. Experimentally, we show that these random features are effective at approximating the Tanimoto coefficient of real-world datasetsand are useful for molecular property prediction and optimization tasks. Future updates to this work will be available at http://arxiv.org/abs/2306.14809.

----

## [0] Probabilistic Inference in Reinforcement Learning Done Right

**Authors**: *Jean Tarbouriech, Tor Lattimore, Brendan O'Donoghue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6e010edde1b8f2812f558b67a1974e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6e010edde1b8f2812f558b67a1974e-Abstract-Conference.html)

**Abstract**:

A popular perspective in Reinforcement learning (RL) casts the problem as probabilistic inference on a graphical model of the Markov decision process (MDP). The core object of study is the probability of each state-action pair being visited under the optimal policy. Previous approaches to approximate this quantity can be arbitrarily poor, leading to algorithms that do not implement genuine statistical inference and consequently do not perform well in challenging problems. In this work, we undertake a rigorous Bayesian treatment of the posterior probability of state-action optimality and clarify how it flows through the MDP. We first reveal that this quantity can indeed be used to generate a policy that explores efficiently, as measured by regret. Unfortunately, computing it is intractable, so we derive a new variational Bayesian approximation yielding a tractable convex optimization problem and establish that the resulting policy also explores efficiently. We call our approach VAPOR and show that it has strong connections to Thompson sampling, K-learning, and maximum entropy exploration. We conclude with some experiments demonstrating the performance advantage of a deep RL version of VAPOR.

----

## [0] Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels

**Authors**: *Zhen Liu, Peitian Ma, Dongliang Chen, Wenbin Pei, Qianli Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6ecedac816a24f92ad1f444b1edcb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6ecedac816a24f92ad1f444b1edcb0-Abstract-Conference.html)

**Abstract**:

Deep Neural Networks (DNNs) have been criticized because they easily overfit noisy (incorrect) labels. To improve the robustness of DNNs, existing methods for image data regard samples with small training losses as correctly labeled data (small-loss criterion). Nevertheless, time series' discriminative patterns are easily distorted by external noises (i.e., frequency perturbations) during the recording process. This results in training losses of some time series samples that do not meet the small-loss criterion. Therefore, this paper proposes a deep learning paradigm called Scale-teaching to cope with time series noisy labels. Specifically, we design a fine-to-coarse cross-scale fusion mechanism for learning discriminative patterns by utilizing time series at different scales to train multiple DNNs simultaneously. Meanwhile, each network is trained in a cross-teaching manner by using complementary information from different scales to select small-loss samples as clean labels. For unselected large-loss samples, we introduce multi-scale embedding graph learning via label propagation to correct their labels by using selected clean samples. Experiments on multiple benchmark time series datasets demonstrate the superiority of the proposed Scale-teaching paradigm over state-of-the-art methods in terms of effectiveness and robustness.

----

## [0] VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning

**Authors**: *Jiayi Guan, Guang Chen, Jiaming Ji, Long Yang, Ao Zhou, Zhijun Li, Changjun Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a7c2a320f5f36bb98f8eb878c6f1180-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a7c2a320f5f36bb98f8eb878c6f1180-Abstract-Conference.html)

**Abstract**:

Offline safe reinforcement learning (RL) algorithms promise to learn policies that satisfy safety constraints directly in offline datasets without interacting with the environment. This arrangement is particularly important in scenarios with high sampling costs and potential dangers, such as autonomous driving and robotics. However, the influence of safety constraints and out-of-distribution (OOD) actions have made it challenging for previous methods to achieve high reward returns while ensuring safety. In this work, we propose a Variational Optimization with Conservative Eestimation algorithm (VOCE) to solve the problem of optimizing safety policies in the offline dataset. Concretely, we reframe the problem of offline safe RL using probabilistic inference, which introduces variational distributions to make the optimization of policies more flexible. Subsequently, we utilize pessimistic estimation methods to estimate the Q-value of cost and reward, which mitigates the extrapolation errors induced by OOD actions. Finally, extensive experiments demonstrate that the VOCE algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety.

----

## [0] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark

**Authors**: *Lasse Hansen, Nabeel Seedat, Mihaela van der Schaar, Andrija Petrovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6aa9a05b929fb08ff46a58cab6cf860d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6aa9a05b929fb08ff46a58cab6cf860d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthetic data serves as an alternative in training machine learning models, particularly when real-world data is limited or inaccessible. However, ensuring that synthetic data mirrors the complex nuances of real-world data is a challenging task. This paper addresses this issue by exploring the potential of integrating data-centric AI techniques which profile the data to guide the synthetic data generation process. Moreover, we shed light on the often ignored consequences of neglecting these data profiles during synthetic data generation --- despite seemingly high statistical fidelity. Subsequently, we propose a novel framework to evaluate the integration of data profiles to guide the creation of more representative synthetic data. In an empirical study, we evaluate the performance of five state-of-the-art models for tabular data generation on eleven distinct tabular datasets. The findings offer critical insights into the successes and limitations of current synthetic data generation techniques. Finally, we provide practical recommendations for integrating data-centric insights into the synthetic data generation process, with a specific focus on classification performance, model selection, and feature selection. This study aims to reevaluate conventional approaches to synthetic data generation and promote the application of data-centric AI techniques in improving the quality and effectiveness of synthetic data.

----

## [0] Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis

**Authors**: *Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html)

**Abstract**:

How can we tell whether two neural networks utilize the same internal processes for a particular computation? This question is pertinent for multiple subfields of neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of dynamics, and two networks performing the same computation with equivalent dynamics need not exhibit the same geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics, called Dynamical Similarity Analysis (DSA). Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare different systems passed through this embedding using a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. In four case studies, we demonstrate that our method disentangles conjugate and non-conjugate recurrent neural networks (RNNs), while geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method opens the door to comparative analyses of the essential temporal structure of computation in neural circuits.

----

## [0] H-nobs: Achieving Certified Fairness and Robustness in Distributed Learning on Heterogeneous Datasets

**Authors**: *Guanqiang Zhou, Ping Xu, Yue Wang, Zhi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ad5d39b10e37915d7dfda2893d8e924-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ad5d39b10e37915d7dfda2893d8e924-Abstract-Conference.html)

**Abstract**:

Fairness and robustness are two important goals in the design of modern distributed learning systems. Despite a few prior works attempting to achieve both fairness and robustness, some key aspects of this direction remain underexplored. In this paper, we try to answer three largely unnoticed and unaddressed questions that are of paramount significance to this topic: (i) What makes jointly satisfying fairness and robustness difficult? (ii) Is it possible to establish theoretical guarantee for the dual property of fairness and robustness? (iii) How much does fairness have to sacrifice at the expense of robustness being incorporated into the system? To address these questions, we first identify data heterogeneity as the key difficulty of combining fairness and robustness. Accordingly, we propose a fair and robust framework called H-nobs which can offer certified fairness and robustness through the adoption of two key components, a fairness-promoting objective function and a simple robust aggregation scheme called norm-based screening (NBS). We explain in detail why NBS is the suitable scheme in our algorithm in contrast to other robust aggregation measures. In addition, we derive three convergence theorems for H-nobs in cases of the learning model being nonconvex, convex, and strongly convex respectively, which provide theoretical guarantees for both fairness and robustness. Further, we empirically investigate the influence of the robust mechanism (NBS) on the fairness performance of H-nobs, the very first attempt of such exploration.

----

## [0] A Randomized Approach to Tight Privacy Accounting

**Authors**: *Jiachen T. Wang, Saeed Mahloujifar, Tong Wu, Ruoxi Jia, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html)

**Abstract**:

Bounding privacy leakage over compositions, i.e., privacy accounting, is a key challenge in differential privacy (DP). However, the privacy parameter ($\varepsilon$ or $\delta$) is often easy to estimate but hard to bound. In this paper, we propose a new differential privacy paradigm called estimate-verify-release (EVR), which tackles the challenges of providing a strict upper bound for the privacy parameter in DP compositions by converting an *estimate* of privacy parameter into a formal guarantee. The EVR paradigm first verifies whether the mechanism meets the *estimated* privacy guarantee, and then releases the query output based on the verification result. The core component of the EVR is privacy verification. We develop a randomized privacy verifier using Monte Carlo (MC) technique. Furthermore, we propose an MC-based DP accountant that outperforms existing DP accounting techniques in terms of accuracy and efficiency. MC-based DP verifier and accountant is applicable to an important and commonly used class of DP algorithms, including the famous DP-SGD. An empirical evaluation shows the proposed EVR paradigm improves the utility-privacy tradeoff for privacy-preserving machine learning.

----

## [0] Triple Eagle: Simple, Fast and Practical Budget-Feasible Mechanisms

**Authors**: *Kai Han, You Wu, He Huang, Shuang Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html)

**Abstract**:

We revisit the classical problem of designing Budget-Feasible Mechanisms (BFMs) for submodular valuation functions, which has been extensively studied since the seminal paper of Singer [FOCSâ€™10] due to its wide applications in crowdsourcing and social marketing. We propose TripleEagle, a novel algorithmic framework for designing BFMs, based on which we present several simple yet effective BFMs thatachieve better approximation ratios than the state-of-the-art work for both monotone and non-monotone submodular valuation functions. Moreover, our BFMs are the first in the literature to achieve linear complexities while ensuring obvious strategyproofness, making them more practical than the previous BFMs. We conduct extensive experiments to evaluate the empirical performance of our BFMs, and the experimental results strongly demonstrate the efficiency and effectiveness of our approach.

----

## [0] VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models

**Authors**: *Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html)

**Abstract**:

Diffusion Models (DMs) are state-of-the-art generative models that learn a reversible corruption process from iterative noise addition and denoising. They are the backbone of many generative AI applications, such as text-to-image conditional generation. However, recent studies have shown that basic unconditional DMs (e.g., DDPM and DDIM) are vulnerable to backdoor injection, a type of output manipulation attack triggered by a maliciously embedded pattern at model input. This paper presents a unified backdoor attack framework (VillanDiffusion) to expand the current scope of backdoor analysis for DMs. Our framework covers mainstream unconditional and conditional DMs (denoising-based and score-based) and various training-free samplers for holistic evaluations. Experiments show that our unified framework facilitates the backdoor analysis of different DM configurations and provides new insights into caption-based backdoor attacks on DMs.

----

## [0] An Information Theory Perspective on Variance-Invariance-Covariance Regularization

**Authors**: *Ravid Shwartz-Ziv, Randall Balestriero, Kenji Kawaguchi, Tim G. J. Rudner, Yann LeCun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract-Conference.html)

**Abstract**:

Variance-Invariance-Covariance Regularization (VICReg) is a self-supervised learning (SSL) method that has shown promising results on a variety of tasks. However, the fundamental mechanisms underlying VICReg remain unexplored. In this paper, we present an information-theoretic perspective on the VICReg objective. We begin by deriving information-theoretic quantities for deterministic networks as an alternative to unrealistic stochastic network assumptions. We then relate the optimization of the VICReg objective to mutual information optimization, highlighting underlying assumptions and facilitating a constructive comparison with other SSL algorithms and derive a generalization bound for VICReg, revealing its inherent advantages for downstream tasks. Building on these results, we introduce a family of SSL methods derived from information-theoretic principles that outperform existing SSL techniques.

----

## [0] Learning and processing the ordinal information of temporal sequences in recurrent neural circuits

**Authors**: *Xiaolong Zou, Zhikun Chu, Qinghai Guo, Jie Cheng, Bo Ho, Si Wu, Yuanyuan Mi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html)

**Abstract**:

Temporal sequence processing is fundamental in brain cognitive functions. Experimental data has indicated that the representations of ordinal information and contents of temporal sequences are disentangled in the brain, but the neural mechanism underlying this disentanglement remains largely unclear. Here, we investigate how recurrent neural circuits learn to represent the abstract order structure of temporal sequences, and how this disentangled representation of order structure from that of contents facilitates the processing of temporal sequences. We show that with an appropriate learn protocol, a recurrent neural circuit can learn a set of tree-structured attractor states to encode the corresponding tree-structured orders of given temporal sequences. This abstract temporal order template can then be bound with different contents, allowing for flexible and robust temporal sequence processing. Using a transfer learning task, we demonstrate that the reuse of a temporal order template facilitates the acquisition of new temporal sequences of the same or similar ordinal structure. Using a key-word spotting task, we demonstrate that the attractor representation of order structure improves the robustness of temporal sequence discrimination, if the ordinal information is the key to differentiate different sequences. We hope this study gives us insights into the neural mechanism of representing the ordinal information of temporal sequences in the brain, and helps us to develop brain-inspired temporal sequence processing algorithms.

----

## [0] UNSSOR: Unsupervised Neural Speech Separation by Leveraging Over-determined Training Mixtures

**Authors**: *Zhong-Qiu Wang, Shinji Watanabe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b44765c9201730a27f7931afb4d7434-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b44765c9201730a27f7931afb4d7434-Abstract-Conference.html)

**Abstract**:

In reverberant conditions with multiple concurrent speakers, each microphone acquires a mixture signal of multiple speakers at a different location. In over-determined conditions where the microphones out-number speakers, we can narrow down the solutions to speaker images and realize unsupervised speech separation by leveraging each mixture signal as a constraint (i.e., the estimated speaker images at a microphone should add up to the mixture). Equipped with this insight, we propose UNSSOR, an algorithm for $\underline{u}$nsupervised $\underline{n}$eural $\underline{s}$peech $\underline{s}$eparation by leveraging $\underline{o}$ver-determined training mixtu$\underline{r}$es. At each training step, we feed an input mixture to a deep neural network (DNN) to produce an intermediate estimate for each speaker, linearly filter the estimates, and optimize a loss so that, at each microphone, the filtered estimates of all the speakers can add up to the mixture to satisfy the above constraint. We show that this loss can promote unsupervised separation of speakers. The linear filters are computed in each sub-band based on the mixture and DNN estimates through the forward convolutive prediction (FCP) algorithm. To address the frequency permutation problem incurred by using sub-band FCP, a loss term based on minimizing intra-source magnitude scattering is proposed. Although UNSSOR requires over-determined training mixtures, we can train DNNs to achieve under-determined separation (e.g., unsupervised monaural speech separation). Evaluation results on two-speaker separation in reverberant conditions show the effectiveness and potential of UNSSOR.

----

## [0] Improving Self-supervised Molecular Representation Learning using Persistent Homology

**Authors**: *Yuankai Luo, Lei Shi, Veronika Thost*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b555e8552240d6dfe0767146c9ebf36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b555e8552240d6dfe0767146c9ebf36-Abstract-Conference.html)

**Abstract**:

Self-supervised learning (SSL) has great potential for molecular representation learning given the complexity of molecular graphs, the large amounts of unlabelled data available, the considerable cost of obtaining labels experimentally, and the hence often only small training datasets. The importance of the topic is reflected in the variety of paradigms and architectures that have been investigated recently, most focus on designing views for contrastive learning.In this paper, we study SSL based on persistent homology (PH), a mathematical tool for modeling topological features of data that persist across multiple scales. It has several unique features which particularly suit SSL, naturally offering: different views of the data, stability in terms of distance preservation, and the opportunity to flexibly incorporate domain knowledge.We (1) investigate an autoencoder, which shows the general representational power of PH, and (2) propose a contrastive loss that complements existing approaches. We rigorously evaluate our approach for molecular property prediction and demonstrate its particular features in improving the embedding space:after SSL, the representations are better and offer considerably more predictive power than the baselines over different probing tasks; our loss increases baseline performance, sometimes largely; and we often obtain substantial improvements over very small datasets, a common scenario in practice.

----

## [0] Characteristic Circuits

**Authors**: *Zhongjie Yu, Martin Trapp, Kristian Kersting*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b61c278e483954fee502b49fe71cd14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b61c278e483954fee502b49fe71cd14-Abstract-Conference.html)

**Abstract**:

In many real-world scenarios it is crucial to be able to reliably and efficiently reason under uncertainty while capturing complex relationships in data.  Probabilistic circuits (PCs), a prominent family of tractable probabilistic models, offer a remedy to this challenge by composing simple, tractable distributions into a high-dimensional probability distribution.   However, learning PCs on heterogeneous data is challenging and densities of some parametric distributions are not available in closed form, limiting their potential use.   We introduce characteristic circuits (CCs), a family of tractable probabilistic models providing a unified formalization of distributions over heterogeneous data in the spectral domain.  The one-to-one relationship between characteristic functions and probability measures enables us to learn high-dimensional distributions on heterogeneous data domains and facilitates efficient probabilistic inference even when no closed-form density function is available.   We show that the structure and parameters of CCs can be learned efficiently from the data and find that CCs outperform state-of-the-art density estimators for heterogeneous data domains on common benchmark data sets.

----

## [0] Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds

**Authors**: *Paul Rosa, Slava Borovitskiy, Alexander Terenin, Judith Rousseau*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b7676588c33d344485eeba1b5653ab1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b7676588c33d344485eeba1b5653ab1-Abstract-Conference.html)

**Abstract**:

Gaussian processes are used in many machine learning applications that rely on uncertainty quantification. Recently, computational tools for working with these models in geometric settings, such as when inputs lie on a Riemannian manifold, have been developed. This raises the question: can these intrinsic models be shown theoretically to lead to better performance, compared to simply embedding all relevant quantities into $\mathbb{R}^d$ and using the restriction of an ordinary Euclidean Gaussian process? To study this, we prove optimal contraction rates for intrinsic Matérn Gaussian processes defined on compact Riemannian manifolds. We also prove analogous rates for extrinsic processes using trace and extension theorems between manifold and ambient Sobolev spaces: somewhat surprisingly, the rates obtained turn out to coincide with those of the intrinsic processes, provided that their smoothness parameters are matched appropriately. We illustrate these rates empirically on a number of examples, which, mirroring prior work, show that intrinsic processes can achieve better performance in practice. Therefore, our work shows that finer-grained analyses are needed to distinguish between different levels of data-efficiency of geometric Gaussian processes, particularly in settings which involve small data set sizes and non-asymptotic behavior.

----

## [0] Causal Context Connects Counterfactual Fairness to Robust Prediction and Group Fairness

**Authors**: *Jacy Reese Anthis, Victor Veitch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b7e1e96243c9edc378f85e7d232e415-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b7e1e96243c9edc378f85e7d232e415-Abstract-Conference.html)

**Abstract**:

Counterfactual fairness requires that a person would have been classified in the same way by an AI or other algorithmic system if they had a different protected class, such as a different race or gender. This is an intuitive standard, as reflected in the U.S. legal system, but its use is limited because counterfactuals cannot be directly observed in real-world data. On the other hand, group fairness metrics (e.g., demographic parity or equalized odds) are less intuitive but more readily observed. In this paper, we use \textit{causal context} to bridge the gaps between counterfactual fairness, robust prediction, and group fairness. First, we motivate counterfactual fairness by showing that there is not necessarily a fundamental trade-off between fairness and accuracy because, under plausible conditions, the counterfactually fair predictor is in fact accuracy-optimal in an unbiased target distribution. Second, we develop a correspondence between the causal graph of the data-generating process and which, if any, group fairness metrics are equivalent to counterfactual fairness. Third, we show that in three common fairness contexts—measurement error, selection on label, and selection on predictors—counterfactual fairness is equivalent to demographic parity, equalized odds, and calibration, respectively. Counterfactual fairness can sometimes be tested by measuring relatively simple group fairness metrics.

----

## [0] Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance

**Authors**: *Congyue Deng, Jiahui Lei, William B. Shen, Kostas Daniilidis, Leonidas J. Guibas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html)

**Abstract**:

Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.

----

## [0] Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents

**Authors**: *Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html)

**Abstract**:

In this paper, we study the problem of planning in Minecraft, a popular, democratized yet challenging open-ended environment for developing multi-task embodied agents. We've found two primary challenges of empowering such agents with planning: 1) planning in an open-ended world like Minecraft requires precise and multi-step reasoning due to the long-term nature of the tasks, and 2) as vanilla planners do not consider the achievability of the current agent when ordering parallel sub-goals within a complicated plan, the resulting plan could be inefficient. To this end, we propose ``$\underline{D}$escribe, $\underline{E}$xplain, $\underline{P}$lan and $\underline{S}$elect'' ($\textbf{DEPS}$), an interactive planning approach based on Large Language Models (LLMs). Our approach helps with better error correction from the feedback during the long-haul planning, while also bringing the sense of proximity via goal $\textbf{Selector}$, a learnable module that ranks parallel sub-goals based on the estimated steps of completion and improves the original plan accordingly. Our experiments mark the milestone of the first zero-shot multi-task agent that can robustly accomplish 70+ Minecraft tasks and nearly double the overall performances. Further testing reveals our method's general effectiveness in popularly adopted non-open-ended domains as well (i.e., ALFWorld and tabletop manipulation). The ablation and exploratory studies detail how our design beats the counterparts and provide a promising update on the $\texttt{ObtainDiamond}$ grand challenge with our approach.

----

## [0] Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage

**Authors**: *Yuheng Jia, Fuchao Yang, Yongqiang Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html)

**Abstract**:

In partial label learning (PLL), each sample is associated with a group of candidate labels, among which only one label is correct. The key of PLL is to disambiguate the candidate label set to find the ground-truth label. To this end, we first construct a constrained regression model to capture the confidence of the candidate labels, and multiply the label confidence matrix by its transpose to build a second-order similarity matrix, whose elements indicate the pairwise similarity relationships of samples globally. Then we develop a semantic dissimilarity matrix by considering the complement of the intersection of the candidate label set, and further propagate the initial dissimilarity relationships to the whole data set by leveraging the local geometric structure of samples. The similarity and dissimilarity matrices form an adversarial relationship, which is further utilized to shrink the solution space of the label confidence matrix and promote the dissimilarity matrix. We finally extend the proposed model to a kernel version to exploit the non-linear structure of samples and solve the proposed model by the inexact augmented Lagrange multiplier method. By exploiting the adversarial prior, the proposed method can significantly outperformstate-of-the-art PLL algorithms when evaluated on 10 artificial and 7 real-world partial label data sets. We also prove the effectiveness of our method with some theoretical guarantees. The code is publicly available at https://github.com/Yangfc-ML/DPCLS.

----

## [0] Data Selection for Language Models via Importance Resampling

**Authors**: *Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html)

**Abstract**:

Selecting a suitable pretraining dataset is crucial for both general-domain (e.g., GPT-3) and domain-specific (e.g., Codex) language models (LMs). We formalize this problem as selecting a subset of a large raw unlabeled dataset to match a desired target distribution given unlabeled target samples. Due to the scale and dimensionality of the raw text data, existing methods use simple heuristics or require human experts to manually curate data. Instead, we extend the classic importance resampling approach used in low-dimensions for LM data selection. We propose Data Selection with Importance Resampling (DSIR), an efficient and scalable framework that estimates importance weights in a reduced feature space for tractability and selects data with importance resampling according to these weights. We instantiate the DSIR framework with hashed n-gram features for efficiency, enabling the selection of 100M documents from the full Pile dataset in 4.5 hours. To measure whether hashed n-gram features preserve the aspects of the data that are relevant to the target, we define KL reduction, a data metric that measures the proximity between the selected pretraining data and the target on some feature space. Across 8 data selection methods (including expert selection), KL reduction on hashed n-gram features highly correlates with average downstream accuracy (r=0.82). When selecting data for continued pretraining on a specific domain, DSIR performs comparably to expert curation across 8 target distributions. When pretraining general-domain models (target is Wikipedia and books), DSIR improves over random selection and heuristic filtering baselines by 2--2.5% on the GLUE benchmark.

----

## [0] Video Dynamics Prior: An Internal Learning Approach for Robust Video Enhancements

**Authors**: *Gaurav Shrivastava, Ser Nam Lim, Abhinav Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ba85c6f1c7656a6a647bc4d63b90bf0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ba85c6f1c7656a6a647bc4d63b90bf0-Abstract-Conference.html)

**Abstract**:

In this paper, we present a novel robust framework for low-level vision tasks, including denoising, object removal, frame interpolation, and super-resolution, that does not require any external training data corpus. Our proposed approach directly learns the weights of neural modules by optimizing over the corrupted test sequence, leveraging the spatio-temporal coherence and internal statistics of videos. Furthermore, we introduce a novel spatial pyramid loss that leverages the property of spatio-temporal patch recurrence in a video across the different scales of the video. This loss enhances robustness to unstructured noise in both the spatial and temporal domains. This further results in our framework being highly robust to degradation in input frames and yields state-of-the-art results on downstream tasks such as denoising, object removal, and frame interpolation. To validate the effectiveness of our approach, we conduct qualitative and quantitative evaluations on standard video datasets such as DAVIS, UCF-101, and VIMEO90K-T.

----

## [0] Glance and Focus: Memory Prompting for Multi-Event Video Question Answering

**Authors**: *Ziyi Bai, Ruiping Wang, Xilin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html)

**Abstract**:

Video Question Answering (VideoQA) has emerged as a vital tool to evaluate agentsâ€™ ability to understand human daily behaviors. Despite the recent success of large vision language models in many multi-modal tasks, complex situation reasoning over videos involving multiple human-object interaction events still remains challenging. In contrast, humans can easily tackle it by using a series of episode memories as anchors to quickly locate question-related key moments for reasoning. To mimic this effective reasoning strategy, we propose the Glance- Focus model. One simple way is to apply an action detection model to predict a set of actions as key memories. However, these actions within a closed set vocabulary are hard to generalize to various video domains. Instead of that, we train an Encoder-Decoder to generate a set of dynamic event memories at the glancing stage. Apart from using supervised bipartite matching to obtain the event memories, we further design an unsupervised memory generation method to get rid of dependence on event annotations. Next, at the focusing stage, these event memories act as a bridge to establish the correlation between the questions with high-level event concepts and low-level lengthy video content. Given the question, the model first focuses on the generated key event memory, then focuses on the most relevant moment for reasoning through our designed multi-level cross- attention mechanism. We conduct extensive experiments on four Multi-Event VideoQA benchmarks including STAR, EgoTaskQA, AGQA, and NExT-QA. Our proposed model achieves state-of-the-art results, surpassing current large models in various challenging reasoning tasks. The code and models are available at https://github.com/ByZ0e/Glance-Focus.

----

## [0] Learning To Dive In Branch And Bound

**Authors**: *Max B. Paulus, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6bbda0824bcc20749f21510fd8b28de5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6bbda0824bcc20749f21510fd8b28de5-Abstract-Conference.html)

**Abstract**:

Primal heuristics are important for solving mixed integer linear programs, because they find feasible solutions that facilitate branch and bound search. A prominent group of primal heuristics are diving heuristics. They iteratively modify and resolve linear programs to conduct a depth-first search from any node in the search tree. Existing divers rely on generic decision rules that fail to exploit structural commonality between similar problem instances that often arise in practice. Therefore, we propose L2Dive to learn specific diving heuristics with graph neural networks: We train generative models to predict variable assignments and leverage the duality of linear programs to make diving decisions based on the model's predictions. L2Dive is fully integrated into the open-source solver SCIP. We find that L2Dive outperforms standard divers to find better feasible solutions on a range of combinatorial optimization problems. For real-world applications from server load balancing and neural network verification, L2Dive improves the primal-dual integral by up to 7% (35%) on average over a tuned (default) solver baseline and reduces average solving time by 20% (29%).

----

## [0] Intriguing Properties of Quantization at Scale

**Authors**: *Arash Ahmadian, Saurabh Dash, Hongyu Chen, Bharat Venkitesh, Stephen Zhen Gou, Phil Blunsom, Ahmet Üstün, Sara Hooker*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html)

**Abstract**:

Emergent properties have been widely adopted as a term to describe behavior not present in smaller models but observed in larger models  (Wei et al., 2022a). Recent work suggests that the trade-off incurred by quantization is also an emergent property, with sharp drops in performance in models over 6B parameters. In this work, we ask are quantization cliffs in performance solely a factor of scale? Against a backdrop of increased research focus on why certain emergent properties surface at scale, this work provides a useful counter-example. We posit that it is possible to optimize for a quantization friendly training recipe that suppresses large activation magnitude outliers. Here, we find that outlier dimensions are not an inherent product of scale, but rather sensitive to the optimization conditions present during pre-training. This both opens up directions for more efficient quantization, and poses the question of whether other emergent properties are inherent or can be altered and conditioned by optimization and architecture design choices. We successfully quantize models ranging in size from 410M to 52B with minimal degradation in performance.

----

## [0] Self-supervised Graph Neural Networks via Low-Rank Decomposition

**Authors**: *Liang Yang, Runjie Shi, Qiuliang Zhang, Bingxin Niu, Zhen Wang, Xiaochun Cao, Chuan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c33e4ea4ddfb05a78541022ab5a1fb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c33e4ea4ddfb05a78541022ab5a1fb9-Abstract-Conference.html)

**Abstract**:

Self-supervised learning is introduced to train graph neural networks (GNNs) by employing propagation-based GNNs designed for semi-supervised learning tasks. Unfortunately, this common choice tends to cause two serious issues. Firstly, global parameters cause the model lack the ability to capture the local property. Secondly, it is difficult to handle networks beyond homophily without label information.This paper tends to break through the common choice of employing propagation-based GNNs, which aggregate representations of nodes belonging to different classes and tend to lose discriminative information. If the propagation in each ego-network is just between the nodes from the same class, the obtained representation matrix should follow the low-rank characteristic. To meet this requirement, this paper proposes the Low-Rank Decomposition-based GNNs (LRD-GNN-Matrix) by employing Low-Rank Decomposition to the attribute matrix. Furthermore, to incorporate long-distance information, Low-Rank Tensor Decomposition-based GNN (LRD-GNN-Tensor) is proposed by constructing the node attribute tensor from selected similar ego-networks and performing Low-Rank Tensor Decomposition. The employed tensor nuclear norm facilitates the capture of the long-distance relationship between original and selected similar ego-networks. Extensive experiments demonstrate the superior performance and the robustness  of  LRD-GNNs.

----

## [0] Cookie Consent Has Disparate Impact on Estimation Accuracy

**Authors**: *Erik Miehling, Rahul Nair, Elizabeth Daly, Karthikeyan Natesan Ramamurthy, Robert Redmond*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html)

**Abstract**:

Cookies are designed to enable more accurate identification and tracking of user behavior, in turn allowing for more personalized ads and better performing ad campaigns. Given the additional information that is recorded, questions related to privacy and fairness naturally arise. How does a user's consent decision influence how much the system can learn about their demographic and tastes? Is the impact of a user's consent decision on the recommender system's ability to learn about their latent attributes uniform across demographics? We investigate these questions in the context of an engagement-driven recommender system using simulation. We empirically demonstrate that when consent rates exhibit demographic-dependence, user consent has a disparate impact on the recommender agent's ability to estimate users' latent attributes. In particular, we find that when consent rates are demographic-dependent, a user disagreeing to share their cookie may counter-intuitively cause the recommender agent to know more about the user than if the user agreed to share their cookie. Furthermore, the gap in base consent rates across demographics serves as an amplifier: users from the lower consent rate demographic who agree to cookie sharing generally experience higher estimation errors than the same users from the higher consent rate demographic, and conversely for users who choose to disagree to cookie sharing, with these differences increasing in consent rate gap. We discuss the need for new notions of fairness that encourage consistency between a user's privacy decisions and the system's ability to estimate their latent attributes.

----

## [0] NPCL: Neural Processes for Uncertainty-Aware Continual Learning

**Authors**: *Saurav Jha, Dong Gong, He Zhao, Lina Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c4a1a3cbe70ef36d7d6332166bba77d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c4a1a3cbe70ef36d7d6332166bba77d-Abstract-Conference.html)

**Abstract**:

Continual learning (CL) aims to train deep neural networks efficiently on streaming data while limiting the forgetting caused by new tasks.  However, learning transferable knowledge with less interference between tasks is difficult, and real-world deployment of CL models is limited by their inability to measure predictive uncertainties. To address these issues, we propose handling CL tasks with neural processes (NPs), a class of meta-learners that encode different tasks into probabilistic distributions over functions all while providing reliable uncertainty estimates. Specifically, we propose an NP-based CL approach (NPCL) with task-specific modules arranged in a hierarchical latent variable model. We tailor regularizers on the learned latent distributions to alleviate forgetting. The uncertainty estimation capabilities of the NPCL can also be used to handle the task head/module inference challenge in CL. Our experiments show that the NPCL  outperforms previous CL approaches. We validate the effectiveness of uncertainty estimation in the NPCL for identifying novel data and evaluating instance-level model confidence. Code is available at https://github.com/srvCodes/NPCL.

----

## [0] From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces

**Authors**: *Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, Kristina Toutanova*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c52a8a4fadc9129c6e1d1745f2dfd0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c52a8a4fadc9129c6e1d1745f2dfd0f-Abstract-Conference.html)

**Abstract**:

Much of the previous work towards digital agents for graphical user interfaces (GUIs) has relied on text-based representations (derived from HTML or other structured data sources), which are not always readily available. These input representations have been often coupled with custom, task-specific action spaces.  This paper focuses on creating agents that interact with the digital world using the same conceptual interface that humans commonly use â€” via pixel-based screenshots and a generic action space corresponding to keyboard and mouse actions. Building upon recent progress in pixel-based pretraining, we show, for the first time, that it is possible for such agents to outperform human crowdworkers on the MiniWob++ benchmark of GUI-based instruction following tasks.

----

## [0] Robust Mean Estimation Without Moments for Symmetric Distributions

**Authors**: *Gleb Novikov, David Steurer, Stefan Tiegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c59ace4fc4872a14df13d91762ad4f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c59ace4fc4872a14df13d91762ad4f0-Abstract-Conference.html)

**Abstract**:

We study the problem of robustly estimating the mean or location parameter without moment assumptions.Known computationally efficient algorithms rely on strong distributional assumptions, such as sub-Gaussianity, or (certifiably) bounded moments.Moreover, the guarantees that they achieve in the heavy-tailed setting are weaker than those for sub-Gaussian distributions with known covariance.In this work, we show that such a tradeoff, between error guarantees and heavy-tails, is not necessary for symmetric distributions.We show that for a large class of symmetric distributions, the same error as in the Gaussian setting can be achieved efficiently.The distributions we study include products of arbitrary symmetric one-dimensional distributions, such as product Cauchy distributions, as well as elliptical distributions, a vast generalization of the Gaussian distribution.For product distributions and elliptical distributions with known scatter (covariance) matrix, we show that given an $\varepsilon$-corrupted sample, we can with probability at least $1-\delta$ estimate its location up to error $O(\varepsilon \sqrt{\log(1/\varepsilon)})$ using $\tfrac{d\log(d) + \log(1/\delta)}{\varepsilon^2 \log(1/\varepsilon)}$ samples.This result matches the best-known guarantees for the Gaussian distribution and known SQ lower bounds (up to the $\log(d)$ factor).For elliptical distributions with unknown scatter (covariance) matrix, we propose a sequence of efficient algorithms that approaches this optimal error.Specifically, for every $k \in \mathbb{N}$, we design an estimator using time and samples $\tilde{O}({d^k})$ achieving error $O(\varepsilon^{1-\frac{1}{2k}})$.This matches the error and running time guarantees when assuming certifiably bounded moments of order up to $k$.For unknown covariance, such error bounds of $o(\sqrt{\varepsilon})$ are not even known for (general) sub-Gaussian distributions.Our algorithms are based on a generalization of the well-known filtering technique [DK22].More specifically, we show how this machinery can be combined with Huber-loss-based techniques to work with projections of the noise that behave more nicely than the initial noise.Moreover, we show how sum-of-squares proofs can be used to obtain algorithmic guarantees even for distributions without a first moment.We believe that this approach may find other applications in future works.

----

## [0] Fast and Simple Spectral Clustering in Theory and Practice

**Authors**: *Peter Macgregor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c5b82193c5d8e6aa5806239676ddc97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c5b82193c5d8e6aa5806239676ddc97-Abstract-Conference.html)

**Abstract**:

Spectral clustering is a popular and effective algorithm designed to find $k$ clusters in a graph $G$.In the classical spectral clustering algorithm, the vertices of $G$ are embedded into $\mathbb{R}^k$ using $k$ eigenvectors of the graph Laplacian matrix.However, computing this embedding is computationally expensive and dominates the running time of the algorithm.In this paper, we present a simple spectral clustering algorithm based on a vertex embedding with $O(\log(k))$ vectors computed by the power method.The vertex embedding is computed in nearly-linear time with respect to the size of the graph, andthe algorithm provably recovers the ground truth clusters under natural assumptions on the input graph.We evaluate the new algorithm on several synthetic and real-world datasets, finding that it is significantly faster than alternative clustering algorithms,while producing results with approximately the same clustering accuracy.

----

## [0] CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation

**Authors**: *Xiuzhe Wu, Peng Dai, Weipeng Deng, Handi Chen, Yang Wu, Yan-Pei Cao, Ying Shan, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html)

**Abstract**:

Existing methods for adapting Neural Radiance Fields (NeRFs) to scene changes require extensive data capture and model retraining, which is both time-consuming and labor-intensive. In this paper, we tackle the challenge of efficiently adapting NeRFs to real-world scene changes over time using a few new images while retaining the memory of unaltered areas, focusing on the continual learning aspect of NeRFs. To this end, we propose CL-NeRF, which consists of two key components: a lightweight expert adaptor for adapting to new changes and evolving scene representations and a conflict-aware knowledge distillation learning objective for memorizing unchanged parts. We also present a new benchmark for evaluating Continual Learning of NeRFs with comprehensive metrics. Our extensive experiments demonstrate that CL-NeRF can synthesize high-quality novel views of both changed and unchanged regions with  high training efficiency, surpassing existing methods in terms of reducing forgetting and adapting to changes. Code and benchmark will be made available.

----

## [0] Generalised f-Mean Aggregation for Graph Neural Networks

**Authors**: *Ryan Kortvelesy, Steven D. Morad, Amanda Prorok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c78ae0c1140902bf3a430b1725bcc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c78ae0c1140902bf3a430b1725bcc4e-Abstract-Conference.html)

**Abstract**:

Graph Neural Network (GNN) architectures are defined by their implementations of update and aggregation modules. While many works focus on new ways to parametrise the update modules, the aggregation modules receive comparatively little attention. Because it is difficult to parametrise aggregation functions, currently most methods select a ``standard aggregator'' such as mean, sum, or max. While this selection is often made without any reasoning, it has been shown that the choice in aggregator has a significant impact on performance, and the best choice in aggregator is problem-dependent. Since aggregation is a lossy operation, it is crucial to select the most appropriate aggregator in order to minimise information loss. In this paper, we present GenAgg, a generalised aggregation operator, which parametrises a function space that includes all standard aggregators. In our experiments, we show that GenAgg is able to represent the standard aggregators with much higher accuracy than baseline methods. We also show that using GenAgg as a drop-in replacement for an existing aggregator in a GNN often leads to a significant boost in performance across various tasks.

----

## [0] Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization

**Authors**: *Mahyar Fazlyab, Taha Entesari, Aniket Roy, Rama Chellappa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c7ca1889f01a9b767c631686fb5fd24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c7ca1889f01a9b767c631686fb5fd24-Abstract-Conference.html)

**Abstract**:

To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. In this paper, we propose a differentiable regularizer that is a lower bound on the distance of the data points to the classification boundary. The proposed regularizer requires knowledge of the model's Lipschitz constant along certain directions. To this end, we develop a scalable method for calculating guaranteed differentiable upper bounds on the Lipschitz constant of neural networks accurately and efficiently.  The relative accuracy of the bounds prevents excessive regularization and allows for more direct manipulation of the decision boundary. Furthermore, our Lipschitz bounding algorithm exploits the monotonicity and Lipschitz continuity of the activation layers, and the resulting bounds can be used to design new layers with controllable bounds on their Lipschitz constant. Experiments on the MNIST, CIFAR-10, and Tiny-ImageNet data sets verify that our proposed algorithm obtains competitively improved results compared to the state-of-the-art.

----

## [0] Unpaired Multi-Domain Causal Representation Learning

**Authors**: *Nils Sturma, Chandler Squires, Mathias Drton, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html)

**Abstract**:

The goal of causal representation learning is to find a representation of data that consists of causally related latent variables. We consider a setup where one has access to data from multiple domains that potentially share a causal representation. Crucially, observations in different domains are assumed to be unpaired, that is, we only observe the marginal distribution in each domain but not their joint distribution. In this paper, we give sufficient conditions for identifiability of the joint distribution and the shared causal graph in a linear setup. Identifiability holds if we can uniquely recover the joint distribution and the shared causal representation from the marginal distributions in each domain. We transform our results into a practical method to recover the shared latent causal graph.

----

## [0] Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection

**Authors**: *Haibao Yu, Yingjuan Tang, Enze Xie, Jilei Mao, Ping Luo, Zaiqing Nie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html)

**Abstract**:

Cooperatively utilizing both ego-vehicle and infrastructure sensor data can significantly enhance autonomous driving perception abilities. However, the uncertain temporal asynchrony and limited communication conditions that are present in traffic environments can lead to fusion misalignment and constrain the exploitation of infrastructure data. To address these issues in vehicle-infrastructure cooperative 3D (VIC3D) object detection, we propose the Feature Flow Net (FFNet), a novel cooperative detection framework. FFNet is a flow-based feature fusion framework that uses a feature flow prediction module to predict future features and compensate for asynchrony. Instead of transmitting feature maps extracted from still-images, FFNet transmits feature flow, leveraging the temporal coherence of sequential infrastructure frames. Furthermore, we introduce a self-supervised training approach that enables FFNet to generate feature flow with feature prediction ability from raw infrastructure sequences. Experimental results demonstrate that our proposed method outperforms existing cooperative detection methods while only requiring about 1/100 of the transmission cost of raw data and covers all latency in one model on the DAIR-V2X dataset. The code  is available https://github.com/haibao-yu/FFNet-VIC3D.

----

## [0] Subspace Identification for Multi-Source Domain Adaptation

**Authors**: *Zijian Li, Ruichu Cai, Guangyi Chen, Boyang Sun, Zhifeng Hao, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html)

**Abstract**:

Multi-source domain adaptation (MSDA) methods aim to transfer knowledge from multiple labeled source domains to an unlabeled target domain. Although current methods achieve target joint distribution identifiability by enforcing minimal changes across domains, they often necessitate stringent conditions, such as an adequate number of domains, monotonic transformation of latent variables, and invariant label distributions. These requirements are challenging to satisfy in real-world applications. To mitigate the need for these strict assumptions, we propose a subspace identification theory that guarantees the disentanglement of domain-invariant and domain-specific variables under less restrictive constraints regarding domain numbers and transformation properties and thereby facilitating domain adaptation by minimizing the impact of domain shifts on invariant variables. Based on this theory, we develop a Subspace Identification Guarantee (SIG) model that leverages variational inference. Furthermore, the SIG model incorporates class-aware conditional alignment to accommodate target shifts where label distributions change with the domain. Experimental results demonstrate that our SIG model outperforms existing MSDA techniques on various benchmark datasets, highlighting its effectiveness in real-world applications.

----

## [0] Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates

**Authors**: *Sarath Sreedharan, Michael Katz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cbd0a1251f41b41aa68e728bcc1ee40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cbd0a1251f41b41aa68e728bcc1ee40-Abstract-Conference.html)

**Abstract**:

There has been an increasing interest in using symbolic models along with reinforcement learning (RL) problems, where these coarser abstract models are used as a way to provide RL agents with higher level guidance. However, most of these works are inherently limited by their assumption of having an access to a symbolic approximation of the underlying problem. To address this issue, we introduce a new method for learning optimistic symbolic approximations of the underlying world model. We will see how these representations, coupled with fast diverse planners developed by the automated planning community, provide us with a new paradigm for optimistic exploration in sparse reward settings. We investigate the possibility of speeding up the learning process by generalizing learned model dynamics across similar actions with minimal human input. Finally, we evaluate the method, by testing it on multiple benchmark domains and compare it with other RL strategies.

----

## [0] Feature learning via mean-field Langevin dynamics: classifying sparse parities and beyond

**Authors**: *Taiji Suzuki, Denny Wu, Kazusato Oko, Atsushi Nitanda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cc321baf0a8611b1d1bdbd18822667b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cc321baf0a8611b1d1bdbd18822667b-Abstract-Conference.html)

**Abstract**:

Neural network in the mean-field regime is known to be capable of \textit{feature learning}, unlike the kernel (NTK) counterpart. Recent works have shown that mean-field neural networks can be globally optimized by a noisy gradient descent update termed the \textit{mean-field Langevin dynamics} (MFLD). However, all existing guarantees for MFLD only considered the \textit{optimization} efficiency, and it is unclear if this algorithm leads to improved \textit{generalization} performance and sample complexity due to the presence of feature learning. To fill this gap, in this work we study the statistical and computational complexity of MFLD in learning a class of binary classification problems. Unlike existing margin bounds for neural networks, we avoid the typical norm control by utilizing the perspective that MFLD optimizes the \textit{distribution} of parameters rather than the parameter itself; this leads to an improved analysis of the sample complexity and convergence rate. We apply our general framework to the learning of $k$-sparse parity functions, where we prove that unlike kernel methods, two-layer neural networks optimized by MFLD achieves a sample complexity where the degree $k$ is ``decoupled'' from the exponent in the dimension dependence.

----



[Go to the previous page](NIPS-2023-list1400.md)

[Go to the next page](NIPS-2023-list1402.md)

[Go to the catalog section](README.md)