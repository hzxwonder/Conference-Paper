## [1400] Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context

**Authors**: *Lakshya A Agrawal, Aditya Kanade, Navin Goyal, Shuvendu K. Lahiri, Sriram K. Rajamani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html)

**Abstract**:

Language models of code (LMs) work well when the surrounding code provides sufficient context. This is not true when it becomes necessary to use types, functionality or APIs defined elsewhere in the repository or a linked library, especially those not seen during training. LMs suffer from limited awareness of such global context and end up hallucinating.Integrated development environments (IDEs) assist developers in understanding repository context using static analysis.  We extend this assistance, enjoyed by developers, to LMs. We propose monitor-guided decoding (MGD) where a monitor uses static analysis to guide the decoding. We construct a repository-level dataset PragmaticCode for method-completion in Java and evaluate MGD on it. On models of varying parameter scale, by monitoring for type-consistent object dereferences, MGD consistently improves compilation rates and agreement with ground truth. Further, LMs with fewer parameters, when augmented with MGD, can outperform larger LMs. With MGD, SantaCoder-1.1B achieves better compilation rate and next-identifier match than the much larger text-davinci-003 model.We also conduct a generalizability study to evaluate the ability of MGD to generalize to multiple programming languages (Java, C# and Rust), coding scenarios (e.g., correct number of arguments to method calls), and to enforce richer semantic constraints (e.g., stateful API protocols). Our data and implementation are available at https://github.com/microsoft/monitors4codegen.

----

## [1401] Towards Federated Foundation Models: Scalable Dataset Pipelines for Group-Structured Learning

**Authors**: *Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, Zachary Garrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/662bb9c4dcc96aeaac8e7cd3fc6a0add-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/662bb9c4dcc96aeaac8e7cd3fc6a0add-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Dataset Grouper, a library to create large-scale group-structured (e.g., federated) datasets, enabling federated learning simulation at the scale of foundation models. This library facilitates the creation of group-structured versions of existing datasets based on user-specified partitions, and directly leads to a variety of useful heterogeneous datasets that can be plugged into existing software frameworks. Dataset Grouper offers three key advantages. First, it scales to settings where even a single group's dataset is too large to fit in memory. Second, it provides flexibility, both in choosing the base (non-partitioned) dataset and in defining partitions. Finally, it is framework-agnostic. We empirically demonstrate that Dataset Grouper enables large-scale federated language modeling simulations on datasets that are orders of magnitude larger than in previous work, allowing for federated training of language models with hundreds of millions, and even billions, of parameters. Our experimental results show that algorithms like FedAvg operate more as meta-learning methods than as empirical risk minimization methods at this scale, suggesting their utility in downstream personalization and task-specific adaptation. Dataset Grouper is available at https://github.com/google-research/dataset_grouper.

----

## [1402] Sharp Spectral Rates for Koopman Operator Learning

**Authors**: *Vladimir Kostic, Karim Lounici, Pietro Novelli, Massimiliano Pontil*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/663bce02a0050c4a11f1eb8a7f1429d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/663bce02a0050c4a11f1eb8a7f1429d3-Abstract-Conference.html)

**Abstract**:

Non-linear dynamical systems can be handily described by the associated Koopman operator, whose action evolves every observable of the system forward in time. Learning the Koopman operator and its spectral decomposition from data is enabled by a number of algorithms. In this work we present for the first time non-asymptotic learning bounds for the Koopman eigenvalues and eigenfunctions. We focus on time-reversal-invariant stochastic dynamical systems, including the important example of Langevin dynamics. We analyze two popular estimators: Extended Dynamic Mode Decomposition (EDMD) and Reduced Rank Regression (RRR). Our results critically hinge on novel {minimax} estimation bounds for the operator norm error, that may be of independent interest. Our spectral learning bounds are driven by the simultaneous control of the operator norm error and a novel metric distortion functional of the estimated eigenfunctions. The bounds indicates that both EDMD and RRR have similar variance, but EDMD suffers from a larger bias which might be detrimental to its learning rate. Our results shed new light on the emergence of spurious eigenvalues, an issue which is well known empirically. Numerical experiments illustrate the implications of the bounds in practice.

----

## [1403] Continual Learning for Instruction Following from Realtime Feedback

**Authors**: *Alane Suhr, Yoav Artzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html)

**Abstract**:

We propose and deploy an approach to  continually train an instruction-following agent from feedback provided by users during collaborative interactions. During interaction, human users instruct an agent using natural language, and provide realtime binary feedback as they observe the agent following their instructions. We design a contextual bandit learning approach, converting  user feedback to immediate reward. We evaluate through thousands of human-agent interactions, demonstrating 15.4% absolute improvement in instruction execution accuracy over time. We also show our approach is robust to several design variations, and that the feedback signal is roughly equivalent to the learning signal of supervised demonstration data.

----

## [1404] Embracing the chaos: analysis and diagnosis of numerical instability in variational flows

**Authors**: *Zuheng Xu, Trevor Campbell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html)

**Abstract**:

In this paper, we investigate the impact of numerical instability on the reliability of sampling, density evaluation, and evidence lower bound (ELBO) estimation in variational flows. We first empirically demonstrate that common flows can exhibit a catastrophic accumulation of error: the numerical flow map deviates significantly from the exact map---which affects sampling---and the numerical inverse flow map does not accurately recover the initial input---which affects density and ELBO computations. Surprisingly though, we find that results produced by flows are often accurate enough for applications despite the presence of serious numerical instability. In this work, we treat variational flows as chaotic dynamical systems, and leverage shadowing theory to elucidate this behavior via theoretical guarantees on the error of sampling, density evaluation, and ELBO estimation. Finally, we develop and empirically test a diagnostic procedure that can be used to validate results produced by numerically unstable flows in practice.

----

## [1405] Masked Two-channel Decoupling Framework for Incomplete Multi-view Weak Multi-label Learning

**Authors**: *Chengliang Liu, Jie Wen, Yabo Liu, Chao Huang, Zhihao Wu, Xiaoling Luo, Yong Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66772e6aa61e54ae16443ae1d78a7319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66772e6aa61e54ae16443ae1d78a7319-Abstract-Conference.html)

**Abstract**:

Multi-view learning has become a popular research topic in recent years, but research on the cross-application of classic multi-label classification and multi-view learning is still in its early stages. In this paper, we focus on the complex yet highly realistic task of incomplete multi-view weak multi-label learning and propose a masked two-channel decoupling framework based on deep neural networks to solve this problem. The core innovation of our method lies in decoupling the single-channel view-level representation, which is common in deep multi-view learning methods, into a shared representation and a view-proprietary representation. We also design a cross-channel contrastive loss to enhance the semantic property of the two channels. Additionally, we exploit supervised information to design a label-guided graph regularization loss, helping the extracted embedding features preserve the geometric structure among samples. Inspired by the success of masking mechanisms in image and text analysis, we develop a random fragment masking strategy for vector features to improve the learning ability of encoders. Finally, it is important to emphasize that our model is fully adaptable to arbitrary view and label absences while also performing well on the ideal full data. We have conducted sufficient and convincing experiments to confirm the effectiveness and advancement of our model.

----

## [1406] Exponential Lower Bounds for Fictitious Play in Potential Games

**Authors**: *Ioannis Panageas, Nikolas Patris, Stratis Skoulakis, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/66820ab16b817d8a6b00d60b3d24b83a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/66820ab16b817d8a6b00d60b3d24b83a-Abstract-Conference.html)

**Abstract**:

Fictitious Play (FP) is a simple and natural dynamic for repeated play with many applications in game theory and multi-agent reinforcement learning. It was introduced by Brown and its convergence properties for two-player zero-sum games was established later by Robinson. Potential games [Monderer and Shapley 1996] is another class of games which exhibit the FP property [Monderer and Shapley 1996], i.e., FP dynamics converges to a Nash equilibrium if all agents follows it. Nevertheless, except for two-player zero-sum games and for specific instances of payoff matrices [Abernethy et. al. 2021] or for adversarial tie-breaking rules [Daskalakis and Pan, 2014], the \textit{convergence rate} of FP is unknown. In this work, we focus on the rate of convergence of FP when applied to potential games and more specifically identical payoff games. We prove that FP can take exponential time (in the number of strategies) to reach a Nash equilibrium, even if the game is restricted to \textit{two agents}. To prove this, we recursively construct a two-player coordination game with a unique Nash equilibrium. Moreover, every approximate Nash equilibrium in the constructed game must be close to the pure Nash equilibrium in $\ell_1$-distance.

----

## [1407] Cocktail: Mixing Multi-Modality Control for Text-Conditional Image Generation

**Authors**: *Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, Tat-Jen Cham*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html)

**Abstract**:

Text-conditional diffusion models are able to generate high-fidelity images with diverse contents.However, linguistic representations frequently exhibit ambiguous descriptions of the envisioned objective imagery, requiring the incorporation of additional control signals to bolster the efficacy of text-guided diffusion models. In this work, we propose Cocktail, a pipeline to mix various modalities into one embedding, amalgamated with a generalized ControlNet (gControlNet), a controllable normalisation (ControlNorm), and a spatial guidance sampling method, to actualize multi-modal and spatially-refined control for text-conditional diffusion models. Specifically, we introduce a hyper-network gControlNet, dedicated to the alignment and infusion of the control signals from disparate modalities into the pre-trained diffusion model. gControlNet is capable of accepting flexible modality signals, encompassing the simultaneous reception of any combination of modality signals, or the supplementary fusion of multiple modality signals. The control signals are then fused and injected into the backbone model according to our proposed ControlNorm.Furthermore, our advanced spatial guidance sampling methodology proficiently incorporates the control signal into the designated region, thereby circumventing the manifestation of undesired objects within the generated image.We demonstrate the results of our method in controlling various modalities, proving high-quality synthesis and fidelity to multiple external signals.

----

## [1408] RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability

**Authors**: *Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6692e1b0e8a31e8de84bd90ad4d8d9e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6692e1b0e8a31e8de84bd90ad4d8d9e0-Abstract-Conference.html)

**Abstract**:

Visual model-based RL methods typically encode image observations into low-dimensional representations in a manner that does not eliminate redundant information. This leaves them susceptible to spurious variations -- changes in task-irrelevant components such as background distractors or lighting conditions. In this paper, we propose a visual model-based RL method that learns a latent representation resilient to such spurious variations. Our training objective encourages the representation to be maximally predictive of dynamics and reward, while constraining the information flow from the observation to the latent representation. We demonstrate that this objective significantly bolsters the resilience of visual model-based RL methods to visual distractors, allowing them to operate in dynamic environments. We then show that while the learned encoder is able to operate in dynamic environments, it is not invariant under significant distribution shift. To address this, we propose a simple reward-free alignment procedure that enables test time adaptation of the encoder. This allows for quick adaptation to widely differing environments without having to relearn the dynamics and policy. Our effort is a step towards making model-based RL a practical and useful tool for dynamic, diverse domains and we show its effectiveness in simulation tasks with significant spurious variations.

----

## [1409] Circuit as Set of Points

**Authors**: *Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, Chang Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6697bb267dc517379bc8aa326e844f8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6697bb267dc517379bc8aa326e844f8d-Abstract-Conference.html)

**Abstract**:

As the size of circuit designs continues to grow rapidly, artificial intelligence technologies are being extensively used in Electronic Design Automation (EDA) to assist with circuit design.Placement and routing are the most time-consuming parts of the physical design process, and how to quickly evaluate the placement has become a hot research topic. Prior works either transformed circuit designs into images using hand-crafted methods and then used Convolutional Neural Networks (CNN) to extract features, which are limited by the quality of the hand-crafted methods and could not achieve end-to-end training, or treated the circuit design as a graph structure and used Graph Neural Networks (GNN) to extract features, which require time-consuming preprocessing.In our work, we propose a novel perspective for circuit design by treating circuit components as point clouds and using Transformer-based point cloud perception methods to extract features from the circuit. This approach enables direct feature extraction from raw data without any preprocessing, allows for end-to-end training, and results in high performance.Experimental results show that our method achieves state-of-the-art performance in congestion prediction tasks on both the CircuitNet and ISPD2015 datasets, as well as in design rule check (DRC) violation prediction tasks on the CircuitNet dataset.Our method establishes a bridge between the relatively mature point cloud perception methods and the fast-developing EDA algorithms, enabling us to leverage more collective intelligence to solve this task. To facilitate the research of open EDA design, source codes and pre-trained models are released at https://github.com/hustvl/circuitformer.

----

## [1410] Causal Component Analysis

**Authors**: *Wendong Liang, Armin Kekic, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67089958e98b243d5cc1881ad60418b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67089958e98b243d5cc1881ad60418b8-Abstract-Conference.html)

**Abstract**:

Independent Component Analysis (ICA) aims to recover independent latent variables from observed mixtures thereof. Causal Representation Learning (CRL) aims instead to infer causally related (thus often statistically dependent) latent variables, together with the unknown graph encoding their causal relationships. We introduce an intermediate problem termed Causal Component Analysis (CauCA). CauCA can be viewed as a generalization of ICA, modelling the causal dependence among the latent components, and as a special case of CRL. In contrast to CRL, it presupposes knowledge of the causal graph, focusing solely on learning the unmixing function and the causal mechanisms. Any impossibility results regarding the recovery of the ground truth in CauCA also apply for CRL, while possibility results may serve as a stepping stone for extensions to CRL. We characterize CauCA identifiability from multiple datasets generated through different types of interventions on the latent causal variables. As a corollary, this interventional perspective also leads to new identifiability results for nonlinear ICA—a special case of CauCA with an empty graph—requiring strictly fewer datasets than previous results. We introduce a likelihood-based approach using normalizing flows to estimate both the unmixing function and the causal mechanisms, and demonstrate its effectiveness through extensive synthetic experiments in the CauCA and ICA setting.

----

## [1411] Latent Graph Inference with Limited Supervision

**Authors**: *Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67101f97dc23fcc10346091181fff6cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67101f97dc23fcc10346091181fff6cb-Abstract-Conference.html)

**Abstract**:

Latent graph inference (LGI) aims to jointly learn the underlying graph structure and node representations from data features. However, existing LGI methods commonly suffer from the issue of supervision starvation, where massive edge weights are learned without semantic supervision and do not contribute to the training loss. Consequently, these supervision-starved weights, which determine the predictions of testing samples, cannot be semantically optimal, resulting in poor generalization. In this paper, we observe that this issue is actually caused by the graph sparsification operation, which severely destroys the important connections established between pivotal nodes and labeled ones. To address this, we propose to restore the corrupted affinities and replenish the missed supervision for better LGI. The key challenge then lies in identifying the critical nodes and recovering the corrupted affinities. We begin by defining the pivotal nodes as k-hop starved nodes, which can be identified based on a given adjacency matrix. Considering the high computational burden, we further present a more efficient alternative inspired by CUR matrix decomposition. Subsequently, we eliminate the starved nodes by reconstructing the destroyed connections. Extensive experiments on representative benchmarks demonstrate that reducing the starved nodes consistently improves the performance of state-of-the-art LGI methods, especially under extremely limited supervision (6.12% improvement on Pubmed with a labeling rate of only 0.3%).

----

## [1412] Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows

**Authors**: *Alexandre Verine, Benjamin Négrevergne, Muni Sreenivas Pydi, Yann Chevaleyre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67159f1c0cab15dd34c76a5dd830a389-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67159f1c0cab15dd34c76a5dd830a389-Abstract-Conference.html)

**Abstract**:

Achieving a balance between image quality (precision) and diversity (recall) is a significant challenge in the domain of generative models. Current state-of-the-art models primarily rely on optimizing heuristics, such as the Fr\'echet Inception Distance. While recent developments have introduced principled methods for evaluating precision and recall, they have yet to be successfully integrated into the training of generative models. Our main contribution is a novel training method for generative models, such as Generative Adversarial Networks and Normalizing Flows, which explicitly optimizes a user-defined trade-off between precision and recall.  More precisely, we show that achieving a specified precision-recall trade-off corresponds to minimizing a unique $f$-divergence from a family we call the \mbox{\em PR-divergences}. Conversely, any $f$-divergence can be written as a linear combination of PR-divergences and  corresponds to a weighted precision-recall trade-off. Through comprehensive evaluations, we show that our approach improves the performance of existing state-of-the-art models like BigGAN in terms of either precision or recall when tested on datasets such as ImageNet.

----

## [1413] Energy Guided Diffusion for Generating Neurally Exciting Images

**Authors**: *Pawel A. Pierzchlewicz, Konstantin Willeke, Arne Nix, Pavithra Elumalai, Kelli Restivo, Tori Shinn, Cate Nealley, Gabrielle Rodriguez, Saumil S. Patel, Katrin Franke, Andreas S. Tolias, Fabian H. Sinz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html)

**Abstract**:

In recent years, most exciting inputs (MEIs) synthesized from encoding models of neuronal activity have become an established method for studying tuning properties of biological and artificial visual systems.    However, as we move up the visual hierarchy, the complexity of neuronal computations increases.     Consequently, it becomes more challenging to model neuronal activity, requiring more complex models.    In this study, we introduce a novel readout architecture inspired by the mechanism of visual attention. This new architecture, which we call attention readout, together with a data-driven convolutional core outperforms previous task-driven models in predicting the activity  of neurons in macaque area V4.    However, as our predictive network becomes deeper and more complex, synthesizing MEIs via straightforward gradient ascent (GA) can struggle to produce qualitatively good results and overfit to idiosyncrasies of a more complex model, potentially decreasing the MEI's model-to-brain transferability.    To solve this problem, we propose a diffusion-based method for generating MEIs via Energy Guidance (EGG).    We show that for models of macaque V4, EGG generates single neuron MEIs that generalize better across varying model architectures than the state-of-the-art GA, while at the same time reducing computational costs by a factor of 4.7x, facilitating experimentally challenging closed-loop experiments.    Furthermore, EGG diffusion can be used to generate other neurally exciting images, like most exciting naturalistic images that are on par with a selection of highly activating natural images, or image reconstructions that generalize better across architectures.    Finally, EGG is simple to implement, requires no retraining of the diffusion model, and can easily be generalized to provide other characterizations of the visual system, such as invariances.    Thus, EGG provides a general and flexible framework to study the coding properties of the visual system in the context of natural images.

----

## [1414] An active learning framework for multi-group mean estimation

**Authors**: *Abdellah Aznag, Rachel Cummings, Adam N. Elmachtoub*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67390075fe466276797f489115582cdc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67390075fe466276797f489115582cdc-Abstract-Conference.html)

**Abstract**:

We consider a fundamental problem where there are multiple groups whose data distributions are unknown, and an analyst would like to learn the mean of each group. We consider an active learning framework to sequentially collect $T$ samples with bandit, each period observing a sample from a chosen group. After observing a sample, the analyst may update their estimate of the mean and variance of that group and choose the next group accordingly.  The objective is to dynamically collect samples to minimize the $p$-norm of the vector of variances of our mean estimators after $T$ rounds. We propose an algorithm, Variance-UCB, that selects groups according to a an upper bound on the variance estimate adjusted to the $p$-norm chosen. We show that the regret of Variance-UCB is $O(T^{-2})$ for finite $p$, and prove that no algorithm can do better. When $p$ is infinite, we recover the $O(T^{-1.5})$ obtained in \cite{activelearning, carpentier2011upper} and provide a new lower bound showing that no algorithm can do better.

----

## [1415] CAT-Walk: Inductive Hypergraph Learning via Set Walks

**Authors**: *Ali Behrouz, Farnoosh Hashemi, Sadaf Sadeghian, Margo I. Seltzer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html)

**Abstract**:

Temporal hypergraphs provide a powerful paradigm for modeling time-dependent, higher-order interactions in complex systems. Representation learning for hypergraphs is essential for extracting patterns of the higher-order interactions that are critically important in real-world problems in social network analysis, neuroscience, finance, etc. However, existing methods are typically designed only for specific tasks or static hypergraphs. We present  CAT-Walk, an inductive method that learns the underlying dynamic laws that govern the temporal and structural processes underlying a temporal hypergraph. CAT-Walk introduces a temporal, higher-order walk on hypergraphs, SetWalk, that extracts higher-order causal patterns. CAT-Walk uses a novel adaptive and permutation invariant pooling strategy, SetMixer, along with a set-based anonymization process that hides the identity of hyperedges. Finally, we present a simple yet effective neural network model to encode hyperedges. Our evaluation on 10 hypergraph benchmark datasets shows that CAT-Walk attains outstanding performance on temporal hyperedge prediction benchmarks in both inductive and transductive settings. It also shows competitive performance with state-of-the-art methods for node classification. (https://github.com/ubc-systopia/CATWalk)

----

## [1416] Unbiased constrained sampling with Self-Concordant Barrier Hamiltonian Monte Carlo

**Authors**: *Maxence Noble, Valentin De Bortoli, Alain Durmus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6745cb9889cc213bda803535f2d3902e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6745cb9889cc213bda803535f2d3902e-Abstract-Conference.html)

**Abstract**:

In this paper, we propose Barrier Hamiltonian Monte Carlo (BHMC), a version of the  HMC algorithm which aims at sampling from a Gibbs distribution $\pi$ on a manifold  $\mathsf{M}$, endowed with a Hessian metric $\mathfrak{g}$ derived from a self-concordant  barrier. Our method relies on Hamiltonian  dynamics which comprises $\mathfrak{g}$. Therefore, it incorporates the constraints defining  $\mathsf{M}$ and is able to exploit its underlying geometry. However,   the corresponding Hamiltonian dynamics is defined via non separable Ordinary Differential Equations (ODEs) in contrast to the Euclidean case. It implies unavoidable bias in existing generalization of HMC to Riemannian manifolds. In this paper, we propose a new filter step, called ``involution checking step'', to address this problem. This step is implemented in two versions of BHMC, coined continuous BHMC (c-bHMC) and  numerical BHMC (n-BHMC) respectively.  Our main results establish that these two new algorithms  generate reversible Markov  chains with respect to $\pi$ and do not suffer from any bias in comparison to previous implementations. Our conclusions are supported by numerical experiments where  we consider target distributions defined on polytopes.

----

## [1417] Directional diffusion models for graph representation learning

**Authors**: *Run Yang, Yuling Yang, Fan Zhou, Qiang Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6751ee6546b31ceb7d4ee12276b9f4d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6751ee6546b31ceb7d4ee12276b9f4d9-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved remarkable success in diverse domains such as image synthesis, super-resolution, and 3D molecule generation. Surprisingly, the application of diffusion models in graph learning has garnered little attention. In this paper, we aim to bridge this gap by exploring the use of diffusion models for unsupervised graph representation learning. Our investigation commences with the identification of anisotropic structures within graphs and the recognition of a crucial limitation in the vanilla forward diffusion process when dealing with these anisotropic structures. The original forward diffusion process continually adds  isotropic Gaussian noise to the data, which may excessively dilute anisotropic signals, leading to rapid signal-to-noise conversion. This rapid conversion poses challenges for training denoising neural networks and obstructs the acquisition of semantically meaningful representations during the reverse process. To overcome this challenge, we introduce a novel class of models termed {\it directional diffusion models}.  These models adopt data-dependent, anisotropic, and directional noises in the forward diffusion process. In order to assess the effectiveness of our proposed models, we conduct extensive experiments on 12 publicly available datasets, with a particular focus on two distinct graph representation learning tasks. The experimental results unequivocally establish the superiority of our models over state-of-the-art baselines, underscoring their effectiveness in capturing meaningful graph representations. Our research not only sheds light on the intricacies of the forward process in diffusion models but also underscores the vast potential of these models in addressing a wide spectrum of graph-related tasks. Our code is available at \url{https://github.com/statsle/DDM}.

----

## [1418] UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition

**Authors**: *Qiufu Li, Xi Jia, Jiancan Zhou, Linlin Shen, Jinming Duan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6776737cd11cf4afa3af226898474418-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6776737cd11cf4afa3af226898474418-Abstract-Conference.html)

**Abstract**:

Sample-to-class-based face recognition models can not fully explore the cross-sample relationship among large amounts of facial images, while sample-to-sample-based models require sophisticated pairing processes for training. Furthermore, neither method satisfies the requirements of real-world face verification applications, which expect a unified threshold separating positive from negative facial pairs. In this paper, we propose a unified threshold integrated sample-to-sample based loss (USS loss), which features an explicit unified threshold for distinguishing positive from negative pairs. Inspired by our USS loss, we also derive the sample-to-sample based softmax and BCE losses, and discuss their relationship. Extensive evaluation on multiple benchmark datasets, including MFR, IJB-C, LFW, CFP-FP, AgeDB, and MegaFace, demonstrates that the proposed USS loss is highly efficient and can work seamlessly with sample-to-class-based losses. The embedded loss (USS and sample-to-class Softmax loss) overcomes the pitfalls of previous approaches and the trained facial model UniTSFace exhibits exceptional performance, outperforming state-of-the-art methods, such as CosFace, ArcFace, VPL, AnchorFace, and UNPG. Our code is available at https://github.com/CVI-SZU/UniTSFace.

----

## [1419] Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks

**Authors**: *Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/677c8dc72c99482507323f313faf4738-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/677c8dc72c99482507323f313faf4738-Abstract-Conference.html)

**Abstract**:

Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP. The code of MDP is publicly available.

----

## [1420] On the Power of SVD in the Stochastic Block Model

**Authors**: *Xinyu Mao, Jiapeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/678594bcff6f99f3b7a8ff459989b1a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/678594bcff6f99f3b7a8ff459989b1a3-Abstract-Conference.html)

**Abstract**:

A popular heuristic method for improving clustering results is to apply dimensionality reduction before running clustering algorithms.It has been observed that spectral-based dimensionality reduction tools, such as PCA or SVD, improve the performance of clustering algorithms in many applications. This phenomenon indicates that spectral method not only serves as a dimensionality reduction tool, but also contributes to the clustering procedure in some sense. It is an interesting question to understand the behavior of spectral steps in clustering problems.As an initial step in this direction, this paper studies the power of vanilla-SVD algorithm in the stochastic block model (SBM). We show that, in the symmetric setting, vanilla-SVD algorithm recovers all clusters correctly. This result answers an open question posed by Van Vu (Combinatorics Probability and Computing, 2018) in the symmetric setting.

----

## [1421] Continuous-time Analysis of Anchor Acceleration

**Authors**: *Jaewook J. Suh, Jisun Park, Ernest K. Ryu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/678cffc05549fdabda971127602084c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/678cffc05549fdabda971127602084c6-Abstract-Conference.html)

**Abstract**:

Recently, the anchor acceleration, an acceleration mechanism distinct from Nesterov's, has been discovered for minimax optimization and fixed-point problems, but its mechanism is not understood well, much less so than Nesterov acceleration. In this work, we analyze continuous-time models of anchor acceleration. We provide tight, unified analyses for characterizing the convergence rate as a function of the anchor coefficient $\beta(t)$, thereby providing insight into the anchor acceleration mechanism and its accelerated $\mathcal{O}(1/k^2)$-convergence rate. Finally, we present an adaptive method inspired by the continuous-time analyses and establish its effectiveness through theoretical analyses and experiments.

----

## [1422] BEDD: The MineRL BASALT Evaluation and Demonstrations Dataset for Training and Benchmarking Agents that Solve Fuzzy Tasks

**Authors**: *Stephanie Milani, Anssi Kanervisto, Karolis Ramanauskas, Sander Schulhoff, Brandon Houghton, Rohin Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67a6726dcd555b982cabb3446ffac01d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/67a6726dcd555b982cabb3446ffac01d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The MineRL BASALT competition has served to catalyze advances in learning from human feedback through four hard-to-specify tasks in Minecraft, such as create and photograph a waterfall. Given the completion of two years of BASALT competitions, we offer to the community a formalized benchmark through the BASALT Evaluation and Demonstrations Dataset (BEDD), which serves as a resource for algorithm development and performance assessment. BEDD consists of a collection of 26 million image-action pairs from nearly 14,000 videos of human players completing the BASALT tasks in Minecraft. It also includes over 3,000 dense pairwise human evaluations of human and algorithmic agents. These comparisons serve as a fixed, preliminary leaderboard for evaluating newly-developed algorithms.  To enable this comparison, we present a streamlined codebase for benchmarking new algorithms against the leaderboard. In addition to presenting these datasets, we conduct a detailed analysis of the data from both datasets to guide algorithm development and evaluation. The released code and data are available at https://github.com/minerllabs/basalt-benchmark.

----

## [1423] Self-supervised Object-Centric Learning for Videos

**Authors**: *Görkay Aydemir, Weidi Xie, Fatma Güney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html)

**Abstract**:

Unsupervised multi-object segmentation has shown impressive results on images by utilizing powerful semantics learned from self-supervised pretraining. An additional modality such as depth or motion is often used to facilitate the segmentation in video sequences. However, the performance improvements observed in synthetic sequences, which rely on the robustness of an additional cue, do not translate to more challenging real-world scenarios. In this paper, we propose the first fully unsupervised method for segmenting multiple objects in real-world sequences. Our object-centric learning framework spatially binds objects to slots on each frame and then relates these slots across frames. From these temporally-aware slots, the training objective is to reconstruct the middle frame in a high-level semantic feature space. We propose a masking strategy by dropping a significant portion of tokens in the feature space for efficiency and regularization. Additionally, we address over-clustering by merging slots based on similarity. Our method can successfully segment multiple instances of complex and high-variety classes in YouTube videos.

----

## [1424] Improving Adversarial Transferability via Intermediate-level Perturbation Decay

**Authors**: *Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67b2e2e895380fa6acd537c2894e490e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67b2e2e895380fa6acd537c2894e490e-Abstract-Conference.html)

**Abstract**:

Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

----

## [1425] GUST: Combinatorial Generalization by Unsupervised Grouping with Neuronal Coherence

**Authors**: *Hao Zheng, Hui Lin, Rong Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67d5c7dd7930dfce2725defdb0552b6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67d5c7dd7930dfce2725defdb0552b6e-Abstract-Conference.html)

**Abstract**:

Dynamically grouping sensory information into structured entities is essential for understanding the world of combinatorial nature. However, the grouping ability and therefore combinatorial generalization are still challenging artificial neural networks. Inspired by the evidence that successful grouping is indicated by neuronal coherence in the human brain, we introduce GUST (Grouping Unsupervisely by Spike Timing network), an iterative network architecture with biological constraints to bias the network towards a dynamical state of neuronal coherence that softly reflects the grouping information in the temporal structure of its spiking activity. We evaluate and analyze the model on synthetic datasets. Interestingly, the segregation ability is directly learned from superimposed stimuli with a succinct unsupervised objective. Two learning stages are present, from coarsely perceiving global features to additionally capturing local features. Further, the learned symbol-like building blocks can be systematically composed to represent novel scenes in a bio-plausible manner.

----

## [1426] State Regularized Policy Optimization on Data with Dynamics Shift

**Authors**: *Zhenghai Xue, Qingpeng Cai, Shuchang Liu, Dong Zheng, Peng Jiang, Kun Gai, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67dd6a41bf9539cffc0fc0165e4d0616-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67dd6a41bf9539cffc0fc0165e4d0616-Abstract-Conference.html)

**Abstract**:

In many real-world scenarios, Reinforcement Learning (RL) algorithms are trained on data with dynamics shift, i.e., with different underlying environment dynamics. A majority of current methods address such issue by training context encoders to identify environment parameters. Data with dynamics shift are separated according to their environment parameters to train the corresponding policy.However, these methods can be sample inefficient as data are used \textit{ad hoc}, and policies trained for one dynamics cannot benefit from data collected in all other environments with different dynamics. In this paper, we find that in many environments with similar structures and different dynamics, optimal policies have similar stationary state distributions. We exploit such property and learn the stationary state distribution from data with dynamics shift for efficient data reuse. Such distribution is used to regularize the policy trained in a new environment, leading to the SRPO (\textbf{S}tate \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization) algorithm. To conduct theoretical analyses, the intuition of similar environment structures is characterized by the notion of homomorphous MDPs. We then demonstrate a lower-bound performance guarantee on policies regularized by the stationary state distribution. In practice, SRPO can be an add-on module to context-based algorithms in both online and offline RL settings.Experimental results show that SRPO can make several context-based algorithms far more data efficient and significantly improve their overall performance.

----

## [1427] Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

**Authors**: *Andy Zhou, Jindong Wang, Yu-Xiong Wang, Haohan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html)

**Abstract**:

We propose a conceptually simple and lightweight framework for improving the robustness of vision models through the combination of knowledge distillation and data augmentation. We address the conjecture that larger models do not make for better teachers by showing strong gains in out-of-distribution robustness when distilling from pretrained foundation models. Following this finding, we propose Discrete Adversarial Distillation (DAD), which leverages a robust teacher to generate adversarial examples and a VQGAN to discretize them, creating more informative samples than standard data augmentation techniques. We provide a theoretical framework for the use of a robust teacher in the knowledge distillation with data augmentation setting and demonstrate strong gains in out-of-distribution robustness and clean accuracy across different student architectures. Notably, our method adds minor computational overhead compared to similar techniques and can be easily combined with other data augmentations for further improvements.

----

## [1428] XES3G5M: A Knowledge Tracing Benchmark Dataset with Auxiliary Information

**Authors**: *Zitao Liu, Qiongqiong Liu, Teng Guo, Jiahao Chen, Shuyan Huang, Xiangyu Zhao, Jiliang Tang, Weiqi Luo, Jian Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/67fc628f17c2ad53621fb961c6bafcaf-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Knowledge tracing (KT) is a task that predicts students' future performance based on their historical learning interactions. With the rapid development of deep learning techniques, existing KT approaches follow a data-driven paradigm that uses massive problem-solving records to model students' learning processes. However, although the educational contexts contain various factors that may have an influence on student learning outcomes, existing public KT datasets mainly consist of anonymized ID-like features, which may hinder the research advances towards this field. Therefore, in this work, we present, \emph{XES3G5M}, a large-scale dataset with rich auxiliary information about questions and their associated knowledge components (KCs)\footnote{\label{ft:kc}A KC is a generalization of everyday terms like concept, principle, fact, or skill.}. The XES3G5M dataset is collected from a real-world online math learning platform, which contains 7,652 questions, and 865 KCs with 5,549,635 interactions from 18,066 students. To the best of our knowledge, the XES3G5M dataset not only has the largest number of KCs in math domain but contains the richest contextual information including tree structured KC relations, question types, textual contents and analysis and student response timestamps. Furthermore, we build a comprehensive benchmark on 19 state-of-the-art deep learning based knowledge tracing (DLKT) models. Extensive experiments demonstrate the effectiveness of leveraging the auxiliary information in our XES3G5M with DLKT models. We hope the proposed dataset can effectively facilitate the KT research work.

----

## [1429] Factorized Contrastive Learning: Going Beyond Multi-view Redundancy

**Authors**: *Paul Pu Liang, Zihao Deng, Martin Q. Ma, James Y. Zou, Louis-Philippe Morency, Ruslan Salakhutdinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html)

**Abstract**:

In a wide range of multimodal tasks, contrastive learning has become a particularly appealing approach since it can successfully learn representations from abundant unlabeled data with only pairing information (e.g., image-caption or video-audio pairs). Underpinning these approaches is the assumption of multi-view redundancy - that shared information between modalities is necessary and sufficient for downstream tasks. However, in many real-world settings, task-relevant information is also contained in modality-unique regions: information that is only present in one modality but still relevant to the task. How can we learn self-supervised multimodal representations to capture both shared and unique information relevant to downstream tasks? This paper proposes FactorCL, a new multimodal representation learning method to go beyond multi-view redundancy. FactorCL is built from three new contributions: (1) factorizing task-relevant information into shared and unique representations, (2) capturing task-relevant information via maximizing MI lower bounds and removing task-irrelevant information via minimizing MI upper bounds, and (3) multimodal data augmentations to approximate task relevance without labels. On large-scale real-world datasets, FactorCL captures both shared and unique information and achieves state-of-the-art results on six benchmarks.

----

## [1430] Semantic Image Synthesis with Unconditional Generator

**Authors**: *Jungwoo Chae, Hyunin Cho, Sooyeon Go, Kyungmook Choi, Youngjung Uh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/683464f40aa1a6b7c939c3e9cd64b1fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/683464f40aa1a6b7c939c3e9cd64b1fd-Abstract-Conference.html)

**Abstract**:

Semantic image synthesis (SIS) aims to generate realistic images according to semantic masks given by a user. Although recent methods produce high quality results with fine spatial control, SIS requires expensive pixel-level annotation of the training images. On the other hand, manipulating intermediate feature maps in a pretrained unconditional generator such as StyleGAN supports coarse spatial control without heavy annotation. In this paper, we introduce a new approach, for reflecting user's detailed guiding masks on a pretrained unconditional generator. Our method converts a user's guiding mask to a proxy mask through a semantic mapper. Then the proxy mask conditions the resulting image through a rearranging network based on cross-attention mechanism. The proxy mask is simple clustering of intermediate feature maps in the generator. The semantic mapper and the rearranging network are easy to train (less than half an hour). Our method is useful for many tasks: semantic image synthesis, spatially editing real images, and unaligned local transplantation. Last but not least, it is generally applicable to various datasets such as human faces, animal faces, and churches.

----

## [1431] Learning Neural Implicit through Volume Rendering with Attentive Depth Fusion Priors

**Authors**: *Pengchong Hu, Zhizhong Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68637ee6b30276f900bc67320466b69f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68637ee6b30276f900bc67320466b69f-Abstract-Conference.html)

**Abstract**:

Learning neural implicit representations has achieved remarkable performance in 3D reconstruction from multi-view images. Current methods use volume rendering to render implicit representations into either RGB or depth images that are supervised by the multi-view ground truth. However, rendering a view each time suffers from incomplete depth at holes and unawareness of occluded structures from the depth supervision, which severely affects the accuracy of geometry inference via volume rendering. To resolve this issue, we propose to learn neural implicit representations from multi-view RGBD images through volume rendering with an attentive depth fusion prior. Our prior allows neural networks to sense coarse 3D structures from the Truncated Signed Distance Function (TSDF) fused from all available depth images for rendering. The TSDF enables accessing the missing depth at holes on one depth image and the occluded parts that are invisible from the current view. By introducing a novel attention mechanism, we allow neural networks to directly use the depth fusion prior with the inferred occupancy as the learned implicit function. Our attention mechanism works with either a one-time fused TSDF that represents a whole scene or an incrementally fused TSDF that represents a partial scene in the context of Simultaneous Localization and Mapping (SLAM). Our evaluations on widely used benchmarks including synthetic and real-world scans show our superiority over the latest neural implicit methods.

----

## [1432] SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning

**Authors**: *Yifan Yang, Peiyao Xiao, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/686a3f32067838c8dbb68da6e9e3cf69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/686a3f32067838c8dbb68da6e9e3cf69-Abstract-Conference.html)

**Abstract**:

Federated bilevel optimization (FBO) has shown great potential recently in machine learning and edge computing due to the emerging nested optimization structure in meta-learning, fine-tuning, hyperparameter tuning, etc. However, existing FBO algorithms often involve complicated computations and require multiple sub-loops per iteration, each of which contains a number of communication rounds. In this paper, we propose a simple and flexible FBO framework named SimFBO, which is easy to implement without sub-loops, and includes a generalized server-side aggregation and update for improving communication efficiency. We further propose System-level heterogeneity robust FBO (ShroFBO) as a variant of SimFBO with stronger resilience to heterogeneous local computation. We show that SimFBO and ShroFBO provably achieve a linear convergence speedup with partial client participation and client sampling without replacement, as well as improved sample and communication complexities. Experiments demonstrate the effectiveness of the proposed methods over existing FBO algorithms.

----

## [1433] PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification

**Authors**: *Qianli Shen, Wai Hoh Tang, Zhun Deng, Apostolos F. Psaros, Kenji Kawaguchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68730224bbf35ffac7a4fbf9b1ea4bfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68730224bbf35ffac7a4fbf9b1ea4bfe-Abstract-Conference.html)

**Abstract**:

Standard approaches for uncertainty quantification in deep learning and physics-informed learning have persistent limitations. Indicatively, strong assumptions regarding the data likelihood are required, the performance highly depends on the selection of priors, and the posterior can be sampled only approximately, which leads to poor approximations because of the associated computational cost.This paper introduces and studies confidence interval (CI) estimation for deterministic partial differential equations as a novel problem.That is, to propagate confidence, in the form of CIs, from data locations to the entire domain with probabilistic guarantees.We propose a method, termed Physics-Informed Confidence Propagation (PICProp), based on bi-level optimization to compute a valid CI without making heavy assumptions.We provide a theorem regarding the validity of our method, and computational experiments, where the focus is on physics-informed learning. Code is available at https://github.com/ShenQianli/PICProp.

----

## [1434] Foundation Model is Efficient Multimodal Multitask Model Selector

**Authors**: *Fanqing Meng, Wenqi Shao, Zhanglin Peng, Chonghe Jiang, Kaipeng Zhang, Yu Qiao, Ping Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/687b7b2bdcc2ced577c0a989b44e7078-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/687b7b2bdcc2ced577c0a989b44e7078-Abstract-Conference.html)

**Abstract**:

This paper investigates an under-explored but important problem: given a collection of pre-trained neural networks, predicting their performance on each multi-modal task without fine-tuning them, such as image recognition, referring, captioning, visual question answering, and text question answering.A brute-force approach is to finetune all models on all target datasets, bringing high computational costs. Although recent-advanced approaches employed lightweight metrics to measure models’ transferability, they often depend heavily on the prior knowledge of a single task, making them inapplicable in a multi-modal multi-task scenario. To tackle this issue, we propose an efficient multi-task model selector (EMMS), which employs large-scale foundation models to transform diverse label formats such as categories, texts, and bounding boxes of different downstream tasks into a unified noisy label embedding. EMMS can estimate a model’s transferability through a simple weighted linear regression, which can be efficiently solved by an alternating minimization algorithm with a convergence guarantee. Extensive experiments on 5 downstream tasks with 24 datasets show that EMMS is fast, effective, and generic enough to assess the transferability of pre-trained models, making it the first model selection method in the multi-task scenario. For instance, compared with the state- of-the-art method LogME enhanced by our label embeddings, EMMS achieves 9.0%, 26.3%, 20.1%, 54.8%, 12.2% performance gain on image recognition, referring, captioning, visual question answering, and text question answering, while bringing 5.13×, 6.29×, 3.59×, 6.19×, and 5.66× speedup in wall-clock time, respectively. The code is available at https://github.com/OpenGVLab/Multitask-Model-Selector.

----

## [1435] Feature Likelihood Score: Evaluating the Generalization of Generative Models Using Samples

**Authors**: *Marco Jiralerspong, Avishek Joey Bose, Ian Gemp, Chongli Qin, Yoram Bachrach, Gauthier Gidel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68b138608ef80b08d65b1bd9594d9559-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68b138608ef80b08d65b1bd9594d9559-Abstract-Conference.html)

**Abstract**:

The past few years have seen impressive progress in the development of deep generative models capable of producing high-dimensional, complex, and photo-realistic data. However, current methods for evaluating such models remain incomplete: standard likelihood-based metrics do not always apply and rarely correlate with perceptual fidelity, while sample-based metrics, such as FID, are insensitive to overfitting, i.e., inability to generalize beyond the training set. To address these limitations, we propose a new metric called the Feature Likelihood Divergence (FLD), a parametric sample-based score that uses density estimation to provide a comprehensive trichotomic evaluation accounting for novelty (i.e., different from the training samples), fidelity, and diversity of generated samples.  We empirically demonstrate the ability of FLD to identify specific overfitting problem cases, where previously proposed metrics fail. We also extensively evaluate FLD on various image datasets and model classes, demonstrating its ability to match intuitions of previous metrics like FID while offering a more comprehensive evaluation of generative models.

----

## [1436] LOVM: Language-Only Vision Model Selection

**Authors**: *Orr Zohar, Shih-Cheng Huang, Kuan-Chieh Wang, Serena Yeung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68c33c4e6fc97f7b31c964dc83303a28-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/68c33c4e6fc97f7b31c964dc83303a28-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Pre-trained multi-modal vision-language models (VLMs) are becoming increasingly popular due to their exceptional performance on downstream vision applications, particularly in the few- and zero-shot settings. However, selecting the best-performing VLM for some downstream applications is non-trivial, as it is dataset and task-dependent. Meanwhile, the exhaustive evaluation of all available VLMs on a novel application is not only time and  computationally demanding but also necessitates the collection of a labeled dataset for evaluation. As the number of open-source VLM variants increases, there is a need for an efficient model selection strategy that does not require access to a curated evaluation dataset. This paper proposes a novel task and benchmark for efficiently evaluating VLMs' zero-shot performance on downstream applications without access to the downstream task dataset. Specifically, we introduce a new task LOVM: Language-Only  Vision  Model Selection , where methods are expected to perform both model selection and performance prediction based solely on a text description of the desired downstream application. We then introduced an extensive LOVM benchmark consisting of ground-truth evaluations of 35 pre-trained VLMs and 23 datasets, where methods are expected to rank the pre-trained VLMs and predict their zero-shot performance.

----

## [1437] Statistical Analysis of Quantum State Learning Process in Quantum Neural Networks

**Authors**: *Hao-Kai Zhang, Chenghong Zhu, Mingrui Jing, Xin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/68efc144ad3b41108f779b51b9fb1300-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/68efc144ad3b41108f779b51b9fb1300-Abstract-Conference.html)

**Abstract**:

Quantum neural networks (QNNs) have been a promising framework in pursuing near-term quantum advantage in various fields, where many applications can be viewed as learning a quantum state that encodes useful data. As a quantum analog of probability distribution learning, quantum state learning is theoretically and practically essential in quantum machine learning. In this paper, we develop a no-go theorem for learning an unknown quantum state with QNNs even starting from a high-fidelity initial state. We prove that when the loss value is lower than a critical threshold, the probability of avoiding local minima vanishes exponentially with the qubit count, while only grows polynomially with the circuit depth. The curvature of local minima is concentrated to the quantum Fisher information times a loss-dependent constant, which characterizes the sensibility of the output state with respect to parameters in QNNs. These results hold for any circuit structures, initialization strategies, and work for both fixed ansatzes and adaptive methods. Extensive numerical simulations are performed to validate our theoretical results. Our findings place generic limits on good initial guesses and adaptive methods for improving the learnability and scalability of QNNs, and deepen the understanding of prior information's role in QNNs.

----

## [1438] SOL: Sampling-based Optimal Linear bounding of arbitrary scalar functions

**Authors**: *Yuriy Biktairov, Jyotirmoy Deshmukh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690b93e9ab0cc3b1d88b32f6f473ce69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690b93e9ab0cc3b1d88b32f6f473ce69-Abstract-Conference.html)

**Abstract**:

Finding tight linear bounds for activation functions in neural networksis an essential part of several state of the art neural network robustness certification tools. An activation function is an arbitrary, nonlinear,scalar function $f: \mathbb{R}^d \rightarrow \mathbb{R}$. In the existing work on robustness certification, such bounds have been computed using human ingenuity for a handful of the most popular activation functions. While a number of heuristics have been proposed for bounding arbitrary functions,no analysis of the tightness optimality for general scalar functions has been offered yet, to the best of our knowledge. We fill this gap by formulating a concise optimality criterion for tightness of the approximation which allows us tobuild optimal bounds for any function convex in the region of interest $R$. Fora more general class of functions Lipshitz-continuous in $R$ we propose a sampling-based approach (SOL) which, given an instance of the bounding problem, efficiently computes the tightest linear bounds within a given $\varepsilon > 0$ threshold. We leverage an adaptive sampling technique to iteratively build a setof sample points suitable for representing the target activation function. While the theoretical worst case time complexity of our approach is$O(\varepsilon^{-2d})$,it typically only takes $O(\log^{\beta} \frac{1}{\varepsilon})$ time for some $\beta \ge 1$ and isthus sufficiently fast in practice. We provide empirical evidence of SOL's practicalityby incorporating it into a robustness certifier and observing that itproduces similar or higher certification rates while taking as low as quarter of the time compared to the other methods.

----

## [1439] Opening the Vocabulary of Egocentric Actions

**Authors**: *Dibyadip Chatterjee, Fadime Sener, Shugao Ma, Angela Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690e82a09bcb3f101831962bf3cb54ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690e82a09bcb3f101831962bf3cb54ec-Abstract-Conference.html)

**Abstract**:

Human actions in egocentric videos often feature hand-object interactions composed of a verb (performed by the hand) applied to an object. Despite their extensive scaling up, egocentric datasets still face two limitations â€” sparsity of action compositions and a closed set of interacting objects. This paper proposes a novel open vocabulary action recognition task. Given a set of verbs and objects observed during training, the goal is to generalize the verbs to an open vocabulary of actions with seen and novel objects. To this end, we decouple the verb and object predictions via an object-agnostic verb encoder and a prompt-based object encoder. The prompting leverages CLIP representations to predict an open vocabulary of interacting objects. We create open vocabulary benchmarks on the EPIC-KITCHENS-100 and Assembly101 datasets; whereas closed-action methods fail to generalize, our proposed method is effective. In addition, our object encoder significantly outperforms existing open-vocabulary visual recognition methods in recognizing novel interacting objects.

----

## [1440] On the Pareto Front of Multilingual Neural Machine Translation

**Authors**: *Liang Chen, Shuming Ma, Dongdong Zhang, Furu Wei, Baobao Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/690eb240baf1180b69dac48fc905c918-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/690eb240baf1180b69dac48fc905c918-Abstract-Conference.html)

**Abstract**:

In this work, we study how the performance of a given direction changes with its sampling ratio in Multilingual Neural Machine Translation (MNMT). By training over 200 multilingual models with various model sizes, data sizes, and language directions, we find it interesting that the performance of certain translation direction does not always improve with the increase of its weight in the multi-task optimization objective. Accordingly, scalarization method leads to a multitask trade-off front that deviates from the traditional Pareto front when there exists data imbalance in the training corpus, which poses a great challenge to improve the overall performance of all directions. Based on our observations, we propose the Double Power Law to predict the unique performance trade-off front in MNMT, which is robust across various languages, data adequacy, and the number of tasks. Finally, we formulate the sample ratio selection problem in MNMT as an optimization problem based on the Double Power Law. Extensive experiments show that it achieves better performance than temperature searching and gradient manipulation methods with only 1/5 to 1/2 of the total training budget. We release the code at https://github.com/pkunlp-icler/ParetoMNMT for reproduction.

----

## [1441] Hierarchically Gated Recurrent Neural Network for Sequence Modeling

**Authors**: *Zhen Qin, Songlin Yang, Yiran Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html)

**Abstract**:

Transformers have surpassed RNNs in popularity due to their superior abilities in parallel training and long-term dependency modeling.Recently, there has been a renewed interest in using linear RNNs for efficient sequence modeling.These linear RNNs often employ gating mechanisms in the output of the linear recurrence layer while ignoring the significance of using forget gates within the recurrence. In this paper, we propose a gated linear RNN model dubbed Hierarchically Gated Recurrent Neural Network (HGRN), which includes forget gates that are lower bounded by a learnable value. The lower bound increases monotonically when moving up layers. This allows the upper layers to model long-term dependencies and the lower layers to model more local, short-term dependencies. Experiments on language modeling, image classification, and long-range arena benchmarks showcase the efficiency and effectiveness of our proposed model. The source code is available at https://github.com/OpenNLPLab/HGRN.

----

## [1442] Why Did This Model Forecast This Future? Information-Theoretic Saliency for Counterfactual Explanations of Probabilistic Regression Models

**Authors**: *Chirag Raman, Alec Nonnemaker, Amelia Villegas-Morcillo, Hayley Hung, Marco Loog*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/694ec0018b9fd0ebe863ec29fa5a89b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/694ec0018b9fd0ebe863ec29fa5a89b9-Abstract-Conference.html)

**Abstract**:

We propose a post hoc saliency-based explanation framework for counterfactual reasoning in probabilistic multivariate time-series forecasting (regression) settings. Building upon Miller's framework of explanations derived from research in multiple social science disciplines, we establish a conceptual link between counterfactual reasoning and saliency-based explanation techniques. To address the lack of a principled notion of saliency, we leverage a unifying definition of information-theoretic saliency grounded in preattentive human visual cognition and extend it to forecasting settings. Specifically, we obtain a closed-form expression for commonly used density functions to identify which observed timesteps appear salient to an underlying model in making its probabilistic forecasts. We empirically validate our framework in a principled manner using synthetic data to establish ground-truth saliency that is unavailable for real-world data. Finally, using real-world data and forecasting models, we demonstrate how our framework can assist domain experts in forming new data-driven hypotheses about the causal relationships between features in the wild.

----

## [1443] Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions

**Authors**: *Kai Liu, Zhihang Fu, Chao Chen, Sheng Jin, Ze Chen, Mingyuan Tao, Rongxin Jiang, Jieping Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/695b6f9490d27d852e439e35c56e73e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/695b6f9490d27d852e439e35c56e73e3-Abstract-Conference.html)

**Abstract**:

The key to OOD detection has two aspects: generalized feature representation and precise category description. Recently, vision-language models such as CLIP provide significant advances in both two issues, but constructing precise category descriptions is still in its infancy due to the absence of unseen categories. This work introduces two hierarchical contexts, namely perceptual context and spurious context, to carefully describe the precise category boundary through automatic prompt tuning. Specifically, perceptual contexts perceive the inter-category difference (e.g., cats vs apples) for current classification tasks, while spurious contexts further identify spurious (similar but exactly not) OOD samples for every single category (e.g., cats vs panthers, apples vs peaches). The two contexts hierarchically construct the precise description for a certain category, which is, first roughly classifying a sample to the predicted category and then delicately identifying whether it is truly an ID sample or actually OOD. Moreover, the precise descriptions for those categories within the vision-language framework present a novel application: CATegory-EXtensible OOD detection (CATEX). One can efficiently extend the set of recognizable categories by simply merging the hierarchical contexts learned under different sub-task settings. And extensive experiments are conducted to demonstrate CATEXâ€™s effectiveness, robustness, and category-extensibility. For instance, CATEX consistently surpasses the rivals by a large margin with several protocols on the challenging ImageNet-1K dataset. In addition, we offer new insights on how to efficiently scale up the prompt engineering in vision-language models to recognize thousands of object categories, as well as how to incorporate large language models (like GPT-3) to boost zero-shot applications.

----

## [1444] Online Corrupted User Detection and Regret Minimization

**Authors**: *Zhiyong Wang, Jize Xie, Tong Yu, Shuai Li, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html)

**Abstract**:

In real-world online web systems, multiple users usually arrive sequentially into the system. For applications like click fraud and fake reviews, some users can maliciously perform corrupted (disrupted) behaviors to trick the system. Therefore, it is crucial to design efficient online learning algorithms to robustly learn from potentially corrupted user behaviors and accurately identify the corrupted users in an online manner.  Existing works propose bandit algorithms robust to adversarial corruption. However, these algorithms are designed for a single user, and cannot leverage the implicit social relations among multiple users for more efficient learning. Moreover, none of them consider how to detect corrupted users online in the multiple-user scenario. In this paper, we present an important online learning problem named LOCUD to learn and utilize unknown user relations from disrupted behaviors to speed up learning, and identify the corrupted users in an online setting. To robustly learn and utilize the unknown relations among potentially corrupted users, we propose a novel bandit algorithm RCLUB-WCU. To detect the corrupted users, we devise a novel online detection algorithm OCCUD based on RCLUB-WCU's inferred user relations. We prove a regret upper bound for RCLUB-WCU, which asymptotically matches the lower bound with respect to $T$ up to logarithmic factors, and matches the state-of-the-art results in degenerate cases. We also give a theoretical guarantee for the detection accuracy of OCCUD. With extensive experiments, our methods achieve superior performance over previous bandit algorithms and high corrupted user detection accuracy.

----

## [1445] Nash Regret Guarantees for Linear Bandits

**Authors**: *Ayush Sawarni, Soumyabrata Pal, Siddharth Barman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69bf9fd8d3b7b792b6c8c19149024d22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69bf9fd8d3b7b792b6c8c19149024d22-Abstract-Conference.html)

**Abstract**:

We obtain essentially tight upper bounds for a strengthened notion of regret in the stochastic linear bandits framework. The strengthening---referred to as Nash regret---is defined as the difference between the (a priori unknown) optimum and the geometric mean of expected rewards accumulated by the linear bandit algorithm. Since the geometric mean corresponds to the well-studied Nash social welfare (NSW) function, this formulation quantifies the performance of a bandit algorithm as the collective welfare it generates across rounds. NSW is known to satisfy fairness axioms and, hence, an upper bound on Nash regret provides a principled fairness guarantee.    We consider the stochastic linear bandits problem over a horizon of $\mathsf{T}$ rounds and with a set of arms ${\cal X}$ in ambient dimension $d$. Furthermore, we focus on settings in which the stochastic reward---associated with each arm in ${\cal X}$---is a non-negative, sub-Poisson random variable. For this setting, we develop an algorithm that achieves a Nash regret of $O\left( \sqrt{\frac{d}{\mathsf{T}}} \log(\mathsf{T} |{\cal X}|)\right)$. In addition, addressing linear bandit instances in which the set of arms ${\cal X}$ is not necessarily finite, we obtain a Nash regret upper bound of $O\left( \frac{d^\frac{5}{4}}{\sqrt{\mathsf{T}}}  \log(\mathsf{T})\right)$. Since bounded random variables are sub-Poisson, these results hold for bounded, non-negative rewards. Our linear bandit algorithm is built upon the successive elimination method with novel technical insights, including tailored concentration bounds and the use of sampling via John ellipsoid in conjunction with the Kieferâ€“Wolfowitz optimal design.

----

## [1446] ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

**Authors**: *Haoran You, Huihong Shi, Yipin Guo, Yingyan Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html)

**Abstract**:

Vision Transformers (ViTs) have shown impressive performance and have become a unified backbone for multiple vision tasks. However, both the attention mechanism and multi-layer perceptrons (MLPs) in ViTs are not sufficiently efficient due to dense multiplications, leading to costly training and inference. To this end, we propose to reparameterize pre-trained ViTs with a mixture of multiplication primitives, e.g., bitwise shifts and additions, towards a new type of multiplication-reduced model, dubbed $\textbf{ShiftAddViT}$, which aims to achieve end-to-end inference speedups on GPUs without requiring training from scratch. Specifically, all $\texttt{MatMuls}$ among queries, keys, and values are reparameterized using additive kernels, after mapping queries and keys to binary codes in Hamming space. The remaining MLPs or linear layers are then reparameterized with shift kernels. We utilize TVM to implement and optimize those customized kernels for practical hardware deployment on GPUs. We find that such a reparameterization on (quadratic or linear) attention maintains model accuracy, while inevitably leading to accuracy drops when being applied to MLPs. To marry the best of both worlds, we further propose a new mixture of experts (MoE) framework to reparameterize MLPs by taking multiplication or its primitives as experts, e.g., multiplication and shift, and designing a new latency-aware load-balancing loss. Such a loss helps to train a generic router for assigning a dynamic amount of input tokens to different experts according to their latency. In principle, the faster the experts run, the more input tokens they are assigned. Extensive experiments on various 2D/3D Transformer-based vision tasks consistently validate the effectiveness of our proposed ShiftAddViT, achieving up to $\textbf{5.18$\times$}$ latency reductions on GPUs and $\textbf{42.9}$% energy savings, while maintaining a comparable accuracy as original or efficient ViTs. Codes and models are available at https://github.com/GATECH-EIC/ShiftAddViT.

----

## [1447] Optimal Extragradient-Based Algorithms for Stochastic Variational Inequalities with Separable Structure

**Authors**: *Angela Yuan, Chris Junchi Li, Gauthier Gidel, Michael I. Jordan, Quanquan Gu, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html)

**Abstract**:

We consider the problem of solving stochastic monotone variational inequalities with a separable structure using a stochastic first-order oracle. Building on standard extragradient for variational inequalities we propose a novel algorithm---stochastic \emph{accelerated gradient-extragradient} (AG-EG)---for strongly monotone variational inequalities (VIs). Our approach combines the strengths of extragradient and Nesterov acceleration. By showing that its iterates remain in a bounded domain and applying scheduled restarting, we prove that AG-EG has an optimal convergence rate for strongly monotone VIs. Furthermore, when specializing to the particular case of bilinearly coupled strongly-convex-strongly-concave saddle-point problems, including bilinear games, our algorithm achieves fine-grained convergence rates that match the respective lower bounds, with the stochasticity being characterized by an additive statistical error term that is optimal up to a constant prefactor.

----

## [1448] Combinatorial Group Testing with Selfish Agents

**Authors**: *Georgios Chionas, Dariusz R. Kowalski, Piotr Krysta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/69f98acf161316ed896047e45da3bc0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/69f98acf161316ed896047e45da3bc0c-Abstract-Conference.html)

**Abstract**:

We study the Combinatorial Group Testing (CGT) problem in a novel game-theoretic framework, with a solution concept of Adversarial Equilibrium (AE). In this new framework, we have $n$ selfish agents corresponding to the elements of the universe $[n] =\{0,1,\ldots,n-1\}$ and a hidden set $K \subseteq [n]$ of active agents of size $|K| = k \ll n$. In each round of the game, each active agent decides if it is present in a query $Q \subseteq [n]$, and all agents receive feedback on $Q \cap K$. The goal of each active agent is to assure that its id could be learned from the feedback as early as possible. We present a comprehensive set of results in this new game, where we design and analyze adaptive algorithmic strategies of agents which are AE's. In particular, if $k$ is known to the agents, then we design adaptive AE strategies with provably near optimal learning time of $O(k \log(n/k))$. In the case of unknown $k$, we design an adaptive AE strategies with learning time of order $n^k$, and we prove a lower bound of $\Omega(n)$ on the learning time of any such algorithmic strategies. This shows a strong separations between the two models of known and unknown $k$, as well as between the classic CGT, i.e., without selfish agents, and our game theoretic CGT model.

----

## [1449] A Hierarchical Spatial Transformer for Massive Point Samples in Continuous Space

**Authors**: *Wenchong He, Zhe Jiang, Tingsong Xiao, Zelin Xu, Shigang Chen, Ronald Fick, Miles Medina, Christine Angelini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a0480190bbe6b622c7f1d3aa9be9c0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a0480190bbe6b622c7f1d3aa9be9c0f-Abstract-Conference.html)

**Abstract**:

Transformers are widely used deep learning architectures. Existing transformers are mostly designed for sequences (texts or time series), images or videos, and graphs. This paper proposes a novel transformer model for massive (up to a million) point samples in continuous space. Such data are ubiquitous in environment sciences (e.g., sensor observations), numerical simulations (e.g., particle-laden flow, astrophysics), and location-based services (e.g., POIs and trajectories). However, designing a transformer for massive spatial points is non-trivial due to several challenges, including implicit long-range and multi-scale dependency on irregular points in continuous space, a non-uniform point distribution, the potential high computational costs of calculating all-pair attention across massive points, and the risks of over-confident predictions due to varying point density. To address these challenges, we propose a new hierarchical spatial transformer model, which includes multi-resolution representation learning within a quad-tree hierarchy and efficient spatial attention via coarse approximation. We also design an uncertainty quantification branch to estimate prediction confidence related to input feature noise and point sparsity. We provide a theoretical analysis of computational time complexity and memory costs. Extensive experiments on both real-world and synthetic datasets show that our method outperforms multiple baselines in prediction accuracy and our model can scale up to one million points on one NVIDIA A100 GPU. The code is available at https://github.com/spatialdatasciencegroup/HST

----

## [1450] Rethinking Gauss-Newton for learning over-parameterized models

**Authors**: *Michael Arbel, Romain Menegaux, Pierre Wolinski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a14c7f9fb3f42645cfa6bd5aa446819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a14c7f9fb3f42645cfa6bd5aa446819-Abstract-Conference.html)

**Abstract**:

This work studies the global convergence and implicit bias of Gauss Newton's (GN) when optimizing over-parameterized one-hidden layer networks in the mean-field regime. We first establish a global convergence result for GN in the continuous-time limit exhibiting a faster convergence rate compared to GD due to improved conditioning. We then perform an empirical study on a synthetic regression task to investigate the implicit bias of GN's method.While GN is consistently faster than GD in finding a global optimum, the learned model generalizes well on test data when starting from random initial weights with a small variance and using a small step size to slow down convergence. Specifically, our study shows that such a setting results in a hidden learning phenomenon, where the dynamics are able to recover features with good generalization properties despite the model having sub-optimal training and test performances due to an under-optimized linear layer. This study exhibits a trade-off between the convergence speed of GN and the generalization ability of the learned solution.

----

## [1451] Knowledge Distillation for High Dimensional Search Index

**Authors**: *Zepu Lu, Jin Chen, Defu Lian, Zaixi Zhang, Yong Ge, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a15378acabd1aef017ec79a3ed744d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a15378acabd1aef017ec79a3ed744d2-Abstract-Conference.html)

**Abstract**:

Lightweight compressed models are prevalent in Approximate Nearest Neighbor Search (ANNS) and Maximum Inner Product Search (MIPS) owing to their superiority of retrieval efficiency in large-scale datasets. However, results given by compressed methods are less accurate due to the curse of dimension and the limitations of optimization objectives (e.g., lacking interactions between queries and documents). Thus, we are encouraged to design a new learning algorithm for the compressed search index on high dimensions to improve retrieval performance. In this paper, we propose a novel KnowledgeDistillation for high dimensional search index framework (KDindex), with the aim of efficiently learning lightweight indexes by distilling knowledge from high-precision ANNS and MIPS models such as graph-based indexes. Specifically, the student is guided to keep the same ranking order of the top-k relevant results yielded by the teacher model, which acts as the additional supervision signals between queries and documents to learn the similarities between documents. Furthermore, to avoid the trivial solutions that all candidates are partitioned to the same centroid, the reconstruction loss that minimizes the compressed error, and the posting list balance strategy that equally allocates the candidates, are integrated into the learning objective. Experiment results demonstrate that KDindex outperforms existing learnable quantization-based indexes and is 40Ã— lighter than the state-of-the-art non-exhaustive methods while achieving comparable recall quality.

----

## [1452] Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations

**Authors**: *Yu Cao, Jingrun Chen, Yixin Luo, Xiang Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a27ee6f66d13557f15f070274c51721-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a27ee6f66d13557f15f070274c51721-Abstract-Conference.html)

**Abstract**:

The diffusion model has shown remarkable success in computer vision, but it remains unclear whether the ODE-based probability flow or the SDE-based diffusion model is more superior and under what circumstances. Comparing the two is challenging due to dependencies on data distributions, score training, and other numerical issues. In this paper, we study the problem mathematically for two limiting scenarios: the zero diffusion (ODE) case and the large diffusion case. We first introduce a pulse-shape error to perturb the score function and analyze error accumulation of sampling quality, followed by a thorough analysis for generalization to arbitrary error. Our findings indicate that when the perturbation occurs at the end of the generative process, the ODE model outperforms the SDE model with a large diffusion coefficient. However, when the perturbation occurs earlier, the SDE model outperforms the ODE model, and we demonstrate that the error of sample generation due to such a pulse-shape perturbation is exponentially suppressed as the diffusion term's magnitude increases to infinity. Numerical validation of this phenomenon is provided using Gaussian, Gaussian mixture, and Swiss roll distribution, as well as realistic datasets like MNIST and CIFAR-10.

----

## [1453] PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance

**Authors**: *Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Although large language models (LLMs) have shown great performance in natural language processing (NLP) in the financial domain, there are no publicly available financially tailored LLMs, instruction tuning datasets, and evaluation benchmarks, which is critical for continually pushing forward the open-source development of financial artificial intelligence (AI). This paper introduces PIXIU, a comprehensive framework including the first financial LLM based on fine-tuning LLaMA with instruction data, the first instruction data with 128K data samples to support the fine-tuning, and an evaluation benchmark with 8 tasks and 15 datasets. We first construct the large-scale multi-task instruction data considering a variety of financial tasks, financial document types, and financial data modalities. We then propose a financial LLM called FinMA by fine-tuning LLaMA with the constructed dataset to be able to follow instructions for various financial tasks. To support the evaluation of financial LLMs, we propose a standardized benchmark that covers a set of critical financial tasks, including six financial NLP tasks and two financial prediction tasks. With this benchmark, we conduct a detailed analysis of FinMA and several existing LLMs, uncovering their strengths and weaknesses in handling critical financial tasks. The model, datasets, benchmark, and experimental results are open-sourced to facilitate future research in financial AI.

----

## [1454] EgoDistill: Egocentric Head Motion Distillation for Efficient Video Understanding

**Authors**: *Shuhan Tan, Tushar Nagarajan, Kristen Grauman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a412f0037b0df295a39a198666ea6a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a412f0037b0df295a39a198666ea6a6-Abstract-Conference.html)

**Abstract**:

Recent advances in egocentric video understanding models are promising, but their heavy computational expense is a barrier for many real-world applications. To address this challenge, we propose EgoDistill, a distillation-based approach that learns to reconstruct heavy ego-centric video clip features by combining the semantics from a sparse set of video frames with head motion from lightweight IMU readings. We further devise a novel IMU-based self-supervised pretraining strategy. Our method leads to significant improvements in efficiency, requiring 200Ã— fewer GFLOPs than equivalent video models. We demonstrate its effectiveness on the Ego4D and EPIC- Kitchens datasets, where our method outperforms state-of-the-art efficient video understanding methods.

----

## [1455] Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework

**Authors**: *Weiduo Liao, Ying Wei, Mingchen Jiang, Qingfu Zhang, Hisao Ishibuchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a42b45af2b72e6e5b5e3a6fe695809f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a42b45af2b72e6e5b5e3a6fe695809f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Compositionality facilitates the comprehension of novel objects using acquired concepts and the maintenance of a knowledge pool. This is particularly crucial for continual learners to prevent catastrophic forgetting and enable compositionally forward transfer of knowledge. However, the existing state-of-the-art benchmarks inadequately evaluate the capability of compositional generalization, leaving an intriguing question unanswered. To comprehensively assess this capability, we introduce two vision benchmarks, namely Compositional GQA (CGQA) and Compositional OBJects365 (COBJ), along with a novel evaluation framework called Compositional Few-Shot Testing (CFST). These benchmarks evaluate the systematicity, productivity, and substitutivity aspects of compositional generalization. Experimental results on five baselines and two modularity-based methods demonstrate that current continual learning techniques do exhibit somewhat favorable compositionality in their learned feature extractors. Nonetheless, further efforts are required in developing modularity-based approaches to enhance compositional generalization. We anticipate that our proposed benchmarks and evaluation protocol will foster research on continual learning and compositionality.

----

## [1456] Unsupervised Protein-Ligand Binding Energy Prediction via Neural Euler's Rotation Equation

**Authors**: *Wengong Jin, Siranush Sarkizova, Xun Chen, Nir Hacohen, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a45a1b0697ee086bd8bf494cacc6567-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a45a1b0697ee086bd8bf494cacc6567-Abstract-Conference.html)

**Abstract**:

Protein-ligand binding prediction is a fundamental problem in AI-driven drug discovery. Previous work focused on supervised learning methods for small molecules where binding affinity data is abundant, but it is hard to apply the same strategy to other ligand classes like antibodies where labelled data is limited. In this paper, we explore unsupervised approaches and reformulate binding energy prediction as a generative modeling task. Specifically, we train an energy-based model on a set of unlabelled protein-ligand complexes using SE(3) denoising score matching (DSM) and interpret its log-likelihood as binding affinity. Our key contribution is a new equivariant rotation prediction network called Neural Euler's Rotation Equations (NERE) for SE(3) DSM. It predicts a rotation by modeling the force and torque between protein and ligand atoms, where the force is defined as the gradient of an energy function with respect to atom coordinates. Using two protein-ligand and antibody-antigen binding affinity prediction benchmarks, we show that NERE outperforms all unsupervised baselines (physics-based potentials and protein language models) in both cases and surpasses supervised baselines in the antibody case.

----

## [1457] ProteinNPT: Improving protein property prediction and design with non-parametric transformers

**Authors**: *Pascal Notin, Ruben Weitzman, Debora S. Marks, Yarin Gal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html)

**Abstract**:

Protein design holds immense potential for optimizing naturally occurring proteins, with broad applications in drug discovery, material design, and sustainability. However, computational methods for protein engineering are  confronted with significant challenges, such as an expansive design space, sparse functional regions, and a scarcity of available labels. These issues are further exacerbated in practice by the fact most real-life design scenarios necessitate the simultaneous optimization of multiple properties. In this work, we introduce ProteinNPT, a non-parametric transformer variant tailored to protein sequences and particularly suited to label-scarce and multi-task learning settings. We first focus on the supervised fitness prediction setting and develop several cross-validation schemes which support robust performance assessment. We subsequently reimplement prior top-performing baselines, introduce several extensions of these baselines by integrating diverse branches of the protein engineering literature, and demonstrate that ProteinNPT consistently outperforms all of them across a diverse set of protein property prediction tasks. Finally, we demonstrate the value of our approach for iterative protein design across extensive in silico Bayesian optimization and conditional sampling experiments.

----

## [1458] Mitigating Source Bias for Fairer Weak Supervision

**Authors**: *Changho Shin, Sonia Cromp, Dyah Adila, Frederic Sala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html)

**Abstract**:

Weak supervision enables efficient development of training sets by reducing the need for ground truth labels. However, the techniques that make weak supervision attractive---such as integrating any source of signal to estimate unknown labels---also entail the danger that the produced pseudolabels are highly biased. Surprisingly, given everyday use and the potential for increased bias, weak supervision has not been studied from the point of view of fairness. We begin such a study, starting with the observation that even when a fair model can be built from a dataset with access to ground-truth labels, the corresponding dataset labeled via weak supervision can be arbitrarily unfair. To address this, we propose and empirically validate a model for source unfairness in weak supervision, then introduce a simple counterfactual fairness-based technique that can mitigate these biases. Theoretically, we show that it is possible for our approach to simultaneously improve both accuracy and fairness---in contrast to standard fairness approaches that suffer from tradeoffs. Empirically, we show that our technique improves accuracy on weak supervision baselines by as much as 32\% while reducing demographic parity gap by 82.5\%. A simple extension of our method aimed at maximizing performance produces state-of-the-art performance in five out of ten datasets in the WRENCH benchmark.

----

## [1459] GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

**Authors**: *Xin Zheng, Miao Zhang, Chunyang Chen, Soheila Molaei, Chuan Zhou, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a55f024db3f771194bdadc8f3a35381-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a55f024db3f771194bdadc8f3a35381-Abstract-Conference.html)

**Abstract**:

Evaluating the performance of graph neural networks (GNNs) is an essential task for practical GNN model deployment and serving, as deployed GNNs face significant performance uncertainty when inferring on unseen and unlabeled test graphs, due to mismatched training-test graph distributions. In this paper, we study a new problem, GNN model evaluation, that aims to assess the performance of a specific GNN model trained on labeled and observed graphs, by precisely estimating its performance (e.g., node classification accuracy) on unseen graphs without labels. Concretely, we propose a two-stage GNN model evaluation framework, including (1) DiscGraph set construction and (2) GNNEvaluator training and inference. The DiscGraph set captures wide-range and diverse graph data distribution discrepancies through a discrepancy measurement function, which exploits the GNN outputs of latent node embeddings and node class predictions. Under the effective training supervision from the DiscGraph set, GNNEvaluator learns to precisely estimate node classification accuracy of the to-be-evaluated GNN model and makes an accurate inference for evaluating GNN model performance. Extensive experiments on real-world unseen and unlabeled test graphs demonstrate the effectiveness of our proposed method for GNN model evaluation.

----

## [1460] Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures

**Authors**: *Runa Eschenhagen, Alexander Immer, Richard E. Turner, Frank Schneider, Philipp Hennig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6679e3d5b9f7d5f09cdb79a5fc3fd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6679e3d5b9f7d5f09cdb79a5fc3fd8-Abstract-Conference.html)

**Abstract**:

The core components of many modern neural network architectures, such as transformers, convolutional, or graph neural networks, can be expressed as linear layers with *weight-sharing*. Kronecker-Factored Approximate Curvature (K-FAC), a second-order optimisation method, has shown promise to speed up neural network training and thereby reduce computational costs. However, there is currently no framework to apply it to generic architectures, specifically ones with linear weight-sharing layers. In this work, we identify two different settings of linear weight-sharing layers which motivate two flavours of K-FAC -- *expand* and *reduce*. We show that they are exact for deep linear networks with weight-sharing in their respective setting. Notably, K-FAC-reduce is generally faster than K-FAC-expand, which we leverage to speed up automatic hyperparameter selection via optimising the marginal likelihood for a Wide ResNet. Finally, we observe little difference between these two K-FAC variations when using them to train both a graph neural network and a vision transformer. However, both variations are able to reach a fixed validation metric target in $50$-$75$\% of the number of steps of a first-order reference run, which translates into a comparable improvement in wall-clock time. This highlights the potential of applying K-FAC to modern neural network architectures.

----

## [1461] Tanimoto Random Features for Scalable Molecular Machine Learning

**Authors**: *Austin Tripp, Sergio Bacallado, Sukriti Singh, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a69d44b3386e50c06f7107ef4f29302-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a69d44b3386e50c06f7107ef4f29302-Abstract-Conference.html)

**Abstract**:

The Tanimoto coefficient is commonly used to measure the similarity between molecules represented as discrete fingerprints,either as a distance metric or a positive definite kernel. While many kernel methods can be accelerated using random feature approximations, at present there is a lack of such approximations for the Tanimoto kernel. In this paper we propose two kinds of novel random features to allow this kernel to scale to large datasets, and in the process discover a novel extension of the kernel to real-valued vectors. We theoretically characterize these random features, and provide error bounds on the spectral norm of the Gram matrix. Experimentally, we show that these random features are effective at approximating the Tanimoto coefficient of real-world datasetsand are useful for molecular property prediction and optimization tasks. Future updates to this work will be available at http://arxiv.org/abs/2306.14809.

----

## [1462] Probabilistic Inference in Reinforcement Learning Done Right

**Authors**: *Jean Tarbouriech, Tor Lattimore, Brendan O'Donoghue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6e010edde1b8f2812f558b67a1974e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6e010edde1b8f2812f558b67a1974e-Abstract-Conference.html)

**Abstract**:

A popular perspective in Reinforcement learning (RL) casts the problem as probabilistic inference on a graphical model of the Markov decision process (MDP). The core object of study is the probability of each state-action pair being visited under the optimal policy. Previous approaches to approximate this quantity can be arbitrarily poor, leading to algorithms that do not implement genuine statistical inference and consequently do not perform well in challenging problems. In this work, we undertake a rigorous Bayesian treatment of the posterior probability of state-action optimality and clarify how it flows through the MDP. We first reveal that this quantity can indeed be used to generate a policy that explores efficiently, as measured by regret. Unfortunately, computing it is intractable, so we derive a new variational Bayesian approximation yielding a tractable convex optimization problem and establish that the resulting policy also explores efficiently. We call our approach VAPOR and show that it has strong connections to Thompson sampling, K-learning, and maximum entropy exploration. We conclude with some experiments demonstrating the performance advantage of a deep RL version of VAPOR.

----

## [1463] Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels

**Authors**: *Zhen Liu, Peitian Ma, Dongliang Chen, Wenbin Pei, Qianli Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a6ecedac816a24f92ad1f444b1edcb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a6ecedac816a24f92ad1f444b1edcb0-Abstract-Conference.html)

**Abstract**:

Deep Neural Networks (DNNs) have been criticized because they easily overfit noisy (incorrect) labels. To improve the robustness of DNNs, existing methods for image data regard samples with small training losses as correctly labeled data (small-loss criterion). Nevertheless, time series' discriminative patterns are easily distorted by external noises (i.e., frequency perturbations) during the recording process. This results in training losses of some time series samples that do not meet the small-loss criterion. Therefore, this paper proposes a deep learning paradigm called Scale-teaching to cope with time series noisy labels. Specifically, we design a fine-to-coarse cross-scale fusion mechanism for learning discriminative patterns by utilizing time series at different scales to train multiple DNNs simultaneously. Meanwhile, each network is trained in a cross-teaching manner by using complementary information from different scales to select small-loss samples as clean labels. For unselected large-loss samples, we introduce multi-scale embedding graph learning via label propagation to correct their labels by using selected clean samples. Experiments on multiple benchmark time series datasets demonstrate the superiority of the proposed Scale-teaching paradigm over state-of-the-art methods in terms of effectiveness and robustness.

----

## [1464] VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning

**Authors**: *Jiayi Guan, Guang Chen, Jiaming Ji, Long Yang, Ao Zhou, Zhijun Li, Changjun Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6a7c2a320f5f36bb98f8eb878c6f1180-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6a7c2a320f5f36bb98f8eb878c6f1180-Abstract-Conference.html)

**Abstract**:

Offline safe reinforcement learning (RL) algorithms promise to learn policies that satisfy safety constraints directly in offline datasets without interacting with the environment. This arrangement is particularly important in scenarios with high sampling costs and potential dangers, such as autonomous driving and robotics. However, the influence of safety constraints and out-of-distribution (OOD) actions have made it challenging for previous methods to achieve high reward returns while ensuring safety. In this work, we propose a Variational Optimization with Conservative Eestimation algorithm (VOCE) to solve the problem of optimizing safety policies in the offline dataset. Concretely, we reframe the problem of offline safe RL using probabilistic inference, which introduces variational distributions to make the optimization of policies more flexible. Subsequently, we utilize pessimistic estimation methods to estimate the Q-value of cost and reward, which mitigates the extrapolation errors induced by OOD actions. Finally, extensive experiments demonstrate that the VOCE algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety.

----

## [1465] Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark

**Authors**: *Lasse Hansen, Nabeel Seedat, Mihaela van der Schaar, Andrija Petrovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6aa9a05b929fb08ff46a58cab6cf860d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6aa9a05b929fb08ff46a58cab6cf860d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthetic data serves as an alternative in training machine learning models, particularly when real-world data is limited or inaccessible. However, ensuring that synthetic data mirrors the complex nuances of real-world data is a challenging task. This paper addresses this issue by exploring the potential of integrating data-centric AI techniques which profile the data to guide the synthetic data generation process. Moreover, we shed light on the often ignored consequences of neglecting these data profiles during synthetic data generation --- despite seemingly high statistical fidelity. Subsequently, we propose a novel framework to evaluate the integration of data profiles to guide the creation of more representative synthetic data. In an empirical study, we evaluate the performance of five state-of-the-art models for tabular data generation on eleven distinct tabular datasets. The findings offer critical insights into the successes and limitations of current synthetic data generation techniques. Finally, we provide practical recommendations for integrating data-centric insights into the synthetic data generation process, with a specific focus on classification performance, model selection, and feature selection. This study aims to reevaluate conventional approaches to synthetic data generation and promote the application of data-centric AI techniques in improving the quality and effectiveness of synthetic data.

----

## [1466] Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis

**Authors**: *Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html)

**Abstract**:

How can we tell whether two neural networks utilize the same internal processes for a particular computation? This question is pertinent for multiple subfields of neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of dynamics, and two networks performing the same computation with equivalent dynamics need not exhibit the same geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics, called Dynamical Similarity Analysis (DSA). Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare different systems passed through this embedding using a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. In four case studies, we demonstrate that our method disentangles conjugate and non-conjugate recurrent neural networks (RNNs), while geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method opens the door to comparative analyses of the essential temporal structure of computation in neural circuits.

----

## [1467] H-nobs: Achieving Certified Fairness and Robustness in Distributed Learning on Heterogeneous Datasets

**Authors**: *Guanqiang Zhou, Ping Xu, Yue Wang, Zhi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ad5d39b10e37915d7dfda2893d8e924-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ad5d39b10e37915d7dfda2893d8e924-Abstract-Conference.html)

**Abstract**:

Fairness and robustness are two important goals in the design of modern distributed learning systems. Despite a few prior works attempting to achieve both fairness and robustness, some key aspects of this direction remain underexplored. In this paper, we try to answer three largely unnoticed and unaddressed questions that are of paramount significance to this topic: (i) What makes jointly satisfying fairness and robustness difficult? (ii) Is it possible to establish theoretical guarantee for the dual property of fairness and robustness? (iii) How much does fairness have to sacrifice at the expense of robustness being incorporated into the system? To address these questions, we first identify data heterogeneity as the key difficulty of combining fairness and robustness. Accordingly, we propose a fair and robust framework called H-nobs which can offer certified fairness and robustness through the adoption of two key components, a fairness-promoting objective function and a simple robust aggregation scheme called norm-based screening (NBS). We explain in detail why NBS is the suitable scheme in our algorithm in contrast to other robust aggregation measures. In addition, we derive three convergence theorems for H-nobs in cases of the learning model being nonconvex, convex, and strongly convex respectively, which provide theoretical guarantees for both fairness and robustness. Further, we empirically investigate the influence of the robust mechanism (NBS) on the fairness performance of H-nobs, the very first attempt of such exploration.

----

## [1468] A Randomized Approach to Tight Privacy Accounting

**Authors**: *Jiachen T. Wang, Saeed Mahloujifar, Tong Wu, Ruoxi Jia, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html)

**Abstract**:

Bounding privacy leakage over compositions, i.e., privacy accounting, is a key challenge in differential privacy (DP). However, the privacy parameter ($\varepsilon$ or $\delta$) is often easy to estimate but hard to bound. In this paper, we propose a new differential privacy paradigm called estimate-verify-release (EVR), which tackles the challenges of providing a strict upper bound for the privacy parameter in DP compositions by converting an *estimate* of privacy parameter into a formal guarantee. The EVR paradigm first verifies whether the mechanism meets the *estimated* privacy guarantee, and then releases the query output based on the verification result. The core component of the EVR is privacy verification. We develop a randomized privacy verifier using Monte Carlo (MC) technique. Furthermore, we propose an MC-based DP accountant that outperforms existing DP accounting techniques in terms of accuracy and efficiency. MC-based DP verifier and accountant is applicable to an important and commonly used class of DP algorithms, including the famous DP-SGD. An empirical evaluation shows the proposed EVR paradigm improves the utility-privacy tradeoff for privacy-preserving machine learning.

----

## [1469] Triple Eagle: Simple, Fast and Practical Budget-Feasible Mechanisms

**Authors**: *Kai Han, You Wu, He Huang, Shuang Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html)

**Abstract**:

We revisit the classical problem of designing Budget-Feasible Mechanisms (BFMs) for submodular valuation functions, which has been extensively studied since the seminal paper of Singer [FOCSâ€™10] due to its wide applications in crowdsourcing and social marketing. We propose TripleEagle, a novel algorithmic framework for designing BFMs, based on which we present several simple yet effective BFMs thatachieve better approximation ratios than the state-of-the-art work for both monotone and non-monotone submodular valuation functions. Moreover, our BFMs are the first in the literature to achieve linear complexities while ensuring obvious strategyproofness, making them more practical than the previous BFMs. We conduct extensive experiments to evaluate the empirical performance of our BFMs, and the experimental results strongly demonstrate the efficiency and effectiveness of our approach.

----

## [1470] VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models

**Authors**: *Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html)

**Abstract**:

Diffusion Models (DMs) are state-of-the-art generative models that learn a reversible corruption process from iterative noise addition and denoising. They are the backbone of many generative AI applications, such as text-to-image conditional generation. However, recent studies have shown that basic unconditional DMs (e.g., DDPM and DDIM) are vulnerable to backdoor injection, a type of output manipulation attack triggered by a maliciously embedded pattern at model input. This paper presents a unified backdoor attack framework (VillanDiffusion) to expand the current scope of backdoor analysis for DMs. Our framework covers mainstream unconditional and conditional DMs (denoising-based and score-based) and various training-free samplers for holistic evaluations. Experiments show that our unified framework facilitates the backdoor analysis of different DM configurations and provides new insights into caption-based backdoor attacks on DMs.

----

## [1471] An Information Theory Perspective on Variance-Invariance-Covariance Regularization

**Authors**: *Ravid Shwartz-Ziv, Randall Balestriero, Kenji Kawaguchi, Tim G. J. Rudner, Yann LeCun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract-Conference.html)

**Abstract**:

Variance-Invariance-Covariance Regularization (VICReg) is a self-supervised learning (SSL) method that has shown promising results on a variety of tasks. However, the fundamental mechanisms underlying VICReg remain unexplored. In this paper, we present an information-theoretic perspective on the VICReg objective. We begin by deriving information-theoretic quantities for deterministic networks as an alternative to unrealistic stochastic network assumptions. We then relate the optimization of the VICReg objective to mutual information optimization, highlighting underlying assumptions and facilitating a constructive comparison with other SSL algorithms and derive a generalization bound for VICReg, revealing its inherent advantages for downstream tasks. Building on these results, we introduce a family of SSL methods derived from information-theoretic principles that outperform existing SSL techniques.

----

## [1472] Learning and processing the ordinal information of temporal sequences in recurrent neural circuits

**Authors**: *Xiaolong Zou, Zhikun Chu, Qinghai Guo, Jie Cheng, Bo Ho, Si Wu, Yuanyuan Mi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html)

**Abstract**:

Temporal sequence processing is fundamental in brain cognitive functions. Experimental data has indicated that the representations of ordinal information and contents of temporal sequences are disentangled in the brain, but the neural mechanism underlying this disentanglement remains largely unclear. Here, we investigate how recurrent neural circuits learn to represent the abstract order structure of temporal sequences, and how this disentangled representation of order structure from that of contents facilitates the processing of temporal sequences. We show that with an appropriate learn protocol, a recurrent neural circuit can learn a set of tree-structured attractor states to encode the corresponding tree-structured orders of given temporal sequences. This abstract temporal order template can then be bound with different contents, allowing for flexible and robust temporal sequence processing. Using a transfer learning task, we demonstrate that the reuse of a temporal order template facilitates the acquisition of new temporal sequences of the same or similar ordinal structure. Using a key-word spotting task, we demonstrate that the attractor representation of order structure improves the robustness of temporal sequence discrimination, if the ordinal information is the key to differentiate different sequences. We hope this study gives us insights into the neural mechanism of representing the ordinal information of temporal sequences in the brain, and helps us to develop brain-inspired temporal sequence processing algorithms.

----

## [1473] UNSSOR: Unsupervised Neural Speech Separation by Leveraging Over-determined Training Mixtures

**Authors**: *Zhong-Qiu Wang, Shinji Watanabe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b44765c9201730a27f7931afb4d7434-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b44765c9201730a27f7931afb4d7434-Abstract-Conference.html)

**Abstract**:

In reverberant conditions with multiple concurrent speakers, each microphone acquires a mixture signal of multiple speakers at a different location. In over-determined conditions where the microphones out-number speakers, we can narrow down the solutions to speaker images and realize unsupervised speech separation by leveraging each mixture signal as a constraint (i.e., the estimated speaker images at a microphone should add up to the mixture). Equipped with this insight, we propose UNSSOR, an algorithm for $\underline{u}$nsupervised $\underline{n}$eural $\underline{s}$peech $\underline{s}$eparation by leveraging $\underline{o}$ver-determined training mixtu$\underline{r}$es. At each training step, we feed an input mixture to a deep neural network (DNN) to produce an intermediate estimate for each speaker, linearly filter the estimates, and optimize a loss so that, at each microphone, the filtered estimates of all the speakers can add up to the mixture to satisfy the above constraint. We show that this loss can promote unsupervised separation of speakers. The linear filters are computed in each sub-band based on the mixture and DNN estimates through the forward convolutive prediction (FCP) algorithm. To address the frequency permutation problem incurred by using sub-band FCP, a loss term based on minimizing intra-source magnitude scattering is proposed. Although UNSSOR requires over-determined training mixtures, we can train DNNs to achieve under-determined separation (e.g., unsupervised monaural speech separation). Evaluation results on two-speaker separation in reverberant conditions show the effectiveness and potential of UNSSOR.

----

## [1474] Improving Self-supervised Molecular Representation Learning using Persistent Homology

**Authors**: *Yuankai Luo, Lei Shi, Veronika Thost*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b555e8552240d6dfe0767146c9ebf36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b555e8552240d6dfe0767146c9ebf36-Abstract-Conference.html)

**Abstract**:

Self-supervised learning (SSL) has great potential for molecular representation learning given the complexity of molecular graphs, the large amounts of unlabelled data available, the considerable cost of obtaining labels experimentally, and the hence often only small training datasets. The importance of the topic is reflected in the variety of paradigms and architectures that have been investigated recently, most focus on designing views for contrastive learning.In this paper, we study SSL based on persistent homology (PH), a mathematical tool for modeling topological features of data that persist across multiple scales. It has several unique features which particularly suit SSL, naturally offering: different views of the data, stability in terms of distance preservation, and the opportunity to flexibly incorporate domain knowledge.We (1) investigate an autoencoder, which shows the general representational power of PH, and (2) propose a contrastive loss that complements existing approaches. We rigorously evaluate our approach for molecular property prediction and demonstrate its particular features in improving the embedding space:after SSL, the representations are better and offer considerably more predictive power than the baselines over different probing tasks; our loss increases baseline performance, sometimes largely; and we often obtain substantial improvements over very small datasets, a common scenario in practice.

----

## [1475] Characteristic Circuits

**Authors**: *Zhongjie Yu, Martin Trapp, Kristian Kersting*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b61c278e483954fee502b49fe71cd14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b61c278e483954fee502b49fe71cd14-Abstract-Conference.html)

**Abstract**:

In many real-world scenarios it is crucial to be able to reliably and efficiently reason under uncertainty while capturing complex relationships in data.  Probabilistic circuits (PCs), a prominent family of tractable probabilistic models, offer a remedy to this challenge by composing simple, tractable distributions into a high-dimensional probability distribution.   However, learning PCs on heterogeneous data is challenging and densities of some parametric distributions are not available in closed form, limiting their potential use.   We introduce characteristic circuits (CCs), a family of tractable probabilistic models providing a unified formalization of distributions over heterogeneous data in the spectral domain.  The one-to-one relationship between characteristic functions and probability measures enables us to learn high-dimensional distributions on heterogeneous data domains and facilitates efficient probabilistic inference even when no closed-form density function is available.   We show that the structure and parameters of CCs can be learned efficiently from the data and find that CCs outperform state-of-the-art density estimators for heterogeneous data domains on common benchmark data sets.

----

## [1476] Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds

**Authors**: *Paul Rosa, Slava Borovitskiy, Alexander Terenin, Judith Rousseau*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b7676588c33d344485eeba1b5653ab1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b7676588c33d344485eeba1b5653ab1-Abstract-Conference.html)

**Abstract**:

Gaussian processes are used in many machine learning applications that rely on uncertainty quantification. Recently, computational tools for working with these models in geometric settings, such as when inputs lie on a Riemannian manifold, have been developed. This raises the question: can these intrinsic models be shown theoretically to lead to better performance, compared to simply embedding all relevant quantities into $\mathbb{R}^d$ and using the restriction of an ordinary Euclidean Gaussian process? To study this, we prove optimal contraction rates for intrinsic Matérn Gaussian processes defined on compact Riemannian manifolds. We also prove analogous rates for extrinsic processes using trace and extension theorems between manifold and ambient Sobolev spaces: somewhat surprisingly, the rates obtained turn out to coincide with those of the intrinsic processes, provided that their smoothness parameters are matched appropriately. We illustrate these rates empirically on a number of examples, which, mirroring prior work, show that intrinsic processes can achieve better performance in practice. Therefore, our work shows that finer-grained analyses are needed to distinguish between different levels of data-efficiency of geometric Gaussian processes, particularly in settings which involve small data set sizes and non-asymptotic behavior.

----

## [1477] Causal Context Connects Counterfactual Fairness to Robust Prediction and Group Fairness

**Authors**: *Jacy Reese Anthis, Victor Veitch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b7e1e96243c9edc378f85e7d232e415-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b7e1e96243c9edc378f85e7d232e415-Abstract-Conference.html)

**Abstract**:

Counterfactual fairness requires that a person would have been classified in the same way by an AI or other algorithmic system if they had a different protected class, such as a different race or gender. This is an intuitive standard, as reflected in the U.S. legal system, but its use is limited because counterfactuals cannot be directly observed in real-world data. On the other hand, group fairness metrics (e.g., demographic parity or equalized odds) are less intuitive but more readily observed. In this paper, we use \textit{causal context} to bridge the gaps between counterfactual fairness, robust prediction, and group fairness. First, we motivate counterfactual fairness by showing that there is not necessarily a fundamental trade-off between fairness and accuracy because, under plausible conditions, the counterfactually fair predictor is in fact accuracy-optimal in an unbiased target distribution. Second, we develop a correspondence between the causal graph of the data-generating process and which, if any, group fairness metrics are equivalent to counterfactual fairness. Third, we show that in three common fairness contexts—measurement error, selection on label, and selection on predictors—counterfactual fairness is equivalent to demographic parity, equalized odds, and calibration, respectively. Counterfactual fairness can sometimes be tested by measuring relatively simple group fairness metrics.

----

## [1478] Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance

**Authors**: *Congyue Deng, Jiahui Lei, William B. Shen, Kostas Daniilidis, Leonidas J. Guibas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b8c6f846c3575e1d1ad496abea28826-Abstract-Conference.html)

**Abstract**:

Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.

----

## [1479] Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents

**Authors**: *Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html)

**Abstract**:

In this paper, we study the problem of planning in Minecraft, a popular, democratized yet challenging open-ended environment for developing multi-task embodied agents. We've found two primary challenges of empowering such agents with planning: 1) planning in an open-ended world like Minecraft requires precise and multi-step reasoning due to the long-term nature of the tasks, and 2) as vanilla planners do not consider the achievability of the current agent when ordering parallel sub-goals within a complicated plan, the resulting plan could be inefficient. To this end, we propose ``$\underline{D}$escribe, $\underline{E}$xplain, $\underline{P}$lan and $\underline{S}$elect'' ($\textbf{DEPS}$), an interactive planning approach based on Large Language Models (LLMs). Our approach helps with better error correction from the feedback during the long-haul planning, while also bringing the sense of proximity via goal $\textbf{Selector}$, a learnable module that ranks parallel sub-goals based on the estimated steps of completion and improves the original plan accordingly. Our experiments mark the milestone of the first zero-shot multi-task agent that can robustly accomplish 70+ Minecraft tasks and nearly double the overall performances. Further testing reveals our method's general effectiveness in popularly adopted non-open-ended domains as well (i.e., ALFWorld and tabletop manipulation). The ablation and exploratory studies detail how our design beats the counterparts and provide a promising update on the $\texttt{ObtainDiamond}$ grand challenge with our approach.

----

## [1480] Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage

**Authors**: *Yuheng Jia, Fuchao Yang, Yongqiang Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b97236d90d945be7c58268207a14f4f-Abstract-Conference.html)

**Abstract**:

In partial label learning (PLL), each sample is associated with a group of candidate labels, among which only one label is correct. The key of PLL is to disambiguate the candidate label set to find the ground-truth label. To this end, we first construct a constrained regression model to capture the confidence of the candidate labels, and multiply the label confidence matrix by its transpose to build a second-order similarity matrix, whose elements indicate the pairwise similarity relationships of samples globally. Then we develop a semantic dissimilarity matrix by considering the complement of the intersection of the candidate label set, and further propagate the initial dissimilarity relationships to the whole data set by leveraging the local geometric structure of samples. The similarity and dissimilarity matrices form an adversarial relationship, which is further utilized to shrink the solution space of the label confidence matrix and promote the dissimilarity matrix. We finally extend the proposed model to a kernel version to exploit the non-linear structure of samples and solve the proposed model by the inexact augmented Lagrange multiplier method. By exploiting the adversarial prior, the proposed method can significantly outperformstate-of-the-art PLL algorithms when evaluated on 10 artificial and 7 real-world partial label data sets. We also prove the effectiveness of our method with some theoretical guarantees. The code is publicly available at https://github.com/Yangfc-ML/DPCLS.

----

## [1481] Data Selection for Language Models via Importance Resampling

**Authors**: *Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html)

**Abstract**:

Selecting a suitable pretraining dataset is crucial for both general-domain (e.g., GPT-3) and domain-specific (e.g., Codex) language models (LMs). We formalize this problem as selecting a subset of a large raw unlabeled dataset to match a desired target distribution given unlabeled target samples. Due to the scale and dimensionality of the raw text data, existing methods use simple heuristics or require human experts to manually curate data. Instead, we extend the classic importance resampling approach used in low-dimensions for LM data selection. We propose Data Selection with Importance Resampling (DSIR), an efficient and scalable framework that estimates importance weights in a reduced feature space for tractability and selects data with importance resampling according to these weights. We instantiate the DSIR framework with hashed n-gram features for efficiency, enabling the selection of 100M documents from the full Pile dataset in 4.5 hours. To measure whether hashed n-gram features preserve the aspects of the data that are relevant to the target, we define KL reduction, a data metric that measures the proximity between the selected pretraining data and the target on some feature space. Across 8 data selection methods (including expert selection), KL reduction on hashed n-gram features highly correlates with average downstream accuracy (r=0.82). When selecting data for continued pretraining on a specific domain, DSIR performs comparably to expert curation across 8 target distributions. When pretraining general-domain models (target is Wikipedia and books), DSIR improves over random selection and heuristic filtering baselines by 2--2.5% on the GLUE benchmark.

----

## [1482] Video Dynamics Prior: An Internal Learning Approach for Robust Video Enhancements

**Authors**: *Gaurav Shrivastava, Ser Nam Lim, Abhinav Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ba85c6f1c7656a6a647bc4d63b90bf0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ba85c6f1c7656a6a647bc4d63b90bf0-Abstract-Conference.html)

**Abstract**:

In this paper, we present a novel robust framework for low-level vision tasks, including denoising, object removal, frame interpolation, and super-resolution, that does not require any external training data corpus. Our proposed approach directly learns the weights of neural modules by optimizing over the corrupted test sequence, leveraging the spatio-temporal coherence and internal statistics of videos. Furthermore, we introduce a novel spatial pyramid loss that leverages the property of spatio-temporal patch recurrence in a video across the different scales of the video. This loss enhances robustness to unstructured noise in both the spatial and temporal domains. This further results in our framework being highly robust to degradation in input frames and yields state-of-the-art results on downstream tasks such as denoising, object removal, and frame interpolation. To validate the effectiveness of our approach, we conduct qualitative and quantitative evaluations on standard video datasets such as DAVIS, UCF-101, and VIMEO90K-T.

----

## [1483] Glance and Focus: Memory Prompting for Multi-Event Video Question Answering

**Authors**: *Ziyi Bai, Ruiping Wang, Xilin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html)

**Abstract**:

Video Question Answering (VideoQA) has emerged as a vital tool to evaluate agentsâ€™ ability to understand human daily behaviors. Despite the recent success of large vision language models in many multi-modal tasks, complex situation reasoning over videos involving multiple human-object interaction events still remains challenging. In contrast, humans can easily tackle it by using a series of episode memories as anchors to quickly locate question-related key moments for reasoning. To mimic this effective reasoning strategy, we propose the Glance- Focus model. One simple way is to apply an action detection model to predict a set of actions as key memories. However, these actions within a closed set vocabulary are hard to generalize to various video domains. Instead of that, we train an Encoder-Decoder to generate a set of dynamic event memories at the glancing stage. Apart from using supervised bipartite matching to obtain the event memories, we further design an unsupervised memory generation method to get rid of dependence on event annotations. Next, at the focusing stage, these event memories act as a bridge to establish the correlation between the questions with high-level event concepts and low-level lengthy video content. Given the question, the model first focuses on the generated key event memory, then focuses on the most relevant moment for reasoning through our designed multi-level cross- attention mechanism. We conduct extensive experiments on four Multi-Event VideoQA benchmarks including STAR, EgoTaskQA, AGQA, and NExT-QA. Our proposed model achieves state-of-the-art results, surpassing current large models in various challenging reasoning tasks. The code and models are available at https://github.com/ByZ0e/Glance-Focus.

----

## [1484] Learning To Dive In Branch And Bound

**Authors**: *Max B. Paulus, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6bbda0824bcc20749f21510fd8b28de5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6bbda0824bcc20749f21510fd8b28de5-Abstract-Conference.html)

**Abstract**:

Primal heuristics are important for solving mixed integer linear programs, because they find feasible solutions that facilitate branch and bound search. A prominent group of primal heuristics are diving heuristics. They iteratively modify and resolve linear programs to conduct a depth-first search from any node in the search tree. Existing divers rely on generic decision rules that fail to exploit structural commonality between similar problem instances that often arise in practice. Therefore, we propose L2Dive to learn specific diving heuristics with graph neural networks: We train generative models to predict variable assignments and leverage the duality of linear programs to make diving decisions based on the model's predictions. L2Dive is fully integrated into the open-source solver SCIP. We find that L2Dive outperforms standard divers to find better feasible solutions on a range of combinatorial optimization problems. For real-world applications from server load balancing and neural network verification, L2Dive improves the primal-dual integral by up to 7% (35%) on average over a tuned (default) solver baseline and reduces average solving time by 20% (29%).

----

## [1485] Intriguing Properties of Quantization at Scale

**Authors**: *Arash Ahmadian, Saurabh Dash, Hongyu Chen, Bharat Venkitesh, Stephen Zhen Gou, Phil Blunsom, Ahmet Üstün, Sara Hooker*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html)

**Abstract**:

Emergent properties have been widely adopted as a term to describe behavior not present in smaller models but observed in larger models  (Wei et al., 2022a). Recent work suggests that the trade-off incurred by quantization is also an emergent property, with sharp drops in performance in models over 6B parameters. In this work, we ask are quantization cliffs in performance solely a factor of scale? Against a backdrop of increased research focus on why certain emergent properties surface at scale, this work provides a useful counter-example. We posit that it is possible to optimize for a quantization friendly training recipe that suppresses large activation magnitude outliers. Here, we find that outlier dimensions are not an inherent product of scale, but rather sensitive to the optimization conditions present during pre-training. This both opens up directions for more efficient quantization, and poses the question of whether other emergent properties are inherent or can be altered and conditioned by optimization and architecture design choices. We successfully quantize models ranging in size from 410M to 52B with minimal degradation in performance.

----

## [1486] Self-supervised Graph Neural Networks via Low-Rank Decomposition

**Authors**: *Liang Yang, Runjie Shi, Qiuliang Zhang, Bingxin Niu, Zhen Wang, Xiaochun Cao, Chuan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c33e4ea4ddfb05a78541022ab5a1fb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c33e4ea4ddfb05a78541022ab5a1fb9-Abstract-Conference.html)

**Abstract**:

Self-supervised learning is introduced to train graph neural networks (GNNs) by employing propagation-based GNNs designed for semi-supervised learning tasks. Unfortunately, this common choice tends to cause two serious issues. Firstly, global parameters cause the model lack the ability to capture the local property. Secondly, it is difficult to handle networks beyond homophily without label information.This paper tends to break through the common choice of employing propagation-based GNNs, which aggregate representations of nodes belonging to different classes and tend to lose discriminative information. If the propagation in each ego-network is just between the nodes from the same class, the obtained representation matrix should follow the low-rank characteristic. To meet this requirement, this paper proposes the Low-Rank Decomposition-based GNNs (LRD-GNN-Matrix) by employing Low-Rank Decomposition to the attribute matrix. Furthermore, to incorporate long-distance information, Low-Rank Tensor Decomposition-based GNN (LRD-GNN-Tensor) is proposed by constructing the node attribute tensor from selected similar ego-networks and performing Low-Rank Tensor Decomposition. The employed tensor nuclear norm facilitates the capture of the long-distance relationship between original and selected similar ego-networks. Extensive experiments demonstrate the superior performance and the robustness  of  LRD-GNNs.

----

## [1487] Cookie Consent Has Disparate Impact on Estimation Accuracy

**Authors**: *Erik Miehling, Rahul Nair, Elizabeth Daly, Karthikeyan Natesan Ramamurthy, Robert Redmond*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html)

**Abstract**:

Cookies are designed to enable more accurate identification and tracking of user behavior, in turn allowing for more personalized ads and better performing ad campaigns. Given the additional information that is recorded, questions related to privacy and fairness naturally arise. How does a user's consent decision influence how much the system can learn about their demographic and tastes? Is the impact of a user's consent decision on the recommender system's ability to learn about their latent attributes uniform across demographics? We investigate these questions in the context of an engagement-driven recommender system using simulation. We empirically demonstrate that when consent rates exhibit demographic-dependence, user consent has a disparate impact on the recommender agent's ability to estimate users' latent attributes. In particular, we find that when consent rates are demographic-dependent, a user disagreeing to share their cookie may counter-intuitively cause the recommender agent to know more about the user than if the user agreed to share their cookie. Furthermore, the gap in base consent rates across demographics serves as an amplifier: users from the lower consent rate demographic who agree to cookie sharing generally experience higher estimation errors than the same users from the higher consent rate demographic, and conversely for users who choose to disagree to cookie sharing, with these differences increasing in consent rate gap. We discuss the need for new notions of fairness that encourage consistency between a user's privacy decisions and the system's ability to estimate their latent attributes.

----

## [1488] NPCL: Neural Processes for Uncertainty-Aware Continual Learning

**Authors**: *Saurav Jha, Dong Gong, He Zhao, Lina Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c4a1a3cbe70ef36d7d6332166bba77d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c4a1a3cbe70ef36d7d6332166bba77d-Abstract-Conference.html)

**Abstract**:

Continual learning (CL) aims to train deep neural networks efficiently on streaming data while limiting the forgetting caused by new tasks.  However, learning transferable knowledge with less interference between tasks is difficult, and real-world deployment of CL models is limited by their inability to measure predictive uncertainties. To address these issues, we propose handling CL tasks with neural processes (NPs), a class of meta-learners that encode different tasks into probabilistic distributions over functions all while providing reliable uncertainty estimates. Specifically, we propose an NP-based CL approach (NPCL) with task-specific modules arranged in a hierarchical latent variable model. We tailor regularizers on the learned latent distributions to alleviate forgetting. The uncertainty estimation capabilities of the NPCL can also be used to handle the task head/module inference challenge in CL. Our experiments show that the NPCL  outperforms previous CL approaches. We validate the effectiveness of uncertainty estimation in the NPCL for identifying novel data and evaluating instance-level model confidence. Code is available at https://github.com/srvCodes/NPCL.

----

## [1489] From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces

**Authors**: *Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, Kristina Toutanova*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c52a8a4fadc9129c6e1d1745f2dfd0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c52a8a4fadc9129c6e1d1745f2dfd0f-Abstract-Conference.html)

**Abstract**:

Much of the previous work towards digital agents for graphical user interfaces (GUIs) has relied on text-based representations (derived from HTML or other structured data sources), which are not always readily available. These input representations have been often coupled with custom, task-specific action spaces.  This paper focuses on creating agents that interact with the digital world using the same conceptual interface that humans commonly use â€” via pixel-based screenshots and a generic action space corresponding to keyboard and mouse actions. Building upon recent progress in pixel-based pretraining, we show, for the first time, that it is possible for such agents to outperform human crowdworkers on the MiniWob++ benchmark of GUI-based instruction following tasks.

----

## [1490] Robust Mean Estimation Without Moments for Symmetric Distributions

**Authors**: *Gleb Novikov, David Steurer, Stefan Tiegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c59ace4fc4872a14df13d91762ad4f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c59ace4fc4872a14df13d91762ad4f0-Abstract-Conference.html)

**Abstract**:

We study the problem of robustly estimating the mean or location parameter without moment assumptions.Known computationally efficient algorithms rely on strong distributional assumptions, such as sub-Gaussianity, or (certifiably) bounded moments.Moreover, the guarantees that they achieve in the heavy-tailed setting are weaker than those for sub-Gaussian distributions with known covariance.In this work, we show that such a tradeoff, between error guarantees and heavy-tails, is not necessary for symmetric distributions.We show that for a large class of symmetric distributions, the same error as in the Gaussian setting can be achieved efficiently.The distributions we study include products of arbitrary symmetric one-dimensional distributions, such as product Cauchy distributions, as well as elliptical distributions, a vast generalization of the Gaussian distribution.For product distributions and elliptical distributions with known scatter (covariance) matrix, we show that given an $\varepsilon$-corrupted sample, we can with probability at least $1-\delta$ estimate its location up to error $O(\varepsilon \sqrt{\log(1/\varepsilon)})$ using $\tfrac{d\log(d) + \log(1/\delta)}{\varepsilon^2 \log(1/\varepsilon)}$ samples.This result matches the best-known guarantees for the Gaussian distribution and known SQ lower bounds (up to the $\log(d)$ factor).For elliptical distributions with unknown scatter (covariance) matrix, we propose a sequence of efficient algorithms that approaches this optimal error.Specifically, for every $k \in \mathbb{N}$, we design an estimator using time and samples $\tilde{O}({d^k})$ achieving error $O(\varepsilon^{1-\frac{1}{2k}})$.This matches the error and running time guarantees when assuming certifiably bounded moments of order up to $k$.For unknown covariance, such error bounds of $o(\sqrt{\varepsilon})$ are not even known for (general) sub-Gaussian distributions.Our algorithms are based on a generalization of the well-known filtering technique [DK22].More specifically, we show how this machinery can be combined with Huber-loss-based techniques to work with projections of the noise that behave more nicely than the initial noise.Moreover, we show how sum-of-squares proofs can be used to obtain algorithmic guarantees even for distributions without a first moment.We believe that this approach may find other applications in future works.

----

## [1491] Fast and Simple Spectral Clustering in Theory and Practice

**Authors**: *Peter Macgregor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c5b82193c5d8e6aa5806239676ddc97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c5b82193c5d8e6aa5806239676ddc97-Abstract-Conference.html)

**Abstract**:

Spectral clustering is a popular and effective algorithm designed to find $k$ clusters in a graph $G$.In the classical spectral clustering algorithm, the vertices of $G$ are embedded into $\mathbb{R}^k$ using $k$ eigenvectors of the graph Laplacian matrix.However, computing this embedding is computationally expensive and dominates the running time of the algorithm.In this paper, we present a simple spectral clustering algorithm based on a vertex embedding with $O(\log(k))$ vectors computed by the power method.The vertex embedding is computed in nearly-linear time with respect to the size of the graph, andthe algorithm provably recovers the ground truth clusters under natural assumptions on the input graph.We evaluate the new algorithm on several synthetic and real-world datasets, finding that it is significantly faster than alternative clustering algorithms,while producing results with approximately the same clustering accuracy.

----

## [1492] CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation

**Authors**: *Xiuzhe Wu, Peng Dai, Weipeng Deng, Handi Chen, Yang Wu, Yan-Pei Cao, Ying Shan, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html)

**Abstract**:

Existing methods for adapting Neural Radiance Fields (NeRFs) to scene changes require extensive data capture and model retraining, which is both time-consuming and labor-intensive. In this paper, we tackle the challenge of efficiently adapting NeRFs to real-world scene changes over time using a few new images while retaining the memory of unaltered areas, focusing on the continual learning aspect of NeRFs. To this end, we propose CL-NeRF, which consists of two key components: a lightweight expert adaptor for adapting to new changes and evolving scene representations and a conflict-aware knowledge distillation learning objective for memorizing unchanged parts. We also present a new benchmark for evaluating Continual Learning of NeRFs with comprehensive metrics. Our extensive experiments demonstrate that CL-NeRF can synthesize high-quality novel views of both changed and unchanged regions with  high training efficiency, surpassing existing methods in terms of reducing forgetting and adapting to changes. Code and benchmark will be made available.

----

## [1493] Generalised f-Mean Aggregation for Graph Neural Networks

**Authors**: *Ryan Kortvelesy, Steven D. Morad, Amanda Prorok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c78ae0c1140902bf3a430b1725bcc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c78ae0c1140902bf3a430b1725bcc4e-Abstract-Conference.html)

**Abstract**:

Graph Neural Network (GNN) architectures are defined by their implementations of update and aggregation modules. While many works focus on new ways to parametrise the update modules, the aggregation modules receive comparatively little attention. Because it is difficult to parametrise aggregation functions, currently most methods select a ``standard aggregator'' such as mean, sum, or max. While this selection is often made without any reasoning, it has been shown that the choice in aggregator has a significant impact on performance, and the best choice in aggregator is problem-dependent. Since aggregation is a lossy operation, it is crucial to select the most appropriate aggregator in order to minimise information loss. In this paper, we present GenAgg, a generalised aggregation operator, which parametrises a function space that includes all standard aggregators. In our experiments, we show that GenAgg is able to represent the standard aggregators with much higher accuracy than baseline methods. We also show that using GenAgg as a drop-in replacement for an existing aggregator in a GNN often leads to a significant boost in performance across various tasks.

----

## [1494] Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization

**Authors**: *Mahyar Fazlyab, Taha Entesari, Aniket Roy, Rama Chellappa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c7ca1889f01a9b767c631686fb5fd24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c7ca1889f01a9b767c631686fb5fd24-Abstract-Conference.html)

**Abstract**:

To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. In this paper, we propose a differentiable regularizer that is a lower bound on the distance of the data points to the classification boundary. The proposed regularizer requires knowledge of the model's Lipschitz constant along certain directions. To this end, we develop a scalable method for calculating guaranteed differentiable upper bounds on the Lipschitz constant of neural networks accurately and efficiently.  The relative accuracy of the bounds prevents excessive regularization and allows for more direct manipulation of the decision boundary. Furthermore, our Lipschitz bounding algorithm exploits the monotonicity and Lipschitz continuity of the activation layers, and the resulting bounds can be used to design new layers with controllable bounds on their Lipschitz constant. Experiments on the MNIST, CIFAR-10, and Tiny-ImageNet data sets verify that our proposed algorithm obtains competitively improved results compared to the state-of-the-art.

----

## [1495] Unpaired Multi-Domain Causal Representation Learning

**Authors**: *Nils Sturma, Chandler Squires, Mathias Drton, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html)

**Abstract**:

The goal of causal representation learning is to find a representation of data that consists of causally related latent variables. We consider a setup where one has access to data from multiple domains that potentially share a causal representation. Crucially, observations in different domains are assumed to be unpaired, that is, we only observe the marginal distribution in each domain but not their joint distribution. In this paper, we give sufficient conditions for identifiability of the joint distribution and the shared causal graph in a linear setup. Identifiability holds if we can uniquely recover the joint distribution and the shared causal representation from the marginal distributions in each domain. We transform our results into a practical method to recover the shared latent causal graph.

----

## [1496] Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection

**Authors**: *Haibao Yu, Yingjuan Tang, Enze Xie, Jilei Mao, Ping Luo, Zaiqing Nie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html)

**Abstract**:

Cooperatively utilizing both ego-vehicle and infrastructure sensor data can significantly enhance autonomous driving perception abilities. However, the uncertain temporal asynchrony and limited communication conditions that are present in traffic environments can lead to fusion misalignment and constrain the exploitation of infrastructure data. To address these issues in vehicle-infrastructure cooperative 3D (VIC3D) object detection, we propose the Feature Flow Net (FFNet), a novel cooperative detection framework. FFNet is a flow-based feature fusion framework that uses a feature flow prediction module to predict future features and compensate for asynchrony. Instead of transmitting feature maps extracted from still-images, FFNet transmits feature flow, leveraging the temporal coherence of sequential infrastructure frames. Furthermore, we introduce a self-supervised training approach that enables FFNet to generate feature flow with feature prediction ability from raw infrastructure sequences. Experimental results demonstrate that our proposed method outperforms existing cooperative detection methods while only requiring about 1/100 of the transmission cost of raw data and covers all latency in one model on the DAIR-V2X dataset. The code  is available https://github.com/haibao-yu/FFNet-VIC3D.

----

## [1497] Subspace Identification for Multi-Source Domain Adaptation

**Authors**: *Zijian Li, Ruichu Cai, Guangyi Chen, Boyang Sun, Zhifeng Hao, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html)

**Abstract**:

Multi-source domain adaptation (MSDA) methods aim to transfer knowledge from multiple labeled source domains to an unlabeled target domain. Although current methods achieve target joint distribution identifiability by enforcing minimal changes across domains, they often necessitate stringent conditions, such as an adequate number of domains, monotonic transformation of latent variables, and invariant label distributions. These requirements are challenging to satisfy in real-world applications. To mitigate the need for these strict assumptions, we propose a subspace identification theory that guarantees the disentanglement of domain-invariant and domain-specific variables under less restrictive constraints regarding domain numbers and transformation properties and thereby facilitating domain adaptation by minimizing the impact of domain shifts on invariant variables. Based on this theory, we develop a Subspace Identification Guarantee (SIG) model that leverages variational inference. Furthermore, the SIG model incorporates class-aware conditional alignment to accommodate target shifts where label distributions change with the domain. Experimental results demonstrate that our SIG model outperforms existing MSDA techniques on various benchmark datasets, highlighting its effectiveness in real-world applications.

----

## [1498] Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates

**Authors**: *Sarath Sreedharan, Michael Katz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cbd0a1251f41b41aa68e728bcc1ee40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cbd0a1251f41b41aa68e728bcc1ee40-Abstract-Conference.html)

**Abstract**:

There has been an increasing interest in using symbolic models along with reinforcement learning (RL) problems, where these coarser abstract models are used as a way to provide RL agents with higher level guidance. However, most of these works are inherently limited by their assumption of having an access to a symbolic approximation of the underlying problem. To address this issue, we introduce a new method for learning optimistic symbolic approximations of the underlying world model. We will see how these representations, coupled with fast diverse planners developed by the automated planning community, provide us with a new paradigm for optimistic exploration in sparse reward settings. We investigate the possibility of speeding up the learning process by generalizing learned model dynamics across similar actions with minimal human input. Finally, we evaluate the method, by testing it on multiple benchmark domains and compare it with other RL strategies.

----

## [1499] Feature learning via mean-field Langevin dynamics: classifying sparse parities and beyond

**Authors**: *Taiji Suzuki, Denny Wu, Kazusato Oko, Atsushi Nitanda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cc321baf0a8611b1d1bdbd18822667b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cc321baf0a8611b1d1bdbd18822667b-Abstract-Conference.html)

**Abstract**:

Neural network in the mean-field regime is known to be capable of \textit{feature learning}, unlike the kernel (NTK) counterpart. Recent works have shown that mean-field neural networks can be globally optimized by a noisy gradient descent update termed the \textit{mean-field Langevin dynamics} (MFLD). However, all existing guarantees for MFLD only considered the \textit{optimization} efficiency, and it is unclear if this algorithm leads to improved \textit{generalization} performance and sample complexity due to the presence of feature learning. To fill this gap, in this work we study the statistical and computational complexity of MFLD in learning a class of binary classification problems. Unlike existing margin bounds for neural networks, we avoid the typical norm control by utilizing the perspective that MFLD optimizes the \textit{distribution} of parameters rather than the parameter itself; this leads to an improved analysis of the sample complexity and convergence rate. We apply our general framework to the learning of $k$-sparse parity functions, where we prove that unlike kernel methods, two-layer neural networks optimized by MFLD achieves a sample complexity where the degree $k$ is ``decoupled'' from the exponent in the dimension dependence.

----

## [1500] Improving Graph Matching with Positional Reconstruction Encoder-Decoder Network

**Authors**: *Yixiao Zhou, Ruiqi Jia, Hongxiang Lin, Hefeng Quan, Yumeng Zhao, Xiaoqing Lyu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html)

**Abstract**:

Deriving from image matching and understanding, semantic keypoint matching aims at establishing correspondence between keypoint sets in images. As graphs are powerful tools to represent points and their complex relationships, graph matching provides an effective way to find desired semantic keypoint correspondences. Recent deep graph matching methods have shown excellent performance, but there is still a lack of exploration and utilization of spatial information of keypoints as nodes in graphs. More specifically, existing methods are insufficient to capture the relative spatial relations through current graph construction approaches from the locations of semantic keypoints. To address these issues, we introduce a positional reconstruction encoder-decoder (PR-EnDec) to model intrinsic graph spatial structure, and present an end-to-end graph matching network PREGM based on PR-EnDec. Our PR-EnDec consists of a positional encoder that learns effective node spatial embedding with the affine transformation invariance, and a spatial relation decoder that further utilizes the high-order spatial information by reconstructing the locational structure of graphs contained in the node coordinates. Extensive experimental results on three public keypoint matching datasets demonstrate the effectiveness of our proposed PREGM.

----

## [1501] A Causal Framework for Decomposing Spurious Variations

**Authors**: *Drago Plecko, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cda6dae05ae5e42ea78be85d5a26f77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cda6dae05ae5e42ea78be85d5a26f77-Abstract-Conference.html)

**Abstract**:

One of the fundamental challenges found throughout the data sciences is to explain why things happen in specific ways, or through which mechanisms a certain variable $X$ exerts influences over another variable $Y$. In statistics and machine learning, significant efforts have been put into developing machinery to estimate correlations across variables efficiently. In causal inference, a large body of literature is concerned with the decomposition of causal effects under the rubric of mediation analysis. However, many variations are spurious in nature, including different phenomena throughout the applied sciences. Despite the statistical power to estimate correlations and the identification power to decompose causal effects, there is still little understanding of the properties of spurious associations and how they can be decomposed in terms of the underlying causal mechanisms. In this manuscript, we develop formal tools for decomposing spurious variations in both Markovian and Semi-Markovian models. We prove the first results that allow a non-parametric decomposition of spurious effects and provide sufficient conditions for the identification of such decompositions. The described approach has several applications, ranging from explainable and fair AI to questions in epidemiology and medicine, and we empirically demonstrate its use.

----

## [1502] Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification

**Authors**: *Tianjun Ke, Haoqun Cao, Zenan Ling, Feng Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html)

**Abstract**:

Meta-learning has demonstrated promising results in few-shot classification (FSC) by learning to solve new problems using prior knowledge. Bayesian methods are effective at characterizing uncertainty in FSC, which is crucial in high-risk fields. In this context, the logistic-softmax likelihood is often employed as an alternative to the softmax likelihood in multi-class Gaussian process classification due to its conditional conjugacy property. However, the theoretical property of logistic-softmax is not clear and previous research indicated that the inherent uncertainty of logistic-softmax leads to suboptimal performance. To mitigate these issues, we revisit and redesign the logistic-softmax likelihood, which enables control of the \textit{a priori} confidence level through a temperature parameter. Furthermore, we theoretically and empirically show that softmax can be viewed as a special case of logistic-softmax and logistic-softmax induces a larger family of data distribution than softmax. Utilizing modified logistic-softmax, we integrate the data augmentation technique into the deep kernel based Gaussian process meta-learning framework, and derive an analytical mean-field approximation for task-specific updates. Our approach yields well-calibrated uncertainty estimates and achieves comparable or superior results on standard benchmark datasets. Code is publicly available at \url{https://github.com/keanson/revisit-logistic-softmax}.

----

## [1503] Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration

**Authors**: *Haitao Lin, Yufei Huang, Odin Zhang, Yunfan Liu, Lirong Wu, Siyuan Li, Zhiyuan Chen, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html)

**Abstract**:

In recent years, AI-assisted drug design methods have been proposed to generate molecules given the pockets' structures of target proteins. Most of them are  {\em atom-level-based} methods, which consider atoms as basic components and generate atom positions and types. In this way, however, it is hard to generate realistic fragments with complicated structures. To solve this, we propose \textsc{D3FG}, a {\em functional-group-based} diffusion model for pocket-specific molecule generation and elaboration. \textsc{D3FG} decomposes molecules into two categories of components: functional groups defined as rigid bodies and linkers as mass points. And the two kinds of components can together form complicated fragments that enhance ligand-protein interactions. To be specific, in the diffusion process, \textsc{D3FG} diffuses the data distribution of the positions, orientations, and types of the components into a prior distribution; In the generative process, the noise is gradually removed from the three variables by denoisers parameterized with designed equivariant graph neural networks.  In the experiments, our method can generate molecules with more realistic 3D structures, competitive affinities toward the protein targets, and better drug properties. Besides, \textsc{D3FG} as a solution to a new task of molecule elaboration, could generate molecules with high affinities based on existing ligands and the hotspots of target proteins.

----

## [1504] Approximately Equivariant Graph Networks

**Authors**: *Ningyuan Huang, Ron Levie, Soledad Villar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cde6435e111671b04f4574006cf3c47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cde6435e111671b04f4574006cf3c47-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are commonly described as being permutation equivariant with respect to node relabeling in the graph. This symmetry of GNNs is often compared to the translation equivariance of Euclidean convolution neural networks (CNNs). However, these two symmetries are fundamentally different: The translation equivariance of CNNs corresponds to symmetries of the fixed domain acting on the image signals (sometimes known as active symmetries), whereas in GNNs any permutation acts on both the graph signals and the graph domain (sometimes described as passive symmetries). In this work, we focus on the active symmetries of GNNs, by considering a learning setting where signals are supported on a fixed graph. In this case, the natural symmetries of GNNs are the automorphisms of the graph. Since real-world graphs tend to be asymmetric, we relax the notion of symmetries by formalizing approximate symmetries via graph coarsening. We present a bias-variance formula that quantifies the tradeoff between the loss in expressivity and the gain in the regularity of the learned estimator, depending on the chosen symmetry group. To illustrate our approach, we conduct extensive experiments on image inpainting, traffic flow prediction, and human pose estimation with different choices of symmetries. We show theoretically and empirically that the best generalization performance can be achieved by choosing a suitably larger group than the graph automorphism, but smaller than the permutation group.

----

## [1505] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Authors**: *Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, Beidi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs), despite their recent impressive accomplishments, are notably cost-prohibitive to deploy, particularly for applications involving long-content generation, such as dialogue systems and story writing. Often, a large amount of transient state information, referred to as the $\mathsf{KV}$ $\mathsf{cache}$, is stored in GPU memory in addition to model parameters, scaling linearly with the sequence length and batch size. In this paper, we introduce a novel approach for implementing the  $\mathsf{KV}$ $\mathsf{cache}$ which significantly reduces its memory footprint.  Our approach is based on the noteworthy observation that a small portion of tokens contributes most of the value when computing attention scores.  We call these tokens Heavy Hitters ($\mathsf{H_2}$). Through a comprehensive investigation, we find that ($i$) the emergence of $\mathsf{H_2}$ is natural and strongly correlates with the frequent co-occurrence of tokens in the text, and ($ii$) removing them results in significant performance degradation. Based on these insights, we propose Heavy Hitter Oracle ($\mathsf{H_2O}$), a $\mathsf{KV}$ $\mathsf{cache}$ eviction policy that dynamically retains a balance of recent and $\mathsf{H_2}$ tokens. We formulate the $\mathsf{KV}$ $\mathsf{cache}$ eviction as a dynamic submodular problem and prove (under mild assumptions) a theoretical guarantee for our novel eviction algorithm which could help guide future work. We validate the accuracy of our algorithm with OPT, LLaMA, and GPT-NeoX across a wide range of tasks. Our implementation of $\mathsf{H_2O}$ with 20\% heavy hitters improves the throughput over three leading inference systems DeepSpeed Zero-Inference, Hugging Face Accelerate, and FlexGen by up to $29\times$, $29\times$, and $3\times$ on OPT-6.7B and OPT-30B. With the same batch size, $\mathsf{H_2O}$ can reduce the latency by up to $1.9\times$.

----

## [1506] Uncovering motifs of concurrent signaling across multiple neuronal populations

**Authors**: *Evren Gokcen, Anna Jasper, Alison Xu, Adam Kohn, Christian K. Machens, Byron M. Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cf7a37e761f55b642cf0939b4c64bb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cf7a37e761f55b642cf0939b4c64bb8-Abstract-Conference.html)

**Abstract**:

Modern recording techniques now allow us to record from distinct neuronal populations in different brain networks. However, especially as we consider multiple (more than two) populations, new conceptual and statistical frameworks are needed to characterize the multi-dimensional, concurrent flow of signals among these populations. Here, we develop a dimensionality reduction framework that determines (1) the subset of populations described by each latent dimension, (2) the direction of signal flow among those populations, and (3) how those signals evolve over time within and across experimental trials. We illustrate these features in simulation, and further validate the method by applying it to previously studied recordings from neuronal populations in macaque visual areas V1 and V2. Then we study interactions across select laminar compartments of areas V1, V2, and V3d, recorded simultaneously with multiple Neuropixels probes. Our approach uncovered signatures of selective communication across these three areas that related to their retinotopic alignment. This work advances the study of concurrent signaling across multiple neuronal populations.

----

## [1507] NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos

**Authors**: *Jinxi Li, Ziyang Song, Bo Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0942e288ce41db8d4ebd041e7d1100-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0942e288ce41db8d4ebd041e7d1100-Abstract-Conference.html)

**Abstract**:

In this paper, we aim to model 3D scene dynamics from multi-view videos. Unlike the majority of existing works which usually focus on the common task of novel view synthesis within the training time period, we propose to simultaneously learn the geometry, appearance, and physical velocity of 3D scenes only from video frames, such that multiple desirable applications can be supported, including future frame extrapolation, unsupervised 3D semantic scene decomposition, and dynamic motion transfer. Our method consists of three major components, 1) the keyframe dynamic radiance field, 2) the interframe velocity field, and 3) a joint keyframe and interframe optimization module which is the core of our framework to effectively  train both networks. To validate our method, we further introduce two dynamic 3D datasets: 1) Dynamic Object dataset, and 2) Dynamic Indoor Scene dataset. We conduct extensive experiments on multiple datasets, demonstrating the superior performance of our method over all baselines, particularly in the critical tasks of future frame extrapolation and unsupervised 3D semantic scene decomposition.

----

## [1508] Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parameterized Models

**Authors**: *Leonardo Galli, Holger Rauhut, Mark Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html)

**Abstract**:

Recent works have shown that line search methods can speed up Stochastic Gradient Descent (SGD) and Adam in modern over-parameterized settings. However, existing line searches may take steps that are smaller than necessary since they require a monotone decrease of the (mini-)batch objective function. We explore nonmonotone line search methods to relax this condition and possibly accept larger step sizes. Despite the lack of a monotonic decrease, we prove the same fast rates of convergence as in the monotone case. Our experiments show that nonmonotone methods improve the speed of convergence and generalization properties of SGD/Adam even beyond the previous monotone line searches. We propose a POlyak NOnmonotone Stochastic (PoNoS) method, obtained by combining a nonmonotone line search with a Polyak initial step size. Furthermore, we develop a new resetting technique that in the majority of the iterations reduces the amount of backtracks to zero while still maintaining a large initial step size. To the best of our knowledge, a first runtime comparison shows that the epoch-wise advantage of line-search-based methods gets reflected in the overall computational time.

----

## [1509] OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

**Authors**: *Yiheng Zhu, Yang Zhan, Xuankun Huang, Yuwei Chen, Yujie Chen, Jiangwen Wei, Wei Feng, Yinzhi Zhou, Haoyuan Hu, Jieping Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0cfc5db3feeabf6762129ba91bd3a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0cfc5db3feeabf6762129ba91bd3a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The dramatic growth of global e-commerce has led to a surge in demand for efficient and cost-effective order fulfillment which can increase customers' service levels and sellers' competitiveness. However, managing order fulfillment is challenging due to a series of interdependent online sequential decision-making problems. To clear this hurdle, rather than solving the problems separately as attempted in some recent researches, this paper proposes a method based on multi-agent reinforcement learning to integratively solve the series of interconnected problems, encompassing order handling, packing and pickup, storage, order consolidation, and last-mile delivery. In particular, we model the integrated problem as a Markov game, wherein a team of agents learns a joint policy via interacting with a simulated environment. Since no simulated environment supporting the complete order fulfillment problem exists, we devise Order Fulfillment COoperative mUlti-agent Reinforcement learning Scalable Environment (OFCOURSE) in the OpenAI Gym style, which allows reproduction and re-utilization to build customized applications. By constructing the fulfillment system in OFCOURSE, we optimize a joint policy that solves the integrated problem, facilitating sequential order-wise operations across all fulfillment units and minimizing the total cost of fulfilling all orders within the promised time. With OFCOURSE, we also demonstrate that the joint policy learned by multi-agent reinforcement learning outperforms the combination of locally optimal policies. The source code of OFCOURSE is available at: https://github.com/GitYiheng/ofcourse.

----

## [1510] On Private and Robust Bandits

**Authors**: *Yulian Wu, Xingyu Zhou, Youming Tao, Di Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)

**Abstract**:

We study private and robust multi-armed bandits (MABs), where the agent receives Huber's contaminated heavy-tailed rewards and meanwhile needs to ensure differential privacy. We consider both the finite $k$-th raw moment and the finite $k$-th central moment settings for heavy-tailed rewards distributions with $k\ge 2$. We first present its minimax lower bound, characterizing the information-theoretic limit of regret with respect to privacy budget, contamination level, and heavy-tailedness. Then, we propose a meta-algorithm that builds on a private and robust mean estimation sub-routine \texttt{PRM} that essentially relies on reward truncation and the Laplace mechanism.  For the above two different heavy-tailed settings, we give corresponding schemes of \texttt{PRM}, which enable us to achieve nearly-optimal regrets.  Moreover, our two proposed truncation-based or histogram-based \texttt{PRM} schemes achieve the optimal trade-off between estimation accuracy, privacy and robustness. Finally, we support our theoretical results and show the effectiveness of our algorithms with experimental studies.

----

## [1511] RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization

**Authors**: *Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d3040941a2d57ead4043556a70dd728-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d3040941a2d57ead4043556a70dd728-Abstract-Conference.html)

**Abstract**:

Multi-agent systems are characterized by environmental uncertainty, varying policies of agents, and partial observability, which result in significant risks. In the context of Multi-Agent Reinforcement Learning (MARL), learning coordinated and decentralized policies that are sensitive to risk is challenging. To formulate the coordination requirements in risk-sensitive MARL, we introduce the Risk-sensitive Individual-Global-Max (RIGM) principle as a generalization of the Individual-Global-Max (IGM) and Distributional IGM (DIGM) principles. This principle requires that the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy. Current MARL value factorization methods do not satisfy the RIGM principle for common risk metrics such as the Value at Risk (VaR) metric or distorted risk measurements. Therefore, we propose RiskQ to address this limitation, which models the joint return distribution by modeling quantiles of it as weighted quantile mixtures of per-agent return distribution utilities. RiskQ satisfies the RIGM principle for the VaR and distorted risk metrics. We show that RiskQ can obtain promising performance through extensive experiments. The source code of RiskQ is available in https://github.com/xmu-rl-3dv/RiskQ.

----

## [1512] Learning Exponential Families from Truncated Samples

**Authors**: *Jane Lee, Andre Wibisono, Emmanouil Zampetakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d5f304fb4ed0243851e41699dca4287-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d5f304fb4ed0243851e41699dca4287-Abstract-Conference.html)

**Abstract**:

Missing data problems have many manifestations across many scientific fields. A fundamental type of missing data problem arises when samples are \textit{truncated}, i.e., samples that lie in a subset of the support are not observed. Statistical estimation from truncated samples is a classical problem in statistics which dates back to Galton, Pearson, and Fisher. A recent line of work provides the first efficient estimation algorithms for the parameters of a Gaussian distribution and for linear regression with Gaussian noise.In this paper we generalize these results to log-concave exponential families. We provide an estimation algorithm that shows that \textit{extrapolation} is possible for a much larger class of distributions while it maintains a polynomial sample and time complexity on average. Our algorithm is based on Projected Stochastic Gradient Descent and is not only applicable in a more general setting but is also simpler and more efficient than recent algorithms. Our work also has interesting implications for learning general log-concave distributions and sampling given only access to truncated data.

----

## [1513] XAGen: 3D Expressive Human Avatars Generation

**Authors**: *Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Jiashi Feng, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d6f9908ea35313dd7566f5ce8c6e815-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d6f9908ea35313dd7566f5ce8c6e815-Abstract-Conference.html)

**Abstract**:

Recent advances in 3D-aware GAN models have enabled the generation of realistic and controllable human body images. However, existing methods focus on the control of major body joints, neglecting the manipulation of expressive attributes, such as facial expressions, jaw poses, hand poses, and so on. In this work, we present XAGen, the first 3D generative model for human avatars capable of expressive control over body, face, and hands. To enhance the fidelity of small-scale regions like face and hands, we devise a multi-scale and multi-part 3D representation that models fine details. Based on this representation, we propose a multi-part rendering technique that disentangles the synthesis of body, face, and hands to ease model training and enhance geometric quality. Furthermore, we design multi-part discriminators that evaluate the quality of the generated avatars with respect to their appearance and fine-grained control capabilities. Experiments show that XAGen surpasses state-of-the-art methods in terms of realism, diversity, and expressive control abilities. Code and data will be made available at https://showlab.github.io/xagen.

----

## [1514] HIQL: Offline Goal-Conditioned RL with Latent States as Actions

**Authors**: *Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html)

**Abstract**:

Unsupervised pre-training has recently become the bedrock for computer vision and natural language processing. In reinforcement learning (RL), goal-conditioned RL can potentially provide an analogous self-supervised approach for making use of large quantities of unlabeled (reward-free) data. However, building effective algorithms for goal-conditioned RL that can learn directly from diverse offline data is challenging, because it is hard to accurately estimate the exact value function for faraway goals. Nonetheless, goal-reaching problems exhibit structure, such that reaching distant goals entails first passing through closer subgoals. This structure can be very useful, as assessing the quality of actions for nearby goals is typically easier than for more distant goals. Based on this idea, we propose a hierarchical algorithm for goal-conditioned RL from offline data. Using one action-free value function, we learn two policies that allow us to exploit this structure: a high-level policy that treats states as actions and predicts (a latent representation of) a subgoal and a low-level policy that predicts the action for reaching this subgoal. Through analysis and didactic examples, we show how this hierarchical decomposition makes our method robust to noise in the estimated value function. We then apply our method to offline goal-reaching benchmarks, showing that our method can solve long-horizon tasks that stymie prior methods, can scale to high-dimensional image observations, and can readily make use of action-free data. Our code is available at https://seohong.me/projects/hiql/

----

## [1515] Visual Instruction Tuning

**Authors**: *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)

**Abstract**:

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks. Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.

----

## [1516] A Fast and Accurate Estimator for Large Scale Linear Model via Data Averaging

**Authors**: *Rui Wang, Yanyan Ouyang, Panpan Yu, Wangli Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6de668dab370194fa304a08be5aacd85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6de668dab370194fa304a08be5aacd85-Abstract-Conference.html)

**Abstract**:

This work is concerned with the estimation problem of linear model when thesample size is extremely large and the data dimension can vary with the samplesize. In this setting, the least square estimator based on the full data is not feasiblewith limited computational resources. Many existing methods for this problem arebased on the sketching technique which uses the sketched data to perform leastsquare estimation. We derive fine-grained lower bounds of the conditional meansquared error for sketching methods. For sampling methods, our lower boundprovides an attainable optimal convergence rate. Our result implies that when thedimension is large, there is hardly a sampling method can have a faster convergencerate than the uniform sampling method. To achieve a better statistical performance,we propose a new sketching method based on data averaging. The proposedmethod reduces the original data to a few averaged observations. These averagedobservations still satisfy the linear model and are used to estimate the regressioncoefficients. The asymptotic behavior of the proposed estimation procedure isstudied. Our theoretical results show that the proposed method can achieve afaster convergence rate than the optimal convergence rate for sampling methods.Theoretical and numerical results show that the proposed estimator has goodstatistical performance as well as low computational cost.

----

## [1517] Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry

**Authors**: *Bariscan Bozkurt, Cengiz Pehlevan, Alper T. Erdogan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6dea02c16a492682d66c6f626c306db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6dea02c16a492682d66c6f626c306db2-Abstract-Conference.html)

**Abstract**:

The backpropagation algorithm has experienced remarkable success in training large-scale artificial neural networks; however, its biological plausibility has been strongly criticized, and it remains an open question whether the brain employs supervised learning mechanisms akin to it. Here, we propose correlative information maximization between layer activations as an alternative normative approach to describe the signal propagation in biological neural networks in both forward and backward directions. This new framework addresses many concerns about the biological-plausibility of conventional artificial neural networks and the backpropagation algorithm. The coordinate descent-based optimization of the corresponding objective, combined with the mean square error loss function for fitting labeled supervision data, gives rise to a neural network structure that emulates a more biologically realistic network of multi-compartment pyramidal neurons with dendritic processing and lateral inhibitory neurons. Furthermore, our approach provides a natural resolution to the weight symmetry problem between forward and backward signal propagation paths, a significant critique against the plausibility of the conventional backpropagation algorithm. This is achieved by leveraging two alternative, yet equivalent forms of the correlative mutual information objective. These alternatives intrinsically lead to forward and backward prediction networks without weight symmetry issues, providing a compelling solution to this long-standing challenge.

----

## [1518] What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?

**Authors**: *Fnu Suya, Xiao Zhang, Yuan Tian, David Evans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e2986deda273d8fb903342841fcc4dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e2986deda273d8fb903342841fcc4dc-Abstract-Conference.html)

**Abstract**:

We study indiscriminate poisoning for linear learners where an adversary injects a few  crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.

----

## [1519] Expressive probabilistic sampling in recurrent neural networks

**Authors**: *Shirui Chen, Linxing Jiang, Rajesh P. N. Rao, Eric Shea-Brown*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e2a1a8a037f9a06004fe651054e8938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e2a1a8a037f9a06004fe651054e8938-Abstract-Conference.html)

**Abstract**:

In sampling-based Bayesian models of brain function, neural activities are assumed to be samples from probability distributions that the brain uses for probabilistic computation. However, a comprehensive understanding of how mechanistic models of neural dynamics can sample from arbitrary distributions is still lacking. We use tools from functional analysis and stochastic differential equations to explore the minimum architectural requirements for $\textit{recurrent}$ neural circuits to sample from complex distributions. We first consider the traditional sampling model consisting of a network of neurons whose outputs directly represent the samples ($\textit{sampler-only}$ network). We argue that synaptic current and firing-rate dynamics in the traditional model have limited capacity to sample from a complex probability distribution. We show that the firing rate dynamics of a recurrent neural circuit with a separate set of output units can sample from an arbitrary probability distribution. We call such circuits $\textit{reservoir-sampler networks}$ (RSNs). We propose an efficient training procedure based on denoising score matching that finds recurrent and output weights such that the RSN implements Langevin sampling. We empirically demonstrate our model's ability to sample from several complex data distributions using the proposed neural dynamics and discuss its applicability to developing the next generation of sampling-based Bayesian brain models.

----

## [1520] Counting Distinct Elements Under Person-Level Differential Privacy

**Authors**: *Thomas Steinke, Alexander Knop*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e32c247076c2c0fb381e022c02d2c78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e32c247076c2c0fb381e022c02d2c78-Abstract-Conference.html)

**Abstract**:

We study the problem of counting the number of distinct elements in a dataset subject to the constraint of differential privacy. We consider the challenging setting of person-level DP (a.k.a. user-level DP) where each person may contribute an unbounded number of items and hence the sensitivity is unbounded.Our approach is to compute a bounded-sensitivity version of this query, which reduces to solving a max-flow problem. The sensitivity bound is optimized to balance the noise we must add to privatize the answer against the error of the approximation of the bounded-sensitivity query to the true number of unique elements.

----

## [1521] Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks

**Authors**: *Feng Chen, Daniel Kunin, Atsushi Yamamura, Surya Ganguli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e4432b912599d11609b9cdf98c823c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e4432b912599d11609b9cdf98c823c5-Abstract-Conference.html)

**Abstract**:

In this work, we reveal a strong implicit bias of stochastic gradient descent (SGD) that drives overly expressive networks to much simpler subnetworks, thereby dramatically reducing the number of independent parameters, and improving generalization. To reveal this bias, we identify invariant sets, or subsets of parameter space that remain unmodified by SGD. We focus on two classes of invariant sets that correspond to simpler (sparse or low-rank) subnetworks and commonly appear in modern architectures. Our analysis uncovers that SGD exhibits a property of stochastic attractivity towards these simpler invariant sets. We establish a sufficient condition for stochastic attractivity based on a competition between the loss landscape's curvature around the invariant set and the noise introduced by stochastic gradients. Remarkably, we find that an increased level of noise strengthens attractivity, leading to the emergence of attractive invariant sets associated with saddle-points or local maxima of the train loss. We observe empirically the existence of attractive invariant sets in trained deep neural networks, implying that SGD dynamics often collapses to simple subnetworks with either vanishing or redundant neurons. We further demonstrate how this simplifying process of stochastic collapse benefits generalization in a linear teacher-student framework. Finally, through this analysis, we mechanistically explain why early training with large learning rates for extended periods benefits subsequent generalization.

----

## [1522] Conservative State Value Estimation for Offline Reinforcement Learning

**Authors**: *Liting Chen, Jie Yan, Zhengdao Shao, Lu Wang, Qingwei Lin, Saravanakumar Rajmohan, Thomas Moscibroda, Dongmei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning faces a significant challenge of value over-estimation due to the distributional drift between the dataset and the current learned policy, leading to learning failure in practice. The common approach is to incorporate a penalty term to reward or value estimation in the Bellman iterations. Meanwhile, to avoid extrapolation on out-of-distribution (OOD) states and actions, existing methods focus on conservative Q-function estimation. In this paper, we propose Conservative State Value Estimation (CSVE), a new approach that learns conservative V-function via directly imposing penalty on OOD states. Compared to prior work, CSVE allows more effective state value estimation with conservative guarantees and further better policy optimization. Further, we apply CSVE and develop a practical actor-critic algorithm in which the critic does the conservative value estimation by additionally sampling and penalizing the states around the dataset, and the actor applies advantage weighted updates extended with state exploration to improve the policy. We evaluate in classic continual control tasks of D4RL, showing that our method performs better than the conservative Q-function learning methods and is strongly competitive among recent SOTA methods.

----

## [1523] Demystifying Oversmoothing in Attention-Based Graph Neural Networks

**Authors**: *Xinyi Wu, Amir Ajorlou, Zihui Wu, Ali Jadbabaie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e4cdfdd909ea4e34bfc85a12774cba0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e4cdfdd909ea4e34bfc85a12774cba0-Abstract-Conference.html)

**Abstract**:

Oversmoothing in Graph Neural Networks (GNNs) refers to the phenomenon where increasing network depth leads to homogeneous node representations. While previous work has established that Graph Convolutional Networks (GCNs) exponentially lose expressive power, it remains controversial whether the graph attention mechanism can mitigate oversmoothing. In this work, we provide a definitive answer to this question through a rigorous mathematical analysis, by viewing attention-based GNNs as nonlinear time-varying dynamical systems and incorporating tools and techniques from the theory of products of inhomogeneous matrices and the joint spectral radius. We establish that, contrary to popular belief, the graph attention mechanism cannot prevent oversmoothing and loses expressive power exponentially. The proposed framework extends the existing results on oversmoothing for symmetric GCNs to a significantly broader class of GNN models, including random walk GCNs, Graph Attention Networks (GATs) and (graph) transformers. In particular, our analysis accounts for asymmetric, state-dependent and time-varying aggregation operators and a wide range of common nonlinear activation functions, such as ReLU, LeakyReLU, GELU and SiLU.

----

## [1524] A Comprehensive Benchmark for Neural Human Radiance Fields

**Authors**: *Kenkun Liu, Derong Jin, Ailing Zeng, Xiaoguang Han, Lei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The past two years have witnessed a significant increase in interest concerning NeRF-based human body rendering. While this surge has propelled considerable advancements, it has also led to an influx of methods and datasets. This explosion complicates experimental settings and makes fair comparisons challenging. In this work, we design and execute thorough studies into unified evaluation settings and metrics to establish a fair and reasonable benchmark for human NeRF models. To reveal the effects of extant models, we benchmark them against diverse and hard scenes. Additionally, we construct a cross-subject benchmark pre-trained on large-scale datasets to assess generalizable methods. Finally, we analyze the essential components for animatability and generalizability, and make HumanNeRF from monocular videos generalizable, as the inaugural baseline. We hope these benchmarks and analyses could serve the community.

----

## [1525] Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms

**Authors**: *Qian Yu, Yining Wang, Baihe Huang, Qi Lei, Jason D. Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html)

**Abstract**:

In stochastic zeroth-order optimization, a problem of practical relevance is understanding how to fully exploit the local geometry of the underlying objective function. We consider a fundamental setting in which the objective function is quadratic, and provide the first tight characterization of the optimal Hessian-dependent sample complexity. Our contribution is twofold. First, from an information-theoretic point of view, we prove tight lower bounds on Hessian-dependent complexities by introducing a concept called \emph{energy allocation}, which captures the interaction between the searching algorithm and the geometry of objective functions. A matching upper bound is obtained by solving the optimal energy spectrum. Then, algorithmically, we show the existence of a Hessian-independent algorithm that universally achieves the asymptotic optimal sample complexities for all Hessian instances. The optimal sample complexities achieved by our algorithm remain valid for heavy-tailed noise distributions, which are enabled by a truncation method.

----

## [1526] Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

**Authors**: *Erin George, Michael Murray, William Swartworth, Deanna Needell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e73c39cc428c7d264d9820319f31e79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e73c39cc428c7d264d9820319f31e79-Abstract-Conference.html)

**Abstract**:

We study benign overfitting in two-layer ReLU networks trained using gradient descent and hinge loss on noisy data for binary classification. In particular, we consider linearly separable data for which a relatively small proportion of labels are corrupted or flipped. We identify conditions on the margin of the clean data that give rise to three distinct training outcomes: benign overfitting, in which zero loss is achieved and with high probability test data is classified correctly; overfitting, in which zero loss is achieved but test data is misclassified with probability lower bounded by a constant; and non-overfitting, in which clean points, but not corrupt points, achieve zero loss and again with high probability test data is classified correctly. Our analysis provides a fine-grained description of the dynamics of neurons throughout training and reveals two distinct phases: in the first phase clean points achieve close to zero loss, in the second phase clean points oscillate on the boundary of zero loss while corrupt points either converge towards zero loss or are eventually zeroed by the network. We prove these results using a combinatorial approach that involves bounding the number of clean versus corrupt updates during these phases of training.

----

## [1527] Adaptive Algorithms for Relaxed Pareto Set Identification

**Authors**: *Cyrille Kone, Emilie Kaufmann, Laura Richert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e976e7930460b5c3167a104ba8cc39c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e976e7930460b5c3167a104ba8cc39c-Abstract-Conference.html)

**Abstract**:

In this paper we revisit the fixed-confidence identification of the Pareto optimal set in a multi-objective multi-armed bandit model. As the sample complexity to identify the exact Pareto set can be very large, a relaxation allowing to output some additional near-optimal arms has been studied. In this work we also tackle alternative relaxations that allow instead to identify a relevant \emph{subset} of the Pareto set. Notably, we propose a single sampling strategy, called Adaptive Pareto Exploration, that can be used in conjunction with different stopping rules to take into account different relaxations of the Pareto Set Identification problem. We analyze the sample complexity of these different combinations, quantifying in particular the reduction in sample complexity that occurs when one seeks to identify at most $k$ Pareto optimal arms. We showcase the good practical performance of Adaptive Pareto Exploration on a real-world scenario, in which we adaptively explore several vaccination strategies against Covid-19 in order to find the optimal ones when multiple immunogenicity criteria are taken into account.

----

## [1528] PHOTOSWAP: Personalized Subject Swapping in Images

**Authors**: *Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, Hyunjoon Jung, Xin Eric Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e9a0a72da9b76c3ebc8cc33ff10ac29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e9a0a72da9b76c3ebc8cc33ff10ac29-Abstract-Conference.html)

**Abstract**:

In an era where images and visual content dominate our digital landscape, the ability to manipulate and personalize these images has become a necessity.Envision seamlessly substituting a tabby cat lounging on a sunlit window sill in a photograph with your own playful puppy, all while preserving the original charm and composition of the image. We present \emph{Photoswap}, a novel approach that enables this immersive image editing experience through personalized subject swapping in existing images.\emph{Photoswap} first learns the visual concept of the subject from reference images and then swaps it into the target image using pre-trained diffusion models in a training-free manner. We establish that a well-conceptualized visual subject can be seamlessly transferred to any image with appropriate self-attention and cross-attention manipulation, maintaining the pose of the swapped subject and the overall coherence of the image. Comprehensive experiments underscore the efficacy and controllability of \emph{Photoswap} in personalized subject swapping. Furthermore, \emph{Photoswap} significantly outperforms baseline methods in human ratings across subject swapping, background preservation, and overall quality, revealing its vast application potential, from entertainment to professional editing.

----

## [1529] Simplifying Neural Network Training Under Class Imbalance

**Authors**: *Ravid Shwartz-Ziv, Micah Goldblum, Yucen Lily Li, C. Bayan Bruss, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ea69f8116b7c01e3c3e43b62e6868fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ea69f8116b7c01e3c3e43b62e6868fc-Abstract-Conference.html)

**Abstract**:

Real-world datasets are often highly class-imbalanced, which can adversely impact the performance of deep learning models. The majority of research on training neural networks under class imbalance has focused on specialized loss functions and sampling techniques. Notably, we demonstrate that simply tuning existing components of standard deep learning pipelines, such as the batch size, data augmentation, architecture size, pre-training, optimizer, and label smoothing, can achieve state-of-the-art performance without any specialized loss functions or samplers. We also provide key prescriptions and considerations for training under class imbalance, and an understanding of why imbalance methods succeed or fail.

----

## [1530] Regret Minimization via Saddle Point Optimization

**Authors**: *Johannes Kirschner, Seyed Alireza Bakhtiari, Kushagra Chandak, Volodymyr Tkachuk, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6eaf8c729af4fbeb18006dc2e6a41d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6eaf8c729af4fbeb18006dc2e6a41d9b-Abstract-Conference.html)

**Abstract**:

A long line of works characterizes the sample complexity of regret minimization in sequential decision-making by min-max programs. In the corresponding saddle-point game, the min-player optimizes the sampling distribution against an adversarial max-player that chooses confusing models leading to large regret. The most recent instantiation of this idea is the decision-estimation coefficient (DEC), which was shown to provide nearly tight lower and upper bounds on the worst-case expected regret in structured bandits and reinforcement learning. By re-parametrizing the offset DEC with the confidence radius and solving the corresponding min-max program, we derive an anytime variant of the Estimation-To-Decisions algorithm (Anytime-E2D). Importantly, the algorithm optimizes the exploration-exploitation trade-off online instead of via the analysis. Our formulation leads to a practical algorithm for finite model classes and linear feedback models. We further point out connections to the information ratio, decoupling coefficient and PAC-DEC, and numerically evaluate the performance of E2D on simple examples.

----

## [1531] On the Sublinear Regret of GP-UCB

**Authors**: *Justin Whitehouse, Aaditya Ramdas, Zhiwei Steven Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html)

**Abstract**:

In the kernelized bandit problem, a learner aims to sequentially compute the optimum of a function lying in a reproducing kernel Hilbert space given only noisy evaluations at sequentially chosen points. In particular, the learner aims to minimize regret, which is a measure of the suboptimality of the choices made.Arguably the most popular algorithm is the Gaussian Process Upper Confidence Bound (GP-UCB) algorithm, which involves acting based on a simple linear estimator of the unknown function.Despite its popularity, existing analyses of GP-UCB give a suboptimal regret rate, which fails to be sublinear for many commonly used kernels such as the Matern kernel. This has led to a longstanding open question: are existing regret analyses for GP-UCB tight, or can bounds be improved by using more sophisticated analytical techniques?In this work, we resolve this open question and show that GP-UCB enjoys nearly optimal regret. In particular, our results yield sublinear regret rates for the Matern kernel, improving over the state-of-the-art analyses and partially resolving a COLT open problem posed by Vakili et al. Our improvements rely on a key technical contribution --- regularizing kernel ridge estimators in proportion to the smoothness of the underlying kernel $k$. Applying this key idea together with a largely overlooked concentration result in separable Hilbert spaces (for which we provide an independent, simplified derivation), we are able to provide a tighter analysis of the GP-UCB algorithm.

----

## [1532] Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks

**Authors**: *Jun Yin, Chaozhuo Li, Hao Yan, Jianxun Lian, Senzhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ecd51685e2d765bc0ad32a2e73faf62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ecd51685e2d765bc0ad32a2e73faf62-Abstract-Conference.html)

**Abstract**:

Intrinsic interpretable graph neural networks aim to provide transparent predictions by identifying the influential fraction of the input graph that guides the model prediction, i.e., the explanatory subgraph. However, current interpretable GNNs mostly are dataset-specific and hard to generalize to different graphs. A more generalizable GNN interpretation model which can effectively distill the universal structural patterns of different graphs is until-now unexplored. Motivated by the great success of recent pre-training techniques, we for the first time propose the Pre-training Interpretable Graph Neural Network ($\pi$-GNN) to distill the universal interpretability of GNNs by pre-training over synthetic graphs with ground-truth explanations. Specifically, we introduce a structural pattern learning module to extract diverse universal structure patterns and integrate them together to comprehensively represent the graphs of different types. Next, a hypergraph refining module is proposed to identify the explanatory subgraph by incorporating the universal structure patterns with local edge interactions. Finally, the task-specific predictor is cascaded with the pre-trained $\pi$-GNN model and fine-tuned over downstream tasks. Extensive experiments demonstrate that $\pi$-GNN significantly surpasses the leading interpretable GNN baselines with up to 9.98\% interpretation improvement and 16.06\% classification accuracy improvement. Meanwhile, $\pi$-GNN pre-trained on graph classification task also achieves the top-tier interpretation performance on node classification task, which further verifies its promising generalization performance among different downstream tasks. Our code and datasets are available at https://anonymous.4open.science/r/PI-GNN-F86C

----

## [1533] Quantum speedups for stochastic optimization

**Authors**: *Aaron Sidford, Chenyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ed9931d6e1fb6a85efa1b2c014a47e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ed9931d6e1fb6a85efa1b2c014a47e1-Abstract-Conference.html)

**Abstract**:

We consider the problem of minimizing a continuous function given given access to a natural quantum generalization of a stochastic gradient oracle. We provide two new methods for the special case of minimizing a Lipschitz convex function. Each method obtains a dimension versus accuracy trade-off which is provably unachievable classically and we prove that one method is asymptotically optimal in low-dimensional settings. Additionally, we provide quantum algorithms for computing a critical point of a smooth non-convex function at rates not known to be achievable classically. To obtain these results we build upon the quantum multivariate mean estimation result of Cornelissen et al. and provide a general quantum variance reduction technique of independent interest.

----

## [1534] Concept Algebra for (Score-Based) Text-Controlled Generative Models

**Authors**: *Zihao Wang, Lin Gui, Jeffrey Negrea, Victor Veitch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f125214c86439d107ccb58e549e828f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f125214c86439d107ccb58e549e828f-Abstract-Conference.html)

**Abstract**:

This paper concerns the structure of learned representations in text-guided generative models, focusing on score-based models. A key property of such models is that they can compose disparate concepts in a 'disentangled' manner.This suggests these models have internal representations that encode concepts in a 'disentangled' manner. Here, we focus on the idea that concepts are encoded as subspaces of some representation space.  We formalize what this means, show there's a natural choice for the representation, and develop a simple method for identifying the part of the representation corresponding to a given concept. In particular, this allows us to manipulate the concepts expressed by the model through algebraic manipulation of the representation. We demonstrate the idea with examples using Stable Diffusion.

----

## [1535] Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics

**Authors**: *Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, Nicolas Courty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)

**Abstract**:

Wasserstein distance (WD) and the associated optimal transport plan have been proven useful in many applications where probability measures are at stake. In this paper, we propose a new proxy of the squared WD, coined $\textnormal{min-SWGG}$, that is based on the transport map induced by an optimal one-dimensional projection of the two input distributions. We draw connections between  $\textnormal{min-SWGG}$, and Wasserstein generalized geodesics in which the pivot measure is supported on a line. We notably provide a new closed form for the exact Wasserstein distance in the particular case of one of the distributions supported on a line allowing us to derive a fast computational scheme that is amenable to gradient descent optimization. We show that  $\textnormal{min-SWGG}$, is an upper bound of WD and that it has a complexity similar to as Sliced-Wasserstein, with the additional feature of providing an associated transport plan. We also investigate some theoretical properties such as metricity, weak convergence, computational and topological properties. Empirical evidences support the benefits of  $\textnormal{min-SWGG}$, in various contexts, from gradient flows, shape matching and image colorization, among others.

----

## [1536] Aggregating Capacity in FL through Successive Layer Training for Computationally-Constrained Devices

**Authors**: *Kilian Pfeiffer, Ramin Khalili, Jörg Henkel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f43166f50f26e8d8f3edc5545b0749f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f43166f50f26e8d8f3edc5545b0749f-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is usually performed on resource-constrained edge devices, e.g., with limited memory for the computation. If the required memory to train a model exceeds this limit, the device will be excluded from the training. This can lead to a lower accuracy as valuable data and computation resources are excluded from training, also causing bias and unfairness. The FL training process should be adjusted to such constraints. The state-of-the-art techniques propose training subsets of the FL model at constrained devices, reducing their resource requirements for training. However, these techniques largely limit the co-adaptation among parameters of the model and are highly inefficient, as we show: it is actually better to train a smaller (less accurate) model by the system where all the devices can train the model end-to-end than applying such techniques. We propose a new method that enables successive freezing and training of the parameters of the FL model at devices, reducing the trainingâ€™s resource requirements at the devices while still allowing enough co-adaptation between parameters. We show through extensive experimental evaluation that our technique greatly improves the accuracy of the trained model (by 52.4 p.p. ) compared with the state of the art, efficiently aggregating the computation capacity available on distributed devices.

----

## [1537] FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations

**Authors**: *Chanakya Ekbote, Ajinkya Pankaj Deshpande, Arun Iyer, Sundararajan Sellamanickam, Ramakrishna Bairi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html)

**Abstract**:

Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe, achieves an average gain of up to 4.4\%, compared to the state-of-the-art unsupervised models, across all datasets in consideration, both homophilic and heterophilic. Our code can be found at: https://github.com/Microsoft/figure.

----

## [1538] Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams

**Authors**: *Esmaeil Seraj, Jerry Xiong, Mariah Schrum, Matthew C. Gombolay*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f5288d7059cbe3f5a19dad1b3bf17e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f5288d7059cbe3f5a19dad1b3bf17e1-Abstract-Conference.html)

**Abstract**:

Extending recent advances in Learning from Demonstration (LfD) frameworks to multi-robot settings poses critical challenges such as environment non-stationarity due to partial observability which is detrimental to the applicability of existing methods. Although prior work has shown that enabling communication among agents of a robot team can alleviate such issues, creating inter-agent communication under existing Multi-Agent LfD (MA-LfD) frameworks requires the human expert to provide demonstrations for both environment actions and communication actions, which necessitates an efficient communication strategy on a known message spaces. To address this problem, we propose Mixed-Initiative Multi-Agent Apprenticeship Learning (MixTURE). MixTURE enables robot teams to learn from a human expert-generated data a preferred policy to accomplish a collaborative task, while simultaneously learning emergent inter-agent communication to enhance team coordination. The key ingredient to MixTURE's success is automatically learning a communication policy, enhanced by a mutual-information maximizing reverse model that rationalizes the underlying expert demonstrations without the need for human generated data or an auxiliary reward function. MixTURE outperforms a variety of relevant baselines on diverse data generated by human experts in complex heterogeneous domains. MixTURE is the first MA-LfD framework to enable learning multi-robot collaborative policies directly from real human data, resulting in ~44% less human workload, and ~46% higher usability score.

----

## [1539] Meta-Learning Adversarial Bandit Algorithms

**Authors**: *Misha Khodak, Ilya Osadchiy, Keegan Harris, Maria-Florina Balcan, Kfir Y. Levy, Ron Meir, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f627c706a7d9961cc1ff55f37f07f97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f627c706a7d9961cc1ff55f37f07f97-Abstract-Conference.html)

**Abstract**:

We study online meta-learning with bandit feedback, with the goal of improving performance across multiple tasks if they are similar according to some natural similarity measure.  As the first to target the adversarial online-within-online partial-information setting, we design meta-algorithms that combine outer learners to simultaneously tune the initialization and other hyperparameters of an inner learner for two important cases:  multi-armed bandits (MAB) and bandit linear optimization (BLO).  For MAB, the meta-learners initialize and set hyperparameters of the Tsallis-entropy generalization of Exp3, with the task-averaged regret improving if the entropy of the optima-in-hindsight is small.  For BLO, we learn to initialize and tune online mirror descent (OMD) with self-concordant barrier regularizers, showing that task-averaged regret varies directly with an action space-dependent measure they induce.  Our guarantees rely on proving that unregularized follow-the-leader combined with two levels of low-dimensional hyperparameter tuning is enough to learn a sequence of affine functions of non-Lipschitz and sometimes non-convex Bregman divergences bounding the regret of OMD.

----

## [1540] Geometric Algebra Transformer

**Authors**: *Johann Brehmer, Pim de Haan, Sönke Behrends, Taco S. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html)

**Abstract**:

Problems involving geometric data arise in physics, chemistry, robotics, computer vision, and many other fields. Such data can take numerous forms, for instance points, direction vectors, translations, or rotations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric (or Clifford) algebra, which offers an efficient 16-dimensional vector-space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a Transformer, GATr is versatile, efficient, and scalable. We demonstrate GATr in problems from n-body modeling to wall-shear-stress estimation on large arterial meshes to robotic motion planning. GATr consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability.

----

## [1541] Top-Ambiguity Samples Matter: Understanding Why Deep Ensemble Works in Selective Classification

**Authors**: *Qiang Ding, Yixuan Cao, Ping Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html)

**Abstract**:

Selective classification allows a machine learning model to reject some hard inputs and thus improve the reliability of its predictions. In this area, the ensemble method is powerful in practice, but there has been no solid analysis on why the ensemble method works. Inspired by an interesting empirical result that the improvement of the ensemble largely comes from top-ambiguity samples where its member models diverge, we prove that, based on some assumptions, the ensemble has a lower selective risk than the member model for any coverage within a range. The proof is nontrivial since the selective risk is a non-convex function of the model prediction. The assumptions and the theoretical results are supported by systematic experiments on both computer vision and natural language processing tasks.

----

## [1542] Unlimiformer: Long-Range Transformers with Unlimited Length Input

**Authors**: *Amanda Bertsch, Uri Alon, Graham Neubig, Matthew R. Gormley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f9806a5adc72b5b834b27e4c7c0df9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f9806a5adc72b5b834b27e4c7c0df9b-Abstract-Conference.html)

**Abstract**:

Since the proposal of transformers, these models have been limited to bounded input lengths, because of their need to attend to every token in the input. In this work, we propose Unlimiformer: a general approach that wraps any existing pretrained encoder-decoder transformer, and offloads the cross-attention computation to a single $k$-nearest-neighbor ($k$NN) index, while the returned $k$NN distances are the attention dot-product scores. This $k$NN index can be kept on either the GPU or CPU memory and queried in sub-linear time; this way, we can index practically unlimited input sequences, while every attention head in every decoder layer retrieves its top-$k$ keys, instead of attending to every key.  We evaluate Unlimiformer on several long-document and book-summarization benchmarks, showing that it can process even **500k** token-long inputs from the BookSum dataset, without any input truncation at test time. We demonstrate that Unlimiformer improves pretrained models such as BART and Longformer by extending them to unlimited inputs without additional learned weights and without modifying their code. Our code and models are publicly available at https://github.com/abertsch72/unlimiformer , and support LLaMA-2 as well.

----

## [1543] Improving CLIP Training with Language Rewrites

**Authors**: *Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) stands as one of the most effective and scalable methods for training transferable vision models using paired image and text data. CLIP models are trained using contrastive loss, which typically relies on data augmentations to prevent overfitting and shortcuts. However, in the CLIP training paradigm, data augmentations are exclusively applied to image inputs, while language inputs remain unchanged throughout the entire training process, limiting the exposure of diverse texts to the same image. In this paper, we introduce Language augmented CLIP (LaCLIP), a simple yet highly effective approach to enhance CLIP training through language rewrites. Leveraging the in-context learning capability of large language models, we rewrite the text descriptions associated with each image. These rewritten texts exhibit diversity in sentence structure and vocabulary while preserving the original key concepts and meanings. During training, LaCLIP randomly selects either the original texts or the rewritten versions as text augmentations for each image. Extensive experiments on CC3M, CC12M, RedCaps and LAION-400M datasets show that CLIP pre-training with language rewrites significantly improves the transfer performance without computation or memory overhead during training. Specifically for ImageNet zero-shot accuracy, LaCLIP outperforms CLIP by 8.2% on CC12M and 2.4% on LAION-400M.

----

## [1544] Extensible Prompts for Language Models on Zero-shot Language Style Customization

**Authors**: *Tao Ge, Jing Hu, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fcbfb3721c1781728b10c6685cc2f6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fcbfb3721c1781728b10c6685cc2f6c-Abstract-Conference.html)

**Abstract**:

We propose eXtensible Prompt (X-Prompt) for prompting a large language model (LLM) beyond natural language (NL). X-Prompt instructs an LLM with not only NL but also an extensible vocabulary of imaginary words. Registering new imaginary words allows us to instruct the LLM to comprehend concepts that are difficult to describe with NL words, thereby making a prompt more descriptive. Also, these imaginary words are designed to be out-of-distribution (OOD) robust so that they can be (re)used like NL words in various prompts, distinguishing X-Prompt from soft prompt that is for fitting in-distribution data. We propose context-augmented learning (CAL) to learn imaginary words for general usability, enabling them to work properly in OOD (unseen) prompts. We experiment X-Prompt for zero-shot language style customization as a case study. The promising results of X-Prompt demonstrate its potential to facilitate advanced interaction beyond the natural language interface, bridging the communication gap between humans and LLMs.

----

## [1545] MIMEx: Intrinsic Rewards from Masked Input Modeling

**Authors**: *Toru Lin, Allan Jabri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fe10a4c0d680609f0560920bd9ade4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fe10a4c0d680609f0560920bd9ade4a-Abstract-Conference.html)

**Abstract**:

Exploring in environments with high-dimensional observations is hard. One promising approach for exploration is to use intrinsic rewards, which often boils down to estimating "novelty" of states, transitions, or trajectories with deep networks. Prior works have shown that conditional prediction objectives such as masked autoencoding can be seen as stochastic estimation of pseudo-likelihood. We show how this perspective naturally leads to a unified view on existing intrinsic reward approaches: they are special cases of conditional prediction, where the estimation of novelty can be seen as pseudo-likelihood estimation with different mask distributions. From this view, we propose a general framework for deriving intrinsic rewards -- Masked Input Modeling for Exploration (MIMEx) -- where the mask distribution can be flexibly tuned to control the difficulty of the underlying conditional prediction task. We demonstrate that MIMEx can achieve superior results when compared against competitive baselines on a suite of challenging sparse-reward visuomotor tasks.

----

## [1546] RGMIL: Guide Your Multiple-Instance Learning Model with Regressor

**Authors**: *Zhaolong Du, Shasha Mao, Yimeng Zhang, Shuiping Gou, Licheng Jiao, Lin Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html)

**Abstract**:

In video analysis, an important challenge is insufficient annotated data due to the rare occurrence of the critical patterns, and we need to provide discriminative frame-level representation with limited annotation in some applications. Multiple Instance Learning (MIL) is suitable for this scenario. However, many MIL models paid attention to analyzing the relationships between instance representations and aggregating them, but neglecting the critical information from the MIL problem itself, which causes difficultly achieving ideal instance-level performance compared with the supervised model.To address this issue, we propose the $\textbf{\textit{Regressor-Guided MIL network} (RGMIL)}$, which effectively produces discriminative instance-level representations in a general multi-classification scenario. In the proposed method, we make full use of the $\textit{regressor}$ through our newly introduced $\textit{aggregator}$, $\textbf{\textit{Regressor-Guided Pooling} (RGP)}$. RGP focuses on simulating the correct inference process of humans while facing similar problems without introducing new parameters, and the MIL problem can be accurately described through the critical information from the $\textit{regressor}$ in our method. In experiments, RGP shows dominance on more than 20 MIL benchmark datasets, with the average bag-level classification accuracy close to 1. We also perform a series of comprehensive experiments on the MMNIST dataset. Experimental results illustrate that our $\textit{aggregator}$ outperforms existing methods under different challenging circumstances. Instance-level predictions are even possible under the guidance of RGP information table in a long sequence. RGMIL also presents comparable instance-level performance with S-O-T-A supervised models in complicated applications. Statistical results demonstrate the assumption that a MIL model can compete with a supervised model at the instance level, as long as a structure that accurately describes the MIL problem is provided. The codes are available on $\url{https://github.com/LMBDA-design/RGMIL}$.

----

## [1547] Stochastic Multi-armed Bandits: Optimal Trade-off among Optimality, Consistency, and Tail Risk

**Authors**: *David Simchi-Levi, Zeyu Zheng, Feng Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ffa1f5ad26addef897dcb938e525db7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ffa1f5ad26addef897dcb938e525db7-Abstract-Conference.html)

**Abstract**:

We consider the stochastic multi-armed bandit problem and fully characterize the interplays among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to achieve the optimal regret tail risk for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, 1)$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ and instance-dependent expected regret of $\tilde O(T^\beta)$, while enjoys a probability of incurring an $\Omega(T^\delta)$ regret that decays exponentially with a polynomial $T$ term. Such decaying rate is proved to be best achievable. We also generalize our analysis to the stochastic multi-armed bandit problem with non-stationary baseline rewards, where in each time period $t$, the decision maker pulls one of $K$ arms and collects a reward which is the sum of three terms: the mean of the pulled arm, an independent noise, and a non-stationary baseline reward as a function of $t$. Our results reveal insights on the trade-off between expected regret and tail risk for both worst-case and instance-dependent scenario, indicating that more sub-optimality and inconsistency leaves space for more light-tailed risk of incurring a large regret.

----

## [1548] Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

**Authors**: *Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ffe484a646db13891bb6435ca39d667-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ffe484a646db13891bb6435ca39d667-Abstract-Conference.html)

**Abstract**:

Recently, pre-trained vision-language models have been increasingly used to tackle the challenging zero-shot segmentation task. Typical solutions follow the paradigm of first generating mask proposals and then adopting CLIP to classify them. To maintain the CLIP's zero-shot transferability, previous practices favour to freeze CLIP during training. However, in the paper, we reveal that CLIP is insensitive to different mask proposals and tends to produce similar predictions for various mask proposals of the same image. This insensitivity results in numerous false positives when classifying mask proposals. This issue mainly relates to the fact that CLIP is trained with image-level supervision. To alleviate this issue, we propose a simple yet effective method, named Mask-aware Fine-tuning (MAFT). Specifically,  Image-Proposals CLIP Encoder (IP-CLIP Encoder) is proposed to handle arbitrary numbers of image and mask proposals simultaneously. Then, mask-aware loss and self-distillation loss are designed to fine-tune IP-CLIP Encoder, ensuring CLIP is responsive to different mask proposals while not sacrificing transferability. In this way, mask-aware representations can be easily learned to make the true positives stand out. Notably, our solution can seamlessly plug into most existing methods without introducing any new parameters during the fine-tuning process. We conduct extensive experiments on the popular zero-shot benchmarks. With MAFT, the performance of the state-of-the-art methods is promoted by a large margin: 50.4\% (+ 8.2\%) on COCO, 81.8\% (+ 3.2\%) on Pascal-VOC, and 8.7\% (+4.3\%) on ADE20K in terms of mIoU for unseen classes. Codes will be provided for reproducibility. Code is available at https://github.com/jiaosiyu1999/MAFT.git .

----

## [1549] Understanding Multi-phase Optimization Dynamics and Rich Nonlinear Behaviors of ReLU Networks

**Authors**: *Mingze Wang, Chao Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7016d7b7b6e3c05b2128ac5b3aae492d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7016d7b7b6e3c05b2128ac5b3aae492d-Abstract-Conference.html)

**Abstract**:

The training process of ReLU neural networks often exhibits complicated nonlinear phenomena. The nonlinearity of models and non-convexity of loss pose significant challenges for theoretical analysis. Therefore, most previous theoretical works on the optimization dynamics of neural networks focus either on local analysis (like the end of training) or approximate linear models (like Neural Tangent Kernel). In this work, we conduct a complete theoretical characterization of the training process of a two-layer ReLU network trained by Gradient Flow on a linearly separable data. In this specific setting, our analysis captures the whole optimization process starting from random initialization to final convergence. Despite the relatively simple model and data that we studied, we reveal four different phases from the whole training process showing a general simplifying-to-complicating learning trend.Specific nonlinear behaviors can also be precisely identified and captured theoretically, such asinitial condensation, saddle-to-plateau dynamics, plateau escape, changes of activation patterns, learning with increasing complexity, etc.

----

## [1550] RD-Suite: A Benchmark for Ranking Distillation

**Authors**: *Zhen Qin, Rolf Jagerman, Rama Kumar Pasumarthi, Honglei Zhuang, He Zhang, Aijun Bai, Kai Hui, Le Yan, Xuanhui Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/701eba0f98c6f28ffee0de5969d8d034-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/701eba0f98c6f28ffee0de5969d8d034-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The distillation of ranking models has become an important topic in both academia and industry. In recent years, several advanced methods have been proposed to tackle this problem, often leveraging ranking information from teacher rankers that is absent in traditional classification settings. To date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide range of tasks and datasets make it difficult to assess or invigorate advances in this field. This paper first examines representative prior arts on ranking distillation, and raises three questions to be answered around methodology and reproducibility. To that end, we propose a systematic and unified benchmark, Ranking Distillation Suite (RD-Suite), which is a suite of tasks with 4 large real-world datasets, encompassing two major modalities (textual and numeric) and two applications (standard distillation and distillation transfer). RD-Suite consists of benchmark results that challenge some of the common wisdom in the field, and the release of datasets with teacher scores and evaluation scripts for future research. RD-Suite paves the way towards better understanding of ranking distillation, facilities more research in this direction, and presents new challenges.

----

## [1551] Gradient Descent with Linearly Correlated Noise: Theory and Applications to Differential Privacy

**Authors**: *Anastasia Koloskova, Ryan McKenna, Zachary Charles, John Keith Rush, H. Brendan McMahan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70255afc962aca0930327c090eb7d8c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70255afc962aca0930327c090eb7d8c5-Abstract-Conference.html)

**Abstract**:

We study gradient descent under linearly correlated noise. Our work is motivated by recent practical methods for optimization with differential privacy (DP), such as DP-FTRL, which achieve strong performance in settings where privacy amplification techniques are infeasible (such as in federated learning). These methods inject privacy noise through a matrix factorization mechanism, making the noise linearly correlated over iterations. We propose a simplified setting that distills key facets of these methods and isolates the impact of linearly correlated noise. We analyze the behavior of gradient descent in this setting, for both convex and non-convex functions. Our analysis is demonstrably tighter than prior work and recovers multiple important special cases exactly (including anticorrelated perturbed gradient descent). We use our results to develop new, effective matrix factorizations for differentially private optimization, and highlight the benefits of these factorizations theoretically and empirically.

----

## [1552] A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions

**Authors**: *David Loiseaux, Mathieu Carrière, Andrew J. Blumberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html)

**Abstract**:

Topological data analysis (TDA) is an area of data science that focuses on using invariants from algebraic topology to provide multiscale shape descriptors for geometric data sets such as point clouds. One of the most important such descriptors is persistent homology, which encodes the change in shape as a filtration parameter changes; a typical parameter is the feature scale. For many data sets, it is useful to simultaneously vary multiple filtration parameters, for example feature scale and density. While the theoretical properties of single parameter persistent homology are well understood, less is known about the multiparameter case.  A central question is the problem of representing multiparameter persistent homology by elements of a vector space for integration with standard machine learning algorithms. Existing approaches to this problem either ignore most of the multiparameter information to reduce to the one-parameter case or are heuristic and potentially unstable in the face of noise. In this article, we introduce a new general representation framework that leverages recent results on decompositions of multiparameter persistent homology. This framework is rich in information, fast to compute, and encompasses previous approaches. Moreover, we establish theoretical stability guarantees under this framework as well as efficient algorithms for practical computation, making this framework an applicable and versatile tool for analyzing geometric and point cloud data. We validate our stability results and algorithms with numerical experiments that demonstrate statistical convergence, prediction accuracy, and fast running times on several real data sets.

----

## [1553] Objaverse-XL: A Universe of 10M+ 3D Objects

**Authors**: *Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, Ali Farhadi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70364304877b5e767de4e9a2a511be0c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/70364304877b5e767de4e9a2a511be0c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Natural language processing and 2D vision models have attained remarkable proficiency on many tasks primarily by escalating the scale of training data. However, 3D vision tasks have not seen the same progress, in part due to the challenges of acquiring high-quality 3D data. In this work, we present Objaverse-XL, a dataset of over 10 million 3D objects. Our compilation comprises deduplicated 3D objects from a diverse set of sources, including manually designed objects, photogrammetry scans of landmarks and everyday items, and professional scans of historic and antique artifacts. Representing the largest scale and diversity in the realm of 3D datasets, Objaverse-XL enables significant new possibilities for 3D vision. Our experiments demonstrate the vast improvements enabled with the scale provided by Objaverse-XL. We show that by training Zero123 on novel view synthesis, utilizing over 100 million multi-view rendered images, we achieve strong zero-shot generalization abilities. We hope that releasing Objaverse-XL will enable further innovations in the field of 3D vision at scale.

----

## [1554] Should We Learn Most Likely Functions or Parameters?

**Authors**: *Shikai Qiu, Tim G. J. Rudner, Sanyam Kapoor, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/703f727ec10190b2fddcf8e24f52df48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/703f727ec10190b2fddcf8e24f52df48-Abstract-Conference.html)

**Abstract**:

Standard regularized training procedures correspond to maximizing a posterior distribution over parameters, known as maximum a posteriori (MAP) estimation. However, model parameters are of interest only insomuch as they combine with the functional form of a model to provide a function that can make good predictions. Moreover, the most likely parameters under the parameter posterior do not generally correspond to the most likely function induced by the parameter posterior. In fact, we can re-parametrize a model such that any setting of parameters can maximize the parameter posterior. As an alternative, we investigate the benefits and drawbacks of directly estimating the most likely function implied by the model and the data. We show that this procedure leads to pathological solutions when using neural networks and prove conditions under which the procedure is well-behaved, as well as a scalable approximation. Under these conditions, we find that function-space MAP estimation can lead to flatter minima, better generalization, and improved robustness to overfitting.

----

## [1555] Geometry-Informed Neural Operator for Large-Scale 3D PDEs

**Authors**: *Zongyi Li, Nikola B. Kovachki, Christopher B. Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Animashree Anandkumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)

**Abstract**:

We propose the geometry-informed neural operator (GINO), a highly efficient approach for learning the solution operator of large-scale partial differential equations with varying geometries. GINO uses a signed distance function (SDF) representation of the input shape and neural operators based on graph and Fourier architectures to learn the solution operator. The graph neural operator handles irregular grids and transforms them into and from regular latent grids on which Fourier neural operator can be efficiently applied. We provide an efficient implementation of GINO using an optimized hashing approach, which allows efficient learning  in a shared, compressed latent space with reduced computation and memory costs. GINO is discretization-invariant, meaning the trained model can be applied to arbitrary discretizations of the continuous domain and applies to any shape or resolution.  To empirically validate the performance of our method on large-scale simulation, we generate the industry-standard aerodynamics dataset of 3D vehicle geometries with Reynolds numbers as high as five million. For this large-scale 3D fluid simulation, numerical methods are expensive to compute surface pressure. We successfully trained GINO to predict the pressure on car surfaces using only five hundred data points. The cost-accuracy experiments show a 26,000x speed-up compared to optimized GPU-based computational fluid dynamics (CFD) simulators on computing the drag coefficient. When tested on new combinations of geometries and boundary conditions (inlet velocities), GINO obtains a one-fourth reduction in error rate compared to deep neural network approaches.

----

## [1556] Differentially Private Image Classification by Learning Priors from Random Processes

**Authors**: *Xinyu Tang, Ashwinee Panda, Vikash Sehwag, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7058bc192a37f5e5a57398887b05f6f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7058bc192a37f5e5a57398887b05f6f6-Abstract-Conference.html)

**Abstract**:

In privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) performs worse than SGD due to per-sample gradient clipping and noise addition.A recent focus in private learning research is improving the performance of DP-SGD on private data by incorporating priors that are learned on real-world public data.In this work, we explore how we can improve the privacy-utility tradeoff of DP-SGD by learning priors from images generated by random processes and transferring these priors to private data. We propose DP-RandP, a three-phase approach. We attain new state-of-the-art accuracy when training from scratch on CIFAR10, CIFAR100, MedMNIST and ImageNet for a range of privacy budgets $\\varepsilon \\in [1, 8]$. In particular, we improve the previous best reported accuracy on CIFAR10 from $60.6 \\%$ to $72.3 \\%$ for $\\varepsilon=1$.

----

## [1557] ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

**Authors**: *Mohammad Reza Taesiri, Giang Nguyen, Sarra Habchi, Cor-Paul Bezemer, Anh Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/706390d6f9208b03bc54f97ac3cfe99e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/706390d6f9208b03bc54f97ac3cfe99e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first zoom to the most discriminative region in the image and then extract features from there to predict image labels, discarding the rest of the image. Studying six popular networks ranging from AlexNet to CLIP, we find that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we  uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zooming, we propose a test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions.Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art (SOTA) TTA method. We introduce ImageNet-Hard, a new benchmark that challenges SOTA classifiers including large vision-language models even when optimal zooming is allowed.

----

## [1558] NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA

**Authors**: *Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, Hyunwoo J. Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/707a2d58641b2192203b4bf4c532cfe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/707a2d58641b2192203b4bf4c532cfe1-Abstract-Conference.html)

**Abstract**:

Multi-hop Knowledge Graph Question Answering (KGQA) is a task that involves retrieving nodes from a knowledge graph (KG) to  answer natural language questions. Recent GNN-based approaches formulate this task as a KG path searching problem, where messages are sequentially propagated from the seed node towards the answer nodes. However, these messages are past-oriented, and they do not consider the full KG context. To make matters worse, KG nodes often represent pronoun entities and are sometimes encrypted, being uninformative in selecting between paths. To address these problems, we propose Neural Tree Search (NuTrea), a tree search-based GNN model that incorporates the broader KG context. Our model adopts a message-passing scheme that probes the unreached subtree regions to boost the past-oriented embeddings. In addition, we introduce the Relation Frequency-Inverse Entity Frequency (RF-IEF) node embedding that considers the global KG context to better characterize ambiguous KG nodes. The general effectiveness of our approach is demonstrated through experiments on three major multi-hop KGQA benchmark datasets, and our extensive analyses further validate its expressiveness and robustness. Overall, NuTrea provides a powerful means to query the KG with complex natural language questions. Code is available at https://github.com/mlvlab/NuTrea.

----

## [1559] Anonymous Learning via Look-Alike Clustering: A Precise Analysis of Model Generalization

**Authors**: *Adel Javanmard, Vahab Mirrokni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70899a5d74f83317c78f1a7d413d1baa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70899a5d74f83317c78f1a7d413d1baa-Abstract-Conference.html)

**Abstract**:

While personalized recommendations systems have become increasingly popular, ensuring user data protection remains a top concern in the development of these learning systems. A common approach to enhancing privacy involves training models using anonymous data rather than individual data. In this paper, we explore a natural technique called "look-alike clustering", which involves replacing sensitive features of individuals with the cluster's average values. We provide a precise analysis of how training models using anonymous cluster centers affects their generalization capabilities. We focus on an asymptotic regime where the size of the training set grows in proportion to the features dimension. Our analysis is based on the  Convex Gaussian Minimax Theorem (CGMT) and allows us to theoretically understand the role of different model components on the generalization error. In addition, we demonstrate that in certain high-dimensional regimes, training over anonymous cluster centers acts as a regularization and improves generalization error of the trained models. Finally, we corroborate our asymptotic theory with finite-sample numerical experiments where we observe a perfect match when the sample size is only of order of a few hundreds.

----

## [1560] Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

**Authors**: *Jinpeng Chen, Runmin Cong, Yuxuan Luo, Horace H. s. Ip, Sam Kwong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html)

**Abstract**:

Existing class-incremental semantic segmentation (CISS) methods mainly tackle catastrophic forgetting and background shift, but often overlook another crucial issue. In CISS, each step focuses on different foreground classes, and the training set for a single step only includes images containing pixels of the current foreground classes, excluding images without them. This leads to an overrepresentation of these foreground classes in the single-step training set, causing the classification biased towards these classes. To address this issue, we present STAR, which preserves the main characteristics of each past class by storing a compact prototype and necessary statistical data, and aligns the class distribution of single-step training samples with the complete dataset by replaying these prototypes and repeating background pixels with appropriate frequency. Compared to the previous works that replay raw images, our method saves over 100 times the storage while achieving better performance. Moreover, STAR incorporates an old-class features maintaining (OCFM) loss, keeping old-class features unchanged while preserving sufficient plasticity for learning new classes. Furthermore, a similarity-aware discriminative (SAD) loss is employed to specifically enhance the feature diversity between similar old-new class pairs. Experiments on two public datasets, Pascal VOC 2012 and ADE20K, reveal that our model surpasses all previous state-of-the-art methods.

----

## [1561] Skill-it! A data-driven skills framework for understanding and training language models

**Authors**: *Mayee F. Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)

**Abstract**:

The quality of training data impacts the performance of pre-trained large language models (LMs). Given a fixed budget of tokens, we study how to best select data that leads to good downstream model performance across tasks. We develop a new framework based on a simple hypothesis: just as humans acquire interdependent skills in a deliberate order, language models also follow a natural order when learning a set of skills from their training data. If such an order exists, it can be utilized for improved understanding of LMs and for data-efficient training. Using this intuition, our framework formalizes the notion of a skill and of an ordered set of skills in terms of the associated data. First, using both synthetic and real data, we demonstrate that these ordered skill sets exist, and that their existence enables more advanced skills to be learned with less data when we train on their prerequisite skills. Second, using our proposed framework, we introduce an online data sampling algorithm, Skill-It, over mixtures of skills for both continual pre-training and fine-tuning regimes, where the objective is to efficiently learn multiple skills in the former and an individual skill in the latter. On the LEGO synthetic in the continual pre-training setting, Skill-It obtains 37.5 points higher accuracy than random sampling. On the Natural Instructions dataset in the fine-tuning setting, Skill-It reduces the validation loss on the target skill by 13.6% versus training on data associated with the target skill itself. We apply our skills framework on the RedPajama dataset to continually pre-train a 3B-parameter LM, achieving higher accuracy on the LM Evaluation Harness with 1B tokens than the baseline approach of sampling uniformly over data sources with 3B tokens.

----

## [1562] Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation

**Authors**: *Stefania Ionescu, Yuhao Du, Kenneth Joseph, Ancsa Hannak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70c6d82d27cd96c501c4def4803d5782-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70c6d82d27cd96c501c4def4803d5782-Abstract-Conference.html)

**Abstract**:

Two-sided matching markets have long existed to pair agents in the absence of regulated exchanges.  A common example is school choice, where a matching mechanism uses student and school preferences to assign students to schools. In such settings, forming preferences is both difficult and critical. Prior work has suggested various prediction mechanisms that help agents make decisions about their preferences. Although often deployed together, these matching and prediction mechanisms are almost always analyzed separately. The present work shows that at the intersection of the two lies a previously unexplored type of strategic behavior: agents returning to the market (e.g., schools) can attack future predictions by interacting short-term non-optimally with their matches. Here, we first introduce this type of strategic behavior, which we call an adversarial interaction attack. Next, we construct a formal economic model that captures the feedback loop between prediction mechanisms designed to assist agents and the matching mechanism used to pair them. Finally, in a simplified setting, we prove that returning agents can benefit from using adversarial interaction attacks and gain progressively more as the trust in and accuracy of predictions increases. We also show that this attack increases inequality in the student population.

----

## [1563] Sample Complexity of Forecast Aggregation

**Authors**: *Tao Lin, Yiling Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html)

**Abstract**:

We consider a Bayesian forecast aggregation model where $n$ experts, after observing private signals about an unknown binary event, report their posterior beliefs about the event to a principal, who then aggregates the reports into a single prediction for the event. The signals of the experts and the outcome of the event follow a joint distribution that is unknown to the principal, but the principal has access to i.i.d. "samples" from the distribution, where each sample is a tuple of the experts' reports (not signals) and the realization of the event. Using these samples, the principal aims to find an $\varepsilon$-approximately optimal aggregator, where optimality is measured in terms of the expected squared distance between the aggregated prediction and the realization of the event. We show that the sample complexity of this problem is at least $\tilde \Omega(m^{n-2} / \varepsilon)$ for arbitrary discrete distributions, where $m$ is the size of each expert's signal space.  This sample complexity grows exponentially in the number of experts $n$. But, if the experts' signals are independent conditioned on the realization of the event, then the sample complexity is significantly reduced, to $\tilde O(1 / \varepsilon^2)$, which does not depend on $n$. Our results can be generalized to non-binary events.  The proof of our results uses a reduction from the distribution learning problem and reveals the fact that forecast aggregation is almost as difficult as distribution learning.

----

## [1564] Learning to Influence Human Behavior with Offline Reinforcement Learning

**Authors**: *Joey Hong, Sergey Levine, Anca D. Dragan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70f286e0fc977c0a3a64ef96849c8d7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70f286e0fc977c0a3a64ef96849c8d7d-Abstract-Conference.html)

**Abstract**:

When interacting with people, AI agents do not just influence the state of the world -- they also influence the actions people take in response to the agent, and even their underlying intentions and strategies. Accounting for and leveraging this influence has mostly been studied in settings where it is sufficient to assume that human behavior is near-optimal: competitive games, or general-sum settings like autonomous driving alongside human drivers. Instead, we focus on influence in settings where there is a need to capture human suboptimality. For instance, imagine a collaborative task in which, due either to cognitive biases or lack of information, people do not perform very well -- how could an agent influence them towards more optimal behavior? Assuming near-optimal human behavior will not work here, and so the agent needs to learn from real human data. But experimenting online with humans is potentially unsafe, and creating a high-fidelity simulator of the environment is often impractical. Hence, we  focus on learning from an offline dataset of human-human interactions. Our observation is that offline reinforcement learning (RL) can learn to effectively influence suboptimal humans by extending and combining elements of observed human-human behavior. We demonstrate that offline RL can solve two challenges with effective influence. First, we show that by learning from a dataset of suboptimal human-human interaction on a variety of tasks -- none of which contains examples of successful influence -- an agent can learn influence strategies to steer humans towards better performance even on new tasks. Second, we show that by also modeling and conditioning on human behavior, offline RL can learn to affect not just the human's actions but also their underlying strategy, and adapt to changes in their strategy.

----

## [1565] Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier

**Authors**: *Yuling Yao, Justin Domke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7103cd82de95a7b30983fcf74ba499ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7103cd82de95a7b30983fcf74ba499ac-Abstract-Conference.html)

**Abstract**:

To check the accuracy of Bayesian computations, it is common to use rank-based simulation-based calibration (SBC). However, SBC has drawbacks: The test statistic is somewhat ad-hoc, interactions are difficult to examine, multiple testing is a challenge, and the resulting p-value is not a divergence metric. We propose to replace the marginal rank test with a flexible classification approach that learns test statistics from data. This measure typically has a higher statistical power than the SBC test and returns an interpretable divergence measure of miscalibration, computed from classification accuracy. This approach can be used with different data generating processes to address simulation-based inference or traditional inference methods like Markov chain Monte Carlo or variational inference. We illustrate an automated implementation using neural networks and statistically-inspired features, and validate the method with numerical and real data experiments.

----

## [1566] Epidemic Learning: Boosting Decentralized Learning with Randomized Communication

**Authors**: *Martijn de Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html)

**Abstract**:

We present Epidemic Learning (EL), a simple yet powerful decentralized learning (DL) algorithm that leverages changing communication topologies to achieve faster model convergence compared to conventional DL approaches. At each round of EL, each node sends its model updates to a random sample of $s$ other nodes (in a system of $n$ nodes). We provide an extensive theoretical analysis of EL, demonstrating that its changing topology culminates in superior convergence properties compared to the state-of-the-art (static and dynamic) topologies. Considering smooth non-convex loss functions, the number of transient iterations for EL, i.e., the rounds required to achieve asymptotic linear speedup, is in $O(n^3/s^2)$ which outperforms the best-known bound $O(n^3)$ by a factor of $s^2$, indicating the benefit of randomized communication for DL. We empirically evaluate EL in a 96-node network and compare its performance with state-of-the-art DL approaches. Our results illustrate that EL converges up to  $ 1.7\times$ quicker than baseline DL algorithms and attains $2.2 $\% higher accuracy for the same communication volume.

----

## [1567] Global Identifiability of 𝓁1-based Dictionary Learning via Matrix Volume Optimization

**Authors**: *Jingzhou Hu, Kejun Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/717b9fd2ede6b8a9971a296d5179df89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/717b9fd2ede6b8a9971a296d5179df89-Abstract-Conference.html)

**Abstract**:

We propose a novel formulation for dictionary learning that minimizes the determinant of the dictionary matrix, also known as its volume, subject to the constraint that each row of the sparse coefficient matrix has unit $\ell_1$ norm. The main motivation for the proposed formulation is that it provides global identifiability guarantee of the groundtruth dictionary and sparse coefficient matrices, up to the inherent and inconsequential permutation and scaling ambiguity, if a set of vectors obtained from the coefficient matrix lies inside the $\ell_\infty$ norm ball but contains the $\ell_2$ norm ball in their convex hull. Unlike existing work on identifiability of dictionary learning, our result is global, meaning that a globally optimal solution to our proposed formulation has to be a permuted and rescaled version of the groundtruth factors. Another major improvement in our result is that there is no additional assumption on the dictionary matrix other than it is nonsingular, unlike most other work that require the atoms of the dictionary to be mutually incoherent. We also provide a probabilistic analysis and show that if the sparse coefficient matrix is generated from the widely adopted Bernoulli-Gaussian model, then it is globally identifiable if the sample size is bigger than a constant times $k\log k$, where $k$ is the number atoms in the dictionary, with overwhelming probability. The bound is essentially the same as those local identifiability results, but we show that it is also global. Finally, we propose algorithms to solve the new proposed formulation, specifically one based on the linearized-ADMM with efficient per-iteration updates. The proposed algorithms exhibit surprisingly effective performance in correctly and efficiently recovering the dictionary, as demonstrated in the numerical experiments.

----

## [1568] Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization

**Authors**: *Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7183f4fc87598f6c6e947b96714acbd6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7183f4fc87598f6c6e947b96714acbd6-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) face the challenges in fine-tuning and deployment due to their high memory demands and computational costs. While parameter-efficient fine-tuning (PEFT) methods aim to reduce the memory usage of the optimizer state during fine-tuning, the inherent size of pre-trained LLM weights continues to be a pressing concern.  Even though quantization techniques are widely proposed to ease memory demands and accelerate LLM inference, most of these techniques are geared towards the deployment phase.To bridge this gap, this paper presents Parameter-Efficient and Quantization-aware Adaptation (PEQA) â€“ a simple yet effective method that combines the advantages of PEFT with quantized LLMs. By updating solely the quantization scales, PEQA can be directly applied to quantized LLMs, ensuring seamless task transitions. Parallel to existing PEFT methods, PEQA significantly reduces the memory overhead associated with the optimizer state. Furthermore, it leverages the advantages of quantization to substantially reduce model sizes. Even after fine-tuning, the quantization structure of a PEQA-tuned LLM remains intact, allowing for accelerated inference on the deployment stage.We employ PEQA-tuning for task-specific adaptation on LLMs with up to $65$ billion parameters. To assess the logical reasoning and language comprehension of PEQA-tuned LLMs, we fine-tune low-bit quantized LLMs using a instruction dataset. Our results show that even when LLMs are quantized to below 4-bit precision, their capabilities in language modeling, few-shot in-context learning, and comprehension can be resiliently restored to (or even improved over) their full-precision original performances with PEQA.

----

## [1569] Corruption-Robust Offline Reinforcement Learning with General Function Approximation

**Authors**: *Chenlu Ye, Rui Yang, Quanquan Gu, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71b52a5b3fe2e9303433a174b60e160d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71b52a5b3fe2e9303433a174b60e160d-Abstract-Conference.html)

**Abstract**:

We investigate the problem of corruption robustness in offline reinforcement learning (RL) with general function approximation, where an adversary can corrupt each sample in the offline dataset, and the corruption level $\zeta\geq0$ quantifies the cumulative corruption amount over $n$ episodes and $H$ steps. Our goal is to find a policy that is robust to such corruption and minimizes the suboptimality gap with respect to the optimal policy for the uncorrupted Markov decision processes (MDPs). Drawing inspiration from the uncertainty-weighting technique from the robust online RL setting \citep{he2022nearly,ye2022corruptionrobust}, we design a new uncertainty weight iteration procedure to efficiently compute on batched samples and propose a corruption-robust algorithm for offline RL. Notably, under the assumption of single policy coverage and the knowledge of $\zeta$, our proposed algorithm achieves a suboptimality bound that is worsened by an additive factor of $\mathcal O(\zeta \cdot (\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H))^{1/2} (C(\hat{\mathcal F},\mu))^{-1/2} n^{-1})$ due to the corruption. Here $\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H)$ is the coverage coefficient that depends on the regularization parameter $\lambda$, the confidence set $\hat{\mathcal F}$, and the dataset $\mathcal Z_n^H$, and $C(\hat{\mathcal F},\mu)$ is a coefficient that depends on $\hat{\mathcal F}$ and the underlying data distribution $\mu$. When specialized to linear MDPs, the corruption-dependent error term reduces to $\mathcal O(\zeta d n^{-1})$ with $d$ being the dimension of the feature map, which matches the existing lower bound for corrupted linear MDPs. This suggests that our analysis is tight in terms of the corruption-dependent term.

----

## [1570] Training Fully Connected Neural Networks is ∃R-Complete

**Authors**: *Daniel Bertschinger, Christoph Hertrich, Paul Jungeblut, Tillmann Miltzow, Simon Weber*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html)

**Abstract**:

We consider the algorithmic problem of finding the optimal weights and biases for a two-layer fully connected neural network to fit a given set of data points, also known as empirical risk minimization. We show that the problem is $\exists\mathbb{R}$-complete. This complexity class can be defined as the set of algorithmic problems that are polynomial-time equivalent to finding real roots of a multivariate polynomial with integer coefficients. Furthermore, we show that arbitrary algebraic numbers are required as weights to be able to train some instances to optimality, even if all data points are rational. Our result already applies to fully connected instances with two inputs, two outputs, and one hidden layer of ReLU neurons. Thereby, we strengthen a result by Abrahamsen, Kleist and Miltzow [NeurIPS 2021]. A consequence of this is that a combinatorial search algorithm like the one by Arora, Basu, Mianjy and Mukherjee [ICLR 2018] is impossible for networks with more than one output dimension, unless $\text{NP} = \exists\mathbb{R}$.

----

## [1571] Uncertainty Estimation for Safety-critical Scene Segmentation via Fine-grained Reward Maximization

**Authors**: *Hongzheng Yang, Cheng Chen, Yueyao Chen, Markus Scheppach, Hon-Chi Yip, Qi Dou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html)

**Abstract**:

Uncertainty estimation plays an important role for future reliable deployment of deep segmentation models in safety-critical scenarios such as medical applications. However, existing methods for uncertainty estimation have been limited by the lack of explicit guidance for calibrating the prediction risk and model confidence. In this work, we propose a novel fine-grained reward maximization (FGRM) framework, to address uncertainty estimation by directly utilizing an uncertainty metric related reward function with a reinforcement learning based model tuning algorithm. This would benefit the model uncertainty estimation with direct optimization guidance for model calibration. Specifically, our method designs a new uncertainty estimation reward function using the calibration metric, which is maximized to fine-tune an evidential learning pre-trained segmentation model for calibrating prediction risk. Importantly, we innovate an effective fine-grained parameter update scheme, which imposes fine-grained reward-weighting of each network parameter according to the parameter importance quantified by the fisher information matrix. To the best of our knowledge, this is the first work exploring reward optimization for model uncertainty estimation in safety-critical vision tasks. The effectiveness of our method is demonstrated on two large safety-critical surgical scene segmentation datasets under two different uncertainty estimation settings. With real-time one forward pass at inference, our method outperforms state-of-the-art methods by a clear margin on all the calibration metrics of uncertainty estimation, while maintaining a high task accuracy for the segmentation results. Code is available at https://github.com/med-air/FGRM.

----

## [1572] GLIME: General, Stable and Local LIME Explanation

**Authors**: *Zeren Tan, Yang Tian, Jian Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71ed042903ed67c7f6355e5dd0539eec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71ed042903ed67c7f6355e5dd0539eec-Abstract-Conference.html)

**Abstract**:

As black-box machine learning models become more complex and are applied in high-stakes settings, the need for providing explanations for their predictions becomes crucial. Although Local Interpretable Model-agnostic Explanations (LIME) \cite{ribeiro2016should} is a widely adopted method for understanding model behavior, it suffers from instability with respect to random seeds \cite{zafar2019dlime, shankaranarayana2019alime, bansal2020sam} and exhibits low local fidelity (i.e., how the explanation explains model's local behaviors) \cite{rahnama2019study, laugel2018defining}. Our study demonstrates that this instability is caused by small sample weights, resulting in the dominance of regularization and slow convergence. Additionally, LIME's sampling approach is non-local and biased towards the reference, leading to diminished local fidelity and instability to references. To address these challenges, we propose \textsc{Glime}, an enhanced framework that extends LIME and unifies several previous methods. Within the \textsc{Glime} framework, we derive an equivalent formulation of LIME that achieves significantly faster convergence and improved stability. By employing a local and unbiased sampling distribution, \textsc{Glime} generates explanations with higher local fidelity compared to LIME, while being independent of the reference choice. Moreover, \textsc{Glime} offers users the flexibility to choose sampling distribution based on their specific scenarios.

----

## [1573] Efficient Symbolic Policy Learning with Differentiable Symbolic Expression

**Authors**: *Jiaming Guo, Rui Zhang, Shaohui Peng, Qi Yi, Xing Hu, Ruizhi Chen, Zidong Du, Xishan Zhang, Ling Li, Qi Guo, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7207ffb9888068c0ee13ae3be023cada-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7207ffb9888068c0ee13ae3be023cada-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (DRL) has led to a wide range of advances in sequential decision-making tasks. However, the complexity of neural network policies makes it difficult to understand and deploy with limited computational resources. Currently, employing compact symbolic expressions as symbolic policies is a promising strategy to obtain simple and interpretable policies. Previous symbolic policy methods usually involve complex training processes and pre-trained neural network policies, which are inefficient and limit the application of symbolic policies. In this paper, we propose an efficient gradient-based learning method named Efficient Symbolic Policy Learning (ESPL) that learns the symbolic policy from scratch in an end-to-end way. We introduce a symbolic network as the search space and employ a path selector to find the compact symbolic policy. By doing so we represent the policy with a differentiable symbolic expression and train it in an off-policy manner which further improves the efficiency. In addition, in contrast with previous symbolic policies which only work in single-task RL because of complexity, we expand ESPL on meta-RL to generate symbolic policies for unseen tasks. Experimentally, we show that our approach generates symbolic policies with higher performance and greatly improves data efficiency for single-task RL. In meta-RL, we demonstrate that compared with neural network policies the proposed symbolic policy achieves higher performance and efficiency and shows the potential to be interpretable.

----

## [1574] PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization

**Authors**: *Jiancong Xiao, Ruoyu Sun, Zhi-Quan Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/720991812855c99df50bc8b36966cd81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/720991812855c99df50bc8b36966cd81-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.

----

## [1575] Neural Graph Generation from Graph Statistics

**Authors**: *Kiarash Zahirnia, Yaochen Hu, Mark Coates, Oliver Schulte*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72153267883fbcafdb6e4662382696c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72153267883fbcafdb6e4662382696c5-Abstract-Conference.html)

**Abstract**:

We describe a new setting for learning a deep graph generative model (GGM) from aggregate graph statistics, rather than from the graph adjacency matrix. Matching the statistics of observed training graphs is the main approach for learning traditional GGMs (e.g, BTER, Chung-Lu, and  Erdos-Renyi models). Privacy researchers have proposed learning from graph statistics as a way to protect privacy. We develop an architecture for training a deep GGM  to match statistics while preserving local differential privacy guarantees. Empirical evaluation on 8 datasets indicates that our deep GGM model generates more realistic graphs than the traditional GGMs when both are learned from graph statistics only. We also benchmark our deep GGM trained on statistics only, against state-of-the-art deep GGM models that are trained on the entire adjacency matrix. The results show that graph statistics are often sufficient to build a competitive deep GGM that generates realistic graphs while protecting local privacy.

----

## [1576] DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction

**Authors**: *Mohammadreza Pourreza, Davood Rafiei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html)

**Abstract**:

There is currently a significant gap between the performance of fine-tuned models and prompting approaches using Large Language Models (LLMs) on the challenging task of text-to-SQL, as evaluated on datasets such as Spider. To improve the performance of LLMs in the reasoning process, we study how decomposing the task into smaller sub-tasks can be effective. In particular, we show that breaking down the generation problem into sub-problems and feeding the solutions of those sub-problems into LLMs can be an effective approach for significantly improving their performance.  Our experiments with three LLMs show that this approach consistently improves their simple few-shot performance by roughly 10%, pushing the accuracy of LLMs towards SOTA or surpassing it. On the holdout test set of Spider, the SOTA, in terms of execution accuracy, was 79.9 and the new SOTA at the time of  this writing using our approach is 85.3. Our approach with in-context learning beats many heavily fine-tuned models by at least 5%. Additionally, when evaluated on the BIRD benchmark, our approach achieved an execution accuracy of 55.9%, setting a new SOTA on its holdout test set.

----

## [1577] Scaling Up Differentially Private LASSO Regularized Logistic Regression via Faster Frank-Wolfe Iterations

**Authors**: *Edward Raff, Amol Khanna, Fred Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72235260ae8d57ac42638a26d3b7d089-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72235260ae8d57ac42638a26d3b7d089-Abstract-Conference.html)

**Abstract**:

To the best of our knowledge, there are no methods today for training differentially private regression models on sparse input data. To remedy this, we adapt the Frank-Wolfe algorithm for $L_1$ penalized linear regression to be aware of sparse inputs and to use them effectively. In doing so, we reduce the training time of the algorithm from $\mathcal{O}( T D S + T N S)$ to $\mathcal{O}(N S + T \sqrt{D} \log{D}  + T S^2)$, where $T$ is the number of iterations and a sparsity rate $S$ of a dataset with $N$ rows and $D$ features. Our results demonstrate that this procedure can reduce runtime by a factor of up to $2,200\times$, depending on the value of the privacy parameter $\epsilon$ and the sparsity of the dataset.

----

## [1578] Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback

**Authors**: *Yang Cai, Haipeng Luo, Chen-Yu Wei, Weiqiang Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/722fcbc1a6667f2075d75ea79a1b3552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/722fcbc1a6667f2075d75ea79a1b3552-Abstract-Conference.html)

**Abstract**:

We revisit the problem of learning in two-player zero-sum Markov games, focusing on developing an algorithm that is *uncoupled*, *convergent*, and *rational*, with non-asymptotic convergence rates to Nash equilibrium. We start from the case of stateless matrix game with bandit feedback as a warm-up, showing an $\tilde{\mathcal{O}}(t^{-\frac{1}{8}})$ last-iterate convergence rate. To the best of our knowledge, this is the first result that obtains finite last-iterate convergence rate given access to only bandit feedback. We extend our result to the case of irreducible Markov games, providing a last-iterate convergence rate of $\tilde{\mathcal{O}}(t^{-\frac{1}{9+\varepsilon}})$ for any $\varepsilon>0$. Finally, we study Markov games without any assumptions on the dynamics, and show a *path convergence* rate, a new notion of convergence we defined, of $\tilde{\mathcal{O}}(t^{-\frac{1}{10}})$. Our algorithm removes the synchronization and prior knowledge requirement of Wei et al. (2021), which pursued the same goals as us for irreducible Markov games. Our algorithm is related to Chen et al. (2021) and Cen et al. (2021)'s and also builds on the entropy regularization technique. However, we remove their requirement of communications on the entropy values, making our algorithm entirely uncoupled.

----

## [1579] Deductive Verification of Chain-of-Thought Reasoning

**Authors**: *Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) significantly benefit from Chain-of-thought (CoT) prompting in performing various reasoning tasks. While CoT allows models to produce more comprehensive reasoning processes, its emphasis on intermediate reasoning steps can inadvertently introduce hallucinations and accumulated errors, thereby limiting modelsâ€™ ability to solve complex reasoning tasks. Inspired by how humans engage in careful and meticulous deductive logical reasoning processes to solve tasks, we seek to enable language models to perform explicit and rigorous deductive reasoning, and also ensure the trustworthiness of their reasoning process through self-verification. However, directly verifying the validity of an entire deductive reasoning process is challenging, even with advanced models like ChatGPT. In light of this, we propose to decompose a reasoning verification process into a series of step-by-step subprocesses, each only receiving their necessary context and premises. To facilitate this procedure, we propose Natural Program, a natural language-based deductive reasoning format. Our approach enables models to generate precise reasoning steps where subsequent steps are more rigorously grounded on prior steps. It also empowers language models to carry out reasoning self-verification in a step-by-step manner. By integrating this verification process into each deductive reasoning stage, we significantly enhance the rigor and trustfulness of generated reasoning steps. Along this process, we also improve the answer correctness on complex reasoning tasks.

----

## [1580] Rigorous Runtime Analysis of MOEA/D for Solving Multi-Objective Minimum Weight Base Problems

**Authors**: *Anh Viet Do, Aneta Neumann, Frank Neumann, Andrew M. Sutton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html)

**Abstract**:

We study the multi-objective minimum weight base problem, an abstraction of classical NP-hard combinatorial problems such as the multi-objective minimum spanning tree problem. We prove some important properties of the convex hull of the non-dominated front, such as its approximation quality and an upper bound on the number of extreme points. Using these properties, we give the first run-time analysis of the MOEA/D algorithm for this problem, an evolutionary algorithm that effectively optimizes by decomposing the objectives into single-objective components. We show that the MOEA/D, given an appropriate decomposition setting, finds all extreme points within expected fixed-parameter polynomial time, in the oracle model. Experiments are conducted on random bi-objective minimum spanning tree instances, and the results agree with our theoretical findings. Furthermore, compared with a previously studied evolutionary algorithm for the problem GSEMO, MOEA/D finds all extreme points much faster across all instances.

----

## [1581] Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics

**Authors**: *Mathias Schreiner, Ole Winther, Simon Olsson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html)

**Abstract**:

Computing properties of molecular systems rely on estimating expectations of the (unnormalized) Boltzmann distribution. Molecular dynamics (MD) is a broadly adopted technique to approximate such quantities. However, stable simulations rely on very small integration time-steps ($10^{-15}\,\mathrm{s}$), whereas convergence of some moments, e.g. binding free energy or rates, might rely on sampling processes on time-scales as long as $10^{-1}\, \mathrm{s}$, and these simulations must be repeated for every molecular system independently. Here, we present Implicit Transfer Operator (ITO) Learning, a framework to learn surrogates of the simulation process with multiple time-resolutions. We implement ITO with denoising diffusion probabilistic models with a new SE(3) equivariant architecture and show the resulting models can generate self-consistent stochastic dynamics across multiple time-scales, even when the system is only partially observed. Finally, we present a coarse-grained CG-SE3-ITO model which can quantitatively model all-atom molecular dynamics using only coarse molecular representations. As such, ITO provides an important step towards multiple time- and space-resolution acceleration of MD. Code is available at \href{https://github.com/olsson-group/ito}{https://github.com/olsson-group/ito}.

----

## [1582] Causal de Finetti: On the Identification of Invariant Causal Structure in Exchangeable Data

**Authors**: *Siyuan Guo, Viktor Tóth, Bernhard Schölkopf, Ferenc Huszar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7279908471a7dd4898d2715f7c6a7413-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7279908471a7dd4898d2715f7c6a7413-Abstract-Conference.html)

**Abstract**:

Constraint-based causal discovery methods leverage conditional independence tests to infer causal relationships in a wide variety of applications. Just as the majority of machine learning methods, existing work focuses on studying $\textit{independent and identically distributed}$ data. However, it is known that even with infinite $i.i.d.\$ data, constraint-based methods can only identify causal structures up to broad Markov equivalence classes, posing a fundamental limitation for causal discovery. In this work, we observe that exchangeable data contains richer conditional independence structure than $i.i.d.\$ data, and show how the richer structure can be leveraged for causal discovery. We first present causal de Finetti theorems, which state that exchangeable distributions with certain non-trivial conditional independences can always be represented as $\textit{independent causal mechanism (ICM)}$ generative processes. We then present our main identifiability theorem, which shows that given data from an ICM generative process, its unique causal structure can be identified through performing conditional independence tests. We finally develop a causal discovery algorithm and demonstrate its applicability to inferring causal relationships from multi-environment data.

----

## [1583] Batch Bayesian Optimization For Replicable Experimental Design

**Authors**: *Zhongxiang Dai, Quoc Phong Nguyen, Sebastian Tay, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html)

**Abstract**:

Many real-world experimental design problems (a) evaluate multiple experimental conditions in parallel and (b) replicate each condition multiple times due to large and heteroscedastic observation noise. Given a fixed total budget, this naturally induces a trade-off between evaluating more unique conditions while replicating each of them fewer times vs. evaluating fewer unique conditions and replicating each more times. Moreover, in these problems, practitioners may be risk-averse and hence prefer an input with both good average performance and small variability. To tackle both challenges, we propose the Batch Thompson Sampling for Replicable Experimental Design (BTS-RED) framework, which encompasses three algorithms. Our BTS-RED-Known and BTS-RED-Unknown algorithms, for, respectively, known and unknown noise variance, choose the number of replications adaptively rather than deterministically such that an input with a larger noise variance is replicated more times. As a result, despite the noise heteroscedasticity, both algorithms enjoy a theoretical guarantee and are asymptotically no-regret. Our Mean-Var-BTS-RED algorithm aims at risk-averse optimization and is also asymptotically no-regret. We also show the effectiveness of our algorithms in two practical real-world applications: precision agriculture and AutoML.

----

## [1584] Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning

**Authors**: *Siming Lan, Rui Zhang, Qi Yi, Jiaming Guo, Shaohui Peng, Yunkai Gao, Fan Wu, Ruizhi Chen, Zidong Du, Xing Hu, Xishan Zhang, Ling Li, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72802bef5cf1a3449e909b20c2ae18d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72802bef5cf1a3449e909b20c2ae18d5-Abstract-Conference.html)

**Abstract**:

In the field of multi-task reinforcement learning, the modular principle, which involves specializing functionalities into different modules and combining them appropriately, has been widely adopted as a promising approach to prevent the negative transfer problem that performance degradation due to conflicts between tasks. However, most of the existing multi-task RL methods only combine shared modules at the task level, ignoring that there may be conflicts within the task. In addition, these methods do not take into account that without constraints, some modules may learn similar functions, resulting in restricting the model's expressiveness and generalization capability of modular methods.In this paper, we propose the Contrastive Modules with Temporal Attention(CMTA) method to address these limitations. CMTA constrains the modules to be different from each other by contrastive learning and combining shared modules at a finer granularity than the task level with temporal attention, alleviating the negative transfer within the task and improving the generalization ability and the performance for multi-task RL.We conducted the experiment on Meta-World, a multi-task RL benchmark containing various robotics manipulation tasks. Experimental results show that CMTA outperforms learning each task individually for the first time and achieves substantial performance improvements over the baselines.

----

## [1585] Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent RL with General Utilities

**Authors**: *Donghao Ying, Yunkai Zhang, Yuhao Ding, Alec Koppel, Javad Lavaei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html)

**Abstract**:

We investigate safe multi-agent reinforcement learning, where agents seek to collectively maximize an aggregate sum of local objectives while satisfying their own safety constraints. The objective and constraints are described by general utilities, i.e., nonlinear functions of the long-term state-action occupancy measure, which encompass broader decision-making goals such as risk, exploration, or imitations. The exponential growth of the state-action space size with the number of agents presents challenges for global observability, further exacerbated by the global coupling arising from agents' safety constraints. To tackle this issue, we propose a primal-dual method utilizing shadow reward and $\kappa$-hop neighbor truncation under a form of correlation decay property, where $\kappa$ is the communication radius. In the exact setting, our algorithm converges to a first-order stationary point (FOSP) at the rate of $\mathcal{O}\left(T^{-2/3}\right)$. In the sample-based setting, we demonstrate that, with high probability, our algorithm requires $\widetilde{\mathcal{O}}\left(\epsilon^{-3.5}\right)$ samples to achieve an $\epsilon$-FOSP with an approximation error of $\mathcal{O}(\phi_0^{2\kappa})$, where $\phi_0\in (0,1)$. Finally, we demonstrate the effectiveness of our model through extensive numerical experiments.

----

## [1586] Optimal Transport-Guided Conditional Score-Based Diffusion Model

**Authors**: *Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html)

**Abstract**:

Conditional score-based diffusion model (SBDM) is for conditional generation of target data with paired data as condition, and has achieved great success in image translation. However, it requires the paired data as condition, and there would be insufficient paired data provided in real-world applications. To tackle the applications with partially paired or even unpaired dataset, we propose a novel Optimal Transport-guided Conditional Score-based diffusion model (OTCS) in this paper. We build the coupling relationship for the unpaired or partially paired dataset based on $L_2$-regularized unsupervised or semi-supervised optimal transport, respectively. Based on the coupling relationship, we develop the objective for training the conditional score-based model for unpaired or partially paired settings, which is based on a reformulation and generalization of the conditional SBDM for paired setting. With the estimated coupling relationship, we effectively train the conditional score-based model by designing  a ``resampling-by-compatibility'' strategy to choose the sampled data with high compatibility as guidance. Extensive experiments on unpaired super-resolution and semi-paired image-to-image translation demonstrated the effectiveness of the proposed OTCS model. From the viewpoint of optimal transport, OTCS provides an approach to transport data across distributions, which is a challenge for OT on large-scale datasets. We theoretically prove that OTCS realizes the data transport in OT with a theoretical bound.

----

## [1587] GNeSF: Generalizable Neural Semantic Fields

**Authors**: *Hanlin Chen, Chen Li, Mengqi Guo, Zhiwen Yan, Gim Hee Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72d32f4fe0b7af03732bd227bf1c4a5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72d32f4fe0b7af03732bd227bf1c4a5f-Abstract-Conference.html)

**Abstract**:

3D scene segmentation based on neural implicit representation has emerged recently with the advantage of training only on 2D supervision. However, existing approaches still requires expensive per-scene optimization that prohibits generalization to novel scenes during inference. To circumvent this problem, we introduce a \textit{generalizable} 3D segmentation framework based on implicit representation. Specifically, our framework takes in multi-view image features and semantic maps as the inputs instead of only spatial information to avoid overfitting to scene-specific geometric and semantic information. We propose a novel soft voting mechanism to aggregate the 2D semantic information from different views for each 3D point. In addition to the image features, view difference information is also encoded in our framework to predict the voting scores. Intuitively, this allows the semantic information from nearby views to contribute more compared to distant ones. Furthermore, a visibility module is also designed to detect and filter out detrimental information from occluded views. Due to the generalizability of our proposed method, we can synthesize semantic maps or conduct 3D semantic segmentation for novel scenes with solely 2D semantic supervision. Experimental results show that our approach achieves comparable performance with scene-specific approaches. More importantly, our approach can even outperform existing strong supervision-based approaches with only 2D annotations.

----

## [1588] When can Regression-Adjusted Control Variate Help? Rare Events, Sobolev Embedding and Minimax Optimality

**Authors**: *Jose H. Blanchet, Haoxuan Chen, Yiping Lu, Lexing Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/730ce0ae730f39e4d77b0f04a8afe4be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/730ce0ae730f39e4d77b0f04a8afe4be-Abstract-Conference.html)

**Abstract**:

This paper studies the use of a machine learning-based estimator as a control variate for mitigating the variance of Monte Carlo sampling. Specifically, we seek to uncover the key factors that influence the efficiency of control variates in reducing variance. We examine a prototype estimation problem that involves simulating the moments of a Sobolev function based on observations obtained from (random) quadrature nodes. Firstly, we establish an information-theoretic lower bound for the problem. We then study a specific quadrature rule that employs a nonparametric regression-adjusted control variate to reduce the variance of the Monte Carlo simulation. We demonstrate that this kind of quadrature rule can improve the Monte Carlo rate and achieve the minimax optimal rate under a sufficient smoothness assumption. Due to the Sobolev Embedding Theorem, the sufficient smoothness assumption eliminates the existence of rare and extreme events. Finally, we show that, in the presence of rare and extreme events, a truncated version of the Monte Carlo algorithm can achieve the minimax optimal rate while the control variate cannot improve the convergence rate.

----

## [1589] Sharp Calibrated Gaussian Processes

**Authors**: *Alexandre Capone, Sandra Hirche, Geoff Pleiss*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7319b7561ffe5e2f6419acd4a2f52d6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7319b7561ffe5e2f6419acd4a2f52d6b-Abstract-Conference.html)

**Abstract**:

While Gaussian processes are a mainstay for various engineering and scientific applications, the uncertainty estimates don't satisfy frequentist guarantees and can be miscalibrated in practice. State-of-the-art approaches for designing calibrated models rely on inflating the Gaussian process posterior variance, which yields confidence intervals that are potentially too coarse. To remedy this, we present a calibration approach that generates predictive quantiles using a computation inspired by the vanilla Gaussian process posterior variance but using a different set of hyperparameters chosen to satisfy an empirical calibration constraint. This results in a calibration approach that is considerably more flexible than existing approaches, which we optimize to yield tight predictive quantiles. Our approach is shown to yield a calibrated model under reasonable assumptions. Furthermore, it outperforms existing approaches in sharpness when employed for calibrated regression.

----

## [1590] GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies

**Authors**: *Takahiro Mimori, Michiaki Hamada*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/732c5757aa5577de9b103332cf7ac0bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/732c5757aa5577de9b103332cf7ac0bf-Abstract-Conference.html)

**Abstract**:

Phylogenetic inference, grounded in molecular evolution models, is essential for understanding the evolutionary relationships in biological data. Accounting for the uncertainty of phylogenetic tree variables, which include tree topologies and evolutionary distances on branches, is crucial for accurately inferring species relationships from molecular data and tasks requiring variable marginalization. Variational Bayesian methods are key to developing scalable, practical models; however, it remains challenging to conduct phylogenetic inference without restricting the combinatorially vast number of possible tree topologies. In this work, we introduce a novel, fully differentiable formulation of phylogenetic inference that leverages a unique representation of topological distributions in continuous geometric spaces. Through practical considerations on design spaces and control variates for gradient estimations, our approach, GeoPhy, enables variational inference without limiting the topological candidates. In experiments using real benchmark datasets, GeoPhy significantly outperformed other approximate Bayesian methods that considered whole topologies.

----

## [1591] AbdomenAtlas-8K: Annotating 8, 000 CT Volumes for Multi-Organ Segmentation in Three Weeks

**Authors**: *Chongyu Qu, Tiezheng Zhang, Hualin Qiao, Jie Liu, Yucheng Tang, Alan L. Yuille, Zongwei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7331077e0449e94a91370c46b4f80f57-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7331077e0449e94a91370c46b4f80f57-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Annotating medical images, particularly for organ segmentation, is laborious and time-consuming. For example, annotating an abdominal organ requires an estimated rate of 30-60 minutes per CT volume based on the expertise of an annotator and the size, visibility, and complexity of the organ. Therefore, publicly available datasets for multi-organ segmentation are often limited in data size and organ diversity. This paper proposes an active learning procedure to expedite the annotation process for organ segmentation and creates the largest multi-organ dataset (by far) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in 8,448 CT volumes, equating to 3.2 million slices. The conventional annotation methods would take an experienced annotator up to 1,600 weeks (or roughly 30.8 years) to complete this task. In contrast, our annotation procedure has accomplished this task in three weeks (based on an 8-hour workday, five days a week) while maintaining a similar or even better annotation quality. This achievement is attributed to three unique properties of our method: (1) label bias reduction using multiple pre-trained segmentation models, (2) effective error detection in the model predictions, and (3) attention guidance for annotators to make corrections on the most salient errors. Furthermore, we summarize the taxonomy of common errors made by AI algorithms and annotators. This allows for continuous improvement of AI and annotations, significantly reducing the annotation costs required to create large-scale datasets for a wider variety of medical imaging tasks. Code and dataset are available at https://github.com/MrGiovanni/AbdomenAtlas

----

## [1592] The Learnability of In-Context Learning

**Authors**: *Noam Wies, Yoav Levine, Amnon Shashua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73950f0eb4ac0925dc71ba2406893320-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73950f0eb4ac0925dc71ba2406893320-Abstract-Conference.html)

**Abstract**:

In-context learning is a surprising and important phenomenon that emerged when modern language models were scaled to billions of learned parameters.   Without modifying a large language model's weights, it can be tuned to perform various downstream natural language tasks simply by including concatenated training examples of these tasks in its input.  Though disruptive for many practical applications of large language models, this emergent learning paradigm is not well understood from a theoretical perspective. In this paper, we propose a first-of-its-kind PAC based framework for in-context learnability, and use it to provide the first finite sample complexity results for the in-context learning setup.  Our framework includes an initial pretraining phase, which fits a function to the pretraining distribution, and then a second in-context learning phase, which keeps this function constant and concatenates training examples of the downstream task in its input.  We use our framework in order to prove that, under mild assumptions, when the pretraining distribution is a mixture of latent tasks (a model often considered for natural language pretraining), these tasks can be efficiently learned via in-context learning, even though the model's weights are unchanged and the input significantly diverges from the pretraining distribution.  Our theoretical analysis reveals that in this setting, in-context learning is more about identifying the task than about learning it, a result which is in line with a series of recent empirical findings.   We hope that the in-context learnability framework presented in this paper will facilitate future progress towards a deeper understanding of this important new learning paradigm.

----

## [1593] Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation

**Authors**: *Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html)

**Abstract**:

The ability to collect a large dataset of human preferences from text-to-image users is usually limited to companies, making such datasets inaccessible to the public. To address this issue, we create a web app that enables text-to-image users to generate images and specify their preferences. Using this web app we build Pick-a-Pic, a large, open dataset of text-to-image prompts and real users’ preferences over generated images. We leverage this dataset to train a CLIP-based scoring function, PickScore, which exhibits superhuman performance on the task of predicting human preferences. Then, we test PickScore’s ability to perform model evaluation and observe that it correlates better with human rankings than other automatic evaluation metrics. Therefore, we recommend using PickScore for evaluating future text-to-image generation models, and using Pick-a-Pic prompts as a more relevant dataset than MS-COCO. Finally, we demonstrate how PickScore can enhance existing text-to-image models via ranking.

----

## [1594] Decorate3D: Text-Driven High-Quality Texture Generation for Mesh Decoration in the Wild

**Authors**: *Yanhui Guo, Xinxin Zuo, Peng Dai, Juwei Lu, Xiaolin Wu, Li Cheng, Youliang Yan, Songcen Xu, Xiaofei Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73af055566f5514b9863315133b84eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73af055566f5514b9863315133b84eda-Abstract-Conference.html)

**Abstract**:

This paper presents Decorate3D, a versatile and user-friendly method for the creation and editing of 3D objects using images. Decorate3D models a real-world object of interest by neural radiance field (NeRF) and decomposes the NeRF representation into an explicit mesh representation, a view-dependent texture, and a diffuse UV texture. Subsequently, users can either manually edit the UV or provide a prompt for the automatic generation of a new 3D-consistent texture.  To achieve high-quality 3D texture generation, we propose a structure-aware score distillation sampling method to optimize a neural UV texture based on user-defined text and empower an image diffusion model with 3D-consistent generation capability. Furthermore, we introduce a few-view resampling training method and utilize a super-resolution model to obtain refined high-resolution UV textures (2048$\times$2048) for 3D texturing. Extensive experiments collectively validate the superior performance of Decorate3D in retexturing real-world 3D objects. Project page: https://decorate3d.github.io/Decorate3D/.

----

## [1595] Representational Strengths and Limitations of Transformers

**Authors**: *Clayton Sanford, Daniel J. Hsu, Matus Telgarsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html)

**Abstract**:

Attention layers, as commonly used in transformers, form the backbone of modern deep learning, yet there is no mathematical description of their benefits and deficiencies as compared with other architectures. In this work we establish both positive and negative results on the representation power of attention layers, with a focus on intrinsic complexity parameters such as width, depth, and embedding dimension. On the positive side, we present a sparse averaging task, where recurrent networks and feedforward networks all have complexity scaling polynomially in the input size, whereas transformers scale merely logarithmically in the input size; furthermore, we use the same construction to show the necessity and role of a large embedding dimension in a transformer. On the negative side, we present a triple detection task, where attention layers in turn have complexity scaling linearly in the input size; as this scenario seems rare in practice, we also present natural variants that can be efficiently solved by attention layers. The proof techniques emphasize the value of communication complexity in the analysis of transformers and related models, and the role of sparse averaging as a prototypical attention task, which even finds use in the analysis of triple detection.

----

## [1596] On the Relationship Between Relevance and Conflict in Online Social Link Recommendations

**Authors**: *Yanbang Wang, Jon M. Kleinberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73d6c3e4b214deebbbf8256e26d2cf45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73d6c3e4b214deebbbf8256e26d2cf45-Abstract-Conference.html)

**Abstract**:

In an online social network, link recommendations are a way for users to discover relevant links to people they may know, thereby potentially increasing their engagement on the platform. However, the addition of links to a social network can also have an effect on the level of conflict in the network --- expressed in terms of polarization and disagreement. To date, however, we have very little understanding of how these two implications of link formation relate to each other: are the goals of high relevance and conflict reduction aligned, or are the links that users are most likely to accept fundamentally different from the ones with the greatest potential for reducing conflict? Here we provide the first analysis of this question, using the recently popular Friedkin-Johnsen model of opinion dynamics. We first present a surprising result on how link additions shift the level of opinion conflict, followed by explanation work that relates the amount of shift to structural features of the added links. We then characterize the gap in conflict reduction between the set of links achieving the largest reduction and the set of links achieving the highest relevance. The gap is measured on real-world data, based on instantiations of relevance defined by 13 link recommendation algorithms. We find that some, but not all, of the more accurate algorithms actually lead to better reduction of conflict. Our work suggests that social links recommended for increasing user engagement may not be as conflict-provoking as people might have thought.

----

## [1597] Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

**Authors**: *Ziba Parsons, Fei Dou, Houyi Du, Zheng Song, Jin Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html)

**Abstract**:

This paper explores the challenges of implementing Federated Learning (FL) in practical scenarios featuring isolated nodes with data heterogeneity, which can only be connected to the server through wireless links in an infrastructure-less environment. To overcome these challenges, we propose a novel mobilizing personalized FL approach, which aims to facilitate mobility and resilience. Specifically, we develop a novel optimization algorithm called Random Walk Stochastic Alternating Direction Method of Multipliers (RWSADMM). RWSADMM capitalizes on the server's random movement toward clients and formulates local proximity among their adjacent clients based on hard inequality constraints rather than requiring consensus updates or introducing bias via regularization methods. To mitigate the computational burden on the clients, an efficient stochastic solver of the approximated optimization problem is designed in RWSADMM, which provably converges to the stationary point almost surely in expectation. Our theoretical and empirical results demonstrate the provable fast convergence and substantial accuracy improvements achieved by RWSADMM compared to baseline methods, along with its benefits of reduced communication costs and enhanced scalability.

----

## [1598] An Optimal Structured Zeroth-order Algorithm for Non-smooth Optimization

**Authors**: *Marco Rando, Cesare Molinari, Lorenzo Rosasco, Silvia Villa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7429f4c1b267cf619f28c4d4f1532f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7429f4c1b267cf619f28c4d4f1532f99-Abstract-Conference.html)

**Abstract**:

Finite-difference methods are a class of algorithms designed to solve black-box optimization problems by approximating a gradient of the target function on a set of directions. In black-box optimization, the non-smooth setting is particularly relevant since, in practice, differentiability and smoothness assumptions cannot be verified. To cope with nonsmoothness, several authors use a smooth approximation of the target function and show that finite difference methods approximate its gradient. Recently, it has been proved that imposing a structure in the directions allows improving performance. However, only the smooth setting was considered. To close this gap, we introduce and analyze O-ZD, the first structured finite-difference algorithm for non-smooth black-box optimization. Our method exploits a smooth approximation of the target function and we prove that it approximates its gradient on a subset of random {\em orthogonal} directions. We analyze the convergence of O-ZD under different assumptions.  For non-smooth convex functions, we obtain the optimal complexity. In the non-smooth non-convex setting, we characterize the number of iterations needed to bound the expected norm of the smoothed gradient. For smooth functions, our analysis recovers existing results for structured zeroth-order methods for the convex case and extends them to the non-convex setting. We conclude with numerical simulations where assumptions are satisfied, observing that our algorithm has very good practical performances.

----

## [1599] Online Control for Meta-optimization

**Authors**: *Xinyi Chen, Elad Hazan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/745b7e084d5ca5afc07fb454ab2be522-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/745b7e084d5ca5afc07fb454ab2be522-Abstract-Conference.html)

**Abstract**:

Choosing the optimal hyperparameters, including learning rate and momentum, for specific optimization instances is a significant yet non-convex challenge. This makes conventional iterative techniques such as hypergradient descent \cite{baydin2017online} insufficient in obtaining global optimality guarantees.We consider the more general task of meta-optimization -- online learning of the best optimization algorithm given problem instances, and introduce a novel approach based on control theory. We show how meta-optimization can be formulated as an optimal control problem, departing from existing literature that use stability-based methods to study optimization. Our approach leverages convex relaxation techniques in the recently-proposed nonstochastic control framework to overcome the challenge of nonconvexity, and obtains regret guarantees vs. the best offline solution. This guarantees that in meta-optimization, we can learn a method that attains convergence comparable to that of the best optimization method in hindsight from a class of methods.

----



[Go to the previous page](NIPS-2023-list7.md)

[Go to the next page](NIPS-2023-list9.md)

[Go to the catalog section](README.md)