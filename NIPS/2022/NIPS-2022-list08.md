## [1400] Agreement-on-the-line: Predicting the Performance of Neural Networks under Distribution Shift

        **Authors**: *Christina Baek, Yiding Jiang, Aditi Raghunathan, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html)

        **Abstract**:

        Recently, Miller et al. showed that a model's in-distribution (ID) accuracy has a strong linear correlation with its out-of-distribution (OOD) accuracy, on several OOD benchmarks, a phenomenon they dubbed ``accuracy-on-the-line''.  While a useful tool for model selection (i.e., the model most likely to perform the best OOD is the one with highest ID accuracy), this fact does not help to estimate the actual OOD performance of models without access to a labeled OOD validation set. In this paper, we show a similar surprising phenomena also holds for the agreement between pairs of neural network classifiers: whenever accuracy-on-the-line holds, we observe that the OOD agreement between the predictions of any two pairs of neural networks (with potentially different architectures) also observes a strong linear correlation with their ID agreement. Furthermore, we observe that the slope and bias of OOD vs ID agreement closely matches that of OOD vs ID accuracy. This phenomenon which we call agreement-on-the-line, has important practical applications: without any labeled data, we can predict the OOD accuracy of classifiers, since OOD agreement can be estimated with just unlabeled data. Our prediction algorithm outperforms previous methods both in shifts where agreement-on-the-line holds and, surprisingly, when accuracy is not on the line. This phenomenon also provides new insights into neural networks: unlike accuracy-on-the-line, agreement-on-the-line only appears to hold for neural network classifiers.

        ----

        ## [1401] Large-Scale Differentiable Causal Discovery of Factor Graphs

        **Authors**: *Romain Lopez, Jan-Christian Hütter, Jonathan K. Pritchard, Aviv Regev*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html)

        **Abstract**:

        A common theme in causal inference is learning causal relationships between observed variables, also known as causal discovery. This is usually a daunting task, given the large number of candidate causal graphs and the combinatorial nature of the search space. Perhaps for this reason, most research has so far focused on relatively small causal graphs, with up to hundreds of nodes. However, recent advances in fields like biology enable generating experimental data sets with thousands of interventions followed by rich profiling of thousands of variables, raising the opportunity and urgent need for large causal graph models.  Here, we introduce the notion of factor directed acyclic graphs ($f$-DAGs) as a way to restrict the search space to non-linear low-rank causal interaction models. Combining this novel structural assumption with recent advances that bridge the gap between causal discovery and continuous optimization, we achieve causal discovery on thousands of variables. Additionally, as a model for the impact of statistical noise on this estimation procedure, we study a model of edge perturbations of the $f$-DAG skeleton based on random graphs and quantify the effect of such perturbations on the $f$-DAG rank. This theoretical analysis suggests that the set of candidate $f$-DAGs is much smaller than the whole DAG space and thus may be more suitable as a search space in the high-dimensional regime where the underlying skeleton is hard to assess. We propose Differentiable Causal Discovery of Factor Graphs (DCD-FG), a scalable implementation of $f$-DAG constrained causal discovery for high-dimensional interventional data. DCD-FG uses a Gaussian non-linear low-rank structural equation model and shows significant improvements compared to state-of-the-art methods in both simulations as well as a recent large-scale single-cell RNA sequencing data set with hundreds of genetic interventions.

        ----

        ## [1402] Diagnosing failures of fairness transfer across distribution shift in real-world medical settings

        **Authors**: *Jessica Schrouff, Natalie Harris, Sanmi Koyejo, Ibrahim M. Alabdulmohsin, Eva Schnider, Krista Opsahl-Ong, Alexander Brown, Subhrajit Roy, Diana Mincu, Christina Chen, Awa Dieng, Yuan Liu, Vivek Natarajan, Alan Karthikesalingam, Katherine A. Heller, Silvia Chiappa, Alexander D'Amour*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a969c30dc7e74d4e891c8ffb217cf79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a969c30dc7e74d4e891c8ffb217cf79-Abstract-Conference.html)

        **Abstract**:

        Diagnosing and mitigating changes in model fairness under distribution shift is an important component of the safe deployment of machine learning in healthcare settings. Importantly, the success of any mitigation strategy strongly depends on the \textit{structure} of the shift. Despite this, there has been little discussion of how to empirically assess the structure of a distribution shift that one is encountering in practice. In this work, we adopt a causal framing to motivate conditional independence tests as a key tool for characterizing distribution shifts. Using our approach in two medical applications, we show that this knowledge can help diagnose failures of fairness transfer, including cases where real-world shifts are more complex than is often assumed in the literature. Based on these results, we discuss potential remedies at each step of the machine learning pipeline.

        ----

        ## [1403] Towards Lightweight Black-Box Attack Against Deep Neural Networks

        **Authors**: *Chenghao Sun, Yonggang Zhang, Chaoqun Wan, Qizhou Wang, Ya Li, Tongliang Liu, Bo Han, Xinmei Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a9745f251508a053425a256490b0665-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a9745f251508a053425a256490b0665-Abstract-Conference.html)

        **Abstract**:

        Black-box attacks can generate adversarial examples without accessing the parameters of target model, largely exacerbating the threats of deployed deep neural networks (DNNs). However, previous works state that black-box attacks fail to mislead target models when their training data and outputs are inaccessible. In this work, we argue that black-box attacks can pose practical attacks in this extremely restrictive scenario where only several test samples are available.  Specifically, we find that attacking the shallow layers of DNNs trained on a few test samples can generate powerful adversarial examples. As only a few samples are required, we refer to these attacks as lightweight black-box attacks. The main challenge to promoting lightweight attacks is to mitigate the adverse impact caused by the approximation error of shallow layers. As it is hard to mitigate the approximation error with few available samples, we propose Error TransFormer (ETF) for lightweight attacks. Namely, ETF transforms the approximation error in the parameter space into a perturbation in the feature space and alleviates the error by disturbing features. In experiments, lightweight black-box attacks with the proposed ETF achieve surprising results. For example, even if only 1 sample per category available, the attack success rate in lightweight black-box attacks is only about 3% lower than that of the black-box attacks with complete training data.

        ----

        ## [1404] Federated Learning from Pre-Trained Models: A Contrastive Learning Approach

        **Authors**: *Yue Tan, Guodong Long, Jie Ma, Lu Liu, Tianyi Zhou, Jing Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7aa320d2b4b8f6400b18f6f77b6c1535-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7aa320d2b4b8f6400b18f6f77b6c1535-Abstract-Conference.html)

        **Abstract**:

        Federated Learning (FL) is a machine learning paradigm that allows decentralized clients to learn collaboratively without sharing their private data. However, excessive computation and communication demands pose challenges to current FL frameworks, especially when training large-scale models. To prevent these issues from hindering the deployment of FL systems, we propose a lightweight framework where clients jointly learn to fuse the representations generated by multiple fixed pre-trained models rather than training a large-scale model from scratch. This leads us to a more practical FL problem by considering how to capture more client-specific and class-relevant information from the pre-trained models and jointly improve each client's ability to exploit those off-the-shelf models. Here, we design a Federated Prototype-wise Contrastive Learning (FedPCL) approach which shares knowledge across clients through their class prototypes and builds client-specific representations in a prototype-wise contrastive manner. Sharing prototypes rather than learnable model parameters allows each client to fuse the representations in a personalized way while keeping the shared knowledge in a compact form for efficient communication. We perform a thorough evaluation of the proposed FedPCL in the lightweight framework, measuring and visualizing its ability to fuse various pre-trained models on popular FL datasets.

        ----

        ## [1405] Theoretically Provable Spiking Neural Networks

        **Authors**: *Shao-Qun Zhang, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks have attracted increasing attention in recent years due to their potential of handling time-dependent data. Many algorithms and techniques have been developed; however, theoretical understandings of many aspects of spiking neural networks are far from clear. A recent work [Zhang and Zhou, 2021] disclosed that typical spiking neural networks could hardly work on spatio-temporal data due to their bifurcation dynamics and suggested that the self-connection structure has to be added. In this paper, we theoretically investigate the approximation ability and computational efficiency of spiking neural networks with self connections, and show that the self-connection structure enables spiking neural networks to approximate discrete dynamical systems using a polynomial number of parameters within polynomial time complexities. Our theoretical results may shed some insight for the future studies of spiking neural networks.

        ----

        ## [1406] Approximate Euclidean lengths and distances beyond Johnson-Lindenstrauss

        **Authors**: *Aleksandros Sobczyk, Mathieu Luisier*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7ac1846846f02b42dcfab5c40bc6ae56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7ac1846846f02b42dcfab5c40bc6ae56-Abstract-Conference.html)

        **Abstract**:

        A classical result of Johnson and Lindenstrauss states that a set of $n$ high dimensional data points can be projected down to $O(\log n/\epsilon^2)$ dimensions such that the square of their pairwise distances is preserved up to a small distortion $\epsilon\in(0,1)$. It has been proved that the JL lemma is optimal for the general case, therefore, improvements can only be explored for special cases. This work aims to improve the $\epsilon^{-2}$ dependency based on techniques inspired by the Hutch++ Algorithm, which reduces $\epsilon^{-2}$ to $\epsilon^{-1}$ for the related problem of implicit matrix trace estimation. We first present an algorithm to estimate the Euclidean lengths of the rows of a matrix. We prove for it element-wise probabilistic bounds that are at least as good as standard JL approximations in the worst-case, but are asymptotically better for matrices with decaying spectrum. Moreover, for any matrix, regardless of its spectrum, the algorithm achieves $\epsilon$-accuracy for the total, Frobenius norm-wise relative error using only $O(\epsilon^{-1})$ queries. This is a quadratic improvement over the norm-wise error of standard JL approximations. We also show how these results can be extended to estimate (i) the Euclidean distances between data points and (ii) the statistical leverage scores of tall-and-skinny data matrices, which are ubiquitous for many applications, with analogous theoretical improvements. Proof-of-concept numerical experiments are presented to validate the theoretical analysis.

        ----

        ## [1407] InsPro: Propagating Instance Query and Proposal for Online Video Instance Segmentation

        **Authors**: *Fei He, Haoyang Zhang, Naiyu Gao, Jian Jia, Yanhu Shan, Xin Zhao, Kaiqi Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html)

        **Abstract**:

        Video instance segmentation (VIS) aims at segmenting and tracking objects in videos. Prior methods typically generate frame-level or clip-level object instances first and then associate them by either additional tracking heads or complex instance matching algorithms. This explicit instance association approach increases system complexity and fails to fully exploit temporal cues in videos. In this paper, we design a simple, fast and yet effective query-based framework for online VIS. Relying on an instance query and proposal propagation mechanism with several specially developed components, this framework can perform accurate instance association implicitly. Specifically, we generate frame-level object instances based on a set of instance query-proposal pairs propagated from previous frames. This instance query-proposal pair is learned to bind with one specific object across frames through conscientiously developed strategies. When using such a pair to predict an object instance on the current frame, not only the generated instance is automatically associated with its precursors on previous frames, but the model gets a good prior for predicting the same object. In this way, we naturally achieve implicit instance association in parallel with segmentation and elegantly take advantage of temporal clues in videos. To show the effectiveness of our method InsPro, we evaluate it on two popular VIS benchmarks, i.e., YouTube-VIS 2019 and YouTube-VIS 2021. Without bells-and-whistles, our InsPro with ResNet-50 backbone achieves 43.2 AP and 37.6 AP on these two benchmarks respectively, outperforming all other online VIS methods.

        ----

        ## [1408] Measuring and Reducing Model Update Regression in Structured Prediction for NLP

        **Authors**: *Deng Cai, Elman Mansimov, Yi-An Lai, Yixuan Su, Lei Shu, Yi Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html)

        **Abstract**:

        Recent advance in deep learning has led to rapid adoption of machine learning based NLP models in a wide range of applications. Despite the continuous gain in accuracy, backward compatibility is also an important aspect for industrial applications, yet it received little research attention. Backward compatibility requires that the new model does not regress on cases that were correctly handled by its predecessor. This work studies model update regression in structured prediction tasks. We choose syntactic dependency parsing and conversational semantic parsing as representative examples of structured prediction tasks in NLP. First, we measure and analyze model update regression in different model update settings. Next, we explore and benchmark existing techniques for reducing model update regression including model ensemble and knowledge distillation. We further propose a simple and effective method, Backward-Congruent Re-ranking (BCR), by taking into account the characteristics of structured output. Experiments show that BCR can better mitigate model update regression than model ensemble and knowledge distillation approaches.

        ----

        ## [1409] Rethinking Lipschitz Neural Networks and Certified Robustness: A Boolean Function Perspective

        **Authors**: *Bohang Zhang, Du Jiang, Di He, Liwei Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b04ec5f2b89d7f601382c422dfe07af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b04ec5f2b89d7f601382c422dfe07af-Abstract-Conference.html)

        **Abstract**:

        Designing neural networks with bounded Lipschitz constant is a promising way to obtain certifiably robust classifiers against adversarial examples. However, the relevant progress for the important $\ell_\infty$ perturbation setting is rather limited, and a principled understanding of how to design expressive $\ell_\infty$ Lipschitz networks is still lacking. In this paper, we bridge the gap by studying certified $\ell_\infty$ robustness from a novel perspective of representing Boolean functions. We derive two fundamental impossibility results that hold for any standard Lipschitz network: one for robust classification on finite datasets, and the other for Lipschitz function approximation. These results identify that networks built upon norm-bounded affine layers and Lipschitz activations intrinsically lose expressive power even in the two-dimensional case, and shed light on how recently proposed Lipschitz networks (e.g., GroupSort and $\ell_\infty$-distance nets) bypass these impossibilities by leveraging order statistic functions. Finally, based on these insights, we develop a unified Lipschitz network that generalizes prior works, and design a practical version that can be efficiently trained (making certified robust training free). Extensive experiments show that our approach is scalable, efficient, and consistently yields better certified robustness across multiple datasets and perturbation radii than prior Lipschitz networks.

        ----

        ## [1410] Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks

        **Authors**: *Yijing Liu, Qinxian Liu, Jian-Wei Zhang, Haozhe Feng, Zhongwei Wang, Zihan Zhou, Wei Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b102c908e9404dd040599c65db4ce3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b102c908e9404dd040599c65db4ce3e-Abstract-Conference.html)

        **Abstract**:

        Modeling multivariate time series (MTS) is critical in modern intelligent systems. The accurate forecast of MTS data is still challenging due to the complicated latent variable correlation. Recent works apply the Graph Neural Networks (GNNs) to the task, with the basic idea of representing the correlation as a static graph. However, predicting with a static graph causes significant bias because the correlation is time-varying in the real-world MTS data. Besides, there is no gap analysis between the actual correlation and the learned one in their works to validate the effectiveness. This paper proposes a temporal polynomial graph neural network (TPGNN) for accurate MTS forecasting, which represents the dynamic variable correlation as a temporal matrix polynomial in two steps. First, we capture the overall correlation with a static matrix basis. Then, we use a set of time-varying coefficients and the matrix basis to construct a matrix polynomial for each time step. The constructed result empirically captures the precise dynamic correlation of six synthetic MTS datasets generated by a non-repeating random walk model. Moreover, the theoretical analysis shows that TPGNN can achieve perfect approximation under a commutative condition. We conduct extensive experiments on two traffic datasets with prior structure and four benchmark datasets. The results indicate that TPGNN achieves the state-of-the-art on both short-term and long-term MTS forecastings.

        ----

        ## [1411] Few-shot Image Generation via Adaptation-Aware Kernel Modulation

        **Authors**: *Yunqing Zhao, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b122d0a0dcb1a86ffa25ccba154652b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b122d0a0dcb1a86ffa25ccba154652b-Abstract-Conference.html)

        **Abstract**:

        Few-shot image generation (FSIG) aims to learn to generate new and diverse samples given an extremely limited number of samples from a domain, e.g., 10 training samples. Recent work has addressed the problem using transfer learning approach, leveraging a GAN pretrained on a large-scale source domain dataset and adapting that model to the target domain based on very limited target domain samples. Central to recent FSIG methods are knowledge preserving criteria, which aim to select a subset of source model's knowledge to be preserved into the adapted model. However, a major limitation of existing methods is that their knowledge preserving criteria consider only source domain/source task, and they fail to consider target domain/adaptation task in selecting source model's knowledge, casting doubt on their suitability for setups of different proximity between source and target domain. Our work makes two contributions. As our first contribution, we re-visit recent FSIG works and their experiments. Our important finding is that, under setups which assumption of close proximity between source and target domains is relaxed, existing state-of-the-art (SOTA) methods which consider only source domain/source task in knowledge preserving perform no better than a baseline fine-tuning method. To address the limitation of existing methods, as our second contribution, we propose Adaptation-Aware kernel Modulation (AdAM) to address general FSIG of different source-target domain proximity. Extensive experimental results show that the proposed method consistently achieves SOTA performance across source/target domains of different proximity, including challenging setups when source and target domains are more apart. Project Page: https://yunqing-me.github.io/AdAM/

        ----

        ## [1412] Learning to Follow Instructions in Text-Based Games

        **Authors**: *Mathieu Tuli, Andrew C. Li, Pashootan Vaezipoor, Toryn Q. Klassen, Scott Sanner, Sheila A. McIlraith*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b24015f3af598e1d9179f6e06353780-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b24015f3af598e1d9179f6e06353780-Abstract-Conference.html)

        **Abstract**:

        Text-based games present a unique class of sequential decision making problem in which agents interact with a partially observable, simulated environment via actions and observations conveyed through natural language. Such observations typically include instructions that, in a reinforcement learning (RL) setting, can directly or indirectly guide a player towards completing reward-worthy tasks. In this work, we study the ability of RL agents to follow such instructions. We conduct experiments that show that the performance of state-of-the-art text-based game agents is largely unaffected by the presence or absence of such instructions, and that these agents are typically unable to execute tasks to completion. To further study and address the task of instruction following, we equip RL agents with an internal structured representation of natural language instructions in the form of Linear Temporal Logic (LTL), a formal language that is increasingly used for temporally extended reward specification in RL. Our framework both supports and highlights the benefit of understanding the temporal semantics of instructions and in measuring progress towards achievement of such a temporally extended behaviour. Experiments with 500+ games in TextWorld demonstrate the superior performance of our approach.

        ----

        ## [1413] Concentration of Data Encoding in Parameterized Quantum Circuits

        **Authors**: *Guangxi Li, Ruilin Ye, Xuanqiang Zhao, Xin Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b2d0730df1edd8c97df4bf83696025d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b2d0730df1edd8c97df4bf83696025d-Abstract-Conference.html)

        **Abstract**:

        Variational quantum algorithms have been acknowledged as the leading strategy to realize near-term quantum advantages in meaningful tasks, including machine learning and optimization. When applied to tasks involving classical data, such algorithms generally begin with data encoding circuits and train quantum neural networks (QNNs) to minimize target functions. Although QNNs have been widely studied to improve these algorithms' performance on practical tasks, there is a gap in systematically understanding the influence of data encoding on the eventual performance. In this paper, we make progress in filling this gap by considering the common data encoding strategies based on parameterized quantum circuits. We prove that, under reasonable assumptions, the distance between the average encoded state and the maximally mixed state could be explicitly upper-bounded with respect to the width and depth of the encoding circuit. This result in particular implies that the average encoded state will concentrate on the maximally mixed state at an exponential speed on depth. Such concentration seriously limits the capabilities of quantum classifiers, and strictly restricts the distinguishability of encoded states from a quantum information perspective. To support our findings, we numerically verify these results on both synthetic and public data sets. Our results highlight the significance of quantum data encoding and may shed light on the future design of quantum encoding strategies.

        ----

        ## [1414] Improving Variational Autoencoders with Density Gap-based Regularization

        **Authors**: *Jianfei Zhang, Jun Bai, Chenghua Lin, Yanmeng Wang, Wenge Rong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html)

        **Abstract**:

        Variational autoencoders (VAEs) are one of the most powerful unsupervised learning frameworks in NLP for latent representation learning and latent-directed generation. The classic optimization goal of VAEs is to maximize the Evidence Lower Bound (ELBo), which consists of a conditional likelihood for generation and a negative Kullback-Leibler (KL) divergence for regularization. In practice, optimizing ELBo often leads the posterior distribution of all samples converging to the same degenerated local optimum, namely posterior collapse or KL vanishing. There are effective ways proposed to prevent posterior collapse in VAEs, but we observe that they in essence make trade-offs between posterior collapse and the hole problem, i.e., the mismatch between the aggregated posterior distribution and the prior distribution. To this end, we introduce new training objectives to tackle both problems through a novel regularization based on the probabilistic density gap between the aggregated posterior distribution and the prior distribution. Through experiments on language modeling, latent space visualization, and interpolation, we show that our proposed method can solve both problems effectively and thus outperforms the existing methods in latent-directed generation. To the best of our knowledge, we are the first to jointly solve the hole problem and posterior collapse.

        ----

        ## [1415] RISE: Robust Individualized Decision Learning with Sensitive Variables

        **Authors**: *Xiaoqing Tan, Zhengling Qi, Christopher W. Seymour, Lu Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html)

        **Abstract**:

        This paper introduces RISE, a robust individualized decision learning framework with sensitive variables, where sensitive variables are collectible data and important to the intervention decision, but their inclusion in decision making is prohibited due to reasons such as delayed availability or fairness concerns. A naive baseline is to ignore these sensitive variables in learning decision rules, leading to significant uncertainty and bias. To address this, we propose a decision learning framework to incorporate sensitive variables during offline training but not include them in the input of the learned decision rule during model deployment. Specifically, from a causal perspective, the proposed framework intends to improve the worst-case outcomes of individuals caused by sensitive variables that are unavailable at the time of decision. Unlike most existing literature that uses mean-optimal objectives, we propose a robust learning framework by finding a newly defined quantile- or infimum-optimal decision rule. The reliable performance of the proposed method is demonstrated through synthetic experiments and three real-world applications.

        ----

        ## [1416] Bayesian Clustering of Neural Spiking Activity Using a Mixture of Dynamic Poisson Factor Analyzers

        **Authors**: *Ganchao Wei, Ian H. Stevenson, Xiaojing Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b39f4512a2e3899edcc59c7501f3cd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b39f4512a2e3899edcc59c7501f3cd4-Abstract-Conference.html)

        **Abstract**:

        Modern neural recording techniques allow neuroscientists to observe the spiking activity of many neurons simultaneously. Although previous work has illustrated how activity within and between known populations of neurons can be summarized by low-dimensional latent vectors, in many cases what determines a unique population may be unclear. Neurons differ in their anatomical location, but also, in their cell types and response properties. Moreover, multiple distinct populations may not be well described by a single low-dimensional, linear representation.To tackle these challenges, we develop a clustering method based on a mixture of dynamic Poisson factor analyzers (DPFA) model, with the number of clusters treated as an unknown parameter. To do the analysis of DPFA model, we propose a novel Markov chain Monte Carlo (MCMC) algorithm to efficiently sample its posterior distribution. Validating our proposed MCMC algorithm with simulations, we find that it can accurately recover the true clustering and latent states and is insensitive to the initial cluster assignments. We then apply the proposed mixture of DPFA model to multi-region experimental recordings, where we find that the proposed method can identify novel, reliable clusters of neurons based on their activity, and may, thus, be a useful tool for neural data analysis.

        ----

        ## [1417] Understanding Deep Contrastive Learning via Coordinate-wise Optimization

        **Authors**: *Yuandong Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b5c9cc08960df40615c1d858961eb8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b5c9cc08960df40615c1d858961eb8b-Abstract-Conference.html)

        **Abstract**:

        We show that Contrastive Learning (CL) under a broad family of loss functions (including InfoNCE) has a unified formulation of coordinate-wise optimization on the network parameter $\vtheta$ and pairwise importance $\alpha$, where the \emph{max player} $\vtheta$ learns representation for contrastiveness, and the \emph{min player} $\alpha$ puts more weights on pairs of distinct samples that share similar representations. The resulting formulation, called \boldmethod{}, unifies not only various existing contrastive losses, which differ by how sample-pair importance $\alpha$ is constructed, but also is able to extrapolate to give novel contrastive losses beyond popular ones, opening a new avenue of contrastive loss design. These novel losses yield comparable (or better) performance on CIFAR10, STL-10 and CIFAR-100 than classic InfoNCE. Furthermore, we also analyze the max player in detail: we prove that with fixed $\alpha$, max player is equivalent to Principal Component Analysis (PCA) for deep linear network, and almost all local minima are global and rank-1, recovering optimal PCA solutions. Finally, we extend our analysis on max player to 2-layer ReLU networks, showing that its fixed points can have higher ranks. Codes are available in https://github.com/facebookresearch/luckmatters/tree/main/ssl/real-dataset.

        ----

        ## [1418] Beyond neural scaling laws: beating power law scaling via data pruning

        **Authors**: *Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html)

        **Abstract**:

        Widely observed neural scaling laws, in which error falls off as a power of the training set size, model size, or both, have driven substantial performance improvements in deep learning. However, these improvements through scaling alone require considerable costs in compute and energy. Here we focus on the scaling of error with dataset size and show how in theory we can break beyond power law scaling and potentially even reduce it to exponential scaling instead if we have access to a high-quality data pruning metric that ranks the order in which training examples should be discarded to achieve any pruned dataset size. We then test this improved scaling prediction with pruned dataset size empirically, and indeed observe better than power law scaling in practice on ResNets trained on CIFAR-10, SVHN, and ImageNet. Next, given the importance of finding high-quality pruning metrics, we perform the first large-scale benchmarking study of ten different data pruning metrics on ImageNet. We find most existing high performing metrics scale poorly to ImageNet, while the best are computationally intensive and require labels for every image. We therefore developed a new simple, cheap and scalable self-supervised pruning metric that demonstrates comparable performance to the best supervised metrics. Overall, our work suggests that the discovery of good data-pruning metrics may provide a viable path forward to substantially improved neural scaling laws, thereby reducing the resource costs of modern deep learning.

        ----

        ## [1419] Neural Estimation of Submodular Functions with Applications to Differentiable Subset Selection

        **Authors**: *Abir De, Soumen Chakrabarti*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html)

        **Abstract**:

        Submodular functions and variants, through their ability to characterize diversity and coverage, have emerged as a key tool for data selection and summarization.  Many recent approaches to learn submodular functions suffer from limited expressiveness. In this work, we propose FlexSubNet, a family of flexible neural models for both monotone and non-monotone submodular functions. To fit a latent submodular function from (set, value) observations, our method applies a concave function on modular functions in a recursive manner. We do not draw the concave function from a restricted family, but rather learn from data using a highly expressive neural network that implements a differentiable quadrature procedure. Such an expressive neural model for concave functions may be of independent interest.  Next, we extend this setup to provide a novel characterization of monotone $\alpha$-submodular functions, a recently introduced notion of approximate submodular functions.  We then use this characterization to design a novel neural model for such functions. Finally, we consider learning submodular set functions under distant supervision in the form of  (perimeter, high-value-subset) pairs.  This yields a novel subset selection method based on an order-invariant, yet greedy sampler built around the above neural set functions. Our experiments on synthetic and real data show that FlexSubNet outperforms several baselines.

        ----

        ## [1420] Maximum Class Separation as Inductive Bias in One Matrix

        **Authors**: *Tejaswi Kasarla, Gertjan J. Burghouts, Max van Spengler, Elise van der Pol, Rita Cucchiara, Pascal Mettes*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b95e1ca9d7347da59cefd362e60a0b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b95e1ca9d7347da59cefd362e60a0b6-Abstract-Conference.html)

        **Abstract**:

        Maximizing the separation between classes constitutes a well-known inductive bias in machine learning and a pillar of many traditional algorithms. By default, deep networks are not equipped with this inductive bias and therefore many alternative solutions have been proposed through differential optimization. Current approaches tend to optimize classification and separation jointly: aligning inputs with class vectors and separating class vectors angularly. This paper proposes a simple alternative: encoding maximum separation as an inductive bias in the network by adding one fixed matrix multiplication before computing the softmax activations. The main observation behind our approach is that separation does not require optimization but can be solved in closed-form prior to training and plugged into a network. We outline a recursive approach to obtain the matrix consisting of maximally separable vectors for any number of classes, which can be added with negligible engineering effort and computational overhead. Despite its simple nature, this one matrix multiplication provides real impact. We show that our proposal directly boosts classification, long-tailed recognition, out-of-distribution detection, and open-set recognition, from CIFAR to ImageNet. We find empirically that maximum separation works best as a fixed bias; making the matrix learnable adds nothing to the performance. The closed-form implementation and code to reproduce the experiments are available on github.

        ----

        ## [1421] Surprising Instabilities in Training Deep Networks and a Theoretical Analysis

        **Authors**: *Yuxin Sun, Dong Lao, Ganesh Sundaramoorthi, Anthony J. Yezzi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html)

        **Abstract**:

        We empirically demonstrate numerical instabilities in training standard deep networks with SGD. Specifically, we show numerical error (on the order of the smallest floating point bit) induced from floating point arithmetic in training deep nets can be amplified significantly and result in significant test accuracy variance, comparable to the test accuracy variance due to stochasticity in SGD. We show how this is likely traced to instabilities of the optimization dynamics that are localized over iterations and regions of the weight tensor space. We do this by presenting a theoretical framework using numerical analysis of partial differential equations (PDE), and analyzing the gradient descent PDE of a one-layer convolutional neural network, which is sufficient to illustrate these instabilities. We show that it is stable only under certain conditions on the learning rate and weight decay. We reproduce the localized instabilities in the PDE for the one-layer network, which arise when the conditions are violated.

        ----

        ## [1422] WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting

        **Authors**: *Youngin Cho, Daejin Kim, Dongmin Kim, Mohammad Azam Khan, Jaegul Choo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b99e3c648898b9e4923dea0aeb4afa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b99e3c648898b9e4923dea0aeb4afa1-Abstract-Conference.html)

        **Abstract**:

        Time series forecasting has become a critical task due to its high practicality in real-world applications such as traffic, energy consumption, economics and finance, and disease analysis. Recent deep-learning-based approaches have shown remarkable success in time series forecasting. Nonetheless, due to the dynamics of time series data, deep networks still suffer from unstable training and overfitting. Inconsistent patterns appearing in real-world data lead the model to be biased to a particular pattern, thus limiting the generalization. In this work, we introduce the dynamic error bounds on training loss to address the overfitting issue in time series forecasting. Consequently, we propose a regularization method called WaveBound which estimates the adequate error bounds of training loss for each time step and feature at each iteration. By allowing the model to focus less on unpredictable data, WaveBound stabilizes the training process, thus significantly improving generalization. With the extensive experiments, we show that WaveBound consistently improves upon the existing models in large margins, including the state-of-the-art model.

        ----

        ## [1423] Finite-Time Analysis of Adaptive Temporal Difference Learning with Deep Neural Networks

        **Authors**: *Tao Sun, Dongsheng Li, Bao Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7b9ebf1a1c149960c3452dc94cbd158e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7b9ebf1a1c149960c3452dc94cbd158e-Abstract-Conference.html)

        **Abstract**:

        Temporal difference (TD) learning with function approximations (linear functions or neural networks) has achieved remarkable empirical success, giving impetus to the development of finite-time analysis. As an accelerated version of TD, the adaptive TD has been proposed and proved to enjoy finite-time convergence under the linear function approximation. Existing numerical results have demonstrated the superiority of adaptive algorithms to vanilla ones. Nevertheless, the performance guarantee of adaptive TD with neural network approximation remains widely unknown. This paper establishes the finite-time analysis for the adaptive TD with multi-layer ReLU network approximation whose samples are generated from a Markov decision process. Our established theory shows that if the width of the deep neural network is large enough, the adaptive TD using neural network approximation can find the (optimal) value function with high probabilities under the same iteration complexity as TD in general cases. Furthermore, we show that the adaptive TD using neural network approximation, with the same width and searching area, can achieve theoretical acceleration when the stochastic semi-gradients decay fast.

        ----

        ## [1424] Benign Underfitting of Stochastic Gradient Descent

        **Authors**: *Tomer Koren, Roi Livni, Yishay Mansour, Uri Sherman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html)

        **Abstract**:

        We study to what extent may stochastic gradient descent (SGD) be understood as a ``conventional'' learning rule that achieves generalization performance by obtaining a good fit to training data. We consider the fundamental stochastic convex optimization framework, where (one pass, $\textit{without}$-replacement) SGD is classically known to minimize the population risk at rate $O(1/\sqrt n)$, and prove that, surprisingly, there exist problem instances where the SGD solution exhibits both empirical risk and generalization gap of $\Omega(1)$. Consequently, it turns out that SGD is not algorithmically stable in $\textit{any}$ sense, and its generalization ability cannot be explained by uniform convergence or any other currently known generalization bound technique for that matter (other than that of its classical analysis). We then continue to analyze the closely related $\textit{with}$-replacement SGD, for which we show that an analogous phenomenon does not occur and prove that its population risk does in fact converge at the optimal rate. Finally, we interpret our main results in the context of without-replacement SGD for finite-sum convex optimization problems, and derive upper and lower bounds for the multi-epoch regime that significantly improve upon previously known results.

        ----

        ## [1425] A Geometric Perspective on Variational Autoencoders

        **Authors**: *Clément Chadebec, Stéphanie Allassonnière*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7bf1dc45f850b8ae1b5a1dd4f475f8b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7bf1dc45f850b8ae1b5a1dd4f475f8b6-Abstract-Conference.html)

        **Abstract**:

        This paper introduces a new interpretation of the Variational Autoencoder framework by taking a fully geometric point of view. We argue that vanilla VAE models unveil naturally a Riemannian structure in their latent space and that taking into consideration those geometrical aspects can lead to better interpolations and an improved generation procedure. This new proposed sampling method consists in sampling from the uniform distribution deriving intrinsically from the learned Riemannian latent space and we show that using this scheme can make a vanilla VAE competitive and even better than more advanced versions on several benchmark datasets. Since generative models are known to be sensitive to the number of training samples we also stress the method's robustness in the low data regime.

        ----

        ## [1426] A Data-Augmentation Is Worth A Thousand Samples: Analytical Moments And Sampling-Free Training

        **Authors**: *Randall Balestriero, Ishan Misra, Yann LeCun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c080cab957edab671ac49ae11e51337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c080cab957edab671ac49ae11e51337-Abstract-Conference.html)

        **Abstract**:

        Data-Augmentation (DA) is known to improve performance across tasks and datasets. We propose a method to theoretically analyze the effect of DA and study questions such as: how many augmented samples are needed to correctly estimate the information encoded by that DA? How does the augmentation policy impact the final parameters of a model? We derive several quantities in close-form, such as the expectation and variance of an image, loss, and model's output under a given DA distribution. Up to our knowledge, we obtain the first explicit regularizer that corresponds to using DA during training for non-trivial transformations such as affine transformations, color jittering, or Gaussian blur. Those derivations open new avenues to quantify the benefits and limitations of DA. For example, given a loss at hand, we find that common DAs require tens of thousands of samples for the loss to be correctly estimated and for the model training to converge. We then show that for a training loss to have reduced variance under DA sampling, the model's saliency map (gradient of the loss with respect to the model's input) must align with the smallest eigenvector of the sample's covariance matrix under the considered DA augmentation; this is exactly the quantity estimated and regularized by TangentProp. Those findings also hint at a possible explanation on why models tend to shift their focus from edges to textures when specific DAs are employed.

        ----

        ## [1427] Effective Adaptation in Multi-Task Co-Training for Unified Autonomous Driving

        **Authors**: *Xiwen Liang, Yangxin Wu, Jianhua Han, Hang Xu, Chunjing Xu, Xiaodan Liang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html)

        **Abstract**:

        Aiming towards a holistic understanding of multiple downstream tasks simultaneously, there is a need for extracting features with better transferability. Though many latest self-supervised pre-training methods have achieved impressive performance on various vision tasks under the prevailing pretrain-finetune paradigm, their generalization capacity to multi-task learning scenarios is yet to be explored. In this paper, we extensively investigate the transfer performance of various types of self-supervised methods, e.g., MoCo and SimCLR, on three downstream tasks, including semantic segmentation, drivable area segmentation, and traffic object detection, on the large-scale driving dataset BDD100K. We surprisingly find that their performances are sub-optimal or even lag far behind the single-task baseline, which may be due to the distinctions of training objectives and architectural design lied in the pretrain-finetune paradigm. To overcome this dilemma as well as avoid redesigning the resource-intensive pre-training stage, we propose a simple yet effective pretrain-adapt-finetune paradigm for general multi-task training, where the off-the-shelf pretrained models can be effectively adapted without increasing the training overhead. During the adapt stage, we utilize learnable multi-scale adapters to dynamically adjust the pretrained model weights supervised by multi-task objectives while leaving the pretrained knowledge untouched. Furthermore, we regard the vision-language pre-training model CLIP as a strong complement to the pretrain-adapt-finetune paradigm and propose a novel adapter named LV-Adapter, which incorporates language priors in the multi-task model via task-specific prompting and alignment between visual and textual features. Our experiments demonstrate that the adapt stage significantly improves the overall performance of those off-the-shelf pretrained models and the contextual features generated by LV-Adapter are of general benefits for downstream tasks.

        ----

        ## [1428] Redundant representations help generalization in wide neural networks

        **Authors**: *Diego Doimo, Aldo Glielmo, Sebastian Goldt, Alessandro Laio*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) defy the classical bias-variance trade-off: adding parameters to a DNN that interpolates its training data will typically improve its generalization performance. Explaining the mechanism behind this ``benign overfitting'' in deep networks remains an outstanding challenge. Here, we study the last hidden layer representations of various state-of-the-art convolutional neural networks and find that  if the last hidden representation is wide enough, its neurons tend to split into groups that carry identical information and differ from each other only by statistically independent noise. The number of such groups increases linearly with the width of the layer, but only if the width is above a critical value. We show that redundant neurons appear only when the training is regularized and the training error is zero.

        ----

        ## [1429] Mean Estimation in High-Dimensional Binary Markov Gaussian Mixture Models

        **Authors**: *Yihan Zhang, Nir Weinberger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c40c5050bd029a3ea7ff8b01412f735-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c40c5050bd029a3ea7ff8b01412f735-Abstract-Conference.html)

        **Abstract**:

        We consider a high-dimensional mean estimation problem over a binary hidden Markov model, which illuminates the interplay between memory in data, sample size, dimension, and signal strength in statistical inference. In this model, an estimator observes $n$ samples of a $d$-dimensional parameter vector $\theta_{*}\in\mathbb{R}^{d}$, multiplied by a random sign $ S_i $ ($1\le i\le n$), and corrupted by isotropic standard Gaussian noise. The sequence of signs $\{S_{i}\}_{i\in[n]}\in\{-1,1\}^{n}$ is drawn from a stationary homogeneous Markov chain with flip probability $\delta\in[0,1/2]$. As $\delta$ varies, this model smoothly interpolates two well-studied models: the Gaussian Location Model for which $\delta=0$ and the Gaussian Mixture Model for which $\delta=1/2$. Assuming that the estimator knows $\delta$, we establish a nearly minimax optimal (up to logarithmic factors) estimation error rate, as a function of $\|\theta_{*}\|,\delta,d,n$. We then provide an upper bound to the case of estimating $\delta$, assuming a (possibly inaccurate) knowledge of $\theta_{*}$. The bound is proved to be tight when $\theta_{*}$ is an accurately known constant. These results are then combined to an algorithm which estimates $\theta_{*}$ with $\delta$ unknown a priori, and theoretical guarantees on its error are stated.

        ----

        ## [1430] Rate-Distortion Theoretic Bounds on Generalization Error for Distributed Learning

        **Authors**: *Milad Sefidgaran, Romain Chor, Abdellatif Zaidi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c61aa6a4dcf12042fe12a31d49d6390-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c61aa6a4dcf12042fe12a31d49d6390-Abstract-Conference.html)

        **Abstract**:

        In this paper, we use tools from rate-distortion theory to establish new upper bounds on the generalization error of statistical distributed learning algorithms. Specifically, there are $K$ clients whose individually chosen models are aggregated by a central server. The bounds depend on the compressibility of each client's algorithm while keeping other clients' algorithms un-compressed, and leveraging the fact that small changes in each local model change the aggregated model by a factor of only $1/K$. Adopting a recently proposed approach by Sefidgaran et al., and extending it suitably to the distributed setting, enables smaller rate-distortion terms which are shown to translate into tighter generalization bounds. The bounds are then applied to the distributed support vector machines (SVM), suggesting that the generalization error of the distributed setting decays faster than that of the centralized one with a factor of $\mathcal{O}(\sqrt{\log(K)/K})$. This finding is validated also experimentally. A similar conclusion is obtained for a multiple-round federated learning setup where each client uses stochastic gradient Langevin dynamics (SGLD).

        ----

        ## [1431] Online Frank-Wolfe with Arbitrary Delays

        **Authors**: *Yuanyu Wan, Wei-Wei Tu, Lijun Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c799b09cc40973ceaa47da50131dc63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c799b09cc40973ceaa47da50131dc63-Abstract-Conference.html)

        **Abstract**:

        The online Frank-Wolfe (OFW) method has gained much popularity for online convex optimization due to its projection-free property. Previous studies show that OFW can attain an $O(T^{3/4})$ regret bound for convex losses and an $O(T^{2/3})$ regret bound for strongly convex losses. However, they assume that each gradient queried by OFW is revealed immediately, which may not hold in practice and limits the application of OFW. To address this limitation, we propose a delayed variant of OFW, which allows gradients to be delayed by arbitrary rounds. The main idea is to perform an update similar to OFW after receiving any delayed gradient, and play the latest decision for each round. Despite its simplicity, we prove that our delayed variant of OFW is able to achieve an $O(T^{3/4}+dT^{1/4})$ regret bound for convex losses and an $O(T^{2/3}+d\log T)$ regret bound for strongly convex losses, where $d$ is the maximum delay. This is quite surprising since under a relatively large amount of delay (e.g., $d=O(\sqrt{T})$ for convex losses and $d=O(T^{2/3}/\log T)$ for strongly convex losses), the delayed variant of OFW enjoys the same regret bound as that of the original OFW.

        ----

        ## [1432] Isometric 3D Adversarial Examples in the Physical World

        **Authors**: *Yibo Miao, Yinpeng Dong, Jun Zhu, Xiao-Shan Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c818dd40651b420873af70b8a790e3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c818dd40651b420873af70b8a790e3f-Abstract-Conference.html)

        **Abstract**:

        Recently, several attempts have demonstrated that 3D deep learning models are as vulnerable to adversarial example attacks as 2D models. However, these methods are still far from stealthy and suffer from severe performance degradation in the physical world. Although 3D data is highly structured, it is difficult to bound the perturbations with simple metrics in the Euclidean space. In this paper, we propose a novel $\epsilon$-isometric ($\epsilon$-ISO) attack method to generate natural and robust 3D adversarial examples in the physical world by considering the geometric properties of 3D objects and the invariance to physical transformations. For naturalness, we constrain the adversarial example and the original one to be $\epsilon$-isometric by adopting the Gaussian curvature as the surrogate metric under a theoretical analysis. For robustness under physical transformations, we propose a maxima over transformation (MaxOT) method to actively search for the most difficult transformations rather than random ones to make the generated adversarial example more robust in the physical world. Extensive experiments on typical point cloud recognition models validate that our approach can improve the attack success rate and naturalness of the generated 3D adversarial examples than the state-of-the-art attack methods.

        ----

        ## [1433] Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes

        **Authors**: *Alessio Mazzetto, Cristina Menghini, Andrew Yuan, Eli Upfal, Stephen H. Bach*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7c9b9ecc92a679f1b6c76ed8f99f2636-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7c9b9ecc92a679f1b6c76ed8f99f2636-Abstract-Conference.html)

        **Abstract**:

        We develop a rigorous mathematical analysis of zero-shot learning with attributes. In this setting, the goal is to label novel classes with no training data, only detectors for attributes and a description of how those attributes are correlated with the target classes, called the class-attribute matrix. We develop the first non-trivial lower bound on the worst-case error of the best map from attributes to classes for this setting, even with perfect attribute detectors. The lower bound characterizes the theoretical intrinsic difficulty of the zero-shot problem based on the available information---the class-attribute matrix---and the bound is practically computable from it. Our lower bound is tight, as we show that we can always find a randomized map from attributes to classes whose expected error is upper bounded by the value of the lower bound. We show that our analysis can be predictive of how standard zero-shot methods behave in practice, including which classes will likely be confused with others.

        ----

        ## [1434] Graph Neural Networks with Adaptive Readouts

        **Authors**: *David Buterez, Jon Paul Janet, Steven J. Kiddle, Dino Oglic, Pietro Liò*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html)

        **Abstract**:

        An effective aggregation of node features into a graph-level representation via readout functions is an essential step in numerous learning tasks involving graph neural networks. Typically, readouts are simple and non-adaptive functions designed such that the resulting hypothesis space is permutation invariant. Prior work on deep sets indicates that such readouts might require complex node embeddings that can be difficult to learn via standard neighborhood aggregation schemes. Motivated by this, we investigate the potential of adaptive readouts given by neural networks that do not necessarily give rise to permutation invariant hypothesis spaces. We argue that in some problems such as binding affinity prediction where molecules are typically presented in a canonical form it might be possible to relax the constraints on permutation invariance of the hypothesis space and learn a more effective model of the affinity by employing an adaptive readout function. Our empirical results demonstrate the effectiveness of neural readouts on more than 40 datasets spanning different domains and graph characteristics. Moreover, we observe a consistent improvement over standard readouts (i.e., sum, max, and mean) relative to the number of neighborhood aggregation iterations and different convolutional operators.

        ----

        ## [1435] Robust Imitation of a Few Demonstrations with a Backwards Model

        **Authors**: *Jung Yeon Park, Lawson L. S. Wong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7ce5da35e01cfa8d303c2dc71e61a470-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7ce5da35e01cfa8d303c2dc71e61a470-Abstract-Conference.html)

        **Abstract**:

        Behavior cloning of expert demonstrations can speed up learning optimal policies in a more sample-efficient way over reinforcement learning. However, the policy cannot extrapolate well to unseen states outside of the demonstration data, creating covariate shift (agent drifting away from demonstrations) and compounding errors. In this work, we tackle this issue by extending the region of attraction around the demonstrations so that the agent can learn how to get back onto the demonstrated trajectories if it veers off-course. We train a generative backwards dynamics model and generate short imagined trajectories from states in the demonstrations. By imitating both demonstrations and these model rollouts, the agent learns the demonstrated paths and how to get back onto these paths. With optimal or near-optimal demonstrations, the learned policy will be both optimal and robust to deviations, with a wider region of attraction. On continuous control domains, we evaluate the robustness when starting from different initial states unseen in the demonstration data. While both our method and other imitation learning baselines can successfully solve the tasks for initial states in the training distribution, our method exhibits considerably more robustness to different initial states.

        ----

        ## [1436] Communication Efficient Distributed Learning for Kernelized Contextual Bandits

        **Authors**: *Chuanhao Li, Huazheng Wang, Mengdi Wang, Hongning Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d1043b688002734b49b766cc2fc478d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d1043b688002734b49b766cc2fc478d-Abstract-Conference.html)

        **Abstract**:

        We tackle the communication efficiency challenge of learning kernelized contextual bandits in a distributed setting. Despite the recent advances in communication-efficient distributed bandit learning, existing solutions are restricted to simple models like multi-armed bandits and linear bandits, which hamper their practical utility. In this paper, instead of assuming the existence of a linear reward mapping from the features to the expected rewards, we consider non-linear reward mappings, by letting agents collaboratively search in a reproducing kernel Hilbert space (RKHS). This introduces significant challenges in communication efficiency as distributed kernel learning requires the transfer of raw data, leading to a communication cost that grows linearly w.r.t. time horizon $T$. We addresses this issue by equipping all agents to communicate via a common Nystr\"{o}m embedding that gets updated adaptively as more data points are collected. We rigorously proved that our algorithm can attain sub-linear rate in both regret and communication cost.

        ----

        ## [1437] Trust Region Policy Optimization with Optimal Transport Discrepancies: Duality and Algorithm for Continuous Actions

        **Authors**: *Antonio Terpin, Nicolas Lanzetti, Batuhan Yardim, Florian Dörfler, Giorgia Ramponi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d3298e48220b289318b533a848ea069-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d3298e48220b289318b533a848ea069-Abstract-Conference.html)

        **Abstract**:

        Policy Optimization (PO) algorithms have been proven particularly suited to handle the high-dimensionality of real-world continuous control tasks. In this context, Trust Region Policy Optimization methods represent a popular approach to stabilize the policy updates. These usually rely on the Kullback-Leibler (KL) divergence to limit the change in the policy. The Wasserstein distance represents a natural alternative, in place of the KL divergence, to define trust regions or to regularize the objective function. However, state-of-the-art works either resort to its approximations or do not provide an algorithm for continuous state-action spaces, reducing the applicability of the method.In this paper, we explore optimal transport discrepancies (which include the Wasserstein distance) to define trust regions, and we propose a novel algorithm - Optimal Transport Trust Region Policy Optimization (OT-TRPO) - for continuous state-action spaces. We circumvent the infinite-dimensional optimization problem for PO by providing a one-dimensional dual reformulation for which strong duality holds.We then analytically derive the optimal policy update given the solution of the dual problem. This way, we bypass the computation of optimal transport costs and of optimal transport maps, which we implicitly characterize by solving the dual formulation.Finally, we provide an experimental evaluation of our approach across various control tasks. Our results show that optimal transport discrepancies can offer an advantage over state-of-the-art approaches.

        ----

        ## [1438] ELIAS: End-to-End Learning to Index and Search in Large Output Spaces

        **Authors**: *Nilesh Gupta, Patrick H. Chen, Hsiang-Fu Yu, Cho-Jui Hsieh, Inderjit S. Dhillon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d4f98f916494121aca3da02e36a4d18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d4f98f916494121aca3da02e36a4d18-Abstract-Conference.html)

        **Abstract**:

        Extreme multi-label classification (XMC) is a popular framework for solving many real-world problems that require accurate prediction from a very large number of potential output choices. A popular approach for dealing with the large label space is to arrange the labels into a shallow tree-based index and then learn an ML model to efficiently search this index via beam search. Existing methods initialize the tree index by clustering the label space into a few mutually exclusive clusters based on pre-defined features and keep it fixed throughout the training procedure. This approach results in a sub-optimal indexing structure over the label space and limits the search performance to the quality of choices made during the initialization of the index. In this paper, we propose a novel method ELIAS which relaxes the tree-based index to a specialized weighted graph-based index which is learned end-to-end with the final task objective. More specifically, ELIAS models the discrete cluster-to-label assignments in the existing tree-based index as soft learnable parameters that are learned jointly with the rest of the ML model. ELIAS achieves state-of-the-art performance on several large-scale extreme classification benchmarks with millions of labels. In particular, ELIAS can be up to 2.5% better at precision@$1$ and up to 4% better at recall@$100$ than existing XMC methods. A PyTorch implementation of ELIAS along with other resources is available at https://github.com/nilesh2797/ELIAS.

        ----

        ## [1439] GStarX: Explaining Graph Neural Networks with Structure-Aware Cooperative Games

        **Authors**: *Shichang Zhang, Yozen Liu, Neil Shah, Yizhou Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d53575463291ea6b5a23cf6e571f59b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d53575463291ea6b5a23cf6e571f59b-Abstract-Conference.html)

        **Abstract**:

        Explaining machine learning models is an important and increasingly popular area of research interest. The Shapley value from game theory has been proposed as a prime approach to compute feature importance towards model predictions on images, text, tabular data, and recently graph neural networks (GNNs) on graphs. In this work, we revisit the appropriateness of the Shapley value for GNN explanation, where the task is to identify the most important subgraph and constituent nodes for GNN predictions. We claim that the Shapley value is a non-ideal choice for graph data because it is by definition not structure-aware. We propose a Graph Structure-aware eXplanation (GStarX) method to leverage the critical graph structure information to improve the explanation. Specifically, we define a scoring function based on a new structure-aware value from the cooperative game theory proposed by Hamiache and Navarro (HN). When used to score node importance, the HN value utilizes graph structures to attribute cooperation surplus between neighbor nodes, resembling message passing in GNNs, so that node importance scores reflect not only the node feature importance, but also the node structural roles. We demonstrate that GStarX produces qualitatively more intuitive explanations, and quantitatively improves explanation fidelity over strong baselines on chemical graph property prediction and text graph sentiment classification. Code: https://github.com/ShichangZh/GStarX

        ----

        ## [1440] Bridging the Gap from Asymmetry Tricks to Decorrelation Principles in Non-contrastive Self-supervised Learning

        **Authors**: *Kang-Jun Liu, Masanori Suganuma, Takayuki Okatani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html)

        **Abstract**:

        Recent non-contrastive methods for self-supervised representation learning show promising performance. While they are attractive since they do not need negative samples, it necessitates some mechanism to avoid collapsing into a trivial solution. Currently, there are two approaches to collapse prevention. One uses an asymmetric architecture on a joint embedding of input, e.g., BYOL and SimSiam, and the other imposes decorrelation criteria on the same joint embedding, e.g., Barlow-Twins and VICReg. The latter methods have theoretical support from information theory as to why they can learn good representation. However, it is not fully understood why the former performs equally well. In this paper, focusing on BYOL/SimSiam, which uses the stop-gradient and a predictor as asymmetric tricks, we present a novel interpretation of these tricks; they implicitly impose a constraint that encourages feature decorrelation similar to Barlow-Twins/VICReg. We then present a novel non-contrastive method, which replaces the stop-gradient in BYOL/SimSiam with the derived constraint; the method empirically shows comparable performance to the above SOTA methods in the standard benchmark test using ImageNet. This result builds a bridge from BYOL/SimSiam to the decorrelation-based methods, contributing to demystifying their secrets.

        ----

        ## [1441] Lipschitz Bandits with Batched Feedback

        **Authors**: *Yasong Feng, Zengfeng Huang, Tianyu Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d6ab81bfaa3a3ea45833e21907453c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d6ab81bfaa3a3ea45833e21907453c3-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study Lipschitz bandit problems with batched feedback, where the expected reward is Lipschitz and the reward observations are communicated to the player in batches. We introduce a novel landscape-aware algorithm, called Batched Lipschitz Narrowing (BLiN), that optimally solves this problem. Specifically, we show that for a $T$-step problem with Lipschitz reward of zooming dimension $d_z$, our algorithm achieves theoretically optimal (up to logarithmic factors) regret rate $\widetilde{\mathcal{O}}\left(T^{\frac{d_z+1}{d_z+2}}\right)$ using only $ \mathcal{O} \left( \log\log T\right) $ batches. We also provide complexity analysis for this problem. Our theoretical lower bound implies that $\Omega(\log\log T)$ batches are necessary for any algorithm to achieve the optimal regret. Thus, BLiN achieves optimal regret rate using minimal communication.

        ----

        ## [1442] Improving Barely Supervised Learning by Discriminating Unlabeled Samples with Super-Class

        **Authors**: *Guan Gui, Zhen Zhao, Lei Qi, Luping Zhou, Lei Wang, Yinghuan Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7d90c28e7820709792d969211815a2b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7d90c28e7820709792d969211815a2b3-Abstract-Conference.html)

        **Abstract**:

        In semi-supervised learning (SSL),  a common practice is to learn consistent information from unlabeled data and discriminative information from labeled data to ensure both the immutability and the separability of the classification model.  Existing SSL methods  suffer from failures in barely-supervised learning (BSL), where only one or two labels per class are available, as the insufficient labels cause the discriminative information being difficult or even infeasible to learn. To bridge this gap, we investigate a simple yet effective way to leverage unlabeled samples for discriminative learning, and propose a novel discriminative information learning module to benefit model training. Specifically, we formulate the learning objective of discriminative information at the super-class level and dynamically assign different classes into different super-classes based on  model performance improvement. On top of this on-the-fly process, we further propose a distribution-based loss to learn discriminative information by utilizing the similarity relationship between samples and super-classes. It encourages the  unlabeled samples to stay closer to the distribution of their corresponding super-class than those of others. Such a constraint is softer than the direct assignment of pseudo labels, while the latter could be very noisy in BSL. We compare our method with state-of-the-art SSL and BSL methods through extensive experiments on standard SSL benchmarks. Our method can achieve superior results, \eg, an average accuracy of 76.76\% on CIFAR-10 with merely 1 label per class.

        ----

        ## [1443] Low-Rank Modular Reinforcement Learning via Muscle Synergy

        **Authors**: *Heng Dong, Tonghan Wang, Jiayuan Liu, Chongjie Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7da6005a8d6942e8b328357da2872aed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7da6005a8d6942e8b328357da2872aed-Abstract-Conference.html)

        **Abstract**:

        Modular Reinforcement Learning (RL) decentralizes the control of multi-joint robots by learning policies for each actuator. Previous work on modular RL has proven its ability to control morphologically different agents with a shared actuator policy. However, with the increase in the Degree of Freedom (DoF) of robots, training a morphology-generalizable modular controller becomes exponentially difficult. Motivated by the way the human central nervous system controls numerous muscles, we propose a Synergy-Oriented LeARning (SOLAR) framework that exploits the redundant nature of DoF in robot control. Actuators are grouped into synergies by an unsupervised learning method, and a synergy action is learned to control multiple actuators in synchrony. In this way, we achieve a low-rank control at the synergy level. We extensively evaluate our method on a variety of robot morphologies, and the results show its superior efficiency and generalizability, especially on robots with a large DoF like Humanoids++ and UNIMALs.

        ----

        ## [1444] Neural Temporal Walks: Motif-Aware Representation Learning on Continuous-Time Dynamic Graphs

        **Authors**: *Ming Jin, Yuan-Fang Li, Shirui Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7dadc855cef7494d5d956a8d28add871-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7dadc855cef7494d5d956a8d28add871-Abstract-Conference.html)

        **Abstract**:

        Continuous-time dynamic graphs naturally abstract many real-world systems, such as social and transactional networks. While the research on continuous-time dynamic graph representation learning has made significant advances recently, neither graph topological properties nor temporal dependencies have been well-considered and explicitly modeled in capturing dynamic patterns. In this paper, we introduce a new approach, Neural Temporal Walks (NeurTWs), for representation learning on continuous-time dynamic graphs. By considering not only time constraints but also structural and tree traversal properties, our method conducts spatiotemporal-biased random walks to retrieve a set of representative motifs, enabling temporal nodes to be characterized effectively. With a component based on neural ordinary differential equations, the extracted motifs allow for irregularly-sampled temporal nodes to be embedded explicitly over multiple different interaction time intervals, enabling the effective capture of the underlying spatiotemporal dynamics. To enrich supervision signals, we further design a harder contrastive pretext task for model optimization. Our method demonstrates overwhelming superiority under both transductive and inductive settings on six real-world datasets.

        ----

        ## [1445] Understanding Benign Overfitting in Gradient-Based Meta Learning

        **Authors**: *Lisha Chen, Songtao Lu, Tianyi Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7db3470825421b6a7e52d95fb00de62e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7db3470825421b6a7e52d95fb00de62e-Abstract-Conference.html)

        **Abstract**:

        Meta learning has demonstrated tremendous success in few-shot learning with  limited supervised data. In those settings, the meta model is usually overparameterized. While the conventional statistical learning theory suggests that overparameterized models tend to overfit, empirical evidence reveals that overparameterized meta learning methods still work well -- a phenomenon often called ``benign overfitting.'' To understand this phenomenon, we focus on the meta learning settings with a challenging bilevel structure that we term the gradient-based meta learning, and analyze its generalization performance under an overparameterized meta linear regression model. While our analysis uses the relatively tractable linear models, our theory contributes to understanding the delicate interplay among data heterogeneity, model adaptation and benign overfitting in gradient-based meta learning tasks. We corroborate our theoretical claims through numerical simulations.

        ----

        ## [1446] Generative Neural Articulated Radiance Fields

        **Authors**: *Alexander W. Bergman, Petr Kellnhofer, Yifan Wang, Eric R. Chan, David B. Lindell, Gordon Wetzstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7dbafa7d2051218f364c9a38ef1150de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7dbafa7d2051218f364c9a38ef1150de-Abstract-Conference.html)

        **Abstract**:

        Unsupervised learning of 3D-aware generative adversarial networks (GANs) using only collections of single-view 2D photographs has very recently made much progress. These 3D GANs, however, have not been demonstrated for human bodies and the generated radiance fields of existing frameworks are not directly editable, limiting their applicability in downstream tasks. We propose a solution to these challenges by developing a 3D GAN framework that learns to generate radiance fields of human bodies or faces in a canonical pose and warp them using an explicit deformation field into a desired body pose or facial expression. Using our framework, we demonstrate the first high-quality radiance field generation results for human bodies. Moreover, we show that our deformation-aware training procedure significantly improves the quality of generated bodies or faces when editing their poses or facial expressions compared to a 3D GAN that is not trained with explicit deformations.

        ----

        ## [1447] AutoMS: Automatic Model Selection for Novelty Detection with Error Rate Control

        **Authors**: *Yifan Zhang, Haiyan Jiang, Haojie Ren, Changliang Zou, Dejing Dou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7dced224614f3c50b1626473f48312bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7dced224614f3c50b1626473f48312bf-Abstract-Conference.html)

        **Abstract**:

        Given an unsupervised novelty detection task on a new dataset, how can we automatically select a ''best'' detection model while simultaneously controlling the error rate of the best model? For novelty detection analysis, numerous detectors have been proposed to detect outliers on a new unseen dataset based on a score function trained on available clean data. However, due to the absence of labeled data for model evaluation and comparison, there is a lack of systematic approaches that are able to select a ''best'' model/detector (i.e., the algorithm as well as its hyperparameters) and achieve certain error rate control simultaneously. In this paper, we introduce a unified data-driven procedure to address this issue. The key idea is to maximize the number of detected outliers while controlling the false discovery rate (FDR) with the help of Jackknife prediction. We establish non-asymptotic bounds for the false discovery proportions and show that the proposed procedure yields valid FDR control under some mild conditions. Numerical experiments on both synthetic and real data validate the theoretical results and demonstrate the effectiveness of our proposed AutoMS method. The code is available at https://github.com/ZhangYifan1996/AutoMS.

        ----

        ## [1448] GALOIS: Boosting Deep Reinforcement Learning via Generalizable Logic Synthesis

        **Authors**: *Yushi Cao, Zhiming Li, Tianpei Yang, Hao Zhang, Yan Zheng, Yi Li, Jianye Hao, Yang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html)

        **Abstract**:

        Despite achieving superior performance in human-level control problems, unlike humans, deep reinforcement learning (DRL) lacks high-order intelligence (e.g., logic deduction and reuse), thus it behaves ineffectively than humans regarding learning and generalization in complex problems. Previous works attempt to directly synthesize a white-box logic program as the DRL policy, manifesting logic-driven behaviors. However, most synthesis methods are built on imperative or declarative programming, and each has a distinct limitation, respectively. The former ignores the cause-effect logic during synthesis, resulting in low generalizability across tasks. The latter is strictly proof-based, thus failing to synthesize programs with complex hierarchical logic. In this paper, we combine the above two paradigms together and propose a novel Generalizable Logic Synthesis (GALOIS) framework to synthesize hierarchical and strict cause-effect logic programs. GALOIS leverages the program sketch and defines a new sketch-based hybrid program language for guiding the synthesis. Based on that, GALOIS proposes a sketch-based program synthesis method to automatically generate white-box programs with generalizable and interpretable cause-effect logic. Extensive evaluations on various decision-making tasks with complex logic demonstrate the superiority of GALOIS over mainstream baselines regarding the asymptotic performance, generalizability, and great knowledge reusability across different environments.

        ----

        ## [1449] Faster Deep Reinforcement Learning with Slower Online Network

        **Authors**: *Kavosh Asadi, Rasool Fakoor, Omer Gottesman, Taesup Kim, Michael L. Littman, Alexander J. Smola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7dfa77fcef807c9a078b58fd619ad897-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7dfa77fcef807c9a078b58fd619ad897-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning algorithms often use two networks for value function optimization: an online network, and a target network that tracks the online network with some delay. Using two separate networks enables the agent to hedge against issues that arise when performing bootstrapping. In this paper we endow two popular deep reinforcement learning algorithms, namely DQN and Rainbow, with updates that incentivize the online network to remain in the proximity of the target network. This improves the robustness of deep reinforcement learning in presence of noisy updates. The resultant agents, called DQN Pro and Rainbow Pro, exhibit significant performance improvements over their original counterparts on the Atari benchmark demonstrating the effectiveness of this simple idea in deep reinforcement learning. The code for our paper is available here: Github.com/amazon-research/fast-rl-with-slow-updates.

        ----

        ## [1450] Learn to Match with No Regret: Reinforcement Learning in Markov Matching Markets

        **Authors**: *Yifei Min, Tianhao Wang, Ruitu Xu, Zhaoran Wang, Michael I. Jordan, Zhuoran Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html)

        **Abstract**:

        We study a Markov matching market involving a planner and a set of strategic agents on the two sides of the market.At each step, the agents are presented with a dynamical context, where the contexts determine the utilities. The planner controls the transition of the contexts to maximize the cumulative social welfare, while the agents aim to find a myopic stable matching at each step. Such a setting captures a range of applications including ridesharing platforms. We formalize the problem by proposing a reinforcement learning framework that integrates optimistic value iteration with maximum weight matching. The proposed algorithm addresses the coupled challenges of sequential exploration, matching stability, and function approximation. We prove that the algorithm achieves sublinear regret.

        ----

        ## [1451] Efficient Frameworks for Generalized Low-Rank Matrix Bandit Problems

        **Authors**: *Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e0dc9ccba0f1333be13a3f9dc2b3138-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e0dc9ccba0f1333be13a3f9dc2b3138-Abstract-Conference.html)

        **Abstract**:

        In the stochastic contextual low-rank matrix bandit problem, the expected reward of an action is given by the inner product between the action's feature matrix and some fixed, but initially unknown $d_1$ by $d_2$ matrix $\Theta^*$ with rank $r \ll \{d_1, d_2\}$, and an agent sequentially takes actions based on past experience to maximize the cumulative reward. In this paper, we study the generalized low-rank matrix bandit problem, which has been recently proposed in \cite{lu2021low} under the Generalized Linear Model (GLM) framework. To overcome the computational infeasibility and theoretical restrain of existing algorithms on this problem, we first propose the G-ESTT framework that modifies the idea from \cite{jun2019bilinear} by using Stein's method on the subspace estimation and then leverage the estimated subspaces via a regularization idea. Furthermore, we remarkably improve the efficiency of G-ESTT by using a novel exclusion idea on the estimated subspace instead, and propose the G-ESTS framework. We also show that both of our methods are the first algorithm to achieve the optimal $\tilde{O}((d_1+d_2)r\sqrt{T})$ bound of regret presented in \cite{lu2021low} up to logarithm terms under some mild conditions, which improves upon the current regret of $\tilde{O}((d_1+d_2)^{3/2} \sqrt{rT})$~\citep{lu2021low}. For completeness, we conduct experiments to illustrate that our proposed algorithms, especially G-ESTS, are also computationally tractable and consistently outperform other state-of-the-art (generalized) linear matrix bandit methods based on a suite of simulations.

        ----

        ## [1452] A Projection-free Algorithm for Constrained Stochastic Multi-level Composition Optimization

        **Authors**: *Tesi Xiao, Krishnakumar Balasubramanian, Saeed Ghadimi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html)

        **Abstract**:

        We propose a projection-free conditional gradient-type algorithm for smooth stochastic multi-level composition optimization, where the objective function is a nested composition of $T$ functions and the constraint set is a closed convex set. Our algorithm assumes access to noisy evaluations of the functions and their gradients, through a stochastic first-order oracle satisfying certain standard unbiasedness and second-moment assumptions. We show that the number of calls to the stochastic first-order oracle and the linear-minimization oracle required by the proposed algorithm, to obtain an $\epsilon$-stationary solution, are of order $\mathcal{O}_T(\epsilon^{-2})$ and $\mathcal{O}_T(\epsilon^{-3})$ respectively, where $\mathcal{O}_T$ hides constants in $T$. Notably, the dependence of these complexity bounds on $\epsilon$ and $T$ are separate in the sense that changing one does not impact the dependence of the bounds on the other. For the case of $T=1$, we also provide a high-probability convergence result that depends poly-logarithmically on the inverse confidence level. Moreover, our algorithm is parameter-free and does not require any (increasing) order of mini-batches to converge unlike the common practice in the analysis of stochastic conditional gradient-type algorithms.

        ----

        ## [1453] Green Hierarchical Vision Transformer for Masked Image Modeling

        **Authors**: *Lang Huang, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Toshihiko Yamasaki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e487c72fce6e45879a78ee0872d991d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e487c72fce6e45879a78ee0872d991d-Abstract-Conference.html)

        **Abstract**:

        We present an efficient approach for Masked Image Modeling (MIM) with hierarchical Vision Transformers (ViTs), allowing the hierarchical ViTs to discard masked patches and operate only on the visible ones. Our approach consists of three key designs. First, for window attention, we propose a Group Window Attention scheme following the Divide-and-Conquer strategy. To mitigate the quadratic complexity of the self-attention w.r.t. the number of patches, group attention encourages a uniform partition that visible patches within each local window of arbitrary size can be grouped with equal size, where masked self-attention is then performed within each group. Second, we further improve the grouping strategy via the Dynamic Programming algorithm to minimize the overall computation cost of the attention on the grouped patches. Third, as for the convolution layers, we convert them to the Sparse Convolution that works seamlessly with the sparse data, i.e., the visible patches in MIM. As a result, MIM can now work on most, if not all, hierarchical ViTs in a green and efficient way. For example, we can train the hierarchical ViTs, e.g., Swin Transformer and Twins Transformer, about 2.7$\times$ faster and reduce the GPU memory usage by 70%, while still enjoying competitive performance on ImageNet classification and the superiority on downstream COCO object detection benchmarks.

        ----

        ## [1454] Simultaneous Missing Value Imputation and Structure Learning with Groups

        **Authors**: *Pablo Morales-Alvarez, Wenbo Gong, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html)

        **Abstract**:

        Learning structures between groups of variables from data with missing values is an important task in the real world, yet difficult to solve. One typical scenario is discovering the structure among topics in the education domain to identify learning pathways. Here, the observations are student performances for questions under each topic which contain missing values. However, most existing methods focus on learning structures between a few individual variables from the complete data. In this work, we propose VISL, a novel scalable structure learning approach that can simultaneously infer structures between groups of variables under missing data and perform missing value imputations with deep learning. Particularly, we propose a generative model with a structured latent space and a graph neural network-based architecture, scaling to a large number of variables. Empirically, we conduct extensive experiments on synthetic, semi-synthetic, and real-world education data sets. We show improved performances on both imputation and structure learning accuracy compared to popular and recent approaches.

        ----

        ## [1455] Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation

        **Authors**: *Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, Gilles Louppe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e6288bfb68182db7d6e328b0aefa89a-Abstract-Conference.html)

        **Abstract**:

        Modern approaches for simulation-based inference build upon deep learning surrogates to enable approximate Bayesian inference with computer simulators. In practice, the estimated posteriors' computational faithfulness is, however, rarely guaranteed. For example, Hermans et al., 2021 have shown that current simulation-based inference algorithms can produce posteriors that are overconfident, hence risking false inferences. In this work, we introduce Balanced Neural Ratio Estimation (BNRE), a variation of the NRE algorithm designed to produce posterior approximations that tend to be more conservative, hence improving their reliability, while sharing the same Bayes optimal solution. We achieve this by enforcing a balancing condition that increases the quantified uncertainty in low simulation budget regimes while still converging to the exact posterior as the budget increases. We provide theoretical arguments showing that BNRE tends to produce posterior surrogates that are more conservative than NRE's. We evaluate BNRE on a wide variety of tasks and show that it produces conservative posterior surrogates on all tested benchmarks and simulation budgets. Finally, we emphasize that BNRE is straightforward to implement over NRE and does not introduce any computational overhead.

        ----

        ## [1456] Interaction Modeling with Multiplex Attention

        **Authors**: *Fan-Yun Sun, Isaac Kauvar, Ruohan Zhang, Jiachen Li, Mykel J. Kochenderfer, Jiajun Wu, Nick Haber*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e6361a5d73a8fab093dd8453e0b106f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e6361a5d73a8fab093dd8453e0b106f-Abstract-Conference.html)

        **Abstract**:

        Modeling multi-agent systems requires understanding how agents interact. Such systems are often difficult to model because they can involve a variety of types of interactions that layer together to drive rich social behavioral dynamics. Here we introduce a method for accurately modeling multi-agent systems. We present Interaction Modeling with Multiplex Attention (IMMA), a forward prediction model that uses a multiplex latent graph to represent multiple independent types of interactions and attention to account for relations of different strengths. We also introduce Progressive Layer Training, a training strategy for this architecture. We show that our approach outperforms state-of-the-art models in trajectory forecasting and relation inference, spanning three multi-agent scenarios: social navigation, cooperative task achievement, and team sports. We further demonstrate that our approach can improve zero-shot generalization and allows us to probe how different interactions impact agent behavior.

        ----

        ## [1457] Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations

        **Authors**: *Steffen Schotthöfer, Emanuele Zangrando, Jonas Kusch, Gianluca Ceruti, Francesco Tudisco*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e98b00eeafcdaeb0c5661fb9355be3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e98b00eeafcdaeb0c5661fb9355be3a-Abstract-Conference.html)

        **Abstract**:

        Neural networks have achieved tremendous success in a large variety of applications. However, their memory footprint and computational demand can render them impractical in application settings with limited hardware or energy resources. In this work, we propose a novel algorithm to find efficient low-rank subnetworks. Remarkably, these subnetworks are determined and adapted already during the training phase and the overall time and memory resources required by both training and evaluating them is significantly reduced. The main idea is to restrict the weight matrices to a low-rank manifold and to update the low-rank factors rather than the full matrix during training. To derive training updates that are restricted to the prescribed manifold, we employ techniques from dynamic model order reduction for matrix differential equations. Moreover, our method automatically and dynamically adapts the ranks during training to achieve a desired approximation accuracy.The efficiency of the proposed method is demonstrated through a variety of numerical experiments on fully-connected and convolutional networks.

        ----

        ## [1458] Causality-driven Hierarchical Structure Discovery for Reinforcement Learning

        **Authors**: *Shaohui Peng, Xing Hu, Rui Zhang, Ke Tang, Jiaming Guo, Qi Yi, Ruizhi Chen, Xishan Zhang, Zidong Du, Ling Li, Qi Guo, Yunji Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7e9fbd01b3084956dd8a070c7bf30bad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7e9fbd01b3084956dd8a070c7bf30bad-Abstract-Conference.html)

        **Abstract**:

        Hierarchical reinforcement learning (HRL) has been proven to be effective for tasks with sparse rewards, for it can improve the agent's exploration efficiency by discovering high-quality hierarchical structures (e.g., subgoals or options). However, automatically discovering high-quality hierarchical structures is still a great challenge. Previous HRL methods can only find the hierarchical structures in simple environments, as they are mainly achieved through the randomness of agent's policies during exploration. In complicated environments, such a randomness-driven exploration paradigm can hardly discover high-quality hierarchical structures because of the low exploration efficiency. In this paper, we propose CDHRL, a causality-driven hierarchical reinforcement learning framework, to build high-quality hierarchical structures efficiently in complicated environments. The key insight is that the causalities among environment variables are naturally fit for modeling reachable subgoals and their dependencies; thus, the causality is suitable to be the guidance in building high-quality hierarchical structures. Roughly, we build the hierarchy of subgoals based on causality autonomously, and utilize the subgoal-based policies to unfold further causality efficiently. Therefore, CDHRL leverages a causality-driven discovery instead of a randomness-driven exploration for high-quality hierarchical structure construction. The results in two complex environments, 2D-Minecraft and Eden, show that CDHRL can discover high-quality hierarchical structures and significantly enhance exploration efficiency.

        ----

        ## [1459] Pay attention to your loss : understanding misconceptions about Lipschitz neural networks

        **Authors**: *Louis Béthune, Thibaut Boissin, Mathieu Serrurier, Franck Mamalet, Corentin Friedrich, Alberto González-Sanz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7eb3d8ae592966543170a65e6b698828-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7eb3d8ae592966543170a65e6b698828-Abstract-Conference.html)

        **Abstract**:

        Lipschitz constrained networks have gathered considerable attention in the deep learning community, with usages ranging from Wasserstein distance estimation to the training of certifiably robust classifiers. However they remain commonly considered as less accurate, and their properties in learning are still not fully understood. In this paper we clarify the matter: when it comes to classification 1-Lipschitz neural networks enjoy several advantages over their unconstrained counterpart. First, we show that these networks are as accurate as classical ones, and can fit arbitrarily difficult boundaries. Then, relying on a robustness metric that reflects operational needs we characterize the most robust classifier: the WGAN discriminator. Next, we show that 1-Lipschitz neural networks generalize well under milder assumptions. Finally, we show that hyper-parameters of the loss are crucial for controlling the accuracy-robustness trade-off. We conclude that they exhibit appealing properties to pave the way toward provably accurate, and provably robust neural networks.

        ----

        ## [1460] Large-Scale Retrieval for Reinforcement Learning

        **Authors**: *Peter C. Humphreys, Arthur Guez, Olivier Tieleman, Laurent Sifre, Theophane Weber, Timothy P. Lillicrap*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7eca17ef54789b0663cab421f2e9dbf5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7eca17ef54789b0663cab421f2e9dbf5-Abstract-Conference.html)

        **Abstract**:

        Effective decision making involves flexibly relating past experiences and relevant contextual information to a novel situation. In deep reinforcement learning (RL), the dominant paradigm is for an agent to amortise information that helps decision-making into its network weights via gradient descent on training losses. Here, we pursue an alternative approach in which agents can utilise large-scale context-sensitive database lookups to support their parametric computations. This allows agents to directly learn in an end-to-end manner to utilise relevant information to inform their outputs. In addition, new information can be attended to by the agent, without retraining, by simply augmenting the retrieval dataset. We study this approach for offline RL in 9x9 Go, a challenging game for which the vast combinatorial state space privileges generalisation over direct matching to past experiences. We leverage fast, approximate nearest neighbor techniques in order to retrieve relevant data from a set of tens of millions of expert demonstration states. Attending to this information provides a significant boost to prediction accuracy and game-play performance over simply using these demonstrations as training trajectories, providing a compelling demonstration of the value of large-scale retrieval in offline RL agents.

        ----

        ## [1461] Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs

        **Authors**: *Etienne Boursier, Loucas Pillaud-Vivien, Nicolas Flammarion*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7eeb9af3eb1f48e29c05e8dd3342b286-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7eeb9af3eb1f48e29c05e8dd3342b286-Abstract-Conference.html)

        **Abstract**:

        The training of neural networks by gradient descent methods is a cornerstone of the deep learning revolution. Yet, despite some recent progress, a complete theory explaining its success is still missing. This article presents, for orthogonal input vectors, a precise description of the gradient flow dynamics of training one-hidden layer ReLU neural networks for the mean squared error at small initialisation. In this setting, despite non-convexity, we show that the gradient flow converges to zero loss and characterise its implicit bias towards minimum variation norm. Furthermore, some interesting phenomena are highlighted: a quantitative description of the initial alignment phenomenon and a proof that the process follows a specific saddle to saddle dynamics.

        ----

        ## [1462] Weisfeiler and Leman Go Walking: Random Walk Kernels Revisited

        **Authors**: *Nils M. Kriege*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html)

        **Abstract**:

        Random walk kernels have been introduced in seminal work on graph learning and were later largely superseded by kernels based on the Weisfeiler-Leman test for graph isomorphism. We give a unified view on both classes of graph kernels. We study walk-based node refinement methods and formally relate them to several widely-used techniques, including Morgan's algorithm for molecule canonization and the Weisfeiler-Leman test. We define corresponding walk-based kernels on nodes that allow fine-grained parameterized neighborhood comparison, reach Weisfeiler-Leman expressiveness, and are computed using the kernel trick. From this we show that classical random walk kernels with only minor modifications regarding definition and computation are as expressive as the widely-used Weisfeiler-Leman subtree kernel but support non-strict neighborhood comparison. We verify experimentally that walk-based kernels reach or even surpass the accuracy of Weisfeiler-Leman kernels in real-world classification tasks.

        ----

        ## [1463] Outsourcing Training without Uploading Data via Efficient Collaborative Open-Source Sampling

        **Authors**: *Junyuan Hong, Lingjuan Lyu, Jiayu Zhou, Michael Spranger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7efe88bb4138d602e56637cfcf713654-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7efe88bb4138d602e56637cfcf713654-Abstract-Conference.html)

        **Abstract**:

        As deep learning blooms with growing demand for computation and data resources, outsourcing model training to a powerful cloud server becomes an attractive alternative to training at a low-power and cost-effective end device. Traditional outsourcing requires uploading device data to the cloud server, which can be infeasible in many real-world applications due to the often sensitive nature of the collected data and the limited communication bandwidth. To tackle these challenges, we propose to leverage widely available open-source data, which is a massive dataset collected from public and heterogeneous sources (e.g., Internet images). We develop a novel strategy called Efficient Collaborative Open-source Sampling (ECOS) to construct a proximal proxy dataset from open-source data for cloud training, in lieu of client data. ECOS probes open-source data on the cloud server to sense the distribution of client data via a communication- and computation-efficient sampling process, which only communicates a few compressed public features and client scalar responses. Extensive empirical studies show that the proposed ECOS improves the quality of automated client labeling, model compression, and label outsourcing when applied in various learning scenarios. Source codes will be released.

        ----

        ## [1464] Multi-agent Dynamic Algorithm Configuration

        **Authors**: *Ke Xue, Jiacheng Xu, Lei Yuan, Miqing Li, Chao Qian, Zongzhang Zhang, Yang Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)

        **Abstract**:

        Automated algorithm configuration relieves users from tedious, trial-and-error tuning tasks. A popular algorithm configuration tuning paradigm is dynamic algorithm configuration (DAC), in which an agent learns dynamic configuration policies across instances by reinforcement learning (RL). However, in many complex algorithms, there may exist different types of configuration hyperparameters, and such heterogeneity may bring difficulties for classic DAC which uses a single-agent RL policy. In this paper, we aim to address this issue and propose multi-agent DAC (MA-DAC), with one agent working for one type of configuration hyperparameter. MA-DAC formulates the dynamic configuration of a complex algorithm with multiple types of hyperparameters as a contextual multi-agent Markov decision process and solves it by a cooperative multi-agent RL (MARL) algorithm. To instantiate, we apply MA-DAC to a well-known optimization algorithm for multi-objective optimization problems. Experimental results show the effectiveness of MA-DAC in not only achieving superior performance compared with other configuration tuning approaches based on heuristic rules, multi-armed bandits, and single-agent RL, but also being capable of generalizing to different problem classes. Furthermore, we release the environments in this paper as a benchmark for testing MARL algorithms, with the hope of facilitating the application of MARL.

        ----

        ## [1465] TaSIL: Taylor Series Imitation Learning

        **Authors**: *Daniel Pfrommer, Thomas T. C. K. Zhang, Stephen Tu, Nikolai Matni*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f10c3d66c3b7863a9cda255dcac5bb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f10c3d66c3b7863a9cda255dcac5bb7-Abstract-Conference.html)

        **Abstract**:

        We propose Taylor Series Imitation Learning (TaSIL), a simple augmentation to standard behavior cloning losses in the context of continuous control. TaSIL penalizes deviations in the higher-order Tayler series terms between the learned and expert policies. We show that experts satisfying a notion of incremental input-to-state stability are easy to learn, in the sense that that a small TaSIL-augmented imitation loss over expert trajectories guarantees a small imitation loss over trajectories generated by the learned policy.  We provide sample-complexity bounds for TaSIL that scale as $\tilde{\mathcal{O}}(1/n)$ in the realizable setting, for $n$ the number of expert demonstrations. Finally, we demonstrate experimentally the relationship between the robustness of the expert policy and the order of Taylor expansion required in TaSIL, and compare standard Behavior Cloning, DART, and DAgger with TaSIL-loss-augmented variants.  In all cases, we show significant improvement over baselines across a variety of MuJoCo tasks.

        ----

        ## [1466] Experimental Design for Linear Functionals in Reproducing Kernel Hilbert Spaces

        **Authors**: *Mojmir Mutny, Andreas Krause*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html)

        **Abstract**:

        Optimal experimental design seeks to determine the most informative allocation of experiments  to infer an unknown statistical quantity. In this work, we investigate optimal design of experiments for {\em estimation of linear functionals in reproducing kernel Hilbert spaces (RKHSs)}. This problem has been extensively studied in the linear regression setting under an estimability condition, which allows estimating parameters without bias. We generalize this framework to RKHSs, and allow for the linear functional to be only approximately inferred, i.e., with a fixed bias. This scenario captures many important modern applications such as estimation of gradient maps, integrals and solutions to differential equations. We provide algorithms for constructing bias-aware designs for linear functionals. We derive non-asymptotic confidence sets for fixed and adaptive designs under sub-Gaussian noise, enabling us to certify estimation with bounded error with high probability.

        ----

        ## [1467] Continuous MDP Homomorphisms and Homomorphic Policy Gradient

        **Authors**: *Sahand Rezaei-Shoshtari, Rosie Zhao, Prakash Panangaden, David Meger, Doina Precup*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f44f98e5e70dea605d0c5baca231c58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f44f98e5e70dea605d0c5baca231c58-Abstract-Conference.html)

        **Abstract**:

        Abstraction has been widely studied as a way to improve the efficiency and generalization of reinforcement learning algorithms. In this paper, we study abstraction in the continuous-control setting. We extend the definition of MDP homomorphisms to encompass continuous actions in continuous state spaces.  We derive a policy gradient theorem on the abstract MDP, which allows us to leverage approximate symmetries of the environment for policy optimization. Based on this theorem, we propose an actor-critic algorithm that is able to learn the policy and the MDP homomorphism map simultaneously, using the lax bisimulation metric.  We demonstrate the effectiveness of our method on benchmark tasks in the DeepMind Control Suite.  Our method's ability to utilize MDP homomorphisms for representation learning leads to improved performance when learning from pixel observations.

        ----

        ## [1468] Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance

        **Authors**: *Dohyun Kwon, Ying Fan, Kangwook Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f52f6b8f107931127eefe15429ee278-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f52f6b8f107931127eefe15429ee278-Abstract-Conference.html)

        **Abstract**:

        Score-based generative models are shown to achieve remarkable empirical performances in various applications such as image generation and audio synthesis. However, a theoretical understanding of score-based diffusion models is still incomplete. Recently, Song et al. showed that the training objective of score-based generative models is equivalent to minimizing the Kullback-Leibler divergence of the generated distribution from the data distribution. In this work, we show that score-based models also minimize the Wasserstein distance between them. Specifically, we prove that the Wasserstein distance is upper bounded by the square root of the objective function up to multiplicative constants and a fixed constant offset. Our proof is based on a novel application of the theory of optimal transport, which can be of independent interest to the society. Our numerical experiments support our findings. By analyzing our upper bounds, we provide a few techniques to obtain tighter upper bounds.

        ----

        ## [1469] Uncertainty-Aware Reinforcement Learning for Risk-Sensitive Player Evaluation in Sports Game

        **Authors**: *Guiliang Liu, Yudong Luo, Oliver Schulte, Pascal Poupart*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f6e51d8298aa01b084b700ab91aff94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f6e51d8298aa01b084b700ab91aff94-Abstract-Conference.html)

        **Abstract**:

        A major task of sports analytics is player evaluation. Previous methods commonly measured the impact of players' actions on desirable outcomes (e.g., goals or winning) without considering the risk induced by stochastic game dynamics.  In this paper, we design an uncertainty-aware Reinforcement Learning (RL) framework to learn a risk-sensitive player evaluation metric from stochastic game dynamics. To embed the risk of a playerâ€™s movements into the distribution of action-values, we model their 1) aleatoric uncertainty, which represents the intrinsic stochasticity in a sports game, and 2) epistemic uncertainty, which is due to a model's insufficient knowledge regarding Out-of-Distribution (OoD) samples. We demonstrate how a distributional Bellman operator and a feature-space density model can capture these uncertainties. Based on such uncertainty estimation, we propose a Risk-sensitive Game Impact Metric (RiGIM) that measures players' performance over a season by conditioning on a specific confidence level. Empirical evaluation, based on over 9M play-by-play ice hockey and soccer events, shows that RiGIM correlates highly with standard success measures and has a consistent risk sensitivity.

        ----

        ## [1470] End-to-end Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking

        **Authors**: *Arpit Bansal, Avi Schwarzschild, Eitan Borgnia, Zeyad Emam, Furong Huang, Micah Goldblum, Tom Goldstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html)

        **Abstract**:

        Machine learning systems perform well on pattern matching tasks, but their ability to perform algorithmic or logical reasoning is not well understood. One important reasoning capability is algorithmic extrapolation, in which models trained only on small/simple reasoning problems can synthesize complex strategies for large/complex problems at test time. Algorithmic extrapolation can be achieved through recurrent systems, which can be iterated many times to solve difficult reasoning problems. We observe that this approach fails to scale to highly complex problems because behavior degenerates when many iterations are applied -- an issue we refer to as "overthinking." We propose a recall architecture that keeps an explicit copy of the problem instance in memory so that it cannot be forgotten. We also employ a progressive training routine that prevents the model from learning behaviors that are specific to iteration number and instead pushes it to learn behaviors that can be repeated indefinitely. These innovations prevent the overthinking problem, and enable recurrent systems to solve extremely hard extrapolation tasks.

        ----

        ## [1471] Smooth Fictitious Play in Stochastic Games with Perturbed Payoffs and Unknown Transitions

        **Authors**: *Lucas Baudin, Rida Laraki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html)

        **Abstract**:

        Recent extensions to dynamic games of the well known fictitious play learning procedure in static games were proved to globally converge to stationary Nash equilibria in two important classes of dynamic games (zero-sum and identical-interest discounted stochastic games). However, those decentralized algorithms need the players to know exactly the model (the transition probabilities and their payoffs at every stage). To overcome these strong assumptions, our paper introduces regularizations of the recent algorithms which are moreover, model-free (players don't know the transitions and their payoffs are perturbed at every stage). Our novel procedures can be interpreted as extensions to stochastic games of the classical smooth fictitious play learning procedures in static games (where players best responses are regularized, thanks to a smooth perturbation of their payoff functions). We prove the convergence of our family of procedures to stationary regularized Nash equilibria in the same classes of dynamic games (zero-sum and identical interests discounted stochastic games). The proof uses the continuous smooth best-response dynamics counterparts, and stochastic approximation methods. In the case of a MDP (a one-player stochastic game), our procedures globally converge to the optimal stationary policy of the regularized problem. In that sense, they can be seen as an alternative to the well known Q-learning procedure.

        ----

        ## [1472] OOD Link Prediction Generalization Capabilities of Message-Passing GNNs in Larger Test Graphs

        **Authors**: *Yangze Zhou, Gitta Kutyniok, Bruno Ribeiro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f88a8478c4ae97819ccffa1e80e7a7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f88a8478c4ae97819ccffa1e80e7a7b-Abstract-Conference.html)

        **Abstract**:

        This work provides the first theoretical study on the ability of graph Message Passing Neural Networks (gMPNNs) ---such as Graph Neural Networks (GNNs)--- to perform inductive out-of-distribution (OOD) link prediction tasks, where deployment (test) graph sizes are larger than training graphs. We first prove non-asymptotic bounds showing that link predictors based on permutation-equivariant (structural) node embeddings obtained by gMPNNs can converge to a random guess as test graphs get larger. We then propose a theoretically-sound gMPNN that outputs structural pairwise (2-node) embeddings and prove non-asymptotic bounds showing that, as test graphs grow, these embeddings converge to embeddings of a continuous function that retains its ability to predict links OOD. Empirical results on random graphs show agreement with our theoretical results.

        ----

        ## [1473] Algorithms with Prediction Portfolios

        **Authors**: *Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, Sergei Vassilvitskii*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f9220f90cc85b0da693643add6618e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f9220f90cc85b0da693643add6618e6-Abstract-Conference.html)

        **Abstract**:

        The research area of algorithms with predictions has seen recent success showing how to incorporate machine learning into algorithm design to improve performance when the predictions are correct, while retaining worst-case guarantees when they are not.  Most previous work has assumed that the algorithm has access to a single predictor. However, in practice, there are many machine learning methods available, often with incomparable generalization guarantees, making it hard to pick a best method a priori. In this work we consider scenarios where multiple predictors are available to the algorithm and the question is how to best utilize them. Ideally, we would like the algorithm's performance to depend on the quality of the {\em best} predictor.  However, utilizing more predictions comes with a cost, since we now have to identify which prediction is best.  We study the use of multiple predictors for a number of fundamental problems, including matching, load balancing, and non-clairvoyant scheduling, which have been well-studied in the single predictor setting. For each of these problems we introduce new algorithms that take advantage of multiple predictors, and prove bounds on the resulting performance.

        ----

        ## [1474] A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs

        **Authors**: *Songming Liu, Zhongkai Hao, Chengyang Ying, Hang Su, Jun Zhu, Ze Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f970edb14104b81e70e3b03e1f5214f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f970edb14104b81e70e3b03e1f5214f-Abstract-Conference.html)

        **Abstract**:

        We present a unified hard-constraint framework for solving geometrically complex PDEs with neural networks, where the most commonly used Dirichlet, Neumann, and Robin boundary conditions (BCs) are considered. Specifically, we first introduce the "extra fields'' from the mixed finite element method to reformulate the PDEs so as to equivalently transform the three types of BCs into linear forms. Based on the reformulation, we derive the general solutions of the BCs analytically, which are employed to construct an ansatz that automatically satisfies the BCs. With such a framework, we can train the neural networks without adding extra loss terms and thus efficiently handle geometrically complex PDEs, alleviating the unbalanced competition between the loss terms corresponding to the BCs and PDEs. We theoretically demonstrate that the "extra fields'' can stabilize the training process. Experimental results on real-world geometrically complex PDEs showcase the effectiveness of our method compared with state-of-the-art baselines.

        ----

        ## [1475] Instability and Local Minima in GAN Training with Kernel Discriminators

        **Authors**: *Evan Becker, Parthe Pandit, Sundeep Rangan, Alyson K. Fletcher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7f9a44cb707ede42a659ad85d940dd55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7f9a44cb707ede42a659ad85d940dd55-Abstract-Conference.html)

        **Abstract**:

        Generative Adversarial Networks (GANs) are a widely-used tool for generative modeling of complex data.  Despite their empirical success, the training of GANs is not fully understood due to the joint training of the generator and discriminator. This paper analyzes these joint dynamics when the true samples, as well as the generated samples, are discrete, finite sets, and the discriminator is kernel-based. A simple yet expressive framework for analyzing training called the $\textit{Isolated Points Model}$ is introduced. In the proposed model, the distance between true samples greatly exceeds the kernel width so that each generated point is influenced by at most one true point. The model enables precise characterization of the conditions for convergence both to good and bad minima. In particular, the analysis explains two common failure modes: (i) an approximate mode collapse and (ii) divergence. Numerical simulations are provided that predictably replicate these behaviors.

        ----

        ## [1476] Versatile Multi-stage Graph Neural Network for Circuit Representation

        **Authors**: *Shuwen Yang, Zhihao Yang, Dong Li, Yingxue Zhang, Zhanguang Zhang, Guojie Song, Jianye Hao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7fa548155f40c014372146be387c4f6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7fa548155f40c014372146be387c4f6a-Abstract-Conference.html)

        **Abstract**:

        Due to the rapid growth in the scale of circuits and the desire for knowledge transfer from old designs to new ones, deep learning technologies have been widely exploited in Electronic Design Automation (EDA) to assist circuit design. In chip design cycles, we might encounter heterogeneous and diverse information sources, including the two most informative ones: the netlist and the design layout. However, handling each information source independently is sub-optimal. In this paper, we propose a novel way to integrate the multiple information sources under a unified heterogeneous graph named Circuit Graph, where topological and geometrical information is well integrated. Then, we propose Circuit GNN to fully utilize the features of vertices, edges as well as heterogeneous information during the message passing process. It is the first attempt to design a versatile circuit representation that is compatible across multiple EDA tasks and stages. Experiments on the two most representative prediction tasks in EDA show that our solution reaches state-of-the-art performance in both logic synthesis and global placement chip design stages. Besides, it achieves a 10x speed-up on congestion prediction compared to the state-of-the-art model.

        ----

        ## [1477] Non-Stationary Bandits under Recharging Payoffs: Improved Planning with Sublinear Regret

        **Authors**: *Orestis Papadigenopoulos, Constantine Caramanis, Sanjay Shakkottai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7fccdff3f1457cb7b846596c76c23abd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7fccdff3f1457cb7b846596c76c23abd-Abstract-Conference.html)

        **Abstract**:

        The stochastic multi-armed bandit setting has been recently studied in the non-stationary regime, where the mean payoff of each action is a non-decreasing function of the number of rounds passed since it was last played. This model captures natural behavioral aspects of the users which crucially determine the performance of recommendation platforms, ad placement systems, and more. Even assuming prior knowledge of the mean payoff functions, computing an optimal planning in the above model is NP-hard, while the state-of-the-art is a $1/4$-approximation algorithm for the case where at most one arm can be played per round. We first focus on the setting where the mean payoff functions are known. In this setting, we significantly improve the best-known guarantees for the planning problem by developing a polynomial-time $(1-{1}/{e})$-approximation algorithm (asymptotically and in expectation), based on a novel combination of randomized LP rounding and a time-correlated (interleaved) scheduling method. Furthermore, our algorithm achieves improved guarantees -- compared to prior work -- for the case where more than one arms can be played at each round. Moving to the bandit setting, when the mean payoff functions are initially unknown, we show how our algorithm can be transformed into a bandit algorithm with sublinear regret.

        ----

        ## [1478] Optimal and Adaptive Monteiro-Svaiter Acceleration

        **Authors**: *Yair Carmon, Danielle Hausler, Arun Jambulapati, Yujia Jin, Aaron Sidford*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7ff97417474268e6b5a38bcbfae04944-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7ff97417474268e6b5a38bcbfae04944-Abstract-Conference.html)

        **Abstract**:

        We develop a variant of the Monteiro-Svaiter (MS) acceleration framework that removes the need to solve an expensive implicit equation at every iteration. Consequently, for any $p\ge 2$ we improve the complexity of convex optimization with Lipschitz $p$th derivative by a logarithmic factor,  matching a lower bound. We also introduce an MS subproblem solver that requires no knowledge of problem parameters, and implement it as either a second- or first-order method by solving linear systems or applying MinRes, respectively. On logistic regression problems our method outperforms previous accelerated second-order methods, but under-performs Newton's method; simply iterating our first-order adaptive subproblem solver is competitive with L-BFGS.

        ----

        ## [1479] Chroma-VAE: Mitigating Shortcut Learning with Generative Classifiers

        **Authors**: *Wanqian Yang, Polina Kirichenko, Micah Goldblum, Andrew Gordon Wilson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/80098914b3b3bad79b80377751a85430-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/80098914b3b3bad79b80377751a85430-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks are susceptible to shortcut learning, using simple features to achieve low training loss without discovering essential semantic structure. Contrary to prior belief, we show that generative models alone are not sufficient to prevent shortcut learning, despite an incentive to recover a more comprehensive representation of the data than discriminative approaches. However, we observe that shortcuts are preferentially encoded with minimal information, a fact that generative models can exploit to mitigate shortcut learning. In particular, we propose Chroma-VAE, a two-pronged approach where a VAE classifier is initially trained to isolate the shortcut in a small latent subspace, allowing a secondary classifier to be trained on the complementary, shortcut-free latent subspace. In addition to demonstrating the efficacy of Chroma-VAE on benchmark and real-world shortcut learning tasks, our work highlights the potential for manipulating the latent space of generative classifiers to isolate or interpret specific correlations.

        ----

        ## [1480] SparCL: Sparse Continual Learning on the Edge

        **Authors**: *Zifeng Wang, Zheng Zhan, Yifan Gong, Geng Yuan, Wei Niu, Tong Jian, Bin Ren, Stratis Ioannidis, Yanzhi Wang, Jennifer G. Dy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/80133d0f6eccaace15508f91e3c5a93c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/80133d0f6eccaace15508f91e3c5a93c-Abstract-Conference.html)

        **Abstract**:

        Existing work in continual learning (CL) focuses on mitigating catastrophic forgetting, i.e., model performance deterioration on past tasks when learning a new task. However, the training efficiency of a CL system is under-investigated, which limits the real-world application of CL systems under resource-limited scenarios. In this work, we propose a novel framework called Sparse Continual Learning (SparCL), which is the first study that leverages sparsity to enable cost-effective continual learning on edge devices. SparCL achieves both training acceleration and accuracy preservation through the synergy of three aspects: weight sparsity, data efficiency, and gradient sparsity. Specifically, we propose task-aware dynamic masking (TDM) to learn a sparse network throughout the entire CL process, dynamic data removal (DDR) to remove less informative training data, and dynamic gradient masking (DGM) to sparsify the gradient updates. Each of them not only improves efficiency, but also further mitigates catastrophic forgetting.  SparCL consistently improves the training efficiency of existing state-of-the-art (SOTA) CL methods by at most 23X less training FLOPs, and, surprisingly, further improves the SOTA accuracy by at most 1.7%. SparCL also outperforms competitive baselines obtained from adapting SOTA sparse training methods to the CL setting in both efficiency and accuracy. We also evaluate the effectiveness of SparCL on a real mobile phone, further indicating the practical potential of our method.

        ----

        ## [1481] Adaptively Exploiting d-Separators with Causal Bandits

        **Authors**: *Blair L. Bilodeau, Linbo Wang, Daniel M. Roy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html)

        **Abstract**:

        Multi-armed bandit problems provide a framework to identify the optimal intervention over a sequence of repeated experiments. Without additional assumptions, minimax optimal performance (measured by cumulative regret) is well-understood. With access to additional observed variables that d-separate the intervention from the outcome (i.e., they are a d-separator), recent "causal bandit" algorithms provably incur less regret. However, in practice it is desirable to be agnostic to whether observed variables are a d-separator. Ideally, an algorithm should be adaptive; that is, perform nearly as well as an algorithm with oracle knowledge of the presence or absence of a d-separator. In this work, we formalize and study this notion of adaptivity, and provide a novel algorithm that simultaneously achieves (a) optimal regret when a d-separator is observed, improving on classical minimax algorithms, and (b) significantly smaller regret than recent causal bandit algorithms when the observed variables are not a d-separator. Crucially, our algorithm does not require any oracle knowledge of whether a d-separator is observed. We also generalize this adaptivity to other conditions, such as the front-door criterion.

        ----

        ## [1482] Spectrum Random Masking for Generalization in Image-based Reinforcement Learning

        **Authors**: *Yangru Huang, Peixi Peng, Yifan Zhao, Guangyao Chen, Yonghong Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html)

        **Abstract**:

        Generalization in image-based reinforcement learning (RL) aims to learn a robust policy that could be applied directly on unseen visual environments, which is a challenging task since agents usually tend to overfit to their training environment. To handle this problem, a natural approach is to increase the data diversity by image based augmentations. However, different with most vision tasks such as classification and detection, RL tasks are not always invariant to spatial based augmentations due to the entanglement of environment dynamics and visual appearance.  In this paper, we argue with two principles for augmentations in RL: First, the augmented observations should facilitate learning a universal policy, which is robust to various distribution shifts. Second, the augmented data should be invariant to the learning signals such as action and reward. Following these rules, we revisit image-based RL tasks from the view of frequency domain and propose a novel augmentation method, namely Spectrum Random Masking (SRM),which is able to help agents to learn the whole frequency spectrum of observation for coping with various distributions and compatible with the pre-collected action and reward corresponding to original observation. Extensive experiments conducted on DMControl Generalization Benchmark   demonstrate the proposed SRM achieves the state-of-the-art performance with strong generalization potentials.

        ----

        ## [1483] Contrastive Graph Structure Learning via Information Bottleneck for Recommendation

        **Authors**: *Chunyu Wei, Jian Liang, Di Liu, Fei Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html)

        **Abstract**:

        Graph convolution networks (GCNs) for recommendations have emerged as an important research topic due to their ability to exploit higher-order neighbors. Despite their success, most of them suffer from the popularity bias brought by a small number of active users and popular items. Also, a real-world user-item bipartite graph contains many noisy interactions, which may hamper the sensitive GCNs. Graph contrastive learning show promising performance for solving the above challenges in recommender systems. Most existing works typically perform graph augmentation to create multiple views of the original graph by randomly dropping edges/nodes or relying on predefined rules, and these augmented views always serve as an auxiliary task by maximizing their correspondence. However, we argue that the graph structures generated from these vanilla approaches may be suboptimal, and maximizing their correspondence will force the representation to capture information irrelevant for the recommendation task. Here, we propose a Contrastive Graph Structure Learning via Information Bottleneck (CGI) for recommendation, which adaptively learns whether to drop an edge or node to obtain optimized graph structures in an end-to-end manner. Moreover, we innovatively introduce the Information Bottleneck into the contrastive learning process to avoid capturing irrelevant information among different views and help enrich the final representation for recommendation. Extensive experiments on public datasets are provided to show that our model significantly outperforms strong baselines.

        ----

        ## [1484] Pyramid Attention For Source Code Summarization

        **Authors**: *Lei Chai, Ming Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/803cb038c7df56122e55a06c2856938f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/803cb038c7df56122e55a06c2856938f-Abstract-Conference.html)

        **Abstract**:

        This paper presents a multi-granularity method for source code summarization, which generates a concise functional description for the given code snippet. We notice that skilled programmers write and read source codes hierarchically and pay close attention to conceptual entities like statements, tokens, sub-tokens, and the mapping relations between them. The entities have specific emphasis according to their granularities, e.g., statements in coarse-granularity reveal the global logical semantics of code, and the sub-tokens in fine-granularity are more related to the textual semantics. Driven by this observation, we demonstrate that a multi-granularity formulation incorporating these conceptual entities benefit the code summarization task. Concretely, the source code is transformed into a pyramidal representation, and then a pyramid attention mechanism is applied for efficient feature aggregation among different hierarchies in it. We instantiate our multi-granularity method using the proposed pyramid attention and name it PA-former (Pyramid Attention transformer). We evaluated it on two source code summarization benchmarks where it surpasses the prior works and achieves new state-of-the-art results. Our code and data are available at https://github.com/leichainju/pa-former.

        ----

        ## [1485] SIREN: Shaping Representations for Detecting Out-of-Distribution Objects

        **Authors**: *Xuefeng Du, Gabriel Gozum, Yifei Ming, Yixuan Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/804dbf8d3b8eee1ef875c6857efc64eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/804dbf8d3b8eee1ef875c6857efc64eb-Abstract-Conference.html)

        **Abstract**:

        Detecting out-of-distribution (OOD) objects is indispensable for safely deploying object detectors in the wild. Although distance-based OOD detection methods have demonstrated promise in image classification, they remain largely unexplored in object-level OOD detection. This paper bridges the gap by proposing a distance-based framework for detecting OOD objects, which relies on the model-agnostic representation space and provides strong generality across different neural architectures. Our proposed framework SIREN contributes two novel components: (1) a representation learning component that uses a trainable loss function to shape the representations into a mixture of von Mises-Fisher (vMF) distributions on the unit hypersphere, and (2) a test-time OOD detection score leveraging the learned vMF distributions in a parametric or non-parametric way. SIREN achieves competitive performance on both the recent detection transformers and CNN-based models, improving the AUROC by a large margin compared to the previous best method. Code is publicly available at https://github.com/deeplearning-wisc/siren.

        ----

        ## [1486] CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP

        **Authors**: *Andreas Fürst, Elisabeth Rumetshofer, Johannes Lehner, Viet T. Tran, Fei Tang, Hubert Ramsauer, David P. Kreil, Michael Kopp, Günter Klambauer, Angela Bitto, Sepp Hochreiter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8078e76f913e31b8467e85b4c0f0d22b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8078e76f913e31b8467e85b4c0f0d22b-Abstract-Conference.html)

        **Abstract**:

        CLIP yielded impressive results on zero-shot transfer learning tasks and is considered as a foundation model like BERT or GPT3. CLIP vision models that have a rich representation are pre-trained using the InfoNCE objective and natural language supervision before they are fine-tuned on particular tasks. Though CLIP excels at zero-shot transfer learning, it suffers from an explaining away problem, that is, it focuses on one or few features, while neglecting other relevant features. This problem is caused by insufficiently extracting the covariance structure in the original multi-modal data. We suggest to use modern Hopfield networks to tackle the problem of explaining away. Their retrieved embeddings have an enriched covariance structure derived from co-occurrences of features in the stored embeddings. However, modern Hopfield networks increase the saturation effect of the InfoNCE objective which hampers learning. We propose to use the InfoLOOB objective to mitigate this saturation effect. We introduce the novel "Contrastive Leave One Out Boost" (CLOOB), which uses modern Hopfield networks for covariance enrichment together with the InfoLOOB objective. In experiments we compare CLOOB to CLIP after pre-training on the Conceptual Captions and the YFCC dataset with respect to their zero-shot transfer learning performance on other datasets. CLOOB consistently outperforms CLIP at zero-shot transfer learning across all considered architectures and datasets.

        ----

        ## [1487] I2Q: A Fully Decentralized Q-Learning Algorithm

        **Authors**: *Jiechuan Jiang, Zongqing Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html)

        **Abstract**:

        Fully decentralized multi-agent reinforcement learning has shown great potentials for many real-world cooperative tasks, where the global information, \textit{e.g.}, the actions of other agents, is not accessible. Although independent Q-learning is widely used for decentralized training, the transition probabilities are non-stationary since other agents are updating policies simultaneously, which leads to non-guaranteed convergence of independent Q-learning. To deal with non-stationarity, we first introduce stationary ideal transition probabilities, on which independent Q-learning could converge to the global optimum. Further, we propose a fully decentralized method, I2Q, which performs independent Q-learning on the modeled ideal transition function to reach the global optimum. The modeling of ideal transition function in I2Q is fully decentralized and independent from the learned policies of other agents, helping I2Q be free from non-stationarity and learn the optimal policy. Empirically, we show that I2Q can achieve remarkable improvement in a variety of cooperative multi-agent tasks.

        ----

        ## [1488] Planning to the Information Horizon of BAMDPs via Epistemic State Abstraction

        **Authors**: *Dilip Arumugam, Satinder Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/80b7bec60081f95d900973509744a306-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/80b7bec60081f95d900973509744a306-Abstract-Conference.html)

        **Abstract**:

        The Bayes-Adaptive Markov Decision Process (BAMDP) formalism pursues the Bayes-optimal solution to the exploration-exploitation trade-off in reinforcement learning. As the computation of exact solutions to Bayesian reinforcement-learning problems is intractable, much of the literature has focused on developing suitable approximation algorithms. In this work, before diving into algorithm design, we first define, under mild structural assumptions, a complexity measure for BAMDP planning. As efficient exploration in BAMDPs hinges upon the judicious acquisition of information, our complexity measure highlights the worst-case difficulty of gathering information and exhausting epistemic uncertainty. To illustrate its significance, we establish a computationally-intractable, exact planning algorithm that takes advantage of this measure to show more efficient planning. We then conclude by introducing a specific form of state abstraction with the potential to reduce BAMDP complexity and gives rise to a computationally-tractable, approximate planning algorithm.

        ----

        ## [1489] AutoST: Towards the Universal Modeling of Spatio-temporal Sequences

        **Authors**: *Jianxin Li, Shuai Zhang, Hui Xiong, Haoyi Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/80d46bb66ea003f4b29fa6013905d50a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/80d46bb66ea003f4b29fa6013905d50a-Abstract-Conference.html)

        **Abstract**:

        The analysis of spatio-temporal sequences plays an important role in many real-world applications, demanding a high model capacity to capture the interdependence among spatial and temporal dimensions. Previous studies provided separated network design in three categories: spatial first, temporal first, and spatio-temporal synchronous. However, the manually-designed heterogeneous models can hardly meet the spatio-temporal dependency capturing priority for various tasks. To address this, we proposed a universal modeling framework with three distinctive characteristics: (i) Attention-based network backbone, including S2T Layer (spatial first), T2S Layer (temporal first), and STS Layer (spatio-temporal synchronous). (ii) The universal modeling framework, named UniST, with a unified architecture that enables flexible modeling priorities with the proposed three different modules. (iii) An automatic search strategy, named AutoST, automatically searches the optimal spatio-temporal modeling priority by network architecture search. Extensive experiments on five real-world datasets demonstrate that UniST with any single type of our three proposed modules can achieve state-of-the-art performance. Furthermore, AutoST can achieve overwhelming performance with UniST.

        ----

        ## [1490] Constrained Langevin Algorithms with L-mixing External Random Variables

        **Authors**: *Yuping Zheng, Andrew G. Lamperski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/811d35e47edbb191c19151f3c5f80f53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/811d35e47edbb191c19151f3c5f80f53-Abstract-Conference.html)

        **Abstract**:

        Langevin algorithms are gradient descent methods augmented with additive noise, and are widely used in Markov Chain Monte Carlo (MCMC) sampling, optimization, and machine learning. In recent years, the non-asymptotic analysis of Langevin algorithms for non-convex learning has been extensively explored. For constrained problems with non-convex losses over a compact convex domain with IID data variables, the projected Langevin algorithm achieves a deviation of $O(T^{-1/4} (\log T)^{1/2})$ from its target distribution \cite{lamperski2021projected} in $1$-Wasserstein distance. In this paper, we obtain a deviation of $O(T^{-1/2} \log T)$ in $1$-Wasserstein distance for non-convex losses with $L$-mixing data variables and polyhedral constraints (which are not necessarily bounded). This improves on the previous bound for constrained problems and matches the best-known bound for unconstrained problems.

        ----

        ## [1491] Language Conditioned Spatial Relation Reasoning for 3D Object Grounding

        **Authors**: *Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, Ivan Laptev*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/819aaee144cb40e887a4aa9e781b1547-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/819aaee144cb40e887a4aa9e781b1547-Abstract-Conference.html)

        **Abstract**:

        Localizing objects in 3D scenes based on natural language requires understanding and reasoning about spatial relations. In particular, it is often crucial to distinguish similar objects referred by the text, such as "the left most chair" and "a chair next to the window". In this work we propose a language-conditioned transformer model for grounding 3D objects and their spatial relations. To this end, we design a spatial self-attention layer that accounts for relative distances and orientations between objects in input 3D point clouds. Training such a layer with visual and language inputs enables to disambiguate spatial relations and to localize objects referred by the text. To facilitate the cross-modal learning of relations, we further propose a teacher-student approach where the teacher model is first trained using ground-truth object labels, and then helps to train a student model using point cloud inputs. We perform ablation studies showing advantages of our approach. We also demonstrate our model to significantly outperform the state of the art on the challenging Nr3D, Sr3D and ScanRefer 3D object grounding datasets.

        ----

        ## [1492] MVP-N: A Dataset and Benchmark for Real-World Multi-View Object Classification

        **Authors**: *Ren Wang, Jiayue Wang, Tae Sung Kim, Jinsung Kim, Hyuk-Jae Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Combining information from multiple views is essential for discriminating similar objects. However, existing datasets for multi-view object classification have several limitations, such as synthetic and coarse-grained objects, no validation split for hyperparameter tuning, and a lack of view-level information quantity annotations for analyzing multi-view-based methods. To address this issue, this study proposes a new dataset, MVP-N, which contains 44 retail products, 16k real captured views with human-perceived information quantity annotations, and 9k multi-view sets. The fine-grained categorization of objects naturally generates multi-view label noise owing to the inter-class view similarity, allowing the study of learning from noisy labels in the multi-view case. Moreover, this study benchmarks four multi-view-based feature aggregation methods and twelve soft label methods on MVP-N. Experimental results show that MVP-N will be a valuable resource for facilitating the development of real-world multi-view object classification methods. The dataset and code are publicly available at https://github.com/SMNUResearch/MVP-N.

        ----

        ## [1493] Data Augmentation for Compositional Data: Advancing Predictive Models of the Microbiome

        **Authors**: *Elliott Gordon-Rodríguez, Thomas P. Quinn, John P. Cunningham*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/81a28be483155f802ddef448d6fc4b57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/81a28be483155f802ddef448d6fc4b57-Abstract-Conference.html)

        **Abstract**:

        Data augmentation plays a key role in modern machine learning pipelines. While numerous augmentation strategies have been studied in the context of computer vision and natural language processing, less is known for other data modalities. Our work extends the success of data augmentation to compositional data, i.e., simplex-valued data, which is of particular interest in microbiology, geochemistry, and other applications. Drawing on key principles from compositional data analysis, such as the \emph{Aitchison geometry of the simplex} and subcompositions, we define novel augmentation strategies for this data modality. Incorporating our data augmentations into standard supervised learning pipelines results in consistent performance gains across a wide range of standard benchmark datasets. In particular, we set a new state-of-the-art for key disease prediction tasks including colorectal cancer, type 2 diabetes, and Crohn's disease. In addition, our data augmentations enable us to define a novel contrastive learning model, which improves on previous representation learning approaches for microbiome compositional data.

        ----

        ## [1494] On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses

        **Authors**: *Anshuman Chhabra, Ashwin Sekhari, Prasant Mohapatra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)

        **Abstract**:

        Clustering models constitute a class of unsupervised machine learning methods which are used in a number of application pipelines, and play a vital role in modern data science. With recent advancements in deep learning-- deep clustering models have emerged as the current state-of-the-art over traditional clustering approaches, especially for high-dimensional image datasets. While traditional clustering approaches have been analyzed from a robustness perspective, no prior work has investigated adversarial attacks and robustness for deep clustering models in a principled manner. To bridge this gap, we propose a blackbox attack using Generative Adversarial Networks (GANs) where the adversary does not know which deep clustering model is being used, but can query it for outputs. We analyze our attack against multiple state-of-the-art deep clustering models and real-world datasets, and find that it is highly successful. We then employ some natural unsupervised defense approaches, but find that these are unable to mitigate our attack. Finally, we attack Face++, a production-level face clustering API service, and find that we can significantly reduce its performance as well. Through this work, we thus aim to motivate the need for truly robust deep clustering models.

        ----

        ## [1495] Exploiting the Relationship Between Kendall's Rank Correlation and Cosine Similarity for Attribution Protection

        **Authors**: *Fan Wang, Adams Wai-Kin Kong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html)

        **Abstract**:

        Model attributions are important in deep neural networks as they aid practitioners in understanding the models, but recent studies reveal that attributions can be easily perturbed by adding imperceptible noise to the input. The non-differentiable Kendall's rank correlation is a key performance index for attribution protection. In this paper, we first show that the expected Kendall's rank correlation is positively correlated to cosine similarity and then indicate that the direction of attribution is the key to attribution robustness. Based on these findings, we explore the vector space of attribution to explain the shortcomings of attribution defense methods using $\ell_p$ norm and propose integrated gradient regularizer (IGR), which maximizes the cosine similarity between natural and perturbed attributions. Our analysis further exposes that IGR encourages neurons with the same activation states for natural samples and the corresponding perturbed samples. Our experiments on different models and datasets confirm our analysis on attribution protection and demonstrate a decent improvement in adversarial robustness.

        ----

        ## [1496] Wavelet Feature Maps Compression for Image-to-Image CNNs

        **Authors**: *Shahaf E. Finder, Yair Zohav, Maor Ashkenazi, Eran Treister*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/81f19c0e9f3e06c831630ab6662fd8ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/81f19c0e9f3e06c831630ab6662fd8ea-Abstract-Conference.html)

        **Abstract**:

        Convolutional Neural Networks (CNNs) are known for requiring extensive computational resources, and quantization is among the best and most common methods for compressing them. While aggressive quantization (i.e., less than 4-bits) performs well for classification, it may cause severe performance degradation in image-to-image tasks such as semantic segmentation and depth estimation. In this paper, we propose Wavelet Compressed Convolution (WCC)---a novel approach for high-resolution activation maps compression integrated with point-wise convolutions, which are the main computational cost of modern architectures. To this end, we use an efficient and hardware-friendly Haar-wavelet transform, known for its effectiveness in image compression, and define the convolution on the compressed activation map. We experiment with various tasks that benefit from high-resolution input. By combining WCC with light quantization, we achieve compression rates equivalent to 1-4bit activation quantization with relatively small and much more graceful degradation in performance. Our code is available at https://github.com/BGUCompSci/WaveletCompressedConvolution.

        ----

        ## [1497] Biologically plausible solutions for spiking networks with efficient coding

        **Authors**: *Veronika Koren, Stefano Panzeri*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html)

        **Abstract**:

        Understanding how the dynamics of neural networks is shaped by the computations they perform is a fundamental question in neuroscience. Recently, the framework of efficient coding proposed a theory of how spiking neural networks can compute low-dimensional stimulus signals with high efficiency. Efficient spiking networks are based on time-dependent minimization of a loss function related to information coding with spikes. To inform the understanding of the function and dynamics of biological networks in the brain, however, the mathematical models have to be informed by biology and obey the same constraints as biological networks. Currently, spiking network models of efficient coding have been extended to include some features of biological plausibility, such as architectures with excitatory and inhibitory neurons. However, biological realism of efficient coding theories is still limited to simple cases and does not include  single neuron and network properties that are known to be key in biological circuits. Here, we revisit the theory of efficient coding with spikes to  develop spiking neural networks that are closer to biological circuits. Namely, we find a biologically plausible spiking model realizing efficient coding in the case of a generalized leaky integrate-and-fire network with excitatory and inhibitory units, equipped with fast and slow synaptic currents, local homeostatic currents such as spike-triggered adaptation, hyperpolarization-activated rebound current, heterogeneous firing thresholds and resets, heterogeneous postsynaptic potentials, and structured, low-rank connectivity. We show how the rank of E-E connectivity matrix shapes network responses.

        ----

        ## [1498] Finding Optimal Arms in Non-stochastic Combinatorial Bandits with Semi-bandit Feedback and Finite Budget

        **Authors**: *Jasmin Brandt, Viktor Bengs, Björn Haddenhorst, Eyke Hüllermeier*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/820e95997d050178323230e316897c38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/820e95997d050178323230e316897c38-Abstract-Conference.html)

        **Abstract**:

        We consider the combinatorial bandits problem with semi-bandit feedback under finite sampling budget constraints, in which the learner can carry out its action only for a limited number of times specified by an overall budget. The action is to choose a set of arms, whereupon feedback for each arm in the chosen set is received. Unlike existing works, we study this problem in a non-stochastic setting with subset-dependent feedback, i.e., the semi-bandit feedback received could be generated by an oblivious adversary and also might depend on the chosen set of arms. In addition, we consider a general feedback scenario covering both the numerical-based as well as preference-based case and introduce a sound theoretical framework for this setting guaranteeing sensible notions of optimal arms, which a learner seeks to find. We suggest a generic algorithm suitable to cover the full spectrum of conceivable arm elimination strategies from aggressive to conservative. Theoretical questions about the sufficient and necessary budget of the algorithm to find the best arm are answered and complemented by deriving lower bounds for any learning algorithm for this problem scenario.

        ----

        ## [1499] Graph Neural Networks are Dynamic Programmers

        **Authors**: *Andrew Joseph Dudzik, Petar Velickovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8248b1ded388fcdbbd121bcdfea3068c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8248b1ded388fcdbbd121bcdfea3068c-Abstract-Conference.html)

        **Abstract**:

        Recent advances in neural algorithmic reasoning with graph neural networks (GNNs) are propped up by the notion of algorithmic alignment. Broadly, a neural network will be better at learning to execute a reasoning task (in terms of sample complexity) if its individual components align well with the target algorithm. Specifically, GNNs are claimed to align with dynamic programming (DP), a general problem-solving strategy which expresses many polynomial-time algorithms. However, has this alignment truly been demonstrated and theoretically quantified? Here we show, using methods from category theory and abstract algebra, that there exists an intricate connection between GNNs and DP, going well beyond the initial observations over individual algorithms such as Bellman-Ford. Exposing this connection, we easily verify several prior findings in the literature, produce better-grounded GNN architectures for edge-centric tasks, and demonstrate empirical results on the CLRS algorithmic reasoning benchmark. We hope our exposition will serve as a foundation for building stronger algorithmically aligned GNNs.

        ----

        ## [1500] Evaluated CMI Bounds for Meta Learning: Tightness and Expressiveness

        **Authors**: *Fredrik Hellström, Giuseppe Durisi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/824c9b06e0b21b2a8bb74fcc8a558be4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/824c9b06e0b21b2a8bb74fcc8a558be4-Abstract-Conference.html)

        **Abstract**:

        Recent work has established that the conditional mutual information (CMI) framework of Steinke and Zakynthinou (2020) is expressive enough to capture generalization guarantees in terms of algorithmic stability, VC dimension, and related complexity measures for conventional learning (Harutyunyan et al., 2021, Haghifam et al., 2021). Hence, it provides a unified method for establishing generalization bounds. In meta learning, there has so far been a divide between information-theoretic results and results from classical learning theory. In this work, we take a first step toward bridging this divide. Specifically, we present novel generalization bounds for meta learning in terms of the evaluated CMI (e-CMI). To demonstrate the expressiveness of the e-CMI framework, we apply our bounds to a representation learning setting, with $n$ samples from $\hat n$ tasks parameterized by functions of the form $f_i \circ h$. Here, each $f_i \in \mathcal F$ is a task-specific function, and $h \in \mathcal H$ is the shared representation. For this setup, we show that the e-CMI framework yields a bound that scales as $\sqrt{ \mathcal C(\mathcal H)/(n\hat n) + \mathcal C(\mathcal F)/n} $, where $\mathcal C(\cdot)$ denotes a complexity measure of the hypothesis class. This scaling behavior coincides with the one reported in Tripuraneni et al. (2020) using Gaussian complexity.

        ----

        ## [1501] SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning

        **Authors**: *Yuhang Jiang, Jianzhun Shao, Shuncheng He, Hongchang Zhang, Xiangyang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/825341ab91db01bf063add41ac022702-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/825341ab91db01bf063add41ac022702-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning typically relies heavily on a well-designed reward signal, which gets more challenging in cooperative multi-agent reinforcement learning. Alternatively, unsupervised reinforcement learning (URL) has delivered on its promise in the recent past to learn useful skills and explore the environment without external supervised signals. These approaches mainly aimed for the single agent to reach distinguishable states, insufficient for multi-agent systems due to that each agent interacts with not only the environment, but also the other agents. We propose Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning (SPD) to learn generic coordination policies for agents with no extrinsic reward. Specifically, we devise the Synergy Pattern Graph (SPG), a graph depicting the relationships of agents at each time step. Furthermore, we propose an episode-wise divergence measurement to approximate the discrepancy of synergy patterns. To overcome the challenge of sparse return, we decompose the discrepancy of synergy patterns to per-time-step pseudo-reward. Empirically, we show the capacity of SPD to acquire meaningful coordination policies, such as maintaining specific formations in Multi-Agent Particle Environment and pass-and-shoot in Google Research Football. Furthermore, we demonstrate that the same instructive pretrained policy's parameters can serve as a good initialization for a series of downstream tasks' policies, achieving higher data efficiency and outperforming state-of-the-art approaches in Google Research Football.

        ----

        ## [1502] Self-Aware Personalized Federated Learning

        **Authors**: *Huili Chen, Jie Ding, Eric W. Tramel, Shuang Wu, Anit Kumar Sahu, Salman Avestimehr, Tao Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8265d7bb2db42e86637001db2c46619f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8265d7bb2db42e86637001db2c46619f-Abstract-Conference.html)

        **Abstract**:

        In the context of personalized federated learning (FL), the critical challenge is to balance local model improvement and global model tuning when the personal and global objectives may not be exactly aligned. Inspired by Bayesian hierarchical models, we develop a self-aware personalized FL method where each client can automatically balance the training of its local personal model and the global model that implicitly contributes to other clients' training. Such a balance is derived from the inter-client and intra-client uncertainty quantification. A larger inter-client variation implies more personalization is needed. Correspondingly, our method uses uncertainty-driven local training steps an aggregation rule instead of conventional local fine-tuning and sample size-based aggregation. With experimental studies on synthetic data, Amazon Alexa audio data, and public datasets such as MNIST, FEMNIST, CIFAR10, and Sent140, we show that our proposed method can achieve significantly improved personalization performance compared with the existing counterparts.

        ----

        ## [1503] Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology

        **Authors**: *Syed Ashar Javed, Dinkar Juyal, Harshith Padigela, Amaro Taylor-Weiner, Limin Yu, Aaditya Prakash*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html)

        **Abstract**:

        Multiple Instance Learning (MIL) has been widely applied in pathology towards solving critical problems such as automating cancer diagnosis and grading, predicting patient prognosis, and therapy response. Deploying these models in a clinical setting requires careful inspection of these black boxes during development and deployment to identify failures and maintain physician trust. In this work, we propose a simple formulation of MIL models, which enables interpretability while maintaining similar predictive performance. Our Additive MIL models enable spatial credit assignment such that the contribution of each region in the image can be exactly computed and visualized. We show that our spatial credit assignment coincides with regions used by pathologists during diagnosis and improves upon classical attention heatmaps from attention MIL models. We show that any existing MIL model can be made additive with a simple change in function composition. We also show how these models can debug model failures, identify spurious features, and highlight class-wise regions of interest, enabling their use in high-stakes environments such as clinical decision-making.

        ----

        ## [1504] Model-Based Imitation Learning for Urban Driving

        **Authors**: *Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zachary Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, Jamie Shotton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html)

        **Abstract**:

        An accurate model of the environment and the dynamic agents acting in it offers great potential for improving motion planning. We present MILE: a Model-based Imitation LEarning approach to jointly learn a model of the world and a policy for autonomous driving. Our method leverages 3D geometry as an inductive bias and learns a highly compact latent space directly from high-resolution videos of expert demonstrations. Our model is trained on an offline corpus of urban driving data, without any online interaction with the environment. MILE improves upon prior state-of-the-art by 31% in driving score on the CARLA simulator when deployed in a completely new town and new weather conditions. Our model can predict diverse and plausible states and actions, that can be interpretably decoded to bird's-eye view semantic segmentation. Further, we demonstrate that it can execute complex driving manoeuvres from plans entirely predicted in imagination. Our approach is the first camera-only method that models static scene, dynamic scene, and ego-behaviour in an urban driving environment. The code and model weights are available at https://github.com/wayveai/mile.

        ----

        ## [1505] Online Training Through Time for Spiking Neural Networks

        **Authors**: *Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Di He, Zhouchen Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82846e19e6d42ebfd4ace4361def29ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82846e19e6d42ebfd4ace4361def29ae-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks (SNNs) are promising brain-inspired energy-efficient models. Recent progress in training methods has enabled successful deep SNNs on large-scale tasks with low latency. Particularly, backpropagation through time (BPTT) with surrogate gradients (SG) is popularly used to enable models to achieve high performance in a very small number of time steps. However, it is at the cost of large memory consumption for training, lack of theoretical clarity for optimization, and inconsistency with the online property of biological learning rules and rules on neuromorphic hardware. Other works connect the spike representations of SNNs with equivalent artificial neural network formulation and train SNNs by gradients from equivalent mappings to ensure descent directions. But they fail to achieve low latency and are also not online. In this work, we propose online training through time (OTTT) for SNNs, which is derived from BPTT to enable forward-in-time learning by tracking presynaptic activities and leveraging instantaneous loss and gradients. Meanwhile, we theoretically analyze and prove that the gradients of OTTT can provide a similar descent direction for optimization as gradients from equivalent mapping between spike representations under both feedforward and recurrent conditions. OTTT only requires constant training memory costs agnostic to time steps, avoiding the significant memory costs of BPTT for GPU training. Furthermore, the update rule of OTTT is in the form of three-factor Hebbian learning, which could pave a path for online on-chip learning. With OTTT, it is the first time that the two mainstream supervised SNN training methods, BPTT with SG and spike representation-based training, are connected, and meanwhile it is in a biologically plausible form. Experiments on CIFAR-10, CIFAR-100, ImageNet, and CIFAR10-DVS demonstrate the superior performance of our method on large-scale static and neuromorphic datasets in a small number of time steps. Our code is available at https://github.com/pkuxmq/OTTT-SNN.

        ----

        ## [1506] SCONE: Surface Coverage Optimization in Unknown Environments by Volumetric Integration

        **Authors**: *Antoine Guédon, Pascal Monasse, Vincent Lepetit*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/828c6d69bdf91fca7f2b97c4dc214e94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/828c6d69bdf91fca7f2b97c4dc214e94-Abstract-Conference.html)

        **Abstract**:

        Next Best View computation (NBV) is a long-standing problem in robotics, and consists in identifying the next most informative sensor position(s) for reconstructing a 3D object or scene efficiently and accurately. Like most current methods, we consider NBV prediction from a depth sensor like Lidar systems. Learning-based methods relying on a volumetric representation of the scene are suitable for path planning, but have lower accuracy than methods using a surface-based representation. However, the latter do not scale well with the size of the scene and constrain the camera to a small number of poses. To obtain the advantages of both representations, we show that we can maximize surface metrics by Monte Carlo integration over a volumetric representation. In particular, we propose an approach, SCONE, that relies on two neural modules: The first module predicts occupancy probability in the entire volume of the scene. Given any new camera pose, the second module samples points in the scene based on their occupancy probability and leverages a self-attention mechanism to predict the visibility of the samples. Finally, we integrate the visibility to evaluate the gain in surface coverage for the new camera pose. NBV is selected as the pose that maximizes the gain in total surface coverage. Our method scales to large scenes and handles free camera motion: It takes as input an arbitrarily large point cloud gathered by a depth sensor as well as camera poses to predict NBV. We demonstrate our approach on a novel dataset made of large and complex 3D scenes.

        ----

        ## [1507] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents

        **Authors**: *Shunyu Yao, Howard Chen, John Yang, Karthik Narasimhan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html)

        **Abstract**:

        Most existing benchmarks for grounding language in interactive environments either lack realistic linguistic elements, or prove difficult to scale up due to substantial human involvement in the collection of data or feedback signals. We develop WebShop â€“ a simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. In this environment, an agent needs to navigate multiple types of webpages and issue diverse actions to find, customize, and purchase a product given an instruction. WebShop provides several challenges including understanding compositional instructions, query (re-)formulation, dealing with noisy text in webpages, and performing strategic exploration. We collect over 1,600 human trajectories to first validate the benchmark, then train and evaluate a diverse range of agents using reinforcement learning, imitation learning, and pre-trained image and language models. Our best model achieves a task success rate of 29%, which significantly outperforms rule heuristics but is far lower than expert human performance (59%). We also analyze agent and human trajectories and ablate various model components to provide insights for developing future agents with stronger language understanding and decision making abilities. Finally, we show our agent trained on WebShop exhibits non-trivial sim-to-real transfer when evaluated on amazon.com and ebay.com, indicating the potential value of our benchmark for developing practical web agents that can operate in the wild.

        ----

        ## [1508] Boosting Out-of-distribution Detection with Typical Features

        **Authors**: *Yao Zhu, Yuefeng Chen, Chuanlong Xie, Xiaodan Li, Rong Zhang, Hui Xue, Xiang Tian, Bolun Zheng, Yaowu Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82b0c1b954b6ef9f3cfb664a82b201bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82b0c1b954b6ef9f3cfb664a82b201bb-Abstract-Conference.html)

        **Abstract**:

        Out-of-distribution (OOD) detection is a critical task for ensuring the reliability and safety of deep neural networks in real-world scenarios. Different from most previous OOD detection methods that focus on designing OOD scores or introducing diverse outlier examples to retrain the model, we delve into the obstacle factors in OOD detection from the perspective of typicality and regard the feature's high-probability region of the deep model as the feature's typical set. We propose to rectify the feature into its typical set and calculate the OOD score with the typical features to achieve reliable uncertainty estimation. The feature rectification can be conducted as a plug-and-play module with various OOD scores. We evaluate the superiority of our method on both the commonly used benchmark (CIFAR) and the more challenging high-resolution benchmark with large label space (ImageNet). Notably, our approach outperforms state-of-the-art methods by up to 5.11% in the average FPR95 on the ImageNet benchmark.

        ----

        ## [1509] Dynamic Learning in Large Matching Markets

        **Authors**: *Anand Kalvit, Assaf Zeevi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html)

        **Abstract**:

        We study a sequential matching problem faced by "large" centralized platforms where "jobs" must be matched to "workers" subject to uncertainty about worker skill proficiencies. Jobs arrive at discrete times with "job-types" observable upon arrival. To capture the "choice overload" phenomenon, we posit an unlimited supply of workers where each worker is characterized by a vector of attributes (aka "worker-types") drawn from an underlying population-level distribution. The distribution as well as mean payoffs for possible worker-job type-pairs are unobservables and the platform's goal is to sequentially match incoming jobs to workers in a way that maximizes its cumulative payoffs over the planning horizon. We establish lower bounds on the "regret" of any matching algorithm in this setting and propose a novel rate-optimal learning algorithm that adapts to aforementioned primitives "online." Our learning guarantees highlight a distinctive characteristic of the problem: achievable performance only has a "second-order" dependence on worker-type distributions; we believe this finding may be of interest more broadly.

        ----

        ## [1510] Invariant and Transportable Representations for Anti-Causal Domain Shifts

        **Authors**: *Yibo Jiang, Victor Veitch*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82e330c1b962ee1e6adb60d01f695366-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82e330c1b962ee1e6adb60d01f695366-Abstract-Conference.html)

        **Abstract**:

        Real-world classification problems must contend with domain shift, the (potential) mismatch between the domain where a model is deployed and the domain(s) where the training data was gathered. Methods to handle such problems must specify what structure is held in common between the domains and what is allowed to vary. A natural assumption is that causal (structural) relationships are invariant in all domains. Then, it is tempting to learn a predictor for label $Y$ that depends only on its causal parents. However, many real-world problems are ``anti-causal'' in the sense that $Y$ is a cause of the covariates $X$---in this case, $Y$ has no causal parents and the naive causal invariance is useless. In this paper, we study representation learning under a particular notion of domain shift that both respects causal invariance and that naturally handles the ``anti-causal'' structure. We show how to leverage the shared causal structure of the domains to learn a representation that both admits an invariant predictor and that also allows fast adaptation in new domains. The key is to translate causal assumptions into learning principles that disentangle ``invariant'' and ``non-stable'' features. Experiments on both synthetic and real-world data demonstrate the effectiveness of the proposed learning algorithm.

        ----

        ## [1511] Symplectic Spectrum Gaussian Processes: Learning Hamiltonians from Noisy and Sparse Data

        **Authors**: *Yusuke Tanaka, Tomoharu Iwata, Naonori Ueda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html)

        **Abstract**:

        Hamiltonian mechanics is a well-established theory for modeling the time evolution of systems with conserved quantities (called Hamiltonian), such as the total energy of the system. Recent works have parameterized the Hamiltonian by machine learning models (e.g., neural networks), allowing Hamiltonian dynamics to be obtained from state trajectories without explicit mathematical modeling. However, the performance of existing models is limited as we can observe only noisy and sparse trajectories in practice. This paper proposes a probabilistic model that can learn the dynamics of conservative or dissipative systems from noisy and sparse data. We introduce a Gaussian process that incorporates the symplectic geometric structure of Hamiltonian systems, which is used as a prior distribution for estimating Hamiltonian systems with additive dissipation. We then present its spectral representation, Symplectic Spectrum Gaussian Processes (SSGPs), for which we newly derive random Fourier features with symplectic structures. This allows us to construct an efficient variational inference algorithm for training the models while simulating the dynamics via ordinary differential equation solvers. Experiments on several physical systems show that SSGP offers excellent performance in predicting dynamics that follow the energy conservation or dissipation law from noisy and sparse data.

        ----

        ## [1512] Multiclass Learnability Beyond the PAC Framework: Universal Rates and Partial Concept Classes

        **Authors**: *Alkis Kalavasis, Grigoris Velegkas, Amin Karbasi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html)

        **Abstract**:

        In this paper we study the problem of multiclass classification with a bounded number of different labels $k$, in the realizable setting. We extend the traditional PAC model to a) distribution-dependent learning rates, and b) learning rates under data-dependent assumptions. First, we consider the universal learning setting (Bousquet, Hanneke, Moran, van Handel and Yehudayoff, STOC'21), for which we provide a complete characterization of the achievable learning rates that holds for every fixed distribution. In particular, we show the following trichotomy: for any concept class, the optimal learning rate is either exponential, linear or arbitrarily slow. Additionally, we provide complexity measures of the underlying hypothesis class that characterize when these rates occur. Second, we consider the problem of multiclass classification with structured data (such as data lying on a low dimensional manifold or satisfying margin conditions), a setting which is captured by partial concept classes (Alon, Hanneke, Holzman and Moran, FOCS'21). Partial concepts are functions that can be undefined in certain parts of the input space. We extend the traditional PAC learnability of total concept classes to partial concept classes in the multiclass setting and investigate differences between partial and total concepts.

        ----

        ## [1513] MetaTeacher: Coordinating Multi-Model Domain Adaptation for Medical Image Classification

        **Authors**: *Zhenbin Wang, Mao Ye, Xiatian Zhu, Liuhan Peng, Liang Tian, Yingying Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html)

        **Abstract**:

        In medical image analysis, we often need to build an image recognition system for a target scenario with the access to small labeled data and abundant unlabeled data, as well as multiple related models pretrained on different source scenarios. This presents the combined challenges of multi-source-free domain adaptation and semi-supervised learning simultaneously. However, both problems are typically studied independently in the literature, and how to effectively combine existing methods is non-trivial in design. In this work, we introduce a novel MetaTeacher framework with three key components: (1) A learnable coordinating scheme for adaptive domain adaptation of individual source models, (2) A mutual feedback mechanism between the target model and source models for more coherent learning, and (3) A semi-supervised bilevel optimization algorithm for consistently organizing the adaption of source models and the learning of target model. It aims to leverage the knowledge of source models adaptively whilst maximize their complementary benefits collectively to counter the challenge of limited supervision. Extensive experiments on five chest x-ray image datasets show that our method outperforms clearly all the state-of-the-art alternatives. The code is available at https://github.com/wongzbb/metateacher.

        ----

        ## [1514] Biological Learning of Irreducible Representations of Commuting Transformations

        **Authors**: *Alexander Genkin, David Lipshutz, Siavash Golkar, Tiberiu Tesileanu, Dmitri B. Chklovskii*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/834f4c0b8d241b4943a9dcb77fd85675-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/834f4c0b8d241b4943a9dcb77fd85675-Abstract-Conference.html)

        **Abstract**:

        A longstanding challenge in neuroscience is to understand neural mechanisms underlying the brain’s remarkable ability to learn and detect transformations of objects due to motion. Translations and rotations of images can be viewed as orthogonal transformations in the space of pixel intensity vectors. Every orthogonal transformation can be decomposed into rotations within irreducible two-dimensional subspaces (or representations). For sets of commuting transformations, known as toroidal groups, Cohen and Welling proposed a mathematical framework for learning the irreducible representations. We explore the possibility that the brain also learns irreducible representations using a biologically plausible learning mechanism. The first is based on SVD of the anti-symmetrized outer product of the vectors representing consecutive images and is implemented by a single-layer neural network. The second is based on PCA of the difference between consecutive frames and is implemented in a two-layer network but with greater biological plausibility. Both networks learn image rotations (replicating Cohen and Welling’s results) as well as  translations. It would be interesting to search for the proposed networks in nascent connectomics and physiology datasets.

        ----

        ## [1515] A Fourier Approach to Mixture Learning

        **Authors**: *Mingda Qiao, Guru Guruganesh, Ankit Singh Rawat, Kumar Avinava Dubey, Manzil Zaheer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8353c5035fe18b4fadd350228b4e0688-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8353c5035fe18b4fadd350228b4e0688-Abstract-Conference.html)

        **Abstract**:

        We revisit the problem of learning mixtures of spherical Gaussians. Given samples from a mixture $\frac{1}{k}\sum_{j=1}^{k}\mathcal{N}(\mu_j, I_d)$, the goal is to estimate the means $\mu_1, \mu_2, \ldots, \mu_k \in \mathbb{R}^d$ up to a small error. The hardness of this learning problem can be measured by the \emph{separation} $\Delta$ defined as the minimum distance between all pairs of means. Regev and Vijayaraghavan (2017) showed that with $\Delta = \Omega(\sqrt{\log k})$ separation, the means can be learned using $\mathrm{poly}(k, d)$ samples, whereas super-polynomially many samples are required if $\Delta = o(\sqrt{\log k})$ and $d = \Omega(\log k)$. This leaves open the low-dimensional regime where $d = o(\log k)$.    In this work, we give an algorithm that efficiently learns the means in $d = O(\log k/\log\log k)$ dimensions under separation $d/\sqrt{\log k}$ (modulo doubly logarithmic factors). This separation is strictly smaller than $\sqrt{\log k}$, and is also shown to be necessary. Along with the results of Regev and Vijayaraghavan (2017), our work almost pins down the critical separation threshold at which efficient parameter learning becomes possible for spherical Gaussian mixtures. More generally, our algorithm runs in time $\mathrm{poly}(k)\cdot f(d, \Delta, \epsilon)$, and is thus fixed-parameter tractable in parameters $d$, $\Delta$ and $\epsilon$.    Our approach is based on estimating the Fourier transform of the mixture at carefully chosen frequencies, and both the algorithm and its analysis are simple and elementary. Our positive results can be easily extended to learning mixtures of non-Gaussian distributions, under a mild condition on the Fourier spectrum of the distribution.

        ----

        ## [1516] Structured Energy Network As a Loss

        **Authors**: *Jay Yoon Lee, Dhruvesh Patel, Purujit Goyal, Wenlong Zhao, Zhiyang Xu, Andrew McCallum*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/83ae75c127e2a3ea3315379020f8c19f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/83ae75c127e2a3ea3315379020f8c19f-Abstract-Conference.html)

        **Abstract**:

        Belanger & McCallum (2016) and Gygli et al. (2017) have shown that an energy network can capture arbitrary dependencies amongst the output variables in structured prediction; however, their reliance on gradient-based inference (GBI) makes the inference slow and unstable. In this work, we propose Structured Energy As Loss (SEAL) to take advantage of the expressivity of energy networks without incurring the high inference cost. This is a novel learning framework that uses an energy network as a trainable loss function (loss-net) to train a separate neural network (task-net), which is then used to perform the inference through a forward pass. We establish SEAL as a general framework wherein various learning strategies like margin-based, regression, and noise-contrastive, could be employed to learn the parameters of loss-net.  Through extensive evaluation on multi-label classification, semantic role labeling, and image segmentation, we demonstrate that SEAL provides various useful design choices, is faster at inference than GBI, and leads to significant performance gains over the baselines.

        ----

        ## [1517] Bayesian inference via sparse Hamiltonian flows

        **Authors**: *Naitong Chen, Zuheng Xu, Trevor Campbell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/83b17fb3369b1effa97ca5409526b02e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/83b17fb3369b1effa97ca5409526b02e-Abstract-Conference.html)

        **Abstract**:

        A Bayesian coreset is a small, weighted subset of data that replaces the full dataset during Bayesian inference, with the goal of reducing computational cost.  Although past work has shown empirically that there often exists a coreset with low inferential error, efficiently constructing such a coreset remains a challenge.  Current methods tend to be slow, require a secondary inference step after coreset construction, and do not provide bounds on the data marginal evidence.  In this work, we introduce a new method---sparse Hamiltonian flows---that addresses all three of these challenges.  The method involves first subsampling the data uniformly, and then optimizing a Hamiltonian flow parametrized by coreset weights and including periodic momentum quasi-refreshment steps.  Theoretical results show that the method enables an exponential compression of the dataset in a representative model, and that the quasi-refreshment steps reduce the KL divergence to the target.  Real and synthetic experiments demonstrate that sparse Hamiltonian flows provide accurate posterior approximations with significantly reduced runtime compared with competing dynamical-system-based inference methods.

        ----

        ## [1518] SAPA: Similarity-Aware Point Affiliation for Feature Upsampling

        **Authors**: *Hao Lu, Wenze Liu, Zixuan Ye, Hongtao Fu, Yuliang Liu, Zhiguo Cao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/83ccb398f3ce9c4d137011f36a03c7d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/83ccb398f3ce9c4d137011f36a03c7d4-Abstract-Conference.html)

        **Abstract**:

        We introduce point affiliation into feature upsampling, a notion that describes the affiliation of each upsampled point to a semantic cluster formed by local decoder feature points with semantic similarity. By rethinking point affiliation, we present a generic formulation for generating upsampling kernels. The kernels encourage not only semantic smoothness but also boundary sharpness in the upsampled feature maps. Such properties are particularly useful for some dense prediction tasks such as semantic segmentation. The key idea of our formulation is to generate similarity-aware kernels by comparing the similarity between each encoder feature point and the spatially associated local region of decoder features. In this way, the encoder feature point can function as a cue to inform the semantic cluster of upsampled feature points. To embody the formulation, we further instantiate a lightweight upsampling operator, termed Similarity-Aware Point Affiliation (SAPA), and investigate its variants. SAPA invites consistent performance improvements on a number of dense prediction tasks, including semantic segmentation, object detection, depth estimation, and image matting. Code is available at: https://github.com/poppinace/sapa

        ----

        ## [1519] Losses Can Be Blessings: Routing Self-Supervised Speech Representations Towards Efficient Multilingual and Multitask Speech Processing

        **Authors**: *Yonggan Fu, Yang Zhang, Kaizhi Qian, Zhifan Ye, Zhongzhi Yu, Cheng-I Jeff Lai, Celine Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/83d349b6eb8125588b5f091e2d47525c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/83d349b6eb8125588b5f091e2d47525c-Abstract-Conference.html)

        **Abstract**:

        Self-supervised learning (SSL) for rich speech representations has achieved empirical success in low-resource Automatic Speech Recognition (ASR) and other speech processing tasks, which can mitigate the necessity of a large amount of transcribed speech and thus has driven a growing demand for on-device ASR and other speech processing. However, advanced speech SSL models have become increasingly large, which contradicts the limited on-device resources. This gap could be more severe in multilingual/multitask scenarios requiring simultaneously recognizing multiple languages or executing multiple speech processing tasks. Additionally, strongly overparameterized speech SSL models tend to suffer from overfitting when being finetuned on low-resource speech corpus. This work aims to enhance the practical usage of speech SSL models towards a win-win in both enhanced efficiency and alleviated overfitting via our proposed S$^3$-Router framework, which for the first time discovers that simply discarding no more than 10% of model weights via only finetuning model connections of speech SSL models can achieve better accuracy over standard weight finetuning on downstream speech processing tasks. More importantly, S$^3$-Router can serve as an all-in-one technique to enable (1) a new finetuning scheme, (2) an efficient multilingual/multitask solution, (3) a state-of-the-art pruning technique, and (4) a new tool to quantitatively analyze the learned speech representation. We believe S$^3$-Router has provided a new perspective for practical deployment of speech SSL models. Our codes are available at: https://github.com/GATECH-EIC/S3-Router.

        ----

        ## [1520] Gradient Methods Provably Converge to Non-Robust Networks

        **Authors**: *Gal Vardi, Gilad Yehudai, Ohad Shamir*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/83e6913572ba09b0ab53c64c016c7d1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/83e6913572ba09b0ab53c64c016c7d1a-Abstract-Conference.html)

        **Abstract**:

        Despite a great deal of research, it is still unclear why neural networks are so susceptible to adversarial examples. In this work, we identify natural settings where depth-$2$ ReLU networks trained with gradient flow are provably non-robust (susceptible to small adversarial $\ell_2$-perturbations), even when robust networks that classify the training dataset correctly exist.Perhaps surprisingly, we show that the well-known implicit bias towards margin maximization induces bias towards non-robust networks, by proving that every network which satisfies the KKT conditions of the max-margin problem is non-robust.

        ----

        ## [1521] Diversity vs Recognizability: Human-like generalization in one-shot generative models

        **Authors**: *Victor Boutin, Lakshya Singhal, Xavier Thomas, Thomas Serre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8402cf3031d649066ada24514739f0dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8402cf3031d649066ada24514739f0dd-Abstract-Conference.html)

        **Abstract**:

        Robust generalization to new concepts has long remained a distinctive feature of human intelligence. However, recent progress in deep generative models has now led to neural architectures capable of synthesizing novel instances of unknown visual concepts from a single training example. Yet, a more precise comparison between these models and humans is not possible because existing performance metrics for generative models (i.e., FID, IS, likelihood) are not appropriate for the one-shot generation scenario. Here, we propose a new framework to evaluate one-shot generative models along two axes: sample recognizability vs. diversity  (i.e., intra-class variability). Using this framework, we perform a systematic evaluation of representative one-shot generative models on the Omniglot handwritten dataset. We first show that GAN-like and VAE-like models fall on opposite ends of the diversity-recognizability space. Extensive analyses of the effect of key model parameters further revealed that spatial attention and context integration have a linear contribution to the diversity-recognizability trade-off. In contrast, disentanglement transports the model along a parabolic curve that could be used to maximize recognizability. Using the diversity-recognizability framework, we were able to identify models and parameters that closely approximate human data.

        ----

        ## [1522] Task-level Differentially Private Meta Learning

        **Authors**: *Xinyu Zhou, Raef Bassily*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/842b82470c93a6c72284a3e83bdaced5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/842b82470c93a6c72284a3e83bdaced5-Abstract-Conference.html)

        **Abstract**:

        We study the problem of meta-learning with task-level differential privacy. Meta-learning has received increasing attention recently because of its ability to enable fast generalization to new task with small number of data points. However, the training process of meta learning likely involves exchange of task specific information, which may pose privacy risk especially in some privacy-sensitive applications. Therefore, it is important to provide strong privacy guarantees such that the learning process will not reveal any task sensitive information. To this end, existing works have proposed meta learning algorithms with record-level differential privacy, which is not sufficient in many scenarios since it does not protect the aggregated statistics based on the task dataset as a whole. Moreover, the utility guarantees in the prior work are based on assuming that the loss function satisfies both smoothness and quadratic growth conditions, which do not necessarily hold in practice. To address these issues, we propose meta learning algorithms with task-level differential privacy; that is, our algorithms protect the privacy of the entire dataset for each task. In the case when a single meta model is trained, we give both privacy and utility guarantees assuming only that the loss is convex and Lipschitz. Moreover, we propose a new private clustering-based meta-learning algorithm that enables private meta learning of multiple meta models. This can provide significant accuracy gains over the single meta model paradigm, especially when the tasks distribution cannot be well represented by a single meta model. Finally, we conduct several experiments demonstrating the effectiveness of our proposed algorithms.

        ----

        ## [1523] On the Statistical Efficiency of Reward-Free Exploration in Non-Linear RL

        **Authors**: *Jinglin Chen, Aditya Modi, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8433bb4f7477bf8202614ce1ae8b1169-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8433bb4f7477bf8202614ce1ae8b1169-Abstract-Conference.html)

        **Abstract**:

        We study reward-free reinforcement learning (RL) under general non-linear function approximation, and establish sample efficiency and hardness results under various standard structural assumptions. On the positive side, we propose the RFOLIVE (Reward-Free OLIVE) algorithm for sample-efficient reward-free exploration under minimal structural assumptions, which covers the previously studied settings of linear MDPs (Jin et al., 2020b), linear completeness (Zanette et al., 2020b) and low-rank MDPs with unknown representation (Modi et al., 2021). Our analyses indicate that the explorability or reachability assumptions, previously made for the latter two settings, are not necessary statistically for reward-free exploration. On the negative side, we provide a statistical hardness result for both reward-free and reward-aware exploration under linear completeness assumptions when the underlying features are unknown, showing an exponential separation between low-rank and linear completeness settings.

        ----

        ## [1524] Pessimism for Offline Linear Contextual Bandits using $\ell_p$ Confidence Sets

        **Authors**: *Gene Li, Cong Ma, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8443219a991f068c34d9491ad68ffa94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8443219a991f068c34d9491ad68ffa94-Abstract-Conference.html)

        **Abstract**:

        We present a family $\{\widehat{\pi}_p\}_{p\ge 1}$ of pessimistic learning rules for offline learning of linear contextual bandits, relying on confidence sets with respect to different $\ell_p$ norms, where $\widehat{\pi}_2$ corresponds to Bellman-consistent pessimism (BCP), while $\widehat{\pi}_\infty$ is a novel generalization of lower confidence bound (LCB) to the linear setting.  We show that the novel $\widehat{\pi}_\infty$ learning rule is, in a sense, adaptively optimal, as it achieves the minimax performance (up to log factors) against all $\ell_q$-constrained problems, and as such it strictly dominates all other predictors in the family, including $\widehat{\pi}_2$.

        ----

        ## [1525] Discrete-Convex-Analysis-Based Framework for Warm-Starting Algorithms with Predictions

        **Authors**: *Shinsaku Sakaue, Taihei Oki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/844e61124d9e1f58632bf0c8968ad728-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/844e61124d9e1f58632bf0c8968ad728-Abstract-Conference.html)

        **Abstract**:

        Augmenting algorithms with learned predictions is a promising approach for going beyond worst-case bounds. Dinitz, Im, Lavastida, Moseley, and Vassilvitskii~(2021) have demonstrated that warm-starts with learned dual solutions can improve the time complexity of the Hungarian method for weighted perfect bipartite matching. We extend and improve their framework in a principled manner via \textit{discrete convex analysis} (DCA), a discrete analog of convex analysis. We show the usefulness of our DCA-based framework by applying it to weighted perfect bipartite matching, weighted matroid intersection, and discrete energy minimization for computer vision. Our DCA-based framework yields time complexity bounds that depend on the $\ell_\infty$-distance from a predicted solution to an optimal solution, which has two advantages relative to the previous $\ell_1$-distance-dependent bounds: time complexity bounds are smaller, and learning of predictions is more sample efficient. We also discuss whether to learn primal or dual solutions from the DCA perspective.

        ----

        ## [1526] CAESAR: An Embodied Simulator for Generating Multimodal Referring Expression Datasets

        **Authors**: *Md Mofijul Islam, Reza Mirzaiee, Alexi Gladstone, Haley N. Green, Tariq Iqbal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Humans naturally use verbal utterances and nonverbal gestures to refer to various objects (known as $\textit{referring expressions}$) in different interactional scenarios. As collecting real human interaction datasets are costly and laborious, synthetic datasets are often used to train models to unambiguously detect relationships among objects. However, existing synthetic data generation tools that provide referring expressions generally neglect nonverbal gestures. Additionally, while a few small-scale datasets contain multimodal cues (verbal and nonverbal), these datasets only capture the nonverbal gestures from an exo-centric perspective (observer). As models can use complementary information from multimodal cues to recognize referring expressions, generating multimodal data from multiple views can help to develop robust models. To address these critical issues, in this paper, we present a novel embodied simulator, CAESAR, to generate multimodal referring expressions containing both verbal utterances and nonverbal cues captured from multiple views. Using our simulator, we have generated two large-scale embodied referring expression datasets, which we have released publicly. We have conducted experimental analyses on embodied spatial relation grounding using various state-of-the-art baseline models. Our experimental results suggest that visual perspective affects the models' performance; and that nonverbal cues improve spatial relation grounding accuracy. Finally, we will release the simulator publicly to allow researchers to generate new embodied interaction datasets.

        ----

        ## [1527] Generalizing Bayesian Optimization with Decision-theoretic Entropies

        **Authors**: *Willie Neiswanger, Lantao Yu, Shengjia Zhao, Chenlin Meng, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8451a20c5a7e0ee5671dda28f7daf7f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8451a20c5a7e0ee5671dda28f7daf7f3-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimization (BO) is a popular method for efficiently inferring optima of an expensive black-box function via a sequence of queries. Existing information-theoretic BO procedures aim to make queries that most reduce the uncertainty about optima, where the uncertainty is captured by Shannon entropy. However, an optimal measure of uncertainty would, ideally, factor in how we intend to use the inferred quantity in some downstream procedure. In this paper, we instead consider a generalization of Shannon entropy from work in statistical decision theory (DeGroot 1962, Rao 1984), which contains a broad class of uncertainty measures parameterized by a problem-specific loss function corresponding to a downstream task. We first show that special cases of this entropy lead to popular acquisition functions used in BO procedures such as knowledge gradient, expected improvement, and entropy search. We then show how alternative choices for the loss yield a flexible family of acquisition functions that can be customized for use in novel optimization settings. Additionally, we develop gradient-based methods to efficiently optimize our proposed family of acquisition functions, and demonstrate strong empirical performance on a diverse set of sequential decision making tasks, including variants of top-$k$ optimization, multi-level set estimation, and sequence search.

        ----

        ## [1528] Ordered Subgraph Aggregation Networks

        **Authors**: *Chendi Qian, Gaurav Rattan, Floris Geerts, Mathias Niepert, Christopher Morris*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8471dc3f5180df17e2fa84f106f1ee8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8471dc3f5180df17e2fa84f106f1ee8e-Abstract-Conference.html)

        **Abstract**:

        Numerous subgraph-enhanced graph neural networks (GNNs) have emerged recently, provably boosting the expressive power of standard (message-passing) GNNs. However, there is a limited understanding of how these approaches relate to each other and to the Weisfeiler-Leman hierarchy. Moreover, current approaches either use all subgraphs of a given size, sample them uniformly at random, or use hand-crafted heuristics instead of learning to select subgraphs in a data-driven manner. Here, we offer a unified way to study such architectures by introducing a theoretical framework and extending the known expressivity results of subgraph-enhanced GNNs. Concretely, we show that increasing subgraph size always increases the expressive power and develop a better understanding of their limitations by relating them to the established $k\mathsf{\text{-}WL}$ hierarchy. In addition, we explore different approaches for learning to sample subgraphs using recent methods for backpropagating through complex discrete probability distributions. Empirically, we study the predictive performance of different subgraph-enhanced GNNs, showing that our data-driven architectures increase prediction accuracy on standard benchmark datasets compared to non-data-driven subgraph-enhanced graph neural networks while reducing computation time.

        ----

        ## [1529] Neuron with Steady Response Leads to Better Generalization

        **Authors**: *Qiang Fu, Lun Du, Haitao Mao, Xu Chen, Wei Fang, Shi Han, Dongmei Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/848784373188ddf641079524e89e0ac9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/848784373188ddf641079524e89e0ac9-Abstract-Conference.html)

        **Abstract**:

        Regularization can mitigate the generalization gap between training and inference by introducing inductive bias. Existing works have already proposed various inductive biases from diverse perspectives. However, none of them explores inductive bias from the perspective of class-dependent response distribution of individual neurons. In this paper, we conduct a substantial analysis of the characteristics of such distribution. Based on the analysis results, we articulate the Neuron Steadiness Hypothesis: the neuron with similar responses to instances of the same class leads to better generalization. Accordingly, we propose a new regularization method called Neuron Steadiness Regularization (NSR) to reduce neuron intra-class response variance. Based on the Complexity Measure, we theoretically guarantee the effectiveness of NSR for improving generalization. We conduct extensive experiments on Multilayer Perceptron, Convolutional Neural Networks, and Graph Neural Networks with popular benchmark datasets of diverse domains, which show that our Neuron Steadiness Regularization consistently outperforms the vanilla version of models with significant gain and low additional computational overhead.

        ----

        ## [1530] Laplacian Autoencoders for Learning Stochastic Representations

        **Authors**: *Marco Miani, Frederik Warburg, Pablo Moreno-Muñoz, Nicki Skafte Detlefsen, Søren Hauberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/84880289c9fcba0d4bdb198cdb8f5080-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/84880289c9fcba0d4bdb198cdb8f5080-Abstract-Conference.html)

        **Abstract**:

        Established methods for unsupervised representation learning such as variational autoencoders produce none or poorly calibrated uncertainty estimates making it difficult to evaluate if learned representations are stable and reliable. In this work, we present a Bayesian autoencoder for unsupervised representation learning, which is trained using a novel variational lower-bound of the autoencoder evidence. This is maximized using Monte Carlo EM with a variational distribution that takes the shape of a Laplace approximation. We develop a new Hessian approximation that scales linearly with data size allowing us to model high-dimensional data. Empirically, we show that our Laplacian autoencoder estimates well-calibrated uncertainties in both latent and output space. We demonstrate that this results in improved performance across a multitude of downstream tasks.

        ----

        ## [1531] Alleviating the Sample Selection Bias in Few-shot Learning by Removing Projection to the Centroid

        **Authors**: *Jing Xu, Xu Luo, Xinglin Pan, Yanan Li, Wenjie Pei, Zenglin Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html)

        **Abstract**:

        Few-shot learning (FSL) targets at generalization of vision models towards unseen tasks without sufficient annotations. Despite the emergence of a number of few-shot learning methods, the sample selection bias problem, i.e., the sensitivity to the limited amount of support data, has not been well understood. In this paper, we find that this problem usually occurs when the positions of support samples are in the vicinity of task centroidâ€”the mean of all class centroids in the task. This motivates us to propose an extremely simple feature transformation to alleviate this problem, dubbed Task Centroid Projection Removing (TCPR). TCPR is applied directly to all image features in a given task, aiming at removing the dimension of features along the direction of the task centroid. While the exact task centoid cannot be accurately obtained from limited data, we estimate it using base features that are each similar to one of the support features. Our method effectively prevents features from being too close to the task centroid. Extensive experiments over ten datasets from different domains show that TCPR can reliably improve classification accuracy across various feature extractors, training algorithms and datasets. The code has been made available at https://github.com/KikimorMay/FSL-TCBR.

        ----

        ## [1532] A Coupled Design of Exploiting Record Similarity for Practical Vertical Federated Learning

        **Authors**: *Zhaomin Wu, Qinbin Li, Bingsheng He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/84b744165a0597360caad96b06e69313-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/84b744165a0597360caad96b06e69313-Abstract-Conference.html)

        **Abstract**:

        Federated learning is a learning paradigm to enable collaborative learning across different parties without revealing raw data. Notably, vertical federated learning (VFL), where parties share the same set of samples but only hold partial features, has a wide range of real-world applications. However, most existing studies in VFL disregard the "record linkage'' process. They design algorithms either assuming the data from different parties can be exactly linked or simply linking each record with its most similar neighboring record. These approaches may fail to capture the key features from other less similar records. Moreover, such improper linkage cannot be corrected by training since existing approaches provide no feedback on linkage during training. In this paper, we design a novel coupled training paradigm, FedSim, that integrates one-to-many linkage into the training process. Besides enabling VFL in many real-world applications with fuzzy identifiers, FedSim also achieves better performance in traditional VFL tasks. Moreover, we theoretically analyze the additional privacy risk incurred by sharing similarities. Our experiments on eight datasets with various similarity metrics show that FedSim outperforms other state-of-the-art baselines. The codes of FedSim are available at https://github.com/Xtra-Computing/FedSim.

        ----

        ## [1533] Cooperative Distribution Alignment via JSD Upper Bound

        **Authors**: *Wonwoong Cho, Ziyu Gong, David I. Inouye*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/84b8d9fcb4e262fcd429544697e1e720-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/84b8d9fcb4e262fcd429544697e1e720-Abstract-Conference.html)

        **Abstract**:

        Unsupervised distribution alignment estimates a transformation that maps two or more source distributions to a shared aligned distribution given only samples from each distribution. This task has many applications including generative modeling, unsupervised domain adaptation, and socially aware learning. Most prior works use adversarial learning (i.e., min-max optimization), which can be challenging to optimize and evaluate. A few recent works explore non-adversarial flow-based (i.e., invertible) approaches, but they lack a unified perspective and are limited in efficiently aligning multiple distributions. Therefore, we propose to unify and generalize previous flow-based approaches under a single non-adversarial framework, which we prove is equivalent to minimizing an upper bound on the Jensen-Shannon Divergence (JSD). Importantly, our problem reduces to a min-min, i.e., cooperative, problem and can provide a natural evaluation metric for unsupervised distribution alignment. We show empirical results on both simulated and real-world datasets to demonstrate the benefits of our approach. Code is available at https://github.com/inouye-lab/alignment-upper-bound.

        ----

        ## [1534] An Analytical Theory of Curriculum Learning in Teacher-Student Networks

        **Authors**: *Luca Saglietti, Stefano Sarao Mannelli, Andrew M. Saxe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html)

        **Abstract**:

        In animals and humans, curriculum learning---presenting data in a curated order---is critical to rapid learning and effective pedagogy.     A long history of experiments has demonstrated the impact of curricula in a variety of animals but, despite its ubiquitous presence, a theoretical understanding of the phenomenon is still lacking.     Surprisingly, in contrast to animal learning, curricula strategies are not widely used in machine learning and recent simulation studies reach the conclusion that curricula are moderately effective or ineffective in most cases.     This stark difference in the importance of curriculum raises a fundamental theoretical question: when and why does curriculum learning help?     In this work, we analyse a prototypical neural network model of curriculum learning in the high-dimensional limit, employing statistical physics methods.     We study a task in which a sparse set of informative features are embedded amidst a large set of noisy features. We analytically derive average learning trajectories for simple neural networks on this task, which establish a clear speed benefit for curriculum learning in the online setting. However, when training experiences can be stored and replayed (for instance, during sleep), the advantage of curriculum in standard neural networks disappears, in line with observations from the deep learning literature.     Inspired by synaptic consolidation techniques developed to combat catastrophic forgetting, we investigate whether consolidating synapses at curriculum change points can boost the benefits of curricula. We derive generalisation performance as a function of consolidation strength (implemented as a Gaussian prior connecting learning phases), and show that this consolidation mechanism can yield a large improvement in test performance.    Our reduced analytical descriptions help reconcile apparently conflicting empirical results, trace regimes where curriculum learning yields the largest gains, and provide experimentally-accessible predictions for the impact of task parameters on curriculum benefits. More broadly, our results suggest that fully exploiting a curriculum may require explicit consolidation at curriculum boundaries.

        ----

        ## [1535] RSA: Reducing Semantic Shift from Aggressive Augmentations for Self-supervised Learning

        **Authors**: *Yingbin Bai, Erkun Yang, Zhaoqing Wang, Yuxuan Du, Bo Han, Cheng Deng, Dadong Wang, Tongliang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/850e8063d902e0825d3c5504d183bafe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/850e8063d902e0825d3c5504d183bafe-Abstract-Conference.html)

        **Abstract**:

        Most recent self-supervised learning methods learn visual representation by contrasting different augmented views of images. Compared with supervised learning, more aggressive augmentations have been introduced to further improve the diversity of training pairs. However, aggressive augmentations may distort images' structures leading to a severe semantic shift problem that augmented views of the same image may not share the same semantics, thus degrading the transfer performance. To address this problem, we propose a new SSL paradigm, which counteracts the impact of semantic shift by balancing the role of weak and aggressively augmented pairs. Specifically, semantically inconsistent pairs are of minority, and we treat them as noisy pairs. Note that deep neural networks (DNNs) have a crucial memorization effect that DNNs tend to first memorize clean (majority) examples before overfitting to noisy (minority) examples. Therefore, we set a relatively large weight for aggressively augmented data pairs at the early learning stage. With the training going on, the model begins to overfit noisy pairs. Accordingly, we gradually reduce the weights of aggressively augmented pairs. In doing so, our method can better embrace aggressive augmentations and neutralize the semantic shift problem. Experiments show that our model achieves 73.1% top-1 accuracy on ImageNet-1K with ResNet-50 for 200 epochs, which is a 2.5% improvement over BYOL. Moreover, experiments also demonstrate that the learned representations can transfer well for various downstream tasks. Code is released at: https://github.com/tmllab/RSA.

        ----

        ## [1536] Evaluating Latent Space Robustness and Uncertainty of EEG-ML Models under Realistic Distribution Shifts

        **Authors**: *Neeraj Wagh, Jionghao Wei, Samarth Rawal, Brent M. Berry, Yogatheesan Varatharajah*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html)

        **Abstract**:

        The recent availability of large datasets in bio-medicine has inspired the development of representation learning methods for multiple healthcare applications. Despite advances in predictive performance, the clinical utility of such methods is limited when exposed to real-world data. This study develops model diagnostic measures to detect potential pitfalls before deployment without assuming access to external data. Specifically, we focus on modeling realistic data shifts in electrophysiological signals (EEGs) via data transforms and extend the conventional task-based evaluations with analyses of a) the model's latent space and b) predictive uncertainty under these transforms. We conduct experiments on multiple EEG feature encoders and two clinically relevant downstream tasks using publicly available large-scale clinical EEGs. Within this experimental setting, our results suggest that measures of latent space integrity and model uncertainty under the proposed data shifts may help anticipate performance degradation during deployment.

        ----

        ## [1537] u-HuBERT: Unified Mixed-Modal Speech Pretraining And Zero-Shot Transfer to Unlabeled Modality

        **Authors**: *Wei-Ning Hsu, Bowen Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/853e781cb2af58956ed5c89aa59da3fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/853e781cb2af58956ed5c89aa59da3fc-Abstract-Conference.html)

        **Abstract**:

        While audio-visual speech models can yield superior performance and robustness compared to audio-only models, their development and adoption are hindered by the lack of labeled and unlabeled audio-visual data and the cost to deploy one model per modality. In this paper, we present u-HuBERT, a self-supervised pre-training framework that can leverage both multimodal and unimodal speech with a unified masked cluster prediction objective. By utilizing modality dropout during pre-training, we demonstrate that a single fine-tuned model can achieve performance on par or better than the state-of-the-art modality-specific models. Moreover, our model fine-tuned only on audio can perform well with audio-visual and visual speech input, achieving zero-shot modality generalization for multiple speech processing tasks. In particular, our single model yields 1.2%/1.4%/27.2% speech recognition word error rate on LRS3 with audio-visual/audio/visual input.

        ----

        ## [1538] Hierarchical Graph Transformer with Adaptive Node Sampling

        **Authors**: *Zaixi Zhang, Qi Liu, Qingyong Hu, Chee-Kong Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html)

        **Abstract**:

        The Transformer architecture has achieved remarkable success in a number of domains including natural language processing and computer vision. However, when it comes to graph-structured data, transformers have not achieved competitive performance, especially on large graphs. In this paper, we identify the main deficiencies of current graph transformers: (1) Existing node sampling strategies in Graph Transformers are agnostic to the graph characteristics and the training process. (2) Most sampling strategies only focus on local neighbors and neglect the long-range dependencies in the graph. We conduct experimental investigations on synthetic datasets to show that existing sampling strategies are sub-optimal. To tackle the aforementioned problems, we formulate the optimization strategies of node sampling in Graph Transformer as an adversary bandit problem, where the rewards are related to the attention weights and can vary in the training procedure. Meanwhile, we propose a hierarchical attention scheme with graph coarsening to capture the long-range interactions while reducing computational complexity. Finally, we conduct extensive experiments on real-world datasets to demonstrate the superiority of our method over existing graph transformers and popular GNNs.

        ----

        ## [1539] Learning Options via Compression

        **Authors**: *Yiding Jiang, Evan Zheran Liu, Benjamin Eysenbach, J. Zico Kolter, Chelsea Finn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8567a53e58a9fa4823af356c76ed943c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8567a53e58a9fa4823af356c76ed943c-Abstract-Conference.html)

        **Abstract**:

        Identifying statistical regularities in solutions to some tasks in multi-task reinforcement learning can accelerate the learning of new tasks.Skill learning offers one way of identifying these regularities by decomposing pre-collected experiences into a sequence of skills.A popular approach to skill learning is maximizing the likelihood of the pre-collected experience with latent variable models,where the latent variables represent the skills. However, there are often many solutions that maximize the likelihood equally well, including degenerate solutions. To address this underspecification, we propose a new objective that combines the maximum likelihood objective with a penalty on the description length of the skills. This penalty incentivizes the skills to maximally extract common structures from the experiences. Empirically, our objective learns skills that solve downstream tasks in fewer samples compared to skills learned from only maximizing likelihood. Further, while most prior works in the offline multi-task setting focus on tasks with low-dimensional observations, our objective can scale to challenging tasks with high-dimensional image observations.

        ----

        ## [1540] Fixed-Distance Hamiltonian Monte Carlo

        **Authors**: *Hadi Mohasel Afshar, Sally Cripps*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/857b34d81f0a8bfe3e18879dee3b5086-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/857b34d81f0a8bfe3e18879dee3b5086-Abstract-Conference.html)

        **Abstract**:

        We propose a variation of the Hamiltonian Monte Carlo sampling (HMC) where the equations of motion are simulated for a fixed traversed distance rather than the conventional fixed simulation time. This new mechanism tends to generate proposals that have higher target probability values. The momentum distribution that is naturally joint with our Fixed-Distance HMC (FDHMC), and keeps the proposal acceptance probability close to 1, is not Gaussian and generates momentums that have  a higher expected magnitude. This translates into a reduced correlation between the successive MCMC states and according to our experimental results, leads to an  improvement in terms of the effective sample size per gradient when compared to the baseline HMC and No-U-Turn (NUTS) samplers.

        ----

        ## [1541] GlanceNets: Interpretable, Leak-proof Concept-based Models

        **Authors**: *Emanuele Marconato, Andrea Passerini, Stefano Teso*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html)

        **Abstract**:

        There is growing interest in concept-based models (CBMs) that combine high-performance and interpretability by acquiring and reasoning with a vocabulary of high-level concepts. A key requirement is that the concepts be interpretable. Existing CBMs tackle this desideratum using a variety of heuristics based on unclear notions of interpretability, and fail to acquire concepts with the intended semantics. We address this by providing a clear definition of interpretability in terms of alignment between the modelâ€™s representation and an underlying data generation process, and introduce GlanceNets, a new CBM that exploits techniques from disentangled representation learning and open-set recognition to achieve alignment, thus improving the interpretability of the learned concepts. We show that GlanceNets, paired with concept-level supervision, achieve better alignment than state-of-the-art approaches while preventing spurious information from unintendedly leaking into the learned concepts.

        ----

        ## [1542] 3DOS: Towards 3D Open Set Learning - Benchmarking and Understanding Semantic Novelty Detection on Point Clouds

        **Authors**: *Antonio Alliegro, Francesco Cappio Borlino, Tatiana Tommasi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/85b6841eaf79327b1777f9e64af3835d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/85b6841eaf79327b1777f9e64af3835d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In recent years there has been significant progress in the field of 3D learning on classification, detection and segmentation problems. The vast majority of the existing studies focus on canonical closed-set conditions, neglecting the intrinsic open nature of the real-world. This limits the abilities of robots and autonomous systems involved in safety-critical applications that require managing novel and unknown signals. In this context exploiting 3D data can be a valuable asset since it provides rich information about the geometry of perceived objects and scenes. With this paper we provide the first broad study on 3D Open Set learning. We introduce 3DOS: a novel testbed for semantic novelty detection that considers several settings with increasing difficulties in terms of semantic (category) shift, and covers both in-domain (synthetic-to-synthetic, real-to-real) and cross-domain (synthetic-to-real) scenarios. Moreover, we investigate the related 2D Open Set literature to understand if and how its recent improvements are effective on 3D data. Our extensive benchmark positions several algorithms in the same coherent picture, revealing their strengths and limitations. The results of our analysis may serve as a reliable foothold for future tailored 3D Open Set methods.

        ----

        ## [1543] Masked Prediction: A Parameter Identifiability View

        **Authors**: *Bingbin Liu, Daniel J. Hsu, Pradeep Ravikumar, Andrej Risteski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/85dd09d356ca561169b2c03e43cf305e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/85dd09d356ca561169b2c03e43cf305e-Abstract-Conference.html)

        **Abstract**:

        The vast majority of work in self-supervised learning have focused on assessing recovered features by a chosen set of downstream tasks. While there are several commonly used benchmark datasets, this lens of feature learning requires assumptions on the downstream tasks which are not inherent to the data distribution itself. In this paper, we present an alternative lens, one of parameter identifiability: assuming data comes from a parametric probabilistic model, we train a self-supervised learning predictor with a suitable parametric form, and ask whether the parameters of the optimal predictor can be used to extract the parameters of the ground truth generative model.Specifically, we focus on latent-variable models capturing sequential structures, namely Hidden Markov Models with both discrete and conditionally Gaussian observations. We focus on masked prediction as the self-supervised learning task and study the optimal masked predictor. We show that parameter identifiability is governed by the task difficulty, which is determined by the choice of data model and the amount of tokens to predict. Technique-wise, we uncover close connections with the uniqueness of tensor rank decompositions, a widely used tool in studying identifiability through the lens of the method of moments.

        ----

        ## [1544] Self-Supervised Learning of Brain Dynamics from Broad Neuroimaging Data

        **Authors**: *Armin W. Thomas, Christopher Ré, Russell A. Poldrack*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8600a9df1a087a9a66900cc8c948c3f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8600a9df1a087a9a66900cc8c948c3f0-Abstract-Conference.html)

        **Abstract**:

        Self-supervised learning techniques are celebrating immense success in natural language processing (NLP) by enabling models to learn from broad language data at unprecedented scales. Here, we aim to leverage the success of these techniques for mental state decoding, where researchers aim to identify specific mental states (e.g., the experience of anger or joy) from brain activity. To this end, we devise a set of novel self-supervised learning frameworks for neuroimaging data inspired by prominent learning frameworks in NLP. At their core, these frameworks learn the dynamics of brain activity by modeling sequences of activity akin to how sequences of text are modeled in NLP. We evaluate the frameworks by pre-training models on a broad neuroimaging dataset spanning functional Magnetic Resonance Imaging data from 11,980 experimental runs of 1,726 individuals across 34 datasets, and subsequently adapting the pre-trained models to benchmark mental state decoding datasets. The pre-trained models transfer well, generally outperforming baseline models trained from scratch, while models trained in a learning framework based on causal language modeling clearly outperform the others.

        ----

        ## [1545] Characterization of Excess Risk for Locally Strongly Convex Population Risk

        **Authors**: *Mingyang Yi, Ruoyu Wang, Zhi-Ming Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8607450734c14f1e0e1c89d35e1a9218-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8607450734c14f1e0e1c89d35e1a9218-Abstract-Conference.html)

        **Abstract**:

        We establish upper bounds for the expected excess risk of models trained by proper iterative algorithms which approximate the local minima. Unlike the results built upon the strong globally strongly convexity or global growth conditions e.g., PL-inequality, we only require the population risk to be \emph{locally} strongly convex around its local minima. Concretely, our bound under convex problems is of order $\tilde{\mathcal{O}}(1/n)$. For non-convex problems with $d$ model parameters such that $d/n$ is smaller than a threshold independent of $n$, the order of $\tilde{\mathcal{O}}(1/n)$ can be maintained if the empirical risk has no spurious local minima with high probability. Moreover, the bound for non-convex problem becomes $\tilde{\mathcal{O}}(1/\sqrt{n})$ without such assumption. Our results are derived via algorithmic stability and characterization of the empirical risk's landscape. Compared with the existing algorithmic stability based results, our bounds are dimensional insensitive and without restrictions on the algorithm's implementation, learning rate, and the number of iterations. Our bounds underscore that with locally strongly convex population risk, the models trained by any proper iterative algorithm can generalize well, even for non-convex problems, and $d$ is large.

        ----

        ## [1546] A Non-Asymptotic Moreau Envelope Theory for High-Dimensional Generalized Linear Models

        **Authors**: *Lijia Zhou, Frederic Koehler, Pragya Sur, Danica J. Sutherland, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/861f7dad098aec1c3560fb7add468d41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/861f7dad098aec1c3560fb7add468d41-Abstract-Conference.html)

        **Abstract**:

        We prove a new generalization bound that shows for any class of linear predictors in Gaussian space, the Rademacher complexity of the class and the training error under any continuous loss $\ell$ can control the test error under all Moreau envelopes of the loss $\ell$ . We use our finite-sample bound to directly recover the “optimistic rate” of Zhou et al. (2021) for linear regression with the square loss, which is known to be tight for minimal $\ell_2$-norm interpolation, but we also handle more general settings where the label is generated by a potentially misspecified multi-index model. The same argument can analyze noisy interpolation of max-margin classifiers through the squared hinge loss, and establishes consistency results in spiked-covariance settings. More generally, when the loss is only assumed to be Lipschitz, our bound effectively improves Talagrand’s well-known contraction lemma by a factor of two, and we prove uniform convergence of interpolators (Koehler et al. 2021) for all smooth, non-negative losses. Finally, we show that application of our generalization bound using localized Gaussian width will generally be sharp for empirical risk minimizers, establishing a non-asymptotic Moreau envelope theory for generalization that applies outside of proportional scaling regimes, handles model misspecification, and complements existing asymptotic Moreau envelope theories for M-estimation.

        ----

        ## [1547] Towards Efficient 3D Object Detection with Knowledge Distillation

        **Authors**: *Jihan Yang, Shaoshuai Shi, Runyu Ding, Zhe Wang, Xiaojuan Qi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8625a8c2be8ba5197b7a14833dbea8ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8625a8c2be8ba5197b7a14833dbea8ac-Abstract-Conference.html)

        **Abstract**:

        Despite substantial progress in 3D object detection, advanced 3D detectors often suffer from heavy computation overheads. To this end, we explore the potential of knowledge distillation (KD) for developing efficient 3D object detectors, focusing on popular pillar- and voxel-based detectors. In the absence of well-developed teacher-student pairs, we first study how to obtain student models with good trade offs between accuracy and efficiency from the perspectives of model compression and input resolution reduction. Then, we build a benchmark to assess existing KD methods developed in the 2D domain for 3D object detection upon six well-constructed teacher-student pairs. Further, we propose an improved KD pipeline incorporating an enhanced logit KD method that performs KD on only a few pivotal positions determined by teacher classification response and a teacher-guided student model initialization to facilitate transferring teacher model's feature extraction ability to students through weight inheritance. Finally, we conduct extensive experiments on the Waymo dataset. Our best performing model achieves $65.75\%$ LEVEL 2 mAPH surpassing its teacher model and requiring only $44\%$ of teacher flops. Our most efficient model runs 51 FPS on an NVIDIA A100, which is $2.2\times$ faster than PointPillar with even higher accuracy. Code will be available.

        ----

        ## [1548] CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning

        **Authors**: *Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven Chu-Hong Hoi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html)

        **Abstract**:

        Program synthesis or code generation aims to generate a program that satisfies a problem specification. Recent approaches using large-scale pretrained language models (LMs) have shown promising results, yet they have some critical limitations. In particular, they often follow a standard supervised fine-tuning procedure to train a code generation model from natural language problem descriptions and ground-truth programs only. Such paradigm largely ignores some important but potentially useful signals in the problem specification such as unit tests, which thus results in poor performance when solving complex unseen coding tasks. We propose “CodeRL” to address the limitations, a new framework for program synthesis tasks through pretrained LMs and deep reinforcement learning (RL). Specifically, during training, we treat the code-generating LM as an actor network, and introduce a critic network that is trained to predict the functional correctness of generated programs and provide dense feedback signals to the actor. During inference, we introduce a new generation procedure with a critical sampling strategy that allows a model to automatically regenerate programs based on feedback from example unit tests and critic scores. For the model backbones, we extended the encoder-decoder architecture of CodeT5 with enhanced learning objectives, larger model sizes, and better pretraining data. Our method not only achieves new SOTA results on the challenging APPS benchmark, but also shows strong zero-shot transfer capability with new SOTA results on the simpler MBPP benchmark.

        ----

        ## [1549] Leveraging the Hints: Adaptive Bidding in Repeated First-Price Auctions

        **Authors**: *Wei Zhang, Yanjun Han, Zhengyuan Zhou, Aaron Flores, Tsachy Weissman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html)

        **Abstract**:

        With the advent and increasing consolidation of e-commerce, digital advertising has very recently replaced traditional advertising as the main marketing force in the economy. In the past four years, a particularly important development in the digital advertising industry is the shift from second-price auctions to first-price auctions for online display ads. This shift immediately motivated the intellectually challenging question of how to bid in first-price auctions, because unlike in second-price auctions, bidding one's private value truthfully is no longer optimal. Following a series of recent works in this area, we consider a differentiated setup: we do not make any assumption about other bidders' maximum bid (i.e. it can be adversarial over time), and instead assume that we have access to a hint that serves as a prediction of other bidders' maximum bid, where the prediction is learned through some blackbox machine learning model. We consider two types of hints: one where a single point-prediction is available, and the other where a hint interval (representing a type of confidence region into which others' maximum bid falls) is available. We establish minimax optimal regret bounds for both cases and highlight the quantitatively different behavior between the two settings. We also provide improved regret bounds when the others' maximum bid exhibits the further structure of sparsity. Finally, we complement the theoretical results with demonstrations using real bidding data.

        ----

        ## [1550] Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization

        **Authors**: *Wenhao Gao, Tianfan Fu, Jimeng Sun, Connor W. Coley*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Molecular optimization is a fundamental goal in the chemical sciences and is of central interest to drug and material design. In recent years, significant progress has been made in solving challenging problems across various aspects of computational molecular optimizations, emphasizing high validity, diversity, and, most recently, synthesizability. Despite this progress, many papers report results on trivial or self-designed tasks, bringing additional challenges to directly assessing the performance of new methods. Moreover, the sample efficiency of the optimization---the number of molecules evaluated by the oracle---is rarely discussed, despite being an essential consideration for realistic discovery applications.To fill this gap, we have created an open-source benchmark for practical molecular optimization, PMO, to facilitate the transparent and reproducible evaluation of algorithmic advances in molecular optimization. This paper thoroughly investigates the performance of 25 molecular design algorithms on 23 single-objective (scalar) optimization tasks with a particular focus on sample efficiency. Our results show that most ``state-of-the-art'' methods fail to outperform their predecessors under a limited oracle budget allowing 10K queries and that no existing algorithm can efficiently solve certain molecular optimization problems in this setting. We analyze the influence of the optimization algorithm choices, molecular assembly strategies, and oracle landscapes on the optimization performance to inform future algorithm development and benchmarking. PMO provides a standardized experimental setup to comprehensively evaluate and compare new molecule optimization methods with existing ones. All code can be found at https://github.com/wenhao-gao/mol_opt.

        ----

        ## [1551] MGNNI: Multiscale Graph Neural Networks with Implicit Layers

        **Authors**: *Juncheng Liu, Bryan Hooi, Kenji Kawaguchi, Xiaokui Xiao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86485259bbe57852dc93477b1deb1f2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86485259bbe57852dc93477b1deb1f2b-Abstract-Conference.html)

        **Abstract**:

        Recently, implicit graph neural networks (GNNs) have been proposed to capture long-range dependencies in underlying graphs. In this paper, we introduce and justify two weaknesses of implicit GNNs: the constrained expressiveness due to their limited effective range for capturing long-range dependencies, and their lack of ability to capture multiscale information on graphs at multiple resolutions. To show the limited effective range of previous implicit GNNs, we first provide a theoretical analysis and point out the intrinsic relationship between the effective range and the convergence of iterative equations used in these models. To mitigate the mentioned weaknesses, we propose a multiscale graph neural network with implicit layers (MGNNI) which is able to model multiscale structures on graphs and has an expanded effective range for capturing long-range dependencies. We conduct comprehensive experiments for both node classification and graph classification to show that MGNNI outperforms representative baselines and has a better ability for multiscale modeling and capturing of long-range dependencies.

        ----

        ## [1552] UQGAN: A Unified Model for Uncertainty Quantification of Deep Classifiers trained via Conditional GANs

        **Authors**: *Philipp Oberdiek, Gernot A. Fink, Matthias Rottmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8648e249887ccb0fe8c067d596e35b40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8648e249887ccb0fe8c067d596e35b40-Abstract-Conference.html)

        **Abstract**:

        We present an approach to quantifying both aleatoric and epistemic uncertainty for deep neural networks in image classification, based on generative adversarial networks (GANs). While most works in the literature that use GANs to generate out-of-distribution (OoD) examples only focus on the evaluation of OoD detection, we present a GAN based approach to learn a classifier that produces proper uncertainties for OoD examples as well as for false positives (FPs). Instead of shielding the entire in-distribution data with GAN generated OoD examples which is state-of-the-art, we shield each class separately with out-of-class examples generated by a conditional GAN and complement this with a one-vs-all image classifier. In our experiments, in particular on CIFAR10, CIFAR100 and Tiny ImageNet, we improve over the OoD detection and FP detection performance of state-of-the-art GAN-training based classifiers. Furthermore, we also find that the generated GAN examples do not significantly affect the calibration error of our classifier and result in a significant gain in model accuracy.

        ----

        ## [1553] Audio-Driven Co-Speech Gesture Video Generation

        **Authors**: *Xian Liu, Qianyi Wu, Hang Zhou, Yuanqi Du, Wayne Wu, Dahua Lin, Ziwei Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html)

        **Abstract**:

        Co-speech gesture is crucial for human-machine interaction and digital entertainment. While previous works mostly map speech audio to human skeletons (e.g., 2D keypoints), directly generating speakers' gestures in the image domain remains unsolved. In this work, we formally define and study this challenging problem of audio-driven co-speech gesture video generation, i.e., using a unified framework to generate speaker image sequence driven by speech audio. Our key insight is that the co-speech gestures can be decomposed into common motion patterns and subtle rhythmic dynamics. To this end, we propose a novel framework, Audio-driveN Gesture vIdeo gEneration (ANGIE), to effectively capture the reusable co-speech gesture patterns as well as fine-grained rhythmic movements. To achieve high-fidelity image sequence generation, we leverage an unsupervised motion representation instead of a structural human body prior (e.g., 2D skeletons). Specifically, 1) we propose a vector quantized motion extractor (VQ-Motion Extractor) to summarize common co-speech gesture patterns from implicit motion representation to codebooks. 2) Moreover, a co-speech gesture GPT with motion refinement (Co-Speech GPT) is devised to complement the subtle prosodic motion details. Extensive experiments demonstrate that our framework renders realistic and vivid co-speech gesture video. Demo video and more resources can be found in: https://alvinliu0.github.io/projects/ANGIE

        ----

        ## [1554] Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off

        **Authors**: *Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frédéric Precioso, Stefano Melacci, Adrian Weller, Pietro Lió, Mateja Jamnik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/867c06823281e506e8059f5c13a57f75-Abstract-Conference.html)

        **Abstract**:

        Deploying AI-powered systems requires trustworthy models supporting effective human interactions, going beyond raw prediction accuracy. Concept bottleneck models promote trustworthiness by conditioning classification tasks on an intermediate level of human-like concepts. This enables human interventions which can correct mispredicted concepts to improve the model's performance. However, existing concept bottleneck models are unable to find optimal compromises between high task accuracy, robust concept-based explanations, and effective interventions on concepts---particularly in real-world conditions where complete and accurate concept supervisions are scarce. To address this, we propose Concept Embedding Models, a novel family of concept bottleneck models which goes beyond the current accuracy-vs-interpretability trade-off by learning interpretable high-dimensional concept representations. Our experiments demonstrate that Concept Embedding Models  (1) attain better or competitive task accuracy w.r.t. standard neural models without concepts, (2) provide concept representations capturing meaningful semantics including and beyond their ground truth labels, (3) support test-time concept interventions whose effect in test accuracy surpasses that in standard concept bottleneck models, and (4) scale to real-world conditions where complete concept supervisions are scarce.

        ----

        ## [1555] DENSE: Data-Free One-Shot Federated Learning

        **Authors**: *Jie Zhang, Chen Chen, Bo Li, Lingjuan Lyu, Shuang Wu, Shouhong Ding, Chunhua Shen, Chao Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract-Conference.html)

        **Abstract**:

        One-shot Federated Learning (FL) has recently emerged as a promising approach, which allows the central server to learn a model in a single communication round. Despite the low communication cost, existing one-shot FL methods are mostly impractical or face inherent limitations, \eg a public dataset is required, clients' models are homogeneous, and additional data/model information need to be uploaded. To overcome these issues, we propose a novel two-stage \textbf{D}ata-fre\textbf{E} o\textbf{N}e-\textbf{S}hot federated l\textbf{E}arning (DENSE) framework, which trains the global model by a data generation stage and a model distillation stage. DENSE is a practical one-shot FL method that can be applied in reality due to the following advantages:(1) DENSE requires no additional information compared with other methods (except the model parameters) to be transferred between clients and the server;(2) DENSE does not require any auxiliary dataset for training;(3) DENSE considers model heterogeneity in FL, \ie different clients can have different model architectures.Experiments on a variety of real-world datasets demonstrate the superiority of our method.For example, DENSE outperforms the best baseline method Fed-ADI by 5.08\% on CIFAR10 dataset.

        ----

        ## [1556] Searching for Better Spatio-temporal Alignment in Few-Shot Action Recognition

        **Authors**: *Yichao Cao, Xiu Su, Qingfei Tang, Shan You, Xiaobo Lu, Chang Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8693ee1ea821666f8569228d1ab38baf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8693ee1ea821666f8569228d1ab38baf-Abstract-Conference.html)

        **Abstract**:

        Spatio-Temporal feature matching and alignment are essential for few-shot action recognition as they determine the coherence and effectiveness of the temporal patterns. Nevertheless, this process could be not reliable, especially when dealing with complex video scenarios. In this paper, we propose to improve the performance of matching and alignment from the end-to-end design of models. Our solution comes at two-folds. First, we encourage to enhance the extracted Spatio-Temporal representations from few-shot videos in the perspective of architectures. With this aim, we propose a specialized transformer search method for videos, thus the spatial and temporal attention can be well-organized and optimized for stronger feature representations. Second, we also design an efficient non-parametric spatio-temporal prototype alignment strategy to better handle the high variability of motion. In particular, a query-specific class prototype will be generated for each query sample and category, which can better match query sequences against all support sequences. By doing so, our method SST enjoys significant superiority over the benchmark UCF101 and HMDB51 datasets. For example, with no pretraining, our method achieves 17.1\% Top-1 accuracy improvement than the baseline TRX on UCF101 5-way 1-shot setting but with only 3x fewer FLOPs.

        ----

        ## [1557] Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively

        **Authors**: *Haojie Zhang, Ge Li, Jia Li, Zhongjin Zhang, Yuqi Zhu, Zhi Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/869bfd807a513755bef25e3896a19a21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/869bfd807a513755bef25e3896a19a21-Abstract-Conference.html)

        **Abstract**:

        Large-scale pre-trained language models have achieved impressive results on a wide range of downstream tasks recently. However, fine-tuning an extremely large-scale pre-trained language model on limited target datasets is often plagued by overfitting and representation degradation. In this paper, we propose a Dynamic Parameter Selection (DPS) algorithm for the large-scale pre-trained models during fine-tuning, which adaptively selects a more promising subnetwork to perform staging updates based on gradients of back-propagation. Experiments on the GLUE benchmark show that DPS outperforms previous fine-tuning methods in terms of overall performance and stability, and consistently achieves better results with variable pre-trained language models. In addition, DPS brings a large magnitude of improvement in out-of-domain transferring experiments and low-resource scenarios, which shows that it can maintain stable general contextual features and reduce the representation collapse. We release our code at \url{https://github.com/ZhangHaojie077/DPS}.

        ----

        ## [1558] Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP

        **Authors**: *Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, Ludwig Schmidt*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86a8a512b27f49519594ebe89f66d708-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86a8a512b27f49519594ebe89f66d708-Abstract-Conference.html)

        **Abstract**:

        Web-crawled datasets have enabled remarkable generalization capabilities in recent image-text models such as CLIP (Contrastive Language-Image pre-training) or Flamingo, but little is known about the dataset creation processes. In this work, we introduce a testbed of six publicly available data sources---YFCC, LAION, Conceptual Captions, WIT, RedCaps, Shutterstock---to investigate how pre-training distributions induce robustness in CLIP. We find that the performance of the pre-training data varies substantially across distribution shifts, with no single data source dominating. Moreover, we systematically study the interactions between these data sources and find that mixing multiple sources does not necessarily yield better models, but rather dilutes the robustness of the best individual data source. We complement our empirical findings with theoretical insights from a simple setting, where combining the training data also results in diluted robustness. In addition, our theoretical model provides a candidate explanation for the success of the CLIP-based data filtering technique recently employed in the LAION dataset. Overall our results demonstrate that simply gathering a large amount of data from the web is not the most effective way to build a pre-training dataset for robust generalization, necessitating further study into dataset design. Code is available at https://github.com/mlfoundations/clipqualitynot_quantity.

        ----

        ## [1559] Stars: Tera-Scale Graph Building for Clustering and Learning

        **Authors**: *CJ Carey, Jonathan Halcrow, Rajesh Jayaram, Vahab Mirrokni, Warren Schudy, Peilin Zhong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86ab3ff2c1387c895766f5c5fc2b610c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86ab3ff2c1387c895766f5c5fc2b610c-Abstract-Conference.html)

        **Abstract**:

        A fundamental procedure in the analysis of massive datasets is the construction of similarity graphs. Such graphs play a key role for many downstream tasks, including clustering, classification, graph learning, and nearest neighbor search. For these tasks, it is critical to build graphs which are sparse yet still representative of the underlying data. The benefits of sparsity are twofold: firstly, constructing dense graphs is infeasible in practice for large datasets, and secondly, the runtime of downstream tasks is directly influenced by the sparsity of the similarity graph. In this work, we present Stars: a highly scalable method for building extremely sparse graphs via two-hop spanners, which are graphs where similar points are connected by a path of length at most two. Stars can construct two-hop spanners with significantly fewer similarity comparisons, which are a major bottleneck for learning based models where comparisons are expensive to evaluate. Theoretically, we demonstrate that Stars builds a graph in nearly-linear time, where approximate nearest neighbors are contained within two-hop neighborhoods. In practice, we have deployed Stars for multiple data sets allowing for graph building at the Tera-Scale, i.e., for graphs with hundreds of billions of nodes and tens of trillions of edges. We evaluate the performance of Stars for clustering and graph learning, and demonstrate 10~1000-fold improvements in pairwise similarity comparisons and significant running time speedups with negligible quality loss.

        ----

        ## [1560] Score-Based Diffusion meets Annealed Importance Sampling

        **Authors**: *Arnaud Doucet, Will Grathwohl, Alexander G. de G. Matthews, Heiko Strathmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86b7128efa3950df7c0f6c0342e6dcc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86b7128efa3950df7c0f6c0342e6dcc1-Abstract-Conference.html)

        **Abstract**:

        More than twenty years after its introduction, Annealed Importance Sampling (AIS) remains one of the most effective methods for marginal likelihood estimation. It relies on a sequence of distributions interpolating between a tractable initial distribution and the target distribution of interest which we simulate from approximately using a non-homogeneous Markov chain. To obtain an importance sampling estimate of the marginal likelihood, AIS introduces an extended target distribution to reweight the Markov chain proposal. While much effort has been devoted to improving the proposal distribution used by AIS, by changing the intermediate distributions and corresponding Markov kernels, an underappreciated issue is that AIS uses a convenient but suboptimal extended target distribution. This can hinder its performance. We here leverage recent progress in score-based generative modeling (SGM) to approximate the optimal extended target distribution for AIS proposals corresponding to the discretization of Langevin and Hamiltonian dynamics. We demonstrate these novel, differentiable, AIS procedures on a number of synthetic benchmark distributions and variational auto-encoders.

        ----

        ## [1561] PaCo: Parameter-Compositional Multi-task Reinforcement Learning

        **Authors**: *Lingfeng Sun, Haichao Zhang, Wei Xu, Masayoshi Tomizuka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html)

        **Abstract**:

        The purpose of multi-task reinforcement learning (MTRL) is to train a single policy that can be applied to a set of different tasks. Sharing parameters allows us to take advantage of the similarities among tasks. However, the gaps between contents and difficulties of different tasks bring us challenges on both which tasks should share the parameters and what parameters should be shared, as well as the optimization challenges due to parameter sharing. In this work, we introduce a parameter-compositional approach (PaCo) as an attempt to address these challenges. In this framework, a policy subspace represented by a set of parameters is learned. Policies for all the single tasks lie in this subspace and can be composed by interpolating with the learned set. It allows not only flexible parameter sharing, but also a natural way to improve training.We demonstrate the state-of-the-art performance on Meta-World benchmarks, verifying the effectiveness of the proposed approach.

        ----

        ## [1562] Entropy-Driven Mixed-Precision Quantization for Deep Network Design

        **Authors**: *Zhenhong Sun, Ce Ge, Junyan Wang, Ming C. Lin, Hesen Chen, Hao Li, Xiuyu Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/86e7ebb16d33d59e62d1b0a079ea058d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/86e7ebb16d33d59e62d1b0a079ea058d-Abstract-Conference.html)

        **Abstract**:

        Deploying deep convolutional neural networks on Internet-of-Things (IoT) devices is challenging due to the limited computational resources, such as limited SRAM memory and Flash storage. Previous works re-design a small network for IoT devices, and then compress the network size by mixed-precision quantization. This two-stage procedure cannot optimize the architecture and the corresponding quantization jointly, leading to sub-optimal tiny deep models. In this work, we propose a one-stage solution that optimizes both jointly and automatically. The key idea of our approach is to cast the joint architecture design and quantization as an Entropy Maximization process. Particularly, our algorithm automatically designs a tiny deep model such that: 1) Its representation capacity measured by entropy is maximized under the given computational budget; 2) Each layer is assigned with a proper quantization precision; 3) The overall design loop can be done on CPU, and no GPU is required. More impressively, our method can directly search high-expressiveness architecture for IoT devices within less than half a CPU hour. Extensive experiments on three widely adopted benchmarks, ImageNet, VWW and WIDER FACE, demonstrate that our method can achieve the state-of-the-art performance in the tiny deep model regime. Code and pre-trained models are available at https://github.com/alibaba/lightweight-neural-architecture-search.

        ----

        ## [1563] Non-Gaussian Tensor Programs

        **Authors**: *Eugene Golikov, Greg Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html)

        **Abstract**:

        Does it matter whether one randomly initializes a neural network (NN) from Gaussian, uniform, or other distributions? We show the answer is ”yes” in some parameter tensors (the so-called matrix-like parameters) but ”no” in others when the NN is wide. This is a specific instance of a more general universality principle for Tensor Programs (TP) that informs precisely when the limit of a program depends on the distribution of its initial matrices and vectors. To obtain this principle, we develop the theory of non-Gaussian Tensor Programs. As corollaries, we obtain all previous consequences of the TP framework (such as NNGP/NTK correspondence, Free Independence Principle, Dynamical Dichotomy Theorem, and μ-parametrization) for NNs with non-Gaussian weights.

        ----

        ## [1564] Adaptation Accelerating Sampling-based Bayesian Inference in Attractor Neural Networks

        **Authors**: *Xingsi Dong, Zilong Ji, Tianhao Chu, Tiejun Huang, Wenhao Zhang, Si Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html)

        **Abstract**:

        The brain performs probabilistic Bayesian inference to interpret the external world. The sampling-based view assumes that the brain represents the stimulus posterior distribution via samples of stochastic neuronal responses. Although the idea of sampling-based inference is appealing, it faces a critical challenge of whether stochastic sampling is fast enough to match the rapid computation of the brain. In this study, we explore how latent stimulus sampling can be accelerated in neural circuits. Specifically, we consider a canonical neural circuit model called continuous attractor neural networks (CANNs) and investigate how sampling-based inference of latent continuous variables is accelerated in CANNs. Intriguingly, we find that by including noisy adaptation in the neuronal dynamics, the CANN is able to speed up the sampling process significantly. We theoretically derive that the CANN with noisy adaptation implements the efficient sampling method called Hamiltonian dynamics with friction, where noisy adaption effectively plays the role of momentum. We theoretically analyze the sampling performances of the network and derive the condition when the acceleration has the maximum effect. Simulation results confirm our theoretical analyses. We further extend the model to coupled CANNs and demonstrate that noisy adaptation accelerates the sampling of the posterior distribution of multivariate stimuli. We hope that this study enhances our understanding of how Bayesian inference is realized in the brain.

        ----

        ## [1565] A Contrastive Framework for Neural Text Generation

        **Authors**: *Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/871cae8f599cb8bbfcb0f58fe1af95ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/871cae8f599cb8bbfcb0f58fe1af95ad-Abstract-Conference.html)

        **Abstract**:

        Text generation is of great importance to many natural language processing applications. However, maximization-based decoding methods (e.g., beam search) of neural language models often lead to degenerate solutions---the generated text is unnatural and contains undesirable repetitions. Existing approaches introduce stochasticity via sampling or modify training objectives to decrease the probabilities of certain tokens (e.g., unlikelihood training). However, they often lead to solutions that lack coherence. In this work, we show that an underlying reason for model degeneration is the anisotropic distribution of token representations. We present a contrastive solution: (i) SimCTG, a contrastive training objective to calibrate the model's representation space, and (ii) a decoding method---contrastive search---to encourage diversity while maintaining coherence in the generated text. Extensive experiments and analyses on three benchmarks from two languages demonstrate that our proposed approach outperforms state-of-the-art text generation methods as evaluated by both human and automatic metrics.

        ----

        ## [1566] Exploring the Latent Space of Autoencoders with Interventional Assays

        **Authors**: *Felix Leeb, Stefan Bauer, Michel Besserve, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/87213955efbe48b46586e37bf2f1fe5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/87213955efbe48b46586e37bf2f1fe5b-Abstract-Conference.html)

        **Abstract**:

        Autoencoders exhibit impressive abilities to embed the data manifold into a low-dimensional latent space, making them a staple of representation learning methods. However, without explicit supervision, which is often unavailable, the representation is usually uninterpretable, making analysis and principled progress challenging. We propose a framework, called latent responses, which exploits the locally contractive behavior exhibited by variational autoencoders to explore the learned manifold. More specifically, we develop tools to probe the representation using interventions in the latent space to quantify the relationships between latent variables. We extend the notion of disentanglement to take the learned generative process into account and consequently avoid the limitations of existing metrics that may rely on spurious correlations. Our analyses underscore the importance of studying the causal structure of the representation to improve performance on downstream tasks such as generation, interpolation, and inference of the factors of variation.

        ----

        ## [1567] Pythae: Unifying Generative Autoencoders in Python - A Benchmarking Use Case

        **Authors**: *Clément Chadebec, Louis J. Vincent, Stéphanie Allassonnière*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/872f0e04ef95be7970d9a9d74b198fdf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/872f0e04ef95be7970d9a9d74b198fdf-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In recent years, deep generative models have attracted increasing interest due to their capacity to model complex distributions. Among those models, variational autoencoders have gained popularity as they have proven both to be computationally efficient and yield impressive results in multiple fields. Following this breakthrough, extensive research has been done in order to improve the original publication, resulting in a variety of different VAE models in response to different tasks. In this paper we present \textbf{Pythae}, a versatile \textit{open-source} Python library providing both a \textit{unified implementation} and a dedicated framework allowing \textit{straightforward}, \emph{reproducible} and \textit{reliable} use of generative autoencoder models. We then propose to use this library to perform a case study benchmark where we present and compare 19 generative autoencoder models representative of some of the main improvements on downstream tasks such as image reconstruction, generation, classification, clustering and interpolation. The open-source library can be found at \url{https://github.com/clementchadebec/benchmark_VAE}.

        ----

        ## [1568] Adaptive Multi-stage Density Ratio Estimation for Learning Latent Space Energy-based Model

        **Authors**: *Zhisheng Xiao, Tian Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html)

        **Abstract**:

        This paper studies the fundamental problem of learning energy-based model (EBM) in the latent space of the generator model. Learning such prior model typically requires running costly Markov Chain Monte Carlo (MCMC). Instead, we propose to use noise contrastive estimation (NCE) to discriminatively learn the EBM through density ratio estimation between the latent prior density and latent posterior density. However, the NCE typically fails to accurately estimate such density ratio given large gap between two densities. To effectively tackle this issue and further learn more expressive prior model, we develop the adaptive multi-stage density ratio estimation which breaks the estimation into multiple stages and learn different stages of density ratio sequentially and adaptively. The latent prior model can be gradually learned using ratio estimated in previous stage so that the final latent space EBM prior can be naturally formed by product of ratios in different stages. The proposed method enables informative and much sharper prior than existing baselines, and can be trained efficiently. Our experiments demonstrate strong performances in terms of image generation and reconstruction as well as anomaly detection.

        ----

        ## [1569] An Analysis of Ensemble Sampling

        **Authors**: *Chao Qin, Zheng Wen, Xiuyuan Lu, Benjamin Van Roy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/874f5e53d7ce44f65fbf27a7b9406983-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/874f5e53d7ce44f65fbf27a7b9406983-Abstract-Conference.html)

        **Abstract**:

        Ensemble sampling serves as a practical approximation to Thompson sampling when maintaining an exact posterior distribution over model parameters is computationally intractable. In this paper, we establish a regret bound that ensures desirable behavior when ensemble sampling is applied to the linear bandit problem. This represents the first rigorous regret analysis of ensemble sampling and is made possible by leveraging information-theoretic concepts and novel analytic techniques that may prove useful beyond the scope of this paper.

        ----

        ## [1570] Fair Wrapping for Black-box Predictions

        **Authors**: *Alexander Soen, Ibrahim M. Alabdulmohsin, Sanmi Koyejo, Yishay Mansour, Nyalleng Moorosi, Richard Nock, Ke Sun, Lexing Xie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/876b45367d9069f0e91e359c57155ab1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/876b45367d9069f0e91e359c57155ab1-Abstract-Conference.html)

        **Abstract**:

        We introduce a new family of techniques to post-process (``wrap") a black-box classifier in order to reduce its bias. Our technique builds on the recent analysis of improper loss functions whose optimization can correct any twist in prediction, unfairness being treated as a twist. In the post-processing, we learn a wrapper function which we define as an $\alpha$-tree, which modifies the prediction. We provide two generic boosting algorithms to learn $\alpha$-trees. We show that our modification has appealing properties in terms of composition of $\alpha$-trees, generalization, interpretability, and KL divergence between modified and original predictions. We exemplify the use of our technique in three fairness notions: conditional value-at-risk, equality of opportunity, and statistical parity; and provide experiments on several readily available datasets.

        ----

        ## [1571] Bridging Central and Local Differential Privacy in Data Acquisition Mechanisms

        **Authors**: *Alireza Fallah, Ali Makhdoumi, Azarakhsh Malekian, Asuman E. Ozdaglar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/878bbdf6227315995c207561211ddb53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/878bbdf6227315995c207561211ddb53-Abstract-Conference.html)

        **Abstract**:

        We study the design of optimal Bayesian data acquisition mechanisms for a platform interested in estimating the mean of a distribution by collecting data from privacy-conscious users. In our setting, users have heterogeneous sensitivities for two types of privacy losses corresponding to local and central differential privacy measures. The local privacy loss is due to the leakage of a user's information when she shares her data with the platform, and the central privacy loss is due to the released estimate by the platform to the public. The users share their data in exchange for a payment (e.g., through monetary transfers or services) that compensates for their privacy losses. The platform does not know the privacy sensitivity of users and must design a mechanism to solicit their preferences and then deliver both local and central privacy guarantees while minimizing the estimation error plus the expected payment to users. We first establish minimax lower bounds for the estimation error, given a vector of privacy guarantees, and show that a linear estimator is (near) optimal. We then turn to our main goal: designing an optimal data acquisition mechanism. We establish that the design of such mechanisms in a Bayesian setting (where the platform knows the distribution of users' sensitivities and not their realizations) can be cast as a nonconvex optimization problem. Additionally, for the class of linear estimators, we prove that finding the optimal mechanism admits a Polynomial Time Approximation Scheme.

        ----

        ## [1572] Meta-Learning Dynamics Forecasting Using Task Inference

        **Authors**: *Rui Wang, Robin Walters, Rose Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/87f476af4053961667c2c08e9f4b850e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/87f476af4053961667c2c08e9f4b850e-Abstract-Conference.html)

        **Abstract**:

        Current deep learning models for dynamics forecasting struggle with generalization. They can only forecast in a specific domain and fail when applied to systems with different parameters, external forces, or boundary conditions.  We propose a model-based meta-learning method called DyAd which can generalize across heterogeneous domains by partitioning them into different tasks.  DyAd has two parts: an encoder that infers the time-invariant hidden features of the task with weak supervision, and a forecaster which learns the shared dynamics of the entire domain. The encoder adapts and controls the forecaster during inference using adaptive instance normalization and adaptive padding.  Theoretically, we prove that the generalization error of such a procedure is related to the task relatedness in the source domain, as well as the domain differences between source and target. Experimentally, we demonstrate that our model outperforms state-of-the-art approaches on forecasting complex physical dynamics including turbulent flow, real-world sea surface temperature, and ocean currents.

        ----

        ## [1573] Multi-fidelity Monte Carlo: a pseudo-marginal approach

        **Authors**: *Diana Cai, Ryan P. Adams*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8803b9ae0b13011f28e6dd57da2ebbd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8803b9ae0b13011f28e6dd57da2ebbd8-Abstract-Conference.html)

        **Abstract**:

        Markov chain Monte Carlo (MCMC) is an established approach for uncertainty quantification and propagation in scientific applications.  A key challenge in applying MCMC to scientific domains is computation: the target density of interest is often a function of expensive computations, such as a high-fidelity physical simulation, an intractable integral, or a slowly-converging iterative algorithm.  Thus, using an MCMC algorithms with an expensive target density becomes impractical, as these expensive computations need  to be evaluated at each iteration of the algorithm.  In practice, these computations often approximated via a cheaper, low-fidelity computation, leading to bias in the resulting target density.  Multi-fidelity MCMC algorithms combine models of varying fidelities in order to obtain an approximate target density with lower computational cost.  In this paper, we describe a class of asymptotically exact multi-fidelity MCMC algorithms for the setting where a sequence of models of increasing fidelity can be computed that approximates the expensive target density of interest.  We take a pseudo-marginal MCMC approach for multi-fidelity inference that utilizes a cheaper, randomized-fidelity unbiased estimator of the target fidelity constructed via  random truncation of a telescoping series of the low-fidelity sequence of models.  Finally, we discuss and evaluate the proposed multi-fidelity MCMC approach on several applications, including log-Gaussian Cox process modeling, Bayesian ODE system identification, PDE-constrained optimization, and Gaussian process parameter inference.

        ----

        ## [1574] SAPD+: An Accelerated Stochastic Method for Nonconvex-Concave Minimax Problems

        **Authors**: *Xuan Zhang, Necdet Serhat Aybat, Mert Gürbüzbalaban*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/880d8999c07a8efc9bbbeb0c38f50765-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/880d8999c07a8efc9bbbeb0c38f50765-Abstract-Conference.html)

        **Abstract**:

        We propose a new stochastic method SAPD+ for solving nonconvex-concave minimax problems of the form $\min\max\mathcal{L}(x,y)=f(x)+\Phi(x,y)-g(y)$, where $f,g$ are closed convex and $\Phi(x,y)$ is a smooth function that is weakly convex in $x$, (strongly) concave in $y$. For both strongly concave and merely concave settings, SAPD+ achieves the best known oracle complexities of $\mathcal{O}(L\kappa_y\epsilon^{-4})$ and $\mathcal{O}(L^3\epsilon^{-6})$, respectively, without assuming compactness of the problem domain, where $\kappa_y$ is the condition number, and $L$ is the Lipschitz constant.  We also propose SAPD+ with variance reduction, which enjoys the best known oracle complexity of $\mathcal{O}(L\kappa_y^2\epsilon^{-3})$ for weakly convex-strongly concave setting. We demonstrate the efficiency of SAPD+ on a distributionally robust learning problem with a nonconvex regularizer and also on a multi-class classification problem in deep learning.

        ----

        ## [1575] Contrastive Adapters for Foundation Model Group Robustness

        **Authors**: *Michael Zhang, Christopher Ré*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8829f586a1ac0e6c41143f5d57b63c4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8829f586a1ac0e6c41143f5d57b63c4b-Abstract-Conference.html)

        **Abstract**:

        While large pretrained foundation models (FMs) have shown remarkable zero-shot classification robustness to dataset-level distribution shifts, their robustness to subpopulation or group shifts is relatively underexplored. We study this problem, and find that foundation models such as CLIP may not be robust to various group shifts. Across 9 robustness benchmarks, zero-shot classification with their embeddings results in gaps of up to 80.7 percentage points (pp) between average and worst-group accuracy. Unfortunately, existing methods to improve robustness require retraining, which can be prohibitively expensive on large foundation models. We also find that efficient ways to improve model inference (e.g. via adapters, lightweight networks that transform FM embeddings) do not consistently improve and can sometimes hurt group robustness compared to zero-shot. We therefore develop an adapter training strategy to effectively and efficiently improve FM group robustness. Our motivating observation is that while poor robustness results from groups in the same class being embedded far apart in the foundation model "embedding space," standard adapter training may not actually bring these points closer together. We thus propose contrastive adapting, which contrastively trains adapters to bring sample embeddings close to both their ground-truth class embeddings and same-class sample embeddings. Across the 9 robustness benchmarks, contrastive adapting consistently improves group robustness, raising worst-group accuracy by 8.5 to 56.0 pp over zero-shot. Our approach is also efficient, doing so without any FM finetuning and only a fixed set of FM embeddings. On popular benchmarks such as Waterbirds and CelebA, this leads to worst-group accuracy comparable to state-of-the-art methods, while only training <1% of the model parameters.

        ----

        ## [1576] Decoupled Context Processing for Context Augmented Language Modeling

        **Authors**: *Zonglin Li, Ruiqi Guo, Sanjiv Kumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/882d801fb1017f955547d5a816ade0fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/882d801fb1017f955547d5a816ade0fc-Abstract-Conference.html)

        **Abstract**:

        Language models can be augmented with context retriever to incorporate knowledge from large external databases. By leveraging retrieved context, the neural network does not have to memorize the massive amount of world knowledge within its internal parameters, leading to better parameter efficiency, interpretability and modularity. In this paper we examined a simple yet effective architecture for incorporating external context into language models based on decoupled $\texttt{Encoder-Decoder}$ architecture. We showed that such a simple architecture achieves competitive results on auto-regressive language modeling and open domain question answering tasks. We also analyzed the behavior of the proposed model which performs grounded context transfer. Finally we discussed the computational implications of such retrieval augmented models.

        ----

        ## [1577] LISA: Learning Interpretable Skill Abstractions from Language

        **Authors**: *Divyansh Garg, Skanda Vaidyanath, Kuno Kim, Jiaming Song, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)

        **Abstract**:

        Learning policies that effectively utilize language instructions in complex, multi-task environments is an important problem in imitation learning. While it is possible to condition on the entire language instruction directly, such an approach could suffer from generalization issues. To encode complex instructions into skills that can generalize to unseen instructions, we propose Learning Interpretable Skill Abstractions (LISA), a hierarchical imitation learning framework that can learn diverse, interpretable skills from language-conditioned demonstrations. LISA uses vector quantization to learn discrete skill codes that are highly correlated with language instructions and the behavior of the learned policy. In navigation and robotic manipulation environments, LISA is able to outperform a strong non-hierarchical baseline in the low data regime and compose learned skills to solve tasks containing unseen long-range instructions. Our method demonstrates a more natural way to condition on language in sequential decision-making problems and achieve interpretable and controllable behavior with the learned skills.

        ----

        ## [1578] Accelerated Primal-Dual Gradient Method for Smooth and Convex-Concave Saddle-Point Problems with Bilinear Coupling

        **Authors**: *Dmitry Kovalev, Alexander V. Gasnikov, Peter Richtárik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/883f66687a521536c505f9b2fbdcbf1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/883f66687a521536c505f9b2fbdcbf1e-Abstract-Conference.html)

        **Abstract**:

        In this paper we study the convex-concave saddle-point problem $\min_x \max_y f(x) + y^\top\mathbf{A}x - g(y)$, where $f(x)$ and $g(y)$ are smooth and convex functions. We propose an Accelerated Primal-Dual Gradient Method (APDG) for solving this problem, achieving (i) an optimal linear convergence rate in the strongly-convex-strongly-concave regime, matching the lower complexity bound (Zhang et al., 2021), and (ii) an accelerated linear convergence rate in the case when only one of the functions $f(x)$ and $g(y)$ is strongly convex or even none of them are. Finally, we obtain a linearly convergent algorithm for the general smooth and convex-concave saddle point problem $\min_x \max_y F(x,y)$ without the requirement of strong convexity or strong concavity.

        ----

        ## [1579] Diffusion Curvature for Estimating Local Curvature in High Dimensional Data

        **Authors**: *Dhananjay Bhaskar, Kincaid MacDonald, Oluwadamilola Fasina, Dawson Thomas, Bastian Rieck, Ian Adelstein, Smita Krishnaswamy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/88438dc62fc5c8777e2b5f1b4f6d37a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/88438dc62fc5c8777e2b5f1b4f6d37a2-Abstract-Conference.html)

        **Abstract**:

        We introduce a new intrinsic measure of local curvature on point-cloud data called diffusion curvature. Our measure uses the framework of diffusion maps, including the data diffusion operator, to structure point cloud data and define local curvature based on the laziness of a random walk starting at a point or region of the data. We show that this laziness directly relates to volume comparison results from Riemannian geometry. We then extend this scalar curvature notion to an entire quadratic form using neural network estimations based on the diffusion map of point-cloud data. We show applications of both estimations on toy data, single-cell data, and on estimating local Hessian matrices of neural network loss landscapes.

        ----

        ## [1580] Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit

        **Authors**: *Boaz Barak, Benjamin L. Edelman, Surbhi Goel, Sham M. Kakade, Eran Malach, Cyril Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html)

        **Abstract**:

        There is mounting evidence of emergent phenomena in the capabilities of deep learning methods as we scale up datasets, model sizes, and training times. While there are some accounts of how these resources modulate statistical capacity, far less is known about their effect on the computational problem of model training. This work conducts such an exploration through the lens of learning a $k$-sparse parity of $n$ bits, a canonical discrete search problem which is statistically easy but computationally hard. Empirically, we find that a variety of neural networks successfully learn sparse parities, with discontinuous phase transitions in the training curves. On small instances, learning abruptly occurs at approximately $n^{O(k)}$ iterations; this nearly matches SQ lower bounds, despite the apparent lack of a sparse prior. Our theoretical analysis shows that these observations are not explained by a Langevin-like mechanism, whereby SGD "stumbles in the dark" until it finds the hidden set of features (a natural algorithm which also runs in $n^{O(k)}$ time). Instead, we show that SGD gradually amplifies the sparse solution via a Fourier gap in the population gradient, making continual progress that is invisible to loss and error metrics.

        ----

        ## [1581] One Positive Label is Sufficient: Single-Positive Multi-Label Learning with Label Enhancement

        **Authors**: *Ning Xu, Congyu Qiao, Jiaqi Lv, Xin Geng, Min-Ling Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/888a66ed219c281f448babae80f3b8e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/888a66ed219c281f448babae80f3b8e8-Abstract-Conference.html)

        **Abstract**:

        Multi-label learning (MLL) learns from the examples each associated with multiple labels simultaneously, where the high cost of annotating all relevant labels for each training example is challenging for real-world applications. To cope with the challenge, we investigate single-positive multi-label learning (SPMLL) where each example is annotated with only one relevant label and show that one can successfully learn a theoretically grounded multi-label classifier for the problem. In this paper,  a novel  SPMLL method named SMILE, i.e., Single-positive MultI-label learning with Label Enhancement, is proposed. Specifically, an unbiased risk estimator is derived, which could be guaranteed to approximately converge to the optimal risk minimizer of fully supervised learning and shows that one positive label of each instance is sufficient to train the predictive model. Then, the corresponding empirical risk estimator is established via recovering the latent soft label as a label enhancement process, where the posterior density of the latent soft labels is approximate to the variational Beta density parameterized by an inference model. Experiments on benchmark datasets validate the effectiveness of the proposed method.

        ----

        ## [1582] Communication Acceleration of Local Gradient Methods via an Accelerated Primal-Dual Algorithm with an Inexact Prox

        **Authors**: *Abdurakhmon Sadiev, Dmitry Kovalev, Peter Richtárik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html)

        **Abstract**:

        Inspired by a recent breakthrough of Mishchenko et al. [2022], who for the first time showed that local gradient steps can lead to provable communication acceleration, we propose an alternative algorithm which obtains the same communication acceleration as their method (ProxSkip). Our approach is very different, however: it is based on the celebrated  method of Chambolle and Pock [2011], with several nontrivial modifications: i) we allow for an inexact computation of the prox operator of a certain smooth strongly convex function via a suitable gradient-based method (e.g., GD or Fast GD), ii) we perform a careful modification of the dual update step in order to retain linear convergence. Our general results offer the new state-of-the-art rates for the class of strongly convex-concave saddle-point problems with bilinear coupling characterized by the absence of smoothness in the dual function. When applied to federated learning, we obtain a theoretically better alternative to ProxSkip: our method requires fewer local steps ($\mathcal{O}(\kappa^{1/3})$ or $\mathcal{O}(\kappa^{1/4})$, compared to $\mathcal{O}(\kappa^{1/2})$ of ProxSkip), and performs a deterministic number of local steps instead. Like ProxSkip, our method can be applied to optimization over a connected network, and we obtain theoretical improvements here as well.

        ----

        ## [1583] Aligning individual brains with fused unbalanced Gromov Wasserstein

        **Authors**: *Alexis Thual, Quang Huy Tran, Tatiana Zemskova, Nicolas Courty, Rémi Flamary, Stanislas Dehaene, Bertrand Thirion*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8906cac4ca58dcaf17e97a0486ad57ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8906cac4ca58dcaf17e97a0486ad57ca-Abstract-Conference.html)

        **Abstract**:

        Individual brains vary in both anatomy and functional organization, even within a given species. Inter-individual variability is a major impediment when trying to draw generalizable conclusions from neuroimaging data collected on groups of subjects. Current co-registration procedures rely on limited data, and thus lead to very coarse inter-subject alignments. In this work, we present a novel method for inter-subject alignment based on Optimal Transport, denoted as Fused Unbalanced Gromov Wasserstein (FUGW). The method aligns two cortical surfaces based on the similarity of their functional signatures in response to a variety of stimuli, while penalizing large deformations of individual topographic organization.We demonstrate that FUGW is suited for whole-brain landmark-free alignment. The unbalanced feature allows to deal with the fact that functional areas vary in size across subjects. Results show that FUGW alignment significantly increases between-subject correlation of activity during new independent fMRI tasks and runs, and leads to more precise maps of fMRI results at the group level.

        ----

        ## [1584] This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish

        **Authors**: *Lukasz Augustyniak, Kamil Tagowski, Albert Sawczyn, Denis Janiak, Roman Bartusiak, Adrian Szymczak, Arkadiusz Janz, Piotr Szymanski, Marcin Watroba, Mikolaj Morzy, Tomasz Kajdanowicz, Maciej Piasecki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/890b206ebb79e550f3988cb8db936f42-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/890b206ebb79e550f3988cb8db936f42-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The availability of compute and data to train larger and larger language models increases the demand for robust methods of benchmarking the true progress of LM training. Recent years witnessed significant progress in standardized benchmarking for English. Benchmarks such as GLUE, SuperGLUE, or KILT have become a de facto standard tools to compare large language models. Following the trend to replicate GLUE for other languages, the KLEJ benchmark\ (klej is the word for glue in Polish) has been released for Polish. In this paper, we evaluate the progress in benchmarking for low-resourced languages. We note that only a handful of languages have such comprehensive benchmarks. We also note the gap in the number of tasks being evaluated by benchmarks for resource-rich English/Chinese and the rest of the world.In this paper, we introduce LEPISZCZE (lepiszcze is the Polish word for glew, the Middle English predecessor of glue), a new, comprehensive benchmark for Polish NLP with a large variety of tasks and high-quality operationalization of the benchmark.We design LEPISZCZE with flexibility in mind. Including new models, datasets, and tasks is as simple as possible while still offering data versioning and model tracking. In the first run of the benchmark, we test 13 experiments (task and dataset pairs) based on the five most recent LMs for Polish. We use five datasets from the Polish benchmark and add eight novel datasets. As the paper's main contribution, apart from LEPISZCZE, we provide insights and experiences learned while creating the benchmark for Polish as the blueprint to design similar benchmarks for other low-resourced languages.

        ----

        ## [1585] An In-depth Study of Stochastic Backpropagation

        **Authors**: *Jun Fang, Mingze Xu, Hao Chen, Bing Shuai, Zhuowen Tu, Joseph Tighe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/890e018ca9c879c5ac01757239538f7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/890e018ca9c879c5ac01757239538f7c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we provide an in-depth study of Stochastic Backpropagation (SBP) when training deep neural networks for standard image classification and object detection tasks. During backward propagation, SBP calculates gradients by using only a subset of feature maps to save GPU memory and computational cost. We interpret SBP as an efficient way to implement stochastic gradient decent by performing backpropagation dropout, which leads to significant memory saving and training run-time reduction, with a minimal impact on the overall model accuracy. We offer best practices to apply SBP for training image recognition models, which can be adopted in learning a wide range of deep neural networks. Experiments on image classification and object detection show that SBP can save up to 40% of GPU memory with less than 1% accuracy degradation. Code is available at: https://github.com/amazon-research/stochastic-backpropagation

        ----

        ## [1586] Transformer Memory as a Differentiable Search Index

        **Authors**: *Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W. Cohen, Donald Metzler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html)

        **Abstract**:

        In this paper, we demonstrate that information retrieval can be accomplished with a single Transformer, in which all information about the corpus is encoded in the parameters of the model. To this end, we introduce the Differentiable Search Index (DSI), a new paradigm that learns a text-to-text model that maps string queries directly to relevant docids; in other words, a DSI model answers queries directly using only its parameters, dramatically simplifying the whole retrieval process. We study variations in how documents and their identifiers are represented, variations in training procedures, and the interplay between models and corpus sizes. Experiments demonstrate that given appropriate design choices, DSI significantly outperforms strong baselines such as dual encoder models. Moreover, DSI demonstrates strong generalization capabilities, outperforming a BM25 baseline in a zero-shot setup.

        ----

        ## [1587] Insights into Pre-training via Simpler Synthetic Tasks

        **Authors**: *Yuhuai Wu, Felix Li, Percy Liang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html)

        **Abstract**:

        Pre-training produces representations that are effective for a wide range of downstream tasks, but it is still unclear what properties of pre-training are necessary for effective gains. Notably, recent work shows that even pre-training on synthetic tasks can achieve significant gains in downstream tasks. In this work, we perform three experiments that iteratively simplify pre-training and show that the simplifications still retain much of its gains. First, building on prior work, we perform a systematic evaluation of three existing synthetic pre-training methods on six downstream tasks. We find the best synthetic pre-training method, LIME, attains an average of $67\%$ of the benefits of natural pre-training. Second, to our surprise, we find that pre-training on a simple and generic synthetic task defined by the set function achieves $65\%$ of the benefits, almost matching LIME. Third, we find that $39\%$ of the benefits can be attained by using merely the parameter statistics of synthetic pre-training. We release the source code at \url{https://github.com/felixzli/synthetic_pretraining}.

        ----

        ## [1588] Last-Iterate Convergence of Optimistic Gradient Method for Monotone Variational Inequalities

        **Authors**: *Eduard Gorbunov, Adrien B. Taylor, Gauthier Gidel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/893cd874ba98afa54ae9e385a24a83ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/893cd874ba98afa54ae9e385a24a83ac-Abstract-Conference.html)

        **Abstract**:

        The Past Extragradient (PEG) [Popov, 1980] method, also known as the Optimistic Gradient method, has known a recent gain in interest in the optimization community with the emergence of variational inequality formulations for machine learning. Recently, in the unconstrained case, Golowich et al. [2020] proved that a $O(1/N)$ last-iterate convergence rate in terms of the squared norm of the operator can be achieved for Lipschitz and monotone operators with a Lipchitz Jacobian. In this work, by introducing a novel analysis through potential functions, we show that (i) this $O(1/N)$ last-iterate convergence can be achieved without any assumption on the Jacobian of the operator, and (ii) it can be extended to the constrained case, which was not derived before even under Lipschitzness of the Jacobian. The proof is significantly different from the one known from Golowich et al. [2020], and its discovery was computer-aided. Those results close the open question of the last iterate convergence of PEG for monotone variational inequalities.

        ----

        ## [1589] 3DILG: Irregular Latent Grids for 3D Generative Modeling

        **Authors**: *Biao Zhang, Matthias Nießner, Peter Wonka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/894ca1c4bc1c6abc4d4998ab94635fdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/894ca1c4bc1c6abc4d4998ab94635fdf-Abstract-Conference.html)

        **Abstract**:

        We propose a new representation for encoding 3D shapes as neural fields. The representation is designed to be compatible with the transformer architecture and to benefit both shape reconstruction and shape generation. Existing works on neural fields are grid-based representations with latents being defined on a regular grid. In contrast, we define latents on irregular grids which facilitates our representation to be sparse and adaptive. In the context of shape reconstruction from point clouds, our shape representation built on irregular grids improves upon grid-based methods in terms of reconstruction accuracy. For shape generation, our representation promotes high-quality shape generation using auto-regressive probabilistic models. We show different applications that improve over the current state of the art. First, we show results of probabilistic shape reconstruction from a single higher resolution image. Second, we train a probabilistic model conditioned on very low resolution images. Third, we apply our model to category-conditioned generation. All probabilistic experiments confirm that we are able to generate detailed and high quality shapes to yield the new state of the art in generative 3D shape modeling.

        ----

        ## [1590] Policy Optimization for Markov Games: Unified Framework and Faster Convergence

        **Authors**: *Runyu Zhang, Qinghua Liu, Huan Wang, Caiming Xiong, Na Li, Yu Bai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8951f484e8242b7f74817fdc390dd954-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8951f484e8242b7f74817fdc390dd954-Abstract-Conference.html)

        **Abstract**:

        This paper studies policy optimization algorithms for multi-agent reinforcement learning. We begin by proposing an algorithm framework for two-player zero-sum Markov Games in the full-information setting, where each iteration consists of a policy update step at each state using a certain matrix game algorithm, and a value update step with a certain learning rate. This framework unifies many existing and new policy optimization algorithms. We show that the \emph{state-wise average policy} of this algorithm converges to an approximate Nash equilibrium (NE) of the game, as long as the matrix game algorithms achieve low weighted regret at each state, with respect to weights determined by the speed of the value updates. Next, we show that this framework instantiated with the Optimistic Follow-The-Regularized-Leader (OFTRL) algorithm at each state (and smooth value updates) can find an $\mathcal{\widetilde{O}}(T^{-5/6})$ approximate NE in $T$ iterations, and a similar algorithm with slightly modified value update rule achieves a faster $\mathcal{\widetilde{O}}(T^{-1})$ convergence rate. These improve over the current best $\mathcal{\widetilde{O}}(T^{-1/2})$ rate of symmetric policy optimization type algorithms. We also extend this algorithm to multi-player general-sum Markov Games and show an $\mathcal{\widetilde{O}}(T^{-3/4})$ convergence rate to Coarse Correlated Equilibria (CCE). Finally, we provide a numerical example to verify our theory and investigate the importance of smooth value updates, and find that using ''eager'' value updates instead (equivalent to the independent natural policy gradient algorithm) may significantly slow down the convergence, even on a simple game with $H=2$ layers.

        ----

        ## [1591] C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting

        **Authors**: *Shane Bergsma, Timothy Zeyl, Javad Rahimipour Anaraki, Lei Guo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html)

        **Abstract**:

        We present coarse-to-fine autoregressive networks (C2FAR), a method for modeling the probability distribution of univariate, numeric random variables.  C2FAR generates a hierarchical, coarse-to-fine discretization of a variable autoregressively; progressively finer intervals of support are generated from a sequence of binned distributions, where each distribution is conditioned on previously-generated coarser intervals.  Unlike prior (flat) binned distributions, C2FAR can represent values with exponentially higher precision, for only a linear increase in complexity.  We use C2FAR for probabilistic forecasting via a recurrent neural network, thus modeling time series autoregressively in both space and time.  C2FAR is the first method to simultaneously handle discrete and continuous series of arbitrary scale and distribution shape.  This flexibility enables a variety of time series use cases, including anomaly detection, interpolation, and compression.  C2FAR achieves improvements over the state-of-the-art on several benchmark forecasting datasets.

        ----

        ## [1592] METS-CoV: A Dataset of Medical Entity and Targeted Sentiment on COVID-19 Related Tweets

        **Authors**: *Peilin Zhou, Zeqiang Wang, Dading Chong, Zhijiang Guo, Yining Hua, Zichang Su, Zhiyang Teng, Jiageng Wu, Jie Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89a7ddfbc08b25ef8ff9029d7dd9e3d3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/89a7ddfbc08b25ef8ff9029d7dd9e3d3-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The COVID-19 pandemic continues to bring up various topics discussed or debated on social media. In order to explore the impact of pandemics on people's lives, it is crucial to understand the public's concerns and attitudes towards pandemic-related entities (e.g., drugs, vaccines) on social media. However, models trained on existing named entity recognition (NER) or targeted sentiment analysis (TSA) datasets have limited ability to understand COVID-19-related social media texts because these datasets are not designed or annotated from a medical perspective. In this paper, we release METS-CoV, a dataset containing medical entities and targeted sentiments from COVID-19 related tweets. METS-CoV contains 10,000 tweets with 7 types of entities, including 4 medical entity types (Disease, Drug, Symptom, and Vaccine) and 3 general entity types (Person, Location, and Organization). To further investigate tweet users' attitudes toward specific entities, 4 types of entities (Person, Organization, Drug, and Vaccine) are selected and annotated with user sentiments, resulting in a targeted sentiment dataset with 9,101 entities (in 5,278 tweets). To the best of our knowledge, METS-CoV is the first dataset to collect medical entities and corresponding sentiments of COVID-19 related tweets. We benchmark the performance of classical machine learning models and state-of-the-art deep learning models on NER and TSA tasks with extensive experiments. Results show that this dataset has vast room for improvement for both NER and TSA tasks. With rich annotations and comprehensive benchmark results, we believe METS-CoV is a fundamental resource for building better medical social media understanding tools and facilitating computational social science research, especially on epidemiological topics. Our data, annotation guidelines, benchmark models, and source code are publicly available (\url{https://github.com/YLab-Open/METS-CoV}) to ensure reproducibility.

        ----

        ## [1593] Respecting Transfer Gap in Knowledge Distillation

        **Authors**: *Yulei Niu, Long Chen, Chang Zhou, Hanwang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89b0e466b46292ce0bfe53618aadd3de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/89b0e466b46292ce0bfe53618aadd3de-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation (KD) is essentially a process of transferring a teacher model's behavior, e.g., network response, to a student model. The network response serves as additional supervision to formulate the machine domain, which uses the data collected from the human domain as a transfer set. Traditional KD methods hold an underlying assumption that the data collected in both human domain and machine domain are both independent and identically distributed (IID). We point out that this naive assumption is unrealistic and there is indeed a transfer gap between the two domains. Although the gap offers the student model external knowledge from the machine domain, the imbalanced teacher knowledge would make us incorrectly estimate how much to transfer from teacher to student per sample on the non-IID transfer set. To tackle this challenge, we propose Inverse Probability Weighting Distillation (IPWD) that estimates the propensity of a training sample belonging to the machine domain, and assigns its inverse amount to compensate for under-represented samples. Experiments on CIFAR-100 and ImageNet demonstrate the effectiveness of \ours~for both two-stage distillation and one-stage self-distillation.

        ----

        ## [1594] Target alignment in truncated kernel ridge regression

        **Authors**: *Arash A. Amini, Richard Baumgartner, Dai Feng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89b89c04f55ea7c7ca989992bb6a98c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/89b89c04f55ea7c7ca989992bb6a98c0-Abstract-Conference.html)

        **Abstract**:

        Kernel ridge regression (KRR) has recently attracted renewed interest due to its potential for explaining the transient effects, such as double descent, that emerge during neural network training. In this work, we study how the alignment between the target function and the kernel affects the performance of the KRR. We focus on the truncated KRR (TKRR) which utilizes an additional parameter that controls the spectral truncation of the kernel matrix. We show that for polynomial alignment, there is an over-aligned regime, in which TKRR can achieve a faster rate than what is achievable by full KRR. The rate of TKRR can improve all the way to the parametric rate, while that of full KRR is capped at a sub-optimal value. This shows that target alignemnt can be better leveraged by utilizing spectral truncation in kernel methods. We also consider the bandlimited alignment setting and show that the regularization surface of TKRR can exhibit transient effects including multiple descent and non-monotonic behavior. Our results show that there is a strong and quantifable relation between the shape of the alignment spectrum and the generalization performance of kernel methods, both in terms of rates and in finite samples.

        ----

        ## [1595] Continual Learning In Environments With Polynomial Mixing Times

        **Authors**: *Matthew Riemer, Sharath Chandra Raparthy, Ignacio Cases, Gopeshh Subbaraj, Maximilian Puelma Touzel, Irina Rish*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89c61fce5a8b73871d1c4073f486b134-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/89c61fce5a8b73871d1c4073f486b134-Abstract-Conference.html)

        **Abstract**:

        The mixing time of the Markov chain induced by a policy limits performance in real-world continual learning scenarios. Yet, the effect of mixing times on learning in continual reinforcement learning (RL) remains underexplored. In this paper, we characterize problems that are of long-term interest to the development of continual RL, which we call scalable MDPs, through the lens of mixing times. In particular, we theoretically establish that scalable MDPs have mixing times that scale polynomially with the size of the problem. We go on to demonstrate that polynomial mixing times present significant difficulties for existing approaches that suffer from myopic bias and stale bootstrapped estimates. To validate the proposed theory, we study the empirical scaling behavior of mixing times with respect to the number of tasks and task switching frequency for pretrained high performing policies on seven Atari games. Our analysis demonstrates both that polynomial mixing times do emerge in practice and how their existence may lead to unstable learning behavior like catastrophic forgetting in continual learning settings.

        ----

        ## [1596] ENS-10: A Dataset For Post-Processing Ensemble Weather Forecasts

        **Authors**: *Saleh Ashkboos, Langwen Huang, Nikoli Dryden, Tal Ben-Nun, Peter Dueben, Lukas Gianinazzi, Luca Kummer, Torsten Hoefler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Post-processing ensemble prediction systems can improve the reliability of weather forecasting, especially for extreme event prediction. In recent years, different machine learning models have been developed to improve the quality of weather post-processing. However, these models require a comprehensive dataset of weather simulations to produce high-accuracy results, which comes at a high computational cost to generate. This paper introduces the ENS-10 dataset, consisting of ten ensemble members spanning 20 years (1998--2017). The ensemble members are generated by perturbing numerical weather simulations to capture the chaotic behavior of the Earth. To represent the three-dimensional state of the atmosphere, ENS-10 provides the most relevant atmospheric variables at 11 distinct pressure levels and the surface at \ang{0.5} resolution for forecast lead times T=0, 24, and 48 hours (two data points per week). We propose the ENS-10 prediction correction task for improving the forecast quality at a 48-hour lead time through ensemble post-processing. We provide a set of baselines and compare their skill at correcting the predictions of three important atmospheric variables. Moreover, we measure the baselines' skill at improving predictions of extreme weather events using our dataset. The ENS-10 dataset is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

        ----

        ## [1597] Panchromatic and Multispectral Image Fusion via Alternating Reverse Filtering Network

        **Authors**: *Keyu Yan, Man Zhou, Jie Huang, Feng Zhao, Chengjun Xie, Chongyi Li, Danfeng Hong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/89ef9ce35c7833cba14bb2381ead6c54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/89ef9ce35c7833cba14bb2381ead6c54-Abstract-Conference.html)

        **Abstract**:

        Panchromatic (PAN) and multi-spectral (MS) image fusion, named Pan-sharpening, refers to super-resolve the low-resolution (LR) multi-spectral (MS) images in the spatial domain to generate the expected high-resolution (HR) MS images, conditioning on the corresponding high-resolution PAN images. In this paper, we present a simple yet effective alternating reverse filtering network for pan-sharpening. Inspired by the classical reverse filtering that reverses images to the status before filtering, we formulate pan-sharpening as an alternately iterative reverse filtering process, which fuses LR MS and HR MS in an interpretable manner. Different from existing model-driven methods that require well-designed priors and degradation assumptions, the reverse filtering process avoids the dependency on pre-defined exact priors. To guarantee the stability and convergence of the iterative process via contraction mapping on a metric space, we develop the learnable multi-scale Gaussian kernel module, instead of using specific filters. We demonstrate the theoretical feasibility of such formulations. Extensive experiments on diverse scenes to thoroughly verify the performance of our method, significantly outperforming the state of the arts.

        ----

        ## [1598] Unsupervised Cross-Task Generalization via Retrieval Augmentation

        **Authors**: *Bill Yuchen Lin, Kangmin Tan, Chris Miller, Beiwen Tian, Xiang Ren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8a0d3ae989a382ce6e50312bc35bf7e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8a0d3ae989a382ce6e50312bc35bf7e1-Abstract-Conference.html)

        **Abstract**:

        Humans can perform unseen tasks by recalling relevant skills acquired previously and then generalizing them to the target tasks, even if there is no supervision at all. In this paper, we aim to improve this kind of cross-task generalization ability of massive multi-task language models, such as T0 and FLAN, in an unsupervised setting. We propose a retrieval-augmentation method named ReCross that takes a few unlabelled examples as queries to retrieve a small subset of upstream data and uses them to update the multi-task model for better generalization. ReCross is a straightforward yet effective retrieval method that combines both efficient dense retrieval and effective pair-wise reranking. Our results and analysis show that it significantly outperforms both non-retrieval methods and other baseline methods.

        ----

        ## [1599] Natural gradient enables fast sampling in spiking neural networks

        **Authors**: *Paul Masset, Jacob A. Zavatone-Veth, J. Patrick Connor, Venkatesh Murthy, Cengiz Pehlevan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8a0fd48510590071e3c129a79b8b8527-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8a0fd48510590071e3c129a79b8b8527-Abstract-Conference.html)

        **Abstract**:

        For animals to navigate an uncertain world, their brains need to estimate uncertainty at the timescales of sensations and actions. Sampling-based algorithms afford a theoretically-grounded framework for probabilistic inference in neural circuits, but it remains unknown how one can implement fast sampling algorithms in biologically-plausible spiking networks. Here, we propose to leverage the population geometry, controlled by the neural code and the neural dynamics, to implement fast samplers in spiking neural networks. We first show that two classes of spiking samplers---efficient balanced spiking networks that simulate Langevin sampling, and networks with probabilistic spike rules that implement Metropolis-Hastings sampling---can be unified within a common framework. We then show that careful choice of population geometry, corresponding to the natural space of parameters, enables rapid inference of parameters drawn from strongly-correlated high-dimensional distributions in both networks. Our results suggest design principles for algorithms for sampling-based probabilistic inference in spiking neural networks, yielding potential inspiration for neuromorphic computing and testable predictions for neurobiology.

        ----

        

[Go to the previous page](NIPS-2022-list07.md)

[Go to the next page](NIPS-2022-list09.md)

[Go to the catalog section](README.md)