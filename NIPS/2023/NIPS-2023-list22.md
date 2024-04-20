## [0] Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming

**Authors**: *Gregory Dexter, Petros Drineas, David P. Woodruff, Taisuke Yasuda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9768645621c2cd6c5b851a06205b92cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9768645621c2cd6c5b851a06205b92cf-Abstract-Conference.html)

**Abstract**:

Sketching algorithms have recently proven to be a powerful approach both for designing low-space streaming algorithms as well as fast polynomial time approximation schemes (PTAS). In this work, we develop new techniques to extend the applicability of sketching-based approaches to the sparse dictionary learning and the Euclidean $k$-means clustering problems. In particular, we initiate the study of the challenging setting where the dictionary/clustering assignment for each of the $n$ input points must be output, which has surprisingly received little attention in prior work. On the fast algorithms front, we obtain a new approach for designing PTAS's for the $k$-means clustering problem, which generalizes to the first PTAS for the sparse dictionary learning problem. On the streaming algorithms front, we obtain new upper bounds and lower bounds for dictionary learning and $k$-means clustering. In particular, given a design matrix $\mathbf A\in\mathbb R^{n\times d}$ in a turnstile stream, we show an $\tilde O(nr/\epsilon^2 + dk/\epsilon)$ space upper bound for $r$-sparse dictionary learning of size $k$, an $\tilde O(n/\epsilon^2 + dk/\epsilon)$ space upper bound for $k$-means clustering, as well as an $\tilde O(n)$ space upper bound for $k$-means clustering on random order row insertion streams with a natural "bounded sensitivity" assumption. On the lower bounds side, we obtain a general $\tilde\Omega(n/\epsilon + dk/\epsilon)$ lower bound for $k$-means clustering, as well as an $\tilde\Omega(n/\epsilon^2)$ lower bound for algorithms which can estimate the cost of a single fixed set of candidate centers.

----

## [0] Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation

**Authors**: *Binhui Xie, Shuang Li, Qingju Guo, Chi Harold Liu, Xinjing Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/976cc04f0cbaad7790ce0d665e44f90f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/976cc04f0cbaad7790ce0d665e44f90f-Abstract-Conference.html)

**Abstract**:

Active learning, a label-efficient paradigm, empowers models to interactively query an oracle for labeling new data. In the realm of LiDAR semantic segmentation, the challenges stem from the sheer volume of point clouds, rendering annotation labor-intensive and cost-prohibitive. This paper presents Annotator, a general and efficient active learning baseline, in which a voxel-centric online selection strategy is tailored to efficiently probe and annotate the salient and exemplar voxel girds within each LiDAR scan, even under distribution shift. Concretely, we first execute an in-depth analysis of several common selection strategies such as Random, Entropy, Margin, and then develop voxel confusion degree (VCD) to exploit the local topology relations and structures of point clouds. Annotator excels in diverse settings, with a particular focus on active learning (AL), active source-free domain adaptation (ASFDA), and active domain adaptation (ADA). It consistently delivers exceptional performance across LiDAR semantic segmentation benchmarks, spanning both simulation-to-real and real-to-real scenarios. Surprisingly, Annotator exhibits remarkable efficiency, requiring significantly fewer annotations, e.g., just labeling five voxels per scan in the SynLiDAR â†’ SemanticKITTI task. This results in impressive performance, achieving 87.8% fully-supervised performance under AL, 88.5% under ASFDA, and 94.4% under ADA. We envision that Annotator will offer a simple, general, and efficient solution for label-efficient 3D applications.

----

## [0] Kissing to Find a Match: Efficient Low-Rank Permutation Representation

**Authors**: *Hannah Dröge, Zorah Lähner, Yuval Bahat, Onofre Martorell Nadal, Felix Heide, Michael Moeller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97826456fb8c02fa368d673a49bbc563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97826456fb8c02fa368d673a49bbc563-Abstract-Conference.html)

**Abstract**:

Permutation matrices play a key role in matching and assignment problems across the fields, especially in computer vision and robotics. However, memory for explicitly representing permutation matrices grows quadratically with the size of the problem, prohibiting large problem instances. In this work, we propose to tackle the curse of dimensionality of large  permutation matrices by approximating them using low-rank matrix factorization, followed by a nonlinearity. To this end, we rely on the Kissing number theory to infer the minimal rank required for representing a permutation matrix of a given size, which is significantly smaller than the problem size. This leads to a drastic reduction in computation and memory costs, e.g., up to $3$ orders of magnitude less memory for a problem of size $n=20000$, represented using $8.4\times10^5$ elements in two small matrices instead of using a single huge matrix with $4\times 10^8$ elements. The proposed representation allows for accurate representations of large permutation matrices, which in turn enables handling large problems that would have been infeasible otherwise. We demonstrate the applicability and merits of the proposed approach through a series of experiments on a range of problems that involve predicting permutation matrices, from linear and quadratic assignment to shape matching problems.

----

## [0] Creating Multi-Level Skill Hierarchies in Reinforcement Learning

**Authors**: *Joshua B. Evans, Özgür Simsek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html)

**Abstract**:

What is a useful skill hierarchy for an autonomous agent? We propose an answer based on a graphical representation of how the interaction between an agent and its environment may unfold. Our approach uses modularity maximisation as a central organising principle to expose the structure of the interaction graph at multiple levels of abstraction. The result is a collection of skills that operate at varying time scales, organised into a hierarchy, where skills that operate over longer time scales are composed of skills that operate over shorter time scales. The entire skill hierarchy is generated automatically, with no human input, including the skills themselves (their behaviour, when they can be called, and when they terminate) as well as the dependency structure between them. In a wide range of environments, this approach generates skill hierarchies that are intuitively appealing and that considerably improve the learning performance of the agent.

----

## [0] Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization

**Authors**: *Nathan Grinsztajn, Daniel Furelos-Blanco, Shikha Surana, Clément Bonnet, Tom Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html)

**Abstract**:

Applying reinforcement learning (RL) to combinatorial optimization problems is attractive as it removes the need for expert knowledge or pre-solved instances. However, it is unrealistic to expect an agent to solve these (often NP-)hard problems in a single shot at inference due to their inherent complexity. Thus, leading approaches often implement additional search strategies, from stochastic sampling and beam-search to explicit fine-tuning. In this paper, we argue for the benefits of learning a population of complementary policies, which can be simultaneously rolled out at inference. To this end, we introduce Poppy, a simple training procedure for populations. Instead of relying on a predefined or hand-crafted notion of diversity, Poppy induces an unsupervised specialization targeted solely at maximizing the performance of the population. We show that Poppy produces a set of complementary policies, and obtains state-of-the-art RL results on three popular NP-hard problems: traveling salesman, capacitated vehicle routing, and job-shop scheduling.

----

## [0] Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective

**Authors**: *Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, Han Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html)

**Abstract**:

Linear scalarization, i.e., combining all loss functions by a weighted sum, has been the default choice in the literature of multi-task learning (MTL) since its inception. In recent years, there is a surge of interest in developing Specialized Multi-Task Optimizers (SMTOs) that treat MTL as a multi-objective optimization problem. However, it remains open whether there is a fundamental advantage of SMTOs over scalarization. In fact, heated debates exist in the community comparing these two types of algorithms, mostly from an empirical perspective. To approach the above question, in this paper, we revisit scalarization from a theoretical perspective. We focus on linear MTL models and study whether scalarization is capable of fully exploring the Pareto front. Our findings reveal that, in contrast to recent works that claimed empirical advantages of scalarization, scalarization is inherently incapable of full exploration, especially for those Pareto optimal solutions that strike the balanced trade-offs between multiple tasks. More concretely, when the model is under-parametrized, we reveal a multi-surface structure of the feasible region and identify necessary and sufficient conditions for full exploration. This leads to the conclusion that scalarization is in general incapable of tracing out the Pareto front. Our theoretical results partially answer the open questions in Xin et al. (2021), and provide a more intuitive explanation on why scalarization fails beyond non-convexity. We additionally perform experiments on a real-world dataset using both scalarization and state-of-the-art SMTOs. The experimental results not only corroborate our theoretical findings, but also unveil the potential of SMTOs in finding balanced solutions, which cannot be achieved by scalarization.

----

## [0] Provable Guarantees for Generative Behavior Cloning: Bridging Low-Level Stability and High-Level Behavior

**Authors**: *Adam Block, Ali Jadbabaie, Daniel Pfrommer, Max Simchowitz, Russ Tedrake*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html)

**Abstract**:

We propose a theoretical framework for studying behavior cloning of complex expert demonstrations using generative modeling.Our framework invokes low-level controllers - either learned or implicit in position-command control - to stabilize imitation around expert demonstrations. We show that with (a) a suitable low-level stability guarantee and (b) a powerful enough generative model as our imitation learner,  pure supervised behavior cloning can generate trajectories matching the per-time step distribution of essentially arbitrary expert trajectories in an optimal transport cost. Our analysis relies on a stochastic continuity property of the learned policy we call "total variation continuity" (TVC). We then show that TVC can be ensured with minimal degradation of accuracy by combining a popular data-augmentation regimen with a novel algorithmic trick: adding augmentation noise at execution time. We instantiate our guarantees for policies parameterized by diffusion models and prove that if the learner accurately estimates the score of the (noise-augmented) expert policy, then the distribution of imitator trajectories is close to the demonstrator distribution in a natural optimal transport distance. Our analysis constructs intricate couplings between noise-augmented trajectories, a technique that may be of independent interest. We conclude by empirically validating our algorithmic recommendations, and discussing implications for future research directions for better behavior cloning with generative modeling.

----

## [0] Prefix-Tree Decoding for Predicting Mass Spectra from Molecules

**Authors**: *Samuel Goldman, John Bradshaw, Jiayi Xin, Connor W. Coley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html)

**Abstract**:

Computational predictions of mass spectra from molecules have enabled the discovery of clinically relevant metabolites. However, such predictive tools are still limited as they occupy one of two extremes, either operating  (a) by fragmenting molecules combinatorially with overly rigid constraints on potential rearrangements and poor time complexity or (b) by decoding lossy and nonphysical discretized spectra vectors. In this work, we use a new intermediate strategy for predicting mass spectra from molecules by treating mass spectra as sets of molecular formulae, which are themselves multisets of atoms. After first encoding an input molecular graph, we decode a set of molecular subformulae, each of which specify a predicted peak in the mass spectrum, the intensities of which are predicted by a second model. Our key insight is to overcome the combinatorial possibilities for molecular subformulae by decoding the formula set using a prefix tree structure, atom-type by atom-type, representing a general method for ordered multiset decoding. We show promising empirical results on mass spectra prediction tasks.

----

## [0] Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks

**Authors**: *Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, Sung Ju Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) have shown promising performance in knowledge-intensive reasoning tasks that require a compound understanding of knowledge. However, deployment of the LLMs in real-world applications can be challenging due to their high computational requirements and concerns on data privacy.Previous studies have focused on building task-specific small Language Models (LMs) by fine-tuning them with labeled data or distilling LLMs. However, these approaches are ill-suited for knowledge-intensive reasoning tasks due to the limited capacity of small LMs in memorizing the knowledge required.Motivated by our theoretical analysis on memorization, we propose Knowledge-Augmented Reasoning Distillation (KARD), a novel method that fine-tunes small LMs to generate rationales obtained from LLMs with augmented knowledge retrieved from an external knowledge base. Moreover, we further propose a neural reranker to obtain documents relevant to rationale generation. We empirically show that KARD significantly improves the performance of small T5 and GPT models on the challenging knowledge-intensive reasoning datasets, namely MedQA-USMLE, StrategyQA, and OpenbookQA.Notably, our method makes the 250M T5 models achieve superior performance against the fine-tuned 3B models, having 12 times larger parameters, on both MedQA-USMLE and StrategyQA benchmarks.

----

## [0] Nonparametric Identifiability of Causal Representations from Unknown Interventions

**Authors**: *Julius von Kügelgen, Michel Besserve, Wendong Liang, Luigi Gresele, Armin Kekic, Elias Bareinboim, David M. Blei, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)

**Abstract**:

We study causal representation learning, the task of inferring latent causal variables and their causal relations from high-dimensional functions (“mixtures”) of the variables. Prior work relies on weak supervision, in the form of counterfactual pre- and post-intervention views or temporal structure; places restrictive assumptions, such as linearity, on the mixing function or latent causal model; or requires partial knowledge of the generative process, such as the causal graph or intervention targets. We instead consider the general setting in which both the causal model and the mixing function are nonparametric. The learning signal takes the form of multiple datasets, or environments, arising from unknown interventions in the underlying causal model. Our goal is to identify both the ground truth latents and their causal graph up to a set of ambiguities which we show to be irresolvable from interventional data. We study the fundamental setting of two causal variables and prove that the observational distribution and one perfect intervention per node suffice for identifiability, subject to a genericity condition. This condition rules out spurious solutions that involve fine-tuning of the intervened and observational distributions, mirroring similar conditions for nonlinear cause-effect inference. For an arbitrary number of variables, we show that at least one pair of distinct perfect interventional domains per node guarantees identifiability. Further, we demonstrate that the strengths of causal influences among the latent variables are preserved by all equivalent solutions, rendering the inferred representation appropriate for drawing causal conclusions from new data. Our study provides the first identifiability results for the general nonparametric setting with unknown interventions, and elucidates what is possible and impossible for causal representation learning without more direct supervision.

----

## [0] Dual Mean-Teacher: An Unbiased Semi-Supervised Framework for Audio-Visual Source Localization

**Authors**: *Yuxin Guo, Shijie Ma, Hu Su, Zhiqing Wang, Yuhao Zhao, Wei Zou, Siyang Sun, Yun Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html)

**Abstract**:

Audio-Visual Source Localization (AVSL) aims to locate sounding objects within video frames given the paired audio clips. Existing methods predominantly rely on self-supervised contrastive learning of audio-visual correspondence. Without any bounding-box annotations, they struggle to achieve precise localization, especially for small objects, and suffer from blurry boundaries and false positives. Moreover, the naive semi-supervised method is poor in effectively utilizing the abundance of unlabeled audio-visual pairs. In this paper, we propose a novel Semi-Supervised Learning framework for AVSL, namely Dual Mean-Teacher (DMT), comprising two teacher-student structures to circumvent the confirmation bias issue. Specifically, two teachers, pre-trained on limited labeled data, are employed to filter out noisy samples via the consensus between their predictions, and then generate high-quality pseudo-labels by intersecting their confidence maps. The optimal utilization of both labeled and unlabeled data combined with this unbiased framework enable DMT to outperform current state-of-the-art methods by a large margin, with CIoU of $\textbf{90.4\%}$ and $\textbf{48.8\%}$ on Flickr-SoundNet and VGG-Sound Source, obtaining $\textbf{8.9\%}$ and $\textbf{9.6\%}$ improvements respectively, given only $3\%$ of data positional-annotated. We also extend our framework to some existing AVSL methods and consistently boost their performance. Our code is publicly available at https://github.com/gyx-gloria/DMT.

----

## [0] Hierarchical Gaussian Mixture based Task Generative Model for Robust Meta-Learning

**Authors**: *Yizhou Zhang, Jingchao Ni, Wei Cheng, Zhengzhang Chen, Liang Tong, Haifeng Chen, Yan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/982ca2640e64bf7a1908b028ebc8734a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/982ca2640e64bf7a1908b028ebc8734a-Abstract-Conference.html)

**Abstract**:

Meta-learning enables quick adaptation of machine learning models to new tasks with limited data. While tasks could come from varying distributions in reality, most of the existing meta-learning methods consider both training and testing tasks as from the same uni-component distribution, overlooking two critical needs of a practical solution: (1) the various sources of tasks may compose a multi-component mixture distribution, and (2) novel tasks may come from a distribution that is unseen during meta-training. In this paper, we demonstrate these two challenges can be solved jointly by modeling the density of task instances. We develop a meta-training framework underlain by a novel Hierarchical Gaussian Mixture based Task Generative Model (HTGM). HTGM extends the widely used empirical process of sampling tasks to a theoretical model, which learns task embeddings, fits the mixture distribution of tasks, and enables density-based scoring of novel tasks. The framework is agnostic to the encoder and scales well with large backbone networks. The model parameters are learned end-to-end by maximum likelihood estimation via an Expectation-Maximization (EM) algorithm. Extensive experiments on benchmark datasets indicate the effectiveness of our method for both sample classification and novel task detection.

----

## [0] Temporal Dynamic Quantization for Diffusion Models

**Authors**: *Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, Eunhyeok Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/983591c3e9a0dc94a99134b3238bbe52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/983591c3e9a0dc94a99134b3238bbe52-Abstract-Conference.html)

**Abstract**:

Diffusion model has gained popularity in vision applications due to its remarkable generative performance and versatility. However, its high storage and computation demands, resulting from the model size and iterative generation, hinder its use on mobile devices. Existing quantization techniques struggle to maintain performance even in 8-bit precision due to the diffusion model's unique property of temporal variation in activation. We introduce a novel quantization method that dynamically adjusts the quantization interval based on time step information, significantly improving output quality. Unlike conventional dynamic quantization techniques, our approach has no computational overhead during inference and is compatible with both post-training quantization (PTQ) and quantization-aware training (QAT). Our extensive experiments demonstrate substantial improvements in output quality with the quantized model across various configurations.

----

## [0] Learning Interpretable Low-dimensional Representation via Physical Symmetry

**Authors**: *Xuanjie Liu, Daniel Chin, Yichen Huang, Gus Xia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9850e6a5410331290dc1deefb7514448-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9850e6a5410331290dc1deefb7514448-Abstract-Conference.html)

**Abstract**:

We have recently seen great progress in learning interpretable music representations, ranging from basic factors, such as pitch and timbre, to high-level concepts, such as chord and texture. However, most methods rely heavily on music domain knowledge. It remains an open question what general computational principles give rise to interpretable representations, especially low-dim factors that agree with human perception. In this study, we take inspiration from modern physics and use physical symmetry as a self-consistency constraint for the latent space. Specifically, it requires the prior model that characterises the dynamics of the latent states to be equivariant with respect to certain group transformations. We show that physical symmetry leads the model to learn a linear pitch factor from unlabelled monophonic music audio in a self-supervised fashion. In addition, the same methodology can be applied to computer vision, learning a 3D Cartesian space from videos of a simple moving object without labels. Furthermore, physical symmetry naturally leads to counterfactual representation augmentation, a new technique which improves sample efficiency.

----

## [0] ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation

**Authors**: *Yasheng Sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98530736e5d94e62b689dfc1fda89bd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98530736e5d94e62b689dfc1fda89bd1-Abstract-Conference.html)

**Abstract**:

While language-guided image manipulation has made remarkable progress, the challenge of how to instruct the manipulation process faithfully reflecting human intentions persists. An accurate and comprehensive description of a manipulation task using natural language is laborious and sometimes even impossible, primarily due to the inherent uncertainty and ambiguity present in linguistic expressions. Is it feasible to accomplish image manipulation without resorting to external cross-modal language information? If this possibility exists, the inherent modality gap would be effortlessly eliminated. In this paper, we propose a novel  manipulation methodology, dubbed ImageBrush, that learns visual instructions for more accurate image editing.Our key idea is to employ a pair of transformation images as visual instructions, which not only precisely captures human intention but also facilitates accessibility in real-world scenarios. Capturing visual instructions is particularly challenging because it involves extracting the underlying intentions solely from visual demonstrations and then applying this operation to a new image. To address this challenge, we formulate visual instruction learning as a diffusion-based inpainting problem, where the contextual information is fully exploited through an iterative process of generation. A visual prompting encoder is carefully devised to enhance the model's capacity in uncovering human intent behind the visual instructions. Extensive experiments show that our method generates engaging manipulation results conforming to the transformations entailed in demonstrations. Moreover, our model exhibits robust generalization capabilities on various downstream tasks such as pose transfer, image translation and video inpainting.

----

## [0] Meek Separators and Their Applications in Targeted Causal Discovery

**Authors**: *Kirankumar Shiragur, Jiaqi Zhang, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/985786d06c1e45e9e8c65f7aca3547e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/985786d06c1e45e9e8c65f7aca3547e4-Abstract-Conference.html)

**Abstract**:

Learning causal structures from interventional data is a fundamental problem with broad applications across various fields. While many previous works have focused on recovering the entire causal graph, in practice, there are scenarios where learning only part of the causal graph suffices. This is called \emph{targeted} causal discovery. In our work, we focus on two such well-motivated problems: subset search and causal matching. We aim to minimize the number of interventions in both cases.Towards this, we introduce the \emph{Meek separator}, which is a subset of vertices that, when intervened, decomposes the remaining unoriented edges into smaller connected components. We then present an efficient algorithm to find Meek separators that are of small sizes. Such a procedure is helpful in designing various divide-and-conquer-based approaches. In particular, we propose two randomized algorithms that achieve logarithmic approximation for subset search and causal matching, respectively. Our results provide the first known average-case provable guarantees for both problems. We believe that this opens up possibilities to design near-optimal methods for many other targeted causal structure learning problems arising from various applications.

----

## [0] CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence

**Authors**: *Bong Gyun Kang, HyunGi Kim, Dahuin Jung, Sungroh Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/986e0caad271b59417287737416d8594-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/986e0caad271b59417287737416d8594-Abstract-Conference.html)

**Abstract**:

Continual learning (CL) aims to incrementally learn multiple tasks that are presented sequentially. The significance of CL lies not only in the practical importance but also in studying the learning mechanisms of humans who are excellent continual learners. While most research on CL has been done on structured data such as images, there is a lack of research on CL for abstract logical concepts such as counting, sorting, and arithmetic, which humans learn gradually over time in the real world. In this work, for the first time, we introduce novel algorithmic reasoning (AR) methodology for continual tasks of abstract concepts: CLeAR. Our methodology proposes a one-to-many mapping of input distribution to a shared mapping space, which allows the alignment of various tasks of different dimensions and shared semantics. Our tasks of abstract logical concepts, in the form of formal language, can be classified into Chomsky hierarchies based on their difficulty. In this study, we conducted extensive experiments consisting of 15 tasks with various levels of Chomsky hierarchy, ranging from in-hierarchy to inter-hierarchy scenarios. CLeAR not only achieved near zero forgetting but also improved accuracy during following tasks, a phenomenon known as backward transfer, while previous CL methods designed for image classification drastically failed.

----

## [0] SLIBO-Net: Floorplan Reconstruction via Slicing Box Representation with Local Geometry Regularization

**Authors**: *Jheng-Wei Su, Kuei-Yu Tung, Chi-Han Peng, Peter Wonka, Hung-Kuo Chu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html)

**Abstract**:

This paper focuses on improving the reconstruction of 2D floorplans from unstructured 3D point clouds. We identify opportunities for enhancement over the existing methods in three main areas: semantic quality, efficient representation, and local geometric details. To address these, we presents SLIBO-Net, an innovative approach to reconstructing 2D floorplans from unstructured 3D point clouds. We propose a novel transformer-based architecture that employs an efficient floorplan representation, providing improved room shape supervision and allowing for manageable token numbers. By incorporating geometric priors as a regularization mechanism and post-processing step, we enhance the capture of local geometric details. We also propose a scale-independent evaluation metric, correcting the discrepancy in error treatment between varying floorplan sizes. Our approach notably achieves a new state-of-the-art on the Structure3D dataset. The resultant floorplans exhibit enhanced semantic plausibility, substantially improving the overall quality and realism of the reconstructions. Our code and dataset are available online.

----

## [0] Fantastic Robustness Measures: The Secrets of Robust Generalization

**Authors**: *Hoki Kim, Jinseong Park, Yujin Choi, Jaewook Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html)

**Abstract**:

Adversarial training has become the de-facto standard method for improving the robustness of models against adversarial examples. However, robust overfitting remains a significant challenge, leading to a large gap between the robustness on the training and test datasets. To understand and improve robust generalization, various measures have been developed, including margin, smoothness, and flatness-based measures. In this study, we present a large-scale analysis of robust generalization to empirically verify whether the relationship between these measures and robust generalization remains valid in diverse settings. We demonstrate when and how these measures effectively capture the robust generalization gap by comparing over 1,300 models trained on CIFAR-10 under the $L_\infty$ norm and further validate our findings through an evaluation of more than 100 models from RobustBench across CIFAR-10, CIFAR-100, and ImageNet. We hope this work can help the community better understand adversarial robustness and motivate the development of more robust defense methods against adversarial attacks.

----

## [0] A Spectral Algorithm for List-Decodable Covariance Estimation in Relative Frobenius Norm

**Authors**: *Ilias Diakonikolas, Daniel Kane, Jasper C. H. Lee, Ankit Pensia, Thanasis Pittas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html)

**Abstract**:

We study the problem of list-decodable Gaussian covariance estimation. Given a multiset $T$ of $n$ points in $\mathbb{R}^d$ such that an unknown $\alpha<1/2$ fraction of points in $T$ are i.i.d. samples from an unknown Gaussian $\mathcal{N}(\mu, \Sigma)$, the goal is to output a list of $O(1/\alpha)$ hypotheses at least one of which is close to $\Sigma$ in relative Frobenius norm. Our main result is a $\mathrm{poly}(d,1/\alpha)$ sample and time algorithm for this task that guarantees relative Frobenius norm error of $\mathrm{poly}(1/\alpha)$. Importantly, our algorithm relies purely on spectral techniques. As a corollary, we obtain an efficient spectral algorithm for robust partial clustering of Gaussian mixture models (GMMs) --- a key ingredient in the recent work of [BakDJKKV22] on robustly learning arbitrary GMMs. Combined with the other components of [BakDJKKV22], our new method yields the first Sum-of-Squares-free algorithm for robustly learning GMMs, resolving an open problem proposed by Vempala and Kothari. At the technical level, we develop a novel multi-filtering method for list-decodable covariance estimation that may be useful in other settings.

----

## [0] Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models

**Authors**: *Simian Luo, Chuanhao Yan, Chenxu Hu, Hang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98c50f47a37f63477c01558600dd225a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98c50f47a37f63477c01558600dd225a-Abstract-Conference.html)

**Abstract**:

The Video-to-Audio (V2A) model has recently gained attention for its practical application in generating audio directly from silent videos, particularly in video/film production. However, previous methods in V2A have limited generation quality in terms of temporal synchronization and audio-visual relevance. We present Diff-Foley, a synchronized Video-to-Audio synthesis method with a latent diffusion model (LDM) that generates high-quality audio with improved synchronization and audio-visual relevance. We adopt contrastive audio-visual pretraining (CAVP) to learn more temporally and semantically aligned features, then train an LDM with CAVP-aligned visual features on spectrogram latent space. The CAVP-aligned features enable LDM to capture the subtler audio-visual correlation via a cross-attention module. We further significantly improve sample quality with `double guidance'. Diff-Foley achieves state-of-the-art V2A performance on current large scale V2A dataset. Furthermore, we demonstrate Diff-Foley practical applicability and adaptability via customized downstream finetuning. Project Page: https://diff-foley.github.io/

----

## [0] The Drunkard's Odometry: Estimating Camera Motion in Deforming Scenes

**Authors**: *David Recasens, Martin R. Oswald, Marc Pollefeys, Javier Civera*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98c9b79e9c686aadd4d81e34a7773dd1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/98c9b79e9c686aadd4d81e34a7773dd1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Estimating camera motion in deformable scenes poses a complex and open research challenge. Most existing non-rigid structure from motion techniques assume to observe also static scene parts besides deforming scene parts in order to establish an anchoring reference. However, this assumption does not hold true in certain relevant application cases such as endoscopies. Deformable odometry and SLAM pipelines, which tackle the most challenging scenario of exploratory trajectories, suffer from a lack of robustness and proper quantitative evaluation methodologies. To tackle this issue with a common benchmark, we introduce the Drunkard's Dataset, a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments. This dataset is the first large set of exploratory camera trajectories with ground truth inside 3D scenes where every surface exhibits non-rigid deformations over time. Simulations in realistic 3D buildings lets us obtain a vast amount of data and ground truth labels, including camera poses, RGB images and depth, optical flow and normal maps at high resolution and quality. We further present a novel deformable odometry method, dubbed the Drunkard’s Odometry, which decomposes optical flow estimates into rigid-body camera motion and non-rigid scene deformations. In order to validate our data, our work contains an evaluation of several baselines as well as a novel tracking error metric which does not require ground truth data. Dataset and code: https://davidrecasens.github.io/TheDrunkard'sOdometry/

----

## [0] Optimal Treatment Allocation for Efficient Policy Evaluation in Sequential Decision Making

**Authors**: *Ting Li, Chengchun Shi, Jianing Wang, Fan Zhou, Hongtu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98d0ad88db1e51bd0aa341a823290ece-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98d0ad88db1e51bd0aa341a823290ece-Abstract-Conference.html)

**Abstract**:

A/B testing is critical for modern technological companies to evaluate the effectiveness of newly developed products against standard baselines. This paper studies optimal designs that aim to maximize the amount of information obtained from online experiments to estimate treatment effects accurately. We propose three optimal allocation strategies in a dynamic setting where treatments are sequentially assigned over time. These strategies are designed to minimize the variance of the treatment effect estimator when data follow a non Markov decision process or a (time-varying) Markov decision process. We further develop estimation procedures based on existing off-policy evaluation (OPE) methods and conduct extensive experiments in various environments to demonstrate the effectiveness of the proposed methodologies. In theory, we prove the optimality of the proposed treatment allocation design and establish upper bounds for the mean squared errors of the resulting treatment effect estimators.

----

## [0] Advancing Bayesian Optimization via Learning Correlated Latent Space

**Authors**: *Seunghun Lee, Jaewon Chu, Sihyeon Kim, Juyeon Ko, Hyunwoo J. Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html)

**Abstract**:

Bayesian optimization is a powerful method for optimizing black-box functions with limited function evaluations. Recent works have shown that optimization in a latent space through deep generative models such as variational autoencoders leads to effective and efficient Bayesian optimization for structured or discrete data. However, as the optimization does not take place in the input space, it leads to an inherent gap that results in potentially suboptimal solutions. To alleviate the discrepancy, we propose Correlated latent space Bayesian Optimization (CoBO), which focuses on learning correlated latent spaces characterized by a strong correlation between the distances in the latent space and the distances within the objective function. Specifically, our method introduces Lipschitz regularization, loss weighting, and trust region recoordination to minimize the inherent gap around the promising areas. We demonstrate the effectiveness of our approach on several optimization tasks in discrete data, such as molecule design and arithmetic expression fitting, and achieve high performance within a small budget.

----

## [0] Generalization bounds for neural ordinary differential equations and deep residual networks

**Authors**: *Pierre Marion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equations (neural ODEs) are a popular family of continuous-depth deep learning models.  In this work, we consider a large family of parameterized ODEs with continuous-in-time parameters, which include time-dependent neural ODEs.  We derive a generalization bound for this class by a Lipschitz-based argument. By leveraging the analogy between neural ODEs and deep residual networks, our approach yields in particular a generalization bound for a class of deep residual networks. The bound involves the magnitude of the difference between successive weight matrices. We illustrate numerically how this quantity affects the generalization capability of neural networks.

----

## [0] Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data

**Authors**: *Sai Aparna Aketi, Abolfazl Hashemi, Kaushik Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html)

**Abstract**:

Decentralized learning enables the training of deep learning models over large distributed datasets generated at different locations, without the need for a central server. However, in practical scenarios, the data distribution across these devices can be significantly different, leading to a degradation in model performance. In this paper, we focus on designing a decentralized learning algorithm that is less susceptible to variations in data distribution across devices. We propose Global Update Tracking (GUT), a novel tracking-based method that aims to mitigate the impact of heterogeneous data in decentralized learning without introducing any communication overhead. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, and ImageNette), model architectures, and network topologies. Our experiments show that the proposed method achieves state-of-the-art performance for decentralized learning on heterogeneous data via a 1-6% improvement in test accuracy compared to other existing techniques.

----

## [0] QuadAttacK: A Quadratic Programming Approach to Learning Ordered Top-K Adversarial Attacks

**Authors**: *Thomas Paniagua, Ryan Grainger, Tianfu Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9902a53031ebbbab73898028073d4790-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9902a53031ebbbab73898028073d4790-Abstract-Conference.html)

**Abstract**:

The adversarial vulnerability of Deep Neural Networks (DNNs) has been well-known and widely concerned, often under the context of learning top-$1$ attacks (e.g., fooling a DNN to classify a cat image as dog). This paper shows that the concern is much more serious by learning significantly more aggressive ordered top-$K$ clear-box targeted attacks proposed in~\citep{zhang2020learning}. We propose a novel and rigorous quadratic programming (QP) method of learning ordered top-$K$ attacks with low computing cost, dubbed as \textbf{QuadAttac$K$}. Our QuadAttac$K$ directly solves the QP to satisfy the attack constraint in the feature embedding space (i.e., the input space to the final linear classifier), which thus exploits the semantics of the feature embedding space (i.e., the principle of class coherence). With the optimized feature embedding  vector perturbation, it then computes the adversarial perturbation in the data space via the vanilla one-step back-propagation. In experiments, the proposed QuadAttac$K$ is tested in the ImageNet-1k  classification using ResNet-50, DenseNet-121, and Vision Transformers (ViT-B and DEiT-S). It successfully pushes the boundary of successful ordered top-$K$ attacks from $K=10$ up to $K=20$ at a cheap budget ($1\times 60$) and further improves attack success rates for $K=5$ for all tested models, while retaining the performance for $K=1$.

----

## [0] Predicting mutational effects on protein-protein binding via a side-chain diffusion probabilistic model

**Authors**: *Shiwei Liu, Tian Zhu, Milong Ren, Chungong Yu, Dongbo Bu, Haicang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html)

**Abstract**:

Many crucial biological processes rely on networks of protein-protein interactions. Predicting the effect of amino acid mutations on protein-protein binding is important in protein engineering, including therapeutic discovery. However, the scarcity of annotated experimental data on binding energy poses a significant challenge for developing computational approaches, particularly deep learning-based methods. In this work, we propose SidechainDiff, a novel representation learning-based approach that leverages unlabelled experimental protein structures. SidechainDiff utilizes a Riemannian diffusion model to learn the generative process of side-chain conformations and can also give the structural context representations of mutations on the protein-protein interface. Leveraging the learned representations, we achieve state-of-the-art performance in predicting the mutational effects on protein-protein binding. Furthermore, SidechainDiff is the first diffusion-based generative model for side-chains, distinguishing it from prior efforts that have predominantly focused on the generation of protein backbone structures.

----

## [0] PETAL: Physics Emulation Through Averaged Linearizations for Solving Inverse Problems

**Authors**: *Jihui Jin, Etienne Ollivier, Richard Touret, Matthew McKinley, Karim Sabra, Justin Romberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/991c9324ca71aa85ab4dd11146b35fc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/991c9324ca71aa85ab4dd11146b35fc3-Abstract-Conference.html)

**Abstract**:

Inverse problems describe the task of recovering an underlying signal of interest given observables. Typically, the observables are related via some non-linear forward model applied to the underlying unknown signal. Inverting the non-linear forward model can be computationally expensive, as it often involves computing and inverting a linearization at a series of estimates. Rather than inverting the physics-based model, we instead train a surrogate forward model (emulator) and leverage modern auto-grad libraries to solve for the input within a classical optimization framework. Current methods to train emulators are done in a black box supervised machine learning fashion and fail to take advantage of any existing knowledge of the forward model. In this article, we propose a simple learned weighted average model that embeds linearizations of the forward model around various reference points into the model itself, explicitly incorporating known physics. Grounding the learned model with physics based linearizations improves the forward modeling accuracy and provides richer physics based gradient information during the inversion process leading to more accurate signal recovery. We demonstrate the efficacy on an ocean acoustic tomography (OAT) example that aims to recover ocean sound speed profile (SSP) variations from acoustic observations (e.g. eigenray arrival times) within simulation of ocean dynamics in the Gulf of Mexico.

----

## [0] RealTime QA: What's the Answer Right Now?

**Authors**: *Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, Kentaro Inui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9941624ef7f867a502732b5154d30cb7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9941624ef7f867a502732b5154d30cb7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce RealTime QA, a dynamic question answering (QA) platform that announces questions and evaluates systems on a regular basis (weekly in this version). RealTime QA inquires about the current world, and QA systems need to answer questions about novel events or information. It therefore challenges static, conventional assumptions in open-domain QA datasets and pursues instantaneous applications. We build strong baseline models upon large pretrained language models, including GPT-3 and T5. Our benchmark is an ongoing effort, and this paper presents real-time evaluation results over the past year. Our experimental results show that GPT-3 can often properly update its generation results, based on newly-retrieved documents, highlighting the importance of up-to-date information retrieval. Nonetheless, we find that GPT-3 tends to return outdated answers when retrieved documents do not provide sufficient information to find an answer. This suggests an important avenue for future research: can an open-domain QA system identify such unanswerable cases and communicate with the user or even the retrieval module to modify the retrieval results? We hope that RealTime QA will spur progress in instantaneous applications of question answering and beyond.

----

## [0] Learning Transformer Programs

**Authors**: *Dan Friedman, Alexander Wettig, Danqi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract-Conference.html)

**Abstract**:

Recent research in mechanistic interpretability has attempted to reverse-engineer Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of providing complete, faithful descriptions of the underlying algorithms. In this work, we introduce a procedure for training Transformers that are mechanistically interpretable by design. We build on RASP [Weiss et al., 2021], a programming language that can be compiled into Transformer weights. Instead of compiling human-written programs into Transformers, we design a modified Transformer that can be trained using gradient-based optimization and then automatically converted into a discrete, human-readable program. We refer to these models as Transformer Programs. To validate our approach, we learn Transformer Programs for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck languages), and NLP tasks including named entity recognition and text classification. The Transformer Programs can automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To demonstrate these advantages, we convert Transformers into Python programs and use off-the-shelf code analysis tools to debug model errors and identify the “circuits” used to solve different sub-problems. We hope that Transformer Programs open a new path toward the goal of intrinsically interpretable machine learning.

----

## [0] An Inverse Scaling Law for CLIP Training

**Authors**: *Xianhang Li, Zeyu Wang, Cihang Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html)

**Abstract**:

CLIP, one of the pioneering foundation models that connect images and text, has enabled many recent breakthroughs in computer vision. However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. In this paper, we present a surprising finding that there exists an inverse scaling law for CLIP training, whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law.As a result of this finding, we are able to successfully train CLIP even with limited computational resources. For example, using 8 A100 GPUs, our CLIP models achieve zero-shot top-1 ImageNet-1k accuracies of 63.2% in ~2 days, 67.8% in ~3 days, and 69.3% in ~4 days. Our method also works well when scaling up --- with G/14, we register a new record of 83.0% ImageNet-1k zero-shot accuracy, and meanwhile accelerate the training by ~33x compared to its OpenCLIP counterpart.By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics. Our code is available at https://github.com/UCSC-VLAA/CLIPA.

----

## [0] Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback

**Authors**: *Minyoung Hwang, Gunmin Lee, Hogun Kee, Chan Woo Kim, Kyungjae Lee, Songhwai Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99766cda865be123d55a1d9666c7b9fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99766cda865be123d55a1d9666c7b9fc-Abstract-Conference.html)

**Abstract**:

Reinforcement learning from human feedback (RLHF) alleviates the problem of designing a task-specific reward function in reinforcement learning by learning it from human preference. However, existing RLHF models are considered inefficient as they produce only a single preference data from each human feedback. To tackle this problem, we propose a novel RLHF framework called SeqRank, that uses sequential preference ranking to enhance the feedback efficiency. Our method samples trajectories in a sequential manner by iteratively selecting a defender from the set of previously chosen trajectories $\mathcal{K}$ and a challenger from the set of unchosen trajectories $\mathcal{U}\setminus\mathcal{K}$, where $\mathcal{U}$ is the replay buffer. We propose two trajectory comparison methods with different defender sampling strategies: (1) sequential pairwise comparison that selects the most recent trajectory and (2) root pairwise comparison that selects the most preferred trajectory from $\mathcal{K}$. We construct a data structure and rank trajectories by preference to augment additional queries. The proposed method results in at least 39.2% higher average feedback efficiency than the baseline and also achieves a balance between feedback efficiency and data dependency. We examine the convergence of the empirical risk and the generalization bound of the reward model with Rademacher complexity. While both trajectory comparison methods outperform conventional pairwise comparison, root pairwise comparison improves the average reward in locomotion tasks and the average success rate in manipulation tasks by 29.0% and 25.0%, respectively. The source code and the videos are provided in the supplementary material.

----

## [0] Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection

**Authors**: *Cheng-Ju Ho, Chen-Hsuan Tai, Yen-Yu Lin, Ming-Hsuan Yang, Yi-Hsuan Tsai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99786eed5e16920f908572fb00e151c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99786eed5e16920f908572fb00e151c3-Abstract-Conference.html)

**Abstract**:

Semi-supervised object detection is crucial for 3D scene understanding, efficiently addressing the limitation of acquiring large-scale 3D bounding box annotations. Existing methods typically employ a teacher-student framework with pseudo-labeling to leverage unlabeled point clouds. However, producing reliable pseudo-labels in a diverse 3D space still remains challenging. In this work, we propose Diffusion-SS3D, a new perspective of enhancing the quality of pseudo-labels via the diffusion model for semi-supervised 3D object detection. Specifically, we include noises to produce corrupted 3D object size and class label distributions, and then utilize the diffusion model as a denoising process to obtain bounding box outputs. Moreover, we integrate the diffusion model into the teacher-student framework, so that the denoised bounding boxes can be used to improve pseudo-label generation, as well as the entire semi-supervised learning process. We conduct experiments on the ScanNet and SUN RGB-D benchmark datasets to demonstrate that our approach achieves state-of-the-art performance against existing methods. We also present extensive analysis to understand how our diffusion model design affects performance in semi-supervised learning. The source code will be available at https://github.com/luluho1208/Diffusion-SS3D.

----

## [0] Aligning Language Models with Human Preferences via a Bayesian Approach

**Authors**: *Jiashuo Wang, Haozhao Wang, Shichao Sun, Wenjie Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html)

**Abstract**:

In the quest to advance human-centric natural language generation (NLG) systems, ensuring alignment between NLG models and human preferences is crucial. For this alignment, current popular methods leverage a reinforcement learning (RL) approach with a reward model trained on feedback from humans. However, inherent disagreements due to the subjective nature of human preferences pose a significant challenge for training the reward model, resulting in a deterioration of the NLG performance. To tackle this issue, previous approaches typically rely on majority voting or averaging to consolidate multiple inconsistent preferences into a merged one. Although straightforward to understand and execute, such methods suffer from an inability to capture the nuanced degrees of disaggregation among humans and may only represent a specialized subset of individuals, thereby lacking the ability to quantitatively disclose the universality of human preferences. To address this challenge, this paper proposes a novel approach, which employs a Bayesian framework to account for the distribution of disagreements among human preferences as training a preference model, and names it as $\textbf{d-PM}$. Besides, considering the RL strategy's inefficient and complex training process over the training efficiency, we further propose utilizing the contrastive learning strategy to train the NLG model with the preference scores derived from the d-PM model. Extensive experiments on two human-centric NLG tasks, i.e., emotional support conversation and integrity ``Rule-of-Thumb'' generation, show that our method consistently exceeds previous SOTA models in both automatic and human evaluations.

----

## [0] A Smooth Binary Mechanism for Efficient Private Continual Observation

**Authors**: *Joel Daniel Andersson, Rasmus Pagh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99c41fb9fd53abfdd4a0259560ef1c9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99c41fb9fd53abfdd4a0259560ef1c9d-Abstract-Conference.html)

**Abstract**:

In privacy under continual observation we study how to release differentially private estimates based on a dataset that evolves over time. The problem of releasing private prefix sums of $x_1, x_2, x_3,\dots\in${$0,1$} (where the value of each $x_i$ is to be private) is particularly well-studied, and a generalized form is used in state-of-the-art methods for private stochastic gradient descent (SGD).The seminal binary mechanism privately releases the first $t$ prefix sums with noise of variance polylogarithmic in $t$. Recently, Henzinger et al. and Denisov et al. showed that it is possible to improve on the binary mechanism in two ways: The variance of the noise can be reduced by a (large) constant factor, and also made more even across time steps. However, their algorithms for generating the noise distribution are not as efficient as one would like in terms of computation time and (in particular) space.We address the efficiency problem by presenting a simple alternative to the binary mechanism in which 1) generating the noise takes constant average time per value, 2) the variance is reduced by a factor about 4 compared to the binary mechanism, and 3) the noise distribution at each step is identical. Empirically, a simple Python implementation of our approach outperforms the running time of the approach of Henzinger et al., as well as an attempt to improve their algorithm using high-performance algorithms for multiplication with Toeplitz matrices.

----

## [0] Training Transformers with 4-bit Integers

**Authors**: *Haocheng Xi, Changhao Li, Jianfei Chen, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html)

**Abstract**:

Quantizing the activation, weight, and gradient to 4-bit is promising to accelerate neural network training. However, existing 4-bit training methods require custom numerical formats  which are not supported by contemporary hardware. In this work, we propose a training method for transformers with all matrix multiplications implemented with the INT4 arithmetic. Training with an ultra-low INT4 precision is challenging. To achieve this, we carefully analyze the specific structures of activation and gradients in transformers to propose dedicated quantizers for them. For forward propagation, we identify the challenge of outliers and propose a Hadamard quantizer to suppress the outliers. For backpropagation, we leverage the structural sparsity of gradients by proposing bit splitting and leverage score sampling techniques to quantize gradients accurately. Our algorithm achieves competitive accuracy on a wide range of tasks including natural language understanding, machine translation, and image classification. Unlike previous 4-bit training methods, our algorithm can be implemented on the current generation of GPUs. Our prototypical linear operator implementation is up to 2.2 times faster than the FP16 counterparts and speeds up the training by 17.8\% on average for sufficiently large models. Our code is available at https://github.com/xijiu9/Train_Transformers_with_INT4.

----

## [0] TD Convergence: An Optimization Perspective

**Authors**: *Kavosh Asadi, Shoham Sabach, Yao Liu, Omer Gottesman, Rasool Fakoor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a08fbb992f15faa695c42b6a2c8e000-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a08fbb992f15faa695c42b6a2c8e000-Abstract-Conference.html)

**Abstract**:

We study the convergence behavior of the celebrated temporal-difference (TD) learning algorithm. By looking at the algorithm through the lens of optimization, we first argue that TD can be viewed as an iterative optimization algorithm where the function to be minimized changes per iteration. By carefully investigating the divergence displayed by TD on a classical counter example, we identify two forces that determine the convergent or divergent behavior of the algorithm. We next formalize our discovery in the linear TD setting with quadratic loss and prove that convergence of TD hinges on the interplay between these two forces. We extend this optimization perspective to prove convergence of TD in a much broader setting than just linear approximation and squared loss. Our results provide a theoretical explanation for the successful application of TD in reinforcement learning.

----

## [0] Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

**Authors**: *Zekun Li, Shiyang Li, Xifeng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a17c1eb808cf012065e9db47b7ca80d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a17c1eb808cf012065e9db47b7ca80d-Abstract-Conference.html)

**Abstract**:

Irregularly sampled time series are increasingly prevalent, particularly in medical domains. While various specialized methods have been developed to handle these irregularities, effectively modeling their complex dynamics and pronounced sparsity remains a challenge. This paper introduces a novel perspective by converting irregularly sampled time series into line graph images, then utilizing powerful pre-trained vision transformers for time series classification in the same way as image classification. This method not only largely simplifies specialized algorithm designs but also presents the potential to serve as a universal framework for time series modeling. Remarkably, despite its simplicity, our approach outperforms state-of-the-art specialized algorithms on several popular healthcare and human activity datasets. Especially in the rigorous leave-sensors-out setting where a portion of variables is omitted during testing, our method exhibits strong robustness against varying degrees of missing observations, achieving an impressive improvement of 42.8% in absolute F1 score points over leading specialized baselines even with half the variables masked. Code and data are available at https://github.com/Leezekun/ViTST.

----

## [0] Symbolic Discovery of Optimization Algorithms

**Authors**: *Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html)

**Abstract**:

We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training. We leverage efficient search techniques to explore an infinite and sparse program space. To bridge the large generalization gap between proxy and target tasks, we also introduce program selection and simplification strategies.Our method discovers a simple and effective optimization algorithm, $\textbf{Lion}$ ($\textit{Evo$\textbf{L}$ved S$\textbf{i}$gn M$\textbf{o}$me$\textbf{n}$tum}$). It is more memory-efficient than Adam as it only keeps track of the momentum. Different from adaptive optimizers, its update has the same magnitude for each parameter calculated through the sign operation.We compare Lion with widely used optimizers, such as Adam and Adafactor, for training a variety of models on different tasks. On image classification, Lion boosts the accuracy of ViT by up to 2\% on ImageNet and saves up to 5x the pre-training compute on JFT. On vision-language contrastive learning, we achieve 88.3\% $\textit{zero-shot}$ and 91.1\% $\textit{fine-tuning}$ accuracy on ImageNet, surpassing the previous best results by 2\% and 0.1\%, respectively. On diffusion models, Lion outperforms Adam by achieving a better FID score and reducing the training compute by up to 2.3x. For autoregressive, masked language modeling, and fine-tuning, Lion exhibits a similar or better performance compared to Adam. Our analysis of Lion reveals that its performance gain grows with the training batch size. It also requires a smaller learning rate than Adam due to the larger norm of the update produced by the sign function. Additionally, we examine the limitations of Lion and identify scenarios where its improvements are small or not statistically significant.

----

## [0] On Calibrating Diffusion Probabilistic Models

**Authors**: *Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a645c38d4ec6f94633a35aeb2079596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a645c38d4ec6f94633a35aeb2079596-Abstract-Conference.html)

**Abstract**:

Recently, diffusion probabilistic models (DPMs) have achieved promising results in diverse generative tasks. A typical DPM framework includes a forward process that gradually diffuses the data distribution and a reverse process that recovers the data distribution from time-dependent data scores. In this work, we observe that the stochastic reverse process of data scores is a martingale, from which concentration bounds and the optional stopping theorem for data scores can be derived. Then, we discover a simple way for calibrating an arbitrary pretrained DPM, with which the score matching loss can be reduced and the lower bounds of model likelihood can consequently be increased. We provide general calibration guidelines under various model parametrizations. Our calibration method is performed only once and the resulting models can be used repeatedly for sampling. We conduct experiments on multiple datasets to empirically validate our proposal. Our code is available at https://github.com/thudzj/Calibrated-DPMs.

----

## [0] InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

**Authors**: *Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven C. H. Hoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html)

**Abstract**:

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-source.

----

## [0] Privacy Auditing with One (1) Training Run

**Authors**: *Thomas Steinke, Milad Nasr, Matthew Jagielski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a6f6e0d6781d1cb8689192408946d73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a6f6e0d6781d1cb8689192408946d73-Abstract-Conference.html)

**Abstract**:

We propose a scheme for auditing differentially private machine learning systems with a single training run. This exploits the parallelism of being able to add or remove multiple training examples independently. We analyze this using the connection between differential privacy and statistical generalization, which avoids the cost of group privacy. Our auditing scheme requires minimal assumptions about the algorithm and can be applied in the black-box or white-box setting. We demonstrate the effectiveness of our framework by applying it to DP-SGD, where we can achieve meaningful empirical privacy lower bounds by training only one model. In contrast, standard methods would require training hundreds of models.

----

## [0] Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization

**Authors**: *Clément Bénard, Brian Staber, Sébastien Da Veiga*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html)

**Abstract**:

Stein thinning is a promising algorithm proposed by (Riabiz et al., 2022) for post-processing outputs of Markov chain Monte Carlo (MCMC). The main principle is to greedily minimize the kernelized Stein discrepancy (KSD), which only requires the gradient of the log-target distribution, and is thus well-suited for Bayesian inference. The main advantages of Stein thinning are the automatic remove of the burn-in period, the correction of the bias introduced by recent MCMC algorithms, and the asymptotic properties of convergence towards the target distribution. Nevertheless, Stein thinning suffers from several empirical pathologies, which may result in poor approximations, as observed in the literature. In this article, we conduct a theoretical analysis of these pathologies, to clearly identify the mechanisms at stake, and suggest improved strategies. Then, we introduce the regularized Stein thinning algorithm to alleviate the identified pathologies. Finally, theoretical guarantees and extensive experiments show the high efficiency of the proposed algorithm. An implementation of regularized Stein thinning as the kernax library in python and JAX is available at https://gitlab.com/drti/kernax.

----

## [0] Punctuation-level Attack: Single-shot and Single Punctuation Can Fool Text Models

**Authors**: *Wenqiang Wang, Chongyang Du, Tao Wang, Kaihao Zhang, Wenhan Luo, Lin Ma, Wei Liu, Xiaochun Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html)

**Abstract**:

The adversarial attacks have attracted increasing attention in various fields including natural language processing. The current textual attacking models primarily focus on fooling models by adding character-/word-/sentence-level perturbations, ignoring their influence on human perception. In this paper, for the first time in the community, we propose a novel mode of textual attack, punctuation-level attack. With various types of perturbations, including insertion, displacement, deletion, and replacement, the punctuation-level attack achieves promising fooling rates against SOTA models on typical textual tasks and maintains minimal influence on human perception and understanding of the text by mere perturbation of single-shot single punctuation. Furthermore, we propose a search method named Text Position Punctuation Embedding and Paraphrase (TPPEP) to accelerate the pursuit of optimal position to deploy the attack, without exhaustive search, and we present a mathematical interpretation of TPPEP. Thanks to the integrated Text Position Punctuation Embedding (TPPE), the punctuation attack can be applied at a constant cost of time. Experimental results on public datasets and SOTA models demonstrate the effectiveness of the punctuation attack and the proposed TPPE. We additionally apply the single punctuation attack to summarization, semantic-similarity-scoring, and text-to-image tasks, and achieve encouraging results.

----

## [0] Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network

**Authors**: *Fuyuan Lyu, Xing Tang, Dugang Liu, Chen Ma, Weihong Luo, Liang Chen, Xiuqiang He, Xue (Steve) Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ab8da29b1eb3bec912a06e0879065cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ab8da29b1eb3bec912a06e0879065cd-Abstract-Conference.html)

**Abstract**:

Deep sparse networks are widely investigated as a neural network architecture for prediction tasks with high-dimensional sparse features, with which feature interaction selection is a critical component. While previous methods primarily focus on how to search feature interaction in a coarse-grained space, less attention has been given to a finer granularity. In this work, we introduce a hybrid-grained feature interaction selection approach that targets both feature field and feature value for deep sparse networks. To explore such expansive space, we propose a decomposed space which is calculated on the fly. We then develop a selection algorithm called OptFeature, which efficiently selects the feature interaction from both the feature field and the feature value simultaneously. Results from experiments on three large real-world benchmark datasets demonstrate that OptFeature performs well in terms of accuracy and efficiency. Additional studies support the feasibility of our method. All source code are publicly available\footnote{https://anonymous.4open.science/r/OptFeature-Anonymous}.

----

## [0] On the Asymptotic Learning Curves of Kernel Ridge Regression under Power-law Decay

**Authors**: *Yicheng Li, Haobo Zhang, Qian Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9adc8ada9183f4b9a007a02773fd8114-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9adc8ada9183f4b9a007a02773fd8114-Abstract-Conference.html)

**Abstract**:

The widely observed 'benign overfitting phenomenon' in the neural network literature raises the challenge to the `bias-variance trade-off' doctrine in the statistical learning theory.Since the generalization ability of the 'lazy trained' over-parametrized neural network can be well approximated by that of the neural tangent kernel regression,the curve of the excess risk (namely, the learning curve) of kernel ridge regression attracts increasing attention recently.However, most recent arguments on the learning curve are heuristic and are based on the 'Gaussian design' assumption.In this paper, under mild and more realistic assumptions, we rigorously provide a full characterization of the learning curve in the asymptotic senseunder a power-law decay condition of the eigenvalues of the kernel and also the target function.The learning curve elaborates the effect and the interplay of the choice of the regularization parameter, the source condition and the noise.In particular, our results suggest that the 'benign overfitting phenomenon' exists in over-parametrized neural networks only when the noise level is small.

----

## [0] Mechanism Design for Collaborative Normal Mean Estimation

**Authors**: *Yiding Chen, Jerry Zhu, Kirthevasan Kandasamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html)

**Abstract**:

We study collaborative normal mean estimation, where $m$ strategic agents collect i.i.d samples from a normal distribution $\mathcal{N}(\mu, \sigma^2)$ at a cost. They all wish to estimate the mean $\mu$. By sharing data with each other, agents can obtain better estimates while keeping the cost of data collection small. To facilitate this collaboration, we wish to design mechanisms that encourage agents to collect a sufficient amount of data and share it truthfully, so that they are all better off than working alone. In naive mechanisms, such as simply pooling and sharing all the data, an individual agent might find it beneficial to under-collect and/or fabricate data, which can lead to poor social outcomes. We design a novel mechanism that overcomes these challenges via two key techniques: first, when sharing the others' data with an agent, the mechanism corrupts this dataset proportional to how much the data reported by the agent differs from the others; second, we design minimax optimal estimators for the corrupted dataset. Our mechanism, which is Nash incentive compatible and individually rational, achieves a social penalty (sum of all agents' estimation errors and data collection costs) that is at most a factor 2 of the global minimum. When applied to high dimensional (non-Gaussian) distributions with bounded variance, this mechanism retains these three properties, but with slightly weaker results. Finally, in two special cases where we restrict the strategy space of the agents, we design mechanisms that essentially achieve the global minimum.

----

## [0] DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

**Authors**: *Kaipeng Zheng, Huishuai Zhang, Weiran Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html)

**Abstract**:

Few-shot learning aims to adapt models trained on the base dataset to novel tasks where the categories were not seen by the model before. This often leads to a relatively concentrated distribution of feature values across channels on novel classes, posing challenges in determining channel importance for novel tasks. Standard few-shot learning methods employ geometric similarity metrics such as cosine similarity and negative Euclidean distance to gauge the semantic relatedness between two features. However, features with high geometric similarities may carry distinct semantics, especially in the context of few-shot learning. In this paper, we demonstrate that the importance ranking of feature channels is a more reliable indicator for few-shot learning than geometric similarity metrics. We observe that replacing the geometric similarity metric with Kendall’s rank correlation only during inference is able to improve the performance of few-shot learning across a wide range of methods and datasets with different domains. Furthermore, we propose a carefully designed differentiable loss for meta-training to address the non-differentiability issue of Kendall’s rank correlation. By replacing geometric similarity with differentiable Kendall’s rank correlation, our method can integrate with numerous existing few-shot approaches and is ready for integrating with future state-of-the-art methods that rely on geometric similarity metrics. Extensive experiments validate the efficacy of the rank-correlation-based approach, showcasing a significant improvement in few-shot learning.

----

## [0] High-dimensional Contextual Bandit Problem without Sparsity

**Authors**: *Junpei Komiyama, Masaaki Imaizumi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html)

**Abstract**:

In this research, we investigate the high-dimensional linear contextual bandit problem where the number of features $p$ is greater than the budget $T$, or it may even be infinite. Differing from the majority of previous works in this field, we do not impose sparsity on the regression coefficients. Instead, we rely on recent findings on overparameterized models, which enables us to analyze the performance of the minimum-norm interpolating estimator when data distributions have small effective ranks. We propose an explore-then-commit (EtC) algorithm to address this problem and examine its performance. Through our analysis, we derive the optimal rate of the ETC algorithm in terms of $T$ and show that this rate can be achieved by balancing exploration and exploitation. Moreover, we introduce an adaptive explore-then-commit (AEtC) algorithm that adaptively finds the optimal balance. We assess the performance of the proposed algorithms through a series of simulations.

----

## [0] VidChapters-7M: Video Chapters at Scale

**Authors**: *Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

----

## [0] Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach

**Authors**: *Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, Frank C. Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html)

**Abstract**:

We present a new method of training energy-based models (EBMs) for anomaly detection that leverages low-dimensional structures within data. The proposed algorithm, Manifold Projection-Diffusion Recovery (MPDR), first perturbs a data point along a low-dimensional manifold that approximates the training dataset. Then, EBM is trained to maximize the probability of recovering the original data. The training involves the generation of negative samples via MCMC, as in conventional EBM training, but from a different distribution concentrated near the manifold. The resulting near-manifold negative samples are highly informative, reflecting relevant modes of variation in data. An energy function of MPDR effectively learns accurate boundaries of the training data distribution and excels at detecting out-of-distribution samples. Experimental results show that MPDR exhibits strong performance across various anomaly detection tasks involving diverse data types, such as images, vectors, and acoustic signals.

----

## [0] Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker

**Authors**: *Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Heather Zheng, Ben Zhao, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b867f0e56c4c085ef1cfdad691db5f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b867f0e56c4c085ef1cfdad691db5f6-Abstract-Conference.html)

**Abstract**:

Finding classifiers robust to adversarial examples is critical for their safedeployment. Determining the robustness of the best possible classifier under agiven threat model for a fixed data distribution and comparing it to thatachieved by state-of-the-art training methods is thus an important diagnostictool. In this paper, we find achievable information-theoretic lower bounds onrobust loss in the presence of a test-time attacker for *multi-classclassifiers on any discrete dataset*. We provide a general framework for findingthe optimal $0-1$ loss that revolves around the construction of a conflicthypergraph from the data and adversarial constraints. The prohibitive cost ofthis formulation in practice leads us to formulate other variants of the attacker-classifiergame that more efficiently determine the range of the optimal loss. Ourvaluation shows, for the first time, an analysis of the gap to optimalrobustness for classifiers in the multi-class setting on benchmark datasets.

----

## [0] Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection

**Authors**: *Hezhe Qiao, Guansong Pang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b905031125e56a557db38dff4fa8d21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b905031125e56a557db38dff4fa8d21-Abstract-Conference.html)

**Abstract**:

We reveal a one-class homophily phenomenon, which is one prevalent property we find empirically in real-world graph anomaly detection (GAD) datasets, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. However, this anomaly-discriminative property is ignored by existing GAD methods that are typically built using a conventional anomaly detection objective, such as data reconstruction.In this work, we explore this property to introduce a novel unsupervised anomaly scoring measure for GAD -- local node affinity-- that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations.  We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors. Optimizing on the original graph structure can be biased by non-homophily edges(i.e., edges connecting normal and abnormal nodes). Thus, TAM is instead optimized on truncated graphs where non-homophily edges are removed iteratively to mitigate this bias. The learned representations result in significantly stronger local affinity for normal nodes than abnormal nodes. Extensive empirical results on 10 real-world GAD datasets show that TAM substantially outperforms seven competing models, achieving over 10% increase in AUROC/AUPRC compared to the best contenders on challenging datasets. Our code is available at https://github.com/mala-lab/TAM-master/.

----

## [0] Sample-Conditioned Hypothesis Stability Sharpens Information-Theoretic Generalization Bounds

**Authors**: *Ziqiao Wang, Yongyi Mao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b912f91a5e299472764377db6ca2431-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b912f91a5e299472764377db6ca2431-Abstract-Conference.html)

**Abstract**:

We present new information-theoretic generalization guarantees through the a novel construction of the "neighboring-hypothesis" matrix and a new family of stability notions termed sample-conditioned hypothesis (SCH) stability.  Our approach yields sharper bounds that improve upon previous information-theoretic bounds in various learning scenarios. Notably, these bounds address the limitations of existing information-theoretic bounds in the context of stochastic convex optimization (SCO) problems, as explored in the recent work by Haghifam et al. (2023).

----

## [0] Exploiting Contextual Objects and Relations for 3D Visual Grounding

**Authors**: *Li Yang, Chunfeng Yuan, Ziqi Zhang, Zhongang Qi, Yan Xu, Wei Liu, Ying Shan, Bing Li, Weiping Yang, Peng Li, Yan Wang, Weiming Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html)

**Abstract**:

3D visual grounding, the task of identifying visual objects in 3D scenes based on natural language inputs, plays a critical role in enabling machines to understand and engage with the real-world environment. However, this task is challenging due to the necessity to capture 3D contextual information to distinguish target objects from complex 3D scenes. The absence of annotations for contextual objects and relations further exacerbates the difficulties. In this paper, we propose a novel model, CORE-3DVG, to address these challenges by explicitly learning about contextual objects and relations. Our method accomplishes 3D visual grounding via three sequential modular networks, including a text-guided object detection network, a relation matching network, and a target identification network. During training, we introduce a pseudo-label self-generation strategy and a weakly-supervised method to facilitate the learning of contextual objects and relations, respectively. The proposed techniques allow the networks to focus more effectively on referred objects within 3D scenes by understanding their context better. We validate our model on the challenging Nr3D, Sr3D, and ScanRefer datasets and demonstrate state-of-the-art performance. Our code will be public at https://github.com/yangli18/CORE-3DVG.

----

## [0] Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt

**Authors**: *Yining Ma, Zhiguang Cao, Yeow Meng Chee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html)

**Abstract**:

In this paper, we present Neural k-Opt (NeuOpt), a novel learning-to-search (L2S) solver for routing problems. It learns to perform flexible k-opt exchanges based on a tailored action factorization method and a customized recurrent dual-stream decoder. As a pioneering work to circumvent the pure feasibility masking scheme and enable the autonomous exploration of both feasible and infeasible regions, we then propose the Guided Infeasible Region Exploration (GIRE) scheme, which supplements the NeuOpt policy network with feasibility-related features and leverages reward shaping to steer reinforcement learning more effectively. Additionally, we equip NeuOpt with Dynamic Data Augmentation (D2A) for more diverse searches during inference. Extensive experiments on the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that our NeuOpt not only significantly outstrips existing (masking-based) L2S solvers, but also showcases superiority over the learning-to-construct (L2C) and learning-to-predict (L2P) solvers. Notably, we offer fresh perspectives on how neural solvers can handle VRP constraints. Our code is available: https://github.com/yining043/NeuOpt.

----

## [0] Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning

**Authors**: *Hanlin Zhu, Paria Rashidinejad, Jiantao Jiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bb93a3c1a424654aaea6f5b594e94d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bb93a3c1a424654aaea6f5b594e94d5-Abstract-Conference.html)

**Abstract**:

We propose A-Crab (Actor-Critic Regularized by Average Bellman error), a new practical algorithm for offline reinforcement learning (RL) in complex environments with insufficient data coverage. Our algorithm combines the marginalized importance sampling framework with the actor-critic paradigm, where the critic returns evaluations of the actor (policy) that are pessimistic relative to the offline data and have a small average (importance-weighted) Bellman error. Compared to existing methods, our algorithm simultaneously offers a number of advantages:(1) It achieves the optimal statistical rate of $1/\sqrt{N}$---where $N$ is the size of offline dataset---in converging to the best policy covered in the offline dataset, even when combined with general function approximators.(2) It relies on a weaker \textit{average} notion of policy coverage (compared to the $\ell_\infty$ single-policy concentrability) that exploits the structure of policy visitations.(3) It outperforms the data-collection behavior policy over a wide range of specific hyperparameters. We provide both theoretical analysis and experimental results to validate the effectiveness of our proposed algorithm. The code is available at https://github.com/zhuhl98/ACrab.

----

## [0] Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration

**Authors**: *Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bbb3c3aa33616c55521e2f826c132bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bbb3c3aa33616c55521e2f826c132bd-Abstract-Conference.html)

**Abstract**:

Semi-implicit variational inference (SIVI) has been introduced to expand the analytical variational families by defining expressive semi-implicit distributions in a hierarchical manner. However, the single-layer architecture commonly used in current SIVI methods can be insufficient when the target posterior has complicated structures. In this paper, we propose hierarchical semi-implicit variational inference, called HSIVI, which generalizes SIVI to allow more expressive multi-layer construction of semi-implicit distributions. By introducing auxiliary distributions that interpolate between a simple base distribution and the target distribution, the conditional layers can be trained by progressively matching these auxiliary distributions one layer after another. Moreover, given pre-trained score networks, HSIVI can be used to accelerate the sampling process of diffusion models with the score matching objective. We show that HSIVI significantly enhances the expressiveness of SIVI on several Bayesian inference problems with complicated target distributions. When used for diffusion model acceleration, we show that HSIVI can produce high quality samples comparable to or better than the existing fast diffusion model based samplers with a small number of function evaluations on various datasets.

----

## [0] Geometry-Aware Adaptation for Pretrained Models

**Authors**: *Nicholas Roberts, Xintong Li, Dyah Adila, Sonia Cromp, Tzu-Heng Huang, Jitian Zhao, Frederic Sala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bbc8b6038603e6170e35f89e3c3e296-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bbc8b6038603e6170e35f89e3c3e296-Abstract-Conference.html)

**Abstract**:

Machine learning models---including prominent zero-shot models---are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes---or, in the case of zero-shot prediction, to improve its performance---without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping $\text{argmax}$ with the Fr√©chet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes for when it is not possible to predict the entire range of unobserved classes. Empirically, using easily-available external metrics, our proposed approach, Loki, gains up to 29.7% relative improvement over SimCLR on ImageNet and scales to hundreds of thousands of classes. When no such metric is available, Loki can use self-derived metrics from class embeddings and obtains a 10.5% improvement on pretrained zero-shot models such as CLIP.

----

## [0] JourneyDB: A Benchmark for Generative Image Understanding

**Authors**: *Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, Hongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bc59aff4685e39e1a8175d5303248a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bc59aff4685e39e1a8175d5303248a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While recent advancements in vision-language models have had a transformative impact on multi-modal comprehension, the extent to which these models possess the ability to comprehend generated images remains uncertain. Synthetic images, in comparison to real data, encompass a higher level of diversity in terms of both content and style, thereby presenting significant challenges for the models to fully grasp. In light of this challenge, we introduce a comprehensive dataset, referred to as JourneyDB, that caters to the domain of generative images within the context of multi-modal visual understanding. Our meticulously curated dataset comprises 4 million distinct and high-quality generated images, each paired with the corresponding text prompts that were employed in their creation. Furthermore, we additionally introduce an external subset with results of another 22 text-to-image generative models, which makes JourneyDB a comprehensive benchmark for evaluating the comprehension of generated images. On our dataset, we have devised four benchmarks to assess the performance of generated image comprehension in relation to both content and style interpretation. These benchmarks encompass prompt inversion, style retrieval, image captioning, and visual question answering. Lastly, we evaluate the performance of state-of-the-art multi-modal models when applied to the JourneyDB dataset, providing a comprehensive analysis of their strengths and limitations in comprehending generated content. We anticipate that the proposed dataset and benchmarks will facilitate further research in the field of generative content understanding. The dataset is publicly available at https://journeydb.github.io.

----

## [0] A fast heuristic to optimize time-space tradeoff for large models

**Authors**: *Akifumi Imanishi, Zijian Xu, Masayuki Takagi, Sixue Wang, Emilio Castillo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9be39b35906526b8d240056daac72c6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9be39b35906526b8d240056daac72c6f-Abstract-Conference.html)

**Abstract**:

Training large-scale neural networks is heavily constrained by GPU memory. In order to circumvent this limitation, gradient checkpointing, or recomputation is a powerful technique. There is active research in this area with methods such as Checkmake or Moccasin. However, both Checkmate and Moccasin rely on mixed integer linear programming or constraint programming, resulting in limited scalability due to their exponentially large search space.This paper proposes a novel algorithm for recomputation (FastSA) based on a simulated annealing heuristic that achieves comparable or even better solutions than state-of-the-art alternatives. FastSA can optimize computational graphs with thousands of nodes within 3 to 30 seconds, several orders of magnitude faster than current solutions.We applied FastSA to PyTorch models and verified its effectiveness through popular large vision and text models, including recent language models with the transformer architecture. The results demonstrate significant memory reductions by 73% with extra 18% computational overheads on average. Our experiments demonstrate the practicality and effectiveness of our recomputation algorithm, further highlighting its potential for wide application in various deep learning domains.

----

## [0] A Unified Conditional Framework for Diffusion-based Image Restoration

**Authors**: *Yi Zhang, Xiaoyu Shi, Dasong Li, Xiaogang Wang, Jian Wang, Hongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html)

**Abstract**:

Diffusion Probabilistic Models (DPMs) have recently shown remarkable performance in image generation tasks, which are capable of generating highly realistic images. When adopting DPMs for image restoration tasks, the crucial aspect lies in how to integrate the conditional information to guide the DPMs to generate accurate and natural output, which has been largely overlooked in existing works. In this paper, we present a unified conditional framework based on diffusion models for image restoration. We leverage a lightweight UNet to predict initial guidance and the diffusion model to learn the residual of the guidance. By carefully designing the basic module and integration module for the diffusion model block, we integrate the guidance and other auxiliary conditional information into every block of the diffusion model to achieve spatially-adaptive generation conditioning. To handle high-resolution images, we propose a simple yet effective inter-step patch-splitting strategy to produce arbitrary-resolution images without grid artifacts. We evaluate our conditional framework on three challenging tasks: extreme low-light denoising, deblurring, and JPEG restoration, demonstrating its significant improvements in perceptual quality and the generalization to restoration tasks. The code will be released at https://zhangyi-3.github.io/project/UCDIR/.

----

## [0] Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization

**Authors**: *Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng, Jianxin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf12308ece130daa083fb21f7faf1b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf12308ece130daa083fb21f7faf1b6-Abstract-Conference.html)

**Abstract**:

Dynamic graph neural networks (DGNNs) are increasingly pervasive in exploiting spatio-temporal patterns on dynamic graphs. However, existing works fail to generalize under distribution shifts, which are common in real-world scenarios. As the generation of dynamic graphs is heavily influenced by latent environments, investigating their impacts on the out-of-distribution (OOD) generalization is critical. However, it remains unexplored with the following two major challenges: (1) How to properly model and infer the complex environments on dynamic graphs with distribution shifts? (2) How to discover invariant patterns given inferred spatio-temporal environments? To solve these challenges, we propose a novel Environment-Aware dynamic Graph LEarning (EAGLE) framework for OOD generalization by modeling complex coupled environments and exploiting spatio-temporal invariant patterns. Specifically, we first design the environment-aware EA-DGNN to model environments by multi-channel environments disentangling. Then, we propose an environment instantiation mechanism for environment diversification with inferred distributions. Finally, we discriminate spatio-temporal invariant patterns for out-of-distribution prediction by the invariant pattern recognition mechanism and perform fine-grained causal interventions node-wisely with a mixture of instantiated environment samples. Experiments on real-world and synthetic dynamic graph datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts. To the best of our knowledge, we are the first to study OOD generalization on dynamic graphs from the environment learning perspective.

----

## [0] Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation

**Authors**: *Aniket Das, Dheeraj Nagaraj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf1962c5b65a243ee243bb03ff2c506-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf1962c5b65a243ee243bb03ff2c506-Abstract-Conference.html)

**Abstract**:

Stein Variational Gradient Descent (SVGD) is a popular particle-based variational inference algorithm with impressive empirical performance across various domains. Although the population (i.e, infinite-particle) limit dynamics of SVGD is well characterized, its behavior in the finite-particle regime is far less understood. To this end, our work introduces the notion of *virtual particles* to develop novel stochastic approximations of population-limit SVGD dynamics in the space of probability measures, that are exactly realizable using finite particles. As a result, we design two computationally efficient variants of SVGD, namely VP-SVGD and GB-SVGD, with provably fast finite-particle convergence rates. Our algorithms can be viewed as specific random-batch approximations of SVGD, which are computationally more efficient than ordinary SVGD. We show that the $n$ particles output by VP-SVGD and GB-SVGD, run for $T$ steps with batch-size $K$, are at-least as good as i.i.d samples from a distribution whose Kernel Stein Discrepancy to the target is at most $O(\tfrac{d^{1/3}}{(KT)^{1/6}})$ under standard assumptions. Our results also hold under a mild growth condition on the potential function, which is much weaker than the isoperimetric (e.g. Poincare Inequality) or information-transport conditions (e.g. Talagrand's Inequality $\mathsf{T}_1$) generally considered in prior works. As a corollary, we analyze the convergence of the empirical measure (of the particles output by VP-SVGD and GB-SVGD) to the target distribution and demonstrate a **double exponential improvement** over the best known finite-particle analysis of SVGD. Beyond this, our results present the **first known oracle complexities for this setting with polynomial dimension dependence**, thereby completely eliminating the curse of dimensionality exhibited by previously known finite-particle rates.

----

## [0] Flow Factorized Representation Learning

**Authors**: *Yue Song, Andy Keller, Nicu Sebe, Max Welling*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bfc2c20fa2f56a18397eafe1be8a50a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bfc2c20fa2f56a18397eafe1be8a50a-Abstract-Conference.html)

**Abstract**:

A prominent goal of representation learning research is to achieve representations which are factorized in a useful manner with respect to the ground truth factors of variation. The fields of disentangled and equivariant representation learning have approached this ideal from a range of complimentary perspectives; however, to date, most approaches have proven to either be ill-specified or insufficiently flexible to effectively separate all realistic factors of interest in a learned latent space. In this work, we propose an alternative viewpoint on such structured representation learning which we call Flow Factorized Representation Learning, and demonstrate it to learn both more efficient and more usefully structured representations than existing frameworks. Specifically, we introduce a generative model which specifies a distinct set of latent probability paths that define different input transformations. Each latent flow is generated by the gradient field of a learned potential following dynamic optimal transport. Our novel setup brings new understandings to both \textit{disentanglement} and \textit{equivariance}. We show that our model achieves higher likelihoods on standard representation learning benchmarks while simultaneously being closer to approximately equivariant models. Furthermore, we demonstrate that the transformations learned by our model are flexibly composable and can also extrapolate to new data, implying a degree of robustness and generalizability approaching the ultimate goal of usefully factorized representation learning.

----

## [0] Hierarchical Randomized Smoothing

**Authors**: *Yan Scholten, Jan Schuchardt, Aleksandar Bojchevski, Stephan Günnemann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c0efc0d84c263972af72bf70a2de533-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c0efc0d84c263972af72bf70a2de533-Abstract-Conference.html)

**Abstract**:

Real-world data is complex and often consists of objects that can be decomposed into multiple entities (e.g. images into pixels, graphs into interconnected nodes). Randomized smoothing is a powerful framework for making models provably robust against small changes to their inputs - by guaranteeing robustness of the majority vote when randomly adding noise before classification. Yet, certifying robustness on such complex data via randomized smoothing is challenging when adversaries do not arbitrarily perturb entire objects (e.g. images) but only a subset of their entities (e.g. pixels). As a solution, we introduce hierarchical randomized smoothing: We partially smooth objects by adding random noise only on a randomly selected subset of their entities. By adding noise in a more targeted manner than existing methods we obtain stronger robustness guarantees while maintaining high accuracy. We initialize hierarchical smoothing using different noising distributions, yielding novel robustness certificates for discrete and continuous domains. We experimentally demonstrate the importance of hierarchical smoothing in image and node classification, where it yields superior robustness-accuracy trade-offs. Overall, hierarchical smoothing is an important contribution towards models that are both - certifiably robust to perturbations and accurate.

----

## [0] BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing

**Authors**: *Subhro Roy, Samuel Thomson, Tongfei Chen, Richard Shin, Adam Pauls, Jason Eisner, Benjamin Van Durme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent work has shown that generation from a prompted or fine-tuned language model can perform well at semantic parsing when the output is constrained to be a valid semantic representation. We introduce BenchCLAMP, a Benchmark to evaluate Constrained LAnguage Model Parsing, that includes context-free grammars for seven semantic parsing datasets and two syntactic parsing datasets with varied output meaning representations, as well as a constrained decoding interface to generate only valid outputs covered by these grammars. We provide low, medium, and high resource splits for each dataset, allowing accurate comparison of various language models under different data regimes. Our benchmark supports evaluation of language models using prompt-based learning as well as fine-tuning. We benchmark seven language models, including two GPT-3 variants available only through an API. Our experiments show that encoder-decoder pretrained language models can achieve similar performance or even surpass state-of-the-art methods for both syntactic and semantic parsing when the model output is constrained to be valid.

----

## [0] Stable Nonconvex-Nonconcave Training via Linear Interpolation

**Authors**: *Thomas Pethick, Wanyun Xie, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html)

**Abstract**:

This paper presents a theoretical analysis of linear interpolation as a principled method for stabilizing (large-scale) neural network training. We argue that instabilities in the optimization process are often caused by the nonmonotonicity of the loss landscape and show how linear interpolation can help by leveraging the theory of nonexpansive operators. We construct a new optimization scheme called relaxed approximate proximal point (RAPP), which is the first 1-SCLI method to achieve last iterate convergence rates for $\rho$-comonotone problems while only requiring $\rho > -\tfrac{1}{2L}$. The construction extends to constrained and regularized settings. By replacing the inner optimizer in RAPP we rediscover the family of Lookahead algorithms for which we establish convergence in cohypomonotone problems even when the base optimizer is taken to be gradient descent ascent. The range of cohypomonotone problems in which Lookahead converges is further expanded by exploiting that Lookahead inherits the properties of the base optimizer. We corroborate the results with experiments on generative adversarial networks which demonstrates the benefits of the linear interpolation present in both RAPP and Lookahead.

----

## [0] UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

**Authors**: *Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html)

**Abstract**:

Diffusion probabilistic models (DPMs) have demonstrated a very promising ability in high-resolution image synthesis. However, sampling from a pre-trained DPM is time-consuming due to the multiple evaluations of the denoising network, making it more and more important to accelerate the sampling of DPMs. Despite recent progress in designing fast samplers, existing methods still cannot generate satisfying images in many applications where fewer steps (e.g., $<$10) are favored. In this paper, we develop a unified corrector (UniC) that can be applied after any existing DPM sampler to increase the order of accuracy without extra model evaluations, and derive a unified predictor (UniP) that supports arbitrary order as a byproduct. Combining UniP and UniC, we propose a unified predictor-corrector framework called UniPC for the fast sampling of DPMs, which has a unified analytical form for any order and can significantly improve the sampling quality over previous methods, especially in extremely few steps. We evaluate our methods through extensive experiments including both unconditional and conditional sampling using pixel-space and latent-space DPMs. Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256$\times$256 (conditional) with only 10 function evaluations. Code is available at https://github.com/wl-zhao/UniPC.

----

## [0] Coop: Memory is not a Commodity

**Authors**: *Jianhao Zhang, Shihan Ma, Peihong Liu, Jinhui Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c534edc7ac1d6438216311be6d42eb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c534edc7ac1d6438216311be6d42eb2-Abstract-Conference.html)

**Abstract**:

Tensor rematerialization allows the training of deep neural networks (DNNs) under limited memory budgets by checkpointing the models and recomputing the evicted tensors as needed. However, the existing tensor rematerialization techniques overlook the memory system in deep learning frameworks and implicitly assume that free memory blocks at different addresses are identical. Under this flawed assumption, discontiguous tensors are evicted, among which some are not used to allocate the new tensor. This leads to severe memory fragmentation and increases the cost of potential rematerializations.To address this issue, we propose to evict tensors within a sliding window to ensure all evictions are contiguous and are immediately used. Furthermore, we proposed cheap tensor partitioning and recomputable in-place to further reduce the rematerialization cost by optimizing the tensor allocation.We named our method Coop as it is a co-optimization of tensor allocation and tensor rematerialization. We evaluated Coop on eight representative DNNs. The experimental results demonstrate that Coop achieves up to $2\times$ memory saving and hugely reduces compute overhead, search latency, and memory fragmentation compared to the state-of-the-art baselines.

----

## [0] Learning with Explanation Constraints

**Authors**: *Rattana Pukdee, Dylan Sam, J. Zico Kolter, Maria-Florina Balcan, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c537882044c8b5352c363e840872ddb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c537882044c8b5352c363e840872ddb-Abstract-Conference.html)

**Abstract**:

As larger deep learning models are hard to interpret, there has been a recent focus on generating explanations of these black-box models. In contrast, we may have apriori explanations of how models should behave. In this paper, we formalize this notion as learning from explanation constraints and provide a learning theoretic framework to analyze how such explanations can improve the learning of our models.  One may naturally ask, "When would these explanations be helpful?"Our first key contribution addresses this question via a class of models that satisfies these explanation constraints in expectation over new data. We provide a characterization of the benefits of these models (in terms of the reduction of their Rademacher complexities) for a canonical class of explanations given by gradient information in the settings of both linear models and two layer neural networks. In addition, we provide an algorithmic solution for our framework, via a variational approximation that achieves better performance and satisfies these constraints more frequently, when compared to simpler augmented Lagrangian methods to incorporate these explanations. We demonstrate the benefits of our approach over a large array of synthetic and real-world experiments.

----

## [0] On the Interplay between Social Welfare and Tractability of Equilibria

**Authors**: *Ioannis Anagnostides, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c6d29852a049218d70108bbf5c48dfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c6d29852a049218d70108bbf5c48dfe-Abstract-Conference.html)

**Abstract**:

Computational tractability and social welfare (aka. efficiency) of equilibria are two fundamental but in general orthogonal considerations in algorithmic game theory. Nevertheless, we show that when (approximate) full efficiency can be guaranteed via a smoothness argument a la Roughgarden, Nash equilibria are approachable under a family of no-regret learning algorithms, thereby enabling fast and decentralized computation. We leverage this connection to obtain new convergence results in large games---wherein the number of players $n \gg 1$---under the well-documented property of full efficiency via smoothness in the limit. Surprisingly, our framework unifies equilibrium computation in disparate classes of problems including games with vanishing strategic sensitivity and two-player zero-sum games, illuminating en route an immediate but overlooked equivalence between smoothness and a well-studied condition in the optimization literature known as the Minty property. Finally, we establish that a family of no-regret dynamics attains a welfare bound that improves over the smoothness framework while at the same time guaranteeing convergence to the set of coarse correlated equilibria. We show this by employing the clairvoyant mirror descent algortihm recently introduced by Piliouras et al.

----

## [0] Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

**Authors**: *Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, Sanjay Shakkottai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c70cfa2e7d9328c649c94d50cbf8faf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c70cfa2e7d9328c649c94d50cbf8faf-Abstract-Conference.html)

**Abstract**:

We present the first framework to solve linear inverse problems leveraging pre-trained \textit{latent} diffusion models.  Previously proposed algorithms (such as DPS and DDRM) only apply to \textit{pixel-space} diffusion models.  We theoretically analyze our algorithm showing provable sample recovery in a linear model setting. The algorithmic insight obtained from our analysis extends to more general settings often considered in practice. Experimentally, we outperform previously proposed posterior sampling algorithms in a wide variety of problems including random inpainting, block inpainting, denoising, deblurring, destriping, and super-resolution.

----

## [0] Maximum State Entropy Exploration using Predecessor and Successor Representations

**Authors**: *Arnav Kumar Jain, Lucas Lehnert, Irina Rish, Glen Berseth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c7900fac04a701cbed83256b76dbaa3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c7900fac04a701cbed83256b76dbaa3-Abstract-Conference.html)

**Abstract**:

Animals have a developed ability to explore that aids them in important tasks such as locating food, exploring for shelter, and finding misplaced items. These exploration skills necessarily track where they have been so that they can plan for finding items with relative efficiency. Contemporary exploration algorithms often learn a less efficient exploration strategy because they either condition only on the current state or simply rely on making random open-loop exploratory moves. In this work, we propose $\eta\psi$-Learning, a method to learn efficient exploratory policies by conditioning on past episodic experience to make the next exploratory move. Specifically, $\eta\psi$-Learning learns an exploration policy that maximizes the entropy of the state visitation distribution of a single trajectory. Furthermore, we demonstrate how variants of the predecessor representation and successor representations can be combined to predict the state visitation entropy. Our experiments demonstrate the efficacy of $\eta\psi$-Learning to strategically explore the environment and maximize the state coverage with limited samples.

----

## [0] From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization

**Authors**: *Yang Li, Jinpei Guo, Runzhong Wang, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html)

**Abstract**:

Extensive experiments have gradually revealed the potential performance bottleneck of modeling Combinatorial Optimization (CO) solving as neural solution prediction tasks. The neural networks, in their pursuit of minimizing the average objective score across the distribution of historical problem instances, diverge from the core target of CO of seeking optimal solutions for every test instance. This calls for an effective search on each problem instance, while the model should serve to provide supporting knowledge that benefits the search. To this end, we propose T2T (Training to Testing) framework that first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. The proposed neural search paradigm consistently leverages generative modeling, specifically diffusion, for graduated solution improvement. It disrupts the local structure of the given solution by introducing noise and reconstructs a lower-cost solution guided by the optimization objective.  Experimental results on Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS) show the significant superiority of T2T, demonstrating an average performance gain of 49.15% for TSP solving and 17.27% for MIS solving compared to the previous state-of-the-art.

----

## [0] Learning Curves for Noisy Heterogeneous Feature-Subsampled Ridge Ensembles

**Authors**: *Benjamin S. Ruben, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c940ba3be5bc9020ec74279d6e37c8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c940ba3be5bc9020ec74279d6e37c8a-Abstract-Conference.html)

**Abstract**:

Feature bagging is a well-established ensembling method which aims to reduceprediction variance by combining predictions of many estimators trained on subsetsor projections of features. Here, we develop a theory of feature-bagging in noisyleast-squares ridge ensembles and simplify the resulting learning curves in the specialcase of equicorrelated data. Using analytical learning curves, we demonstratethat subsampling shifts the double-descent peak of a linear predictor. This leadsus to introduce heterogeneous feature ensembling, with estimators built on varyingnumbers of feature dimensions, as a computationally efficient method to mitigatedouble-descent. Then, we compare the performance of a feature-subsamplingensemble to a single linear predictor, describing a trade-off between noise amplificationdue to subsampling and noise reduction due to ensembling. Our qualitativeinsights carry over to linear classifiers applied to image classification tasks withrealistic datasets constructed using a state-of-the-art deep learning feature map.

----

## [0] Neural MMO 20: A Massively Multi-task Addition to Massively Multi-agent Learning

**Authors**: *Joseph Suarez, David Bloomin, Kyoung Whan Choe, Hao Xiang Li, Ryan Sullivan, Nishaanth Kanna, Daniel Scott, Rose S. Shuman, Herbie Bradley, Louis Castricato, Phillip Isola, Chenghui Yu, Yuhao Jiang, Qimai Li, Jiaxin Chen, Xiaolong Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Neural MMO 2.0 is a massively multi-agent and multi-task environment for reinforcement learning research. This version features a novel task-system that broadens the range of training settings and poses a new challenge in generalization: evaluation on and against tasks, maps, and opponents never seen during training. Maps are procedurally generated with 128 agents in the standard setting and 1-1024 supported overall. Version 2.0 is a complete rewrite of its predecessor with three-fold improved performance, effectively addressing simulation bottlenecks in online training. Enhancements to compatibility enable training with standard reinforcement learning frameworks designed for much simpler environments. Neural MMO 2.0 is free and open-source with comprehensive documentation available at neuralmmo.github.io and an active community Discord. To spark initial research on this new platform, we are concurrently running a competition at NeurIPS 2023.

----

## [0] Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models

**Authors**: *Lin Li, Jun Xiao, Guikun Chen, Jian Shao, Yueting Zhuang, Long Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ca825deb6ce588c96f880728d3b8aea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ca825deb6ce588c96f880728d3b8aea-Abstract-Conference.html)

**Abstract**:

Pretrained vision-language models, such as CLIP, have demonstrated strong generalization capabilities, making them promising tools in the realm of zero-shot visual recognition. Visual relation detection (VRD) is a typical task that identifies relationship (or interaction) types between object pairs within an image. However, naively utilizing CLIP with prevalent class-based prompts for zero-shot VRD has several weaknesses, e.g., it struggles to distinguish between different fine-grained relation types and it neglects essential spatial information of two objects. To this end, we propose a novel method for zero-shot VRD: RECODE, which solves RElation detection via COmposite DEscription prompts. Specifically, RECODE first decomposes each predicate category into subject, object, and spatial components. Then, it leverages large language models (LLMs) to generate description-based prompts (or visual cues) for each component. Different visual cues enhance the discriminability of similar relation categories from different perspectives, which significantly boosts performance in VRD. To dynamically fuse different cues, we further introduce a chain-of-thought method that prompts LLMs to generate reasonable weights for different visual cues. Extensive experiments on four VRD benchmarks have demonstrated the effectiveness and interpretability of RECODE.

----

## [0] ToolQA: A Dataset for LLM Question Answering with External Tools

**Authors**: *Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9cb2a7495900f8b602cb10159246a016-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9cb2a7495900f8b602cb10159246a016-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks, but they still suffer from challenges such as hallucination and weak numerical reasoning. To overcome these challenges, external tools can be used to enhance LLMs' question-answering abilities. However, current evaluation methods do not distinguish between questions that can be answered using LLMs' internal knowledge and those that require external information through tool use. To address this issue, we introduce a new dataset called ToolQA, which is designed to faithfully evaluate LLMs' ability to use external tools for question answering. Our development of ToolQA involved a scalable, automated process for dataset curation, along with 13 specialized tools designed for interaction with external knowledge in order to answer questions. Importantly, we strive to minimize the overlap between our benchmark data and LLMs' pre-training data, enabling a more precise evaluation of LLMs' tool-use reasoning abilities. We conducted an in-depth diagnosis of existing tool-use LLMs to highlight their strengths, weaknesses, and potential improvements. Our findings set a new benchmark for evaluating LLMs and suggest new directions for future advancements. Our data and code are freely available for the broader scientific community on GitHub.

----

## [0] BiSLS/SPS: Auto-tune Step Sizes for Stable Bi-level Optimization

**Authors**: *Chen Fan, Gaspard Choné-Ducasse, Mark Schmidt, Christos Thrampoulidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html)

**Abstract**:

The popularity of bi-level optimization (BO) in deep learning has spurred a growing interest in studying gradient-based BO algorithms.However, existing algorithms involve two coupled learning rates that can be affected by approximation errors when computing hypergradients, making careful fine-tuning necessary to ensure fast convergence. To alleviate this issue, we investigate the use of recently proposed adaptive step-size methods, namely stochastic line search (SLS) and stochastic Polyak step size (SPS), for computing both the upper and lower-level learning rates. First, we revisit the use of SLS and SPS in single-level optimization without the additional interpolation condition that is typically assumed in prior works. For such settings, we investigate new variants of SLS and SPS that improve upon existing suggestions in the literature and are simpler to implement. Importantly, these two variants can be seen as special instances of general family of methods with an envelope-type step-size. This unified envelope strategy allows for the extension of the algorithms and their convergence guarantees to BO settings. Finally, our extensive experiments demonstrate that the new algorithms, which are available in both SGD and Adam versions, can find large learning rates with minimal tuning and converge faster than corresponding vanilla SGD or Adam BO algorithms that require fine-tuning.

----

## [0] Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task

**Authors**: *Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html)

**Abstract**:

Modern generative models exhibit unprecedented capabilities to generate extremely realistic data. However, given the inherent compositionality of the real world, reliable use of these models in practical applications requires that they exhibit the capability to compose a novel set of concepts to generate outputs not seen in the training data set. Prior work demonstrates that recent diffusion models do exhibit intriguing compositional generalization abilities, but also fail unpredictably. Motivated by this, we perform a controlled study for understanding compositional generalization in conditional diffusion models in a synthetic setting, varying different attributes of the training data and measuring the model's ability to generate samples out-of-distribution. Our results show: (i) the order in which the ability to generate samples from a concept and compose them emerges is governed by the structure of the underlying data-generating process; (ii) performance on compositional tasks exhibits a sudden "emergence" due to multiplicative reliance on the performance of constituent tasks, partially explaining emergent phenomena seen in generative models; and (iii) composing concepts with lower frequency in the training data to generate out-of-distribution samples requires considerably more optimization steps compared to generating in-distribution samples. Overall, our study lays a foundation for understanding emergent capabilities and compositionality in generative models from a data-centric perspective.

----

## [0] Extracting Reward Functions from Diffusion Models

**Authors**: *Felipe Nuti, Tim Franzmeyer, João F. Henriques*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d23562fcedc078e27a3be813ff6feb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d23562fcedc078e27a3be813ff6feb5-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved remarkable results in image generation, and have similarly been used to learn high-performing policies in sequential decision-making tasks. Decision-making diffusion models can be trained on lower-quality data, and then be steered with a reward function to generate near-optimal trajectories.We consider the problem of extracting a reward function by comparing a decision-making diffusion model that models low-reward behavior and one that models high-reward behavior; a setting related to inverse reinforcement learning. We first define the notion of a \emph{relative reward function of two diffusion models} and show conditions under which it exists and is unique. We then devise a practical learning algorithm for extracting it by aligning the gradients of a reward function -- parametrized by a neural network -- to the difference in outputs of both diffusion models.Our method finds correct reward functions in navigation environments, and we demonstrate that steering the base model with the learned reward functions results in significantly increased performance in standard locomotion benchmarks.Finally, we demonstrate that our approach generalizes beyond sequential decision-making by learning a reward-like function from two large-scale image generation diffusion models. The extracted reward function successfully assigns lower rewards to harmful images.

----

## [0] Disentangling Voice and Content with Self-Supervision for Speaker Recognition

**Authors**: *Tianchi Liu, Kong Aik Lee, Qiongqiong Wang, Haizhou Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html)

**Abstract**:

For speaker recognition, it is difficult to extract an accurate  speaker representation from speech because of its mixture of speaker traits and content. This paper proposes a disentanglement framework that simultaneously models speaker traits and content variability in speech. It is realized with the use of three Gaussian inference layers, each consisting of a learnable transition model that extracts distinct speech components. Notably, a strengthened transition model is specifically designed to model complex speech dynamics. We also propose a self-supervision method to dynamically disentangle content without the use of labels other than speaker identities.  The efficacy of the proposed framework is validated via experiments conducted on the VoxCeleb and SITW datasets with 9.56\% and 8.24\% average reductions in EER and minDCF, respectively. Since neither additional model training nor data is specifically needed, it is easily applicable in practical use.

----

## [0] Automatic Integration for Spatiotemporal Neural Point Processes

**Authors**: *Zihao Zhou, Rose Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d30c2def27b5c6a5fb21a9aa5c16f8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d30c2def27b5c6a5fb21a9aa5c16f8f-Abstract-Conference.html)

**Abstract**:

Learning continuous-time point processes is essential to many discrete event forecasting tasks. However, integration poses a major challenge, particularly for spatiotemporal point processes (STPPs), as it involves calculating the likelihood through triple integrals over space and time. Existing methods for integrating STPP either assume a parametric form of the intensity function, which lacks flexibility; or approximating the intensity with Monte Carlo sampling, which introduces numerical errors. Recent work by Omi et al. proposes a dual network approach for efficient integration of flexible intensity function. However, their method only focuses on the 1D temporal point process. In this paper, we introduce a novel paradigm: Auto-STPP (Automatic Integration for Spatiotemporal Neural Point Processes) that extends the dual network approach to 3D STPP. While previous work provides a foundation, its direct extension overly restricts the intensity function and leads to computational challenges. In response, we introduce a decomposable parametrization for the integral network using ProdNet. This approach, leveraging the product of simplified univariate graphs, effectively sidesteps the computational complexities inherent in multivariate computational graphs. We prove the consistency of Auto-STPP and validate it on synthetic data and benchmark real-world datasets. Auto-STPP shows a significant advantage in recovering complex intensity functions from irregular spatiotemporal events, particularly when the intensity is sharply localized. Our code is open-source at https://github.com/Rose-STL-Lab/AutoSTPP.

----

## [0] Identifiability Guarantees for Causal Disentanglement from Soft Interventions

**Authors**: *Jiaqi Zhang, Kristjan H. Greenewald, Chandler Squires, Akash Srivastava, Karthikeyan Shanmugam, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d3a4cdf6f70559e8c6fe02170fba568-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d3a4cdf6f70559e8c6fe02170fba568-Abstract-Conference.html)

**Abstract**:

Causal disentanglement aims to uncover a representation of data using latent variables that are interrelated through a causal model. Such a representation is identifiable if the latent model that explains the data is unique. In this paper, we focus on the scenario where unpaired observational and interventional data are available, with each intervention changing the mechanism of a latent variable. When the causal variables are fully observed, statistically consistent algorithms have been developed to identify the causal model under faithfulness assumptions. We here show that identifiability can still be achieved with unobserved causal variables, given a generalized notion of faithfulness. Our results guarantee that we can recover the latent causal model up to an equivalence class and predict the effect of unseen combinations of interventions, in the limit of infinite data. We implement our causal disentanglement framework by developing an autoencoding variational Bayes algorithm and apply it to the problem of predicting combinatorial perturbation effects in genomics.

----

## [0] Equivariant Adaptation of Large Pretrained Models

**Authors**: *Arnab Kumar Mondal, Siba Smarak Panigrahi, Oumar Kaba, Sai Mudumba, Siamak Ravanbakhsh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d5856318032ef3630cb580f4e24f823-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d5856318032ef3630cb580f4e24f823-Abstract-Conference.html)

**Abstract**:

Equivariant networks are specifically designed to ensure consistent behavior with respect to a set of input transformations, leading to higher sample efficiency and more accurate and robust predictions. However, redesigning each component of prevalent deep neural network architectures to achieve chosen equivariance is a difficult problem and can result in a computationally expensive network during both training and inference. A recently proposed alternative towards equivariance that removes the architectural constraints is to use a simple canonicalization network that transforms the input to a canonical form before feeding it to an unconstrained prediction network. We show here that this approach can effectively be used to make a large pretrained network equivariant. However, we observe that the produced canonical orientations can be misaligned with those of the training distribution, hindering performance. Using dataset-dependent priors to inform the canonicalization function, we are able to make large pretrained models equivariant while maintaining their performance. This significantly improves the robustness of these models to deterministic transformations of the data, such as rotations. We believe this equivariant adaptation of large pretrained models can help their domain-specific applications with known symmetry priors.

----

## [0] HT-Step: Aligning Instructional Articles with How-To Videos

**Authors**: *Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, Lorenzo Torresani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d58d85bfc041b4f901c62ba37a3f322-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d58d85bfc041b4f901c62ba37a3f322-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce HT-Step, a large-scale dataset containing temporal annotations of instructional article steps in cooking videos. It includes 122k segment-level annotations over 20k narrated videos (approximately 2.3k hours) of the HowTo100M dataset.Each annotation provides a temporal interval, and a categorical step label from a taxonomy of 4,958 unique steps automatically mined from wikiHow articles which include rich descriptions of each step.Our dataset significantly surpasses existing labeled step datasets in terms of scale, number of tasks, and richness of natural language step descriptions. Based on these annotations, we introduce a strongly supervised benchmark for aligning instructional articles with how-to videos and present a comprehensive evaluation of baseline methods for this task.By publicly releasing these annotations and defining rigorous evaluation protocols and metrics,we hope to significantly accelerate research in the field of procedural activity understanding.

----

## [0] Provable Training for Graph Contrastive Learning

**Authors**: *Yue Yu, Xiao Wang, Mengmei Zhang, Nian Liu, Chuan Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d75de47462ffe77addaa7b985fc6d8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d75de47462ffe77addaa7b985fc6d8e-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has emerged as a popular training approach for learning node embeddings from augmented graphs without labels. Despite the key principle that maximizing the similarity between positive node pairs while minimizing it between negative node pairs is well established, some fundamental problems are still unclear. Considering the complex graph structure, are some nodes consistently well-trained and following this principle even with different graph augmentations? Or are there some nodes more likely to be untrained across graph augmentations and violate the principle? How to distinguish these nodes and further guide the training of GCL? To answer these questions, we first present experimental evidence showing that the training of GCL is indeed imbalanced across all nodes. To address this problem, we propose the metric "node compactness", which is the lower bound of how a node follows the GCL principle related to the range of augmentations. We further derive the form of node compactness theoretically through bound propagation, which can be integrated into binary cross-entropy as a regularization. To this end, we propose the PrOvable Training (POT) for GCL, which regularizes the training of GCL to encode node embeddings that follows the GCL principle better. Through extensive experiments on various benchmarks, POT consistently improves the existing GCL approaches, serving as a friendly plugin.

----

## [0] Generalized Weighted Path Consistency for Mastering Atari Games

**Authors**: *Dengwei Zhao, Shikui Tu, Lei Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d87a0c38431d0ec8d8b8ece95198c04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d87a0c38431d0ec8d8b8ece95198c04-Abstract-Conference.html)

**Abstract**:

Reinforcement learning with the help of neural-guided search consumes huge computational resources to achieve remarkable performance. Path consistency (PC), i.e., $f$ values on one optimal path should be identical, was previously imposed on MCTS by PCZero to improve the learning efficiency of AlphaZero. Not only  PCZero still lacks a theoretical support but also considers  merely board games. In this paper,  PCZero is  generalized into GW-PCZero for  real applications with non-zero immediate reward. A weighting mechanism is introduced to reduce the variance caused by scouting's uncertainty on the $f$ value estimation. For the first time, it is theoretically proved that neural-guided MCTS is guaranteed to find the optimal solution under the constraint of PC. Experiments are conducted on the Atari $100$k benchmark with $26$ games and GW-PCZero achieves $198\%$ mean human performance, higher than the state-of-the-art EfficientZero's $194\\%$, while consuming only $25\\%$ of the computational resources consumed by EfficientZero.

----

## [0] Scaling Data-Constrained Language Models

**Authors**: *Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin A. Raffel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html)

**Abstract**:

The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity, including augmenting the training dataset with code data or removing commonly used filters. Models and datasets from our 400 training runs are freely available at https://github.com/huggingface/datablations.

----

## [0] A Definition of Continual Reinforcement Learning

**Authors**: *David Abel, André Barreto, Benjamin Van Roy, Doina Precup, Hado Philip van Hasselt, Satinder Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html)

**Abstract**:

In a standard view of the reinforcement learning problem, an agent’s goal is to efficiently identify a policy that maximizes long-term reward. However, this perspective is based on a restricted view of learning as finding a solution, rather than treating learning as endless adaptation. In contrast, continual reinforcement learning refers to the setting in which the best agents never stop learning. Despite the importance of continual reinforcement learning, the community lacks a simple definition of the problem that highlights its commitments and makes its primary concepts precise and clear. To this end, this paper is dedicated to carefully defining the continual reinforcement learning problem. We formalize the notion of agents that “never stop learning” through a new mathematical language for analyzing and cataloging agents. Using this new language, we define a continual learning agent as one that can be understood as carrying out an implicit search process indefinitely, and continual reinforcement learning as the setting in which the best agents are all continual learning agents. We provide two motivating examples, illustrating that traditional views of multi-task reinforcement learning and continual supervised learning are special cases of our definition. Collectively, these definitions and perspectives formalize many intuitive concepts at the heart of learning, and open new research pathways surrounding continual learning agents.

----

## [0] A Dual-Stream Neural Network Explains the Functional Segregation of Dorsal and Ventral Visual Pathways in Human Brains

**Authors**: *Minkyu Choi, Kuan Han, Xiaokai Wang, Yizhen Zhang, Zhongming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d8ed3c9e27a9265ee60c8edba3dec1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d8ed3c9e27a9265ee60c8edba3dec1d-Abstract-Conference.html)

**Abstract**:

The human visual system uses two parallel pathways for spatial processing and object recognition. In contrast, computer vision systems tend to use a single feedforward pathway, rendering them less robust, adaptive, or efficient than human vision. To bridge this gap, we developed a dual-stream vision model inspired by the human eyes and brain. At the input level, the model samples two complementary visual patterns to mimic how the human eyes use magnocellular and parvocellular retinal ganglion cells to separate retinal inputs to the brain. At the backend, the model processes the separate input patterns through two branches of convolutional neural networks (CNN) to mimic how the human brain uses the dorsal and ventral cortical pathways for parallel visual processing. The first branch (WhereCNN) samples a global view to learn spatial attention and control eye movements. The second branch (WhatCNN) samples a local view to represent the object around the fixation. Over time, the two branches interact recurrently to build a scene representation from moving fixations. We compared this model with the human brains processing the same movie and evaluated their functional alignment by linear transformation. The WhereCNN and WhatCNN branches were found to differentially match the dorsal and ventral pathways of the visual cortex, respectively, primarily due to their different learning objectives, rather than their distinctions in retinal sampling or sensitivity to attention-driven eye movements. These model-based results lead us to speculate that the distinct responses and representations of the ventral and dorsal streams are more influenced by their distinct goals in visual attention and object recognition than by their specific bias or selectivity in retinal inputs. This dual-stream model takes a further step in brain-inspired computer vision, enabling parallel neural networks to actively explore and understand the visual surroundings.

----

## [0] When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment

**Authors**: *Tianwei Ni, Michel Ma, Benjamin Eysenbach, Pierre-Luc Bacon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) algorithms face two distinct challenges: learning effective representations of past and present observations, and determining how actions influence future returns. Both challenges involve modeling long-term dependencies. The Transformer architecture has been very successful to solve problems that involve long-term dependencies, including in the RL domain. However, the underlying reason for the strong performance of Transformer-based RL methods remains unclear: is it because they learn effective memory, or because they perform effective credit assignment? After introducing formal definitions of memory length and credit assignment length, we design simple configurable tasks to measure these distinct quantities. Our empirical results reveal that Transformers can enhance the memory capability of RL algorithms, scaling up to tasks that require memorizing observations $1500$ steps ago. However, Transformers do not improve long-term credit assignment. In summary, our results provide an explanation for the success of Transformers in RL, while also highlighting an important area for future research and benchmark design. Our code is open-sourced at https://github.com/twni2016/Memory-RL.

----

## [0] Hypothesis Selection with Memory Constraints

**Authors**: *Maryam Aliakbarpour, Mark Bun, Adam Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dd67d30e0edd53581363c1b49006e1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dd67d30e0edd53581363c1b49006e1d-Abstract-Conference.html)

**Abstract**:

Hypothesis selection is a fundamental problem in learning theory and statistics. Given a dataset and a finite set of candidate distributions, the goal is to select a distribution that matches the data as well as possible. More specifically, suppose we have sample access to an unknown distribution $P$ over a domain $\mathcal{X}$ that we know is well-approximated by one of a a class of $n$  distributions (a.k.a. hypotheses), $\mathcal{H} \coloneqq \{H_1, H_2, \ldots, H_n\}$. The goal is to design an algorithm that outputs a distribution $\hat{H} \in \mathcal{H}$ whose total variation distance from $P$ is nearly minimal.In this work, we study the hypothesis selection problem under memory constraints. We consider a model where samples from $P$ are presented in a stream and we access each sample $x$ via ``PDF-comparison'' queries that allow us to compare the probability densities of any pair of hypothesesat the domain point $x$ (i.e., is $H_i(x) < H_j(x)$?). This model allows us to study how much memory is needed at any point in time to store information about the portion of the stream seen so far.Our main result is an algorithm that achieves a nearly optimal tradeoff between memory usage and the number of samples required. In particular, given $b$ bits of memory (for $b$ roughly between $\log n$ and $n$), our algorithm solves the hypothesis selection problem with $s$ samples, where $b \cdot s = O(n \log n)$. This result is optimal up to an $O(\log n)$ factor, for all $b$.

----

## [0] Optimization or Architecture: How to Hack Kalman Filtering

**Authors**: *Ido Greenberg, Netanel Yannay, Shie Mannor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dfcc83c01e94d02c751c47517855c9f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dfcc83c01e94d02c751c47517855c9f-Abstract-Conference.html)

**Abstract**:

In non-linear filtering, it is traditional to compare non-linear architectures such as neural networks to the standard linear Kalman Filter (KF). We observe that this mixes the evaluation of two separate components: the non-linear architecture, and the parameters optimization method. In particular, the non-linear model is often optimized, whereas the reference KF model is not. We argue that both should be optimized similarly, and to that end present the Optimized KF (OKF). We demonstrate that the KF may become competitive to neural models â€“ if optimized using OKF. This implies that experimental conclusions of certain previous studies were derived from a flawed process. The advantage of OKF over the standard KF is further studied theoretically and empirically, in a variety of problems. Conveniently, OKF can replace the KF in real-world systems by merely updating the parameters.

----

## [0] Online robust non-stationary estimation

**Authors**: *Abishek Sankararaman, Balakrishnan Narayanaswamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e15d892c63903ecc278e0dd05536951-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e15d892c63903ecc278e0dd05536951-Abstract-Conference.html)

**Abstract**:

The real-time estimation of time-varying parameters from high-dimensional, heavy-tailed and corrupted data-streams is a common sub-routine in systems ranging from those for network monitoring and anomaly detection to those for traffic scheduling in data-centers. For estimation tasks that can be cast as minimizing a strongly convex loss function, we prove that an appropriately tuned version of the {\ttfamily clipped Stochastic Gradient Descent} (SGD) is simultaneously {\em(i)} adaptive to drift, {\em (ii)} robust to heavy-tailed inliers and arbitrary corruptions,  {\em(iii)} requires no distributional knowledge and {\em (iv)} can be implemented in an online streaming fashion. All prior estimation algorithms have only been proven to posses a subset of these practical desiderata. A observation we make is that, neither the $\mathcal{O}\left(\frac{1}{t}\right)$ learning rate for {\ttfamily clipped SGD} known to be optimal for strongly convex loss functions of a \emph{stationary} data-stream, nor the $\mathcal{O}(1)$ learning rate known to be optimal for being adaptive to drift in a \emph{noiseless} environment can be used. Instead, a learning rate of $T^{-\alpha}$ for $ \alpha < 1$ where $T$ is the stream-length is needed to balance adaptivity to potential drift and to combat noise. We develop a new inductive argument and combine it with a martingale concentration result to derive high-probability under \emph{any learning rate} on data-streams exhibiting \emph{arbitrary distribution shift} - a proof strategy that may be of independent interest. Further, using the classical doubling-trick, we relax the knowledge of the stream length $T$. Ours is the first online estimation algorithm that is provably robust to heavy-tails, corruptions and distribution shift simultaneously. We complement our theoretical results empirically on synthetic and real data.

----

## [0] POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images

**Authors**: *Antonín Vobecký, Oriane Siméoni, David Hurych, Spyridon Gidaris, Andrei Bursuc, Patrick Pérez, Josef Sivic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e30acdeff572463c1db9b7de59de64c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e30acdeff572463c1db9b7de59de64c-Abstract-Conference.html)

**Abstract**:

We describe an approach to predict open-vocabulary 3D semantic voxel occupancy map from input 2D images with the objective of enabling 3D grounding, segmentation and retrieval of free-form language queries. This is a challenging problem because of the 2D-3D ambiguity and the open-vocabulary nature of the target tasks, where obtaining annotated training data in 3D is difficult. The contributions of this work are three-fold. First, we design a new model architecture for open-vocabulary 3D semantic occupancy prediction.  The architecture consists of a 2D-3D encoder together with occupancy prediction and 3D-language heads. The output is a dense voxel map of 3D grounded language embeddings enabling a range of open-vocabulary tasks. Second, we develop a tri-modal self-supervised learning algorithm that leverages three modalities: (i) images, (ii) language and (iii) LiDAR point clouds, and enables training the proposed architecture using a strong pre-trained vision-language model without the need for any 3D manual language annotations. Finally, we demonstrate quantitatively the strengths of the proposed model on several open-vocabulary tasks:Zero-shot 3D semantic segmentation using existing datasets; 3D grounding and retrieval of free-form language queries, using a small dataset that we propose as an extension of nuScenes. You can find the project page here https://vobecant.github.io/POP3D.

----

## [0] Faster Relative Entropy Coding with Greedy Rejection Coding

**Authors**: *Gergely Flamich, Stratis Markou, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html)

**Abstract**:

Relative entropy coding (REC) algorithms encode a sample from a target distribution $Q$ using a proposal distribution $P$ using as few bits as possible. Unlike entropy coding, REC does not assume discrete distributions and require quantisation.As such, it can be naturally integrated into communication pipelines such as learnt compression and differentially private federated learning. Unfortunately, despite their practical benefits, REC algorithms have not seen widespread application, due to their prohibitively slow runtimes or restrictive assumptions. In this paper, we make progress towards addressing these issues. We introduce Greedy Rejection Coding (GRC), which generalises the rejection sampling-based algorithm of Harsha et al. (2007) to arbitrary probability spaces and partitioning schemes. We first show that GRC terminates almost surely and returns unbiased samples from $Q$, and then focus on two variants of GRC, namely GRCS and GRCD. We show that for continuous $Q$ and $P$ over $\mathbb{R}$ with unimodal $dQ/dP$, the expected runtime of GRCS is upper bounded by $\beta D_{KL}(Q||P) + \mathcal{O}(1)$ where $\beta \approx 4.82$, and its expected codelength is optimal. This makes GRCS the first REC algorithm with guaranteed optimal runtime for this class of distributions, up to the multiplicative constant $\beta$. This significantly improves upon the previous state-of-the-art method, A* coding (Flamich et al., 2022). Under the same assumptions, we experimentally observe and conjecture that the expected runtime and codelength of GRCD are upper bounded by $D_{KL}(Q||P) + \mathcal{O}(1)$. Finally, we evaluate GRC in a compression pipeline with variational autoencoders on MNIST, and show that a modified training objective and a codelength-compression method can further improve compression efficiency.

----

## [0] Sparse Parameterization for Epitomic Dataset Distillation

**Authors**: *Xing Wei, Anjia Cao, Funing Yang, Zhiheng Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e8889198d16fb79926e71adbe38cae4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e8889198d16fb79926e71adbe38cae4-Abstract-Conference.html)

**Abstract**:

The success of deep learning relies heavily on large and diverse datasets, but the storage, preprocessing, and training of such data present significant challenges. To address these challenges, dataset distillation techniques have been proposed to obtain smaller synthetic datasets that capture the essential information of the originals. In this paper, we introduce a Sparse Parameterization for Epitomic datasEt Distillation (SPEED) framework, which leverages the concept of dictionary learning and sparse coding to distill epitomes that represent pivotal information of the dataset. SPEED prioritizes proper parameterization of the synthetic dataset and introduces techniques to capture spatial redundancy within and between synthetic images. We propose Spatial-Agnostic Epitomic Tokens (SAETs) and Sparse Coding Matrices (SCMs) to efficiently represent and select significant features. Additionally, we build a Feature-Recurrent Network (FReeNet) to generate hierarchical features with high compression and storage efficiency. Experimental results demonstrate the superiority of SPEED in handling high-resolution datasets, achieving state-of-the-art performance on multiple benchmarks and downstream applications. Our framework is compatible with a variety of dataset matching approaches, generally enhancing their performance. This work highlights the importance of proper parameterization in epitomic dataset distillation and opens avenues for efficient representation learning. Source code is available at https://github.com/MIV-XJTU/SPEED.

----



[Go to the previous page](NIPS-2023-list2100.md)

[Go to the next page](NIPS-2023-list2102.md)

[Go to the catalog section](README.md)