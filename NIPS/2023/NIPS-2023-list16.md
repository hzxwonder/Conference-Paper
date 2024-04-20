## [0] Improving Graph Matching with Positional Reconstruction Encoder-Decoder Network

**Authors**: *Yixiao Zhou, Ruiqi Jia, Hongxiang Lin, Hefeng Quan, Yumeng Zhao, Xiaoqing Lyu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html)

**Abstract**:

Deriving from image matching and understanding, semantic keypoint matching aims at establishing correspondence between keypoint sets in images. As graphs are powerful tools to represent points and their complex relationships, graph matching provides an effective way to find desired semantic keypoint correspondences. Recent deep graph matching methods have shown excellent performance, but there is still a lack of exploration and utilization of spatial information of keypoints as nodes in graphs. More specifically, existing methods are insufficient to capture the relative spatial relations through current graph construction approaches from the locations of semantic keypoints. To address these issues, we introduce a positional reconstruction encoder-decoder (PR-EnDec) to model intrinsic graph spatial structure, and present an end-to-end graph matching network PREGM based on PR-EnDec. Our PR-EnDec consists of a positional encoder that learns effective node spatial embedding with the affine transformation invariance, and a spatial relation decoder that further utilizes the high-order spatial information by reconstructing the locational structure of graphs contained in the node coordinates. Extensive experimental results on three public keypoint matching datasets demonstrate the effectiveness of our proposed PREGM.

----

## [0] A Causal Framework for Decomposing Spurious Variations

**Authors**: *Drago Plecko, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cda6dae05ae5e42ea78be85d5a26f77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cda6dae05ae5e42ea78be85d5a26f77-Abstract-Conference.html)

**Abstract**:

One of the fundamental challenges found throughout the data sciences is to explain why things happen in specific ways, or through which mechanisms a certain variable $X$ exerts influences over another variable $Y$. In statistics and machine learning, significant efforts have been put into developing machinery to estimate correlations across variables efficiently. In causal inference, a large body of literature is concerned with the decomposition of causal effects under the rubric of mediation analysis. However, many variations are spurious in nature, including different phenomena throughout the applied sciences. Despite the statistical power to estimate correlations and the identification power to decompose causal effects, there is still little understanding of the properties of spurious associations and how they can be decomposed in terms of the underlying causal mechanisms. In this manuscript, we develop formal tools for decomposing spurious variations in both Markovian and Semi-Markovian models. We prove the first results that allow a non-parametric decomposition of spurious effects and provide sufficient conditions for the identification of such decompositions. The described approach has several applications, ranging from explainable and fair AI to questions in epidemiology and medicine, and we empirically demonstrate its use.

----

## [0] Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification

**Authors**: *Tianjun Ke, Haoqun Cao, Zenan Ling, Feng Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html)

**Abstract**:

Meta-learning has demonstrated promising results in few-shot classification (FSC) by learning to solve new problems using prior knowledge. Bayesian methods are effective at characterizing uncertainty in FSC, which is crucial in high-risk fields. In this context, the logistic-softmax likelihood is often employed as an alternative to the softmax likelihood in multi-class Gaussian process classification due to its conditional conjugacy property. However, the theoretical property of logistic-softmax is not clear and previous research indicated that the inherent uncertainty of logistic-softmax leads to suboptimal performance. To mitigate these issues, we revisit and redesign the logistic-softmax likelihood, which enables control of the \textit{a priori} confidence level through a temperature parameter. Furthermore, we theoretically and empirically show that softmax can be viewed as a special case of logistic-softmax and logistic-softmax induces a larger family of data distribution than softmax. Utilizing modified logistic-softmax, we integrate the data augmentation technique into the deep kernel based Gaussian process meta-learning framework, and derive an analytical mean-field approximation for task-specific updates. Our approach yields well-calibrated uncertainty estimates and achieves comparable or superior results on standard benchmark datasets. Code is publicly available at \url{https://github.com/keanson/revisit-logistic-softmax}.

----

## [0] Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration

**Authors**: *Haitao Lin, Yufei Huang, Odin Zhang, Yunfan Liu, Lirong Wu, Siyuan Li, Zhiyuan Chen, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html)

**Abstract**:

In recent years, AI-assisted drug design methods have been proposed to generate molecules given the pockets' structures of target proteins. Most of them are  {\em atom-level-based} methods, which consider atoms as basic components and generate atom positions and types. In this way, however, it is hard to generate realistic fragments with complicated structures. To solve this, we propose \textsc{D3FG}, a {\em functional-group-based} diffusion model for pocket-specific molecule generation and elaboration. \textsc{D3FG} decomposes molecules into two categories of components: functional groups defined as rigid bodies and linkers as mass points. And the two kinds of components can together form complicated fragments that enhance ligand-protein interactions. To be specific, in the diffusion process, \textsc{D3FG} diffuses the data distribution of the positions, orientations, and types of the components into a prior distribution; In the generative process, the noise is gradually removed from the three variables by denoisers parameterized with designed equivariant graph neural networks.  In the experiments, our method can generate molecules with more realistic 3D structures, competitive affinities toward the protein targets, and better drug properties. Besides, \textsc{D3FG} as a solution to a new task of molecule elaboration, could generate molecules with high affinities based on existing ligands and the hotspots of target proteins.

----

## [0] Approximately Equivariant Graph Networks

**Authors**: *Ningyuan Huang, Ron Levie, Soledad Villar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cde6435e111671b04f4574006cf3c47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cde6435e111671b04f4574006cf3c47-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are commonly described as being permutation equivariant with respect to node relabeling in the graph. This symmetry of GNNs is often compared to the translation equivariance of Euclidean convolution neural networks (CNNs). However, these two symmetries are fundamentally different: The translation equivariance of CNNs corresponds to symmetries of the fixed domain acting on the image signals (sometimes known as active symmetries), whereas in GNNs any permutation acts on both the graph signals and the graph domain (sometimes described as passive symmetries). In this work, we focus on the active symmetries of GNNs, by considering a learning setting where signals are supported on a fixed graph. In this case, the natural symmetries of GNNs are the automorphisms of the graph. Since real-world graphs tend to be asymmetric, we relax the notion of symmetries by formalizing approximate symmetries via graph coarsening. We present a bias-variance formula that quantifies the tradeoff between the loss in expressivity and the gain in the regularity of the learned estimator, depending on the chosen symmetry group. To illustrate our approach, we conduct extensive experiments on image inpainting, traffic flow prediction, and human pose estimation with different choices of symmetries. We show theoretically and empirically that the best generalization performance can be achieved by choosing a suitably larger group than the graph automorphism, but smaller than the permutation group.

----

## [0] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Authors**: *Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, Beidi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs), despite their recent impressive accomplishments, are notably cost-prohibitive to deploy, particularly for applications involving long-content generation, such as dialogue systems and story writing. Often, a large amount of transient state information, referred to as the $\mathsf{KV}$ $\mathsf{cache}$, is stored in GPU memory in addition to model parameters, scaling linearly with the sequence length and batch size. In this paper, we introduce a novel approach for implementing the  $\mathsf{KV}$ $\mathsf{cache}$ which significantly reduces its memory footprint.  Our approach is based on the noteworthy observation that a small portion of tokens contributes most of the value when computing attention scores.  We call these tokens Heavy Hitters ($\mathsf{H_2}$). Through a comprehensive investigation, we find that ($i$) the emergence of $\mathsf{H_2}$ is natural and strongly correlates with the frequent co-occurrence of tokens in the text, and ($ii$) removing them results in significant performance degradation. Based on these insights, we propose Heavy Hitter Oracle ($\mathsf{H_2O}$), a $\mathsf{KV}$ $\mathsf{cache}$ eviction policy that dynamically retains a balance of recent and $\mathsf{H_2}$ tokens. We formulate the $\mathsf{KV}$ $\mathsf{cache}$ eviction as a dynamic submodular problem and prove (under mild assumptions) a theoretical guarantee for our novel eviction algorithm which could help guide future work. We validate the accuracy of our algorithm with OPT, LLaMA, and GPT-NeoX across a wide range of tasks. Our implementation of $\mathsf{H_2O}$ with 20\% heavy hitters improves the throughput over three leading inference systems DeepSpeed Zero-Inference, Hugging Face Accelerate, and FlexGen by up to $29\times$, $29\times$, and $3\times$ on OPT-6.7B and OPT-30B. With the same batch size, $\mathsf{H_2O}$ can reduce the latency by up to $1.9\times$.

----

## [0] Uncovering motifs of concurrent signaling across multiple neuronal populations

**Authors**: *Evren Gokcen, Anna Jasper, Alison Xu, Adam Kohn, Christian K. Machens, Byron M. Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6cf7a37e761f55b642cf0939b4c64bb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6cf7a37e761f55b642cf0939b4c64bb8-Abstract-Conference.html)

**Abstract**:

Modern recording techniques now allow us to record from distinct neuronal populations in different brain networks. However, especially as we consider multiple (more than two) populations, new conceptual and statistical frameworks are needed to characterize the multi-dimensional, concurrent flow of signals among these populations. Here, we develop a dimensionality reduction framework that determines (1) the subset of populations described by each latent dimension, (2) the direction of signal flow among those populations, and (3) how those signals evolve over time within and across experimental trials. We illustrate these features in simulation, and further validate the method by applying it to previously studied recordings from neuronal populations in macaque visual areas V1 and V2. Then we study interactions across select laminar compartments of areas V1, V2, and V3d, recorded simultaneously with multiple Neuropixels probes. Our approach uncovered signatures of selective communication across these three areas that related to their retinotopic alignment. This work advances the study of concurrent signaling across multiple neuronal populations.

----

## [0] NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos

**Authors**: *Jinxi Li, Ziyang Song, Bo Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0942e288ce41db8d4ebd041e7d1100-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0942e288ce41db8d4ebd041e7d1100-Abstract-Conference.html)

**Abstract**:

In this paper, we aim to model 3D scene dynamics from multi-view videos. Unlike the majority of existing works which usually focus on the common task of novel view synthesis within the training time period, we propose to simultaneously learn the geometry, appearance, and physical velocity of 3D scenes only from video frames, such that multiple desirable applications can be supported, including future frame extrapolation, unsupervised 3D semantic scene decomposition, and dynamic motion transfer. Our method consists of three major components, 1) the keyframe dynamic radiance field, 2) the interframe velocity field, and 3) a joint keyframe and interframe optimization module which is the core of our framework to effectively  train both networks. To validate our method, we further introduce two dynamic 3D datasets: 1) Dynamic Object dataset, and 2) Dynamic Indoor Scene dataset. We conduct extensive experiments on multiple datasets, demonstrating the superior performance of our method over all baselines, particularly in the critical tasks of future frame extrapolation and unsupervised 3D semantic scene decomposition.

----

## [0] Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parameterized Models

**Authors**: *Leonardo Galli, Holger Rauhut, Mark Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0bf1265ea9635fb4f9d56f16d7efb2-Abstract-Conference.html)

**Abstract**:

Recent works have shown that line search methods can speed up Stochastic Gradient Descent (SGD) and Adam in modern over-parameterized settings. However, existing line searches may take steps that are smaller than necessary since they require a monotone decrease of the (mini-)batch objective function. We explore nonmonotone line search methods to relax this condition and possibly accept larger step sizes. Despite the lack of a monotonic decrease, we prove the same fast rates of convergence as in the monotone case. Our experiments show that nonmonotone methods improve the speed of convergence and generalization properties of SGD/Adam even beyond the previous monotone line searches. We propose a POlyak NOnmonotone Stochastic (PoNoS) method, obtained by combining a nonmonotone line search with a Polyak initial step size. Furthermore, we develop a new resetting technique that in the majority of the iterations reduces the amount of backtracks to zero while still maintaining a large initial step size. To the best of our knowledge, a first runtime comparison shows that the epoch-wise advantage of line-search-based methods gets reflected in the overall computational time.

----

## [0] OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

**Authors**: *Yiheng Zhu, Yang Zhan, Xuankun Huang, Yuwei Chen, Yujie Chen, Jiangwen Wei, Wei Feng, Yinzhi Zhou, Haoyuan Hu, Jieping Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d0cfc5db3feeabf6762129ba91bd3a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d0cfc5db3feeabf6762129ba91bd3a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The dramatic growth of global e-commerce has led to a surge in demand for efficient and cost-effective order fulfillment which can increase customers' service levels and sellers' competitiveness. However, managing order fulfillment is challenging due to a series of interdependent online sequential decision-making problems. To clear this hurdle, rather than solving the problems separately as attempted in some recent researches, this paper proposes a method based on multi-agent reinforcement learning to integratively solve the series of interconnected problems, encompassing order handling, packing and pickup, storage, order consolidation, and last-mile delivery. In particular, we model the integrated problem as a Markov game, wherein a team of agents learns a joint policy via interacting with a simulated environment. Since no simulated environment supporting the complete order fulfillment problem exists, we devise Order Fulfillment COoperative mUlti-agent Reinforcement learning Scalable Environment (OFCOURSE) in the OpenAI Gym style, which allows reproduction and re-utilization to build customized applications. By constructing the fulfillment system in OFCOURSE, we optimize a joint policy that solves the integrated problem, facilitating sequential order-wise operations across all fulfillment units and minimizing the total cost of fulfilling all orders within the promised time. With OFCOURSE, we also demonstrate that the joint policy learned by multi-agent reinforcement learning outperforms the combination of locally optimal policies. The source code of OFCOURSE is available at: https://github.com/GitYiheng/ofcourse.

----

## [0] On Private and Robust Bandits

**Authors**: *Yulian Wu, Xingyu Zhou, Youming Tao, Di Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html)

**Abstract**:

We study private and robust multi-armed bandits (MABs), where the agent receives Huber's contaminated heavy-tailed rewards and meanwhile needs to ensure differential privacy. We consider both the finite $k$-th raw moment and the finite $k$-th central moment settings for heavy-tailed rewards distributions with $k\ge 2$. We first present its minimax lower bound, characterizing the information-theoretic limit of regret with respect to privacy budget, contamination level, and heavy-tailedness. Then, we propose a meta-algorithm that builds on a private and robust mean estimation sub-routine \texttt{PRM} that essentially relies on reward truncation and the Laplace mechanism.  For the above two different heavy-tailed settings, we give corresponding schemes of \texttt{PRM}, which enable us to achieve nearly-optimal regrets.  Moreover, our two proposed truncation-based or histogram-based \texttt{PRM} schemes achieve the optimal trade-off between estimation accuracy, privacy and robustness. Finally, we support our theoretical results and show the effectiveness of our algorithms with experimental studies.

----

## [0] RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization

**Authors**: *Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d3040941a2d57ead4043556a70dd728-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d3040941a2d57ead4043556a70dd728-Abstract-Conference.html)

**Abstract**:

Multi-agent systems are characterized by environmental uncertainty, varying policies of agents, and partial observability, which result in significant risks. In the context of Multi-Agent Reinforcement Learning (MARL), learning coordinated and decentralized policies that are sensitive to risk is challenging. To formulate the coordination requirements in risk-sensitive MARL, we introduce the Risk-sensitive Individual-Global-Max (RIGM) principle as a generalization of the Individual-Global-Max (IGM) and Distributional IGM (DIGM) principles. This principle requires that the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy. Current MARL value factorization methods do not satisfy the RIGM principle for common risk metrics such as the Value at Risk (VaR) metric or distorted risk measurements. Therefore, we propose RiskQ to address this limitation, which models the joint return distribution by modeling quantiles of it as weighted quantile mixtures of per-agent return distribution utilities. RiskQ satisfies the RIGM principle for the VaR and distorted risk metrics. We show that RiskQ can obtain promising performance through extensive experiments. The source code of RiskQ is available in https://github.com/xmu-rl-3dv/RiskQ.

----

## [0] Learning Exponential Families from Truncated Samples

**Authors**: *Jane Lee, Andre Wibisono, Emmanouil Zampetakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d5f304fb4ed0243851e41699dca4287-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d5f304fb4ed0243851e41699dca4287-Abstract-Conference.html)

**Abstract**:

Missing data problems have many manifestations across many scientific fields. A fundamental type of missing data problem arises when samples are \textit{truncated}, i.e., samples that lie in a subset of the support are not observed. Statistical estimation from truncated samples is a classical problem in statistics which dates back to Galton, Pearson, and Fisher. A recent line of work provides the first efficient estimation algorithms for the parameters of a Gaussian distribution and for linear regression with Gaussian noise.In this paper we generalize these results to log-concave exponential families. We provide an estimation algorithm that shows that \textit{extrapolation} is possible for a much larger class of distributions while it maintains a polynomial sample and time complexity on average. Our algorithm is based on Projected Stochastic Gradient Descent and is not only applicable in a more general setting but is also simpler and more efficient than recent algorithms. Our work also has interesting implications for learning general log-concave distributions and sampling given only access to truncated data.

----

## [0] XAGen: 3D Expressive Human Avatars Generation

**Authors**: *Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Jiashi Feng, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d6f9908ea35313dd7566f5ce8c6e815-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d6f9908ea35313dd7566f5ce8c6e815-Abstract-Conference.html)

**Abstract**:

Recent advances in 3D-aware GAN models have enabled the generation of realistic and controllable human body images. However, existing methods focus on the control of major body joints, neglecting the manipulation of expressive attributes, such as facial expressions, jaw poses, hand poses, and so on. In this work, we present XAGen, the first 3D generative model for human avatars capable of expressive control over body, face, and hands. To enhance the fidelity of small-scale regions like face and hands, we devise a multi-scale and multi-part 3D representation that models fine details. Based on this representation, we propose a multi-part rendering technique that disentangles the synthesis of body, face, and hands to ease model training and enhance geometric quality. Furthermore, we design multi-part discriminators that evaluate the quality of the generated avatars with respect to their appearance and fine-grained control capabilities. Experiments show that XAGen surpasses state-of-the-art methods in terms of realism, diversity, and expressive control abilities. Code and data will be made available at https://showlab.github.io/xagen.

----

## [0] HIQL: Offline Goal-Conditioned RL with Latent States as Actions

**Authors**: *Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html)

**Abstract**:

Unsupervised pre-training has recently become the bedrock for computer vision and natural language processing. In reinforcement learning (RL), goal-conditioned RL can potentially provide an analogous self-supervised approach for making use of large quantities of unlabeled (reward-free) data. However, building effective algorithms for goal-conditioned RL that can learn directly from diverse offline data is challenging, because it is hard to accurately estimate the exact value function for faraway goals. Nonetheless, goal-reaching problems exhibit structure, such that reaching distant goals entails first passing through closer subgoals. This structure can be very useful, as assessing the quality of actions for nearby goals is typically easier than for more distant goals. Based on this idea, we propose a hierarchical algorithm for goal-conditioned RL from offline data. Using one action-free value function, we learn two policies that allow us to exploit this structure: a high-level policy that treats states as actions and predicts (a latent representation of) a subgoal and a low-level policy that predicts the action for reaching this subgoal. Through analysis and didactic examples, we show how this hierarchical decomposition makes our method robust to noise in the estimated value function. We then apply our method to offline goal-reaching benchmarks, showing that our method can solve long-horizon tasks that stymie prior methods, can scale to high-dimensional image observations, and can readily make use of action-free data. Our code is available at https://seohong.me/projects/hiql/

----

## [0] Visual Instruction Tuning

**Authors**: *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)

**Abstract**:

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks. Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.

----

## [0] A Fast and Accurate Estimator for Large Scale Linear Model via Data Averaging

**Authors**: *Rui Wang, Yanyan Ouyang, Panpan Yu, Wangli Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6de668dab370194fa304a08be5aacd85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6de668dab370194fa304a08be5aacd85-Abstract-Conference.html)

**Abstract**:

This work is concerned with the estimation problem of linear model when thesample size is extremely large and the data dimension can vary with the samplesize. In this setting, the least square estimator based on the full data is not feasiblewith limited computational resources. Many existing methods for this problem arebased on the sketching technique which uses the sketched data to perform leastsquare estimation. We derive fine-grained lower bounds of the conditional meansquared error for sketching methods. For sampling methods, our lower boundprovides an attainable optimal convergence rate. Our result implies that when thedimension is large, there is hardly a sampling method can have a faster convergencerate than the uniform sampling method. To achieve a better statistical performance,we propose a new sketching method based on data averaging. The proposedmethod reduces the original data to a few averaged observations. These averagedobservations still satisfy the linear model and are used to estimate the regressioncoefficients. The asymptotic behavior of the proposed estimation procedure isstudied. Our theoretical results show that the proposed method can achieve afaster convergence rate than the optimal convergence rate for sampling methods.Theoretical and numerical results show that the proposed estimator has goodstatistical performance as well as low computational cost.

----

## [0] Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry

**Authors**: *Bariscan Bozkurt, Cengiz Pehlevan, Alper T. Erdogan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6dea02c16a492682d66c6f626c306db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6dea02c16a492682d66c6f626c306db2-Abstract-Conference.html)

**Abstract**:

The backpropagation algorithm has experienced remarkable success in training large-scale artificial neural networks; however, its biological plausibility has been strongly criticized, and it remains an open question whether the brain employs supervised learning mechanisms akin to it. Here, we propose correlative information maximization between layer activations as an alternative normative approach to describe the signal propagation in biological neural networks in both forward and backward directions. This new framework addresses many concerns about the biological-plausibility of conventional artificial neural networks and the backpropagation algorithm. The coordinate descent-based optimization of the corresponding objective, combined with the mean square error loss function for fitting labeled supervision data, gives rise to a neural network structure that emulates a more biologically realistic network of multi-compartment pyramidal neurons with dendritic processing and lateral inhibitory neurons. Furthermore, our approach provides a natural resolution to the weight symmetry problem between forward and backward signal propagation paths, a significant critique against the plausibility of the conventional backpropagation algorithm. This is achieved by leveraging two alternative, yet equivalent forms of the correlative mutual information objective. These alternatives intrinsically lead to forward and backward prediction networks without weight symmetry issues, providing a compelling solution to this long-standing challenge.

----

## [0] What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?

**Authors**: *Fnu Suya, Xiao Zhang, Yuan Tian, David Evans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e2986deda273d8fb903342841fcc4dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e2986deda273d8fb903342841fcc4dc-Abstract-Conference.html)

**Abstract**:

We study indiscriminate poisoning for linear learners where an adversary injects a few  crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.

----

## [0] Expressive probabilistic sampling in recurrent neural networks

**Authors**: *Shirui Chen, Linxing Jiang, Rajesh P. N. Rao, Eric Shea-Brown*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e2a1a8a037f9a06004fe651054e8938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e2a1a8a037f9a06004fe651054e8938-Abstract-Conference.html)

**Abstract**:

In sampling-based Bayesian models of brain function, neural activities are assumed to be samples from probability distributions that the brain uses for probabilistic computation. However, a comprehensive understanding of how mechanistic models of neural dynamics can sample from arbitrary distributions is still lacking. We use tools from functional analysis and stochastic differential equations to explore the minimum architectural requirements for $\textit{recurrent}$ neural circuits to sample from complex distributions. We first consider the traditional sampling model consisting of a network of neurons whose outputs directly represent the samples ($\textit{sampler-only}$ network). We argue that synaptic current and firing-rate dynamics in the traditional model have limited capacity to sample from a complex probability distribution. We show that the firing rate dynamics of a recurrent neural circuit with a separate set of output units can sample from an arbitrary probability distribution. We call such circuits $\textit{reservoir-sampler networks}$ (RSNs). We propose an efficient training procedure based on denoising score matching that finds recurrent and output weights such that the RSN implements Langevin sampling. We empirically demonstrate our model's ability to sample from several complex data distributions using the proposed neural dynamics and discuss its applicability to developing the next generation of sampling-based Bayesian brain models.

----

## [0] Counting Distinct Elements Under Person-Level Differential Privacy

**Authors**: *Thomas Steinke, Alexander Knop*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e32c247076c2c0fb381e022c02d2c78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e32c247076c2c0fb381e022c02d2c78-Abstract-Conference.html)

**Abstract**:

We study the problem of counting the number of distinct elements in a dataset subject to the constraint of differential privacy. We consider the challenging setting of person-level DP (a.k.a. user-level DP) where each person may contribute an unbounded number of items and hence the sensitivity is unbounded.Our approach is to compute a bounded-sensitivity version of this query, which reduces to solving a max-flow problem. The sensitivity bound is optimized to balance the noise we must add to privatize the answer against the error of the approximation of the bounded-sensitivity query to the true number of unique elements.

----

## [0] Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks

**Authors**: *Feng Chen, Daniel Kunin, Atsushi Yamamura, Surya Ganguli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e4432b912599d11609b9cdf98c823c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e4432b912599d11609b9cdf98c823c5-Abstract-Conference.html)

**Abstract**:

In this work, we reveal a strong implicit bias of stochastic gradient descent (SGD) that drives overly expressive networks to much simpler subnetworks, thereby dramatically reducing the number of independent parameters, and improving generalization. To reveal this bias, we identify invariant sets, or subsets of parameter space that remain unmodified by SGD. We focus on two classes of invariant sets that correspond to simpler (sparse or low-rank) subnetworks and commonly appear in modern architectures. Our analysis uncovers that SGD exhibits a property of stochastic attractivity towards these simpler invariant sets. We establish a sufficient condition for stochastic attractivity based on a competition between the loss landscape's curvature around the invariant set and the noise introduced by stochastic gradients. Remarkably, we find that an increased level of noise strengthens attractivity, leading to the emergence of attractive invariant sets associated with saddle-points or local maxima of the train loss. We observe empirically the existence of attractive invariant sets in trained deep neural networks, implying that SGD dynamics often collapses to simple subnetworks with either vanishing or redundant neurons. We further demonstrate how this simplifying process of stochastic collapse benefits generalization in a linear teacher-student framework. Finally, through this analysis, we mechanistically explain why early training with large learning rates for extended periods benefits subsequent generalization.

----

## [0] Conservative State Value Estimation for Offline Reinforcement Learning

**Authors**: *Liting Chen, Jie Yan, Zhengdao Shao, Lu Wang, Qingwei Lin, Saravanakumar Rajmohan, Thomas Moscibroda, Dongmei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning faces a significant challenge of value over-estimation due to the distributional drift between the dataset and the current learned policy, leading to learning failure in practice. The common approach is to incorporate a penalty term to reward or value estimation in the Bellman iterations. Meanwhile, to avoid extrapolation on out-of-distribution (OOD) states and actions, existing methods focus on conservative Q-function estimation. In this paper, we propose Conservative State Value Estimation (CSVE), a new approach that learns conservative V-function via directly imposing penalty on OOD states. Compared to prior work, CSVE allows more effective state value estimation with conservative guarantees and further better policy optimization. Further, we apply CSVE and develop a practical actor-critic algorithm in which the critic does the conservative value estimation by additionally sampling and penalizing the states around the dataset, and the actor applies advantage weighted updates extended with state exploration to improve the policy. We evaluate in classic continual control tasks of D4RL, showing that our method performs better than the conservative Q-function learning methods and is strongly competitive among recent SOTA methods.

----

## [0] Demystifying Oversmoothing in Attention-Based Graph Neural Networks

**Authors**: *Xinyi Wu, Amir Ajorlou, Zihui Wu, Ali Jadbabaie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e4cdfdd909ea4e34bfc85a12774cba0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e4cdfdd909ea4e34bfc85a12774cba0-Abstract-Conference.html)

**Abstract**:

Oversmoothing in Graph Neural Networks (GNNs) refers to the phenomenon where increasing network depth leads to homogeneous node representations. While previous work has established that Graph Convolutional Networks (GCNs) exponentially lose expressive power, it remains controversial whether the graph attention mechanism can mitigate oversmoothing. In this work, we provide a definitive answer to this question through a rigorous mathematical analysis, by viewing attention-based GNNs as nonlinear time-varying dynamical systems and incorporating tools and techniques from the theory of products of inhomogeneous matrices and the joint spectral radius. We establish that, contrary to popular belief, the graph attention mechanism cannot prevent oversmoothing and loses expressive power exponentially. The proposed framework extends the existing results on oversmoothing for symmetric GCNs to a significantly broader class of GNN models, including random walk GCNs, Graph Attention Networks (GATs) and (graph) transformers. In particular, our analysis accounts for asymmetric, state-dependent and time-varying aggregation operators and a wide range of common nonlinear activation functions, such as ReLU, LeakyReLU, GELU and SiLU.

----

## [0] A Comprehensive Benchmark for Neural Human Radiance Fields

**Authors**: *Kenkun Liu, Derong Jin, Ailing Zeng, Xiaoguang Han, Lei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The past two years have witnessed a significant increase in interest concerning NeRF-based human body rendering. While this surge has propelled considerable advancements, it has also led to an influx of methods and datasets. This explosion complicates experimental settings and makes fair comparisons challenging. In this work, we design and execute thorough studies into unified evaluation settings and metrics to establish a fair and reasonable benchmark for human NeRF models. To reveal the effects of extant models, we benchmark them against diverse and hard scenes. Additionally, we construct a cross-subject benchmark pre-trained on large-scale datasets to assess generalizable methods. Finally, we analyze the essential components for animatability and generalizability, and make HumanNeRF from monocular videos generalizable, as the inaugural baseline. We hope these benchmarks and analyses could serve the community.

----

## [0] Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms

**Authors**: *Qian Yu, Yining Wang, Baihe Huang, Qi Lei, Jason D. Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html)

**Abstract**:

In stochastic zeroth-order optimization, a problem of practical relevance is understanding how to fully exploit the local geometry of the underlying objective function. We consider a fundamental setting in which the objective function is quadratic, and provide the first tight characterization of the optimal Hessian-dependent sample complexity. Our contribution is twofold. First, from an information-theoretic point of view, we prove tight lower bounds on Hessian-dependent complexities by introducing a concept called \emph{energy allocation}, which captures the interaction between the searching algorithm and the geometry of objective functions. A matching upper bound is obtained by solving the optimal energy spectrum. Then, algorithmically, we show the existence of a Hessian-independent algorithm that universally achieves the asymptotic optimal sample complexities for all Hessian instances. The optimal sample complexities achieved by our algorithm remain valid for heavy-tailed noise distributions, which are enabled by a truncation method.

----

## [0] Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

**Authors**: *Erin George, Michael Murray, William Swartworth, Deanna Needell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e73c39cc428c7d264d9820319f31e79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e73c39cc428c7d264d9820319f31e79-Abstract-Conference.html)

**Abstract**:

We study benign overfitting in two-layer ReLU networks trained using gradient descent and hinge loss on noisy data for binary classification. In particular, we consider linearly separable data for which a relatively small proportion of labels are corrupted or flipped. We identify conditions on the margin of the clean data that give rise to three distinct training outcomes: benign overfitting, in which zero loss is achieved and with high probability test data is classified correctly; overfitting, in which zero loss is achieved but test data is misclassified with probability lower bounded by a constant; and non-overfitting, in which clean points, but not corrupt points, achieve zero loss and again with high probability test data is classified correctly. Our analysis provides a fine-grained description of the dynamics of neurons throughout training and reveals two distinct phases: in the first phase clean points achieve close to zero loss, in the second phase clean points oscillate on the boundary of zero loss while corrupt points either converge towards zero loss or are eventually zeroed by the network. We prove these results using a combinatorial approach that involves bounding the number of clean versus corrupt updates during these phases of training.

----

## [0] Adaptive Algorithms for Relaxed Pareto Set Identification

**Authors**: *Cyrille Kone, Emilie Kaufmann, Laura Richert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e976e7930460b5c3167a104ba8cc39c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e976e7930460b5c3167a104ba8cc39c-Abstract-Conference.html)

**Abstract**:

In this paper we revisit the fixed-confidence identification of the Pareto optimal set in a multi-objective multi-armed bandit model. As the sample complexity to identify the exact Pareto set can be very large, a relaxation allowing to output some additional near-optimal arms has been studied. In this work we also tackle alternative relaxations that allow instead to identify a relevant \emph{subset} of the Pareto set. Notably, we propose a single sampling strategy, called Adaptive Pareto Exploration, that can be used in conjunction with different stopping rules to take into account different relaxations of the Pareto Set Identification problem. We analyze the sample complexity of these different combinations, quantifying in particular the reduction in sample complexity that occurs when one seeks to identify at most $k$ Pareto optimal arms. We showcase the good practical performance of Adaptive Pareto Exploration on a real-world scenario, in which we adaptively explore several vaccination strategies against Covid-19 in order to find the optimal ones when multiple immunogenicity criteria are taken into account.

----

## [0] PHOTOSWAP: Personalized Subject Swapping in Images

**Authors**: *Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, Hyunjoon Jung, Xin Eric Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6e9a0a72da9b76c3ebc8cc33ff10ac29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6e9a0a72da9b76c3ebc8cc33ff10ac29-Abstract-Conference.html)

**Abstract**:

In an era where images and visual content dominate our digital landscape, the ability to manipulate and personalize these images has become a necessity.Envision seamlessly substituting a tabby cat lounging on a sunlit window sill in a photograph with your own playful puppy, all while preserving the original charm and composition of the image. We present \emph{Photoswap}, a novel approach that enables this immersive image editing experience through personalized subject swapping in existing images.\emph{Photoswap} first learns the visual concept of the subject from reference images and then swaps it into the target image using pre-trained diffusion models in a training-free manner. We establish that a well-conceptualized visual subject can be seamlessly transferred to any image with appropriate self-attention and cross-attention manipulation, maintaining the pose of the swapped subject and the overall coherence of the image. Comprehensive experiments underscore the efficacy and controllability of \emph{Photoswap} in personalized subject swapping. Furthermore, \emph{Photoswap} significantly outperforms baseline methods in human ratings across subject swapping, background preservation, and overall quality, revealing its vast application potential, from entertainment to professional editing.

----

## [0] Simplifying Neural Network Training Under Class Imbalance

**Authors**: *Ravid Shwartz-Ziv, Micah Goldblum, Yucen Lily Li, C. Bayan Bruss, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ea69f8116b7c01e3c3e43b62e6868fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ea69f8116b7c01e3c3e43b62e6868fc-Abstract-Conference.html)

**Abstract**:

Real-world datasets are often highly class-imbalanced, which can adversely impact the performance of deep learning models. The majority of research on training neural networks under class imbalance has focused on specialized loss functions and sampling techniques. Notably, we demonstrate that simply tuning existing components of standard deep learning pipelines, such as the batch size, data augmentation, architecture size, pre-training, optimizer, and label smoothing, can achieve state-of-the-art performance without any specialized loss functions or samplers. We also provide key prescriptions and considerations for training under class imbalance, and an understanding of why imbalance methods succeed or fail.

----

## [0] Regret Minimization via Saddle Point Optimization

**Authors**: *Johannes Kirschner, Seyed Alireza Bakhtiari, Kushagra Chandak, Volodymyr Tkachuk, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6eaf8c729af4fbeb18006dc2e6a41d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6eaf8c729af4fbeb18006dc2e6a41d9b-Abstract-Conference.html)

**Abstract**:

A long line of works characterizes the sample complexity of regret minimization in sequential decision-making by min-max programs. In the corresponding saddle-point game, the min-player optimizes the sampling distribution against an adversarial max-player that chooses confusing models leading to large regret. The most recent instantiation of this idea is the decision-estimation coefficient (DEC), which was shown to provide nearly tight lower and upper bounds on the worst-case expected regret in structured bandits and reinforcement learning. By re-parametrizing the offset DEC with the confidence radius and solving the corresponding min-max program, we derive an anytime variant of the Estimation-To-Decisions algorithm (Anytime-E2D). Importantly, the algorithm optimizes the exploration-exploitation trade-off online instead of via the analysis. Our formulation leads to a practical algorithm for finite model classes and linear feedback models. We further point out connections to the information ratio, decoupling coefficient and PAC-DEC, and numerically evaluate the performance of E2D on simple examples.

----

## [0] On the Sublinear Regret of GP-UCB

**Authors**: *Justin Whitehouse, Aaditya Ramdas, Zhiwei Steven Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ec2be0bb10be9a0e5db4cc2a921f301-Abstract-Conference.html)

**Abstract**:

In the kernelized bandit problem, a learner aims to sequentially compute the optimum of a function lying in a reproducing kernel Hilbert space given only noisy evaluations at sequentially chosen points. In particular, the learner aims to minimize regret, which is a measure of the suboptimality of the choices made.Arguably the most popular algorithm is the Gaussian Process Upper Confidence Bound (GP-UCB) algorithm, which involves acting based on a simple linear estimator of the unknown function.Despite its popularity, existing analyses of GP-UCB give a suboptimal regret rate, which fails to be sublinear for many commonly used kernels such as the Matern kernel. This has led to a longstanding open question: are existing regret analyses for GP-UCB tight, or can bounds be improved by using more sophisticated analytical techniques?In this work, we resolve this open question and show that GP-UCB enjoys nearly optimal regret. In particular, our results yield sublinear regret rates for the Matern kernel, improving over the state-of-the-art analyses and partially resolving a COLT open problem posed by Vakili et al. Our improvements rely on a key technical contribution --- regularizing kernel ridge estimators in proportion to the smoothness of the underlying kernel $k$. Applying this key idea together with a largely overlooked concentration result in separable Hilbert spaces (for which we provide an independent, simplified derivation), we are able to provide a tighter analysis of the GP-UCB algorithm.

----

## [0] Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks

**Authors**: *Jun Yin, Chaozhuo Li, Hao Yan, Jianxun Lian, Senzhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ecd51685e2d765bc0ad32a2e73faf62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ecd51685e2d765bc0ad32a2e73faf62-Abstract-Conference.html)

**Abstract**:

Intrinsic interpretable graph neural networks aim to provide transparent predictions by identifying the influential fraction of the input graph that guides the model prediction, i.e., the explanatory subgraph. However, current interpretable GNNs mostly are dataset-specific and hard to generalize to different graphs. A more generalizable GNN interpretation model which can effectively distill the universal structural patterns of different graphs is until-now unexplored. Motivated by the great success of recent pre-training techniques, we for the first time propose the Pre-training Interpretable Graph Neural Network ($\pi$-GNN) to distill the universal interpretability of GNNs by pre-training over synthetic graphs with ground-truth explanations. Specifically, we introduce a structural pattern learning module to extract diverse universal structure patterns and integrate them together to comprehensively represent the graphs of different types. Next, a hypergraph refining module is proposed to identify the explanatory subgraph by incorporating the universal structure patterns with local edge interactions. Finally, the task-specific predictor is cascaded with the pre-trained $\pi$-GNN model and fine-tuned over downstream tasks. Extensive experiments demonstrate that $\pi$-GNN significantly surpasses the leading interpretable GNN baselines with up to 9.98\% interpretation improvement and 16.06\% classification accuracy improvement. Meanwhile, $\pi$-GNN pre-trained on graph classification task also achieves the top-tier interpretation performance on node classification task, which further verifies its promising generalization performance among different downstream tasks. Our code and datasets are available at https://anonymous.4open.science/r/PI-GNN-F86C

----

## [0] Quantum speedups for stochastic optimization

**Authors**: *Aaron Sidford, Chenyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ed9931d6e1fb6a85efa1b2c014a47e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ed9931d6e1fb6a85efa1b2c014a47e1-Abstract-Conference.html)

**Abstract**:

We consider the problem of minimizing a continuous function given given access to a natural quantum generalization of a stochastic gradient oracle. We provide two new methods for the special case of minimizing a Lipschitz convex function. Each method obtains a dimension versus accuracy trade-off which is provably unachievable classically and we prove that one method is asymptotically optimal in low-dimensional settings. Additionally, we provide quantum algorithms for computing a critical point of a smooth non-convex function at rates not known to be achievable classically. To obtain these results we build upon the quantum multivariate mean estimation result of Cornelissen et al. and provide a general quantum variance reduction technique of independent interest.

----

## [0] Concept Algebra for (Score-Based) Text-Controlled Generative Models

**Authors**: *Zihao Wang, Lin Gui, Jeffrey Negrea, Victor Veitch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f125214c86439d107ccb58e549e828f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f125214c86439d107ccb58e549e828f-Abstract-Conference.html)

**Abstract**:

This paper concerns the structure of learned representations in text-guided generative models, focusing on score-based models. A key property of such models is that they can compose disparate concepts in a 'disentangled' manner.This suggests these models have internal representations that encode concepts in a 'disentangled' manner. Here, we focus on the idea that concepts are encoded as subspaces of some representation space.  We formalize what this means, show there's a natural choice for the representation, and develop a simple method for identifying the part of the representation corresponding to a given concept. In particular, this allows us to manipulate the concepts expressed by the model through algebraic manipulation of the representation. We demonstrate the idea with examples using Stable Diffusion.

----

## [0] Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics

**Authors**: *Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, Nicolas Courty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)

**Abstract**:

Wasserstein distance (WD) and the associated optimal transport plan have been proven useful in many applications where probability measures are at stake. In this paper, we propose a new proxy of the squared WD, coined $\textnormal{min-SWGG}$, that is based on the transport map induced by an optimal one-dimensional projection of the two input distributions. We draw connections between  $\textnormal{min-SWGG}$, and Wasserstein generalized geodesics in which the pivot measure is supported on a line. We notably provide a new closed form for the exact Wasserstein distance in the particular case of one of the distributions supported on a line allowing us to derive a fast computational scheme that is amenable to gradient descent optimization. We show that  $\textnormal{min-SWGG}$, is an upper bound of WD and that it has a complexity similar to as Sliced-Wasserstein, with the additional feature of providing an associated transport plan. We also investigate some theoretical properties such as metricity, weak convergence, computational and topological properties. Empirical evidences support the benefits of  $\textnormal{min-SWGG}$, in various contexts, from gradient flows, shape matching and image colorization, among others.

----

## [0] Aggregating Capacity in FL through Successive Layer Training for Computationally-Constrained Devices

**Authors**: *Kilian Pfeiffer, Ramin Khalili, Jörg Henkel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f43166f50f26e8d8f3edc5545b0749f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f43166f50f26e8d8f3edc5545b0749f-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is usually performed on resource-constrained edge devices, e.g., with limited memory for the computation. If the required memory to train a model exceeds this limit, the device will be excluded from the training. This can lead to a lower accuracy as valuable data and computation resources are excluded from training, also causing bias and unfairness. The FL training process should be adjusted to such constraints. The state-of-the-art techniques propose training subsets of the FL model at constrained devices, reducing their resource requirements for training. However, these techniques largely limit the co-adaptation among parameters of the model and are highly inefficient, as we show: it is actually better to train a smaller (less accurate) model by the system where all the devices can train the model end-to-end than applying such techniques. We propose a new method that enables successive freezing and training of the parameters of the FL model at devices, reducing the trainingâ€™s resource requirements at the devices while still allowing enough co-adaptation between parameters. We show through extensive experimental evaluation that our technique greatly improves the accuracy of the trained model (by 52.4 p.p. ) compared with the state of the art, efficiently aggregating the computation capacity available on distributed devices.

----

## [0] FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations

**Authors**: *Chanakya Ekbote, Ajinkya Pankaj Deshpande, Arun Iyer, Sundararajan Sellamanickam, Ramakrishna Bairi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html)

**Abstract**:

Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe, achieves an average gain of up to 4.4\%, compared to the state-of-the-art unsupervised models, across all datasets in consideration, both homophilic and heterophilic. Our code can be found at: https://github.com/Microsoft/figure.

----

## [0] Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams

**Authors**: *Esmaeil Seraj, Jerry Xiong, Mariah Schrum, Matthew C. Gombolay*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f5288d7059cbe3f5a19dad1b3bf17e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f5288d7059cbe3f5a19dad1b3bf17e1-Abstract-Conference.html)

**Abstract**:

Extending recent advances in Learning from Demonstration (LfD) frameworks to multi-robot settings poses critical challenges such as environment non-stationarity due to partial observability which is detrimental to the applicability of existing methods. Although prior work has shown that enabling communication among agents of a robot team can alleviate such issues, creating inter-agent communication under existing Multi-Agent LfD (MA-LfD) frameworks requires the human expert to provide demonstrations for both environment actions and communication actions, which necessitates an efficient communication strategy on a known message spaces. To address this problem, we propose Mixed-Initiative Multi-Agent Apprenticeship Learning (MixTURE). MixTURE enables robot teams to learn from a human expert-generated data a preferred policy to accomplish a collaborative task, while simultaneously learning emergent inter-agent communication to enhance team coordination. The key ingredient to MixTURE's success is automatically learning a communication policy, enhanced by a mutual-information maximizing reverse model that rationalizes the underlying expert demonstrations without the need for human generated data or an auxiliary reward function. MixTURE outperforms a variety of relevant baselines on diverse data generated by human experts in complex heterogeneous domains. MixTURE is the first MA-LfD framework to enable learning multi-robot collaborative policies directly from real human data, resulting in ~44% less human workload, and ~46% higher usability score.

----

## [0] Meta-Learning Adversarial Bandit Algorithms

**Authors**: *Misha Khodak, Ilya Osadchiy, Keegan Harris, Maria-Florina Balcan, Kfir Y. Levy, Ron Meir, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f627c706a7d9961cc1ff55f37f07f97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f627c706a7d9961cc1ff55f37f07f97-Abstract-Conference.html)

**Abstract**:

We study online meta-learning with bandit feedback, with the goal of improving performance across multiple tasks if they are similar according to some natural similarity measure.  As the first to target the adversarial online-within-online partial-information setting, we design meta-algorithms that combine outer learners to simultaneously tune the initialization and other hyperparameters of an inner learner for two important cases:  multi-armed bandits (MAB) and bandit linear optimization (BLO).  For MAB, the meta-learners initialize and set hyperparameters of the Tsallis-entropy generalization of Exp3, with the task-averaged regret improving if the entropy of the optima-in-hindsight is small.  For BLO, we learn to initialize and tune online mirror descent (OMD) with self-concordant barrier regularizers, showing that task-averaged regret varies directly with an action space-dependent measure they induce.  Our guarantees rely on proving that unregularized follow-the-leader combined with two levels of low-dimensional hyperparameter tuning is enough to learn a sequence of affine functions of non-Lipschitz and sometimes non-convex Bregman divergences bounding the regret of OMD.

----

## [0] Geometric Algebra Transformer

**Authors**: *Johann Brehmer, Pim de Haan, Sönke Behrends, Taco S. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html)

**Abstract**:

Problems involving geometric data arise in physics, chemistry, robotics, computer vision, and many other fields. Such data can take numerous forms, for instance points, direction vectors, translations, or rotations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric (or Clifford) algebra, which offers an efficient 16-dimensional vector-space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a Transformer, GATr is versatile, efficient, and scalable. We demonstrate GATr in problems from n-body modeling to wall-shear-stress estimation on large arterial meshes to robotic motion planning. GATr consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability.

----

## [0] Top-Ambiguity Samples Matter: Understanding Why Deep Ensemble Works in Selective Classification

**Authors**: *Qiang Ding, Yixuan Cao, Ping Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html)

**Abstract**:

Selective classification allows a machine learning model to reject some hard inputs and thus improve the reliability of its predictions. In this area, the ensemble method is powerful in practice, but there has been no solid analysis on why the ensemble method works. Inspired by an interesting empirical result that the improvement of the ensemble largely comes from top-ambiguity samples where its member models diverge, we prove that, based on some assumptions, the ensemble has a lower selective risk than the member model for any coverage within a range. The proof is nontrivial since the selective risk is a non-convex function of the model prediction. The assumptions and the theoretical results are supported by systematic experiments on both computer vision and natural language processing tasks.

----

## [0] Unlimiformer: Long-Range Transformers with Unlimited Length Input

**Authors**: *Amanda Bertsch, Uri Alon, Graham Neubig, Matthew R. Gormley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6f9806a5adc72b5b834b27e4c7c0df9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6f9806a5adc72b5b834b27e4c7c0df9b-Abstract-Conference.html)

**Abstract**:

Since the proposal of transformers, these models have been limited to bounded input lengths, because of their need to attend to every token in the input. In this work, we propose Unlimiformer: a general approach that wraps any existing pretrained encoder-decoder transformer, and offloads the cross-attention computation to a single $k$-nearest-neighbor ($k$NN) index, while the returned $k$NN distances are the attention dot-product scores. This $k$NN index can be kept on either the GPU or CPU memory and queried in sub-linear time; this way, we can index practically unlimited input sequences, while every attention head in every decoder layer retrieves its top-$k$ keys, instead of attending to every key.  We evaluate Unlimiformer on several long-document and book-summarization benchmarks, showing that it can process even **500k** token-long inputs from the BookSum dataset, without any input truncation at test time. We demonstrate that Unlimiformer improves pretrained models such as BART and Longformer by extending them to unlimited inputs without additional learned weights and without modifying their code. Our code and models are publicly available at https://github.com/abertsch72/unlimiformer , and support LLaMA-2 as well.

----

## [0] Improving CLIP Training with Language Rewrites

**Authors**: *Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) stands as one of the most effective and scalable methods for training transferable vision models using paired image and text data. CLIP models are trained using contrastive loss, which typically relies on data augmentations to prevent overfitting and shortcuts. However, in the CLIP training paradigm, data augmentations are exclusively applied to image inputs, while language inputs remain unchanged throughout the entire training process, limiting the exposure of diverse texts to the same image. In this paper, we introduce Language augmented CLIP (LaCLIP), a simple yet highly effective approach to enhance CLIP training through language rewrites. Leveraging the in-context learning capability of large language models, we rewrite the text descriptions associated with each image. These rewritten texts exhibit diversity in sentence structure and vocabulary while preserving the original key concepts and meanings. During training, LaCLIP randomly selects either the original texts or the rewritten versions as text augmentations for each image. Extensive experiments on CC3M, CC12M, RedCaps and LAION-400M datasets show that CLIP pre-training with language rewrites significantly improves the transfer performance without computation or memory overhead during training. Specifically for ImageNet zero-shot accuracy, LaCLIP outperforms CLIP by 8.2% on CC12M and 2.4% on LAION-400M.

----

## [0] Extensible Prompts for Language Models on Zero-shot Language Style Customization

**Authors**: *Tao Ge, Jing Hu, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fcbfb3721c1781728b10c6685cc2f6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fcbfb3721c1781728b10c6685cc2f6c-Abstract-Conference.html)

**Abstract**:

We propose eXtensible Prompt (X-Prompt) for prompting a large language model (LLM) beyond natural language (NL). X-Prompt instructs an LLM with not only NL but also an extensible vocabulary of imaginary words. Registering new imaginary words allows us to instruct the LLM to comprehend concepts that are difficult to describe with NL words, thereby making a prompt more descriptive. Also, these imaginary words are designed to be out-of-distribution (OOD) robust so that they can be (re)used like NL words in various prompts, distinguishing X-Prompt from soft prompt that is for fitting in-distribution data. We propose context-augmented learning (CAL) to learn imaginary words for general usability, enabling them to work properly in OOD (unseen) prompts. We experiment X-Prompt for zero-shot language style customization as a case study. The promising results of X-Prompt demonstrate its potential to facilitate advanced interaction beyond the natural language interface, bridging the communication gap between humans and LLMs.

----

## [0] MIMEx: Intrinsic Rewards from Masked Input Modeling

**Authors**: *Toru Lin, Allan Jabri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6fe10a4c0d680609f0560920bd9ade4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6fe10a4c0d680609f0560920bd9ade4a-Abstract-Conference.html)

**Abstract**:

Exploring in environments with high-dimensional observations is hard. One promising approach for exploration is to use intrinsic rewards, which often boils down to estimating "novelty" of states, transitions, or trajectories with deep networks. Prior works have shown that conditional prediction objectives such as masked autoencoding can be seen as stochastic estimation of pseudo-likelihood. We show how this perspective naturally leads to a unified view on existing intrinsic reward approaches: they are special cases of conditional prediction, where the estimation of novelty can be seen as pseudo-likelihood estimation with different mask distributions. From this view, we propose a general framework for deriving intrinsic rewards -- Masked Input Modeling for Exploration (MIMEx) -- where the mask distribution can be flexibly tuned to control the difficulty of the underlying conditional prediction task. We demonstrate that MIMEx can achieve superior results when compared against competitive baselines on a suite of challenging sparse-reward visuomotor tasks.

----

## [0] RGMIL: Guide Your Multiple-Instance Learning Model with Regressor

**Authors**: *Zhaolong Du, Shasha Mao, Yimeng Zhang, Shuiping Gou, Licheng Jiao, Lin Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html)

**Abstract**:

In video analysis, an important challenge is insufficient annotated data due to the rare occurrence of the critical patterns, and we need to provide discriminative frame-level representation with limited annotation in some applications. Multiple Instance Learning (MIL) is suitable for this scenario. However, many MIL models paid attention to analyzing the relationships between instance representations and aggregating them, but neglecting the critical information from the MIL problem itself, which causes difficultly achieving ideal instance-level performance compared with the supervised model.To address this issue, we propose the $\textbf{\textit{Regressor-Guided MIL network} (RGMIL)}$, which effectively produces discriminative instance-level representations in a general multi-classification scenario. In the proposed method, we make full use of the $\textit{regressor}$ through our newly introduced $\textit{aggregator}$, $\textbf{\textit{Regressor-Guided Pooling} (RGP)}$. RGP focuses on simulating the correct inference process of humans while facing similar problems without introducing new parameters, and the MIL problem can be accurately described through the critical information from the $\textit{regressor}$ in our method. In experiments, RGP shows dominance on more than 20 MIL benchmark datasets, with the average bag-level classification accuracy close to 1. We also perform a series of comprehensive experiments on the MMNIST dataset. Experimental results illustrate that our $\textit{aggregator}$ outperforms existing methods under different challenging circumstances. Instance-level predictions are even possible under the guidance of RGP information table in a long sequence. RGMIL also presents comparable instance-level performance with S-O-T-A supervised models in complicated applications. Statistical results demonstrate the assumption that a MIL model can compete with a supervised model at the instance level, as long as a structure that accurately describes the MIL problem is provided. The codes are available on $\url{https://github.com/LMBDA-design/RGMIL}$.

----

## [0] Stochastic Multi-armed Bandits: Optimal Trade-off among Optimality, Consistency, and Tail Risk

**Authors**: *David Simchi-Levi, Zeyu Zheng, Feng Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ffa1f5ad26addef897dcb938e525db7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ffa1f5ad26addef897dcb938e525db7-Abstract-Conference.html)

**Abstract**:

We consider the stochastic multi-armed bandit problem and fully characterize the interplays among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to achieve the optimal regret tail risk for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, 1)$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ and instance-dependent expected regret of $\tilde O(T^\beta)$, while enjoys a probability of incurring an $\Omega(T^\delta)$ regret that decays exponentially with a polynomial $T$ term. Such decaying rate is proved to be best achievable. We also generalize our analysis to the stochastic multi-armed bandit problem with non-stationary baseline rewards, where in each time period $t$, the decision maker pulls one of $K$ arms and collects a reward which is the sum of three terms: the mean of the pulled arm, an independent noise, and a non-stationary baseline reward as a function of $t$. Our results reveal insights on the trade-off between expected regret and tail risk for both worst-case and instance-dependent scenario, indicating that more sub-optimality and inconsistency leaves space for more light-tailed risk of incurring a large regret.

----

## [0] Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

**Authors**: *Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/6ffe484a646db13891bb6435ca39d667-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/6ffe484a646db13891bb6435ca39d667-Abstract-Conference.html)

**Abstract**:

Recently, pre-trained vision-language models have been increasingly used to tackle the challenging zero-shot segmentation task. Typical solutions follow the paradigm of first generating mask proposals and then adopting CLIP to classify them. To maintain the CLIP's zero-shot transferability, previous practices favour to freeze CLIP during training. However, in the paper, we reveal that CLIP is insensitive to different mask proposals and tends to produce similar predictions for various mask proposals of the same image. This insensitivity results in numerous false positives when classifying mask proposals. This issue mainly relates to the fact that CLIP is trained with image-level supervision. To alleviate this issue, we propose a simple yet effective method, named Mask-aware Fine-tuning (MAFT). Specifically,  Image-Proposals CLIP Encoder (IP-CLIP Encoder) is proposed to handle arbitrary numbers of image and mask proposals simultaneously. Then, mask-aware loss and self-distillation loss are designed to fine-tune IP-CLIP Encoder, ensuring CLIP is responsive to different mask proposals while not sacrificing transferability. In this way, mask-aware representations can be easily learned to make the true positives stand out. Notably, our solution can seamlessly plug into most existing methods without introducing any new parameters during the fine-tuning process. We conduct extensive experiments on the popular zero-shot benchmarks. With MAFT, the performance of the state-of-the-art methods is promoted by a large margin: 50.4\% (+ 8.2\%) on COCO, 81.8\% (+ 3.2\%) on Pascal-VOC, and 8.7\% (+4.3\%) on ADE20K in terms of mIoU for unseen classes. Codes will be provided for reproducibility. Code is available at https://github.com/jiaosiyu1999/MAFT.git .

----

## [0] Understanding Multi-phase Optimization Dynamics and Rich Nonlinear Behaviors of ReLU Networks

**Authors**: *Mingze Wang, Chao Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7016d7b7b6e3c05b2128ac5b3aae492d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7016d7b7b6e3c05b2128ac5b3aae492d-Abstract-Conference.html)

**Abstract**:

The training process of ReLU neural networks often exhibits complicated nonlinear phenomena. The nonlinearity of models and non-convexity of loss pose significant challenges for theoretical analysis. Therefore, most previous theoretical works on the optimization dynamics of neural networks focus either on local analysis (like the end of training) or approximate linear models (like Neural Tangent Kernel). In this work, we conduct a complete theoretical characterization of the training process of a two-layer ReLU network trained by Gradient Flow on a linearly separable data. In this specific setting, our analysis captures the whole optimization process starting from random initialization to final convergence. Despite the relatively simple model and data that we studied, we reveal four different phases from the whole training process showing a general simplifying-to-complicating learning trend.Specific nonlinear behaviors can also be precisely identified and captured theoretically, such asinitial condensation, saddle-to-plateau dynamics, plateau escape, changes of activation patterns, learning with increasing complexity, etc.

----

## [0] RD-Suite: A Benchmark for Ranking Distillation

**Authors**: *Zhen Qin, Rolf Jagerman, Rama Kumar Pasumarthi, Honglei Zhuang, He Zhang, Aijun Bai, Kai Hui, Le Yan, Xuanhui Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/701eba0f98c6f28ffee0de5969d8d034-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/701eba0f98c6f28ffee0de5969d8d034-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The distillation of ranking models has become an important topic in both academia and industry. In recent years, several advanced methods have been proposed to tackle this problem, often leveraging ranking information from teacher rankers that is absent in traditional classification settings. To date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide range of tasks and datasets make it difficult to assess or invigorate advances in this field. This paper first examines representative prior arts on ranking distillation, and raises three questions to be answered around methodology and reproducibility. To that end, we propose a systematic and unified benchmark, Ranking Distillation Suite (RD-Suite), which is a suite of tasks with 4 large real-world datasets, encompassing two major modalities (textual and numeric) and two applications (standard distillation and distillation transfer). RD-Suite consists of benchmark results that challenge some of the common wisdom in the field, and the release of datasets with teacher scores and evaluation scripts for future research. RD-Suite paves the way towards better understanding of ranking distillation, facilities more research in this direction, and presents new challenges.

----

## [0] Gradient Descent with Linearly Correlated Noise: Theory and Applications to Differential Privacy

**Authors**: *Anastasia Koloskova, Ryan McKenna, Zachary Charles, John Keith Rush, H. Brendan McMahan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70255afc962aca0930327c090eb7d8c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70255afc962aca0930327c090eb7d8c5-Abstract-Conference.html)

**Abstract**:

We study gradient descent under linearly correlated noise. Our work is motivated by recent practical methods for optimization with differential privacy (DP), such as DP-FTRL, which achieve strong performance in settings where privacy amplification techniques are infeasible (such as in federated learning). These methods inject privacy noise through a matrix factorization mechanism, making the noise linearly correlated over iterations. We propose a simplified setting that distills key facets of these methods and isolates the impact of linearly correlated noise. We analyze the behavior of gradient descent in this setting, for both convex and non-convex functions. Our analysis is demonstrably tighter than prior work and recovers multiple important special cases exactly (including anticorrelated perturbed gradient descent). We use our results to develop new, effective matrix factorizations for differentially private optimization, and highlight the benefits of these factorizations theoretically and empirically.

----

## [0] A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions

**Authors**: *David Loiseaux, Mathieu Carrière, Andrew J. Blumberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html)

**Abstract**:

Topological data analysis (TDA) is an area of data science that focuses on using invariants from algebraic topology to provide multiscale shape descriptors for geometric data sets such as point clouds. One of the most important such descriptors is persistent homology, which encodes the change in shape as a filtration parameter changes; a typical parameter is the feature scale. For many data sets, it is useful to simultaneously vary multiple filtration parameters, for example feature scale and density. While the theoretical properties of single parameter persistent homology are well understood, less is known about the multiparameter case.  A central question is the problem of representing multiparameter persistent homology by elements of a vector space for integration with standard machine learning algorithms. Existing approaches to this problem either ignore most of the multiparameter information to reduce to the one-parameter case or are heuristic and potentially unstable in the face of noise. In this article, we introduce a new general representation framework that leverages recent results on decompositions of multiparameter persistent homology. This framework is rich in information, fast to compute, and encompasses previous approaches. Moreover, we establish theoretical stability guarantees under this framework as well as efficient algorithms for practical computation, making this framework an applicable and versatile tool for analyzing geometric and point cloud data. We validate our stability results and algorithms with numerical experiments that demonstrate statistical convergence, prediction accuracy, and fast running times on several real data sets.

----

## [0] Objaverse-XL: A Universe of 10M+ 3D Objects

**Authors**: *Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, Ali Farhadi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70364304877b5e767de4e9a2a511be0c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/70364304877b5e767de4e9a2a511be0c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Natural language processing and 2D vision models have attained remarkable proficiency on many tasks primarily by escalating the scale of training data. However, 3D vision tasks have not seen the same progress, in part due to the challenges of acquiring high-quality 3D data. In this work, we present Objaverse-XL, a dataset of over 10 million 3D objects. Our compilation comprises deduplicated 3D objects from a diverse set of sources, including manually designed objects, photogrammetry scans of landmarks and everyday items, and professional scans of historic and antique artifacts. Representing the largest scale and diversity in the realm of 3D datasets, Objaverse-XL enables significant new possibilities for 3D vision. Our experiments demonstrate the vast improvements enabled with the scale provided by Objaverse-XL. We show that by training Zero123 on novel view synthesis, utilizing over 100 million multi-view rendered images, we achieve strong zero-shot generalization abilities. We hope that releasing Objaverse-XL will enable further innovations in the field of 3D vision at scale.

----

## [0] Should We Learn Most Likely Functions or Parameters?

**Authors**: *Shikai Qiu, Tim G. J. Rudner, Sanyam Kapoor, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/703f727ec10190b2fddcf8e24f52df48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/703f727ec10190b2fddcf8e24f52df48-Abstract-Conference.html)

**Abstract**:

Standard regularized training procedures correspond to maximizing a posterior distribution over parameters, known as maximum a posteriori (MAP) estimation. However, model parameters are of interest only insomuch as they combine with the functional form of a model to provide a function that can make good predictions. Moreover, the most likely parameters under the parameter posterior do not generally correspond to the most likely function induced by the parameter posterior. In fact, we can re-parametrize a model such that any setting of parameters can maximize the parameter posterior. As an alternative, we investigate the benefits and drawbacks of directly estimating the most likely function implied by the model and the data. We show that this procedure leads to pathological solutions when using neural networks and prove conditions under which the procedure is well-behaved, as well as a scalable approximation. Under these conditions, we find that function-space MAP estimation can lead to flatter minima, better generalization, and improved robustness to overfitting.

----

## [0] Geometry-Informed Neural Operator for Large-Scale 3D PDEs

**Authors**: *Zongyi Li, Nikola B. Kovachki, Christopher B. Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Animashree Anandkumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)

**Abstract**:

We propose the geometry-informed neural operator (GINO), a highly efficient approach for learning the solution operator of large-scale partial differential equations with varying geometries. GINO uses a signed distance function (SDF) representation of the input shape and neural operators based on graph and Fourier architectures to learn the solution operator. The graph neural operator handles irregular grids and transforms them into and from regular latent grids on which Fourier neural operator can be efficiently applied. We provide an efficient implementation of GINO using an optimized hashing approach, which allows efficient learning  in a shared, compressed latent space with reduced computation and memory costs. GINO is discretization-invariant, meaning the trained model can be applied to arbitrary discretizations of the continuous domain and applies to any shape or resolution.  To empirically validate the performance of our method on large-scale simulation, we generate the industry-standard aerodynamics dataset of 3D vehicle geometries with Reynolds numbers as high as five million. For this large-scale 3D fluid simulation, numerical methods are expensive to compute surface pressure. We successfully trained GINO to predict the pressure on car surfaces using only five hundred data points. The cost-accuracy experiments show a 26,000x speed-up compared to optimized GPU-based computational fluid dynamics (CFD) simulators on computing the drag coefficient. When tested on new combinations of geometries and boundary conditions (inlet velocities), GINO obtains a one-fourth reduction in error rate compared to deep neural network approaches.

----

## [0] Differentially Private Image Classification by Learning Priors from Random Processes

**Authors**: *Xinyu Tang, Ashwinee Panda, Vikash Sehwag, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7058bc192a37f5e5a57398887b05f6f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7058bc192a37f5e5a57398887b05f6f6-Abstract-Conference.html)

**Abstract**:

In privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) performs worse than SGD due to per-sample gradient clipping and noise addition.A recent focus in private learning research is improving the performance of DP-SGD on private data by incorporating priors that are learned on real-world public data.In this work, we explore how we can improve the privacy-utility tradeoff of DP-SGD by learning priors from images generated by random processes and transferring these priors to private data. We propose DP-RandP, a three-phase approach. We attain new state-of-the-art accuracy when training from scratch on CIFAR10, CIFAR100, MedMNIST and ImageNet for a range of privacy budgets $\\varepsilon \\in [1, 8]$. In particular, we improve the previous best reported accuracy on CIFAR10 from $60.6 \\%$ to $72.3 \\%$ for $\\varepsilon=1$.

----

## [0] ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

**Authors**: *Mohammad Reza Taesiri, Giang Nguyen, Sarra Habchi, Cor-Paul Bezemer, Anh Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/706390d6f9208b03bc54f97ac3cfe99e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/706390d6f9208b03bc54f97ac3cfe99e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first zoom to the most discriminative region in the image and then extract features from there to predict image labels, discarding the rest of the image. Studying six popular networks ranging from AlexNet to CLIP, we find that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we  uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zooming, we propose a test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions.Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art (SOTA) TTA method. We introduce ImageNet-Hard, a new benchmark that challenges SOTA classifiers including large vision-language models even when optimal zooming is allowed.

----

## [0] NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA

**Authors**: *Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, Hyunwoo J. Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/707a2d58641b2192203b4bf4c532cfe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/707a2d58641b2192203b4bf4c532cfe1-Abstract-Conference.html)

**Abstract**:

Multi-hop Knowledge Graph Question Answering (KGQA) is a task that involves retrieving nodes from a knowledge graph (KG) to  answer natural language questions. Recent GNN-based approaches formulate this task as a KG path searching problem, where messages are sequentially propagated from the seed node towards the answer nodes. However, these messages are past-oriented, and they do not consider the full KG context. To make matters worse, KG nodes often represent pronoun entities and are sometimes encrypted, being uninformative in selecting between paths. To address these problems, we propose Neural Tree Search (NuTrea), a tree search-based GNN model that incorporates the broader KG context. Our model adopts a message-passing scheme that probes the unreached subtree regions to boost the past-oriented embeddings. In addition, we introduce the Relation Frequency-Inverse Entity Frequency (RF-IEF) node embedding that considers the global KG context to better characterize ambiguous KG nodes. The general effectiveness of our approach is demonstrated through experiments on three major multi-hop KGQA benchmark datasets, and our extensive analyses further validate its expressiveness and robustness. Overall, NuTrea provides a powerful means to query the KG with complex natural language questions. Code is available at https://github.com/mlvlab/NuTrea.

----

## [0] Anonymous Learning via Look-Alike Clustering: A Precise Analysis of Model Generalization

**Authors**: *Adel Javanmard, Vahab Mirrokni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70899a5d74f83317c78f1a7d413d1baa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70899a5d74f83317c78f1a7d413d1baa-Abstract-Conference.html)

**Abstract**:

While personalized recommendations systems have become increasingly popular, ensuring user data protection remains a top concern in the development of these learning systems. A common approach to enhancing privacy involves training models using anonymous data rather than individual data. In this paper, we explore a natural technique called "look-alike clustering", which involves replacing sensitive features of individuals with the cluster's average values. We provide a precise analysis of how training models using anonymous cluster centers affects their generalization capabilities. We focus on an asymptotic regime where the size of the training set grows in proportion to the features dimension. Our analysis is based on the  Convex Gaussian Minimax Theorem (CGMT) and allows us to theoretically understand the role of different model components on the generalization error. In addition, we demonstrate that in certain high-dimensional regimes, training over anonymous cluster centers acts as a regularization and improves generalization error of the trained models. Finally, we corroborate our asymptotic theory with finite-sample numerical experiments where we observe a perfect match when the sample size is only of order of a few hundreds.

----

## [0] Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

**Authors**: *Jinpeng Chen, Runmin Cong, Yuxuan Luo, Horace H. s. Ip, Sam Kwong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html)

**Abstract**:

Existing class-incremental semantic segmentation (CISS) methods mainly tackle catastrophic forgetting and background shift, but often overlook another crucial issue. In CISS, each step focuses on different foreground classes, and the training set for a single step only includes images containing pixels of the current foreground classes, excluding images without them. This leads to an overrepresentation of these foreground classes in the single-step training set, causing the classification biased towards these classes. To address this issue, we present STAR, which preserves the main characteristics of each past class by storing a compact prototype and necessary statistical data, and aligns the class distribution of single-step training samples with the complete dataset by replaying these prototypes and repeating background pixels with appropriate frequency. Compared to the previous works that replay raw images, our method saves over 100 times the storage while achieving better performance. Moreover, STAR incorporates an old-class features maintaining (OCFM) loss, keeping old-class features unchanged while preserving sufficient plasticity for learning new classes. Furthermore, a similarity-aware discriminative (SAD) loss is employed to specifically enhance the feature diversity between similar old-new class pairs. Experiments on two public datasets, Pascal VOC 2012 and ADE20K, reveal that our model surpasses all previous state-of-the-art methods.

----

## [0] Skill-it! A data-driven skills framework for understanding and training language models

**Authors**: *Mayee F. Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)

**Abstract**:

The quality of training data impacts the performance of pre-trained large language models (LMs). Given a fixed budget of tokens, we study how to best select data that leads to good downstream model performance across tasks. We develop a new framework based on a simple hypothesis: just as humans acquire interdependent skills in a deliberate order, language models also follow a natural order when learning a set of skills from their training data. If such an order exists, it can be utilized for improved understanding of LMs and for data-efficient training. Using this intuition, our framework formalizes the notion of a skill and of an ordered set of skills in terms of the associated data. First, using both synthetic and real data, we demonstrate that these ordered skill sets exist, and that their existence enables more advanced skills to be learned with less data when we train on their prerequisite skills. Second, using our proposed framework, we introduce an online data sampling algorithm, Skill-It, over mixtures of skills for both continual pre-training and fine-tuning regimes, where the objective is to efficiently learn multiple skills in the former and an individual skill in the latter. On the LEGO synthetic in the continual pre-training setting, Skill-It obtains 37.5 points higher accuracy than random sampling. On the Natural Instructions dataset in the fine-tuning setting, Skill-It reduces the validation loss on the target skill by 13.6% versus training on data associated with the target skill itself. We apply our skills framework on the RedPajama dataset to continually pre-train a 3B-parameter LM, achieving higher accuracy on the LM Evaluation Harness with 1B tokens than the baseline approach of sampling uniformly over data sources with 3B tokens.

----

## [0] Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation

**Authors**: *Stefania Ionescu, Yuhao Du, Kenneth Joseph, Ancsa Hannak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70c6d82d27cd96c501c4def4803d5782-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70c6d82d27cd96c501c4def4803d5782-Abstract-Conference.html)

**Abstract**:

Two-sided matching markets have long existed to pair agents in the absence of regulated exchanges.  A common example is school choice, where a matching mechanism uses student and school preferences to assign students to schools. In such settings, forming preferences is both difficult and critical. Prior work has suggested various prediction mechanisms that help agents make decisions about their preferences. Although often deployed together, these matching and prediction mechanisms are almost always analyzed separately. The present work shows that at the intersection of the two lies a previously unexplored type of strategic behavior: agents returning to the market (e.g., schools) can attack future predictions by interacting short-term non-optimally with their matches. Here, we first introduce this type of strategic behavior, which we call an adversarial interaction attack. Next, we construct a formal economic model that captures the feedback loop between prediction mechanisms designed to assist agents and the matching mechanism used to pair them. Finally, in a simplified setting, we prove that returning agents can benefit from using adversarial interaction attacks and gain progressively more as the trust in and accuracy of predictions increases. We also show that this attack increases inequality in the student population.

----

## [0] Sample Complexity of Forecast Aggregation

**Authors**: *Tao Lin, Yiling Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html)

**Abstract**:

We consider a Bayesian forecast aggregation model where $n$ experts, after observing private signals about an unknown binary event, report their posterior beliefs about the event to a principal, who then aggregates the reports into a single prediction for the event. The signals of the experts and the outcome of the event follow a joint distribution that is unknown to the principal, but the principal has access to i.i.d. "samples" from the distribution, where each sample is a tuple of the experts' reports (not signals) and the realization of the event. Using these samples, the principal aims to find an $\varepsilon$-approximately optimal aggregator, where optimality is measured in terms of the expected squared distance between the aggregated prediction and the realization of the event. We show that the sample complexity of this problem is at least $\tilde \Omega(m^{n-2} / \varepsilon)$ for arbitrary discrete distributions, where $m$ is the size of each expert's signal space.  This sample complexity grows exponentially in the number of experts $n$. But, if the experts' signals are independent conditioned on the realization of the event, then the sample complexity is significantly reduced, to $\tilde O(1 / \varepsilon^2)$, which does not depend on $n$. Our results can be generalized to non-binary events.  The proof of our results uses a reduction from the distribution learning problem and reveals the fact that forecast aggregation is almost as difficult as distribution learning.

----

## [0] Learning to Influence Human Behavior with Offline Reinforcement Learning

**Authors**: *Joey Hong, Sergey Levine, Anca D. Dragan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/70f286e0fc977c0a3a64ef96849c8d7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/70f286e0fc977c0a3a64ef96849c8d7d-Abstract-Conference.html)

**Abstract**:

When interacting with people, AI agents do not just influence the state of the world -- they also influence the actions people take in response to the agent, and even their underlying intentions and strategies. Accounting for and leveraging this influence has mostly been studied in settings where it is sufficient to assume that human behavior is near-optimal: competitive games, or general-sum settings like autonomous driving alongside human drivers. Instead, we focus on influence in settings where there is a need to capture human suboptimality. For instance, imagine a collaborative task in which, due either to cognitive biases or lack of information, people do not perform very well -- how could an agent influence them towards more optimal behavior? Assuming near-optimal human behavior will not work here, and so the agent needs to learn from real human data. But experimenting online with humans is potentially unsafe, and creating a high-fidelity simulator of the environment is often impractical. Hence, we  focus on learning from an offline dataset of human-human interactions. Our observation is that offline reinforcement learning (RL) can learn to effectively influence suboptimal humans by extending and combining elements of observed human-human behavior. We demonstrate that offline RL can solve two challenges with effective influence. First, we show that by learning from a dataset of suboptimal human-human interaction on a variety of tasks -- none of which contains examples of successful influence -- an agent can learn influence strategies to steer humans towards better performance even on new tasks. Second, we show that by also modeling and conditioning on human behavior, offline RL can learn to affect not just the human's actions but also their underlying strategy, and adapt to changes in their strategy.

----

## [0] Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier

**Authors**: *Yuling Yao, Justin Domke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7103cd82de95a7b30983fcf74ba499ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7103cd82de95a7b30983fcf74ba499ac-Abstract-Conference.html)

**Abstract**:

To check the accuracy of Bayesian computations, it is common to use rank-based simulation-based calibration (SBC). However, SBC has drawbacks: The test statistic is somewhat ad-hoc, interactions are difficult to examine, multiple testing is a challenge, and the resulting p-value is not a divergence metric. We propose to replace the marginal rank test with a flexible classification approach that learns test statistics from data. This measure typically has a higher statistical power than the SBC test and returns an interpretable divergence measure of miscalibration, computed from classification accuracy. This approach can be used with different data generating processes to address simulation-based inference or traditional inference methods like Markov chain Monte Carlo or variational inference. We illustrate an automated implementation using neural networks and statistically-inspired features, and validate the method with numerical and real data experiments.

----

## [0] Epidemic Learning: Boosting Decentralized Learning with Randomized Communication

**Authors**: *Martijn de Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html)

**Abstract**:

We present Epidemic Learning (EL), a simple yet powerful decentralized learning (DL) algorithm that leverages changing communication topologies to achieve faster model convergence compared to conventional DL approaches. At each round of EL, each node sends its model updates to a random sample of $s$ other nodes (in a system of $n$ nodes). We provide an extensive theoretical analysis of EL, demonstrating that its changing topology culminates in superior convergence properties compared to the state-of-the-art (static and dynamic) topologies. Considering smooth non-convex loss functions, the number of transient iterations for EL, i.e., the rounds required to achieve asymptotic linear speedup, is in $O(n^3/s^2)$ which outperforms the best-known bound $O(n^3)$ by a factor of $s^2$, indicating the benefit of randomized communication for DL. We empirically evaluate EL in a 96-node network and compare its performance with state-of-the-art DL approaches. Our results illustrate that EL converges up to  $ 1.7\times$ quicker than baseline DL algorithms and attains $2.2 $\% higher accuracy for the same communication volume.

----

## [0] Global Identifiability of 𝓁1-based Dictionary Learning via Matrix Volume Optimization

**Authors**: *Jingzhou Hu, Kejun Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/717b9fd2ede6b8a9971a296d5179df89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/717b9fd2ede6b8a9971a296d5179df89-Abstract-Conference.html)

**Abstract**:

We propose a novel formulation for dictionary learning that minimizes the determinant of the dictionary matrix, also known as its volume, subject to the constraint that each row of the sparse coefficient matrix has unit $\ell_1$ norm. The main motivation for the proposed formulation is that it provides global identifiability guarantee of the groundtruth dictionary and sparse coefficient matrices, up to the inherent and inconsequential permutation and scaling ambiguity, if a set of vectors obtained from the coefficient matrix lies inside the $\ell_\infty$ norm ball but contains the $\ell_2$ norm ball in their convex hull. Unlike existing work on identifiability of dictionary learning, our result is global, meaning that a globally optimal solution to our proposed formulation has to be a permuted and rescaled version of the groundtruth factors. Another major improvement in our result is that there is no additional assumption on the dictionary matrix other than it is nonsingular, unlike most other work that require the atoms of the dictionary to be mutually incoherent. We also provide a probabilistic analysis and show that if the sparse coefficient matrix is generated from the widely adopted Bernoulli-Gaussian model, then it is globally identifiable if the sample size is bigger than a constant times $k\log k$, where $k$ is the number atoms in the dictionary, with overwhelming probability. The bound is essentially the same as those local identifiability results, but we show that it is also global. Finally, we propose algorithms to solve the new proposed formulation, specifically one based on the linearized-ADMM with efficient per-iteration updates. The proposed algorithms exhibit surprisingly effective performance in correctly and efficiently recovering the dictionary, as demonstrated in the numerical experiments.

----

## [0] Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization

**Authors**: *Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7183f4fc87598f6c6e947b96714acbd6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7183f4fc87598f6c6e947b96714acbd6-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) face the challenges in fine-tuning and deployment due to their high memory demands and computational costs. While parameter-efficient fine-tuning (PEFT) methods aim to reduce the memory usage of the optimizer state during fine-tuning, the inherent size of pre-trained LLM weights continues to be a pressing concern.  Even though quantization techniques are widely proposed to ease memory demands and accelerate LLM inference, most of these techniques are geared towards the deployment phase.To bridge this gap, this paper presents Parameter-Efficient and Quantization-aware Adaptation (PEQA) â€“ a simple yet effective method that combines the advantages of PEFT with quantized LLMs. By updating solely the quantization scales, PEQA can be directly applied to quantized LLMs, ensuring seamless task transitions. Parallel to existing PEFT methods, PEQA significantly reduces the memory overhead associated with the optimizer state. Furthermore, it leverages the advantages of quantization to substantially reduce model sizes. Even after fine-tuning, the quantization structure of a PEQA-tuned LLM remains intact, allowing for accelerated inference on the deployment stage.We employ PEQA-tuning for task-specific adaptation on LLMs with up to $65$ billion parameters. To assess the logical reasoning and language comprehension of PEQA-tuned LLMs, we fine-tune low-bit quantized LLMs using a instruction dataset. Our results show that even when LLMs are quantized to below 4-bit precision, their capabilities in language modeling, few-shot in-context learning, and comprehension can be resiliently restored to (or even improved over) their full-precision original performances with PEQA.

----

## [0] Corruption-Robust Offline Reinforcement Learning with General Function Approximation

**Authors**: *Chenlu Ye, Rui Yang, Quanquan Gu, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71b52a5b3fe2e9303433a174b60e160d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71b52a5b3fe2e9303433a174b60e160d-Abstract-Conference.html)

**Abstract**:

We investigate the problem of corruption robustness in offline reinforcement learning (RL) with general function approximation, where an adversary can corrupt each sample in the offline dataset, and the corruption level $\zeta\geq0$ quantifies the cumulative corruption amount over $n$ episodes and $H$ steps. Our goal is to find a policy that is robust to such corruption and minimizes the suboptimality gap with respect to the optimal policy for the uncorrupted Markov decision processes (MDPs). Drawing inspiration from the uncertainty-weighting technique from the robust online RL setting \citep{he2022nearly,ye2022corruptionrobust}, we design a new uncertainty weight iteration procedure to efficiently compute on batched samples and propose a corruption-robust algorithm for offline RL. Notably, under the assumption of single policy coverage and the knowledge of $\zeta$, our proposed algorithm achieves a suboptimality bound that is worsened by an additive factor of $\mathcal O(\zeta \cdot (\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H))^{1/2} (C(\hat{\mathcal F},\mu))^{-1/2} n^{-1})$ due to the corruption. Here $\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H)$ is the coverage coefficient that depends on the regularization parameter $\lambda$, the confidence set $\hat{\mathcal F}$, and the dataset $\mathcal Z_n^H$, and $C(\hat{\mathcal F},\mu)$ is a coefficient that depends on $\hat{\mathcal F}$ and the underlying data distribution $\mu$. When specialized to linear MDPs, the corruption-dependent error term reduces to $\mathcal O(\zeta d n^{-1})$ with $d$ being the dimension of the feature map, which matches the existing lower bound for corrupted linear MDPs. This suggests that our analysis is tight in terms of the corruption-dependent term.

----

## [0] Training Fully Connected Neural Networks is ∃R-Complete

**Authors**: *Daniel Bertschinger, Christoph Hertrich, Paul Jungeblut, Tillmann Miltzow, Simon Weber*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html)

**Abstract**:

We consider the algorithmic problem of finding the optimal weights and biases for a two-layer fully connected neural network to fit a given set of data points, also known as empirical risk minimization. We show that the problem is $\exists\mathbb{R}$-complete. This complexity class can be defined as the set of algorithmic problems that are polynomial-time equivalent to finding real roots of a multivariate polynomial with integer coefficients. Furthermore, we show that arbitrary algebraic numbers are required as weights to be able to train some instances to optimality, even if all data points are rational. Our result already applies to fully connected instances with two inputs, two outputs, and one hidden layer of ReLU neurons. Thereby, we strengthen a result by Abrahamsen, Kleist and Miltzow [NeurIPS 2021]. A consequence of this is that a combinatorial search algorithm like the one by Arora, Basu, Mianjy and Mukherjee [ICLR 2018] is impossible for networks with more than one output dimension, unless $\text{NP} = \exists\mathbb{R}$.

----

## [0] Uncertainty Estimation for Safety-critical Scene Segmentation via Fine-grained Reward Maximization

**Authors**: *Hongzheng Yang, Cheng Chen, Yueyao Chen, Markus Scheppach, Hon-Chi Yip, Qi Dou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html)

**Abstract**:

Uncertainty estimation plays an important role for future reliable deployment of deep segmentation models in safety-critical scenarios such as medical applications. However, existing methods for uncertainty estimation have been limited by the lack of explicit guidance for calibrating the prediction risk and model confidence. In this work, we propose a novel fine-grained reward maximization (FGRM) framework, to address uncertainty estimation by directly utilizing an uncertainty metric related reward function with a reinforcement learning based model tuning algorithm. This would benefit the model uncertainty estimation with direct optimization guidance for model calibration. Specifically, our method designs a new uncertainty estimation reward function using the calibration metric, which is maximized to fine-tune an evidential learning pre-trained segmentation model for calibrating prediction risk. Importantly, we innovate an effective fine-grained parameter update scheme, which imposes fine-grained reward-weighting of each network parameter according to the parameter importance quantified by the fisher information matrix. To the best of our knowledge, this is the first work exploring reward optimization for model uncertainty estimation in safety-critical vision tasks. The effectiveness of our method is demonstrated on two large safety-critical surgical scene segmentation datasets under two different uncertainty estimation settings. With real-time one forward pass at inference, our method outperforms state-of-the-art methods by a clear margin on all the calibration metrics of uncertainty estimation, while maintaining a high task accuracy for the segmentation results. Code is available at https://github.com/med-air/FGRM.

----

## [0] GLIME: General, Stable and Local LIME Explanation

**Authors**: *Zeren Tan, Yang Tian, Jian Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/71ed042903ed67c7f6355e5dd0539eec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/71ed042903ed67c7f6355e5dd0539eec-Abstract-Conference.html)

**Abstract**:

As black-box machine learning models become more complex and are applied in high-stakes settings, the need for providing explanations for their predictions becomes crucial. Although Local Interpretable Model-agnostic Explanations (LIME) \cite{ribeiro2016should} is a widely adopted method for understanding model behavior, it suffers from instability with respect to random seeds \cite{zafar2019dlime, shankaranarayana2019alime, bansal2020sam} and exhibits low local fidelity (i.e., how the explanation explains model's local behaviors) \cite{rahnama2019study, laugel2018defining}. Our study demonstrates that this instability is caused by small sample weights, resulting in the dominance of regularization and slow convergence. Additionally, LIME's sampling approach is non-local and biased towards the reference, leading to diminished local fidelity and instability to references. To address these challenges, we propose \textsc{Glime}, an enhanced framework that extends LIME and unifies several previous methods. Within the \textsc{Glime} framework, we derive an equivalent formulation of LIME that achieves significantly faster convergence and improved stability. By employing a local and unbiased sampling distribution, \textsc{Glime} generates explanations with higher local fidelity compared to LIME, while being independent of the reference choice. Moreover, \textsc{Glime} offers users the flexibility to choose sampling distribution based on their specific scenarios.

----

## [0] Efficient Symbolic Policy Learning with Differentiable Symbolic Expression

**Authors**: *Jiaming Guo, Rui Zhang, Shaohui Peng, Qi Yi, Xing Hu, Ruizhi Chen, Zidong Du, Xishan Zhang, Ling Li, Qi Guo, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7207ffb9888068c0ee13ae3be023cada-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7207ffb9888068c0ee13ae3be023cada-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (DRL) has led to a wide range of advances in sequential decision-making tasks. However, the complexity of neural network policies makes it difficult to understand and deploy with limited computational resources. Currently, employing compact symbolic expressions as symbolic policies is a promising strategy to obtain simple and interpretable policies. Previous symbolic policy methods usually involve complex training processes and pre-trained neural network policies, which are inefficient and limit the application of symbolic policies. In this paper, we propose an efficient gradient-based learning method named Efficient Symbolic Policy Learning (ESPL) that learns the symbolic policy from scratch in an end-to-end way. We introduce a symbolic network as the search space and employ a path selector to find the compact symbolic policy. By doing so we represent the policy with a differentiable symbolic expression and train it in an off-policy manner which further improves the efficiency. In addition, in contrast with previous symbolic policies which only work in single-task RL because of complexity, we expand ESPL on meta-RL to generate symbolic policies for unseen tasks. Experimentally, we show that our approach generates symbolic policies with higher performance and greatly improves data efficiency for single-task RL. In meta-RL, we demonstrate that compared with neural network policies the proposed symbolic policy achieves higher performance and efficiency and shows the potential to be interpretable.

----

## [0] PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization

**Authors**: *Jiancong Xiao, Ruoyu Sun, Zhi-Quan Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/720991812855c99df50bc8b36966cd81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/720991812855c99df50bc8b36966cd81-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.

----

## [0] Neural Graph Generation from Graph Statistics

**Authors**: *Kiarash Zahirnia, Yaochen Hu, Mark Coates, Oliver Schulte*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72153267883fbcafdb6e4662382696c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72153267883fbcafdb6e4662382696c5-Abstract-Conference.html)

**Abstract**:

We describe a new setting for learning a deep graph generative model (GGM) from aggregate graph statistics, rather than from the graph adjacency matrix. Matching the statistics of observed training graphs is the main approach for learning traditional GGMs (e.g, BTER, Chung-Lu, and  Erdos-Renyi models). Privacy researchers have proposed learning from graph statistics as a way to protect privacy. We develop an architecture for training a deep GGM  to match statistics while preserving local differential privacy guarantees. Empirical evaluation on 8 datasets indicates that our deep GGM model generates more realistic graphs than the traditional GGMs when both are learned from graph statistics only. We also benchmark our deep GGM trained on statistics only, against state-of-the-art deep GGM models that are trained on the entire adjacency matrix. The results show that graph statistics are often sufficient to build a competitive deep GGM that generates realistic graphs while protecting local privacy.

----

## [0] DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction

**Authors**: *Mohammadreza Pourreza, Davood Rafiei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html)

**Abstract**:

There is currently a significant gap between the performance of fine-tuned models and prompting approaches using Large Language Models (LLMs) on the challenging task of text-to-SQL, as evaluated on datasets such as Spider. To improve the performance of LLMs in the reasoning process, we study how decomposing the task into smaller sub-tasks can be effective. In particular, we show that breaking down the generation problem into sub-problems and feeding the solutions of those sub-problems into LLMs can be an effective approach for significantly improving their performance.  Our experiments with three LLMs show that this approach consistently improves their simple few-shot performance by roughly 10%, pushing the accuracy of LLMs towards SOTA or surpassing it. On the holdout test set of Spider, the SOTA, in terms of execution accuracy, was 79.9 and the new SOTA at the time of  this writing using our approach is 85.3. Our approach with in-context learning beats many heavily fine-tuned models by at least 5%. Additionally, when evaluated on the BIRD benchmark, our approach achieved an execution accuracy of 55.9%, setting a new SOTA on its holdout test set.

----

## [0] Scaling Up Differentially Private LASSO Regularized Logistic Regression via Faster Frank-Wolfe Iterations

**Authors**: *Edward Raff, Amol Khanna, Fred Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72235260ae8d57ac42638a26d3b7d089-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72235260ae8d57ac42638a26d3b7d089-Abstract-Conference.html)

**Abstract**:

To the best of our knowledge, there are no methods today for training differentially private regression models on sparse input data. To remedy this, we adapt the Frank-Wolfe algorithm for $L_1$ penalized linear regression to be aware of sparse inputs and to use them effectively. In doing so, we reduce the training time of the algorithm from $\mathcal{O}( T D S + T N S)$ to $\mathcal{O}(N S + T \sqrt{D} \log{D}  + T S^2)$, where $T$ is the number of iterations and a sparsity rate $S$ of a dataset with $N$ rows and $D$ features. Our results demonstrate that this procedure can reduce runtime by a factor of up to $2,200\times$, depending on the value of the privacy parameter $\epsilon$ and the sparsity of the dataset.

----

## [0] Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback

**Authors**: *Yang Cai, Haipeng Luo, Chen-Yu Wei, Weiqiang Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/722fcbc1a6667f2075d75ea79a1b3552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/722fcbc1a6667f2075d75ea79a1b3552-Abstract-Conference.html)

**Abstract**:

We revisit the problem of learning in two-player zero-sum Markov games, focusing on developing an algorithm that is *uncoupled*, *convergent*, and *rational*, with non-asymptotic convergence rates to Nash equilibrium. We start from the case of stateless matrix game with bandit feedback as a warm-up, showing an $\tilde{\mathcal{O}}(t^{-\frac{1}{8}})$ last-iterate convergence rate. To the best of our knowledge, this is the first result that obtains finite last-iterate convergence rate given access to only bandit feedback. We extend our result to the case of irreducible Markov games, providing a last-iterate convergence rate of $\tilde{\mathcal{O}}(t^{-\frac{1}{9+\varepsilon}})$ for any $\varepsilon>0$. Finally, we study Markov games without any assumptions on the dynamics, and show a *path convergence* rate, a new notion of convergence we defined, of $\tilde{\mathcal{O}}(t^{-\frac{1}{10}})$. Our algorithm removes the synchronization and prior knowledge requirement of Wei et al. (2021), which pursued the same goals as us for irreducible Markov games. Our algorithm is related to Chen et al. (2021) and Cen et al. (2021)'s and also builds on the entropy regularization technique. However, we remove their requirement of communications on the entropy values, making our algorithm entirely uncoupled.

----

## [0] Deductive Verification of Chain-of-Thought Reasoning

**Authors**: *Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) significantly benefit from Chain-of-thought (CoT) prompting in performing various reasoning tasks. While CoT allows models to produce more comprehensive reasoning processes, its emphasis on intermediate reasoning steps can inadvertently introduce hallucinations and accumulated errors, thereby limiting modelsâ€™ ability to solve complex reasoning tasks. Inspired by how humans engage in careful and meticulous deductive logical reasoning processes to solve tasks, we seek to enable language models to perform explicit and rigorous deductive reasoning, and also ensure the trustworthiness of their reasoning process through self-verification. However, directly verifying the validity of an entire deductive reasoning process is challenging, even with advanced models like ChatGPT. In light of this, we propose to decompose a reasoning verification process into a series of step-by-step subprocesses, each only receiving their necessary context and premises. To facilitate this procedure, we propose Natural Program, a natural language-based deductive reasoning format. Our approach enables models to generate precise reasoning steps where subsequent steps are more rigorously grounded on prior steps. It also empowers language models to carry out reasoning self-verification in a step-by-step manner. By integrating this verification process into each deductive reasoning stage, we significantly enhance the rigor and trustfulness of generated reasoning steps. Along this process, we also improve the answer correctness on complex reasoning tasks.

----

## [0] Rigorous Runtime Analysis of MOEA/D for Solving Multi-Objective Minimum Weight Base Problems

**Authors**: *Anh Viet Do, Aneta Neumann, Frank Neumann, Andrew M. Sutton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html)

**Abstract**:

We study the multi-objective minimum weight base problem, an abstraction of classical NP-hard combinatorial problems such as the multi-objective minimum spanning tree problem. We prove some important properties of the convex hull of the non-dominated front, such as its approximation quality and an upper bound on the number of extreme points. Using these properties, we give the first run-time analysis of the MOEA/D algorithm for this problem, an evolutionary algorithm that effectively optimizes by decomposing the objectives into single-objective components. We show that the MOEA/D, given an appropriate decomposition setting, finds all extreme points within expected fixed-parameter polynomial time, in the oracle model. Experiments are conducted on random bi-objective minimum spanning tree instances, and the results agree with our theoretical findings. Furthermore, compared with a previously studied evolutionary algorithm for the problem GSEMO, MOEA/D finds all extreme points much faster across all instances.

----

## [0] Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics

**Authors**: *Mathias Schreiner, Ole Winther, Simon Olsson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html)

**Abstract**:

Computing properties of molecular systems rely on estimating expectations of the (unnormalized) Boltzmann distribution. Molecular dynamics (MD) is a broadly adopted technique to approximate such quantities. However, stable simulations rely on very small integration time-steps ($10^{-15}\,\mathrm{s}$), whereas convergence of some moments, e.g. binding free energy or rates, might rely on sampling processes on time-scales as long as $10^{-1}\, \mathrm{s}$, and these simulations must be repeated for every molecular system independently. Here, we present Implicit Transfer Operator (ITO) Learning, a framework to learn surrogates of the simulation process with multiple time-resolutions. We implement ITO with denoising diffusion probabilistic models with a new SE(3) equivariant architecture and show the resulting models can generate self-consistent stochastic dynamics across multiple time-scales, even when the system is only partially observed. Finally, we present a coarse-grained CG-SE3-ITO model which can quantitatively model all-atom molecular dynamics using only coarse molecular representations. As such, ITO provides an important step towards multiple time- and space-resolution acceleration of MD. Code is available at \href{https://github.com/olsson-group/ito}{https://github.com/olsson-group/ito}.

----

## [0] Causal de Finetti: On the Identification of Invariant Causal Structure in Exchangeable Data

**Authors**: *Siyuan Guo, Viktor Tóth, Bernhard Schölkopf, Ferenc Huszar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7279908471a7dd4898d2715f7c6a7413-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7279908471a7dd4898d2715f7c6a7413-Abstract-Conference.html)

**Abstract**:

Constraint-based causal discovery methods leverage conditional independence tests to infer causal relationships in a wide variety of applications. Just as the majority of machine learning methods, existing work focuses on studying $\textit{independent and identically distributed}$ data. However, it is known that even with infinite $i.i.d.\$ data, constraint-based methods can only identify causal structures up to broad Markov equivalence classes, posing a fundamental limitation for causal discovery. In this work, we observe that exchangeable data contains richer conditional independence structure than $i.i.d.\$ data, and show how the richer structure can be leveraged for causal discovery. We first present causal de Finetti theorems, which state that exchangeable distributions with certain non-trivial conditional independences can always be represented as $\textit{independent causal mechanism (ICM)}$ generative processes. We then present our main identifiability theorem, which shows that given data from an ICM generative process, its unique causal structure can be identified through performing conditional independence tests. We finally develop a causal discovery algorithm and demonstrate its applicability to inferring causal relationships from multi-environment data.

----

## [0] Batch Bayesian Optimization For Replicable Experimental Design

**Authors**: *Zhongxiang Dai, Quoc Phong Nguyen, Sebastian Tay, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html)

**Abstract**:

Many real-world experimental design problems (a) evaluate multiple experimental conditions in parallel and (b) replicate each condition multiple times due to large and heteroscedastic observation noise. Given a fixed total budget, this naturally induces a trade-off between evaluating more unique conditions while replicating each of them fewer times vs. evaluating fewer unique conditions and replicating each more times. Moreover, in these problems, practitioners may be risk-averse and hence prefer an input with both good average performance and small variability. To tackle both challenges, we propose the Batch Thompson Sampling for Replicable Experimental Design (BTS-RED) framework, which encompasses three algorithms. Our BTS-RED-Known and BTS-RED-Unknown algorithms, for, respectively, known and unknown noise variance, choose the number of replications adaptively rather than deterministically such that an input with a larger noise variance is replicated more times. As a result, despite the noise heteroscedasticity, both algorithms enjoy a theoretical guarantee and are asymptotically no-regret. Our Mean-Var-BTS-RED algorithm aims at risk-averse optimization and is also asymptotically no-regret. We also show the effectiveness of our algorithms in two practical real-world applications: precision agriculture and AutoML.

----

## [0] Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning

**Authors**: *Siming Lan, Rui Zhang, Qi Yi, Jiaming Guo, Shaohui Peng, Yunkai Gao, Fan Wu, Ruizhi Chen, Zidong Du, Xing Hu, Xishan Zhang, Ling Li, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72802bef5cf1a3449e909b20c2ae18d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72802bef5cf1a3449e909b20c2ae18d5-Abstract-Conference.html)

**Abstract**:

In the field of multi-task reinforcement learning, the modular principle, which involves specializing functionalities into different modules and combining them appropriately, has been widely adopted as a promising approach to prevent the negative transfer problem that performance degradation due to conflicts between tasks. However, most of the existing multi-task RL methods only combine shared modules at the task level, ignoring that there may be conflicts within the task. In addition, these methods do not take into account that without constraints, some modules may learn similar functions, resulting in restricting the model's expressiveness and generalization capability of modular methods.In this paper, we propose the Contrastive Modules with Temporal Attention(CMTA) method to address these limitations. CMTA constrains the modules to be different from each other by contrastive learning and combining shared modules at a finer granularity than the task level with temporal attention, alleviating the negative transfer within the task and improving the generalization ability and the performance for multi-task RL.We conducted the experiment on Meta-World, a multi-task RL benchmark containing various robotics manipulation tasks. Experimental results show that CMTA outperforms learning each task individually for the first time and achieves substantial performance improvements over the baselines.

----

## [0] Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent RL with General Utilities

**Authors**: *Donghao Ying, Yunkai Zhang, Yuhao Ding, Alec Koppel, Javad Lavaei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html)

**Abstract**:

We investigate safe multi-agent reinforcement learning, where agents seek to collectively maximize an aggregate sum of local objectives while satisfying their own safety constraints. The objective and constraints are described by general utilities, i.e., nonlinear functions of the long-term state-action occupancy measure, which encompass broader decision-making goals such as risk, exploration, or imitations. The exponential growth of the state-action space size with the number of agents presents challenges for global observability, further exacerbated by the global coupling arising from agents' safety constraints. To tackle this issue, we propose a primal-dual method utilizing shadow reward and $\kappa$-hop neighbor truncation under a form of correlation decay property, where $\kappa$ is the communication radius. In the exact setting, our algorithm converges to a first-order stationary point (FOSP) at the rate of $\mathcal{O}\left(T^{-2/3}\right)$. In the sample-based setting, we demonstrate that, with high probability, our algorithm requires $\widetilde{\mathcal{O}}\left(\epsilon^{-3.5}\right)$ samples to achieve an $\epsilon$-FOSP with an approximation error of $\mathcal{O}(\phi_0^{2\kappa})$, where $\phi_0\in (0,1)$. Finally, we demonstrate the effectiveness of our model through extensive numerical experiments.

----

## [0] Optimal Transport-Guided Conditional Score-Based Diffusion Model

**Authors**: *Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72c12e48c6135762f56bf188cd2479d2-Abstract-Conference.html)

**Abstract**:

Conditional score-based diffusion model (SBDM) is for conditional generation of target data with paired data as condition, and has achieved great success in image translation. However, it requires the paired data as condition, and there would be insufficient paired data provided in real-world applications. To tackle the applications with partially paired or even unpaired dataset, we propose a novel Optimal Transport-guided Conditional Score-based diffusion model (OTCS) in this paper. We build the coupling relationship for the unpaired or partially paired dataset based on $L_2$-regularized unsupervised or semi-supervised optimal transport, respectively. Based on the coupling relationship, we develop the objective for training the conditional score-based model for unpaired or partially paired settings, which is based on a reformulation and generalization of the conditional SBDM for paired setting. With the estimated coupling relationship, we effectively train the conditional score-based model by designing  a ``resampling-by-compatibility'' strategy to choose the sampled data with high compatibility as guidance. Extensive experiments on unpaired super-resolution and semi-paired image-to-image translation demonstrated the effectiveness of the proposed OTCS model. From the viewpoint of optimal transport, OTCS provides an approach to transport data across distributions, which is a challenge for OT on large-scale datasets. We theoretically prove that OTCS realizes the data transport in OT with a theoretical bound.

----

## [0] GNeSF: Generalizable Neural Semantic Fields

**Authors**: *Hanlin Chen, Chen Li, Mengqi Guo, Zhiwen Yan, Gim Hee Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/72d32f4fe0b7af03732bd227bf1c4a5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/72d32f4fe0b7af03732bd227bf1c4a5f-Abstract-Conference.html)

**Abstract**:

3D scene segmentation based on neural implicit representation has emerged recently with the advantage of training only on 2D supervision. However, existing approaches still requires expensive per-scene optimization that prohibits generalization to novel scenes during inference. To circumvent this problem, we introduce a \textit{generalizable} 3D segmentation framework based on implicit representation. Specifically, our framework takes in multi-view image features and semantic maps as the inputs instead of only spatial information to avoid overfitting to scene-specific geometric and semantic information. We propose a novel soft voting mechanism to aggregate the 2D semantic information from different views for each 3D point. In addition to the image features, view difference information is also encoded in our framework to predict the voting scores. Intuitively, this allows the semantic information from nearby views to contribute more compared to distant ones. Furthermore, a visibility module is also designed to detect and filter out detrimental information from occluded views. Due to the generalizability of our proposed method, we can synthesize semantic maps or conduct 3D semantic segmentation for novel scenes with solely 2D semantic supervision. Experimental results show that our approach achieves comparable performance with scene-specific approaches. More importantly, our approach can even outperform existing strong supervision-based approaches with only 2D annotations.

----

## [0] When can Regression-Adjusted Control Variate Help? Rare Events, Sobolev Embedding and Minimax Optimality

**Authors**: *Jose H. Blanchet, Haoxuan Chen, Yiping Lu, Lexing Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/730ce0ae730f39e4d77b0f04a8afe4be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/730ce0ae730f39e4d77b0f04a8afe4be-Abstract-Conference.html)

**Abstract**:

This paper studies the use of a machine learning-based estimator as a control variate for mitigating the variance of Monte Carlo sampling. Specifically, we seek to uncover the key factors that influence the efficiency of control variates in reducing variance. We examine a prototype estimation problem that involves simulating the moments of a Sobolev function based on observations obtained from (random) quadrature nodes. Firstly, we establish an information-theoretic lower bound for the problem. We then study a specific quadrature rule that employs a nonparametric regression-adjusted control variate to reduce the variance of the Monte Carlo simulation. We demonstrate that this kind of quadrature rule can improve the Monte Carlo rate and achieve the minimax optimal rate under a sufficient smoothness assumption. Due to the Sobolev Embedding Theorem, the sufficient smoothness assumption eliminates the existence of rare and extreme events. Finally, we show that, in the presence of rare and extreme events, a truncated version of the Monte Carlo algorithm can achieve the minimax optimal rate while the control variate cannot improve the convergence rate.

----

## [0] Sharp Calibrated Gaussian Processes

**Authors**: *Alexandre Capone, Sandra Hirche, Geoff Pleiss*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7319b7561ffe5e2f6419acd4a2f52d6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7319b7561ffe5e2f6419acd4a2f52d6b-Abstract-Conference.html)

**Abstract**:

While Gaussian processes are a mainstay for various engineering and scientific applications, the uncertainty estimates don't satisfy frequentist guarantees and can be miscalibrated in practice. State-of-the-art approaches for designing calibrated models rely on inflating the Gaussian process posterior variance, which yields confidence intervals that are potentially too coarse. To remedy this, we present a calibration approach that generates predictive quantiles using a computation inspired by the vanilla Gaussian process posterior variance but using a different set of hyperparameters chosen to satisfy an empirical calibration constraint. This results in a calibration approach that is considerably more flexible than existing approaches, which we optimize to yield tight predictive quantiles. Our approach is shown to yield a calibrated model under reasonable assumptions. Furthermore, it outperforms existing approaches in sharpness when employed for calibrated regression.

----

## [0] GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies

**Authors**: *Takahiro Mimori, Michiaki Hamada*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/732c5757aa5577de9b103332cf7ac0bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/732c5757aa5577de9b103332cf7ac0bf-Abstract-Conference.html)

**Abstract**:

Phylogenetic inference, grounded in molecular evolution models, is essential for understanding the evolutionary relationships in biological data. Accounting for the uncertainty of phylogenetic tree variables, which include tree topologies and evolutionary distances on branches, is crucial for accurately inferring species relationships from molecular data and tasks requiring variable marginalization. Variational Bayesian methods are key to developing scalable, practical models; however, it remains challenging to conduct phylogenetic inference without restricting the combinatorially vast number of possible tree topologies. In this work, we introduce a novel, fully differentiable formulation of phylogenetic inference that leverages a unique representation of topological distributions in continuous geometric spaces. Through practical considerations on design spaces and control variates for gradient estimations, our approach, GeoPhy, enables variational inference without limiting the topological candidates. In experiments using real benchmark datasets, GeoPhy significantly outperformed other approximate Bayesian methods that considered whole topologies.

----

## [0] AbdomenAtlas-8K: Annotating 8, 000 CT Volumes for Multi-Organ Segmentation in Three Weeks

**Authors**: *Chongyu Qu, Tiezheng Zhang, Hualin Qiao, Jie Liu, Yucheng Tang, Alan L. Yuille, Zongwei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7331077e0449e94a91370c46b4f80f57-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7331077e0449e94a91370c46b4f80f57-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Annotating medical images, particularly for organ segmentation, is laborious and time-consuming. For example, annotating an abdominal organ requires an estimated rate of 30-60 minutes per CT volume based on the expertise of an annotator and the size, visibility, and complexity of the organ. Therefore, publicly available datasets for multi-organ segmentation are often limited in data size and organ diversity. This paper proposes an active learning procedure to expedite the annotation process for organ segmentation and creates the largest multi-organ dataset (by far) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in 8,448 CT volumes, equating to 3.2 million slices. The conventional annotation methods would take an experienced annotator up to 1,600 weeks (or roughly 30.8 years) to complete this task. In contrast, our annotation procedure has accomplished this task in three weeks (based on an 8-hour workday, five days a week) while maintaining a similar or even better annotation quality. This achievement is attributed to three unique properties of our method: (1) label bias reduction using multiple pre-trained segmentation models, (2) effective error detection in the model predictions, and (3) attention guidance for annotators to make corrections on the most salient errors. Furthermore, we summarize the taxonomy of common errors made by AI algorithms and annotators. This allows for continuous improvement of AI and annotations, significantly reducing the annotation costs required to create large-scale datasets for a wider variety of medical imaging tasks. Code and dataset are available at https://github.com/MrGiovanni/AbdomenAtlas

----

## [0] The Learnability of In-Context Learning

**Authors**: *Noam Wies, Yoav Levine, Amnon Shashua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73950f0eb4ac0925dc71ba2406893320-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73950f0eb4ac0925dc71ba2406893320-Abstract-Conference.html)

**Abstract**:

In-context learning is a surprising and important phenomenon that emerged when modern language models were scaled to billions of learned parameters.   Without modifying a large language model's weights, it can be tuned to perform various downstream natural language tasks simply by including concatenated training examples of these tasks in its input.  Though disruptive for many practical applications of large language models, this emergent learning paradigm is not well understood from a theoretical perspective. In this paper, we propose a first-of-its-kind PAC based framework for in-context learnability, and use it to provide the first finite sample complexity results for the in-context learning setup.  Our framework includes an initial pretraining phase, which fits a function to the pretraining distribution, and then a second in-context learning phase, which keeps this function constant and concatenates training examples of the downstream task in its input.  We use our framework in order to prove that, under mild assumptions, when the pretraining distribution is a mixture of latent tasks (a model often considered for natural language pretraining), these tasks can be efficiently learned via in-context learning, even though the model's weights are unchanged and the input significantly diverges from the pretraining distribution.  Our theoretical analysis reveals that in this setting, in-context learning is more about identifying the task than about learning it, a result which is in line with a series of recent empirical findings.   We hope that the in-context learnability framework presented in this paper will facilitate future progress towards a deeper understanding of this important new learning paradigm.

----

## [0] Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation

**Authors**: *Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html)

**Abstract**:

The ability to collect a large dataset of human preferences from text-to-image users is usually limited to companies, making such datasets inaccessible to the public. To address this issue, we create a web app that enables text-to-image users to generate images and specify their preferences. Using this web app we build Pick-a-Pic, a large, open dataset of text-to-image prompts and real users’ preferences over generated images. We leverage this dataset to train a CLIP-based scoring function, PickScore, which exhibits superhuman performance on the task of predicting human preferences. Then, we test PickScore’s ability to perform model evaluation and observe that it correlates better with human rankings than other automatic evaluation metrics. Therefore, we recommend using PickScore for evaluating future text-to-image generation models, and using Pick-a-Pic prompts as a more relevant dataset than MS-COCO. Finally, we demonstrate how PickScore can enhance existing text-to-image models via ranking.

----

## [0] Decorate3D: Text-Driven High-Quality Texture Generation for Mesh Decoration in the Wild

**Authors**: *Yanhui Guo, Xinxin Zuo, Peng Dai, Juwei Lu, Xiaolin Wu, Li Cheng, Youliang Yan, Songcen Xu, Xiaofei Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73af055566f5514b9863315133b84eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73af055566f5514b9863315133b84eda-Abstract-Conference.html)

**Abstract**:

This paper presents Decorate3D, a versatile and user-friendly method for the creation and editing of 3D objects using images. Decorate3D models a real-world object of interest by neural radiance field (NeRF) and decomposes the NeRF representation into an explicit mesh representation, a view-dependent texture, and a diffuse UV texture. Subsequently, users can either manually edit the UV or provide a prompt for the automatic generation of a new 3D-consistent texture.  To achieve high-quality 3D texture generation, we propose a structure-aware score distillation sampling method to optimize a neural UV texture based on user-defined text and empower an image diffusion model with 3D-consistent generation capability. Furthermore, we introduce a few-view resampling training method and utilize a super-resolution model to obtain refined high-resolution UV textures (2048$\times$2048) for 3D texturing. Extensive experiments collectively validate the superior performance of Decorate3D in retexturing real-world 3D objects. Project page: https://decorate3d.github.io/Decorate3D/.

----

## [0] Representational Strengths and Limitations of Transformers

**Authors**: *Clayton Sanford, Daniel J. Hsu, Matus Telgarsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html)

**Abstract**:

Attention layers, as commonly used in transformers, form the backbone of modern deep learning, yet there is no mathematical description of their benefits and deficiencies as compared with other architectures. In this work we establish both positive and negative results on the representation power of attention layers, with a focus on intrinsic complexity parameters such as width, depth, and embedding dimension. On the positive side, we present a sparse averaging task, where recurrent networks and feedforward networks all have complexity scaling polynomially in the input size, whereas transformers scale merely logarithmically in the input size; furthermore, we use the same construction to show the necessity and role of a large embedding dimension in a transformer. On the negative side, we present a triple detection task, where attention layers in turn have complexity scaling linearly in the input size; as this scenario seems rare in practice, we also present natural variants that can be efficiently solved by attention layers. The proof techniques emphasize the value of communication complexity in the analysis of transformers and related models, and the role of sparse averaging as a prototypical attention task, which even finds use in the analysis of triple detection.

----

## [0] On the Relationship Between Relevance and Conflict in Online Social Link Recommendations

**Authors**: *Yanbang Wang, Jon M. Kleinberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/73d6c3e4b214deebbbf8256e26d2cf45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/73d6c3e4b214deebbbf8256e26d2cf45-Abstract-Conference.html)

**Abstract**:

In an online social network, link recommendations are a way for users to discover relevant links to people they may know, thereby potentially increasing their engagement on the platform. However, the addition of links to a social network can also have an effect on the level of conflict in the network --- expressed in terms of polarization and disagreement. To date, however, we have very little understanding of how these two implications of link formation relate to each other: are the goals of high relevance and conflict reduction aligned, or are the links that users are most likely to accept fundamentally different from the ones with the greatest potential for reducing conflict? Here we provide the first analysis of this question, using the recently popular Friedkin-Johnsen model of opinion dynamics. We first present a surprising result on how link additions shift the level of opinion conflict, followed by explanation work that relates the amount of shift to structural features of the added links. We then characterize the gap in conflict reduction between the set of links achieving the largest reduction and the set of links achieving the highest relevance. The gap is measured on real-world data, based on instantiations of relevance defined by 13 link recommendation algorithms. We find that some, but not all, of the more accurate algorithms actually lead to better reduction of conflict. Our work suggests that social links recommended for increasing user engagement may not be as conflict-provoking as people might have thought.

----

## [0] Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

**Authors**: *Ziba Parsons, Fei Dou, Houyi Du, Zheng Song, Jin Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html)

**Abstract**:

This paper explores the challenges of implementing Federated Learning (FL) in practical scenarios featuring isolated nodes with data heterogeneity, which can only be connected to the server through wireless links in an infrastructure-less environment. To overcome these challenges, we propose a novel mobilizing personalized FL approach, which aims to facilitate mobility and resilience. Specifically, we develop a novel optimization algorithm called Random Walk Stochastic Alternating Direction Method of Multipliers (RWSADMM). RWSADMM capitalizes on the server's random movement toward clients and formulates local proximity among their adjacent clients based on hard inequality constraints rather than requiring consensus updates or introducing bias via regularization methods. To mitigate the computational burden on the clients, an efficient stochastic solver of the approximated optimization problem is designed in RWSADMM, which provably converges to the stationary point almost surely in expectation. Our theoretical and empirical results demonstrate the provable fast convergence and substantial accuracy improvements achieved by RWSADMM compared to baseline methods, along with its benefits of reduced communication costs and enhanced scalability.

----

## [0] An Optimal Structured Zeroth-order Algorithm for Non-smooth Optimization

**Authors**: *Marco Rando, Cesare Molinari, Lorenzo Rosasco, Silvia Villa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7429f4c1b267cf619f28c4d4f1532f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7429f4c1b267cf619f28c4d4f1532f99-Abstract-Conference.html)

**Abstract**:

Finite-difference methods are a class of algorithms designed to solve black-box optimization problems by approximating a gradient of the target function on a set of directions. In black-box optimization, the non-smooth setting is particularly relevant since, in practice, differentiability and smoothness assumptions cannot be verified. To cope with nonsmoothness, several authors use a smooth approximation of the target function and show that finite difference methods approximate its gradient. Recently, it has been proved that imposing a structure in the directions allows improving performance. However, only the smooth setting was considered. To close this gap, we introduce and analyze O-ZD, the first structured finite-difference algorithm for non-smooth black-box optimization. Our method exploits a smooth approximation of the target function and we prove that it approximates its gradient on a subset of random {\em orthogonal} directions. We analyze the convergence of O-ZD under different assumptions.  For non-smooth convex functions, we obtain the optimal complexity. In the non-smooth non-convex setting, we characterize the number of iterations needed to bound the expected norm of the smoothed gradient. For smooth functions, our analysis recovers existing results for structured zeroth-order methods for the convex case and extends them to the non-convex setting. We conclude with numerical simulations where assumptions are satisfied, observing that our algorithm has very good practical performances.

----

## [0] Online Control for Meta-optimization

**Authors**: *Xinyi Chen, Elad Hazan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/745b7e084d5ca5afc07fb454ab2be522-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/745b7e084d5ca5afc07fb454ab2be522-Abstract-Conference.html)

**Abstract**:

Choosing the optimal hyperparameters, including learning rate and momentum, for specific optimization instances is a significant yet non-convex challenge. This makes conventional iterative techniques such as hypergradient descent \cite{baydin2017online} insufficient in obtaining global optimality guarantees.We consider the more general task of meta-optimization -- online learning of the best optimization algorithm given problem instances, and introduce a novel approach based on control theory. We show how meta-optimization can be formulated as an optimal control problem, departing from existing literature that use stability-based methods to study optimization. Our approach leverages convex relaxation techniques in the recently-proposed nonstochastic control framework to overcome the challenge of nonconvexity, and obtains regret guarantees vs. the best offline solution. This guarantees that in meta-optimization, we can learn a method that attains convergence comparable to that of the best optimization method in hindsight from a class of methods.

----



[Go to the previous page](NIPS-2023-list1500.md)

[Go to the next page](NIPS-2023-list1502.md)

[Go to the catalog section](README.md)