## [600] Bootstrapping Multi-View Representations for Fake News Detection

        **Authors**: *Qichao Ying, Xiaoxiao Hu, Yangming Zhou, Zhenxing Qian, Dan Zeng, Shiming Ge*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25670](https://doi.org/10.1609/aaai.v37i4.25670)

        **Abstract**:

        Previous researches on multimedia fake news detection include a series of complex feature extraction and fusion networks to gather useful information from the news. However, how cross-modal consistency relates to the fidelity of news and how features from different modalities affect the decision-making are still open questions. This paper presents a novel scheme of Bootstrapping Multi-view Representations (BMR) for fake news detection. Given a multi-modal news, we extract representations respectively from the views of the text, the image pattern and the image semantics. Improved Multi-gate Mixture-of-Expert networks (iMMoE) are proposed for feature refinement and fusion. Representations from each view are separately used to coarsely predict the fidelity of the whole news, and the multimodal representations are able to predict the cross-modal consistency. With the prediction scores, we reweigh each view of the representations and bootstrap them for fake news detection. Extensive experiments conducted on typical fake news detection datasets prove that BMR outperforms state-of-the-art schemes.

        ----

        ## [601] Overcoming Forgetting in Fine-Grained Urban Flow Inference via Adaptive Knowledge Replay

        **Authors**: *Haoyang Yu, Xovee Xu, Ting Zhong, Fan Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25671](https://doi.org/10.1609/aaai.v37i4.25671)

        **Abstract**:

        Fine-grained urban flow inference (FUFI) problem aims at inferring the high-resolution flow maps from the coarse-grained ones, which plays an important role in sustainable and economic urban computing and traffic management. Previous models addressed the FUFI problem from spatial constraint, external factors, and memory cost. However, utilizing the new urban flow maps to calibrate the learned model is very challenging due to the "catastrophic forgetting" problem and is still under-explored. In this paper, we make the first step in FUFI and present CUFAR -- Continual Urban Flow inference with Adaptive knowledge Replay -- a novel framework for inferring the fine-grained citywide traffic flows. Specifically, (1) we design a spatial-temporal inference network that can extract better flow map features from both local and global levels; (2) then we present an adaptive knowledge replay (AKR) training algorithm to selectively replay the learned knowledge to facilitate the learning process of the model on new knowledge without forgetting. In addition, we also propose a knowledge discriminator to avoid "negative replaying" issue introduced by noisy urban flow maps. Extensive experiments on four large-scale real-world FUFI datasets demonstrate that our proposed model consistently outperforms strong baselines and effectively mitigates the forgetting problem. Source code is available at: https://github.com/PattonYu/CUFAR.

        ----

        ## [602] Generalized Cell Type Annotation and Discovery for Single-Cell RNA-Seq Data

        **Authors**: *Yuyao Zhai, Liang Chen, Minghua Deng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25672](https://doi.org/10.1609/aaai.v37i4.25672)

        **Abstract**:

        The rapid development of single-cell RNA sequencing (scRNA-seq) technology allows us to study gene expression heterogeneity at the cellular level. Cell annotation is the basis for subsequent downstream analysis in single-cell data mining. Existing methods rarely explore the fine-grained semantic knowledge of novel cell types absent from the reference data and usually susceptible to batch effects on the classification of seen cell types.
Taking into consideration these limitations, this paper proposes a new and practical task called generalized cell type annotation and discovery for scRNA-seq data. In this task, cells of seen cell types are given class labels, while cells of novel cell types are given cluster labels instead of a unified “unassigned” label. To address this problem, we carefully design a comprehensive evaluation benchmark and propose a novel end-to-end algorithm framework called scGAD. Specifically, scGAD first builds the intrinsic correspondence across the reference and target data by retrieving the geometrically and semantically mutual nearest neighbors as anchor pairs. Then we introduce an anchor-based self-supervised learning module with a connectivity-aware attention mechanism to facilitate model prediction capability on unlabeled target data. To enhance the inter-type separation and intra-type compactness, we further propose a confidential prototypical self-supervised learning module to uncover the consensus category structure of the reference and target data. Extensive results on massive real datasets demonstrate the superiority of scGAD over various state-of-the-art clustering and annotation methods.

        ----

        ## [603] Mining and Applying Composition Knowledge of Dance Moves for Style-Concentrated Dance Generation

        **Authors**: *Xinjian Zhang, Su Yang, Yi Xu, Weishan Zhang, Longwen Gao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25673](https://doi.org/10.1609/aaai.v37i4.25673)

        **Abstract**:

        Choreography refers to creation of dance motions according to both music and dance knowledge, where the created dances should be style-specific and consistent. However, most of the existing methods generate dances using the given music as the only reference, lacking the stylized dancing knowledge, namely, the flag motion patterns contained in different styles. Without the stylized prior knowledge, these approaches are not promising to generate controllable style or diverse moves for each dance style, nor new dances complying with stylized knowledge. To address this issue, we propose a novel music-to-dance generation framework guided by style embedding, considering both input music and stylized dancing knowledge. These style embeddings are learnt representations of style-consistent kinematic abstraction of reference dance videos, which can act as controllable factors to impose style constraints on dance generation in a latent manner. Hence, we can make the style embedding fit into any given style while allowing the flexibility to generate new compatible dance moves by modifying the style embedding according to the learnt representations of a certain style. We are the first to achieve knowledge-driven style control in dance generation tasks. To support this study, we build a large multi-style music-to-dance dataset referred to as I-Dance. The qualitative and quantitative evaluations demonstrate the advantage of the proposed framework, as well as the ability to synthesize diverse moves under a dance style directed by style embedding.

        ----

        ## [604] Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation

        **Authors**: *Ruijie Zhao, Mingwei Zhan, Xianwen Deng, Yanhao Wang, Yijun Wang, Guan Gui, Zhi Xue*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25674](https://doi.org/10.1609/aaai.v37i4.25674)

        **Abstract**:

        Traffic classification is a critical task in network security and management. Recent research has demonstrated the effectiveness of the deep learning-based traffic classification method. However, the following limitations remain: (1) the traffic representation is simply generated from raw packet bytes, resulting in the absence of important information; (2) the model structure of directly applying deep learning algorithms does not take traffic characteristics into account; and (3) scenario-specific classifier training usually requires a labor-intensive and time-consuming process to label data. In this paper, we introduce a masked autoencoder (MAE) based traffic transformer with multi-level flow representation to tackle these problems. To model raw traffic data, we design a formatted traffic representation matrix with hierarchical flow information. After that, we develop an efficient Traffic Transformer, in which packet-level and flow-level attention mechanisms implement more efficient feature extraction with lower complexity. At last, we utilize the MAE paradigm to pre-train our classifier with a large amount of unlabeled data, and perform fine-tuning with a few labeled data for a series of traffic classification tasks. Experiment findings reveal that our method outperforms state-of-the-art methods on five real-world traffic datasets by a large margin. The code is available at https://github.com/NSSL-SJTU/YaTC.

        ----

        ## [605] Loan Fraud Users Detection in Online Lending Leveraging Multiple Data Views

        **Authors**: *Sha Zhao, Yongrui Huang, Ling Chen, Chunping Wang, Shijian Li, Lei Chen, Gang Pan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25675](https://doi.org/10.1609/aaai.v37i4.25675)

        **Abstract**:

        In recent years, online lending platforms have been becoming attractive for micro-financing and popular in financial industries. However, such online lending platforms face a high risk of failure due to the lack of expertise on borrowers' creditworthness. Thus, risk forecasting is important to avoid economic loss. Detecting loan fraud users in advance is at the heart of risk forecasting. The purpose of fraud user (borrower) detection is to predict whether one user will fail to make required payments in the future. Detecting fraud users depend on historical loan records. However, a large proportion of users lack such information, especially for new users. In this paper, we attempt to detect loan fraud users from cross domain heterogeneous data views, including user attributes, installed app lists, app installation behaviors, and app-in logs, which compensate for the lack of historical loan records. However, it is difficult to effectively fuse the multiple heterogeneous data views. Moreover, some samples miss one or even more data views, increasing the difficulty in fusion. To address the challenges, we propose a novel end-to-end deep multiview learning approach, which encodes heterogeneous data views into homogeneous ones, generates the missing views based on the learned relationship among all the views, and then fuses all the views together to a comprehensive view for identifying fraud users. Our model is evaluated on a real-world large-scale dataset consisting of 401,978 loan records of 228,117 users from January 1, 2019, to September 30, 2019, achieving the state-of-the-art performance.

        ----

        ## [606] Sparse Maximum Margin Learning from Multimodal Human Behavioral Patterns

        **Authors**: *Ervine Zheng, Qi Yu, Zhi Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25676](https://doi.org/10.1609/aaai.v37i4.25676)

        **Abstract**:

        We propose a multimodal data fusion framework to systematically analyze human behavioral data from specialized domains that are inherently dynamic, sparse, and heterogeneous. We develop a two-tier architecture of probabilistic mixtures, where the lower tier leverages parametric distributions from the exponential family to extract significant behavioral patterns from each data modality. These patterns are then organized into a dynamic latent state space at the higher tier to fuse patterns from different modalities. In addition, our framework jointly performs pattern discovery and maximum-margin learning for downstream classification tasks by using a group-wise sparse prior that regularizes the coefficients of the maximum-margin classifier. Therefore, the discovered patterns are highly interpretable and discriminative to support downstream classification tasks. Experiments on real-world behavioral data from medical and psychological domains demonstrate that our framework discovers meaningful multimodal behavioral patterns with improved interpretability and prediction performance.

        ----

        ## [607] Direct Heterogeneous Causal Learning for Resource Allocation Problems in Marketing

        **Authors**: *Hao Zhou, Shaoming Li, Guibin Jiang, Jiaqi Zheng, Dong Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25677](https://doi.org/10.1609/aaai.v37i4.25677)

        **Abstract**:

        Marketing is an important mechanism to increase user engagement and improve platform revenue, and heterogeneous causal learning can help develop more effective strategies. Most decision-making problems in marketing can be formulated as resource allocation problems and have been studied for decades. Existing works usually divide the solution procedure into two fully decoupled stages, i.e., machine learning (ML) and operation research (OR) --- the first stage predicts the model parameters and they are fed to the optimization in the second stage. However, the error of the predicted parameters in ML cannot be respected and a series of complex mathematical operations in OR lead to the increased accumulative errors. Essentially, the improved precision on the prediction parameters may not have a positive correlation on the final solution due to the side-effect from the decoupled design.

In this paper, we propose a novel approach for solving resource allocation problems to mitigate the side-effects. Our key intuition is that we introduce the decision factor to establish a bridge between ML and OR such that the solution can be directly obtained in OR by only performing the sorting or comparison operations on the decision factor. Furthermore, we design a customized loss function that can conduct direct heterogeneous causal learning on the decision factor, an unbiased estimation of which can be guaranteed when the loss convergences.  As a case study, we apply our approach to two crucial problems in marketing: the binary treatment assignment problem and the budget allocation problem with multiple treatments. Both large-scale simulations and online A/B Tests demonstrate that our approach achieves significant improvement compared with state-of-the-art.

        ----

        ## [608] Mediated Cheap Talk Design

        **Authors**: *Itai Arieli, Ivan Geffner, Moshe Tennenholtz*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25678](https://doi.org/10.1609/aaai.v37i5.25678)

        **Abstract**:

        We study an information design problem with two informed senders and a receiver in which, in contrast to traditional Bayesian persuasion settings, senders do not have commitment power. In our setting, a trusted mediator/platform gathers data from the senders and recommends the receiver which action to play. We characterize the set of feasible action distributions that can be obtained in equilibrium, and provide an O(n log n) algorithm (where n is the number of states) that computes the optimal equilibrium for the senders.  Additionally, we show that the optimal equilibrium for the receiver can be obtained by a simple revelation mechanism.

        ----

        ## [609] Bidding Graph Games with Partially-Observable Budgets

        **Authors**: *Guy Avni, Ismaël Jecker, Dorde Zikelic*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25679](https://doi.org/10.1609/aaai.v37i5.25679)

        **Abstract**:

        Two-player zero-sum "graph games" are central in logic, verification, and multi-agent systems. The game proceeds by placing a token on a vertex of a graph, and allowing the players to move it to produce an infinite path, which determines the winner or payoff of the game. Traditionally, the players alternate turns in moving the token. In "bidding games", however, the players have budgets and in each turn, an auction (bidding) determines which player moves the token. So far, bidding games have only been studied as full-information games. 
In this work we initiate the study of partial-information bidding games: we study bidding games in which a player's initial budget is drawn from a known probability distribution. 
We show that while for some bidding mechanisms and objectives, it is straightforward to adapt the results from the full-information setting to the partial-information setting, for others, the analysis is significantly more challenging, requires new techniques, and gives rise to interesting results. 
Specifically, we study games with "mean-payoff" objectives in combination with "poorman" bidding. We construct optimal strategies for a partially-informed player who plays against a fully-informed adversary. We show that, somewhat surprisingly, the "value" under pure strategies does not necessarily exist in such games.

        ----

        ## [610] Fairness Concepts for Indivisible Items with Externalities

        **Authors**: *Haris Aziz, Warut Suksompong, Zhaohong Sun, Toby Walsh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25680](https://doi.org/10.1609/aaai.v37i5.25680)

        **Abstract**:

        We study a fair allocation problem of indivisible items under additive externalities in which each agent also receives utility from items that are assigned to other agents. This allows us to capture scenarios in which agents benefit from or compete against one another. We extend the well-studied properties of envy-freeness up to one item (EF1) and envy-freeness up to any item (EFX) to this setting, and we propose a new fairness concept called general fair share (GFS), which applies to a more general public decision making model. We undertake a detailed study and present algorithms for finding fair allocations.

        ----

        ## [611] Finding Fair Allocations under Budget Constraints

        **Authors**: *Siddharth Barman, Arindam Khan, Sudarshan Shyam, K. V. N. Sreenivas*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25681](https://doi.org/10.1609/aaai.v37i5.25681)

        **Abstract**:

        We study the fair allocation of indivisible goods among agents with identical, additive valuations but individual budget constraints. Here, the indivisible goods--each with a specific size and value--need to be allocated such that the bundle assigned to each agent is of total size at most the agent's budget. Since envy-free allocations do not necessarily exist in the indivisible goods context, compelling relaxations--in particular, the notion of envy-freeness up to k goods (EFk)--have received significant attention in recent years. In an EFk allocation, each agent prefers its own bundle over that of any other agent, up to the removal of k goods, and the agents have similarly bounded envy against the charity (which corresponds to the set of all unallocated goods). It has been shown in prior work that an allocation that satisfies the budget constraints and maximizes the Nash social welfare is 1/4-approximately EF1. However, the computation (or even existence) of exact EFk allocations remained an intriguing open problem.

We make notable progress towards this by proposing a simple, greedy, polynomial-time algorithm that computes EF2 allocations under budget constraints. Our algorithmic result  implies the universal existence of EF2 allocations in this fair division context. The analysis of the algorithm exploits intricate structural properties of envy-freeness. Interestingly, the same algorithm also provides EF1 guarantees for important special cases. Specifically, we settle the existence of EF1 allocations for instances in which: (i) the value of each good is proportional to its size, (ii) all the goods have the same size, or (iii) all the goods have the same value. Our EF2 result even extends to the setting wherein the goods' sizes are agent specific.

        ----

        ## [612] Now We're Talking: Better Deliberation Groups through Submodular Optimization

        **Authors**: *Jake Barrett, Kobi Gal, Paul Gölz, Rose M. Hong, Ariel D. Procaccia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25682](https://doi.org/10.1609/aaai.v37i5.25682)

        **Abstract**:

        Citizens’ assemblies are groups of randomly selected constituents who are tasked with providing recommendations on policy questions. Assembly members form their recommendations through a sequence of discussions in small groups (deliberation), in which group members exchange arguments and experiences. We seek to support this process through optimization, by studying how to assign participants to discussion groups over multiple sessions, in a way that maximizes interaction between participants and satisfies diversity constraints within each group. Since repeated meetings between a given pair of participants have diminishing marginal returns, we capture interaction through a submodular function, which is approximately optimized by a greedy algorithm making calls to an ILP solver. This framework supports different submodular objective functions, and we identify sensible options, but we also show it is not necessary to commit to a particular choice: Our main theoretical result is a (practically efficient) algorithm that simultaneously approximates every possible objective function of the form we are interested in. Experiments with data from real citizens' assemblies demonstrate that our approach substantially outperforms the heuristic algorithm currently used by practitioners.

        ----

        ## [613] Causes of Stability in Dynamic Coalition Formation

        **Authors**: *Niclas Boehmer, Martin Bullinger, Anna Maria Kerkmann*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25683](https://doi.org/10.1609/aaai.v37i5.25683)

        **Abstract**:

        We study the formation of stable outcomes via simple dynamics in cardinal hedonic games, where the utilities of agents change over time depending on the history of the coalition formation process. Specifically, we analyze situations where members of a coalition decrease their utility for a leaving agent (resent) or increase their utility for a joining agent (appreciation). We show that in contrast to classical dynamics, for resentful or appreciative agents, dynamics are guaranteed to converge under mild conditions for various stability concepts. Thereby, we establish that both resent and appreciation are strong stability-driving forces.

        ----

        ## [614] Properties of Position Matrices and Their Elections

        **Authors**: *Niclas Boehmer, Jin-Yi Cai, Piotr Faliszewski, Austen Z. Fan, Lukasz Janeczko, Andrzej Kaczmarczyk, Tomasz Was*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25684](https://doi.org/10.1609/aaai.v37i5.25684)

        **Abstract**:

        We study the properties of elections that have a given position matrix (in such elections each candidate is ranked on each position by a number of voters specified in the matrix).  We show that counting elections that generate a given position matrix is #P-complete. Consequently, sampling such elections uniformly at random seems challenging and we propose a simpler algorithm, without hard guarantees. Next, we consider the problem of testing if a given matrix can be implemented by an election with a certain structure (such as single-peakedness or group-separability). Finally, we consider the problem of checking if a given position matrix can be implemented by an election with a Condorcet winner.  We complement our theoretical findings with experiments.

        ----

        ## [615] Rank Aggregation Using Scoring Rules

        **Authors**: *Niclas Boehmer, Robert Bredereck, Dominik Peters*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25685](https://doi.org/10.1609/aaai.v37i5.25685)

        **Abstract**:

        To aggregate rankings into a social ranking, one can use scoring systems such as Plurality, Veto, and Borda. We distinguish three types of methods: ranking by score, ranking by repeatedly choosing a winner that we delete and rank at the top, and ranking by repeatedly choosing a loser that we delete and rank at the bottom. The latter method captures the frequently studied voting rules Single Transferable Vote (aka Instant Runoff Voting), Coombs, and Baldwin. In an experimental analysis, we show that the three types of methods produce different rankings in practice. We also provide evidence that sequentially selecting winners is most suitable to detect the "true" ranking of candidates. For different rules in our classes, we then study the (parameterized) computational complexity of deciding in which positions a given candidate can appear in the chosen ranking. As part of our analysis, we also consider the Winner Determination problem for STV, Coombs, and Baldwin and determine their complexity when there are few voters or candidates.

        ----

        ## [616] Proportionality in Approval-Based Participatory Budgeting

        **Authors**: *Markus Brill, Stefan Forster, Martin Lackner, Jan Maly, Jannik Peters*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25686](https://doi.org/10.1609/aaai.v37i5.25686)

        **Abstract**:

        The ability to measure the satisfaction of (groups of) voters is a crucial prerequisite for formulating proportionality axioms in approval-based participatory budgeting elections. Two common -- but very different -- ways to measure the satisfaction of a voter consider (i) the number of approved projects and (ii) the total cost of approved projects, respectively. In general, it is difficult to decide which measure of satisfaction best reflects the voters' true utilities. In this paper, we study proportionality axioms with respect to large classes of approval-based satisfaction functions. We establish logical implications among our axioms and related notions from the literature, and we ask whether outcomes can be achieved that are proportional with respect to more than one satisfaction function. We show that this is impossible for the two commonly used satisfaction functions when considering proportionality notions based on extended justified representation, but achievable for a notion based on proportional justified representation. For the latter result, we introduce a strengthening of priceability and show that it is satisfied by several polynomial-time computable rules, including the Method of Equal Shares and Phragmén's sequential rule.

        ----

        ## [617] Multiwinner Voting with Possibly Unavailable Candidates

        **Authors**: *Markus Brill, Hayrullah Dindar, Jonas Israel, Jérôme Lang, Jannik Peters, Ulrike Schmidt-Kraepelin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25687](https://doi.org/10.1609/aaai.v37i5.25687)

        **Abstract**:

        Selecting a committee that meets diversity and proportionality criteria is a challenging endeavor that has been studied extensively in recent years. This task becomes even more challenging when some of the selected candidates decline the invitation to join the committee. Since the unavailability of one candidate may impact the rest of the selection, inviting all candidates at the same time may lead to a suboptimal committee. Instead, invitations should be sequential and conditional on which candidates invited so far accepted the invitation: the solution to the committee selection problem is a query policy. If invitation queries are binding, they should be safe: one should not query a candidate without being sure that whatever the set of available candidates possible at that stage, her inclusion will not jeopardize committee optimality. Assuming approval-based inputs, we characterize the set of rules for which a safe query exists at every stage. In order to parallelize the invitation process, we investigate the computation of safe parallel queries, and show that it is often hard. We also study the existence of safe parallel queries with respect to proportionality axioms such as extended justified representation.

        ----

        ## [618] Fair Division with Prioritized Agents

        **Authors**: *Xiaolin Bu, Zihao Li, Shengxin Liu, Jiaxin Song, Biaoshuai Tao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25688](https://doi.org/10.1609/aaai.v37i5.25688)

        **Abstract**:

        We consider the fair division problem of indivisible items. It is well-known that an envy-free allocation may not exist, and a relaxed version of envy-freeness, envy-freeness up to one item (EF1), has been widely considered. In an EF1 allocation, an agent may envy others' allocated shares, but only up to one item. In many applications, we may wish to specify a subset of prioritized agents where strict envy-freeness needs to be guaranteed from these agents to the remaining agents, while ensuring the whole allocation is still EF1. Prioritized agents may be those agents who are envious in a previous EF1 allocation, those agents who belong to underrepresented groups, etc. Motivated by this, we propose a new fairness notion named envy-freeness with prioritized agents EFprior, and study the existence and the algorithmic aspects for the problem of computing an EFprior allocation. With additive valuations, the simple round-robin algorithm is able to compute an EFprior allocation. In this paper, we mainly focus on general valuations. In particular, we present a polynomial-time algorithm that outputs an EFprior allocation with most of the items allocated. When all the items need to be allocated, we also present polynomial-time algorithms for some well-motivated special cases.

        ----

        ## [619] Topological Distance Games

        **Authors**: *Martin Bullinger, Warut Suksompong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25689](https://doi.org/10.1609/aaai.v37i5.25689)

        **Abstract**:

        We introduce a class of strategic games in which agents are assigned to nodes of a topology graph and the utility of an agent depends on both the agent's inherent utilities for other agents as well as her distance from these agents on the topology graph. This model of topological distance games (TDGs) offers an appealing combination of important aspects of several prominent settings in coalition formation, including (additively separable) hedonic games, social distance games, and Schelling games. We study the existence and complexity of stable outcomes in TDGs—for instance, while a jump stable assignment may not exist in general, we show that the existence is guaranteed in several special cases. We also investigate the dynamics induced by performing beneficial jumps.

        ----

        ## [620] Game Implementation: What Are the Obstructions?

        **Authors**: *Jiehua Chen, Seyedeh Negar Layegh Khavidaki, Sebastian Vincent Haydn, Sofia Simola, Manuel Sorge*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25690](https://doi.org/10.1609/aaai.v37i5.25690)

        **Abstract**:

        In many applications, we want to influence the decisions of independent agents by designing incentives for their actions. We revisit a fundamental problem in this area, called GAME IMPLEMENTATION: Given a game in standard form and a set of desired strategies, can we design a set of payment promises such that if the players take the payment promises into account, then all undominated strategies are desired? Furthermore, we aim to minimize the cost, that is, the worst-case amount of payments.

We study the tractability of computing such payment promises and determine more closely what obstructions we may have to overcome in doing so. We show that GAME IMPLEMENTATION is NP-hard even for two players, solving in particular a long-standing open question and suggesting more restrictions are necessary to obtain tractability results. We thus study the regime in which players have only a small constant number of strategies and obtain the following. First, this case remains NP-hard even if each player’s utility depends only on three others. Second, we repair a flawed efficient algorithm for the case of both small number of strategies and small number of players. Among further results, we characterize sets of desired strategies that can be implemented at zero cost as a generalization of Nash equilibria.

        ----

        ## [621] A Pair-Approximation Method for Modelling the Dynamics of Multi-Agent Stochastic Games

        **Authors**: *Chen Chu, Zheng Yuan, Shuyue Hu, Chunjiang Mu, Zhen Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25691](https://doi.org/10.1609/aaai.v37i5.25691)

        **Abstract**:

        Developing a dynamical model for learning in games has attracted much recent interest. In stochastic games, agents need to make decisions in multiple states, and transitions between states, in turn, influence the dynamics of strategies. While previous works typically focus either on 2-agent stochastic games or on normal form games under an infinite-agent setting, we aim at formally modelling the learning dynamics in stochastic games under the infinite-agent setting. With a novel use of pair-approximation method, we develop a formal model for myopic Q-learning in stochastic games with symmetric state transition. We verify the descriptive power of our model (a partial differential equation) across various games through comparisons with agent-based simulation results. Based on our proposed model, we can gain qualitative and quantitative insights into the influence of transition probabilities on the dynamics of strategies. In particular, we illustrate that a careful design of transition probabilities can help players overcome the social dilemmas and promote cooperation, even if agents are myopic learners.

        ----

        ## [622] Complexity of Probabilistic Inference in Random Dichotomous Hedonic Games

        **Authors**: *Saar Cohen, Noa Agmon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25692](https://doi.org/10.1609/aaai.v37i5.25692)

        **Abstract**:

        Hedonic games model cooperative games where agents desire to form coalitions, and only care about the composition of the coalitions of which they are members. Focusing on various classes of dichotomous hedonic games, where each agent either approves or disapproves a given coalition, we propose the random extension, where players have an independent participation probability. We initiate the research on the computational complexity of computing the probability that coalitions and partitions are optimal or stable. While some cases admit efficient algorithms (e.g., agents approve only few coalitions), they become computationally hard (#P-hard) in their complementary scenario. We then investigate the distribution of coalitions in perfect partitions and their performance in majority games, where an agent approves coalitions in which the agent is friends with the majority of its members. When friendships independently form with a constant probability, we prove that the number of coalitions of size 3 converges in distribution to a Poisson random variable.

        ----

        ## [623] Combinatorial Civic Crowdfunding with Budgeted Agents: Welfare Optimality at Equilibrium and Optimal Deviation

        **Authors**: *Sankarshan Damle, Manisha Padala, Sujit Gujar*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25693](https://doi.org/10.1609/aaai.v37i5.25693)

        **Abstract**:

        Civic Crowdfunding (CC) uses the ``power of the crowd" to garner contributions towards public projects. As these projects are non-excludable, agents may prefer to ``free-ride,"  resulting in the project not being funded. Researchers introduce refunds for single project CC to incentivize agents to contribute, guaranteeing the project's funding. These funding guarantees are applicable only when agents have an unlimited budget. This paper focuses on a combinatorial setting, where multiple projects are available for CC and agents have a limited budget. We study specific conditions where funding can be guaranteed. Naturally, funding the optimal social welfare subset of projects is desirable when every available project cannot be funded due to budget restrictions. We prove the impossibility of achieving optimal welfare at equilibrium for any monotone refund scheme. Further, given the contributions of other agents, we prove that it is NP-Hard for an agent to determine its optimal strategy. That is, while profitable deviations may exist for agents instead of funding the optimal welfare subset, it is computationally hard for an agent to find its optimal deviation. Consequently, we study different heuristics agents can use to contribute to the projects in practice. We demonstrate the heuristics' performance as the average-case trade-off between the welfare obtained and an agent's utility through simulations.

        ----

        ## [624] Strategyproofness and Proportionality in Party-Approval Multiwinner Elections

        **Authors**: *Théo Delemazure, Tom Demeulemeester, Manuel Eberl, Jonas Israel, Patrick Lederer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25694](https://doi.org/10.1609/aaai.v37i5.25694)

        **Abstract**:

        In party-approval multiwinner elections the goal is to allocate the seats of a fixed-size committee to parties based on the approval ballots of the voters over the parties. In particular, each voter can approve multiple parties and each party can be assigned multiple seats. Two central requirements in this setting are proportional representation and strategyproofness. Intuitively, proportional representation requires that every sufficiently large group of voters with similar preferences is represented in the committee. Strategyproofness demands that no voter can benefit by misreporting her true preferences. We show that these two axioms are incompatible for anonymous party-approval multiwinner voting rules, thus proving a far-reaching impossibility theorem. The proof of this result is obtained by formulating the problem in propositional logic and then letting a SAT solver show that the formula is unsatisfiable. Additionally, we demonstrate how to circumvent this impossibility by considering a weakening of strategyproofness which requires that only voters who do not approve any elected party cannot manipulate. While most common voting rules fail even this weak notion of strategyproofness, we characterize Chamberlin-Courant approval voting within the class of Thiele rules based on this strategyproofness notion.

        ----

        ## [625] Tight Inapproximability for Graphical Games

        **Authors**: *Argyrios Deligkas, John Fearnley, Alexandros Hollender, Themistoklis Melissourgos*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25695](https://doi.org/10.1609/aaai.v37i5.25695)

        **Abstract**:

        We provide a complete characterization for the computational complexity of finding approximate equilibria in two-action graphical games. We consider the two most well-studied approximation notions: ε-Nash equilibria (ε-NE) and ε-well-supported Nash equilibria (ε-WSNE), where ε is in [0,1]. We prove that computing an ε-NE is PPAD-complete for any constant ε smaller than 1/2, while a very simple algorithm (namely, letting all players mix uniformly between their two actions) yields a 1/2-NE. On the other hand, we show that computing an ε-WSNE is PPAD-complete for any constant ε smaller than 1, while a 1-WSNE is trivial to achieve, because any strategy profile is a 1-WSNE. All of our lower bounds immediately also apply to graphical games with more than two actions per player.

        ----

        ## [626] From Monopoly to Competition: Optimal Contests Prevail

        **Authors**: *Xiaotie Deng, Yotam Gafni, Ron Lavi, Tao Lin, Hongyi Ling*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25696](https://doi.org/10.1609/aaai.v37i5.25696)

        **Abstract**:

        We study competition among contests in a general model that allows for an arbitrary and heterogeneous space of contest design and symmetric contestants. The goal of the contest designers is to maximize the contestants' sum of efforts. Our main result shows that optimal contests in the monopolistic setting (i.e., those that maximize the sum of efforts in a model with a single contest) form an equilibrium in the model with competition among contests. Under a very natural assumption these contests are in fact dominant, and the equilibria that they form are unique. Moreover, equilibria with the optimal contests are Pareto-optimal even in cases where other equilibria emerge. In many natural cases, they also maximize the social welfare.

        ----

        ## [627] Commitment Games with Conditional Information Disclosure

        **Authors**: *Anthony DiGiovanni, Jesse Clifton*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25697](https://doi.org/10.1609/aaai.v37i5.25697)

        **Abstract**:

        The conditional commitment abilities of mutually transparent computer agents have been studied in previous work on commitment games and program equilibrium. This literature has shown how these abilities can help resolve Prisoner’s Dilemmas and other failures of cooperation in complete information settings. But inefficiencies due to private information have been neglected thus far in this literature, despite the fact that these problems are pervasive and might also be addressed by greater mutual transparency. In this work, we introduce a framework for commitment games with a new kind of conditional commitment device, which agents can use to conditionally disclose private information. We prove a folk theorem for this setting that provides sufficient conditions for ex post efficiency, and thus represents a model of ideal cooperation between agents without a third-party mediator. Further, extending previous work on program equilibrium, we develop an implementation of conditional information disclosure. We show that this implementation forms program ε-Bayesian Nash equilibria corresponding to the Bayesian Nash equilibria of these commitment games.

        ----

        ## [628] Rawlsian Fairness in Online Bipartite Matching: Two-Sided, Group, and Individual

        **Authors**: *Seyed A. Esmaeili, Sharmila Duppala, Davidson Cheng, Vedant Nanda, Aravind Srinivasan, John P. Dickerson*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25698](https://doi.org/10.1609/aaai.v37i5.25698)

        **Abstract**:

        Online bipartite-matching platforms are ubiquitous and find applications in important areas such as crowdsourcing and ridesharing. In the most general form, the platform consists of three entities: two sides to be matched and a platform operator that decides the matching. The design of algorithms for such platforms has traditionally focused on the operator’s (expected) profit. Since fairness has become an important consideration that was ignored in the existing algorithms a collection of online matching algorithms have been developed that give a fair treatment guarantee for one side of the market at the expense of a drop in the operator’s profit. In this paper, we generalize the existing work to offer fair treatment guarantees to both sides of the market simultaneously, at a calculated worst case drop to operator profit. We consider group and individual Rawlsian fairness criteria. Moreover, our algorithms have theoretical guarantees and have adjustable parameters that can be tuned as desired to balance the trade-off between the utilities of the three sides. We also derive hardness results that give clear upper bounds over the performance of any algorithm.

        ----

        ## [629] Participatory Budgeting Designs for the Real World

        **Authors**: *Roy Fairstein, Gerdus Benadè, Kobi Gal*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25699](https://doi.org/10.1609/aaai.v37i5.25699)

        **Abstract**:

        Participatory budgeting engages the public in the process of allocating public money to different types of projects. PB designs differ in how voters are asked to express their preferences over candidate projects and how these preferences are aggregated to determine which projects to fund. This paper studies two fundamental questions in PB design. Which voting format and aggregation method to use, and how to evaluate the outcomes of these design decisions? We conduct an extensive empirical study in which 1 800 participants vote in four participatory budgeting elections in a controlled setting to evaluate the practical effects of the choice of voting format and aggregation rule.We find that k-approval leads to the best user experience. With respect to the aggregation rule, greedy aggregation leads to outcomes that are highly sensitive to the input format used and the fraction of the population that participates. The method of equal shares, in contrast, leads to outcomes that are not sensitive to the type of voting format used, and these outcomes are remarkably stable even when the majority of the population does not participate in the election. These results carry valuable insights for PB practitioners and social choice researchers.

        ----

        ## [630] PAC Learning and Stabilizing Hedonic Games: Towards a Unifying Approach

        **Authors**: *Simone Fioravanti, Michele Flammini, Bojana Kodric, Giovanna Varricchio*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25700](https://doi.org/10.1609/aaai.v37i5.25700)

        **Abstract**:

        We study PAC learnability and PAC stabilizability of Hedonic Games (HGs), i.e., efficiently inferring preferences or core-stable partitions from samples. We first expand the known learnability/stabilizability landscape for some of the most prominent HGs classes, providing results for Friends and Enemies Games, Bottom Responsive, and Anonymous HGs. Then, having a broader view in mind, we attempt to shed light on the structural properties leading to learnability/stabilizability, or lack thereof, for specific HGs classes. Along this path, we focus on the fully expressive Hedonic Coalition Nets representation of HGs. We identify two sets of conditions that lead to efficient learnability, and which encompass all of the known positive learnability results. On the side of stability, we reveal that, while the freedom of choosing an ad hoc adversarial distribution is the most obvious hurdle to achieving PAC stability, it is not the only one. First, we show a distribution independent necessary condition for PAC stability. Then, we focus on W-games, where players have individual preferences over other players and evaluate coalitions based on the least preferred member. We prove that these games are PAC stabilizable under the class of bounded distributions, which assign positive probability mass to all coalitions. Finally, we discuss why such a result is not easily extendable to other HGs classes even in this promising scenario. Namely, we establish a purely computational property necessary for achieving PAC stability.

        ----

        ## [631] Scalable Edge Blocking Algorithms for Defending Active Directory Style Attack Graphs

        **Authors**: *Mingyu Guo, Max Ward, Aneta Neumann, Frank Neumann, Hung Nguyen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25701](https://doi.org/10.1609/aaai.v37i5.25701)

        **Abstract**:

        Active Directory (AD) is the default security management system for Windows domain networks.  An AD environment naturally describes an attack graph where nodes represent computers/accounts/security groups, and edges represent existing accesses/known exploits that allow the attacker to gain access from one node to another.  Motivated by practical AD use cases, we study a Stackelberg game between one attacker and one defender.  There are multiple entry nodes for the attacker to choose from and there is a single target (Domain Admin).  Every edge has a failure rate.  The attacker chooses the attack path with the maximum success rate.  The defender can block a limited number of edges (i.e., revoke accesses) from a set of blockable edges, limited by budget. The defender's aim is to minimize the attacker's success rate.

We exploit the tree-likeness of practical AD graphs to design scalable algorithms.  We propose two novel methods that combine theoretical fixed parameter analysis and practical optimisation techniques.

For graphs with small tree widths, we propose a tree decomposition based dynamic program.  We then propose a general method for converting tree decomposition based dynamic programs to reinforcement learning environments, which leads to an anytime algorithm that scales better, but loses the optimality guarantee.

For graphs with small numbers of non-splitting paths (a parameter we invent specifically for AD graphs), we propose a kernelization technique that significantly downsizes the model, which is then solved via mixed-integer programming.

Experimentally, our algorithms scale to handle synthetic AD graphs with tens of thousands of nodes.

        ----

        ## [632] Representation with Incomplete Votes

        **Authors**: *Daniel Halpern, Gregory Kehne, Ariel D. Procaccia, Jamie Tucker-Foltz, Manuel Wüthrich*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25702](https://doi.org/10.1609/aaai.v37i5.25702)

        **Abstract**:

        Platforms for online civic participation rely heavily on methods for condensing thousands of comments into a relevant handful, based on whether participants agree or disagree with them. These methods should guarantee fair representation of the participants, as their outcomes may affect the health of the conversation and inform impactful downstream decisions. To that end, we draw on the literature on approval-based committee elections. Our setting is novel in that the approval votes are incomplete since participants will typically not vote on all comments. We prove that this complication renders non-adaptive algorithms impractical in terms of the amount of information they must gather. Therefore, we develop an adaptive algorithm that uses information more efficiently by presenting incoming participants with statements that appear promising based on votes by previous participants. We prove that this method satisfies commonly used notions of fair representation, even when participants only vote on a small fraction of comments. Finally, an empirical evaluation using real data shows that the proposed algorithm provides representative outcomes in practice.

        ----

        ## [633] Optimizing Multiple Simultaneous Objectives for Voting and Facility Location

        **Authors**: *Yue Han, Christopher Jerrett, Elliot Anshelevich*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25703](https://doi.org/10.1609/aaai.v37i5.25703)

        **Abstract**:

        We study the classic facility location setting, where we are given n clients and m possible facility locations in some arbitrary metric space, and want to choose a location to build a facility. The exact same setting also arises in spatial social choice, where voters are the clients and the goal is to choose a candidate or outcome, with the distance from a voter to an outcome representing the cost of this outcome for the voter (e.g., based on their ideological differences). Unlike most previous work, we do not focus on a single objective to optimize (e.g., the total distance from clients to the facility, or the maximum distance, etc.), but instead attempt to optimize several different objectives simultaneously. More specifically, we consider the l-centrum family of objectives, which includes the total distance, max distance, and many others. We present tight bounds on how well any pair of such objectives (e.g., max and sum) can be simultaneously approximated compared to their optimum outcomes. In particular, we show that for any such pair of objectives, it is always possible to choose an outcome which simultaneously approximates both objectives within a factor of 1 plus square root of 2, and give a precise characterization of how this factor improves as the two objectives being optimized become more similar. For q>2 different centrum objectives, we show that it is always possible to approximate all q of these objectives within a small constant, and that this constant approaches 3 as q increases. Our results show that when optimizing only a few simultaneous objectives, it is always possible to form an outcome which is a significantly better than 3 approximation for all of these objectives.

        ----

        ## [634] Class Fairness in Online Matching

        **Authors**: *Hadi Hosseini, Zhiyi Huang, Ayumi Igarashi, Nisarg Shah*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25704](https://doi.org/10.1609/aaai.v37i5.25704)

        **Abstract**:

        We initiate the study of fairness among classes of agents in online bipartite matching where there is a given set of offline vertices (aka agents) and another set of vertices (aka items) that arrive online and must be matched irrevocably upon arrival. In this setting, agents are partitioned into a set of classes and the matching is required to be fair with respect to the classes. We adopt popular fairness notions (e.g. envy-freeness, proportionality, and maximin share) and their relaxations to this setting and study deterministic and randomized algorithms for matching indivisible items (leading to integral matchings) and for matching divisible items (leading to fractional matchings).
For matching indivisible items, we propose an adaptive-priority-based algorithm, MATCH-AND-SHIFT, prove that it achieves (1/2)-approximation of both class envy-freeness up to one item and class maximin share fairness, and show that each guarantee is tight. For matching divisible items, we design a water-filling-based algorithm, EQUAL-FILLING, that achieves (1-1/e)-approximation of class envy-freeness and class proportionality; we prove (1-1/e) to be tight for class proportionality and establish a 3/4 upper bound on class envy-freeness.

        ----

        ## [635] How to Cut a Discrete Cake Fairly

        **Authors**: *Ayumi Igarashi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25705](https://doi.org/10.1609/aaai.v37i5.25705)

        **Abstract**:

        Cake-cutting is a fundamental model of dividing a heterogeneous resource, such as land, broadcast time, and advertisement space. In this study, we consider the problem of dividing indivisible goods fairly under the connectivity constraints of a path. We prove that a connected division of indivisible items satisfying a discrete counterpart of envy-freeness, called envy-freeness up to one good (EF1), always exists for any number of agents n with monotone valuations. Our result settles an open question raised by Bilò et al. (2019), who proved that an EF1 connected division always exists for four agents with monotone valuations. Moreover, the proof can be extended to show the following (1) ``secretive" and (2) ``extra" versions: (1) for n agents with monotone valuations, the path can be divided into n connected bundles such that an EF1 assignment of the remaining bundles can be made to the other agents for any selection made by the “secretive agent”; (2) for n+1 agents with monotone valuations, the path can be divided into n connected bundles such that when any ``extra agent” leaves, an EF1 assignment of the bundles can be made to the remaining agents.

        ----

        ## [636] Competition, Alignment, and Equilibria in Digital Marketplaces

        **Authors**: *Meena Jagadeesan, Michael I. Jordan, Nika Haghtalab*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25706](https://doi.org/10.1609/aaai.v37i5.25706)

        **Abstract**:

        Competition between traditional platforms is known to improve user utility by aligning the platform's actions with user preferences. But to what extent is alignment exhibited in data-driven marketplaces? To study this question from a theoretical perspective, we introduce a duopoly market where platform actions are bandit algorithms and the two platforms compete for user participation. A salient feature of this market is that the quality of recommendations depends on both the bandit algorithm and the amount of data provided by interactions from users. This interdependency between the algorithm performance and the actions of users complicates the structure of market equilibria and their quality in terms of user utility. Our main finding is that competition in this market does not perfectly align market outcomes with user utility. Interestingly, market outcomes exhibit misalignment not only when the platforms have separate data repositories, but also when the platforms have a shared data repository. Nonetheless, the data sharing assumptions impact what mechanism drives misalignment and also affect the specific form of misalignment (e.g. the quality of the best-case and worst-case market outcomes). More broadly, our work illustrates that competition in digital marketplaces has subtle consequences for user utility that merit further investigation.

        ----

        ## [637] Voting with Preference Intensities

        **Authors**: *Anson Kahng, Mohamad Latifian, Nisarg Shah*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25707](https://doi.org/10.1609/aaai.v37i5.25707)

        **Abstract**:

        When an agent votes, she typically ranks the set of available alternatives. Occasionally, she may also wish to report the intensity of her preferences by indicating adjacent pairs of alternatives in her ranking between which her preference is acutely decisive; for instance, she may suggest that she likes alternative a more than b, but b much more than c. We design near-optimal voting rules which aggregate such preference rankings with intensities using the recently-popular distortion framework. We also show that traditional voting rules, which aggregate preference rankings while ignoring (or not eliciting) intensities, can incur significant welfare loss.

        ----

        ## [638] Approximations for Indivisible Concave Allocations with Applications to Nash Welfare Maximization

        **Authors**: *Nathaniel Kell, Kevin Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25708](https://doi.org/10.1609/aaai.v37i5.25708)

        **Abstract**:

        We study a general allocation setting where agent valuations are concave additive. In this model, a collection of items must be uniquely distributed among a set of agents, where each agent-item pair has a specified utility. The objective is to maximize the sum of agent valuations, each of which is an arbitrary non-decreasing concave function of the agent's total additive utility. This setting was studied by Devanur and Jain (STOC 2012) in the online setting for divisible items. In this paper, we obtain both multiplicative and additive approximations in the offline setting for indivisible items. Our approximations depend on novel parameters that measure the local multiplicative/additive curvatures of each agent valuation, which we show correspond directly to the integrality gap of the natural assignment convex program of the problem. Furthermore, we extend our additive guarantees to obtain constant multiplicative approximations for Asymmetric Nash Welfare Maximization when agents have smooth valuations. This algorithm also yields an interesting tatonnement-style interpretation, where agents adjust uniform prices and items are assigned according to maximum weighted bang-per-buck ratios.

        ----

        ## [639] Strategic Facility Location with Clients That Minimize Total Waiting Time

        **Authors**: *Simon Krogmann, Pascal Lenzner, Alexander Skopalik*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25709](https://doi.org/10.1609/aaai.v37i5.25709)

        **Abstract**:

        We study a non-cooperative two-sided facility location game in which facilities and clients behave strategically.
This is in contrast to many other facility location games in which clients simply visit their closest facility.
Facility agents select a location on a graph to open a facility to attract as much purchasing power as possible, while client agents choose which facilities to patronize by strategically distributing their purchasing power in order to minimize their total waiting time. Here, the waiting time of a facility depends on its received total purchasing power.    
We show that our client stage is an atomic splittable congestion game, which implies existence, uniqueness and efficient computation of a client equilibrium.
Therefore, facility agents can efficiently predict client behavior and make strategic decisions accordingly.
Despite that, we prove that subgame perfect equilibria do not exist in all instances of this game and that their existence is NP-hard to decide.
On the positive side, we provide a simple and efficient algorithm to compute 3-approximate subgame perfect equilibria.

        ----

        ## [640] Proportional Decisions in Perpetual Voting

        **Authors**: *Martin Lackner, Jan Maly*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25710](https://doi.org/10.1609/aaai.v37i5.25710)

        **Abstract**:

        Perpetual voting is a framework for long-term collective decision making. In this framework, we consider a sequence of subsequent approval-based elections and try to achieve a fair overall outcome. To achieve fairness over time, perpetual voting rules take the history of previous decisions into account and identify voters that were dissatisfied with previous decisions. In this paper, we look at perpetual voting rules from an axiomatic perspective. First, we define two classes of perpetual voting rules that are particularly easy to explain to voters and explore the bounds imposed by this simplicity. Second, we study proportionality in the perpetual setting and identify two rules with strong proportionality guarantees. However, both rules yield different guarantees and we prove them to be incompatible with each other.

        ----

        ## [641] Multiagent MST Cover: Pleasing All Optimally via a Simple Voting Rule

        **Authors**: *Bo Li, Xiaowei Wu, Chenyang Xu, Ruilong Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25711](https://doi.org/10.1609/aaai.v37i5.25711)

        **Abstract**:

        Given a connected graph on whose edges we can build roads to connect the nodes, a number of agents hold possibly different perspectives on which edges should be selected by assigning different edge weights. Our task is to build a minimum number of roads so that every agent has a spanning tree in the built subgraph whose weight is the same as a minimum spanning tree in the original graph. We first show that this problem is NP-hard and does not admit better than ((1-o(1)) ln k)-approximation polynomial-time algorithms unless P = NP, where k is the number of agents. We then give a simple voting algorithm with an optimal approximation ratio. Moreover, our algorithm only needs to access the agents' rankings on the edges. Finally, we extend our problem to submodular objective functions and Matroid rank constraints.

        ----

        ## [642] When Congestion Games Meet Mobile Crowdsourcing: Selective Information Disclosure

        **Authors**: *Hongbo Li, Lingjie Duan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25712](https://doi.org/10.1609/aaai.v37i5.25712)

        **Abstract**:

        In congestion games, users make myopic routing decisions to jam each other, and the social planner with the full information designs mechanisms on information or payment side to regulate. However, it is difficult to obtain time-varying traffic conditions, and emerging crowdsourcing platforms (e.g., Waze and Google Maps) provide a convenient way for mobile users travelling on the paths to learn and share the traffic conditions over time. When congestion games meet mobile crowdsourcing, it is critical to incentive selfish users to change their myopic routing policy and reach the best exploitation-exploration trade-off. By considering a simple but fundamental parallel routing network with one deterministic path and multiple stochastic paths for atomic users, we prove that the myopic routing policy's price of anarchy (PoA) can be arbitrarily large as the discount factor approaches 1. To remedy such huge efficiency loss, we propose a selective information disclosure (SID) mechanism: we only reveal the latest traffic information to users when they intend to over-explore the stochastic paths, while hiding such information when they want to under-explore. We prove that our mechanism reduces PoA to less than 2. Besides the worst-case performance, we further examine our mechanism's average-case performance by using extensive simulations.

        ----

        ## [643] Partitioning Friends Fairly

        **Authors**: *Lily Li, Evi Micha, Aleksandar Nikolov, Nisarg Shah*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25713](https://doi.org/10.1609/aaai.v37i5.25713)

        **Abstract**:

        We consider the problem of partitioning n agents in an undirected social network into k almost equal in size (differing by at most one) groups, where the utility of an agent for a group is the number of her neighbors in the group. The core and envy-freeness are two compelling axiomatic fairness guarantees in such settings. The former demands that there be no coalition of agents such that each agent in the coalition has more utility for that coalition than for her own group, while the latter demands that no agent envy another agent for the group they are in. We provide (often tight) approximations to both fairness guarantees, and many of our positive results are obtained via efficient algorithms.

        ----

        ## [644] Differentially Private Condorcet Voting

        **Authors**: *Zhechen Li, Ao Liu, Lirong Xia, Yongzhi Cao, Hanpin Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25714](https://doi.org/10.1609/aaai.v37i5.25714)

        **Abstract**:

        Designing private voting rules is an important and pressing problem for trustworthy democracy. In this paper, under the framework of differential privacy, we propose a novel famliy of randomized voting rules based on the well-known Condorcet method, and focus on three classes of voting rules in this family: Laplacian Condorcet method (CMLAP), exponential Condorcet method (CMEXP), and randomized response Condorcet method (CMRR), where λ represents the level of noise. We prove that all of our rules satisfy absolute monotonicity, lexi-participation, probabilistic Pareto efficiency, approximate probabilistic Condorcet criterion, and approximate SD-strategyproofness. In addition, CMRR satisfies (non-approximate) probabilistic Condorcet criterion, while CMLAP and CMEXP satisfy strong lexi-participation. Finally, we regard differential privacy as a voting axiom, and discuss its relations to other axioms.

        ----

        ## [645] Function Approximation for Solving Stackelberg Equilibrium in Large Perfect Information Games

        **Authors**: *Chun Kai Ling, J. Zico Kolter, Fei Fang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25715](https://doi.org/10.1609/aaai.v37i5.25715)

        **Abstract**:

        Function approximation (FA) has been a critical component in solving large zero-sum games. Yet, little attention has been given towards FA in solving general-sum extensive-form games, despite them being widely regarded as being computationally more challenging than their fully competitive  or cooperative counterparts. A key challenge is that for many equilibria in general-sum games, no simple analogue to the state value function used in Markov Decision Processes and zero-sum games exists. In this paper, we propose learning the Enforceable Payoff Frontier (EPF)---a generalization of the state value function for general-sum games. We approximate the optimal Stackelberg extensive-form correlated equilibrium by representing EPFs with neural networks and training them by using appropriate backup operations and loss functions. This is the first method that applies FA to the Stackelberg setting, allowing us to scale to much larger games while still enjoying performance guarantees based on FA error. Additionally, our proposed method guarantees incentive compatibility and is easy to evaluate without having to depend on self-play or approximate best-response oracles.

        ----

        ## [646] Optimal Pricing Schemes for Identical Items with Time-Sensitive Buyers

        **Authors**: *Zhengyang Liu, Liang Shan, Zihe Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25716](https://doi.org/10.1609/aaai.v37i5.25716)

        **Abstract**:

        Time or money? That is a question! In this paper, we consider this dilemma in the pricing regime, in which we try to find the optimal pricing scheme for identical items with heterogenous time-sensitive buyers. We characterize the revenue-optimal solution and propose an efficient algorithm to find it in a Bayesian setting. Our results also demonstrate the tight ratio between the value of wasted time and the seller's revenue, as well as that of two common-used pricing schemes, the k-step function and the fixed pricing. To explore the nature of the optimal scheme in the general setting, we present the closed forms over the product distribution and show by examples that positive correlation between the valuation of the item and the cost per unit time could help increase revenue. To the best of our knowledge, it is the first step towards understanding the impact of the time factor as a part of the buyer cost in pricing problems, in the computational view.

        ----

        ## [647] Approval-Based Voting with Mixed Goods

        **Authors**: *Xinhang Lu, Jannik Peters, Haris Aziz, Xiaohui Bei, Warut Suksompong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25717](https://doi.org/10.1609/aaai.v37i5.25717)

        **Abstract**:

        We consider a voting scenario in which the resource to be voted upon may consist of both indivisible and divisible goods. This generalizes both the well-studied model of multiwinner voting and the recently introduced model of cake sharing. Under approval votes, we propose two variants of the extended justified representation (EJR) notion from multiwinner voting, a stronger one called EJR for mixed goods (EJR-M) and a weaker one called EJR up to 1 (EJR-1). We extend three multiwinner voting rules to our setting—GreedyEJR, the method of equal shares (MES), and proportional approval voting (PAV)—and show that while all three generalizations satisfy EJR-1, only the first one provides EJR-M. In addition, we derive tight bounds on the proportionality degree implied by EJR-M and EJR-1, and investigate the proportionality degree of our proposed rules.

        ----

        ## [648] Utility Maximizer or Value Maximizer: Mechanism Design for Mixed Bidders in Online Advertising

        **Authors**: *Hongtao Lv, Zhilin Zhang, Zhenzhe Zheng, Jinghan Liu, Chuan Yu, Lei Liu, Lizhen Cui, Fan Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25718](https://doi.org/10.1609/aaai.v37i5.25718)

        **Abstract**:

        Digital advertising constitutes one of the main revenue sources for online platforms. In recent years, some advertisers tend to adopt auto-bidding tools to facilitate advertising performance optimization, making the classical utility maximizer model in auction theory not fit well. Some recent studies proposed a new model, called value maximizer, for auto-bidding advertisers with return-on-investment (ROI) constraints. However, the model of either utility maximizer or value maximizer could only characterize partial advertisers in real-world advertising platforms. In a mixed environment where utility maximizers and value maximizers coexist, the truthful ad auction design would be challenging since bidders could manipulate both their values and affiliated classes, leading to a multi-parameter mechanism design problem. In this work, we address this issue by proposing a payment rule which combines the corresponding ones in classical VCG and GSP mechanisms in a novel way. Based on this payment rule, we propose a truthful auction mechanism with an approximation ratio of 2 on social welfare, which is close to the lower bound of at least 5/4 that we also prove. The designed auction mechanism is a generalization of VCG for utility maximizers and GSP for value maximizers.

        ----

        ## [649] Facility Location Games with Entrance Fees

        **Authors**: *Mengfan Ma, Mingyu Xiao, Tian Bai, Bakh Khoussainov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25719](https://doi.org/10.1609/aaai.v37i5.25719)

        **Abstract**:

        The facility location game is an extensively studied problem in mechanism design. In the classical model, the cost of each agent is her distance to the nearest facility. In this paper, we consider a novel model where each facility charges an entrance fee, which is a function of the facility's location. Thus, in our model, the cost of each agent is the sum of the distance to the facility and the entrance fee of the facility. The generalized model captures more real-life scenarios. In our model, the entrance fee function can be an arbitrary function, and the corresponding preferences of agents may not be single-peaked anymore: this makes the problem complex and requires new techniques in the analysis. We systematically study the model and design strategyproof mechanisms with nice approximation ratios and also complement these with nearly-tight impossibility results. Specifically, for one-facility and two-facility games, we provide upper and lower bounds for the approximation ratios given by deterministic and randomized mechanisms, with respect to the utilitarian and egalitarian objectives. Most of our bounds are tight, and these bounds are independent of the entrance fee functions. Our results also match the results of the classical model.

        ----

        ## [650] Securing Lifelines: Safe Delivery of Critical Services in Areas with Volatile Security Situation via a Stackelberg Game Approach

        **Authors**: *Tien Mai, Arunesh Sinha*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25720](https://doi.org/10.1609/aaai.v37i5.25720)

        **Abstract**:

        Vaccine delivery in under-resourced locations with security risks is not just challenging but also life threatening. The COVID pandemic and the need to vaccinate added even more urgency to this issue. Motivated by this problem, we propose a general framework to set-up limited temporary (vaccination) centers that balance physical security and desired (vaccine) service coverage with limited resources. We set-up the problem as a Stackelberg game between the centers operator (defender) and an adversary, where the set of centers is not fixed a priori but is part of the decision output. This results in a mixed combinatorial and continuous optimization problem. As part of our scalable approximation solution, we provide a fundamental contribution by identifying general duality conditions of switching max and min when both discrete and continuous variables are involved. Via detailed experiments, we show that the solution proposed is scalable in practice.

        ----

        ## [651] Differentially Private Fair Division

        **Authors**: *Pasin Manurangsi, Warut Suksompong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25721](https://doi.org/10.1609/aaai.v37i5.25721)

        **Abstract**:

        Fairness and privacy are two important concerns in social decision-making processes such as resource allocation. We study privacy in the fair allocation of indivisible resources using the well-established framework of differential privacy. We present algorithms for approximate envy-freeness and proportionality when two instances are considered to be adjacent if they differ only on the utility of a single agent for a single item. On the other hand, we provide strong negative results for both fairness criteria when the adjacency notion allows the entire utility function of a single agent to change.

        ----

        ## [652] An Efficient Deep Reinforcement Learning Algorithm for Solving Imperfect Information Extensive-Form Games

        **Authors**: *Linjian Meng, Zhenxing Ge, Pinzhuo Tian, Bo An, Yang Gao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25722](https://doi.org/10.1609/aaai.v37i5.25722)

        **Abstract**:

        One of the most popular methods for learning Nash equilibrium (NE) in large-scale imperfect information extensive-form games (IIEFGs) is the neural variants of counterfactual regret minimization (CFR). CFR is a special case of Follow-The-Regularized-Leader (FTRL). At each iteration, the neural variants of CFR update the agent's strategy via the estimated counterfactual regrets. Then, they use neural networks to approximate the new strategy, which incurs an approximation error. These approximation errors will accumulate since the counterfactual regrets at iteration t are estimated using the agent's past approximated strategies. Such accumulated approximation error causes poor performance. To address this accumulated approximation error, we propose a novel FTRL algorithm called FTRL-ORW, which does not utilize the agent's past strategies to pick the next iteration strategy. More importantly, FTRL-ORW can update its strategy via the trajectories sampled from the game, which is suitable to solve large-scale IIEFGs since sampling multiple actions for each information set is too expensive in such games. However, it remains unclear which algorithm to use to compute the next iteration strategy for FTRL-ORW when only such sampled trajectories are revealed at iteration t. To address this problem and scale FTRL-ORW to large-scale games, we provide a model-free method called Deep FTRL-ORW, which computes the next iteration strategy using model-free Maximum Entropy Deep Reinforcement Learning. Experimental results on two-player zero-sum IIEFGs show that Deep FTRL-ORW significantly outperforms existing model-free neural methods and OS-MCCFR.

        ----

        ## [653] Fast and Interpretable Dynamics for Fisher Markets via Block-Coordinate Updates

        **Authors**: *Tianlong Nan, Yuan Gao, Christian Kroer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25723](https://doi.org/10.1609/aaai.v37i5.25723)

        **Abstract**:

        We consider the problem of large-scale Fisher market equilibrium computation through scalable first-order optimization methods. It is well-known that market equilibria can be captured using structured convex programs such as the Eisenberg-Gale and Shmyrev convex programs. Highly performant deterministic full-gradient first-order methods have been developed for these programs. In this paper, we develop new block-coordinate first-order methods for computing Fisher market equilibria, and show that these methods have interpretations as tâtonnement-style or proportional response-style dynamics where either buyers or items show up one at a time. We reformulate these convex programs and solve them using proximal block coordinate descent methods, a class of methods that update only a small number of coordinates of the decision variable in each iteration. Leveraging recent advances in the convergence analysis of these methods and structures of the equilibrium-capturing convex programs, we establish fast convergence rates of these methods.

        ----

        ## [654] Ballot Length in Instant Runoff Voting

        **Authors**: *Kiran Tomlinson, Johan Ugander, Jon M. Kleinberg*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25724](https://doi.org/10.1609/aaai.v37i5.25724)

        **Abstract**:

        Instant runoff voting (IRV) is an increasingly-popular alternative to traditional plurality voting in which voters submit rankings over the candidates rather than single votes. In practice, elections using IRV often restrict the ballot length, the number of candidates a voter is allowed to rank on their ballot. We theoretically and empirically analyze how ballot length can influence the outcome of an election, given fixed voter preferences. We show that there exist preference profiles over k candidates such that up to k-1 different candidates win at different ballot lengths. We derive exact lower bounds on the number of voters required for such profiles and provide a construction matching the lower bound for unrestricted voter preferences. Additionally, we characterize which sequences of winners are possible over ballot lengths and provide explicit profile constructions achieving any feasible winner sequence. We also examine how classic preference restrictions influence our results—for instance, single-peakedness makes k-1 different winners impossible but still allows at least Ω(√k). Finally, we analyze a collection of 168 real-world elections, where we truncate rankings to simulate shorter ballots. We find that shorter ballots could have changed the outcome in one quarter of these elections. Our results highlight ballot length as a consequential degree of freedom in the design of IRV elections.

        ----

        ## [655] Multi-Stage Facility Location Problems with Transient Agents

        **Authors**: *Xuezhen Wang, Vincent Chau, Hau Chan, Ken C. K. Fong, Minming Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25725](https://doi.org/10.1609/aaai.v37i5.25725)

        **Abstract**:

        We study various models for the one-dimensional multi-stage facility location problems with transient agents, where a transient agent arrives in some stage and stays for a number of consecutive stages. In the problems, we need to serve each agent in one of their stages by determining the location of the facility at each stage. In the first model, we assume there is no cost for moving the facility across the stages. We focus on optimal algorithms to minimize both the social cost objective, defined as the total distance of all agents to the facility over all stages, and the maximum cost objective, defined as the max distance of any agent to the facility over all stages. For each objective, we give a slice-wise polynomial (XP) algorithm (i.e., solvable in m^f(k) for some fixed parameter k and computable function f, where m is the input size) and show that there is a polynomial-time algorithm when a natural first-come-first-serve (FCFS) order of agent serving is enforced. We then consider the mechanism design problem, where the agents' locations and arrival stages are private, and design a group strategy-proof mechanism that achieves good approximation ratios for both objectives and settings with and without FCFS ordering. In the second model, we consider the facility's moving cost between adjacent stages under the social cost objective, which accounts for the total moving distance of the facility. Correspondingly, we design XP (and polynomial time) algorithms and a group strategy-proof mechanism for settings with or without the FCFS ordering.

        ----

        ## [656] Bayesian Optimization-Based Combinatorial Assignment

        **Authors**: *Jakob Weissteiner, Jakob Heiss, Julien Siems, Sven Seuken*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25726](https://doi.org/10.1609/aaai.v37i5.25726)

        **Abstract**:

        We study the combinatorial assignment domain, which includes combinatorial auctions and course allocation. The main challenge in this domain is that the bundle space grows exponentially in the number of items. To address this, several papers have recently proposed machine learning-based preference elicitation algorithms that aim to elicit only the most important information from agents. However, the main shortcoming of this prior work is that it does not model a mechanism's uncertainty over values for not yet elicited bundles. In this paper, we address this shortcoming by presenting a Bayesian optimization-based combinatorial assignment (BOCA) mechanism. Our key technical contribution is to integrate a method for capturing model uncertainty into an iterative combinatorial auction mechanism. Concretely, we design a new method for estimating an upper uncertainty bound that can be used to define an acquisition function to determine the next query to the agents. This enables the mechanism to properly explore (and not just exploit) the bundle space during its preference elicitation phase. We run computational experiments in several spectrum auction domains to evaluate BOCA's performance. Our results show that BOCA achieves higher allocative efficiency than state-of-the-art approaches.

        ----

        ## [657] Semi-random Impossibilities of Condorcet Criterion

        **Authors**: *Lirong Xia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25727](https://doi.org/10.1609/aaai.v37i5.25727)

        **Abstract**:

        The Condorcet criterion (CC) is a classical and well-accepted criterion for voting. Unfortunately, it is incompatible with many other desiderata including participation (PAR), half-way monotonicity (HM), Maskin monotonicity (MM), and strategy-proofness (SP). Such incompatibilities are often known as impossibility theorems, and are proved by worst-case analysis. Previous work has investigated the likelihood for these impossibilities to occur under certain models, which are often criticized of being unrealistic.

We strengthen previous work by proving the first set of semi-random impossibilities for voting rules to satisfy CC and the more general, group versions of the four desiderata: for any sufficiently large number of voters n, any size of the group 1<= B<= \sqrt n, any voting rule r, and under a large class of semi-random models that include Impartial Culture, the likelihood for r to satisfy CC and PAR, CC and HM, CC and MM, or CC and SP  is 1-\Omega(B/\sqrt n). This matches existing lower bounds for CC&PAR (B=1) and CC&SP and CC&HM (B<=\sqrt n), showing that many commonly-studied voting rules  are already asymptotically optimal in such cases.

        ----

        ## [658] Tournament Fixing Parameterized by Feedback Vertex Set Number Is FPT

        **Authors**: *Meirav Zehavi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25728](https://doi.org/10.1609/aaai.v37i5.25728)

        **Abstract**:

        A knockout (or single-elimination) tournament is a format of a competition that is very popular in practice (particularly in sports, elections and decision making), and which has been extensively and intensively studied from a theoretical point of view for more than a decade. Particular attention has been devoted to the Tournament Fixing problem, where, roughly speaking, the objective is to determine whether we can conduct the knockout tournament in a way that makes our favorite player win. Here, part of the input is a tournament graph D that encodes the winner of each possible match. A sequence of papers has studied the parameterized complexity of  Tournament Fixing with respect to the feedback arc set number (fas) of D Given that this parameter yielded tractability, it has been asked explicitly and repeatedly whether Tournament Fixing is FPT also with respect to the  feedback vertex set number (fvs) of D. We answer this question positively. In fact, although fvs can be arbitrarily smaller than fas, we attain the same dependency on the parameter in the time complexity. So, additionally, our work subsumes the best known algorithm for Tournament Fixing with respect to as.

        ----

        ## [659] Truthful Mechanisms for Steiner Tree Problems

        **Authors**: *Jinshan Zhang, Zhengyang Liu, Xiaotie Deng, Jianwei Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25729](https://doi.org/10.1609/aaai.v37i5.25729)

        **Abstract**:

        Consider an undirected graph G=(V,E) model for a communication network, where  each edge is owned by a selfish agent, who reports the cost for offering the use of her edge. Note that each edge agent may misreport her own cost for the use of the edge for her own benefit. In such a non-cooperative setting, we aim at designing an approximately truthful mechanism for establishing a Steiner tree, a minimum cost tree spanning over all the terminals. We present a truthful-in-expectation mechanism that achieves the approximation ratio ln 4 + ε ≈ 1.39, which matches the current best algorithmic ratio for STP.

        ----

        ## [660] Collusion-Proof and Sybil-Proof Reward Mechanisms for Query Incentive Networks

        **Authors**: *Youjia Zhang, Pingzhong Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25730](https://doi.org/10.1609/aaai.v37i5.25730)

        **Abstract**:

        This paper explores reward mechanisms for a query incentive network in which agents seek information from social networks. In a query tree issued by the task owner, each agent is rewarded by the owner for contributing to the solution, for instance, solving the task or inviting others to solve it. The reward mechanism determines the reward for each agent and motivates all agents to propagate and report their information truthfully. In particular, the reward cannot exceed the budget set by the task owner. However, our impossibility results demonstrate that a reward mechanism cannot simultaneously achieve Sybil-proof (agents benefit from manipulating multiple fake identities), collusion-proof (multiple agents pretend as a single agent to improve the reward), and other essential properties. In order to address these issues, we propose two novel reward mechanisms. The first mechanism achieves Sybil-proof and collusion-proof, respectively; the second mechanism sacrifices Sybil-proof to achieve the approximate versions of Sybil-proof and collusion-proof. Additionally, we show experimentally that our second reward mechanism outperforms the existing ones.

        ----

        ## [661] Fisher Markets with Social Influence

        **Authors**: *Jiayi Zhao, Denizalp Goktas, Amy Greenwald*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25731](https://doi.org/10.1609/aaai.v37i5.25731)

        **Abstract**:

        A Fisher market is an economic model of buyer and seller interactions in which each buyer’s utility depends only on the bundle of goods she obtains. Many people’s interests, however, are affected by their social interactions with others. In this paper, we introduce a generalization of Fisher markets, namely influence Fisher markets, which captures the impact of social influence on buyers’ utilities. We show that competitive equilibria in influence Fisher markets correspond to generalized Nash equilibria in an associated pseudo-game, which implies the existence of competitive equilibria in all influence Fisher markets with continuous and concave utility functions. We then construct a monotone pseudo-game, whose variational equilibria and their duals together characterize competitive equilibria in influence Fisher markets with continuous, jointly concave, and homogeneous utility functions. This observation implies that competitive equilibria in these markets can be computed in polynomial time under standard smoothness assumptions on the utility functions. The dual of this second pseudo-game enables us to interpret the competitive equilibria of influence CCH Fisher markets as the solutions to a system of simultaneous Stackelberg games. Finally, we derive a novel first-order method that solves this Stackelberg system in polynomial time, prove that it is equivalent to computing competitive equilibrium prices via tâtonnement, and run experiments that confirm our theoretical results.

        ----

        ## [662] Probably Approximate Shapley Fairness with Applications in Machine Learning

        **Authors**: *Zijian Zhou, Xinyi Xu, Rachael Hwee Ling Sim, Chuan Sheng Foo, Bryan Kian Hsiang Low*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25732](https://doi.org/10.1609/aaai.v37i5.25732)

        **Abstract**:

        The Shapley value (SV) is adopted in various scenarios in machine learning (ML), including data valuation, agent valuation, and feature attribution, as it satisfies their fairness requirements. However, as exact SVs are infeasible to compute in practice, SV estimates are approximated instead. This approximation step raises an important question: do the SV estimates preserve the fairness guarantees of exact SVs? We observe that the fairness guarantees of exact SVs are too restrictive for SV estimates. Thus, we generalise Shapley fairness to probably approximate Shapley fairness and propose fidelity score, a metric to measure the variation of SV estimates, that determines how probable the fairness guarantees hold. Our last theoretical contribution is a novel greedy active estimation (GAE) algorithm that will maximise the lowest fidelity score and achieve a better fairness guarantee than the de facto Monte-Carlo estimation. We empirically verify GAE outperforms several existing methods in guaranteeing fairness while remaining competitive in estimation accuracy in various ML scenarios using real-world datasets.

        ----

        ## [663] The Perils of Trial-and-Error Reward Design: Misdesign through Overfitting and Invalid Task Specifications

        **Authors**: *Serena Booth, W. Bradley Knox, Julie Shah, Scott Niekum, Peter Stone, Alessandro Allievi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25733](https://doi.org/10.1609/aaai.v37i5.25733)

        **Abstract**:

        In reinforcement learning (RL), a reward function that aligns exactly with a task's true performance metric is often necessarily sparse. For example, a true task metric might encode a reward of 1 upon success and 0 otherwise. The sparsity of these true task metrics can make them hard to learn from, so in practice they are often replaced with alternative dense reward functions. These dense reward functions are typically designed by experts through an ad hoc process of trial and error. In this process, experts manually search for a reward function that improves performance with respect to the task metric while also enabling an RL algorithm to learn faster. This process raises the question of whether the same reward function is optimal for all algorithms, i.e., whether the reward function can be overfit to a particular algorithm. In this paper, we study the consequences of this wide yet unexamined practice of trial-and-error reward design. We first conduct computational experiments that confirm that reward functions can be overfit to learning algorithms and their hyperparameters. We then conduct a controlled observation study which emulates expert practitioners' typical experiences of reward design, in which we similarly find evidence of reward function overfitting. We also find that experts' typical approach to reward design---of adopting a myopic strategy and weighing the relative goodness of each state-action pair---leads to misdesign through invalid task specifications, since RL algorithms use cumulative reward rather than rewards for individual state-action pairs as an optimization target.

Code, data: github.com/serenabooth/reward-design-perils

        ----

        ## [664] The Value of AI Guidance in Human Examination of Synthetically-Generated Faces

        **Authors**: *Aidan Boyd, Patrick Tinsley, Kevin W. Bowyer, Adam Czajka*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25734](https://doi.org/10.1609/aaai.v37i5.25734)

        **Abstract**:

        Face image synthesis has progressed beyond the point at which humans can effectively distinguish authentic faces from synthetically-generated ones. Recently developed synthetic face image detectors boast ``better-than-human'' discriminative ability, especially those guided by human perceptual intelligence during the model's training process. In this paper, we investigate whether these human-guided synthetic face detectors can assist non-expert human operators in the task of synthetic image detection when compared to models trained without human-guidance. We conducted a large-scale experiment with more than 1,560 subjects classifying whether an image shows an authentic or synthetically-generated face, and annotating regions supporting their decisions. In total, 56,015 annotations across 3,780 unique face images were collected. All subjects first examined samples without any AI support, followed by samples given (a) the AI's decision (``synthetic'' or ``authentic''), (b) class activation maps illustrating where the model deems salient for its decision, 
or (c) both the AI's decision and AI's saliency map. Synthetic faces were generated with six modern Generative Adversarial Networks. Interesting observations from this experiment include: (1) models trained with human-guidance, which are also more accurate in our experiments, offer better support to human examination of face images when compared to models trained traditionally using cross-entropy loss, (2) binary decisions presented to humans results in their better performance than when saliency maps are presented, (3) understanding the AI's accuracy helps humans to increase trust in a given model and thus increase their overall accuracy. This work demonstrates that although humans supported by machines achieve better-than-random accuracy of synthetic face detection, the approaches of supplying humans with AI support and of building trust are key factors determining high effectiveness of the human-AI tandem.

        ----

        ## [665] Teaching to Learn: Sequential Teaching of Learners with Internal States

        **Authors**: *Mustafa Mert Çelikok, Pierre-Alexandre Murena, Samuel Kaski*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25735](https://doi.org/10.1609/aaai.v37i5.25735)

        **Abstract**:

        In sequential machine teaching, a teacher’s objective is to provide the optimal sequence of inputs to sequential learners in order to guide them towards the best model. However, this teaching objective considers a restricted class of learners with fixed inductive biases. In this paper, we extend the machine teaching framework to learners that can improve their inductive biases, represented as latent internal states, in order to generalize to new datasets.
We introduce a novel framework in which learners’ inductive biases may change with the teaching interaction, which affects the learning performance in future tasks. In order to teach such learners, we propose a multi-objective control approach that takes the future performance of the learner after teaching into account. This framework provides tools for modelling learners with internal states, humans and meta-learning algorithms alike. Furthermore, we distinguish manipulative teaching, which can be done by effectively hiding data and also used for indoctrination, from teaching to learn which aims to help the learner become better at learning from new datasets in the absence of a teacher. Our empirical results demonstrate that our framework is able to reduce the number of required tasks for online meta-learning, and increases independent learning performance of simulated human users in future tasks.

        ----

        ## [666] Interactive Concept Bottleneck Models

        **Authors**: *Kushal Chauhan, Rishabh Tiwari, Jan Freyberg, Pradeep Shenoy, Krishnamurthy Dvijotham*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25736](https://doi.org/10.1609/aaai.v37i5.25736)

        **Abstract**:

        Concept bottleneck models (CBMs) are interpretable neural networks that first predict labels for human-interpretable concepts relevant to the prediction task, and then predict the final label based on the concept label predictions. We extend CBMs to interactive prediction settings where the model can query a human collaborator for the label to some concepts. We develop an interaction policy that, at prediction time, chooses which concepts to request a label for so as to maximally improve the final prediction. We demonstrate that a simple policy combining concept prediction uncertainty and influence of the concept on the final prediction achieves strong performance and outperforms static approaches as well as active feature acquisition methods proposed in the literature. We show that the interactive CBM can achieve accuracy gains of 5-10% with only 5 interactions over competitive baselines on the Caltech-UCSD Birds, CheXpert and OAI datasets.

        ----

        ## [667] Local Justice and Machine Learning: Modeling and Inferring Dynamic Ethical Preferences toward Allocations

        **Authors**: *Violet Xinying Chen, Joshua Williams, Derek Leben, Hoda Heidari*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25737](https://doi.org/10.1609/aaai.v37i5.25737)

        **Abstract**:

        We consider a setting in which a social planner has to make a sequence of decisions to allocate scarce resources in a high-stakes domain. Our goal is to understand stakeholders' dynamic moral preferences toward such allocational policies. In particular, we evaluate the sensitivity of moral preferences to the history of allocations and their perceived future impact on various socially salient groups.  We propose a mathematical model to capture and infer such dynamic moral preferences. We illustrate our model through small-scale human-subject experiments focused on the allocation of scarce medical resource distributions during a hypothetical viral epidemic. We observe that participants' preferences are indeed history- and impact-dependent. Additionally, our preliminary experimental results reveal intriguing patterns specific to medical resources---a topic that is particularly salient against the backdrop of the global covid-19 pandemic.

        ----

        ## [668] Extracting Semantic-Dynamic Features for Long-Term Stable Brain Computer Interface

        **Authors**: *Tao Fang, Qian Zheng, Yu Qi, Gang Pan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25738](https://doi.org/10.1609/aaai.v37i5.25738)

        **Abstract**:

        Brain-computer Interface (BCI) builds a neural signal to the motor command pathway, which is a prerequisite for the realization of neural prosthetics. However, a long-term stable BCI suffers from the neural data drift across days while retraining the BCI decoder is expensive and restricts its application scenarios. Recent solutions of neural signal recalibration treat the continuous neural signals as discrete, which is less effective in temporal feature extraction. Inspired by the observation from biologists that low-dimensional dynamics could describe high-dimensional neural signals, we model the underlying neural dynamics and propose a semantic-dynamic feature that represents the semantics and dynamics in a shared feature space facilitating the BCI recalibration. Besides, we present the joint distribution alignment instead of the common used marginal alignment strategy, dealing with the various complex changes in neural data distribution. Our recalibration approach achieves state-of-the-art performance on the real neural data of two monkeys in both classification and regression tasks. Our approach is also evaluated on a simulated dataset, which indicates its robustness in dealing with various common causes of neural signal instability.

        ----

        ## [669] Moral Machine or Tyranny of the Majority?

        **Authors**: *Michael Feffer, Hoda Heidari, Zachary C. Lipton*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25739](https://doi.org/10.1609/aaai.v37i5.25739)

        **Abstract**:

        With artificial intelligence systems increasingly applied in consequential domains, researchers have begun to ask how AI systems ought to act in ethically charged situations where even humans lack consensus. In the Moral Machine project, researchers crowdsourced answers to "Trolley Problems" concerning autonomous vehicles. Subsequently, Noothigattu et al. (2018) proposed inferring linear functions that approximate each individual's preferences and aggregating these linear models by averaging parameters across the population. In this paper, we examine this averaging mechanism, focusing on fairness concerns and strategic effects. We investigate a simple setting where the population consists of two groups, the minority constitutes an α < 0.5 share of the population, and within-group preferences are homogeneous. Focusing on the fraction of contested cases where the minority group prevails, we make the following observations: (a) even when all parties report their preferences truthfully, the fraction of disputes where the minority prevails is less than proportionate in α; (b) the degree of sub-proportionality grows more severe as the level of disagreement between the groups increases; (c) when parties report preferences strategically, pure strategy equilibria do not always exist; and (d) whenever a pure strategy equilibrium exists, the majority group prevails 100% of the time. These findings raise concerns about stability and fairness of averaging as a mechanism for aggregating diverging voices. Finally, we discuss alternatives, including randomized dictatorship and median-based mechanisms.

        ----

        ## [670] The Effect of Modeling Human Rationality Level on Learning Rewards from Multiple Feedback Types

        **Authors**: *Gaurav R. Ghosal, Matthew Zurek, Daniel S. Brown, Anca D. Dragan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25740](https://doi.org/10.1609/aaai.v37i5.25740)

        **Abstract**:

        When inferring reward functions from human behavior (be it demonstrations, comparisons, physical corrections, or e-stops), it has proven useful to model the human as making noisy-rational choices, with a "rationality coefficient" capturing how much noise or entropy we expect to see in the human behavior. Prior work typically sets the rationality level to a constant value, regardless of the type, or quality, of human feedback. However, in many settings, giving one type of feedback (e.g. a demonstration) may be much more difficult than a different type of feedback (e.g. answering a comparison query). Thus, we expect to see more or less noise depending on the type of human feedback. In this work, we advocate that grounding the rationality coefficient in real data for each feedback type, rather than assuming a default value, has a significant positive effect on reward learning. We test this in both simulated experiments and in a user study with real human feedback. We find that overestimating human rationality can have dire effects on reward learning accuracy and regret. We also find that fitting the rationality coefficient to human data enables better reward learning, even when the human deviates significantly from the noisy-rational choice model due to systematic biases. Further, we find that the rationality level affects the informativeness of each feedback type: surprisingly, demonstrations are not always the most informative---when the human acts very suboptimally, comparisons actually become more informative, even when the rationality level is the same for both.  Ultimately, our results emphasize the importance and advantage of paying attention to the assumed human-rationality-level, especially when agents actively learn from multiple types of human feedback.

        ----

        ## [671] The Role of Heuristics and Biases during Complex Choices with an AI Teammate

        **Authors**: *Nikolos Gurney, John H. Miller, David V. Pynadath*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25741](https://doi.org/10.1609/aaai.v37i5.25741)

        **Abstract**:

        Behavioral scientists have classically documented aversion to algorithmic decision aids, from simple linear models to AI. Sentiment, however, is changing and possibly accelerating AI helper usage. AI assistance is, arguably, most valuable when humans must make complex choices. We argue that classic experimental methods used to study heuristics and biases are insufficient for studying complex choices made with AI helpers. We adapted an experimental paradigm designed for studying complex choices in such contexts. We show that framing and anchoring effects impact how people work with an AI helper and are predictive of choice outcomes. The evidence suggests that some participants, particularly those in a loss frame, put too much faith in the AI helper and experienced worse choice outcomes by doing so. The paradigm also generates computational modeling-friendly data allowing future studies of human-AI decision making.

        ----

        ## [672] Learning to Defer with Limited Expert Predictions

        **Authors**: *Patrick Hemmer, Lukas Thede, Michael Vössing, Johannes Jakubik, Niklas Kühl*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25742](https://doi.org/10.1609/aaai.v37i5.25742)

        **Abstract**:

        Recent research suggests that combining AI models with a human expert can exceed the performance of either alone. The combination of their capabilities is often realized by learning to defer algorithms that enable the AI to learn to decide whether to make a prediction for a particular instance or defer it to the human expert. However, to accurately learn which instances should be deferred to the human expert, a large number of expert predictions that accurately reflect the expert's capabilities are required—in addition to the ground truth labels needed to train the AI. This requirement shared by many learning to defer algorithms hinders their adoption in scenarios where the responsible expert regularly changes or where acquiring a sufficient number of expert predictions is costly. In this paper, we propose a three-step approach to reduce the number of expert predictions required to train learning to defer algorithms. It encompasses (1) the training of an embedding model with ground truth labels to generate feature representations that serve as a basis for (2) the training of an expertise predictor model to approximate the expert's capabilities. (3) The expertise predictor generates artificial expert predictions for instances not yet labeled by the expert, which are required by the learning to defer algorithms. We evaluate our approach on two public datasets. One with "synthetically" generated human experts and another from the medical domain containing real-world radiologists' predictions. Our experiments show that the approach allows the training of various learning to defer algorithms with a minimal number of human expert predictions. Furthermore, we demonstrate that even a small number of expert predictions per class is sufficient for these algorithms to exceed the performance the AI and the human expert can achieve individually.

        ----

        ## [673] SWL-Adapt: An Unsupervised Domain Adaptation Model with Sample Weight Learning for Cross-User Wearable Human Activity Recognition

        **Authors**: *Rong Hu, Ling Chen, Shenghuan Miao, Xing Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25743](https://doi.org/10.1609/aaai.v37i5.25743)

        **Abstract**:

        In practice, Wearable Human Activity Recognition (WHAR) models usually face performance degradation on the new user due to user variance. Unsupervised domain adaptation (UDA) becomes the natural solution to cross-user WHAR under annotation scarcity. Existing UDA models usually align samples across domains without differentiation, which ignores the difference among samples. In this paper, we propose an unsupervised domain adaptation model with sample weight learning (SWL-Adapt) for cross-user WHAR. SWL-Adapt calculates sample weights according to the classification loss and domain discrimination loss of each sample with a parameterized network. We introduce the meta-optimization based update rule to learn this network end-to-end, which is guided by meta-classification loss on the selected pseudo-labeled target samples. Therefore, this network can fit a weighting function according to the cross-user WHAR task at hand, which is superior to existing sample differentiation rules fixed for special scenarios. Extensive experiments on three public WHAR datasets demonstrate that SWL-Adapt achieves the state-of-the-art performance on the cross-user WHAR task, outperforming the best baseline by an average of 3.1% and 5.3% in accuracy and macro F1 score, respectively.

        ----

        ## [674] Incentive-Boosted Federated Crowdsourcing

        **Authors**: *Xiangping Kang, Guoxian Yu, Jun Wang, Wei Guo, Carlotta Domeniconi, Jinglin Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25744](https://doi.org/10.1609/aaai.v37i5.25744)

        **Abstract**:

        Crowdsourcing is a favorable computing paradigm for processing computer-hard tasks by harnessing human intelligence. However, generic crowdsourcing systems may lead to privacy-leakage through the sharing of worker data. To tackle this problem, we propose a novel approach, called iFedCrowd (incentive-boosted Federated Crowdsourcing), to manage the privacy and quality of crowdsourcing projects. iFedCrowd allows participants to locally process sensitive data and only upload encrypted training models, and then aggregates the model parameters to build a shared server model to protect data privacy. To motivate workers to build a high-quality global model in an efficacy way, we introduce an incentive mechanism that encourages workers to constantly collect fresh data to train accurate client models and boosts the global model training. We model the incentive-based interaction between the crowdsourcing platform and participating workers as a Stackelberg game, in which each side maximizes its own profit. We derive the Nash Equilibrium of the game to find the optimal solutions for the two sides. Experimental results confirm that iFedCrowd can complete secure crowdsourcing projects with high quality and efficiency.

        ----

        ## [675] Towards Voice Reconstruction from EEG during Imagined Speech

        **Authors**: *Young-Eun Lee, Seo-Hyun Lee, Sang-Ho Kim, Seong-Whan Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25745](https://doi.org/10.1609/aaai.v37i5.25745)

        **Abstract**:

        Translating imagined speech from human brain activity into voice is a challenging and absorbing research issue that can provide new means of human communication via brain signals. Efforts to reconstruct speech from brain activity have shown their potential using invasive measures of spoken speech data, but have faced challenges in reconstructing imagined speech. In this paper, we propose NeuroTalk, which converts non-invasive brain signals of imagined speech into the user's own voice. Our model was trained with spoken speech EEG which was generalized to adapt to the domain of imagined speech, thus allowing natural correspondence between the imagined speech and the voice as a ground truth. In our framework, an automatic speech recognition decoder contributed to decomposing the phonemes of the generated speech, demonstrating the potential of voice reconstruction from unseen words. Our results imply the potential of speech synthesis from human EEG signals, not only from spoken speech but also from the brain signals of imagined speech.

        ----

        ## [676] Evaluating and Improving Interactions with Hazy Oracles

        **Authors**: *Stephan J. Lemmer, Jason J. Corso*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25746](https://doi.org/10.1609/aaai.v37i5.25746)

        **Abstract**:

        Many AI systems integrate sensor inputs, world knowledge, and human-provided information to perform inference. While such systems often treat the human input as flawless, humans are better thought of as hazy oracles whose input may be ambiguous or outside of the AI system's understanding. In such situations it makes sense for the AI system to defer its inference while it disambiguates the human-provided information by, for example, asking the human to rephrase the query. Though this approach has been considered in the past, current work is typically limited to application-specific methods and non-standardized human experiments. We instead introduce and formalize a general notion of deferred inference. Using this formulation, we then propose a novel evaluation centered around the Deferred Error Volume (DEV) metric, which explicitly considers the tradeoff between error reduction and the additional human effort required to achieve it. We demonstrate this new formalization and an innovative deferred inference method on the disparate tasks of Single-Target Video Object Tracking and Referring Expression Comprehension, ultimately reducing error by up to 48% without any change to the underlying model or its parameters.

        ----

        ## [677] Human-in-the-Loop Vehicle ReID

        **Authors**: *Zepeng Li, Dongxiang Zhang, Yanyan Shen, Gang Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25747](https://doi.org/10.1609/aaai.v37i5.25747)

        **Abstract**:

        Vehicle ReID has been an active topic in computer vision, with a substantial number of deep neural models proposed as end-to-end solutions. In this paper, we solve the problem from a new perspective and present an interesting variant called human-in-the-loop vehicle ReID to leverage interactive (and possibly wrong) human feedback signal for performance enhancement. Such human-machine cooperation mode is orthogonal to existing ReID models. To avoid incremental training overhead, we propose an Interaction ReID Network (IRIN) that can directly accept the feedback signal as an input and adjust the embedding of query image in an online fashion. IRIN is offline trained by simulating the human interaction process, with multiple optimization strategies to fully exploit the feedback signal. Experimental results show that even by interacting  with flawed feedback generated by non-experts, IRIN still outperforms state-of-the-art ReID models by a considerable margin. If the feedback contains no false positive, IRIN boosts the mAP in Veri776 from 81.6% to 95.2% with only 5 rounds of interaction per query image.

        ----

        ## [678] Modeling Human Trust and Reliance in AI-Assisted Decision Making: A Markovian Approach

        **Authors**: *Zhuoyan Li, Zhuoran Lu, Ming Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25748](https://doi.org/10.1609/aaai.v37i5.25748)

        **Abstract**:

        The increased integration of artificial intelligence (AI) technologies in human workflows has resulted in a new paradigm of AI-assisted decision making,
in which an AI model provides decision recommendations while humans make the final decisions. To best support humans in decision making, it is critical to obtain a quantitative understanding of how humans interact with and rely on AI. Previous studies often model humans' reliance on AI as an analytical process, i.e., reliance decisions are made based on cost-benefit analysis. However, theoretical models in psychology suggest that the reliance decisions can often be driven by emotions like humans' trust in AI models. In this paper, we propose a hidden Markov model to capture the affective process underlying the human-AI interaction in AI-assisted decision making, by characterizing how decision makers adjust their trust in AI over time and make reliance decisions based on their trust. Evaluations on real human behavior data collected from human-subject experiments show that the proposed model outperforms various baselines in accurately predicting humans' reliance behavior in AI-assisted decision making. Based on the proposed model, we further provide insights into how humans' trust and reliance dynamics in AI-assisted decision making is influenced by contextual factors like decision stakes and their interaction experiences.

        ----

        ## [679] Learning Deep Hierarchical Features with Spatial Regularization for One-Class Facial Expression Recognition

        **Authors**: *Bingjun Luo, Junjie Zhu, Tianyu Yang, Sicheng Zhao, Chao Hu, Xibin Zhao, Yue Gao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25749](https://doi.org/10.1609/aaai.v37i5.25749)

        **Abstract**:

        Existing methods on facial expression recognition (FER) are mainly trained in the setting when multi-class data is available. However, to detect the alien expressions that are absent during training, this type of methods cannot work. To address this problem, we develop a Hierarchical Spatial One Class Facial Expression Recognition Network (HS-OCFER) which can construct the decision boundary of a given expression class (called normal class) by training on only one-class data. Specifically, HS-OCFER consists of three novel components. First, hierarchical bottleneck modules are proposed to enrich the representation power of the model and extract detailed feature hierarchy from different levels. Second, multi-scale spatial regularization with facial geometric information is employed to guide the feature extraction towards emotional facial representations and prevent the model from overfitting extraneous disturbing factors. Third, compact intra-class variation is adopted to separate the normal class from alien classes in the decision space. Extensive evaluations on 4 typical FER datasets from both laboratory and wild scenarios show that our method consistently outperforms state-of-the-art One-Class Classification (OCC) approaches.

        ----

        ## [680] Frustratingly Easy Truth Discovery

        **Authors**: *Reshef Meir, Ofra Amir, Omer Ben-Porat, Tsviel Ben Shabat, Gal Cohensius, Lirong Xia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25750](https://doi.org/10.1609/aaai.v37i5.25750)

        **Abstract**:

        Truth discovery is a general name for a broad range of statistical methods aimed to extract the correct answers to questions, based on multiple answers coming from noisy sources. For example, workers in a crowdsourcing platform.
In this paper, we consider an extremely simple heuristic for estimating workers' competence using  average proximity to other workers. We prove that this  estimates well the actual competence level and enables separating high and low quality workers in a wide spectrum of domains and statistical models. Under Gaussian noise,  this simple estimate is the unique solution to the MLE with a constant regularization factor.  

Finally,  weighing workers according to their average proximity in a crowdsourcing setting, results in  substantial improvement over unweighted aggregation and other truth discovery algorithms in practice.

        ----

        ## [681] Beam Search Optimized Batch Bayesian Active Learning

        **Authors**: *Jingyu Sun, Hongjie Zhai, Osamu Saisho, Susumu Takeuchi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25751](https://doi.org/10.1609/aaai.v37i5.25751)

        **Abstract**:

        Active Learning is an essential method for label-efficient deep learning. As a Bayesian active learning method, Bayesian Active Learning by Disagreement (BALD) successfully selects the most representative samples by maximizing the mutual information between the model prediction and model parameters. However, when applied to a batch acquisition mode, like batch construction with greedy search, BALD suffers from poor performance, especially with noises of near-duplicate data. To address this shortcoming, we propose a diverse beam search optimized batch active learning method, which explores a graph for every batch construction by expanding the highest-scored samples of a predetermined number. To avoid near duplicate beam branches (very similar beams generated from the same root and similar samples), which is undesirable for lacking diverse representations in the feature space, we design a self-adapted constraint within candidate beams. The proposed method is able to acquire data that can better represent the distribution of the unlabeled pool, and at the same time, be significantly different from existing beams. We observe that the proposed method achieves higher batch performance than the baseline methods on three benchmark datasets.

        ----

        ## [682] Multi-Scale Control Signal-Aware Transformer for Motion Synthesis without Phase

        **Authors**: *Lintao Wang, Kun Hu, Lei Bai, Yu Ding, Wanli Ouyang, Zhiyong Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25752](https://doi.org/10.1609/aaai.v37i5.25752)

        **Abstract**:

        Synthesizing controllable motion for a character using deep learning has been a promising approach due to its potential to learn a compact model without laborious feature engineering. To produce dynamic motion from weak control signals such as desired paths, existing methods often require auxiliary information such as phases for alleviating motion ambiguity, which limits their generalisation capability. As past poses often contain useful auxiliary hints, in this paper, we propose a task-agnostic deep learning method, namely Multi-scale Control Signal-aware Transformer (MCS-T), with an attention based encoder-decoder architecture to discover the auxiliary information implicitly for synthesizing controllable motion without explicitly requiring auxiliary information such as phase. Specifically, an encoder is devised to adaptively formulate the motion patterns of a character's past poses with multi-scale skeletons, and  a decoder driven by control signals to further synthesize and predict the character's state by paying context-specialised attention to the encoded past motion patterns. As a result, it helps alleviate the issues of low responsiveness and slow transition which often happen in conventional methods not using auxiliary information. Both qualitative and quantitative experimental results on an existing biped locomotion dataset, which involves diverse types of motion transitions, demonstrate the effectiveness of our method. In particular, MCS-T is able to successfully generate motions comparable to those generated by the methods using auxiliary information.

        ----

        ## [683] SwiftAvatar: Efficient Auto-Creation of Parameterized Stylized Character on Arbitrary Avatar Engines

        **Authors**: *Shizun Wang, Weihong Zeng, Xu Wang, Hao Yang, Li Chen, Chuang Zhang, Ming Wu, Yi Yuan, Yunzhao Zeng, Min Zheng, Jing Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25753](https://doi.org/10.1609/aaai.v37i5.25753)

        **Abstract**:

        The creation of a parameterized stylized character involves careful selection of numerous parameters, also known as the "avatar vectors" that can be interpreted by the avatar engine. Existing unsupervised avatar vector estimation methods that auto-create avatars for users, however, often fail to work because of the domain gap between realistic faces and stylized avatar images. To this end, we propose SwiftAvatar, a novel avatar auto-creation framework that is evidently superior to previous works. SwiftAvatar introduces dual-domain generators to create pairs of realistic faces and avatar images using shared latent codes. The latent codes can then be bridged with the avatar vectors as pairs, by performing GAN inversion on the avatar images rendered from the engine using avatar vectors. Through this way, we are able to synthesize paired data in high-quality as many as possible, consisting of avatar vectors and their corresponding realistic faces. We also propose semantic augmentation to improve the diversity of synthesis. Finally, a light-weight avatar vector estimator is trained on the synthetic pairs to implement efficient auto-creation. Our experiments demonstrate the effectiveness and efficiency of SwiftAvatar on two different avatar engines. The superiority and advantageous flexibility of SwiftAvatar are also verified in both subjective and objective evaluations.

        ----

        ## [684] Human Joint Kinematics Diffusion-Refinement for Stochastic Motion Prediction

        **Authors**: *Dong Wei, Huaijiang Sun, Bin Li, Jianfeng Lu, Weiqing Li, Xiaoning Sun, Shengxiang Hu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25754](https://doi.org/10.1609/aaai.v37i5.25754)

        **Abstract**:

        Stochastic human motion prediction aims to forecast multiple plausible future motions given a single pose sequence from the past. Most previous works focus on designing elaborate losses to improve the accuracy, while the diversity is typically characterized by randomly sampling a set of latent variables from the latent prior, which is then decoded into possible motions. This joint training of sampling and decoding, however, suffers from posterior collapse as the learned latent variables tend to be ignored by a strong decoder, leading to limited diversity. Alternatively, inspired by the diffusion process in nonequilibrium thermodynamics, we propose MotionDiff, a diffusion probabilistic model to treat the kinematics of human joints as heated particles, which will diffuse from original states to a noise distribution. This process not only offers a natural way to obtain the "whitened'' latents without any trainable parameters, but also introduces a new noise in each diffusion step, both of which facilitate more diverse motions. Human motion prediction is then regarded as the reverse diffusion process that converts the noise distribution into realistic future motions conditioned on the observed sequence. Specifically, MotionDiff consists of two parts: a spatial-temporal transformer-based diffusion network to generate diverse yet plausible motions, and a flexible refinement network to further enable geometric losses and align with the ground truth. Experimental results on two datasets demonstrate that our model yields the competitive performance in terms of both diversity and accuracy.

        ----

        ## [685] Collective Intelligence in Human-AI Teams: A Bayesian Theory of Mind Approach

        **Authors**: *Samuel Westby, Christoph Riedl*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25755](https://doi.org/10.1609/aaai.v37i5.25755)

        **Abstract**:

        We develop a network of Bayesian agents that collectively model the mental states of teammates from the observed communication. Using a generative computational approach to cognition, we make two contributions. First, we show that our agent could generate interventions that improve the collective intelligence of a human-AI team beyond what humans alone would achieve. Second, we develop a real-time measure of human's theory of mind ability and test theories about human cognition. We use data collected from an online experiment in which 145 individuals in 29 human-only teams of five communicate through a chat-based system to solve a cognitive task. We find that humans (a) struggle to fully integrate information from teammates into their decisions, especially when communication load is high, and (b) have cognitive biases which lead them to underweight certain useful, but ambiguous, information. Our theory of mind ability measure predicts both individual- and team-level performance. Observing teams' first 25% of messages explains about 8% of the variation in final team performance, a 170% improvement compared to the current state of the art.

        ----

        ## [686] Learning to Select Pivotal Samples for Meta Re-weighting

        **Authors**: *Yinjun Wu, Adam Stein, Jacob Gardner, Mayur Naik*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25756](https://doi.org/10.1609/aaai.v37i5.25756)

        **Abstract**:

        Sample re-weighting strategies provide a promising mechanism to deal with imperfect training data in machine learning, such as noisily labeled or class-imbalanced data. One such strategy involves formulating a bi-level optimization problem called the meta re-weighting problem, whose goal is to optimize performance on a small set of perfect pivotal samples, called meta samples. Many approaches have been proposed to efficiently solve this problem. However, all of them assume that a perfect meta sample set is already provided while we observe that the selections of meta sample set is performance-critical. In this paper, we study how to learn to identify such a meta sample set from a large, imperfect training set, that is subsequently cleaned and used to optimize performance in the meta re-weighting setting. We propose a learning framework which reduces the meta samples selection problem to a weighted K-means clustering problem through rigorously theoretical analysis. We propose two clustering methods within our learning framework, Representation-based clustering method (RBC) and Gradient-based clustering method (GBC), for balancing performance and computational efficiency. Empirical studies demonstrate the performance advantage of our methods over various baseline methods

        ----

        ## [687] Better Peer Grading through Bayesian Inference

        **Authors**: *Hedayat Zarkoob, Greg d'Eon, Lena Podina, Kevin Leyton-Brown*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25757](https://doi.org/10.1609/aaai.v37i5.25757)

        **Abstract**:

        Peer grading systems aggregate noisy reports from multiple students to approximate a "true" grade as closely as possible. Most current systems either take the mean or median of reported grades; others aim to estimate students’ grading accuracy under a probabilistic model. This paper extends the state of the art in the latter approach in three key ways: 
(1) recognizing that students can behave strategically (e.g., reporting grades close to the class average without doing the work); (2) appropriately handling censored data that arises from discrete-valued grading rubrics; and (3) using mixed integer programming to improve the interpretability of the grades assigned to students. We demonstrate how to make Bayesian inference practical in this model and evaluate our approach on both synthetic and real-world data obtained by using our implemented system in four large classes. These extensive experiments show that grade aggregation using our model accurately estimates true grades, students' likelihood of submitting uninformative grades, and the variation in their inherent grading error; we also characterize our models' robustness.

        ----

        ## [688] Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination

        **Authors**: *Rui Zhao, Jinming Song, Yufeng Yuan, Haifeng Hu, Yang Gao, Yi Wu, Zhongqian Sun, Wei Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25758](https://doi.org/10.1609/aaai.v37i5.25758)

        **Abstract**:

        We study the problem of training a Reinforcement Learning (RL) agent that is collaborative with humans without using human data. Although such agents can be obtained through self-play training, they can suffer significantly from the distributional shift when paired with unencountered partners, such as humans. In this paper, we propose Maximum Entropy Population-based training (MEP) to mitigate such distributional shift. In MEP, agents in the population are trained with our derived Population Entropy bonus to promote the pairwise diversity between agents and the individual diversity of agents themselves. After obtaining this diversified population, a common best agent is trained by paring with agents in this population via prioritized sampling, where the prioritization is dynamically adjusted based on the training progress. We demonstrate the effectiveness of our method MEP, with comparison to Self-Play PPO (SP), Population-Based Training (PBT), Trajectory Diversity (TrajeDi), and Fictitious Co-Play (FCP) in both matrix game and Overcooked game environments, with partners being human proxy models and real humans. A supplementary video showing experimental results is available at https://youtu.be/Xh-FKD0AAKE.

        ----

        ## [689] A Set of Control Points Conditioned Pedestrian Trajectory Prediction

        **Authors**: *Inhwan Bae, Hae-Gon Jeon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25759](https://doi.org/10.1609/aaai.v37i5.25759)

        **Abstract**:

        Predicting the trajectories of pedestrians in crowded conditions is an important task for applications like autonomous navigation systems. Previous studies have tackled this problem using two strategies. They (1) infer all future steps recursively, or (2) predict the potential destinations of pedestrians at once and interpolate the intermediate steps to arrive there. However, these strategies often suffer from the accumulated errors of the recursive inference, or restrictive assumptions about social relations in the intermediate path. In this paper, we present a graph convolutional network-based trajectory prediction. Firstly, we propose a control point prediction that divides the future path into three sections and infers the intermediate destinations of pedestrians to reduce the accumulated error. To do this, we construct multi-relational weighted graphs to account for their physical and complex social relations. We then introduce a trajectory refinement step based on a spatio-temporal and multi-relational graph. By considering the social interactions between neighbors, better prediction results are achievable. In experiments, the proposed network achieves state-of-the-art performance on various real-world trajectory prediction benchmarks.

        ----

        ## [690] Meta-Auxiliary Learning for Adaptive Human Pose Prediction

        **Authors**: *Qiongjie Cui, Huaijiang Sun, Jianfeng Lu, Bin Li, Weiqing Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25760](https://doi.org/10.1609/aaai.v37i5.25760)

        **Abstract**:

        Predicting high-fidelity future human poses, from a historically observed sequence, is crucial for intelligent robots to interact with humans. Deep end-to-end learning approaches, which typically train a generic pre-trained model on external datasets and then directly apply it to all test samples, emerge as the dominant solution to solve this issue. Despite encouraging progress, they remain non-optimal, as the unique properties (e.g., motion style, rhythm) of a specific sequence cannot be adapted. More generally, once encountering out-of-distributions, the predicted poses tend to be unreliable. Motivated by this observation, we propose a novel test-time adaptation framework that leverages two self-supervised auxiliary tasks to help the primary forecasting network adapt to the test sequence. In the testing phase, our model can adjust the model parameters by several gradient updates to improve the generation quality. However, due to catastrophic forgetting, both auxiliary tasks typically have a low ability to automatically present the desired positive incentives for the final prediction performance. For this reason, we also propose a meta-auxiliary learning scheme for better adaptation. Extensive experiments show that the proposed approach achieves higher accuracy and more realistic visualization.

        ----

        ## [691] Moving-Landmark Assisted Distributed Learning Based Decentralized Cooperative Localization (DL-DCL) with Fault Tolerance

        **Authors**: *Shubhankar Gupta, Suresh Sundaram*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25761](https://doi.org/10.1609/aaai.v37i5.25761)

        **Abstract**:

        This paper considers the problem of cooperative localization of multiple robots under uncertainty, communicating over a partially connected, dynamic communication network and assisted by an agile landmark. Each robot owns an IMU and a relative pose sensing suite, which can get faulty due to system or environmental uncertainty, and therefore exhibit large bias in their estimation output. For the robots to localize accurately under sensor failure and system or environmental uncertainty, a novel Distributed Learning based Decentralized Cooperative Localization (DL-DCL) algorithm is proposed that involves real-time learning of an information fusion strategy by each robot for combining pose estimates from its own sensors as well as from those of its neighboring robots, and utilizing the moving landmark's pose information as a feedback to the learning process. Convergence analysis shows that the learning process converges exponentially under certain reasonable assumptions. Simulations involving sensor failures inducing around 40-60 times increase in the nominal bias show DL-DCL's estimation performance to be approximately 40% better than the well-known covariance-based estimate fusion methods. For the evaluation of DL-DCL's implementability and fault-tolerance capability in practice, a high-fidelity simulation is carried out in Gazebo with ROS2.

        ----

        ## [692] Periodic Multi-Agent Path Planning

        **Authors**: *Kazumi Kasaura, Ryo Yonetani, Mai Nishimura*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25762](https://doi.org/10.1609/aaai.v37i5.25762)

        **Abstract**:

        Multi-agent path planning (MAPP) is the problem of planning collision-free trajectories from start to goal locations for a team of agents. This work explores a relatively unexplored setting of MAPP where streams of agents have to go through the starts and goals with high throughput. We tackle this problem by formulating a new variant of MAPP called periodic MAPP in which the timing of agent appearances is periodic. The objective with periodic MAPP is to find a periodic plan, a set of collision-free trajectories that the agent streams can use repeatedly over periods, with periods that are as small as possible. To meet this objective, we propose a solution method that is based on constraint relaxation and optimization. We show that the periodic plans once found can be used for a more practical case in which agents in a stream can appear at random times. We confirm the effectiveness of our method compared with baseline methods in terms of throughput in several scenarios that abstract autonomous intersection management tasks.

        ----

        ## [693] Improving Robotic Tactile Localization Super-resolution via Spatiotemporal Continuity Learning and Overlapping Air Chambers

        **Authors**: *Xuyang Li, Yipu Zhang, Xuemei Xie, Jiawei Li, Guangming Shi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25763](https://doi.org/10.1609/aaai.v37i5.25763)

        **Abstract**:

        Human hand has amazing super-resolution ability in sensing the force and position of contact and this ability can be strengthened by practice. Inspired by this, we propose a method for robotic tactile super-resolution enhancement by learning spatiotemporal continuity of contact position and a tactile sensor composed of overlapping air chambers. Each overlapping air chamber is constructed of soft material and seals the barometer inside to mimic adapting receptors of human skin. Each barometer obtains the global receptive field of the contact surface with the pressure propagation in the hyperelastic seal overlapping air chambers. 
Neural networks with causal convolution are employed to resolve the pressure data sampled by barometers and to predict the contact position. The temporal consistency of spatial position contributes to the accuracy and stability of positioning. We obtain an average super-resolution (SR) factor of over 2500 with only four physical sensing nodes on the rubber surface (0.1 mm in the best case on 38 × 26 mm²), which outperforms the state-of-the-art. The effect of time series length on the location prediction accuracy of causal convolution is quantitatively analyzed in this article. 
We show that robots can accomplish challenging tasks such as haptic trajectory following, adaptive grasping, and human-robot interaction with the tactile sensor. This research provides new insight into tactile super-resolution sensing and could be beneficial to various applications in the robotics field.

        ----

        ## [694] Co-imitation: Learning Design and Behaviour by Imitation

        **Authors**: *Chang Rajani, Karol Arndt, David Blanco Mulero, Kevin Sebastian Luck, Ville Kyrki*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25764](https://doi.org/10.1609/aaai.v37i5.25764)

        **Abstract**:

        The co-adaptation of robots has been a long-standing research endeavour with the goal of adapting both body and behaviour of a robot for a given task, inspired by the natural evolution of animals. Co-adaptation has the potential to eliminate costly manual hardware engineering as well as improve the performance of systems.
The standard approach to co-adaptation is to use a reward function for optimizing behaviour and morphology. However, defining and constructing such reward functions is notoriously difficult and often a significant engineering effort.
This paper introduces a new viewpoint on the co-adaptation problem, which we call co-imitation: finding a morphology and a policy that allow an imitator to closely match the behaviour of a demonstrator. To this end we propose a co-imitation methodology for adapting behaviour and morphology by matching state-distributions of the demonstrator. Specifically, we focus on the challenging scenario with mismatched state- and action-spaces between both agents. We find that co-imitation increases behaviour similarity across a variety of tasks and settings, and demonstrate co-imitation by transferring human walking, jogging and kicking skills onto a simulated humanoid.

        ----

        ## [695] RobustLoc: Robust Camera Pose Regression in Challenging Driving Environments

        **Authors**: *Sijie Wang, Qiyu Kang, Rui She, Wee Peng Tay, Andreas Hartmannsgruber, Diego Navarro Navarro*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25765](https://doi.org/10.1609/aaai.v37i5.25765)

        **Abstract**:

        Camera relocalization has various applications in autonomous driving. Previous camera pose regression models consider only ideal scenarios where there is little environmental perturbation. To deal with challenging driving environments that may have changing seasons, weather, illumination, and the presence of unstable objects, we propose RobustLoc, which derives its robustness against perturbations from neural differential equations. Our model uses a convolutional neural network to extract feature maps from multi-view images, a robust neural differential equation diffusion block module to diffuse information interactively, and a branched pose decoder with multi-layer training to estimate the vehicle poses. Experiments demonstrate that RobustLoc surpasses current state-of-the-art camera pose regression models and achieves robust performance in various environments. Our code is released at: https://github.com/sijieaaa/RobustLoc

        ----

        ## [696] Abstract Argumentation Framework with Conditional Preferences

        **Authors**: *Gianvincenzo Alfano, Sergio Greco, Francesco Parisi, Irina Trubitsyna*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25766](https://doi.org/10.1609/aaai.v37i5.25766)

        **Abstract**:

        Dung's abstract Argumentation Framework (AF) has emerged as a central formalism in the area of knowledge representation and reasoning.
Preferences in AF allow to represent the comparative strength of arguments in a simple yet expressive way. 
Preference-based AF (PAF) has been proposed to extend AF with preferences of the form a > b, whose intuitive meaning is that argument a is better than b. 
In this paper we generalize PAF by introducing conditional preferences of the form a > b \leftarrow body that informally state that a is better than b whenever the condition expressed by body is true.
The resulting framework, namely Conditional Preference-based AF (CPAF), extends the PAF semantics under three well-known preference criteria, i.e. democratic, elitist, and KTV.  
After introducing CPAF, we study the complexity of the verification problem (deciding whether a set of arguments is a ``best'' extension) as well as of the credulous and skeptical acceptance problems (deciding whether a given argument belongs to any or all ``best'' extensions, respectively) under multiple-status semantics (that is, complete, preferred, stable, and semi-stable semantics) for the above-mentioned preference criteria.

        ----

        ## [697] Reactive Synthesis of Dominant Strategies

        **Authors**: *Benjamin Aminof, Giuseppe De Giacomo, Sasha Rubin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25767](https://doi.org/10.1609/aaai.v37i5.25767)

        **Abstract**:

        We study the synthesis under environment specifications problem for LTL/LTLf which, in particular, generalizes FOND (strong) planning with these temporal goals. We consider the case where the agent cannot enforce its goal --- for which the argument for using best-effort strategies has been made  --- and study the intermediate ground, between enforcing and best-effort strategies, of dominant strategies. Intuitively, such strategies achieve the goal against any environment for which it is achievable.  

We show that dominant strategies may exist when enforcing ones do not, while still sharing with the latter many desirable properties such as being interchangeable with each other, and being monotone with respect to tightening of environment specifications. We give necessary and sufficient conditions for the existence of dominant strategies, and show that deciding if they exist is 2EXPTIME-complete --- the same as for enforcing strategies. Finally, we give a uniform, optimal, game-theoretic algorithm for simultaneously solving the three synthesis problems of enforcing, dominant, and best-effort strategies.

        ----

        ## [698] Complexity of Safety and coSafety Fragments of Linear Temporal Logic

        **Authors**: *Alessandro Artale, Luca Geatti, Nicola Gigante, Andrea Mazzullo, Angelo Montanari*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25768](https://doi.org/10.1609/aaai.v37i5.25768)

        **Abstract**:

        Linear Temporal Logic (LTL) is the de-facto standard temporal logic for system specification, whose foundational properties have been studied for over five decades. Safety and cosafety properties of LTL define notable fragments of LTL, where a prefix of a trace suffices to establish whether a formula is true or not over that trace. In this paper, we study the complexity of the problems of satisfiability, validity, and realizability over infinite and finite traces for the safety and cosafety fragments of LTL. As for satisfiability and validity over infinite traces, we prove that the majority of the fragments have the same complexity as full LTL, that is, they are PSPACE-complete. The picture is radically different for realizability: we find fragments with the same expressive power whose complexity varies from 2EXPTIME-complete (as full LTL) to EXPTIME-complete. Notably, for all cosafety fragments, the complexity of the three problems does not change passing from infinite to finite traces, while for all safety fragments the complexity of satisfiability (resp., realizability) over finite traces drops to NP-complete (resp., Πᴾ₂- complete).

        ----

        ## [699] Automatically Verifying Expressive Epistemic Properties of Programs

        **Authors**: *Francesco Belardinelli, Ioana Boureanu, Vadim Malvone, Fortunat Rajaona*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25769](https://doi.org/10.1609/aaai.v37i5.25769)

        **Abstract**:

        We propose a new approach to the verification of epistemic properties of programmes. First, we introduce the new ``program-epistemic'' logic L_PK, which is strictly richer and more general than similar formalisms appearing in the literature. To solve the verification problem in an efficient way, we introduce a translation from our language L_PK into first-order logic. Then, we show and prove correct a reduction from the model checking problem for program-epistemic formulas to the satisfiability of their first-order translation. Both our logic and our translation can handle richer specification w.r.t. the state of the art, allowing us to express the knowledge of agents about facts pertaining to programs (i.e., agents' knowledge before a program is executed as well as after is has been executed). Furthermore, we implement our translation in Haskell in a general way (i.e., independently of the programs in the logical statements), and we use existing SMT-solvers to check satisfaction of L_PK formulas on a benchmark example in the AI/agency field.

        ----

        ## [700] The Effect of Preferences in Abstract Argumentation under a Claim-Centric View

        **Authors**: *Michael Bernreiter, Wolfgang Dvorák, Anna Rapberger, Stefan Woltran*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25770](https://doi.org/10.1609/aaai.v37i5.25770)

        **Abstract**:

        In this paper, we study the effect of preferences in abstract argumentation under a claim-centric perspective. Recent work has revealed that semantical and computational properties can change when reasoning is performed on 
  claim-level rather than on the argument-level, while under certain 
  natural restrictions (arguments with the same claims have the 
  same outgoing attacks) these properties are conserved. We now investigate
  these effects when, in addition, preferences have to be taken into account and consider four prominent reductions to handle preferences between arguments.
  As we shall see, these reductions give rise to 
  different classes of claim-augmented argumentation frameworks, and behave 
  differently in terms of semantic properties and computational complexity.
  This strengthens the view that the actual choice for handling preferences 
  has to be taken with care.

        ----

        ## [701] The Parameterized Complexity of Network Microaggregation

        **Authors**: *Václav Blazej, Robert Ganian, Dusan Knop, Jan Pokorný, Simon Schierreich, Kirill Simonov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25771](https://doi.org/10.1609/aaai.v37i5.25771)

        **Abstract**:

        Microaggregation is a classical statistical disclosure control technique which requires the input data to be partitioned into clusters while adhering to specified size constraints. We provide novel exact algorithms and lower bounds for the task of microaggregating a given network while considering both unrestricted and connected clusterings, and analyze these from the perspective of the parameterized complexity paradigm. Altogether, our results assemble a complete complexity-theoretic picture for the network microaggregation problem with respect to the most natural parameterizations of the problem, including input-specified parameters capturing the size and homogeneity of the clusters as well as the treewidth and vertex cover number of the network.

        ----

        ## [702] SMT Safety Verification of Ontology-Based Processes

        **Authors**: *Diego Calvanese, Alessandro Gianola, Andrea Mazzullo, Marco Montali*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25772](https://doi.org/10.1609/aaai.v37i5.25772)

        **Abstract**:

        In the context of verification of data-aware processes, a formal approach based on satisfiability modulo theories (SMT) has been considered to verify parameterised safety properties. This approach requires a combination of model-theoretic notions and algorithmic techniques based on backward reachability. We introduce here Ontology-Based Processes, which are a variant of one of the most investigated models in this spectrum, namely simple artifact systems (SASs), where, instead of managing a database, we operate over a description logic (DL) ontology.  We prove that when the DL is expressed in (a slight extension of) RDFS, it enjoys suitable model-theoretic properties, and that by relying on such DL we can define Ontology-Based Processes to which backward reachability can still be applied.  Relying on these results we are able to show that in this novel setting, verification of safety properties is decidable in PSPACE.

        ----

        ## [703] Epistemic Disjunctive Datalog for Querying Knowledge Bases

        **Authors**: *Gianluca Cima, Marco Console, Maurizio Lenzerini, Antonella Poggi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25773](https://doi.org/10.1609/aaai.v37i5.25773)

        **Abstract**:

        The Datalog query language can express several powerful recursive properties, often crucial in real-world scenarios. While answering such queries is feasible over relational databases, the picture changes dramatically when data is enriched with intensional knowledge. It is indeed well-known that answering Datalog queries is undecidable already over lightweight knowledge bases (KBs) of the DL-Lite family. To overcome this issue, we propose a new query language based on Disjunctive Datalog rules combined with a modal epistemic operator. Rules in this language interact with the queried KB exclusively via the epistemic operator, thus extracting only the information true in every model of the KB. This form of interaction is crucial for not falling into undecidability. The contribution provided by this paper is threefold. First, we illustrate the syntax and the semantics of the novel query language. Second, we study the expressive power of different fragments of our new language and compare it with Disjunctive Datalog and its variants. Third, we outline the precise data complexity of answering queries in our new language over KBs expressed in various well-known formalisms.

        ----

        ## [704] Learning Logic Programs by Discovering Where Not to Search

        **Authors**: *Andrew Cropper, Céline Hocquette*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25774](https://doi.org/10.1609/aaai.v37i5.25774)

        **Abstract**:

        The goal of inductive logic programming (ILP) is to search for a hypothesis that generalises training examples and background knowledge (BK). To improve performance, we introduce an approach that, before searching for a hypothesis, first discovers "where not to search". We use given BK to discover constraints on hypotheses, such as that a number cannot be both even and odd. We use the constraints to bootstrap a constraint-driven ILP system. Our experiments on multiple domains (including program synthesis and inductive general game playing) show that our approach can (i) substantially reduce learning times by up to 97%, and (ii) can scale to domains with millions of facts.

        ----

        ## [705] From Width-Based Model Checking to Width-Based Automated Theorem Proving

        **Authors**: *Mateus de Oliveira Oliveira, Farhad Vadiee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25775](https://doi.org/10.1609/aaai.v37i5.25775)

        **Abstract**:

        In the field of parameterized complexity theory, the study of graph width measures has been intimately connected with the development of width-based model checking algorithms for combinatorial properties on graphs. In this work, we introduce a general framework to convert a large class of width-based model-checking algorithms into algorithms that can be used to test the validity of graph-theoretic conjectures on classes of graphs of bounded width. Our framework is modular and can be applied with respect to several well-studied width measures for graphs, including treewidth and cliquewidth.

As a quantitative application of our framework, we prove analytically that for several long-standing graph-theoretic conjectures, there exists an algorithm that takes a number k as input and correctly determines in time double-exponential in a polynomial of k whether the conjecture is valid on all graphs of treewidth at most k. These upper bounds, which may be regarded as upper-bounds on the size of proofs/disproofs for these conjectures on the class of graphs of treewidth at most k, improve significantly on theoretical upper bounds obtained using previously available techniques.

        ----

        ## [706] Model-Checking for Ability-Based Logics with Constrained Plans

        **Authors**: *Stéphane Demri, Raul Fervari*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25776](https://doi.org/10.1609/aaai.v37i5.25776)

        **Abstract**:

        We investigate the complexity of the model-checking problem for a family of modal logics capturing the notion of “knowing how”. We consider the most standard ability-based knowing how logic, for which we show that model-checking is PSpace-complete. By contrast, a multi-agent variant based on an uncertainty relation between plans in which uncertainty is encoded by a regular language, is shown to admit a PTime model-checking problem. We extend with budgets the above-mentioned ability-logics, as done for ATL-like logics. We show that for the former logic enriched with budgets, the complexity increases to at least ExpSpace-hardness, whereas for the latter, the PTime bound is preserved. Other variant logics are discussed along the paper.

        ----

        ## [707] A Structural Complexity Analysis of Synchronous Dynamical Systems

        **Authors**: *Eduard Eiben, Robert Ganian, Thekla Hamm, Viktoriia Korchemna*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25777](https://doi.org/10.1609/aaai.v37i5.25777)

        **Abstract**:

        Synchronous dynamical systems are well-established models that have been used to capture a range of phenomena in networks, including opinion diffusion, spread of disease and product adoption. We study the three most notable problems in synchronous dynamical systems: whether the system will transition to a target configuration from a starting configuration, whether the system will reach convergence from a starting configuration, and whether the system is guaranteed to converge from every possible starting configuration. While all three problems were known to be intractable in the classical sense, we initiate the study of their exact boundaries of tractability from the perspective of structural parameters of the network by making use of the more fine-grained parameterized complexity paradigm. 

As our first result, we consider treewidth - as the most prominent and ubiquitous structural parameter - and show that all three problems remain intractable even on instances of constant treewidth. We complement this negative finding with fixed-parameter algorithms for the former two problems parameterized by treedepth, a well-studied restriction of treewidth. While it is possible to rule out a similar algorithm for convergence guarantee under treedepth, we conclude with a fixed-parameter algorithm for this last problem when parameterized by treedepth and the maximum in-degree.

        ----

        ## [708] Evaluating Epistemic Logic Programs via Answer Set Programming with Quantifiers

        **Authors**: *Wolfgang Faber, Michael Morak*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25778](https://doi.org/10.1609/aaai.v37i5.25778)

        **Abstract**:

        In this paper we introduce a simple way to evaluate epistemic logic programs by means of answer set programming with quantifiers, a recently proposed extension of answer set programming. The method can easily be adapted for most of the many semantics that were proposed for epistemic logic programs. We evaluate the proposed transformation on existing benchmarks using a recently proposed solver for answer set programming with quantifiers, which relies on QBF solvers.

        ----

        ## [709] Reachability Games Modulo Theories with a Bounded Safety Player

        **Authors**: *Marco Faella, Gennaro Parlato*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25779](https://doi.org/10.1609/aaai.v37i5.25779)

        **Abstract**:

        Solving reachability games is a fundamental problem for the analysis, verification, and synthesis of reactive systems.
We consider logical reachability games modulo theories (in short, GMTs), i.e.,
infinite-state games whose rules are defined by logical formulas over a multi-sorted first-order theory. 
Our games have an asymmetric constraint: the safety player has at most k possible moves from each game configuration, whereas the reachability player has no such limitation.
Even though determining the winner of such a GMT is undecidable, it can be reduced to the well-studied problem of checking the satisfiability of a system of  constrained Horn clauses (CHCs), for which many off-the-shelf solvers have been developed.
Winning strategies for GMTs can also be computed by resorting to suitable CHC queries. 
We demonstrate that GMTs can model various relevant real-world games, and that our approach can effectively solve several problems from different domains, using Z3 as the backend CHC solver.

        ----

        ## [710] Splitting Answer Set Programs with Respect to Intensionality Statements

        **Authors**: *Jorge Fandinno, Yuliya Lierler*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25780](https://doi.org/10.1609/aaai.v37i5.25780)

        **Abstract**:

        Splitting a logic program allows us to reduce the task of computing its stable models to similar tasks for its subprograms. This can be used to increase solving performance and to prove the correctness of programs. We generalize the conditions under which this technique is applicable, by considering not only dependencies between predicates but also their arguments and context. This allows splitting  programs commonly used in practice to which previous results were not applicable.

        ----

        ## [711] Monitoring Arithmetic Temporal Properties on Finite Traces

        **Authors**: *Paolo Felli, Marco Montali, Fabio Patrizi, Sarah Winkler*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25781](https://doi.org/10.1609/aaai.v37i5.25781)

        **Abstract**:

        We study monitoring of linear-time arithmetic properties against finite traces generated by an unknown dynamic system. The monitoring state is determined by considering at once the trace prefix seen so far, and all its possible finite-length, future continuations. This makes monitoring at least as hard as satisfiability and validity. Traces consist of finite sequences of assignments of a fixed set of variables to numerical values. Properties are specified in a logic we call ALTLf, combining LTLf (LTL on finite traces) with linear arithmetic constraints that may carry lookahead, i.e., variables may be compared over multiple instants of the trace. While the monitoring problem for this setting is undecidable in general, we show decidability for (a) properties without lookahead, and (b) properties with lookahead that satisfy the abstract, semantic condition of finite summary, studied before in the context of model checking. We then single out concrete, practically relevant classes of constraints guaranteeing finite summary. Feasibility is witnessed by a prototype implementation.

        ----

        ## [712] Untangled: A Complete Dynamic Topological Logic

        **Authors**: *David Fernández-Duque, Yoàv Montacute*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25782](https://doi.org/10.1609/aaai.v37i5.25782)

        **Abstract**:

        Dynamical systems are general models of change or movement over time with a broad area of applicability to many branches of science, including computer science and AI. Dynamic topological logic (DTL) is a formal framework for symbolic reasoning about dynamical systems. DTL can express various liveness and reachability conditions on such systems, but has the drawback that the only known axiomatisation requires an extended language. In this paper, we consider dynamic topological logic restricted to the class of scattered spaces. Scattered spaces appear in the context of computational logic as they provide semantics for provability and enjoy definable fixed points. We exhibit the first sound and complete dynamic topological logic in the original language of DTL. In particular, we show that the version of DTL based on the class of scattered spaces is finitely axiomatisable, and that the natural axiomatisation is sound and complete.

        ----

        ## [713] Inconsistent Cores for ASP: The Perks and Perils of Non-monotonicity

        **Authors**: *Johannes Klaus Fichte, Markus Hecher, Stefan Szeider*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25783](https://doi.org/10.1609/aaai.v37i5.25783)

        **Abstract**:

        Answer Set Programming (ASP) is a prominent modeling and solving framework. An inconsistent core (IC) of an ASP program is an inconsistent subset of rules. In the case of inconsistent programs, a smallest or subset-minimal IC contains crucial rules for the inconsistency. In this work, we study fnding minimal ICs of ASP programs and key fragments from a complexity-theoretic perspective. Interestingly, due to ASP’s non-monotonic behavior, also consistent programs admit ICs. It turns out that there is an entire landscape of problems involving ICs with a diverse range of complexities up to the fourth level of the Polynomial Hierarchy. Deciding the existence of an IC is, already for tight programs, on the second level of the Polynomial Hierarchy. Furthermore, we give encodings for IC-related problems on the fragment of tight programs and illustrate feasibility on small instance sets.

        ----

        ## [714] General Acyclicity and Cyclicity Notions for the Disjunctive Skolem Chase

        **Authors**: *Lukas Gerlach, David Carral*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25784](https://doi.org/10.1609/aaai.v37i5.25784)

        **Abstract**:

        The disjunctive skolem chase is a sound, complete, and potentially non-terminating procedure for solving boolean conjunctive query entailment over knowledge bases of disjunctive existential rules. We develop novel acyclicity and cyclicity notions for this procedure; that is, we develop sufficient conditions to determine chase termination and non-termination. Our empirical evaluation shows that our novel notions are significantly more general than existing criteria.

        ----

        ## [715] GANTEE: Generative Adversarial Network for Taxonomy Enterance Evaluation

        **Authors**: *Zhouhong Gu, Sihang Jiang, Jingping Liu, Yanghua Xiao, Hongwei Feng, Zhixu Li, Jiaqing Liang, Jian Zhong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25785](https://doi.org/10.1609/aaai.v37i5.25785)

        **Abstract**:

        Taxonomy is formulated as directed acyclic graphs or trees of concepts that support many downstream tasks.
Many new coming concepts need to be added to an existing taxonomy.
The traditional taxonomy expansion task aims only at finding the best position for new coming concepts in the existing taxonomy. 
However, they have two drawbacks when being applied to the real-scenarios.
The previous methods suffer from low-efficiency since they waste much time when most of the new coming concepts are indeed noisy concepts. They also suffer from low-effectiveness since they collect training samples only from the existing taxonomy, which limits the ability of the model to mine more hypernym-hyponym relationships among real concepts.
This paper proposes a pluggable framework called Generative Adversarial Network for Taxonomy Entering Evaluation (GANTEE) to alleviate these drawbacks.
A generative adversarial network is designed in this framework by discriminative models to alleviate the first drawback and the generative model to alleviate the second drawback.
Two discriminators are used in GANTEE to provide long-term and short-term rewards, respectively.
Moreover, to further improve the efficiency, pre-trained language models are used to retrieve the representation of the concepts quickly.
The experiments on three real-world large-scale datasets with two different languages show that GANTEE improves the performance of the existing taxonomy expansion methods in both effectiveness and efficiency.

        ----

        ## [716] Finite Based Contraction and Expansion via Models

        **Authors**: *Ricardo Guimarães, Ana Ozaki, Jandson S. Ribeiro*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25786](https://doi.org/10.1609/aaai.v37i5.25786)

        **Abstract**:

        We propose a new paradigm for Belief Change in which the new information is represented as sets of models, while the agent's body of knowledge is represented as a finite set of formulae, that is, a finite base. The focus on finiteness is crucial when we consider limited agents and reasoning algorithms. Moreover, having the input as arbitrary set of models is more general than the usual treatment of formulas as input. In this setting, we define new Belief Change operations akin to traditional expansion and contraction, and we identify the rationality postulates that emerge due to the finite representability requirement. We also analyse different logics concerning compatibility with our framework.

        ----

        ## [717] MAPS-KB: A Million-Scale Probabilistic Simile Knowledge Base

        **Authors**: *Qianyu He, Xintao Wang, Jiaqing Liang, Yanghua Xiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25787](https://doi.org/10.1609/aaai.v37i5.25787)

        **Abstract**:

        The ability to understand and generate similes is an imperative step to realize human-level AI. However, there is still a considerable gap between machine intelligence and human cognition in similes, since deep models based on statistical distribution tend to favour high-frequency similes. Hence, a large-scale symbolic knowledge base of similes is required, as it contributes to the modeling of diverse yet unpopular similes while facilitating additional evaluation and reasoning. To bridge the gap, we propose a novel framework for large-scale simile knowledge base construction, as well as two probabilistic metrics which enable an improved understanding of simile phenomena in natural language. Overall, we construct MAPS-KB, a million-scale probabilistic simile knowledge base, covering 4.3 million triplets over 0.4 million terms from 70 GB corpora. We conduct sufficient experiments to justify the effectiveness and necessity of the methods of our framework. We also apply MAPS-KB on three downstream tasks to achieve state-of-the-art performance, further demonstrating the value of MAPS-KB. Resources of MAPS-KB are publicly available at https://github.com/Abbey4799/MAPS-KB.

        ----

        ## [718] Characterizing Structural Hardness of Logic Programs: What Makes Cycles and Reachability Hard for Treewidth?

        **Authors**: *Markus Hecher*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25788](https://doi.org/10.1609/aaai.v37i5.25788)

        **Abstract**:

        Answer Set Programming (ASP) is a problem modeling and solving framework for several problems in KR with growing industrial applications. Also for studies of computational complexity and deeper insights into the hardness and its sources, ASP has been attracting researchers for many years. These studies resulted in fruitful characterizations in terms of complexity classes, fine-grained insights in form of dichotomy-style results, as well as detailed parameterized complexity landscapes. Recently, this lead to a novel result establishing that for the measure treewidth, which captures structural density of a program, the evaluation of the well-known class of normal programs is expected to be slightly harder than deciding satisfiability (SAT). However, it is unclear how to utilize this structural power of ASP. This paper deals with a novel reduction from SAT to normal ASP that goes beyond well-known encodings: We explicitly utilize the structural power of ASP, whereby we sublinearly decrease the treewidth, which probably cannot be significantly improved. Then, compared to existing results, this characterizes hardness in a fine-grained way by establishing the required functional dependency of the dependency graph’s cycle length (SCC size) on the treewidth.

        ----

        ## [719] Conditional Syntax Splitting for Non-monotonic Inference Operators

        **Authors**: *Jesse Heyninck, Gabriele Kern-Isberner, Thomas Andreas Meyer, Jonas Philipp Haldimann, Christoph Beierle*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25789](https://doi.org/10.1609/aaai.v37i5.25789)

        **Abstract**:

        Syntax splitting is a property of inductive inference operators that ensures we can restrict our attention to parts of the conditional belief base that share atoms with a given query. To apply syntax splitting, a conditional belief base needs to consist of syntactically disjoint conditionals. This requirement is often too strong in practice, as conditionals might share atoms. In this paper we introduce the concept of conditional syntax splitting, inspired by the notion of conditional independence as known from probability theory. We show that lexicographic inference and system W satisfy conditional syntax splitting, and connect conditional syntax splitting to several known properties from the literature on non-monotonic reasoning, including the drowning effect.

        ----

        ## [720] Relational Program Synthesis with Numerical Reasoning

        **Authors**: *Céline Hocquette, Andrew Cropper*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25790](https://doi.org/10.1609/aaai.v37i5.25790)

        **Abstract**:

        Learning programs with numerical values is fundamental to many AI applications, including bio-informatics and drug design. However, current program synthesis approaches struggle to learn programs with numerical values.
An especially difficult problem is learning continuous values from multiple examples, such as intervals. To overcome this limitation, we introduce an inductive logic programming approach which combines relational learning with numerical reasoning. Our approach, which we call NumSynth, uses satisfiability modulo theories solvers to efficiently learn programs with numerical values. Our approach can identify numerical values in linear arithmetic fragments, such as real difference logic, and from infinite domains, such as real numbers or integers. Our experiments on four diverse domains, including game playing and program synthesis, show that our approach can (i) learn programs with numerical values from linear arithmetical reasoning, and (ii) outperform existing approaches in terms of predictive accuracies and learning times.

        ----

        ## [721] Common Knowledge of Abstract Groups

        **Authors**: *Merlin Humml, Lutz Schröder*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25791](https://doi.org/10.1609/aaai.v37i5.25791)

        **Abstract**:

        Epistemic logics typically talk about knowledge of individual agents or groups of explicitly listed agents. Often, however, one wishes to express knowledge of groups of agents specified by a given property, as in ‘it is common knowledge among economists’. We introduce such a logic of common knowledge, which we term abstract-group epistemic logic (AGEL). That is, AGEL features a common knowledge operator for groups of agents given by concepts in a separate agent logic that we keep generic, with one possible agent logic being ALC. We show that AGEL is EXPTIME-complete, with the lower bound established by reduction from standard group epistemic logic, and the upper bound by a satisfiability-preserving embedding into the full µ-calculus. Further main results include a finite model property (not enjoyed by the full µ-calculus) and a complete axiomatization.

        ----

        ## [722] FASTDIAGP: An Algorithm for Parallelized Direct Diagnosis

        **Authors**: *Viet-Man Le, Cristian Vidal Silva, Alexander Felfernig, David Benavides, José A. Galindo, Thi Ngoc Trang Tran*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25792](https://doi.org/10.1609/aaai.v37i5.25792)

        **Abstract**:

        Constraint-based applications attempt to identify a solution that meets all defined user requirements. If the requirements are inconsistent with the underlying constraint set, algorithms that compute diagnoses for inconsistent constraints should be implemented to help users resolve the “no solution could be found” dilemma. FastDiag is a typical direct diagnosis algorithm that supports diagnosis calculation without pre-determining conflicts. However, this approach faces runtime performance issues, especially when analyzing complex and large-scale knowledge bases. In this paper, we propose a novel algorithm, so-called FastDiagP, which is based on the idea of speculative programming. This algorithm extends FastDiag by integrating a parallelization mechanism that anticipates and pre-calculates consistency checks requested by FastDiag. This mechanism helps to provide consistency checks with fast answers and boosts the algorithm’s runtime performance. The performance improvements of our proposed algorithm have been shown through empirical results using the Linux-2.6.3.33 configuration knowledge base.

        ----

        ## [723] Two Views of Constrained Differential Privacy: Belief Revision and Update

        **Authors**: *Likang Liu, Keke Sun, Chunlai Zhou, Yuan Feng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25793](https://doi.org/10.1609/aaai.v37i5.25793)

        **Abstract**:

        In this paper, we provide two views of constrained differential private (DP) mechanisms. The first one is as belief revision.  A constrained DP mechanism is obtained by standard probabilistic conditioning, and hence can be naturally implemented by Monte Carlo algorithms.  The other is as belief update.  A constrained DP is defined according to l2-distance minimization postprocessing or projection and hence can be naturally implemented by optimization algorithms.  The main advantage of these two perspectives is that we can make full use of the machinery of belief revision and update to show basic properties for constrained differential privacy especially some important new composition properties.  Within the framework established in this paper, constrained DP algorithms in the literature can be classified either as belief revision or belief update.  At the end of the paper, we demonstrate their differences especially in utility on a couple of scenarios.

        ----

        ## [724] Copyright-Certified Distillation Dataset: Distilling One Million Coins into One Bitcoin with Your Private Key

        **Authors**: *Tengjun Liu, Ying Chen, Wanxuan Gu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25794](https://doi.org/10.1609/aaai.v37i5.25794)

        **Abstract**:

        The rapid development of neural network dataset distillation in recent years has provided new ideas in many areas such as continuous learning, neural network architecture search and privacy preservation. Dataset distillation is a very effective method to distill large training datasets into small data, thus ensuring that the test accuracy of models trained on their synthesized small datasets matches that of models trained on the full dataset. Thus, dataset distillation itself is commercially valuable, not only for reducing training costs, but also for compressing storage costs and significantly reducing the training costs of deep learning. However, copyright protection for dataset distillation has not been proposed yet, so we propose the first method to protect intellectual property by embedding watermarks in the dataset distillation process. Our approach not only popularizes the dataset distillation technique, but also authenticates the ownership of the distilled dataset by the models trained on that distilled dataset.

        ----

        ## [725] DHGE: Dual-View Hyper-Relational Knowledge Graph Embedding for Link Prediction and Entity Typing

        **Authors**: *Haoran Luo, Haihong E, Ling Tan, Gengxian Zhou, Tianyu Yao, Kaiyang Wan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25795](https://doi.org/10.1609/aaai.v37i5.25795)

        **Abstract**:

        In the field of representation learning on knowledge graphs (KGs), a hyper-relational fact consists of a main triple and several auxiliary attribute-value descriptions, which is considered more comprehensive and specific than a triple-based fact. However, currently available hyper-relational KG embedding methods in a single view are limited in application because they weaken the hierarchical structure that represents the affiliation between entities. To overcome this limitation, we propose a dual-view hyper-relational KG structure (DH-KG) that contains a hyper-relational instance view for entities and a hyper-relational ontology view for concepts that are abstracted hierarchically from the entities. This paper defines link prediction and entity typing tasks on DH-KG for the first time and constructs two DH-KG datasets, JW44K-6K, extracted from Wikidata, and HTDM based on medical data. Furthermore, we propose DHGE, a DH-KG embedding model based on GRAN encoders, HGNNs, and joint learning. DHGE outperforms baseline models on DH-KG, according to experimental results. Finally, we provide an example of how this technology can be used to treat hypertension. Our model and new datasets are publicly available.

        ----

        ## [726] Automated Verification of Propositional Agent Abstraction for Classical Planning via CTLK Model Checking

        **Authors**: *Kailun Luo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25796](https://doi.org/10.1609/aaai.v37i5.25796)

        **Abstract**:

        Abstraction has long been an effective mechanism to help find a solution in classical planning. Agent abstraction, based on the situation calculus, is a promising explainable framework for agent planning, yet its automation is still far from being tackled. In this paper, we focus on a propositional version of agent abstraction designed for finite-state systems. We investigate the automated verification of the existence of propositional agent abstraction, given a finite-state system and a mapping indicating an abstraction for it. By formalizing sound, complete and deterministic properties of abstractions in a general framework, we show that the verification task can be reduced to the task of model checking against CTLK specifications. We implemented a prototype system, and validated the viability of our approach through experimentation on several domains from classical planning.

        ----

        ## [727] Efficient Answer Enumeration in Description Logics with Functional Roles

        **Authors**: *Carsten Lutz, Marcin Przybylko*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25797](https://doi.org/10.1609/aaai.v37i5.25797)

        **Abstract**:

        We study the enumeration of answers to ontology-mediated queries
when the ontology is formulated in a description logic that supports
functional roles and the query is a CQ. In particular, we show that
enumeration is possible with linear preprocessing and constant delay
when a certain extension of the CQ (pertaining to functional roles)
is acyclic and free-connex acyclic. This holds both for complete answers and
for partial answers. We provide matching lower bounds for the
case where the query is self-join free.

        ----

        ## [728] Distributed Spectrum-Based Fault Localization

        **Authors**: *Avraham Natan, Roni Stern, Meir Kalech*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25798](https://doi.org/10.1609/aaai.v37i5.25798)

        **Abstract**:

        Spectrum-Based Fault Localization (SFL) is a popular approach for diagnosing faulty systems. SFL algorithms are inherently centralized, where observations are collected and analyzed by a single diagnoser. Applying SFL to diagnose distributed systems is challenging, especially when communication is costly and there are privacy concerns. We propose two SFL-based algorithms that are designed for distributed systems: one for diagnosing a single faulty component and one for diagnosing multiple faults. We analyze these algorithms theoretically and empirically. Our analysis shows that the distributed SFL algorithms we developed output identical diagnoses to centralized SFL while preserving privacy.

        ----

        ## [729] Multi-Level Wavelet Mapping Correlation for Statistical Dependence Measurement: Methodology and Performance

        **Authors**: *Yixin Ren, Hao Zhang, Yewei Xia, Jihong Guan, Shuigeng Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25799](https://doi.org/10.1609/aaai.v37i5.25799)

        **Abstract**:

        We propose a new criterion for measuring dependence between two real variables, namely, Multi-level Wavelet Mapping Correlation (MWMC). MWMC can capture the nonlinear dependencies between variables by measuring their correlation under different levels of wavelet mappings. We show that the empirical estimate of MWMC converges exponentially to its population quantity. To support independence test better with MWMC, we further design a permutation test based on MWMC and prove that our test can not only control the type I error rate (the rate of false positives) well but also ensure that the type II error rate (the rate of false negatives) is upper bounded by O(1/n) (n is the sample size) with finite permutations. By extensive experiments on (conditional) independence tests and causal discovery, we show that our method outperforms existing independence test methods.

        ----

        ## [730] Learning Interpretable Temporal Properties from Positive Examples Only

        **Authors**: *Rajarshi Roy, Jean-Raphaël Gaglione, Nasim Baharisangari, Daniel Neider, Zhe Xu, Ufuk Topcu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25800](https://doi.org/10.1609/aaai.v37i5.25800)

        **Abstract**:

        We consider the problem of explaining the temporal behavior of black-box systems using human-interpretable models. Following recent research trends, we rely on the fundamental yet interpretable models of deterministic finite automata (DFAs) and linear temporal logic (LTL_f) formulas. In contrast to most existing works for learning DFAs and LTL_f formulas, we consider learning from only positive examples. Our motivation is that negative examples are generally difficult to observe, in particular, from black-box systems. To learn meaningful models from positive examples only, we design algorithms that rely on conciseness and language minimality of models as regularizers. Our learning algorithms are based on two approaches: a symbolic and a counterexample-guided one. The symbolic approach exploits an efficient encoding of language minimality as a constraint satisfaction problem, whereas the counterexample-guided one relies on generating suitable negative examples to guide the learning. Both approaches provide us with effective algorithms with minimality guarantees on the learned models. To assess the effectiveness of our algorithms, we evaluate them on a few practical case studies.

        ----

        ## [731] Editing Boolean Classifiers: A Belief Change Perspective

        **Authors**: *Nicolas Schwind, Katsumi Inoue, Pierre Marquis*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25801](https://doi.org/10.1609/aaai.v37i5.25801)

        **Abstract**:

        This paper is about editing Boolean classifiers, i.e., determining how a Boolean classifier should be modified when new pieces of evidence must be incorporated. Our main goal is to delineate what are the rational ways of making such edits. This goes through a number of rationality postulates inspired from those considered so far for belief revision. We give a representation theorem and present some families of edit operators satisfying the postulates.

        ----

        ## [732] Implementing Bounded Revision via Lexicographic Revision and C-revision

        **Authors**: *Meliha Sezgin, Gabriele Kern-Isberner*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25802](https://doi.org/10.1609/aaai.v37i5.25802)

        **Abstract**:

        New information in the context of real life settings usually is accompanied by some kind of supplementary information that indicates context, reliability, or expertise of the information's source.
Bounded Revision (BR) displays an iterated belief revision mechanism that takes as input a new information accompanied by a reference sentence acting as supplementary information, which specifies the depth with which the new input shall be integrated in the posterior belief state. The reference sentence specifies which worlds in the prior belief state are affected by the change mechanism. We show that Bounded Revision can be characterized by three simple, yet elegant postulates and corresponds to a special case of a lexicographic revision, which inherits all relevant features of BR. Furthermore, we present methodological implementations of BR including conditional revision with c-revisions, making it directly usable for conditional revision tools.

        ----

        ## [733] Multi-Aspect Explainable Inductive Relation Prediction by Sentence Transformer

        **Authors**: *Zhixiang Su, Di Wang, Chunyan Miao, Lizhen Cui*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25803](https://doi.org/10.1609/aaai.v37i5.25803)

        **Abstract**:

        Recent studies on knowledge graphs (KGs) show that path-based methods empowered by pre-trained language models perform well in the provision of inductive and explainable relation predictions. In this paper, we introduce the concepts of relation path coverage and relation path confidence to filter out unreliable paths prior to model training to elevate the model performance. Moreover, we propose Knowledge Reasoning Sentence Transformer (KRST) to predict inductive relations in KGs. KRST is designed to encode the extracted reliable paths in KGs, allowing us to properly cluster paths and provide multi-aspect explanations. We conduct extensive experiments on three real-world datasets. The experimental results show that compared to SOTA models, KRST achieves the best performance in most transductive and inductive test cases (4 of 6), and in 11 of 12 few-shot test cases.

        ----

        ## [734] Learning to Break Symmetries for Efficient Optimization in Answer Set Programming

        **Authors**: *Alice Tarzariol, Martin Gebser, Konstantin Schekotihin, Mark Law*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25804](https://doi.org/10.1609/aaai.v37i5.25804)

        **Abstract**:

        The ability to efficiently solve hard combinatorial optimization problems is a key prerequisite to various applications of declarative programming paradigms. Symmetries in solution candidates pose a significant challenge to modern optimization algorithms since the enumeration of such candidates might substantially reduce their performance.

This paper proposes a novel approach using Inductive Logic Programming (ILP) to lift symmetry-breaking constraints for optimization problems modeled in Answer Set Programming (ASP). Given an ASP encoding with optimization statements and a set of small representative instances, our method augments ground ASP programs with auxiliary normal rules enabling the identification of symmetries using existing tools, like SBASS. Then, the obtained symmetries are lifted to first-order constraints with ILP. 
We prove the correctness of our method and evaluate it on real-world optimization problems from the domain of automated configuration. Our experiments show significant improvements of optimization performance due to the learned first-order constraints.

        ----

        ## [735] On Undisputed Sets in Abstract Argumentation

        **Authors**: *Matthias Thimm*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25805](https://doi.org/10.1609/aaai.v37i5.25805)

        **Abstract**:

        We introduce the notion of an undisputed set for abstract argumentation frameworks, which is a conflict-free set of arguments, such that its reduct contains no non-empty admissible set. We show that undisputed sets, and the stronger notion of strongly undisputed sets, provide a meaningful approach to weaken admissibility and deal with the problem of attacks from self-attacking arguments, in a similar manner as the recently introduced notion of weak admissibility. We investigate the properties of our new semantical notions and show certain relationships to classical semantics, in particular that undisputed sets are a generalisation of preferred extensions and strongly undisputed sets are a generalisation of stable extensions. We also investigate the computational complexity of standard reasoning tasks with these new notions and show that they lie on the second and third level of the polynomial hierarchy, respectively.

        ----

        ## [736] Neurosymbolic Reasoning and Learning with Restricted Boltzmann Machines

        **Authors**: *Son N. Tran, Artur S. d'Avila Garcez*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25806](https://doi.org/10.1609/aaai.v37i5.25806)

        **Abstract**:

        Knowledge representation and reasoning in neural networks has been a long-standing endeavour which has attracted much attention recently. The principled integration of reasoning and learning in neural networks is a main objective of the area of neurosymbolic Artificial Intelligence. In this paper, a neurosymbolic system is introduced that can represent any propositional logic formula. A proof of equivalence is presented showing that energy minimization in restricted Boltzmann machines corresponds to logical reasoning. We demonstrate the application of our approach empirically on logical reasoning and learning from data and knowledge. Experimental results show that reasoning can be performed effectively for a class of logical formulae. Learning from data and knowledge is also evaluated in comparison with learning of logic programs using neural networks. The results show that our approach can improve on state-of-the-art neurosymbolic systems. The theorems and empirical results presented in this paper are expected to reignite the research on the use of neural networks as massively-parallel models for logical reasoning and promote the principled integration of reasoning and learning in deep networks.

        ----

        ## [737] Materialisation-Based Reasoning in DatalogMTL with Bounded Intervals

        **Authors**: *Przemyslaw Andrzej Walega, Michal Zawidzki, Dingmin Wang, Bernardo Cuenca Grau*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25807](https://doi.org/10.1609/aaai.v37i5.25807)

        **Abstract**:

        DatalogMTL is a powerful extension of Datalog with operators from metric temporal logic (MTL), which has received significant attention in recent years. In this paper, we investigate materialisation-based reasoning (a.k.a. forward chaining) in the context of DatalogMTL programs and datasets with bounded intervals, where partial representations of the canonical model are obtained through successive rounds of rule applications. Although materialisation does not naturally terminate in this setting, it is known that the structure of canonical models is ultimately periodic. Our first contribution in this paper is a detailed analysis of the periodic structure of canonical models; in particular, we formulate saturation conditions whose satisfaction by a partial materialisation implies an ability to recover the full canonical model via unfolding; this allows us to compute the actual periods describing the repeating parts of the canonical model as well as to establish concrete bounds on the number of rounds of rule applications required to achieve saturation. Based on these theoretical results, we propose a practical reasoning algorithm where saturation can be efficiently detected as materialisation progresses, and where the relevant periods used to evaluate entailment of queries via unfolding are efficiently computed. We have implemented our algorithm  and our experiments suggest that  our approach is both scalable and robust.

        ----

        ## [738] Efficient Extraction of EL-Ontology Deductive Modules

        **Authors**: *Hui Yang, Yue Ma, Nicole Bidoit*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25808](https://doi.org/10.1609/aaai.v37i5.25808)

        **Abstract**:

        Because widely used real-world ontologies are often complex and large, one important challenge has emerged: designing tools for users to focus on sub-ontologies corresponding to their specific interests. To this end, various modules have been introduced to provide concise ontology views. This work concentrates on extracting deductive modules that preserve logical entailment over a given vocabulary. Existing deductive module proposals are either inefficient from a computing point of view or unsatisfactory from a quality point of view because the modules extracted are not concise enough. For example, minimal modules guarantee the most concise results, but their computation is highly time-consuming, while ⊥⊤∗-modules are easy to compute but usually contain many redundant items. To overcome computation cost and lack of quality, we propose to compute two kinds of deductive modules called pseudo-minimal modules and complete modules for EL-ontology. Our deductive module definitions rely on associating a tree representation with an ontology, and their computation is based on SAT encoding. Our experiments on real-world ontologies show that our pseudo-minimal modules are indeed minimal modules in almost all cases (98.9%), and computing pseudo-minimal modules is more efficient (99.79 times faster on average) than the state-of-the-art method Zoom for computing minimal modules. Also, our complete modules are more compact than ⊥⊤∗-modules, but their computation time remains comparable. Finally, note that our proposal applies to EL-ontologies while Zoom only works for EL-terminologies.

        ----

        ## [739] Visually Grounded Commonsense Knowledge Acquisition

        **Authors**: *Yuan Yao, Tianyu Yu, Ao Zhang, Mengdi Li, Ruobing Xie, Cornelius Weber, Zhiyuan Liu, Hai-Tao Zheng, Stefan Wermter, Tat-Seng Chua, Maosong Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25809](https://doi.org/10.1609/aaai.v37i5.25809)

        **Abstract**:

        Large-scale commonsense knowledge bases empower a broad range of AI applications, where the automatic extraction of commonsense knowledge (CKE) is a fundamental and challenging problem. CKE from text is known for suffering from the inherent sparsity and reporting bias of commonsense in text. Visual perception, on the other hand, contains rich commonsense knowledge about real-world entities, e.g., (person, can_hold, bottle), which can serve as promising sources for acquiring grounded commonsense knowledge. In this work, we present CLEVER, which formulates CKE as a distantly supervised multi-instance learning problem, where models learn to summarize commonsense relations from a bag of images about an entity pair without any human annotation on image instances. To address the problem, CLEVER leverages vision-language pre-training models for deep understanding of each image in the bag, and selects informative instances from the bag to summarize commonsense entity relations via a novel contrastive attention mechanism. Comprehensive experimental results in held-out and human evaluation show that CLEVER can extract commonsense knowledge in promising quality, outperforming pre-trained language model-based methods by 3.9 AUC and 6.4 mAUC points. The predicted commonsense scores show strong correlation with human judgment with a 0.78 Spearman coefficient. Moreover, the extracted commonsense can also be grounded into images with reasonable interpretability. The data and codes can be obtained at https://github.com/thunlp/CLEVER.

        ----

        ## [740] DNG: Taxonomy Expansion by Exploring the Intrinsic Directed Structure on Non-gaussian Space

        **Authors**: *Songlin Zhai, Weiqing Wang, Yuan-Fang Li, Yuan Meng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25810](https://doi.org/10.1609/aaai.v37i5.25810)

        **Abstract**:

        Taxonomy expansion is the process of incorporating a large number of additional nodes (i.e., ''queries'') into an existing taxonomy (i.e., ''seed''), with the most important step being the selection of appropriate positions for each query.
Enormous efforts have been made by exploring the seed's structure.
However, existing approaches are deficient in their mining of structural information in two ways: poor modeling of the hierarchical semantics and failure to capture directionality of the is-a relation.
This paper seeks to address these issues by explicitly denoting each node as the combination of inherited feature (i.e., structural part) and incremental feature (i.e., supplementary part).
Specifically, the inherited feature originates from ''parent'' nodes and is weighted by an inheritance factor.
With this node representation, the hierarchy of semantics in taxonomies (i.e., the inheritance and accumulation of features from ''parent'' to ''child'') could be embodied.
Additionally, based on this representation, the directionality of the is-a relation could be easily translated into the irreversible inheritance of features.
Inspired by the Darmois-Skitovich Theorem, we implement this irreversibility by a non-Gaussian constraint on the supplementary feature.
A log-likelihood learning objective is further utilized to optimize the proposed model (dubbed DNG), whereby the required non-Gaussianity is also theoretically ensured.
Extensive experimental results on two real-world datasets verify the superiority of DNG relative to several strong baselines.

        ----

        ## [741] Quality-Aware Self-Training on Differentiable Synthesis of Rare Relational Data

        **Authors**: *Chongsheng Zhang, Yaxin Hou, Ke Chen, Shuang Cao, Gaojuan Fan, Ji Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25811](https://doi.org/10.1609/aaai.v37i5.25811)

        **Abstract**:

        Data scarcity is a very common real-world problem that poses a major  challenge to data-driven analytics. Although a lot of data-balancing approaches have been proposed to mitigate this problem, they may drop some useful information or fall into the overfitting problem. Generative Adversarial Network (GAN) based data synthesis methods can alleviate such a problem but lack of quality control over the generated samples. Moreover, the latent associations between the attribute set and the class labels in a relational data cannot be easily captured by a vanilla GAN. In light of this, we introduce an end-to-end self-training scheme (namely, Quality-Aware Self-Training) for rare relational data synthesis, which generates labeled synthetic data via pseudo labeling on GAN-based synthesis. We design a semantic pseudo labeling module to first control the quality of the generated features/samples, then calibrate their semantic labels via a classifier committee consisting of multiple pre-trained shallow classifiers. The high-confident generated samples with calibrated pseudo labels are then fed into a semantic classification network as augmented samples for self-training. We conduct extensive experiments on 20 benchmark datasets of different domains, including 14 industrial datasets. The results show that our method significantly outperforms state-of-the-art methods, including two recent GAN-based data synthesis schemes. Codes are available at https://github.com/yaxinhou/QAST.

        ----

        ## [742] Learning to Select Prototypical Parts for Interpretable Sequential Data Modeling

        **Authors**: *Yifei Zhang, Neng Gao, Cunqing Ma*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25812](https://doi.org/10.1609/aaai.v37i5.25812)

        **Abstract**:

        Prototype-based interpretability methods provide intuitive explanations of model prediction by comparing samples to a reference set of memorized exemplars or typical representatives in terms of similarity. In the field of sequential data modeling, similarity calculations of prototypes are usually based on encoded representation vectors. However, due to highly recursive functions, there is usually a non-negligible disparity between the prototype-based explanations and the original input. In this work, we propose a Self-Explaining Selective Model (SESM) that uses a linear combination of prototypical concepts to explain its own predictions. The model employs the idea of case-based reasoning by selecting sub-sequences of the input that mostly activate different concepts as prototypical parts, which users can compare to sub-sequences selected from different example inputs to understand model decisions. For better interpretability, we design multiple constraints including diversity, stability, and locality as training objectives. Extensive experiments in different domains demonstrate that our method exhibits promising interpretability and competitive accuracy.

        ----

        ## [743] McOmet: Multimodal Fusion Transformer for Physical Audiovisual Commonsense Reasoning

        **Authors**: *Daoming Zong, Shiliang Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i5.25813](https://doi.org/10.1609/aaai.v37i5.25813)

        **Abstract**:

        Abstract
Referred to by: Retraction Note to: McOmet: Multimodal Fusion Transformer for Physical Audiovisual Commonsense Reasoning.
This article, which was published in Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI 2023), has been retracted by agreement between the authors and the journal.

        ----

        ## [744] Approximating Full Conformal Prediction at Scale via Influence Functions

        **Authors**: *Javier Abad Martinez, Umang Bhatt, Adrian Weller, Giovanni Cherubin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25814](https://doi.org/10.1609/aaai.v37i6.25814)

        **Abstract**:

        Conformal prediction (CP) is a wrapper around traditional machine learning models, giving coverage guarantees under the sole assumption of exchangeability; in classification problems, a CP guarantees that the error rate is at most a chosen significance level, irrespective of whether the underlying model is misspecified. However, the prohibitive computational costs of full CP led researchers to design scalable alternatives, which alas do not attain the same guarantees or statistical power of full CP. In this paper, we use influence functions to efficiently approximate full CP. We prove that our method is a consistent approximation of full CP, and empirically show that the approximation error becomes smaller as the training set increases; e.g., for 1,000 training points the two methods output p-values that are <0.001 apart: a negligible error for any practical application. Our methods enable scaling full CP to large real-world datasets. We compare our full CP approximation (ACP) to mainstream CP alternatives, and observe that our method is computationally competitive whilst enjoying the statistical predictive power of full CP.

        ----

        ## [745] Efficient Distributed Inference of Deep Neural Networks via Restructuring and Pruning

        **Authors**: *Afshin Abdi, Saeed Rashidi, Faramarz Fekri, Tushar Krishna*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25815](https://doi.org/10.1609/aaai.v37i6.25815)

        **Abstract**:

        In this paper, we consider the parallel implementation of an already-trained deep model on multiple processing nodes (a.k.a. workers). Specifically, we investigate as to how a deep model should be divided into several parallel sub-models, each of which is executed efficiently by a worker. Since latency due to synchronization and data transfer among workers negatively impacts the performance of the parallel implementation, it is desirable to have minimum interdependency among parallel sub-models. To achieve this goal, we propose to rearrange the neurons in the neural network, partition them (without changing the general topology of the neural network), and modify the weights such that the interdependency among sub-models is minimized under the computations and communications constraints of the workers while minimizing its impact on the performance of the model. We propose RePurpose, a layer-wise model restructuring and pruning technique that guarantees the performance of the overall parallelized model. To efficiently apply RePurpose, we propose an approach based on L0 optimization and the Munkres assignment algorithm. We show that, compared to the existing methods, RePurpose significantly improves the efficiency of the distributed inference via parallel implementation, both in terms of communication and computational complexity.

        ----

        ## [746] Symbolic Metamodels for Interpreting Black-Boxes Using Primitive Functions

        **Authors**: *Mahed Abroshan, Saumitra Mishra, Mohammad Mahdi Khalili*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25816](https://doi.org/10.1609/aaai.v37i6.25816)

        **Abstract**:

        One approach for interpreting black-box machine learning models is to find a global approximation of the model using simple interpretable functions, which is called a metamodel (a model of the model). Approximating the black-box with
a metamodel can be used to 1) estimate instance-wise feature importance; 2) understand the functional form of the model; 3) analyze feature interactions. In this work, we propose a new method for finding interpretable metamodels. Our approach utilizes Kolmogorov superposition theorem, which expresses multivariate functions as a composition of univariate functions (our primitive parameterized
functions). This composition can be represented in the form of a tree. Inspired by symbolic regression, we use a modified form of genetic programming to search over different tree configurations. Gradient descent (GD) is used to optimize the parameters of a given configuration. Our method is a novel memetic algorithm that uses GD  not only for training numerical constants but also for the training
of building blocks. Using several experiments, we show that our method outperforms recent metamodeling approaches suggested for interpreting black-boxes.

        ----

        ## [747] Utilizing Prior Solutions for Reward Shaping and Composition in Entropy-Regularized Reinforcement Learning

        **Authors**: *Jacob Adamczyk, Argenis Arriojas, Stas Tiomkin, Rahul V. Kulkarni*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25817](https://doi.org/10.1609/aaai.v37i6.25817)

        **Abstract**:

        In reinforcement learning (RL), the ability to utilize prior knowledge from previously solved tasks can allow agents to quickly solve new problems. In some cases, these new problems may be approximately solved by composing the solutions of previously solved primitive tasks (task composition). Otherwise, prior knowledge can be used to adjust the reward function for a new problem, in a way that leaves the optimal policy unchanged but enables quicker learning (reward shaping). In this work, we develop a general framework for reward shaping and task composition in entropy-regularized RL. To do so, we derive an exact relation connecting the optimal soft value functions for two entropy-regularized RL problems with different reward functions and dynamics. We show how the derived relation leads to a general result for reward shaping in entropy-regularized RL. We then generalize this approach to derive an exact relation connecting optimal value functions for the composition of multiple tasks in entropy-regularized RL. We validate these theoretical contributions with experiments showing that reward shaping and task composition lead to faster learning in various settings.

        ----

        ## [748] Clustering What Matters: Optimal Approximation for Clustering with Outliers

        **Authors**: *Akanksha Agrawal, Tanmay Inamdar, Saket Saurabh, Jie Xue*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25818](https://doi.org/10.1609/aaai.v37i6.25818)

        **Abstract**:

        Clustering with outliers is one of the most fundamental problems in Computer Science.  Given a set X of n points and two numbers k and m, the clustering with outliers aims to exclude m points from X, and partition the remaining points into k clusters that minimizes a certain cost function. In this paper, we give a general approach for solving clustering with outliers, which results in a fixed-parameter tractable (FPT) algorithm in k and m (i.e., an algorithm with running time of the form f(k, m) * poly(n) for some function f), that almost matches the approximation ratio for its outlier-free counterpart.

As a corollary, we obtain FPT approximation algorithms with optimal approximation ratios for k-Median and k-Means with outliers in general and Euclidean metrics. We also exhibit more applications of our approach to other variants of the problem that impose additional constraints on the clustering, such as fairness or matroid constraints.

        ----

        ## [749] Contrastive Classification and Representation Learning with Probabilistic Interpretation

        **Authors**: *Rahaf Aljundi, Yash Patel, Milan Sulc, Nikolay Chumerin, Daniel Olmeda Reino*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25819](https://doi.org/10.1609/aaai.v37i6.25819)

        **Abstract**:

        Cross entropy loss has served as the main objective function for classification-based tasks. Widely deployed for learning neural network classifiers, it shows both effectiveness and a probabilistic interpretation.  Recently, after the success of self supervised contrastive representation learning methods, supervised contrastive methods have been proposed to learn representations and have shown superior and more robust performance, compared to solely training with cross entropy loss. However, cross entropy loss is still needed to train the final classification layer. In this work, we investigate the possibility of learning both the representation and the classifier using one objective function that combines the robustness of contrastive learning and the probabilistic interpretation of  cross entropy loss. First,  we revisit a previously proposed contrastive-based objective function that approximates cross entropy loss and present a simple extension to learn  the classifier jointly. Second, we propose a new version of the supervised contrastive training that learns jointly the parameters of the classifier and the backbone of the network. We empirically show that these proposed objective functions demonstrate state-of-the-art performance and show a significant improvement over the standard cross entropy loss with more training stability and robustness in various challenging settings.

        ----

        ## [750] Simulating Network Paths with Recurrent Buffering Units

        **Authors**: *Divyam Anshumaan, Sriram Balasubramanian, Shubham Tiwari, Nagarajan Natarajan, Sundararajan Sellamanickam, Venkat N. Padmanabhan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25820](https://doi.org/10.1609/aaai.v37i6.25820)

        **Abstract**:

        Simulating physical network paths (e.g., Internet) is a cornerstone research problem in the emerging sub-field of AI-for-networking. We seek a model that generates end-to-end packet delay values in response to the time-varying load offered by a sender, which is typically a function of the previously output delays. The problem setting is unique, and renders the state-of-the-art text and time-series generative models inapplicable or ineffective. We formulate an ML problem at the intersection of dynamical systems, sequential decision making, and time-series modeling. We propose a novel grey-box approach to network simulation that embeds the semantics of physical network path in a new RNN-style model called Recurrent Buffering Unit, providing the interpretability of standard network simulator tools, the power of neural models, the efficiency of SGD-based techniques for learning, and yielding promising results on synthetic and real-world network traces.

        ----

        ## [751] Fully Dynamic Online Selection through Online Contention Resolution Schemes

        **Authors**: *Vashist Avadhanula, Andrea Celli, Riccardo Colini-Baldeschi, Stefano Leonardi, Matteo Russo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25821](https://doi.org/10.1609/aaai.v37i6.25821)

        **Abstract**:

        We study fully dynamic online selection problems in an adversarial/stochastic setting that includes Bayesian online selection, prophet inequalities, posted price mechanisms, and stochastic probing problems subject to combinatorial constraints.   In the classical ``incremental'' version of the problem, selected elements remain active until the end of the input sequence. On the other hand, in the fully dynamic version of the problem, elements stay active for a limited time interval, and then leave. This models, for example, the online matching of tasks to workers with task/worker-dependent working times, and sequential posted pricing of perishable goods. A successful approach to online selection problems in the adversarial setting is given by the notion of  Online Contention Resolution Scheme (OCRS), that uses  a priori information to formulate a linear relaxation of the underlying optimization problem, whose optimal fractional solution is rounded online for any adversarial order of the input sequence. Our main contribution is providing a general method for constructing an OCRS for fully dynamic online selection problems. Then, we show how to employ such OCRS to construct no-regret algorithms in a partial information model with semi-bandit feedback and adversarial inputs.

        ----

        ## [752] Tree Learning: Optimal Sample Complexity and Algorithms

        **Authors**: *Dmitrii Avdiukhin, Grigory Yaroslavtsev, Danny Vainstein, Orr Fischer, Sauman Das, Faraz Mirza*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25822](https://doi.org/10.1609/aaai.v37i6.25822)

        **Abstract**:

        We study the problem of learning a hierarchical tree representation of data from labeled samples, taken from an arbitrary (and possibly adversarial) distribution. Consider a collection of data tuples labeled according to their hierarchical structure. The smallest number of such tuples required in order to be able to accurately label subsequent tuples is of interest for data collection in machine learning. We present optimal sample complexity bounds for this problem in several learning settings, including (agnostic) PAC learning and online learning. Our results are based on tight bounds of the Natarajan and Littlestone dimensions of the associated problem. The corresponding tree classifiers can be constructed efficiently in near-linear time.

        ----

        ## [753] Meta-Learning for Simple Regret Minimization

        **Authors**: *Mohammad Javad Azizi, Branislav Kveton, Mohammad Ghavamzadeh, Sumeet Katariya*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25823](https://doi.org/10.1609/aaai.v37i6.25823)

        **Abstract**:

        We develop a meta-learning framework for simple regret minimization in bandits. In this framework, a learning agent interacts with a sequence of bandit tasks, which are sampled i.i.d. from an unknown prior distribution, and learns its meta-parameters to perform better on future tasks. We propose the first Bayesian and frequentist meta-learning algorithms for this setting. The Bayesian algorithm has access to a prior distribution over the meta-parameters and its meta simple regret over m bandit tasks with horizon n is mere O(m / √n). On the other hand, the meta simple regret of the frequentist algorithm is O(n√m + m/ √n). While its regret is worse, the frequentist algorithm is more general because it does not need a prior distribution over the meta-parameters. It can also be analyzed in more settings. We instantiate our algorithms for several classes of bandit problems. Our algorithms are general and we complement our theory by evaluating them empirically in several environments.

        ----

        ## [754] Generalizing Downsampling from Regular Data to Graphs

        **Authors**: *Davide Bacciu, Alessio Conte, Francesco Landolfi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25824](https://doi.org/10.1609/aaai.v37i6.25824)

        **Abstract**:

        Downsampling produces coarsened, multi-resolution representations of data and it is used, for example, to produce lossy compression and visualization of large images, reduce computational costs, and boost deep neural representation learning. 
Unfortunately, due to their lack of a regular structure, there is still no consensus on how downsampling should apply to graphs and linked data. Indeed reductions in graph data are still needed for the goals described above, but reduction mechanisms do not have the same focus on preserving topological structures and properties, while allowing for resolution-tuning, as is the case in regular data downsampling.
In this paper, we take a step in this direction, introducing a unifying interpretation of downsampling in regular and graph data. In particular, we define a graph coarsening mechanism which is a graph-structured counterpart of controllable equispaced coarsening mechanisms in regular data. We prove theoretical guarantees for distortion bounds on path lengths, as well as the ability to preserve key topological properties in the coarsened graphs. We leverage these concepts to define a graph pooling mechanism that we empirically assess in graph classification tasks, providing a greedy algorithm that allows efficient parallel implementation on GPUs, and showing that it compares favorably against pooling methods in literature.

        ----

        ## [755] PiCor: Multi-Task Deep Reinforcement Learning with Policy Correction

        **Authors**: *Fengshuo Bai, Hongming Zhang, Tianyang Tao, Zhiheng Wu, Yanna Wang, Bo Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25825](https://doi.org/10.1609/aaai.v37i6.25825)

        **Abstract**:

        Multi-task deep reinforcement learning (DRL) ambitiously aims to train a general agent that masters multiple tasks simultaneously. However, varying learning speeds of different tasks compounding with negative gradients interference makes policy learning inefficient. In this work, we propose PiCor, an efficient multi-task DRL framework that splits learning into policy optimization and policy correction phases. The policy optimization phase improves the policy by any DRL algothrim on the sampled single task without considering other tasks. The policy correction phase first constructs an adaptive adjusted performance constraint set. Then the intermediate policy learned by the first phase is constrained to the set, which controls the negative interference and balances the learning speeds across tasks. Empirically, we demonstrate that PiCor outperforms previous methods and significantly improves sample efficiency on simulated robotic manipulation and continuous control tasks. We additionally show that adaptive weight adjusting can further improve data efficiency and performance.

        ----

        ## [756] Achieving Zero Constraint Violation for Constrained Reinforcement Learning via Conservative Natural Policy Gradient Primal-Dual Algorithm

        **Authors**: *Qinbo Bai, Amrit Singh Bedi, Vaneet Aggarwal*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25826](https://doi.org/10.1609/aaai.v37i6.25826)

        **Abstract**:

        We consider the problem of constrained Markov decision process (CMDP) in continuous state actions spaces where the goal is to maximize the expected cumulative reward subject to some constraints. We propose a novel Conservative Natural Policy Gradient Primal Dual Algorithm (CNPGPD) to achieve zero constraint violation while achieving state of the art convergence results for the objective value function. For general policy parametrization, we prove convergence of value function to global optimal upto an approximation error due to restricted policy class. We improve the sample complexity of existing constrained NPGPD algorithm. To the best of our knowledge, this is the first work to establish zero constraint violation with Natural policy gradient style algorithms for infinite horizon discounted CMDPs. We demonstrate the merits of proposed algorithm via experimental evaluations.

        ----

        ## [757] Optimal Sparse Recovery with Decision Stumps

        **Authors**: *Kiarash Banihashem, Mohammad Hajiaghayi, Max Springer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25827](https://doi.org/10.1609/aaai.v37i6.25827)

        **Abstract**:

        Decision trees are widely used for their low computational cost, good
  predictive performance, and ability to assess the importance of features.
  Though often used in practice for feature selection, the theoretical
  guarantees of these methods are not well understood. We here obtain a tight
  finite sample bound for the feature selection problem in linear regression
  using single-depth decision trees. We examine the statistical properties of
  these "decision stumps" for the recovery of the s active features from p
  total features, where s << p. Our analysis provides tight sample performance guarantees on
  high-dimensional sparse systems which align with the finite sample bound of
  O(s log p) as obtained by Lasso, improving upon previous bounds for both
  the median and optimal splitting criteria. Our results extend to the
  non-linear regime as well as arbitrary sub-Gaussian distributions,
  demonstrating that tree based methods attain strong feature selection
  properties under a wide variety of settings and further shedding light on the
  success of these methods in practice. As a byproduct of our analysis, we show
  that we can provably guarantee recovery even when the number of active
  features s is unknown.
  We further validate our theoretical results and proof methodology
  using computational experiments.

        ----

        ## [758] Towards Efficient and Domain-Agnostic Evasion Attack with High-Dimensional Categorical Inputs

        **Authors**: *Hongyan Bao, Yufei Han, Yujun Zhou, Xin Gao, Xiangliang Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25828](https://doi.org/10.1609/aaai.v37i6.25828)

        **Abstract**:

        Our work targets at searching feasible adversarial perturbation to attack a classifier with  high-dimensional categorical inputs in a domain-agnostic setting.
This is intrinsically a NP-hard knapsack problem where the exploration space becomes explosively larger as the feature dimension increases. Without the help of domain knowledge, solving this problem via heuristic method, such as Branch-and-Bound, suffers from exponential complexity, yet can bring arbitrarily bad attack results. We address the challenge via the lens of multi-armed bandit based combinatorial search. Our proposed method, namely FEAT, treats modifying each categorical feature as pulling an arm in multi-armed bandit programming. Our objective is to achieve highly efficient and effective attack using an Orthogonal Matching Pursuit (OMP)-enhanced Upper Confidence Bound (UCB) exploration strategy. Our theoretical analysis bounding the regret gap of FEAT guarantees its practical attack performance. In empirical analysis, we compare FEAT with other state-of-the-art domain-agnostic attack methods over various real-world categorical data sets of different applications. Substantial experimental observations confirm the expected efficiency and attack effectiveness of FEAT applied in different application scenarios. Our work further hints the applicability of FEAT for assessing the adversarial vulnerability of classification systems with high-dimensional categorical inputs.

        ----

        ## [759] Fairness and Welfare Quantification for Regret in Multi-Armed Bandits

        **Authors**: *Siddharth Barman, Arindam Khan, Arnab Maiti, Ayush Sawarni*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25829](https://doi.org/10.1609/aaai.v37i6.25829)

        **Abstract**:

        We extend the notion of regret with a welfarist perspective. Focussing on the classic multi-armed bandit (MAB) framework, the current work quantifies the performance of bandit algorithms by applying a fundamental welfare function, namely the Nash social welfare (NSW) function. This corresponds to equating algorithm's performance to the geometric mean of its expected rewards and leads us to the study of   Nash regret, defined as the difference between the - a priori unknown - optimal mean (among the arms) and the algorithm's performance. Since NSW is known to satisfy fairness axioms, our approach complements the utilitarian considerations of average (cumulative) regret, wherein the algorithm is evaluated via the arithmetic mean of its expected rewards. 

This work develops an algorithm that, given the horizon of play T, achieves a Nash regret of O ( sqrt{(k log T)/T} ), here k denotes the number of arms in the MAB instance. Since, for any algorithm, the Nash regret is at least as much as its average regret (the AM-GM inequality), the known lower bound on average regret holds for Nash regret as well. Therefore, our Nash regret guarantee is essentially tight. In addition, we develop an anytime algorithm with a Nash regret guarantee of O( sqrt{(k log T)/T} log T ).

        ----

        ## [760] Alternating Layered Variational Quantum Circuits Can Be Classically Optimized Efficiently Using Classical Shadows

        **Authors**: *Afrad Basheer, Yuan Feng, Christopher Ferrie, Sanjiang Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25830](https://doi.org/10.1609/aaai.v37i6.25830)

        **Abstract**:

        Variational quantum algorithms (VQAs) are the quantum analog of classical neural networks (NNs). A VQA consists of a parameterized quantum circuit (PQC) which is composed of multiple layers of ansatzes (simpler PQCs, which are an analogy of NN layers) that differ only in selections of parameters. Previous work has identified the alternating layered ansatz as potentially a new standard ansatz in near-term quantum computing. Indeed, shallow alternating layered VQAs are easy to implement and have been shown to be both trainable and expressive. In this work, we introduce a training algorithm with an exponential reduction in training cost of such VQAs. Moreover, our algorithm uses classical shadows of quantum input data, and can hence be run on a classical computer with rigorous performance guarantees. We demonstrate 2-3 orders of magnitude improvement in the training cost using our algorithm for the example problems of finding state preparation circuits and the quantum autoencoder.

        ----

        ## [761] Learnable Spectral Wavelets on Dynamic Graphs to Capture Global Interactions

        **Authors**: *Anson Bastos, Abhishek Nadgeri, Kuldeep Singh, Toyotaro Suzumura, Manish Singh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25831](https://doi.org/10.1609/aaai.v37i6.25831)

        **Abstract**:

        Learning on evolving(dynamic) graphs has caught the attention of researchers as static methods exhibit limited performance in this setting. The existing methods for dynamic graphs learn spatial features by local neighborhood aggregation, which essentially only captures the low pass signals and local interactions. In this work, we go beyond current approaches to incorporate global features for effectively learning representations of a dynamically evolving graph. 
We propose to do so by capturing the spectrum of the dynamic graph. Since static methods to learn the graph spectrum would not consider the history of the evolution of the spectrum as the graph evolves with time, we propose an approach to learn the graph wavelets to capture this evolving spectra.
Further, we propose a framework that integrates the dynamically captured spectra in the form of these learnable wavelets into spatial features for incorporating local and global interactions. Experiments on eight standard datasets show that our method significantly outperforms related methods on various tasks for dynamic graphs.

        ----

        ## [762] Equi-Tuning: Group Equivariant Fine-Tuning of Pretrained Models

        **Authors**: *Sourya Basu, Prasanna Sattigeri, Karthikeyan Natesan Ramamurthy, Vijil Chenthamarakshan, Kush R. Varshney, Lav R. Varshney, Payel Das*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25832](https://doi.org/10.1609/aaai.v37i6.25832)

        **Abstract**:

        We introduce equi-tuning, a novel fine-tuning method that transforms (potentially non-equivariant) pretrained models into group equivariant models while incurring minimum L_2 loss between the feature representations of the pretrained and the equivariant models. Large pretrained models can be equi-tuned for different groups to satisfy the needs of various downstream tasks. Equi-tuned models benefit from both group equivariance as an inductive bias and semantic priors from pretrained models. We provide applications of equi-tuning on three different tasks: image classification, compositional generalization in language, and fairness in natural language generation (NLG). We also provide a novel group-theoretic definition for fairness in NLG. The effectiveness of this definition is shown by testing it against a standard empirical method of fairness in NLG. We provide experimental results for equi-tuning using a variety of pretrained models: Alexnet, Resnet, VGG, and Densenet for image classification; RNNs, GRUs, and LSTMs for compositional generalization; and GPT2 for fairness in NLG. We test these models on benchmark datasets across all considered tasks to show the generality and effectiveness of the proposed method.

        ----

        ## [763] Sustaining Fairness via Incremental Learning

        **Authors**: *Somnath Basu Roy Chowdhury, Snigdha Chaturvedi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25833](https://doi.org/10.1609/aaai.v37i6.25833)

        **Abstract**:

        Machine learning systems are often deployed for making critical decisions like credit lending, hiring, etc. While making decisions, such systems often encode the user's demographic information (like gender, age) in their intermediate representations. This can lead to decisions that are biased towards specific demographics. Prior work has focused on  debiasing intermediate representations to ensure fair decisions. However, these approaches fail to remain fair with changes in the task or demographic distribution. To ensure fairness in the wild, it is important for a system to adapt to such changes as it accesses new data in an incremental fashion.  In this work, we propose to address this issue by introducing the problem of learning fair representations in an incremental learning setting. To this end, we present Fairness-aware Incremental Representation Learning (FaIRL), a representation learning system that can sustain fairness while incrementally learning new tasks. FaIRL is able to achieve fairness and learn new tasks by controlling the rate-distortion function of the learned representations. Our empirical evaluations show that FaIRL is able to make fair decisions while achieving high performance on the target task, outperforming several baselines.

        ----

        ## [764] Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling

        **Authors**: *Lucas Berry, David Meger*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25834](https://doi.org/10.1609/aaai.v37i6.25834)

        **Abstract**:

        In this work, we demonstrate how to reliably estimate epistemic uncertainty while maintaining the flexibility needed to capture complicated aleatoric distributions. To this end, we propose an ensemble of Normalizing Flows (NF), which are state-of-the-art in modeling aleatoric uncertainty. The ensembles are created via sets of fixed dropout masks, making them less expensive than creating separate NF models. We demonstrate how to leverage the unique structure of NFs, base distributions, to estimate aleatoric uncertainty without relying on samples, provide a comprehensive set of baselines, and derive unbiased estimates for differential entropy. The methods were applied to a variety of experiments, commonly used to benchmark aleatoric and epistemic uncertainty estimation: 1D sinusoidal data, 2D windy grid-world (Wet Chicken), Pendulum, and Hopper. In these experiments, we setup an active learning framework and evaluate each model's capability at measuring aleatoric and epistemic uncertainty. The results show the advantages of using NF ensembles in capturing complicated aleatoric while maintaining accurate epistemic uncertainty estimates.

        ----

        ## [765] An Improved Algorithm for Online Min-Sum Set Cover

        **Authors**: *Marcin Bienkowski, Marcin Mucha*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25835](https://doi.org/10.1609/aaai.v37i6.25835)

        **Abstract**:

        We study a fundamental model of online preference aggregation, where an algorithm maintains an ordered list of n elements. An input is a stream of preferred sets R_1, R_2, ..., R_t, ... Upon seeing R_t and without knowledge of any future sets, an algorithm has to rerank elements (change the list ordering), so that at least one element of R_t is found near the list front. The incurred cost is a sum of the list update costs (the number of swaps of neighboring list elements) and access cost (the position of the first element of R_t on the list). This scenario occurs naturally in applications such as ordering items in an online shop using aggregated preferences of shop customers. The theoretical underpinning of this problem is known as Min-Sum Set Cover.

Unlike previous work that mostly studied the performance of an online algorithm ALG in comparison to the static optimal solution (a single optimal list ordering), in this paper, we study an arguably harder variant where the benchmark is the provably stronger optimal dynamic solution OPT (that may also modify the list ordering). In terms of an online shop, this means that the aggregated preferences of its user base evolve with time. We construct a computationally efficient randomized algorithm whose competitive ratio (ALG-to-OPT cost ratio) is O(r^2) and prove the existence of a deterministic O(r^4)-competitive algorithm. Here, r is the maximum cardinality of sets R_t. This is the first algorithm whose ratio does not depend on n: the previously best algorithm for this problem was O(r^(3/2) * n^(1/2))-competitive and Ω(r) is a lower bound on the performance of any deterministic online algorithm.

        ----

        ## [766] AutoInit: Analytic Signal-Preserving Weight Initialization for Neural Networks

        **Authors**: *Garrett Bingham, Risto Miikkulainen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25836](https://doi.org/10.1609/aaai.v37i6.25836)

        **Abstract**:

        Neural networks require careful weight initialization to prevent signals from exploding or vanishing.  Existing initialization schemes solve this problem in specific cases by assuming that the network has a certain activation function or topology.  It is difficult to derive such weight initialization strategies, and modern architectures therefore often use these same initialization schemes even though their assumptions do not hold. This paper introduces AutoInit, a weight initialization algorithm that automatically adapts to different neural network architectures.  By analytically tracking the mean and variance of signals as they propagate through the network, AutoInit appropriately scales the weights at each layer to avoid exploding or vanishing signals.  Experiments demonstrate that AutoInit improves performance of convolutional, residual, and transformer networks across a range of activation function, dropout, weight decay, learning rate, and normalizer settings, and does so more reliably than data-dependent initialization methods.  This flexibility allows AutoInit to initialize models for everything from small tabular tasks to large datasets such as ImageNet.  Such generality turns out particularly useful in neural architecture search and in activation function discovery.  In these settings, AutoInit initializes each candidate appropriately, making performance evaluations more accurate. AutoInit thus serves as an automatic configuration tool that makes design of new neural network architectures more robust. The AutoInit package provides a wrapper around TensorFlow models and is available at https://github.com/cognizant-ai-labs/autoinit.

        ----

        ## [767] A Parameterized Theory of PAC Learning

        **Authors**: *Cornelius Brand, Robert Ganian, Kirill Simonov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25837](https://doi.org/10.1609/aaai.v37i6.25837)

        **Abstract**:

        Probably Approximately Correct (i.e., PAC) learning is a core concept of sample complexity theory, and efficient PAC learnability is often seen as a natural counterpart to the class P in classical computational complexity. But while the nascent theory of parameterized complexity has allowed us to push beyond the P-NP "dichotomy" in classical computational complexity and identify the exact boundaries of tractability for numerous problems, there is no analogue in the domain of sample complexity that could push beyond efficient PAC learnability.

As our core contribution, we fill this gap by developing a theory of parameterized PAC learning  which allows us to shed new light on several recent PAC learning results that incorporated elements of parameterized complexity. Within the theory, we identify not one but two notions of fixed-parameter learnability that both form distinct counterparts to the class FPT - the core concept at the center of the parameterized complexity paradigm - and develop the machinery required to exclude fixed-parameter learnability. We then showcase the applications of this theory to identify refined boundaries of tractability for CNF and DNF learning as well as for a range of learning problems on graphs.

        ----

        ## [768] Fully-Dynamic Decision Trees

        **Authors**: *Marco Bressan, Gabriel Damay, Mauro Sozio*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25838](https://doi.org/10.1609/aaai.v37i6.25838)

        **Abstract**:

        We develop the first fully dynamic algorithm that maintains a decision tree over an arbitrary sequence of insertions and deletions of labeled examples. Given ε>0 our algorithm guarantees that, at every point in time, every node of the decision tree uses a split with Gini gain within an additive ε of the optimum. For real-valued features the algorithm has an amortized running time per insertion/deletion of O((d·log³n)/ε²), which improves to O((d·log²n)/ε) for binary or categorical features, while it uses space O(n·d), where n is the maximum number of examples at any point in time and d is the number of features. Our algorithm is nearly optimal, as we show that any algorithm with similar guarantees requires amortized running time Ω(d) and space Ω(n·d/polylog(nd)). We complement our theoretical results with an extensive experimental evaluation on real-world data, showing the effectiveness of our algorithm.

        ----

        ## [769] Scalable Theory-Driven Regularization of Scene Graph Generation Models

        **Authors**: *Davide Buffelli, Efthymia Tsamoura*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25839](https://doi.org/10.1609/aaai.v37i6.25839)

        **Abstract**:

        Several techniques have recently aimed to improve the performance of deep learning models for Scene Graph Generation (SGG) by incorporating background knowledge. State-of-the-art techniques can be divided into two families: one where the background knowledge is incorporated into the model in a subsymbolic fashion, and another in which the background knowledge is maintained in symbolic form. Despite promising results, both families of techniques face several shortcomings: the first one requires ad-hoc, more complex neural architectures increasing the training or inference cost; the second one suffers from limited scalability w.r.t. the size of the background knowledge. Our work introduces a regularization technique for injecting symbolic background knowledge into neural SGG models that overcomes the limitations of prior art. Our technique is model-agnostic, does not incur any cost at inference time, and scales to previously unmanageable background knowledge sizes. We demonstrate that our technique can improve the accuracy of state-of-the-art SGG models, by up to 33%.

        ----

        ## [770] Toward a Perspectivist Turn in Ground Truthing for Predictive Computing

        **Authors**: *Federico Cabitza, Andrea Campagner, Valerio Basile*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25840](https://doi.org/10.1609/aaai.v37i6.25840)

        **Abstract**:

        Most current Artificial Intelligence applications are based on supervised Machine Learning (ML), which ultimately grounds on data annotated by small teams of experts or large ensemble of volunteers. The annotation process is often performed in terms of a majority vote, however this has been proved to be often problematic by recent evaluation studies.
In this article, we describe and advocate for a different paradigm, which we call perspectivism: this counters the removal of disagreement and, consequently, the assumption of correctness of traditionally aggregated gold-standard datasets, and proposes the adoption of methods that preserve divergence of opinions and integrate multiple perspectives in the ground truthing process of ML development. Drawing on previous works which inspired it, mainly from the crowdsourcing and multi-rater labeling settings, we survey the state-of-the-art and describe the potential of our proposal for not only the more subjective tasks (e.g. those related to human language) but also those tasks commonly understood as objective (e.g. medical decision making). We present the main benefits of adopting a perspectivist stance in ML, as well as possible disadvantages, and various ways in which such a stance can be implemented in practice. Finally, we share a set of recommendations and outline a research agenda to advance the perspectivist stance in ML.

        ----

        ## [771] Semantic-Enhanced Image Clustering

        **Authors**: *Shaotian Cai, Liping Qiu, Xiaojun Chen, Qin Zhang, Longteng Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25841](https://doi.org/10.1609/aaai.v37i6.25841)

        **Abstract**:

        Image clustering is an important and open challenging task in computer vision. Although many methods have been proposed to solve the image clustering task, they only explore images and uncover clusters according to the image features, thus being unable to distinguish visually similar but semantically different images. In this paper, we propose to investigate the task of image clustering with the help of visual-language pre-training model. Different from the zero-shot setting, in which the class names are known, we only know the number of clusters in this setting. Therefore, how to map images to a proper semantic space and how to cluster images from both image and semantic spaces are two key problems. To solve the above problems, we propose a novel image clustering method guided by the visual-language pre-training model CLIP, named Semantic-Enhanced Image Clustering (SIC). In this new method, we propose a method to map the given images to a proper semantic space first and efficient methods to generate pseudo-labels according to the relationships between images and semantics. Finally, we propose to perform clustering with consistency learning in both image space and semantic space, in a self-supervised learning fashion. The theoretical result of convergence analysis shows that our proposed method can converge at a sublinear speed. Theoretical analysis of expectation risk also shows that we can reduce the expectation risk by improving neighborhood consistency, increasing prediction confidence, or reducing neighborhood imbalance. Experimental results on five benchmark datasets clearly show the superiority of our new method.

        ----

        ## [772] RePreM: Representation Pre-training with Masked Model for Reinforcement Learning

        **Authors**: *Yuanying Cai, Chuheng Zhang, Wei Shen, Xuyun Zhang, Wenjie Ruan, Longbo Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25842](https://doi.org/10.1609/aaai.v37i6.25842)

        **Abstract**:

        Inspired by the recent success of sequence modeling in RL and the use of   masked language model for pre-training, we propose a masked model for pre-training in RL, RePreM (Representation Pre-training with Masked Model), which trains the encoder combined with transformer blocks to predict the masked states or actions in a trajectory. RePreM is simple but effective compared to existing representation pre-training methods in RL. It avoids algorithmic sophistication (such as data augmentation or estimating multiple models) with sequence modeling and generates a representation that captures long-term dynamics well. Empirically, we demonstrate the effectiveness of RePreM in various tasks, including dynamic prediction, transfer learning, and sample-efficient RL with both value-based and actor-critic methods. Moreover, we show that RePreM scales well with dataset size, dataset quality, and the scale of the encoder, which indicates its potential towards big RL models.

        ----

        ## [773] FTM: A Frame-Level Timeline Modeling Method for Temporal Graph Representation Learning

        **Authors**: *Bowen Cao, Qichen Ye, Weiyuan Xu, Yuexian Zou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25843](https://doi.org/10.1609/aaai.v37i6.25843)

        **Abstract**:

        Learning representations for graph-structured data is essential for graph analytical tasks. While remarkable progress has been made on static graphs, researches on temporal graphs are still in its beginning stage. The bottleneck of the temporal graph representation learning approach is the neighborhood aggregation strategy, based on which graph attributes share and gather information explicitly. Existing neighborhood aggregation strategies fail to capture either the short-term features or the long-term features of temporal graph attributes, leading to unsatisfactory model performance and even poor robustness and domain generality of the representation learning method. To address this problem, we propose a Frame-level Timeline Modeling (FTM) method that helps to capture both short-term and long-term features and thus learns more informative representations on temporal graphs. In particular, we present a novel link-based framing technique to preserve the short-term features and then incorporate a timeline aggregator module to capture the intrinsic dynamics of graph evolution as long-term features. Our method can be easily assembled with most temporal GNNs. Extensive experiments on common datasets show that our method brings great improvements to the capability, robustness, and domain generality of backbone methods in downstream tasks. Our code can be found at https://github.com/yeeeqichen/FTM.

        ----

        ## [774] Estimating Treatment Effects from Irregular Time Series Observations with Hidden Confounders

        **Authors**: *Defu Cao, James Enouen, Yujing Wang, Xiangchen Song, Chuizheng Meng, Hao Niu, Yan Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25844](https://doi.org/10.1609/aaai.v37i6.25844)

        **Abstract**:

        Causal analysis for time series data, in particular estimating individualized treatment effect (ITE), is a key task in many real world applications, such as finance, retail, healthcare,  etc. Real world time series, i.e., large-scale irregular or sparse and intermittent time series, raise significant challenges to existing work attempting to estimate treatment effects. Specifically, the existence of hidden confounders can lead to  biased treatment estimates and complicate the causal inference process. In particular, anomaly hidden confounders which exceed the typical range can lead to high variance estimates. Moreover,  in continuous time settings with irregular samples, it is challenging to directly handle the dynamics of causality. In this paper, we leverage recent advances in Lipschitz regularization and neural controlled differential equations (CDE)  to develop an effective and scalable solution, namely LipCDE, to address the above challenges. LipCDE can directly model the dynamic causal relationships between historical data and outcomes with irregular samples by considering the boundary of hidden confounders given by Lipschitz constrained neural networks. Furthermore, we conduct extensive experiments on both synthetic and real world datasets to demonstrate the effectiveness and scalability of LipCDE.

        ----

        ## [775] InParformer: Evolutionary Decomposition Transformers with Interactive Parallel Attention for Long-Term Time Series Forecasting

        **Authors**: *Haizhou Cao, Zhenhao Huang, Tiechui Yao, Jue Wang, Hui He, Yangang Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25845](https://doi.org/10.1609/aaai.v37i6.25845)

        **Abstract**:

        Long-term time series forecasting (LTSF) provides substantial benefits for numerous real-world applications, whereas places essential demands on the model capacity to capture long-range dependencies. Recent Transformer-based models have significantly improved LTSF performance. It is worth noting that Transformer with the self-attention mechanism was originally proposed to model language sequences whose tokens (i.e., words) are discrete and highly semantic. However, unlike language sequences, most time series are sequential and continuous numeric points. Time steps with temporal redundancy are weakly semantic, and only leveraging time-domain tokens is hard to depict the overall properties of time series (e.g., the overall trend and periodic variations). To address these problems, we propose a novel Transformer-based forecasting model named InParformer with an Interactive Parallel Attention (InPar Attention) mechanism. The InPar Attention is proposed to learn long-range dependencies comprehensively in both frequency and time domains. To improve its learning capacity and efficiency, we further design several mechanisms, including query selection, key-value pair compression, and recombination. Moreover, InParformer is constructed with evolutionary seasonal-trend decomposition modules to enhance intricate temporal pattern extraction. Extensive experiments on six real-world benchmarks show that InParformer outperforms the state-of-the-art forecasting Transformers.

        ----

        ## [776] Meta-Sketch: A Neural Data Structure for Estimating Item Frequencies of Data Streams

        **Authors**: *Yukun Cao, Yuan Feng, Xike Xie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25846](https://doi.org/10.1609/aaai.v37i6.25846)

        **Abstract**:

        To estimate item frequencies of data streams with limited space, sketches are widely used in real applications, including real-time web analytics, network monitoring, and self-driving. Sketches can be viewed as a model which maps the identifier of a stream item to the corresponding frequency domain. Starting from the premise, we envision a neural data structure, which we term the meta-sketch, to go beyond the basic structure of conventional sketches. The meta-sketch learns basic sketching abilities from meta-tasks constituted with synthetic datasets following Zipf distributions in the pre-training phase, and can be fast adapted to real (skewed) distributions in the adaption phase. Extensive experiments demonstrate the performance gains of the meta-sketch and offer insights into our proposals.

        ----

        ## [777] Unfooling Perturbation-Based Post Hoc Explainers

        **Authors**: *Zachariah Carmichael, Walter J. Scheirer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25847](https://doi.org/10.1609/aaai.v37i6.25847)

        **Abstract**:

        Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP. The code for this work is available at https://github.com/craymichael/unfooling.

        ----

        ## [778] Very Fast, Approximate Counterfactual Explanations for Decision Forests

        **Authors**: *Miguel Á. Carreira-Perpiñán, Suryabhan Singh Hada*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25848](https://doi.org/10.1609/aaai.v37i6.25848)

        **Abstract**:

        We consider finding a counterfactual explanation for a classification or regression forest, such as a random forest. This requires solving an optimization problem to find the closest input instance to a given instance for which the forest outputs a desired value. Finding an exact solution has a cost that is exponential on the number of leaves in the forest. We propose a simple but very effective approach: we constrain the optimization to input space regions populated by actual data points. The problem reduces to a form of nearest-neighbor search using a certain distance on a certain dataset. This has two advantages: first, the solution can be found very quickly, scaling to large forests and high-dimensional data, and enabling interactive use. Second, the solution found is more likely to be realistic in that it is guided towards high-density areas of input space.

        ----

        ## [779] An Equivalence Analysis of Binary Quantification Methods

        **Authors**: *Alberto Castaño, Jaime Alonso, Pablo González, Juan José del Coz*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25849](https://doi.org/10.1609/aaai.v37i6.25849)

        **Abstract**:

        Quantification (or prevalence estimation) algorithms aim at predicting the class distribution of unseen sets (or bags) of examples. These methods are useful for two main tasks: 1) quantification applications, for instance when we need to track the proportions of several groups of interest over time, and 2) domain adaptation problems, in which we usually need to adapt a previously trained classifier to a different --albeit related-- target distribution according to the estimated prevalences. This paper analyzes several binary quantification algorithms showing that not only do they share a common framework but are, in fact, equivalent. Inspired by this study, we propose a new method that extends one of the approaches analyzed. After an empirical evaluation of all these methods using synthetic and benchmark datasets, the paper concludes recommending three of them due to their precision, efficiency, and diversity.

        ----

        ## [780] Soft Action Priors: Towards Robust Policy Transfer

        **Authors**: *Matheus Centa, Philippe Preux*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25850](https://doi.org/10.1609/aaai.v37i6.25850)

        **Abstract**:

        Despite success in many challenging problems, reinforcement learning (RL) is still confronted with sample inefficiency, which can be mitigated by introducing prior knowledge to agents. However, many transfer techniques in reinforcement learning make the limiting assumption that the teacher is an expert. In this paper, we use the action prior from the Reinforcement Learning as Inference framework - that is, a distribution over actions at each state which resembles a teacher policy, rather than a Bayesian prior - to recover state-of-the-art policy distillation techniques. Then, we propose a class of adaptive methods that can robustly exploit action priors by combining reward shaping and auxiliary regularization losses. In contrast to prior work, we develop algorithms for leveraging suboptimal action priors that may nevertheless impart valuable knowledge - which we call soft action priors. The proposed algorithms adapt by adjusting the strength of teacher feedback according to an estimate of the teacher's usefulness in each state. We perform tabular experiments, which show that the proposed methods achieve state-of-the-art performance, surpassing it when learning from suboptimal priors. Finally, we demonstrate the robustness of the adaptive algorithms in continuous action deep RL problems, in which adaptive algorithms considerably improved stability when compared to existing policy distillation methods.

        ----

        ## [781] Invariant Representations with Stochastically Quantized Neural Networks

        **Authors**: *Mattia Cerrato, Marius Köppel, Roberto Esposito, Stefan Kramer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25851](https://doi.org/10.1609/aaai.v37i6.25851)

        **Abstract**:

        Representation learning algorithms offer the opportunity to learn invariant representations of the input data with regard to nuisance factors.
Many authors have leveraged such strategies to learn fair representations, i.e., vectors where information about sensitive attributes is removed. These methods are attractive as they may be interpreted as minimizing the mutual information between a neural layer's activations and a sensitive attribute.
However, the theoretical grounding of such methods relies either on the computation of infinitely accurate adversaries or on minimizing a variational upper bound of a mutual information estimate.
In this paper, we propose a methodology for direct computation of the mutual information between neurons in a layer and a sensitive attribute. We employ stochastically-activated binary neural networks, which lets us treat neurons as random variables.
Our method is therefore able to minimize an upper bound on the mutual information between the neural representations and a sensitive attribute.
We show that this method compares favorably with the state of the art in fair representation learning and that the learned representations display a higher level of invariance compared to full-precision neural networks.

        ----

        ## [782] Learning Pessimism for Reinforcement Learning

        **Authors**: *Edoardo Cetin, Oya Çeliktutan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25852](https://doi.org/10.1609/aaai.v37i6.25852)

        **Abstract**:

        Off-policy deep reinforcement learning algorithms commonly compensate for overestimation bias during temporal-difference learning by utilizing pessimistic estimates of the expected target returns. In this work, we propose Generalized Pessimism Learning (GPL), a strategy employing a novel learnable penalty to enact such pessimism. In particular, we propose to learn this penalty alongside the critic with dual TD-learning, a new procedure to estimate and minimize the magnitude of the target returns bias with trivial computational cost. GPL enables us to accurately counteract overestimation bias throughout training without incurring the downsides of overly pessimistic targets. By integrating GPL with popular off-policy algorithms, we achieve state-of-the-art results in both competitive proprioceptive and pixel-based benchmarks.

        ----

        ## [783] Posterior Coreset Construction with Kernelized Stein Discrepancy for Model-Based Reinforcement Learning

        **Authors**: *Souradip Chakraborty, Amrit Singh Bedi, Pratap Tokekar, Alec Koppel, Brian M. Sadler, Furong Huang, Dinesh Manocha*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25853](https://doi.org/10.1609/aaai.v37i6.25853)

        **Abstract**:

        Model-based approaches to reinforcement learning (MBRL) exhibit favorable performance in practice, but their theoretical guarantees in large spaces are mostly restricted to the setting when transition model is Gaussian or Lipschitz, and demands a posterior estimate whose representational complexity grows unbounded with time. In this work, we develop a novel MBRL method (i) which relaxes the assumptions on the target transition model to belong to a generic family of mixture models; (ii) is applicable to large-scale training by incorporating a compression step such that the posterior estimate consists of a Bayesian coreset of only statistically significant past state-action pairs; and (iii) exhibits a sublinear Bayesian regret.
To achieve these results, we adopt an approach based upon Stein's method, which, under a smoothness condition on the constructed posterior and target, allows distributional distance to be evaluated in closed form as the kernelized Stein discrepancy (KSD). The aforementioned compression step is then computed in terms of greedily retaining only those samples which are more than a certain KSD away from the previous model estimate.
Experimentally, we observe that this approach is competitive with several state-of-the-art RL methodologies, and can achieve up-to 50 percent reduction in wall clock time in some continuous control environments.

        ----

        ## [784] NHITS: Neural Hierarchical Interpolation for Time Series Forecasting

        **Authors**: *Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza Ramírez, Max Mergenthaler Canseco, Artur Dubrawski*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25854](https://doi.org/10.1609/aaai.v37i6.25854)

        **Abstract**:

        Recent progress in neural forecasting accelerated improvements in the performance of large-scale forecasting systems. Yet, long-horizon forecasting remains a very difficult task. Two common challenges afflicting the task are the volatility of the predictions and their computational complexity. We introduce NHITS, a model which addresses both challenges by incorporating novel hierarchical interpolation and multi-rate data sampling techniques. These techniques enable the proposed method to assemble its predictions sequentially, emphasizing components with different frequencies and scales while decomposing the input signal and synthesizing the forecast. We prove that the hierarchical interpolation technique can efficiently approximate arbitrarily long horizons in the presence of smoothness. Additionally, we conduct extensive large-scale dataset experiments from the long-horizon forecasting literature, demonstrating the advantages of our method over the state-of-the-art methods, where NHITS provides an average accuracy improvement of almost 20% over the latest Transformer architectures while reducing the computation time by an order of magnitude (50 times). Our code is available at https://github.com/Nixtla/neuralforecast.

        ----

        ## [785] Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton

        **Authors**: *Kai-Shiang Chang, Wei-Yao Wang, Wen-Chih Peng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25855](https://doi.org/10.1609/aaai.v37i6.25855)

        **Abstract**:

        Sports analytics has captured increasing attention since analysis of the various data enables insights for training strategies, player evaluation, etc. In this paper, we focus on predicting what types of returning strokes will be made, and where players will move to based on previous strokes. As this problem has not been addressed to date, movement forecasting can be tackled through sequence-based and graph-based models by formulating as a sequence prediction task. However, existing sequence-based models neglect the effects of interactions between players, and graph-based models still suffer from multifaceted perspectives on the next movement. Moreover, there is no existing work on representing strategic relations among players' shot types and movements. To address these challenges, we first introduce the procedure of the Player Movements (PM) graph to exploit the structural movements of players with strategic relations. Based on the PM graph, we propose a novel Dynamic Graphs and Hierarchical Fusion for Movement Forecasting model (DyMF) with interaction style extractors to capture the mutual interactions of players themselves and between both players within a rally, and dynamic players' tactics across time. In addition, hierarchical fusion modules are designed to incorporate the style influence of both players and rally interactions. Extensive experiments show that our model empirically outperforms both sequence- and graph-based methods and demonstrate the practical usage of movement forecasting. Code is available at https://github.com/wywyWang/CoachAI-Projects/tree/main/Movement%20Forecasting.

        ----

        ## [786] Graph Ordering Attention Networks

        **Authors**: *Michail Chatzianastasis, Johannes F. Lutzeyer, George Dasoulas, Michalis Vazirgiannis*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25856](https://doi.org/10.1609/aaai.v37i6.25856)

        **Abstract**:

        Graph Neural Networks (GNNs) have been successfully used in many problems involving graph-structured data, achieving state-of-the-art performance. 
GNNs typically employ a message-passing scheme, in which every node aggregates information from its neighbors using a permutation-invariant aggregation function.
Standard well-examined choices such as the mean or sum aggregation functions have limited capabilities, as they are not able to capture interactions among neighbors. 
In this work, we formalize these interactions using an information-theoretic framework that notably includes synergistic information. 
Driven by this definition, we introduce the Graph Ordering Attention (GOAT) layer, a novel GNN component that captures interactions between nodes in a neighborhood. 
This is achieved by learning local node orderings via an attention mechanism and processing the ordered representations using a recurrent neural network aggregator. 
This design allows us to make use of a permutation-sensitive aggregator while maintaining the permutation-equivariance of the proposed GOAT layer. 
The GOAT model demonstrates its increased performance in modeling graph metrics that capture complex information, such as the betweenness centrality and the effective size of a node. In practical use-cases, its superior modeling capability is confirmed through its success in several real-world node classification benchmarks.

        ----

        ## [787] Scalable and Globally Optimal Generalized L₁ K-center Clustering via Constraint Generation in Mixed Integer Linear Programming

        **Authors**: *Aravinth Chembu, Scott Sanner, Hassan Khurram, Akshat Kumar*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25857](https://doi.org/10.1609/aaai.v37i6.25857)

        **Abstract**:

        The k-center clustering algorithm, introduced over 35 years ago, is known to be robust to class imbalance prevalent in many clustering problems and has various applications such as data summarization, document clustering, and facility location determination. Unfortunately, existing k-center algorithms provide highly suboptimal solutions that can limit their practical application, reproducibility, and clustering quality. In this paper, we provide a novel scalable and globally optimal solution to a popular variant of the k-center problem known as generalized L_1 k-center clustering that uses L_1 distance and allows the selection of arbitrary vectors as cluster centers.  We show that this clustering objective can be reduced to a mixed-integer linear program (MILP) that facilitates globally optimal clustering solutions. However, solving such a MILP may be intractable for large datasets; to remedy this, we present a scalable algorithm that leverages constraint generation to efficiently and provably converge to its global optimum. We further enhance outlier handling through a simple but elegant extension to our MILP objective. We first evaluate our algorithm on a variety of synthetic datasets to better understand its properties and then validate on 20 real benchmark datasets where we compare its performance to both traditional L_1 distance k-center and k-medians baselines. Our results demonstrate significant suboptimality of existing algorithms in comparison to our approach and further demonstrate that we can find optimal generalized L_1 k-center clustering solutions up to an unprecedented 1,000,000 data points.

        ----

        ## [788] Attribute and Structure Preserving Graph Contrastive Learning

        **Authors**: *Jialu Chen, Gang Kou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25858](https://doi.org/10.1609/aaai.v37i6.25858)

        **Abstract**:

        Graph Contrastive Learning (GCL) has drawn much research interest due to its strong ability to capture both graph structure and node attribute information in a self-supervised manner. Current GCL methods usually adopt Graph Neural Networks (GNNs) as the base encoder, which typically relies on the homophily assumption of networks and overlooks node similarity in the attribute space. There are many scenarios where such assumption cannot be satisfied, or node similarity plays a crucial role. In order to design a more robust mechanism, we develop a novel attribute and structure preserving graph contrastive learning framework, named ASP, which comprehensively and efficiently preserves node attributes while exploiting graph structure. Specifically, we consider three different graph views in our framework, i.e., original view, attribute view, and global structure view. Then, we perform contrastive learning across three views in a joint fashion, mining comprehensive graph information. We validate the effectiveness of the proposed framework on various real-world networks with different levels of homophily. The results demonstrate the superior performance of our model over the representative baselines.

        ----

        ## [789] On the Stability and Generalization of Triplet Learning

        **Authors**: *Jun Chen, Hong Chen, Xue Jiang, Bin Gu, Weifu Li, Tieliang Gong, Feng Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25859](https://doi.org/10.1609/aaai.v37i6.25859)

        **Abstract**:

        Triplet learning, i.e. learning from triplet data, has attracted much attention in computer vision tasks with an extremely large number of categories, e.g., face recognition and person re-identification. Albeit with rapid progress in designing and applying triplet learning algorithms, there is a lacking study on the theoretical understanding of their generalization performance. To fill this gap, this paper investigates the generalization guarantees of triplet learning by leveraging the stability analysis.  Specifically, we establish the first general high-probability generalization bound for the triplet learning algorithm satisfying the uniform stability, and then obtain the excess risk bounds of the order O(log(n)/(√n) ) for both stochastic gradient descent (SGD) and regularized risk minimization (RRM), where 2n is approximately equal to the number of training samples. Moreover, an optimistic generalization bound in expectation as fast as O(1/n) is derived for RRM in a low noise case via the on-average stability analysis. Finally, our results are applied to triplet metric learning to characterize its theoretical underpinning.

        ----

        ## [790] CF-ViT: A General Coarse-to-Fine Method for Vision Transformer

        **Authors**: *Mengzhao Chen, Mingbao Lin, Ke Li, Yunhang Shen, Yongjian Wu, Fei Chao, Rongrong Ji*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25860](https://doi.org/10.1609/aaai.v37i6.25860)

        **Abstract**:

        Vision Transformers (ViT) have made many breakthroughs in computer vision tasks. However, considerable redundancy arises in the spatial dimension of an input image, leading to massive computational costs. Therefore, We propose a coarse-to-fine vision transformer (CF-ViT) to relieve computational burden while retaining performance in this paper. Our proposed CF-ViT is motivated by two important observations in modern ViT models: (1) The coarse-grained patch splitting can locate informative regions of an input image. (2) Most images can be well recognized by a ViT model in a small-length token sequence.  Therefore, our CF-ViT implements network inference in a two-stage manner. At coarse inference stage, an input image is split into a small-length patch sequence for a computationally economical classification. If not well recognized, the informative patches are identified and further re-split in a fine-grained granularity.  Extensive experiments demonstrate the efficacy of our CF-ViT. For example, without any compromise on performance, CF-ViT reduces 53% FLOPs of LV-ViT, and also achieves 2.01x throughput. Code of this project is at https://github.com/ChenMnZ/CF-V

        ----

        ## [791] Context-Aware Safe Medication Recommendations with Molecular Graph and DDI Graph Embedding

        **Authors**: *Qianyu Chen, Xin Li, Kunnan Geng, Mingzhong Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25861](https://doi.org/10.1609/aaai.v37i6.25861)

        **Abstract**:

        Molecular structures and Drug-Drug Interactions (DDI) are recognized as important knowledge to guide medication recommendation (MR) tasks, and medical concept embedding has been applied to boost their performance. Though promising performance has been achieved by leveraging Graph Neural Network (GNN) models to encode the molecular structures of medications or/and DDI, we observe that existing models are still defective: 1) to differentiate medications with similar molecules but different functionality; or/and 2) to properly capture the unintended reactions between drugs in the embedding space. To alleviate this limitation, we propose Carmen, a cautiously designed graph embedding-based MR framework. Carmen consists of four components, including patient representation learning, context information extraction, a context-aware GNN, and DDI encoding. Carmen incorporates the visit history into the representation learning of molecular graphs to distinguish molecules with similar topology but dissimilar activity. Its DDI encoding module is specially devised for the non-transitive interaction DDI graphs. The experiments on real-world datasets demonstrate that Carmen achieves remarkable performance improvement over state-of-the-art models and can improve the safety of recommended drugs with a proper DDI graph encoding.

        ----

        ## [792] Min-Max Submodular Ranking for Multiple Agents

        **Authors**: *Qingyun Chen, Sungjin Im, Benjamin Moseley, Chenyang Xu, Ruilong Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25862](https://doi.org/10.1609/aaai.v37i6.25862)

        **Abstract**:

        In the submodular ranking (SR) problem, the input consists of a set of submodular functions defined on a ground set of elements. The goal is to order elements for all the functions to have value above a certain threshold as soon on average as possible, assuming we choose one element per time. The problem is flexible enough to capture various applications in machine learning, including decision trees. 

This paper considers the min-max version of SR where multiple instances share the ground set. With the view of each instance being associated with an agent, the min-max problem is to order the common elements to minimize the maximum objective of all agents---thus, finding a fair solution for all agents. We give approximation algorithms for this problem and demonstrate their effectiveness in the application of finding a decision tree for multiple agents.

        ----

        ## [793] Supervised Contrastive Few-Shot Learning for High-Frequency Time Series

        **Authors**: *Xi Chen, Cheng Ge, Ming Wang, Jin Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25863](https://doi.org/10.1609/aaai.v37i6.25863)

        **Abstract**:

        Significant progress has been made in representation learning, especially with recent success on self-supervised contrastive learning. However, for time series with less intuitive or semantic meaning, sampling bias may be inevitably encountered in unsupervised approaches. Although supervised contrastive learning has shown superior performance by leveraging label information, it may also suffer from class collapse. In this study, we consider a realistic scenario in industry with limited annotation information available. A supervised contrastive framework is developed for high-frequency time series representation and classification, wherein a novel variant of supervised contrastive loss is proposed to include multiple augmentations while induce spread within each class. Experiments on four mainstream public datasets as well as a series of sensitivity and ablation analyses demonstrate that the learned representations are effective and robust compared with the direct supervised learning and self-supervised learning, notably under the minimal few-shot situation.

        ----

        ## [794] The Sufficiency of Off-Policyness and Soft Clipping: PPO Is Still Insufficient according to an Off-Policy Measure

        **Authors**: *Xing Chen, Dongcui Diao, Hechang Chen, Hengshuai Yao, Haiyin Piao, Zhixiao Sun, Zhiwei Yang, Randy Goebel, Bei Jiang, Yi Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25864](https://doi.org/10.1609/aaai.v37i6.25864)

        **Abstract**:

        The popular Proximal Policy Optimization (PPO) algorithm approximates the solution in a clipped policy space. Does there exist better policies outside of this space? By using a novel surrogate objective that employs the sigmoid function (which provides an interesting way of exploration), we found that the answer is "YES", and the better policies are in fact located very far from the clipped space. We show that PPO is insufficient in "off-policyness", according to an off-policy metric called DEON. Our algorithm explores in a much larger policy space than PPO, and it maximizes the Conservative Policy Iteration (CPI) objective better than PPO during training. To the best of our knowledge, all current PPO methods have the clipping operation and optimize in the clipped policy space. Our method is the first of this kind, which advances the understanding of CPI optimization and policy gradient methods. Code is available at https://github.com/raincchio/P3O.

        ----

        ## [795] Global Convergence of Two-Timescale Actor-Critic for Solving Linear Quadratic Regulator

        **Authors**: *Xuyang Chen, Jingliang Duan, Yingbin Liang, Lin Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25865](https://doi.org/10.1609/aaai.v37i6.25865)

        **Abstract**:

        The actor-critic (AC) reinforcement learning algorithms have been the powerhouse behind many challenging applications. Nevertheless, its convergence is fragile in general. To study its instability, existing works mostly consider the uncommon double-loop variant or basic models with finite state and action space. We investigate the more practical single-sample two-timescale AC for solving the canonical linear quadratic regulator (LQR) problem, where the actor and the critic update only once with a single sample in each iteration on an unbounded continuous state and action space. Existing analysis cannot conclude the convergence for such a challenging case. We develop a new analysis framework that allows establishing the global convergence to an epsilon-optimal solution with at most an order of epsilon to -2.5 sample complexity. To our knowledge, this is the first finite-time convergence analysis for the single sample two-timescale AC for solving LQR with global optimality. The sample complexity improves those of other variants by orders, which sheds light on the practical wisdom of single sample algorithms. We also further validate our theoretical findings via comprehensive simulation comparisons.

        ----

        ## [796] Topological Pooling on Graphs

        **Authors**: *Yuzhou Chen, Yulia R. Gel*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25866](https://doi.org/10.1609/aaai.v37i6.25866)

        **Abstract**:

        Graph neural networks (GNNs) have demonstrated a significant success in various graph learning tasks, from graph classification to anomaly detection. There recently has emerged a number of approaches adopting a graph pooling operation within GNNs, with a goal to preserve graph attributive and structural features during the graph representation learning. However, most existing graph pooling operations suffer from the limitations of relying on node-wise neighbor weighting and embedding, which leads to insufficient encoding of rich topological structures and node attributes exhibited by real-world networks. By invoking the machinery of persistent homology and the concept of landmarks, we propose a novel topological pooling layer and witness complex-based topological embedding mechanism that allow us to systematically integrate hidden topological information at both local and global levels. Specifically, we design new learnable local and global topological representations Wit-TopoPool which allow us to simultaneously extract rich discriminative topological information from graphs. Experiments on 11 diverse benchmark datasets against 18 baseline models in conjunction with graph classification tasks indicate that Wit-TopoPool significantly outperforms all competitors across all datasets.

        ----

        ## [797] Riemannian Local Mechanism for SPD Neural Networks

        **Authors**: *Ziheng Chen, Tianyang Xu, Xiao-Jun Wu, Rui Wang, Zhiwu Huang, Josef Kittler*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25867](https://doi.org/10.1609/aaai.v37i6.25867)

        **Abstract**:

        The Symmetric Positive Definite (SPD) matrices have received wide attention for data representation in many scientific areas. Although there are many different attempts to develop effective deep architectures for data processing on the Riemannian manifold of SPD matrices, very few solutions explicitly mine the local geometrical information in deep SPD feature representations. Given the great success of local mechanisms in Euclidean methods, we argue that it is of utmost importance to ensure the preservation of local geometric information in the SPD networks. We first analyse the convolution operator commonly used for capturing local information in Euclidean deep networks from the perspective of a higher level of abstraction afforded by category theory. Based on this analysis, we define the local information in the SPD manifold and design a multi-scale submanifold block for mining local geometry. Experiments involving multiple visual tasks validate the effectiveness of our approach.

        ----

        ## [798] TC-DWA: Text Clustering with Dual Word-Level Augmentation

        **Authors**: *Bo Cheng, Ximing Li, Yi Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25868](https://doi.org/10.1609/aaai.v37i6.25868)

        **Abstract**:

        The pre-trained language models, e.g., ELMo and BERT, have recently achieved promising performance improvement in a wide range of NLP tasks, because they can output strong contextualized embedded features of words. Inspired by their great success, in this paper we target at fine-tuning them to effectively handle the text clustering task, i.e., a classic and fundamental challenge in machine learning. Accordingly, we propose a novel BERT-based method, namely Text Clustering with Dual Word-level Augmentation (TCDWA). To be specific, we formulate a self-training objective and enhance it with a dual word-level augmentation technique. First, we suppose that each text contains several most informative words, called anchor words, supporting the full text semantics. We use the embedded features of anchor words as augmented data, which are selected by ranking the norm-based attention weights of words. Second, we formulate an expectation form of word augmentation, which is equivalent to generating infinite augmented features, and further suggest a tractable approximation of Taylor expansion for efficient optimization. To evaluate the effectiveness of TCDWA, we conduct extensive experiments on several benchmark text datasets. The results demonstrate that TCDWA consistently outperforms the state-of-the-art baseline methods. Code available: https://github.com/BoCheng-96/TC-DWA.

        ----

        ## [799] Causal Inference with Conditional Instruments Using Deep Generative Models

        **Authors**: *Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, Thuc Duy Le*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i6.25869](https://doi.org/10.1609/aaai.v37i6.25869)

        **Abstract**:

        The instrumental variable (IV) approach is a widely used way to estimate the causal effects of a treatment on an outcome of interest from observational data with latent confounders. A standard IV is expected to be related to the treatment variable and independent of all other variables in the system. However, it is challenging to search for a standard IV from data directly due to the strict conditions. The conditional IV (CIV) method has been proposed to allow a variable to be an instrument conditioning on a set of variables, allowing a wider choice of possible IVs and enabling broader practical applications of the IV approach. Nevertheless, there is not a data-driven method to discover a CIV and its conditioning set directly from data. To fill this gap, in this paper, we propose to learn the representations of the information of a CIV and its conditioning set from data with latent confounders for average causal effect estimation. By taking advantage of deep generative models, we develop a novel data-driven approach for simultaneously learning the representation of a CIV from measured variables and generating the representation of its conditioning set given measured variables. Extensive experiments on synthetic and real-world datasets show that our method outperforms the existing IV methods.

        ----

        

[Go to the previous page](AAAI-2023-list03.md)

[Go to the next page](AAAI-2023-list05.md)

[Go to the catalog section](README.md)