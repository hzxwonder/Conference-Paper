## [1000] STEM: Unleashing the Power of Embeddings for Multi-Task Recommendation

**Authors**: *Liangcai Su, Junwei Pan, Ximei Wang, Xi Xiao, Shijie Quan, Xihua Chen, Jie Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28749](https://doi.org/10.1609/aaai.v38i8.28749)

**Abstract**:

Multi-task learning (MTL) has gained significant popularity in recommender systems as it enables simultaneous optimization of multiple objectives. A key challenge in MTL is negative transfer, but existing studies explored negative transfer on all samples, overlooking the inherent complexities within them. We split the samples according to the relative amount of positive feedback among tasks. Surprisingly, negative transfer still occurs in existing MTL methods on samples that receive comparable feedback across tasks. Existing work commonly employs a shared-embedding paradigm, limiting the ability of modeling diverse user preferences on different tasks. In this paper, we introduce a novel Shared and Task-specific EMbeddings (STEM) paradigm that aims to incorporate both shared and task-specific embeddings to effectively capture task-specific user preferences. Under this paradigm, we propose a simple model STEM-Net, which is equipped with an All Forward Task-specific Backward gating network to facilitate the learning of task-specific embeddings and direct knowledge transfer across tasks. Remarkably, STEM-Net demonstrates exceptional performance on comparable samples, achieving positive transfer. Comprehensive evaluation on three public MTL recommendation datasets demonstrates that STEM-Net outperforms state-of-the-art models by a substantial margin. Our code is released at https://github.com/LiangcaiSu/STEM.

----

## [1001] Anchoring Path for Inductive Relation Prediction in Knowledge Graphs

**Authors**: *Zhixiang Su, Di Wang, Chunyan Miao, Lizhen Cui*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28750](https://doi.org/10.1609/aaai.v38i8.28750)

**Abstract**:

Aiming to accurately predict missing edges representing relations between entities, which are pervasive in real-world Knowledge Graphs (KGs), relation prediction plays a critical role in enhancing the comprehensiveness and utility of KGs. Recent research focuses on path-based methods due to their inductive and explainable properties. However, these methods face a great challenge when lots of reasoning paths do not form Closed Paths (CPs) in the KG. To address this challenge, we propose Anchoring Path Sentence Transformer (APST) by introducing Anchoring Paths (APs) to alleviate the reliance of CPs. Specifically, we develop a search-based description retrieval method to enrich entity descriptions and an assessment mechanism to evaluate the rationality of APs. APST takes both APs and CPs as the inputs of a unified Sentence Transformer architecture, enabling comprehensive predictions and high-quality explanations. We evaluate APST on three public datasets and achieve state-of-the-art (SOTA) performance in 30 of 36 transductive, inductive, and few-shot experimental settings.

----

## [1002] MAPTree: Beating "Optimal" Decision Trees with Bayesian Decision Trees

**Authors**: *Colin Sullivan, Mo Tiwari, Sebastian Thrun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28751](https://doi.org/10.1609/aaai.v38i8.28751)

**Abstract**:

Decision trees remain one of the most popular machine learning models today, largely due to their out-of-the-box performance and interpretability. In this work, we present a Bayesian approach to decision tree induction via maximum a posteriori inference of a posterior distribution over trees. We first demonstrate a connection between maximum a posteriori inference of decision trees and AND/OR search. Using this connection, we propose an AND/OR search algorithm, dubbed MAPTree, which is able to recover the maximum a posteriori tree. Lastly, we demonstrate the empirical performance of the maximum a posteriori tree both on synthetic data and in real world settings. On 16 real world datasets, MAPTree either outperforms baselines or demonstrates comparable performance but with much smaller trees. On a synthetic dataset, MAPTree also demonstrates greater robustness to noise and better generalization than existing approaches. Finally, MAPTree recovers the maxiumum a posteriori tree faster than existing sampling approaches and, in contrast with those algorithms, is able to provide a certificate of optimality. The code for our experiments is available at https://github.com/ThrunGroup/maptree.

----

## [1003] CREAD: A Classification-Restoration Framework with Error Adaptive Discretization for Watch Time Prediction in Video Recommender Systems

**Authors**: *Jie Sun, Zhaoying Ding, Xiaoshuang Chen, Qi Chen, Yincheng Wang, Kaiqiao Zhan, Ben Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28752](https://doi.org/10.1609/aaai.v38i8.28752)

**Abstract**:

The watch time is a significant indicator of user satisfaction in video recommender systems. However, the prediction of watch time as a target variable is often hindered by its highly imbalanced distribution with a scarcity of observations for larger target values and over-populated samples for small values. State-of-the-art watch time prediction models discretize the continuous watch time into a set of buckets in order to consider the distribution of watch time. However, it is highly uninvestigated how these discrete buckets should be created from the continuous watch time distribution, and existing discretization approaches suffer from either a large learning error or a large restoration error. To address this challenge, we propose a Classification-Restoration framework with Error-Adaptive-Discretization (CREAD) to accurately predict the watch time. The proposed framework contains a discretization module, a classification module, and a restoration module. It predicts the watch time through multiple classification problems. The discretization process is a key contribution of the CREAD framework. We theoretically analyze the impacts of the discretization on the learning error and the restoration error, and then propose the error-adaptive discretization (EAD) technique to better balance the two errors, which achieves better performance over traditional discretization approaches. We conduct detailed offline evaluations on a public dataset and an industrial dataset, both showing performance gains through the proposed approach. Moreover, We have fully launched our framework to an online video platform, which resulted in a significant increase in users' video watch time by 0.29% through A/B testing. These results highlight the effectiveness of the CREAD framework in watch time prediction in video recommender systems.

----

## [1004] ModWaveMLP: MLP-Based Mode Decomposition and Wavelet Denoising Model to Defeat Complex Structures in Traffic Forecasting

**Authors**: *Ke Sun, Pei Liu, Pengfei Li, Zhifang Liao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28753](https://doi.org/10.1609/aaai.v38i8.28753)

**Abstract**:

Traffic prediction is the core issue of Intelligent Transportation Systems. Recently, researchers have tended to use complex structures, such as transformer-based structures, for tasks such as traffic prediction. Notably, traffic data is simpler to process compared to text and images, which raises questions about the necessity of these structures. Additionally, when handling traffic data, researchers tend to manually design the model structure based on the data features, which makes the structure of traffic prediction redundant and the model generalizability limited. To address the above, we introduce the ‘ModWaveMLP’—A multilayer perceptron (MLP) based model designed according to mode decomposition and wavelet noise reduction information learning concepts. The model is based on simple MLP structure, which achieves the separation and prediction of different traffic modes and does not depend on additional features introduced such as the topology of the traffic network. By performing experiments on real-world datasets METR-LA and PEMS-BAY, our model achieves SOTA, outperforms GNN and transformer-based models, and outperforms those that introduce additional feature data with better generalizability, and we further demonstrate the effectiveness of the various parts of the model through ablation experiments. This offers new insights to subsequent researchers involved in traffic model design. The code is available at: https://github.com/Kqingzheng/ModWaveMLP.

----

## [1005] Motif-Aware Riemannian Graph Neural Network with Generative-Contrastive Learning

**Authors**: *Li Sun, Zhenhao Huang, Zixi Wang, Feiyang Wang, Hao Peng, Philip S. Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28754](https://doi.org/10.1609/aaai.v38i8.28754)

**Abstract**:

Graphs are typical non-Euclidean data of complex structures. In recent years, Riemannian graph representation learning has emerged as an exciting alternative to Euclidean ones. However, Riemannian methods are still in an early stage: most of them present a single curvature (radius) regardless of structural complexity, suffer from numerical instability due to the exponential/logarithmic map, and lack the ability to capture motif regularity. In light of the issues above, we propose the problem of Motif-aware Riemannian Graph Representation Learning, seeking a numerically stable encoder to capture motif regularity in a diverse-curvature manifold without labels. To this end, we present a novel Motif-aware Riemannian model with Generative-Contrastive learning (MotifRGC), which conducts a minmax game in Riemannian manifold in a self-supervised manner. First, we propose a new type of Riemannian GCN (D-GCN), in which we construct a diverse-curvature manifold by a product layer with the diversified factor, and replace the exponential/logarithmic map by a stable kernel layer. Second, we introduce a motif-aware Riemannian generative-contrastive learning to capture motif regularity in the constructed manifold and learn motif-aware node representation without external labels. Empirical results show the superiority of MofitRGC.

----

## [1006] Fine-Tuning Graph Neural Networks by Preserving Graph Generative Patterns

**Authors**: *Yifei Sun, Qi Zhu, Yang Yang, Chunping Wang, Tianyu Fan, Jiajun Zhu, Lei Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28755](https://doi.org/10.1609/aaai.v38i8.28755)

**Abstract**:

Recently, the paradigm of pre-training and fine-tuning graph neural networks has been intensively studied and applied in a wide range of graph mining tasks. 
Its success is generally attributed to the structural consistency between pre-training and downstream datasets, which, however, does not hold in many real-world scenarios. 
Existing works have shown that the structural divergence between pre-training and downstream graphs significantly limits the transferability when using the vanilla fine-tuning strategy. This divergence leads to model overfitting on pre-training graphs and causes difficulties in capturing the structural properties of the downstream graphs. 
In this paper, we identify the fundamental cause of structural divergence as the discrepancy of generative patterns between the pre-training and downstream graphs.
Furthermore, we propose G-Tuning to preserve the generative patterns of downstream graphs. 
Given a downstream graph G, the core idea is to tune the pre-trained GNN so that it can reconstruct the generative patterns of G, the graphon W.
However, the exact reconstruction of a graphon is known to be computationally expensive. To overcome this challenge, we provide a theoretical analysis that establishes the existence of a set of alternative graphons called graphon bases for any given graphon. By utilizing a linear combination of these graphon bases, we can efficiently approximate W. This theoretical finding forms the basis of our model, as it enables effective learning of the graphon bases and their associated coefficients.
Compared with existing algorithms, G-Tuning demonstrates consistent performance improvement in 7 in-domain and 7 out-of-domain transfer learning experiments.

----

## [1007] Finding Interpretable Class-Specific Patterns through Efficient Neural Search

**Authors**: *Nils Philipp Walter, Jonas Fischer, Jilles Vreeken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28756](https://doi.org/10.1609/aaai.v38i8.28756)

**Abstract**:

Discovering patterns in data that best describe the differences between classes allows to
hypothesize and reason about class-specific mechanisms. In molecular biology, for example,
these bear the promise of advancing the understanding of cellular processes differing between
tissues or diseases, which could lead to novel treatments. To be useful in practice, methods
that tackle the problem of finding such differential patterns have to be readily interpretable by
domain experts, and scalable to the extremely high-dimensional data.

In this work, we propose a novel, inherently interpretable binary neural network architecture
Diffnaps that extracts differential patterns from data. Diffnaps is scalable to hundreds
of thousands of features and robust to noise, thus overcoming the limitations of current
state-of-the-art methods in large-scale applications such as in biology. We show on synthetic
and real world data, including three biological applications, that unlike its competitors,
Diffnaps consistently yields accurate, succinct, and interpretable class descriptions.

----

## [1008] End-to-End Learning of LTLf Formulae by Faithful LTLf Encoding

**Authors**: *Hai Wan, Pingjia Liang, Jianfeng Du, Weilin Luo, Rongzhen Ye, Bo Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28757](https://doi.org/10.1609/aaai.v38i8.28757)

**Abstract**:

It is important to automatically discover the underlying tree-structured formulae from large amounts of data. In this paper, we examine learning linear temporal logic on finite traces (LTLf) formulae, which is a tree structure syntactically and characterizes temporal properties semantically. Its core challenge is to bridge the gap between the concise tree-structured syntax and the complex LTLf semantics. Besides, the learning quality is endangered by explosion of the search space and wrong search bias guided by imperfect data. We tackle these challenges by proposing an LTLf encoding method to parameterize a neural network so that the neural computation is able to simulate the inference of LTLf formulae. We first identify faithful LTLf encoding, a subclass of LTLf encoding, which has a one-to-one correspondence to LTLf formulae. Faithful encoding guarantees that the learned parameter assignment of the neural network can directly be interpreted to an LTLf formula. With such an encoding method, we then propose an end-to-end approach, TLTLf, to learn LTLf formulae through neural networks parameterized by our LTLf encoding method. Experimental results demonstrate that our approach achieves state-of-the-art performance with up to 7% improvement in accuracy, highlighting the benefits of introducing the faithful LTLf encoding.

----

## [1009] Contributing Dimension Structure of Deep Feature for Coreset Selection

**Authors**: *Zhijing Wan, Zhixiang Wang, Yuran Wang, Zheng Wang, Hongyuan Zhu, Shin'ichi Satoh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28758](https://doi.org/10.1609/aaai.v38i8.28758)

**Abstract**:

Coreset selection seeks to choose a subset of crucial training samples for efficient learning. It has gained traction in deep learning, particularly with the surge in training dataset sizes. Sample selection hinges on two main aspects: a sample's representation in enhancing performance and the role of sample diversity in averting overfitting. Existing methods typically measure both the representation and diversity of data based on similarity metrics, such as L2-norm. They have capably tackled representation via distribution matching guided by the similarities of features, gradients, or other information between data. However, the results of effectively diverse sample selection are mired in sub-optimality. This is because the similarity metrics usually simply aggregate dimension similarities without acknowledging disparities among the dimensions that significantly contribute to the final similarity. As a result, they fall short of adequately capturing diversity. To address this, we propose a feature-based diversity constraint, compelling the chosen subset to exhibit maximum diversity. Our key lies in the introduction of a novel Contributing Dimension Structure (CDS) metric. Different from similarity metrics that measure the overall similarity of high-dimensional features, our CDS metric considers not only the reduction of redundancy in feature dimensions, but also the difference between dimensions that contribute significantly to the final similarity. We reveal that existing methods tend to favor samples with similar CDS, leading to a reduced variety of CDS types within the coreset and subsequently hindering model performance. In response, we enhance the performance of five classical selection methods by integrating the CDS constraint. Our experiments on three datasets demonstrate the general effectiveness of the proposed method in boosting existing methods.

----

## [1010] Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective

**Authors**: *Binwu Wang, Pengkun Wang, Yudong Zhang, Xu Wang, Zhengyang Zhou, Lei Bai, Yang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28759](https://doi.org/10.1609/aaai.v38i8.28759)

**Abstract**:

With the progress of urban transportation systems, a significant amount of high-quality traffic data is continuously collected through streaming manners, which has propelled the prosperity of the field of spatial-temporal graph prediction.  In this paper, rather than solely focusing on designing powerful models for static graphs, we shift our focus to spatial-temporal graph prediction in the dynamic scenario, which involves a continuously expanding and evolving underlying graph.  To address inherent challenges, a decoupled learning framework (DLF) is proposed in this paper, which consists of a spatial-temporal graph learning network (DSTG) with a specialized decoupling training strategy.  Incorporating inductive biases of time-series structures, DSTG can interpret time dependencies into latent trend and seasonal terms.  To enable prompt adaptation to the evolving distribution of the dynamic graph, our decoupling training strategy is devised to iteratively update these two types of patterns.  Specifically, for learning seasonal patterns, we conduct thorough training for the model using a long time series (e.g., three months of data).  To enhance the learning ability of the model, we also introduce the masked auto-encoding mechanism.  During this period, we frequently update trend patterns to expand new information from dynamic graphs.  Considering both effectiveness and efficiency, we develop a subnet sampling strategy to select a few representative nodes for fine-tuning the weights of the model.  These sampled nodes cover unseen patterns and previously learned patterns.  Experiments on dynamic spatial-temporal graph datasets further demonstrate the competitive performance, superior efficiency, and strong scalability of the proposed framework.

----

## [1011] EnMatch: Matchmaking for Better Player Engagement via Neural Combinatorial Optimization

**Authors**: *Kai Wang, Haoyu Liu, Zhipeng Hu, Xiaochuan Feng, Minghao Zhao, Shiwei Zhao, Runze Wu, Xudong Shen, Tangjie Lv, Changjie Fan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28760](https://doi.org/10.1609/aaai.v38i8.28760)

**Abstract**:

Matchmaking is a core task in e-sports and online games, as it contributes to player engagement and further influences the game's lifecycle. Previous methods focus on creating fair games at all times. They divide players into different tiers based on skill levels and only select players from the same tier for each game. Though this strategy can ensure fair matchmaking, it is not always good for player engagement. In this paper, we propose a novel Engagement-oriented Matchmaking (EnMatch) framework to ensure fair games and simultaneously enhance player engagement. Two main issues need to be addressed. First, it is unclear how to measure the impact of different team compositions and confrontations on player engagement during the game considering the variety of player characteristics. Second, such a detailed consideration on every single player during matchmaking will result in an NP-hard combinatorial optimization problem with non-linear objectives. In light of these challenges, we turn to real-world data analysis to reveal engagement-related factors. The resulting insights guide the development of engagement modeling, enabling the estimation of quantified engagement before a match is completed. To handle the combinatorial optimization problem, we formulate the problem into a reinforcement learning framework, in which a neural combinatorial optimization problem is built and solved. The performance of EnMatch is finally demonstrated through the comparison with other state-of-the-art methods based on several real-world datasets and online deployments on two games.

----

## [1012] Review-Enhanced Hierarchical Contrastive Learning for Recommendation

**Authors**: *Ke Wang, Yanmin Zhu, Tianzi Zang, Chunyang Wang, Mengyuan Jing*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28761](https://doi.org/10.1609/aaai.v38i8.28761)

**Abstract**:

Designed to establish potential relations and distill high-order representations, graph-based recommendation systems continue to reveal promising results by jointly modeling ratings and reviews. However, existing studies capture simple review relations, failing to (1) completely explore hidden connections between users (or items), (2) filter out redundant information derived from reviews, and (3) model the behavioral association between rating and review interactions. To address these challenges, we propose a review-enhanced hierarchical contrastive learning, namely ReHCL. First, ReHCL constructs topic and semantic graphs to fully mine review relations from different views. Moreover, a cross-view graph contrastive learning is used to achieve enhancement of node representations and extract useful review knowledge. Meanwhile, we design a neighbor-based positive sampling to capture the graph-structured similarity between topic and semantic views, further performing efficient contrast and reducing redundant noise. Next, we propose a cross-modal contrastive learning to match the rating and review representations, by exploring the association between ratings and reviews. Lastly, these two contrastive learning modes form a hierarchical contrastive learning task, which is applied to enhance the final recommendation task. Extensive experiments verify the superiority of ReHCL compared with state-of-the-arts.

----

## [1013] Pseudo-Label Calibration Semi-supervised Multi-Modal Entity Alignment

**Authors**: *Luyao Wang, Pengnian Qi, Xigang Bao, Chunlai Zhou, Biao Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28762](https://doi.org/10.1609/aaai.v38i8.28762)

**Abstract**:

Multi-modal entity alignment (MMEA) aims to identify equivalent entities between two multi-modal knowledge graphs for integration. Unfortunately, prior arts have attempted to improve the interaction and fusion of multi-modal information, which have overlooked the influence of modal-specific noise and the usage of labeled and unlabeled data in semi-supervised settings. In this work, we introduce a Pseudo-label Calibration Multi-modal Entity Alignment (PCMEA) in a semi-supervised way. Specifically, in order to generate holistic entity representations, we first devise various embedding modules and attention mechanisms to extract visual, structural, relational, and attribute features. Different from the prior direct fusion methods, we next propose to exploit mutual information maximization to filter the modal-specific noise and to augment modal-invariant commonality. Then, we combine pseudo-label calibration with momentum-based contrastive learning to make full use of the labeled and unlabeled data, which improves the quality of pseudo-label and pulls aligned entities closer. Finally, extensive experiments on two MMEA datasets demonstrate the effectiveness of our PCMEA, which yields state-of-the-art performance.

----

## [1014] Preference Aware Dual Contrastive Learning for Item Cold-Start Recommendation

**Authors**: *Wenbo Wang, Bingquan Liu, Lili Shan, Chengjie Sun, Ben Chen, Jian Guan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28763](https://doi.org/10.1609/aaai.v38i8.28763)

**Abstract**:

Existing cold-start recommendation methods often adopt item-level alignment strategies to align the content feature and the collaborative feature of warm items for model training, however, cold items in the test stage have no historical interactions with users to obtain the collaborative feature. These existing models ignore the aforementioned condition of cold items in the training stage, resulting in the performance limitation. In this paper, we propose a preference aware dual contrastive learning based recommendation model (PAD-CLRec), where the user preference is explored to take into account the condition of cold items for feature alignment. Here, the user preference is obtained by aggregating a group of collaborative feature of the warm items in the user's purchase records. Then, a group-level alignment between the user preference and the item's content feature can be realized via a proposed preference aware contrastive function for enhancing cold-item recommendation. In addition, a joint objective function is introduced to achieve a better trade-off between the recommendation performance of warm items and cold items from both item-level and group-level perspectives, yielding better overall recommendation performance. Extensive experiments are conducted to demonstrate the effectiveness of the proposed method, and the results show the superiority of our method, as compared with the state-of-the-arts.

----

## [1015] Deciphering Compatibility Relationships with Textual Descriptions via Extraction and Explanation

**Authors**: *Yu Wang, Zexue He, Zhankui He, Hao Xu, Julian J. McAuley*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28764](https://doi.org/10.1609/aaai.v38i8.28764)

**Abstract**:

Understanding and accurately explaining compatibility relationships between fashion items is a challenging problem in the burgeoning domain of AI-driven outfit recommendations. Present models, while making strides in this area, still occasionally fall short, offering explanations that can be elementary and repetitive. This work aims to address these shortcomings by introducing the Pair Fashion Explanation (PFE) dataset, a unique resource that has been curated to illuminate these compatibility relationships. Furthermore, we propose an innovative two stage pipeline model that leverages this dataset. This fine-tuning allows the model to generate explanations that convey the compatibility relationships between items. Our experiments showcase the model's potential in crafting descriptions that are knowledgeable, aligned with ground-truth matching correlations, and that produce understandable and informative descriptions, as assessed by both automatic metrics and human evaluation. Our code and data are released at https://github.com/wangyu-ustc/PairFashionExplanation.

----

## [1016] Open-Set Graph Domain Adaptation via Separate Domain Alignment

**Authors**: *Yu Wang, Ronghang Zhu, Pengsheng Ji, Sheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28765](https://doi.org/10.1609/aaai.v38i8.28765)

**Abstract**:

Domain adaptation has become an attractive learning paradigm, as it can leverage source domains with rich labels to deal with classification tasks in an unlabeled target domain. A few recent studies develop domain adaptation approaches for graph-structured data. In the case of node classification task, current domain adaptation methods only focus on the closed-set setting, where source and target domains share the same label space. A more practical assumption is that the target domain may contain new classes that are not included in the source domain. Therefore, in this paper, we introduce a novel and challenging problem for graphs, i.e., open-set domain adaptive node classification, and propose a new approach to solve it. Specifically, we develop an algorithm for efficient knowledge transfer from a labeled source graph to an unlabeled target graph under a separate domain alignment (SDA) strategy, in order to learn discriminative feature representations for the target graph. Our goal is to not only correctly classify target nodes into the known classes, but also classify unseen types of nodes into an unknown class. Experimental results on real-world datasets show that our method outperforms existing methods on graph domain adaptation.

----

## [1017] G^2SAM: Graph-Based Global Semantic Awareness Method for Multimodal Sarcasm Detection

**Authors**: *Yiwei Wei, Shaozu Yuan, Hengyang Zhou, Longbiao Wang, Zhiling Yan, Ruosong Yang, Meng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28766](https://doi.org/10.1609/aaai.v38i8.28766)

**Abstract**:

Multimodal sarcasm detection, aiming to detect the ironic sentiment within multimodal social data, has gained substantial popularity in both the natural language processing and computer vision communities. Recently, graph-based studies by drawing sentimental relations to detect multimodal sarcasm have made notable advancements. However, they have neglected exploiting graph-based global semantic congruity from existing instances to facilitate the prediction, which ultimately hinders the model's performance. In this paper, we introduce a new inference paradigm that leverages global graph-based semantic awareness to handle this task. Firstly, we construct fine-grained multimodal graphs for each instance and integrate them into semantic space to draw graph-based relations. During inference, we leverage global semantic congruity to retrieve k-nearest neighbor instances in semantic space as references for voting on the final prediction. To enhance the semantic correlation of representation in semantic space, we also introduce label-aware graph contrastive learning to further improve the performance. Experimental results demonstrate that our model achieves state-of-the-art (SOTA) performance in multimodal sarcasm detection. The code will be available at https://github.com/upccpu/G2SAM.

----

## [1018] Poincaré Differential Privacy for Hierarchy-Aware Graph Embedding

**Authors**: *Yuecen Wei, Haonan Yuan, Xingcheng Fu, Qingyun Sun, Hao Peng, Xianxian Li, Chunming Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28767](https://doi.org/10.1609/aaai.v38i8.28767)

**Abstract**:

Hierarchy is an important and commonly observed topological property in real-world graphs that indicate the relationships between supervisors and subordinates or the organizational behavior of human groups. As hierarchy is introduced as a new inductive bias into the Graph Neural Networks (GNNs) in various tasks, it implies latent topological relations for attackers to improve their inference attack performance, leading to serious privacy leakage issues. In addition, existing privacy-preserving frameworks suffer from reduced protection ability in hierarchical propagation due to the deficiency of adaptive upper-bound estimation of the hierarchical perturbation boundary. It is of great urgency to effectively leverage the hierarchical property of data while satisfying privacy guarantees. To solve the problem, we propose the Poincar\'e Differential Privacy framework, named PoinDP, to protect the hierarchy-aware graph embedding based on hyperbolic geometry. Specifically, PoinDP first learns the hierarchy weights for each entity based on the Poincar\'e model in hyperbolic space. Then, the Personalized Hierarchy-aware Sensitivity is designed to measure the sensitivity of the hierarchical structure and adaptively allocate the privacy protection strength. Besides, Hyperbolic Gaussian Mechanism (HGM) is proposed to extend the Gaussian mechanism in Euclidean space to hyperbolic space to realize random perturbations that satisfy differential privacy under the hyperbolic space metric. Extensive experiment results on five real-world datasets demonstrate the proposed PoinDP’s advantages of effective privacy protection while maintaining good performance on the node classification task.

----

## [1019] Pairwise-Label-Based Deep Incremental Hashing with Simultaneous Code Expansion

**Authors**: *Dayan Wu, Qinghang Su, Bo Li, Weiping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28768](https://doi.org/10.1609/aaai.v38i8.28768)

**Abstract**:

Deep incremental hashing has become a subject of considerable interest due to its capability to learn hash codes in an incremental manner, eliminating the need to generate codes for classes that have already been learned. However, accommodating more classes requires longer hash codes, and regenerating database codes becomes inevitable when code expansion is required.
In this paper, we present a unified deep hash framework that can simultaneously learn new classes and increase hash code capacity. Specifically, we design a triple-channel asymmetric framework to optimize a new CNN model with a target code length and a code projection matrix. This enables us to directly generate hash codes for new images, and efficiently generate expanded hash codes for original database images from the old ones with the learned projection matrix.
Meanwhile, we propose a pairwise-label-based incremental similarity-preserving loss to optimize the new CNN model, which can incrementally preserve new similarities while maintaining the old ones. Additionally, we design a double-end quantization loss to reduce the quantization error from new and original query images. As a result, our method efficiently embeds both new and original similarities into the expanded hash codes, while keeping the original database codes unchanged.
We conduct extensive experiments on three widely-used image retrieval benchmarks, demonstrating that our method can significantly reduce the time required to expand existing database codes, while maintaining state-of-the-art retrieval performance.

----

## [1020] Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations

**Authors**: *Likang Wu, Zhaopeng Qiu, Zhi Zheng, Hengshu Zhu, Enhong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28769](https://doi.org/10.1609/aaai.v38i8.28769)

**Abstract**:

Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for graph semantic mining in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including promoting out-of-distribution (OOD) applications. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that aids LLM recommender in grasping the semantics of behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By facilitating this capability, our framework enables personalized and accurate job recommendations for individual users. We evaluate the effectiveness of our approach on comprehensive real-world datasets and demonstrate its ability to improve the relevance and quality of recommended results. This research not only sheds light on the untapped potential of large language models but also provides valuable insights for developing advanced recommendation systems in the recruitment market. The findings contribute to the growing field of natural language processing and offer practical implications for enhancing job search experiences.

----

## [1021] CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph

**Authors**: *Hongjie Xia, Huijie Ao, Long Li, Yu Liu, Sen Liu, Guangnan Ye, Hongfeng Chai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28770](https://doi.org/10.1609/aaai.v38i8.28770)

**Abstract**:

Quantitative stock selection is one of the most challenging FinTech tasks due to the non-stationary dynamics and complex market dependencies. Existing studies rely on channel mixing methods, exacerbating the issue of distribution shift in financial time series. Additionally, complex model structures they build make it difficult to handle very long sequences. Furthermore, most of them are based on predefined stock relationships thus making it difficult to capture the dynamic and highly volatile stock markets. To address the above issues, in this paper, we propose Channel-Independent based Spatio-Temporal Hypergraph Pre-trained Attention Networks (CI-STHPAN), a two-stage framework for stock selection, involving Transformer and HGAT based stock time series self-supervised pre-training and stock-ranking based downstream task fine-tuning. We calculate the similarity of stock time series of different channel in dynamic intervals based on Dynamic Time Warping (DTW), and further construct channel-independent stock dynamic hypergraph based on the similarity. Experiments with NASDAQ and NYSE markets data over five years show that our framework outperforms SOTA approaches in terms of investment return ratio (IRR) and Sharpe ratio (SR). Additionally, we find that even without introducing graph information, self-supervised learning based on the vanilla Transformer Encoder also surpasses SOTA results. Notable improvements are gained on the NYSE market. It is mainly attributed to the improvement of fine-tuning approach on Information Coefficient (IC) and Information Ratio based IC (ICIR), indicating that the fine-tuning method enhances the accuracy and stability of the model prediction.

----

## [1022] Feature Distribution Matching by Optimal Transport for Effective and Robust Coreset Selection

**Authors**: *Weiwei Xiao, Yongyong Chen, Qiben Shan, Yaowei Wang, Jingyong Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28771](https://doi.org/10.1609/aaai.v38i8.28771)

**Abstract**:

Training neural networks with good generalization requires large computational costs in many deep learning methods due to large-scale datasets and over-parameterized models. Despite the emergence of a number of coreset selection methods to reduce the computational costs, the problem of coreset distribution bias, i.e., the skewed distribution between the coreset and the entire dataset, has not been well studied. In this paper, we find that the closer the feature distribution of the coreset is to that of the entire dataset, the better the generalization performance of the coreset, particularly under extreme pruning. This motivates us to propose a simple yet effective method for coreset selection to alleviate the distribution bias between the coreset and the entire dataset, called feature distribution matching (FDMat). Unlike gradient-based methods, which selects samples with larger gradient values or approximates gradient values of the entire dataset, FDMat aims to select coreset that is closest to feature distribution of the entire dataset. Specifically, FDMat transfers coreset selection as an optimal transport problem from the coreset to the entire dataset in feature embedding spaces. Moreover, our method shows strong robustness due to the removal of samples far from the distribution, especially for the entire dataset containing noisy and class-imbalanced samples. Extensive experiments on multiple benchmarks show that FDMat can improve the performance of coreset selection than existing coreset methods. The code is available at https://github.com/successhaha/FDMat.

----

## [1023] NestE: Modeling Nested Relational Structures for Knowledge Graph Reasoning

**Authors**: *Bo Xiong, Mojtaba Nayyeri, Linhao Luo, Zihao Wang, Shirui Pan, Steffen Staab*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28772](https://doi.org/10.1609/aaai.v38i8.28772)

**Abstract**:

Reasoning with knowledge graphs (KGs) has primarily focused on triple-shaped facts. Recent advancements have been explored to enhance the semantics of these facts by incorporating more potent representations, such as hyper-relational facts. However, these approaches are limited to atomic facts, which describe a single piece of information. This paper extends beyond atomic facts and delves into nested facts, represented by quoted triples where subjects and objects are triples themselves (e.g., ((BarackObama, holds_position, President), succeed_by, (DonaldTrump, holds_position, President))). These nested facts enable the expression of complex semantics like situations over time and logical patterns} over entities and relations. In response, we introduce NestE, a novel KG embedding approach that captures the semantics of both atomic and nested factual knowledge. NestE represents each atomic fact as a 1*3 matrix, and each nested relation is modeled as a 3*3 matrix that rotates the 1*3 atomic fact matrix through matrix multiplication. Each element of the matrix is represented as a complex number in the generalized 4D hypercomplex space, including (spherical) quaternions, hyperbolic quaternions, and split-quaternions. Through thorough analysis, we demonstrate the embedding's efficacy in capturing diverse logical patterns over nested facts, surpassing the confines of first-order logic-like expressions. Our experimental results showcase NestE's significant performance gains over current baselines in triple prediction and conditional link prediction. The code and pre-trained models are open available at https://github.com/xiongbo010/NestE.

----

## [1024] Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum

**Authors**: *Fan Xu, Nan Wang, Hao Wu, Xuezhi Wen, Xibin Zhao, Hai Wan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28773](https://doi.org/10.1609/aaai.v38i8.28773)

**Abstract**:

Graph-based fraud detection (GFD) can be regarded as a challenging semi-supervised node binary classification task. In recent years, Graph Neural Networks (GNN) have been widely applied to GFD, characterizing the anomalous possibility of a node by aggregating neighbor information. However, fraud graphs are inherently heterophilic, thus most of GNNs perform poorly due to their assumption of homophily. In addition, due to the existence of heterophily and class imbalance problem, the existing models do not fully utilize the precious node label information. To address the above issues, this paper proposes a semi-supervised GNN-based fraud detector SEC-GFD. This detector includes a hybrid filtering module and a local environmental constraint module, the two modules are utilized to solve heterophily and label utilization problem respectively. The first module starts from the perspective of the spectral domain, and solves the heterophily problem to a certain extent. Specifically, it divides the spectrum into various mixed-frequency bands based on the correlation between spectrum energy distribution and heterophily. Then in order to make full use of the node label information, a local environmental constraint module is adaptively designed. The comprehensive experimental results on four real-world fraud detection datasets denote that SEC-GFD outperforms other competitive graph-based fraud detectors. We release our code at https://github.com/Sunxkissed/SEC-GFD.

----

## [1025] Empowering Dual-Level Graph Self-Supervised Pretraining with Motif Discovery

**Authors**: *Pengwei Yan, Kaisong Song, Zhuoren Jiang, Yangyang Kang, Tianqianjin Lin, Changlong Sun, Xiaozhong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28774](https://doi.org/10.1609/aaai.v38i8.28774)

**Abstract**:

While self-supervised graph pretraining techniques have shown promising results in various domains, their application still experiences challenges of limited topology learning, human knowledge dependency, and incompetent multi-level interactions. To address these issues, we propose a novel solution, Dual-level Graph self-supervised Pretraining with Motif discovery (DGPM), which introduces a unique dual-level pretraining structure that orchestrates node-level and subgraph-level pretext tasks. Unlike prior approaches, DGPM autonomously uncovers significant graph motifs through an edge pooling module, aligning learned motif similarities with graph kernel-based similarities. A cross-matching task enables sophisticated node-motif interactions and novel representation learning. Extensive experiments on 15 datasets validate DGPM's effectiveness and generalizability, outperforming state-of-the-art methods in unsupervised representation learning and transfer learning settings. The autonomously discovered motifs demonstrate the potential of DGPM to enhance robustness and interpretability.

----

## [1026] Hypergraph Joint Representation Learning for Hypervertices and Hyperedges via Cross Expansion

**Authors**: *Yuguang Yan, Yuanlin Chen, Shibo Wang, Hanrui Wu, Ruichu Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28775](https://doi.org/10.1609/aaai.v38i8.28775)

**Abstract**:

Hypergraph captures high-order information in structured data and obtains much attention in machine learning and data mining. Existing approaches mainly learn representations for hypervertices by transforming a hypergraph to a standard graph, or learn representations for hypervertices and hyperedges in separate spaces. In this paper, we propose a hypergraph expansion method to transform a hypergraph to a standard graph while preserving high-order information. Different from previous hypergraph expansion approaches like clique expansion and star expansion, we transform both hypervertices and hyperedges in the hypergraph to vertices in the expanded graph, and construct connections between hypervertices or hyperedges, so that richer relationships can be used in graph learning. Based on the expanded graph, we propose a learning model to embed hypervertices and hyperedges in a joint representation space. Compared with the method of learning separate spaces for hypervertices and hyperedges, our method is able to capture common knowledge involved in hypervertices and hyperedges, and also improve the data efficiency and computational efficiency. To better leverage structure information, we minimize the graph reconstruction loss to preserve the structure information in the model. We perform experiments on both hypervertex classification and hyperedge classification tasks to demonstrate the effectiveness of our proposed method.

----

## [1027] FairSIN: Achieving Fairness in Graph Neural Networks through Sensitive Information Neutralization

**Authors**: *Cheng Yang, Jixi Liu, Yunhe Yan, Chuan Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28776](https://doi.org/10.1609/aaai.v38i8.28776)

**Abstract**:

Despite the remarkable success of graph neural networks (GNNs) in modeling graph-structured data, like other machine learning models, GNNs are also susceptible to making biased predictions based on sensitive attributes, such as race and gender. For fairness consideration, recent state-of-the-art (SOTA) methods propose to filter out sensitive information from inputs or representations, e.g., edge dropping or feature masking. However, we argue that such filtering-based strategies may also filter out some non-sensitive feature information, leading to a sub-optimal trade-off between predictive performance and fairness. To address this issue, we unveil an innovative neutralization-based paradigm, where additional Fairness-facilitating Features (F3) are incorporated into node features or representations before message passing. The F3 are expected to statistically neutralize the sensitive bias in node representations and provide additional nonsensitive information. We also provide theoretical explanations for our rationale, concluding that F3 can be realized by emphasizing the features of each node’s heterogeneous neighbors (neighbors with different sensitive attributes). We name our method as FairSIN, and present three implementation variants from both data-centric and model-centric perspectives. Experimental results on five benchmark datasets with three different GNN backbones show that FairSIN significantly improves fairness metrics while maintaining high prediction accuracies. Codes and appendix can be found at https://github.com/BUPT-GAMMA/FariSIN.

----

## [1028] Fine-Tuning Large Language Model Based Explainable Recommendation with Explainable Quality Reward

**Authors**: *Mengyuan Yang, Mengying Zhu, Yan Wang, Linxun Chen, Yilei Zhao, Xiuyuan Wang, Bing Han, Xiaolin Zheng, Jianwei Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28777](https://doi.org/10.1609/aaai.v38i8.28777)

**Abstract**:

Large language model-based explainable recommendation (LLM-based ER) systems can provide remarkable human-like explanations and have widely received attention from researchers. However, the original LLM-based ER systems face three low-quality problems in their generated explanations, i.e., lack of personalization, inconsistency, and questionable explanation data. To address these problems, we propose a novel LLM-based ER model denoted as LLM2ER to serve as a backbone and devise two innovative explainable quality reward models for fine-tuning such a backbone in a reinforcement learning paradigm, ultimately yielding a fine-tuned model denoted as LLM2ER-EQR, which can provide high-quality explanations. LLM2ER-EQR can generate personalized, informative, and consistent high-quality explanations learned from questionable-quality explanation datasets. Extensive experiments conducted on three real-world datasets demonstrate that our model can generate fluent, diverse, informative, and highly personalized explanations.

----

## [1029] Graph Neural Networks with Soft Association between Topology and Attribute

**Authors**: *Yachao Yang, Yanfeng Sun, Shaofan Wang, Jipeng Guo, Junbin Gao, Fujiao Ju, Baocai Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28778](https://doi.org/10.1609/aaai.v38i8.28778)

**Abstract**:

Graph Neural Networks (GNNs) have shown great performance in learning representations for graph-structured data. However, recent studies have found that the interference between topology and attribute can lead to distorted node representations. Most GNNs are designed based on homophily assumptions, thus they cannot be applied to graphs with heterophily. This research critically analyzes the propagation principles of various GNNs and the corresponding challenges from an optimization perspective. A novel GNN called Graph Neural Networks with Soft Association between Topology and Attribute (GNN-SATA) is proposed. Different embeddings are utilized to gain insights into attributes and structures while establishing their interconnections through soft association. Further as integral components of the soft association, a Graph Pruning Module (GPM) and Graph Augmentation Module (GAM) are developed. These modules dynamically remove or add edges to the adjacency relationships to make the model better fit with graphs with homophily or heterophily. Experimental results on homophilic and heterophilic graph datasets convincingly demonstrate that the proposed GNN-SATA effectively captures more accurate adjacency relationships and outperforms state-of-the-art approaches. Especially on the heterophilic graph dataset Squirrel, GNN-SATA achieves a 2.81% improvement in accuracy, utilizing merely 27.19% of the original number of adjacency relationships. Our code is released at https://github.com/wwwfadecom/GNN-SATA.

----

## [1030] TriSampler: A Better Negative Sampling Principle for Dense Retrieval

**Authors**: *Zhen Yang, Zhou Shao, Yuxiao Dong, Jie Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28779](https://doi.org/10.1609/aaai.v38i8.28779)

**Abstract**:

Negative sampling stands as a pivotal technique in dense retrieval, essential for training effective retrieval models and significantly impacting retrieval performance. While existing negative sampling methods have made commendable progress by leveraging hard negatives, a comprehensive guiding principle for constructing negative candidates and designing negative sampling distributions is still lacking. To bridge this gap, we embark on a theoretical analysis of negative sampling in dense retrieval. This exploration culminates in the unveiling of the quasi-triangular principle, a novel framework that elucidates the triangular-like interplay between query, positive document, and negative document. Fueled by this guiding principle, we introduce TriSampler, a straightforward yet highly effective negative sampling method. The keypoint of TriSampler lies in its ability to selectively sample more informative negatives within a prescribed constrained region. Experimental evaluation show that TriSampler consistently attains superior retrieval performance across a diverse of representative retrieval models.

----

## [1031] Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems

**Authors**: *Zhiguang Yang, Liufang Sang, Haoran Wang, Wenlong Chen, Lu Wang, Jie He, Changping Peng, Zhangang Lin, Chun Gan, Jingping Shao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28780](https://doi.org/10.1609/aaai.v38i8.28780)

**Abstract**:

Creativity is the heart and soul of advertising services. Effective creatives can create a win-win scenario: advertisers each target users and achieve marketing objectives more effectively, users more quickly find products of interest, and platforms generate more advertising revenue. With the advent of AI-Generated Content, advertisers now can produce vast amounts of creative content at a minimal cost. The current challenge lies in how advertising systems can select the most pertinent creative in real-time for each user personally. Existing methods typically perform serial ranking of ads or creatives, limiting the creative module in terms of both effectiveness and efficiency. In this paper, we propose for the first time a novel architecture for online parallel estimation of ads and creatives ranking, as well as the corresponding offline joint optimization model. The online architecture enables sophisticated personalized creative modeling while reducing overall latency. The offline joint model for CTR estimation allows mutual awareness and collaborative optimization between ads and creatives. Additionally, we optimize the offline evaluation metrics for the implicit feedback sorting task involved in ad creative ranking. We conduct extensive experiments to compare ours with two state-of-the-art approaches. The results demonstrate the effectiveness of our approach in both offline evaluations and real-world advertising platforms online in terms of response time, CTR, and CPM.

----

## [1032] WaveNet: Tackling Non-stationary Graph Signals via Graph Spectral Wavelets

**Authors**: *Zhirui Yang, Yulan Hu, Sheng Ouyang, Jingyu Liu, Shuqiang Wang, Xibo Ma, Wenhan Wang, Hanjing Su, Yong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28781](https://doi.org/10.1609/aaai.v38i8.28781)

**Abstract**:

In the existing spectral GNNs, polynomial-based methods occupy the mainstream in designing a filter through the Laplacian matrix. However, polynomial combinations factored by the Laplacian matrix naturally have limitations in message passing (e.g., over-smoothing). Furthermore, most existing spectral GNNs are based on polynomial bases, which struggle to capture the high-frequency parts of the graph spectral signal. Additionally, we also find that even increasing the polynomial order does not change this situation, which means polynomial-based models have a natural deficiency when facing high-frequency signals. To tackle these problems, we propose WaveNet, which aims to effectively capture the high-frequency part of the graph spectral signal from the perspective of wavelet bases through reconstructing the message propagation matrix. We utilize Multi-Resolution Analysis (MRA) to model this question, and our proposed method can reconstruct arbitrary filters theoretically. We also conduct node classification experiments on real-world graph benchmarks and achieve superior performance on most datasets. Our code is available at https://github.com/Bufordyang/WaveNet

----

## [1033] RRL: Recommendation Reverse Learning

**Authors**: *Xiaoyu You, Jianwei Xu, Mi Zhang, Zechen Gao, Min Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28782](https://doi.org/10.1609/aaai.v38i8.28782)

**Abstract**:

As societies become increasingly aware of data privacy, regulations require that private information about users must be removed from both database and ML models, which is more colloquially called `the right to be forgotten`. Such privacy problems of recommendation systems, which hold large amounts of private data, are drawing increasing attention. Recent research suggests dividing the preference data into multiple shards and training submodels with these shards and forgetting users' personal preference data by retraining the submodels of marked shards. Despite the computational efficiency development compared with retraining from scratch, the overall recommendation performance deteriorates after dividing the shards because the collaborative information contained in the training data is broken. In this paper, we aim to propose a forgetting framework for recommendation models that neither separate the training data nor jeopardizes the recommendation performance, named Recommendation Reverse Learning (RRL). Given the trained recommendation model and marked preference data, we devise Reverse BPR Objective (RBPR Objective) to fine-tune the recommendation model to force it to forget the marked data. Nevertheless, as the recommendation model encode the complex collaborative information among users, we propose to utilize Fisher Information Matrix (FIM) to estimate the influence of reverse learning on other users' collaborative information and guide the updates of representations. We conduct experiments on two representative recommendation models and three public benchmark datasets to verify the efficiency of RRL. To verify the forgetting completeness, we use RRL to make the recommendation model poisoned by shilling attacks forget malicious users.

----

## [1034] UNEX-RL: Reinforcing Long-Term Rewards in Multi-Stage Recommender Systems with UNidirectional EXecution

**Authors**: *Gengrui Zhang, Yao Wang, Xiaoshuang Chen, Hongyi Qian, Kaiqiao Zhan, Ben Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28783](https://doi.org/10.1609/aaai.v38i8.28783)

**Abstract**:

In recent years, there has been a growing interest in utilizing reinforcement learning (RL) to optimize long-term rewards in recommender systems. Since industrial recommender systems are typically designed as multi-stage systems, RL methods with a single agent face challenges when optimizing multiple stages simultaneously. The reason is that different stages have different observation spaces, and thus cannot be modeled by a single agent. To address this issue, we propose a novel UNidirectional-EXecution-based multi-agent Reinforcement Learning (UNEX-RL) framework to reinforce the long-term rewards in multi-stage recommender systems.
We show that the unidirectional execution is a key feature of multi-stage recommender systems, bringing new challenges to the applications of multi-agent reinforcement learning (MARL), namely the observation dependency and the cascading effect. To tackle these challenges, we provide a cascading information chain (CIC) method to separate the independent observations from action-dependent observations and use CIC to train UNEX-RL effectively. We also discuss practical variance reduction techniques for UNEX-RL. Finally, we show the effectiveness of UNEX-RL on both public datasets and an online recommender system with over 100 million users. Specifically, UNEX-RL reveals a 0.558% increase in users' usage time compared with single-agent RL algorithms in online A/B experiments, highlighting the effectiveness of UNEX-RL in industrial recommender systems.

----

## [1035] M3D: Dataset Condensation by Minimizing Maximum Mean Discrepancy

**Authors**: *Hansong Zhang, Shikun Li, Pengju Wang, Dan Zeng, Shiming Ge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28784](https://doi.org/10.1609/aaai.v38i8.28784)

**Abstract**:

Training state-of-the-art (SOTA) deep models often requires extensive data, resulting in substantial training and storage costs. To address these challenges, dataset condensation has been developed to learn a small synthetic set that preserves essential information from the original large-scale dataset. Nowadays, optimization-oriented methods have been the primary method in the field of dataset condensation for achieving SOTA results. However, the bi-level optimization process hinders the practical application of such methods to realistic and larger datasets. To enhance condensation efficiency, previous works proposed Distribution-Matching (DM) as an alternative, which significantly reduces the condensation cost. Nonetheless, current DM-based methods still yield less comparable results to SOTA optimization-oriented methods. In this paper, we argue that existing DM-based methods overlook the higher-order alignment of the distributions, which may lead to sub-optimal matching results. Inspired by this, we present a novel DM-based method named M3D for dataset condensation by Minimizing the Maximum Mean Discrepancy  between feature representations of the synthetic and real images. By embedding their distributions in a reproducing kernel Hilbert space, we align all orders of moments of the distributions of real and synthetic images, resulting in a more generalized condensed set. Notably, our method even surpasses the SOTA optimization-oriented method IDC on the high-resolution ImageNet dataset. Extensive analysis is conducted to verify the effectiveness of the proposed method. Source codes are available at https://github.com/Hansong-Zhang/M3D.

----

## [1036] DiG-In-GNN: Discriminative Feature Guided GNN-Based Fraud Detector against Inconsistencies in Multi-Relation Fraud Graph

**Authors**: *Jinghui Zhang, Zhengjia Xu, Dingyang Lv, Zhan Shi, Dian Shen, Jiahui Jin, Fang Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28785](https://doi.org/10.1609/aaai.v38i8.28785)

**Abstract**:

Fraud detection on multi-relation graphs aims to identify fraudsters in graphs. Graph Neural Network (GNN) models leverage graph structures to pass messages from neighbors to the target nodes, thereby enriching the representations of those target nodes. However, feature and structural inconsistency in the graph, owing to fraudsters' camouflage behaviors, diminish the suspiciousness of fraud nodes which hinders the effectiveness of GNN-based models. In this work, we propose DiG-In-GNN, Discriminative Feature Guided GNN against Inconsistency, to dig into graphs for fraudsters. Specifically, we use multi-scale contrastive learning from the perspective of the neighborhood subgraph where the target node is located to generate guidance nodes to cope with the feature inconsistency. Then, guided by the guidance nodes, we conduct fine-grained neighbor selection through reinforcement learning for each neighbor node to precisely filter nodes that can enhance the message passing and therefore alleviate structural inconsistency. Finally, the two modules are integrated together to obtain discriminable representations of the nodes. Experiments on three fraud detection datasets demonstrate the superiority of the proposed method DiG-In-GNN, which obtains up to 20.73% improvement over previous state-of-the-art methods. Our code can be found at https://github.com/GraphBerry/DiG-In-GNN.

----

## [1037] Dual-View Whitening on Pre-trained Text Embeddings for Sequential Recommendation

**Authors**: *Lingzi Zhang, Xin Zhou, Zhiwei Zeng, Zhiqi Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28786](https://doi.org/10.1609/aaai.v38i8.28786)

**Abstract**:

Recent advances in sequential recommendation models have demonstrated the efficacy of integrating pre-trained text embeddings with item ID embeddings to achieve superior performance. However, our study takes a unique perspective by exclusively focusing on the untapped potential of text embeddings, obviating the need for ID embeddings. We begin by implementing a pre-processing strategy known as whitening, which effectively transforms the anisotropic semantic space of pre-trained text embeddings into an isotropic Gaussian distribution. Comprehensive experiments reveal that applying whitening to pre-trained text embeddings in sequential recommendation models significantly enhances performance. Yet, a full whitening operation might break the potential manifold of items with similar text semantics. To retain the original semantics while benefiting from the isotropy of the whitened text features, we propose a Dual-view Whitening method for Sequential Recommendation (DWSRec), which leverages both fully whitened and relaxed whitened item representations as dual views for effective recommendations. We further examine the advantages of our approach through both empirical and theoretical analyses. Experiments on three public benchmark datasets show that DWSRec outperforms state-of-the-art methods for sequential recommendation.

----

## [1038] CAMEL: Capturing Metaphorical Alignment with Context Disentangling for Multimodal Emotion Recognition

**Authors**: *Linhao Zhang, Li Jin, Guangluan Xu, Xiaoyu Li, Cai Xu, Kaiwen Wei, Nayu Liu, Haonan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28787](https://doi.org/10.1609/aaai.v38i8.28787)

**Abstract**:

Understanding the emotional polarity of multimodal content with metaphorical characteristics, such as memes, poses a significant challenge in Multimodal Emotion Recognition (MER). Previous MER researches have overlooked the phenomenon of metaphorical alignment in multimedia content, which involves non-literal associations between concepts to convey implicit emotional tones.  Metaphor-agnostic MER methods may be misinformed by the isolated unimodal emotions, which are distinct from the real emotions blended in multimodal metaphors. Moreover, contextual semantics can further affect the emotions associated with similar metaphors, leading to the challenge of maintaining contextual compatibility. To address the issue of metaphorical alignment in MER, we propose to leverage a conditional generative approach for capturing metaphorical analogies. Our approach formulates schematic prompts and corresponding references based on theoretical foundations, which allows the model to better grasp metaphorical nuances. In order to maintain contextual sensitivity, we incorporate a disentangled contrastive matching mechanism, which undergoes curricular adjustment to regulate its intensity during the learning process. The automatic and human evaluation experiments on two benchmarks prove that, our model provides considerable and stable improvements in recognizing multimodal emotion with metaphor attributes.

----

## [1039] ROG_PL: Robust Open-Set Graph Learning via Region-Based Prototype Learning

**Authors**: *Qin Zhang, Xiaowei Li, Jiexin Lu, Liping Qiu, Shirui Pan, Xiaojun Chen, Junyang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28788](https://doi.org/10.1609/aaai.v38i8.28788)

**Abstract**:

Open-set graph learning is a practical task that aims to classify the known class nodes and to identify unknown class samples as unknowns. Conventional node classification methods usually perform unsatisfactorily in open-set scenarios due to the complex data they encounter, such as out-of-distribution (OOD) data and in-distribution (IND) noise. OOD data are samples that do not belong to any known classes. They are outliers if they occur in training (OOD noise), and open-set samples if they occur in testing. IND noise are  training samples which are assigned incorrect labels. The existence of IND noise and OOD noise is prevalent, which usually cause the ambiguity problem, including the intra-class variety problem and the inter-class confusion problem. Thus, to explore robust open-set learning methods is necessary and difficult, and it becomes even more difficult for non-IID graph data. To this end, we propose a unified framework named ROG_PL to achieve robust open-set learning on complex noisy graph data, by introducing prototype learning. In specific, ROG_PL consists of two modules, i.e.,  denoising via label propagation and open-set prototype learning via regions. The first module corrects noisy labels through similarity-based label propagation and removes low-confidence samples, to solve the intra-class variety problem caused by noise. The second module learns open-set prototypes for each known class via non-overlapped regions and remains both interior and border prototypes to remedy the inter-class confusion problem. The two modules are iteratively updated under the constraints of  classification loss and  prototype diversity loss. To the best of our knowledge, the proposed ROG_PL is the first robust open-set node classification method for graph data with complex noise. Experimental evaluations of ROG_PL on several benchmark graph datasets demonstrate that it has good performance.

----

## [1040] Temporal Graph Contrastive Learning for Sequential Recommendation

**Authors**: *Shengzhe Zhang, Liyi Chen, Chao Wang, Shuangli Li, Hui Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28789](https://doi.org/10.1609/aaai.v38i8.28789)

**Abstract**:

Sequential recommendation is a crucial task in understanding users' evolving interests and predicting their future behaviors.  While existing approaches on sequence or graph modeling to learn interaction sequences of users have shown promising performance, how to effectively exploit temporal information and deal with the uncertainty noise in evolving user behaviors is still quite challenging. To this end, in this paper, we propose a Temporal Graph Contrastive Learning method for Sequential Recommendation (TGCL4SR) which leverages not only local interaction sequences but also global temporal graphs to comprehend item correlations and analyze user behaviors from a temporal perspective. Specifically, we first devise a Temporal Item Transition Graph (TITG) to fully leverage global interactions to understand item correlations, and augment this graph by dual transformations based on neighbor sampling and time disturbance. Accordingly, we design a Temporal item Transition graph Convolutional network (TiTConv) to capture temporal item transition patterns in TITG. Then, a novel Temporal Graph Contrastive Learning (TGCL) mechanism is designed to enhance the uniformity of representations between augmented graphs from identical sequences. For local interaction sequences, we design a temporal sequence encoder to incorporate time interval embeddings into the architecture of Transformer. At the training stage, we take maximum mean discrepancy and TGCL losses as auxiliary objectives. Extensive experiments on several real-world datasets show the effectiveness of TGCL4SR against state-of-the-art baselines of sequential recommendation.

----

## [1041] Influential Exemplar Replay for Incremental Learning in Recommender Systems

**Authors**: *Xinni Zhang, Yankai Chen, Chenhao Ma, Yixiang Fang, Irwin King*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28790](https://doi.org/10.1609/aaai.v38i8.28790)

**Abstract**:

Personalized recommender systems have found widespread applications for effective information filtering. Conventional models engage in knowledge mining within the static setting to reconstruct singular historical data. Nonetheless, the dynamics of real-world environments are in a constant state of flux, rendering acquired model knowledge inadequate for accommodating emergent trends and thus leading to notable recommendation performance decline. Given the typically prohibitive cost of exhaustive model retraining, it has emerged to study incremental learning for recommender systems with ever-growing data. In this paper, we propose an effective model-agnostic framework, namely INFluential Exemplar Replay (INFER). INFER facilitates recommender models in retaining the earlier assimilated knowledge, e.g., users' enduring preferences, while concurrently accommodating evolving trends manifested in users' new interaction behaviors. We commence with a vanilla implementation that centers on identifying the most representative data samples for effective consolidation of early knowledge. Subsequently, we propose an advanced solution, namely INFERONCE, to optimize the computational overhead associated with the vanilla implementation. Extensive experiments on four prototypical backbone models, two classic recommendation tasks, and four widely used benchmarks consistently demonstrate the effectiveness of our method as well as its compatibility for extending to several incremental recommender models.

----

## [1042] Another Way to the Top: Exploit Contextual Clustering in Learned Image Coding

**Authors**: *Yichi Zhang, Zhihao Duan, Ming Lu, Dandan Ding, Fengqing Zhu, Zhan Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28791](https://doi.org/10.1609/aaai.v38i8.28791)

**Abstract**:

While convolution and self-attention are extensively used in learned image compression (LIC) for transform coding, this paper proposes an alternative called Contextual Clustering based LIC (CLIC) which primarily relies on clustering operations and local attention for correlation characterization and compact representation of an image. As seen, CLIC expands the receptive field into the entire image for intra-cluster feature aggregation. Afterward, features are reordered to their original spatial positions to pass through the local attention units for inter-cluster embedding. Additionally, we introduce the Guided Post-Quantization Filtering (GuidedPQF)  into CLIC, effectively mitigating the propagation and accumulation of quantization errors at the initial decoding stage. Extensive experiments demonstrate the superior performance of CLIC over state-of-the-art works: when optimized using MSE, it outperforms VVC by about 10% BD-Rate in three widely-used benchmark datasets; when optimized using MS-SSIM, it saves more than 50% BD-Rate over VVC. Our CLIC offers a new way to generate compact representations for image compression, which also provides a novel direction along the line of LIC development.

----

## [1043] Multi-Domain Deep Learning from a Multi-View Perspective for Cross-Border E-commerce Search

**Authors**: *Yiqian Zhang, Yinfu Feng, Wen-Ji Zhou, Yunan Ye, Min Tan, Rong Xiao, Haihong Tang, Jiajun Ding, Jun Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28792](https://doi.org/10.1609/aaai.v38i8.28792)

**Abstract**:

Building click-through rate (CTR) and conversion rate (CVR) prediction models for cross-border e-commerce search requires modeling the correlations among multi-domains. Existing multi-domain methods would suffer severely from poor scalability and low efficiency when number of domains increases. To this end, we propose a Domain-Aware Multi-view mOdel (DAMO), which is domain-number-invariant, to effectively leverage cross-domain relations from a multi-view perspective. Specifically, instead of working in the original feature space defined by different domains, DAMO maps everything to a new low-rank multi-view space. To achieve this, DAMO firstly extracts multi-domain features in an explicit feature-interactive manner. These features are parsed to a multi-view extractor to obtain view-invariant and view-specific features. Then a multi-view predictor inputs these two sets of features and outputs view-based predictions. To enforce view-awareness in the predictor, we further propose a lightweight view-attention estimator to dynamically learn the optimal view-specific weights w.r.t. a view-guided loss. Extensive experiments on public and industrial datasets show that compared with state-of-the-art models, our DAMO achieves better performance with lower storage and computational costs. In addition, deploying DAMO to a large-scale cross-border e-commence platform leads to 1.21%, 1.76%, and 1.66% improvements over the existing CGC-based model in the online AB-testing experiment in terms of CTR, CVR, and Gross Merchandises Value, respectively.

----

## [1044] Spatial-Temporal Interplay in Human Mobility: A Hierarchical Reinforcement Learning Approach with Hypergraph Representation

**Authors**: *Zhaofan Zhang, Yanan Xiao, Lu Jiang, Dingqi Yang, Minghao Yin, Pengyang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28793](https://doi.org/10.1609/aaai.v38i8.28793)

**Abstract**:

In the realm of human mobility, the decision-making process for selecting the next-visit location is intricately influenced by a trade-off between spatial and temporal constraints, which are reflective of individual needs and preferences. This trade-off, however, varies across individuals, making the modeling of these spatial-temporal dynamics a formidable challenge. To address the problem, in this work, we introduce the "Spatial-temporal Induced Hierarchical Reinforcement Learning" (STI-HRL) framework, for capturing the interplay between spatial and temporal factors in human mobility decision-making. Specifically, STI-HRL employs a two-tiered decision-making process: the low-level focuses on disentangling spatial and temporal preferences using dedicated agents, while the high-level integrates these considerations to finalize the decision. To complement the hierarchical decision setting, we construct a hypergraph to organize historical data, encapsulating the multi-aspect semantics of human mobility. We propose a cross-channel hypergraph embedding module to learn the representations as the states to facilitate the decision-making cycle. Our extensive experiments on two real-world datasets validate the superiority of STI-HRL over state-of-the-art methods in predicting users' next visits across various performance metrics.

----

## [1045] FacetCRS: Multi-Faceted Preference Learning for Pricking Filter Bubbles in Conversational Recommender System

**Authors**: *Yongsen Zheng, Ziliang Chen, Jinghui Qin, Liang Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28794](https://doi.org/10.1609/aaai.v38i8.28794)

**Abstract**:

The filter bubble is a notorious issue in Recommender Systems (RSs), which describes the phenomenon whereby users are exposed to a limited and narrow range of information or content that reinforces their existing dominant preferences and beliefs. This results in a lack of exposure to diverse and varied content. Many existing works have predominantly examined filter bubbles in static or relatively-static recommendation settings. However, filter bubbles will be continuously intensified over time due to the feedback loop between the user and the system in the real-world online recommendation. To address these issues, we propose a novel paradigm, Multi-Facet Preference Learning for Pricking Filter Bubbles in Conversational Recommender System (FacetCRS), which aims to burst filter bubbles in the conversational recommender system (CRS) through timely user-item interactions via natural language conversations. By considering diverse user preferences and intentions, FacetCRS automatically model user preference into multi-facets, including entity-, word-, context-, and review-facet, to capture diverse and dynamic user preferences to prick filter bubbles in the CRS. It is an end-to-end CRS framework to adaptively learn representations of various levels of preference facet and diverse types of external knowledge. Extensive experiments on two publicly available benchmark datasets demonstrate that our proposed method achieves state-of-the-art performance in mitigating filter bubbles and enhancing recommendation quality in CRS.

----

## [1046] GMP-AR: Granularity Message Passing and Adaptive Reconciliation for Temporal Hierarchy Forecasting

**Authors**: *Fan Zhou, Chen Pan, Lintao Ma, Yu Liu, Siqiao Xue, James Zhang, Jun Zhou, Hongyuan Mei, Weitao Lin, Zi Zhuang, Wenxin Ning, Yunhua Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28795](https://doi.org/10.1609/aaai.v38i8.28795)

**Abstract**:

Time series forecasts of different temporal granularity are widely used in real-world applications, e.g., sales prediction in days and weeks for making different inventory plans. However, these tasks are usually solved separately without ensuring coherence, which is crucial for aligning downstream decisions. Previous works mainly focus on ensuring coherence with some straightforward methods, e.g., aggregation from the forecasts of fine granularity to the coarse ones, and allocation from the coarse granularity to the fine ones. These methods merely take the temporal hierarchical structure to maintain coherence without improving the forecasting accuracy. In this paper, we propose a novel granularity message-passing mechanism (GMP) that leverages temporal hierarchy information to improve forecasting performance and also utilizes an adaptive reconciliation (AR) strategy to maintain coherence without performance loss. Furthermore, we introduce an optimization module to achieve task-based targets while adhering to more real-world constraints. Experiments on real-world datasets demonstrate that our framework (GMP-AR) achieves superior performances on temporal hierarchical forecasting tasks compared to state-of-the-art methods. In addition, our framework has been successfully applied to a real-world task of payment traffic management in Alipay by integrating with the task-based optimization module.

----

## [1047] Explainable Origin-Destination Crowd Flow Interpolation via Variational Multi-Modal Recurrent Graph Auto-Encoder

**Authors**: *Qiang Zhou, Xinjiang Lu, Jingjing Gu, Zhe Zheng, Bo Jin, Jingbo Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28796](https://doi.org/10.1609/aaai.v38i8.28796)

**Abstract**:

Origin-destination (OD) crowd flow, if more accurately inferred at a fine-grained level, has the potential to enhance the efficacy of various urban applications. While in practice for mining OD crowd flow with effect, the problem of spatially interpolating OD crowd flow occurs since the ineluctable missing values. This problem is further complicated by the inherent scarcity and noise nature of OD crowd flow data. In this paper, we propose an uncertainty-aware interpolative and explainable framework, namely UApex, for realizing reliable and trustworthy OD crowd flow interpolation. Specifically, we first design a Variational Multi-modal Recurrent Graph Auto-Encoder (VMR-GAE) for uncertainty-aware OD crowd flow interpolation. A key idea here is to formulate the problem as semi-supervised learning on directed graphs. Next, to mitigate the data scarcity, we incorporate a distribution alignment mechanism that can introduce supplementary modals into variational inference. Then, a dedicated decoder with a Poisson prior is proposed for OD crowd flow interpolation. Moreover, to make VMR-GAE more trustworthy, we develop an efficient and uncertainty-aware explainer that can provide explanations from the spatiotemporal topology perspective via the Shapley value. Extensive experiments on two real-world datasets validate that VMR-GAE outperforms the state-of-the-art baselines. Also, an exploratory empirical study shows that the proposed explainer can generate meaningful spatiotemporal explanations.

----

## [1048] An Efficient Subgraph-Inferring Framework for Large-Scale Heterogeneous Graphs

**Authors**: *Wei Zhou, Hong Huang, Ruize Shi, Kehan Yin, Hai Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28797](https://doi.org/10.1609/aaai.v38i8.28797)

**Abstract**:

Heterogeneous Graph Neural Networks (HGNNs) play a vital role in advancing the field of graph representation learning by addressing the complexities arising from diverse data types and interconnected relationships in real-world scenarios. However, traditional HGNNs face challenges when applied to large-scale graphs due to the necessity of training or inferring on the entire graph. As the size of the heterogeneous graphs increases, the time and memory overhead required by these models escalates rapidly, even reaching unacceptable levels. To address this issue, in this paper, we present a novel framework named (SubInfer), which conducts training and inferring on subgraphs instead of the entire graphs, hence efficiently handling large-scale heterogeneous graphs. The proposed framework comprises three main steps: 1) partitioning the heterogeneous graph from multiple perspectives to preserve various semantic information, 2) completing the subgraphs to improve the convergence speed of subgraph training and the performance of subgraph inference, and 3) training and inferring the HGNN model on distributed clusters to further reduce the time overhead. The framework is applicable to the vast majority of HGNN models. Experiments on five benchmark datasets demonstrate that SubInfer effectively optimizes the training and inference phase, delivering comparable performance to traditional HGNN models while significantly reducing time and memory overhead.

----

## [1049] Analytically Tractable Models for Decision Making under Present Bias

**Authors**: *Yasunori Akagi, Naoki Marumo, Takeshi Kurashima*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28798](https://doi.org/10.1609/aaai.v38i9.28798)

**Abstract**:

Time-inconsistency is a characteristic of human behavior in which people plan for long-term benefits but take actions that differ from the plan due to conflicts with short-term benefits. Such time-inconsistent behavior is believed to be caused by present bias, a tendency to overestimate immediate rewards and underestimate future rewards. It is essential in behavioral economics to investigate the relationship between present bias and time-inconsistency. In this paper, we propose a model for analyzing agent behavior with present bias in tasks to make progress toward a goal over a specific period. Unlike previous models, the state sequence of the agent can be described analytically in our model. Based on this property, we analyze three crucial problems related to agents under present bias: task abandonment, optimal goal setting, and optimal reward scheduling. Extensive analysis reveals how present bias affects the condition under which task abandonment occurs and optimal intervention strategies. Our findings are meaningful for preventing task abandonment and intervening through incentives in the real world.

----

## [1050] Optimistic Policy Gradient in Multi-Player Markov Games with a Single Controller: Convergence beyond the Minty Property

**Authors**: *Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28799](https://doi.org/10.1609/aaai.v38i9.28799)

**Abstract**:

Policy gradient methods enjoy strong practical performance in numerous tasks in reinforcement learning. Their theoretical understanding in multiagent settings, however, remains limited, especially beyond two-player competitive and potential Markov games. In this paper, we develop a new framework to characterize optimistic policy gradient methods in multi-player Markov games with a single controller. Specifically, under the further assumption that the game exhibits an equilibrium collapse, in that the marginals of coarse correlated equilibria (CCE) induce Nash equilibria (NE), we show convergence to stationary epsilon-NE in O(1/epsilon^2) iterations, where O suppresses polynomial factors in the natural parameters of the game. Such an equilibrium collapse is well-known to manifest itself in two-player zero-sum Markov games, but also occurs even in a class of multi-player Markov games with separable interactions, as established by recent work. As a result, we bypass known complexity barriers for computing stationary NE when either of our assumptions fails. Our approach relies on a natural generalization of the classical Minty property that we introduce, which we anticipate to have further applications beyond Markov games.

----

## [1051] Improved Metric Distortion via Threshold Approvals

**Authors**: *Elliot Anshelevich, Aris Filos-Ratsikas, Christopher Jerrett, Alexandros A. Voudouris*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28800](https://doi.org/10.1609/aaai.v38i9.28800)

**Abstract**:

We consider a social choice setting in which agents and alternatives are represented by points in a metric space, and the cost of an agent for an alternative is the distance between the corresponding points in the space. The goal is to choose a single alternative to (approximately) minimize the social cost (cost of all agents) or the maximum cost of any agent, when only limited information about the preferences of the agents is given. Previous work has shown that the best possible distortion one can hope to achieve is 3 when access to the ordinal preferences of the agents is given, even when the distances between alternatives in the metric space are known. We improve upon this bound of 3 by designing deterministic mechanisms that exploit a bit of cardinal information. We show that it is possible to achieve distortion 1+sqrt(2) by using the ordinal preferences of the agents, the distances between alternatives, and a threshold approval set per agent that contains all alternatives for whom her cost is within an appropriately chosen factor of her cost for her most-preferred alternative. We show that this bound is the best possible for any deterministic mechanism in general metric spaces, and also provide improved bounds for the fundamental case of a line metric.

----

## [1052] Fair Lotteries for Participatory Budgeting

**Authors**: *Haris Aziz, Xinhang Lu, Mashbat Suzuki, Jeremy Vollen, Toby Walsh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28801](https://doi.org/10.1609/aaai.v38i9.28801)

**Abstract**:

In pursuit of participatory budgeting (PB) outcomes with broader fairness guarantees, we initiate the study of lotteries over discrete PB outcomes. As the projects have heterogeneous costs, the amount spent may not be equal ex ante and ex post. To address this, we develop a technique to bound the amount by which the ex-post spend differs from the ex-ante spend---the property is termed budget balanced up to one project (BB1). With respect to fairness, we take a best-of-both-worlds perspective, seeking outcomes that are both ex-ante and ex-post fair. Towards this goal, we initiate a study of ex-ante fairness properties in PB, including Individual Fair Share (IFS), Unanimous Fair Share (UFS) and their stronger variants, as well as Group Fair Share (GFS). We show several incompatibility results between these ex-ante fairness notions and existing ex-post concepts based on justified representation. One of our main contributions is a randomized algorithm which simultaneously satisfies ex-ante Strong UFS, ex-post full justified representation (FJR) and ex-post BB1 for PB with binary utilities.

----

## [1053] Envy-Free House Allocation under Uncertain Preferences

**Authors**: *Haris Aziz, Isaiah Iliffe, Bo Li, Angus Ritossa, Ankang Sun, Mashbat Suzuki*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28802](https://doi.org/10.1609/aaai.v38i9.28802)

**Abstract**:

Envy-freeness is one of the most important fairness concerns when allocating items. We study envy-free house allocation when agents have uncertain preferences over items and consider several well-studied preference uncertainty models. The central problem that we focus on is computing an allocation that has the highest probability of being envy-free. We show that each model leads to a distinct set of algorithmic and complexity results, including detailed results on (in-)approximability. En route, we consider two related problems of checking whether there exists an allocation that is possibly or necessarily envy-free. We give a complete picture of the computational complexity of these two problems for all the uncertainty models we consider.

----

## [1054] Content Filtering with Inattentive Information Consumers

**Authors**: *Ian Ball, James W. Bono, Justin Grana, Nicole Immorlica, Brendan Lucier, Aleksandrs Slivkins*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28803](https://doi.org/10.1609/aaai.v38i9.28803)

**Abstract**:

We develop a model of content filtering as a game between the filter and the content consumer, where the latter incurs information costs for examining the content. Motivating examples include censoring misinformation, spam/phish filtering, and recommender systems acting on a stream of content. When the attacker is exogenous, we show that improving the filter’s quality is weakly Pareto improving, but has no impact on equilibrium payoffs until the filter becomes sufficiently accurate. Further, if the filter does not internalize the consumer’s information costs, its lack of commitment power may render it useless and lead to inefficient outcomes. When the attacker is also strategic, improvements in filter quality may decrease equilibrium payoffs.

----

## [1055] Nearly Equitable Allocations beyond Additivity and Monotonicity

**Authors**: *Siddharth Barman, Umang Bhaskar, Yeshwant Pandit, Soumyajit Pyne*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28804](https://doi.org/10.1609/aaai.v38i9.28804)

**Abstract**:

Equitability (EQ) in fair division requires that items be allocated such that all agents value the bundle they receive equally. With indivisible items, an equitable allocation may not exist, and hence we instead consider a meaningful analog, EQx, that requires equitability up to any item. EQx allocations exist for monotone, additive valuations. However, if (1) the agents' valuations are not additive or (2) the set of indivisible items includes both goods and chores (positively and negatively valued items), then prior to the current work it was not known whether EQx allocations exist or not. 

We study both the existence and efficient computation of EQx allocations. (1) For monotone valuations (not necessarily additive), we show that EQx allocations always exist. Also, for the large class of weakly well-layered valuations, EQx allocations can be found in polynomial time. Further, we prove that approximately EQx allocations can be computed efficiently under general monotone valuations.  (2) For non-monotone valuations, we show that an EQx allocation may not exist, even for two agents with additive valuations. Under some special cases, however, we show existence and efficient computability of EQx allocations. This includes the case of two agents with additive valuations where each item is either a good or a chore, and there are no mixed items.

----

## [1056] Principal-Agent Reward Shaping in MDPs

**Authors**: *Omer Ben-Porat, Yishay Mansour, Michal Moshkovitz, Boaz Taitler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28805](https://doi.org/10.1609/aaai.v38i9.28805)

**Abstract**:

Principal-agent problems arise when one party acts on behalf of another, leading to conflicts of interest. The economic literature has extensively studied principal-agent problems, and recent work has extended this to more complex scenarios such as Markov Decision Processes (MDPs). In this paper, we further explore this line of research by investigating how reward shaping under budget constraints can improve the principal's utility. We study a two-player Stackelberg game where the principal and the agent have different reward functions, and the agent chooses an MDP policy for both players. The principal offers an additional reward to the agent, and the agent picks their policy selfishly to maximize their reward, which is the sum of the original and the offered reward. Our results establish the NP-hardness of the problem and offer polynomial approximation algorithms for two classes of instances: Stochastic trees and deterministic decision processes with a finite horizon.

----

## [1057] Enhancing the Efficiency of Altruism and Taxes in Affine Congestion Games through Signalling

**Authors**: *Vittorio Bilò, Cosimo Vinci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28806](https://doi.org/10.1609/aaai.v38i9.28806)

**Abstract**:

We address the problem of improving the worst-case efficiency of pure Nash equilibria (aka, the price of anarchy) in affine congestion games, through a novel use of signalling. We assume that, for each player in the game, a most preferred strategy is publicly signalled. This can be done either distributedly by the players themselves, or be the outcome of some centralized algorithm. We apply this signalling scheme to two well-studied scenarios: games with partially altruistic players and games with resource taxation. We show a significant improvement in the price of anarchy of these games, whenever the aggregate signalled strategy profile is a good approximation of the game social optimum.

----

## [1058] Approval-Based Committee Voting in Practice: A Case Study of (over-)Representation in the Polkadot Blockchain

**Authors**: *Niclas Boehmer, Markus Brill, Alfonso Cevallos, Jonas Gehrlein, Luis Sánchez Fernández, Ulrike Schmidt-Kraepelin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28807](https://doi.org/10.1609/aaai.v38i9.28807)

**Abstract**:

We provide the first large-scale data collection of real-world approval-based committee elections. These elections have been conducted on the Polkadot blockchain as part of their Nominated Proof-of-Stake mechanism and contain around one thousand candidates and tens of thousands of (weighted) voters each. We conduct an in-depth study of application-relevant questions, including a quantitative and qualitative analysis of the outcomes returned by different voting rules. Besides considering proportionality measures that are standard in the multiwinner voting literature, we pay particular attention to less-studied measures of overrepresentation, as these are closely related to the security of the Polkadot network. We also analyze how different design decisions such as the committee size affect the examined measures.

----

## [1059] Completing Priceable Committees: Utilitarian and Representation Guarantees for Proportional Multiwinner Voting

**Authors**: *Markus Brill, Jannik Peters*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28808](https://doi.org/10.1609/aaai.v38i9.28808)

**Abstract**:

When selecting committees based on preferences of voters, a variety of different criteria can be considered. Two natural objectives are maximizing the utilitarian welfare (the sum of voters' utilities) and coverage (the number of represented voters) of the selected committee. Previous work has studied the impact on utilitarian welfare and coverage when requiring the committee to satisfy minimal requirements such as justified representation or weak proportionality. In this paper, we consider the impact of imposing much more demanding proportionality axioms. We identify a class of voting rules that achieve strong guarantees on utilitarian welfare and coverage when combined with appropriate completions. This class is defined via a weakening of priceability and contains prominent rules such as the Method of Equal Shares. We show that committees selected by these rules (i) can be completed to achieve optimal coverage and (ii) can be completed to achieve an asymptotically optimal approximation to the utilitarian welfare if they additionally satisfy EJR+. Answering an open question of Elkind et al. (2022), we use the Greedy Justified Candidate Rule to obtain the best possible utilitarian guarantee subject to proportionality. We also consider completion methods suggested in the participatory budgeting literature and other objectives besides welfare and coverage.

----

## [1060] Stability in Online Coalition Formation

**Authors**: *Martin Bullinger, René Romen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28809](https://doi.org/10.1609/aaai.v38i9.28809)

**Abstract**:

Coalition formation is concerned with the question of how to partition a set of agents into disjoint coalitions according to their preferences. Deviating from most of the previous work, we consider an online variant of the problem, where agents arrive in sequence and whenever an agent arrives, they have to be assigned to a coalition immediately and irrevocably. The scarce existing literature on online coalition formation has focused on the objective of maximizing social welfare, a demanding requirement, even in the offline setting. Instead, we seek to achieve stable coalition structures in an online setting, and focus on stability concepts based on deviations by single agents. We present a comprehensive picture in additively separable hedonic games, leading to dichotomies, where positive results are obtained by deterministic algorithms and negative results even hold for randomized algorithms.

----

## [1061] Participation Incentives in Approval-Based Committee Elections

**Authors**: *Martin Bullinger, Chris Dong, Patrick Lederer, Clara Mehler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28810](https://doi.org/10.1609/aaai.v38i9.28810)

**Abstract**:

In approval-based committee (ABC) voting, the goal is to
choose a subset of predefined size of the candidates based on
the voters’ approval preferences over the candidates. While
this problem has attracted significant attention in recent years,
the incentives for voters to participate in an election for a
given ABC voting rule have been neglected so far. This paper
is thus the first to explicitly study this property, typically called
participation, for ABC voting rules. In particular, we show
that all ABC scoring rules even satisfy group participation,
whereas most sequential rules severely fail participation. We
furthermore explore several escape routes to the impossibility
for sequential ABC voting rules: we prove for many sequential
rules that (i) they satisfy participation on laminar profiles, (ii)
voters who approve none of the elected candidates cannot
benefit by abstaining, and (iii) it is NP-hard for a voter to
decide whether she benefits from abstaining

----

## [1062] Low-Distortion Clustering with Ordinal and Limited Cardinal Information

**Authors**: *Jakob Burkhardt, Ioannis Caragiannis, Karl Fehrs, Matteo Russo, Chris Schwiegelshohn, Sudarshan Shyam*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28811](https://doi.org/10.1609/aaai.v38i9.28811)

**Abstract**:

Motivated by recent work in computational social choice, we extend the metric distortion framework to clustering problems. Given a set of n agents located in an underlying metric space, our goal is to partition them into k clusters, optimizing some social cost objective. The metric space is defined by a distance function d between the agent locations. Information about d is available only implicitly via n rankings, through which each agent ranks all other agents in terms of their distance from her. Still, even though no cardinal information (i.e., the exact distance values) is available, we would like to evaluate clustering algorithms in terms of social cost objectives that are defined using d. This is done using the notion of distortion, which measures how far from optimality a clustering can be, taking into account all underlying metrics that are consistent with the ordinal information available.

Unfortunately, the most important clustering objectives (e.g., those used in the well-known k-median and k-center problems) do not admit algorithms with finite distortion. To sidestep this disappointing fact, we follow two alternative approaches: We first explore whether resource augmentation can be beneficial. We consider algorithms that use more than k clusters but compare their social cost to that of the optimal k-clusterings. We show that using exponentially (in terms of k) many clusters, we can get low (constant or logarithmic) distortion for the k-center and k-median objectives. Interestingly, such an exponential blowup is shown to be necessary. More importantly, we explore whether limited cardinal information can be used to obtain better results. Somewhat surprisingly, for k-median and k-center, we show that a number of queries that is polynomial in k and only logarithmic in n (i.e., only sublinear in the number of agents for the most relevant scenarios in practice) is enough to get constant distortion.

----

## [1063] Efficient Learning in Polyhedral Games via Best-Response Oracles

**Authors**: *Darshan Chakrabarti, Gabriele Farina, Christian Kroer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28812](https://doi.org/10.1609/aaai.v38i9.28812)

**Abstract**:

We study online learning and equilibrium computation in games with polyhedral decision sets, a property shared by normal-form games (NFGs) and extensive-form games (EFGs), when the learning agent is restricted to utilizing a best-response oracle. We show how to achieve constant regret in zero-sum games and O(T^0.25) regret in general-sum games while using only O(log t) best-response queries at a given iteration t, thus improving over the best prior result, which required O(T) queries per iteration. Moreover, our framework yields the first last-iterate convergence guarantees for self-play with best-response oracles in zero-sum games. This convergence occurs at a linear rate, though with a condition-number dependence. We go on to show a O(T^(-0.5)) best-iterate convergence rate without such a dependence. Our results build on linear-rate convergence results for variants of the Frank-Wolfe (FW) algorithm for strongly convex and smooth minimization problems over polyhedral domains. These FW results depend on a condition number of the polytope, known as facial distance. In order to enable application to settings such as EFGs, we show two broad new results: 1) the facial distance for polytopes in standard form is at least γ/k where γ is the minimum value of a nonzero coordinate of a vertex of the polytope and k≤n is the number of tight inequality constraints in the optimal face, and 2) the facial distance for polytopes of the form Ax=b, Cx≤d, x≥0 where x∈R^n, C≥0 is a nonzero integral matrix, and d≥0, is at least 1/(c√n), where c is the infinity norm of C. This yields the first such results for several problems such as sequence-form polytopes, flow polytopes, and matching polytopes.

----

## [1064] Proportional Aggregation of Preferences for Sequential Decision Making

**Authors**: *Nikhil Chandak, Shashwat Goel, Dominik Peters*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28813](https://doi.org/10.1609/aaai.v38i9.28813)

**Abstract**:

We study the problem of fair sequential decision making given voter preferences. In each round, a decision rule must choose a decision from a set of alternatives where each voter reports which of these alternatives they approve. Instead of going with the most popular choice in each round, we aim for proportional representation, using axioms inspired by the multi-winner voting literature. The axioms require that every group of α% of the voters, if it agrees in every round (i.e., approves a common alternative), then those voters must approve at least α% of the decisions. A stronger version of the axioms requires that every group of α% of the voters that agrees in a β fraction of rounds must approve β⋅α% of the decisions. We show that three attractive voting rules satisfy axioms of this style. One of them (Sequential Phragmén) makes its decisions online, and the other two satisfy strengthened versions of the axioms but make decisions semi-online (Method of Equal Shares) or fully offline (Proportional Approval Voting). We present empirical results for these rules based on synthetic data and U.S. political elections. We also run experiments using the moral machine dataset about ethical dilemmas. We train preference models on user responses from different countries and let the models cast votes. We find that aggregating these votes using our rules leads to a more equal utility distribution across demographics than making decisions using a single global preference model.

----

## [1065] How to Make Knockout Tournaments More Popular?

**Authors**: *Juhi Chaudhary, Hendrik Molter, Meirav Zehavi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28814](https://doi.org/10.1609/aaai.v38i9.28814)

**Abstract**:

Given a mapping from a set of players to the leaves of a complete binary tree (called a seeding), a knockout tournament is conducted as follows: every round, every two players with a common parent compete against each other, and the winner is promoted to the common parent; then, the leaves are deleted. When only one player remains, it is declared the winner.  This is a popular competition format in sports, elections, and decision-making. Over the past decade, it has been studied intensively from both theoretical and practical points of view. Most frequently, the objective is to seed the tournament in a way that ``assists'' (or even guarantees) some particular player to win the competition. We introduce a new objective, which is very sensible from the perspective of the directors of the competition:  maximize the profit or popularity of the tournament. Specifically, we associate a ``score'' with every possible match, and aim to seed the tournament to maximize the sum of the scores of the matches that take place. We focus on the case where we assume a total order on the players' strengths, and provide a wide spectrum of results on the computational complexity of the problem.

----

## [1066] 1/2-Approximate MMS Allocation for Separable Piecewise Linear Concave Valuations

**Authors**: *Chandra Chekuri, Pooja Kulkarni, Rucha Kulkarni, Ruta Mehta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28815](https://doi.org/10.1609/aaai.v38i9.28815)

**Abstract**:

We study fair distribution of a collection of m indivisible goods among a group of n agents, using the widely recognized fairness principles of Maximin Share (MMS) and Any Price Share (APS). These principles have undergone thorough investigation within the context of additive valuations. We explore these notions for valuations that extend beyond additivity.

First, we study approximate MMS under the separable (piecewise-linear) concave (SPLC) valuations, an important class generalizing additive, where the best known factor was 1/3-MMS. We show that 1/2-MMS allocation exists and can be computed in polynomial time, significantly improving the state-of-the-art.
We note that SPLC valuations introduce an elevated level of intricacy in contrast to additive. For instance, the MMS value of an agent can be as high as her value for the entire set of items. We use a relax-and-round paradigm that goes through competitive equilibrium and LP relaxation. Our result extends to give (symmetric) 1/2-APS, a stronger guarantee than MMS.

APS is a stronger notion that generalizes MMS by allowing agents with arbitrary entitlements. We study the approximation of APS under submodular valuation functions. We design and analyze a simple greedy algorithm using concave extensions of submodular functions. We prove that the algorithm gives a 1/3-APS allocation which matches the best-known factor. Concave extensions are hard to compute in polynomial time and are, therefore, generally not used in approximation algorithms. Our approach shows a way to utilize it within analysis (while bypassing its computation), and hence might be of independent interest.

----

## [1067] Dynamic Budget Throttling in Repeated Second-Price Auctions

**Authors**: *Zhaohua Chen, Chang Wang, Qian Wang, Yuqi Pan, Zhuming Shi, Zheng Cai, Yukun Ren, Zhihua Zhu, Xiaotie Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28816](https://doi.org/10.1609/aaai.v38i9.28816)

**Abstract**:

In today's online advertising markets, a crucial requirement for an advertiser is to control her total expenditure within a time horizon under some budget. 
Among various budget control methods, throttling has emerged as a popular choice, managing an advertiser's total expenditure by selecting only a subset of auctions to participate in.
This paper provides a theoretical panorama of a single advertiser's dynamic budget throttling process in repeated second-price auctions.
We first establish a lower bound on the regret and an upper bound on the asymptotic competitive ratio for any throttling algorithm, respectively, when the advertiser's values are stochastic and adversarial. 
Regarding the algorithmic side, we propose the OGD-CB algorithm, which guarantees a near-optimal expected regret with stochastic values. 
On the other hand, when values are adversarial, we prove that this algorithm also reaches the upper bound on the asymptotic competitive ratio. 
We further compare throttling with pacing, another widely adopted budget control method, in repeated second-price auctions. 
In the stochastic case, we demonstrate that pacing is generally superior to throttling for the advertiser, supporting the well-known result that pacing is asymptotically optimal in this scenario. 
However, in the adversarial case, we give an exciting result indicating that throttling is also an asymptotically optimal dynamic bidding strategy. 
Our results bridge the gaps in theoretical research of throttling in repeated auctions and comprehensively reveal the ability of this popular budget-smoothing strategy.

----

## [1068] The Complexity of Computing Robust Mediated Equilibria in Ordinal Games

**Authors**: *Vincent Conitzer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28817](https://doi.org/10.1609/aaai.v38i9.28817)

**Abstract**:

Usually, to apply game-theoretic methods, we must specify utilities
precisely, and we run the risk that the solutions we compute are not
robust to errors in this specification.  Ordinal games provide an
attractive alternative: they require specifying only which outcomes
are preferred to which other ones.  Unfortunately, they provide little
guidance for how to play unless there are pure Nash equilibria;
evaluating mixed strategies appears to fundamentally require cardinal
utilities.

In this paper, we observe that we can in fact make good use of mixed
strategies in ordinal games if we consider settings that allow for
folk theorems.  These allow us to find equilibria that are robust, in
the sense that they remain equilibria no matter which cardinal
utilities are the correct ones -- as long as they are consistent with
the specified ordinal preferences.  We analyze this concept and study
the computational complexity of finding such equilibria in a range of
settings.

----

## [1069] Learning Discrete-Time Major-Minor Mean Field Games

**Authors**: *Kai Cui, Gökçe Dayanikli, Mathieu Laurière, Matthieu Geist, Olivier Pietquin, Heinz Koeppl*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28818](https://doi.org/10.1609/aaai.v38i9.28818)

**Abstract**:

Recent techniques based on Mean Field Games (MFGs) allow the scalable analysis of multi-player games with many similar, rational agents. However, standard MFGs remain limited to homogeneous players that weakly influence each other, and cannot model major players that strongly influence other players, severely limiting the class of problems that can be handled. We propose a novel discrete time version of major-minor MFGs (M3FGs), along with a learning algorithm based on fictitious play and partitioning the probability simplex. Importantly, M3FGs generalize MFGs with common noise and can handle not only random exogeneous environment states but also major players. A key challenge is that the mean field is stochastic and not deterministic as in standard MFGs. Our theoretical investigation verifies both the M3FG model and its algorithmic solution, showing firstly the well-posedness of the M3FG model starting from a finite game of interest, and secondly convergence and approximation guarantees of the fictitious play algorithm. Then, we empirically verify the obtained theoretical results, ablating some of the theoretical assumptions made, and show successful equilibrium learning in three example problems. Overall, we establish a learning framework for a novel and broad class of tractable games.

----

## [1070] Automated Design of Affine Maximizer Mechanisms in Dynamic Settings

**Authors**: *Michael J. Curry, Vinzenz Thoma, Darshan Chakrabarti, Stephen McAleer, Christian Kroer, Tuomas Sandholm, Niao He, Sven Seuken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28819](https://doi.org/10.1609/aaai.v38i9.28819)

**Abstract**:

Dynamic mechanism design is a challenging extension to ordinary mechanism design in which the mechanism designer must make a sequence of decisions over time in the face of possibly untruthful reports of participating agents.
Optimizing dynamic mechanisms for welfare is relatively well understood. However, there has been less work on optimizing for other goals (e.g., revenue), and without restrictive assumptions on valuations, it is remarkably challenging to characterize good mechanisms. Instead, we turn to automated mechanism design to find mechanisms with good performance in specific problem instances.
We extend the class of affine maximizer mechanisms to MDPs where agents may untruthfully report their rewards. This extension results in a challenging bilevel optimization problem in which the upper problem involves choosing optimal mechanism parameters, and the lower problem involves solving the resulting MDP. 
Our approach can find truthful dynamic mechanisms that achieve strong performance on goals other than welfare, and can be applied to essentially any problem setting---without restrictions on valuations---for which RL can learn optimal policies.

----

## [1071] How to Evaluate Behavioral Models

**Authors**: *Greg d'Eon, Sophie Greenwood, Kevin Leyton-Brown, James R. Wright*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28820](https://doi.org/10.1609/aaai.v38i9.28820)

**Abstract**:

Researchers building behavioral models, such as behavioral game theorists, use experimental data to evaluate predictive models of human behavior. However, there is little agreement about which loss function should be used in evaluations, with error rate, negative log-likelihood, cross-entropy, Brier score, and squared L2 error all being common choices. We attempt to offer a principled answer to the question of which loss functions should be used for this task, formalizing axioms that we argue loss functions should satisfy. We construct a family of loss functions, which we dub ``diagonal bounded Bregman divergences'', that satisfy all of these axioms. These rule out many loss functions used in practice, but notably include squared L2 error; we thus recommend its use for evaluating behavioral models.

----

## [1072] Independence of Irrelevant Alternatives under the Lens of Pairwise Distortion

**Authors**: *Théo Delemazure, Jérôme Lang, Grzegorz Pierczynski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28821](https://doi.org/10.1609/aaai.v38i9.28821)

**Abstract**:

We give a quantitative analysis of the independence of irrelevant alternatives (IIA) axiom. IIA says that the society's preference between x and y should depend only on individual preferences between x and y: we show that, in several contexts, if the individuals express their preferences about additional (``irrelevant'') alternatives, this information helps to estimate better which of x and y has higher social welfare.
Our contribution is threefold: (1) we provide a new tool to measure the impact of IIA on social welfare (pairwise distortion), based on the well-established notion of voting distortion, (2) we study the average impact of IIA in both general and metric settings, with experiments on synthetic and real data and (3)  we study the worst-case impact of IIA in the 1D-Euclidean metric space.

----

## [1073] The Complexity of Fair Division of Indivisible Items with Externalities

**Authors**: *Argyrios Deligkas, Eduard Eiben, Viktoriia Korchemna, Simon Schierreich*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28822](https://doi.org/10.1609/aaai.v38i9.28822)

**Abstract**:

We study the computational complexity of fairly allocating a set of indivisible items under externalities. In this recently-proposed setting, in addition to the utility the agent gets from their bundle, they also receive utility from items allocated to other agents.
We focus on the extended definitions of envy-freeness up to one item (EF1) and of envy-freeness up to any item (EFX), and we provide the landscape of their complexity for several different scenarios. We prove that it is NP-complete to decide whether there exists an EFX allocation, even when there are only three agents, or even when there are only six different values for the items. We complement these negative results by showing that when both the number of agents and the number of different values for items are bounded by a parameter the problem becomes fixed-parameter tractable. Furthermore, we prove that two-valued and binary-valued instances are equivalent and that EFX and EF1 allocations coincide for this class of instances. Finally, motivated from real-life scenarios, we focus on a class of structured valuation functions, which we term agent/item-correlated. We prove their equivalence to the "standard" setting without externalities. Therefore, all previous results for EF1 and EFX apply immediately for these valuations.

----

## [1074] Competition among Pairwise Lottery Contests

**Authors**: *Xiaotie Deng, Hangxin Gan, Ningyuan Li, Weian Li, Qi Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28823](https://doi.org/10.1609/aaai.v38i9.28823)

**Abstract**:

We investigate a two-stage competitive model involving multiple contests. In this model, each contest designer chooses two participants from a pool of candidate contestants and determines the biases. Contestants strategically distribute their efforts across various contests within their budget. We first show the existence of a pure strategy Nash equilibrium (PNE) for the contestants, and propose a fully polynomial-time approximation scheme to compute an approximate PNE. In the scenario where designers simultaneously decide the participants and biases, the subgame perfect equilibrium (SPE) may not exist. Nonetheless, when designers' decisions are made in two substages, the existence of SPE is established. In the scenario where designers can hold multiple contests, we show that the SPE always exists under mild conditions and can be computed efficiently.

----

## [1075] Refined Characterizations of Approval-Based Committee Scoring Rules

**Authors**: *Chris Dong, Patrick Lederer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28824](https://doi.org/10.1609/aaai.v38i9.28824)

**Abstract**:

In approval-based committee (ABC) elections, the goal is to select a fixed-size subset of the candidates, a so-called committee, based on the voters' approval ballots over the candidates. One of the most popular classes of ABC voting rules are ABC scoring rules, for which voters give points to each committee and the committees with maximal total points are chosen. While the set of ABC scoring rules has recently been characterized in a model where the output is a ranking of all committees, no full characterization of these rules exists in the standard model where a set of winning committees is returned. We address this issue by characterizing two important subclasses of ABC scoring rules in the standard ABC election model, thereby both extending the result for ABC ranking rules to the standard setting and refining it to subclasses. In more detail, by relying on a consistency axiom for variable electorates, we characterize (i) the prominent class of Thiele rules and (ii) a new class of ABC voting rules called ballot size weighted approval voting. Based on these theorems, we also infer characterizations of three well-known ABC voting rules, namely multi-winner approval voting, proportional approval voting, and satisfaction approval voting.

----

## [1076] Implications of Distance over Redistricting Maps: Central and Outlier Maps

**Authors**: *Seyed A. Esmaeili, Darshan Chakrabarti, Hayley Grape, Brian Brubach*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28825](https://doi.org/10.1609/aaai.v38i9.28825)

**Abstract**:

In representative democracy, a redistricting map is chosen to partition an electorate into districts which each elects a representative. A valid redistricting map must satisfy a collection of constraints such as being compact, contiguous, and of almost-equal population. However, these constraints are loose enough to enable an enormous ensemble of valid redistricting maps. This enables a partisan legislature to gerrymander by choosing a map which unfairly favors it. In this paper, we introduce an interpretable and tractable distance measure over redistricting maps which does not use election results and study its implications over the ensemble of redistricting maps. Specifically, we define a central map which may be considered "most typical" and give a rigorous justification for it by showing that it mirrors the Kemeny ranking in a scenario where we have a committee voting over a collection of redistricting maps to be drawn. We include runnning time and sample complexity analysis for our algorithms, including some negative results which hold using any algorithm. We further study outlier detection based on this distance measure and show that our framework can detect some gerrymandered maps. More precisely, we show some maps that are widely considered to be gerrymandered that lie very far away from our central maps in comparison to a large ensemble of valid redistricting maps. Since our distance measure does not rely on election results, this gives a significant advantage in gerrymandering detection which is lacking in all previous methods.

----

## [1077] On Optimal Tradeoffs between EFX and Nash Welfare

**Authors**: *Michal Feldman, Simon Mauras, Tomasz Ponitka*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28826](https://doi.org/10.1609/aaai.v38i9.28826)

**Abstract**:

A major problem in fair division is how to allocate a set of indivisible resources among agents fairly and efficiently. The goal of this work is to characterize the tradeoffs between two well-studied measures of fairness and efficiency --- envy freeness up to any item (EFX) for fairness, and Nash welfare for efficiency --- by saying, for given constants α and β, whether there exists an α-EFX allocation that guarantees a β-fraction of the maximum Nash welfare (β-MNW). For additive valuations, we show that for any α ∈ [0,1], there exists a partial allocation that is α-EFX and 1/(α+1)-MNW. This tradeoff turns out to be tight (for every α) as demonstrated by an impossibility result that we give. We also show that for α ∈ [0, φ-1 ≃ 0.618] these partial allocations can be turned into complete allocations where all items are assigned. Furthermore, for any α ∈ [0, 1/2], we show that the tight tradeoff of α-EFX and 1/(α+1)-MNW with complete allocations holds for the more general setting of subadditive valuations. Our results improve upon the current state of the art, for both additive and subadditive valuations, and match the best-known approximations of EFX under complete allocations, regardless of Nash welfare guarantees. Notably, our constructions for additive valuations also provide EF1 and constant approximations for maximin share guarantees.

----

## [1078] Manipulation-Robust Selection of Citizens' Assemblies

**Authors**: *Bailey Flanigan, Jennifer Liang, Ariel D. Procaccia, Sven Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28827](https://doi.org/10.1609/aaai.v38i9.28827)

**Abstract**:

Among the recent work on designing algorithms for selecting citizens' assembly participants, one key property of these algorithms has not yet been studied: their manipulability. Strategic manipulation is a concern because these algorithms must satisfy representation constraints according to volunteers' self-reported features; misreporting these features could thereby increase a volunteer's chance of being selected, decrease someone else's chance, and/or increase the expected number of seats given to their group. Strikingly, we show that Leximin — an algorithm that is widely used for its fairness — is highly manipulable in this way. We then introduce a new class of selection algorithms that use Lp norms as objective functions. We show that the manipulability of the Lp-based algorithm decreases in O(1/n^(1-1/p)) as the number of volunteers n grows, approaching the optimal rate of O(1/n) as p approaches infinity. These theoretical results are confirmed via experiments in eight real-world datasets.

----

## [1079] Project-Fair and Truthful Mechanisms for Budget Aggregation

**Authors**: *Rupert Freeman, Ulrike Schmidt-Kraepelin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28828](https://doi.org/10.1609/aaai.v38i9.28828)

**Abstract**:

We study the budget aggregation problem in which a set of strategic voters must split a finite divisible resource (such as money or time) among a set of competing projects. Our goal is twofold: We seek truthful mechanisms that provide fairness guarantees to the projects. For the first objective, we focus on the class of moving phantom mechanisms, which are -- to this day -- essentially the only known truthful mechanisms in this setting. For project fairness, we consider the mean division as a fair baseline, and bound the maximum difference between the funding received by any project and this baseline. We propose a novel and simple moving phantom mechanism that provides optimal project fairness guarantees. As a corollary of our results, we show that our new mechanism minimizes the L1 distance to the mean for three projects and gives the first non-trivial bounds on this quantity for more than three projects.

----

## [1080] Maxileximin Envy Allocations and Connected Goods

**Authors**: *Gianluigi Greco, Francesco Scarcello*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28829](https://doi.org/10.1609/aaai.v38i9.28829)

**Abstract**:

Fair allocation of indivisible goods presents intriguing challenges from both a social choice perspective and an algorithmic standpoint. Due to the indivisibility of goods, it is common for one agent to envy the bundle of goods assigned to another agent and, indeed, envy-free solutions do not exist in general. In line with the classical game-theoretic concept of Nucleolus in coalitional games, we propose that a fair allocation should minimize the agents’ dissatisfaction profile in a lexicographic manner, where the dissatisfaction of an agent is defined as her maximum envy towards other agents. Therefore, we seek allocations that minimize the maximum envy. In cases where multiple solutions have an equal maximum value, we minimize the second-worst value, and so on. Additionally, as is customary in fair division problems, we also consider an efficiency requirement: among the allocations with the best agents’ dissatisfaction profile, we prioritize those that maximize the sum of agents’ utilities, known as maximum social welfare. Such allocations, referred to as maxileximin allocations, always exist.
In this study, we analyze the computational properties of maxileximin allocations in the context of fair allocation problems with constraints. Specifically, we focus on the Connected Fair Division problem, where goods correspond to the nodes of a graph, and a bundle of goods is allowed if the subgraph formed by those goods is connected. We demonstrate that the problem is F∆P2 -complete, even for instances with simple graphical structures such as path and star graphs.
However, we identify islands of tractability for instances with more intricate graphs, such as those having bounded treewidth, provided that the number of agents is bounded by a fixed number and utility functions use small values.

----

## [1081] Information Design for Congestion Games with Unknown Demand

**Authors**: *Svenja M. Griesbach, Martin Hoefer, Max Klimm, Tim Koglin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28830](https://doi.org/10.1609/aaai.v38i9.28830)

**Abstract**:

We study a novel approach to information design in the standard traffic model of network congestion games. It captures the natural condition that the demand is unknown to the users of the network. A principal (e.g., a mobility service) commits to a signaling strategy, observes the realized demand and sends a (public) signal to agents (i.e., users of the network). Based on the induced belief about the demand, the users then form an equilibrium. We consider the algorithmic goal of the principal: Compute a signaling scheme that minimizes the expected total cost of the induced equilibrium. We concentrate on single-commodity networks and affine cost functions, for which we obtain the following results.

First, we devise a fully polynomial-time approximation scheme (FPTAS) for the case that the demand can only take two values. It relies on several structural properties of the cost of the induced equilibrium as a function of the updated belief about the distribution of demands. We show that this function is piecewise linear for any number of demands, and monotonic for two demands. 
Second, we give a complete characterization of the graph structures for which it is optimal to fully reveal the information about the realized demand. This signaling scheme turns out to be optimal for all cost functions and probability distributions over demands if and only if the graph is series-parallel. Third, we propose an algorithm that computes the optimal signaling scheme for any number of demands whose time complexity is polynomial in the number of supports that occur in a Wardrop equilibrium for some demand. Finally, we conduct a computational study that tests this algorithm on real-world instances.

----

## [1082] Zero-Sum Games between Mean-Field Teams: Reachability-Based Analysis under Mean-Field Sharing

**Authors**: *Yue Guan, Mohammad Afshari, Panagiotis Tsiotras*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28831](https://doi.org/10.1609/aaai.v38i9.28831)

**Abstract**:

This work studies the behaviors of two large-population teams competing in a discrete environment. The team-level interactions are modeled as a zero-sum game while the agent dynamics within each team is formulated as a collaborative mean-field team problem. Drawing inspiration from the mean-field literature, we first approximate the large-population team game with its infinite-population limit. Subsequently, we construct a fictitious centralized system and transform the infinite-population game to an equivalent zero-sum game between two coordinators. Via a novel reachability analysis, we study the optimality of coordination strategies, which induce decentralized strategies under the original information structure. The optimality of the resulting strategies is established in the original finite-population game, and the theoretical guarantees are verified by numerical examples.

----

## [1083] Worst-Case VCG Redistribution Mechanism Design Based on the Lottery Ticket Hypothesis

**Authors**: *Mingyu Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28832](https://doi.org/10.1609/aaai.v38i9.28832)

**Abstract**:

We study worst-case VCG redistribution mechanism design for the public project problem.  The mechanism design task comes down to designing a payment function that maximizes the worst-case allocative efficiency ratio.

We use a multilayer perceptron (MLP) with ReLU activation to model the payment function and use mixed integer programming (MIP) to solve for the worst-case type profiles that maximally violate the mechanism design constraints. We collect these worst-case type profiles and use them as training samples to train toward better worst-case mechanisms.

In practice, we require a tiny neural network structure for the above approach to scale.  The Lottery Ticket Hypothesis states that a large network is likely to contain a "winning ticket" -- a much smaller subnetwork that "won the initialization lottery", which makes its training particularly effective.  Motivated by this hypothesis, we train a large network and prune it into a tiny subnetwork.  We run MIP-based worst-case training on the drawn subnetwork and evaluate the resulting mechanism's worst-case performance.  If the subnetwork does not achieve good worst-case performance, then we record the type profiles that cause the current draw to be bad.  To draw again, we restore the large network to its initial weights and prune using recorded type profiles from earlier draws, therefore avoiding drawing the same ticket twice.  We expect to eventually encounter a tiny subnetwork that leads to effective training for our worst-case mechanism design task.  Lastly, a by-product of multiple ticket draws is an ensemble of mechanisms with different worst cases, which improves the worst-case performance further.

Using our approach, we find previously unknown optimal mechanisms for up to 5 agents. Our results confirm the tightness of existing theoretical upper bounds.  For up to 20 agents, we derive significantly improved worst-case mechanisms, surpassing a long list of existing manual results.

----

## [1084] An Exercise in Tournament Design: When Some Matches Must Be Scheduled

**Authors**: *Sushmita Gupta, Ramanujan Sridharan, Peter Strulo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28833](https://doi.org/10.1609/aaai.v38i9.28833)

**Abstract**:

Single-elimination (SE) tournaments are a popular format used in competitive environments and decision making. Algorithms for SE tournament manipulation  have been an active topic of research in recent years. In this paper, we initiate the algorithmic study of a novel variant of SE tournament manipulation that aims to model the fact that certain matchups are highly desired in a sporting context, incentivizing an organizer to manipulate the bracket to make such matchups take place.  We obtain both hardness and tractability results. We show that while the problem of computing a bracket enforcing a given set of matches in an SE tournament is NP-hard, there are natural restrictions that lead to polynomial-time solvability. In particular, we show polynomial-time solvability if there is a linear ordering on the ability of players with only a constant number of exceptions where a player with lower ability beats a player with higher ability.

----

## [1085] Regret Analysis of Repeated Delegated Choice

**Authors**: *Mohammad Hajiaghayi, Mohammad Mahdavi, Keivan Rezaei, Suho Shin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28834](https://doi.org/10.1609/aaai.v38i9.28834)

**Abstract**:

We present a study on a repeated delegated choice problem, which is the first to consider an online learning variant of Kleinberg and Kleinberg, EC'18. In this model, a principal interacts repeatedly with an agent who possesses an exogenous set of solutions to search for efficient ones. Each solution can yield varying utility for both the principal and the agent, and the agent may propose a solution to maximize its own utility in a selfish manner. To mitigate this behavior, the principal announces an eligible set which screens out a certain set of solutions. The principal, however, does not have any information on the distribution of solutions nor the number of solutions in advance. Therefore, the principal dynamically announces various eligible sets to efficiently learn the distribution. The principal's objective is to minimize cumulative regret compared to the optimal eligible set in hindsight. We explore two dimensions of the problem setup, whether the agent behaves myopically or strategizes across the rounds, and whether the solutions yield deterministic or stochastic utility. We obtain sublinear regret upper bounds in various regimes, and derive corresponding lower bounds which implies the tightness of the results. Overall, we bridge a well-known problem in economics to the evolving area of online learning, and present a comprehensive study in this problem.

----

## [1086] Cost Minimization for Equilibrium Transition

**Authors**: *Haoqiang Huang, Zihe Wang, Zhide Wei, Jie Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28835](https://doi.org/10.1609/aaai.v38i9.28835)

**Abstract**:

In this paper, we delve into the problem of using monetary incentives to encourage players to shift from an initial Nash equilibrium to a more favorable one within a game. Our main focus revolves around computing the minimum reward required to facilitate this equilibrium transition. The game involves a single row player who possesses m strategies and k column players, each endowed with n strategies. Our findings reveal that determining whether the minimum reward is zero is NP-complete, and computing the minimum reward becomes APX-hard. Nonetheless, we bring some positive news, as this problem can be efficiently handled if either k or n is a fixed constant. Furthermore, we have devised an approximation algorithm with an additive error that runs in polynomial time. Lastly, we explore a specific case wherein the utility functions exhibit single-peaked characteristics, and we successfully demonstrate that the optimal reward can be computed in polynomial time.

----

## [1087] Reachability of Fair Allocations via Sequential Exchanges

**Authors**: *Ayumi Igarashi, Naoyuki Kamiyama, Warut Suksompong, Sheung Man Yuen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28836](https://doi.org/10.1609/aaai.v38i9.28836)

**Abstract**:

In the allocation of indivisible goods, a prominent fairness notion is envy-freeness up to one good (EF1). We initiate the study of reachability problems in fair division by investigating the problem of whether one EF1 allocation can be reached from another EF1 allocation via a sequence of exchanges such that every intermediate allocation is also EF1. We show that two EF1 allocations may not be reachable from each other even in the case of two agents, and deciding their reachability is PSPACE-complete in general. On the other hand, we prove that reachability is guaranteed for two agents with identical or binary utilities as well as for any number of agents with identical binary utilities. We also examine the complexity of deciding whether there is an EF1 exchange sequence that is optimal in the number of exchanges required.

----

## [1088] Repeated Fair Allocation of Indivisible Items

**Authors**: *Ayumi Igarashi, Martin Lackner, Oliviero Nardi, Arianna Novaro*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28837](https://doi.org/10.1609/aaai.v38i9.28837)

**Abstract**:

The problem of fairly allocating a set of indivisible items is a well-known challenge in the field of (computational) social choice. In this scenario, there is a fundamental incompatibility between notions of fairness (such as envy-freeness and proportionality) and economic efficiency (such as Pareto-optimality). However, in the real world, items are not always allocated once and for all, but often repeatedly. For example, the items may be recurring chores to distribute in a household. Motivated by this, we initiate the study of the repeated fair division of indivisible goods and chores, and propose a formal model for this scenario. In this paper, we show that, if the number of repetitions is a multiple of the number of agents, there always exists a sequence of allocations that is proportional and Pareto-optimal. On the other hand, irrespective of the number of repetitions, an envy-free and Pareto-optimal sequence of allocations may not exist. For the case of two agents, we show that if the number of repetitions is even, it is always possible to find a sequence of allocations that is overall envy-free and Pareto-optimal. We then prove even stronger fairness guarantees, showing that every allocation in such a sequence satisfies some relaxation of envy-freeness. Finally, in case that the number of repetitions can be chosen freely, we show that envy-free and Pareto-optimal allocations are achievable for any number of agents.

----

## [1089] Spatial Voting with Incomplete Voter Information

**Authors**: *Aviram Imber, Jonas Israel, Markus Brill, Hadas Shachnai, Benny Kimelfeld*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28838](https://doi.org/10.1609/aaai.v38i9.28838)

**Abstract**:

We consider spatial voting where candidates are located in the Euclidean d-dimensional space, and each voter ranks candidates based on their distance from the voter's ideal point. We explore the case where information about the location of voters' ideal points is incomplete: for each dimension, we are given an interval of possible values. We study the computational complexity of finding the possible and necessary winners for positional scoring rules. Our results show that we retain tractable cases of the classic model where voters have partial-order preferences. Moreover, we show that there are positional scoring rules under which the possible-winner problem is intractable for partial orders, but tractable in the one-dimensional spatial setting. 
We also consider approval voting in this setting. We show that for up to two dimensions, the necessary-winner problem is tractable, while the possible-winner problem is hard for any number of dimensions.

----

## [1090] Maximizing Nash Social Welfare under Two-Sided Preferences

**Authors**: *Pallavi Jain, Rohit Vaish*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28839](https://doi.org/10.1609/aaai.v38i9.28839)

**Abstract**:

The maximum Nash social welfare (NSW)---which maximizes the geometric mean of agents' utilities---is a fundamental solution concept with remarkable fairness and efficiency guarantees. The computational aspects of NSW have been extensively studied for *one-sided* preferences where a set of agents have preferences over a set of resources. Our work deviates from this trend and studies NSW maximization for *two-sided* preferences, wherein a set of workers and firms, each having a cardinal valuation function, are matched with each other. We provide a systematic study of the computational complexity of maximizing NSW for many-to-one matchings under two-sided preferences. Our main negative result is that maximizing NSW is NP-hard even in a highly restricted setting where each firm has capacity 2, all valuations are in the range {0,1,2}, and each agent positively values at most three other agents. In search of positive results, we develop approximation algorithms as well as parameterized algorithms in terms of natural parameters such as the number of workers, the number of firms, and the firms' capacities. We also provide algorithms for restricted domains such as symmetric binary valuations and bounded degree instances.

----

## [1091] Optimal Mechanism in a Dynamic Stochastic Knapsack Environment

**Authors**: *Jihyeok Jung, Chan-Oi Song, Deok-Joo Lee, Kiho Yoon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28840](https://doi.org/10.1609/aaai.v38i9.28840)

**Abstract**:

This study introduces an optimal mechanism in a dynamic stochastic knapsack environment. The model features a single seller who has a fixed quantity of a perfectly divisible item. Impatient buyers with a piece-wise linear utility function arrive randomly and they report the two-dimensional private information: marginal value and demanded quantity. We derive a revenue-maximizing dynamic mechanism in a finite discrete time framework that satisfies incentive compatibility, individual rationality, and feasibility conditions. This is achieved by characterizing buyers' utility and utilizing the Bellman equation. Moreover, we establish the essential penalty scheme for incentive compatibility, as well as the allocation and payment policies. Lastly, we propose algorithms to approximate the optimal policy, based on the Monte Carlo simulation-based regression method and reinforcement learning.

----

## [1092] Proportional Representation in Metric Spaces and Low-Distortion Committee Selection

**Authors**: *Yusuf Hakan Kalayci, David Kempe, Vikram Kher*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28841](https://doi.org/10.1609/aaai.v38i9.28841)

**Abstract**:

We introduce a novel definition for a small set R of k points being "representative" of a larger set in a metric space. Given a set V (e.g., documents or voters) to represent, and a set C of possible representatives, our criterion requires that for any subset S comprising a theta fraction of V, the average distance of S to their best theta*k points in R should not be more than a factor gamma compared to their average distance to the best theta*k points among all of C. This definition is a strengthening of proportional fairness and core fairness, but - different from those notions - requires that large cohesive clusters be represented proportionally to their size.

Since there are instances for which - unless gamma is polynomially large - no solutions exist, we study this notion in a resource augmentation framework, implicitly stating the constraints for a set R of size k as though its size were only k/alpha, for alpha > 1. Furthermore, motivated by the application to elections, we mostly focus on the "ordinal" model, where the algorithm does not learn the actual distances; instead, it learns only for each point v in V and each candidate pairs c, c' which of c, c' is closer to v. Our main result is that the Expanding Approvals Rule (EAR) of Aziz and Lee is (alpha, gamma) representative with gamma <= 1 + 6.71 * (alpha)/(alpha-1).

Our results lead to three notable byproducts. First, we show that the EAR achieves constant proportional fairness in the ordinal model, giving the first positive result on metric proportional fairness with ordinal information. Second, we show that for the core fairness objective, the EAR achieves the same asymptotic tradeoff between resource augmentation and approximation as the recent results of Li et al., which used full knowledge of the metric. Finally, our results imply a very simple single-winner voting rule with metric distortion at most 44.

----

## [1093] Towards Optimal Subsidy Bounds for Envy-Freeable Allocations

**Authors**: *Yasushi Kawase, Kazuhisa Makino, Hanna Sumita, Akihisa Tamura, Makoto Yokoo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28842](https://doi.org/10.1609/aaai.v38i9.28842)

**Abstract**:

We study the fair division of indivisible items with subsidies among n agents, where the absolute marginal valuation of each item is at most one. Under monotone valuations (where each item is a good), it is known that a maximum subsidy of 2(n-1) and a total subsidy of 2(n-1)² are sufficient to guarantee the existence of an envy-freeable allocation. In this paper, we improve upon these bounds, even in a wider model. Namely, we show that, given an EF1 allocation, we can compute in polynomial time an envy-free allocation with a subsidy of at most n-1 per agent and a total subsidy of at most n(n-1)/2. Moreover, we present further improved bounds for monotone valuations.

----

## [1094] Strategyproof Mechanisms for Group-Fair Obnoxious Facility Location Problems

**Authors**: *Jiaqian Li, Minming Li, Hau Chan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28843](https://doi.org/10.1609/aaai.v38i9.28843)

**Abstract**:

We study the group-fair obnoxious facility location problems from the mechanism design perspective where agents belong to different groups and have private location preferences on the undesirable locations of the facility. Our main goal is to design strategyproof mechanisms that elicit the true location preferences from the agents and determine a facility location that approximately optimizes several group-fair objectives. We first consider the maximum total and average group cost (group-fair) objectives. For these objectives, we propose deterministic mechanisms that achieve 3-approximation ratios and provide matching lower bounds. We then provide the characterization of 2-candidate strategyproof randomized mechanisms. Leveraging the characterization, we design randomized mechanisms with improved approximation ratios of 2 for both objectives. We also provide randomized lower bounds of 5/4 for both objectives. Moreover, we investigate intergroup and intragroup fairness (IIF) objectives, addressing fairness between groups and within each group. We present a mechanism that achieves a 4-approximation for the IIF objectives and provide tight lower bounds.

----

## [1095] Opponent-Model Search in Games with Incomplete Information

**Authors**: *Junkang Li, Bruno Zanuttini, Véronique Ventos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28844](https://doi.org/10.1609/aaai.v38i9.28844)

**Abstract**:

Games with incomplete information are games that model situations where players do not have common knowledge about the game they play, e.g. card games such as poker or bridge. Opponent models can be of crucial importance for decision-making in such games. We propose algorithms for computing optimal and/or robust strategies in games with incomplete information, given various types of knowledge about opponent models. As an application, we describe a framework for reasoning about an opponent's reasoning in such games, where opponent models arise naturally.

----

## [1096] Double Auction on Diffusion Network

**Authors**: *Miao Li, Yuhan Cao, Dengji Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28845](https://doi.org/10.1609/aaai.v38i9.28845)

**Abstract**:

Mechanism design on social networks has attracted extensive attention recently. The goal is to design mechanisms to incentivize participants to invite more participants via their social networks, and the challenge is that the participants are competitors. Various mechanisms have been proposed for single-/multiple-unit auctions, but it has been shown that it is challenging to design such mechanisms for more complex settings. We move this forward to investigate a double auction on a network where each trader (a buyer or a seller) can link to other buyers and sellers. Incentiving invitation is more difficult than in multi-unit one-sided auctions, because there are two different roles and a buyer (seller) seems happy to invite a seller (buyer), but again the invited seller (buyer) may invite another buyer (seller) to compete with the original buyer (seller). To combat this, we propose a solution called dynamic trade reduction (DTR), which also guarantees a non-negative revenue for the market owner. Interestingly, our solution is also applicable to the multi-unit one-sided auction when there is only one seller linking to only buyers on the network. We believe that the principle of our solution has the potential to be extended to design the multi-item one-sided auction.

----

## [1097] Pay to (Not) Play: Monetizing Impatience in Mobile Games

**Authors**: *Taylor Lundy, Narun Raman, Hu Fu, Kevin Leyton-Brown*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28846](https://doi.org/10.1609/aaai.v38i9.28846)

**Abstract**:

Mobile gaming is a rapidly growing and incredibly profitable sector; having grown seven-fold over the past 10 years, it now grosses over $100 billion annually. This growth was due in large part to a shift in monetization strategies: rather than charging players an upfront cost ("pay-to-play"), games often request optional microtransactions throughout gameplay ("free-to-play"). We focus on a common scenario in which games include wait times---gating either items or game progression---that players can pay to skip. Game designers typically say that they optimize for player happiness rather than revenue; however, prices for skips are typically set at levels that few players are willing to pay, leading to low purchase rates. Under a traditional analysis, it would seem that game designers fail at their stated goal if few players buy what they are selling. We argue that an alternate model can better explain this dynamic: players value tasks more highly as they are perceived to be more difficult. While skips can increase players' utilities by providing instant gratification, pricing skips too cheaply can lower players' utilities by decreasing the perceived amount of work needed to complete a task. We show that high revenue, high player utility, and low purchase rates can all coexist under this model, particularly under a realistic distribution of players having few buyers but a few big-spending "whales." We also investigate how a game designer should optimize prices under our model. An appendix of the paper with proofs, more comprehensive results and visualizations can be found at https://arxiv.org/abs/2312.10205.

----

## [1098] Weighted Envy-Freeness for Submodular Valuations

**Authors**: *Luisa Montanari, Ulrike Schmidt-Kraepelin, Warut Suksompong, Nicholas Teh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28847](https://doi.org/10.1609/aaai.v38i9.28847)

**Abstract**:

We investigate the fair allocation of indivisible goods to agents with possibly different entitlements represented by weights. Previous work has shown that guarantees for additive valuations with existing envy-based notions cannot be extended to the case where agents have matroid-rank (i.e., binary submodular) valuations. We propose two families of envy-based notions for matroid-rank and general submodular valuations, one based on the idea of transferability and the other on marginal values. We show that our notions can be satisfied via generalizations of rules such as picking sequences and maximum weighted Nash welfare. In addition, we introduce welfare measures based on harmonic numbers, and show that variants of maximum weighted harmonic welfare offer stronger fairness guarantees than maximum weighted Nash welfare under matroid-rank valuations.

----

## [1099] Computing Nash Equilibria in Potential Games with Private Uncoupled Constraints

**Authors**: *Nikolas Patris, Stelios Stavroulakis, Fivos Kalogiannis, Rose Zhang, Ioannis Panageas*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28848](https://doi.org/10.1609/aaai.v38i9.28848)

**Abstract**:

We consider the problem of computing Nash equilibria in potential games where each player's strategy set is subject to private uncoupled constraints. This scenario is frequently encountered in real-world applications like road network congestion games where individual drivers adhere to personal budget and fuel limitations. Despite the plethora of algorithms that efficiently compute Nash equilibria (NE) in potential games, the domain of constrained potential games remains largely unexplored. We introduce an algorithm that leverages the Lagrangian formulation of NE. The algorithm is implemented independently by each player and runs in polynomial time with respect to the approximation error, the sum of the size of the action-spaces, and the game's inherit parameters.

----

## [1100] Peer Neighborhood Mechanisms: A Framework for Mechanism Generalization

**Authors**: *Adam Richardson, Boi Faltings*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28849](https://doi.org/10.1609/aaai.v38i9.28849)

**Abstract**:

Peer prediction incentive mechanisms for crowdsourcing are generally limited to eliciting samples from categorical distributions. Prior work on extending peer prediction to arbitrary distributions has largely relied on assumptions on the structures of the distributions or known properties of the data providers. We introduce a novel class of incentive mechanisms that extend peer prediction mechanisms to arbitrary distributions by replacing the notion of an exact match with a concept of neighborhood matching. We present conditions on the belief updates of the data providers that guarantee incentive-compatibility for rational data providers, and admit a broad class of possible reasonable updates.

----

## [1101] Machine Learning-Powered Combinatorial Clock Auction

**Authors**: *Ermis Nikiforos Soumalias, Jakob Weissteiner, Jakob Heiss, Sven Seuken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28850](https://doi.org/10.1609/aaai.v38i9.28850)

**Abstract**:

We study the design of iterative combinatorial auctions (ICAs). The main challenge in this domain is that the bundle space grows exponentially in the number of items. To address this, several papers have recently proposed machine learning (ML)-based preference elicitation algorithms that aim to elicit only the most important information from bidders. However, from a practical point of view, the main shortcoming of this prior work is that those designs elicit bidders' preferences via value queries (i.e., “What is your value for the bundle {A, B}?''). In most real-world ICA domains, value queries are considered impractical, since they impose an unrealistically high cognitive burden on bidders, which is why they are not used in practice. In this paper, we address this shortcoming by designing an ML-powered combinatorial clock auction that elicits information from the bidders only via demand queries (i.e., “At prices p, what is your most preferred bundle of items?''). We make two key technical contributions: First, we present a novel method for training an ML model on demand queries. Second, based on those trained ML models, we introduce an efficient method for determining the demand query with the highest clearing potential, for which we also provide a theoretical foundation. We experimentally evaluate our ML-based demand query mechanism in several spectrum auction domains and compare it against the most established real-world ICA: the combinatorial clock auction (CCA). Our mechanism significantly outperforms the CCA in terms of efficiency in all domains, it achieves higher efficiency in a significantly reduced number of rounds, and, using linear prices, it exhibits vastly higher clearing potential. Thus, with this paper we bridge the gap between research and practice and propose the first practical ML-powered ICA.

----

## [1102] Almost Envy-Free Allocations of Indivisible Goods or Chores with Entitlements

**Authors**: *Max Springer, MohammadTaghi Hajiaghayi, Hadi Yami*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28851](https://doi.org/10.1609/aaai.v38i9.28851)

**Abstract**:

We here address the problem of fairly allocating indivisible goods or chores to n agents with weights that define their entitlement to the set of indivisible resources. Stemming from well-studied fairness concepts such as envy-freeness up to one good (EF1) and envy-freeness up to any good (EFX) for agents with equal entitlements, we present, in this study, the first set of impossibility results alongside algorithmic guarantees for fairness among agents with unequal entitlements.

Within this paper, we expand the concept of envy-freeness up to any good or chore to the weighted context (WEFX and XWEF respectively), demonstrating that these allocations are not guaranteed to exist for two or three agents. Despite these negative results, we develop a WEFX procedure for two agents with integer weights, and furthermore, we devise an approximate WEFX procedure for two agents with normalized weights. We further present a polynomial-time algorithm that guarantees a weighted envy-free allocation up to one chore (1WEF) for any number of agents with additive cost functions. Our work underscores the heightened complexity of the weighted fair division problem when compared to its unweighted counterpart.

----

## [1103] The Moderating Effect of Instant Runoff Voting

**Authors**: *Kiran Tomlinson, Johan Ugander, Jon M. Kleinberg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28852](https://doi.org/10.1609/aaai.v38i9.28852)

**Abstract**:

Instant runoff voting (IRV) has recently gained popularity as an alternative to plurality voting for political elections, with advocates claiming a range of advantages, including that it produces more moderate winners than plurality and could thus help address polarization. However, there is little theoretical backing for this claim, with existing evidence focused on case studies and simulations. In this work, we prove that IRV has a moderating effect relative to plurality voting in a precise sense, developed in a 1-dimensional Euclidean model of voter preferences.  We develop a theory of exclusion zones, derived from properties of the voter distribution, which serve to show how moderate and extreme candidates interact during IRV vote tabulation. The theory allows us to prove that if voters are symmetrically distributed and not too concentrated at the extremes, IRV cannot elect an extreme candidate over a moderate. In contrast, we show plurality can and validate our results computationally. Our methods provide new frameworks for the analysis of voting systems, deriving exact winner distributions geometrically and establishing a connection between plurality voting and stick-breaking processes.

----

## [1104] Unravelling Expressive Delegations: Complexity and Normative Analysis

**Authors**: *Giannis Tyrovolas, Andrei Constantinescu, Edith Elkind*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28853](https://doi.org/10.1609/aaai.v38i9.28853)

**Abstract**:

We consider binary group decision-making under a rich model of liquid democracy: agents submit ranked delegation options, where each option may be a function of multiple agents' votes; e.g., "I vote yes if a majority of my friends vote yes." Such ballots are unravelled into a profile of direct votes by selecting one entry from each ballot so as not to introduce cyclic dependencies. We study delegation via monotonic Boolean functions, and two unravelling procedures: MinSum, which minimises the sum of the ranks of the chosen entries, and its egalitarian counterpart, MinMax. We provide complete computational dichotomies: MinSum is hard to compute (and approximate) as soon as any non-trivial functions are permitted, and polynomial otherwise; for MinMax the easiness results extend to arbitrary-arity logical ORs and ANDs taken in isolation, but not beyond. For the classic model of delegating to individual agents, we give asymptotically near-tight algorithms for carrying out the two procedures and efficient algorithms for finding optimal unravellings with the highest vote count for a given alternative. These algorithms inspire novel tie-breaking rules for the setup of voting to change a status quo. We then introduce a new axiom, which can be viewed as a variant of the participation axiom, and use algorithmic techniques developed earlier in the paper to show that it is satisfied by MinSum and a lexicographic refinement of MinMax (but not MinMax itself).

----

## [1105] Predicting Real-World Penny Auction Durations by Integrating Game Theory and Machine Learning

**Authors**: *Yujia Wang, Haoran Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28854](https://doi.org/10.1609/aaai.v38i9.28854)

**Abstract**:

Game theory and machine learning are two widely used techniques for predicting the outcomes of strategic interactions among humans. However, the game theory-based approach often relies on strong rationality and informational assumptions, while the machine learning-based approach typically requires the testing data to come from the same distribution as the training data. Our work studies how to integrate the two techniques to address these weaknesses. We focus on the interactions among real bidders in penny auctions, and develop a three-stage framework to predict the distributions of auction durations, which indicate the numbers of bids and auctioneer revenues. Specifically, we first leverage a pre-trained neural network to encode the descriptions of products in auctions into embeddings. Second, we apply game theory models to make preliminary predictions of auction durations. In particular, we tackle the challenge of accurately inferring parameters in game theory models. Third, we develop a Multi-Branch Mixture Density Network to learn the mapping from product embeddings and game-theoretic predictions to the distributions of actual auction durations. Experiments on real-world penny auction data demonstrate that our framework outperforms both game theory-based and machine learning-based prediction approaches.

----

## [1106] Simultaneous Optimization of Bid Shading and Internal Auction for Demand-Side Platforms

**Authors**: *Yadong Xu, Bonan Ni, Weiran Shen, Xun Wang, Zichen Wang, Yinsong Xue, Pingzhong Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28855](https://doi.org/10.1609/aaai.v38i9.28855)

**Abstract**:

Online advertising has been one of the most important sources for industry's growth, where the demand-side platforms (DSP) play an important role via bidding to the ad exchanges on behalf of their advertiser clients. Since more and more ad exchanges have shifted from second to first price auctions, it is challenging for DSPs to adjust bidding strategy in the volatile environment. Recent studies on bid shading in first-price auctions may have limited performance due to relatively strong hypotheses about winning probability distribution. Moreover, these studies do not consider the incentive of advertiser clients, which can be crucial for a reliable advertising platform. In this work, we consider both the optimization of bid shading technique and the design of internal auction which is ex-post incentive compatible (IC) for the management of a DSP. Firstly, we prove that the joint design of bid shading and ex-post IC auction can be reduced to choosing one monotone bid function for each advertiser without loss of optimality. Then we propose a parameterized neural network to implement the monotone bid functions. With well-designed surrogate loss, the objective can be optimized in an end-to-end manner. Finally, our experimental results demonstrate the effectiveness and superiority of our algorithm.

----

## [1107] Learning Coalition Structures with Games

**Authors**: *Yixuan Even Xu, Chun Kai Ling, Fei Fang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28856](https://doi.org/10.1609/aaai.v38i9.28856)

**Abstract**:

Coalitions naturally exist in many real-world systems involving multiple decision makers such as ridesharing, security, and online ad auctions, but the coalition structure among the agents is often unknown. We propose and study an important yet previously overseen problem -- Coalition Structure Learning (CSL), where we aim to carefully design a series of games for the agents and infer the underlying coalition structure by observing their interactions in those games. We establish a lower bound on the sample complexity  -- defined as the number of games needed to learn the structure -- of any algorithms for CSL and propose the Iterative Grouping (IG) algorithm for designing normal-form games to achieve the lower bound. We show that IG can be extended to other succinct games such as congestion games and graphical games. Moreover, we solve CSL in a more restrictive and practical setting: auctions. We show a variant of IG to solve CSL in the auction setting even if we cannot design the bidder valuations. Finally, we conduct experiments to evaluate IG in the auction setting and the results align with our theoretical analysis.

----

## [1108] Non-excludable Bilateral Trade between Groups

**Authors**: *Yixuan Even Xu, Hanrui Zhang, Vincent Conitzer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28857](https://doi.org/10.1609/aaai.v38i9.28857)

**Abstract**:

Bilateral trade is one of the most natural and important forms of economic interaction: A seller has a single, indivisible item for sale, and a buyer is potentially interested. The two parties typically have different, privately known valuations for the item, and ideally, they would like to trade if the buyer values the item more than the seller. The celebrated impossibility result by Myerson and Satterthwaite shows that any mechanism for this setting must violate at least one important desideratum. In this paper, we investigate a richer paradigm of bilateral trade, with many self-interested buyers and sellers on both sides of a single trade who cannot be excluded from the trade. We show that this allows for more positive results. In fact, we establish a dichotomy in the possibility of trading efficiently. If in expectation, the buyers value the item more, we can achieve efficiency in the limit. If this is not the case, then efficiency cannot be achieved in general. En route, we characterize trading mechanisms that encourage truth-telling, which may be of independent interest. We also evaluate our trading mechanisms experimentally, and the experiments align with our theoretical results.

----

## [1109] Greedy-Based Online Fair Allocation with Adversarial Input: Enabling Best-of-Many-Worlds Guarantees

**Authors**: *Zongjun Yang, Luofeng Liao, Christian Kroer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28858](https://doi.org/10.1609/aaai.v38i9.28858)

**Abstract**:

We study an online allocation problem with sequentially arriving items and adversarially chosen agent values, with the goal of balancing fairness and efficiency. Our goal is to study the performance of algorithms that achieve strong guarantees under other input models such as stochastic inputs, in order to achieve robust guarantees against a variety of inputs. To that end, we study the PACE (Pacing According to Current Estimated utility) algorithm, an existing algorithm designed for stochastic input. We show that in the equal-budgets case, PACE is equivalent to an integral greedy algorithm. We go on to show that with natural restrictions on the adversarial input model, both the greedy allocation and PACE have asymptotically bounded multiplicative envy as well as competitive ratio for Nash welfare, with the multiplicative factors either constant or with optimal order dependence on the number of agents. This completes a "best-of-many-worlds" guarantee for PACE, since past work showed that PACE achieves guarantees for stationary and stochastic-but-non-stationary input models.

----

## [1110] On the Outcome Equivalence of Extensive-Form and Behavioral Correlated Equilibria

**Authors**: *Brian Hu Zhang, Tuomas Sandholm*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28859](https://doi.org/10.1609/aaai.v38i9.28859)

**Abstract**:

We investigate two notions of correlated equilibrium for extensive-form games: the extensive-form correlated equilibrium (EFCE) and the behavioral correlated equilibrium (BCE). We show that the two are outcome-equivalent, in the sense that every outcome distribution achievable under one notion is achievable under the other. Our result implies, to our knowledge, the first polynomial-time algorithm for computing a BCE.

----

## [1111] Eliciting Honest Information from Authors Using Sequential Review

**Authors**: *Yichi Zhang, Grant Schoenebeck, Weijie Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28860](https://doi.org/10.1609/aaai.v38i9.28860)

**Abstract**:

In the setting of conference peer review, the conference aims to accept high-quality papers and reject low-quality papers based on noisy review scores. A recent work proposes the isotonic mechanism, which can elicit the ranking of paper qualities from an author with multiple submissions to help improve the conference's decisions. However, the isotonic mechanism relies on the assumption that the author's utility is both an increasing and a convex function with respect to the review score, which is often violated in realistic settings (e.g.~when authors aim to maximize the number of accepted papers). In this paper, we propose a sequential review mechanism that can truthfully elicit the ranking information from authors while only assuming the agent's utility is increasing with respect to the true quality of her accepted papers. The key idea is to review the papers of an author in a sequence based on the provided ranking and conditioning the review of the next paper on the review scores of the previous papers. Advantages of the sequential review mechanism include: 1) eliciting truthful ranking information in a more realistic setting than prior work; 2) reducing the reviewing workload and increasing the average quality of papers being reviewed; 3) incentivizing authors to write fewer papers of higher quality.

----

## [1112] Fair Allocation of Items in Multiple Regions

**Authors**: *Houyu Zhou, Tianze Wei, Biaoshuai Tao, Minming Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28861](https://doi.org/10.1609/aaai.v38i9.28861)

**Abstract**:

We initiate the study of fair allocation with the set of divisible or indivisible items distributed in multiple regions. The key requirement is that each agent can only obtain items from one region. In this work, we consider two kinds of fairness concepts: envy-based notions including envy-freeness (EF) and envy-freeness up to one/any item (EF1/EFX), and share-based notions including proportionality (PROP) and proportionality up to one/any item (PROP1/PROPX).  On the negative side, we show NP-hardness and inapproximability results about the aforementioned fairness notions. On the positive side, we propose several algorithms to compute the partial allocations that satisfy envy-based notions and allocations that approximate the above fairness notions.

----

## [1113] Altruism in Facility Location Problems

**Authors**: *Houyu Zhou, Hau Chan, Minming Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28862](https://doi.org/10.1609/aaai.v38i9.28862)

**Abstract**:

We study the facility location problems (FLPs) with altruistic agents who act to benefit others in their affiliated groups. Our aim is to design mechanisms that elicit true locations from the agents in different overlapping groups and place a facility to serve agents to approximately optimize a given objective based on agents' costs to the facility. Existing studies of FLPs consider myopic agents who aim to minimize their own costs to the facility. We mainly consider altruistic agents with well-motivated group costs that are defined over costs incurred by all agents in their groups. Accordingly, we define Pareto strategyproofness to account for altruistic agents and their multiple group memberships with incomparable group costs. We consider mechanisms satisfying this strategyproofness under various combinations of the planner's objectives and agents' group costs. For each of these settings, we provide upper and lower bounds of approximation ratios of the mechanisms satisfying Pareto strategyproofness.

----

## [1114] Explaining Reinforcement Learning Agents through Counterfactual Action Outcomes

**Authors**: *Yotam Amitai, Yael Septon, Ofra Amir*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28863](https://doi.org/10.1609/aaai.v38i9.28863)

**Abstract**:

Explainable reinforcement learning (XRL) methods aim to help elucidate agent policies and decision-making processes. The majority of XRL approaches focus on local explanations, seeking to shed light on the reasons an agent acts the way it does at a specific world state. While such explanations are both useful and necessary, they typically do not portray the outcomes of the agent's selected choice of action. 
In this work, we propose ``COViz'', a new local explanation method that visually compares the outcome of an agent's chosen action to a counterfactual one. In contrast to most local explanations that provide state-limited observations of the agent's motivation, our method depicts alternative trajectories the agent could have taken from the given state and their outcomes. 
We evaluated the usefulness of COViz in supporting people's understanding of agents' preferences and compare it with reward decomposition, a local explanation method that describes an agent's expected utility for different actions by decomposing it into meaningful reward types. Furthermore, we examine the complementary benefits of integrating both methods. Our results show that such integration significantly improved participants' performance.

----

## [1115] Data-Driven Knowledge-Aware Inference of Private Information in Continuous Double Auctions

**Authors**: *Lvye Cui, Haoran Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28864](https://doi.org/10.1609/aaai.v38i9.28864)

**Abstract**:

Inferring the private information of humans from their strategic behavioral data is crucial and challenging. The main approach is first obtaining human behavior functions (which map public information and human private information to behavior), enabling subsequent inference of private information from observed behavior. Most existing studies rely on strong equilibrium assumptions to obtain behavior functions. Our work focuses on continuous double auctions, where multiple traders with heterogeneous rationalities and beliefs dynamically trade commodities and deriving equilibria is generally intractable. We develop a knowledge-aware machine learning-based framework to infer each trader's private cost vectors for producing different units of its commodity. Our key idea is to learn behavior functions by incorporating the statistical knowledge about private costs given the observed trader asking behavior across the population. Specifically, we first use a neural network to characterize each trader's behavior function. Second, we leverage the statistical knowledge to derive the posterior distribution of each trader's private costs given its observed asks. Third, through designing a novel loss function, we utilize the knowledge-based posterior distributions to guide the learning of the neural network. We conduct extensive experiments on a large experimental dataset, and demonstrate the superior performance of our framework over baselines in inferring the private information of humans.

----

## [1116] Procedural Level Generation with Diffusion Models from a Single Example

**Authors**: *Shiqi Dai, Xuanyu Zhu, Naiqi Li, Tao Dai, Zhi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28865](https://doi.org/10.1609/aaai.v38i9.28865)

**Abstract**:

Level generation is a central focus of Procedural Content Generation (PCG), yet deep learning-based approaches are limited by scarce training data, i.e., human-designed levels. Despite being a dominant framework, Generative Adversarial Networks (GANs) exhibit a substantial quality gap between generated and human-authored levels, alongside rising training costs, particularly with increasing token complexity. In this paper, we introduce a diffusion-based generative model that learns from just one example. Our approach involves two core components: 1) an efficient yet expressive level representation, and 2) a latent denoising network with constrained receptive fields. To start with, our method utilizes token semantic labels, similar to word embeddings, to provide dense representations. This strategy not only surpasses one-hot encoding in representing larger game levels but also improves stability and accelerates convergence in latent diffusion. In addition, we adapt the denoising network architecture to confine the receptive field to localized patches of the data, aiming to facilitate single-example learning. Extensive experiments demonstrate that our model is capable of generating stylistically congruent samples of arbitrary sizes compared to manually designed levels. It suits a wide range of level structures with fewer artifacts than GAN-based approaches. The source code is available at https://github.com/shiqi-dai/diffusioncraft.

----

## [1117] When Are Two Lists Better than One?: Benefits and Harms in Joint Decision-Making

**Authors**: *Kate Donahue, Sreenivas Gollapudi, Kostas Kollias*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28866](https://doi.org/10.1609/aaai.v38i9.28866)

**Abstract**:

Historically, much of machine learning research has focused on the performance of the algorithm alone, but recently more attention has been focused on optimizing joint human-algorithm performance. Here, we analyze a specific type of human-algorithm collaboration where the algorithm has access to a set of n items, and presents a subset of size k to the human, who selects a final item from among those k. This scenario could model content recommendation, route planning, or any type of labeling task. Because both the human and algorithm have imperfect, noisy information about the true ordering of items, the key question is: which value of k maximizes the probability that the best item will be ultimately selected? For k=1, performance is optimized by the algorithm acting alone, and for k=n it is optimized by the human acting alone. 
Surprisingly, we show that for multiple of noise models, it is optimal to set k in [2, n-1] - that is, there are strict benefits to collaborating, even when the human and algorithm have equal accuracy separately. We demonstrate this theoretically for the Mallows model and experimentally for the Random Utilities models of noisy permutations. However, we show this pattern is *reversed* when the human is anchored on the algorithm's presented ordering -  the joint system always has strictly worse performance. We extend these results to the case where the human and algorithm differ in their accuracy levels, showing that there always exist regimes where a more accurate agent would strictly benefit from collaborating with a less accurate one, but these regimes are asymmetric between the human and the algorithm's accuracy.

----

## [1118] A Local-Ascending-Global Learning Strategy for Brain-Computer Interface

**Authors**: *Dongrui Gao, Haokai Zhang, Pengrui Li, Tian Tang, Shihong Liu, Zhihong Zhou, Shaofei Ying, Ye Zhu, Yongqing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28867](https://doi.org/10.1609/aaai.v38i9.28867)

**Abstract**:

Neuroscience research indicates that the interaction among different functional regions of the brain plays a crucial role in driving various cognitive tasks. Existing studies have primarily focused on constructing either local or global functional connectivity maps within the brain, often lacking an adaptive approach to fuse functional brain regions and explore latent relationships between localization during different cognitive tasks. This paper introduces a novel approach called the Local-Ascending-Global Learning Strategy (LAG) to uncover higher-level latent topological patterns among functional brain regions. The strategy initiates from the local connectivity of individual brain functional regions and develops a K-Level Self-Adaptive Ascending Network (SALK) to dynamically capture strong connectivity patterns among brain regions during different cognitive tasks. Through the step-by-step fusion of brain regions, this approach captures higher-level latent patterns, shedding light on the progressively adaptive fusion of various brain functional regions under different cognitive tasks. Notably, this study represents the first exploration of higher-level latent patterns through progressively adaptive fusion of diverse brain functional regions under different cognitive tasks. The proposed LAG strategy is validated using datasets related to fatigue (SEED-VIG), emotion (SEED-IV), and motor imagery (BCI_C_IV_2a). The results demonstrate the generalizability of LAG, achieving satisfactory outcomes in independent-subject experiments across all three datasets. This suggests that LAG effectively characterizes higher-level latent patterns associated with different cognitive tasks, presenting a novel approach to understanding brain patterns in varying cognitive contexts.

----

## [1119] Working Memory Capacity of ChatGPT: An Empirical Study

**Authors**: *Dongyu Gong, Xingchen Wan, Dingmin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28868](https://doi.org/10.1609/aaai.v38i9.28868)

**Abstract**:

Working memory is a critical aspect of both human intelligence and artificial intelligence, serving as a workspace for the temporary storage and manipulation of information. In this paper, we systematically assess the working memory capacity of ChatGPT, a large language model developed by OpenAI, by examining its performance in verbal and spatial n-back tasks under various conditions. Our experiments reveal that ChatGPT has a working memory capacity limit strikingly similar to that of humans. Furthermore, we investigate the impact of different instruction strategies on ChatGPT's performance and observe that the fundamental patterns of a capacity limit persist. From our empirical findings, we propose that n-back tasks may serve as tools for benchmarking the working memory capacity of large language models and hold potential for informing future efforts aimed at enhancing AI working memory.

----

## [1120] Count What You Want: Exemplar Identification and Few-Shot Counting of Human Actions in the Wild

**Authors**: *Yifeng Huang, Duc Duy Nguyen, Lam Nguyen, Cuong Pham, Minh Hoai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28869](https://doi.org/10.1609/aaai.v38i9.28869)

**Abstract**:

This paper addresses the task of counting human actions of interest using sensor data from wearable devices. We propose a novel exemplar-based framework, allowing users to provide exemplars of the actions they want to count by vocalizing predefined sounds ``one'', ``two'', and ``three''. Our method first localizes temporal positions of these utterances from the audio sequence. These positions serve as the basis for identifying exemplars representing the action class of interest. A similarity map is then computed between the exemplars and the entire sensor data sequence, which is further fed into a density estimation module to generate a sequence of estimated density values. Summing these density values provides the final count. To develop and evaluate our approach, we introduce a diverse and realistic dataset consisting of real-world data from 37 subjects and 50 action categories, encompassing both sensor and audio data. The experiments on this dataset demonstrate the viability of the proposed method in counting instances of actions from new classes and subjects that were not part of the training data. On average, the discrepancy between the predicted count and the ground truth value is 7.47, significantly lower than the errors of the frequency-based and transformer-based methods. Our project, code and dataset can be found at https://github.com/cvlab-stonybrook/ExRAC.

----

## [1121] Learning Optimal Advantage from Preferences and Mistaking It for Reward

**Authors**: *W. Bradley Knox, Stephane Hatgis-Kessell, Sigurdur O. Adalgeirsson, Serena Booth, Anca D. Dragan, Peter Stone, Scott Niekum*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28870](https://doi.org/10.1609/aaai.v38i9.28870)

**Abstract**:

We consider algorithms for learning reward functions from human preferences over pairs of trajectory segments, as used in reinforcement learning from human feedback (RLHF).
Most recent work assumes that human preferences are generated based only upon the reward accrued within those segments, or their partial return. Recent work casts doubt on the validity of this assumption, proposing an alternative preference model based upon regret. We investigate the consequences of assuming preferences are based upon partial return when they actually arise from regret. We argue that the learned function is an approximation of the optimal advantage function, not a reward function. We find that if a specific pitfall is addressed, this incorrect assumption is not particularly harmful, resulting in a highly shaped reward function. Nonetheless, this incorrect usage of the approximation of the optimal advantage function is less desirable than the appropriate and simpler approach of greedy maximization of it. From the perspective of the regret preference model, we also provide a clearer interpretation of fine tuning contemporary large language models with RLHF. This paper overall provides insight regarding why learning under the partial return preference model tends to work so well in practice, despite it conforming poorly to how humans give preferences.

----

## [1122] A Unified Self-Distillation Framework for Multimodal Sentiment Analysis with Uncertain Missing Modalities

**Authors**: *Mingcheng Li, Dingkang Yang, Yuxuan Lei, Shunli Wang, Shuaibing Wang, Liuzhen Su, Kun Yang, Yuzheng Wang, Mingyang Sun, Lihua Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28871](https://doi.org/10.1609/aaai.v38i9.28871)

**Abstract**:

Multimodal Sentiment Analysis (MSA) has attracted widespread research attention recently. Most MSA studies are based on the assumption of modality completeness. However, many inevitable factors in real-world scenarios lead to uncertain missing modalities, which invalidate the fixed multimodal fusion approaches. To this end, we propose a Unified multimodal Missing modality self-Distillation Framework (UMDF) to handle the problem of uncertain missing modalities in MSA. Specifically, a unified self-distillation mechanism in UMDF drives a single network to automatically learn robust inherent representations from the consistent distribution of multimodal data. Moreover, we present a multi-grained crossmodal interaction module to deeply mine the complementary semantics among modalities through coarse- and fine-grained crossmodal attention. Eventually, a dynamic feature integration module is introduced to enhance the beneficial semantics in incomplete modalities while filtering the redundant information therein to obtain a refined and robust multimodal representation. Comprehensive experiments on three datasets demonstrate that our framework significantly improves MSA performance under both uncertain missing-modality and complete-modality testing conditions.

----

## [1123] Decoding AI's Nudge: A Unified Framework to Predict Human Behavior in AI-Assisted Decision Making

**Authors**: *Zhuoyan Li, Zhuoran Lu, Ming Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28872](https://doi.org/10.1609/aaai.v38i9.28872)

**Abstract**:

With the rapid development of AI-based decision aids,  different forms of AI assistance have been increasingly integrated into the human decision making processes.  To best support humans in decision making, it is essential to quantitatively understand how diverse forms of AI assistance influence humans' decision making behavior. To this end, much of the current research focuses on the end-to-end prediction of human behavior using ``black-box'' models, often lacking interpretations of the nuanced ways in which AI assistance impacts the human decision making process. 
Meanwhile, methods that prioritize the interpretability of human behavior predictions are often tailored for one specific form of AI assistance, making adaptations to other forms of assistance difficult.  In this paper, we propose a computational framework that can provide an interpretable characterization of the influence of different forms of AI assistance on decision makers in AI-assisted decision making.  By conceptualizing  AI assistance as the ``nudge'' in human decision making processes, our approach centers around modelling how different forms of AI assistance modify humans' strategy in weighing different information in making their decisions. Evaluations on behavior data collected from real human decision makers 
show that the proposed framework outperforms various baselines in accurately predicting human behavior in AI-assisted decision making. Based on the proposed framework, we further provide insights into how individuals with different cognitive styles are nudged by AI assistance differently.

----

## [1124] GigaHumanDet: Exploring Full-Body Detection on Gigapixel-Level Images

**Authors**: *Chenglong Liu, Haoran Wei, Jinze Yang, Jintao Liu, Wenxi Li, Yuchen Guo, Lu Fang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28873](https://doi.org/10.1609/aaai.v38i9.28873)

**Abstract**:

Performing person detection in super-high-resolution images has been a challenging task. For such a task, modern detectors, which usually encode a box using center and width/height, struggle with accuracy due to two factors: 1) Human characteristic: people come in various postures and the center with high freedom is difficult to capture robust visual pattern; 2) Image characteristic: due to vast scale diversity of input (gigapixel-level), distance regression (for width and height) is hard to pinpoint, especially for a person, with substantial scale, who is near the camera. To address these challenges, we propose GigaHumanDet, an innovative solution aimed at further enhancing detection accuracy for gigapixel-level images. GigaHumanDet employs the corner modeling method to avoid the potential issues of a high degree of freedom in center pinpointing. To better distinguish similar-looking persons and enforce instance consistency of corner pairs, an instance-guided learning approach is designed to capture discriminative individual semantics. Further, we devise reliable shape-aware bodyness equipped with a multi-precision strategy as the human corner matching guidance to be appropriately adapted to the single-view large scene. Experimental results on PANDA and STCrowd datasets show the superiority and strong applicability of our design. Notably, our model achieves 82.4% in term of AP, outperforming current state-of-the-arts by more than 10%.

----

## [1125] Hypergraph-Guided Disentangled Spectrum Transformer Networks for Near-Infrared Facial Expression Recognition

**Authors**: *Bingjun Luo, Haowen Wang, Jinpeng Wang, Junjie Zhu, Xibin Zhao, Yue Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28874](https://doi.org/10.1609/aaai.v38i9.28874)

**Abstract**:

With the strong robusticity on illumination variations, near-infrared (NIR) can be an effective and essential complement to visible (VIS) facial expression recognition in low lighting or complete darkness conditions. However, facial expression recognition (FER) from NIR images presents a more challenging problem than traditional FER due to the limitations imposed by the data scale and the difficulty of extracting discriminative features from incomplete visible lighting contents. In this paper, we give the first attempt at deep NIR facial expression recognition and propose a novel method called near-infrared facial expression transformer (NFER-Former). Specifically, to make full use of the abundant label information in the field of VIS, we introduce a Self-Attention Orthogonal Decomposition mechanism that disentangles the expression information and spectrum information from the input image, so that the expression features can be extracted without the interference of spectrum variation. We also propose a Hypergraph-Guided Feature Embedding method that models some key facial behaviors and learns the structure of the complex correlations between them, thereby alleviating the interference of inter-class similarity. Additionally, we construct a large NIR-VIS Facial Expression dataset that includes 360 subjects to better validate the efficiency of NFER-Former. Extensive experiments and ablation studies show that NFER-Former significantly improves the performance of NIR FER and achieves state-of-the-art results on the only two available NIR FER datasets, Oulu-CASIA and Large-HFE.

----

## [1126] Goal Alignment: Re-analyzing Value Alignment Problems Using Human-Aware AI

**Authors**: *Malek Mechergui, Sarath Sreedharan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28875](https://doi.org/10.1609/aaai.v38i9.28875)

**Abstract**:

While the question of misspecified objectives has gotten much attention in recent years, most works in this area primarily focus on the challenges related to the complexity of the objective specification mechanism (for example, the use of reward functions). However, the complexity of the objective specification mechanism is just one of many reasons why the user may have misspecified their objective. A foundational cause for misspecification that is being overlooked by these works is the inherent asymmetry in human expectations about the agent's behavior and the behavior generated by the agent for the specified objective. To address this, we propose a novel formulation for the objective misspecification problem that builds on the human-aware planning literature, which was originally introduced to support explanation and explicable behavioral generation. Additionally, we propose a first-of-its-kind interactive algorithm that is capable of using information generated under incorrect beliefs about the agent to determine the true underlying goal of the user.

----

## [1127] Efficient Online Crowdsourcing with Complex Annotations

**Authors**: *Reshef Meir, Viet-An Nguyen, Xu Chen, Jagdish Ramakrishnan, Udi Weinsberg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28876](https://doi.org/10.1609/aaai.v38i9.28876)

**Abstract**:

Crowdsourcing platforms use various truth discovery algorithms to aggregate annotations from multiple labelers. In an online setting, however, the main challenge is to decide whether to ask for more annotations for each item to efficiently trade off cost (i.e., the number of annotations) for quality of the aggregated annotations. In this paper, we propose a novel approach for general complex annotation (such as bounding boxes and taxonomy paths), that works in an online crowdsourcing setting. We prove that the expected average similarity of a labeler is linear in their accuracy conditional on the reported label. This enables us to infer reported label accuracy in a broad range of scenarios. We conduct extensive evaluations on real-world crowdsourcing data from Meta and show the effectiveness of our proposed online algorithms in improving the cost-quality trade-off.

----

## [1128] Can You Rely on Synthetic Labellers in Preference-Based Reinforcement Learning? It's Complicated

**Authors**: *Katherine Metcalf, Miguel Sarabia, Masha Fedzechkina, Barry-John Theobald*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28877](https://doi.org/10.1609/aaai.v38i9.28877)

**Abstract**:

Preference-based Reinforcement Learning (PbRL) enables non-experts to train Reinforcement Learning models using preference feedback. However, the effort required to collect preference labels from real humans means that PbRL research primarily relies on synthetic labellers. We validate the most common synthetic labelling strategy by comparing against labels collected from a crowd of humans on three Deep Mind Control (DMC) suite tasks: stand, walk, and run. We find that: (1) the synthetic labels are a good proxy for real humans under some circumstances, (2) strong preference label agreement between human and synthetic labels is not necessary for similar policy performance, (3) policy performance is higher at the start of training from human feedback and is higher at the end of training from synthetic feedback, and (4) training on only examples with high levels of inter-annotator agreement does not meaningfully improve policy performance. Our results justify the use of synthetic labellers to develop and ablate PbRL methods, and provide insight into how human labelling changes over the course of policy training.

----

## [1129] When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming

**Authors**: *Hussein Mozannar, Gagan Bansal, Adam Fourney, Eric Horvitz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28878](https://doi.org/10.1609/aaai.v38i9.28878)

**Abstract**:

AI powered code-recommendation systems, such as Copilot and CodeWhisperer, provide code suggestions inside a programmer's environment (e.g., an IDE) with the aim of improving productivity. We pursue mechanisms for leveraging signals about programmers' acceptance and rejection of code suggestions to guide recommendations. We harness data drawn from interactions with GitHub Copilot, a system used by millions of programmers, to develop interventions that can save time for programmers. We introduce a utility-theoretic framework to drive decisions about suggestions to display versus withhold. The approach, conditional suggestion display from human feedback (CDHF), relies on a cascade of models that provide the likelihood that recommended code will be accepted. These likelihoods are used to selectively hide suggestions, reducing both latency and programmer verification time. Using data from 535 programmers, we perform a retrospective evaluation of CDHF and show that we can avoid displaying a significant fraction of suggestions that would have been rejected. We further demonstrate the importance of incorporating the programmer's latent unobserved state in decisions about when to display suggestions through an ablation study. Finally, we showcase how using suggestion acceptance as a reward signal for guiding the display of suggestions can lead to suggestions of reduced quality, indicating an unexpected pitfall.

----

## [1130] Improving Transferability for Cross-Domain Trajectory Prediction via Neural Stochastic Differential Equation

**Authors**: *Daehee Park, Jaewoo Jeong, Kuk-Jin Yoon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28879](https://doi.org/10.1609/aaai.v38i9.28879)

**Abstract**:

Multi-agent trajectory prediction is crucial for various practical applications, spurring the construction of many large-scale trajectory datasets, including vehicles and pedestrians. 
However, discrepancies exist among datasets due to external factors and data acquisition strategies. External factors include geographical differences and driving styles, while data acquisition strategies include data acquisition rate, history/prediction length, and detector/tracker error. 
Consequently, the proficient performance of models trained on large-scale datasets has limited transferability on other small-size datasets, bounding the utilization of existing large-scale datasets.
To address this limitation, we propose a method based on continuous and stochastic representations of Neural Stochastic Differential Equations (NSDE) for alleviating discrepancies due to data acquisition strategy. 
We utilize the benefits of continuous representation for handling arbitrary time steps and the use of stochastic representation for handling detector/tracker errors. 
Additionally, we propose a dataset-specific diffusion network and its training framework to handle dataset-specific detection/tracking errors. 
The effectiveness of our method is validated against state-of-the-art trajectory prediction models on the popular benchmark datasets: nuScenes, Argoverse, Lyft, INTERACTION, and Waymo Open Motion Dataset (WOMD). 
Improvement in performance gain on various source and target dataset configurations shows the generalized competence of our approach in addressing cross-dataset discrepancies.

----

## [1131] Exploring Domain Incremental Video Highlights Detection with the LiveFood Benchmark

**Authors**: *Sen Pei, Shixiong Xu, Xiaojie Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28880](https://doi.org/10.1609/aaai.v38i9.28880)

**Abstract**:

Video highlights detection (VHD) is an active research field in computer vision, aiming to locate the most user-appealing clips given raw video inputs. However, most VHD methods are based on the closed world assumption, i.e., a fixed number of highlight categories is defined in advance and all training data are available beforehand. Consequently, existing methods have poor scalability with respect to increasing highlight domains and training data. To address above issues, we propose a novel video highlights detection method named Global Prototype Encoding (GPE) to learn incrementally for adapting to new domains via parameterized prototypes. To facilitate this new research direction, we collect a finely annotated dataset termed LiveFood, including over 5,100 live gourmet videos that consist of four domains: ingredients, cooking, presentation, and eating. To the best of our knowledge, this is the first work to explore video highlights detection in the incremental learning setting, opening up new land to apply VHD for practical scenarios where both the concerned highlight domains and training data increase over time. We demonstrate the effectiveness of GPE through extensive experiments. Notably,  GPE surpasses popular domain incremental learning methods on LiveFood, achieving significant mAP improvements on all domains. Concerning the classic datasets, GPE also yields comparable performance as previous arts. The code is available at: https://github.com/ForeverPs/IncrementalVHD_GPE.

----

## [1132] Sample-Constrained Black Box Optimization for Audio Personalization

**Authors**: *Rajalaxmi Rajagopalan, Yu-Lin Wei, Romit Roy Choudhury*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28881](https://doi.org/10.1609/aaai.v38i9.28881)

**Abstract**:

We consider the problem of personalizing audio to maximize user experience. Briefly, we aim to find a filter h*, which applied to any music or speech, will maximize the user’s satisfaction. This is a black-box optimization problem since the user’s satisfaction function is unknown. Substantive work has been done on this topic where the key idea is to play audio samples to the user, each shaped by a different filter hi, and query the user for their satisfaction scores f(hi). A family of “surrogate” functions is then designed to fit these scores and the optimization method gradually refines these functions to arrive at the filter ˆh* that maximizes satisfaction. 

In certain applications, we observe that a second type of querying is possible where users can tell us the individual elements h*[j] of the optimal filter h*. Consider an analogy from cooking where the goal is to cook a recipe that maximizes user satisfaction. A user can be asked to score various cooked recipes (e.g., tofu fried rice) or to score individual ingredients (say, salt, sugar, rice, chicken, etc.). Given a budget of B queries, where a query can be of either type, our goal is to find the recipe that will maximize this user’s satisfaction. 

Our proposal builds on Sparse Gaussian Process Regression (GPR) and shows how a hybrid approach can outperform any one type of querying. Our results are validated through simulations and real world experiments, where volunteers gave feedback on music/speech audio and were able to achieve high satisfaction levels. We believe this idea of hybrid querying opens new problems in black-box optimization and solutions can benefit other applications beyond audio personalization.

----

## [1133] Intelligent Calibration for Bias Reduction in Sentiment Corpora Annotation Process

**Authors**: *Idan Toker, David Sarne, Jonathan Schler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28882](https://doi.org/10.1609/aaai.v38i9.28882)

**Abstract**:

This paper focuses in the inherent anchoring bias present in sequential reviews-sentiment corpora annotation processes. It proposes employing a limited subset of meticulously chosen reviews at the outset of the process, as a means of calibration, effectively mitigating the phenomenon. Through extensive experimentation we validate the phenomenon of sentiment bias in the annotation process and show that its magnitude can be influenced by pre-calibration. Furthermore, we show that the choice of the calibration set matters, hence the need for effective guidelines for choosing the reviews to be included in it. A comparison of annotators performance with the proposed calibration to annotation processes that do not use calibration or use a randomly-picked calibration set, reveals that indeed the calibration set picked is highly effective---it manages to substantially reduce the average absolute error compared to the other cases. Furthermore, the proposed selection guidelines are found to be highly robust in picking an effective calibration set also for domains different than the one based on which these rules were extracted.

----

## [1134] TransGOP: Transformer-Based Gaze Object Prediction

**Authors**: *Binglu Wang, Chenxi Guo, Yang Jin, Haisheng Xia, Nian Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28883](https://doi.org/10.1609/aaai.v38i9.28883)

**Abstract**:

Gaze object prediction aims to predict the location and category of the object that is watched by a human. Previous gaze object prediction works use CNN-based object detectors to predict the object's location. However, we find that Transformer-based object detectors can predict more accurate object location for dense objects in retail scenarios. Moreover, the long-distance modeling capability of the Transformer can help to build relationships between the human head and the gaze object, which is important for the GOP task. To this end,  this paper introduces Transformer into the fields of gaze object prediction and proposes an end-to-end Transformer-based gaze object prediction method named TransGOP. Specifically, TransGOP uses an off-the-shelf Transformer-based object detector to detect the location of objects and designs a Transformer-based gaze autoencoder in the gaze regressor to establish long-distance gaze relationships. Moreover, to improve gaze heatmap regression, we propose an object-to-gaze cross-attention mechanism to let the queries of the gaze autoencoder learn the global-memory position knowledge from the object detector. Finally, to make the whole framework end-to-end trained, we propose a Gaze Box loss to jointly optimize the object detector and gaze regressor by enhancing the gaze heatmap energy in the box of the gaze object. Extensive experiments on the GOO-Synth and GOO-Real datasets demonstrate that our TransGOP achieves state-of-the-art performance on all tracks, i.e., object detection, gaze estimation, and gaze object prediction. Our code will be available at https://github.com/chenxi-Guo/TransGOP.git.

----

## [1135] Visual Redundancy Removal for Composite Images: A Benchmark Dataset and a Multi-Visual-Effects Driven Incremental Method

**Authors**: *Miaohui Wang, Rong Zhang, Lirong Huang, Yanshan Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28884](https://doi.org/10.1609/aaai.v38i9.28884)

**Abstract**:

Composite images (CIs) typically combine various elements from different scenes, views, and styles, which are a very important information carrier in the era of mixed media such as virtual reality, mixed reality, metaverse, etc. However, the complexity of CI content presents a significant challenge for subsequent visual perception modeling and compression. In addition, the lack of benchmark CI databases also hinders the use of recent advanced data-driven methods. To address these challenges, we first establish one of the earliest visual redundancy prediction (VRP) databases for CIs. Moreover, we propose a multi-visual effect (MVE)-driven incremental learning method that combines the strengths of hand-crafted and data-driven approaches to achieve more accurate VRP modeling. Specifically, we design special incremental rules to learn the visual knowledge flow of MVE. To effectively capture the associated features of MVE, we further develop a three-stage incremental learning approach for VRP based on an encoder-decoder network. Extensive experimental results validate the superiority of the proposed method in terms of subjective, objective, and compression experiments.

----

## [1136] TexFit: Text-Driven Fashion Image Editing with Diffusion Models

**Authors**: *Tongxin Wang, Mang Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28885](https://doi.org/10.1609/aaai.v38i9.28885)

**Abstract**:

Fashion image editing aims to edit an input image to obtain richer or distinct visual clothing matching effects. Existing global fashion image editing methods are difficult to achieve rich outfit combination effects while local fashion image editing is more in line with the needs of diverse and personalized outfit matching. The local editing techniques typically depend on text and auxiliary modalities (e.g., human poses, human keypoints, garment sketches, etc.) for image manipulation, where the auxiliary modalities essentially assist in locating the editing region. Since these auxiliary modalities usually involve additional efforts in practical application scenarios, text-driven fashion image editing shows high flexibility. In this paper, we propose TexFit, a Text-driven Fashion image Editing method using diffusion models, which performs the local image editing only with the easily accessible text. Our approach employs a text-based editing region location module to predict precise editing region in the fashion image. Then, we take the predicted region as the generation condition of diffusion models together with the text prompt to achieve precise local editing of fashion images while keeping the rest part intact. In addition, previous fashion datasets usually focus on global description, lacking local descriptive information that can guide the precise local editing. Therefore, we develop a new DFMM-Spotlight dataset by using region extraction and attribute combination strategies. It focuses locally on clothes and accessories, enabling local editing with text input. Experimental results on the DFMM-Spotlight dataset demonstrate the effectiveness of our model. Code and Datasets are available at https://texfit.github.io/.

----

## [1137] Rating-Based Reinforcement Learning

**Authors**: *Devin White, Mingkang Wu, Ellen R. Novoseller, Vernon J. Lawhern, Nicholas R. Waytowich, Yongcan Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28886](https://doi.org/10.1609/aaai.v38i9.28886)

**Abstract**:

This paper develops a novel rating-based reinforcement learning approach that uses human ratings to obtain human guidance in reinforcement learning. Different from the existing preference-based and ranking-based reinforcement learning paradigms, based on human relative preferences over sample pairs, the proposed rating-based reinforcement learning approach is based on human evaluation of individual trajectories without relative comparisons between sample pairs. The rating-based reinforcement learning approach builds on a new prediction model for human ratings and a novel multi-class loss function. We conduct several experimental studies based on synthetic ratings and real human ratings to evaluate the effectiveness and benefits of the new rating-based reinforcement learning approach.

----

## [1138] MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug-Drug Interaction Prediction

**Authors**: *Di Wu, Wu Sun, Yi He, Zhong Chen, Xin Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28887](https://doi.org/10.1609/aaai.v38i9.28887)

**Abstract**:

Taking incompatible multiple drugs together may cause adverse interactions and side effects on the body. Accurate prediction of drug-drug interaction (DDI) events is essential for avoiding this issue. Recently, various artificial intelligence-based approaches have been proposed for predicting DDI events. However, DDI events are associated with complex relationships and mechanisms among drugs, targets, enzymes, transporters, molecular structures, etc. Existing approaches either partially or loosely consider these relationships and mechanisms by a non-end-to-end learning framework, resulting in sub-optimal feature extractions and fusions for prediction. Different from them, this paper proposes a Multimodal Knowledge Graph Fused End-to-end Neural Network (MKGFENN) that consists of two main parts: multimodal knowledge graph (MKG) and fused end-to-end neural network (FENN). First, MKG is constructed by comprehensively exploiting DDI events-associated relationships and mechanisms from four knowledge graphs of drugs-chemical entities, drug-substructures, drugs-drugs, and molecular structures. Correspondingly, a four channels graph neural network is designed to extract high-order and semantic features from MKG. Second, FENN designs a multi-layer perceptron to fuse the extracted features by end-to-end learning. With such designs, the feature extractions and fusions of DDI events are guaranteed to be comprehensive and optimal for prediction. Through extensive experiments on real drug datasets, we demonstrate that MKG-FENN exhibits high accuracy and significantly outperforms state-of-the-art models in predicting DDI events. The source code and supplementary file of this article are available on: https://github.com/wudi1989/MKG-FENN.

----

## [1139] Spatial-Related Sensors Matters: 3D Human Motion Reconstruction Assisted with Textual Semantics

**Authors**: *Xueyuan Yang, Chao Yao, Xiaojuan Ban*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28888](https://doi.org/10.1609/aaai.v38i9.28888)

**Abstract**:

Leveraging wearable devices for motion reconstruction has emerged as an economical and viable technique. Certain methodologies employ sparse Inertial Measurement Units (IMUs) on the human body and harness data-driven strategies to model human poses. However, the reconstruction of motion based solely on sparse IMU data is inherently fraught with ambiguity, a consequence of numerous identical IMU readings corresponding to different poses. In this paper, we explore the spatial importance of sparse sensors, supervised by text that describes specific actions. Specifically, uncertainty is introduced to derive weighted features for each IMU. We also design a Hierarchical Temporal Transformer (HTT) and apply contrastive learning to achieve precise temporal and feature alignment of sensor data with textual semantics. Experimental results demonstrate our proposed approach achieves significant improvements in multiple metrics compared to existing methods. Notably, with textual supervision, our method not only differentiates between ambiguous actions such as sitting and standing but also produces more precise and natural motion.

----

## [1140] Scalable Motion Style Transfer with Constrained Diffusion Generation

**Authors**: *Wenjie Yin, Yi Yu, Hang Yin, Danica Kragic, Mårten Björkman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28889](https://doi.org/10.1609/aaai.v38i9.28889)

**Abstract**:

Current training of motion style transfer systems relies on consistency losses across style domains to preserve contents, hindering its scalable application to a large number of domains and private data. Recent image transfer works show the potential of independent training on each domain by leveraging implicit bridging between diffusion models, with the content preservation, however, limited to simple data patterns. We address this by imposing biased sampling in backward diffusion while maintaining the domain independence in the training stage. We construct the bias from the source domain keyframes and apply them as the gradient of content constraints, yielding a framework with keyframe manifold constraint gradients (KMCGs). Our validation demonstrates the success of training separate models to transfer between as many as ten dance motion styles. Comprehensive experiments find a significant improvement in preserving motion contents in comparison to baseline and ablative diffusion-based style transfer models. In addition, we perform a human study for a subjective assessment of the quality of generated dance motions. The results validate the competitiveness of KMCGs.

----

## [1141] 'Why Didn't You Allocate This Task to Them?' Negotiation-Aware Task Allocation and Contrastive Explanation Generation

**Authors**: *Zahra Zahedi, Sailik Sengupta, Subbarao Kambhampati*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28890](https://doi.org/10.1609/aaai.v38i9.28890)

**Abstract**:

In this work, we design an Artificially Intelligent Task Allocator (AITA) that proposes a task allocation for a team of humans. A key property of this allocation is that when an agent with imperfect knowledge (about their teammate's costs and/or the team's performance metric) contests the allocation with a counterfactual, a contrastive explanation can always be provided to showcase why the proposed allocation is better than the proposed counterfactual. For this, we consider a negotiation process that produces a negotiation-aware task allocation and, when contested, leverages a negotiation tree to provide a contrastive explanation. With human subject studies, we show that the proposed allocation indeed appears fair to a majority of participants and, when not, the explanations generated are judged as convincing and easy to comprehend.

----

## [1142] Beyond Mimicking Under-Represented Emotions: Deep Data Augmentation with Emotional Subspace Constraints for EEG-Based Emotion Recognition

**Authors**: *Zhi Zhang, Shenghua Zhong, Yan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28891](https://doi.org/10.1609/aaai.v38i9.28891)

**Abstract**:

In recent years, using Electroencephalography (EEG) to recognize emotions has garnered considerable attention. Despite advancements, limited EEG data restricts its potential. Thus, Generative Adversarial Networks (GANs) are proposed to mimic the observed distributions and generate EEG data. However, for imbalanced datasets, GANs struggle to produce reliable augmentations for under-represented minority emotions by merely mimicking them. Thus, we introduce Emotional Subspace Constrained Generative Adversarial Networks (ESC-GAN) as an alternative to existing frameworks. We first propose the EEG editing paradigm, editing reference EEG signals from well-represented to under-represented emotional subspaces. Then, we introduce diversity-aware and
boundary-aware losses to constrain the augmented subspace. Here, the diversity-aware loss encourages a diverse emotional subspace by enlarging the sample difference, while boundary-aware loss constrains the augmented subspace near the decision boundary where recognition models can be vulnerable. Experiments show ESC-GAN boosts emotion recognition performance on benchmark datasets, DEAP, AMIGOS, and SEED, while protecting against potential adversarial attacks. Finally, the proposed method opens new avenues for editing EEG signals under emotional subspace constraints, facilitating unbiased and secure EEG data augmentation.

----

## [1143] MetaRLEC: Meta-Reinforcement Learning for Discovery of Brain Effective Connectivity

**Authors**: *Zuozhen Zhang, Junzhong Ji, Jinduo Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28892](https://doi.org/10.1609/aaai.v38i9.28892)

**Abstract**:

In recent years, the discovery of brain effective connectivity (EC) networks through computational analysis of functional magnetic resonance imaging (fMRI) data has gained prominence in neuroscience and neuroimaging. However, owing to the influence of diverse factors during data collection and processing, fMRI data typically exhibits high noise and limited sample characteristics, consequently leading to suboptimal performance of current methods. In this paper, we propose a novel brain effective connectivity discovery method based on meta-reinforcement learning, called MetaRLEC. The method mainly consists of three modules: actor, critic, and meta-critic. MetaRLEC first employs an encoder-decoder framework: the encoder utilizing a Transformer, converts noisy fMRI data into a state embedding; the decoder employing bidirectional LSTM, discovers brain region dependencies from the state and generates actions (EC networks). Then a critic network evaluates these actions, incentivizing the actor to learn higher-reward actions amidst the high-noise setting. Finally, a meta-critic framework facilitates online learning of historical state-action pairs, integrating an action-value neural network and supplementary training losses to enhance the model's adaptability to small-sample fMRI data. We conduct comprehensive experiments on both simulated and real-world data to demonstrate the efficacy of our proposed method.

----

## [1144] DanceMVP: Self-Supervised Learning for Multi-Task Primitive-Based Dance Performance Assessment via Transformer Text Prompting

**Authors**: *Yun Zhong, Yiannis Demiris*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28893](https://doi.org/10.1609/aaai.v38i9.28893)

**Abstract**:

Dance is generally considered to be complex for most people as it requires coordination of numerous body motions and accurate responses to the musical content and rhythm. Studies on automatic dance performance assessment could help people improve their sensorimotor skills and promote research in many fields, including human motion analysis and motion generation. Recent papers on dance performance assessment usually evaluate simple dance motions with a single task - estimating final performance scores. In this paper, we propose DanceMVP: multi-task dance performance assessment via text prompting that solves three related tasks - (i) dance vocabulary recognition, (ii) dance performance scoring and (iii) dance rhythm evaluation. In the pre-training phase, we contrastively learn the primitive-based features of complex dance motion and music using the InfoNCE loss. For the downstream task, we propose a transformer-based text prompter to perform multi-task evaluations for the three proposed assessment tasks. Also, we build a multimodal dance-music dataset named ImperialDance. The novelty of our ImperialDance is that it contains dance motions for diverse expertise levels and a significant amount of repeating dance sequences for the same choreography to keep track of the dance performance progression. Qualitative results show that our pre-trained feature representation could cluster dance pieces for different dance genres, choreographies, expertise levels and primitives, which generalizes well on both ours and other dance-music datasets. The downstream experiments demonstrate the robustness and improvement of our method over several ablations and baselines across all three tasks, as well as monitoring the users' dance level progression.

----

## [1145] Optimal Makespan in a Minute Timespan! A Scalable Multi-Robot Goal Assignment Algorithm for Minimizing Mission Time

**Authors**: *Aakash, Indranil Saha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28894](https://doi.org/10.1609/aaai.v38i9.28894)

**Abstract**:

We study a variant of the multi-robot goal assignment problem where a unique goal to each robot needs to be assigned while minimizing the largest cost of movement among the robots, called makespan. A significant step in solving this problem is to find the cost associated with the robot-goal pairs, which requires solving a complex path planning problem.  We present OM, a scalable optimal algorithm that solves the multi-robot goal assignment problem by computing the paths for a significantly less number of robot-goal pairs compared to the state-of-the-art algorithms, leading to a computationally superior mechanism to solve the problem. We extensively evaluate our algorithm for hundreds of robots on randomly generated and standard workspaces. Our experimental results demonstrate that the proposed algorithm achieves a noticeable speedup over two state-of-the-art baseline algorithms.

----

## [1146] On Computing Makespan-Optimal Solutions for Generalized Sliding-Tile Puzzles

**Authors**: *Marcus Gozon, Jingjin Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28895](https://doi.org/10.1609/aaai.v38i9.28895)

**Abstract**:

In the 15-puzzle game, 15 labeled square tiles are reconfigured on a 4 × 4 board through an escort, wherein each (time) step, a single tile neighboring it may slide into it, leaving the space previously occupied by the tile as the new escort. We study a generalized sliding-tile puzzle (GSTP) in which (1) there are 1+ escorts and (2) multiple tiles can move synchronously in a single time step. Compared with popular discrete multi-agent/robot motion models, GSTP provides a more accurate model for a broad array of high-utility applications, including warehouse automation and autonomous garage parking, but is less studied due to the more involved tile interactions. In this work, we analyze optimal GSTP solution structures, establishing that computing makespan optimal solutions for GSTP is NP-complete and developing polynomial time algorithms yielding makespans approximating the minimum with expected/high probability constant factors, assuming randomized start and goal configurations.

----

## [1147] Interactive Visual Task Learning for Robots

**Authors**: *Weiwei Gu, Anant Sah, Nakul Gopalan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28896](https://doi.org/10.1609/aaai.v38i9.28896)

**Abstract**:

We present a framework for robots to learn novel visual concepts and tasks via in-situ linguistic interactions with human users. Previous approaches have either used large pre-trained visual models to infer novel objects zero-shot, or added novel concepts along with their attributes and representations to a concept hierarchy. We extend the approaches that focus on learning visual concept hierarchies by enabling them to learn novel concepts and solve unseen robotics tasks with them. To enable a visual concept learner to solve robotics tasks one-shot, we developed two distinct techniques. Firstly, we propose a novel approach, Hi-Viscont(HIerarchical VISual CONcept learner for Task), which augments information of a novel concept to its parent nodes within a concept hierarchy. This information propagation allows all concepts in a hierarchy to update as novel concepts are taught in a continual learning setting. Secondly, we represent a visual task as a scene graph with language annotations, allowing us to create novel permutations of a demonstrated task zero-shot in-situ. We present two sets of results. Firstly, we compare Hi-Viscont with the baseline model (FALCON) on visual question answering(VQA) in three domains. While being comparable to the baseline model on leaf level concepts, Hi-Viscont achieves an improvement of over 9% on non-leaf concepts on average. Secondly, we conduct a human-subjects experiment where users teach our robot visual tasks in-situ. We compare our model’s performance against the baseline FALCON model. Our framework achieves 33% improvements in success rate metric, and 19% improvements in the object level accuracy compared to the baseline model. With both of these results we demonstrate the ability of our model to learn tasks and concepts in a continual learning setting on the robot.

----

## [1148] DexFuncGrasp: A Robotic Dexterous Functional Grasp Dataset Constructed from a Cost-Effective Real-Simulation Annotation System

**Authors**: *Jinglue Hang, Xiangbo Lin, Tianqiang Zhu, Xuanheng Li, Rina Wu, Xiaohong Ma, Yi Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28897](https://doi.org/10.1609/aaai.v38i9.28897)

**Abstract**:

Robot grasp dataset is the basis of designing the robot's grasp generation model. Compared with the building grasp dataset for Low-DOF grippers, it is harder for High-DOF dexterous robot hand. Most current datasets meet the needs of generating stable grasps, but they are not suitable for dexterous hands to complete human-like functional grasp, such as grasp the handle of a cup or pressing the button of a flashlight, so as to enable robots to complete subsequent functional manipulation action autonomously, and there is no dataset with functional grasp pose annotations at present. This paper develops a unique Cost-Effective Real-Simulation Annotation System by leveraging natural hand's actions. The system is able to capture a functional grasp of a dexterous hand in a simulated environment assisted by human demonstration in real world. By using this system, dexterous grasp data can be collected efficiently as well as cost-effective. Finally, we construct the first dexterous functional grasp dataset with rich pose annotations. A Functional Grasp Synthesis Model is also provided to validate the effectiveness of the proposed system and dataset. Our project page is: https://hjlllll.github.io/DFG/.

----

## [1149] LINGO-Space: Language-Conditioned Incremental Grounding for Space

**Authors**: *Dohyun Kim, Nayoung Oh, Deokmin Hwang, Daehyung Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28898](https://doi.org/10.1609/aaai.v38i9.28898)

**Abstract**:

We aim to solve the problem of spatially localizing composite instructions referring to space: space grounding. Compared to current instance grounding, space grounding is challenging due to the ill-posedness of identifying locations referred to by discrete expressions and the compositional ambiguity of referring expressions. Therefore, we propose a novel probabilistic space-grounding methodology (LINGO-Space) that accurately identifies a probabilistic distribution of space being referred to and incrementally updates it, given subsequent referring expressions leveraging configurable polar distributions. Our evaluations show that the estimation using polar distributions enables a robot to ground locations successfully through 20 table-top manipulation benchmark tests. We also show that updating the distribution helps the grounding method accurately narrow the referring space. We finally demonstrate the robustness of the space grounding with simulated manipulation and real quadruped robot navigation tasks. Code and videos are available at https://lingo-space.github.io.

----

## [1150] CTO-SLAM: Contour Tracking for Object-Level Robust 4D SLAM

**Authors**: *Xiaohan Li, Dong Liu, Jun Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28899](https://doi.org/10.1609/aaai.v38i9.28899)

**Abstract**:

The demand for 4D ( 3D+time ) SLAM system is increasingly urgent, especially for decision-making and scene understanding. However, most of the existing simultaneous localization and mapping ( SLAM ) systems primarily assume static environments. They fail to represent dynamic scenarios due to the challenge of establishing robust long-term spatiotemporal associations in dynamic object tracking. We address this limitation and propose CTO-SLAM, a monocular and RGB-D object-level 4D SLAM system to track moving objects and estimate their motion simultaneously. In this paper, we propose contour tracking, which introduces contour features to enhance the keypoint representation of dynamic objects and coupled with pixel tracking to achieve long-term robust object tracking. Based on contour tracking, we propose a novel sampling-based object pose initialization algorithm and the following adapted bundle adjustment ( BA ) optimization algorithm to estimate dynamic object poses with high accuracy. The CTO-SLAM system is verified on both KITTI and VKITTI datasets. The experimental results demonstrate that our system effectively addresses cumulative errors in long-term spatiotemporal association and hence obtains substantial improvements over the state-of-the-art systems. The source code is available at https://github.com/realXiaohan/CTO-SLAM.

----

## [1151] BAT: Behavior-Aware Human-Like Trajectory Prediction for Autonomous Driving

**Authors**: *Haicheng Liao, Zhenning Li, Huanming Shen, Wenxuan Zeng, Dongping Liao, Guofa Li, Chengzhong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28900](https://doi.org/10.1609/aaai.v38i9.28900)

**Abstract**:

The ability to accurately predict the trajectory of surrounding vehicles is a critical hurdle to overcome on the journey to fully autonomous vehicles. To address this challenge, we pioneer a novel behavior-aware trajectory prediction model (BAT) that incorporates insights and findings from traffic psychology, human behavior, and decision-making. Our model consists of behavior-aware, interaction-aware, priority-aware, and position-aware modules that perceive and understand the underlying interactions and account for uncertainty and variability in prediction, enabling higher-level learning and flexibility without rigid categorization of driving behavior. Importantly, this approach eliminates the need for manual labeling in the training process and addresses the challenges of non-continuous behavior labeling and the selection of appropriate time windows.
We evaluate BAT's performance across the Next Generation Simulation (NGSIM), Highway Drone (HighD), Roundabout Drone (RounD), and Macao Connected Autonomous Driving (MoCAD) datasets, showcasing its superiority over prevailing state-of-the-art (SOTA) benchmarks in terms of prediction accuracy and efficiency. Remarkably, even when trained on reduced portions of the training data (25%), our model outperforms most of the baselines, demonstrating its robustness and efficiency in predicting vehicle trajectories, and the potential to reduce the amount of data required to train autonomous vehicles, especially in corner cases. In conclusion, the behavior-aware model represents a significant advancement in the development of autonomous vehicles capable of predicting trajectories with the same level of proficiency as human drivers. The project page is available on our GitHub.

----

## [1152] Deep Homography Estimation for Visual Place Recognition

**Authors**: *Feng Lu, Shuting Dong, Lijun Zhang, Bingxi Liu, Xiangyuan Lan, Dongmei Jiang, Chun Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28901](https://doi.org/10.1609/aaai.v38i9.28901)

**Abstract**:

Visual place recognition (VPR) is a fundamental task for many applications such as robot localization and augmented reality. Recently, the hierarchical VPR methods have received considerable attention due to the trade-off between accuracy and efficiency. They usually first use global features to retrieve the candidate images, then verify the spatial consistency of matched local features for re-ranking. However, the latter typically relies on the RANSAC algorithm for fitting homography, which is time-consuming and non-differentiable. This makes existing methods compromise to train the network only in global feature extraction. Here, we propose a transformer-based deep homography estimation (DHE) network that takes the dense feature map extracted by a backbone network as input and fits homography for fast and learnable geometric verification. Moreover, we design a re-projection error of inliers loss to train the DHE network without additional homography labels, which can also be jointly trained with the backbone network to help it extract the features that are more suitable for local matching. Extensive experiments on benchmark datasets show that our method can outperform several state-of-the-art methods. And it is more than one order of magnitude faster than the mainstream hierarchical VPR methods using RANSAC. The code is released at https://github.com/Lu-Feng/DHE-VPR.

----

## [1153] Task Planning for Object Rearrangement in Multi-Room Environments

**Authors**: *Karan Mirakhor, Sourav Ghosh, Dipanjan Das, Brojeshwar Bhowmick*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28902](https://doi.org/10.1609/aaai.v38i9.28902)

**Abstract**:

Object rearrangement in a multi-room setup should produce a reasonable plan that reduces the agent's overall travel and the number of steps. Recent state-of-the-art methods fail to produce such plans because they rely on explicit exploration for discovering unseen objects due to partial observability and a heuristic planner to sequence the actions for rearrangement. This paper proposes a novel task planner to efficiently plan a sequence of actions to discover unseen objects and rearrange misplaced objects within an untidy house to achieve a desired tidy state. The proposed method introduces several innovative techniques, including (i) a method for discovering unseen objects using commonsense knowledge from large language models, (ii) a collision resolution and buffer prediction method based on Cross-Entropy Method to handle blocked goal and swap cases, (iii) a directed spatial graph-based state space for scalability, and (iv) deep reinforcement learning (RL) for producing an efficient plan to simultaneously discover unseen objects and rearrange the visible misplaced ones to minimize the overall traversal. The paper also presents new metrics and a benchmark dataset called MoPOR to evaluate the effectiveness of the rearrangement planning in a multi-room setting. The experimental results demonstrate that the proposed method effectively addresses the multi-room rearrangement problem.

----

## [1154] Hierarchical Planning and Learning for Robots in Stochastic Settings Using Zero-Shot Option Invention

**Authors**: *Naman Shah, Siddharth Srivastava*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28903](https://doi.org/10.1609/aaai.v38i9.28903)

**Abstract**:

This paper addresses the problem of inventing and using hierarchical representations for stochastic robot-planning problems. Rather than using hand-coded state or action representations as input, it presents new methods for learning how to create a high-level action representation for long-horizon, sparse reward robot planning problems in stochastic settings with unknown dynamics. After training, this system yields a robot-specific but environment independent planning system. Given new problem instances in unseen stochastic environments, it first creates zero-shot options (without any experience on the new environment) with dense pseudo-rewards and then uses them to solve the input problem in a hierarchical planning and refinement process. Theoretical results identify sufficient conditions for completeness of the presented approach. Extensive empirical analysis shows that even in settings that go beyond these sufficient conditions, this approach convincingly outperforms baselines by 2x in terms of solution time with orders of magnitude improvement in solution quality.

----

## [1155] MorphVAE: Advancing Morphological Design of Voxel-Based Soft Robots with Variational Autoencoders

**Authors**: *Junru Song, Yang Yang, Wei Peng, Weien Zhou, Feifei Wang, Wen Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28904](https://doi.org/10.1609/aaai.v38i9.28904)

**Abstract**:

Soft robot design is an intricate field with unique challenges due to its complex and vast search space. In the past literature, evolutionary computation algorithms, including novel probabilistic generative models (PGMs), have shown potential in this realm. However, these methods are sample inefficient and predominantly focus on rigid robots in locomotion tasks, which limit their performance and application in robot design automation. In this work, we propose MorphVAE, an innovative PGM that incorporates a multi-task training scheme and a meticulously crafted sampling technique termed ``continuous natural selection'', aimed at bolstering sample efficiency. This method empowers us to gain insights from assessed samples across diverse tasks and temporal evolutionary stages, while simultaneously maintaining a delicate balance between optimization efficiency and biodiversity. Through extensive experiments in various locomotion and manipulation tasks, we substantiate the efficiency of MorphVAE in generating high-performing and diverse designs, surpassing the performance of competitive baselines.

----

## [1156] DistilVPR: Cross-Modal Knowledge Distillation for Visual Place Recognition

**Authors**: *Sijie Wang, Rui She, Qiyu Kang, Xingchao Jian, Kai Zhao, Yang Song, Wee Peng Tay*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28905](https://doi.org/10.1609/aaai.v38i9.28905)

**Abstract**:

The utilization of multi-modal sensor data in visual place recognition (VPR) has demonstrated enhanced performance compared to single-modal counterparts. Nonetheless, integrating additional sensors comes with elevated costs and may not be feasible for systems that demand lightweight operation, thereby impacting the practical deployment of VPR. To address this issue, we resort to knowledge distillation, which empowers single-modal students to learn from cross-modal teachers without introducing additional sensors during inference. Despite the notable advancements achieved by current distillation approaches, the exploration of feature relationships remains an under-explored area. In order to tackle the challenge of cross-modal distillation in VPR, we present DistilVPR, a novel distillation pipeline for VPR. We propose leveraging feature relationships from multiple agents, including self-agents and cross-agents for teacher and student neural networks. Furthermore, we integrate various manifolds, characterized by different space curvatures for exploring feature relationships. This approach enhances the diversity of feature relationships, including Euclidean, spherical, and hyperbolic relationship modules, thereby enhancing the overall representational capacity. The experiments demonstrate that our proposed pipeline achieves state-of-the-art performance compared to other distillation baselines. We also conduct necessary ablation studies to show design effectiveness. The code is released at: https://github.com/sijieaaa/DistilVPR

----

## [1157] Angle Robustness Unmanned Aerial Vehicle Navigation in GNSS-Denied Scenarios

**Authors**: *Yuxin Wang, Zunlei Feng, Haofei Zhang, Yang Gao, Jie Lei, Li Sun, Mingli Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28906](https://doi.org/10.1609/aaai.v38i9.28906)

**Abstract**:

Due to the inability to receive signals from the Global Navigation Satellite System (GNSS) in extreme conditions, achieving accurate and robust navigation for Unmanned Aerial Vehicles (UAVs) is a challenging task. Recently emerged, vision-based navigation has been a promising and feasible alternative to GNSS-based navigation. However, existing vision-based techniques are inadequate in addressing flight deviation caused by environmental disturbances and inaccurate position predictions in practical settings. In this paper, we present a novel angle robustness navigation paradigm to deal with flight deviation in point-to-point navigation tasks. Additionally, we propose a model that includes the Adaptive Feature Enhance Module, Cross-knowledge Attention-guided Module and Robust Task-oriented Head Module to accurately predict direction angles for high-precision navigation. To evaluate the vision-based navigation methods, we collect a new dataset termed as UAV_AR368. Furthermore, we design the Simulation Flight Testing Instrument (SFTI) using Google Earth to simulate different flight environments, thereby reducing the expenses associated with real flight testing. Experiment results demonstrate that the proposed model outperforms the state-of-the-art by achieving improvements of 26.0% and 45.6% in the success rate of arrival under ideal and disturbed circumstances, respectively.

----

## [1158] Learning from Ambiguous Demonstrations with Self-Explanation Guided Reinforcement Learning

**Authors**: *Yantian Zha, Lin Guan, Subbarao Kambhampati*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28907](https://doi.org/10.1609/aaai.v38i9.28907)

**Abstract**:

Our work aims at efficiently leveraging ambiguous demonstrations for the training of a reinforcement learning (RL) agent. An ambiguous demonstration can usually be interpreted in multiple ways, which severely hinders the RL agent from learning stably and efficiently. Since an optimal demonstration may also suffer from being ambiguous, previous works that combine RL and learning from demonstration (RLfD works) may not work well. Inspired by how humans handle such situations, we propose to use self-explanation (an agent generates explanations for itself) to recognize valuable high-level relational features as an interpretation of why a successful trajectory is successful. This way, the agent can leverage the explained important relations as guidance for its RL learning. Our main contribution is to propose the Self-Explanation for RL from Demonstrations (SERLfD) framework, which can overcome the limitations of existing RLfD works. Our experimental results show that an RLfD model can be improved by using our SERLfD framework in terms of training stability and performance. To foster further research in self-explanation-guided robot learning, we have made our demonstrations and code publicly accessible at https://github.com/YantianZha/SERLfD. For a deeper understanding of our work, interested readers can refer to our arXiv version at https://arxiv.org/pdf/2110.05286.pdf, including an accompanying appendix.

----

## [1159] Multi-Constellation-Inspired Single-Shot Global LiDAR Localization

**Authors**: *Tongzhou Zhang, Gang Wang, Yu Chen, Hai Zhang, Jue Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28908](https://doi.org/10.1609/aaai.v38i9.28908)

**Abstract**:

Global localization is a challenging task for intelligent robots, as its accuracy directly contributes to the performance of downstream navigation and planning tasks. However, existing literature focus more on the place retrieval and the success rate of localization, with limited attention given to the metrics of position estimation. In this paper, a single-shot global LiDAR localization method is proposed with the ultimate goal of achieving high position accuracy, inspired by the positioning approach of multi-constellation localization systems. Initially, we perform coarse localization using global descriptors and select observation points along with their corresponding coordinates based on the obtained coarse localization results. Coordinates can be acquired from a pre-built map, GNSS, or other devices. Then, a lightweight LiDAR odometry method is designed to estimate the distance between the retrieved data and the observation points. Ultimately, the localization problem is transformed into an optimization problem of solving a system of multiple sphere equations. The experimental results on the KITTI dataset and the self-collected dataset demonstrate that our method achieves an average localization error (including errors in the z-axis) of 0.89 meters. In addition, it achieves retrieval efficiency of 0.357 s per frame on the former dataset and 0.214 s per frame on the latter one. Code and data are available at https://github.com/jlurobot/multi-constellation-localization.

----

## [1160] DeepPointMap: Advancing LiDAR SLAM with Unified Neural Descriptors

**Authors**: *Xiaze Zhang, Ziheng Ding, Qi Jing, Yuejie Zhang, Wenchao Ding, Rui Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28909](https://doi.org/10.1609/aaai.v38i9.28909)

**Abstract**:

Point clouds have shown significant potential in various domains, including Simultaneous Localization and Mapping (SLAM). However, existing approaches either rely on dense point clouds to achieve high localization accuracy or use generalized descriptors to reduce map size. Unfortunately, these two aspects seem to conflict with each other. To address this limitation, we propose an unified architecture, DeepPointMap, achieving excellent preference on both aspects. We utilize neural network to extract highly representative and sparse neural descriptors from point clouds, enabling memory-efficient map representation and accurate multi-scale localization tasks (e.g., odometry and loop-closure). Moreover, we showcase the versatility of our framework by extending it to more challenging multi-agent collaborative SLAM. The promising results obtained in these scenarios further emphasize the effectiveness and potential of our approach.

----

## [1161] Complexity of Credulous and Skeptical Acceptance in Epistemic Argumentation Framework

**Authors**: *Gianvincenzo Alfano, Sergio Greco, Francesco Parisi, Irina Trubitsyna*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28910](https://doi.org/10.1609/aaai.v38i9.28910)

**Abstract**:

Dung’s Argumentation Framework (AF) has been extended in several directions. Among the numerous proposed extensions, three of them seem to be of particular interest and have correlations between them. These extensions are: constrained AF (CAF), where AF is augmented with (strong) constraints; epistemic AF (EAF), where AF is augmented with epistemic constraints; and incomplete AF (iAF), where arguments and attacks can be uncertain. While the complexity and expressiveness of CAF and iAF have been studied, that of EAF has not been explored so far. In this paper we investigate the complexity and expressivity of EAF. To this end, we first introduce the Labeled CAF (LCAF), a variation of CAF where constraints are defined over the alphabet of labeled arguments. Then, we investigate the complexity of credulous and skeptical reasoning and show that: i) EAF is more expressive than iAF (under preferred semantics), ii) although LCAF is a restriction of EAF where modal operators are not allowed, these frameworks have the same complexity, iii) the results for LCAF close a gap in the characterization of the complexity of CAF. Interestingly, even though EAF has the same complexity as LCAF, it allows modeling domain knowledge in a more natural and easy-to-understand way.

----

## [1162] Approximation Algorithms for Preference Aggregation Using CP-Nets

**Authors**: *Abu Mohammad Hammad Ali, Boting Yang, Sandra Zilles*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28911](https://doi.org/10.1609/aaai.v38i9.28911)

**Abstract**:

This paper studies the design and analysis of approximation algorithms for aggregating preferences over combinatorial domains, represented using Conditional Preference Networks (CP-nets). Its focus is on aggregating preferences over so-called swaps, for which optimal solutions in general are already known to be of exponential size. We first analyze a trivial 2-approximation algorithm that simply outputs the best of the given input preferences, and establish a structural condition under which the approximation ratio of this algorithm is improved to 4/3. We then propose a polynomial-time approximation algorithm whose outputs are provably no worse than those of the trivial algorithm, but often substantially better. A family of problem instances is presented for which our improved algorithm produces optimal solutions, while, for any ε, the trivial algorithm cannot attain a (2- ε)-approximation. These results may lead to the first polynomial-time approximation algorithm that solves the CP-net aggregation problem for swaps with an approximation ratio substantially better than 2.

----

## [1163] What Does a Query Answer Tell You? Informativeness of Query Answers for Knowledge Bases

**Authors**: *Luca Andolfi, Gianluca Cima, Marco Console, Maurizio Lenzerini*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28912](https://doi.org/10.1609/aaai.v38i9.28912)

**Abstract**:

Query answering for Knowledge Bases (KBs) amounts to extracting information from the various models of a KB, and presenting the user with an object that represents such information. In the vast majority of cases, this object consists of those tuples of constants that satisfy the query expression either in every model (certain answers) or in some model (possible answers). However, similarly to the case of incomplete databases, both these forms of answers are a lossy representation of all the knowledge inferable from the query and the queried KB. In this paper, we illustrate a formal framework to characterize the information that query answers for KBs are able to represent. As a first application of the framework, we study the informativeness of current query answering approaches, including the recently introduced partial answers. We then define a novel notion of answers, allowing repetition of variables across answer tuples. We show that these answers are capable of representing a meaningful form of information, and we also study their data complexity properties.

----

## [1164] Defeasible Normative Reasoning: A Proof-Theoretic Integration of Logical Argumentation

**Authors**: *Ofer Arieli, Kees van Berkel, Christian Straßer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28913](https://doi.org/10.1609/aaai.v38i9.28913)

**Abstract**:

We present a novel computational approach to resolving conflicts among norms by nonmonotonic normative reasoning (in constrained I/O logics). Our approach extends standard sequent-based proof systems and makes them more adequate to nonmonotonic reasoning by adding to the sequents annotations that keep track of what is known about the defeasible status of the derived sequents. This makes transparent the reasons according to which norms should be applicable or inapplicable, and accordingly the sequents that make use of such norms are accepted or retracted. We also show that this proof theoretic method has tight links to the semantics of formal argumentation frameworks. The outcome of this paper is thus a threefold characterization result that relates, in the context of nonmonotonic normative reasoning, three traditional ingredients of AI-based reasoning methods: maximally consistent sets of premises (in constrained I/O logics), derived sequents (which are accepted in corresponding annotated sequent calculi), and logical arguments (that belong to the grounded extensions of the induced logical argumentation frameworks).

----

## [1165] Computing the Why-Provenance for Datalog Queries via SAT Solvers

**Authors**: *Marco Calautti, Ester Livshits, Andreas Pieris, Markus Schneider*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28914](https://doi.org/10.1609/aaai.v38i9.28914)

**Abstract**:

Explaining an answer to a Datalog query is an essential task towards Explainable AI, especially nowadays where Datalog plays a critical role in the development of ontology-based applications. A well-established approach for explaining a query answer is the so-called why-provenance, which essentially collects all the subsets of the input database that can be used to obtain that answer via some derivation process, typically represented as a proof tree. It is well known, however, that computing the why-provenance for Datalog queries is computationally expensive, and thus, very few attempts can be found in the literature. The goal of this work is to demonstrate how off-the-shelf SAT solvers can be exploited towards an efficient computation of the why-provenance for Datalog queries. Interestingly, our SAT-based approach allows us to build the why-provenance in an incremental fashion, that is, one explanation at a time, which is much more useful in a practical context than the one-shot computation of the whole set of explanations as done by existing approaches.

----

## [1166] Generalisation through Negation and Predicate Invention

**Authors**: *David M. Cerna, Andrew Cropper*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28915](https://doi.org/10.1609/aaai.v38i9.28915)

**Abstract**:

The ability to generalise from a small number of examples is a fundamental challenge in machine learning. To tackle this challenge, we introduce an inductive logic programming (ILP) approach that combines negation and predicate invention. 
Combining these two features allows an ILP system to generalise better by learning rules with universally quantified body-only variables. We implement our idea in NOPI, which can learn normal logic programs with predicate invention, including Datalog programs with stratified negation. Our experimental results on multiple domains show that our approach can improve predictive accuracies and learning times.

----

## [1167] Learning Small Decision Trees for Data of Low Rank-Width

**Authors**: *Konrad K. Dabrowski, Eduard Eiben, Sebastian Ordyniak, Giacomo Paesani, Stefan Szeider*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28916](https://doi.org/10.1609/aaai.v38i9.28916)

**Abstract**:

We consider the NP-hard problem of finding a smallest decision tree
representing a classification instance in terms of a partially defined
Boolean function. Small decision trees are desirable to provide an
interpretable model for the given data. We show that the problem is
fixed-parameter tractable when parameterized by the rank-width of the
incidence graph of the given classification instance. Our algorithm
proceeds by dynamic programming using an NLC decomposition obtained
from a rank-width decomposition. The key to the algorithm is a
succinct representation of partial solutions. This allows us to limit
the space and time requirements for each dynamic programming step in
terms of the parameter.

----

## [1168] Stable Model Semantics for Description Logic Terminologies

**Authors**: *Federica Di Stefano, Mantas Simkus*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28917](https://doi.org/10.1609/aaai.v38i9.28917)

**Abstract**:

This paper studies a stable model semantics for Description Logic (DL) knowledge bases (KBs) and for (possibly cyclic) terminologies, ultimately showing that terminologies under the proposed semantics can be equipped with effective reasoning algorithms. The semantics is derived using Quantified Equilibrium Logic, and---in contrast to the usual semantics of DLs based on classical logic---supports default negation and allows to combine the open-world and the closed-world assumptions in a natural way. Towards understanding the computational properties of this and related formalisms, we show a strong undecidability result that applies not only to KBs under the stable model semantics, but also to the more basic setting of minimal model reasoning. Specifically, we show that concept satisfiability in minimal models of an ALCIO KB is undecidable. We then turn our attention to (possibly cyclic) DL terminologies, where ontological axioms are limited to definitions of concept names in terms of complex concepts. This restriction still yields a very rich setting. We show that standard reasoning problems, like concept satisfiability and subsumption, are ExpTime-complete for terminologies expressed in ALCI under the stable model semantics.

----

## [1169] Redefining ABA+ Semantics via Abstract Set-to-Set Attacks

**Authors**: *Yannis Dimopoulos, Wolfgang Dvorák, Matthias König, Anna Rapberger, Markus Ulbricht, Stefan Woltran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28918](https://doi.org/10.1609/aaai.v38i9.28918)

**Abstract**:

Assumption-based argumentation (ABA) is a powerful defeasible reasoning formalism which is based on the interplay of assumptions, their contraries, and inference rules. ABA with preferences (ABA+) generalizes the basic model by allowing qualitative comparison between assumptions. The integration of preferences however comes with a cost. In ABA+, the evaluation under two central and well-established semantics---grounded and complete semantics---is not guaranteed to yield an outcome. Moreover, while ABA frameworks without preferences allow for a graph-based representation in Dung-style frameworks, an according instantiation for general ABA+ frameworks has not been established so far. In this work, we tackle both issues: First, we develop a novel abstract argumentation formalism based on set-to-set attacks. We show that our so-called Hyper Argumentation Frameworks (HYPAFs) capture ABA+. Second, we propose relaxed variants of complete and grounded semantics for HYPAFs that yield an extension for all frameworks by design, while still faithfully generalizing the established semantics of Dung-style Argumentation Frameworks. We exploit the newly established correspondence between ABA+ and HYPAFs to obtain variants for grounded and complete ABA+ semantics that are guaranteed to yield an outcome. Finally, we discuss basic properties and provide a complexity analysis. Along the way, we settle the computational complexity of several ABA+ semantics.

----

## [1170] Towards Epistemic-Doxastic Planning with Observation and Revision

**Authors**: *Thorsten Engesser, Andreas Herzig, Elise Perrotin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28919](https://doi.org/10.1609/aaai.v38i9.28919)

**Abstract**:

Epistemic planning is useful in situations where multiple agents have different knowledge and beliefs about the world, such as in robot-human interaction. One aspect that has been largely neglected in the literature is planning with observations in the presence of false beliefs. This is a particularly challenging problem because it requires belief revision. We introduce a simple specification language for reasoning about actions with knowledge and belief. We demonstrate our approach on well-known false-belief tasks such as the Sally-Anne Task and compare it to other action languages. Our logic leads to an epistemic planning formalism that is expressive enough to model second-order false-belief tasks, yet has the same computational complexity as classical planning.

----

## [1171] Dynamic Tangled Derivative Logic of Metric Spaces

**Authors**: *David Fernández-Duque, Yoàv Montacute*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28920](https://doi.org/10.1609/aaai.v38i9.28920)

**Abstract**:

Dynamical systems are abstract models of interaction between space and time. They are often used in fields such as physics and engineering to understand complex processes, but due to their general nature, they have found applications for studying computational processes, interaction in multi-agent systems, machine learning algorithms and other computer science related phenomena. In the vast majority of applications, a dynamical system consists of the action of a continuous `transition function' on a metric space. In this work, we consider decidable formal systems for reasoning about such structures. Spatial logics can be traced back to the 1940's, but our work follows a more dynamic turn that these logics have taken due to two recent developments: the study of the topological mu-calculus, and the the integration of linear temporal logic with logics based on the Cantor derivative. In this paper, we combine dynamic topological logics based on the Cantor derivative and the `next point in time' operators with an expressively complete fixed point operator to produce a combination of the topological mu-calculus with linear temporal logic. We show that the resulting logics are decidable and have a natural axiomatisation. Moreover, we prove that these logics are complete for interpretations on the Cantor space, the rational numbers, and subspaces thereof.

----

## [1172] Submodel Enumeration for CTL Is Hard

**Authors**: *Nicolas Fröhlich, Arne Meier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28921](https://doi.org/10.1609/aaai.v38i9.28921)

**Abstract**:

Expressing system specifications using Computation Tree Logic (CTL) formulas, formalising programs using Kripke structures, and then model checking the system is an established workflow in program verification and has wide applications in AI. In this paper, we consider the task of model enumeration, which asks for a uniform stream of output systems that satisfy the given specification. We show that, given a CTL formula and a system (potentially falsified by the formula), enumerating satisfying submodels is always hard for CTL--regardless of which subset of CTL-operators is considered. As a silver lining on the horizon, we present fragments via restrictions on the allowed Boolean functions that still allow for fast enumeration.

----

## [1173] Linear-Time Verification of Data-Aware Processes Modulo Theories via Covers and Automata

**Authors**: *Alessandro Gianola, Marco Montali, Sarah Winkler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28922](https://doi.org/10.1609/aaai.v38i9.28922)

**Abstract**:

The need to model and analyse dynamic systems operating over complex data is ubiquitous in AI and neighboring areas, in particular business process management. Analysing such data-aware systems is a notoriously difficult problem, as they are intrinsically infinite-state. Existing approaches work for specific datatypes, and/or limit themselves to the verification of safety properties. In this paper, we lift both such limitations, studying for the first time linear-time verification for so-called data-aware processes modulo theories (DMTs), from the foundational and practical point of view. The DMT model is very general, as it supports processes operating over variables that can store arbitrary types of data, ranging over infinite domains and equipped with domain-specific predicates. Specifically, we provide four contributions. First, we devise a semi-decision procedure for linear-time verification of DMTs, which works for a very large class of datatypes obeying to mild model-theoretic assumptions. The procedure relies on a unique combination of automata-theoretic and cover computation techniques to respectively deal with linear-time properties and datatypes. Second, we identify an abstract, semantic property that guarantees the existence of a faithful finite-state abstraction of the original system, and show that our method becomes a decision procedure in this case. Third, we identify concrete, checkable classes of systems that satisfy this property, generalising several results in the literature. Finally, we present an implementation and an experimental evaluation over a benchmark of real-world data-aware business processes.

----

## [1174] On the Structural Hardness of Answer Set Programming: Can Structure Efficiently Confine the Power of Disjunctions?

**Authors**: *Markus Hecher, Rafael Kiesel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28923](https://doi.org/10.1609/aaai.v38i9.28923)

**Abstract**:

Answer Set Programming (ASP) is a generic problem modeling and solving framework with a strong focus on knowledge representation and a rapid growth of industrial applications. So far, the study of complexity resulted in characterizing hardness and determining their sources, fine-grained insights in the form of dichotomy-style results, as well as detailed parameterized complexity landscapes. Unfortunately, for the well-known parameter treewidth disjunctive programs require double-exponential runtime under reasonable complexity assumptions. This quickly becomes out of reach. We deal with the classification of structural parameters for disjunctive ASP on the program's rule structure (incidence graph). 
First, we provide a polynomial kernel to obtain single-exponential runtime in terms of vertex cover size, despite subset-minimization being not represented in the program’s structure. Then we turn our attention to strictly better structural parameters between vertex cover size and treewidth. Here, we provide double-exponential lower bounds for the most prominent parameters in that range: treedepth, feedback vertex size, and cliquewidth. Based on this, we argue that unfortunately our options beyond vertex cover size are limited. Our results provide an in-depth hardness study, relying on a novel reduction from normal to disjunctive programs, trading the increase of complexity for an exponential parameter compression.

----

## [1175] Knowledge Enhanced Representation Learning for Drug Discovery

**Authors**: *Thanh Lam Hoang, Marco Luca Sbodio, Marcos Martínez Galindo, Mykhaylo Zayats, Raúl Fernández-Díaz, Victor Valls, Gabriele Picco, Cesar Berrospi, Vanessa López*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28924](https://doi.org/10.1609/aaai.v38i9.28924)

**Abstract**:

Recent research on predicting the binding affinity between drug molecules and proteins use representations learned, through unsupervised learning techniques, from large databases of molecule SMILES and protein sequences. While these representations have significantly enhanced the predictions, they are usually based on a limited set of modalities, and they do not exploit available knowledge about existing relations among molecules and proteins. Our study reveals that enhanced representations, derived from multimodal knowledge graphs describing relations among molecules and proteins, lead to state-of-the-art results in well-established benchmarks (first place in the leaderboard for Therapeutics Data Commons benchmark ``Drug-Target Interaction Domain Generalization Benchmark", with an improvement of 8 points with respect to previous best result). Moreover, our results significantly surpass those achieved in standard benchmarks by using conventional pre-trained representations that rely only on sequence or SMILES data. We release our multimodal knowledge graphs, integrating data from seven public data sources, and which contain over 30 million triples.  Pretrained models from our proposed graphs and benchmark task source code are also released.

----

## [1176] Learning MDL Logic Programs from Noisy Data

**Authors**: *Céline Hocquette, Andreas Niskanen, Matti Järvisalo, Andrew Cropper*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28925](https://doi.org/10.1609/aaai.v38i9.28925)

**Abstract**:

Many inductive logic programming approaches struggle to learn programs from noisy data. To overcome this limitation, we introduce an approach that learns minimal description length programs from noisy data, including recursive programs. Our experiments on several domains, including drug design, game playing, and program synthesis, show that our approach can outperform existing approaches in terms of predictive accuracies and scale to moderate amounts of noise.

----

## [1177] A Compiler for Weak Decomposable Negation Normal Form

**Authors**: *Petr Illner, Petr Kucera*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28926](https://doi.org/10.1609/aaai.v38i9.28926)

**Abstract**:

This paper integrates weak decomposable negation normal form (wDNNF) circuits, introduced by Akshay et al. in 2018, into the knowledge compilation map. This circuit type generalises decomposable negation normal form (DNNF) circuits in such a way that they allow a restricted form of sharing variables among the inputs of a conjunction node. We show that wDNNF circuits have the same properties as DNNF circuits regarding the queries and transformations presented in the knowledge compilation map, whilst being strictly more succinct than DNNF circuits (that is, they can represent Boolean functions compactly). We also present and evaluate a knowledge compiler, called Bella, for converting CNF formulae into wDNNF circuits. Our experiments demonstrate that wDNNF circuits are suitable for configuration instances.

----

## [1178] Exact ASP Counting with Compact Encodings

**Authors**: *Mohimenul Kabir, Supratik Chakraborty, Kuldeep S. Meel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28927](https://doi.org/10.1609/aaai.v38i9.28927)

**Abstract**:

Answer Set Programming (ASP) has emerged as a promising
paradigm in knowledge representation and automated reason-
ing owing to its ability to model hard combinatorial problems
from diverse domains in a natural way. Building on advances
in propositional SAT solving, the past two decades have wit-
nessed the emergence of well-engineered systems for solv-
ing the answer set satisfiability problem, i.e., finding mod-
els or answer sets for a given answer set program. In re-
cent years, there has been growing interest in problems be-
yond satisfiability, such as model counting, in the context of
ASP. Akin to the early days of propositional model count-
ing, state-of-the-art exact answer set counters do not scale
well beyond small instances. Exact ASP counters struggle
with handling larger input formulas. The primary contribu-
tion of this paper is a new ASP counting framework, called
sharpASP, which counts answer sets avoiding larger input
formulas. This relies on an alternative way of defining answer
sets that allows lifting of key techniques developed in the con-
text of propositional model counting. Our extensive empirical
analysis over 1470 benchmarks demonstrates significant per-
formance gain over current state-of-the-art exact answer set
counters. Specifically, by using sharpASP, we were able to
solve 1062 benchmarks with PAR2 score of 3082 whereas
using prior state-of-the-art, we could only solve 895 bench-
marks with PAR2 score of 4205, all other experimental con-
ditions being the same.

----

## [1179] Minimal Macro-Based Rewritings of Formal Languages: Theory and Applications in Ontology Engineering (and Beyond)

**Authors**: *Christian Kindermann, Anne-Marie George, Bijan Parsia, Uli Sattler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28928](https://doi.org/10.1609/aaai.v38i9.28928)

**Abstract**:

In this paper, we introduce the problem of rewriting finite formal languages using syntactic macros such that the rewriting is minimal in size. We present polynomial-time algorithms to solve variants of this problem and show their correctness. To demonstrate the practical relevance of the proposed problems and the feasibility and effectiveness of our algorithms in practice, we apply these to biomedical ontologies authored in OWL. We find that such rewritings can significantly reduce the size of ontologies by capturing repeated expressions with macros. This approach not only offers valuable assistance in enhancing ontology quality and comprehension but can also be seen as a general methodology for evaluating features of rewriting systems (including syntactic macros, templates, or other forms of rewriting rules), which can be analyzed in terms of their influence on computational problems.

----

## [1180] On the Expressivity of Recurrent Neural Cascades

**Authors**: *Nadezda Alexandrovna Knorozova, Alessandro Ronca*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28929](https://doi.org/10.1609/aaai.v38i9.28929)

**Abstract**:

Recurrent Neural Cascades (RNCs) are the recurrent neural networks with no cyclic dependencies among recurrent neurons. This class of recurrent networks has received a lot of attention in practice. Besides training methods for a fixed architecture such as backpropagation, the cascade architecture naturally allows for constructive learning methods, where recurrent nodes are added incrementally one at a time, often yielding smaller networks. Furthermore, acyclicity amounts to a structural prior that even for the same number of neurons yields a more favourable sample complexity compared to a fully-connected architecture.
A central question is whether the advantages of the cascade architecture come at the cost of a reduced expressivity. We provide new insights into this question. We show that the regular languages captured by RNCs with sign and tanh activation with positive recurrent weights are the star-free regular languages. In order to establish our results we developed a novel framework where capabilities of RNCs are assessed by analysing which semigroups and groups a single neuron is able to implement. A notable implication of our framework is that RNCs can achieve the expressivity of all regular languages by introducing neurons that can implement groups.

----

## [1181] Efficient Axiomatization of OWL 2 EL Ontologies from Data by Means of Formal Concept Analysis

**Authors**: *Francesco Kriegel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28930](https://doi.org/10.1609/aaai.v38i9.28930)

**Abstract**:

We present an FCA-based axiomatization method that produces a complete OWL 2 EL TBox (the terminological part of an OWL 2 EL ontology) from a graph dataset in at most exponential time.  We describe technical details that allow for efficient implementation as well as variations that dispense with the computation of extremely large axioms, thereby rendering the approach applicable albeit some completeness is lost.  Moreover, we evaluate the prototype on real-world datasets.

----

## [1182] BAIT: Benchmarking (Embedding) Architectures for Interactive Theorem-Proving

**Authors**: *Sean Lamont, Michael Norrish, Amir Dezfouli, Christian Walder, Paul Montague*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28931](https://doi.org/10.1609/aaai.v38i9.28931)

**Abstract**:

Artificial Intelligence for Theorem Proving (AITP) has given
rise to a plethora of benchmarks and methodologies, particularly in Interactive Theorem Proving (ITP). Research in the
area is fragmented, with a diverse set of approaches being
spread across several ITP systems. This presents a significant challenge to the comparison of methods, which are often
complex and difficult to replicate.
Addressing this, we present BAIT, a framework for the fair
and streamlined comparison of learning approaches in ITP.
We demonstrate BAIT’s capabilities with an in-depth comparison, across several ITP benchmarks, of state-of-the-art
architectures applied to the problem of formula embedding.
We find that Structure Aware Transformers perform particularly well, improving on techniques associated with the original problem sets. BAIT also allows us to assess the end-to-end proving performance of systems built on interactive
environments. This unified perspective reveals a novel end-to-end system that improves on prior work. We also provide
a qualitative analysis, illustrating that improved performance
is associated with more semantically-aware embeddings. By
streamlining the implementation and comparison of Machine
Learning algorithms in the ITP context, we anticipate BAIT
will be a springboard for future research.

----

## [1183] INFORMEDQX: Informed Conflict Detection for Over-Constrained Problems

**Authors**: *Viet-Man Le, Alexander Felfernig, Thi Ngoc Trang Tran, Mathias Uta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28932](https://doi.org/10.1609/aaai.v38i9.28932)

**Abstract**:

Conflict detection is relevant in various application scenarios, ranging from interactive decision-making to the diagnosis of faulty knowledge bases. Conflicts can be regarded as sets of constraints that cause an inconsistency. In many scenarios (e.g., constraint-based configuration), conflicts are repeatedly determined for the same or similar sets of constraints. This misses out on the valuable opportunity for leveraging knowledge reuse and related potential performance improvements, which are extremely important, specifically interactive constraint-based applications. In this paper, we show how to integrate knowledge reuse concepts into non-instructive conflict detection. We introduce the InformedQX algorithm, which is a reuse-aware variant of QuickXPlain. The results of a related performance analysis with the Linux-2.6.3.33 configuration knowledge base show significant improvements in terms of runtime performance compared to QuickXPlain.

----

## [1184] Abstraction of Situation Calculus Concurrent Game Structures

**Authors**: *Yves Lespérance, Giuseppe De Giacomo, Maryam Rostamigiv, Shakil M. Khan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28933](https://doi.org/10.1609/aaai.v38i9.28933)

**Abstract**:

We present a general framework for abstracting agent behavior in multi-agent synchronous games in the situation calculus, which provides a first-order representation of the state and allows us to model how plays depend on the data and objects involved.  We represent such games as action theories of a special form called situation calculus synchronous game structures (SCSGSs), in which we have a single action "tick" whose effects depend on the combination of moves selected by the players.  In our framework, one specifies both an abstract SCSGS and a concrete SCSGS, as well as a refinement mapping that specifies how each abstract move is implemented by a Golog program defined over the concrete SCSGS.  We define notions of sound and complete abstraction with respect to a mapping over such SCSGS.  To express strategic properties on the abstract and concrete games we adopt a first-order variant of alternating-time mu-calculus mu-ATL-FO.  We show that we can exploit abstraction in verifying mu-ATL-FO properties of SCSGSs under the assumption that agents can always execute abstract moves to completion even if not fully controlling their outcomes.

----

## [1185] Relational Programming with Foundational Models

**Authors**: *Ziyang Li, Jiani Huang, Jason Liu, Felix Zhu, Eric Zhao, William Dodds, Neelay Velingker, Rajeev Alur, Mayur Naik*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28934](https://doi.org/10.1609/aaai.v38i9.28934)

**Abstract**:

Foundation models have vast potential to enable diverse AI applications. The powerful yet incomplete nature of these models has spurred a wide range of mechanisms to augment them with capabilities such as in-context learning, information retrieval, and code interpreting. We propose Vieira, a declarative framework that unifies these mechanisms in a general solution for programming with foundation models. Vieira follows a probabilistic relational paradigm and treats foundation models as stateless functions with relational inputs and outputs. It supports neuro-symbolic applications by enabling the seamless combination of such models with logic programs, as well as complex, multi-modal applications by streamlining the composition of diverse sub-models. We implement Vieira by extending the Scallop compiler with a foreign interface that supports foundation models as plugins. We implement plugins for 12 foundation models including GPT, CLIP, and SAM. We evaluate Vieira on 9 challenging tasks that span language, vision, and structured and vector databases. Our evaluation shows that programs in Vieira are concise, can incorporate modern foundation models, and have comparable or better accuracy than competitive baselines.

----

## [1186] MINES: Message Intercommunication for Inductive Relation Reasoning over Neighbor-Enhanced Subgraphs

**Authors**: *Ke Liang, Lingyuan Meng, Sihang Zhou, Wenxuan Tu, Siwei Wang, Yue Liu, Meng Liu, Long Zhao, Xiangjun Dong, Xinwang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28935](https://doi.org/10.1609/aaai.v38i9.28935)

**Abstract**:

GraIL and its variants have shown their promising capacities for inductive relation reasoning on knowledge graphs. However, the uni-directional message-passing mechanism hinders such models from exploiting hidden mutual relations between entities in directed graphs. Besides, the enclosing subgraph extraction in most GraIL-based models restricts the model from extracting enough discriminative information for reasoning. Consequently, the expressive ability of these models is limited. To address the problems, we propose a novel GraIL-based framework, termed MINES, by introducing a Message Intercommunication mechanism on the Neighbor-Enhanced Subgraph. Concretely, the message intercommunication mechanism is designed to capture the omitted hidden mutual information. It introduces bi-directed information interactions between connected entities by inserting an undirected/bi-directed GCN layer between uni-directed RGCN layers. Moreover, inspired by the success of involving more neighbors in other graph-based tasks, we extend the neighborhood area beyond the enclosing subgraph to enhance the information collection for inductive relation reasoning. Extensive experiments prove the promising capacity of the proposed MINES from various aspects, especially for the superiority, effectiveness, and transfer ability.

----

## [1187] Auditable Algorithms for Approximate Model Counting

**Authors**: *Kuldeep S. Meel, Supratik Chakraborty, S. Akshay*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28936](https://doi.org/10.1609/aaai.v38i9.28936)

**Abstract**:

The problem of model counting, i.e., counting satisfying assignments of a Boolean formula, is a fundamental problem in computer science, with diverse applications. Given #P-hardness of the problem, many algorithms have been developed over the years to provide an approximate model count. Recently, building on the practical success of SAT-solvers used as NP oracles, the focus has shifted from theory to practical implementations of such algorithms. This has brought to focus new challenges. In this paper, we consider one such challenge – that of auditable deterministic approximate model counters wherein a counter should also generate a certificate, which allows a user (often with limited computational power) to independently audit whether the count returned by an invocation of the algorithm is indeed within the promised bounds. 

We start by examining a celebrated approximate model counting algorithm due to Stockmeyer that uses polynomially many calls to a \Sigma^2_P oracle, and show that it can be audited via a \Pi^2_P formula on (n^2 log^2 n) variables, where n is the number of variables in the original formula. Since n is often large (10’s to 100’s of thousands) for typical instances, we ask if the count of variables in the certificate formula can be reduced – a critical question towards potential implementation. We show that this improvement in certification can be achieved with a tradeoff in the counting algorithm’s complexity. Specifically, we develop new deterministic approximate model counting algorithms that invoke a \Sigma^3_P oracle, but can be certified using a \Pi^2_P formula on fewer variables: our final algorithm uses just (n log n) variables.

Our study demonstrates that one can simplify certificate checking significantly if we allow the counting algorithm to access a slightly more powerful oracle. We believe this shows for the first time how the audit complexity can be traded for the complexity of approximate counting.

----

## [1188] A General Theoretical Framework for Learning Smallest Interpretable Models

**Authors**: *Sebastian Ordyniak, Giacomo Paesani, Mateusz Rychlicki, Stefan Szeider*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28937](https://doi.org/10.1609/aaai.v38i9.28937)

**Abstract**:

We develop a general algorithmic framework that allows us to obtain fixed-parameter tractability for computing smallest symbolic models that represent given data. Our framework applies to all ML model types that admit a certain extension property. By showing this extension property for decision trees, decision sets, decision lists, and binary decision diagrams, we obtain that minimizing these fundamental model types is fixed-parameter tractable. Our framework even applies to ensembles, which combine individual models by majority decision.

----

## [1189] Reinforcement Learning and Data-Generation for Syntax-Guided Synthesis

**Authors**: *Julian Parsert, Elizabeth Polgreen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28938](https://doi.org/10.1609/aaai.v38i9.28938)

**Abstract**:

Program synthesis is the task of automatically generating code based on a specification. In Syntax-Guided Synthesis (SyGuS) this specification is a combination of a syntactic template and a logical formula, and the result is guaranteed to satisfy both. We present a reinforcement-learning guided algorithm for SyGuS which uses Monte-Carlo Tree Search (MCTS) to search the space of candidate solutions. Our algorithm learns policy and value functions which, combined with the upper confidence bound for trees, allow it to balance exploration and exploitation. A common challenge in applying machine learning approaches to syntax-guided synthesis is the scarcity of training data. To address this, we present a method for automatically generating training data for SyGuS based on anti-unification of existing first-order satisfiability problems, which we use to train our MCTS policy. We implement and evaluate this setup and demonstrate that learned policy and value improve the synthesis performance over a baseline by over 26 percentage points in the training and testing sets. Our tool outperforms state-of-the-art tool cvc5 on the training set and performs comparably in terms of the total number of problems solved on the testing set (solving 23% of the benchmarks on which cvc5 fails). We make our data set publicly available, to enable further application of machine learning methods to the SyGuS problem.

----

## [1190] Adaptive Reactive Synthesis for LTL and LTLf Modulo Theories

**Authors**: *Andoni Rodríguez, César Sánchez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28939](https://doi.org/10.1609/aaai.v38i9.28939)

**Abstract**:

Reactive synthesis is the process of generate correct con- trollers from temporal logic specifications. Typically, synthesis is restricted to Boolean specifications in LTL. Recently, a Boolean abstraction technique allows to translate LTLT specifications that contain literals in theories into equi-realizable LTL specifications, but no full synthesis procedure exists yet. In synthesis modulo theories, the system receives valuations of environment variables (from a first-order theory T ) and outputs valuations of system variables from T . In this paper, we address how to syntheize a full controller using a combination of the static Boolean controller obtained from the Booleanized LTL specification together with on-the-fly queries to a solver that produces models of satisfiable existential T formulae. This is the first synthesis method for LTL modulo theories. Additionally, our method can produce adaptive responses which increases explainability and can improve runtime properties like performance. Our approach is applicable to both LTL modulo theories and LTLf modulo theories.

----

## [1191] A Unified View on Forgetting and Strong Equivalence Notions in Answer Set Programming

**Authors**: *Zeynep G. Saribatur, Stefan Woltran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28940](https://doi.org/10.1609/aaai.v38i9.28940)

**Abstract**:

Answer Set Programming (ASP) is a prominent rule-based language for knowledge representation and reasoning with roots in logic programming and non-monotonic reasoning. The aim to capture the essence of removing (ir)relevant details in ASP programs led to the investigation of different notions, from strong persistence (SP) forgetting, to faithful abstractions, and, recently, strong simplifications, where the latter two can be seen as relaxed and strengthened notions of forgetting, respectively. Although it was observed that these notions are related, especially given that they have characterizations through the semantics for strong equivalence, it remained unclear whether they can be brought together. In this work, we bridge this gap by introducing a novel relativized equivalence notion, which is a relaxation of the recent simplification notion, that is able to capture all related notions from the literature. We provide the necessary and sufficient conditions for relativized simplifiability, which shows that the challenging part is for when the context programs do not contain all the atoms to remove. We then introduce an operator that combines projection and a relaxation of SP-forgetting to obtain the relativized simplifications. We furthermore provide complexity results that complete the overall picture.

----

## [1192] BeliefFlow: A Framework for Logic-Based Belief Diffusion via Iterated Belief Change

**Authors**: *Nicolas Schwind, Katsumi Inoue, Sébastien Konieczny, Pierre Marquis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28941](https://doi.org/10.1609/aaai.v38i9.28941)

**Abstract**:

This paper presents BeliefFlow, a novel framework for representing how logical beliefs spread among interacting agents within a network. In a Belief Flow Network (BFN), agents communicate asynchronously. The agents' beliefs are represented using epistemic states, which encompass their current beliefs and conditional beliefs guiding future changes. When communication occurs between two connected agents, the receiving agent changes its epistemic state using an improvement operator, a well-known type of rational iterated belief change operator that generalizes belief revision operators. We show that BFNs satisfy appealing properties, leading to two significant outcomes. First, in any BFN with strong network connectivity, the beliefs of all agents converge towards a global consensus. Second, within any BFN, we show that it is possible to compute an optimal strategy for influencing the global beliefs. This strategy, which involves controlling the beliefs of a least number of agents through bribery, can be identified from the topology of the network and can be computed in polynomial time.

----

## [1193] NegVSR: Augmenting Negatives for Generalized Noise Modeling in Real-world Video Super-Resolution

**Authors**: *Yexing Song, Meilin Wang, Zhijing Yang, Xiaoyu Xian, Yukai Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28942](https://doi.org/10.1609/aaai.v38i9.28942)

**Abstract**:

The capability of video super-resolution (VSR) to synthesize high-resolution (HR) video from ideal datasets has been demonstrated in many works. However, applying the VSR model to real-world video with unknown and complex degradation remains a challenging task. First, existing degradation metrics in most VSR methods are not able to effectively simulate real-world noise and blur. On the contrary, simple combinations of classical degradation are used for real-world noise modeling, which led to the VSR model often being violated by out-of-distribution noise. Second, many SR models focus on noise simulation and transfer. Nevertheless, the sampled noise is monotonous and limited. To address the aforementioned problems, we propose a Negatives augmentation strategy for generalized noise modeling in Video Super-Resolution (NegVSR) task. Specifically, we first propose sequential noise generation toward real-world data to extract practical noise sequences. Then, the degeneration domain is widely expanded by negative augmentation to build up various yet challenging real-world noise sets. We further propose the augmented negative guidance loss to learn robust features among augmented negatives effectively. Extensive experiments on real-world datasets (e.g., VideoLQ and FLIR) show that our method outperforms state-of-the-art methods with clear margins, especially in visual quality. Project page is available at: https://negvsr.github.io/.

----

## [1194] Scalable Enumeration of Trap Spaces in Boolean Networks via Answer Set Programming

**Authors**: *Giang V. Trinh, Belaid Benhamou, Samuel Pastva, Sylvain Soliman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28943](https://doi.org/10.1609/aaai.v38i9.28943)

**Abstract**:

Boolean Networks (BNs) are widely used as a modeling formalism in several domains, notably systems biology and computer science. A fundamental problem in BN analysis is the enumeration of trap spaces, which are hypercubes in the state space that cannot be escaped once entered. Several methods have been proposed for enumerating trap spaces, however they often suffer from scalability and efficiency issues, particularly for large and complex models. To our knowledge, the most efficient and recent methods for the trap space enumeration all rely on Answer Set Programming (ASP), which has been widely applied to the analysis of BNs. Motivated by these considerations, our work proposes a new method for enumerating trap spaces in BNs using ASP. We evaluate the method on a mix of 250+ real-world and 400+ randomly generated BNs, showing that it enables analysis of models beyond the capabilities of existing tools (namely pyboolnet, mpbn, trappist, and trapmvn).

----

## [1195] Non-flat ABA Is an Instance of Bipolar Argumentation

**Authors**: *Markus Ulbricht, Nico Potyka, Anna Rapberger, Francesca Toni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28944](https://doi.org/10.1609/aaai.v38i9.28944)

**Abstract**:

Assumption-based Argumentation (ABA) is a well-known structured argumentation formalism, whereby arguments and attacks between them are drawn from rules, defeasible assumptions and their contraries. 
A common restriction imposed on ABA frameworks (ABAFs) is that they are flat, i.e. each of the defeasible assumptions can only be assumed, but not derived. While it is known that flat ABAFs can be translated into abstract argumentation frameworks (AFs) as proposed by Dung, no translation exists from general, possibly non-flat ABAFs into any kind of abstract argumentation formalism. 
In this paper, we close this gap and show that bipolar AFs (BAFs) can instantiate general ABAFs. To this end we develop suitable, novel BAF semantics which borrow from the notion of deductive support. We investigate basic properties of our BAFs, including computational complexity, and prove the desired relation to ABAFs under several semantics.

----

## [1196] Bilateral Gradual Semantics for Weighted Argumentation

**Authors**: *Zongshun Wang, Yuping Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28945](https://doi.org/10.1609/aaai.v38i9.28945)

**Abstract**:

Abstract argumentation is a reasoning model for evaluating arguments. Recently, gradual semantics has received considerable attention in weighted argumentation, which assigns an acceptability degree to each argument as its strength. In this paper, we aim to enhance gradual semantics by non-reciprocally incorporating the notion of rejectability degree. Such a setting offers a bilateral perspective on argument strength, enabling more comprehensive argument evaluations in practical situations. To this end, we first provide a set of principles for our semantics, taking both the acceptability and rejectability degrees into account, and propose three novel semantics conforming to the above principles. These semantics are defined as the limits of iterative sequences that always converge in any given weighted argumentation system, making them preferable for real-world applications.

----

## [1197] Decomposing Constraint Networks for Calculating c-Representations

**Authors**: *Marco Wilhelm, Gabriele Kern-Isberner*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28946](https://doi.org/10.1609/aaai.v38i9.28946)

**Abstract**:

It is well-known from probability theory that network-based methods like Bayesian networks constitute remarkable frameworks for efficient probabilistic reasoning. In this paper, we focus on qualitative default reasoning based on Spohn’s ranking functions for which network-based methods have not yet been studied satisfactorily. With constraint networks, we develop a framework for iterative calculations of c-representations, a family of ranking models of conditional belief bases which show outstanding properties from a commonsense and formal point of view, that are characterized by assigning possible worlds a degree of implausibility via penalizing the falsification of conditionals. Constraint networks unveil the dependencies among these penalty points (and hence among the conditionals) and make it possible to compute the penalty points locally on so-called safe sub-bases. As an application of our framework, we show that skeptical c-inferences can be drawn locally from safe sub-bases without losing validity.

----

## [1198] Optimised Storage for Datalog Reasoning

**Authors**: *Xinyue Zhang, Pan Hu, Yavor Nenov, Ian Horrocks*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i9.28947](https://doi.org/10.1609/aaai.v38i9.28947)

**Abstract**:

Materialisation facilitates Datalog reasoning by precomputing all consequences of the facts and the rules so that queries can be directly answered over the materialised facts. However, storing all materialised facts may be infeasible in practice, especially when the rules are complex and the given set of facts is large. We observe that for certain combinations of rules, there exist data structures that compactly represent the reasoning result and can be efficiently queried when necessary. In this paper, we present a general framework that allows for the integration of such optimised storage schemes with standard materialisation algorithms. Moreover, we devise optimised storage schemes targeting at transitive rules and union rules, two types of (combination of) rules that commonly occur in practice. Our experimental evaluation shows that our approach significantly improves memory consumption, sometimes by orders of magnitude, while remaining competitive in terms of query answering time.

----

## [1199] Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers

**Authors**: *Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos, Michalis Vazirgiannis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28948](https://doi.org/10.1609/aaai.v38i10.28948)

**Abstract**:

In recent years, significant progress has been made in the field of protein function prediction with the development of various machine-learning approaches.
However, most existing methods formulate the task as a multi-classification problem, i.e. assigning predefined labels to proteins.
In this work, we propose a novel approach, Prot2Text, which predicts a protein's function in a free text style, moving beyond the conventional binary or categorical classifications.
By combining Graph Neural Networks(GNNs) and Large Language Models(LLMs), in an encoder-decoder framework, our model effectively integrates diverse data types including protein sequence, structure, and textual annotation and description.
This multimodal approach allows for a holistic representation of proteins' functions, enabling the generation of detailed and accurate functional descriptions.
To evaluate our model, we extracted a multimodal protein dataset from SwissProt, and demonstrate empirically the effectiveness of Prot2Text.
These results highlight the transformative impact of multimodal models, specifically the fusion of GNNs and LLMs, empowering researchers with powerful tools for more accurate function prediction of existing as well as first-to-see proteins.

----



[Go to the previous page](AAAI-2024-list05.md)

[Go to the next page](AAAI-2024-list07.md)

[Go to the catalog section](README.md)