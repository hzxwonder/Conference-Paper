## [200] Learning Stochastic Equivalence based on Discrete Ricci Curvature

**Authors**: *Xuan Guo, Qiang Tian, Wang Zhang, Wenjun Wang, Pengfei Jiao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/201](https://doi.org/10.24963/ijcai.2021/201)

**Abstract**:

Role-based network embedding methods aim to preserve node-centric connectivity patterns, which are expressions of node roles, into low-dimensional vectors. However, almost all the existing methods are designed for capturing a relaxation of automorphic equivalence or regular equivalence. They may be good at structure identification but could show poorer performance on role identification. Because automorphic equivalence and regular equivalence strictly tie the role of a node to the identities of all its neighbors. To mitigate this problem, we construct a framework called Curvature-based Network Embedding with Stochastic Equivalence (CNESE) to embed stochastic equivalence. More specifically, we estimate the role distribution of nodes based on discrete Ricci curvature for its excellent ability to concisely representing local topology. We use a Variational Auto-Encoder to generate embeddings while a degree-guided regularizer and a contrastive learning regularizer are leveraged to improving both its robustness and discrimination ability. The effectiveness of our proposed CNESE is demonstrated by extensive experiments on real-world networks.

----

## [201] Federated Learning with Sparsification-Amplified Privacy and Adaptive Optimization

**Authors**: *Rui Hu, Yanmin Gong, Yuanxiong Guo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/202](https://doi.org/10.24963/ijcai.2021/202)

**Abstract**:

Federated learning (FL) enables distributed agents to collaboratively learn a centralized model without sharing their raw data with each other. However, data locality does not provide sufficient privacy protection, and it is desirable to facilitate FL with rigorous differential privacy (DP) guarantee. Existing DP mechanisms would introduce random noise with magnitude proportional to the model size, which can be quite large in deep neural networks. In this paper, we propose a new FL framework with sparsification-amplified privacy. Our approach integrates random sparsification with gradient perturbation on each agent to amplify privacy guarantee. Since sparsification would increase the number of communication rounds required to achieve a certain target accuracy, which is unfavorable for DP guarantee, we further introduce acceleration techniques to help reduce the privacy cost. We rigorously analyze the convergence of our approach and utilize Renyi DP to tightly account the end-to-end DP guarantee. Extensive experiments on benchmark datasets validate that our approach outperforms previous differentially-private FL approaches in both privacy guarantee and communication efficiency.

----

## [202] Temporal Heterogeneous Information Network Embedding

**Authors**: *Hong Huang, Ruize Shi, Wei Zhou, Xiao Wang, Hai Jin, Xiaoming Fu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/203](https://doi.org/10.24963/ijcai.2021/203)

**Abstract**:

Heterogeneous information network (HIN) embedding, learning the low-dimensional representation of multi-type nodes, has been applied widely and achieved excellent performance. However, most of the previous works focus more on static heterogeneous networks or learning node embedding within specific snapshots, and seldom attention has been paid to the whole evolution process and capturing all temporal dynamics. In order to fill the gap of obtaining multi-type node embeddings by considering all temporal dynamics during the evolution, we propose a novel temporal HIN embedding method (THINE). THINE not only uses attention mechanism and meta-path to preserve structures and semantics in HIN but also combines the Hawkes process to simulate the evolution of the temporal network. Our extensive evaluations with various real-world temporal HINs demonstrate that THINE achieves state-of-the-art performance in both static and dynamic tasks, including node classification, link prediction, and temporal link recommendation.

----

## [203] Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning

**Authors**: *Ming Jin, Yizhen Zheng, Yuan-Fang Li, Chen Gong, Chuan Zhou, Shirui Pan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/204](https://doi.org/10.24963/ijcai.2021/204)

**Abstract**:

Graph representation learning plays a vital role in processing graph-structured data. However, prior arts on graph representation learning heavily rely on labeling information. To overcome this problem, inspired by the recent success of graph contrastive learning and Siamese networks in visual representation learning, we propose a novel self-supervised approach in this paper to learn node representations by enhancing Siamese self-distillation with multi-scale contrastive learning. Specifically, we first generate two augmented views from the input graph based on local and global perspectives. Then, we employ two objectives called cross-view and cross-network contrastiveness to maximize the agreement between node representations across different views and networks. To demonstrate the effectiveness of our approach, we perform empirical experiments on five real-world datasets. Our method not only achieves new state-of-the-art results but also surpasses some semi-supervised counterparts by large margins. Code is made available at https://github.com/GRAND-Lab/MERIT

----

## [204] Practical One-Shot Federated Learning for Cross-Silo Setting

**Authors**: *Qinbin Li, Bingsheng He, Dawn Song*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/205](https://doi.org/10.24963/ijcai.2021/205)

**Abstract**:

Federated learning enables multiple parties to collaboratively learn a model without exchanging their data. While most existing federated learning algorithms need many rounds to converge, one-shot federated learning (i.e., federated learning with a single communication round) is a promising approach to make federated learning applicable in cross-silo setting in practice. However, existing one-shot algorithms only support specific models and do not provide any privacy guarantees, which significantly limit the applications in practice. In this paper, we propose a practical one-shot federated learning algorithm named FedKT. By utilizing the knowledge transfer technique, FedKT can be applied to any classification models and can flexibly achieve differential privacy guarantees. Our experiments on various tasks show that FedKT can significantly outperform the other state-of-the-art federated learning algorithms with a single communication round.

----

## [205] Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation

**Authors**: *Yang Li, Tong Chen, Yadan Luo, Hongzhi Yin, Zi Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/206](https://doi.org/10.24963/ijcai.2021/206)

**Abstract**:

Being an indispensable component in location-based social networks, next point-of-interest (POI) recommendation recommends users unexplored POIs based on their recent visiting histories. However, existing work mainly models check-in data as isolated POI sequences, neglecting the crucial collaborative signals from cross-sequence check-in information. Furthermore, the sparse POI-POI transitions restrict the ability of a model to learn effective sequential patterns for recommendation. In this paper, we propose Sequence-to-Graph (Seq2Graph) augmentation for each POI sequence, allowing collaborative signals to be propagated from correlated POIs belonging to other sequences. We then devise a novel Sequence-to-Graph POI Recommender (SGRec), which jointly learns POI embeddings and infers a user's temporal preferences from the graph-augmented POI sequence. To overcome the sparsity of POI-level interactions, we further infuse category-awareness into SGRec with a multi-task learning scheme that captures the denser category-wise transitions. As such, SGRec makes full use of the collaborative signals for learning expressive POI representations, and also comprehensively uncovers multi-level sequential patterns for user preference modelling. Extensive experiments on two real-world datasets demonstrate the superiority of SGRec against state-of-the-art methods in next POI recommendation.

----

## [206] Modeling Trajectories with Neural Ordinary Differential Equations

**Authors**: *Yuxuan Liang, Kun Ouyang, Hanshu Yan, Yiwei Wang, Zekun Tong, Roger Zimmermann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/207](https://doi.org/10.24963/ijcai.2021/207)

**Abstract**:

Recent advances in location-acquisition techniques have generated massive spatial trajectory data. Recurrent Neural Networks (RNNs) are modern tools for modeling such trajectory data. After revisiting RNN-based methods for trajectory modeling, we expose two common critical drawbacks in the existing uses. First, RNNs are discrete-time models that only update the hidden states upon the arrival of new observations, which makes them an awkward fit for learning real-world trajectories with continuous-time dynamics. Second, real-world trajectories are never perfectly accurate due to unexpected sensor noise. Most RNN-based approaches are deterministic and thereby vulnerable to such noise. To tackle these challenges, we devise a novel method entitled TrajODE for more natural modeling of trajectories. It combines the continuous-time characteristic of Neural Ordinary Differential Equations (ODE) with the robustness of stochastic latent spaces. Extensive experiments on the task of trajectory classification demonstrate the superiority of our framework against the RNN counterparts.

----

## [207] RCA: A Deep Collaborative Autoencoder Approach for Anomaly Detection

**Authors**: *Boyang Liu, Ding Wang, Kaixiang Lin, Pang-Ning Tan, Jiayu Zhou*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/208](https://doi.org/10.24963/ijcai.2021/208)

**Abstract**:

Unsupervised anomaly detection plays a crucial role in many critical applications. Driven by the success of deep learning, recent years have witnessed growing interests in applying deep neural networks (DNNs) to anomaly detection problems. A common approach is using autoencoders to learn a feature representation for the normal observations in the data. The reconstruction error of the autoencoder is then used as outlier scores to detect the anomalies. However, due to the high complexity brought upon by the over-parameterization of DNNs, the reconstruction error of the anomalies could also be small, which hampers the effectiveness of these methods. To alleviate this problem, we propose a robust framework using collaborative autoencoders to jointly identify normal observations from the data while learning its feature representation. We investigate the theoretical properties of the framework and empirically show its outstanding performance as compared to other DNN-based methods. Our experimental results also show the resiliency of the framework to missing values compared to other baseline methods.

----

## [208] MG-DVD: A Real-time Framework for Malware Variant Detection Based on Dynamic Heterogeneous Graph Learning

**Authors**: *Chen Liu, Bo Li, Jun Zhao, Ming Su, Xudong Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/209](https://doi.org/10.24963/ijcai.2021/209)

**Abstract**:

Detecting the newly emerging malware variants in real time is crucial for mitigating cyber risks and proactively blocking intrusions.  In this paper, we propose  MG-DVD,  a novel detection framework based on dynamic heterogeneous graph learning, to detect malware variants in real time.  Particularly, MG-DVD first models the fine-grained execution event streams of malware variants into dynamic heterogeneous graphs and investigates real-world meta-graphs between malware objects,  which can effectively characterize more discriminative malicious evolutionary patterns between malware and their variants. Then,  MG-DVD presents two dynamic walk-based heterogeneous graph learning methods to learn more comprehensive representations of malware variants,  which significantly reduces the cost of the entire graph retraining. As a result, MG-DVD  is equipped with the ability to detect malware variants in real time, and it presents better interpretability by introducing meaningful meta-graphs. Comprehensive experiments on large-scale samples prove that our proposed  MG-DVD  outperforms state-of-the-art methods in detecting malware variants in terms of effectiveness and efficiency.

----

## [209] Node-wise Localization of Graph Neural Networks

**Authors**: *Zemin Liu, Yuan Fang, Chenghao Liu, Steven C. H. Hoi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/210](https://doi.org/10.24963/ijcai.2021/210)

**Abstract**:

Graph neural networks (GNNs) emerge as a powerful family of representation learning models on graphs. To derive node representations, they utilize a global model that recursively aggregates information from the neighboring nodes. However, different nodes reside at different parts of the graph in different local contexts, making their distributions vary across the graph. Ideally, how a node receives its neighborhood information should be a function of its local context, to diverge from the global GNN model shared by all nodes. To utilize node locality without overfitting, we propose a node-wise localization of GNNs by accounting for both global and local aspects of the graph. Globally, all nodes on the graph depend on an underlying global GNN to encode the general patterns across the graph; locally, each node is localized into a unique model as a function of the global model and its local context. Finally, we conduct extensive experiments on four benchmark graphs, and consistently obtain promising performance surpassing the state-of-the-art GNNs.

----

## [210] GraphReach: Position-Aware Graph Neural Network using Reachability Estimations

**Authors**: *Sunil Nishad, Shubhangi Agarwal, Arnab Bhattacharya, Sayan Ranu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/211](https://doi.org/10.24963/ijcai.2021/211)

**Abstract**:

Majority of the existing graph neural networks(GNN) learn node embeddings that encode their local neighborhoods but not their positions. Consequently, two nodes that are vastly distant but located in similar local neighborhoods map to similar embeddings in those networks. This limitation prevents accurate performance in predictive tasks that rely on position information. In this paper, we develop GRAPHREACH , a position-aware inductive GNN that captures the global positions of nodes through reachability estimations with respect to a set of anchor nodes. The anchors are strategically selected so that reachability estimations across all the nodes are maximized. We show that this combinatorial anchor selection problem is NP-hard and, consequently, develop a greedy (1âˆ’1/e) approximation heuristic. Empirical evaluation against state-of-the-art GNN architectures reveal that GRAPHREACH provides up to 40% relative improvement in accuracy. In addition, it is more robust to adversarial attacks.

----

## [211] Graph Edit Distance Learning via Modeling Optimum Matchings with Constraints

**Authors**: *Yun Peng, Byron Choi, Jianliang Xu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/212](https://doi.org/10.24963/ijcai.2021/212)

**Abstract**:

Graph edit distance (GED) is a fundamental measure for graph similarity analysis in many real applications. GED computation has known to be NP-hard and many heuristic methods are proposed. GED has two inherent characteristics: multiple optimum node matchings and one-to-one node matching constraints. However, these two characteristics have not been well considered in the existing learning-based methods, which leads to suboptimal models. In this paper, we propose a novel GED-specific loss function that simultaneously encodes the two characteristics. First, we propose an optimal partial node matching-based regularizer to encode multiple optimum node matchings. Second, we propose a plane intersection-based regularizer to impose the one-to-one constraints for the encoded node matchings. We use the graph neural network on the association graph of the two input graphs to learn the cross-graph representation. Our experiments show that our method is 4.2x-103.8x more accurate than the state-of-the-art methods on real-world benchmark graphs.

----

## [212] GAEN: Graph Attention Evolving Networks

**Authors**: *Min Shi, Yu Huang, Xingquan Zhu, Yufei Tang, Yuan Zhuang, Jianxun Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/213](https://doi.org/10.24963/ijcai.2021/213)

**Abstract**:

Real-world networked systems often show dynamic properties with continuously evolving network nodes and topology over time. When learning from dynamic networks, it is beneficial to correlate all temporal networks to fully capture the similarity/relevance between nodes. Recent work for dynamic network representation learning typically trains each single network independently and imposes relevance regularization on the network learning at different time steps. Such a snapshot scheme fails to leverage topology similarity between temporal networks for progressive training. In addition to the static node relationships within each network, nodes could show similar variation patterns (e.g., change of local structures) within the temporal network sequence. Both static node structures and temporal variation patterns can be combined to better characterize node affinities for unified embedding learning. In this paper, we propose Graph Attention Evolving Networks (GAEN) for dynamic network embedding with preserved similarities between nodes  derived from their temporal variation patterns. Instead of training graph attention weights for each network independently, we allow model weights to share and evolve across all temporal networks based on their respective topology discrepancies. Experiments and validations, on four real-world dynamic graphs, demonstrate that GAEN outperforms the state-of-the-art in both link prediction and node classification tasks.

----

## [213] Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification

**Authors**: *Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjing Wang, Yu Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/214](https://doi.org/10.24963/ijcai.2021/214)

**Abstract**:

Graph neural network (GNN) and label propagation algorithm (LPA) are both message passing algorithms, which have achieved superior performance in semi-supervised classification. GNN performs feature propagation by a neural network to make predictions, while LPA uses label propagation across graph adjacency matrix to get results. However, there is still no effective way to directly combine these two kinds of algorithms. To address this issue, we propose a novel Unified Message Passaging Model (UniMP) that can incorporate feature and label propagation at both training and inference time. First, UniMP adopts a Graph Transformer network, taking feature embedding and label embedding as input information for propagation. Second, to train the network without overfitting in self-loop input label information, UniMP introduces a masked label prediction strategy, in which some percentage of input label information are masked at random, and then predicted. UniMP conceptually unifies feature propagation and label propagation and is empirically powerful. It obtains new state-of-the-art semi-supervised classification results in Open Graph Benchmark (OGB).

----

## [214] Keyword-Based Knowledge Graph Exploration Based on Quadratic Group Steiner Trees

**Authors**: *Yuxuan Shi, Gong Cheng, Trung-Kien Tran, Jie Tang, Evgeny Kharlamov*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/215](https://doi.org/10.24963/ijcai.2021/215)

**Abstract**:

Exploring complex structured knowledge graphs (KGs) is challenging for non-experts as it requires knowledge of query languages and the underlying structure of the KGs. Keyword-based exploration is a convenient paradigm, and computing a group Steiner tree (GST) as an answer is a popular implementation. Recent studies suggested improving the cohesiveness of an answer where entities have small semantic distances from each other. However, how to efficiently compute such an answer is open. In this paper, to model cohesiveness in a generalized way, the quadratic group Steiner tree problem (QGSTP) is formulated where the cost function extends GST with quadratic terms representing semantic distances. For QGSTP we design a branch-and-bound best-first (B3F) algorithm where we exploit combinatorial methods to estimate lower bounds for costs. This exact algorithm shows practical performance on medium-sized KGs.

----

## [215] Federated Model Distillation with Noise-Free Differential Privacy

**Authors**: *Lichao Sun, Lingjuan Lyu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/216](https://doi.org/10.24963/ijcai.2021/216)

**Abstract**:

Conventional federated learning directly averages model weights, which is only possible for collaboration between models with homogeneous architectures. Sharing prediction instead of weight removes this obstacle and eliminates the risk of white-box inference attacks in conventional federated learning. However, the predictions from local models are sensitive and would leak training data privacy to the public. To address this issue, one naive approach is adding the differentially private random noise to the predictions, which however brings a substantial trade-off between privacy budget and model performance. In this paper, we propose a novel framework called FEDMD-NFDP, which applies a Noise-FreeDifferential Privacy (NFDP) mechanism into a federated model distillation framework. Our extensive experimental results on various datasets validate that FEDMD-NFDP can deliver not only comparable utility and communication efficiency but also provide a noise-free differential privacy guarantee. We also demonstrate the feasibility of our FEDMD-NFDP by considering both IID and Non-IID settings, heterogeneous model architectures, and unlabelled public datasets from a different distribution.

----

## [216] LDP-FL: Practical Private Aggregation in Federated Learning with Local Differential Privacy

**Authors**: *Lichao Sun, Jianwei Qian, Xun Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/217](https://doi.org/10.24963/ijcai.2021/217)

**Abstract**:

Training deep learning models on sensitive user data has raised increasing privacy concerns in many areas. Federated learning is a popular approach for privacy protection that collects the local gradient information instead of raw data. One way to achieve a strict privacy guarantee is to apply local differential privacy into federated learning. However, previous works do not give a practical solution due to two issues. First, the range difference of weights in different deep learning model layers has not been explicitly considered when applying local differential privacy mechanism. Second, the privacy budget explodes due to the high dimensionality of weights in deep learning models and many query iterations of federated learning. In this paper, we proposed a novel design of local differential privacy mechanism for federated learning to address the abovementioned issues. It makes the local weights update differentially private by adapting to the varying ranges at different layers of a deep neural network, which introduces a smaller variance of the estimated model weights, especially for deeper models. Moreover, the proposed mechanism bypasses the curse of dimensionality by parameter shuffling aggregation. A series of empirical evaluations on three commonly used datasets in prior differential privacy works, MNIST, Fashion-MNIST and CIFAR-10, demonstrate that our solution can not only achieve superior deep learning performance but also provide a strong privacy guarantee at the same time.

----

## [217] Does Every Data Instance Matter? Enhancing Sequential Recommendation by Eliminating Unreliable Data

**Authors**: *Yatong Sun, Bin Wang, Zhu Sun, Xiaochun Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/218](https://doi.org/10.24963/ijcai.2021/218)

**Abstract**:

Most sequential recommender systems (SRSs) predict next-item as target for each user given its preceding  items  as  input,  assuming  that  each  input is  related  to  its  target.   However,  users  may  unintentionally  click  on  items  that  are  inconsistent with  their  preference.   We  empirically  verify  that SRSs  can  be  misguided  with  such  unreliable  instances  (i.e.   targets  mismatch  inputs).   This  inspires  us  to  design  a  novel  SRS By Eliminating unReliable Data (BERD) guided with two observations:  (1) unreliable instances generally have high training  loss;  and  (2)  high-loss  instances  are  not necessarily  unreliable  but  uncertain  ones  caused by blurry sequential pattern.  Accordingly, BERD models both loss and uncertainty of each instance via a Gaussian distribution to better distinguish unreliable instances; meanwhile an uncertainty-aware graph  convolution  network  is  exploited  to  assist in mining unreliable instances by lowering uncertainty.   Extensive  experiments  on  four  real-world datasets  demonstrate  the  superiority  of  our  proposed BERD.

----

## [218] Cooperative Joint Attentive Network for Patient Outcome Prediction on Irregular Multi-Rate Multivariate Health Data

**Authors**: *Qingxiong Tan, Mang Ye, Grace Lai-Hung Wong, Pong Chi Yuen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/219](https://doi.org/10.24963/ijcai.2021/219)

**Abstract**:

Due to the dynamic health status of patients and discrepant stability of physiological variables, health data often presents as irregular multi-rate multivariate time series (IMR-MTS) with significantly varying sampling rates. Existing methods mainly study changes of IMR-MTS values in the time domain, without considering their different dominant frequencies and varying data quality. Hence, we propose a novel Cooperative Joint Attentive Network (CJANet) to analyze IMR-MTS in frequency domain, which adaptively handling discrepant dominant frequencies while tackling diverse data qualities caused by irregular sampling. In particular, novel dual-channel joint attention is designed to jointly identify important magnitude and phase signals while detecting their dominant frequencies, automatically enlarging the positive influence of key variables and frequencies. Furthermore, a new cooperative learning module is introduced to enhance information exchange between magnitude and phase channels, effectively integrating global signals to optimize the network. A frequency-aware fusion strategy is finally designed to aggregate the learned features. Extensive experimental results on real-world medical datasets indicate that CJANet significantly outperforms existing methods and provides highly interpretable results.

----

## [219] Pattern-enhanced Contrastive Policy Learning Network for Sequential Recommendation

**Authors**: *Xiaohai Tong, Pengfei Wang, Chenliang Li, Long Xia, Shaozhang Niu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/220](https://doi.org/10.24963/ijcai.2021/220)

**Abstract**:

Sequential recommendation aims to predict users’ future behaviors given their historical interactions. However, due to the randomness and diversity of a user’s behaviors, not all historical items are informative to tell his/her next choice. It is obvious that identifying relevant items and extracting meaningful sequential patterns are necessary for a better recommendation. Unfortunately, few works have focused on this sequence denoising process.
In this paper, we propose a PatteRn-enhanced ContrAstive Policy Learning Network (RAP for short) for sequential recommendation, RAP formalizes the denoising problem in the form of Markov Decision Process (MDP), and sample actions for each item to determine whether it is relevant with the target item. To tackle the lack of relevance supervision, RAP fuses a series of mined sequential patterns into the policy learning process, which work as a prior knowledge to guide the denoising process. After that, RAP splits the initial item sequence into two disjoint subsequences: a positive subsequence and a negative subsequence. At this, a novel contrastive learning mechanism is introduced to guide the sequence denoising and achieve preference estimation from the positive subsequence simultaneously. Extensive experiments on four public real-world datasets demonstrate the effectiveness of our approach for sequential recommendation.

----

## [220] Heuristic Search for Approximating One Matrix in Terms of Another Matrix

**Authors**: *Guihong Wan, Haim Schweitzer*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/221](https://doi.org/10.24963/ijcai.2021/221)

**Abstract**:

We study the approximation of a target matrix in terms of several selected columns of another matrix, sometimes called "a dictionary".
This approximation problem arises in various domains, such as signal processing, computer vision, and machine learning. 
An optimal column selection algorithm for the special case where the target matrix has only one column is known since the 1970's,
but most previously proposed column selection algorithms for the general case are greedy.
We propose the first nontrivial optimal algorithm for the general case, using a heuristic search setting similar to the classical A* algorithm.
We also propose practical sub-optimal algorithms in a setting similar to the classical Weighted A* algorithm.
Experimental results show that our sub-optimal algorithms compare favorably with the current state-of-the-art greedy algorithms.
They also provide bounds on how close their solutions are to the optimal solution.

----

## [221] Preference-Adaptive Meta-Learning for Cold-Start Recommendation

**Authors**: *Li Wang, Binbin Jin, Zhenya Huang, Hongke Zhao, Defu Lian, Qi Liu, Enhong Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/222](https://doi.org/10.24963/ijcai.2021/222)

**Abstract**:

In recommender systems, the cold-start problem is a critical issue. To alleviate this problem, an emerging direction adopts meta-learning frameworks and achieves success. Most existing works aim to learn globally shared prior knowledge across all users so that it can be quickly adapted to a new user with sparse interactions. However, globally shared prior knowledge may be inadequate to discern users’ complicated behaviors and causes poor generalization. Therefore, we argue that prior knowledge should be locally shared by users with similar preferences who can be recognized by social relations. To this end, in this paper, we propose a Preference-Adaptive Meta-Learning approach (PAML) to improve existing meta-learning frameworks with better generalization capacity. Specifically, to address two challenges imposed by social relations, we first identify reliable implicit friends to strengthen a user’s social relations based on our defined palindrome paths. Then, a coarse-fine preference modeling method is proposed to leverage social relations and capture the preference. Afterwards, a novel preference-specific adapter is designed to adapt the globally shared prior knowledge to the preference-specific knowledge so that users who have similar tastes share similar knowledge. We conduct extensive experiments on two publicly available datasets. Experimental results validate the power of social relations and the effectiveness of PAML.

----

## [222] Federated Learning with Fair Averaging

**Authors**: *Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang, Rongshan Yu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/223](https://doi.org/10.24963/ijcai.2021/223)

**Abstract**:

Fairness has emerged as a critical problem in federated learning (FL). In this work, we identify a cause of unfairness in FL -- conflicting gradients with large differences in the magnitudes. To address this issue, we propose the federated fair averaging (FedFV) algorithm to mitigate potential conflicts among clients before averaging their gradients. We first use the cosine similarity to detect gradient conflicts, and then iteratively eliminate such conflicts by modifying both the direction and the magnitude of the gradients. We further show the theoretical foundation of FedFV to mitigate the issue conflicting gradients and converge to Pareto stationary solutions. Extensive  experiments on a suite of federated datasets confirm that FedFV compares favorably against state-of-the-art methods in terms of fairness, accuracy and efficiency. The source code is available at https://github.com/WwZzz/easyFL.

----

## [223] User-as-Graph: User Modeling with Heterogeneous Graph Pooling for News Recommendation

**Authors**: *Chuhan Wu, Fangzhao Wu, Yongfeng Huang, Xing Xie*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/224](https://doi.org/10.24963/ijcai.2021/224)

**Abstract**:

Accurate user modeling is critical for news recommendation. Existing news recommendation methods usually model users' interest from their behaviors via sequential or attentive models. However, they cannot model the rich relatedness between user behaviors, which can provide useful contexts of these behaviors for user interest modeling.  In this paper, we propose a novel user modeling approach for news recommendation, which models each user as a personalized heterogeneous graph built from user behaviors to better capture the fine-grained behavior relatedness. In addition, in order to learn user interest embedding from the personalized heterogeneous graph, we propose a novel heterogeneous graph pooling method, which can summarize both node features and graph topology, and be aware of the varied characteristics of different types of nodes. Experiments on large-scale benchmark dataset show the proposed methods can effectively improve the performance of user modeling for news recommendation.

----

## [224] Spatial-Temporal Sequential Hypergraph Network for Crime Prediction with Dynamic Multiplex Relation Learning

**Authors**: *Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Liefeng Bo, Xiyue Zhang, Tianyi Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/225](https://doi.org/10.24963/ijcai.2021/225)

**Abstract**:

Crime prediction is crucial for public safety and resource optimization, yet is very challenging due to two aspects: i) the dynamics of criminal patterns across time and space, crime events are distributed unevenly on both spatial and temporal domains; ii) time-evolving dependencies between different types of crimes (e.g., Theft, Robbery, Assault, Damage) which reveal fine-grained semantics of crimes. To tackle these challenges, we propose Spatial-Temporal Sequential Hypergraph Network (ST-SHN) to collectively encode complex crime spatial-temporal patterns as well as the underlying category-wise crime semantic relationships. In specific, to handle spatial-temporal dynamics under the long-range and global context, we design a graph-structured message passing architecture with the integration of the hypergraph learning paradigm. To capture category-wise crime heterogeneous relations in a dynamic environment, we introduce a multi-channel routing mechanism to learn the time-evolving structural dependency across crime types. We conduct extensive experiments on two real-word datasets, showing that our proposed ST-SHN framework can significantly improve the prediction performance as compared to various state-of-the-art baselines. The source code is available at https://github.com/akaxlh/ST-SHN.

----

## [225] Heterogeneous Graph Information Bottleneck

**Authors**: *Liang Yang, Fan Wu, Zichen Zheng, Bingxin Niu, Junhua Gu, Chuan Wang, Xiaochun Cao, Yuanfang Guo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/226](https://doi.org/10.24963/ijcai.2021/226)

**Abstract**:

Most attempts on extending  Graph Neural Networks (GNNs) to Heterogeneous Information Networks (HINs)  implicitly take the direct assumption that the multiple homogeneous attributed networks induced by different meta-paths are complementary.  The doubts about the hypothesis of complementary motivate an alternative  assumption of consensus. That is, the aggregated node attributes shared by multiple homogeneous attributed networks are essential for node representations, while the specific ones in each homogeneous attributed network should be discarded. In this paper, a novel Heterogeneous Graph Information Bottleneck (HGIB) is proposed to implement the consensus hypothesis in an unsupervised manner. To this end, information bottleneck (IB) is extended to unsupervised representation learning  by leveraging self-supervision strategy.  Specifically, HGIB simultaneously maximizes the mutual information between   one homogeneous network and the representation learned from another homogeneous network, while minimizes the mutual information between the specific information contained in one homogeneous network and the representation learned from this homogeneous network. Model analysis reveals that the two extreme cases of HGIB correspond to the supervised heterogeneous GNN and the infomax on homogeneous graph, respectively. Extensive experiments on real datasets demonstrate that the consensus-based unsupervised HGIB significantly outperforms most semi-supervised SOTA methods based on complementary assumption.

----

## [226] Graph Deformer Network

**Authors**: *Wenting Zhao, Yuan Fang, Zhen Cui, Tong Zhang, Jian Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/227](https://doi.org/10.24963/ijcai.2021/227)

**Abstract**:

Convolution learning on graphs draws increasing attention recently due to its potential applications to a large amount of irregular data. Most graph convolution methods leverage the plain summation/average aggregation to avoid the discrepancy of responses from isomorphic graphs. However, such an extreme collapsing way would result in a structural loss and signal entanglement of nodes, which further cause the degradation of the learning ability. In this paper, we propose a simple yet effective Graph Deformer Network (GDN) to fulfill anisotropic convolution filtering on graphs, analogous to the standard convolution operation on images. Local neighborhood subgraphs (acting like receptive fields) with different structures are deformed into a unified virtual space, coordinated by several anchor nodes. In the deformation process, we transfer components of nodes therein into affinitive anchors by learning their correlations, and build a multi-granularity feature space calibrated with anchors. Anisotropic convolutional kernels can be further performed over the anchor-coordinated space to well encode local variations of receptive fields. By parameterizing anchors and stacking coarsening layers, we build a graph deformer network in an end-to-end fashion. Theoretical analysis indicates its connection to previous work and shows the promising property of graph isomorphism testing. Extensive experiments on widely-used datasets validate the effectiveness of GDN in graph and node classifications.

----

## [227] Knowledge-based Residual Learning

**Authors**: *Guanjie Zheng, Chang Liu, Hua Wei, Porter Jenkins, Chacha Chen, Tao Wen, Zhenhui Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/228](https://doi.org/10.24963/ijcai.2021/228)

**Abstract**:

Small data has been a barrier for many machine learning tasks, especially when applied in scientific domains. Fortunately, we can utilize domain knowledge to make up the lack of data. Hence, in this paper, we propose a hybrid model KRL that treats domain knowledge model as a weak learner and uses another neural net model to boost it. We prove that KRL is guaranteed to improve over pure domain knowledge model and pure neural net model under certain loss functions. Extensive experiments have shown the superior performance of KRL over baselines. In addition, several case studies have explained how the domain knowledge can assist the prediction.

----

## [228] Faster Guarantees of Evolutionary Algorithms for Maximization of Monotone Submodular Functions

**Authors**: *Victoria G. Crawford*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/229](https://doi.org/10.24963/ijcai.2021/229)

**Abstract**:

In this paper, the monotone submodular maximization problem (SM) is studied. SM is to find a subset of size kappa from a universe of size n that maximizes a monotone submodular objective function f . We show using a novel analysis that the Pareto optimization algorithm achieves a worst-case ratio of (1 − epsilon)(1 − 1/e) in expectation for every cardinality constraint kappa < P , where P ≤ n + 1 is an input, in O(nP ln(1/epsilon)) queries of f . In addition, a novel evolutionary algorithm called the biased Pareto optimization algorithm, is proposed that achieves a worst-case ratio of (1 − epsilon)(1 − 1/e − epsilon) in expectation for every cardinality constraint kappa < P in O(n ln(P ) ln(1/epsilon)) queries of f . Further, the biased Pareto optimization algorithm can be modified in order to achieve a a worst-case ratio of (1 − epsilon)(1 − 1/e − epsilon) in expectation for cardinality constraint kappa in O(n ln(1/epsilon)) queries of f . An empirical evaluation corroborates our theoretical analysis of the algorithms, as the algorithms exceed the stochastic greedy solution value at roughly when one would expect based upon our analysis.

----

## [229] DACBench: A Benchmark Library for Dynamic Algorithm Configuration

**Authors**: *Theresa Eimer, André Biedenkapp, Maximilian Reimer, Steven Adriaensen, Frank Hutter, Marius Lindauer*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/230](https://doi.org/10.24963/ijcai.2021/230)

**Abstract**:

Dynamic Algorithm Configuration (DAC) aims to dynamically control a target algorithm's hyperparameters in order to improve its performance.
Several theoretical and empirical results have demonstrated the benefits of dynamically controlling hyperparameters in domains like evolutionary computation, AI Planning or deep learning.
Replicating these results, as well as studying new methods for DAC, however, is difficult since existing benchmarks are often specialized and incompatible with the same interfaces.
To facilitate benchmarking and thus research on DAC, we propose DACBench, a benchmark library that seeks to collect and standardize existing DAC benchmarks from different AI domains, as well as provide a template for new ones.
For the design of DACBench, we focused on important desiderata, such as (i) flexibility, (ii) reproducibility, (iii) extensibility and (iv) automatic documentation and visualization.
To show the potential, broad applicability and challenges of DAC, we explore how a set of six initial benchmarks compare in several dimensions of difficulty.

----

## [230] Bounded-cost Search Using Estimates of Uncertainty

**Authors**: *Maximilian Fickert, Tianyi Gu, Wheeler Ruml*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/231](https://doi.org/10.24963/ijcai.2021/231)

**Abstract**:

Many planning problems are too hard to solve optimally.  In bounded-cost search, one attempts to find, as quickly as possible, a plan that costs no more than a user-provided absolute cost bound.  Several algorithms have been previously proposed for this setting, including Potential Search (PTS) and Bounded-cost Explicit Estimation Search (BEES).  BEES attempts to improve on PTS by predicting whether nodes will lead to plans within the cost bound or not.  This paper introduces a relatively simple algorithm, Expected Effort Search (XES), which uses not just point estimates but belief distributions in order to estimate the probability that a node will lead to a plan within the bound.  XES's expansion order minimizes expected search time in a simplified formal model.  Experimental results on standard planning and search benchmarks show that it consistently exhibits strong performance, outperforming both PTS and BEES.  We also derive improved variants of BEES that can exploit belief distributions.  These new methods advance the recent trend of taking advantage of uncertainty estimates in deterministic single-agent search.

----

## [231] A Runtime Analysis of Typical Decomposition Approaches in MOEA/D Framework for Many-objective Optimization Problems

**Authors**: *Zhengxin Huang, Yuren Zhou, Chuan Luo, Qingwei Lin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/232](https://doi.org/10.24963/ijcai.2021/232)

**Abstract**:

Decomposition approach is an important component in multi-objective evolutionary algorithm based on decomposition (MOEA/D), which is a popular method for handing many-objective optimization problems (MaOPs). This paper presents a theoretical analysis on the convergence ability of using the typical weighted sum (WS), Tchebycheff (TCH) or penalty-based boundary intersection (PBI) approach in a basic MOEA/D for solving two benchmark MaOPs. The results show that using WS, the algorithm can always find an optimal solution for any subproblem in polynomial expected runtime. In contrast, the algorithm needs at least exponential expected runtime for some subproblems if using TCH or PBI. Moreover, our analyses discover an obvious shortcoming of using WS, that is, the optimal solutions of different subproblems are easily corresponding to the same solution. In addition, this analysis indicates that if using PBI, a small value of the penalty parameter is a good choice for faster converging to the Pareto front, but it may lose the diversity. This study reveals some optimization behaviors of using three typical decomposition approaches in the well-known MOEA/D framework for solving MaOPs.

----

## [232] A New Upper Bound Based on Vertex Partitioning for the Maximum K-plex Problem

**Authors**: *Hua Jiang, Dongming Zhu, Zhichao Xie, Shaowen Yao, Zhang-Hua Fu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/233](https://doi.org/10.24963/ijcai.2021/233)

**Abstract**:

Given an undirected graph, the Maximum k-plex Problem (MKP) is to find a largest induced subgraph in which each vertex has at most kâˆ’1 non-adjacent vertices. The problem arises in social network analysis and has found applications in many important areas employing graph-based data mining. Existing exact algorithms usually implement a branch-and-bound approach that requires a tight upper bound to reduce the search space. In this paper, we propose a new upper bound for MKP, which is a partitioning of the candidate vertex set with respect to the constructing solution. We implement a new branch-and-bound algorithm that employs the upper bound to reduce the number of branches. Experimental results show that the upper bound is very effective in reducing the search space. The new algorithm outperforms the state-of-the-art algorithms significantly on real-world massive graphs, DIMACS graphs and random graphs.

----

## [233] Choosing the Right Algorithm With Hints From Complexity Theory

**Authors**: *Shouda Wang, Weijie Zheng, Benjamin Doerr*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/234](https://doi.org/10.24963/ijcai.2021/234)

**Abstract**:

Choosing a suitable algorithm from the myriads of different search heuristics is difficult when faced with a novel optimization problem. In this work, we argue that the purely academic question of what could be the best possible algorithm in a certain broad class of black-box optimizers can give fruitful indications in which direction to search for good established optimization heuristics. We demonstrate this approach on the recently proposed DLB benchmark, for which the only known results are O(n^3) runtimes for several classic evolutionary algorithms and an O(n^2 log n) runtime for an estimation-of-distribution algorithm. Our finding that the unary unbiased black-box complexity is only O(n^2) suggests the Metropolis algorithm as an interesting candidate and we prove that it solves the DLB problem in quadratic time. Since we also prove that better runtimes cannot be obtained in the class of unary unbiased algorithms, we shift our attention to algorithms that use the information of more parents to generate new solutions. An artificial algorithm of this type having an O(n log n) runtime leads to the result that the significance-based compact genetic algorithm (sig-cGA) can solve the DLB problem also in time O(n log n). Our experiments show a remarkably good performance of the Metropolis algorithm, clearly the best of all algorithms regarded for reasonable problem sizes.

----

## [234] UIBert: Learning Generic Multimodal Representations for UI Understanding

**Authors**: *Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, Blaise Agüera y Arcas*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/235](https://doi.org/10.24963/ijcai.2021/235)

**Abstract**:

To improve the accessibility of smart devices and to simplify their usage, building models which understand user interfaces (UIs) and assist users to complete their tasks is critical. However, unique challenges are proposed by UI-specific characteristics, such as how to effectively leverage multimodal UI features that involve image, text, and structural metadata and how to achieve good performance when high-quality labeled data is unavailable. To address such challenges we introduce UIBert, a transformer-based joint image-text model trained through novel pre-training tasks on large-scale unlabeled UI data to learn generic feature representations for a UI and its components. Our key intuition is that the heterogeneous features in a UI are self-aligned, i.e., the image and text features of UI components, are predictive of each other. We propose five pretraining tasks utilizing this self-alignment among different features of a UI component and across various components in the same UI. We evaluate our method on nine real-world downstream UI tasks where UIBert outperforms strong multimodal baselines by up to 9.26% accuracy.

----

## [235] Pruning of Deep Spiking Neural Networks through Gradient Rewiring

**Authors**: *Yanqi Chen, Zhaofei Yu, Wei Fang, Tiejun Huang, Yonghong Tian*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/236](https://doi.org/10.24963/ijcai.2021/236)

**Abstract**:

Spiking Neural Networks (SNNs) have been attached great importance due to their biological plausibility and high energy-efficiency on neuromorphic chips. As these chips are usually resource-constrained, the compression of SNNs is thus crucial along the road of practical use of SNNs. Most existing methods directly apply pruning approaches in artificial neural networks (ANNs) to SNNs, which ignore the difference between ANNs and SNNs, thus limiting the performance of the pruned SNNs. Besides, these methods are only suitable for shallow SNNs. In this paper, inspired by synaptogenesis and synapse elimination in the neural system, we propose gradient rewiring (Grad R), a joint learning algorithm of connectivity and weight for SNNs, that enables us to seamlessly optimize network structure without retraining. Our key innovation is to redefine the gradient to a new synaptic parameter, allowing better exploration of network structures by taking full advantage of the competition between pruning and regrowth of connections. The experimental results show that the proposed method achieves minimal loss of SNNs' performance on MNIST and CIFAR-10 datasets so far. Moreover, it reaches a ~3.5% accuracy loss under unprecedented 0.73% connectivity, which reveals remarkable structure refining capability in SNNs. Our work suggests that there exists extremely high redundancy in deep SNNs. Our codes are available at https://github.com/Yanqi-Chen/Gradient-Rewiring.

----

## [236] Human-AI Collaboration with Bandit Feedback

**Authors**: *Ruijiang Gao, Maytal Saar-Tsechansky, Maria De-Arteaga, Ligong Han, Min Kyung Lee, Matthew Lease*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/237](https://doi.org/10.24963/ijcai.2021/237)

**Abstract**:

Human-machine complementarity is important when neither the algorithm nor the human yield dominant performance across all instances in a given domain. Most research on algorithmic decision-making solely centers on the algorithm's performance, while recent work that explores human-machine collaboration has framed the decision-making problems as classification tasks. In this paper, we first propose and then develop a solution for a novel human-machine collaboration problem in a bandit feedback setting. Our solution aims to exploit the human-machine complementarity to maximize decision rewards. We then extend our approach to settings with multiple human decision makers. We demonstrate the effectiveness of our proposed methods using both synthetic and real human responses, and find that our methods outperform both the algorithm and the human when they each make decisions on their own. We also show how personalized routing in the presence of multiple human decision-makers can further improve the human-machine team performance.

----

## [237] Accounting for Confirmation Bias in Crowdsourced Label Aggregation

**Authors**: *Meric Altug Gemalmaz, Ming Yin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/238](https://doi.org/10.24963/ijcai.2021/238)

**Abstract**:

Collecting large-scale human-annotated datasets via crowdsourcing to train and improve automated models is a prominent human-in-the-loop approach to integrate human and machine intelligence. However, together with their unique intelligence, humans also come with their biases and subjective beliefs, which may influence the quality of the annotated data and negatively impact the effectiveness of the human-in-the-loop systems. One of the most common types of cognitive biases that humans are subject to is the confirmation bias, which is people's tendency to favor information that confirms their existing beliefs and values. In this paper, we present an algorithmic approach to infer the correct answers of tasks by aggregating the annotations from multiple crowd workers, while taking workers' various levels of confirmation bias into consideration. Evaluations on real-world crowd annotations show that the proposed bias-aware label aggregation algorithm outperforms baseline methods in accurately inferring the ground-truth labels of different tasks when crowd workers indeed exhibit some degree of confirmation bias. Through simulations on synthetic data, we further identify the conditions when the proposed algorithm has the largest advantages over baseline methods.

----

## [238] An Entanglement-driven Fusion Neural Network for Video Sentiment Analysis

**Authors**: *Dimitris Gkoumas, Qiuchi Li, Yijun Yu, Dawei Song*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/239](https://doi.org/10.24963/ijcai.2021/239)

**Abstract**:

Video data is multimodal in its nature, where an utterance can involve linguistic, visual and acoustic information. Therefore, a key challenge for video sentiment analysis is how to combine different modalities for sentiment recognition effectively. The latest neural network approaches achieve state-of-the-art performance, but they neglect to a large degree of how humans understand and reason about sentiment states. By contrast, recent advances in quantum probabilistic neural models have achieved comparable performance to the state-of-the-art, yet with better transparency and increased level of interpretability. However, the existing quantum-inspired models treat quantum states as either a classical mixture or as a separable tensor product across modalities, without triggering their interactions in a way that they are correlated or non-separable (i.e., entangled). This means that the current models have not fully exploited the expressive power of quantum probabilities. To fill this gap, we propose a transparent quantum probabilistic neural model. The model induces different modalities to interact in such a way that they may not be separable, encoding crossmodal information in the form of non-classical correlations. Comprehensive evaluation on two benchmarking datasets for video sentiment analysis shows that the model achieves significant performance improvement. We also show that the degree of non-separability between modalities optimizes the post-hoc interpretability.

----

## [239] Event-based Action Recognition Using Motion Information and Spiking Neural Networks

**Authors**: *Qianhui Liu, Dong Xing, Huajin Tang, De Ma, Gang Pan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/240](https://doi.org/10.24963/ijcai.2021/240)

**Abstract**:

Event-based cameras have attracted increasing attention due to their advantages of biologically inspired paradigm and low power consumption. Since event-based cameras record the visual input as asynchronous discrete events, they are inherently suitable to cooperate with the spiking neural network (SNN). Existing works of SNNs for processing events mainly focus on the task of object recognition. However, events from the event-based camera are triggered by dynamic changes, which makes it an ideal choice to capture actions in the visual scene. Inspired by the dorsal stream in visual cortex, we propose a hierarchical SNN architecture for event-based action recognition using motion information. Motion features are extracted and utilized from events to local and finally to global perception for action recognition. To the best of the authorsâ€™ knowledge, it is the first attempt of SNN to apply motion information to event-based action recognition. We evaluate our proposed SNN on three event-based action recognition datasets, including our newly published DailyAction-DVS dataset comprising 12 actions collected under diverse recording conditions. Extensive experimental results show the effectiveness of motion information and our proposed SNN architecture for event-based action recognition.

----

## [240] Item Response Ranking for Cognitive Diagnosis

**Authors**: *Shiwei Tong, Qi Liu, Runlong Yu, Wei Huang, Zhenya Huang, Zachary A. Pardos, Weijie Jiang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/241](https://doi.org/10.24963/ijcai.2021/241)

**Abstract**:

Cognitive diagnosis, a fundamental task in education area, aims at providing an approach to reveal the proficiency level of students on knowledge concepts. Actually, monotonicity is one of the basic conditions in cognitive diagnosis theory, which assumes that student's proficiency is monotonic with the probability of giving the right response to a test item. However, few of previous methods consider the monotonicity during optimization. To this end, we propose Item Response Ranking framework (IRR), aiming at introducing pairwise learning into cognitive diagnosis to well model the monotonicity between item responses. Specifically, we first use an item specific sampling method to sample item responses and construct response pairs based on their partial order, where we propose the two-branch sampling methods to handle the unobserved responses. After that, we use a pairwise objective function to exploit the monotonicity in the pair formulation. In fact, IRR is a general framework which can be applied to most of contemporary cognitive diagnosis models. Extensive experiments demonstrate the effectiveness and interpretability of our method.

----

## [241] Type Anywhere You Want: An Introduction to Invisible Mobile Keyboard

**Authors**: *Sahng-Min Yoo, Ue-Hwan Kim, Yewon Hwang, Jong-Hwan Kim*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/242](https://doi.org/10.24963/ijcai.2021/242)

**Abstract**:

Contemporary soft keyboards possess limitations: the lack of physical feedback results in an increase of typos, and the interface of soft keyboards degrades the utility of the screen. To overcome these limitations, we propose an Invisible Mobile Keyboard (IMK), which lets users freely type on the desired area without any constraints. To facilitate a data-driven IMK decoding task, we have collected the most extensive text-entry dataset (approximately 2M pairs of typing positions and the corresponding characters). Additionally, we propose our baseline decoder along with a semantic typo correction mechanism based on self-attention, which decodes such unconstrained inputs with high accuracy (96.0%). Moreover, the user study reveals that the users could type faster and feel convenience and satisfaction to IMK with our decoder. Lastly, we make the source code and the dataset public to contribute to the research community.

----

## [242] Best-Effort Synthesis: Doing Your Best Is Not Harder Than Giving Up

**Authors**: *Benjamin Aminof, Giuseppe De Giacomo, Sasha Rubin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/243](https://doi.org/10.24963/ijcai.2021/243)

**Abstract**:

We study best-effort synthesis under environment assumptions specified in LTL, and show that this problem has exactly the same computational complexity of standard LTL synthesis: 2EXPTIME-complete. We provide optimal algorithms for computing best-effort strategies, both in the case of LTL over infinite traces and LTL over finite traces (i.e., LTLf). The latter are particularly well suited for implementation.

----

## [243] A Game-Theoretic Account of Responsibility Allocation

**Authors**: *Christel Baier, Florian Funke, Rupak Majumdar*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/244](https://doi.org/10.24963/ijcai.2021/244)

**Abstract**:

When designing or analyzing multi-agent systems, a fundamental problem is responsibility ascription: to specify which agents are responsible for the joint outcome of their behaviors and to which extent. We model strategic multi-agent interaction as an extensive form game of imperfect information and define notions of forward (prospective) and backward (retrospective) responsibility. Forward responsibility identifies the responsibility of a group of agents for an outcome along all possible plays, whereas backward responsibility identifies the responsibility along a given play. We further distinguish between strategic and causal backward responsibility, where the former captures the epistemic knowledge of players along a play, while the latter formalizes which players – possibly unknowingly – caused the outcome. A formal connection between forward and backward notions is established in the case of perfect recall. We further ascribe quantitative responsibility through cooperative game theory. We show through a number of examples that our approach encompasses several prior formal accounts of responsibility attribution.

----

## [244] On Cycles, Attackers and Supporters - A Contribution to The Investigation of Dynamics in Abstract Argumentation

**Authors**: *Ringo Baumann, Markus Ulbricht*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/245](https://doi.org/10.24963/ijcai.2021/245)

**Abstract**:

Abstract argumentation as defined by Dung in his seminal 1995 paper is by now a major research area in knowledge representation and reasoning. Dynamics of abstract argumentation frameworks (AFs) as well as syntactical consequences of semantical facts of them are the central issues of this paper. The first main part is engaged with the systematical study of the influence of attackers and supporters regarding the acceptability status of whole sets and/or single arguments. In particular, we investigate the impact of addition or removal of arguments, a line of research that has been around for more than a decade. Apart from entirely new results, we revisit, generalize and sum up similar results from the literature. To gain a comprehensive formal and intuitive understanding of the behavior of AFs we put special effort in comparing different kind of semantics. We concentrate on classical admissibility-based semantics and also give pointers to semantics based on naivity and weak admissibility, a recently introduced mediating approach. In the second main part we show how to infer syntactical information from semantical one. For instance, it is well-known that if a finite AF possesses no stable extension, then it has to contain an odd-cycle. In this paper, we even present a characterization of this issue. Moreover, we show that the change of the number of extensions if adding or removing an argument allows to conclude the existence of certain even or odd cycles in the considered AF without having further information.

----

## [245] Reasoning About Agents That May Know Other Agents' Strategies

**Authors**: *Francesco Belardinelli, Sophia Knight, Alessio Lomuscio, Bastien Maubert, Aniello Murano, Sasha Rubin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/246](https://doi.org/10.24963/ijcai.2021/246)

**Abstract**:

We study the semantics of knowledge in strategic reasoning. Most existing works either implicitly assume that agents do not know one another’s strategies, or that all strategies are known to all; and some works present inconsistent mixes of both features. We put forward a novel semantics for Strategy Logic with Knowledge that cleanly models whose strategies each agent knows. We study how adopting this semantics impacts agents’ knowledge and strategic ability, as well as the complexity of the model-checking problem.

----

## [246] Choice Logics and Their Computational Properties

**Authors**: *Michael Bernreiter, Jan Maly, Stefan Woltran*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/247](https://doi.org/10.24963/ijcai.2021/247)

**Abstract**:

Qualitative Choice Logic (QCL) and Conjunctive Choice Logic (CCL) are formalisms for preference handling, with especially QCL being well established in the field of AI. So far, analyses of these logics need to be done on a case-by-case basis, albeit they share several common features. This calls for a more general choice logic framework, with QCL and CCL as well as some of their derivatives being particular instantiations. We provide such a framework, which allows us, on the one hand, to easily define new choice logics and, on the other hand, to examine properties of different choice logics in a uniform setting. In particular, we investigate strong equivalence, a core concept in non-classical logics for understanding formula simplification, and computational complexity. Our analysis also yields new results for QCL and CCL. For example, we show that the main reasoning task regarding preferred models is ϴ₂P-complete for QCL and CCL, while being Δ₂P-complete for a newly introduced choice logic.

----

## [247] Cardinality Queries over DL-Lite Ontologies

**Authors**: *Meghyn Bienvenu, Quentin Manière, Michaël Thomazo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/248](https://doi.org/10.24963/ijcai.2021/248)

**Abstract**:

Ontology-mediated query answering (OMQA) employs structured knowledge and automated reasoning in order to facilitate access to incomplete and possibly heterogeneous data. While most research on OMQA adopts (unions of) conjunctive queries as the query language, there has been recent interest in handling queries that involve counting. In this paper, we advance this line of research by investigating cardinality queries (which correspond to Boolean atomic counting queries) coupled with DL-Lite ontologies. Despite its apparent simplicity, we show that such an OMQA setting gives rise to rich and complex behaviour. While we prove that cardinality query answering is tractable (TC0) in data complexity when the ontology is formulated in DL-Lite-core, the problem becomes coNP-hard as soon as role inclusions are allowed. For DL-Lite-pos-H (which allows only positive axioms), we establish a P-coNP dichotomy and pinpoint the TC0 cases; for DL-Lite-core-H (allowing also negative axioms), we identify new sources of coNP complexity and also exhibit L-complete cases. Interestingly, and in contrast to related tractability results, we observe that the canonical model may not give the optimal count value in the tractable cases, which led us to develop an entirely new approach based upon exploring a space of strategies to determine the minimum possible number of query matches.

----

## [248] Budget-Constrained Coalition Strategies with Discounting

**Authors**: *Lia Bozzone, Pavel Naumov*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/249](https://doi.org/10.24963/ijcai.2021/249)

**Abstract**:

Discounting future costs and rewards is a common practice in accounting, game theory, and machine learning. In spite of this, existing logics for reasoning about strategies with cost and resource constraints do not account for discounting. The paper proposes a sound and complete logical system for reasoning about budget-constrained strategic abilities that incorporates discounting into its semantics.

----

## [249] Abductive Learning with Ground Knowledge Base

**Authors**: *Le-Wen Cai, Wang-Zhou Dai, Yu-Xuan Huang, Yu-Feng Li, Stephen H. Muggleton, Yuan Jiang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/250](https://doi.org/10.24963/ijcai.2021/250)

**Abstract**:

Abductive Learning is a framework that combines machine learning with first-order logical reasoning. It allows machine learning models to exploit complex symbolic domain knowledge represented by first-order logic rules. However, it is challenging to obtain or express the ground-truth domain knowledge explicitly as first-order logic rules in many applications. The only accessible knowledge base is implicitly represented by groundings, i.e., propositions or atomic formulas without variables. This paper proposes Grounded Abductive Learning (GABL) to enhance machine learning models with abductive reasoning in a ground domain knowledge base, which offers inexact supervision through a set of logic propositions. We apply GABL on two weakly supervised learning problems and found that the model's initial accuracy plays a crucial role in learning. The results on a real-world OCR task show that GABL can significantly reduce the effort of data labeling than the compared methods.

----

## [250] Intensional and Extensional Views in DL-Lite Ontologies

**Authors**: *Marco Console, Giuseppe De Giacomo, Maurizio Lenzerini, Manuel Namici*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/251](https://doi.org/10.24963/ijcai.2021/251)

**Abstract**:

The use of virtual collections of data is often essential in several
data and knowledge management tasks. In the literature, the standard
way to define virtual data collections is via views, i.e., virtual
relations defined using queries.  In data and knowledge bases, the
notion of views is a staple of data access, data integration and
exchange, query optimization, and data privacy.  In this work, we
study views in Ontology-Based Data Access (OBDA) systems.
OBDA is a powerful paradigm for accessing data through an
ontology, i.e., a conceptual specification of the domain of interest
written using logical axioms. Intuitively, users of an OBDA system
interact with the data only through the ontology's conceptual lens. We
present a novel framework to express natural and sophisticated forms
of views in OBDA systems and introduce fundamental reasoning tasks for
these views. We study the computational complexity of these tasks and
present classes of views for which these tasks are tractable or at
least decidable.

----

## [251] On Belief Change for Multi-Label Classifier Encodings

**Authors**: *Sylvie Coste-Marquis, Pierre Marquis*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/252](https://doi.org/10.24963/ijcai.2021/252)

**Abstract**:

An important issue in ML consists in developing approaches exploiting background knowledge T for improving the accuracy and the robustness of learned classifiers C. Delegating the classification task to a Boolean circuit Σ exhibiting the same input-output behaviour as C, the problem of exploiting T within C can be viewed as a belief change scenario. However, usual change operations are not suited to the task of modifying the classifier encoding Σ in a minimal way, to make it complying with T. To fill the gap, we present a new belief change operation, called rectification. We characterize the family of rectification operators from an axiomatic perspective and exhibit operators from this family. We identify the standard belief change postulates that every rectification operator satisfies and those it does not. We also focus on some computational aspects of rectification and compliance.

----

## [252] A Uniform Abstraction Framework for Generalized Planning

**Authors**: *Zhenhe Cui, Yongmei Liu, Kailun Luo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/253](https://doi.org/10.24963/ijcai.2021/253)

**Abstract**:

Generalized planning aims at finding a general solution for a set of similar planning problems. Abstractions are widely used to solve such problems. However, the connections among these abstraction works remain vague. Thus, to facilitate a deep understanding and further exploration of abstraction approaches for generalized planning, it is important to develop a uniform abstraction framework for generalized planning. Recently, Banihashemi et al. proposed an agent abstraction framework based on the situation calculus. However, expressiveness of such an abstraction framework is limited. In this paper, by extending their abstraction framework, we propose a uniform abstraction framework for generalized planning. We formalize a generalized planning problem as a triple of a basic action theory, a trajectory constraint, and a goal. Then we define the concepts of sound abstractions of a generalized planning problem. We show that solutions to a generalized planning problem are nicely related to those of its sound abstractions. We also define and analyze the dual notion of complete abstractions. Finally, we review some important abstraction works for generalized planning and show that they can be formalized in our framework.

----

## [253] Abductive Knowledge Induction from Raw Data

**Authors**: *Wang-Zhou Dai, Stephen H. Muggleton*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/254](https://doi.org/10.24963/ijcai.2021/254)

**Abstract**:

For many reasoning-heavy tasks with raw inputs, it is challenging to design an appropriate end-to-end pipeline to formulate the problem-solving process. Some modern AI systems, e.g., Neuro-Symbolic Learning, divide the pipeline into sub-symbolic perception and symbolic reasoning, trying to utilise data-driven machine learning and knowledge-driven problem-solving simultaneously. However, these systems suffer from the exponential computational complexity caused by the interface between the two components, where the sub-symbolic learning model lacks direct supervision, and the symbolic model lacks accurate input facts. Hence, they usually focus on learning the sub-symbolic model with a complete symbolic knowledge base while avoiding a crucial problem: where does the knowledge come from? In this paper, we present Abductive Meta-Interpretive Learning (MetaAbd) that unites abduction and induction to learn neural networks and logic theories jointly from raw data. Experimental results demonstrate that MetaAbd not only outperforms the compared systems in predictive accuracy and data efficiency but also induces logic programs that can be re-used as background knowledge in subsequent learning tasks. To the best of our knowledge, MetaAbd is the first system that can jointly learn neural networks from scratch and induce recursive first-order logic theories with predicate invention.

----

## [254] Finite-Trace and Generalized-Reactivity Specifications in Temporal Synthesis

**Authors**: *Giuseppe De Giacomo, Antonio Di Stasio, Lucas M. Tabajara, Moshe Y. Vardi, Shufang Zhu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/255](https://doi.org/10.24963/ijcai.2021/255)

**Abstract**:

Linear Temporal Logic (LTL) synthesis aims at automatically synthesizing a program that complies with desired properties expressed in LTL. Unfortunately it has been proved to be too difficult computationally to perform full LTL synthesis. There have been two success stories with LTL synthesis, both having to do with the form of the specification. The first is the GR(1) approach: use safety conditions to determine the possible transitions in a game between the environment and the agent, plus one powerful notion of fairness, Generalized Reactivity(1), or GR(1). The second, inspired by AI planning, is focusing on finite-trace temporal synthesis, with LTLf (LTL on finite traces) as the specification language. In this paper we take these two lines of work and bring them together. We first study the case in which we have an LTLf agent goal and a GR(1) assumption. We then add to the framework safety conditions for both the environment and the agent, obtaining a highly expressive yet still scalable form of LTL synthesis.

----

## [255] HyperLDLf: a Logic for Checking Properties of Finite Traces Process Logs

**Authors**: *Giuseppe De Giacomo, Paolo Felli, Marco Montali, Giuseppe Perelli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/256](https://doi.org/10.24963/ijcai.2021/256)

**Abstract**:

Temporal logics over finite traces, such as LTLf and its extension LDLf, have been adopted in several areas, including Business Process Management (BPM),  to check properties of processes whose executions have an unbounded, but finite, length. These logics express properties of single traces in isolation, however, especially in BPM it is also of interest to express properties over the entire log, i.e., properties that relate multiple traces of the log at once.  In the case of infinite-traces, HyperLTL has been proposed to express these ``hyper'' properties. In this paper, motivated by BPM, we introduce HyperLDLf, a logic that extends LDLf with the hyper features of HyperLTL. We provide a sound, complete and computationally optimal technique, based on DFAs manipulation, for the model checking problem in the relevant case where the set of traces (i.e., the log) is a regular language. We illustrate how this form of model checking can be used for verifying log of business processes and for advanced forms of process mining.

----

## [256] How Hard to Tell? Complexity of Belief Manipulation Through Propositional Announcements

**Authors**: *Thomas Eiter, Aaron Hunter, François Schwarzentruber*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/257](https://doi.org/10.24963/ijcai.2021/257)

**Abstract**:

Consider a set of agents with initial beliefs and a formal operator for incorporating new information. Now suppose that, for each agent, we have a formula that we would like them to believe.  Does there exist a single announcement that will lead all agents to believe the corresponding formula? This paper studies the problem of the existence of such an announcement in the context of model-preference definable revision operators. First, we provide two characterisation theorems for the existence of announcements: one in the general case, the other for total partial orderings. Second, we exploit the characterisation theorems to provide upper bound complexity results. Finally, we also provide matching optimal lower bounds for the Dalal and Ginsberg operators.

----

## [257] Improved Algorithms for Allen's Interval Algebra: a Dynamic Programming Approach

**Authors**: *Leif Eriksson, Victor Lagerkvist*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/258](https://doi.org/10.24963/ijcai.2021/258)

**Abstract**:

The constraint satisfaction problem (CSP) is an important framework in artificial intelligence used to model e.g. qualitative reasoning problems such as Allen's interval algebra A. There is strong practical incitement to solve CSPs as efficiently as possible, and the classical complexity of temporal CSPs, including A, is well understood. However, the situation is more dire with respect to running time bounds of the form O(f(n)) (where n is the number of variables) where existing results gives a best theoretical upper bound 2^O(n * log n) which leaves a significant gap to the best (conditional) lower bound 2^o(n). In this paper we narrow this gap by presenting two novel algorithms for temporal CSPs based on dynamic programming. The first algorithm solves temporal CSPs limited to constraints of arity three in O(3^n) time, and we use this algorithm to solve A in O((1.5922n)^n) time. The second algorithm tackles A directly and solves it in O((1.0615n)^n), implying a remarkable improvement over existing methods since no previously published algorithm belongs to O((cn)^n) for any c. We also extend the latter algorithm to higher dimensions box algebras where we obtain the first explicit upper bound.

----

## [258] Decomposition-Guided Reductions for Argumentation and Treewidth

**Authors**: *Johannes Klaus Fichte, Markus Hecher, Yasir Mahmood, Arne Meier*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/259](https://doi.org/10.24963/ijcai.2021/259)

**Abstract**:

Argumentation is a widely applied framework for modeling and evaluating arguments and its reasoning with various applications. Popular frameworks are abstract argumentation (Dung’s framework) or logic-based argumentation (Besnard-Hunter’s framework). Their computational complexity has been studied quite in-depth. Incorporating treewidth into the complexity analysis is particularly interesting, as solvers oftentimes employ SAT-based solvers, which can solve instances of low treewidth fast. In this paper, we address whether one can design reductions from argumentation problems to SAT-problems while linearly preserving the treewidth, which results in decomposition-guided (DG) reductions. It turns out that the linear treewidth overhead caused by our DG reductions, cannot be significantly improved under reasonable assumptions. Finally, we consider logic-based argumentation and establish new upper bounds using DG reductions and lower bounds.

----

## [259] Actively Learning Concepts and Conjunctive Queries under ELr-Ontologies

**Authors**: *Maurice Funk, Jean Christoph Jung, Carsten Lutz*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/260](https://doi.org/10.24963/ijcai.2021/260)

**Abstract**:

We consider the problem to learn a concept or a query in the presence of an ontology formulated in the description logic ELr, in Angluin's framework of active learning that allows the learning algorithm to interactively query an oracle (such as a domain expert). We show that the following can be learned in polynomial time: (1) EL-concepts, (2) symmetry-free ELI-concepts, and (3) conjunctive queries (CQs) that are chordal, symmetry-free, and of bounded arity. In all cases, the learner can pose to the oracle membership queries based on ABoxes and equivalence queries that ask whether a given concept/query from the considered class is equivalent to the target. The restriction to bounded arity in (3) can be removed when we admit unrestricted CQs in equivalence queries. We also show that EL-concepts are not polynomial query learnable in the presence of ELI-ontologies.

----

## [260] Program Synthesis as Dependency Quantified Formula Modulo Theory

**Authors**: *Priyanka Golia, Subhajit Roy, Kuldeep S. Meel*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/261](https://doi.org/10.24963/ijcai.2021/261)

**Abstract**:

Given a specification φ(X, Y ) over inputs X and output Y and defined over a background theory T, the problem of program synthesis is to design a program f such that Y = f (X), satisfies the specification φ. Over the past decade, syntax-guided synthesis (SyGuS) has emerged as a dominant approach to program synthesis where in addition to the specification φ, the end-user also specifies a grammar L to aid the underlying synthesis engine. This paper investigates the feasibility of synthesis techniques without grammar, a sub-class defined as T constrained synthesis. We show that T-constrained synthesis can be reduced to DQF(T),i.e., to the problem of finding a witness of a dependency quantified formula modulo theory. When the underlying theory is the theory of bitvectors, the corresponding DQF problem can be further reduced to Dependency Quantified Boolean Formulas (DQBF). We rely on the progress in DQBF solving to design DQBF-based synthesizers that outperform the domain-specific program synthesis techniques; thereby positioning DQBF as a core representation language for program synthesis. Our empirical analysis shows that T-constrained synthesis can achieve significantly better performance than syntax-guided approaches. Furthermore, the general-purpose DQBF solvers perform on par with domain-specific synthesis techniques.

----

## [261] Updating the Belief Promotion Operator

**Authors**: *Daniel A. Grimaldi, Maria Vanina Martinez, Ricardo Oscar Rodríguez*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/262](https://doi.org/10.24963/ijcai.2021/262)

**Abstract**:

In this note, we introduce the local version of the operator for belief promotion proposed by Schwind et al. We propose a set of postulates and provide a representation theorem that characterizes the proposal. This family of operators is related to belief promotion in the same way that updating is related to revision, and we provide several results that allow us to show this relationship formally. Furthermore, we also show the relationship of the proposed operator with features of credibility-limited revision theory.

----

## [262] Using Platform Models for a Guided Explanatory Diagnosis Generation for Mobile Robots

**Authors**: *Daniel Habering, Till Hofmann, Gerhard Lakemeyer*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/263](https://doi.org/10.24963/ijcai.2021/263)

**Abstract**:

Plan execution on a mobile robot is inherently error-prone, as the robot
needs to act in a physical world which can never be completely
controlled by the robot. If an error occurs during execution, the true
world state is unknown, as a failure may have unobservable consequences.
One approach to deal with such failures is diagnosis, where the true
world state is determined by identifying a set of faults based on sensed
observations. In this paper, we present a novel approach to explanatory
diagnosis, based on the assumption that most failures occur due to some
robot hardware failure. We model the robot platform components with
state machines and formulate action variants for the robots' actions,
modelling different fault modes. We apply diagnosis as
planning with a top-k planning approach to determine possible diagnosis
candidates and then use active diagnosis to find out which of those
candidates is the true diagnosis.  Finally, based on the platform model,
we recover from the occurred failure such that the robot can continue to
operate. We evaluate our approach in a logistics robots scenario by
comparing it to having no diagnosis and diagnosis without platform
models, showing a significant improvement to both alternatives.

----

## [263] HIP Network: Historical Information Passing Network for Extrapolation Reasoning on Temporal Knowledge Graph

**Authors**: *Yongquan He, Peng Zhang, Luchen Liu, Qi Liang, Wenyuan Zhang, Chuang Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/264](https://doi.org/10.24963/ijcai.2021/264)

**Abstract**:

In recent years, temporal knowledge graph (TKG) reasoning has received significant attention. Most existing methods assume that all timestamps and corresponding graphs are available during training, which makes it difficult to predict future events. To address this issue, recent works learn to infer future events based on historical information. However, these methods do not comprehensively consider the latent patterns behind temporal changes, to pass historical information selectively, update representations appropriately and predict events accurately. In this paper, we propose the Historical Information Passing (HIP) network to predict future events. HIP network passes information from temporal, structural and repetitive perspectives, which are used to model the temporal evolution of events, the interactions of events at the same time step, and the known events respectively. In particular, our method considers the updating of relation representations and adopts three scoring functions corresponding to the above dimensions. Experimental results on five benchmark datasets show the superiority of HIP network, and the significant improvements on Hits@1 prove that our method can more accurately predict what is going to happen.

----

## [264] Multi-Agent Abstract Argumentation Frameworks With Incomplete Knowledge of Attacks

**Authors**: *Andreas Herzig, Antonio Yuste-Ginel*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/265](https://doi.org/10.24963/ijcai.2021/265)

**Abstract**:

We introduce a multi-agent, dynamic extension of abstract argumentation frameworks (AFs), strongly inspired by epistemic logic, where agents have only
partial information about the conflicts between arguments. These frameworks can be used to model a variety of situations. For instance, those in which
agents have bounded logical resources and therefore fail to spot some of the actual attacks, or those where some arguments are not explicitly and fully
stated (enthymematic argumentation). Moreover, we include second-order knowledge and common knowledge of the attack relation in our structures (where the latter accounts for the state of the debate), so as to reason about different kinds of persuasion and about strategic features. This version of multi-agent AFs, as well as their updates with public announcements of attacks (more concretely, the effects of these updates on the acceptability of an argument) can be described using S5-PAL, a well-known dynamic-epistemic logic. We also discuss how to extend our proposal to capture arbitrary higher-order attitudes and uncertainty.

----

## [265] Signature-Based Abduction with Fresh Individuals and Complex Concepts for Description Logics

**Authors**: *Patrick Koopmann*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/266](https://doi.org/10.24963/ijcai.2021/266)

**Abstract**:

Given a knowledge base and an observation as a set of facts, 
ABox abduction aims at computing a hypothesis that, when added to the 
knowledge base, is sufficient to entail the observation. In signature-based 
ABox abduction, the hypothesis is further required to use only names from a 
given set. This form of abduction has applications such as diagnosis, KB 
repair, or explaning missing entailments. It is possible that hypotheses for 
a given observation only exist if we admit the use of fresh individuals
and/or complex concepts built from the given signature, something most 
approaches for ABox abduction so far do not allow or only allow 
with restrictions. In this paper, we investigate the computational complexity 
of this form of abduction---allowing either fresh individuals, complex concepts, 
or both---for various description logics, and give size bounds on the hypotheses 
if they exist.

----

## [266] Scalable Non-observational Predicate Learning in ASP

**Authors**: *Mark Law, Alessandra Russo, Krysia Broda, Elisa Bertino*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/267](https://doi.org/10.24963/ijcai.2021/267)

**Abstract**:

Recently, novel ILP systems under the answer set semantics have been proposed, some of which are robust to noise and scalable over large hypothesis spaces. One such system is FastLAS, which is significantly faster than other state-of-the-art ASP-based ILP systems. FastLAS is, however, only capable of Observational Predicate Learning (OPL), where the learned hypothesis defines predicates that are directly observed in the examples. It cannot learn knowledge that is indirectly observable, such as learning causes of observed events. This class of problems, known as non-OPL, is known to be difficult to handle in the context of non-monotonic semantics. Solving non-OPL learning tasks whilst preserving scalability is a challenging open problem.

We address this problem with a new abductive method for translating examples of a non-OPL task to a set of examples, called possibilities, such that the original example is covered iff at least one of the possibilities is covered. This new method allows an ILP system capable of performing OPL tasks to be "upgraded" to solve non-OPL tasks. In particular, we present our new FastNonOPL system, which upgrades FastLAS with the new possibility generation. We compare it to other state-of-the-art ASP-based ILP systems capable of solving non-OPL tasks, showing that FastNonOPL is significantly faster, and in many cases more accurate, than these other systems.

----

## [267] Inferring Time-delayed Causal Relations in POMDPs from the Principle of Independence of Cause and Mechanism

**Authors**: *Junchi Liang, Abdeslam Boularias*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/268](https://doi.org/10.24963/ijcai.2021/268)

**Abstract**:

This paper introduces an algorithm for discovering implicit and delayed causal relations between events observed by a robot at regular or arbitrary times, with the objective of improving data-efficiency and interpretability of model-based reinforcement learning (RL) techniques. The proposed algorithm initially predicts observations with the Markov assumption, and incrementally introduces new hidden variables to explain and reduce the stochasticity of the observations. The hidden variables are memory units that keep track of pertinent past events. Such events are systematically identified by their information gains. A test of independence between inputs and mechanisms is performed to identify cases when there is a causal link between events and those when the information gain is due to confounding variables. The learned transition and reward models are then used in a Monte Carlo tree search for planning. Experiments on simulated and real robotic tasks, and the challenging 3D game Doom show that this method significantly improves over current RL techniques.

----

## [268] Reasoning about Beliefs and Meta-Beliefs by Regression in an Expressive Probabilistic Action Logic

**Authors**: *Daxin Liu, Gerhard Lakemeyer*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/269](https://doi.org/10.24963/ijcai.2021/269)

**Abstract**:

In a recent paper Belle and Lakemeyer proposed the logic DS, a probabilistic extension of a modal variant of the situation calculus with a model of belief based on weighted possible worlds. Among other things, they were able to precisely capture the beliefs of a probabilistic knowledge base in terms of the concept of only-believing. While intuitively appealing, the logic has a number of shortcomings. Perhaps the most severe is the limited expressiveness in that degrees of belief are restricted to constant rational numbers, which makes it impossible to express arbitrary belief distributions.

In this paper we will address this and other shortcomings by extending the language and modifying the semantics of belief and only-believing.  Among other things, we will show that belief retains many but not all of the properties of DS.  Moreover, it turns out that only-believing arbitrary sentences, including those mentioning belief, is uniquely satisfiable in our logic.   For an interesting class of knowledge bases we also show how reasoning about beliefs and meta-beliefs after performing noisy actions and sensing can be reduced to reasoning about the initial beliefs of an agent using a form of regression.

----

## [269] Multi-Agent Belief Base Revision

**Authors**: *Emiliano Lorini, François Schwarzentruber*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/270](https://doi.org/10.24963/ijcai.2021/270)

**Abstract**:

We present a generalization of belief base revision
to the multi-agent case. In our approach agents
have belief bases containing both propositional beliefs
and higher-order beliefs about their own beliefs
and other agentsâ€™ beliefs. Moreover, their belief
bases are split in two parts: the mutable part,
whose elements may change under belief revision,
and the core part, whose elements do not change.
We study a belief revision operator inspired by
the notion of screened revision. We provide complexity
results of model checking for our approach
as well as an optimal model checking algorithm.
Moreover, we study complexity of epistemic planning
formulated in the context of our framework.

----

## [270] Bounded Predicates in Description Logics with Counting

**Authors**: *Sanja Lukumbuzya, Mantas Simkus*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/271](https://doi.org/10.24963/ijcai.2021/271)

**Abstract**:

Description Logics (DLs) support so-called anonymous objects, which significantly contribute to the expressiveness of these KR languages, but also cause substantial computational challenges. This paper investigates reasoning about upper bounds on predicate sizes for ontologies written in the expressive
DL ALCHOIQ extended with closed predicates. We describe a procedure based on integer programming that allows us to decide the existence of upper bounds on the cardinality of some predicate in the models of a given ontology in a data-independent way. Our results yield a promising supporting tool for constructing higher quality ontologies, and provide a new way to push the decidability frontiers. To wit, we define a new safety condition for Datalog-based
queries over DL ontologies, while retaining decidability of query entailment.

----

## [271] On the Relation Between Approximation Fixpoint Theory and Justification Theory

**Authors**: *Simon Marynissen, Bart Bogaerts, Marc Denecker*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/272](https://doi.org/10.24963/ijcai.2021/272)

**Abstract**:

Approximation Fixpoint Theory (AFT) and Justification Theory (JT) are two frameworks to unify logical formalisms. AFT studies semantics in terms of fixpoints of lattice operators, and JT in terms of so-called justifications, which are explanations of why certain facts do or do not hold in a model. While the approaches differ, the frameworks were designed with similar goals in mind, namely to study the different semantics that arise in (mainly) non-monotonic logics. The First contribution of our current paper is to provide a formal link between the two frameworks. To be precise, we show that every justification frame induces an approximator and that this mapping from JT to AFT preserves all major semantics. The second contribution exploits this correspondence to extend JT with a novel class of semantics, namely ultimate semantics: we formally show that ultimate semantics can be obtained in JT by a syntactic transformation on the justification frame, essentially performing some sort of resolution on the rules.

----

## [272] Faster Smarter Proof by Induction in Isabelle/HOL

**Authors**: *Yutaka Nagashima*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/273](https://doi.org/10.24963/ijcai.2021/273)

**Abstract**:

We present sem_ind, a recommendation tool for proof by induction in Isabelle/HOL.
Given an inductive problem, sem_ind produces candidate arguments for proof by induction, and selects promising ones using heuristics.
Our evaluation based on 1,095 inductive problems from 22 source files shows that sem_ind improves the accuracy of recommendation from 20.1% to 38.2% for the most promising candidates within 5.0 seconds of timeout compared to its predecessor while decreasing the median value of execution time from 2.79 seconds to 1.06 seconds.

----

## [273] Two Forms of Responsibility in Strategic Games

**Authors**: *Pavel Naumov, Jia Tao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/274](https://doi.org/10.24963/ijcai.2021/274)

**Abstract**:

The paper studies two forms of responsibility, seeing to it and being blamable, in the setting of strategic games with imperfect information. The paper shows that being blamable is definable through seeing to it, but not the other way around. In addition, it proposes a bimodal logical system that describes the interplay between the seeing to it modality and the individual knowledge modality.

----

## [274] Compressing Exact Cover Problems with Zero-suppressed Binary Decision Diagrams

**Authors**: *Masaaki Nishino, Norihito Yasuda, Kengo Nakamura*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/275](https://doi.org/10.24963/ijcai.2021/275)

**Abstract**:

Exact cover refers to the problem of finding subfamily
F of a given family of sets S whose universe
is D, where F forms a partition of D. Knuth’s Algorithm
DLX is a state-of-the-art method for solving
exact cover problems. Since DLX’s running
time depends on the cardinality of input S, it can be
slow if S is large. Our proposal can improve DLX
by exploiting a novel data structure, DanceDD,
which extends the zero-suppressed binary decision
diagram (ZDD) by adding links to enable efficient
modifications of the data structure. With DanceDD,
we can represent S in a compressed way and perform
search in linear time with the size of the structure
by using link operations. The experimental results
show that our method is an order of magnitude
faster when the problem is highly compressed.

----

## [275] Modeling Precomputation In Games Played Under Computational Constraints

**Authors**: *Thomas Orton*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/276](https://doi.org/10.24963/ijcai.2021/276)

**Abstract**:

Understanding the properties of games played under computational constraints remains challenging. For example, how do we expect rational (but computationally bounded) players to play games with a prohibitively large number of states, such as chess? This paper presents a novel model for the precomputation (preparing moves in advance) aspect of computationally constrained games. A fundamental trade-off is shown between randomness of play, and susceptibility to precomputation, suggesting that randomization is necessary in games with computational constraints. We present efficient algorithms for computing how susceptible a strategy is to precomputation, and computing an $\epsilon$-Nash equilibrium of our model. Numerical experiments measuring the trade-off between randomness and precomputation are provided for Stockfish (a well-known chess playing algorithm).

----

## [276] A Ladder of Causal Distances

**Authors**: *Maxime Peyrard, Robert West*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/277](https://doi.org/10.24963/ijcai.2021/277)

**Abstract**:

Causal discovery, the task of automatically constructing a causal model from data, is of major significance across the sciences.
Evaluating the performance of causal discovery algorithms should ideally involve comparing the inferred models to ground-truth models available for benchmark datasets, which in turn requires a notion of distance between causal models.
While such distances have been proposed previously, they are limited by focusing on graphical properties of the causal models being compared.
Here, we overcome this limitation by defining distances derived from the causal distributions induced by the models, rather than exclusively from their graphical structure.
Pearl and Mackenzie [2018] have arranged the properties of causal models in a hierarchy called the ``ladder of causation'' spanning three rungs: observational, interventional, and counterfactual.
Following this organization, we introduce a hierarchy of three distances, one for each rung of the ladder.
Our definitions are intuitively appealing as well as efficient to compute approximately.
We put our causal distances to use by benchmarking standard causal discovery systems on both synthetic and real-world datasets for which ground-truth causal models are available.

----

## [277] Unsupervised Knowledge Graph Alignment by Probabilistic Reasoning and Semantic Embedding

**Authors**: *Zhiyuan Qi, Ziheng Zhang, Jiaoyan Chen, Xi Chen, Yuejia Xiang, Ningyu Zhang, Yefeng Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/278](https://doi.org/10.24963/ijcai.2021/278)

**Abstract**:

Knowledge Graph (KG) alignment is to discover the mappings (i.e., equivalent entities, relations, and others) between two KGs. The existing methods can be divided into the embedding-based models, and the conventional reasoning and lexical matching based systems. The former compute the similarity of entities via their cross-KG embeddings, but they usually rely on an ideal supervised learning setting for good performance and lack appropriate reasoning to avoid logically wrong mappings; while the latter address the reasoning issue but are poor at utilizing the KG graph structures and the entity contexts. In this study, we aim at combining the above two solutions and thus propose an iterative framework named PRASE which is based on probabilistic reasoning and semantic embedding. It learns the KG embeddings via entity mappings from a probabilistic reasoning system named PARIS, and feeds the resultant entity mappings and embeddings back into PARIS for augmentation. The PRASE framework is compatible with different embedding-based models, and our experiments on multiple datasets have demonstrated its state-of-the-art performance.

----

## [278] Efficient PAC Reinforcement Learning in Regular Decision Processes

**Authors**: *Alessandro Ronca, Giuseppe De Giacomo*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/279](https://doi.org/10.24963/ijcai.2021/279)

**Abstract**:

Recently regular decision processes have been proposed as a well-behaved form of non-Markov decision process. Regular decision processes are characterised by a transition function and a reward function that depend on the whole history, though regularly (as in regular languages). In practice both the transition and the reward functions can be seen as finite transducers. We study reinforcement learning in regular decision processes. Our main contribution is to show that a near-optimal policy can be PAC-learned in polynomial time in a set of parameters that describe the underlying decision process. We argue that the identified set of parameters is minimal and it reasonably captures the difficulty of a regular decision process.

----

## [279] Inconsistency Measurement for Paraconsistent Inference

**Authors**: *Yakoub Salhi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/280](https://doi.org/10.24963/ijcai.2021/280)

**Abstract**:

One of the main aims of the methods developed for reasoning under inconsistency, in particular paraconsistent inference, is to derive informative conclusions from inconsistent bases. In this paper, we introduce an approach based on inconsistency measurement for defining non-monotonic paraconsistent consequence relations. The main idea consists in adapting properties of classical reasoning under consistency to inconsistent propositional bases by involving inconsistency measures (IM). We first exhibit interesting properties of our consequence relations. We then study situations where they bring about consequences that are always jointly consistent. In particular, we introduce a property of inconsistency measures that guarantees the consistency of the set of all entailed formulas. We also show that this property leads to several interesting properties of our IM-based consequence relations. Finally, we discuss relationships between our framework and well-known consequence relations that are based on maximal consistent subsets. In this setting, we establish direct connections between the latter and properties of inconsistency measures.

----

## [280] A Description Logic for Analogical Reasoning

**Authors**: *Steven Schockaert, Yazmín Ibáñez-García, Víctor Gutiérrez-Basulto*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/281](https://doi.org/10.24963/ijcai.2021/281)

**Abstract**:

Ontologies formalise how the concepts from a given domain are interrelated. Despite their clear potential as a backbone for explainable AI, existing ontologies tend to be highly incomplete, which acts as a significant barrier to their more widespread adoption. To mitigate this issue, we present a mechanism to infer plausible missing knowledge, which relies on reasoning by analogy.  To the best of our knowledge, this is the first paper that studies analogical reasoning within the setting of description logic ontologies. After showing that the standard formalisation of analogical proportion has important limitations in this setting, we introduce an alternative semantics based on bijective mappings between sets of features. We then analyse the properties of analogies under the proposed semantics, and show among others how it enables two plausible inference patterns: rule translation and rule extrapolation.

----

## [281] Ranking Extensions in Abstract Argumentation

**Authors**: *Kenneth Skiba, Tjitze Rienstra, Matthias Thimm, Jesse Heyninck, Gabriele Kern-Isberner*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/282](https://doi.org/10.24963/ijcai.2021/282)

**Abstract**:

Extension-based semantics in abstract argumentation provide a criterion to determine whether a set of arguments is acceptable or not. In this paper, we present the notion of extension-ranking semantics, which determines a preordering over sets of arguments, where one set is deemed more plausible than another if it is somehow more acceptable. We obtain extension-based semantics as a special case of this new approach, but it also allows us to make more fine-grained distinctions, such as one set being "more complete'' or "more admissible'' than another.  We define a number of general principles to classify extension-ranking semantics and develop concrete approaches. We also study the relation between extension-ranking semantics and argument-ranking based semantics, which rank individual arguments instead of sets of arguments.

----

## [282] Physics-informed Spline Learning for Nonlinear Dynamics Discovery

**Authors**: *Fangzheng Sun, Yang Liu, Hao Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/283](https://doi.org/10.24963/ijcai.2021/283)

**Abstract**:

Dynamical systems are typically governed by a set of linear/nonlinear differential equations. Distilling the analytical form of these equations from very limited data remains intractable in many disciplines such as physics, biology, climate science, engineering and social science. To address this fundamental challenge, we propose a novel Physics-informed Spline Learning (PiSL) framework to discover parsimonious governing equations for nonlinear dynamics, based on sparsely sampled noisy data. The key concept is to (1) leverage splines to interpolate locally the dynamics, perform analytical differentiation and build the library of candidate terms, (2) employ sparse representation of the governing equations, and (3) use the physics residual in turn to inform the spline learning. The synergy between splines and discovered underlying physics leads to the robust capacity of dealing with high-level data scarcity and noise. A hybrid sparsity-promoting alternating direction optimization strategy is developed for systematically pruning the sparse coefficients that form the structure and explicit expression of the governing equations. The efficacy and superiority of the proposed method have been demonstrated by multiple well-known nonlinear dynamical systems, in comparison with two state-of-the-art methods.

----

## [283] Lifting Symmetry Breaking Constraints with Inductive Logic Programming

**Authors**: *Alice Tarzariol, Martin Gebser, Konstantin Schekotihin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/284](https://doi.org/10.24963/ijcai.2021/284)

**Abstract**:

Efficient omission of symmetric solution candidates is essential for combinatorial problem solving. Most of the existing approaches are instance-specific and focus on the automatic computation of Symmetry Breaking Constraints (SBCs) for each given problem instance. However, the application of such approaches to large-scale instances or advanced problem encodings might be problematic. Moreover, the computed SBCs are propositional and, therefore, can neither be meaningfully interpreted nor transferred to other instances.
To overcome these limitations, we introduce a new model-oriented approach for Answer Set Programming that lifts the SBCs of small problem instances into a set of interpretable first-order constraints using the Inductive Logic Programming paradigm. 
Experiments demonstrate the ability of our framework to learn general constraints from instance-specific SBCs for a collection of combinatorial problems. The obtained results indicate that our approach significantly outperforms a state-of-the-art instance-specific method as well as the direct application of a solver.

----

## [284] Skeptical Reasoning with Preferred Semantics in Abstract Argumentation without Computing Preferred Extensions

**Authors**: *Matthias Thimm, Federico Cerutti, Mauro Vallati*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/285](https://doi.org/10.24963/ijcai.2021/285)

**Abstract**:

We address the problem of deciding skeptical acceptance wrt. preferred semantics of an argument in abstract argumentation frameworks, i.e., the problem of deciding whether an argument is contained in all maximally admissible sets, a.k.a. preferred extensions. State-of-the-art algorithms solve this problem with iterative calls to an external SAT-solver to determine preferred extensions. We provide a new characterisation of skeptical acceptance wrt. preferred semantics that does not involve the notion of a preferred extension. We then develop a new algorithm that also relies on iterative calls to an external SAT-solver but avoids the costly part of maximising admissible sets. We present the results of an experimental evaluation that shows that this new approach significantly outperforms the state of the art. We also apply similar ideas to develop a new algorithm for computing the ideal extension.

----

## [285] Abstract Argumentation Frameworks with Domain Assignments

**Authors**: *Alexandros Vassiliades, Theodore Patkos, Giorgos Flouris, Antonis Bikakis, Nick Bassiliades, Dimitris Plexousakis*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/286](https://doi.org/10.24963/ijcai.2021/286)

**Abstract**:

Argumentative discourse rarely consists of opinions whose claims apply universally. As with logical statements, an argument applies to specific objects in the universe or relations among them, and may have exceptions. In this paper, we propose
an argumentation formalism that allows associating arguments with a domain of application. Appropriate semantics are given, which formalise the notion of partial argument acceptance, i.e. the set of objects or relations that an argument can be applied
to. We show that our proposal is in fact equivalent to the standard Argumentation Frameworks of Dung, but allows a more intuitive and compact expression of some core concepts of commonsense and non-monotonic reasoning, such as the scope of an argument, exceptions, relevance and others.

----

## [286] Transforming Robotic Plans with Timed Automata to Solve Temporal Platform Constraints

**Authors**: *Tarik Viehmann, Till Hofmann, Gerhard Lakemeyer*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/287](https://doi.org/10.24963/ijcai.2021/287)

**Abstract**:

Task planning for mobile robots typically uses an abstract planning domain
that ignores the low-level details of the specific robot platform.
Therefore, executing a plan on an actual robot often requires
additional steps to deal with the specifics of the robot platform. Such
a platform can be modeled with timed automata and a set of temporal
constraints that need to be satisfied during execution.

In this paper, we describe how to transform an abstract plan into a
platform-specific action sequence that satisfies all platform
constraints. The transformation procedure first transforms the plan into
a timed automaton, which is then combined with the platform automata
while removing all transitions that violate any constraint. We then
apply reachability analysis on the resulting automaton.  From any
solution trace one can obtain the abstract plan extended by additional
platform actions such that all platform constraints are satisfied.  We
describe the transformation procedure in detail and provide an
evaluation in two real-world robotics scenarios.

----

## [287] Neighborhood Intervention Consistency: Measuring Confidence for Knowledge Graph Link Prediction

**Authors**: *Kai Wang, Yu Liu, Quan Z. Sheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/288](https://doi.org/10.24963/ijcai.2021/288)

**Abstract**:

Link prediction based on knowledge graph embeddings (KGE) has recently drawn a considerable momentum. However, existing KGE models suffer from insufficient accuracy and hardly evaluate the confidence probability of each predicted triple. To fill this critical gap, we propose a novel confidence measurement method based on causal intervention, called Neighborhood Intervention Consistency (NIC). Unlike previous confidence measurement methods that focus on the optimal score in a prediction, NIC actively intervenes in the input entity vector to measure the robustness of the prediction result. The experimental results on ten popular KGE models show that our NIC method can effectively estimate the confidence score of each predicted triple. The top 10% triples with high NIC confidence can achieve 30% higher accuracy in the state-of-the-art KGE models.

----

## [288] Causal Discovery with Multi-Domain LiNGAM for Latent Factors

**Authors**: *Yan Zeng, Shohei Shimizu, Ruichu Cai, Feng Xie, Michio Yamamoto, Zhifeng Hao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/289](https://doi.org/10.24963/ijcai.2021/289)

**Abstract**:

Discovering causal structures among latent factors from observed data is a particularly challenging problem. Despite some efforts for this problem, existing methods focus on the single-domain data only. In this paper, we propose Multi-Domain Linear Non-Gaussian Acyclic Models for LAtent Factors (MD-LiNA), where the causal structure among latent factors of interest is shared for all domains, and we provide its identification results. The model enriches the causal representation for multi-domain data. We propose an integrated two-phase algorithm to estimate the model. In particular, we first locate the latent factors and estimate the factor loading matrix. Then to uncover the causal structure among shared latent factors of interest, we derive a score function based on the characterization of independence relations between external influences and the dependence relations between multi-domain latent factors and latent factors of interest. We show that the proposed method provides locally consistent estimators. Experimental results on both synthetic and real-world data demonstrate the efficacy and robustness of our approach.

----

## [289] AMEIR: Automatic Behavior Modeling, Interaction Exploration and MLP Investigation in the Recommender System

**Authors**: *Pengyu Zhao, Kecheng Xiao, Yuanxing Zhang, Kaigui Bian, Wei Yan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/290](https://doi.org/10.24963/ijcai.2021/290)

**Abstract**:

Recently, deep learning models have been widely explored in recommender systems. Though having achieved remarkable success, the design of task-aware recommendation models usually requires manual feature engineering and architecture engineering from domain experts. To relieve those efforts, we explore the potential of neural architecture search (NAS) and introduce AMEIR for Automatic behavior Modeling, interaction Exploration and multi-layer perceptron (MLP) Investigation in the Recommender system. Specifically, AMEIR divides the complete recommendation models into three stages of behavior modeling, interaction exploration, MLP aggregation, and introduces a novel search space containing three tailored subspaces that cover most of the existing methods and thus allow for searching better models. To find the ideal architecture efficiently and effectively, AMEIR realizes the one-shot random search in recommendation progressively on the three stages and assembles the search results as the final outcome. The experiment over various scenarios reveals that AMEIR outperforms competitive baselines of elaborate manual design and leading algorithmic complex NAS methods with lower model complexity and comparable time cost, indicating efficacy, efficiency, and robustness of the proposed method.

----

## [290] The Surprising Power of Graph Neural Networks with Random Node Initialization

**Authors**: *Ralph Abboud, Ismail Ilkan Ceylan, Martin Grohe, Thomas Lukasiewicz*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/291](https://doi.org/10.24963/ijcai.2021/291)

**Abstract**:

Graph neural networks (GNNs) are effective models for representation learning on relational data. However, standard GNNs are limited in their expressive power, as they cannot distinguish graphs beyond the capability of the Weisfeiler-Leman graph isomorphism heuristic. In order to break this expressiveness barrier, GNNs have been enhanced with random node initialization (RNI), where the idea is to train and run the models with randomized initial node features. In this work, we analyze the expressive power of GNNs with RNI, and prove that these models are universal, a first such result for GNNs not relying on computationally demanding higher-order properties. This universality result holds even with partially randomized initial node features, and preserves the invariance properties of GNNs in expectation. We then empirically analyze the effect of RNI on GNNs, based on carefully constructed datasets. Our empirical findings support the superior performance of GNNs with RNI over standard GNNs.

----

## [291] Likelihood-free Out-of-Distribution Detection with Invertible Generative Models

**Authors**: *Amirhossein Ahmadian, Fredrik Lindsten*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/292](https://doi.org/10.24963/ijcai.2021/292)

**Abstract**:

Likelihood of generative models has been used traditionally as a score to detect atypical (Out-of-Distribution, OOD) inputs.  However, several recent studies have found this approach to be highly unreliable, even with invertible generative models, where computing the likelihood is feasible. In this paper, we present a different framework for generative model--based OOD detection that employs the model in constructing a new representation space, instead of using it directly in computing typicality scores, where it is emphasized that the score function should be interpretable as the similarity between the input and training data in the new space. In practice, with a focus on invertible models, we propose to extract low-dimensional features (statistics) based on the model encoder and complexity of input images, and then use a One-Class SVM to score the data. Contrary to recently proposed OOD detection methods for generative models, our method does not require computing likelihood values. Consequently, it is much faster when using invertible models with iteratively approximated likelihood (e.g. iResNet), while it still has a performance competitive with other related methods.

----

## [292] Simulation of Electron-Proton Scattering Events by a Feature-Augmented and Transformed Generative Adversarial Network (FAT-GAN)

**Authors**: *Yasir Alanazi, Nobuo Sato, Tianbo Liu, Wally Melnitchouk, Pawel Ambrozewicz, Florian Hauenstein, Michelle P. Kuchera, Evan Pritchard, Michael Robertson, Ryan R. Strauss, Luisa Velasco, Yaohang Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/293](https://doi.org/10.24963/ijcai.2021/293)

**Abstract**:

We apply generative adversarial network (GAN) technology to build an event generator that simulates particle production in electron-proton scattering that is free of theoretical assumptions about underlying particle dynamics. The difficulty of efficiently training a GAN event simulator lies in learning the complicated patterns of the distributions of the particles physical properties. We develop a GAN that selects a set of transformed features from particle momenta that can be generated easily by the generator, and uses these to produce a set of augmented features that improve the sensitivity of the discriminator. The new Feature-Augmented and Transformed GAN (FAT-GAN) is able to faithfully reproduce the distribution of final state electron momenta in inclusive electron scattering, without the need for input derived from domain-based theoretical assumptions. The developed technology can play a significant role in boosting the science of existing and future accelerator facilities, such as the Electron-Ion Collider.

----

## [293] Deep Reinforcement Learning for Navigation in AAA Video Games

**Authors**: *Eloi Alonso, Maxim Peter, David Goumard, Joshua Romoff*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/294](https://doi.org/10.24963/ijcai.2021/294)

**Abstract**:

In video games, \non-player characters (NPCs) are used to enhance the players' experience in a variety of ways, e.g., as enemies, allies, or innocent bystanders. A crucial component of NPCs is navigation, which allows them to move from one point to another on the map. The most popular approach for NPC navigation in the video game industry is to use a navigation mesh (NavMesh), which is a graph representation of the map, with nodes and edges indicating traversable areas. Unfortunately, complex navigation abilities that extend the character's capacity for movement, e.g., grappling hooks, jetpacks, teleportation, or double-jumps, increase the complexity of the NavMesh, making it intractable in many practical scenarios. Game designers are thus constrained to only add abilities that can be handled by a NavMesh. As an alternative to the NavMesh, we propose to use Deep Reinforcement Learning (Deep RL) to learn how to navigate 3D maps in video games using any navigation ability. We test our approach on complex 3D environments that are notably an order of magnitude larger than maps typically used in the Deep RL literature. One of these environments is from a recently released AAA video game called Hyper Scape. We find that our approach performs surprisingly well, achieving at least 90% success rate in a variety of scenarios using complex navigation abilities.

----

## [294] Conditional Self-Supervised Learning for Few-Shot Classification

**Authors**: *Yuexuan An, Hui Xue, Xingyu Zhao, Lu Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/295](https://doi.org/10.24963/ijcai.2021/295)

**Abstract**:

How to learn a transferable feature representation from limited examples is a key challenge for few-shot classification. Self-supervision as an auxiliary task to the main supervised few-shot task is considered to be a conceivable way to solve the problem since self-supervision can provide additional structural information easily ignored by the main task. However, learning a good representation by traditional self-supervised methods is usually dependent on large training samples. In few-shot scenarios, due to the lack of sufficient samples, these self-supervised methods might learn a biased representation, which more likely leads to the wrong guidance for the main tasks and finally causes the performance degradation. In this paper, we propose conditional self-supervised learning (CSS) to use auxiliary information to guide the representation learning of self-supervised tasks. Specifically, CSS leverages supervised information as prior knowledge to shape and improve the learning feature manifold of self-supervision without auxiliary unlabeled data, so as to reduce representation bias and mine more effective semantic information. Moreover, CSS exploits more meaningful information through supervised and the improved self-supervised learning respectively and integrates the information into a unified distribution, which can further enrich and broaden the original representation. Extensive experiments demonstrate that our proposed method without any fine-tuning can achieve a significant accuracy improvement on the few-shot classification scenarios compared to the state-of-the-art few-shot learning methods.

----

## [295] DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization

**Authors**: *Noor H. Awad, Neeratyoy Mallik, Frank Hutter*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/296](https://doi.org/10.24963/ijcai.2021/296)

**Abstract**:

Modern machine learning algorithms crucially rely on several design decisions to achieve strong performance, making the problem of Hyperparameter Optimization (HPO) more important than ever. Here, we combine the advantages of the popular bandit-based HPO method Hyperband (HB) and the evolutionary search approach of Differential Evolution (DE) to yield a new HPO method which we call DEHB. Comprehensive results on a very broad range of HPO problems, as well as a wide range of tabular benchmarks from neural architecture search, demonstrate that DEHB achieves strong performance far more robustly than all previous HPO methods we are aware of, especially for high-dimensional problems with discrete input dimensions. For example, DEHB is up to 1000x faster than random search. It is also efficient in computational time, conceptually simple and easy to implement, positioning it well to become a new default HPO method.

----

## [296] Verifying Reinforcement Learning up to Infinity

**Authors**: *Edoardo Bacci, Mirco Giacobbe, David Parker*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/297](https://doi.org/10.24963/ijcai.2021/297)

**Abstract**:

Formally verifying that reinforcement learning systems act
safely is increasingly important, but existing methods
only verify over finite time.
This is of limited use for dynamical systems that run indefinitely. 
We introduce the first method for verifying the time-unbounded
safety of neural networks controlling dynamical systems.
We develop a novel abstract interpretation method which,
by constructing adaptable template-based polyhedra using MILP and interval
arithmetic, yields sound---safe and invariant---overapproximations
of the reach set.
This provides stronger safety guarantees
than previous time-bounded methods and shows whether
the agent has generalised beyond the length of its training episodes.
Our method supports ReLU activation functions
and systems with linear, piecewise linear and non-linear dynamics
defined with polynomial and transcendental functions.
We demonstrate its efficacy on a range of benchmark control problems.

----

## [297] Robustly Learning Composable Options in Deep Reinforcement Learning

**Authors**: *Akhil Bagaria, Jason K. Senthil, Matthew Slivinski, George Konidaris*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/298](https://doi.org/10.24963/ijcai.2021/298)

**Abstract**:

Hierarchical reinforcement learning (HRL) is only effective for long-horizon problems  when high-level skills can be reliably sequentially executed. Unfortunately, learning reliably composable skills is difficult, because all the components of every skill are constantly changing during learning. We propose three methods for improving the composability of learned skills: representing skill initiation regions using a combination of pessimistic and optimistic classifiers; learning re-targetable policies that are robust to non-stationary subgoal regions; and learning robust option policies using model-based RL. We test these improvements on four sparse-reward maze navigation tasks involving a simulated quadrupedal robot. Each method successively improves the robustness of a baseline skill discovery method, substantially outperforming state-of-the-art flat and hierarchical methods.

----

## [298] Reconciling Rewards with Predictive State Representations

**Authors**: *Andrea Baisero, Christopher Amato*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/299](https://doi.org/10.24963/ijcai.2021/299)

**Abstract**:

Predictive state representations (PSRs) are models of controlled non-Markov
  observation sequences which exhibit the same generative process governing
  POMDP observations without relying on an underlying latent state.  In that
  respect, a PSR is indistinguishable from the corresponding POMDP.  However,
  PSRs notoriously ignore the notion of rewards, which undermines the general
  utility of PSR models for control, planning, or reinforcement learning.
  Therefore, we describe a sufficient and necessary accuracy condition
  which determines whether a PSR is able to accurately model POMDP rewards, we
  show that rewards can be approximated even when the accuracy condition is not
  satisfied, and we find that a non-trivial number of POMDPs taken from a
  well-known third-party repository do not satisfy the accuracy condition.  
  We propose reward-predictive state representations (R-PSRs), a
  generalization of PSRs which accurately models both observations and rewards,
  and develop value iteration for R-PSRs.  We show that there is a mismatch
  between optimal POMDP policies and the optimal PSR policies derived from
  approximate rewards.  On the other hand, optimal R-PSR policies perfectly
  match optimal POMDP policies, reconfirming R-PSRs as accurate state-less
  generative models of observations and rewards.

----

## [299] Optimal Algorithms for Range Searching over Multi-Armed Bandits

**Authors**: *Siddharth Barman, Ramakrishnan Krishnamurthy, Saladi Rahul*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/300](https://doi.org/10.24963/ijcai.2021/300)

**Abstract**:

This paper studies a multi-armed bandit (MAB) version of the range-searching problem. In its basic form, range searching considers as input a set of points (on the real line) and a collection of (real) intervals. Here, with each specified point, we have an associated weight, and the problem objective is to find a maximum-weight point within every given interval. The current work addresses range searching with stochastic weights: each point corresponds to an arm (that admits sample access) and the point's weight is the (unknown) mean of the underlying distribution. In this MAB setup, we develop sample-efficient algorithms that find, with high probability, near-optimal arms within the given intervals, i.e., we obtain  PAC (probably approximately correct) guarantees. We also provide an algorithm for a generalization wherein the weight of each point is a multi-dimensional vector. The sample complexities of our algorithms depend, in particular, on the size of the {optimal hitting set} of the given intervals. Finally, we establish lower bounds proving that the obtained sample complexities are essentially tight. Our results highlight the significance of geometric constructs (specifically, hitting sets) in our MAB setting.

----

## [300] Efficient Neural Network Verification via Layer-based Semidefinite Relaxations and Linear Cuts

**Authors**: *Ben Batten, Panagiotis Kouvaros, Alessio Lomuscio, Yang Zheng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/301](https://doi.org/10.24963/ijcai.2021/301)

**Abstract**:

We introduce an efficient and tight layer-based semidefinite relaxation for verifying local robustness of neural networks. The improved tightness is the result of the combination between semidefinite relaxations and linear cuts. We obtain a computationally efficient method by decomposing the semidefinite formulation into
layerwise constraints.  By leveraging on chordal graph decompositions, we show that the formulation here presented is provably tighter than current approaches. Experiments on a set of benchmark networks show that the approach here proposed enables the verification of more instances compared to other relaxation methods.  The results also demonstrate that the SDP relaxation here proposed is one order of magnitude faster than previous SDP methods.

----

## [301] Fast Pareto Optimization for Subset Selection with Dynamic Cost Constraints

**Authors**: *Chao Bian, Chao Qian, Frank Neumann, Yang Yu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/302](https://doi.org/10.24963/ijcai.2021/302)

**Abstract**:

Subset selection with cost constraints is a fundamental problem with various applications such as influence maximization and sensor placement. The goal is to select a subset from a ground set to maximize a monotone objective function such that a monotone cost function is upper bounded by a budget. Previous algorithms with bounded approximation guarantees include the generalized greedy algorithm, POMC and EAMC, all of which can achieve the best known approximation guarantee. In real-world scenarios, the resources often vary, i.e., the budget often changes over time, requiring the algorithms to adapt the solutions quickly. However, when the budget changes dynamically, all these three algorithms either achieve arbitrarily bad approximation guarantees, or require a long running time. In this paper, we propose a new algorithm FPOMC by combining the merits of the generalized greedy algorithm and POMC. That is, FPOMC introduces a greedy selection strategy into POMC. We prove that FPOMC can maintain the best known approximation guarantee efficiently.

----

## [302] Partial Multi-Label Optimal Margin Distribution Machine

**Authors**: *Nan Cao, Teng Zhang, Hai Jin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/303](https://doi.org/10.24963/ijcai.2021/303)

**Abstract**:

Partial multi-label learning deals with the circumstance in which the ground-truth labels are not directly available but hidden in a candidate label set. Due to the presence of other irrelevant labels, vanilla multi-label learning methods are prone to be misled and fail to generalize well on unseen data, thus how to enable them to get rid of the noisy labels turns to be the core problem of partial multi-label learning. In this paper, we propose the Partial Multi-Label Optimal margin Distribution Machine (PML-ODM), which distinguishs the noisy labels through explicitly optimizing the distribution of ranking margin, and exhibits better generalization performance than minimum margin based counterparts. In addition, we propose a novel feature prototype representation to further enhance the disambiguation ability, and the non-linear kernels can also be applied to promote the generalization performance for linearly inseparable data. Extensive experiments on real-world data sets validates the superiority of our proposed method.

----

## [303] Towards Understanding the Spectral Bias of Deep Learning

**Authors**: *Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, Quanquan Gu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/304](https://doi.org/10.24963/ijcai.2021/304)

**Abstract**:

An intriguing phenomenon observed during training neural networks is the spectral bias, which states that neural networks are biased towards learning less complex functions. The priority of learning functions with low complexity might be at the core of explaining the generalization ability of neural networks, and certain efforts have been made to provide a theoretical explanation for spectral bias. However, there is still no satisfying theoretical result justifying the underlying mechanism of spectral bias. In this paper, we give a comprehensive and rigorous explanation for spectral bias and relate it with the neural tangent kernel function proposed in recent work. We prove that the training process of neural networks can be decomposed along different directions defined by the eigenfunctions of the neural tangent kernel, where each direction has its own convergence rate and the rate is determined by the corresponding eigenvalue. We then provide a case study when the input data is uniformly distributed over the unit sphere, and show that lower degree spherical harmonics are easier to be learned by over-parameterized neural networks. Finally, we provide numerical experiments to demonstrate the correctness of our theory. Our experimental results also show that our theory can tolerate certain model misspecification in terms of the input data distribution.

----

## [304] Thompson Sampling for Bandits with Clustered Arms

**Authors**: *Emil Carlsson, Devdatt P. Dubhashi, Fredrik D. Johansson*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/305](https://doi.org/10.24963/ijcai.2021/305)

**Abstract**:

We propose algorithms based on a multi-level Thompson sampling scheme, for the stochastic multi-armed bandit and its contextual variant with linear expected rewards, in the setting where arms are clustered. We show, both theoretically and empirically, how exploiting a given cluster structure can significantly improve the regret and computational cost compared to using standard Thompson sampling. In the case of the stochastic multi-armed bandit we give upper bounds on the expected cumulative regret showing how it depends on the quality of the clustering.  Finally, we perform an empirical evaluation showing that our algorithms perform well compared to previously proposed algorithms for bandits with clustered arms.

----

## [305] Reinforcement Learning for Sparse-Reward Object-Interaction Tasks in a First-person Simulated 3D Environment

**Authors**: *Wilka Carvalho, Anthony Liang, Kimin Lee, Sungryull Sohn, Honglak Lee, Richard L. Lewis, Satinder Singh*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/306](https://doi.org/10.24963/ijcai.2021/306)

**Abstract**:

Learning how to execute complex tasks involving multiple objects in a 3D world is challenging when there is no ground-truth information about the objects or any demonstration to learn from.
When an agent only receives a signal from task-completion, this makes it challenging to learn the object-representations which support learning the correct object-interactions needed to complete the task.
In this work, we formulate learning an attentive object dynamics model as a classification problem, using random object-images to define incorrect labels for our object-dynamics model. 
We show empirically that this enables object-representation learning that captures an object's category (is it a toaster?), its properties (is it on?), and object-relations (is something inside of it?).
With this, our core learner (a relational RL agent) receives the dense training signal it needs to rapidly learn object-interaction tasks.
We demonstrate results in the 3D AI2Thor simulated kitchen environment with a range of challenging food preparation tasks.
We compare our method's performance to several related approaches and against the performance of an oracle: an agent that is supplied with ground-truth information about objects in the scene.
We find that our agent achieves performance closest to the oracle in terms of both learning speed and maximum success rate.

----

## [306] Generative Adversarial Neural Architecture Search

**Authors**: *Seyed Saeed Changiz Rezaei, Fred X. Han, Di Niu, Mohammad Salameh, Keith G. Mills, Shuo Lian, Wei Lu, Shangling Jui*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/307](https://doi.org/10.24963/ijcai.2021/307)

**Abstract**:

Despite the empirical success of neural architecture search (NAS) in deep learning applications, the optimality, reproducibility and cost of NAS schemes remain hard to assess.  In this paper, we propose Generative Adversarial NAS (GA-NAS) with theoretically provable convergence guarantees, promoting stability and reproducibility in neural architecture search. Inspired by importance sampling, GA-NAS iteratively fits a generator to previously discovered top architectures, thus increasingly focusing on important parts of a large search space. Furthermore, we propose an efficient adversarial learning approach, where the generator is trained by reinforcement learning based on rewards provided by a discriminator, thus being able to explore the search space without evaluating a large number of architectures. Extensive experiments show that GA-NAS beats the best published results under several cases on three public NAS benchmarks. In the meantime, GA-NAS can handle ad-hoc search constraints and search spaces. We show that GA-NAS can be used to improve already optimized baselines found by other NAS methods, including EfficientNet and ProxylessNAS, in terms of ImageNet accuracy or the number of parameters, in their original search space.

----

## [307] AMA-GCN: Adaptive Multi-layer Aggregation Graph Convolutional Network for Disease Prediction

**Authors**: *Hao Chen, Fuzhen Zhuang, Li Xiao, Ling Ma, Haiyan Liu, Ruifang Zhang, Huiqin Jiang, Qing He*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/308](https://doi.org/10.24963/ijcai.2021/308)

**Abstract**:

Recently, Graph Convolutional Networks (GCNs) have proven to be a powerful mean for Computer Aided Diagnosis (CADx). This approach requires building a population graph to aggregate structural information, where the graph adjacency matrix represents the relationship between nodes. Until now, this adjacency matrix is usually defined manually based on phenotypic information. In this paper, we propose an encoder that automatically selects the appropriate phenotypic measures according to their spatial distribution, and uses the text similarity awareness mechanism to calculate the edge weights between nodes. The encoder can automatically construct the population graph using phenotypic measures which have a positive impact on the final results, and further realizes the fusion of multimodal information. In addition, a novel graph convolution network architecture using multi-layer aggregation mechanism is proposed. The structure can obtain deep structure information while suppressing over-smooth, and increase the similarity between the same type of nodes. Experimental results on two databases show that our method can significantly improve the diagnostic accuracy for Autism spectrum disorder and breast cancer, indicating its universality in leveraging multimodal data for disease prediction.

----

## [308] Learning Attributed Graph Representation with Communicative Message Passing Transformer

**Authors**: *Jianwen Chen, Shuangjia Zheng, Ying Song, Jiahua Rao, Yuedong Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/309](https://doi.org/10.24963/ijcai.2021/309)

**Abstract**:

Constructing appropriate representations of molecules lies at the core of numerous tasks such as material science, chemistry, and drug designs. Recent researches abstract molecules as attributed graphs and employ graph neural networks (GNN) for molecular representation learning, which have made remarkable achievements in molecular graph modeling. Albeit powerful, current models either are based on local aggregation operations and thus miss higher-order graph properties or focus on only node information without fully using the edge information. For this sake, we propose a Communicative Message Passing Transformer (CoMPT) neural network to improve the molecular graph representation by reinforcing message interactions between nodes and edges based on the Transformer architecture. Unlike the previous transformer-style GNNs that treat molecule as a fully connected graph, we introduce a message diffusion mechanism to leverage the graph connectivity inductive bias and reduce the message enrichment explosion. Extensive experiments demonstrated that the proposed model obtained superior performances (around 4% on average) against state-of-the-art baselines on seven chemical property datasets (graph-level tasks) and two chemical shift datasets (node-level tasks). Further visualization studies also indicated a better representation capacity achieved by our model.

----

## [309] Understanding Structural Vulnerability in Graph Convolutional Networks

**Authors**: *Liang Chen, Jintang Li, Qibiao Peng, Yang Liu, Zibin Zheng, Carl Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/310](https://doi.org/10.24963/ijcai.2021/310)

**Abstract**:

Recent studies have shown that Graph Convolutional Networks (GCNs) are vulnerable to adversarial attacks on the graph structure. Although multiple works have been proposed to improve their robustness against such structural adversarial attacks, the reasons for the success of the attacks remain unclear. In this work, we theoretically and empirically demonstrate that structural adversarial examples can be attributed to the non-robust aggregation scheme (i.e., the weighted mean) of GCNs. Specifically, our analysis takes advantage of the breakdown point which can quantitatively measure the robustness of aggregation schemes. The key insight is that weighted mean, as the basic design of GCNs, has a low breakdown point and its output can be dramatically changed by injecting a single edge. We show that adopting the aggregation scheme with a high breakdown point (e.g., median or trimmed mean) could significantly enhance the robustness of GCNs against structural attacks. Extensive experiments on four real-world datasets demonstrate that such a simple but effective method achieves the best robustness performance compared to state-of-the-art models.

----

## [310] Monte Carlo Filtering Objectives

**Authors**: *Shuangshuang Chen, Sihao Ding, Yiannis Karayiannidis, Mårten Björkman*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/311](https://doi.org/10.24963/ijcai.2021/311)

**Abstract**:

Learning generative models and inferring latent trajectories have shown to be challenging for time series due to the intractable marginal likelihoods of flexible generative models. It can be addressed by surrogate objectives for optimization. We propose Monte Carlo filtering objectives (MCFOs), a family of variational objectives for jointly learning parametric generative models and amortized adaptive importance proposals of time series. MCFOs extend the choices of likelihood estimators beyond Sequential Monte Carlo in state-of-the-art objectives, possess important properties revealing the factors for the tightness of objectives, and allow for less biased and variant gradient estimates. We demonstrate that the proposed MCFOs and gradient estimations lead to efficient and stable model learning, and learned generative models well explain data and importance proposals are more sample efficient on various kinds of time series data.

----

## [311] Dependent Multi-Task Learning with Causal Intervention for Image Captioning

**Authors**: *Wenqing Chen, Jidong Tian, Caoyun Fan, Hao He, Yaohui Jin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/312](https://doi.org/10.24963/ijcai.2021/312)

**Abstract**:

Recent work for image captioning mainly followed an extract-then-generate paradigm, pre-extracting a sequence of object-based features and then formulating image captioning as a single sequence-to-sequence task. Although promising, we observed two problems in generated captions: 1) content inconsistency where models would generate contradicting facts; 2) not informative enough where models would miss parts of important information. From a causal perspective, the reason is that models have captured spurious statistical correlations between visual features and certain expressions (e.g., visual features of "long hair" and "woman"). In this paper, we propose a dependent multi-task learning framework with the causal intervention (DMTCI). Firstly, we involve an intermediate task, bag-of-categories generation, before the final task, image captioning. The intermediate task would help the model better understand the visual features and thus alleviate the content inconsistency problem. Secondly, we apply Pearl's do-calculus on the model, cutting off the link between the visual features and possible confounders and thus letting models focus on the causal visual features. Specifically, the high-frequency concept set is considered as the proxy confounders where the real confounders are inferred in the continuous space. Finally, we use a multi-agent reinforcement learning (MARL) strategy to enable end-to-end training and reduce the inter-task error accumulations. The extensive experiments show that our model outperforms the baseline models and achieves competitive performance with state-of-the-art models.

----

## [312] Few-Shot Learning with Part Discovery and Augmentation from Unlabeled Images

**Authors**: *Wentao Chen, Chenyang Si, Wei Wang, Liang Wang, Zilei Wang, Tieniu Tan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/313](https://doi.org/10.24963/ijcai.2021/313)

**Abstract**:

Few-shot learning is a challenging task since only few instances are given for recognizing an unseen class. One way to alleviate this problem is to acquire a strong inductive bias via meta-learning on similar tasks. In this paper, we show that such inductive bias can be learned from a flat collection of unlabeled images, and instantiated as transferable representations among seen and unseen classes.   Specifically, we propose a novel part-based self-supervised representation learning scheme to learn transferable representations by maximizing the similarity of an image to its discriminative part. To mitigate the overfitting in few-shot classification caused by data scarcity, we further propose a part augmentation strategy by retrieving extra images from a base dataset. We conduct systematic studies on miniImageNet and tieredImageNet benchmarks. Remarkably, our method yields impressive results, outperforming the previous best unsupervised methods by 7.74% and 9.24% under 5-way 1-shot and 5-way 5-shot settings, which are comparable with state-of-the-art supervised methods.

----

## [313] On Self-Distilling Graph Neural Network

**Authors**: *Yuzhao Chen, Yatao Bian, Xi Xiao, Yu Rong, Tingyang Xu, Junzhou Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/314](https://doi.org/10.24963/ijcai.2021/314)

**Abstract**:

Recently, the teacher-student knowledge distillation framework has demonstrated its potential in training Graph Neural Networks (GNNs). However, due to the difficulty of training over-parameterized GNN models, one may not easily obtain a satisfactory teacher model for distillation. Furthermore, the inefficient training process of teacher-student knowledge distillation also impedes its applications in GNN models. In this paper, we propose the first teacher-free knowledge distillation method for GNNs, termed GNN Self-Distillation (GNN-SD), that serves as a drop-in replacement of the standard training process. The method is built upon the proposed neighborhood discrepancy rate (NDR), which quantifies the non-smoothness of the embedded graph in an efficient way. Based on this metric, we propose the adaptive discrepancy retaining (ADR) regularizer to empower the transferability of knowledge that maintains high neighborhood discrepancy across GNN layers. We also summarize a generic GNN-SD framework that could be exploited to induce other distillation strategies. Experiments further prove the effectiveness and generalization of our approach, as it brings: 1) state-of-the-art GNN distillation performance with less training cost, 2) consistent and considerable performance enhancement for various popular backbones.

----

## [314] Time-Aware Multi-Scale RNNs for Time Series Modeling

**Authors**: *Zipeng Chen, Qianli Ma, Zhenxi Lin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/315](https://doi.org/10.24963/ijcai.2021/315)

**Abstract**:

Multi-scale information is crucial for modeling time series. Although most existing methods consider multiple scales in the time-series data, they assume all kinds of scales are equally important for each sample, making them unable to capture the dynamic temporal patterns of time series. To this end, we propose Time-Aware Multi-Scale Recurrent Neural Networks (TAMS-RNNs), which disentangle representations of different scales and adaptively select the most important scale for each sample at each time step. First, the hidden state of the RNN is disentangled into multiple independently updated small hidden states, which use different update frequencies to model time-series multi-scale information. Then, at each time step, the temporal context information is used to modulate the features of different scales, selecting the most important time-series scale. Therefore, the proposed model can capture the multi-scale information for each time series at each time step adaptively. Extensive experiments demonstrate that the model outperforms state-of-the-art methods on multivariate time series classification and human motion prediction tasks. Furthermore, visualized analysis on music genre recognition verifies the effectiveness of the model.

----

## [315] Variational Model-based Policy Optimization

**Authors**: *Yinlam Chow, Brandon Cui, Moonkyung Ryu, Mohammad Ghavamzadeh*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/316](https://doi.org/10.24963/ijcai.2021/316)

**Abstract**:

Model-based reinforcement learning (RL) algorithms allow us to combine model-generated data with those collected from interaction with the real system in order to alleviate the data efficiency problem in RL. However, designing such algorithms is often challenging because the bias in simulated data may overshadow the ease of data generation. A potential solution to this challenge is to jointly learn and improve model and policy using a universal objective function. In this paper, we leverage the connection between RL and probabilistic inference, and formulate such an objective function as a variational lower-bound of a log-likelihood. This allows us to use expectation maximization (EM) and iteratively fix a baseline policy and learn a variational distribution, consisting of a model and a policy (E-step), followed by improving the baseline policy given the learned variational distribution (M-step). We propose model-based and model-free policy iteration (actor-critic) style algorithms for the E-step and show how the variational distribution learned by them can be used to optimize the M-step in a fully model-based fashion. Our experiments on a number of continuous control tasks show that our model-based (E-step) algorithm, called variational model-based policy optimization (VMBPO), is more sample-efficient and robust to hyper-parameter tuning than its model-free (E-step) counterpart. Using the same control tasks, we also compare VMBPO with several state-of-the-art model-based and model-free RL algorithms and show its sample efficiency and performance.

----

## [316] CuCo: Graph Representation with Curriculum Contrastive Learning

**Authors**: *Guanyi Chu, Xiao Wang, Chuan Shi, Xunqiang Jiang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/317](https://doi.org/10.24963/ijcai.2021/317)

**Abstract**:

Graph-level representation learning is to learn low-dimensional representation for the entire graph, which has shown a large impact on real-world applications. Recently, limited by expensive labeled data, contrastive learning based graph-level representation learning attracts considerable attention. However, these methods mainly focus on graph augmentation for positive samples, while the effect of negative samples is less explored. In this paper, we study the impact of negative samples on learning graph-level representations, and a novel curriculum contrastive learning framework for self-supervised graph-level representation, called CuCo, is proposed. Specifically, we introduce four graph augmentation techniques to obtain the positive and negative samples, and utilize graph neural networks to learn their representations. Then a scoring function is proposed to sort negative samples from easy to hard and a pacing function is to automatically select the negative samples in each training procedure. Extensive experiments on fifteen graph classification real-world datasets, as well as the parameter analysis, well demonstrate that our proposed CuCo yields truly encouraging results in terms of performance on classification and convergence.

----

## [317] Convexified Graph Neural Networks for Distributed Control in Robotic Swarms

**Authors**: *Saar Cohen, Noa Agmon*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/318](https://doi.org/10.24963/ijcai.2021/318)

**Abstract**:

A network of robots can be viewed as a signal graph, describing the underlying network topology with naturally distributed architectures, whose nodes are assigned to data values associated with each robot. Graph neural networks (GNNs) learn representations from signal graphs, thus making them well-suited candidates for learning distributed controllers. Oftentimes, existing GNN architectures assume ideal scenarios, while ignoring the possibility that this distributed graph may change along time due to link failures or topology variations, which can be found in dynamic settings. A mismatch between the graphs on which GNNs were trained and the ones on which they are tested is thus formed. Utilizing online learning, GNNs can be retrained at testing time, overcoming this issue. However, most online algorithms are centralized and work on convex problems (which GNNs scarcely lead to). This paper introduces novel architectures which solve the convexity restriction and can be easily updated in a distributed, online manner. Finally, we provide experiments, showing how these models can be applied to optimizing formation control in a swarm of flocking robots.

----

## [318] Isotonic Data Augmentation for Knowledge Distillation

**Authors**: *Wanyun Cui, Sen Yan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/319](https://doi.org/10.24963/ijcai.2021/319)

**Abstract**:

Knowledge distillation uses both real hard labels and soft labels predicted by teacher model as supervision. Intuitively, we expect the soft label probabilities and hard label probabilities to be concordant. However, in the real knowledge distillations, we found critical rank violations between hard labels and soft labels for augmented samples. For example, for an augmented sample x = 0.7 * cat + 0.3 * panda, a meaningful soft label distribution should have the same rank: P(cat|x)>P(panda|x)>P(other|x). But real teacher models usually violate the rank: P(tiger|x)>P(panda|x)>P(cat|x). We attribute the rank violations to the increased difficulty of understanding augmented samples for the teacher model. Empirically, we found the violations injuries the knowledge transfer. In this paper, we denote eliminating rank violations in data augmentation for knowledge distillation as isotonic data augmentation (IDA). We use isotonic regression (IR) -- a classic statistical algorithm -- to eliminate the rank violations. We show that IDA can be modeled as a tree-structured IR problem and gives an O(c*log(c)) optimal algorithm, where c is the number of labels. In order to further reduce the time complexity of the optimal algorithm, we also proposed a GPU-friendly approximation algorithm with linear time complexity. We have verified on variant datasets and data augmentation baselines that (1) the rank violation is a general phenomenon for data augmentation in knowledge distillation. And (2) our proposed IDA algorithms effectively increases the accuracy of knowledge distillation by solving the ranking violations.

----

## [319] Graph-Free Knowledge Distillation for Graph Neural Networks

**Authors**: *Xiang Deng, Zhongfei Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/320](https://doi.org/10.24963/ijcai.2021/320)

**Abstract**:

Knowledge distillation (KD) transfers knowledge from a teacher network to a student by enforcing the student to mimic the outputs of the pretrained teacher on training data. However, data samples are not always accessible in many cases due to large data sizes, privacy, or confidentiality. Many efforts have been made on addressing this problem for convolutional neural networks (CNNs) whose inputs lie in a grid domain within a continuous space such as images and videos, but largely overlook graph neural networks (GNNs) that handle non-grid data with different topology structures within a discrete space. The inherent differences between their inputs make these CNN-based approaches not applicable to GNNs. In this paper, we propose to our best knowledge the first dedicated approach to distilling knowledge from a GNN without graph data. The proposed graph-free KD (GFKD) learns graph topology structures for knowledge transfer by modeling them with multinomial distribution. We then introduce a gradient estimator to optimize this framework. Essentially, the gradients w.r.t. graph structures are obtained by only using GNN forward-propagation without back-propagation, which means that GFKD is compatible with modern GNN libraries such as DGL and Geometric. Moreover, we provide the strategies for handling different types of prior knowledge in the graph data or the GNNs. Extensive experiments demonstrate that GFKD achieves the state-of-the-art performance for distilling knowledge from GNNs without training data.

----

## [320] Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks

**Authors**: *Jianhao Ding, Zhaofei Yu, Yonghong Tian, Tiejun Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/321](https://doi.org/10.24963/ijcai.2021/321)

**Abstract**:

Spiking Neural Networks (SNNs), as bio-inspired energy-efficient neural networks, have attracted great attentions from researchers and industry. The most efficient way to train deep SNNs is through ANN-SNN conversion. However, the conversion usually suffers from accuracy loss and long inference time, which impede the practical application of SNN. In this paper, we theoretically analyze ANN-SNN conversion and derive sufficient conditions of the optimal conversion. To better correlate ANN-SNN and get greater accuracy, we propose Rate Norm Layer to replace the ReLU activation function in source ANN training, enabling direct conversion from a trained ANN to an SNN. Moreover, we propose an optimal fit curve to quantify the fit between the activation value of source ANN and the actual firing rate of target SNN. We show that the inference time can be reduced by optimizing the upper bound of the fit curve in the revised ANN to achieve fast inference. Our theory can explain the existing work on fast reasoning and get better results. The experimental results show that the proposed method achieves near loss-less conversion with VGG-16, PreActResNet-18, and deeper structures. Moreover, it can reach 8.6× faster reasoning performance under 0.265× energy consumption of the typical method. The code is available at https://github.com/DingJianhao/OptSNNConvertion-RNL-RIL.

----

## [321] Boosting Variational Inference With Locally Adaptive Step-Sizes

**Authors**: *Gideon Dresdner, Saurav Shekhar, Fabian Pedregosa, Francesco Locatello, Gunnar Rätsch*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/322](https://doi.org/10.24963/ijcai.2021/322)

**Abstract**:

Variational Inference makes a trade-off between the capacity of the variational family and the tractability of finding an approximate posterior distribution. Instead, Boosting Variational Inference allows practitioners to obtain increasingly good posterior approximations by spending more compute. The main obstacle to widespread adoption of Boosting Variational Inference is the amount of resources necessary to improve over a strong Variational Inference baseline. In our work, we trace this limitation back to the global curvature of the KL-divergence. We characterize how the global curvature impacts time and memory consumption, address the problem with the notion of local curvature, and provide a novel approximate backtracking algorithm for estimating local curvature. We give new theoretical convergence rates for our algorithms and provide experimental validation on synthetic and real-world datasets.

----

## [322] Automatic Translation of Music-to-Dance for In-Game Characters

**Authors**: *Yinglin Duan, Tianyang Shi, Zhipeng Hu, Zhengxia Zou, Changjie Fan, Yi Yuan, Xi Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/323](https://doi.org/10.24963/ijcai.2021/323)

**Abstract**:

Music-to-dance translation is an emerging and powerful feature in recent role-playing games. Previous works of this topic consider music-to-dance as a supervised motion generation problem based on time-series data. However, these methods require a large amount of training data pairs and may suffer from the degradation of movements. This paper provides a new solution to this task where we re-formulate the translation as a piece-wise dance phrase retrieval problem based on the choreography theory. With such a design, players are allowed to optionally edit the dance movements on top of our generation while other regression-based methods ignore such user interactivity. Considering that the dance motion capture is expensive that requires the assistance of professional dancers, we train our method under a semi-supervised learning fashion with a large unlabeled music dataset (20x than our labeled one) and also introduce self-supervised pre-training to improve the training stability and generalization performance. Experimental results suggest that our method not only generalizes well over various styles of music but also succeeds in choreography for game players. Our project including the large-scale dataset and supplemental materials is available at https://github.com/FuxiCV/music-to-dance.

----

## [323] Time-Series Representation Learning via Temporal and Contextual Contrasting

**Authors**: *Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, Chee Keong Kwoh, Xiaoli Li, Cuntai Guan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/324](https://doi.org/10.24963/ijcai.2021/324)

**Abstract**:

Learning decent representations from unlabeled time-series data with temporal dynamics is a very challenging task. In this paper, we propose an unsupervised Time-Series representation learning framework via Temporal and Contextual Contrasting (TS-TCC), to learn time-series representation from unlabeled data. First, the raw time-series data are transformed into two different yet correlated views by using weak and strong augmentations. Second, we propose a novel temporal contrasting module to learn robust temporal representations by designing a tough cross-view prediction task. Last, to further learn discriminative representations, we propose a contextual contrasting module built upon the contexts from the temporal contrasting module. It attempts to maximize the similarity among different contexts of the same sample while minimizing similarity among contexts of different samples. Experiments have been carried out on three real-world time-series datasets. The results manifest that training a linear classifier on top of the features learned by our proposed TS-TCC performs comparably with the supervised training. Additionally, our proposed TS-TCC shows high efficiency in few-labeled data and transfer learning scenarios. The code is publicly available at https://github.com/emadeldeen24/TS-TCC.

----

## [324] Jointly Learning Prices and Product Features

**Authors**: *Ehsan Emamjomeh-Zadeh, Renato Paes Leme, Jon Schneider, Balasubramanian Sivan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/325](https://doi.org/10.24963/ijcai.2021/325)

**Abstract**:

Product Design is an important problem in marketing research where a firm tries
to learn what features of a product are more valuable to consumers. We study
this problem from the viewpoint of online learning: a firm repeatedly interacts
with a buyer by choosing a product configuration as well as a price and
observing the buyer's purchasing decision. The goal of the firm is to maximize
revenue throughout the course of $T$ rounds by
learning the buyer's preferences.

We study both the case of a set of discrete products and the case of a continuous set of
allowable product features. In both cases we provide nearly tight
upper and lower regret bounds.

----

## [325] BAMBOO: A Multi-instance Multi-label Approach Towards VDI User Logon Behavior Modeling

**Authors**: *Wen-Ping Fan, Yao Zhang, Qichen Hao, Xinya Wu, Min-Ling Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/326](https://doi.org/10.24963/ijcai.2021/326)

**Abstract**:

Different to traditional on-premise VDI , the virtual desktops in DaaS (Desktop as a Service) are hosted in public cloud where virtual machines are charged based on usage. Accordingly, an adaptive power management system which can turn off spare virtual machines without sacrificing end user experience is of significant customer value as it can greatly help reduce the running cost. Generally, logon behavior modeling for VDI users serves as the key enabling-technique to fulfill intelligent power management. Prior attempts work by modeling logon behavior in a user-dependent manner with tailored single-instance feature representation, where the strong relationships among pool-sharing VDI users are ignored in the modeling framework. In this paper, a novel formulation towards VDI user logon behavior modeling is proposed by employing the multi-instance multi-label (MIML) techniques. Specifically, each user is grouped with supporting users whose behaviors are jointly modeled in the feature space with multi-instance representation as well as in the output space with multi-label prediction. The resulting MIML formulation is optimized by adapting the popular MIML boosting procedure via balanced error-rate minimization. Experimental studies on real VDI customers' data clearly validate the effectiveness of the proposed MIML-based approach against state-of-the-art VDI user logon behavior modeling techniques.

----

## [326] Contrastive Model Invertion for Data-Free Knolwedge Distillation

**Authors**: *Gongfan Fang, Jie Song, Xinchao Wang, Chengchao Shen, Xingen Wang, Mingli Song*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/327](https://doi.org/10.24963/ijcai.2021/327)

**Abstract**:

Model inversion, whose goal is to recover training data from a pre-trained model, has been recently proved feasible. However, existing inversion methods usually suffer from the mode collapse problem, where the synthesized instances are highly similar to each other and thus show limited effectiveness for downstream tasks, such as knowledge distillation. In this paper, we propose Contrastive Model Inversion (CMI), where the data diversity is explicitly modeled as an optimizable objective, to alleviate the mode collapse issue. Our main observation is that, under the constraint of the same amount of data, higher data diversity usually indicates stronger instance discrimination. To this end, we introduce in CMI a contrastive learning objective that encourages the synthesizing instances to be distinguishable from the already synthesized ones in previous batches. Experiments of pre-trained models on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CMI not only generates more visually plausible instances than the state of the arts, but also achieves significantly superior performance when the generated data are used for knowledge distillation. Code is available at https://github.com/zju-vipa/DataFree.

----

## [327] Deep Reinforcement Learning for Multi-contact Motion Planning of Hexapod Robots

**Authors**: *Huiqiao Fu, Kaiqiang Tang, Peng Li, Wenqi Zhang, Xinpeng Wang, Guizhou Deng, Tao Wang, Chunlin Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/328](https://doi.org/10.24963/ijcai.2021/328)

**Abstract**:

Legged locomotion in a complex environment requires careful planning of the footholds of legged robots. In this paper, a novel Deep Reinforcement Learning (DRL) method is proposed to implement multi-contact motion planning for hexapod robots moving on uneven plum-blossom piles. First, the motion of hexapod robots is formulated as a Markov Decision Process (MDP) with a speciﬁed reward function. Second, a transition feasibility model is proposed for hexapod robots, which describes the feasibility of the state transition under the condition of satisfying kinematics and dynamics, and in turn determines the rewards. Third, the footholds and Center-of-Mass (CoM) sequences are sampled from a diagonal Gaussian distribution and the sequences are optimized through learning the optimal policies using the designed DRL algorithm. Both of the simulation and experimental results on physical systems demonstrate the feasibility and efficiency of the proposed method. Videos are shown at https://videoviewpage.wixsite.com/mcrl.

----

## [328] On the Convergence of Stochastic Compositional Gradient Descent Ascent Method

**Authors**: *Hongchang Gao, Xiaoqian Wang, Lei Luo, Xinghua Shi*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/329](https://doi.org/10.24963/ijcai.2021/329)

**Abstract**:

The compositional minimax problem covers plenty of machine learning models such as the distributionally robust compositional optimization problem. However,  it is yet another understudied problem to optimize the compositional minimax problem. In this paper, we develop a novel efficient stochastic compositional gradient descent ascent method for optimizing the compositional minimax problem. Moreover, we establish the theoretical convergence rate of our proposed method. To the best of our knowledge, this is the first work achieving such a convergence rate for the compositional minimax problem. Finally, we conduct extensive experiments to demonstrate the effectiveness of our proposed method.

----

## [329] Learning Groupwise Explanations for Black-Box Models

**Authors**: *Jingyue Gao, Xiting Wang, Yasha Wang, Yulan Yan, Xing Xie*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/330](https://doi.org/10.24963/ijcai.2021/330)

**Abstract**:

We study two user demands that are important during the exploitation of explanations in practice: 1) understanding the overall model behavior faithfully with limited cognitive load and 2) predicting the model behavior accurately on unseen instances. We illustrate that the two user demands correspond to two major sub-processes in the human cognitive process and propose a unified framework to fulfill them simultaneously. Given a local explanation method, our framework jointly 1) learns a limited number of groupwise explanations that interpret the model behavior on most instances with high fidelity and 2) specifies the region where each explanation applies. Experiments on six datasets demonstrate the effectiveness of our method.

----

## [330] Video Summarization via Label Distributions Dual-Reward

**Authors**: *Yongbiao Gao, Ning Xu, Xin Geng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/331](https://doi.org/10.24963/ijcai.2021/331)

**Abstract**:

Reinforcement learning maps from perceived state representation to actions, which is adopted to solve the video summarization problem. The reward is crucial for deal with the video summarization task via reinforcement learning, since the reward signal defines the goal of video summarization. However, existing reward mechanism in reinforcement learning cannot handle the ambiguity which appears frequently in video summarization, i.e., the diverse consciousness by different people on the same video. To solve this problem, in this paper label distributions are mapped from the CNN and LSTM-based state representation to capture the subjectiveness of video summaries. The dual-reward is designed by measuring the similarity between user score distributions and the generated label distributions. Not only the average score but also the the variance of the subjective opinions are considered in summary generation. Experimental results on several benchmark datasets show that our proposed method outperforms other approaches under various settings.

----

## [331] BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing

**Authors**: *Aritra Ghosh, Andrew S. Lan*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/332](https://doi.org/10.24963/ijcai.2021/332)

**Abstract**:

Computerized adaptive testing (CAT) refers to a form of tests that are personalized to every student/test taker. CAT methods adaptively select the next most informative question/item for each student given their responses to previous questions, effectively reducing test length. Existing CAT methods use item response theory (IRT) models to relate student ability to their responses to questions and static question selection algorithms designed to reduce the ability estimation error as quickly as possible; therefore, these algorithms cannot improve by learning from large-scale student response data. In this paper, we propose BOBCAT, a Bilevel Optimization-Based framework for CAT to directly learn a data-driven question selection algorithm from training data. BOBCAT is agnostic to the underlying student response model and is computationally efficient during the adaptive testing process. Through extensive experiments on five real-world student response datasets, we show that BOBCAT outperforms existing CAT methods (sometimes significantly) at reducing test length.

----

## [332] Method of Moments for Topic Models with Mixed Discrete and Continuous Features

**Authors**: *Joachim Giesen, Paul Kahlmeyer, Sören Laue, Matthias Mitterreiter, Frank Nussbaum, Christoph Staudt, Sina Zarrieß*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/333](https://doi.org/10.24963/ijcai.2021/333)

**Abstract**:

Topic models are characterized by a latent class variable that represents the different topics. Traditionally, their observable variables are  modeled as discrete variables like, for instance, in the prototypical latent Dirichlet allocation (LDA) topic model. In LDA, words in text documents are encoded by discrete count vectors with respect to some dictionary. The classical approach for learning topic models optimizes a likelihood function that is non-concave due to the presence of the latent variable. Hence, this approach mostly boils down to using search heuristics like the EM algorithm for parameter estimation. Recently, it was shown that topic models can be learned with strong algorithmic and statistical guarantees through Pearson's method of moments. Here, we extend this line of work to topic models that feature discrete as well as continuous observable variables (features). Moving beyond discrete variables as in LDA allows for more sophisticated features and a natural extension of topic models to other modalities than text, like, for instance, images. We provide algorithmic and statistical guarantees for the method of moments applied to the extended topic model that we corroborate experimentally on synthetic data. We also demonstrate the applicability of our model on real-world document data with embedded images that we preprocess into continuous state-of-the-art feature vectors.

----

## [333] Bayesian Experience Reuse for Learning from Multiple Demonstrators

**Authors**: *Mike Gimelfarb, Scott Sanner, Chi-Guhn Lee*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/334](https://doi.org/10.24963/ijcai.2021/334)

**Abstract**:

Learning from Demonstrations (LfD) is a powerful approach for incorporating advice from experts in the form of demonstrations. However, demonstrations often come from multiple sub-optimal experts with conflicting goals, rendering them difficult to incorporate effectively in online settings. To address this, we formulate a quadratic program whose solution yields an adaptive weighting over experts, that can be used to sample experts with relevant goals. In order to compare different source and target task goals safely, we model their uncertainty using normal-inverse-gamma priors, whose posteriors are learned from demonstrations using Bayesian neural networks with a shared encoder. Our resulting approach, which we call Bayesian Experience Reuse, can be applied for LfD in static and dynamic decision-making settings. We demonstrate its effectiveness for minimizing multi-modal functions, and optimizing a high-dimensional supply chain with cost uncertainty, where it is also shown to improve upon the performance of the demonstrators' policies.

----

## [334] Fast Multi-label Learning

**Authors**: *Xiuwen Gong, Dong Yuan, Wei Bao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/335](https://doi.org/10.24963/ijcai.2021/335)

**Abstract**:

Embedding approaches have become one of the most pervasive techniques for multi-label classification. However, the training process of embedding methods usually involves a complex quadratic or semidefinite programming problem, or the model may even involve an NP-hard problem. Thus, such methods are prohibitive on large-scale applications. More importantly, much of the literature has already shown that the binary relevance (BR) method is usually good enough for some applications.
Unfortunately, BR runs slowly due to its linear dependence on the size of the input data. The goal of this paper is to provide a simple method, yet with provable guarantees, which can achieve competitive performance without a complex training
process. To achieve our goal, we provide a simple stochastic sketch strategy for multi-label classification and present theoretical results from both algorithmic and statistical learning perspectives. Our comprehensive empirical studies corroborate our theoretical findings and demonstrate the superiority of the proposed methods.

----

## [335] InverseNet: Augmenting Model Extraction Attacks with Training Data Inversion

**Authors**: *Xueluan Gong, Yanjiao Chen, Wenbin Yang, Guanghao Mei, Qian Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/336](https://doi.org/10.24963/ijcai.2021/336)

**Abstract**:

Cloud service providers, including Google, Amazon, and Alibaba, have now launched machine-learning-as-a-service (MLaaS) platforms, allowing clients to access sophisticated cloud-based machine learning models via APIs. Unfortunately, however, the commercial value of these models makes them alluring targets for theft, and their strategic position as part of the IT infrastructure of many companies makes them an enticing springboard for conducting further adversarial attacks. In this paper, we put forth a novel and effective attack strategy, dubbed InverseNet, that steals the functionality of black-box cloud-based models with only a small number of queries. The crux of the innovation is that, unlike existing model extraction attacks that rely on public datasets or adversarial samples, InverseNet constructs inversed training samples to increase the similarity between the extracted substitute model and the victim model. Further, only a small number of data samples with high confidence scores (rather than an entire dataset) are used to reconstruct the inversed dataset, which substantially reduces the attack cost. Extensive experiments conducted on three simulated victim models and Alibaba Cloud's commercially-available API demonstrate that InverseNet yields a model with significantly greater functional similarity to the victim model than the current state-of-the-art attacks at a substantially lower query budget.

----

## [336] Hierarchical Class-Based Curriculum Loss

**Authors**: *Palash Goyal, Divya Choudhary, Shalini Ghosh*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/337](https://doi.org/10.24963/ijcai.2021/337)

**Abstract**:

Classification algorithms in machine learning often assume a flat label space. However, most real world data have dependencies between the labels, which can often be captured by using a hierarchy. Utilizing this relation can help develop a model capable of satisfying the dependencies and improving model accuracy and interpretability. Further, as different levels in the hierarchy correspond to different granularities, penalizing each label equally can be detrimental to model learning. In this paper, we propose a loss function, hierarchical curriculum loss, with two properties: (i) satisfy hierarchical constraints present in the label space, and (ii) provide non-uniform weights to labels based on their levels in the hierarchy, learned implicitly by the training paradigm. We theoretically show that the proposed hierarchical class-based curriculum loss is a tight bound of 0-1 loss  among all losses satisfying the hierarchical constraints. We test our loss function on real world image data sets, and show that it significantly outperforms state-of-the-art baselines.

----

## [337] The Successful Ingredients of Policy Gradient Algorithms

**Authors**: *Sven Gronauer, Martin Gottwald, Klaus Diepold*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/338](https://doi.org/10.24963/ijcai.2021/338)

**Abstract**:

Despite the sublime success in recent years, the underlying mechanisms powering the advances of reinforcement learning are yet poorly understood. In this paper, we identify these mechanisms - which we call ingredients - in on-policy policy gradient methods and empirically determine their impact on the learning. To allow an equitable assessment, we conduct our experiments based on a unified and modular implementation. Our results underline the significance of recent algorithmic advances and demonstrate that reaching state-of-the-art performance may not need sophisticated algorithms but can also be accomplished by the combination of a few simple ingredients.

----

## [338] Learning Nash Equilibria in Zero-Sum Stochastic Games via Entropy-Regularized Policy Approximation

**Authors**: *Yue Guan, Qifan Zhang, Panagiotis Tsiotras*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/339](https://doi.org/10.24963/ijcai.2021/339)

**Abstract**:

We explore the use of policy approximations to reduce the computational cost of learning Nash equilibria in zero-sum stochastic games. We propose a new Q-learning type algorithm that uses a sequence of entropy-regularized soft policies to approximate the Nash policy during the Q-function updates. We prove that under certain conditions, by updating the entropy regularization, the algorithm converges to a Nash equilibrium. We also demonstrate the proposed algorithm's ability to transfer previous training experiences, enabling the agents to adapt quickly to new environments. We provide a dynamic hyper-parameter scheduling scheme to further expedite convergence. Empirical results applied to a number of stochastic games verify that the proposed algorithm converges to the Nash equilibrium, while exhibiting a major speed-up over existing algorithms.

----

## [339] Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion

**Authors**: *Xian-Jin Gui, Wei Wang, Zhang-Hao Tian*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/340](https://doi.org/10.24963/ijcai.2021/340)

**Abstract**:

Deep neural networks need large amounts of labeled data to achieve good performance. In real-world applications, labels are usually collected from non-experts such as crowdsourcing to save cost and thus are noisy. In the past few years, deep learning methods for dealing with noisy labels have been developed, many of which are based on the small-loss criterion. However, there are few theoretical analyses to explain why these methods could learn well from noisy labels. In this paper, we theoretically explain why the widely-used small-loss criterion works. Based on the explanation, we reformalize the vanilla small-loss criterion to better tackle noisy labels. The experimental results verify our theoretical explanation and also demonstrate the effectiveness of the reformalization.

----

## [340] Hindsight Value Function for Variance Reduction in Stochastic Dynamic Environment

**Authors**: *Jiaming Guo, Rui Zhang, Xishan Zhang, Shaohui Peng, Qi Yi, Zidong Du, Xing Hu, Qi Guo, Yunji Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/341](https://doi.org/10.24963/ijcai.2021/341)

**Abstract**:

Policy gradient methods are appealing in deep reinforcement learning but suffer from high variance of gradient estimate. To reduce the variance, the state value function is applied commonly. However, the effect of the state value function becomes limited in stochastic dynamic environments, where the unexpected state dynamics and rewards will increase the variance. In this paper, we propose to replace the state value function with a novel hindsight value function, which leverages the information from the future to reduce the variance of the gradient estimate for stochastic dynamic environments. Particularly, to obtain an ideally unbiased gradient estimate, we propose an information-theoretic approach, which optimizes the embeddings of the future to be independent of previous actions. In our experiments, we apply the proposed hindsight value function in stochastic dynamic environments, including discrete-action environments and continuous-action environments. Compared with the standard state value function, the proposed hindsight value function consistently reduces the variance, stabilizes the training, and improves the eventual policy.

----

## [341] DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation

**Authors**: *Lei Guo, Li Tang, Tong Chen, Lei Zhu, Quoc Viet Hung Nguyen, Hongzhi Yin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/342](https://doi.org/10.24963/ijcai.2021/342)

**Abstract**:

Shared-account  Cross-domain  Sequential  Recommendation (SCSR)  is the task of recommending the next item based on a sequence of recorded user behaviors, where multiple users share a single account, and their behaviours are available in multiple domains. Existing work on solving SCSR mainly relies on mining sequential patterns via RNN-based models, which are not expressive enough to capture the relationships among multiple entities. Moreover, all existing algorithms try to bridge two domains via knowledge transfer in the latent space, and the explicit cross-domain graph structure is unexploited. In this work, we propose a novel graph-based solution, namely DA-GCN, to address the above challenges. Specifically, we first link users and items in each domain as a graph. Then, we devise a domain-aware graph convolution network to learn user-specific node representations. To fully account for users' domain-specific preferences on items, two novel attention mechanisms are further developed to selectively guide the message passing process. Extensive experiments on two real-world datasets are conducted to demonstrate the superiority of our DA-GCN method.

----

## [342] Robust Regularization with Adversarial Labelling of Perturbed Samples

**Authors**: *Xiaohui Guo, Richong Zhang, Yaowei Zheng, Yongyi Mao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/343](https://doi.org/10.24963/ijcai.2021/343)

**Abstract**:

Recent researches have suggested that the predictive accuracy of neural network may contend with its adversarial robustness. This presents challenges in designing effective regularization schemes that also provide strong adversarial robustness. Revisiting Vicinal Risk Minimization (VRM) as a unifying regularization principle, we propose Adversarial Labelling of Perturbed Samples (ALPS) as a regularization scheme that aims at improving the generalization ability and adversarial robustness of the trained model. ALPS trains neural networks with synthetic samples formed by perturbing each authentic input sample towards another one along with an adversarially assigned label. The ALPS regularization objective is formulated as a min-max problem, in which the outer problem is minimizing an upper-bound of the VRM loss, and the inner problem is L1-ball constrained adversarial labelling on perturbed sample. The analytic solution to the induced inner maximization problem is elegantly derived, which enables computational efficiency. Experiments on the SVHN, CIFAR-10, CIFAR-100 and Tiny-ImageNet datasets show that the ALPS has a state-of-the-art regularization performance while also serving as an effective adversarial training scheme.

----

## [343] Enabling Retrain-free Deep Neural Network Pruning Using Surrogate Lagrangian Relaxation

**Authors**: *Deniz Gurevin, Mikhail A. Bragin, Caiwen Ding, Shanglin Zhou, Lynn Pepin, Bingbing Li, Fei Miao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/344](https://doi.org/10.24963/ijcai.2021/344)

**Abstract**:

Network pruning is a widely used technique to reduce computation cost and model size for deep neural networks. However, the typical three-stage pipeline, i.e., training, pruning and retraining (fine-tuning) significantly increases the overall training trails. In this paper, we develop a systematic weight-pruning optimization approach based on Surrogate Lagrangian relaxation (SLR), which is tailored to overcome difficulties caused by the discrete nature of the weight-pruning problem while ensuring fast convergence. 
We further accelerate the convergence of the SLR by using quadratic penalties. Model parameters obtained by SLR during the training phase are much closer to their optimal values as compared to those obtained by other state-of-the-art methods. We evaluate the proposed method on image classification tasks using CIFAR-10 and ImageNet, as well as object detection tasks using COCO 2014 and Ultra-Fast-Lane-Detection using TuSimple lane detection dataset. Experimental results demonstrate that our SLR-based weight-pruning optimization approach achieves higher compression rate than state-of-the-arts under the same accuracy requirement. It also achieves a high model accuracy even at the hard-pruning stage without retraining (reduces the traditional three-stage pruning to two-stage). Given a limited budget of retraining epochs, our approach quickly recovers the model accuracy.

----

## [344] Riemannian Stochastic Recursive Momentum Method for non-Convex Optimization

**Authors**: *Andi Han, Junbin Gao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/345](https://doi.org/10.24963/ijcai.2021/345)

**Abstract**:

We propose a stochastic recursive momentum method for Riemannian non-convex optimization that achieves a nearly-optimal complexity to find epsilon-approximate solution with one sample. The new algorithm requires one-sample gradient evaluations per iteration and does not require restarting with a large batch gradient, which is commonly used to obtain a faster rate. Extensive experiment results demonstrate the superiority of the proposed algorithm. Extensions to nonsmooth and constrained optimization settings are also discussed.

----

## [345] Fine-Grained Air Quality Inference via Multi-Channel Attention Model

**Authors**: *Qilong Han, Dan Lu, Rui Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/346](https://doi.org/10.24963/ijcai.2021/346)

**Abstract**:

In this paper, we study the problem of fine-grained air quality inference that predicts the air quality level of any location from air quality readings of nearby monitoring stations. We point out the importance of explicitly modeling both static and dynamic spatial correlations, and consequently propose a novel multi-channel attention model (MCAM) that models static and dynamic spatial correlations as separate channels. The static channel combines the beauty of attention mechanisms and graph-based spatial modeling via an adapted bilateral filtering technique, which considers not only locations' Euclidean distances but also their similarity of geo-context features. The dynamic channel learns stations' time-dependent spatial influence on a target location at each time step via long short-term memory (LSTM) networks and attention mechanisms. In addition, we introduce two novel ideas, atmospheric dispersion theories and the hysteretic nature of air pollutant dispersion, to better model the dynamic spatial correlation. We also devise a multi-channel graph convolutional fusion network to effectively fuse the graph outputs, along with other features, from both channels. Our extensive experiments on real-world benchmark datasets demonstrate that MCAM significantly outperforms the state-of-the-art solutions.

----

## [346] Model-Based Reinforcement Learning for Infinite-Horizon Discounted Constrained Markov Decision Processes

**Authors**: *Aria HasanzadeZonuzy, Dileep M. Kalathil, Srinivas Shakkottai*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/347](https://doi.org/10.24963/ijcai.2021/347)

**Abstract**:

In many real-world reinforcement learning (RL) problems, in addition to maximizing the  objective, the learning agent has to  maintain some necessary safety constraints.  We formulate the problem of learning a safe policy as  an  infinite-horizon discounted Constrained Markov Decision Process (CMDP)  with an unknown transition probability matrix, where the safety requirements are modeled as   constraints on expected cumulative costs.  We propose two model-based constrained reinforcement learning (CRL) algorithms for learning a safe policy, namely, (i) GM-CRL algorithm,  where the algorithm has access to a generative model, and (ii) UC-CRL  algorithm,  where the algorithm learns the model using an upper confidence style online exploration method.   We characterize the sample complexity of these algorithms, i.e., the the number of samples needed to ensure a desired level of accuracy with high probability, both with respect to objective maximization and constraint satisfaction.

----

## [347] State-Based Recurrent SPMNs for Decision-Theoretic Planning under Partial Observability

**Authors**: *Layton Hayes, Prashant Doshi, Swaraj Pawar, Hari Teja Tatavarti*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/348](https://doi.org/10.24963/ijcai.2021/348)

**Abstract**:

The sum-product network (SPN) has been extended to model sequence data with the recurrent SPN (RSPN), and to decision-making problems with sum-product-max networks (SPMN). In this paper, we build on the concepts introduced by these extensions and present state-based recurrent SPMNs (S-RSPMNs) as a generalization of SPMNs to sequential decision-making problems where the state may not be perfectly observed. As with recurrent SPNs, S-RSPMNs utilize a repeatable template network to model sequences of arbitrary lengths. We present an algorithm for learning compact template structures by identifying unique belief states and the transitions between them through a state matching process that utilizes augmented data.  In our knowledge, this is the first data-driven approach that learns graphical models for planning under partial observability, which can be solved efficiently. S-RSPMNs retain the linear solution complexity of SPMNs, and we demonstrate significant improvements in compactness of representation and the run time of structure learning and inference in sequential domains.

----

## [348] Beyond the Spectrum: Detecting Deepfakes via Re-Synthesis

**Authors**: *Yang He, Ning Yu, Margret Keuper, Mario Fritz*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/349](https://doi.org/10.24963/ijcai.2021/349)

**Abstract**:

The rapid advances in deep generative models over the past years have led to highly realistic media, known as deepfakes, that are commonly indistinguishable from real to human eyes. These advances make assessing the authenticity of visual data increasingly difficult and pose a misinformation threat to the trustworthiness of visual content in general. Although recent work has shown strong detection accuracy of such deepfakes, the success largely relies on identifying frequency artifacts in the generated images, which will not yield a sustainable detection approach as generative models continue evolving and closing the gap to real images. In order to overcome this issue, we propose a novel fake detection that is designed to re-synthesize testing images and extract visual cues for detection. The re-synthesis procedure is flexible, allowing us to incorporate a series of visual tasks - we adopt super-resolution, denoising and colorization as the re-synthesis. We demonstrate the improved effectiveness, cross-GAN generalization, and robustness against perturbations of our approach in a variety of detection scenarios involving multiple generators over CelebA-HQ, FFHQ, and LSUN datasets. Source code is available at https://github.com/SSAW14/BeyondtheSpectrum.

----

## [349] Interpretable Minority Synthesis for Imbalanced Classification

**Authors**: *Yi He, Fudong Lin, Xu Yuan, Nian-Feng Tzeng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/350](https://doi.org/10.24963/ijcai.2021/350)

**Abstract**:

This  paper  proposes  a  novel  oversampling  approach that strives to balance the class priors with a considerably imbalanced data distribution of high dimensionality. The  crux  of  our  approach  lies in learning interpretable latent representations that can model the synthetic mechanism of the minority samples by using a generative adversarial network(GAN). A Bayesian regularizer is imposed to guide the GAN to extract a set of salient features that are either disentangled or intensionally entangled, with their  interplay  controlled  by  a  prescribed  structure, defined with human-in-the-loop. As such, our GAN enjoys an improved sample complexity, being able  to  synthesize  high-quality  minority  samples even if the sizes of minority classes are extremely small  during  training.   Empirical  studies  substantiate that our approach can empower simple classifiers  to  achieve  superior  imbalanced  classification performance over the state-of-the-art competitors and is robust across various imbalance settings. Code is released in github.com/fudonglin/IMSIC.

----

## [350] DEEPSPLIT: An Efficient Splitting Method for Neural Network Verification via Indirect Effect Analysis

**Authors**: *Patrick Henriksen, Alessio Lomuscio*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/351](https://doi.org/10.24963/ijcai.2021/351)

**Abstract**:

We propose a novel, complete algorithm for the verification and analysis of feed-forward, ReLU-based neural networks. The algorithm, based on symbolic interval propagation, introduces a new method for determining split-nodes which evaluates the indirect effect that splitting has on the relaxations of successor nodes. We combine this with a new efficient linear-programming encoding of the splitting constraints to further improve the algorithm’s performance. The resulting implementation, DeepSplit, achieved speedups of 1–2 orders of magnitude and 21-34% fewer timeouts when compared to the current SoA toolkits.

----

## [351] Behavior Mimics Distribution: Combining Individual and Group Behaviors for Federated Learning

**Authors**: *Hua Huang, Fanhua Shang, Yuanyuan Liu, Hongying Liu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/352](https://doi.org/10.24963/ijcai.2021/352)

**Abstract**:

Federated Learning (FL) has become an active and promising distributed machine learning paradigm. As a result of statistical heterogeneity, recent studies clearly show that the performance of popular FL methods (e.g., FedAvg) deteriorates dramatically due to the client drift caused by local updates. This paper proposes a novel Federated Learning algorithm (called IGFL), which leverages both Individual and Group behaviors to mimic distribution, thereby improving the ability to deal with heterogeneity.  Unlike existing FL methods, our IGFL can be applied to both client and server optimization. As a by-product, we propose a new attention-based federated learning in the server optimization of IGFL. To the best of our knowledge, this is the first time to incorporate attention
mechanisms into federated optimization. We conduct extensive experiments and show that IGFL can significantly improve the performance of existing federated learning methods. Especially when the distributions of data among individuals are diverse, IGFL can improve the classification accuracy by about 13% compared with prior baselines.

----

## [352] UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks

**Authors**: *Jing Huang, Jie Yang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/353](https://doi.org/10.24963/ijcai.2021/353)

**Abstract**:

Hypergraph, an expressive structure with flexibility to model the higher-order correlations among entities, has recently attracted increasing attention from various research domains. Despite the success of Graph Neural Networks (GNNs) for graph representation learning, how to adapt the powerful GNN-variants directly into hypergraphs remains a challenging problem. In this paper, we propose UniGNN, a unified framework for interpreting the message passing process in graph and hypergraph neural networks, which can generalize general GNN models into hypergraphs. In this framework, meticulously-designed architectures aiming to deepen GNNs can also be incorporated into hypergraphs with the least effort. Extensive experiments have been conducted to demonstrate the effectiveness of UniGNN on multiple real-world datasets, which outperform the state-of-the-art approaches with a large margin. Especially for the DBLP dataset, we increase the accuracy from 77.4% to 88.8% in the semi-supervised hypernode classification task. We further prove that the proposed message-passing based UniGNN models are at most as powerful as the 1-dimensional Generalized Weisfeiler-Leman (1-GWL) algorithm in terms of distinguishing non-isomorphic hypergraphs. Our code is available at https://github.com/OneForward/UniGNN.

----

## [353] Asynchronous Active Learning with Distributed Label Querying

**Authors**: *Sheng-Jun Huang, Chen-Chen Zong, Kun-Peng Ning, Haibo Ye*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/354](https://doi.org/10.24963/ijcai.2021/354)

**Abstract**:

Active learning tries to learn an effective model with lowest labeling cost. Most existing active learning methods work in a synchronous way, which implies that the label querying can be performed only after the model updating in each iteration. While training models is usually time-consuming, it may lead to serious latency between two queries, especially in the crowdsourcing environments where there are many online annotators working simultaneously. This will significantly decrease the labeling efficiency and strongly limit the application of active learning in real tasks. To overcome this challenge, we propose a multi-server multi-worker framework for asynchronous active learning in the distributed environment. By maintaining two shared pools of candidate queries and labeled data respectively, the servers, the workers and the annotators efficiently corporate with each other without synchronization. Moreover, diverse sampling strategies from distributed workers are incorporated to select the most useful instances for model improving. Both theoretical analysis and experimental study validate the effectiveness of the proposed approach.

----

## [354] On the Neural Tangent Kernel of Deep Networks with Orthogonal Initialization

**Authors**: *Wei Huang, Weitao Du, Richard Yi Da Xu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/355](https://doi.org/10.24963/ijcai.2021/355)

**Abstract**:

The prevailing thinking is that orthogonal weights are crucial to enforcing dynamical isometry and speeding up training. The increase in learning speed that results from orthogonal initialization in linear networks has been well-proven. However, while the same is believed to also hold for nonlinear networks when the dynamical isometry condition is satisfied, the training dynamics behind this contention have not been thoroughly explored. In this work, we study the dynamics of ultra-wide networks across a range of architectures, including Fully Connected Networks (FCNs) and Convolutional Neural Networks (CNNs) with orthogonal initialization via neural tangent kernel (NTK). Through a series of propositions and lemmas, we prove that two NTKs, one corresponding to Gaussian weights and one to orthogonal weights, are equal when the network width is infinite. Further, during training, the NTK of an orthogonally-initialized infinite-width network should theoretically remain constant. This suggests that the orthogonal initialization cannot speed up training in the NTK (lazy training) regime, contrary to the prevailing thoughts. In order to explore under what circumstances can orthogonality accelerate training, we conduct a thorough empirical investigation outside the NTK regime. We find that when the hyper-parameters are set to achieve a linear regime in nonlinear activation, orthogonal initialization can improve the learning speed with a large learning rate or large depth.

----

## [355] On Explaining Random Forests with SAT

**Authors**: *Yacine Izza, João Marques-Silva*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/356](https://doi.org/10.24963/ijcai.2021/356)

**Abstract**:

Random Forest (RFs) are among the most widely used Machine Learning  (ML) classifiers. Even though RFs are not interpretable, there are no dedicated  non-heuristic approaches for computing explanations of RFs. Moreover, there is recent work on polynomial algorithms for explaining ML models, including naive Bayes classifiers. Hence, one question is whether finding explanations of RFs  can be solved in polynomial time. This paper answers this question negatively, by proving that computing one PI-explanation of an RF is D^P-hard. Furthermore, the paper proposes a propositional encoding for  computing explanations of RFs, thus enabling finding PI-explanations   with a SAT solver. This contrasts with earlier work on explaining boosted trees  (BTs) and neural networks (NNs), which requires encodings based on  SMT/MILP. Experimental results, obtained on a wide range of publicly available  datasets, demonstrate that the proposed SAT-based approach scales to  RFs of sizes common in practical applications. Perhaps more  importantly, the experimental results demonstrate that, for the vast  majority of examples considered, the SAT-based approach proposed in  this paper significantly outperforms existing heuristic approaches.

----

## [356] Reinforcement Learning for Route Optimization with Robustness Guarantees

**Authors**: *Tobias Jacobs, Francesco Alesiani, Gülcin Ermis*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/357](https://doi.org/10.24963/ijcai.2021/357)

**Abstract**:

Application of deep learning to NP-hard combinatorial optimization problems is an emerging research trend, and a number of interesting approaches have been published over the last few years. In this work we address robust optimization, which is a more complex variant where a max-min problem is to be solved. We obtain robust solutions by solving the inner minimization problem exactly and apply Reinforcement Learning to learn a heuristic for the outer problem. The minimization term in the inner objective represents an obstacle to existing RL-based approaches, as its value depends on the full solution in a non-linear manner and cannot be evaluated for partial solutions constructed by the agent over the course of each episode. We overcome this obstacle by defining the reward in terms of the one-step advantage over a baseline policy whose role can be played by any fast heuristic for the given problem. The agent is trained to maximize the total advantage, which, as we show, is equivalent to the original objective. We validate our approach by solving min-max versions of standard benchmarks for the Capacitated Vehicle Routing and the Traveling Salesperson Problem, where our agents obtain near-optimal solutions and improve upon the baselines.

----

## [357] Learning CNF Theories Using MDL and Predicate Invention

**Authors**: *Arcchit Jain, Clément Gautrais, Angelika Kimmig, Luc De Raedt*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/358](https://doi.org/10.24963/ijcai.2021/358)

**Abstract**:

We revisit the problem of learning logical theories from examples, one of the most quintessential problems in machine learning. More specifically, we develop an approach to learn CNF-formulae from satisfiability. This is a setting in which the examples correspond to partial interpretations and an example is classified as positive when it is logically consistent with the theory.
We present a novel algorithm, called Mistle -- Minimal SAT Theory Learner, for learning such theories. The distinguishing features are that 1) Mistle performs predicate invention and inverse resolution, 2) is based on the MDL principle to compress the data, and 3) combines this with frequent pattern mining to find the most interesting theories. 
The experiments demonstrate that Mistle can learn CNF theories accurately and works well in tasks involving compression and classification.

----

## [358] Learning to Learn Personalized Neural Network for Ventricular Arrhythmias Detection on Intracardiac EGMs

**Authors**: *Zhenge Jia, Zhepeng Wang, Feng Hong, Lichuan Ping, Yiyu Shi, Jingtong Hu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/359](https://doi.org/10.24963/ijcai.2021/359)

**Abstract**:

Life-threatening ventricular arrhythmias (VAs) detection on intracardiac electrograms (IEGMs) is essential to Implantable Cardioverter Defibrillators (ICDs). However, current VAs detection methods count on a variety of heuristic detection criteria, and require frequent manual interventions to personalize criteria parameters for each patient to achieve accurate detection. In this work, we propose a one-dimensional convolutional neural network (1D-CNN) based life-threatening VAs detection on IEGMs. The network architecture is elaborately designed to satisfy the extreme resource constraints of the ICD while maintaining high detection accuracy. We further propose a meta-learning algorithm with a novel patient-wise training tasks formatting strategy to personalize the 1D-CNN. The algorithm generates a well-generalized model initialization containing across-patient knowledge, and performs a quick adaptation of the model to the specific patient's IEGMs. In this way, a new patient could be immediately assigned with personalized 1D-CNN model parameters using limited input data. Compared with the conventional VAs detection method, the proposed method achieves 2.2% increased sensitivity for detecting VAs rhythm and 8.6% increased specificity for non-VAs rhythm.

----

## [359] SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging

**Authors**: *Ziyu Jia, Youfang Lin, Jing Wang, Xuehui Wang, Peiyi Xie, Yingbin Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/360](https://doi.org/10.24963/ijcai.2021/360)

**Abstract**:

Sleep staging is fundamental for sleep assessment and disease diagnosis. Although previous attempts to classify sleep stages have achieved high classification performance, several challenges remain open: 1) How to effectively extract salient waves in multimodal sleep data; 2) How to capture the multi-scale transition rules among sleep stages; 3) How to adaptively seize the key role of specific modality for sleep staging. To address these challenges, we propose SalientSleepNet, a multimodal salient wave detection network for sleep staging. Specifically, SalientSleepNet is a temporal fully convolutional network based on the $U^2$-Net architecture that is originally proposed for salient object detection in computer vision. It is mainly composed of two independent $U^2$-like streams to extract the salient features from multimodal data, respectively. Meanwhile, the multi-scale extraction module is designed to capture multi-scale transition rules among sleep stages. Besides, the multimodal attention module is proposed to adaptively capture valuable information from multimodal data for the specific sleep stage. Experiments on the two datasets demonstrate that SalientSleepNet outperforms the state-of-the-art baselines. It is worth noting that this model has the least amount of parameters compared with the existing deep neural network models.

----

## [360] Knowledge Consolidation based Class Incremental Online Learning with Limited Data

**Authors**: *Mohammed Asad Karim, Vinay Kumar Verma, Pravendra Singh, Vinay P. Namboodiri, Piyush Rai*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/361](https://doi.org/10.24963/ijcai.2021/361)

**Abstract**:

We propose a novel approach for class incremental online learning in a limited data setting. This problem setting is challenging because of the following constraints: (1) Classes are given incrementally, which necessitates a class incremental learning approach; (2)  Data for each class is given in an online fashion, i.e., each training example is seen only once during training; (3) Each class has very few training examples; and (4) We do not use or assume access to any replay/memory to store data from previous classes. Therefore, in this setting, we have to handle twofold problems of catastrophic forgetting and overfitting. In our approach, we learn robust representations that are generalizable across tasks without suffering from the problems of catastrophic forgetting and overfitting to accommodate future classes with limited samples. Our proposed method leverages the meta-learning framework with knowledge consolidation. The meta-learning framework helps the model for rapid learning when samples appear in an online fashion. Simultaneously, knowledge consolidation helps to learn a robust representation against forgetting under online updates to facilitate future learning. Our approach significantly outperforms other methods on several benchmarks.

----

## [361] Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation

**Authors**: *Taehyeon Kim, Jaehoon Oh, Nakyil Kim, Sangwook Cho, Se-Young Yun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/362](https://doi.org/10.24963/ijcai.2021/362)

**Abstract**:

Knowledge distillation (KD), transferring knowledge from a cumbersome teacher model to a lightweight student model, has been investigated to design efficient neural architectures. Generally, the objective function of KD is the Kullback-Leibler (KL) divergence loss between the softened probability distributions of the teacher model and the student model with the temperature scaling hyperparameter τ. Despite its widespread use, few studies have discussed how such softening influences generalization. Here, we theoretically show that the KL divergence loss focuses on the logit matching when τ increases and the label matching when τ goes to 0 and empirically show that the logit matching is positively correlated to performance improvement in general. From this observation, we consider an intuitive KD loss function, the mean squared error (MSE) between the logit vectors, so that the student model can directly learn the logit of the teacher model. The MSE loss outperforms the KL divergence loss, explained by the penultimate layer representations difference between the two losses. Furthermore, we show that sequential distillation can improve performance and that KD, using the KL divergence loss with small τ particularly, mitigates the label noise. The code to reproduce the experiments is publicly available online at https://github.com/jhoon-oh/kd_data/.

----

## [362] Epsilon Best Arm Identification in Spectral Bandits

**Authors**: *Tomás Kocák, Aurélien Garivier*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/363](https://doi.org/10.24963/ijcai.2021/363)

**Abstract**:

We propose an analysis of Probably Approximately Correct (PAC) identification of an ϵ-best arm in graph bandit models with Gaussian distributions. We consider finite but potentially very large bandit models where the set of arms is endowed with a graph structure, and we assume that the arms' expectations μ are smooth with respect to this graph. Our goal is to identify an arm whose expectation is at most ϵ below the largest of all means. We focus on the fixed-confidence setting: given a risk parameter δ, we consider sequential strategies that yield an ϵ-optimal arm with probability at least 1-δ. All such strategies use at least T*(μ)log(1/δ) samples, where R is the smoothness parameter. We identify the complexity term  T*(μ) as the solution of a min-max problem for which we give a game-theoretic analysis and an approximation procedure. This procedure is the key element required by the asymptotically optimal Track-and-Stop strategy.

----

## [363] Towards Scalable Complete Verification of Relu Neural Networks via Dependency-based Branching

**Authors**: *Panagiotis Kouvaros, Alessio Lomuscio*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/364](https://doi.org/10.24963/ijcai.2021/364)

**Abstract**:

We introduce an efficient method for the complete verification of ReLU-based feed-forward neural networks. The method implements branching on the ReLU states on the basis of a notion of dependency between the nodes. This results in dividing the original verification problem into a set of sub-problems whose MILP formulations require fewer integrality constraints. We evaluate the method on all of the ReLU-based fully connected networks from the first competition for neural network verification. The experimental results obtained show 145% performance gains over the present state-of-the-art in complete verification.

----

## [364] Solving Continuous Control with Episodic Memory

**Authors**: *Igor Kuznetsov, Andrey Filchenkov*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/365](https://doi.org/10.24963/ijcai.2021/365)

**Abstract**:

Episodic memory lets reinforcement learning algorithms remember and exploit promising experience from the past to improve agent performance. Previous works on memory mechanisms show benefits of using episodic-based data structures for discrete action problems in terms of sample-efficiency. The application of episodic memory for continuous control with a large action space is not trivial. Our study aims to answer the question: can episodic memory be used to improve agent's performance in continuous control? Our proposed algorithm combines episodic memory with Actor-Critic architecture by modifying critic's objective. We further improve performance by introducing episodic-based replay buffer prioritization. We evaluate our algorithm on OpenAI gym domains and show greater sample-efficiency compared with the state-of-the art model-free off-policy algorithms.

----

## [365] On Guaranteed Optimal Robust Explanations for NLP Models

**Authors**: *Emanuele La Malfa, Rhiannon Michelmore, Agnieszka M. Zbrzezny, Nicola Paoletti, Marta Kwiatkowska*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/366](https://doi.org/10.24963/ijcai.2021/366)

**Abstract**:

We build on abduction-based explanations for machine learning and develop a method for computing local explanations for neural network models in natural language processing (NLP). Our explanations comprise a subset of the words of the input text that satisfies two key features: optimality w.r.t. a user-defined cost function, such as the length of explanation, and robustness, in that they ensure prediction invariance for any bounded perturbation in the embedding space of the left-out words. We present two solution algorithms, respectively based on implicit hitting sets and maximum universal subsets, introducing a number of algorithmic improvements to speed up convergence of hard instances. We show how our method can be configured with different perturbation sets in the embedded space and used to detect bias in predictions by enforcing include/exclude constraints on biased terms, as well as to enhance existing heuristic-based NLP explanation frameworks such as Anchors. We evaluate our framework on three widely used sentiment analysis tasks and texts of up to 100 words from SST, Twitter and IMDB datasets, demonstrating the effectiveness of the derived explanations.

----

## [366] Topological Uncertainty: Monitoring Trained Neural Networks through Persistence of Activation Graphs

**Authors**: *Théo Lacombe, Yuichi Ike, Mathieu Carrière, Frédéric Chazal, Marc Glisse, Yuhei Umeda*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/367](https://doi.org/10.24963/ijcai.2021/367)

**Abstract**:

Although neural networks are capable of reaching astonishing performance on a wide variety of contexts, properly training networks on complicated tasks requires expertise and can be expensive from a computational perspective. In industrial applications, data coming from an open-world setting might widely differ from the benchmark datasets on which a network was trained. Being able to monitor the presence of such variations without retraining the network is of crucial importance. 

In this paper, we develop a method to monitor trained neural networks based on the topological properties of their activation graphs. To each new observation, we assign a Topological Uncertainty, a score that aims to assess the reliability of the predictions by investigating the whole network instead of its final layer only as typically done by practitioners. Our approach entirely works at a post-training level and does not require any assumption on the network architecture, optimization scheme, nor the use of data augmentation or auxiliary datasets; and can be faithfully applied on a large range of network architectures and data types. We showcase experimentally the potential of Topological Uncertainty in the context of trained network selection, Out-Of-Distribution detection, and shift-detection, both on synthetic and real datasets of images and graphs.

----

## [367] RetCL: A Selection-based Approach for Retrosynthesis via Contrastive Learning

**Authors**: *Hankook Lee, Sungsoo Ahn, Seung-Woo Seo, You Young Song, Eunho Yang, Sung Ju Hwang, Jinwoo Shin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/368](https://doi.org/10.24963/ijcai.2021/368)

**Abstract**:

Retrosynthesis, of which the goal is to find a set of reactants for synthesizing a target product, is an emerging research area of deep learning. While the existing approaches have shown promising results, they currently lack the ability to consider availability (e.g., stability or purchasability) of the reactants or generalize to unseen reaction templates (i.e., chemical reaction rules). In this paper, we propose a new approach that mitigates the issues by reformulating retrosynthesis into a selection problem of reactants from a candidate set of commercially available molecules. To this end, we design an efficient reactant selection framework, named RetCL (retrosynthesis via contrastive learning), for enumerating all of the candidate molecules based on selection scores computed by graph neural networks. For learning the score functions, we also propose a novel contrastive training scheme with hard negative mining. Extensive experiments demonstrate the benefits of the proposed selection-based approach. For example, when all 671k reactants in the USPTO database are given as candidates, our RetCL achieves top-1 exact match accuracy of 71.3% for the USPTO-50k benchmark, while a recent transformer-based approach achieves 59.6%. We also demonstrate that RetCL generalizes well to unseen templates in various settings in contrast to template-based approaches.

----

## [368] TextGTL: Graph-based Transductive Learning for Semi-supervised Text Classification via Structure-Sensitive Interpolation

**Authors**: *Chen Li, Xutan Peng, Hao Peng, Jianxin Li, Lihong Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/369](https://doi.org/10.24963/ijcai.2021/369)

**Abstract**:

Compared with traditional sequential learning models, graph-based neural networks exhibit excellent properties when encoding text, such as the capacity of capturing global and local information simultaneously. Especially in the semi-supervised scenario, propagating information along the edge can effectively alleviate the sparsity of labeled data. In this paper, beyond the existing architecture of heterogeneous word-document graphs, for the first time, we investigate how to construct lightweight non-heterogeneous graphs based on different linguistic information to better serve free text representation learning. Then, a novel semi-supervised framework for text classification that refines graph topology under theoretical guidance and shares information across different text graphs, namely Text-oriented Graph-based Transductive Learning (TextGTL), is proposed. TextGTL also performs attribute space interpolation based on dense substructure in graphs to predict low-entropy labels with high-quality feature nodes for data augmentation. To verify the effectiveness of TextGTL, we conduct extensive experiments on various benchmark datasets, observing significant performance gains over conventional heterogeneous graphs.  In addition, we also design ablation studies to dive deep into the validity of components in TextTGL.

----

## [369] Regularising Knowledge Transfer by Meta Functional Learning

**Authors**: *Pan Li, Yanwei Fu, Shaogang Gong*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/370](https://doi.org/10.24963/ijcai.2021/370)

**Abstract**:

Machine learning classifiersâ€™ capability is largely dependent on the scale of available training data and limited by the model overfitting in data-scarce learning tasks. To address this problem, this work proposes a novel Meta Functional Learning (MFL) by meta-learning a generalisable functional model from data-rich tasks whilst simultaneously regularising knowledge transfer to data-scarce tasks. The MFL computes meta-knowledge on functional regularisation generalisable to different learning tasks by which functional training on limited labelled data promotes more discriminative functions to be learned. Moreover, we adopt an Iterative Update strategy on MFL (MFL-IU). This improves knowledge transfer regularisation from MFL by progressively learning the functional regularisation in knowledge transfer. Experiments on three Few-Shot Learning (FSL) benchmarks (miniImageNet, CIFAR-FS and CUB) show that meta functional learning for regularisation knowledge transfer can benefit improving FSL classifiers.

----

## [370] Pairwise Half-graph Discrimination: A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks

**Authors**: *Pengyong Li, Jun Wang, Ziliang Li, Yixuan Qiao, Xianggen Liu, Fei Ma, Peng Gao, Sen Song, Guotong Xie*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/371](https://doi.org/10.24963/ijcai.2021/371)

**Abstract**:

Self-supervised learning has gradually emerged as a powerful technique for graph representation learning. However, transferable, generalizable, and robust representation learning on graph data still remains a challenge for pre-training graph neural networks. In this paper, we propose a simple and effective self-supervised pre-training strategy, named Pairwise Half-graph Discrimination (PHD), that explicitly pre-trains a graph neural network at graph-level. PHD is designed as a simple binary classification task to discriminate whether two half-graphs come from the same source. Experiments demonstrate that the PHD is an effective pre-training strategy that offers comparable or superior performance on 13 graph classification tasks compared with state-of-the-art strategies, and achieves notable improvements when combined with node-level strategies. Moreover, the visualization of learned representation revealed that PHD strategy indeed empowers the model to learn graph-level knowledge like the molecular scaffold. These results have established PHD as a powerful and effective self-supervised learning strategy in graph-level representation learning.

----

## [371] SHPOS: A Theoretical Guaranteed Accelerated Particle Optimization Sampling Method

**Authors**: *Zhijian Li, Chao Zhang, Hui Qian, Xin Du, Lingwei Peng*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/372](https://doi.org/10.24963/ijcai.2021/372)

**Abstract**:

Recently, the Stochastic Particle Optimization Sampling (SPOS) method is proposed to solve the particle-collapsing pitfall of deterministic Particle Variational Inference methods by ultilizing the stochastic Overdamped Langevin dynamics to enhance exploration. In this paper, we propose an accelerated particle optimization sampling method called Stochastic Hamiltonian Particle Optimization Sampling (SHPOS). Compared to the first-order dynamics used in SPOS, SHPOS adopts an augmented second-order dynamics, which involves an extra momentum term to achieve acceleration. We establish a non-asymptotic convergence analysis for SHPOS, and show that it enjoys a faster convergence rate than SPOS. Besides, we also propose a variance-reduced stochastic gradient variant of SHPOS for tasks with large-scale datasets and complex models. Experiments on both synthetic and real data validate our theory and demonstrate the superiority of SHPOS over the state-of-the-art.

----

## [372] An Adaptive News-Driven Method for CVaR-sensitive Online Portfolio Selection in Non-Stationary Financial Markets

**Authors**: *Qianqiao Liang, Mengying Zhu, Xiaolin Zheng, Yan Wang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/373](https://doi.org/10.24963/ijcai.2021/373)

**Abstract**:

CVaR-sensitive online portfolio selection (CS-OLPS) becomes increasingly important for investors because of its effectiveness to minimize conditional value at risk (CVaR) and control extreme losses. However, the non-stationary nature of financial markets makes it very difficult to address the CS-OLPS problem effectively. To address the CS-OLPS problem in non-stationary markets, we propose an effective news-driven method, named CAND, which adaptively exploits news to determine the adjustment tendency and adjustment scale for tracking the dynamic optimal portfolio with minimal CVaR in each trading round. In addition, we devise a filtering mechanism to reduce the errors caused by the noisy news for further improving CAND's effectiveness. We rigorously prove a sub-linear regret of CAND. Extensive experiments on three real-world datasets demonstrate CANDâ€™s superiority over the state-of-the-art portfolio methods in terms of returns and risks.

----

## [373] Residential Electric Load Forecasting via Attentive Transfer of Graph Neural Networks

**Authors**: *Weixuan Lin, Di Wu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/374](https://doi.org/10.24963/ijcai.2021/374)

**Abstract**:

An accurate short-term electric load forecasting is critical for modern electric power systems' safe and economical operation. 
Electric load forecasting can be formulated as a multi-variate time series problem. Residential houses in the same neighborhood may be affected by similar factors and share some latent spatial dependencies. However, most of the existing works on electric load forecasting fail to explore such dependencies. In recent years, graph neural networks (GNNs) have shown impressive success in modeling such dependencies. However, such GNN based models usually would require a large amount of training data. We may have a minimal amount of data available to train a reliable forecasting model for houses in a new neighborhood area. At the same time, we may have a large amount of historical data collected from other houses that can be leveraged to improve the new neighborhood's prediction performance. In this paper, we propose an attentive transfer learning-based GNN model that can utilize the learned prior knowledge to improve the learning process in a new area.  The transfer process is achieved by an attention network, which generically avoids negative transfer by leveraging knowledge from multiple sources.
Extensive experiments have been conducted on real-world data sets.  Results have shown that the proposed framework can consistently outperform baseline models in different areas.

----

## [374] Graph Filter-based Multi-view Attributed Graph Clustering

**Authors**: *Zhiping Lin, Zhao Kang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/375](https://doi.org/10.24963/ijcai.2021/375)

**Abstract**:

Graph clustering has become an important research topic due to the proliferation of graph data. However, existing methods suffer from two major drawbacks. On the one hand, most methods can not simultaneously exploit attribute and graph structure information. On the other hand, most methods are incapable of handling multi-view data which contain sets of different features and graphs. In this paper, we propose a novel Multi-view Attributed Graph Clustering (MvAGC) method, which is simple yet effective. Firstly, a graph filter is applied to features to obtain a smooth representation without the need of learning the parameters of neural networks. Secondly, a novel strategy is designed to select a few anchor points, so as to reduce the computation complexity. Thirdly, a new regularizer is developed to explore high-order neighborhood information. Our extensive experiments indicate that our method works surprisingly well with respect to state-of-the-art deep neural network methods. The source code is available at https://github.com/sckangz/MvAGC.

----

## [375] On the Intrinsic Differential Privacy of Bagging

**Authors**: *Hongbin Liu, Jinyuan Jia, Neil Zhenqiang Gong*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/376](https://doi.org/10.24963/ijcai.2021/376)

**Abstract**:

Differentially private machine learning trains models while protecting privacy of the sensitive training data. The key to obtain differentially private models is to introduce noise/randomness to the training process. In particular, existing differentially private machine learning methods add noise to the training data, the gradients, the loss function, and/or the model itself. Bagging, a popular ensemble learning framework, randomly creates some subsamples of the training data, trains a base model for each subsample using a base learner, and takes majority vote among the base models when making predictions.  Bagging has intrinsic randomness in the training process as it randomly creates subsamples. Our major theoretical results show that such intrinsic randomness already makes Bagging differentially private without the needs of additional noise. Moreover, we prove that if no assumptions about the base learner are made, our derived privacy guarantees are tight. We empirically evaluate Bagging on MNIST and CIFAR10. Our experimental results demonstrate that Bagging achieves significantly higher accuracies than state-of-the-art differentially private machine learning methods with the same privacy budgets.

----

## [376] Two-stage Training for Learning from Label Proportions

**Authors**: *Jiabin Liu, Bo Wang, Xin Shen, Zhiquan Qi, Yingjie Tian*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/377](https://doi.org/10.24963/ijcai.2021/377)

**Abstract**:

Learning from label proportions (LLP) aims at learning an instance-level classifier with label proportions in grouped training data. Existing deep learning based LLP methods utilize end-to-end pipelines to obtain the proportional loss with Kullback-Leibler divergence between the bag-level prior and posterior class distributions. However, the unconstrained optimization on this objective can hardly reach a solution in accordance with the given proportions. Besides, concerning the probabilistic classifier, this strategy unavoidably results in high-entropy conditional class distributions at the instance level. These issues further degrade the performance of the instance-level classification. In this paper, we regard these problems as noisy pseudo labeling, and instead impose the strict proportion consistency on the classifier with a constrained optimization as a continuous training stage for existing LLP classifiers. In addition, we introduce the mixup strategy and symmetric cross-entropy to further reduce the label noise. Our framework is model-agnostic, and demonstrates compelling performance improvement in extensive experiments, when incorporated into other deep LLP models as a post-hoc phase.

----

## [377] Adversarial Spectral Kernel Matching for Unsupervised Time Series Domain Adaptation

**Authors**: *Qiao Liu, Hui Xue*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/378](https://doi.org/10.24963/ijcai.2021/378)

**Abstract**:

Unsupervised domain adaptation (UDA) has been received increasing attention since it does not require labels in target domain. Most existing UDA methods learn domain-invariant features by minimizing discrepancy distance computed by a certain metric between domains. However, these discrepancy-based methods cannot be robustly applied to unsupervised time series domain adaptation (UTSDA). That is because discrepancy metrics in these methods contain only low-order and local statistics, which have limited expression for time series distributions and therefore result in failure of domain matching. Actually, the real-world time series are always non-local distributions, i.e., with non-stationary and non-monotonic statistics. In this paper, we propose an Adversarial Spectral Kernel Matching (AdvSKM) method, where a hybrid spectral kernel network is specifically designed as inner kernel to reform the Maximum Mean Discrepancy (MMD) metric for UTSDA. The hybrid spectral kernel network can precisely characterize non-stationary and non-monotonic statistics in time series distributions. Embedding hybrid spectral kernel network to MMD not only guarantees precise discrepancy metric but also benefits domain matching. Besides, the differentiable architecture of the spectral kernel network enables adversarial kernel learning, which brings more discriminatory expression for discrepancy matching. The results of extensive experiments on several real-world UTSDA tasks verify the effectiveness of our proposed method.

----

## [378] Smart Contract Vulnerability Detection: From Pure Neural Network to Interpretable Graph Feature and Expert Pattern Fusion

**Authors**: *Zhenguang Liu, Peng Qian, Xiang Wang, Lei Zhu, Qinming He, Shouling Ji*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/379](https://doi.org/10.24963/ijcai.2021/379)

**Abstract**:

Smart contracts hold digital coins worth billions of dollars, their security issues have drawn extensive attention in the past years. Towards smart contract vulnerability detection, conventional methods heavily rely on fixed expert rules, leading to low accuracy and poor scalability. Recent deep learning approaches alleviate this issue but fail to encode useful expert knowledge. In this paper, we explore combining deep learning with expert patterns in an explainable fashion. Specifically, we develop automatic tools to extract expert patterns from the source code. We then cast the code into a semantic graph to extract deep graph features. Thereafter, the global graph feature and local expert patterns are fused to cooperate and approach the final prediction, while yielding their interpretable weights. Experiments are conducted on all available smart contracts with source code in two platforms, Ethereum and VNT Chain. Empirically, our system significantly outperforms state-of-the-art methods. Our code is released.

----

## [379] Transfer Learning via Optimal Transportation for Integrative Cancer Patient Stratification

**Authors**: *Ziyu Liu, Wei Shao, Jie Zhang, Min Zhang, Kun Huang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/380](https://doi.org/10.24963/ijcai.2021/380)

**Abstract**:

The Stratification of early-stage cancer patients for the prediction of clinical outcome is a challenging task since cancer is associated with various molecular aberrations. A single biomarker often cannot provide sufficient information to stratify early-stage patients effectively. Understanding the complex mechanism behind cancer development calls for exploiting biomarkers from multiple modalities of data such as histopathology images and genomic data. The integrative analysis of these biomarkers sheds light on cancer diagnosis, subtyping, and prognosis. Another difficulty is that labels for early-stage cancer patients are scarce and not reliable enough for predicting survival times. Given the fact that different cancer types share some commonalities, we explore if the knowledge learned from one cancer type can be utilized to improve prognosis accuracy for another cancer type. We propose a novel unsupervised multi-view transfer learning algorithm to simultaneously analyze multiple biomarkers in different cancer types. We integrate multiple views using non-negative matrix factorization and formulate the transfer learning model based on the Optimal Transport theory to align features of different cancer types. We evaluate the stratification performance on three early-stage cancers from the Cancer Genome Atlas (TCGA) project. Comparing with other benchmark methods, our framework achieves superior accuracy for patient outcome prediction.

----

## [380] Graph Entropy Guided Node Embedding Dimension Selection for Graph Neural Networks

**Authors**: *Gongxu Luo, Jianxin Li, Hao Peng, Carl Yang, Lichao Sun, Philip S. Yu, Lifang He*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/381](https://doi.org/10.24963/ijcai.2021/381)

**Abstract**:

Graph representation learning has achieved great success in many areas, including e-commerce, chemistry, biology, etc. However, the fundamental problem of choosing the appropriate dimension of node embedding for a given graph still remains unsolved. The commonly used strategies for Node Embedding Dimension Selection (NEDS) based on grid search or empirical knowledge suffer from heavy computation and poor model performance. In this paper, we revisit NEDS from the perspective of minimum entropy principle. Subsequently, we propose a novel Minimum Graph Entropy (MinGE) algorithm for NEDS with graph data. To be specific, MinGE considers both feature entropy and structure entropy on graphs, which are carefully designed according to the characteristics of the rich information in them. The feature entropy, which assumes the embeddings of adjacent nodes to be more similar, connects node features and link topology on graphs. The structure entropy takes the normalized degree as basic unit to further measure the higher-order structure of graphs. Based on them, we design MinGE to directly calculate the ideal node embedding dimension for any graph. Finally, comprehensive experiments with popular Graph Neural Networks (GNNs) on benchmark datasets demonstrate the effectiveness and generalizability of our proposed MinGE.

----

## [381] Stochastic Actor-Executor-Critic for Image-to-Image Translation

**Authors**: *Ziwei Luo, Jing Hu, Xin Wang, Siwei Lyu, Bin Kong, Youbing Yin, Qi Song, Xi Wu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/382](https://doi.org/10.24963/ijcai.2021/382)

**Abstract**:

Training a model-free deep reinforcement learning model to solve image-to-image translation is difficult since it involves high-dimensional continuous state and action spaces. In this paper, we draw inspiration from the recent success of the maximum entropy reinforcement learning framework designed for challenging continuous control problems to develop stochastic policies over high dimensional continuous spaces including image representation, generation, and control simultaneously. Central to this method is the Stochastic Actor-Executor-Critic (SAEC) which is an off-policy actor-critic model with an additional executor to generate realistic images. Specifically, the actor focuses on the high-level representation and control policy by a stochastic latent action, as well as explicitly directs the executor to generate low-level actions to manipulate the state. Experiments on several image-to-image translation tasks have demonstrated the effectiveness and robustness of the proposed SAEC when facing high-dimensional continuous space problems.

----

## [382] Hierarchical Temporal Multi-Instance Learning for Video-based Student Learning Engagement Assessment

**Authors**: *Jiayao Ma, Xinbo Jiang, Songhua Xu, Xueying Qin*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/383](https://doi.org/10.24963/ijcai.2021/383)

**Abstract**:

Video-based automatic assessment of a student's learning engagement on the fly can provide immense values for delivering personalized instructional services, a vehicle particularly important for massive online education. To train such an assessor, a major challenge lies in the collection of sufficient labels at the appropriate temporal granularity since a learner's engagement status may continuously change throughout a study session. Supplying labels at either frame or clip level incurs a high annotation cost. To overcome such a challenge, this paper proposes a novel hierarchical multiple instance learning (MIL) solution, which only requires labels anchored on full-length videos to learn to assess student engagement at an arbitrary temporal granularity and for an arbitrary duration in a study session. The hierarchical model mainly comprises a bottom module and a top module, respectively dedicated to learning the latent relationship between a clip and its constituent frames and that between a video and its constituent clips, with the constraints on the training stage that the average engagements of local clips is that of the video label. To verify the effectiveness of our method, we compare the performance of the proposed approach with that of several state-of-the-art peer solutions through extensive experiments.

----

## [383] Multi-Cause Effect Estimation with Disentangled Confounder Representation

**Authors**: *Jing Ma, Ruocheng Guo, Aidong Zhang, Jundong Li*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/384](https://doi.org/10.24963/ijcai.2021/384)

**Abstract**:

One fundamental problem in causality learning is to estimate the causal effects of one or multiple treatments (e.g., medicines in the prescription) on an important outcome (e.g., cure of a disease). One major challenge of causal effect estimation is the existence of unobserved confounders -- the unobserved variables that affect both the treatments and the outcome. Recent studies have shown that by modeling how instances are assigned with different treatments together, the patterns of unobserved confounders can be captured through their learned latent representations. However, the interpretability of the representations in these works is limited. In this paper, we focus on the multi-cause effect estimation problem from a new perspective by learning disentangled representations of confounders. The disentangled representations not only facilitate the treatment effect estimation but also strengthen the understanding of causality learning process. Experimental results on both synthetic and real-world datasets show the superiority of our proposed framework from different aspects.

----

## [384] Average-Reward Reinforcement Learning with Trust Region Methods

**Authors**: *Xiaoteng Ma, Xiaohang Tang, Li Xia, Jun Yang, Qianchuan Zhao*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/385](https://doi.org/10.24963/ijcai.2021/385)

**Abstract**:

Most of reinforcement learning algorithms optimize the discounted criterion which is beneficial to accelerate the convergence and reduce the variance of estimates. Although the discounted criterion is appropriate for certain tasks such as financial related problems, many engineering problems treat future rewards equally and prefer a long-run average criterion. In this paper, we study the reinforcement learning problem with the long-run average criterion. Firstly, we develop a unified trust region theory with discounted and average criteria. With the average criterion, a novel performance bound within the trust region is derived with the Perturbation Analysis (PA) theory. Secondly, we propose a practical algorithm named Average Policy Optimization (APO), which improves the value estimation with a novel technique named Average Value Constraint. To the best of our knowledge, our work is the first one to study the trust region approach with the average criterion and it complements the framework of reinforcement learning beyond the discounted criterion. Finally, experiments are conducted in the continuous control environment MuJoCo. In most tasks, APO performs better than the discounted PPO, which demonstrates the effectiveness of our approach.

----

## [385] Temporal and Object Quantification Networks

**Authors**: *Jiayuan Mao, Zhezheng Luo, Chuang Gan, Joshua B. Tenenbaum, Jiajun Wu, Leslie Pack Kaelbling, Tomer D. Ullman*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/386](https://doi.org/10.24963/ijcai.2021/386)

**Abstract**:

We present Temporal and Object Quantification Networks (TOQ-Nets), a new class of neuro-symbolic networks with a structural bias that enables them to learn to recognize complex relational-temporal events. This is done by including reasoning layers that implement finite-domain quantification over objects and time. The structure allows them to generalize directly to input instances with varying numbers of objects in temporal sequences of varying lengths. We evaluate TOQ-Nets on input domains that require recognizing event-types in terms of complex temporal relational patterns. We demonstrate that TOQ-Nets can generalize from small amounts of data to scenarios containing more objects than were present during training and to temporal warpings of input sequences.

----

## [386] Evaluating Relaxations of Logic for Neural Networks: A Comprehensive Study

**Authors**: *Mattia Medina Grespan, Ashim Gupta, Vivek Srikumar*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/387](https://doi.org/10.24963/ijcai.2021/387)

**Abstract**:

Symbolic knowledge can provide crucial inductive bias for training neural models, especially in low data regimes. A successful strategy for incorporating such knowledge involves relaxing logical statements into sub-differentiable losses for optimization. In this paper, we study the question of how best to relax logical expressions that represent labeled examples and knowledge about a problem; we focus on sub-differentiable t-norm relaxations of logic. We present theoretical and empirical criteria for characterizing which relaxation would perform best in various scenarios. In our theoretical study driven by the goal of preserving tautologies, the Lukasiewicz t-norm performs best. However, in our empirical analysis on the text chunking and digit recognition tasks, the product t-norm achieves best predictive performance. We analyze this apparent discrepancy, and conclude with a list of best practices for defining loss functions via logic.

----

## [387] Minimization of Limit-Average Automata

**Authors**: *Jakub Michaliszyn, Jan Otop*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/388](https://doi.org/10.24963/ijcai.2021/388)

**Abstract**:

LimAvg-automata are weighted automata over infinite words that aggregate weights along runs with the limit-average value function. 
In this paper, we study the minimization problem for (deterministic) LimAvg-automata. 
Our main contribution is an equivalence relation on words characterizing LimAvg-automata, i.e., 
the equivalence classes of this relation correspond to states of an equivalent LimAvg-automaton. 
In contrast to relations characterizing DFA, our relation depends not only on the function defined by the target automaton, but also on its structure. 

We show two applications of this relation. 
First, we present a minimization algorithm for LimAvg-automata, which returns a minimal LimAvg-automaton among those equivalent and structurally similar to the input one. 
Second, we present an extension of Angluin's L^*-algorithm with syntactic queries, which learns in polynomial time a LimAvg-automaton equivalent to the target one.

----

## [388] Details (Don't) Matter: Isolating Cluster Information in Deep Embedded Spaces

**Authors**: *Lukas Miklautz, Lena G. M. Bauer, Dominik Mautz, Sebastian Tschiatschek, Christian Böhm, Claudia Plant*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/389](https://doi.org/10.24963/ijcai.2021/389)

**Abstract**:

Deep clustering techniques combine representation learning with clustering objectives to improve their performance. Among existing deep clustering techniques, autoencoder-based methods are the most prevalent ones. While they achieve promising clustering results, they suffer from an inherent conflict between preserving details, as expressed by the reconstruction loss, and finding similar groups by ignoring details, as expressed by the clustering loss. This conflict leads to brittle training procedures, dependence on trade-off hyperparameters and less interpretable results. We propose our framework, ACe/DeC, that is compatible with Autoencoder Centroid based Deep Clustering methods and automatically learns a latent representation consisting of two separate spaces. The clustering space captures all cluster-specific information and the shared space explains general variation in the data. This separation resolves the above mentioned conflict and allows our method to learn both detailed reconstructions and cluster specific abstractions.
We evaluate our framework with extensive experiments to show several benefits: (1) cluster performance – on various data sets we outperform relevant baselines; (2) no hyperparameter tuning – this improved performance is achieved without introducing new clustering specific hyperparameters; (3) interpretability – isolating the cluster specific information in a separate space is advantageous for data exploration and interpreting the clustering results; and (4) dimensionality of the embedded space – we automatically learn a low dimensional space for clustering. 
Our ACe/DeC framework isolates cluster information, increases stability and interpretability, while improving cluster performance.

----

## [389] Contrastive Losses and Solution Caching for Predict-and-Optimize

**Authors**: *Maxime Mulamba, Jayanta Mandi, Michelangelo Diligenti, Michele Lombardi, Victor Bucarey, Tias Guns*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/390](https://doi.org/10.24963/ijcai.2021/390)

**Abstract**:

Many decision-making processes involve solving a combinatorial optimization problem with uncertain input that can be estimated from historic data. Recently, problems in this class have been successfully addressed via end-to-end learning approaches, which rely on solving one optimization problem for each training instance at every epoch. In this context, we provide two distinct contributions. First, we use a Noise Contrastive approach to motivate a family of surrogate loss functions, based on viewing non-optimal solutions as negative examples. Second, we address a major bottleneck of all predict-and-optimize approaches, i.e. the need to frequently recompute optimal solutions at training time. This is done via a solver-agnostic solution caching scheme, and by replacing optimization calls with a lookup in the solution cache. The method is formally based on an inner approximation of the feasible space and, combined with a cache lookup strategy, provides a controllable trade-off between training time and accuracy of the loss approximation. We empirically show that even a very slow growth rate is enough to match the quality of state-of-the-art methods, at a fraction of the computational cost.

----

## [390] Fine-grained Generalization Analysis of Structured Output Prediction

**Authors**: *Waleed Mustafa, Yunwen Lei, Antoine Ledent, Marius Kloft*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/391](https://doi.org/10.24963/ijcai.2021/391)

**Abstract**:

In machine learning we often encounter structured output prediction problems (SOPPs), i.e. problems where the output space admits a rich internal structure. Application domains where SOPPs naturally occur include natural language processing, speech recognition, and computer vision. Typical SOPPs have an extremely large label set, which grows exponentially as a function of the size of the output. Existing generalization analysis implies generalization bounds with at least a square-root dependency on the cardinality d of the label set, which can be vacuous in practice. In this paper, we significantly improve the state of the art by developing novel high-probability bounds with a logarithmic dependency on d. Furthermore, we leverage the lens of algorithmic stability to develop generalization bounds in expectation without any dependency on d. Our results therefore build a solid theoretical foundation for learning in large-scale SOPPs. Furthermore, we extend our results to learning with weakly dependent data.

----

## [391] Accelerating Neural Architecture Search via Proxy Data

**Authors**: *Byunggook Na, Jisoo Mok, Hyeokjun Choe, Sungroh Yoon*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/392](https://doi.org/10.24963/ijcai.2021/392)

**Abstract**:

Despite the increasing interest in neural architecture search (NAS), the significant computational cost of NAS is a hindrance to researchers. Hence, we propose to reduce the cost of NAS using proxy data, i.e., a representative subset of the target data, without sacrificing search performance. Even though data selection has been used across various fields, our evaluation of existing selection methods for NAS algorithms offered by NAS-Bench-1shot1 reveals that they are not always appropriate for NAS and a new selection method is necessary. By analyzing proxy data constructed using various selection methods through data entropy, we propose a novel proxy data selection method tailored for NAS. To empirically demonstrate the effectiveness, we conduct thorough experiments across diverse datasets, search spaces, and NAS algorithms. Consequently, NAS algorithms with the proposed selection discover architectures that are competitive with those obtained using the entire dataset. It significantly reduces the search cost: executing DARTS with the proposed selection requires only 40 minutes on CIFAR-10 and 7.5 hours on ImageNet with a single GPU. Additionally, when the architecture searched on ImageNet using the proposed selection is inversely transferred to CIFAR-10, a state-of-the-art test error of 2.4% is yielded. Our code is available at https://github.com/nabk89/NAS-with-Proxy-data.

----

## [392] What Changed? Interpretable Model Comparison

**Authors**: *Rahul Nair, Massimiliano Mattetti, Elizabeth Daly, Dennis Wei, Oznur Alkan, Yunfeng Zhang*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/393](https://doi.org/10.24963/ijcai.2021/393)

**Abstract**:

We consider the problem of distinguishing two machine learning (ML) models built for the same task in a human-interpretable way. As models can fail or succeed in different ways, classical accuracy metrics may mask crucial qualitative differences. This problem arises in a few contexts. In business applications with periodically retrained models, an updated model may deviate from its predecessor for some segments without a change in overall accuracy. In automated ML systems, where several ML pipelines are generated, the top pipelines have comparable accuracy but may have more subtle differences. We present a method for interpretable comparison of binary classification models by approximating them with Boolean decision rules. We introduce stabilization conditions that allow for the two rule sets to be more directly comparable. A method is proposed to compare two rule sets based on their statistical and semantic similarity by solving assignment problems and highlighting changes.  An empirical evaluation on several benchmark datasets illustrates the insights that may be obtained and shows that artificially induced changes can be reliably recovered by our method.

----

## [393] TIDOT: A Teacher Imitation Learning Approach for Domain Adaptation with Optimal Transport

**Authors**: *Tuan Nguyen, Trung Le, Nhan Dam, Quan Hung Tran, Truyen Nguyen, Dinh Q. Phung*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/394](https://doi.org/10.24963/ijcai.2021/394)

**Abstract**:

Using the principle of imitation learning and the theory of optimal transport we propose in this paper a novel model for unsupervised domain adaptation named Teacher Imitation Domain Adaptation with Optimal Transport (TIDOT). Our model includes two cooperative agents: a teacher and a student. The former agent is trained to be an expert on labeled data in the source domain, whilst the latter one aims to work with unlabeled data in the target domain. More specifically, optimal transport is applied to quantify the total of the distance between embedded distributions of the source and target data in the joint space, and the distance between predictive distributions of both agents, thus by minimizing this quantity TIDOT could mitigate not only the data shift but also the label shift. Comprehensive empirical studies show that TIDOT outperforms existing state-of-the-art performance on benchmark datasets.

----

## [394] Learning Embeddings from Knowledge Graphs With Numeric Edge Attributes

**Authors**: *Sumit Pai, Luca Costabello*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/395](https://doi.org/10.24963/ijcai.2021/395)

**Abstract**:

Numeric values associated to edges of a knowledge graph have been used to represent uncertainty, edge importance, and even out-of-band knowledge in a growing number of scenarios, ranging from genetic data to social networks. Nevertheless, traditional knowledge graph embedding models are not designed to capture such information, to the detriment of predictive power. 
We propose a novel method that injects numeric edge attributes into the scoring layer of a traditional knowledge graph embedding architecture. Experiments with publicly available numeric-enriched knowledge graphs show that our method outperforms traditional numeric-unaware baselines as well as the recent UKGE model.

----

## [395] Explaining Deep Neural Network Models with Adversarial Gradient Integration

**Authors**: *Deng Pan, Xin Li, Dongxiao Zhu*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/396](https://doi.org/10.24963/ijcai.2021/396)

**Abstract**:

Deep neural networks (DNNs) have became one of the most high performing tools in a broad range
of machine learning areas. However, the multilayer non-linearity of the network architectures prevent
us from gaining a better understanding of the models’ predictions. Gradient based attribution
methods (e.g., Integrated Gradient (IG)) that decipher input features’ contribution to the prediction
task have been shown to be highly effective yet requiring a reference input as the anchor for explaining
model’s output. The performance of DNN model interpretation can be quite inconsistent with
regard to the choice of references. Here we propose an Adversarial Gradient Integration (AGI) method
that integrates the gradients from adversarial examples to the target example along the curve of steepest
ascent to calculate the resulting contributions from all input features. Our method doesn’t rely on
the choice of references, hence can avoid the ambiguity and inconsistency sourced from the reference
selection. We demonstrate the performance of our AGI method and compare with competing methods
in explaining image classification results. Code is available from https://github.com/pd90506/AGI.

----

## [396] Two Birds with One Stone: Series Saliency for Accurate and Interpretable Multivariate Time Series Forecasting

**Authors**: *Qingyi Pan, Wenbo Hu, Ning Chen*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/397](https://doi.org/10.24963/ijcai.2021/397)

**Abstract**:

It is important yet challenging to perform accurate and interpretable time series forecasting. Though deep learning methods can boost forecasting accuracy, they often sacrifice interpretability. In this paper, we present a new scheme of series saliency to boost both accuracy and interpretability. By extracting series images from sliding windows of the time series, we design series saliency as a mixup strategy with a learnable mask between the series images and their perturbed versions. Series saliency is model agnostic and performs as an adaptive data augmentation method for training deep models. Moreover, by slightly changing the objective, we optimize series saliency to find a mask for interpretable forecasting in both feature and time dimensions. Experimental results on several real datasets demonstrate that series saliency is effective to produce accurate time-series forecasting results as well as generate temporal interpretations.

----

## [397] Learning Aggregation Functions

**Authors**: *Giovanni Pellegrini, Alessandro Tibo, Paolo Frasconi, Andrea Passerini, Manfred Jaeger*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/398](https://doi.org/10.24963/ijcai.2021/398)

**Abstract**:

Learning on sets is increasingly gaining attention in the machine learning community, due to its widespread applicability. Typically, representations over sets are computed by using fixed aggregation functions such as sum or maximum. However, recent results showed that universal function representation by sum- (or max-) decomposition requires either highly discontinuous (and thus poorly learnable) mappings, or a latent dimension equal to the maximum number of elements in the set. To mitigate this problem, we introduce LAF (Learning Aggregation Function), a learnable aggregator for sets of arbitrary cardinality. LAF can approximate several extensively used aggregators (such as average, sum, maximum) as well as more complex functions (e.g. variance and skewness). We report experiments on semi-synthetic and real data showing that LAF outperforms state-of-the-art sum- (max-) decomposition architectures such as DeepSets and library-based architectures like Principal Neighborhood Aggregation, and can be effectively combined with attention-based architectures.

----

## [398] Meta-Reinforcement Learning by Tracking Task Non-stationarity

**Authors**: *Riccardo Poiani, Andrea Tirinzoni, Marcello Restelli*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/399](https://doi.org/10.24963/ijcai.2021/399)

**Abstract**:

Many real-world domains are subject to a structured non-stationarity which affects the agent's goals and the environmental dynamics. Meta-reinforcement learning (RL) has been shown successful for training agents that quickly adapt to related tasks. However, most of the existing meta-RL algorithms for non-stationary domains either make strong assumptions on the task generation process or require sampling from it at training time. In this paper, we propose a novel algorithm (TRIO) that optimizes for the future by explicitly tracking the task evolution through time. At training time, TRIO learns a variational module to quickly identify latent parameters from experience samples. This module is learned jointly with an optimal exploration policy that takes task uncertainty into account. At test time, TRIO tracks the evolution of the latent parameters online, hence reducing the uncertainty over future tasks and obtaining fast adaptation through the meta-learned policy. Unlike most existing methods, TRIO does not assume Markovian task-evolution processes, it does not require information about the non-stationarity at training time, and it captures complex changes undergoing in the environment. We evaluate our algorithm on different simulated problems and show it outperforms competitive baselines.

----

## [399] Multi-version Tensor Completion for Time-delayed Spatio-temporal Data

**Authors**: *Cheng Qian, Nikos Kargas, Cao Xiao, Lucas Glass, Nicholas D. Sidiropoulos, Jimeng Sun*

**Conference**: *ijcai 2021*

**URL**: [https://doi.org/10.24963/ijcai.2021/400](https://doi.org/10.24963/ijcai.2021/400)

**Abstract**:

Real-world spatio-temporal data is often incomplete or inaccurate due to various data loading delays. For example, a location-disease-time tensor of case counts can have multiple delayed updates of recent temporal slices for some locations or diseases. Recovering such missing or noisy (under-reported) elements of the input tensor can be viewed as a generalized tensor completion problem. Existing tensor completion methods usually assume that i) missing elements are randomly distributed and ii) noise for each tensor element is i.i.d. zero-mean. Both assumptions can be violated for spatio-temporal tensor data. We often observe multiple versions of the input tensor with different under-reporting noise levels. The amount of noise can be time- or location-dependent as more updates are progressively introduced to the tensor. We model such dynamic data as a multi-version tensor with an extra tensor mode capturing the data updates. We propose a low-rank tensor model to predict the updates over time. We demonstrate that our method can accurately predict the ground-truth values of many real-world tensors. We obtain up to  27.2%  lower root mean-squared-error compared to the best baseline method. Finally, we extend our method to track the tensor data over time, leading to significant computational savings.

----



[Go to the previous page](IJCAI-2021-list01.md)

[Go to the next page](IJCAI-2021-list03.md)

[Go to the catalog section](README.md)