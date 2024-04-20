## [1800] Sparse Variational Student-t Processes

**Authors**: *Jian Xu, Delu Zeng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29549](https://doi.org/10.1609/aaai.v38i14.29549)

**Abstract**:

The theory of Bayesian learning incorporates the use of Student-t Processes to model heavy-tailed distributions and datasets with outliers. However, despite Student-t Processes having a similar computational complexity as Gaussian Processes, there has been limited emphasis on the sparse representation of this model. This is mainly due to the increased difficulty in modeling and computation compared to previous sparse Gaussian Processes. Our motivation is to address the need for a sparse representation framework that reduces computational complexity, allowing Student-t Processes to be more flexible for real-world datasets. To achieve this, we leverage the conditional distribution of Student-t Processes to introduce sparse inducing points. Bayesian methods and variational inference are then utilized to derive a well-defined lower bound, facilitating more efficient optimization of our model through stochastic gradient descent. We propose two methods for computing the variational lower bound, one utilizing Monte Carlo sampling and the other employing Jensen's inequality to compute the KL regularization term in the loss function. We propose adopting these approaches as viable alternatives to Gaussian processes when the data might contain outliers or exhibit heavy-tailed behavior, and we provide specific recommendations for their applicability. We evaluate the two proposed approaches on various synthetic and real-world datasets from UCI and Kaggle, demonstrating their effectiveness compared to baseline methods in terms of computational complexity and accuracy, as well as their robustness to outliers.

----

## [1801] Relative Policy-Transition Optimization for Fast Policy Transfer

**Authors**: *Jiawei Xu, Cheng Zhou, Yizheng Zhang, Baoxiang Wang, Lei Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29550](https://doi.org/10.1609/aaai.v38i14.29550)

**Abstract**:

We consider the problem of policy transfer between two Markov Decision Processes (MDPs). We introduce a lemma based on existing theoretical results in reinforcement learning to measure the relativity gap between two arbitrary MDPs, that is the difference between any two cumulative expected returns defined on different policies and environment dynamics. Based on this lemma, we propose two new algorithms referred to as Relative Policy Optimization (RPO) and Relative Transition Optimization (RTO), which offer fast policy transfer and dynamics modelling, respectively. RPO transfers the policy evaluated in one environment to maximize the return in another, while RTO updates the parameterized dynamics model to reduce the gap between the dynamics of the two environments. Integrating the two algorithms results in the complete Relative Policy-Transition Optimization (RPTO) algorithm, in which the policy interacts with the two environments simultaneously, such that data collections from two environments, policy and transition updates are completed in one closed loop to form a principled learning framework for policy transfer. We demonstrate the effectiveness of RPTO on a set of MuJoCo continuous control tasks by creating policy transfer problems via variant dynamics.

----

## [1802] Union Subgraph Neural Networks

**Authors**: *Jiaxing Xu, Aihu Zhang, Qingtian Bian, Vijay Prakash Dwivedi, Yiping Ke*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29551](https://doi.org/10.1609/aaai.v38i14.29551)

**Abstract**:

Graph Neural Networks (GNNs) are widely used for graph representation learning in many application domains. The expressiveness of vanilla GNNs is upper-bounded by 1-dimensional Weisfeiler-Leman (1-WL) test as they operate on rooted subtrees through iterative message passing. In this paper, we empower GNNs by injecting neighbor-connectivity information extracted from a new type of substructure. We first investigate different kinds of connectivities existing in a local neighborhood and identify a substructure called union subgraph, which is able to capture the complete picture of the 1-hop neighborhood of an edge. We then design a shortest-path-based substructure descriptor that possesses three nice properties and can effectively encode the high-order connectivities in union subgraphs. By infusing the encoded neighbor connectivities, we propose a novel model, namely Union Subgraph Neural Network (UnionSNN), which is proven to be strictly more powerful than 1-WL in distinguishing non-isomorphic graphs. Additionally, the local encoding from union subgraphs can also be injected into arbitrary message-passing neural networks (MPNNs) and Transformer-based models as a plugin. Extensive experiments on 18 benchmarks of both graph-level and node-level tasks demonstrate that UnionSNN outperforms state-of-the-art baseline models, with competitive computational efficiency. The injection of our local encoding to existing models is able to boost the performance by up to 11.09%. Our code is available at https://github.com/AngusMonroe/UnionSNN.

----

## [1803] Enhancing Ensemble Clustering with Adaptive High-Order Topological Weights

**Authors**: *Jiaxuan Xu, Taiyong Li, Lei Duan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29552](https://doi.org/10.1609/aaai.v38i14.29552)

**Abstract**:

Ensemble clustering learns more accurate consensus results from a set of weak base clustering results. This technique is more challenging than other clustering algorithms due to the base clustering result set's randomness and the inaccessibility of data features. Existing ensemble clustering methods rely on the Co-association (CA) matrix quality but lack the capability to handle missing connections in base clustering. Inspired by the neighborhood high-order and topological similarity theories, this paper proposes a topological ensemble model based on high-order information. Specifically, this paper compensates for missing connections by mining neighborhood high-order connection information in the CA matrix and learning optimal connections with adaptive weights. Afterward, the learned excellent connections are embedded into topology learning to capture the topology of the base clustering. Finally, we incorporate adaptive high-order connection representation and topology learning into a unified learning framework. To our knowledge, this is the first ensemble clustering work based on topological similarity and high-order connectivity relations. Extensive experiments on multiple datasets demonstrate the effectiveness of the proposed method. The source code of the proposed approach is available at https://github.com/ltyong/awec.

----

## [1804] PTMQ: Post-training Multi-Bit Quantization of Neural Networks

**Authors**: *Ke Xu, Zhongcheng Li, Shanshan Wang, Xingyi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29553](https://doi.org/10.1609/aaai.v38i14.29553)

**Abstract**:

The ability of model quantization with arbitrary bit-width to dynamically meet diverse bit-width requirements during runtime has attracted significant attention. Recent research has focused on optimizing large-scale training methods to achieve robust bit-width adaptation, which is a time-consuming process requiring hundreds of GPU hours. Furthermore, converting bit-widths requires recalculating statistical parameters of the norm layers, thereby impeding real-time switching of the bit-width. To overcome these challenges, we propose an efficient Post-Training Multi-bit Quantization (PTMQ) scheme that requires only a small amount of calibration data to perform block-wise reconstruction of multi-bit quantization errors. It eliminates the influence of statistical parameters by fusing norm layers, and supports real-time switching bit-widths in uniform quantization and mixed-precision quantization. To improve quantization accuracy and robustness, we propose a Multi-bit Feature Mixer technique (MFM) for fusing features of different bit-widths to enhance robustness across varying bit-widths. Moreover, we introduced the Group-wise Distillation Loss (GD-Loss) to enhance the correlation between different bit-width groups and further improve the overall performance of PTMQ. Extensive experiments demonstrate that PTMQ achieves comparable performance to existing state-of-the-art post-training quantization methods, while optimizing it speeds up by 100$\times$ compared to recent multi-bit quantization works. Code can be available at https://github.com/xuke225/PTMQ.

----

## [1805] LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-identification

**Authors**: *Kunlun Xu, Xu Zou, Jiahuan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29554](https://doi.org/10.1609/aaai.v38i14.29554)

**Abstract**:

Lifelong person re-identification (LReID) aims to train a unified model from diverse data sources step by step. The severe domain gaps between different training steps result in catastrophic forgetting in LReID, and existing methods mainly rely on data replay and knowledge distillation techniques to handle this issue. However, the former solution needs to store historical exemplars which inevitably impedes data privacy. The existing knowledge distillation-based models usually retain all the knowledge of the learned old models without any selections, which will inevitably include erroneous and detrimental knowledge that severely impacts the learning performance of the new model. To address these issues, we propose an exemplar-free LReID method named LongShort Term Knowledge Consolidation (LSTKC) that contains a Rectification-based Short-Term Knowledge Transfer module (R-STKT) and an Estimation-based Long-Term Knowledge Consolidation module (E-LTKC). For each learning iteration within one training step, R-STKT aims to filter and rectify the erroneous knowledge contained in the old model and transfer the rectified knowledge to facilitate the short-term learning of the new model. Meanwhile, once one training step is finished, E-LTKC proposes to further consolidate the learned long-term knowledge via adaptively fusing the parameters of models from different steps. Consequently, experimental results show that our LSTKC exceeds the state-of-the-art methods by 6.3%/9.4% and 7.9%/4.5%, 6.4%/8.0% and 9.0%/5.5% average mAP/R@1 on seen and unseen domains under two different training orders of the challenging LReID benchmark respectively.

----

## [1806] Defying Imbalanced Forgetting in Class Incremental Learning

**Authors**: *Shixiong Xu, Gaofeng Meng, Xing Nie, Bolin Ni, Bin Fan, Shiming Xiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29555](https://doi.org/10.1609/aaai.v38i14.29555)

**Abstract**:

We observe a high level of imbalance in the accuracy of different learned classes in the same old task for the first time. This intriguing phenomenon, discovered in replay-based Class Incremental Learning (CIL), highlights the imbalanced forgetting of learned classes, as their accuracy is similar before the occurrence of catastrophic forgetting. This discovery remains previously unidentified due to the reliance on average incremental accuracy as the measurement for CIL, which assumes that the accuracy of classes within the same task is similar. However, this assumption is invalid in the face of catastrophic forgetting. Further empirical studies indicate that this imbalanced forgetting is caused by conflicts in representation between semantically similar old and new classes. These conflicts are rooted in the data imbalance present in replay-based CIL methods. Building on these insights, we propose CLass-Aware Disentanglement (CLAD) as a means to predict the old classes that are more likely to be forgotten and enhance their accuracy. Importantly, CLAD can be seamlessly integrated into existing CIL methods. Extensive experiments demonstrate that CLAD consistently improves current replay-based methods, resulting in performance gains of up to 2.56%.

----

## [1807] E2E-AT: A Unified Framework for Tackling Uncertainty in Task-Aware End-to-End Learning

**Authors**: *Wangkun Xu, Jianhong Wang, Fei Teng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29556](https://doi.org/10.1609/aaai.v38i14.29556)

**Abstract**:

Successful machine learning involves a complete pipeline of data, model, and downstream applications. Instead of treating them separately, there has been a prominent increase of attention within the constrained optimization (CO) and machine learning (ML) communities towards combining prediction and optimization models. The so-called end-to-end (E2E) learning captures the task-based objective for which they will be used for decision making. Although a large variety of E2E algorithms have been presented, it has not been fully investigated how to systematically address uncertainties involved in such models. Most of the existing work considers the uncertainties of ML in the input space and improves robustness through adversarial training. We extend this idea to E2E learning and prove that there is a robustness certification procedure by solving augmented integer programming. Furthermore, we show that neglecting the uncertainty of COs during training causes a new trigger for generalization errors. To include all these components, we propose a unified framework that covers the uncertainties emerging in both the input feature space of the ML models and the COs. The framework is described as a robust optimization problem and is practically solved via end-to-end adversarial training (E2E-AT). Finally, the performance of E2E-AT is evaluated by a real-world end-to-end power system operation problem, including load forecasting and sequential scheduling tasks.

----

## [1808] LERE: Learning-Based Low-Rank Matrix Recovery with Rank Estimation

**Authors**: *Zhengqin Xu, Yulun Zhang, Chao Ma, Yichao Yan, Zelin Peng, Shoulie Xie, Shiqian Wu, Xiaokang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i14.29557](https://doi.org/10.1609/aaai.v38i14.29557)

**Abstract**:

A fundamental task in the realms of computer vision, Low-Rank Matrix Recovery (LRMR) focuses on the inherent low-rank structure precise recovery from incomplete data and/or corrupted measurements given that the rank is a known prior or accurately estimated. However,  it remains challenging for existing rank estimation methods to accurately estimate the rank of an ill-conditioned matrix. Also, existing LRMR optimization methods are heavily dependent on the chosen parameters, and are therefore difficult to adapt to different situations. Addressing these issues, A novel LEarning-based low-rank matrix recovery with Rank Estimation (LERE) is proposed. More specifically, considering the characteristics of the Gerschgorin disk's center and radius, a new heuristic decision rule in the Gerschgorin Disk Theorem is significantly enhanced and the low-rank boundary can be exactly located, which leads to a marked improvement in the accuracy of rank estimation. According to the estimated rank, we select row and column sub-matrices from the observation matrix by uniformly random sampling. A 17-iteration feedforward-recurrent-mixed neural network is then adapted to learn the parameters in the sub-matrix recovery processing. Finally, by the correlation of the row sub-matrix and column sub-matrix, LERE successfully recovers the underlying low-rank matrix. Overall, LERE is more efficient and robust than existing LRMR methods. Experimental results demonstrate that LERE surpasses state-of-the-art (SOTA) methods. The code for this work is accessible at  https://github.com/zhengqinxu/LERE.

----

## [1809] Multiobjective Lipschitz Bandits under Lexicographic Ordering

**Authors**: *Bo Xue, Ji Cheng, Fei Liu, Yimu Wang, Qingfu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29558](https://doi.org/10.1609/aaai.v38i15.29558)

**Abstract**:

This paper studies the multiobjective bandit problem under lexicographic ordering, wherein the learner aims to simultaneously maximize ? objectives hierarchically. The only existing algorithm for this problem considers the multi-armed bandit model, and its regret bound is O((KT)^(2/3)) under a metric called priority-based regret. However, this bound is suboptimal, as the lower bound for single objective multi-armed bandits is Omega(KlogT). Moreover, this bound becomes vacuous when the arm number K is infinite. To address these limitations, we investigate the multiobjective Lipschitz bandit model, which allows for an infinite arm set. Utilizing a newly designed multi-stage decision-making strategy, we develop an improved algorithm that achieves a general regret bound of O(T^((d_z^i+1)/(d_z^i+2))) for the i-th objective, where d_z^i is the zooming dimension for the i-th objective, with i in {1,2,...,m}. This bound matches the lower bound of the single objective Lipschitz bandit problem in terms of T, indicating that our algorithm is almost optimal. Numerical experiments confirm the effectiveness of our algorithm.

----

## [1810] Residual Hyperbolic Graph Convolution Networks

**Authors**: *Yangkai Xue, Jindou Dai, Zhipeng Lu, Yuwei Wu, Yunde Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29559](https://doi.org/10.1609/aaai.v38i15.29559)

**Abstract**:

Hyperbolic graph convolutional networks (HGCNs) have demonstrated representational capabilities of modeling hierarchical-structured graphs. However, as in general GCNs, over-smoothing may occur as the number of model layers increases, limiting the representation capabilities of most current HGCN models. In this paper, we propose residual hyperbolic graph convolutional networks (R-HGCNs) to address the over-smoothing problem. We introduce a hyperbolic residual connection function to overcome the over-smoothing problem, and also theoretically prove the effectiveness of the hyperbolic residual function. Moreover, we use product manifolds and HyperDrop to facilitate the R-HGCNs. The distinctive features of the R-HGCNs are as follows: (1) The hyperbolic residual connection preserves the initial node information in each layer and adds a hyperbolic identity mapping to prevent node features from being indistinguishable. (2) Product manifolds in R-HGCNs have been set up with different origin points in different components to facilitate the extraction of feature information from a wider range of perspectives, which enhances the representing capability of R-HGCNs. (3) HyperDrop adds multiplicative Gaussian noise into hyperbolic representations, such that perturbations can be added to alleviate the over-fitting problem without deconstructing the hyperbolic geometry.
Experiment results demonstrate the effectiveness of R-HGCNs under various graph convolution layers and different structures of product manifolds.

----

## [1811] GraFITi: Graphs for Forecasting Irregularly Sampled Time Series

**Authors**: *Vijaya Krishna Yalavarthi, Kiran Madhusudhanan, Randolf Scholz, Nourhan Ahmed, Johannes Burchert, Shayan Jawed, Stefan Born, Lars Schmidt-Thieme*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29560](https://doi.org/10.1609/aaai.v38i15.29560)

**Abstract**:

It uses the power of Graph Neural Networks to learn the graph and predict the target edge weights. GraFITi has been tested on 3 real-world and 1 synthetic irregularly sampled time series dataset with missing values and compared with various state-of-the-art models. The experimental results demonstrate that GraFITi improves the forecasting accuracy by up to 17% and reduces the run time up to 5 times compared to the state-of-the-art forecasting models.

----

## [1812] Live and Learn: Continual Action Clustering with Incremental Views

**Authors**: *Xiaoqiang Yan, Yingtao Gan, Yiqiao Mao, Yangdong Ye, Hui Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29561](https://doi.org/10.1609/aaai.v38i15.29561)

**Abstract**:

Multi-view action clustering leverages the complementary information from different camera views to enhance the clustering performance. Although existing approaches have achieved significant progress, they assume all camera views are available in advance, which is impractical when the camera view is incremental over time. Besides, learning the invariant information among multiple camera views is still a challenging issue, especially in continual learning scenario. Aiming at these problems, we propose a novel  continual action clustering (CAC) method, which is capable of learning action categories in a continual learning manner. To be specific, we first devise a category memory library, which captures and stores the learned categories from historical views. Then, as a new camera view arrives, we only need to maintain a consensus partition matrix, which can be updated by leveraging the incoming new camera view rather than keeping all of them. Finally, a three-step alternate optimization is proposed, in which the category memory library and consensus partition matrix are  optimized. The empirical experimental results on 6 realistic multi-view action collections demonstrate the excellent clustering performance and time/space efficiency of the CAC compared with 15 state-of-the-art baselines.

----

## [1813] Federated Partial Label Learning with Local-Adaptive Augmentation and Regularization

**Authors**: *Yan Yan, Yuhong Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29562](https://doi.org/10.1609/aaai.v38i15.29562)

**Abstract**:

Partial label learning (PLL) expands the applicability of supervised machine learning models by enabling effective learning from weakly annotated overcomplete labels. Existing PLL methods however focus on the standard centralized learning scenarios. In this paper, we expand PLL into the distributed computation setting by formalizing a new learning scenario named as federated partial label learning (FedPLL), where the training data with partial labels are distributed across multiple local clients with privacy constraints. To address this challenging problem, we propose a novel Federated PLL method with Local-Adaptive Augmentation and Regularization (FedPLL-LAAR). In addition to alleviating the partial label noise with moving-average label disambiguation, the proposed method performs MixUp-based local-adaptive data augmentation to mitigate the challenge posed by insufficient and imprecisely annotated local data, and dynamically incorporates the guidance of global model to minimize client drift through adaptive gradient alignment regularization between the global and local models. Extensive experiments conducted on multiple datasets under the FedPLL setting demonstrate the effectiveness of the proposed FedPLL-LAAR method for federated partial label learning.

----

## [1814] An Optimal Transport View for Subspace Clustering and Spectral Clustering

**Authors**: *Yuguang Yan, Zhihao Xu, Canlin Yang, Jie Zhang, Ruichu Cai, Michael Kwok-Po Ng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29563](https://doi.org/10.1609/aaai.v38i15.29563)

**Abstract**:

Clustering is one of the most fundamental problems in machine learning and data mining, and many algorithms have been proposed in the past decades. Among them, subspace clustering and spectral clustering are the most famous approaches. In this paper, we provide an explanation for subspace clustering and spectral clustering from the perspective of optimal transport. Optimal transport studies how to move samples from one distribution to another distribution with minimal transport cost, and has shown a powerful ability to extract geometric information. By considering a self optimal transport model with only one group of samples, we observe that both subspace clustering and spectral clustering can be explained in the framework of optimal transport, and the optimal transport matrix bridges the spaces of features and spectral embeddings. Inspired by this connection, we propose a spectral optimal transport barycenter model, which learns spectral embeddings by solving a barycenter problem equipped with an optimal transport discrepancy and guidance of data. Based on our proposed model, we take advantage of optimal transport to exploit both feature and metric information involved in data for learning coupled spectral embeddings and affinity matrix in a unified model. We develop an alternating optimization algorithm to solve the resultant problems, and conduct experiments in different settings to evaluate the performance of our proposed methods.

----

## [1815] Exploiting Geometry for Treatment Effect Estimation via Optimal Transport

**Authors**: *Yuguang Yan, Zeqin Yang, Weilin Chen, Ruichu Cai, Zhifeng Hao, Michael Kwok-Po Ng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29564](https://doi.org/10.1609/aaai.v38i15.29564)

**Abstract**:

Estimating treatment effects from observational data suffers from the issue of confounding bias, which is induced by the imbalanced confounder distributions between the treated and control groups. As an effective approach, re-weighting learns a group of sample weights to balance the confounder distributions. Existing methods of re-weighting highly rely on a propensity score model or moment alignment. However, for complex real-world data, it is difficult to obtain an accurate propensity score prediction. Although moment alignment is free of learning a propensity score model, accurate estimation for high-order moments is computationally difficult and still remains an open challenge, and first and second-order moments are insufficient to align the distributions and easy to be misled by outliers. In this paper, we exploit geometry to capture the intrinsic structure involved in data for balancing the confounder distributions, so that confounding bias can be reduced even with outliers. To achieve this, we construct a connection between treatment effect estimation and optimal transport, a powerful tool to capture geometric information. After that, we propose an optimal transport model to learn sample weights by extracting geometry from confounders, in which geometric information between groups and within groups is leveraged for better confounder balancing. A projected mirror descent algorithm is employed to solve the derived optimization problem. Experimental studies on both synthetic and real-world datasets demonstrate the effectiveness of our proposed method.

----

## [1816] Wasserstein Differential Privacy

**Authors**: *Chengyi Yang, Jiayin Qi, Aimin Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29565](https://doi.org/10.1609/aaai.v38i15.29565)

**Abstract**:

Differential privacy (DP) has achieved remarkable results in the field of privacy-preserving machine learning. However, existing DP frameworks do not satisfy all the conditions for becoming metrics, which prevents them from deriving better basic private properties and leads to exaggerated values on privacy budgets. We propose Wasserstein differential privacy (WDP), an alternative DP framework to measure the risk of privacy leakage, which satisfies the properties of symmetry and triangle inequality. We show and prove that WDP has 13 excellent properties, which can be theoretical supports for the better performance of WDP than other DP frameworks. 
In addition, we derive a general privacy accounting method called Wasserstein accountant, which enables WDP to be applied in stochastic gradient descent (SGD) scenarios containing subsampling. Experiments on basic mechanisms, compositions and deep learning show that the privacy budgets obtained by Wasserstein accountant are relatively stable and less influenced by order. Moreover, the overestimation on privacy budgets can be effectively alleviated. The code is available at https://github.com/Hifipsysta/WDP.

----

## [1817] Federated Causality Learning with Explainable Adaptive Optimization

**Authors**: *Dezhi Yang, Xintong He, Jun Wang, Guoxian Yu, Carlotta Domeniconi, Jinglin Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29566](https://doi.org/10.1609/aaai.v38i15.29566)

**Abstract**:

Discovering the causality from observational data is a crucial task in various scientific domains. With increasing awareness of privacy, data are not allowed to be exposed, and it is very hard to learn causal graphs from dispersed data, since these data may have different distributions. In this paper, we propose a federated causal discovery strategy (FedCausal) to learn the unified global causal graph from decentralized heterogeneous data. We design a global optimization formula to naturally aggregate the causal graphs from client data and constrain the acyclicity of the global graph without exposing local data. Unlike other federated causal learning algorithms, FedCausal unifies the local and global optimizations into a complete directed acyclic graph (DAG) learning process with a flexible optimization objective. We prove that this optimization objective has a high interpretability and can adaptively handle homogeneous and heterogeneous data. Experimental results on synthetic and real datasets show that FedCausal can effectively deal with non-independently and identically distributed (non-iid) data and has a superior performance.

----

## [1818] Multi-Modal Disordered Representation Learning Network for Description-Based Person Search

**Authors**: *Fan Yang, Wei Li, Menglong Yang, Binbin Liang, Jianwei Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29567](https://doi.org/10.1609/aaai.v38i15.29567)

**Abstract**:

Description-based person search aims to retrieve images of the target identity via textual descriptions. One of the challenges for this task is to extract discriminative representation from images and descriptions. Most existing methods apply the part-based split method or external models to explore the fine-grained details of local features, which ignore the global relationship between partial information and cause network instability. To overcome these issues, we propose a Multi-modal Disordered Representation Learning Network (MDRL) for description-based person search to fully extract the visual and textual representations. Specifically, we design a Cross-modality Global Feature Learning Architecture to learn the global features from the two modalities and meet the demand of the task. Based on our global network, we introduce a Disorder Local Learning Module to explore local features by a disordered reorganization strategy from both visual and textual aspects and enhance the robustness of the whole network. Besides, we introduce a Cross-modality Interaction Module to guide the two streams to extract visual or textual representations considering the correlation between modalities. Extensive experiments are conducted on two public datasets, and the results show that our method outperforms the state-of-the-art methods on CUHK-PEDES and ICFG-PEDES datasets and achieves superior performance.

----

## [1819] Exploring One-Shot Semi-supervised Federated Learning with Pre-trained Diffusion Models

**Authors**: *Mingzhao Yang, Shangchao Su, Bin Li, Xiangyang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29568](https://doi.org/10.1609/aaai.v38i15.29568)

**Abstract**:

Recently, semi-supervised federated learning (semi-FL) has been proposed to handle the commonly seen real-world scenarios with labeled data on the server and unlabeled data on the clients. However, existing methods face several challenges such as communication costs, data heterogeneity, and training pressure on client devices. To address these challenges, we introduce the powerful diffusion models (DM) into semi-FL and propose FedDISC, a Federated Diffusion-Inspired Semi-supervised Co-training method. Specifically, we first extract prototypes of the labeled server data and use these prototypes to predict pseudo-labels of the client data. For each category, we compute the cluster centroids and domain-specific representations to signify the semantic and stylistic information of their distributions. After adding noise, these representations are sent back to the server, which uses the pre-trained DM to generate synthetic datasets complying with the client distributions and train a global model on it. With the assistance of vast knowledge within DM, the synthetic datasets have comparable quality and diversity to the client images, subsequently enabling the training of global models that achieve performance equivalent to or even surpassing the ceiling of supervised centralized training. FedDISC works within one communication round, does not require any local training, and involves very minimal information uploading, greatly enhancing its practicality. Extensive experiments on three large-scale datasets demonstrate that FedDISC effectively addresses the semi-FL problem on non-IID clients and outperforms the compared SOTA methods. Sufficient visualization experiments also illustrate that the synthetic dataset generated by FedDISC exhibits comparable diversity and quality to the original client dataset, with a neglectable possibility of leaking privacy-sensitive information of the clients.

----

## [1820] Exploring Sparse Visual Prompt for Domain Adaptive Dense Prediction

**Authors**: *Senqiao Yang, Jiarui Wu, Jiaming Liu, Xiaoqi Li, Qizhe Zhang, Mingjie Pan, Yulu Gan, Zehui Chen, Shanghang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29569](https://doi.org/10.1609/aaai.v38i15.29569)

**Abstract**:

The visual prompts have provided an efficient manner in addressing visual cross-domain problems. Previous works introduce domain prompts to tackle the classification Test-Time Adaptation (TTA) problem by placing image-level prompts on the input and fine-tuning prompts for each target domain. However, since the image-level prompts mask out continuous spatial details in the prompt-allocated region, it will suffer from inaccurate contextual information and limited domain knowledge extraction, particularly when dealing with dense prediction TTA problems. To overcome these challenges, we propose a novel Sparse Visual Domain Prompts (SVDP) approach, which applies minimal trainable parameters (e.g., 0.1%) to pixels across the entire image and reserves more spatial information of the input. To better apply SVDP in extracting domain-specific knowledge, we introduce the Domain Prompt Placement (DPP) method to adaptively allocates trainable parameters of SVDP on the pixels with large distribution shifts. Furthermore, recognizing that each target domain sample exhibits a unique domain shift, we design Domain Prompt Updating (DPU) strategy to optimize prompt parameters differently for each sample, facilitating efficient adaptation to the target domain. Extensive experiments were conducted on widely-used TTA and continual TTA benchmarks, and our proposed method achieves state-of-the-art performance in both semantic segmentation and depth estimation tasks.

----

## [1821] A Variational Autoencoder for Neural Temporal Point Processes with Dynamic Latent Graphs

**Authors**: *Sikun Yang, Hongyuan Zha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29570](https://doi.org/10.1609/aaai.v38i15.29570)

**Abstract**:

Continuously observed event occurrences, often exhibit self and mutually exciting effects, which can be well modeled using temporal point processes. Beyond that, these event dynamics may also change over time, with certain periodic trends. We propose a novel variational autoencoder to capture such a mixture of temporal dynamics. More specifically, the whole time interval of the input sequence is partitioned into a set of sub intervals. The event dynamics are assumed to be stationary within each subinterval, but could be changing across those subintervals. In particular, we use a sequential latent variable model to learn a dependency graph between the observed dimensions, for each subinterval. The model predicts the future event times, by using the learned dependency graph to remove the non contributing influences of past events. By doing so, the proposed model demonstrates its higher accuracy in predicting inter event times and event types for several real world event sequences, compared with existing state of the art neural point processes.

----

## [1822] A Transfer Approach Using Graph Neural Networks in Deep Reinforcement Learning

**Authors**: *Tianpei Yang, Heng You, Jianye Hao, Yan Zheng, Matthew E. Taylor*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29571](https://doi.org/10.1609/aaai.v38i15.29571)

**Abstract**:

Transfer learning (TL) has shown great potential to improve Reinforcement Learning (RL) efficiency by leveraging prior knowledge in new tasks. However, much of the existing TL research focuses on transferring knowledge between tasks that share the same state-action spaces.  Further, transfer from multiple source tasks that have different state-action spaces is more challenging and needs to be solved urgently to improve the generalization and practicality of the method in real-world scenarios. This paper proposes TURRET (Transfer Using gRaph neuRal nETworks), to utilize the generalization capabilities of Graph Neural Networks (GNNs) to facilitate efficient and effective multi-source policy transfer learning in the state-action mismatch setting.  TURRET learns a semantic representation by accounting for the intrinsic property of the agent through GNNs, which leads to a unified state embedding space for all tasks. As a result, TURRET achieves more efficient transfer with strong generalization ability between different tasks and can be easily combined with existing Deep RL algorithms. Experimental results show that TURRET significantly outperforms other TL methods on multiple continuous action control tasks, successfully transferring across robots with different state-action spaces.

----

## [1823] Safe Abductive Learning in the Presence of Inaccurate Rules

**Authors**: *Xiaowen Yang, Jie-Jing Shao, Wei-Wei Tu, Yu-Feng Li, Wang-Zhou Dai, Zhi-Hua Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29572](https://doi.org/10.1609/aaai.v38i15.29572)

**Abstract**:

Integrating complementary strengths of raw data and logical rules to improve the learning generalization has been recently shown promising and effective, e.g., abductive learning is one generic framework that can learn the perception model from data and reason between rules simultaneously. However, the performance would be seriously decreased when inaccurate logical rules appear, which may be even worse than baselines using only raw data.
Efforts on this issue are highly desired while remain to be limited.  This paper proposes a simple and effective safe abductive learning method to alleviate the harm caused by inaccurate rules. Unlike the existing methods which directly use all rules without correctness checks, it utilizes them selectively by constructing a graphical model with an adaptive reasoning process to prevent performance hazards. Theoretically, we show that induction and abduction are mutually beneficial, and can be rigorously justified from a classical maximum likelihood estimation perspective.  Experiments on diverse tasks show that our method can tolerate at least twice as many inaccurate rules as accurate ones and achieve highly competitive performance while other methods can't.  Moreover, the proposal can refine inaccurate rules and works well in extended weakly supervised scenarios.

----

## [1824] Leveraging Normalization Layer in Adapters with Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning

**Authors**: *Yongjin Yang, Taehyeon Kim, Se-Young Yun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29573](https://doi.org/10.1609/aaai.v38i15.29573)

**Abstract**:

Cross-domain few-shot learning presents a formidable challenge, as models must be trained on base classes and then tested on novel classes from various domains with only a few samples at hand. While prior approaches have primarily focused on parameter-efficient methods of using adapters, they often overlook two critical issues: shifts in batch statistics and noisy sample statistics arising from domain discrepancy variations. In this paper, we introduce Leveraging Normalization Layer in Adapters with Progressive Learning and Adaptive Distillation (ProLAD), marking two principal contributions. First, our methodology utilizes two separate adapters: one devoid of a normalization layer, which is more effective for similar domains, and another embedded with a normalization layer, designed to leverage the batch statistics of the target domain, thus proving effective for dissimilar domains. Second, to address the pitfalls of noisy statistics, we deploy two strategies: a progressive training of the two adapters and an adaptive distillation technique derived from features determined by the model solely with the adapter devoid of a normalization layer. Through this adaptive distillation, our approach functions as a modulator, controlling the primary adapter for adaptation, based on each domain. Evaluations on standard cross-domain few-shot learning benchmarks confirm that our technique outperforms existing state-of-the-art methodologies.

----

## [1825] Adversarial Purification with the Manifold Hypothesis

**Authors**: *Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard I. Hartley, Peter H. Tu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29574](https://doi.org/10.1609/aaai.v38i15.29574)

**Abstract**:

In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.

----

## [1826] Dynamic Knowledge Injection for AIXI Agents

**Authors**: *Samuel Yang-Zhao, Kee Siong Ng, Marcus Hutter*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29575](https://doi.org/10.1609/aaai.v38i15.29575)

**Abstract**:

Prior approximations of AIXI, a Bayesian optimality notion for general reinforcement learning, can only approximate AIXI's Bayesian environment model using an a-priori defined set of models. This is a fundamental source of epistemic uncertainty for the agent in settings where the existence of systematic bias in the predefined model class cannot be resolved by simply collecting more data from the environment. We address this issue in the context of Human-AI teaming by considering a setup where additional knowledge for the agent in the form of new candidate models arrives from a human operator in an online fashion. We introduce a new agent called DynamicHedgeAIXI that maintains an exact Bayesian mixture over dynamically changing sets of models via a time-adaptive prior constructed from a variant of the Hedge algorithm. The DynamicHedgeAIXI agent is the richest direct approximation of AIXI known to date and comes with good performance guarantees. Experimental results on epidemic control on contact networks validates the agent's practical utility.

----

## [1827] PerFedRLNAS: One-for-All Personalized Federated Neural Architecture Search

**Authors**: *Dixi Yao, Baochun Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29576](https://doi.org/10.1609/aaai.v38i15.29576)

**Abstract**:

Personalized federated learning is a new paradigm to address heterogeneous problems (e.g. issues with non-i.i.d. data) in federated learning. However, existing personalized federated learning methods lack standards for how personalized and shared parts of the models are designed. Sometimes, manual design can even lead to worse performance than non-personalization. As a result, we propose a new algorithm for personalized federated neural architecture search, called PerFedRLNAS, to automatically personalize the architectures and weights of models on each client. With such an algorithm, we can solve the issues of low efficiency as well as failure to adapt to new search spaces in previous federated neural architecture search work. We further show that with automatically assigning different client architectures can solve heterogeneity of data distribution, efficiency and memory in federated learning.  In our experiments, we empirically show that our framework shows much better performance with respect to personalized accuracy and overall time compared to state-of-the-art methods. Furthermore, PerFedRLNAS has a good generalization ability to new clients, and is easy to be deployed in practice.

----

## [1828] VQ-FONT: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization

**Authors**: *Mingshuai Yao, Yabo Zhang, Xianhui Lin, Xiaoming Li, Wangmeng Zuo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29577](https://doi.org/10.1609/aaai.v38i15.29577)

**Abstract**:

Few-shot font generation is challenging, as it needs to capture the fine-grained stroke styles from a limited set of reference glyphs, and then transfer to other characters, which are expected to have similar styles. However, due to the diversity and complexity of Chinese font styles, the synthesized glyphs of existing methods usually exhibit visible artifacts, such as missing details and distorted strokes. In this paper, we propose a VQGAN-based framework (i.e., VQ-Font) to enhance glyph fidelity through token prior refinement and structure-aware enhancement. Specifically, we pre-train a VQGAN to encapsulate font token prior within a code-book. Subsequently, VQ-Font refines the synthesized glyphs with the codebook to eliminate the domain gap between synthesized and real-world strokes. Furthermore, our VQ-Font leverages the inherent design of Chinese characters, where structure components such as radicals and character components are combined in specific arrangements, to recalibrate fine-grained styles based on references. This process improves the matching and fusion of styles at the structure level. Both modules collaborate to enhance the fidelity of the generated fonts. Experiments on a collected font dataset show that our VQ-Font outperforms the competing methods both quantitatively and qualitatively, especially in generating challenging styles. Our code is available at https://github.com/Yaomingshuai/VQ-Font.

----

## [1829] DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency

**Authors**: *Wenfang Yao, Kejing Yin, William K. Cheung, Jia Liu, Jing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29578](https://doi.org/10.1609/aaai.v38i15.29578)

**Abstract**:

The combination of electronic health records (EHR) and medical images is crucial for clinicians in making diagnoses and forecasting prognoses. Strategically fusing these two data modalities has great potential to improve the accuracy of machine learning models in clinical prediction tasks. However, the asynchronous and complementary nature of EHR and medical images presents unique challenges. Missing modalities due to clinical and administrative factors are inevitable in practice, and the significance of each data modality varies depending on the patient and the prediction target, resulting in inconsistent predictions and suboptimal model performance. To address these challenges, we propose DrFuse to achieve effective clinical multi-modal fusion. It tackles the missing modality issue by disentangling the features shared across modalities and those unique within each modality. Furthermore, we address the modal inconsistency issue via a disease-wise attention layer that produces the patient- and disease-wise weighting for each modality to make the final prediction. We validate the proposed method using real-world large-scale datasets, MIMIC-IV and MIMIC-CXR. Experimental results show that the proposed method significantly outperforms the state-of-the-art models.

----

## [1830] Progressively Knowledge Distillation via Re-parameterizing Diffusion Reverse Process

**Authors**: *Xufeng Yao, Fanbin Lu, Yuechen Zhang, Xinyun Zhang, Wenqian Zhao, Bei Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29579](https://doi.org/10.1609/aaai.v38i15.29579)

**Abstract**:

Knowledge distillation aims at transferring knowledge from the teacher model to the student one by aligning their distributions. 
Feature-level distillation often uses L2 distance or its variants as the loss function, based on the assumption that outputs follow normal distributions. 
This poses a significant challenge when distribution gaps are substantial since this loss function ignores the variance term. 
To address the problem, we propose to decompose the transfer objective into small parts and optimize it progressively. 
This process is inspired by diffusion models from which the noise distribution is mapped to the target distribution step by step.
However, directly employing diffusion models is impractical in the distillation scenario due to its heavy reverse process.
To overcome this challenge, we adopt the structural re-parameterization technique to generate multiple student features to approximate the teacher features sequentially. 
The multiple student features are combined linearly in inference time without extra cost.
We present extensive experiments performed on various transfer scenarios, such as CNN-to-CNN and Transformer-to-CNN, that validate the effectiveness of our approach.

----

## [1831] Data-Augmented Curriculum Graph Neural Architecture Search under Distribution Shifts

**Authors**: *Yang Yao, Xin Wang, Yijian Qin, Ziwei Zhang, Wenwu Zhu, Hong Mei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29580](https://doi.org/10.1609/aaai.v38i15.29580)

**Abstract**:

Graph neural architecture search (NAS) has achieved great success in designing architectures for graph data processing.However, distribution shifts pose great challenges for graph NAS, since the optimal searched architectures for the training graph data may fail to generalize to the unseen test graph data. The sole prior work tackles this problem by customizing architectures for each graph instance through learning graph structural information, but failed to consider data augmentation during training, which has been proven by existing works to be able to improve generalization.In this paper, we propose Data-augmented Curriculum Graph Neural Architecture Search (DCGAS), which learns an architecture customizer with good generalizability to data under distribution shifts. Specifically, we design an embedding-guided data generator, which can generate sufficient graphs for training to help the model better capture graph structural information. In addition, we design a two-factor uncertainty-based curriculum weighting strategy, which can evaluate the importance of data in enabling the model to learn key information in real-world distribution and reweight them during training. Experimental results on synthetic datasets and real datasets with distribution shifts demonstrate that our proposed method learns generalizable mappings and outperforms existing methods.

----

## [1832] Task-Free Dynamic Sparse Vision Transformer for Continual Learning

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29581](https://doi.org/10.1609/aaai.v38i15.29581)

**Abstract**:

Vision Transformers (ViTs) represent self-attention-based network backbones shown to be efficient in many individual tasks, but which have not been explored in Task-Free Continual Learning (TFCL) so far. Most existing ViT-based approaches for Continual Learning (CL) are relying on task information. In this study, we explore the advantages of the ViT in a more challenging CL scenario where the task boundaries are unavailable during training. To address this learning paradigm, we propose the Task-Free Dynamic Sparse Vision Transformer (TFDSViT), which can dynamically build new sparse experts, where each expert leverages sparsity to allocate the model's capacity for capturing different information categories over time. To avoid forgetting and ensure efficiency in reusing the previously learned knowledge in subsequent learning, we propose a new dynamic dual attention mechanism consisting of the Sparse Attention (SA') and Knowledge Transfer Attention (KTA) modules. The SA' refrains from updating some previously learned attention blocks for preserving prior knowledge. The KTA uses and regulates the information flow of all previously learned experts for learning new patterns. The proposed dual attention mechanism can simultaneously relieve forgetting and promote knowledge transfer for a dynamic expansion model in a task-free manner. We also propose an energy-based dynamic expansion mechanism using the energy as a measure of novelty for the incoming samples which provides appropriate expansion signals leading to a compact network architecture for TFDSViT. Extensive empirical studies demonstrate the effectiveness of TFDSViT. The code and supplementary material (SM) are available at https://github.com/dtuzi123/TFDSViT.

----

## [1833] Task-Free Continual Generation and Representation Learning via Dynamic Expansionable Memory Cluster

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29582](https://doi.org/10.1609/aaai.v38i15.29582)

**Abstract**:

Human brains can continually acquire and learn new skills and knowledge over time from a dynamically changing environment without forgetting previously learnt information. Such a capacity can selectively transfer some important and recently seen information to the persistent knowledge regions of the brain. Inspired by this intuition, we propose a new memory-based approach for image reconstruction and generation in continual learning, consisting of a temporary and evolving memory, with two different storage strategies, corresponding to the temporary and permanent memorisation. The temporary memory aims to preserve up-to-date information while the evolving memory can dynamically increase its capacity in order to preserve permanent knowledge information. This is achieved by the proposed memory expansion mechanism that selectively transfers those data samples deemed as important from the temporary memory to new clusters defined within the evolved memory according to an information novelty criterion. Such a mechanism promotes the knowledge diversity among clusters in the evolved memory, resulting in capturing more diverse information by using a compact memory capacity. Furthermore, we propose a two-step optimization strategy for training a Variational Autoencoder (VAE) to implement generation and representation learning tasks, which updates the generator and inference models separately using two optimisation paths. This approach leads to a better trade-off between generation and reconstruction performance. We show empirically and theoretically that the proposed approach can learn meaningful latent representations while generating diverse images from different domains. The source code and supplementary material (SM) are available at https://github.com/dtuzi123/DEMC.

----

## [1834] Uncertainty Regularized Evidential Regression

**Authors**: *Kai Ye, Tiejin Chen, Hua Wei, Liang Zhan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29583](https://doi.org/10.1609/aaai.v38i15.29583)

**Abstract**:

The Evidential Regression Network (ERN) represents a novel approach that integrates deep learning with Dempster-Shafer's theory to predict a target and quantify the associated uncertainty. Guided by the underlying theory, specific activation functions must be employed to enforce non-negative values, which is a constraint that compromises model performance by limiting its ability to learn from all samples. This paper provides a theoretical analysis of this limitation and introduces an improvement to overcome it. Initially, we define the region where the models can't effectively learn from the samples. Following this, we thoroughly analyze the ERN and investigate this constraint. Leveraging the insights from our analysis, we address the limitation by introducing a novel regularization term that empowers the ERN to learn from the whole training set. Our extensive experiments substantiate our theoretical findings and demonstrate the effectiveness of the proposed solution.

----

## [1835] Near-Optimal Resilient Aggregation Rules for Distributed Learning Using 1-Center and 1-Mean Clustering with Outliers

**Authors**: *Yuhao Yi, Ronghui You, Hong Liu, Changxin Liu, Yuan Wang, Jiancheng Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29584](https://doi.org/10.1609/aaai.v38i15.29584)

**Abstract**:

Byzantine machine learning has garnered considerable attention in light of the unpredictable faults that can occur in large-scale distributed learning systems. The key to secure resilience against Byzantine machines in distributed learning is resilient aggregation mechanisms. Although abundant resilient aggregation rules have been proposed, they are designed in ad-hoc manners, imposing extra barriers on comparing, analyzing, and improving the rules across performance criteria. This paper studies near-optimal aggregation rules using clustering in the presence of outliers. Our outlier-robust clustering approach utilizes geometric properties of the update vectors provided by workers. Our analysis show that constant approximations to the 1-center and 1-mean clustering problems with outliers provide near-optimal resilient aggregators for metric-based criteria, which have been proven to be crucial in the homogeneous and heterogeneous cases respectively. In addition, we discuss two contradicting types of attacks under which no single aggregation rule is guaranteed to improve upon the naive average. Based on the discussion, we propose a two-phase resilient aggregation framework. We run experiments for image classification using a non-convex loss function. The proposed algorithms outperform previously known aggregation rules by a large margin with both homogeneous and heterogeneous data distributions among non-faulty workers. Code and appendix are available at https://github.com/jerry907/AAAI24-RASHB.

----

## [1836] Discriminatively Fuzzy Multi-View K-means Clustering with Local Structure Preserving

**Authors**: *Jun Yin, Shiliang Sun, Lai Wei, Pei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29585](https://doi.org/10.1609/aaai.v38i15.29585)

**Abstract**:

Multi-view K-means clustering successfully generalizes K-means from single-view to multi-view, and obtains excellent clustering performance. In every view, it makes each data point close to the center of the corresponding cluster. However, multi-view K-means only considers the compactness of each cluster, but ignores the separability of different clusters, which is of great importance to producing a good clustering result. In this paper, we propose Discriminatively Fuzzy Multi-view K-means clustering with Local Structure Preserving (DFMKLS). On the basis of minimizing the distance between each data point and the center of the corresponding cluster, DFMKLS separates clusters by maximizing the distance between the centers of pairwise clusters. DFMKLS also relaxes its objective by introducing the idea of fuzzy clustering, which calculates the probability that a data point belongs to each cluster. Considering multi-view K-means mainly focuses on the global information of the data, to efficiently use the local information, we integrate the local structure preserving into the framework of DFMKLS. The effectiveness of DFMKLS is evaluated on benchmark multi-view datasets. It obtains superior performances than state-of-the-art multi-view clustering methods, including multi-view K-means.

----

## [1837] Effective Causal Discovery under Identifiable Heteroscedastic Noise Model

**Authors**: *Naiyu Yin, Tian Gao, Yue Yu, Qiang Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29586](https://doi.org/10.1609/aaai.v38i15.29586)

**Abstract**:

Capturing the underlying structural causal relations represented by Directed Acyclic Graphs (DAGs) has been a fundamental task in various AI disciplines. Causal DAG learning via the continuous optimization framework has recently achieved promising performance in terms of accuracy and efficiency. However, most methods make strong assumptions of homoscedastic noise, i.e., exogenous noises have equal variances across variables, observations, or even both. The noises in real data usually violate both assumptions due to the biases introduced by different data collection processes. To address the heteroscedastic noise issue, we introduce relaxed implementable sufficient conditions and prove the identifiability of a general class of SEM subject to those conditions. Based on the identifiable general SEM, we propose a novel formulation for DAG learning which accounts for the noise variance variation across variables and observations. We then propose an effective two-phase iterative DAG learning algorithm to address the increasing optimization difficulties and learn a causal DAG from data with heteroscedastic variables noise under varying variance. We show significant empirical gains of the proposed approaches over state-of-the-art methods on both synthetic data and real data.

----

## [1838] Dynamic Spiking Graph Neural Networks

**Authors**: *Nan Yin, Mengzhu Wang, Zhenghan Chen, Giulia De Masi, Huan Xiong, Bin Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29587](https://doi.org/10.1609/aaai.v38i15.29587)

**Abstract**:

The integration of Spiking Neural Networks (SNNs) and Graph Neural Networks (GNNs) is gradually attracting attention due to the low power consumption and high efficiency in processing the non-Euclidean data represented by graphs. 
However, as a common problem, dynamic graph representation learning faces challenges such as high complexity and large memory overheads. Current work often uses SNNs instead of Recurrent Neural Networks (RNNs) by using binary features instead of continuous ones for efficient training, which overlooks graph structure information and leads to the loss of details during propagation.  
Additionally, optimizing dynamic spiking models typically requires the propagation of information across time steps, which increases memory requirements. To address these challenges, we present a framework named Dynamic Spiking Graph Neural Networks (Dy-SIGN). To mitigate the information loss problem, Dy-SIGN propagates early-layer information directly to the last layer for information compensation. To accommodate the memory requirements, we apply the implicit differentiation on the equilibrium state, which does not rely on the exact reverse of the forward computation. While traditional implicit differentiation methods are usually used for static situations, Dy-SIGN extends it to the dynamic graph setting. Extensive experiments on three large-scale real-world dynamic graph datasets validate the effectiveness of Dy-SIGN on dynamic node classification tasks with lower computational costs.

----

## [1839] Asymmetric Mutual Alignment for Unsupervised Zero-Shot Sketch-Based Image Retrieval

**Authors**: *Zhihui Yin, Jiexi Yan, Chenghao Xu, Cheng Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29588](https://doi.org/10.1609/aaai.v38i15.29588)

**Abstract**:

In recent years, many methods have been proposed to address the zero-shot sketch-based image retrieval (ZS-SBIR) task, which is a practical problem in many applications. However, in real-world scenarios, on the one hand, we can not obtain training data with the same distribution as the test data, and on the other hand, the labels of training data are not available as usual. To tackle this issue, we focus on a new problem, namely unsupervised zero-shot sketch-based image retrieval (UZS-SBIR), where the available training data does not have labels while the training and testing categories are not overlapping. In this paper, we introduce a new asymmetric mutual alignment method (AMA) including a self-distillation module and a cross-modality mutual alignment module. First, we conduct self-distillation to extract the feature embeddings from unlabeled data. Due to the lack of available information in an unsupervised manner, we employ the cross-modality mutual alignment module to further excavate underlying intra-modality and inter-modality relationships from unlabeled data, and take full advantage of these correlations to align the feature embeddings in image and sketch domains. Meanwhile, the feature representations are enhanced by the intra-modality clustering relations, leading to better generalization ability to unseen classes. Moreover, we conduct an asymmetric strategy to update the teacher and student networks, respectively. Extensive experimental results on several benchmark datasets demonstrate the superiority of our method.

----

## [1840] Risk-Conditioned Reinforcement Learning: A Generalized Approach for Adapting to Varying Risk Measures

**Authors**: *Gwangpyo Yoo, Jinwoo Park, Honguk Woo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29589](https://doi.org/10.1609/aaai.v38i15.29589)

**Abstract**:

In application domains requiring mission-critical decision making, such as finance and robotics, the optimal policy derived by reinforcement learning (RL) often hinges on a preference for risk management. Yet, the dynamic nature of risk measures poses considerable challenges to achieving generalization and adaptation of risk-sensitive policies in the context of RL. In this paper, we propose a risk-conditioned RL model that enables rapid policy adaptation to varying risk measures via a unified risk representation, the Weighted Value-at-Risk (WV@R). To sample risk measures that avoid undue optimism, we construct a risk proposal network employing a conditional adversarial auto-encoder and a normalizing flow. This network establishes coherent representations for risk measures, preserving the continuity in terms of the Wasserstein distance on the risk measures. The normalizing flow is used to support non-crossing quantile regression that obtains valid samples for risk measures, and it is also applied to the agent’s critic to ascertain the preservation of monotonicity in quantile estimations. Through experiments with locomotion, finance, and self-driving scenarios, we show that our model is capable of adapting to a range of risk measures, achieving comparable performance to the baseline models individually trained for each measure. Our model often outperforms the baselines, especially in the cases when exploration is required during training but risk-aversion is favored during evaluation.

----

## [1841] Online Boosting Adaptive Learning under Concept Drift for Multistream Classification

**Authors**: *En Yu, Jie Lu, Bin Zhang, Guangquan Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29590](https://doi.org/10.1609/aaai.v38i15.29590)

**Abstract**:

Multistream classification poses significant challenges due to the necessity for rapid adaptation in dynamic streaming processes with concept drift. Despite the growing research outcomes in this area, there has been a notable oversight regarding the temporal dynamic relationships between these streams, leading to the issue of negative transfer arising from irrelevant data. In this paper, we propose a novel Online Boosting Adaptive Learning (OBAL) method that effectively addresses this limitation by adaptively learning the dynamic correlation among different streams. Specifically, OBAL operates in a dual-phase mechanism, in the first of which we design an Adaptive COvariate Shift Adaptation (AdaCOSA) algorithm to construct an initialized ensemble model using archived data from various source streams, thus mitigating the covariate shift while learning the dynamic correlations via an adaptive re-weighting strategy. During the online process, we employ a Gaussian Mixture Model-based weighting mechanism, which is seamlessly integrated with the acquired correlations via AdaCOSA to effectively handle asynchronous drift. This approach significantly improves the predictive performance and stability of the target stream. We conduct comprehensive experiments on several synthetic and real-world data streams, encompassing various drifting scenarios and types. The results clearly demonstrate that OBAL achieves remarkable advancements in addressing multistream classification problems by effectively leveraging positive knowledge derived from multiple sources.

----

## [1842] Chronic Poisoning: Backdoor Attack against Split Learning

**Authors**: *Fangchao Yu, Bo Zeng, Kai Zhao, Zhi Pang, Lina Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29591](https://doi.org/10.1609/aaai.v38i15.29591)

**Abstract**:

Split learning is a computing resource-friendly distributed learning framework that protects client training data by splitting the model between the client and server. Previous work has proved that split learning faces a severe risk of privacy leakage, as a malicious server can recover the client's private data by hijacking the training process. In this paper, we first explore the vulnerability of split learning to server-side backdoor attacks, where our goal is to compromise the model's integrity. Since the server-side attacker cannot access the training data and client model in split learning, the traditional poisoning-based backdoor attack methods are no longer applicable. Therefore, constructing backdoor attacks in split learning poses significant challenges. Our strategy involves the attacker establishing a shadow model on the server side that can encode backdoor samples and guiding the client model to learn from this model during the training process, thereby enabling the client to acquire the same capability. Based on these insights, we propose a three-stage backdoor attack framework named SFI. Our attack framework minimizes assumptions about the attacker's background knowledge and ensures that the attack process remains imperceptible to the client. We implement SFI on various benchmark datasets, and extensive experimental results demonstrate its effectiveness and generality. For example, success rates of our attack on MNIST, Fashion, and CIFAR10 datasets all exceed 90%, with limited impact on the main task.

----

## [1843] Cheaper and Faster: Distributed Deep Reinforcement Learning with Serverless Computing

**Authors**: *Hanfei Yu, Jian Li, Yang Hua, Xu Yuan, Hao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29592](https://doi.org/10.1609/aaai.v38i15.29592)

**Abstract**:

Deep reinforcement learning (DRL) has gained immense success in many applications, including gaming AI, robotics, and system scheduling. Distributed algorithms and architectures have been vastly proposed (e.g., actor-learner architecture) to accelerate DRL training with large-scale server-based clusters. However, training on-policy algorithms with the actor-learner architecture unavoidably induces resource wasting due to synchronization between learners and actors, thus resulting in significantly extra billing. As a promising alternative, serverless computing naturally fits on-policy synchronization and alleviates resource wasting in distributed DRL training with pay-as-you-go pricing. Yet, none has leveraged serverless computing to facilitate DRL training.  This paper proposes MinionsRL, the first serverless distributed DRL training framework that aims to accelerate DRL training- and cost-efficiency with dynamic actor scaling. We prototype MinionsRL on top of Microsoft Azure Container Instances and evaluate it with popular DRL tasks from OpenAI Gym. Extensive experiments show that MinionsRL reduces total training time by up to 52% and training cost by 86% compared to latest solutions.

----

## [1844] Barely Supervised Learning for Graph-Based Fraud Detection

**Authors**: *Hang Yu, Zhengyang Liu, Xiangfeng Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29593](https://doi.org/10.1609/aaai.v38i15.29593)

**Abstract**:

In recent years, graph-based fraud detection methods have garnered increasing attention for their superior ability to tackle the issue of camouflage in fraudulent scenarios. However, these methods often rely on a substantial proportion of samples as the training set, disregarding the reality of scarce annotated samples in real-life scenarios. As a theoretical framework within semi-supervised learning, the principle of consistency regularization posits that unlabeled samples should be classified into the same category as their own perturbations. Inspired by this principle, this study incorporates unlabeled samples as an auxiliary during model training, designing a novel barely supervised learning method to address the challenge of limited annotated samples in fraud detection. Specifically, to tackle the issue of camouflage in fraudulent scenarios, we employ disentangled representation learning based on edge information for a small subset of annotated nodes. This approach partitions node features into three distinct components representing different connected edges, providing a foundation for the subsequent augmentation of unlabeled samples. For the unlabeled nodes used in auxiliary training, we apply both strong and weak augmentation and design regularization losses to enhance the detection performance of the model in the context of extremely limited labeled samples. Across five publicly available datasets, the proposed model showcases its superior detection capability over baseline models.

----

## [1845] A Non-parametric Graph Clustering Framework for Multi-View Data

**Authors**: *Shengju Yu, Siwei Wang, Zhibin Dong, Wenxuan Tu, Suyuan Liu, Zhao Lv, Pan Li, Miao Wang, En Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29594](https://doi.org/10.1609/aaai.v38i15.29594)

**Abstract**:

Multi-view graph clustering (MVGC) derives encouraging grouping results by seamlessly integrating abundant information inside heterogeneous data, and has captured surging  focus recently. 
Nevertheless, the majority of current MVGC works involve at least one hyper-parameter, which not only requires additional efforts for tuning, but also leads to a complicated solving procedure, 
largely harming the flexibility and scalability of corresponding algorithms. To this end, in the article we are devoted to getting rid of hyper-parameters, and devise a non-parametric graph clustering (NpGC) framework to more practically partition multi-view data. To be specific, we hold that hyper-parameters play a role in balancing error item and regularization item so as to form high-quality clustering representations.  Therefore, under without the assistance of hyper-parameters, how to acquire high-quality representations becomes the key. Inspired by this, we adopt two types of anchors, view-related and view-unrelated, to concurrently mine exclusive characteristics and common characteristics among views. Then, all anchors' information is gathered together via a consensus bipartite graph. By such ways, NpGC extracts both complementary and consistent multi-view features, thereby obtaining superior clustering results. Also, linear complexities enable it to handle datasets with over 120000 samples. Numerous experiments reveal NpGC's strong points compared to lots of  classical  approaches.

----

## [1846] DVSAI: Diverse View-Shared Anchors Based Incomplete Multi-View Clustering

**Authors**: *Shengju Yu, Siwei Wang, Pei Zhang, Miao Wang, Ziming Wang, Zhe Liu, Liming Fang, En Zhu, Xinwang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29595](https://doi.org/10.1609/aaai.v38i15.29595)

**Abstract**:

In numerous real-world applications, it is quite common that sample information is partially available for some views due to machine breakdown or sensor failure, causing the problem of incomplete multi-view clustering (IMVC). While several IMVC approaches using view-shared anchors have successfully achieved pleasing performance improvement, (1) they generally construct  anchors with only one dimension, which  could  deteriorate the multi-view diversity, bringing about serious information loss; (2) the constructed anchors are typically with a single size, which could not sufficiently characterize the distribution of the whole samples,  leading to limited clustering performance. For generating view-shared anchors with multi-dimension and multi-size for IMVC, we design a novel framework called Diverse View-Shared Anchors based Incomplete multi-view clustering  (DVSAI). Concretely, we associate each partial view with several potential spaces. 
In each space, we enable anchors to communicate among views and generate the view-shared anchors with space-specific dimension and size. Consequently,  spaces with various scales make  the generated view-shared anchors enjoy diverse dimensions and sizes. Subsequently, we devise an integration scheme with linear computational and memory expenditures to integrate the outputted multi-scale unified anchor graphs  such that running spectral algorithm generates the spectral embedding. Afterwards, we theoretically demonstrate that DVSAI owns linear time and space costs, thus well-suited for tackling  large-size datasets. Finally, comprehensive experiments confirm the effectiveness and advantages of  DVSAI.

----

## [1847] HGPrompt: Bridging Homogeneous and Heterogeneous Graphs for Few-Shot Prompt Learning

**Authors**: *Xingtong Yu, Yuan Fang, Zemin Liu, Xinming Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29596](https://doi.org/10.1609/aaai.v38i15.29596)

**Abstract**:

Graph neural networks (GNNs) and heterogeneous graph neural networks (HGNNs) are prominent techniques for homogeneous and heterogeneous graph representation learning, yet their performance in an end-to-end supervised framework greatly depends on the availability of task-specific supervision. To reduce the labeling cost, pre-training on self-supervised pretext tasks has become a popular paradigm, but there is often a gap between the pre-trained model and downstream tasks, stemming from the divergence in their objectives. To bridge the gap, prompt learning has risen as a promising direction especially in few-shot settings, without the need to fully fine-tune the pre-trained model. While there has been some early exploration of prompt-based learning on graphs, they primarily deal with homogeneous graphs, ignoring the heterogeneous graphs that are prevalent in downstream applications. In this paper, we propose HGPROMPT, a
novel pre-training and prompting framework to unify not only pre-training and downstream tasks but also homogeneous and heterogeneous graphs via a dual-template design. Moreover, we propose dual-prompt in HGPROMPT to assist a downstream task in locating the most relevant prior to bridge the gaps caused by not only feature variations but also heterogeneity differences across tasks. Finally, we thoroughly evaluate and analyze HGPROMPT through extensive experiments
on three public datasets.

----

## [1848] ANEDL: Adaptive Negative Evidential Deep Learning for Open-Set Semi-supervised Learning

**Authors**: *Yang Yu, Danruo Deng, Furui Liu, Qi Dou, Yueming Jin, Guangyong Chen, Pheng-Ann Heng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29597](https://doi.org/10.1609/aaai.v38i15.29597)

**Abstract**:

Semi-supervised learning (SSL) methods assume that labeled
data, unlabeled data and test data are from the same distribution. Open-set semi-supervised learning (Open-set SSL) con-
siders a more practical scenario, where unlabeled data and
test data contain new categories (outliers) not observed in
labeled data (inliers). Most previous works focused on out-
lier detection via binary classifiers, which suffer from insufficient scalability and inability to distinguish different types
of uncertainty. In this paper, we propose a novel framework,
Adaptive Negative Evidential Deep Learning (ANEDL) to
tackle these limitations. Concretely, we first introduce evidential deep learning (EDL) as an outlier detector to quantify
different types of uncertainty, and design different uncertainty
metrics for self-training and inference. Furthermore, we propose a novel adaptive negative optimization strategy, making
EDL more tailored to the unlabeled dataset containing both
inliers and outliers. As demonstrated empirically, our proposed method outperforms existing state-of-the-art methods
across four datasets.

----

## [1849] TIKP: Text-to-Image Knowledge Preservation for Continual Semantic Segmentation

**Authors**: *Zhidong Yu, Wei Yang, Xike Xie, Zhenbo Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29598](https://doi.org/10.1609/aaai.v38i15.29598)

**Abstract**:

Continual Semantic Segmentation (CSS) is an emerging trend, where catastrophic forgetting has been a perplexing problem. In this paper, we propose a Text-to-Image Knowledge Preservation (TIKP) framework to address this issue. TIKP applies Text-to-Image techniques to CSS by automatically generating prompts and content adaptation. It extracts associations between the labels of seen data and constructs text-level prompts based on these associations, which are preserved and maintained at each incremental step. During training, these prompts generate correlated images to mitigate the catastrophic forgetting. Particularly, as the generated images may have different distributions from the original data, TIKP transfers the knowledge by a content adaption loss, which determines the role played by the generated images in incremental training based on the similarity. In addition, for the classifier, we use the previous model from a different perspective: misclassifying new classes into old objects instead of the background. We propose a knowledge distillation loss based on wrong labels, enabling us to attribute varying weights to individual objects during the distillation process. Extensive experiments conducted in the same setting show that TIKP outperforms state-of-the-art methods by a large margin on benchmark datasets.

----

## [1850] Accelerating Text-to-Image Editing via Cache-Enabled Sparse Diffusion Inference

**Authors**: *Zihao Yu, Haoyang Li, Fangcheng Fu, Xupeng Miao, Bin Cui*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29599](https://doi.org/10.1609/aaai.v38i15.29599)

**Abstract**:

Due to the recent success of diffusion models, text-to-image generation is becoming increasingly popular and achieves a wide range of applications. Among them, text-to-image editing, or continuous text-to-image generation, attracts lots of attention and can potentially improve the quality of generated images. It's common to see that users may want to slightly edit the generated image by making minor modifications to their input textual descriptions for several rounds of diffusion inference. However, such an image editing process suffers from the low inference efficiency of many existing diffusion models even using GPU accelerators.

To solve this problem, we introduce Fast Image Semantically Edit (FISEdit), a cached-enabled sparse diffusion model inference engine for efficient text-to-image editing. The key intuition behind our approach is to utilize the semantic mapping between the minor modifications on the input text and the affected regions on the output image. For each text editing step, FISEdit can 1) automatically identify the affected image regions and 2) utilize the cached unchanged regions' feature map to accelerate the inference process. For the former, we measure the differences between cached and ad hoc feature maps given the modified textual description, extract the region with significant differences, and capture the affected region by masks. For the latter, we develop an efficient sparse diffusion inference engine that only computes the feature maps for the affected region while reusing the cached statistics for the rest of the image. Finally, extensive empirical results show that FISEdit can be 3.4 times and 4.4 times faster than existing methods on NVIDIA TITAN RTX and A100 GPUs respectively, and even generates more satisfactory images.

----

## [1851] PDE+: Enhancing Generalization via PDE with Adaptive Distributional Diffusion

**Authors**: *Yige Yuan, Bingbing Xu, Bo Lin, Liang Hou, Fei Sun, Huawei Shen, Xueqi Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29600](https://doi.org/10.1609/aaai.v38i15.29600)

**Abstract**:

The generalization of neural networks is a central challenge in machine learning, especially concerning the performance under distributions that differ from training ones. Current methods, mainly based on the data-driven paradigm such as data augmentation, adversarial training, and noise injection, may encounter limited generalization due to model non-smoothness. In this paper, we propose to investigate generalization from a Partial Differential Equation (PDE) perspective, aiming to enhance it directly through the underlying function of neural networks, rather than focusing on adjusting input data. Specifically, we first establish the connection between neural network generalization and the smoothness of the solution to a specific PDE, namely transport equation. Building upon this, we propose a general framework that introduces adaptive distributional diffusion into transport equation to enhance the smoothness of its solution, thereby improving generalization. In the context of neural networks, we put this theoretical framework into practice as PDE+ (PDE with Adaptive Distributional Diffusion) which diffuses each sample into a distribution covering semantically similar inputs. This enables better coverage of potentially unobserved distributions in training, thus improving generalization beyond merely data-driven methods. The effectiveness of PDE+ is validated through extensive experimental settings, demonstrating its superior performance compared to state-of-the-art methods. Our code is available at https://github.com/yuanyige/pde-add.

----

## [1852] Self-Paced Unified Representation Learning for Hierarchical Multi-Label Classification

**Authors**: *Zixuan Yuan, Hao Liu, Haoyi Zhou, Denghui Zhang, Xiao Zhang, Hao Wang, Hui Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29601](https://doi.org/10.1609/aaai.v38i15.29601)

**Abstract**:

Hierarchical Multi-Label Classification (HMLC) is a well-established problem that aims at assigning data instances to multiple classes stored in a hierarchical structure. Despite its importance, existing approaches often face two key limitations: (i) They employ dense networks to solely explore the class hierarchy as hard criterion for maintaining taxonomic consistency among predicted classes, yet without leveraging rich semantic relationships between instances and classes; (ii) They struggle to generalize in settings with deep class levels, since the mini-batches uniformly sampled from different levels ignore the varying complexities of data and result in a non-smooth model adaptation to sparse data. To mitigate these issues, we present a Self-Paced Unified Representation (SPUR) learning framework, which focuses on the interplay between instance and classes to flexibly organize the training process of HMLC algorithms. Our framework consists of two lightweight encoders designed to capture the semantics of input features and the topological information of the class hierarchy. These encoders generate unified embeddings of instances and class hierarchy, which enable SPUR to exploit semantic dependencies between them and produce predictions in line with taxonomic constraints. Furthermore, we introduce a dynamic hardness measurement strategy that considers both class hierarchy and instance features to estimate the learning difficulty of each instance. This strategy is achieved by incorporating the propagation loss obtained at each hierarchical level, allowing for a more comprehensive assessment of learning complexity. Extensive experiments on several empirical benchmarks demonstrate the effectiveness and efficiency of SPUR compared to state-of-the-art methods, especially in scenarios with missing features.

----

## [1853] A Plug-and-Play Quaternion Message-Passing Module for Molecular Conformation Representation

**Authors**: *Angxiao Yue, Dixin Luo, Hongteng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29602](https://doi.org/10.1609/aaai.v38i15.29602)

**Abstract**:

Graph neural networks have been widely used to represent 3D molecules, which capture molecular attributes and geometric information through various message-passing mechanisms.
This study proposes a novel quaternion message-passing (QMP) module that can be plugged into many existing 3D molecular representation models and enhance their power for distinguishing molecular conformations.
In particular, our QMP module represents the 3D rotations between one chemical bond and its neighbor bonds as a quaternion sequence. 
Then, it aggregates the rotations by the chained Hamilton product of the quaternions. 
The real part of the output quaternion is invariant to the global 3D rotations of molecules but sensitive to the local torsions caused by twisting bonds, providing discriminative information for training molecular conformation representation models. 
In theory, we prove that considering these features enables invariant GNNs to distinguish the conformations caused by bond torsions. 
We encapsulate the QMP module with acceleration, so combining existing models with the QMP requires merely one-line code and little computational cost. 
Experiments on various molecular datasets show that plugging our QMP module into existing invariant GNNs leads to consistent and significant improvements in molecular conformation representation and downstream tasks.

----

## [1854] Efficient Asynchronous Federated Learning with Prospective Momentum Aggregation and Fine-Grained Correction

**Authors**: *Yu Zang, Zhe Xue, Shilong Ou, Lingyang Chu, Junping Du, Yunfei Long*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29603](https://doi.org/10.1609/aaai.v38i15.29603)

**Abstract**:

Asynchronous federated learning (AFL) is a distributed machine learning technique that allows multiple devices to collaboratively train deep learning models without sharing local data. However, AFL suffers from low efficiency due to poor client model training quality and slow server model convergence speed, which are a result of the heterogeneous nature of both data and devices. To address these issues, we propose Efficient Asynchronous Federated Learning with Prospective Momentum Aggregation and Fine-Grained Correction (FedAC). Our framework consists of three key components. The first component is client weight evaluation based on temporal gradient, which evaluates the client weight based on the similarity between the client and server update directions. The second component is adaptive server update with prospective weighted momentum, which uses an asynchronous buffered update strategy and a prospective weighted momentum with adaptive learning rate to update the global model in server. The last component is client update with fine-grained gradient correction, which introduces a fine-grained gradient correction term to mitigate the client drift and correct the client stochastic gradient. We conduct experiments on real and synthetic datasets, and compare with existing federated learning methods. Experimental results demonstrate effective improvements in model training efficiency and AFL performance by our framework.

----

## [1855] Generalizing across Temporal Domains with Koopman Operators

**Authors**: *Qiuhao Zeng, Wei Wang, Fan Zhou, Gezheng Xu, Ruizhi Pu, Changjian Shui, Christian Gagné, Shichun Yang, Charles X. Ling, Boyu Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29604](https://doi.org/10.1609/aaai.v38i15.29604)

**Abstract**:

In the field of domain generalization, the task of constructing a predictive model capable of generalizing to a target domain without access to target data remains challenging. This problem becomes further complicated when considering evolving dynamics between domains. While various approaches have been proposed to address this issue, a comprehensive understanding of the underlying generalization theory is still lacking. In this study, we contribute novel theoretic results that aligning conditional distribution leads to the reduction of generalization bounds. Our analysis serves as a key motivation for solving the Temporal Domain Generalization (TDG) problem through the application of Koopman Neural Operators, resulting in Temporal Koopman Networks (TKNets). By employing Koopman Neural Operators, we effectively address the time-evolving distributions encountered in TDG using the principles of Koopman theory, where measurement functions are sought to establish linear transition relations between evolving domains. Through empirical evaluations conducted on synthetic and real-world datasets, we validate the effectiveness of our proposed approach.

----

## [1856] Hierarchical Multi-Marginal Optimal Transport for Network Alignment

**Authors**: *Zhichen Zeng, Boxin Du, Si Zhang, Yinglong Xia, Zhining Liu, Hanghang Tong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29605](https://doi.org/10.1609/aaai.v38i15.29605)

**Abstract**:

Finding node correspondence across networks, namely multi-network alignment, is an essential prerequisite for joint learning on multiple networks. Despite great success in aligning networks in pairs, the literature on multi-network alignment is sparse due to the exponentially growing solution space and lack of high-order discrepancy measures. To fill this gap, we propose a hierarchical multi-marginal optimal transport framework named HOT for multi-network alignment. To handle the large solution space, multiple networks are decomposed into smaller aligned clusters via the fused Gromov-Wasserstein (FGW) barycenter. To depict high-order relationships across multiple networks, the FGW distance is generalized to the multi-marginal setting, based on which networks can be aligned jointly. A fast proximal point method is further developed with guaranteed convergence to a local optimum. Extensive experiments and analysis show that our proposed HOT achieves significant improvements over the state-of-the-art in both effectiveness and scalability.

----

## [1857] Harnessing the Power of SVD: An SVA Module for Enhanced Signal Classification

**Authors**: *Lei Zhai, Shuyuan Yang, Yitong Li, Zhixi Feng, Zhihao Chang, Quanwei Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29606](https://doi.org/10.1609/aaai.v38i15.29606)

**Abstract**:

Deep learning methods have achieved outstanding performance in various signal tasks. However, due to degraded signals in real electromagnetic environment, it is crucial to seek methods that can improve the representation of signal features. In this paper, a Singular Value decomposition-based Attention, SVA is proposed to explore structure of signal data for adaptively enhancing intrinsic feature. Using a deep neural network as a base model, SVA performs feature semantic subspace learning through a decomposition layer and combines it with an attention layer to achieve adaptive enhancement of signal features. Moreover, we consider the gradient explosion problem brought by SVA and optimize SVA to improve the stability of training. Extensive experimental results demon-strate that applying SVA to a generalized classification model can significantly improve its ability in representations, making its recognition performance competitive with, or even better than, the state-of-the-art task-specific models.

----

## [1858] Optimistic Model Rollouts for Pessimistic Offline Policy Optimization

**Authors**: *Yuanzhao Zhai, Yiying Li, Zijian Gao, Xudong Gong, Kele Xu, Dawei Feng, Bo Ding, Huaimin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29607](https://doi.org/10.1609/aaai.v38i15.29607)

**Abstract**:

Model-based offline reinforcement learning (RL) has made remarkable progress, offering a promising avenue for improving generalization with synthetic model rollouts. Existing works primarily focus on incorporating pessimism for policy optimization, usually via constructing a Pessimistic Markov Decision Process (P-MDP). However, the P-MDP discourages the policies from learning in out-of-distribution (OOD) regions beyond the support of offline datasets, which can under-utilize the generalization ability of dynamics models. In contrast, we propose constructing an Optimistic MDP (O-MDP). We initially observed the potential benefits of optimism brought by encouraging more OOD rollouts. Motivated by this observation, we present ORPO, a simple yet effective model-based offline RL framework. ORPO generates Optimistic model Rollouts for Pessimistic offline policy Optimization. Specifically, we train an optimistic rollout policy in the O-MDP to sample more OOD model rollouts. Then we relabel the sampled state-action pairs with penalized rewards, and optimize the output policy in the P-MDP. Theoretically, we demonstrate that the performance of policies trained with ORPO can be lower-bounded in linear MDPs. Experimental results show that our framework significantly outperforms P-MDP baselines by a margin of 30%, achieving state-of-the-art performance on the widely-used benchmark. Moreover, ORPO exhibits notable advantages in problems that require generalization.

----

## [1859] MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning

**Authors**: *Baoquan Zhang, Chuyao Luo, Demin Yu, Xutao Li, Huiwei Lin, Yunming Ye, Bowen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29608](https://doi.org/10.1609/aaai.v38i15.29608)

**Abstract**:

Equipping a deep model the ability of few-shot learning (FSL) is a core challenge for artificial intelligence. Gradient-based meta-learning effectively addresses the challenge by learning how to learn novel tasks. Its key idea is learning a deep model in a bi-level optimization manner, where the outer-loop process learns a shared gradient descent algorithm (called meta-optimizer), while the inner-loop process leverages it to optimize a task-specific base learner with few examples. Although these methods have shown superior performance on FSL, the outer-loop process requires calculating second-order derivatives along the inner-loop path, which imposes considerable memory burdens and the risk of vanishing gradients. This degrades meta-learning performance. Inspired by recent diffusion models, we find that the inner-loop gradient descent process can be viewed as a reverse process (i.e., denoising) of diffusion where the target of denoising is the weight of base learner but origin data. Based on this fact, we propose to model the gradient descent algorithm as a diffusion model and then present a novel conditional diffusion-based meta-learning, called MetaDiff, that effectively models the optimization process of base learner weights from Gaussian initialization to target weights in a denoising manner. Thanks to the training efficiency of diffusion models, our MetaDiff does not need to differentiate through the inner-loop path such that the memory burdens and the risk of vanishing gradients can be effectively alleviated for improving FSL. Experimental results show that our MetaDiff outperforms state-of-the-art gradient-based meta-learning family on FSL tasks.

----

## [1860] Learning Cluster-Wise Anchors for Multi-View Clustering

**Authors**: *Chao Zhang, Xiuyi Jia, Zechao Li, Chunlin Chen, Huaxiong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29609](https://doi.org/10.1609/aaai.v38i15.29609)

**Abstract**:

Due to its effectiveness and efficiency, anchor based multi-view clustering (MVC) has recently attracted much attention. Most existing approaches try to adaptively learn anchors to construct an anchor graph for clustering. However, they generally focus on improving the diversity among anchors by using orthogonal constraint and ignore the underlying semantic relations, which may make the anchors not representative and discriminative enough. To address this problem, we propose an adaptive Cluster-wise Anchor learning based MVC method, CAMVC for short. We first make an anchor cluster assumption that supposes the prior cluster structure of target anchors by pre-defining a consensus cluster indicator matrix. Based on the prior knowledge, an explicit cluster structure of latent anchors is enforced by learning diverse cluster centroids, which can explore both inter-cluster diversity and intra-cluster consistency of anchors, and improve the subspace representation discrimination. Extensive results demonstrate the effectiveness and superiority of our proposed method compared with some state-of-the-art MVC approaches.

----

## [1861] Targeted Activation Penalties Help CNNs Ignore Spurious Signals

**Authors**: *Dekai Zhang, Matt Williams, Francesca Toni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29610](https://doi.org/10.1609/aaai.v38i15.29610)

**Abstract**:

Neural networks (NNs) can learn to rely on spurious signals in the training data, leading to poor generalisation. Recent methods tackle this problem by training NNs with additional ground-truth annotations of such signals. These methods may, however, let spurious signals re-emerge in deep convolutional NNs (CNNs). We propose Targeted Activation Penalty (TAP), a new method tackling the same problem by penalising activations to control the re-emergence of spurious signals in deep CNNs, while also lowering training times and memory usage. In addition, ground-truth annotations can be expensive to obtain. We show that TAP still works well with annotations generated by pre-trained models as effective substitutes of ground-truth annotations. We demonstrate the power of TAP against two state-of-the-art baselines on the MNIST benchmark and on two clinical image datasets, using four different CNN architectures.

----

## [1862] Robust Test-Time Adaptation for Zero-Shot Prompt Tuning

**Authors**: *Dingchu Zhang, Zhi Zhou, Yu-Feng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29611](https://doi.org/10.1609/aaai.v38i15.29611)

**Abstract**:

CLIP has demonstrated remarkable generalization across diverse downstream tasks. By aligning images and texts in a shared feature space, they enable zero-shot classification via hand-crafted prompts. However, recent studies have shown that hand-crafted prompts may be unsuitable in practical applications. Specifically, choosing an appropriate prompt for a given task requires accurate data and knowledge, which may not be obtainable in practical situations. An inappropriate prompt can result in poor performance. Moreover, if there is no training data, tuning prompts arbitrarily through unlabeled test data may lead to serious performance degradation when giving hand-crafted prompts. Our study reveals that the aforementioned problems are mainly due to the biases in testing data (Data Bias) and pre-trained CLIP model (Model Bias). The Data Bias makes it challenging to choose an appropriate prompt, while Model Bias renders some predictions inaccurate and biased, which leads to error accumulation. To address these biases, we propose robust test-time Adaptation for zeroshot Prompt tuning (ADAPROMPT). Specifically, we ensemble multiple prompts to avoid the worst-case results and dynamically tune prompts to adapt to Data Bias during testing. Furthermore, we adopt a confidence-aware buffer to store balanced and confident unlabeled test data to tune prompts in order to overcome Model Bias. Our extensive experiments on several benchmarks demonstrate that ADAPROMPT alleviates model bias, adapts to data bias and mostly outperforms the state-of-the-art methods at a small time cost. Moreover, our experimental results reveal that ADAPROMPT hardly encounters any performance degradation on these datasets.

----

## [1863] FM-OV3D: Foundation Model-Based Cross-Modal Knowledge Blending for Open-Vocabulary 3D Detection

**Authors**: *Dongmei Zhang, Chang Li, Renrui Zhang, Shenghao Xie, Wei Xue, Xiaodong Xie, Shanghang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29612](https://doi.org/10.1609/aaai.v38i15.29612)

**Abstract**:

The superior performances of pre-trained foundation models in various visual tasks underscore their potential to enhance the 2D models' open-vocabulary ability. Existing methods explore analogous applications in the 3D space. However, most of them only center around knowledge extraction from singular foundation models, which limits the open-vocabulary ability of 3D models. We hypothesize that leveraging complementary pre-trained knowledge from various foundation models can improve knowledge transfer from 2D pre-trained visual language models to the 3D space. In this work, we propose FM-OV3D, a method of Foundation Model-based Cross-modal Knowledge Blending for Open-Vocabulary 3D Detection, which improves the open-vocabulary localization and recognition abilities of 3D model by blending knowledge from multiple pre-trained foundation models, achieving true open-vocabulary without facing constraints from original 3D datasets. Specifically, to learn the open-vocabulary 3D localization ability, we adopt the open-vocabulary localization knowledge of the Grounded-Segment-Anything model. For open-vocabulary 3D recognition ability, We leverage the knowledge of generative foundation models, including GPT-3 and Stable Diffusion models, and cross-modal discriminative models like CLIP. The experimental results on two popular benchmarks for open-vocabulary 3D object detection show that our model efficiently learns knowledge from multiple foundation models to enhance the open-vocabulary ability of the 3D model and successfully achieves state-of-the-art performance in open-vocabulary 3D object detection tasks. Code is released at https://github.com/dmzhang0425/FM-OV3D.git.

----

## [1864] Coupled Confusion Correction: Learning from Crowds with Sparse Annotations

**Authors**: *Hansong Zhang, Shikun Li, Dan Zeng, Chenggang Yan, Shiming Ge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29613](https://doi.org/10.1609/aaai.v38i15.29613)

**Abstract**:

As the size of the datasets getting larger, accurately annotating such datasets is becoming more impractical due to the expensiveness on both time and economy. Therefore, crowd-sourcing has been widely adopted to alleviate the cost of collecting labels, which also inevitably introduces label noise and eventually degrades the performance of the model. To learn from crowd-sourcing annotations, modeling the expertise of each annotator is a common but challenging paradigm, because the annotations collected by crowd-sourcing are usually highly-sparse. To alleviate this problem, we propose Coupled Confusion Correction (CCC), where two models are simultaneously trained to correct the confusion matrices learned by each other. Via bi-level optimization, the confusion matrices learned by one model can be corrected by the distilled data from the other. Moreover, we cluster the ``annotator groups'' who share similar expertise so that their confusion matrices could be corrected together. In this way, the expertise of the annotators, especially of those who provide seldom labels, could be better captured. Remarkably, we point out that the annotation sparsity not only means the average number of labels is low, but also there are always some annotators who provide very few labels, which is neglected by previous works when constructing synthetic crowd-sourcing annotations. Based on that, we propose to use Beta distribution to control the generation of the crowd-sourcing labels so that the synthetic annotations could be more consistent with the real-world ones. Extensive experiments are conducted on two types of synthetic datasets and three real-world datasets, the results of which demonstrate that CCC significantly outperforms state-of-the-art approaches. Source codes are available at: https://github.com/Hansong-Zhang/CCC.

----

## [1865] Exponential Hardness of Optimization from the Locality in Quantum Neural Networks

**Authors**: *Hao-Kai Zhang, Chengkai Zhu, Geng Liu, Xin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29614](https://doi.org/10.1609/aaai.v38i15.29614)

**Abstract**:

Quantum neural networks (QNNs) have become a leading paradigm for establishing near-term quantum applications in recent years. The trainability issue of QNNs has garnered extensive attention, spurring demand for a comprehensive analysis of QNNs in order to identify viable solutions. In this work, we propose a perspective that characterizes the trainability of QNNs based on their locality. We prove that the entire variation range of the loss function via adjusting any local quantum gate vanishes exponentially in the number of qubits with a high probability for a broad class of QNNs. This result reveals extra harsh constraints independent of gradients and unifies the restrictions on gradient-based and gradient-free optimizations naturally. We showcase the validity of our results with numerical simulations of representative models and examples. Our findings, as a fundamental property of random quantum circuits, deepen the understanding of the role of locality in QNNs and serve as a guideline for assessing the effectiveness of diverse training strategies for quantum neural networks.

----

## [1866] HONGAT: Graph Attention Networks in the Presence of High-Order Neighbors

**Authors**: *Heng-Kai Zhang, Yi-Ge Zhang, Zhi Zhou, Yu-Feng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29615](https://doi.org/10.1609/aaai.v38i15.29615)

**Abstract**:

Graph Attention Networks (GATs) that compute node representation by its lower-order neighbors, are state-of-the-art architecture for representation learning with graphs. In practice, however, the high-order neighbors that turn out to be useful, remain largely unemployed in GATs. Efforts on this issue remain to be limited. This paper proposes a simple and effective high-order neighbor GAT (HONGAT) model to both effectively exploit informative high-order neighbors and address over-smoothing at the decision boundary of nodes. Two tightly coupled novel technologies, namely common neighbor similarity and new masking matrix, are introduced. Specifically, high-order neighbors are fully explored by generic high-order common-neighbor-based similarity; in order to prevent severe over-smoothing, typical averaging range no longer works well and a new masking mechanism is employed without any extra hyperparameter. Extensive empirical evaluation on real-world datasets clearly shows the necessity of the new algorithm in the ability of exploring high-order neighbors, which promisingly achieves significant gains over previous state-of-the-art graph attention methods.

----

## [1867] Memory-Efficient Reversible Spiking Neural Networks

**Authors**: *Hong Zhang, Yu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29616](https://doi.org/10.1609/aaai.v38i15.29616)

**Abstract**:

Spiking neural networks (SNNs) are potential competitors to artificial neural networks (ANNs) due to their high energy-efficiency on neuromorphic hardware. However, SNNs are unfolded over simulation time steps during the training process. Thus, SNNs require much more memory than ANNs, which impedes the training of deeper SNN models. In this paper, we propose the reversible spiking neural network to reduce the memory cost of intermediate activations and membrane potentials during training. Firstly, we extend the reversible architecture along temporal dimension and propose the reversible spiking block, which can reconstruct the computational graph and recompute all intermediate variables in forward pass with a reverse process. On this basis, we adopt the state-of-the-art SNN models to the reversible variants, namely reversible spiking ResNet (RevSResNet) and reversible spiking transformer (RevSFormer). Through experiments on static and neuromorphic datasets, we demonstrate that the memory cost per image of our reversible SNNs does not increase with the network depth. On CIFAR10 and CIFAR100 datasets, our RevSResNet37 and RevSFormer-4-384 achieve comparable accuracies and consume 3.79x and 3.00x lower GPU memory per image than their counterparts with roughly identical model complexity and parameters. We believe that this work can unleash the memory constraints in SNN training and pave the way for training extremely large and deep SNNs.

----

## [1868] FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning

**Authors**: *Jianqing Zhang, Yang Liu, Yang Hua, Jian Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29617](https://doi.org/10.1609/aaai.v38i15.29617)

**Abstract**:

Recently, Heterogeneous Federated Learning (HtFL) has attracted attention due to its ability to support heterogeneous models and data. To reduce the high communication cost of transmitting model parameters, a major challenge in HtFL, prototype-based HtFL methods are proposed to solely share class representatives, a.k.a, prototypes, among heterogeneous clients while maintaining the privacy of clients’ models. However, these prototypes are naively aggregated into global prototypes on the server using weighted averaging, resulting in suboptimal global knowledge which negatively impacts the performance of clients. To overcome this challenge, we introduce a novel HtFL approach called FedTGP, which leverages our Adaptive-margin-enhanced Contrastive Learning (ACL) to learn Trainable Global Prototypes (TGP) on the server. By incorporating ACL, our approach enhances prototype separability while preserving semantic meaning. Extensive experiments with twelve heterogeneous models demonstrate that our FedTGP surpasses state-of-the-art methods by up to 9.08% in accuracy while maintaining the communication and privacy advantages of prototype-based HtFL. Our code is available at https://github.com/TsingZ0/FedTGP.

----

## [1869] Reinforced Adaptive Knowledge Learning for Multimodal Fake News Detection

**Authors**: *Litian Zhang, Xiaoming Zhang, Ziyi Zhou, Feiran Huang, Chaozhuo Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29618](https://doi.org/10.1609/aaai.v38i15.29618)

**Abstract**:

Nowadays, detecting multimodal fake news has emerged as a foremost concern since the widespread dissemination of fake news may incur adverse societal impact. Conventional methods generally focus on capturing the linguistic and visual semantics within the multimodal  content, which fall short in effectively distinguishing the heightened level of meticulous fabrications. Recently, external knowledge is introduced to provide valuable background facts as complementary to facilitate news detection. Nevertheless, existing knowledge-enhanced endeavors directly incorporate all knowledge contexts through static entity embeddings, resulting in the potential noisy and content-irrelevant knowledge. Moreover, the integration of knowledge entities makes it intractable to model the sophisticated correlations between multimodal semantics and knowledge entities. In light of these limitations, we propose a novel Adaptive Knowledge-Aware Fake News Detection model, dubbed AKA-Fake. For each news, AKA-Fake learns a compact knowledge subgraph under a reinforcement learning paradigm, which consists of a subset of entities and contextual neighbors in the knowledge graph, restoring the most informative knowledge facts. A novel heterogeneous graph learning module is further proposed to capture the reliable cross-modality correlations via topology refinement and modality-attentive pooling. Our proposal is extensively evaluated over three popular datasets, and experimental results demonstrate the superiority of AKA-Fake.

----

## [1870] Multi-Label Supervised Contrastive Learning

**Authors**: *Pingyue Zhang, Mengyue Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29619](https://doi.org/10.1609/aaai.v38i15.29619)

**Abstract**:

Multi-label classification is an arduous problem given the complication in label correlation. Whilst sharing a common goal with contrastive learning in utilizing correlations for representation learning, how to better leverage label information remains challenging. 
Previous endeavors include extracting label-level presentations or mapping labels to an embedding space, overlooking the correlation between multiple labels. 
It exhibits a great ambiguity in determining positive samples with different extent of label overlap between samples and integrating such relations in loss functions. 
In our work, we propose Multi-Label Supervised Contrastive learning (MulSupCon) with a novel contrastive loss function to adjust weights based on how much overlap one sample shares with the anchor. 
By analyzing gradients, we explain why our method performs better under multi-label circumstances. 
To evaluate, we conduct direct classification and transfer learning on several multi-label datasets, including widely-used image datasets such as MS-COCO and NUS-WIDE.
Validation indicates that our method outperforms the traditional multi-label classification method and shows a competitive performance when comparing to other existing approaches.

----

## [1871] United We Stand: Accelerating Privacy-Preserving Neural Inference by Conjunctive Optimization with Interleaved Nexus

**Authors**: *Qiao Zhang, Tao Xiang, Chunsheng Xin, Hongyi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29620](https://doi.org/10.1609/aaai.v38i15.29620)

**Abstract**:

Privacy-preserving Machine Learning as a Service (MLaaS) enables the powerful cloud server to run its well-trained neural model upon the input from resource-limited client, with both of server's model parameters and client's input data protected. While computation efficiency is critical for the practical implementation of privacy-preserving MLaaS and it is inspiring to witness recent advances towards efficiency improvement, there still exists a significant performance gap to real-world applications. In general, state-of-the-art frameworks perform function-wise efficiency optimization based on specific cryptographic primitives. Although it is logical, such independent optimization for each function makes noticeable amount of expensive operations unremovable and misses the opportunity to further accelerate the performance by jointly considering privacy-preserving computation among adjacent functions. As such, we propose COIN: Conjunctive Optimization with Interleaved Nexus, which remodels mainstream computation for each function to conjunctive counterpart for composite function, with a series of united optimization strategies. Specifically, COIN jointly computes a pair of consecutive nonlinear-linear functions in the neural model by reconstructing the intermediates throughout the whole procedure, which not only eliminates the most expensive crypto operations without invoking extra encryption enabler, but also makes the online crypto complexity independent of filter size. Experimentally, COIN demonstrates 11.2x to 29.6x speedup over various function dimensions from modern networks, and 6.4x to 12x speedup on the total computation time when applied in networks with model input from small-scale CIFAR10 to large-scale ImageNet.

----

## [1872] A Learnable Discrete-Prior Fusion Autoencoder with Contrastive Learning for Tabular Data Synthesis

**Authors**: *Rongchao Zhang, Yiwei Lou, Dexuan Xu, Yongzhi Cao, Hanpin Wang, Yu Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29621](https://doi.org/10.1609/aaai.v38i15.29621)

**Abstract**:

The actual collection of tabular data for sharing involves confidentiality and privacy constraints, leaving the potential risks of machine learning for interventional data analysis unsafely averted. Synthetic data has emerged recently as a privacy-protecting solution to address this challenge. However, existing approaches regard discrete and continuous modal features as separate entities, thus falling short in properly capturing their inherent correlations. In this paper, we propose a novel contrastive learning guided Gaussian Transformer autoencoder, termed GTCoder, to synthesize photo-realistic multimodal tabular data for scientific research. Our approach introduces a transformer-based fusion module that seamlessly integrates multimodal features, permitting for mining more informative latent representations. The attention within the fusion module directs the integrated output features to focus on critical components that facilitate the task of generating latent embeddings. Moreover, we formulate a contrastive learning strategy to implicitly constrain the embeddings from discrete features in the latent feature space by encouraging the similar discrete feature distributions closer while pushing the dissimilar further away, in order to better enhance the representation of the latent embedding. Experimental results indicate that GTCoder is effective to generate photo-realistic synthetic data, with interactive interpretation of latent embedding, and performs favorably against some baselines on most real-world and simulated datasets.

----

## [1873] Efficient Deweahter Mixture-of-Experts with Uncertainty-Aware Feature-Wise Linear Modulation

**Authors**: *Rongyu Zhang, Yulin Luo, Jiaming Liu, Huanrui Yang, Zhen Dong, Denis A. Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Yuan Du, Shanghang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29622](https://doi.org/10.1609/aaai.v38i15.29622)

**Abstract**:

The Mixture-of-Experts (MoE) approach has demonstrated outstanding scalability in multi-task learning including low-level upstream tasks such as concurrent removal of multiple adverse weather effects.
However, the conventional MoE architecture with parallel Feed Forward Network (FFN) experts leads to significant parameter and computational overheads that hinder its efficient deployment. In addition, the naive MoE linear router is suboptimal in assigning task-specific features to multiple experts which limits its further scalability.
In this work, we propose an efficient MoE architecture with weight sharing across the experts. Inspired by the idea of linear feature modulation (FM), our architecture implicitly instantiates multiple experts via learnable activation modulations on a single shared expert block.
The proposed Feature Modulated Expert (FME) serves as a building block for the novel Mixture-of-Feature-Modulation-Experts (MoFME) architecture, which can scale up the number of experts with low overhead.
We further propose an Uncertainty-aware Router (UaR) to assign task-specific features to different FM modules with well-calibrated weights. This enables MoFME to effectively learn diverse expert functions for multiple tasks.
The conducted experiments on the multi-deweather task show that our MoFME outperforms the state-of-the-art in the image restoration quality by 0.1-0.2 dB while saving more than 74% of parameters and 20% inference time over the conventional MoE counterpart. Experiments on the downstream segmentation and classification tasks further demonstrate the generalizability of MoFME to real open-world applications.

----

## [1874] Analyzing Generalization in Policy Networks: A Case Study with the Double-Integrator System

**Authors**: *Ruining Zhang, Haoran Han, Maolong Lv, Qisong Yang, Jian Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29623](https://doi.org/10.1609/aaai.v38i15.29623)

**Abstract**:

Extensive utilization of deep reinforcement learning (DRL) policy networks in diverse continuous control tasks has raised questions regarding performance degradation in expansive state spaces where the input state norm is larger than that in the training environment. This paper aims to uncover the underlying factors contributing to such performance deterioration when dealing with expanded state spaces, using a novel analysis technique known as state division. In contrast to prior approaches that employ state division merely as a post-hoc explanatory tool, our methodology delves into the intrinsic characteristics of DRL policy networks. Specifically, we demonstrate that the expansion of state space induces the activation function $\tanh$ to exhibit saturability, resulting in the transformation of the state division boundary from nonlinear to linear. Our analysis centers on the paradigm of the double-integrator system, revealing that this gradual shift towards linearity imparts a control behavior reminiscent of bang-bang control. However, the inherent linearity of the division boundary prevents the attainment of an ideal bang-bang control, thereby introducing unavoidable overshooting. Our experimental investigations, employing diverse RL algorithms, establish that this performance phenomenon stems from inherent attributes of the DRL policy network, remaining consistent across various optimization algorithms.

----

## [1875] Reviewing the Forgotten Classes for Domain Adaptation of Black-Box Predictors

**Authors**: *Shaojie Zhang, Chun Shen, Shuai Lü, Zeyu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29624](https://doi.org/10.1609/aaai.v38i15.29624)

**Abstract**:

For addressing the data privacy and portability issues of domain adaptation, Domain Adaptation of Black-box Predictors (DABP) aims to adapt a black-box source model to an unlabeled target domain without accessing both the source-domain data and details of the source model. Although existing DABP approaches based on knowledge distillation (KD) have achieved promising results, we experimentally find that these methods all have the minority class forgetting issue, which refers that the trained model completely forgets some minority classes. To address this issue, we propose a method called Reviewing the Forgotten Classes (RFC), which including two main modules. Firstly, we propose a simple but effective component called selection training (ST). ST selects classes that the model tends to forget according to the learning status of the model and obtains clean samples of the selected classes with the small-loss criterion for enhanced training. ST is orthogonal to previous methods and can effectively alleviate their minority class forgetting issue. Secondly, we find that neighborhood clustering (NC) can help the model learn more balanced than KD so that further alleviate the minority class forgetting issue. However, NC is based on the fact that target features from the source model already form some semantic structure, while DABP is unable to obtain the source model. Thus, we use KD and ST to warm up the target model to form a certain semantic structure. Overall, our method inherits the merits of both ST and NC, and achieves state of the art on three DABP benchmarks.

----

## [1876] TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling

**Authors**: *Shimin Zhang, Qu Yang, Chenxiang Ma, Jibin Wu, Haizhou Li, Kay Chen Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29625](https://doi.org/10.1609/aaai.v38i15.29625)

**Abstract**:

The identification of sensory cues associated with potential opportunities and dangers is frequently complicated by unrelated events that separate useful cues by long delays. As a result, it remains a challenging task for state-of-the-art spiking neural networks (SNNs) to establish long-term temporal dependency between distant cues. To address this challenge, we propose a novel biologically inspired Two-Compartment Leaky Integrate-and-Fire spiking neuron model, dubbed TC-LIF. The proposed model incorporates carefully designed somatic and dendritic compartments that are tailored to facilitate learning long-term temporal dependencies. Furthermore, the theoretical analysis is provided to validate the effectiveness of TC-LIF in propagating error gradients over an extended temporal duration. Our experimental results, on a diverse range of temporal classification tasks, demonstrate superior temporal classification capability, rapid training convergence, and high energy efficiency of the proposed TC-LIF model. Therefore, this work opens up a myriad of opportunities for solving challenging temporal processing tasks on emerging neuromorphic computing systems. Our code is publicly available at https://github.com/ZhangShimin1/TC-LIF.

----

## [1877] Learning with Noisy Labels Using Hyperspherical Margin Weighting

**Authors**: *Shuo Zhang, Yuwen Li, Zhongyu Wang, Jianqing Li, Chengyu Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29626](https://doi.org/10.1609/aaai.v38i15.29626)

**Abstract**:

Datasets often include noisy labels, but learning from them is difficult. Since mislabeled examples usually have larger loss values in training, the small-loss trick is regarded as a standard metric to identify the clean example from the training set for better performance. Nonetheless, this proposal ignores that some clean but hard-to-learn examples also generate large losses. They could be misidentified by this criterion. In this paper, we propose a new metric called the Integrated Area Margin (IAM), which is superior to the traditional small-loss trick, particularly in recognizing the clean but hard-to-learn examples. According to the IAM, we further offer the Hyperspherical Margin Weighting (HMW) approach. It is a new sample weighting strategy that restructures the importance of each example. It should be highlighted that our approach is universal and can strengthen various methods in this field. Experiments on both benchmark and real-world datasets indicate that our HMW outperforms many state-of-the-art approaches in learning with noisy label tasks. Codes are available at https://github.com/Zhangshuojackpot/HMW.

----

## [1878] One Step Closer to Unbiased Aleatoric Uncertainty Estimation

**Authors**: *Wang Zhang, Ziwen Martin Ma, Subhro Das, Tsui-Wei Lily Weng, Alexandre Megretski, Luca Daniel, Lam M. Nguyen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29627](https://doi.org/10.1609/aaai.v38i15.29627)

**Abstract**:

Neural networks are powerful tools in various applications, and quantifying their uncertainty is crucial for reliable decision-making. In the deep learning field, the uncertainties are usually categorized into aleatoric (data) and epistemic (model) uncertainty. In this paper, we point out that the existing popular variance attenuation method highly overestimates aleatoric uncertainty. To address this issue, we proposed a new estimation method by actively de-noising the observed data. By conducting a broad range of experiments, we demonstrate that our proposed approach provides a much closer approximation to the actual data uncertainty than the standard method.

----

## [1879] Gaussian Process Neural Additive Models

**Authors**: *Wei Zhang, Brian Barr, John Paisley*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29628](https://doi.org/10.1609/aaai.v38i15.29628)

**Abstract**:

Deep neural networks have revolutionized many fields, but their black-box nature also occasionally prevents their wider adoption in fields such as healthcare and finance where interpretable and explainable models are required. The recent development of Neural Additive Models (NAMs) poses a major step in the direction of interpretable deep learning for tabular datasets. In this paper, we propose a new subclass of NAMs that utilize a single-layer neural network construction of the Gaussian process via random Fourier features, which we call Gaussian Process Neural Additive Models (GP-NAM). GP-NAMs have the advantage of a convex objective function and number of trainable parameters that grows linearly with feature dimensions. It suffers no loss in performance compared with deeper NAM approaches because GPs are well-suited to learning complex non-parametric univariate functions. We demonstrate the performance of GP-NAM on several tabular datasets, showing that it achieves comparable performance in both classification and regression tasks with a massive reduction in the number of parameters.

----

## [1880] From Toxic to Trustworthy: Using Self-Distillation and Semi-supervised Methods to Refine Neural Networks

**Authors**: *Xianda Zhang, Baolin Zheng, Jianbao Hu, Chengyang Li, Xiaoying Bai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29629](https://doi.org/10.1609/aaai.v38i15.29629)

**Abstract**:

Despite the tremendous success of deep neural networks (DNNs) across various fields, their susceptibility to potential backdoor attacks seriously threatens their application security, particularly in safety-critical or security-sensitive ones. Given this growing threat, there is a pressing need for research into purging backdoors from DNNs. However, prior efforts on erasing backdoor triggers not only failed to withstand increasingly powerful attacks but also resulted in reduced model performance. In this paper, we propose From Toxic to Trustworthy (FTT), an innovative approach to eliminate backdoor triggers while simultaneously enhancing model accuracy. Following the stringent and practical assumption of limited availability of clean data, we introduce a self-attention distillation (SAD) method to remove the backdoor by aligning the shallow and deep parts of the network. Furthermore, we first devise a semi-supervised learning (SSL) method that leverages ubiquitous and available poisoned data to further purify backdoors and improve accuracy. Extensive experiments on various attacks and models have shown that our FTT can reduce the attack success rate from 97% to 1% and improve the accuracy of 4% on average, demonstrating its effectiveness in mitigating backdoor attacks and improving model performance. Compared to state-of-the-art (SOTA) methods, our FTT can reduce the attack success rate by 2 times and improve the accuracy by 5%, shedding light on backdoor cleansing.

----

## [1881] Low Category Uncertainty and High Training Potential Instance Learning for Unsupervised Domain Adaptation

**Authors**: *Xinyu Zhang, Meng Kang, Shuai Lü*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29630](https://doi.org/10.1609/aaai.v38i15.29630)

**Abstract**:

Recently, instance contrastive learning achieves good results in unsupervised domain adaptation. It reduces the distances between positive samples and the anchor, increases the distances between negative samples and the anchor, and learns discriminative feature representations for target samples. However, most recent methods for identifying positive and negative samples are based on whether the pseudo-labels of samples and the pseudo-label of the anchor correspond to the same class. Due to the lack of target labels, many uncertain data are mistakenly labeled during the training process, and many low training potential data are also utilized. To address these problems, we propose Low Category Uncertainty and High Training Potential Instance Learning for Unsupervised Domain Adaptation (LUHP). We first propose a weight to measure the category uncertainty of the target sample. We can effectively filter the samples near the decision boundary through category uncertainty thresholds which are calculated by weights. Then we propose a new loss to focus on samples with high training potential. Finally, for anchors with low category uncertainty, we propose a sample reuse strategy to make the model more robust. We demonstrate the effectiveness of LUHP by showing the results of four datasets widely used in unsupervised domain adaptation.

----

## [1882] Class-Attribute Priors: Adapting Optimization to Heterogeneity and Fairness Objective

**Authors**: *Xuechen Zhang, Mingchen Li, Jiasi Chen, Christos Thrampoulidis, Samet Oymak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29631](https://doi.org/10.1609/aaai.v38i15.29631)

**Abstract**:

Modern classification problems exhibit heterogeneities across individual classes: Each class may have unique attributes, such as sample size, label quality, or predictability (easy vs difficult), and variable importance at test-time. Without care, these heterogeneities impede the learning process, most notably, when optimizing fairness objectives. Confirming this, under a gaussian mixture setting, we show that the optimal SVM classifier for balanced accuracy needs to be adaptive to the class attributes. This motivates us to propose CAP: An effective and general method that generates a class-specific learning strategy (e.g.~hyperparameter) based on the attributes of that class. This way, optimization process better adapts to heterogeneities. CAP leads to substantial improvements over the naive approach of assigning separate hyperparameters to each class. We instantiate CAP for loss function design and post-hoc logit adjustment, with emphasis on label-imbalanced problems. We show that CAP is competitive with prior art and its flexibility unlocks clear benefits for fairness objectives beyond balanced accuracy. Finally, we evaluate CAP on problems with label noise as well as weighted test objectives to showcase how CAP can jointly adapt to different heterogeneities.

----

## [1883] Learning Multi-Task Sparse Representation Based on Fisher Information

**Authors**: *Yayu Zhang, Yuhua Qian, Guoshuai Ma, Keyin Zheng, Guoqing Liu, Qingfu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29632](https://doi.org/10.1609/aaai.v38i15.29632)

**Abstract**:

Multi-task learning deals with multiple related tasks simultaneously by sharing knowledge. In a typical deep multi-task learning model, all tasks use the same feature space and share the latent knowledge. If the tasks are weakly correlated or some features are negatively correlated, sharing all knowledge often leads to negative knowledge transfer among. To overcome this issue, this paper proposes a Fisher sparse multi-task learning method. It can obtain a sparse sharing representation for each task. In such a way, tasks share features on a sparse subspace. Our method can ensure that the knowledge transferred among tasks is beneficial. Specifically, we first propose a sparse deep multi-task learning model, and then introduce Fisher sparse module into traditional deep multi-task learning to learn the sparse variables of task. By alternately updating the neural network parameters and sparse variables, a sparse sharing representation can be learned for each task. In addition, in order to reduce the computational overhead, an heuristic method is used to estimate the Fisher information of neural network parameters. Experimental results show that, comparing with other methods, our proposed method can improve the performance for all tasks, and has high sparsity in multi-task learning.

----

## [1884] A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning

**Authors**: *Yinmin Zhang, Jie Liu, Chuming Li, Yazhe Niu, Yaodong Yang, Yu Liu, Wanli Ouyang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29633](https://doi.org/10.1609/aaai.v38i15.29633)

**Abstract**:

Offline-to-online Reinforcement Learning (O2O RL) aims to improve the performance of offline pretrained policy using only a few online samples. Built on offline RL algorithms, most O2O methods focus on the balance between RL objective and pessimism, or the utilization of offline and online samples. In this paper, from a novel perspective, we systematically study the challenges that remain in O2O RL and identify that the reason behind the slow improvement of the performance and the instability of online finetuning lies in the inaccurate Q-value estimation inherited from offline pretraining. Specifically, we demonstrate that the estimation bias and the inaccurate rank of Q-value cause a misleading signal for the policy update, making the standard offline RL algorithms, such as CQL and TD3-BC, ineffective in the online finetuning. Based on this observation, we address the problem of Q-value estimation by two techniques: (1) perturbed value update and (2) increased frequency of Q-value updates. The first technique smooths out biased Q-value estimation with sharp peaks, preventing early-stage policy exploitation of sub-optimal actions. The second one alleviates the estimation bias inherited from offline pretraining by accelerating learning. Extensive experiments on the MuJoco and Adroit environments demonstrate that the proposed method, named SO2, significantly alleviates Q-value estimation issues, and consistently improves the performance against the state-of-the-art methods by up to 83.1%.

----

## [1885] Mitigating Label Bias in Machine Learning: Fairness through Confident Learning

**Authors**: *Yixuan Zhang, Boyu Li, Zenan Ling, Feng Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29634](https://doi.org/10.1609/aaai.v38i15.29634)

**Abstract**:

Discrimination can occur when the underlying unbiased labels are overwritten by an agent with potential bias, resulting in biased datasets that unfairly harm specific groups and cause classifiers to inherit these biases. In this paper, we demonstrate that despite only having access to the biased labels, it is possible to eliminate bias by filtering the fairest instances within the framework of confident learning. In the context of confident learning, low self-confidence usually indicates potential label errors; however, this is not always the case. Instances, particularly those from underrepresented groups, might exhibit low confidence scores for reasons other than labeling errors. To address this limitation, our approach employs truncation of the confidence score and extends the confidence interval of the probabilistic threshold. Additionally, we incorporate with co-teaching paradigm for providing a more robust and reliable selection of fair instances and effectively mitigating the adverse effects of biased labels. Through extensive experimentation and evaluation of various datasets, we demonstrate the efficacy of our approach in promoting fairness and reducing the impact of label bias in machine learning models.

----

## [1886] Enhancing Representation of Spiking Neural Networks via Similarity-Sensitive Contrastive Learning

**Authors**: *Yuhan Zhang, Xiaode Liu, Yuanpei Chen, Weihang Peng, Yufei Guo, Xuhui Huang, Zhe Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29635](https://doi.org/10.1609/aaai.v38i15.29635)

**Abstract**:

Spiking neural networks (SNNs) have attracted intensive attention as a promising energy-efficient alternative to conventional artificial neural networks (ANNs) recently, which could transmit information in form of binary spikes rather than continuous activations thus the  multiplication of activation and weight could be replaced by addition to save energy. However, the binary spike representation form will sacrifice the expression performance of SNNs and lead to accuracy degradation compared with ANNs. Considering improving feature representation is beneficial to training an accurate SNN model, this paper focuses on enhancing the feature representation of the SNN. To this end, we establish a similarity-sensitive contrastive learning framework, where SNN could capture significantly more information from its ANN counterpart to improve representation by Mutual Information (MI) maximization with layer-wise sensitivity to similarity.  In specific, it enriches the SNN’s feature representation by pulling the positive pairs of SNN's and ANN's feature representation of each layer from the same input samples closer together while pushing the negative pairs from different samples further apart. Experimental results show that our method consistently outperforms the current state-of-the-art algorithms on both popular non-spiking static and neuromorphic datasets.

----

## [1887] Cached Transformers: Improving Transformers with Differentiable Memory Cachde

**Authors**: *Zhaoyang Zhang, Wenqi Shao, Yixiao Ge, Xiaogang Wang, Jinwei Gu, Ping Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29636](https://doi.org/10.1609/aaai.v38i15.29636)

**Abstract**:

This work introduces a new Transformer model called Cached Transformer, which uses Gated Recurrent Cached (GRC) attention to extend the self-attention mechanism with a differentiable memory cache of tokens. GRC attention enables attending to both past and current tokens, increasing the receptive field of attention and allowing for exploring long-range dependencies. By utilizing a recurrent gating unit to continuously update the cache, our model achieves significant advancements in \textbf{six} language and vision tasks, including language modeling, machine translation, ListOPs, image classification, object detection, and instance segmentation. Furthermore, our approach surpasses previous memory-based techniques in tasks such as language modeling and displays the ability to be applied to a broader range of situations.

----

## [1888] An Implicit Trust Region Approach to Behavior Regularized Offline Reinforcement Learning

**Authors**: *Zhe Zhang, Xiaoyang Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29637](https://doi.org/10.1609/aaai.v38i15.29637)

**Abstract**:

We revisit behavior regularization, a popular approach to mitigate the extrapolation error in offline reinforcement learning (RL), showing that current behavior regularization may suffer from unstable learning and hinder policy improvement. Motivated by this, a novel reward shaping-based behavior regularization method is proposed, where the log-probability ratio between the learned policy and the behavior policy is monitored during learning. We show that this is equivalent to an implicit but computationally lightweight trust region mechanism, which is beneficial to mitigate the influence of estimation errors of the value function, leading to more stable performance improvement. Empirical results on the popular D4RL benchmark verify the effectiveness of the presented method with promising performance compared with some state-of-the-art offline RL algorithms.

----

## [1889] Towards a Theoretical Understanding of Why Local Search Works for Clustering with Fair-Center Representation

**Authors**: *Zhen Zhang, Junfeng Yang, Limei Liu, Xuesong Xu, Guozhen Rong, Qilong Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29638](https://doi.org/10.1609/aaai.v38i15.29638)

**Abstract**:

The representative k-median problem generalizes the classical clustering formulations in that it partitions the data points into several disjoint demographic groups and poses a lower-bound constraint on the number of opened facilities from each group, such that all the groups are fairly represented by the opened facilities. Due to its simplicity, the local-search heuristic that optimizes an initial solution by iteratively swapping at most a constant number of closed facilities for the same number of opened ones (denoted by the O(1)-swap heuristic) has been frequently used in the representative k-median problem. Unfortunately, despite its good performance exhibited in experiments, whether the O(1)-swap heuristic has provable approximation guarantees for the case where the number of groups is more than 2 remains an open question for a long time. As an answer to this question, we show that the O(1)-swap heuristic
(1) is guaranteed to yield a constant-factor approximation solution if the number of groups is a constant, and
(2) has an unbounded approximation ratio otherwise.
Our main technical contribution is a new approach for theoretically analyzing local-search heuristics, which derives the approximation ratio of the O(1)-swap heuristic via linearly combining the increased clustering costs induced by a set of  hierarchically organized swaps.

----

## [1890] Symmetric Self-Paced Learning for Domain Generalization

**Authors**: *Di Zhao, Yun Sing Koh, Gillian Dobbie, Hongsheng Hu, Philippe Fournier-Viger*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29639](https://doi.org/10.1609/aaai.v38i15.29639)

**Abstract**:

Deep learning methods often suffer performance degradation due to domain shift, where discrepancies exist between training and testing data distributions.
Domain generalization mitigates this problem by leveraging information from multiple source domains to enhance model generalization capabilities for unseen domains.
However, existing domain generalization methods typically present examples to the model in a random manner, overlooking the potential benefits of structured data presentation.
To bridge this gap, we propose a novel learning strategy, Symmetric Self-Paced Learning (SSPL), for domain generalization.
SSPL consists of a Symmetric Self-Paced training scheduler and a Gradient-based Difficulty Measure (GDM).
Specifically, the proposed training scheduler initially focuses on easy examples, gradually shifting emphasis to harder examples as training progresses.
GDM dynamically evaluates example difficulty through the gradient magnitude with respect to the example itself.
Experiments across five popular benchmark datasets demonstrate the effectiveness of the proposed learning strategy.

----

## [1891] Dynamic Reactive Spiking Graph Neural Network

**Authors**: *Han Zhao, Xu Yang, Cheng Deng, Junchi Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29640](https://doi.org/10.1609/aaai.v38i15.29640)

**Abstract**:

Spiking Graph Neural Networks are emerging tools for analyzing graph data along with low energy consumption and certain biological fidelity. Existing methods directly integrate same-reactive spiking neurons into graph neural networks for processing propagated graphs. However, such same-reactive neurons are not biological-functionality enough compared to the brain's dynamic-reactive ones, limiting the model's expression. Meanwhile, insufficient long-range neighbor information can be excavated with the few-step propagated graph, restricting discrimination of graph spiking embeddings. Inspired by the dynamic cognition in the brain, we propose a Dynamic Reactive Spiking Graph Neural Network that can enhance model's expressive ability in higher biological fidelity. Specifically, we design dynamic reactive spiking neurons to process spiking graph inputs, which have unique optimizable thresholds to spontaneously explore dynamic reactive states between neurons. Moreover, discriminative graph positional spikes are learned and integrated adaptively into spiking outputs through our neurons, thereby exploring long-range neighbors more thoroughly. Finally, with the dynamic reactive mechanism and learnable positional integration, we can obtain a powerful and highly bio-fidelity model with low energy consumption. Experiments on various domain-related datasets can demonstrate the effectiveness of our model. Our code is available at https://github.com/hzhao98/DRSGNN.

----

## [1892] Learning Visual Abstract Reasoning through Dual-Stream Networks

**Authors**: *Kai Zhao, Chang Xu, Bailu Si*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29641](https://doi.org/10.1609/aaai.v38i15.29641)

**Abstract**:

Visual abstract reasoning tasks present challenges for deep neural networks, exposing limitations in their capabilities. In this work, we present a neural network model that addresses the challenges posed by Raven’s Progressive Matrices (RPM). Inspired by the two-stream hypothesis of visual processing, we introduce the Dual-stream Reasoning Network (DRNet), which utilizes two parallel branches to capture image features. On top of the two streams, a reasoning module first learns to merge the high-level features of the same image. Then, it employs a rule extractor to handle combinations involving the eight context images and each candidate image, extracting discrete abstract rules and utilizing an multilayer perceptron (MLP) to make predictions. Empirical results demonstrate that the proposed DRNet achieves state-of-the-art average performance across multiple RPM benchmarks. Furthermore, DRNet demonstrates robust generalization capabilities, even extending to various out-of-distribution scenarios. The dual streams within DRNet serve distinct functions by addressing local or spatial information. They are then integrated into the reasoning module, leveraging abstract rules to facilitate the execution of visual reasoning tasks. These findings indicate that the dual-stream architecture could play a crucial role in visual abstract reasoning.

----

## [1893] Robust Visual Recognition with Class-Imbalanced Open-World Noisy Data

**Authors**: *Na Zhao, Gim Hee Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29642](https://doi.org/10.1609/aaai.v38i15.29642)

**Abstract**:

Learning from open-world noisy data, where both closed-set and open-set noise co-exist in the dataset, is a realistic but underexplored setting. Only recently, several efforts have been initialized to tackle this problem. However, these works assume the classes are balanced when dealing with open-world noisy data. This assumption often violates the nature of real-world large-scale datasets, where the label distributions are generally long-tailed, i.e. class-imbalanced. In this paper, we study the problem of robust visual recognition with class-imbalanced open-world noisy data. We propose a probabilistic graphical model-based approach: iMRF to achieve label noise correction that is robust to class imbalance via an efficient iterative inference of a Markov Random Field (MRF) in each training mini-batch. Furthermore, we design an agreement-based thresholding strategy to adaptively collect clean samples from all classes that includes corrected closed-set noisy samples while rejecting open-set noisy samples. We also introduce a noise-aware balanced cross-entropy loss to explicitly eliminate the bias caused by class-imbalanced data. Extensive experiments on several benchmark datasets including synthetic and real-world noisy datasets demonstrate the superior performance robustness of our method over existing methods. Our code is available at https://github.com/Na-Z/LIOND.

----

## [1894] From GARCH to Neural Network for Volatility Forecast

**Authors**: *Pengfei Zhao, Haoren Zhu, Wilfred Siu Hung Ng, Dik Lun Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29643](https://doi.org/10.1609/aaai.v38i15.29643)

**Abstract**:

Volatility, as a measure of uncertainty, plays a crucial role in numerous financial activities such as risk management. The Econometrics and Machine Learning communities have developed two distinct approaches for financial volatility forecasting: the stochastic approach and the neural network (NN) approach. Despite their individual strengths, these methodologies have conventionally evolved in separate research trajectories with little interaction between them. This study endeavors to bridge this gap by establishing an equivalence relationship between models of the GARCH family and their corresponding NN counterparts. With the equivalence relationship established, we introduce an innovative approach, named GARCH-NN, for constructing NN-based volatility models. It obtains the NN counterparts of GARCH models and integrates them as components into an established NN architecture, thereby seamlessly infusing volatility stylized facts (SFs) inherent in the GARCH models into the neural network. We develop the GARCH-LSTM model to showcase the power of GARCH-NN approach. Experiment results validate that amalgamating the NN counterparts of the GARCH family models into established NN models leads to enhanced outcomes compared to employing the stochastic and NN models in isolation.

----

## [1895] Robust Nonparametric Regression under Poisoning Attack

**Authors**: *Puning Zhao, Zhiguo Wan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29644](https://doi.org/10.1609/aaai.v38i15.29644)

**Abstract**:

This paper studies robust nonparametric regression, in which an adversarial attacker can modify the values of up to q samples from a training dataset of size N. Our initial solution is an M-estimator based on Huber loss minimization. Compared with simple kernel regression, i.e. the Nadaraya-Watson estimator, this method can significantly weaken the impact of malicious samples on the regression performance. We provide the convergence rate as well as the corresponding minimax lower bound. The result shows that, with proper bandwidth selection, supremum error is minimax optimal. The L2 error is optimal with relatively small q, but is suboptimal with larger q. The reason is that this estimator is vulnerable if there are many attacked samples concentrating in a small region. To address this issue, we propose a correction method by projecting the initial estimate to the space of Lipschitz functions. The final estimate is nearly minimax optimal for arbitrary q, up to a logarithmic factor.

----

## [1896] Embedded Feature Selection on Graph-Based Multi-View Clustering

**Authors**: *Wenhui Zhao, Guangfei Li, Haizhou Yang, Quanxue Gao, Qianqian Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29645](https://doi.org/10.1609/aaai.v38i15.29645)

**Abstract**:

Recently, anchor graph-based multi-view clustering has been proven to be highly efficient for large-scale data processing. However, most existing anchor graph-based clustering methods necessitate post-processing to obtain clustering labels and are unable to effectively utilize the information within anchor graphs. To solve these problems, we propose an Embedded Feature Selection on Graph-Based Multi-View Clustering (EFSGMC) approach to improve the clustering performance. Our method decomposes anchor graphs, taking advantage of memory efficiency, to obtain clustering labels in a single step without the need for post-processing. Furthermore, we introduce the l2,p-norm for graph-based feature selection, which selects the most relevant data for efficient graph factorization. Lastly, we employ the tensor Schatten p-norm as a tensor rank approximation function to capture the complementary information between different views, ensuring similarity between cluster assignment matrices. Experimental results on five real-world datasets demonstrate that our proposed method outperforms state-of-the-art approaches.

----

## [1897] Domain Invariant Learning for Gaussian Processes and Bayesian Exploration

**Authors**: *Xilong Zhao, Siyuan Bian, Yaoyun Zhang, Yuliang Zhang, Qinying Gu, Xinbing Wang, Chenghu Zhou, Nanyang Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29646](https://doi.org/10.1609/aaai.v38i15.29646)

**Abstract**:

Out-of-distribution (OOD) generalization has long been a challenging problem that remains largely unsolved. Gaussian processes (GP), as popular probabilistic model classes, especially in the small data regime, presume strong OOD generalization abilities. Surprisingly, their OOD generalization abilities have been under-explored before compared with other lines of GP research. In this paper, we identify that GP is not free from the problem and propose a domain invariant learning algorithm for Gaussian processes (DIL-GP) with a min-max optimization on the likelihood. DIL-GP discovers the heterogeneity in the data and forces invariance across partitioned subsets of data. We further extend the DIL-GP to improve Bayesian optimization's adaptability on changing environments. Numerical experiments demonstrate the superiority of DIL-GP for predictions on several synthetic and real-world datasets. We further demonstrate the effectiveness of the DIL-GP Bayesian optimization method on a PID parameters tuning experiment for a quadrotor. The full version and source code are available at: https://github.com/Billzxl/DIL-GP.

----

## [1898] CcDPM: A Continuous Conditional Diffusion Probabilistic Model for Inverse Design

**Authors**: *Yanxuan Zhao, Peng Zhang, Guopeng Sun, Zhigong Yang, Jianqiang Chen, Yueqing Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29647](https://doi.org/10.1609/aaai.v38i15.29647)

**Abstract**:

Engineering design methods aim to generate new designs that meet desired performance requirements. Past work has directly introduced conditional Generative Adversarial Networks (cGANs) into this field and achieved promising results in single-point design problems(one performance requirement under one working condition). However, these methods assume that the performance requirements are distributed in categorical space, which is not reasonable in these scenarios. Although Continuous conditional GANs (CcGANs) introduce Vicinal Risk Minimization (VRM) to reduce the performance loss caused by this assumption, they still face the following challenges: 1) CcGANs can not handle multi-point design problems (multiple performance requirements under multiple working conditions). 2) Their training process is time-consuming due to the high computational complexity of the vicinal loss. To address these issues, A Continuous conditional Diffusion Probabilistic Model (CcDPM) is proposed, which the first time introduces the diffusion model into the engineering design area and VRM into the diffusion model. CcDPM adopts a novel sampling method called multi-point design sampling to deal with multi-point design problems. Moreover, the k-d tree is used in the training process of CcDPM to shorten the calculation time of vicinal loss and speed up the training process by 2-300 times in our experiments. Experiments on a synthetic problem and three real-world design problems demonstrate that CcDPM outperforms the state-of-the-art GAN models.

----

## [1899] A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks

**Authors**: *Zhe Zhao, Pengkun Wang, Haibin Wen, Yudong Zhang, Zhengyang Zhou, Yang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29648](https://doi.org/10.1609/aaai.v38i15.29648)

**Abstract**:

Graph neural networks (GNNs) have achieved state-of-the-art results on many graph representation learning tasks by exploiting statistical correlations. However, numerous observations have shown that such correlations may not reflect the true causal mechanisms underlying the data and thus may hamper the ability of the model to generalize beyond the observed distribution. To address this problem, we propose an Information-based Causal Learning (ICL) framework that combines information theory and causality to analyze and improve graph representation learning to transform information relevance to causal dependence. Specifically, we first introduce a multi-objective mutual information optimization objective derived from information-theoretic analysis and causal learning principles to simultaneously extract invariant and interpretable causal information and reduce reliance on non-causal information in correlations. To optimize this multi-objective objective, we enable a causal disentanglement layer that effectively decouples the causal and non-causal information in the graph representations. Moreover, due to the intractability of mutual information estimation, we derive variational bounds that enable us to transform the above objective into a tractable loss function. To balance the multiple information objectives and avoid optimization conflicts, we leverage multi-objective gradient descent to achieve a stable and efficient transformation from informational correlation to causal dependency. Our approach provides important insights into modulating the information flow in GNNs to enhance their reliability and generalization. Extensive experiments demonstrate that our approach significantly improves the robustness and interpretability of GNNs across different distribution shifts. Visual analysis demonstrates how our method converts informative dependencies in representations into causal dependencies.

----

## [1900] DCLP: Neural Architecture Predictor with Curriculum Contrastive Learning

**Authors**: *Shenghe Zheng, Hongzhi Wang, Tianyu Mu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29649](https://doi.org/10.1609/aaai.v38i15.29649)

**Abstract**:

Neural predictors have shown great potential in the evaluation process of neural architecture search (NAS). However, current predictor-based approaches overlook the fact that training a predictor necessitates a considerable number of trained neural networks as the labeled training set, which is costly to obtain. Therefore, the critical issue in utilizing predictors for NAS is to train a high-performance predictor using as few trained neural networks as possible. Although some methods attempt to address this problem through unsupervised learning, they often result in inaccurate predictions. We argue that the unsupervised tasks intended for the common graph data are too challenging for neural networks, causing unsupervised training to be susceptible to performance crashes in NAS. To address this issue, we propose a CurricuLum-guided Contrastive Learning framework for neural Predictor (DCLP). Our method simplifies the contrastive task by designing a novel curriculum to enhance the stability of unlabeled training data distribution during contrastive training. Specifically, we propose a scheduler that ranks the training data according to the contrastive difficulty of each data and then inputs them to the contrastive learner in order. This approach concentrates the training data distribution and makes contrastive training more efficient. By using our method, the contrastive learner incrementally learns feature representations via unsupervised data on a smooth learning curve, avoiding performance crashes that may occur with excessively variable training data distributions. We experimentally demonstrate that DCLP has high accuracy and efficiency compared with existing predictors, and shows promising potential to discover superior architectures in various search spaces when combined with search strategies. Our code is available at: https://github.com/Zhengsh123/DCLP.

----

## [1901] Confusing Pair Correction Based on Category Prototype for Domain Adaptation under Noisy Environments

**Authors**: *Churan Zhi, Junbao Zhuo, Shuhui Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29650](https://doi.org/10.1609/aaai.v38i15.29650)

**Abstract**:

In this paper, we address unsupervised domain adaptation under noisy environments, which is more challenging and practical than traditional domain adaptation. In this scenario, the model is prone to overfitting noisy labels, resulting in a more pronounced domain shift and a notable decline in the overall model performance. Previous methods employed prototype methods for domain adaptation on robust feature spaces. However, these approaches struggle to effectively classify classes with similar features under noisy environments. To address this issue, we propose a new method to detect and correct confusing class pair. We first divide classes into easy and hard classes based on the small loss criterion. We then leverage the top-2 predictions for each sample after aligning the source and target domain to find the confusing pair in the hard classes. We apply label correction to the noisy samples within the confusing pair. With the proposed label correction method, we can train our model with more accurate labels. Extensive experiments confirm the effectiveness of our method and demonstrate its favorable performance compared with existing state-of-the-art methods. Our codes are publicly available at https://github.com/Hehxcf/CPC/.

----

## [1902] Knowledge-Aware Parameter Coaching for Personalized Federated Learning

**Authors**: *Mingjian Zhi, Yuanguo Bi, Wenchao Xu, Haozhao Wang, Tianao Xiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29651](https://doi.org/10.1609/aaai.v38i15.29651)

**Abstract**:

Personalized Federated Learning (pFL) can effectively exploit the non-IID data from distributed clients by customizing personalized models. Existing pFL methods either simply take the local model as a whole for aggregation or require significant training overhead to induce the inter-client personalized weights, and thus clients cannot efficiently exploit the mutually relevant knowledge from each other. In this paper, we propose a knowledge-aware parameter coaching scheme where each client can swiftly and granularly refer to parameters of other clients to guide the local training, whereby accurate personalized client models can be efficiently produced without contradictory knowledge. Specifically, a novel regularizer is designed to conduct layer-wise parameters coaching via a relation cube, which is constructed based on the knowledge represented by the layered parameters among all clients. Then, we develop an optimization method to update the relation cube and the parameters of each client. It is theoretically demonstrated that the convergence of the proposed method can be guaranteed under both convex and non-convex settings. Extensive experiments are conducted over various datasets, which show that the proposed method can achieve better performance compared with the state-of-the-art baselines in terms of accuracy and convergence speed.

----

## [1903] No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning

**Authors**: *Dianyu Zhong, Yiqin Yang, Qianchuan Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29652](https://doi.org/10.1609/aaai.v38i15.29652)

**Abstract**:

The large action space is one fundamental obstacle to deploying Reinforcement Learning methods in the real world. The numerous redundant actions will cause the agents to make repeated or invalid attempts, even leading to task failure. Although current algorithms conduct some initial explorations for this issue, they either suffer from rule-based systems or depend on expert demonstrations, which significantly limits their applicability in many real-world settings. In this work, we examine the theoretical analysis of what action can be eliminated in policy optimization and propose a novel redundant action filtering mechanism. Unlike other works, our method constructs the similarity factor by estimating the distance between the state distributions, which requires no prior knowledge. In addition, we combine the modified inverse model to avoid extensive computation in high-dimensional state space. We reveal the underlying structure of action spaces and propose a simple yet efficient redundant action filtering mechanism named No Prior Mask (NPM) based on the above techniques. We show the superior performance of our method by conducting extensive experiments on high-dimensional, pixel-input, and stochastic problems with various action redundancy tasks. Our code is public online at https://github.com/zhongdy15/npm.

----

## [1904] PreRoutGNN for Timing Prediction with Order Preserving Partition: Global Circuit Pre-training, Local Delay Learning and Attentional Cell Modeling

**Authors**: *Ruizhe Zhong, Junjie Ye, Zhentao Tang, Shixiong Kai, Mingxuan Yuan, Jianye Hao, Junchi Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29653](https://doi.org/10.1609/aaai.v38i15.29653)

**Abstract**:

Pre-routing timing prediction has been recently studied for evaluating the quality of a candidate cell placement in chip design. It involves directly estimating the timing metrics for both pin-level (slack, slew) and edge-level (net delay, cell delay), without time-consuming routing. However, it often suffers from signal decay and error accumulation due to the long timing paths in large-scale industrial circuits. To address these challenges, we propose a two-stage approach. First, we propose global circuit training to pre-train a graph auto-encoder that learns the global graph embedding from circuit netlist. Second, we use a novel node updating scheme for message passing on GCN, following the topological sorting sequence of the learned graph embedding and circuit graph. This scheme residually models the local time delay between two adjacent pins in the updating sequence, and extracts the lookup table information inside each cell via a new attention mechanism. To handle large-scale circuits efficiently, we introduce an order preserving partition scheme that reduces memory consumption while maintaining the topological dependencies. Experiments on 21 real world circuits achieve a new SOTA R2 of 0.93 for slack prediction, which is significantly surpasses 0.59 by previous SOTA method. Code will be available at: https://github.com/Thinklab-SJTU/EDA-AI.

----

## [1905] Cycle Self-Refinement for Multi-Source Domain Adaptation

**Authors**: *Chaoyang Zhou, Zengmao Wang, Bo Du, Yong Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29654](https://doi.org/10.1609/aaai.v38i15.29654)

**Abstract**:

Multi-source domain adaptation (MSDA) aims to transfer knowledge from multiple source domains to the unlabeled target domain. In this paper, we propose a cycle self-refinement domain adaptation method, which progressively attempts to learn the dominant transferable knowledge in each source domain in a cycle manner. Specifically, several source-specific networks and a domain-ensemble network are adopted in the proposed method. The source-specific networks are adopted to provide the dominant transferable knowledge in each source domain for instance-level ensemble on predictions of the samples in target domain. Then these samples with high-confidence ensemble predictions are adopted to refine the domain-ensemble network. Meanwhile, to guide each source-specific network to learn more dominant transferable knowledge, we force the features of the target domain from the domain-ensemble network and the features of each source domain from the corresponding source-specific network to be aligned with their predictions from the corresponding networks. Thus the adaptation ability of source-specific networks and the domain-ensemble network can be improved progressively. Extensive experiments on Office-31, Office-Home and DomainNet show that the proposed method outperforms the state-of-the-art methods for most tasks.

----

## [1906] Explaining Generalization Power of a DNN Using Interactive Concepts

**Authors**: *Huilin Zhou, Hao Zhang, Huiqi Deng, Dongrui Liu, Wen Shen, Shih-Han Chan, Quanshi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29655](https://doi.org/10.1609/aaai.v38i15.29655)

**Abstract**:

This paper explains the generalization power of a deep neural network (DNN) from the perspective of interactions. Although there is no universally accepted definition of the concepts encoded by a DNN, the sparsity of interactions in a DNN has been proved, i.e., the output score of a DNN can be well explained by a small number of interactions between input variables. In this way, to some extent, we can consider such interactions as interactive concepts encoded by the DNN. Therefore, in this paper, we derive an analytic explanation of inconsistency of concepts of different complexities. This may shed new lights on using the generalization power of concepts to explain the generalization power of the entire DNN. Besides, we discover that the DNN with stronger generalization power usually learns simple concepts more quickly and encodes fewer complex concepts.  We also discover the detouring dynamics of learning complex concepts, which explains both the high learning difficulty and the low generalization power of complex concepts. The code will be released when the paper is accepted.

----

## [1907] Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition

**Authors**: *Qianrui Zhou, Hua Xu, Hao Li, Hanlei Zhang, Xiaohan Zhang, Yifan Wang, Kai Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29656](https://doi.org/10.1609/aaai.v38i15.29656)

**Abstract**:

Multimodal intent recognition aims to leverage diverse modalities such as expressions, body movements and tone of speech to comprehend user's intent, constituting a critical task for understanding human language and behavior in real-world multimodal scenarios. Nevertheless, the majority of existing methods ignore potential correlations among different modalities and own limitations in effectively learning semantic features from nonverbal modalities. In this paper, we introduce a token-level contrastive learning method with modality-aware prompting (TCL-MAP) to address the above challenges. To establish an optimal multimodal semantic environment for text modality, we develop a modality-aware prompting module (MAP), which effectively aligns and fuses features from text, video and audio modalities with similarity-based modality alignment and cross-modality attention mechanism. Based on the modality-aware prompt and ground truth labels, the proposed token-level contrastive learning framework (TCL) constructs augmented samples and employs NT-Xent loss on the label token. Specifically, TCL capitalizes on the optimal textual semantic insights derived from intent labels to guide the learning processes of other modalities in return. Extensive experiments show that our method achieves remarkable improvements compared to state-of-the-art methods. Additionally, ablation analyses demonstrate the superiority of the modality-aware prompt over the handcrafted prompt, which holds substantial significance for multimodal prompt learning. The codes are released at https://github.com/thuiar/TCL-MAP.

----

## [1908] On the Robustness of Neural-Enhanced Video Streaming against Adversarial Attacks

**Authors**: *Qihua Zhou, Jingcai Guo, Song Guo, Ruibin Li, Jie Zhang, Bingjie Wang, Zhenda Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29657](https://doi.org/10.1609/aaai.v38i15.29657)

**Abstract**:

The explosive growth of video traffic on today's Internet promotes the rise of Neural-enhanced Video Streaming (NeVS), which effectively improves the rate-distortion trade-off by employing a cheap neural super-resolution model for quality enhancement on the receiver side. Missing by existing work, we reveal that the NeVS pipeline may suffer from a practical threat, where the crucial codec component (i.e., encoder for compression and decoder for restoration) can trigger adversarial attacks in a man-in-the-middle manner to significantly destroy video recovery performance and finally incurs the malfunction of downstream video perception tasks. In this paper, we are the first attempt to inspect the vulnerability of NeVS and discover a novel adversarial attack, called codec hijacking, where the injected invisible perturbation conspires with the malicious encoding matrix by reorganizing the spatial-temporal bit allocation within the bitstream size budget. Such a zero-day vulnerability makes our attack hard to defend because there is no visual distortion on the recovered videos until the attack happens. More seriously, this attack can be extended to diverse enhancement models, thus exposing a wide range of video perception tasks under threat. Evaluation based on state-of-the-art video codec benchmark illustrates that our attack significantly degrades the recovery performance of NeVS over previous attack methods. The damaged video quality finally leads to obvious malfunction of downstream tasks with over 75% success rate. We hope to arouse public attention on codec hijacking and its defence.

----

## [1909] Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations

**Authors**: *Renzhe Zhou, Chenxiao Gao, Zongzhang Zhang, Yang Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29658](https://doi.org/10.1609/aaai.v38i15.29658)

**Abstract**:

Generalization and sample efficiency have been long-standing issues concerning reinforcement learning, and thus the field of Offline Meta-Reinforcement Learning (OMRL) has gained increasing attention due to its potential of solving a wide range of problems with static and limited offline data. Existing OMRL methods often assume sufficient training tasks and data coverage to apply contrastive learning to extract task representations. However, such assumptions are not applicable in several real-world applications and thus undermine the generalization ability of the representations. In this paper, we consider OMRL with two types of data limitations: limited training tasks and limited behavior diversity and propose a novel algorithm called GENTLE for learning generalizable task representations in the face of data limitations. GENTLE employs Task Auto-Encoder (TAE), which is an encoder-decoder architecture to extract the characteristics of the tasks. Unlike existing methods, TAE is optimized solely by reconstruction of the state transition and reward, which captures the generative structure of the task models and produces generalizable representations when training tasks are limited. To alleviate the effect of limited behavior diversity, we consistently construct pseudo-transitions to align the data distribution used to train TAE with the data distribution encountered during testing. Empirically, GENTLE significantly outperforms existing OMRL methods on both in-distribution tasks and out-of-distribution tasks across both the given-context protocol and the one-shot protocol.

----

## [1910] Federated Label-Noise Learning with Local Diversity Product Regularization

**Authors**: *Xiaochen Zhou, Xudong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29659](https://doi.org/10.1609/aaai.v38i15.29659)

**Abstract**:

Training data in federated learning (FL) frameworks can have label noise, since they must be stored and annotated on clients' devices. 
If trained over such corrupted data, the models learn the wrong knowledge of label noise, which highly degrades their performance.  
Although several FL schemes are designed to combat label noise, they suffer performance degradation when the clients' devices only have limited local training samples.
To this end, a new scheme called federated label-noise learning (FedLNL) is developed in this paper.
The key problem of FedLNL is how to estimate a noise transition matrix (NTM) accurately in the case of limited local training samples.
If a gradient-based update method is used to update the local NTM on each client's device, it can generate too large gradients for the local NTM, causing a high estimation error of the local NTM.
To tackle this issue, an alternating update method for the local NTM and the local classifier is designed in FedLNL, where the local NTM is updated by a Bayesian inference-based update method.
Such an alternating update method makes the loss function of existing NTM-based schemes not applicable to FedLNL.
To enable federated optimization of FedLNL, a new regularizer on the parameters of the classifier called local diversity product regularizer is designed for the loss function of FedLNL. 
The results show that FedLNL improves the test accuracy of a trained model by up to 25.98%, compared with the state-of-the-art FL schemes that tackle label-noise issues.

----

## [1911] Abstract and Explore: A Novel Behavioral Metric with Cyclic Dynamics in Reinforcement Learning

**Authors**: *Anjie Zhu, Peng-Fei Zhang, Ruihong Qiu, Zetao Zheng, Zi Huang, Jie Shao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29660](https://doi.org/10.1609/aaai.v38i15.29660)

**Abstract**:

Intrinsic motivation lies at the heart of the exploration of reinforcement learning, which is primarily driven by the agent's inherent satisfaction rather than external feedback from the environment. However, in recent more challenging procedurally-generated environments with high stochasticity and uninformative extrinsic rewards, we identify two significant issues of applying intrinsic motivation. (1) State representation collapse: In existing methods, the learned representations within intrinsic motivation have a high probability to neglect the distinction among different states and be distracted by the task-irrelevant information brought by the stochasticity. (2) Insufficient interrelation among dynamics: Unsuccessful guidance provided by the uninformative extrinsic reward makes the dynamics learning in intrinsic motivation less effective. In light of the above observations, a novel Behavioral metric with Cyclic Dynamics (BCD) is proposed, which considers both cumulative and immediate effects and facilitates the abstraction and exploration of the agent. For the behavioral metric, the successor feature is utilized to reveal the expected future rewards and alleviate the heavy reliance of previous methods on extrinsic rewards. Moreover, the latent variable and vector quantization techniques are employed to enable an accurate measurement of the transition function in a discrete and interpretable manner. In addition, cyclic dynamics is established to capture the interrelations between state and action, thereby providing a thorough awareness of environmental dynamics. Extensive experiments conducted on procedurally-generated environments demonstrate the state-of-the-art performance of our proposed BCD.

----

## [1912] Adaptive Meta-Learning Probabilistic Inference Framework for Long Sequence Prediction

**Authors**: *Jianping Zhu, Xin Guo, Yang Chen, Yao Yang, Wenbo Li, Bo Jin, Fei Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29661](https://doi.org/10.1609/aaai.v38i15.29661)

**Abstract**:

Long sequence prediction has broad and significant application value in fields such as finance, wind power, and weather. However, the complex long-term dependencies of long sequence data and the potential domain shift problems limit the effectiveness of traditional models in practical scenarios. To this end, we propose an Adaptive Meta-Learning Probabilistic Inference Framework (AMPIF) based on sequence decomposition, which can effectively enhance the long sequence prediction ability of various basic models. Specifically, first, we decouple complex sequences into seasonal and trend components through a frequency domain decomposition module. Then, we design an adaptive meta-learning task construction strategy, which divides the seasonal and trend components into different tasks through a clustering-matching approach. Finally, we design a dual-stream amortized network (ST-DAN) to capture shared information between seasonal-trend tasks and use the support set to generate task-specific parameters for rapid generalization learning on the query set. We conducted extensive experiments on six datasets, including wind power and finance scenarios, and the results show that our method significantly outperforms baseline methods in prediction accuracy, interpretability, and algorithm stability and can effectively enhance the long sequence prediction capabilities of base models. The source code is publicly available at https://github.com/Zhu-JP/AMPIF.

----

## [1913] Towards the Disappearing Truth: Fine-Grained Joint Causal Influences Learning with Hidden Variable-Driven Causal Hypergraphs in Time Series

**Authors**: *Kun Zhu, Chunhui Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29662](https://doi.org/10.1609/aaai.v38i15.29662)

**Abstract**:

Causal discovery under Granger causality framework has yielded widespread concerns in time series analysis task. Nevertheless, most previous methods are unaware of the underlying causality disappearing problem, that is, certain weak causalities are less focusable and may be lost during the modeling process, thus leading to biased causal conclusions. Therefore, we propose to introduce joint causal influences (i.e., causal influences from the union of multiple variables) as additional causal indication information to help identify weak causalities. Further, to break the limitation of existing methods that implicitly and coarsely model joint causal influences, we propose a novel hidden variable-driven causal hypergraph neural network to meticulously explore the locality and diversity of joint causal influences, and realize its explicit and fine-grained modeling. Specifically, we introduce hidden variables to construct a causal hypergraph for explicitly characterizing various fine-grained joint causal influences. Then, we customize a dual causal information transfer mechanism (encompassing a multi-level causal path and an information aggregation path) to realize the free diffusion and meticulous aggregation of joint causal influences and facilitate its adaptive learning. Finally, we design a multi-view collaborative optimization constraint to guarantee the characterization diversity of causal hypergraph and capture remarkable forecasting relationships (i.e., causalities). Experiments are conducted to demonstrate the superiority of the proposed model.

----

## [1914] Contrastive Balancing Representation Learning for Heterogeneous Dose-Response Curves Estimation

**Authors**: *Minqin Zhu, Anpeng Wu, Haoxuan Li, Ruoxuan Xiong, Bo Li, Xiaoqing Yang, Xuan Qin, Peng Zhen, Jiecheng Guo, Fei Wu, Kun Kuang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29663](https://doi.org/10.1609/aaai.v38i15.29663)

**Abstract**:

Estimating the individuals' potential response to varying treatment doses is crucial for decision-making in areas such as precision medicine and management science. Most recent studies predict counterfactual outcomes by learning a covariate representation that is independent of the treatment variable. However, such independence constraints neglect much of the covariate information that is useful for counterfactual prediction, especially when the treatment variables are continuous. To tackle the above issue, in this paper, we first theoretically demonstrate the importance of the balancing and prognostic representations for unbiased estimation of the heterogeneous dose-response curves, that is, the learned representations are constrained to satisfy the conditional independence between the covariates and both of the treatment variables and the potential responses. Based on this, we propose a novel Contrastive balancing Representation learning Network using a partial distance measure, called CRNet, for estimating the heterogeneous dose-response curves without losing the continuity of treatments. Extensive experiments are conducted on synthetic and real-world datasets demonstrating that our proposal significantly outperforms previous methods.

----

## [1915] Every Node Is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering

**Authors**: *Pengfei Zhu, Qian Wang, Yu Wang, Jialu Li, Qinghua Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29664](https://doi.org/10.1609/aaai.v38i15.29664)

**Abstract**:

Attributed graph clustering is an unsupervised task that partitions nodes into different groups. Self-supervised learning (SSL) shows great potential in handling this task, and some recent studies simultaneously learn multiple SSL tasks to further boost performance. Currently, different SSL tasks are assigned the same set of weights for all graph nodes. However, we observe that some graph nodes whose neighbors are in different groups require significantly different emphases on SSL tasks. In this paper, we propose to dynamically learn the weights of SSL tasks for different nodes and fuse the embeddings learned from different SSL tasks to boost performance. We design an innovative graph clustering approach, namely Dynamically Fusing Self-Supervised Learning (DyFSS). Specifically, DyFSS fuses features extracted from diverse SSL tasks using distinct weights derived from a gating network. To effectively learn the gating network, we design a dual-level self-supervised strategy that incorporates pseudo labels and the graph structure. Extensive experiments on five datasets show that DyFSS outperforms the state-of-the-art multi-task SSL methods by up to 8.66% on the accuracy metric. The code of DyFSS is available at: https://github.com/q086/DyFSS.

----

## [1916] Double Buffers CEM-TD3: More Efficient Evolution and Richer Exploration

**Authors**: *Sheng Zhu, Chun Shen, Shuai Lü, Junhong Wu, Daolong An*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29665](https://doi.org/10.1609/aaai.v38i15.29665)

**Abstract**:

CEM-TD3 is a combination scheme using the simple cross-entropy method (CEM) and Twin Delayed Deep Deterministic policy gradient (TD3), and it achieves a satisfactory trade-off between performance and sample efficiency. However, we find that CEM-TD3 cannot fully address the low efficiency of policy search caused by CEM, and the policy gradient learning introduced by TD3 will weaken the diversity of individuals in the population. In this paper, we propose Double Buffers CEM-TD3 (DBCEM-TD3) that optimizes both CEM and TD3. For CEM, DBCEM-TD3 maintains an actor buffer to store the population required for evolution. In each iteration, it only needs to generate a small number of actors to replace the poor actors in the policy buffer to achieve more efficient evolution. The fitness of individuals in the actor buffer decreases exponentially with time, which can avoid premature convergence of the mean actor. For TD3, DBCEM-TD3 maintains a critic buffer with the same number of critics as the number of actors generated in each iteration, and each critic is trained independently by sampling from the shared replay buffer. In each iteration, each newly generated actor uses different critics to guide learning. This ensures more diverse behaviors among the learned actors, enabling richer experiences to be collected during the evaluation phase. We conduct experimental evaluations on five continuous control tasks provided by OpenAI Gym. DBCEM-TD3 outperforms CEM-TD3, TD3, and other classic off-policy reinforcement learning algorithms in terms of performance and sample efficiency.

----

## [1917] Decoding Global Preferences: Temporal and Cooperative Dependency Modeling in Multi-Agent Preference-Based Reinforcement Learning

**Authors**: *Tianchen Zhu, Yue Qiu, Haoyi Zhou, Jianxin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29666](https://doi.org/10.1609/aaai.v38i15.29666)

**Abstract**:

Designing accurate reward functions for reinforcement learning (RL) has long been challenging. Preference-based RL (PbRL) offers a promising approach by using human preferences
to train agents, eliminating the need for manual reward design. While successful in single-agent tasks, extending PbRL to complex multi-agent scenarios is nontrivial. Existing PbRL methods lack the capacity to comprehensively capture both temporal and cooperative aspects, leading to inadequate reward functions. This work introduces an advanced multi-agent preference learning framework that effectively addresses these limitations. Based on a cascading Transformer architecture, our approach captures both temporal and cooperative dependencies, alleviating issues related to reward uniformity and intricate interactions among agents. Experimental results demonstrate substantial performance improvements in multi-agent cooperative tasks, and the reconstructed reward function closely resembles expert-defined reward functions. The source code is available at https://github.com/catezi/MAPT.

----

## [1918] Detection and Defense of Unlearnable Examples

**Authors**: *Yifan Zhu, Lijia Yu, Xiao-Shan Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29667](https://doi.org/10.1609/aaai.v38i15.29667)

**Abstract**:

Privacy preserving has become increasingly critical with the emergence of social media. Unlearnable examples have been proposed to avoid leaking personal information on the Internet by degrading the generalization abilities of deep learning models. However, our study reveals that unlearnable examples are easily detectable. We provide theoretical results on linear separability of certain unlearnable poisoned dataset and simple network-based detection methods that can identify all existing unlearnable examples, as demonstrated by extensive experiments. Detectability of unlearnable examples with simple networks motivates us to design a novel defense method. We propose using stronger data augmentations coupled with adversarial noises generated by simple networks, to degrade the detectability and thus provide effective defense against unlearnable examples with a lower cost. Adversarial training with large budgets is a widely-used defense method on unlearnable examples. We establish quantitative criteria between the poison and adversarial budgets, which determine the existence of robust unlearnable examples or the failure of the adversarial defense.

----

## [1919] Robust Node Classification on Graph Data with Graph and Label Noise

**Authors**: *Yonghua Zhu, Lei Feng, Zhenyun Deng, Yang Chen, Robert Amor, Michael Witbrock*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29668](https://doi.org/10.1609/aaai.v38i15.29668)

**Abstract**:

Current research for node classification focuses on dealing with either graph noise or label noise, but few studies consider both of them. In this paper, we propose a new robust node classification method to simultaneously deal with graph noise and label noise. To do this, we design a graph contrastive loss to conduct local graph learning and employ self-attention to conduct global graph learning. They enable us to improve the expressiveness of node representation by using comprehensive information among nodes. We also utilize pseudo graphs and pseudo labels to deal with graph noise and label noise, respectively. Furthermore, We numerically validate the superiority of our method in terms of robust node classification compared with all comparison methods.

----

## [1920] MFABA: A More Faithful and Accelerated Boundary-Based Attribution Method for Deep Neural Networks

**Authors**: *Zhiyu Zhu, Huaming Chen, Jiayu Zhang, Xinyi Wang, Zhibo Jin, Minhui Xue, Dongxiao Zhu, Kim-Kwang Raymond Choo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29669](https://doi.org/10.1609/aaai.v38i15.29669)

**Abstract**:

To better understand the output of deep neural networks (DNN), attribution based methods have been an important approach for model interpretability, which assign a score for each input dimension to indicate its importance towards the model outcome. Notably, the attribution methods use the ax- ioms of sensitivity and implementation invariance to ensure the validity and reliability of attribution results. Yet, the ex- isting attribution methods present challenges for effective in- terpretation and efficient computation. In this work, we in- troduce MFABA, an attribution algorithm that adheres to ax- ioms, as a novel method for interpreting DNN. Addition- ally, we provide the theoretical proof and in-depth analy- sis for MFABA algorithm, and conduct a large scale exper- iment. The results demonstrate its superiority by achieving over 101.5142 times faster speed than the state-of-the-art at- tribution algorithms. The effectiveness of MFABA is thor- oughly evaluated through the statistical analysis in compar- ison to other methods, and the full implementation package is open-source at: https://github.com/LMBTough/MFABA.

----

## [1921] DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning

**Authors**: *Huiping Zhuang, Run He, Kai Tong, Ziqian Zeng, Cen Chen, Zhiping Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29670](https://doi.org/10.1609/aaai.v38i15.29670)

**Abstract**:

Class-incremental learning (CIL) under an exemplar-free constraint has presented a significant challenge. Existing methods adhering to this constraint are prone to catastrophic forgetting, far more so than replay-based techniques that retain access to past samples. In this paper,  to solve the  exemplar-free CIL problem, we propose a Dual-Stream Analytic Learning (DS-AL) approach. The DS-AL contains a main stream offering an analytical (i.e., closed-form) linear solution, and a compensation stream improving the inherent under-fitting limitation due to adopting linear mapping. The main stream redefines the CIL problem into a Concatenated Recursive Least Squares (C-RLS) task, allowing an equivalence between the CIL and its joint-learning counterpart. The compensation stream is governed by a Dual-Activation Compensation (DAC) module. This module re-activates the embedding with a different activation function from the main stream one, and seeks fitting compensation by projecting the embedding to the null space of the main stream's linear mapping. Empirical results demonstrate that the DS-AL, despite being an exemplar-free technique, delivers performance comparable with or better than that of replay-based methods across various datasets, including CIFAR-100, ImageNet-100 and ImageNet-Full. Additionally, the C-RLS' equivalent property allows the DS-AL to execute CIL in a phase-invariant manner. This is evidenced by a never-before-seen 500-phase CIL ImageNet task, which performs on a level identical to a 5-phase one. Our codes are available at https://github.com/ZHUANGHP/Analytic-continual-learning.

----

## [1922] Patch-Aware Sample Selection for Efficient Masked Image Modeling

**Authors**: *Zhengyang Zhuge, Jiaxing Wang, Yong Li, Yongjun Bao, Peisong Wang, Jian Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29671](https://doi.org/10.1609/aaai.v38i15.29671)

**Abstract**:

Nowadays sample selection is drawing increasing attention. By extracting and training only on the most informative subset, sample selection can effectively reduce the training cost. Although sample selection is effective in conventional supervised learning, applying it to Masked Image Modeling (MIM) still poses challenges due to the gap between sample-level selection and patch-level pre-training. In this paper, we inspect the sample selection in MIM pre-training and find the basic selection suffers from performance degradation. We attribute this degradation primarily to 2 factors: the random mask strategy and the simple averaging function. We then propose Patch-Aware Sample Selection (PASS), including a low-cost Dynamic Trained Mask Predictor (DTMP) and Weighted Selection Score (WSS). DTMP consistently masks the informative patches in samples, ensuring a relatively accurate representation of selection score. WSS enhances the selection score using patch-level disparity. Extensive experiments show the effectiveness of PASS in selecting the most informative subset and accelerating pretraining. PASS exhibits superior performance across various datasets, MIM methods, and downstream tasks. Particularly, PASS improves MAE by 0.7% on ImageNet-1K while utilizing only 37% data budget and achieves ~1.7x speedup.

----

## [1923] Dirichlet-Based Prediction Calibration for Learning with Noisy Labels

**Authors**: *Chen-Chen Zong, Ye-Wen Wang, Ming-Kun Xie, Sheng-Jun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29672](https://doi.org/10.1609/aaai.v38i15.29672)

**Abstract**:

Learning with noisy labels can significantly hinder the generalization performance of deep neural networks (DNNs). Existing approaches address this issue through loss correction or example selection methods. However, these methods often rely on the model's predictions obtained from the softmax function, which can be over-confident and unreliable. In this study, we identify the translation invariance of the softmax function as the underlying cause of this problem and propose the \textit{Dirichlet-based Prediction Calibration} (DPC) method as a solution. Our method introduces a calibrated softmax function that breaks the translation invariance by incorporating a suitable constant in the exponent term, enabling more reliable model predictions. To ensure stable model training, we leverage a Dirichlet distribution to assign probabilities to predicted labels and introduce a novel evidence deep learning (EDL) loss. The proposed loss function encourages positive and sufficiently large logits for the given label, while penalizing negative and small logits for other labels, leading to more distinct logits and facilitating better example selection based on a large-margin criterion. Through extensive experiments on diverse benchmark datasets, we demonstrate that DPC achieves state-of-the-art performance. The code is available at https://github.com/chenchenzong/DPC.

----

## [1924] Coverage-Guaranteed Prediction Sets for Out-of-Distribution Data

**Authors**: *Xin Zou, Weiwei Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29673](https://doi.org/10.1609/aaai.v38i15.29673)

**Abstract**:

Out-of-distribution (OOD) generalization has attracted increasing research attention in recent years, due to its promising experimental results in real-world applications. In this paper, we study the confidence set prediction problem in the OOD generalization setting. Split conformal prediction (SCP) is an efficient framework for handling the confidence set prediction problem. However, the validity of SCP requires the examples to be exchangeable, which is violated in the OOD setting. Empirically, we show that trivially applying SCP results in a failure to maintain the marginal coverage when the unseen target domain is different from the source domain. To address this issue, we develop a method for forming confident prediction sets in the OOD setting and theoretically prove the validity of our method. Finally, we conduct experiments on simulated data to empirically verify the correctness of our theory and the validity of our proposed method.

----

## [1925] Generalization Analysis of Machine Learning Algorithms via the Worst-Case Data-Generating Probability Measure

**Authors**: *Xinying Zou, Samir M. Perlaza, Iñaki Esnaola, Eitan Altman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29674](https://doi.org/10.1609/aaai.v38i15.29674)

**Abstract**:

In this paper, the worst-case probability measure over the data is introduced as a tool for characterizing the generalization capabilities of machine learning algorithms. More specifically, the worst-case probability measure is a Gibbs probability measure and the unique solution to the maximization of the expected loss under a relative entropy constraint with respect to a reference probability measure. Fundamental generalization metrics, such as the sensitivity of the expected loss, the sensitivity of the empirical risk, and the generalization gap are shown to have closed-form expressions involving the worst-case data-generating probability measure. Existing results for the Gibbs algorithm, such as characterizing the generalization gap as a sum of mutual information and lautum information, up to a constant factor, are recovered. A novel parallel is established between the worst-case data-generating probability measure and the Gibbs algorithm. Specifically, the Gibbs probability measure is identified as a fundamental commonality of the model space and the data space for machine learning algorithms.

----

## [1926] Probabilistic Neural Circuits

**Authors**: *Pedro Zuidberg Dos Martires*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i15.29675](https://doi.org/10.1609/aaai.v38i15.29675)

**Abstract**:

Probabilistic circuits (PCs) have gained prominence in recent years as a versatile framework for discussing probabilistic models that support tractable queries and are yet expressive enough to model complex probability distributions. Nevertheless, tractability comes at a cost: PCs are less expressive than neural networks. In this paper we introduce probabilistic neural circuits (PNCs), which strike a balance between PCs and neural nets in terms of tractability and expressive power. Theoretically, we show that PNCs can be interpreted as deep mixtures of Bayesian networks. Experimentally, we demonstrate that PNCs constitute powerful function approximators.

----

## [1927] Improved Anonymous Multi-Agent Path Finding Algorithm

**Authors**: *Zain Alabedeen Ali, Konstantin S. Yakovlev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29676](https://doi.org/10.1609/aaai.v38i16.29676)

**Abstract**:

We consider an Anonymous Multi-Agent Path-Finding (AMAPF) problem where the set of agents is confined to a graph, a set of goal vertices is given and each of these vertices has to be reached by some agent. The problem is to find an assignment of the goals to the agents as well as the collision-free paths, and we are interested in finding the solution with the optimal makespan. A well-established approach to solve this problem is to reduce it to a special type of a graph search problem, i.e. to the problem of finding a maximum flow on an auxiliary graph induced by the input one. The size of the former graph may be very large and the search on it may become a bottleneck. To this end, we suggest a specific search algorithm that leverages the idea of exploring the search space not through considering separate search states but rather bulks of them simultaneously. That is, we implicitly compress, store and expand bulks of the search states as single states, which results in high reduction in runtime and memory. Empirically, the resultant AMAPF solver demonstrates superior performance compared to the state-of-the-art competitor and is able to solve all publicly available MAPF instances from the well-known MovingAI benchmark in less than 30 seconds.

----

## [1928] Cautiously-Optimistic Knowledge Sharing for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Yanwen Ba, Xuan Liu, Xinning Chen, Hao Wang, Yang Xu, Kenli Li, Shigeng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29677](https://doi.org/10.1609/aaai.v38i16.29677)

**Abstract**:

While decentralized training is attractive in multi-agent reinforcement learning (MARL) for its excellent scalability and robustness, its inherent coordination challenges in collaborative tasks result in numerous interactions for agents to learn good policies. To alleviate this problem, action advising methods make experienced agents share their knowledge about what to do, while less experienced agents strictly follow the received advice. However, this method of sharing and utilizing knowledge may hinder the team's exploration of better states, as agents can be unduly influenced by suboptimal or even adverse advice, especially in the early stages of learning. Inspired by the fact that humans can learn not only from the success but also from the failure of others, this paper proposes a novel knowledge sharing framework called Cautiously-Optimistic kNowledge Sharing (CONS). CONS enables each agent to share both positive and negative knowledge and cautiously assimilate knowledge from others, thereby enhancing the efficiency of early-stage exploration and the agents' robustness to adverse advice. Moreover, considering the continuous improvement of policies, agents value negative knowledge more in the early stages of learning and shift their focus to positive knowledge in the later stages. Our framework can be easily integrated into existing Q-learning based methods without introducing additional training costs. We evaluate CONS in several challenging multi-agent tasks and find it excels in environments where optimal behavioral patterns are difficult to discover, surpassing the baselines in terms of convergence rate and final performance.

----

## [1929] Natural Strategic Ability in Stochastic Multi-Agent Systems

**Authors**: *Raphaël Berthon, Joost-Pieter Katoen, Munyque Mittelmann, Aniello Murano*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29678](https://doi.org/10.1609/aaai.v38i16.29678)

**Abstract**:

Strategies synthesized using formal methods can be complex and often require infinite memory, which does not correspond to the expected behavior when trying to model Multi-Agent Systems (MAS). To capture such behaviors, natural strategies are a recently proposed framework striking a balance between the ability of agents to strategize with memory and the complexity of the model-checking problem, but until now has been restricted to fully deterministic settings. For the first time, we consider the probabilistic temporal logics PATL and PATL∗ under natural strategies (NatPATL and NatPATL∗). As main result we show that, in stochastic MAS, NatPATL model-checking is NP-complete when the active coalition is restricted to deterministic strategies. We also give a 2NEXPTIME complexity result for NatPATL∗ with the same restriction. In the unrestricted case, we give an EXPSPACE complexity for NatPATL and 3EXPSPACE complexity for NatPATL*.

----

## [1930] On Alternating-Time Temporal Logic, Hyperproperties, and Strategy Sharing

**Authors**: *Raven Beutner, Bernd Finkbeiner*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29679](https://doi.org/10.1609/aaai.v38i16.29679)

**Abstract**:

Alternating-time temporal logic (ATL*) is a well-established framework for formal reasoning about multi-agent systems. 
However, while ATL* can reason about the strategic ability of agents (e.g., some coalition A can ensure that a goal is reached eventually), we cannot compare multiple strategic interactions, nor can we require multiple agents to follow the same strategy. 
For example, we cannot state that coalition A can reach a goal sooner (or more often) than some other coalition A'. 
In this paper, we propose HyperATL*_S, an extension of ATL* in which we can (1) compare the outcome of multiple strategic interactions w.r.t. a hyperproperty, i.e., a property that refers to multiple paths at the same time, and (2) enforce that some agents share the same strategy.
We show that HyperATL*_S is a rich specification language that captures important AI-related properties that were out of reach of existing logics. 
We prove that model checking of HyperATL*_S on concurrent game structures is decidable.
We implement our model-checking algorithm in a tool we call HyMASMC and evaluate it on a range of benchmarks.

----

## [1931] RGMComm: Return Gap Minimization via Discrete Communications in Multi-Agent Reinforcement Learning

**Authors**: *Jingdi Chen, Tian Lan, Carlee Joe-Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29680](https://doi.org/10.1609/aaai.v38i16.29680)

**Abstract**:

Communication is crucial for solving cooperative Multi-Agent Reinforcement Learning tasks in partially observable Markov Decision Processes. Existing works often rely on black-box methods to encode local information/features into messages shared with other agents, leading to the generation of continuous messages with high communication overhead and poor interpretability. Prior attempts at discrete communication methods generate one-hot vectors trained as part of agents' actions and use the Gumbel softmax operation for calculating message gradients, which are all heuristic designs that do not provide any quantitative guarantees on the expected return. 
This paper establishes an upper bound on the return gap between an ideal policy with full observability and an optimal partially observable policy with discrete communication. This result enables us to recast multi-agent communication into a novel online clustering problem over the local observations at each agent, with messages as cluster labels and the upper bound on the return gap as clustering loss. To minimize the return gap, we propose the Return-Gap-Minimization Communication (RGMComm) algorithm, which is a surprisingly simple design of discrete message generation functions and is integrated with reinforcement learning through the utilization of a novel Regularized Information Maximization loss function, which incorporates cosine-distance as the clustering metric. Evaluations show that RGMComm significantly outperforms state-of-the-art multi-agent communication baselines and can achieve nearly optimal returns with few-bit messages that are naturally interpretable.

----

## [1932] STAS: Spatial-Temporal Return Decomposition for Solving Sparse Rewards Problems in Multi-agent Reinforcement Learning

**Authors**: *Sirui Chen, Zhaowei Zhang, Yaodong Yang, Yali Du*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29681](https://doi.org/10.1609/aaai.v38i16.29681)

**Abstract**:

Centralized Training with Decentralized Execution (CTDE) has been proven to be an effective paradigm in cooperative multi-agent reinforcement learning (MARL). One of the major challenges is credit assignment, which aims to credit agents by their contributions. They lack the functionality to model complicated relations of the delayed global reward in the temporal dimension and suffer from inefficiencies. To tackle this, we introduce Spatial-Temporal Attention with Shapley (STAS), a novel method that learns credit assignment in both temporal and spatial dimensions. It first decomposes the global return back to each time step, then utilizes the Shapley Value to redistribute the individual payoff from the decomposed global reward. To mitigate the computational complexity of the Shapley Value, we introduce an approximation of marginal contribution and utilize Monte Carlo sampling to estimate it. We evaluate our method on an Alice & Bob example and MPE environments across different scenarios. Our results demonstrate that our method effectively assigns spatial-temporal credit, outperforming all state-of-the-art baselines.

----

## [1933] Learning Efficient and Robust Multi-Agent Communication via Graph Information Bottleneck

**Authors**: *Shifei Ding, Wei Du, Ling Ding, Lili Guo, Jian Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29682](https://doi.org/10.1609/aaai.v38i16.29682)

**Abstract**:

Efficient communication learning among agents has been shown crucial for cooperative multi-agent reinforcement learning (MARL), as it can promote the action coordination of agents and ultimately improve performance. Graph neural network (GNN) provide a general paradigm for communication learning, which consider agents and communication channels as nodes and edges in a graph, with the action selection corresponding to node labeling. Under such paradigm, an agent aggregates information from neighbor agents, which can reduce uncertainty in local decision-making and induce implicit action coordination. However, this communication paradigm is vulnerable to adversarial attacks and noise, and how to learn robust and efficient communication under perturbations has largely not been studied. To this end, this paper introduces a novel Multi-Agent communication mechanism via Graph Information bottleneck (MAGI), which can optimally balance the robustness and expressiveness of the message representation learned by agents. This communication mechanism is aim at learning the minimal sufficient message representation for an agent by maximizing the mutual information (MI) between the message representation and the selected action, and simultaneously constraining the MI between the message representation and the agent feature. Empirical results demonstrate that MAGI is more robust and efficient than state-of-the-art GNN-based MARL methods.

----

## [1934] Expressive Multi-Agent Communication via Identity-Aware Learning

**Authors**: *Wei Du, Shifei Ding, Lili Guo, Jian Zhang, Ling Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29683](https://doi.org/10.1609/aaai.v38i16.29683)

**Abstract**:

Information sharing through communication is essential for tackling complex multi-agent reinforcement learning tasks. Many existing multi-agent communication protocols can be viewed as instances of message passing graph neural networks (GNNs). However, due to the significantly limited expressive ability of the standard GNN method, the agent feature representations remain similar and indistinguishable even though the agents have different neighborhood structures. This further results in the homogenization of agent behaviors and reduces the capability to solve tasks effectively. In this paper, we propose a multi-agent communication protocol via identity-aware learning (IDEAL), which explicitly enhances the distinguishability of agent feature representations to break the diversity bottleneck. Specifically, IDEAL extends existing multi-agent communication protocols by inductively considering the agents' identities during the message passing process. To obtain expressive feature representations for a given agent, IDEAL first extracts the ego network centered around that agent and then performs multiple rounds of heterogeneous message passing, where different parameter sets are applied to the central agent and the other surrounding agents within the ego network. IDEAL fosters expressive communication between agents and generates distinguishable feature representations, which promotes action diversity and individuality emergence. Experimental results on various benchmarks demonstrate IDEAL can be flexibly integrated into various multi-agent communication methods and enhances the corresponding performance.

----

## [1935] Situation-Dependent Causal Influence-Based Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Xiao Du, Yutong Ye, Pengyu Zhang, Yaning Yang, Mingsong Chen, Ting Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29684](https://doi.org/10.1609/aaai.v38i16.29684)

**Abstract**:

Learning to collaborate has witnessed significant progress in multi-agent reinforcement learning (MARL). However, promoting coordination among agents and enhancing exploration capabilities remain challenges. In multi-agent environments, interactions between agents are limited in specific situations. Effective collaboration between agents thus requires a nuanced understanding of when and how agents' actions influence others.To this end, in this paper, we propose a novel MARL algorithm named Situation-Dependent Causal Influence-Based Cooperative Multi-agent Reinforcement Learning (SCIC), which incorporates a novel Intrinsic reward mechanism based on a new cooperation criterion measured by situation-dependent causal influence among agents.Our approach aims to detect inter-agent causal influences in specific situations based on the criterion using causal intervention and conditional mutual information. This effectively assists agents in exploring states that can positively impact other agents, thus promoting cooperation between agents.The resulting update links coordinated exploration and intrinsic reward distribution, which enhance overall collaboration and performance.Experimental results on various MARL benchmarks demonstrate the superiority of our method compared to state-of-the-art approaches.

----

## [1936] Learning Multi-Object Positional Relationships via Emergent Communication

**Authors**: *Yicheng Feng, Boshi An, Zongqing Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29685](https://doi.org/10.1609/aaai.v38i16.29685)

**Abstract**:

The study of emergent communication has been dedicated to interactive artificial intelligence. While existing work focuses on communication about single objects or complex image scenes, we argue that communicating relationships between multiple objects is important in more realistic tasks, but understudied. In this paper, we try to fill this gap and focus on emergent communication about positional relationships between two objects. We train agents in the referential game where observations contain two objects, and find that generalization is the major problem when the positional relationship is involved. The key factor affecting the generalization ability of the emergent language is the input variation between Speaker and Listener, which is realized by a random image generator in our work. Further, we find that the learned language can generalize well in a new multi-step MDP task where the positional relationship describes the goal, and performs better than raw-pixel images as well as pre-trained image features, verifying the strong generalization ability of discrete sequences. We also show that language transfer from the referential game performs better in the new task than learning language directly in this task, implying the potential benefits of pre-training in referential games. All in all, our experiments demonstrate the viability and merit of having agents learn to communicate positional relationships between multiple objects through emergent communication.

----

## [1937] Exact Algorithms and Lowerbounds for Multiagent Path Finding: Power of Treelike Topology

**Authors**: *Foivos Fioravantes, Dusan Knop, Jan Matyás Kristan, Nikolaos Melissinos, Michal Opler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29686](https://doi.org/10.1609/aaai.v38i16.29686)

**Abstract**:

In the Multiagent Path Finding (MAPF for short) problem, we focus on efficiently finding non-colliding paths for a set of k agents on a given graph G, where each agent seeks a path from its source vertex to a target.
An important measure of the quality of the solution is the length of the proposed schedule l, that is, the length of a longest path (including the waiting time).
In this work, we propose a systematic study under the parameterized complexity framework. The hardness results we provide align with many heuristics used for this problem, whose running time could potentially be improved based on our Fixed-Parameter Tractability (FPT) results.

We show that MAPF is W[1]-hard with respect to k (even if k is combined with the maximum degree of the input graph).
The problem remains NP-hard in planar graphs even if the maximum degree and the makespan l are fixed constants.
On the positive side, we show an FPT algorithm for k+l.

As we continue, the structure of G comes into play.
We give an FPT algorithm for parameter k plus the diameter of the graph G.
The MAPF problem is W[1]-hard for cliquewidth of G plus l while it is FPT for treewidth of G plus l.

----

## [1938] The Irrelevance of Influencers: Information Diffusion with Re-Activation and Immunity Lasts Exponentially Long on Social Network Models

**Authors**: *Tobias Friedrich, Andreas Göbel, Nicolas Klodt, Martin S. Krejca, Marcus Pappik*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29687](https://doi.org/10.1609/aaai.v38i16.29687)

**Abstract**:

Information diffusion models on networks are at the forefront of AI research. The dynamics of such models typically follow stochastic models from epidemiology, used to model not only infections but various phenomena, including the behavior of computer viruses and viral marketing campaigns. A core question in this setting is how to efficiently detect the most influential vertices in the host graph such that the infection survives the longest. In processes that incorporate re-infection of the vertices, such as the SIS process, theoretical studies identify parameter thresholds where the survival time of the process rapidly transitions from logarithmic to super-polynomial. These results contradict the intuition that the starting configuration is relevant, since the process will always either die out fast or survive almost indefinitely. A shortcoming of these results is that models incorporating short-term immunity (or creative advertisement fatigue) have not been subjected to such a theoretical analysis so far.

We reduce this gap in the literature by studying the SIRS process, a more realistic model, which besides re-infection additionally incorporates short-term immunity. On complex network models, we identify parameter regimes for which the process survives exponentially long, and we get a tight threshold for random graphs. Underlying these results is our main technical contribution, showing a threshold behavior for the survival time of the SIRS process on graphs with large expander subgraphs, such as social network models.

----

## [1939] Memory Asymmetry Creates Heteroclinic Orbits to Nash Equilibrium in Learning in Zero-Sum Games

**Authors**: *Yuma Fujimoto, Kaito Ariu, Kenshi Abe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29688](https://doi.org/10.1609/aaai.v38i16.29688)

**Abstract**:

Learning in games considers how multiple agents maximize their own rewards through repeated games. Memory, an ability that an agent changes his/her action depending on the history of actions in previous games, is often introduced into learning to explore more clever strategies and discuss the decision-making of real agents like humans. However, such games with memory are hard to analyze because they exhibit complex phenomena like chaotic dynamics or divergence from Nash equilibrium. In particular, how asymmetry in memory capacities between agents affects learning in games is still unclear. In response, this study formulates a gradient ascent algorithm in games with asymmetry memory capacities. To obtain theoretical insights into learning dynamics, we first consider a simple case of zero-sum games. We observe complex behavior, where learning dynamics draw a heteroclinic connection from unstable fixed points to stable ones. Despite this complexity, we analyze learning dynamics and prove local convergence to these stable fixed points, i.e., the Nash equilibria. We identify the mechanism driving this convergence: an agent with a longer memory learns to exploit the other, which in turn endows the other's utility function with strict concavity. We further numerically observe such convergence in various initial strategies, action numbers, and memory lengths. This study reveals a novel phenomenon due to memory asymmetry, providing fundamental strides in learning in games and new insights into computing equilibria.

----

## [1940] Factored Online Planning in Many-Agent POMDPs

**Authors**: *Maris F. L. Galesloot, Thiago D. Simão, Sebastian Junges, Nils Jansen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29689](https://doi.org/10.1609/aaai.v38i16.29689)

**Abstract**:

In centralized multi-agent systems, often modeled as multi-agent partially observable Markov decision processes (MPOMDPs), the action and observation spaces grow exponentially with the number of agents, making the value and belief estimation of single-agent online planning ineffective. Prior work partially tackles value estimation by exploiting the inherent structure of multi-agent settings via so-called coordination graphs. Additionally, belief estimation methods have been improved by incorporating the likelihood of observations into the approximation. However, the challenges of value estimation and belief estimation have only been tackled individually, which prevents existing methods from scaling to settings with many agents. Therefore, we address these challenges simultaneously. First, we introduce weighted particle filtering to a sample-based online planner for MPOMDPs. Second, we present a scalable approximation of the belief. Third, we bring an approach that exploits the typical locality of agent interactions to novel online planning algorithms for MPOMDPs operating on a so-called sparse particle filter tree. Our experimental evaluation against several state-of-the-art baselines shows that our methods (1) are competitive in settings with only a few agents and (2) improve over the baselines in the presence of many agents.

----

## [1941] Foundations of Reactive Synthesis for Declarative Process Specifications

**Authors**: *Luca Geatti, Marco Montali, Andrey Rivkin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29690](https://doi.org/10.1609/aaai.v38i16.29690)

**Abstract**:

Given a specification of Linear-time Temporal Logic interpreted over finite traces (LTLf), the reactive synthesis problem asks to find a finitely-representable, terminating controller that reacts to the uncontrollable actions of an environment in order to enforce a desired system specification. In this paper we study, for the first time, the foundations of reactive synthesis for DECLARE, a well-established declarative, pattern-based business process modelling language grounded in LTLf. We provide a threefold contribution. First, we define a reactive synthesis problem for DECLARE. Second, we show how an arbitrary DECLARE specification can be polynomially encoded into an equivalent pure-past one in LTLf, and exploit this to define an EXPTIME algorithm for DECLARE synthesis. Third, we derive a symbolic version of this algorithm, by introducing a novel translation of pure-past temporal formulas into symbolic deterministic finite automata.

----

## [1942] Learning in Online Principal-Agent Interactions: The Power of Menus

**Authors**: *Minbiao Han, Michael Albert, Haifeng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29691](https://doi.org/10.1609/aaai.v38i16.29691)

**Abstract**:

We study a ubiquitous learning challenge in online principal-agent problems during which the principal learns the agent's private information from the agent's revealed preferences in historical interactions. This paradigm includes important special cases such as pricing and contract design, which have been widely studied in recent literature. However, existing work considers the case where the principal can only choose a single strategy at every round to interact with the agent and then observe the agent's revealed preference through their actions. In this paper, we extend this line of study to allow the principal to offer a menu of strategies to the agent and learn additionally from observing the agent's selection from the menu. We provide a thorough investigation of several online principal-agent problem settings and characterize their sample complexities, accompanied by the corresponding algorithms we have developed. We instantiate this paradigm to several important design problems — including Stackelberg (security) games, contract design, and information design. Finally, we also explore the connection between our findings and existing results about online learning in Stackelberg games, and we offer a solution that can overcome a key hard instance of previous work.

----

## [1943] Stability of Multi-Agent Learning in Competitive Networks: Delaying the Onset of Chaos

**Authors**: *Aamal Abbas Hussain, Francesco Belardinelli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29692](https://doi.org/10.1609/aaai.v38i16.29692)

**Abstract**:

The behaviour of multi agent learning in competitive network games is often studied within the context of zero sum games, in which convergence guarantees may be obtained. However, outside of this class the behaviour of learning is known to display complex behaviours and convergence cannot be always guaranteed. Nonetheless, in order to develop a complete picture of the behaviour of multi agent learning in competitive settings, the zero sum assumption must be lifted.
Motivated by this we study the Q Learning dynamics, a popular model of exploration and exploitation in multi agent learning, in competitive network games. We determine how the degree of competition, exploration rate and network connectivity impact the convergence of Q Learning. To study generic competitive games, we parameterise network games in terms of correlations between agent payoffs and study the average behaviour of the Q Learning dynamics across all games drawn from a choice of this parameter. This statistical approach establishes choices of parameters for which Q Learning dynamics converge to a stable fixed point. Differently to previous works, we find that the stability of Q Learning is explicitly dependent only on the network connectivity rather than the total number of agents. Our experiments validate these findings and show that, under certain network structures, the total number of agents can be increased without increasing the likelihood of unstable or chaotic behaviours.

----

## [1944] Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing

**Authors**: *Haobin Jiang, Ziluo Ding, Zongqing Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29693](https://doi.org/10.1609/aaai.v38i16.29693)

**Abstract**:

Exploration in decentralized cooperative multi-agent reinforcement learning faces two challenges. One is that the novelty of global states is unavailable, while the novelty of local observations is biased. The other is how agents can explore in a coordinated way. To address these challenges, we propose MACE, a simple yet effective multi-agent coordinated exploration method. By communicating only local novelty, agents can take into account other agents' local novelty to approximate the global novelty. Further, we newly introduce weighted mutual information to measure the influence of one agent's action on other agents' accumulated novelty. We convert it as an intrinsic reward in hindsight to encourage agents to exert more influence on other agents' exploration and boost coordinated exploration. Empirically, we show that MACE achieves superior performance in three multi-agent environments with sparse rewards.

----

## [1945] Optimistic Value Instructors for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Chao Li, Yupeng Zhang, Jianqi Wang, Yujing Hu, Shaokang Dong, Wenbin Li, Tangjie Lv, Changjie Fan, Yang Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29694](https://doi.org/10.1609/aaai.v38i16.29694)

**Abstract**:

In cooperative multi-agent reinforcement learning, decentralized agents hold the promise of overcoming the combinatorial explosion of joint action space and enabling greater scalability. However, they are susceptible to a game-theoretic pathology called relative overgeneralization that shadows the optimal joint action. Although recent value-decomposition algorithms guide decentralized agents by learning a factored global action value function, the representational limitation and the inaccurate sampling of optimal joint actions during the learning process make this problem still. To address this limitation, this paper proposes a novel algorithm called Optimistic Value Instructors (OVI). The main idea behind OVI is to introduce multiple optimistic instructors into the value-decomposition paradigm, which are capable of suggesting potentially optimal joint actions and rectifying the factored global action value function to recover these optimal actions. Specifically, the instructors maintain optimistic value estimations of per-agent local actions and thus eliminate the negative effects caused by other agents' exploratory or sub-optimal non-cooperation, enabling accurate identification and suggestion of optimal joint actions. Based on the instructors' suggestions, the paper further presents two instructive constraints to rectify the factored global action value function to recover these optimal joint actions, thus overcoming the RO problem. Experimental evaluation of OVI on various cooperative multi-agent tasks demonstrates its superior performance against multiple baselines, highlighting its effectiveness.

----

## [1946] ConcaveQ: Non-monotonic Value Function Factorization via Concave Representations in Deep Multi-Agent Reinforcement Learning

**Authors**: *Huiqun Li, Hanhan Zhou, Yifei Zou, Dongxiao Yu, Tian Lan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29695](https://doi.org/10.1609/aaai.v38i16.29695)

**Abstract**:

Value function factorization has achieved great success in multi-agent reinforcement learning by optimizing joint action-value functions through the maximization of factorized per-agent utilities. To ensure Individual-Global-Maximum property, existing works often focus on value factorization using monotonic functions, which are known to result in restricted representation expressiveness. In this paper, we analyze the limitations of monotonic factorization and present ConcaveQ, a novel non-monotonic value function factorization approach that goes beyond monotonic mixing functions and employs neural network representations of concave mixing functions. Leveraging the concave property in factorization, an iterative action selection scheme is developed to obtain optimal joint actions during training. It is used to update agents’ local policy networks, enabling fully decentralized execution. The effectiveness of the proposed ConcaveQ is validated across scenarios involving multi-agent predator-prey environment and StarCraft II micromanagement tasks. Empirical results exhibit significant improvement of ConcaveQ over state-of-the-art multi-agent reinforcement learning approaches.

----

## [1947] Transition-Informed Reinforcement Learning for Large-Scale Stackelberg Mean-Field Games

**Authors**: *Pengdeng Li, Runsheng Yu, Xinrun Wang, Bo An*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29696](https://doi.org/10.1609/aaai.v38i16.29696)

**Abstract**:

Many real-world scenarios including fleet management and Ad auctions can be modeled as Stackelberg mean-field games (SMFGs) where a leader aims to incentivize a large number of homogeneous self-interested followers to maximize her utility. Existing works focus on cases with a small number of heterogeneous followers, e.g., 5-10, and suffer from scalability issue when the number of followers increases. There are three major challenges in solving large-scale SMFGs: i) classical methods based on solving differential equations fail as they require exact dynamics parameters, ii) learning by interacting with environment is data-inefficient, and iii) complex interaction between the leader and followers makes the learning performance unstable. We address these challenges through transition-informed reinforcement learning. Our main contributions are threefold: i) we first propose an RL framework, the Stackelberg mean-field update, to learn the leader's policy without priors of the environment, ii) to improve the data efficiency and accelerate the learning process, we then propose the Transition-Informed Reinforcement Learning (TIRL) by leveraging the instantiated empirical Fokker-Planck equation, and iii) we develop a regularized TIRL by employing various regularizers to alleviate the sensitivity of the learning performance to the initialization of the leader's policy. Extensive experiments on fleet management and food gathering demonstrate that our approach can scale up to 100,000 followers and significantly outperform existing baselines.

----

## [1948] Decentralized Gradient-Free Methods for Stochastic Non-smooth Non-convex Optimization

**Authors**: *Zhenwei Lin, Jingfan Xia, Qi Deng, Luo Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29697](https://doi.org/10.1609/aaai.v38i16.29697)

**Abstract**:

We consider decentralized gradient-free optimization of minimizing Lipschitz continuous functions that satisfy neither smoothness nor convexity assumption. We propose two novel gradient-free algorithms, the Decentralized Gradient-Free Method (DGFM) and its variant, the Decentralized Gradient-Free Method+ (DGFM+). Based on the techniques of randomized smoothing and gradient tracking, DGFM requires the computation of the zeroth-order oracle of a single sample in each iteration, making it less demanding in terms of computational resources for individual computing nodes. Theoretically, DGFM achieves a complexity of O(d^(3/2)δ^(-1)ε^(-4)) for obtaining a (δ,ε)-Goldstein stationary point. DGFM+, an advanced version of DGFM, incorporates variance reduction to further improve the convergence behavior. It samples a mini-batch at each iteration and periodically draws a larger batch of data, which improves the complexity to O(d^(3/2)δ^(-1)ε^(-3)). Moreover, experimental results underscore the empirical advantages of our proposed algorithms when applied to real-world datasets.

----

## [1949] Imagine, Initialize, and Explore: An Effective Exploration Method in Multi-Agent Reinforcement Learning

**Authors**: *Zeyang Liu, Lipeng Wan, Xinrui Yang, Zhuoran Chen, Xingyu Chen, Xuguang Lan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29698](https://doi.org/10.1609/aaai.v38i16.29698)

**Abstract**:

Effective exploration is crucial to discovering optimal strategies for multi-agent reinforcement learning (MARL) in complex coordination tasks. Existing methods mainly utilize intrinsic rewards to enable committed exploration or use role-based learning for decomposing joint action spaces instead of directly conducting a collective search in the entire action-observation space. However, they often face challenges obtaining specific joint action sequences to reach successful states in long-horizon tasks. To address this limitation, we propose Imagine, Initialize, and Explore (IIE), a novel method that offers a promising solution for efficient multi-agent exploration in complex scenarios. IIE employs a transformer model to imagine how the agents reach a critical state that can influence each other's transition functions. Then, we initialize the environment at this state using a simulator before the exploration phase. We formulate the imagination as a sequence modeling problem, where the states, observations, prompts, actions, and rewards are predicted autoregressively. The prompt consists of timestep-to-go, return-to-go, influence value, and one-shot demonstration, specifying the desired state and trajectory as well as guiding the action generation. By initializing agents at the critical states, IIE significantly increases the likelihood of discovering potentially important under-explored regions. Despite its simplicity, empirical results demonstrate that our method outperforms multi-agent exploration baselines on the StarCraft Multi-Agent Challenge (SMAC) and SMACv2 environments. Particularly, IIE shows improved performance in the sparse-reward SMAC tasks and produces more effective curricula over the initialized states than other generative methods, such as CVAE-GAN and diffusion models.

----

## [1950] TAPE: Leveraging Agent Topology for Cooperative Multi-Agent Policy Gradient

**Authors**: *Xingzhou Lou, Junge Zhang, Timothy J. Norman, Kaiqi Huang, Yali Du*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29699](https://doi.org/10.1609/aaai.v38i16.29699)

**Abstract**:

Multi-Agent Policy Gradient (MAPG) has made significant progress in recent years. However, centralized critics in state-of-the-art MAPG methods still face the centralized-decentralized mismatch (CDM) issue, which means sub-optimal actions by some agents will affect other agent's policy learning. While using individual critics for policy updates can avoid this issue, they severely limit cooperation among agents. To address this issue, we propose an agent topology framework, which decides whether other agents should be considered in policy gradient and achieves compromise between facilitating cooperation and alleviating the CDM issue. The agent topology allows agents to use coalition utility as learning objective instead of global utility by centralized critics or local utility by individual critics. To constitute the agent topology, various models are studied. We propose Topology-based multi-Agent Policy gradiEnt (TAPE) for both stochastic and deterministic MAPG methods. We prove the policy improvement theorem for stochastic TAPE and give a theoretical explanation for the improved cooperation among agents. Experiment results on several benchmarks show the agent topology is able to facilitate agent cooperation and alleviate CDM issue respectively to improve performance of TAPE. Finally, multiple ablation studies and a heuristic graph search algorithm are devised to show the efficacy of the agent topology.

----

## [1951] PMAC: Personalized Multi-Agent Communication

**Authors**: *Xiangrui Meng, Ying Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29700](https://doi.org/10.1609/aaai.v38i16.29700)

**Abstract**:

Communication plays a crucial role in information sharing within the field of multi-agent reinforcement learning (MARL). However, how to transmit information that meets individual needs remains a long-standing challenge. Some existing work focus on using a common channel for information transfer, which limits the capability for local communication. Meanwhile, other work attempt to establish peer-to-peer communication topologies but suffer from quadratic complexity. In this paper, we propose Personalized Multi-Agent Communication (PMAC), which enables the formation of peer-to-peer communication topologies, personalized message sending, and personalized message receiving. All these modules in PMAC are performed using only multilayer perceptrons (MLPs) with linear computational complexity. Empirically, we show the strength of personalized communication in a variety of cooperative scenarios. Our approach exhibits competitive performance compared to existing methods while maintaining notable computational efficiency.

----

## [1952] Adaptive Anytime Multi-Agent Path Finding Using Bandit-Based Large Neighborhood Search

**Authors**: *Thomy Phan, Taoan Huang, Bistra Dilkina, Sven Koenig*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29701](https://doi.org/10.1609/aaai.v38i16.29701)

**Abstract**:

Anytime multi-agent path finding (MAPF) is a promising approach to scalable path optimization in large-scale multi-agent systems. State-of-the-art anytime MAPF is based on Large Neighborhood Search (LNS), where a fast initial solution is iteratively optimized by destroying and repairing a fixed number of parts, i.e., the neighborhood of the solution, using randomized destroy heuristics and prioritized planning. Despite their recent success in various MAPF instances, current LNS-based approaches lack exploration and flexibility due to greedy optimization with a fixed neighborhood size which can lead to low-quality solutions in general. So far, these limitations have been addressed with extensive prior effort in tuning or offline machine learning beyond actual planning. In this paper, we focus on online learning in LNS and propose Bandit-based Adaptive LArge Neighborhood search Combined with Exploration (BALANCE). BALANCE uses a bi-level multi-armed bandit scheme to adapt the selection of destroy heuristics and neighborhood sizes on the fly during search. We evaluate BALANCE on multiple maps from the MAPF benchmark set and empirically demonstrate performance improvements of at least 50% compared to state-of-the-art anytime MAPF in large-scale scenarios. We find that Thompson Sampling performs particularly well compared to alternative multi-armed bandit algorithms.

----

## [1953] Minimum Coverage Sets for Training Robust Ad Hoc Teamwork Agents

**Authors**: *Muhammad Rahman, Jiaxun Cui, Peter Stone*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29702](https://doi.org/10.1609/aaai.v38i16.29702)

**Abstract**:

Robustly cooperating with unseen agents and human partners presents significant challenges due to the diverse cooperative conventions these partners may adopt. Existing Ad Hoc Teamwork (AHT) methods address this challenge by training an agent with a population of diverse teammate policies obtained through maximizing specific diversity metrics. However, prior heuristic-based diversity metrics do not always maximize the agent's robustness in all cooperative problems. In this work, we first propose that maximizing an AHT agent's robustness requires it to emulate policies in the minimum coverage set (MCS), the set of best-response policies to any partner policies in the environment. We then introduce the L-BRDiv algorithm that generates a set of teammate policies that, when used for AHT training, encourage agents to emulate policies from the MCS. L-BRDiv works by solving a constrained optimization problem to jointly train teammate policies for AHT training and approximating AHT agent policies that are members of the MCS. We empirically demonstrate that L-BRDiv produces more robust AHT agents than state-of-the-art methods in a broader range of two-player cooperative problems without the need for extensive hyperparameter tuning for its objectives.  Our study shows that L-BRDiv outperforms the baseline methods by prioritizing discovering distinct members of the MCS instead of repeatedly finding redundant policies.

----

## [1954] Decentralized Monte Carlo Tree Search for Partially Observable Multi-Agent Pathfinding

**Authors**: *Alexey Skrynnik, Anton Andreychuk, Konstantin S. Yakovlev, Aleksandr Panov*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29703](https://doi.org/10.1609/aaai.v38i16.29703)

**Abstract**:

The Multi-Agent Pathfinding (MAPF) problem involves finding a set of conflict-free paths for a group of agents confined to a graph. In typical MAPF scenarios, the graph and the agents' starting and ending vertices are known beforehand, allowing the use of centralized planning algorithms. However, in this study, we focus on the decentralized MAPF setting, where the agents may observe the other agents only locally and are restricted in communications with each other. Specifically, we investigate the lifelong variant of MAPF, where new goals are continually assigned to the agents upon completion of previous ones. Drawing inspiration from the successful AlphaZero approach, we propose a decentralized multi-agent Monte Carlo Tree Search (MCTS) method for MAPF tasks. Our approach utilizes the agent's observations to recreate the intrinsic Markov decision process, which is then used for planning with a tailored for multi-agent tasks version of neural MCTS. The experimental results show that our approach outperforms state-of-the-art learnable MAPF solvers. The source code is available at https://github.com/AIRI-Institute/mats-lp.

----

## [1955] Learn to Follow: Decentralized Lifelong Multi-Agent Pathfinding via Planning and Learning

**Authors**: *Alexey Skrynnik, Anton Andreychuk, Maria Nesterova, Konstantin S. Yakovlev, Aleksandr Panov*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29704](https://doi.org/10.1609/aaai.v38i16.29704)

**Abstract**:

Multi-agent Pathfinding (MAPF) problem generally asks to find a set of conflict-free paths for a set of agents confined to a graph and is typically solved in a centralized fashion. Conversely, in this work, we investigate the decentralized MAPF setting, when the central controller that possesses all the information on the agents' locations and goals is absent and the agents have to sequentially decide the actions on their own without having access to the full state of the environment. We focus on the practically important lifelong variant of MAPF, which involves continuously assigning new goals to the agents upon arrival to the previous ones. To address this complex problem, we propose a method that integrates two complementary approaches: planning with heuristic search and reinforcement learning through policy optimization. Planning is utilized to construct and re-plan individual paths. We enhance our planning algorithm with a dedicated technique tailored to avoid congestion and increase the throughput of the system. We employ reinforcement learning to discover the collision avoidance policies that effectively guide the agents along the paths. The policy is implemented as a neural network and is effectively trained without any reward-shaping or external guidance. We evaluate our method on a wide range of setups comparing it to the state-of-the-art solvers. The results show that our method consistently outperforms the learnable competitors, showing higher throughput and better ability to generalize to the maps that were unseen at the training stage. Moreover our solver outperforms a rule-based one in terms of throughput and is an order of magnitude faster than a state-of-the-art search-based solver. The code is available at https://github.com/AIRI-Institute/learn-to-follow.

----

## [1956] What Makes Good Collaborative Views? Contrastive Mutual Information Maximization for Multi-Agent Perception

**Authors**: *Wanfang Su, Lixing Chen, Yang Bai, Xi Lin, Gaolei Li, Zhe Qu, Pan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29705](https://doi.org/10.1609/aaai.v38i16.29705)

**Abstract**:

Multi-agent perception (MAP) allows autonomous systems to understand complex environments by interpreting data from multiple sources. This paper investigates intermediate collaboration for MAP with a specific focus on exploring "good" properties of collaborative view (i.e., post-collaboration feature) and its underlying relationship to individual views (i.e., pre-collaboration features), which were treated as an opaque procedure by most existing works. We propose a novel framework named CMiMC (Contrastive Mutual Information Maximization for Collaborative Perception) for intermediate collaboration. The core philosophy of CMiMC is to preserve discriminative information of individual views in the collaborative view by maximizing mutual information between pre- and post-collaboration features while enhancing the efficacy of collaborative views by minimizing the loss function of downstream tasks. In particular, we define multi-view mutual information (MVMI) for intermediate collaboration that evaluates correlations between collaborative views and individual views on both global and local scales. We establish CMiMNet based on multi-view contrastive learning to realize estimation and maximization of MVMI, which assists the training of a collaborative encoder for voxel-level feature fusion. We evaluate CMiMC on V2X-Sim 1.0, and it improves the SOTA average precision by 3.08% and 4.44% at 0.5 and 0.7 IoU (Intersection-over-Union) thresholds, respectively. In addition, CMiMC can reduce communication volume to 1/32 while achieving performance comparable to SOTA. Code and Appendix are released at https://github.com/77SWF/CMiMC.

----

## [1957] Bidirectional Temporal Plan Graph: Enabling Switchable Passing Orders for More Efficient Multi-Agent Path Finding Plan Execution

**Authors**: *Yifan Su, Rishi Veerapaneni, Jiaoyang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29706](https://doi.org/10.1609/aaai.v38i16.29706)

**Abstract**:

The Multi-Agent Path Finding (MAPF) problem involves planning collision-free paths for multiple agents in a shared environment. The majority of MAPF solvers rely on the assumption that an agent can arrive at a specific location at a specific timestep. However, real-world execution uncertainties can cause agents to deviate from this assumption, leading to collisions and deadlocks. Prior research solves this problem by having agents follow a Temporal Plan Graph (TPG), enforcing a consistent passing order at every location as defined in the MAPF plan. However, we show that TPGs are overly strict because, in some circumstances, satisfying the passing order requires agents to wait unnecessarily, leading to longer execution time. To overcome this issue, we introduce a new graphical representation called a Bidirectional Temporal Plan Graph (BTPG), which allows switching passing orders during execution to avoid unnecessary waiting time. We design two anytime algorithms for constructing a BTPG: BTPG-naïve and BTPG-optimized. Experimental results show that following BTPGs consistently outperforms following TPGs, reducing unnecessary waits by 8-20%.

----

## [1958] Large-Scale Multi-Robot Coverage Path Planning via Local Search

**Authors**: *Jingtao Tang, Hang Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29707](https://doi.org/10.1609/aaai.v38i16.29707)

**Abstract**:

We study graph-based Multi-Robot Coverage Path Planning (MCPP) that aims to compute coverage paths for multiple robots to cover all vertices of a given 2D grid terrain graph G. Existing graph-based MCPP algorithms first compute a tree cover on G---a forest of multiple trees that cover all vertices---and then employ the Spanning Tree Coverage (STC) paradigm to generate coverage paths on the decomposed graph D of the terrain graph G by circumnavigating the edges of the computed trees, aiming to optimize the makespan (i.e., the maximum coverage path cost among all robots).
In this paper, we take a different approach by exploring how to systematically search for good coverage paths directly on D. We introduce a new algorithmic framework, called LS-MCPP, which leverages a local search to operate directly on D. We propose a novel standalone paradigm, Extended-STC (ESTC), that extends STC to achieve complete coverage for MCPP on any decomposed graph, even those resulting from incomplete terrain graphs. Furthermore, we demonstrate how to integrate ESTC with three novel types of neighborhood operators into our framework to effectively guide its search process. Our extensive experiments demonstrate the effectiveness of LS-MCPP, consistently improving the initial solution returned by two state-of-the-art baseline algorithms that compute suboptimal tree covers on G, with a notable reduction in makespan by up to 35.7% and 30.3%, respectively. Moreover, LS-MCPP consistently matches or surpasses the results of optimal tree cover computation, achieving these outcomes with orders of magnitude faster runtime, thereby showcasing its significant benefits for large-scale real-world coverage tasks.

----

## [1959] Robust Communicative Multi-Agent Reinforcement Learning with Active Defense

**Authors**: *Lebin Yu, Yunbo Qiu, Quanming Yao, Yuan Shen, Xudong Zhang, Jian Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29708](https://doi.org/10.1609/aaai.v38i16.29708)

**Abstract**:

Communication in multi-agent reinforcement learning (MARL) has been proven to effectively promote cooperation among agents recently. Since communication in real-world scenarios is vulnerable to noises and adversarial attacks, it is crucial to develop robust communicative MARL technique. However, existing research in this domain has predominantly focused on passive defense strategies, where agents receive all messages equally, making it hard to balance performance and robustness. We propose an active defense strategy, where agents automatically reduce the impact of potentially harmful messages on the final decision. There are two challenges to implement this strategy, that are defining unreliable messages and adjusting the unreliable messages' impact on the final decision properly. To address them, we design an Active Defense Multi-Agent Communication framework (ADMAC), which estimates the reliability of received messages and adjusts their impact on the final decision accordingly with the help of a decomposable decision structure. The superiority of ADMAC over existing methods is validated by experiments in three communication-critical tasks under four types of attacks.

----

## [1960] Leveraging Partial Symmetry for Multi-Agent Reinforcement Learning

**Authors**: *Xin Yu, Rongye Shi, Pu Feng, Yongkai Tian, Simin Li, Shuhao Liao, Wenjun Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29709](https://doi.org/10.1609/aaai.v38i16.29709)

**Abstract**:

Incorporating symmetry as an inductive bias into multi-agent reinforcement learning (MARL) has led to improvements in generalization, data efficiency, and physical consistency. While prior research has succeeded in using perfect symmetry prior, the realm of partial symmetry in the multi-agent domain remains unexplored. To fill in this gap, we introduce the partially symmetric Markov game, a new subclass of the Markov game. We then theoretically show that the performance error introduced by utilizing symmetry in MARL is bounded, implying that the symmetry prior can still be useful in MARL even in partial symmetry situations. Motivated by this insight, we propose the Partial Symmetry Exploitation (PSE) framework that is able to adaptively incorporate symmetry prior in MARL under different symmetry-breaking conditions. Specifically, by adaptively adjusting the exploitation of symmetry, our framework is able to achieve superior sample efficiency and overall performance of MARL algorithms. Extensive experiments are conducted to demonstrate the superior performance of the proposed framework over baselines. Finally, we implement the proposed framework in real-world multi-robot testbed to show its superiority.

----

## [1961] ProAgent: Building Proactive Cooperative Agents with Large Language Models

**Authors**: *Ceyao Zhang, Kaijie Yang, Siyi Hu, Zihao Wang, Guanghe Li, Yihang Sun, Cheng Zhang, Zhaowei Zhang, Anji Liu, Song-Chun Zhu, Xiaojun Chang, Junge Zhang, Feng Yin, Yitao Liang, Yaodong Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29710](https://doi.org/10.1609/aaai.v38i16.29710)

**Abstract**:

Building agents with adaptive behavior in cooperative tasks stands as a paramount goal in the realm of multi-agent systems. Current approaches to developing cooperative agents rely primarily on learning-based methods, whose policy generalization depends heavily on the diversity of teammates they interact with during the training phase. Such reliance, however, constrains the agents' capacity for strategic adaptation when cooperating with unfamiliar teammates, which becomes a significant challenge in zero-shot coordination scenarios. To address this challenge, we propose ProAgent, a novel framework that harnesses large language models (LLMs) to create proactive agents capable of dynamically adapting their behavior to enhance cooperation with teammates. ProAgent can analyze the present state, and infer the intentions of teammates from observations. It then updates its beliefs in alignment with the teammates' subsequent actual behaviors. Moreover, ProAgent exhibits a high degree of modularity and interpretability, making it easily integrated into various of coordination scenarios. Experimental evaluations conducted within the Overcooked-AI environment unveil the remarkable performance superiority of ProAgent, outperforming five methods based on self-play and population-based training when cooperating with AI agents. Furthermore, in partnered with human proxy models, its performance exhibits an average improvement exceeding 10% compared to the current state-of-the-art method. For more information about our project, please visit https://pku-proagent.github.io.

----

## [1962] Intrinsic Action Tendency Consistency for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Junkai Zhang, Yifan Zhang, Xi Sheryl Zhang, Yifan Zang, Jian Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29711](https://doi.org/10.1609/aaai.v38i16.29711)

**Abstract**:

Efficient collaboration in the centralized training with decentralized execution (CTDE) paradigm remains a challenge in cooperative multi-agent systems. We identify divergent action tendencies among agents as a significant obstacle to CTDE's training efficiency, requiring a large number of training samples to achieve a unified consensus on agents' policies. This divergence stems from the lack of adequate team consensus-related guidance signals during credit assignment in CTDE. To address this, we propose Intrinsic Action Tendency Consistency, a novel approach for cooperative multi-agent reinforcement learning. It integrates intrinsic rewards, obtained through an action model, into a reward-additive CTDE (RA-CTDE) framework. We formulate an action model that enables surrounding agents to predict the central agent's action tendency. Leveraging these predictions, we compute a cooperative intrinsic reward that encourages agents to align their actions with their neighbors' predictions. We establish the equivalence between RA-CTDE and CTDE through theoretical analyses, demonstrating that CTDE's training process can be achieved using N individual targets. Building on this insight, we introduce a novel method to combine intrinsic rewards and RA-CTDE. Extensive experiments on challenging tasks in SMAC, MPE, and GRF benchmarks showcase the improved performance of our method.

----

## [1963] Emergent Communication for Numerical Concepts Generalization

**Authors**: *Enshuai Zhou, Yifan Hao, Rui Zhang, Yuxuan Guo, Zidong Du, Xishan Zhang, Xinkai Song, Chao Wang, Xuehai Zhou, Jiaming Guo, Qi Yi, Shaohui Peng, Di Huang, Ruizhi Chen, Qi Guo, Yunji Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29712](https://doi.org/10.1609/aaai.v38i16.29712)

**Abstract**:

Research on emergent communication has recently gained significant traction as a promising avenue for the linguistic community to unravel human language's origins and explore artificial intelligence's generalization capabilities. Current research has predominantly concentrated on recognizing qualitative patterns of object attributes(e.g., shape and color) and paid little attention to the quantitative relationship among object quantities which is known as the part of numerical concepts. The ability to generalize numerical concepts, i.e., counting and calculations with unseen quantities, is essential, as it mirrors humans' foundational abstract reasoning abilities. In this work, we introduce the NumGame, leveraging the referential game framework, forcing agents to communicate and generalize the numerical concepts effectively. Inspired by the human learning process of numbers, we present a two-stage training approach that sequentially fosters a rudimentary numerical sense followed by the ability of arithmetic calculation, ultimately aiding agents in generating semantically stable and unambiguous language for numerical concepts. The experimental results indicate the impressive generalization capabilities to unseen quantities and regularity of the language emergence from communication.

----

## [1964] Decomposing Temporal Equilibrium Strategy for Coordinated Distributed Multi-Agent Reinforcement Learning

**Authors**: *Chenyang Zhu, Wen Si, Jinyu Zhu, Zhihao Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29713](https://doi.org/10.1609/aaai.v38i16.29713)

**Abstract**:

The increasing demands for system complexity and robustness have prompted the integration of temporal logic into Multi-Agent Reinforcement Learning (MARL) to address tasks with non-Markovian properties. However, incorporating non-Markovian properties introduces additional computational complexities, as agents are required to integrate historical data into their decision-making process. Also, optimizing strategies within a multi-agent environment presents significant challenges due to the exponential growth of the state space with the number of agents. In this study, we introduce an innovative hierarchical MARL framework that synthesizes temporal equilibrium strategies through parity games and subsequently encodes them as individual reward machines for MARL coordination. More specifically, we reduce the strategy synthesis problem into an emptiness problem concerning parity games with optimized states and transitions. Following this synthesis step, the temporal equilibrium strategy is decomposed into individual reward machines for decentralized MARL. Theoretical proofs are provided to verify the consistency of the Nash equilibrium between the parallel composition of decomposed strategies and the original strategy. Empirical evidence confirms the efficacy of the proposed synthesis technique, showcasing its ability to reduce state space compared to the state-of-the-art tool. Furthermore, our study highlights the superior performance of the distributed MARL paradigm over centralized approaches when deploying decomposed strategies.

----

## [1965] Balancing Humans and Machines: A Study on Integration Scale and Its Impact on Collaborative Performance

**Authors**: *Rui Zou, Sannyuya Liu, Yawei Luo, Yaqi Liu, Jintian Feng, Mengqi Wei, Jianwen Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29714](https://doi.org/10.1609/aaai.v38i16.29714)

**Abstract**:

In the evolving artificial intelligence domain, hybrid human-machine systems have emerged as a transformative research area. While many studies have concentrated on individual human-machine interactions, there is a lack of focus on multi-human and multi-machine dynamics. This paper delves into these nuances by introducing a novel statistical framework that discerns integration accuracy in terms of precision and diversity. Empirical studies reveal that performance surges consistently with scale, either in human or machine settings. However, hybrid systems present complexities. Their performance is intricately tied to the human-to-machine ratio. Interestingly, as the scale expands, integration performance growth isn't limitless. It reaches a threshold influenced by model diversity. This introduces a pivotal `knee point', signifying the optimal balance between performance and scale. This knowledge is vital for resource allocation in practical applications. Grounded in rigorous evaluations using public datasets, our findings emphasize the framework's robustness in refining integrated systems.

----

## [1966] Frame Semantic Role Labeling Using Arbitrary-Order Conditional Random Fields

**Authors**: *Chaoyi Ai, Kewei Tu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29715](https://doi.org/10.1609/aaai.v38i16.29715)

**Abstract**:

This paper presents an approach to frame semantic role labeling (FSRL), a task in natural language processing that identifies semantic roles within a text following the theory of frame semantics. Unlike previous approaches which do not adequately model correlations and interactions amongst arguments, we propose arbitrary-order conditional random fields (CRFs) that are capable of modeling full interaction amongst an arbitrary number of arguments of a given predicate. To achieve tractable representation and inference, we apply canonical polyadic decomposition to the arbitrary-order factor in our proposed CRF and utilize mean-field variational inference for approximate inference. We further unfold our iterative inference procedure into a recurrent neural network that is connected to our neural encoder and scorer, enabling end-to-end training and inference. Finally, we also improve our model with several techniques such as span-based scoring and decoding. Our experiments show that our approach achieves state-of-the-art performance in FSRL.

----

## [1967] DTF-AT: Decoupled Time-Frequency Audio Transformer for Event Classification

**Authors**: *Tony Alex, Sara Ahmed, Armin Mustafa, Muhammad Awais, Philip JB Jackson*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29716](https://doi.org/10.1609/aaai.v38i16.29716)

**Abstract**:

Convolutional neural networks (CNNs) and Transformer-based networks have recently enjoyed significant attention for various audio classification and tagging tasks following their wide adoption in the computer vision domain.
Despite the difference in information distribution between audio spectrograms and natural images, there has been limited exploration of effective information retrieval from spectrograms using domain-specific layers tailored for the audio domain. In this paper, we leverage the power of the Multi-Axis Vision Transformer (MaxViT) to create DTF-AT (Decoupled Time-Frequency Audio Transformer) that facilitates interactions across time, frequency, spatial, and channel dimensions.
The proposed DTF-AT architecture is rigorously evaluated across diverse audio and speech classification tasks, consistently establishing new benchmarks for state-of-the-art (SOTA) performance. Notably, on the challenging AudioSet 2M classification task, our approach demonstrates a substantial improvement of 4.4% when the model is trained from scratch and 3.2% when the model is initialised from ImageNet-1K pretrained weights. In addition, we present comprehensive ablation studies to investigate the impact and efficacy of our proposed approach. The codebase and pretrained weights are available on https://github.com/ta012/DTFAT.git

----

## [1968] WikiSQE: A Large-Scale Dataset for Sentence Quality Estimation in Wikipedia

**Authors**: *Kenichiro Ando, Satoshi Sekine, Mamoru Komachi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29717](https://doi.org/10.1609/aaai.v38i16.29717)

**Abstract**:

Wikipedia can be edited by anyone and thus contains various quality sentences. Therefore, Wikipedia includes some poor-quality edits, which are often marked up by other editors. While editors' reviews enhance the credibility of Wikipedia, it is hard to check all edited text. Assisting in this process is very important, but a large and comprehensive dataset for studying it does not currently exist. Here, we propose WikiSQE, the first large-scale dataset for sentence quality estimation in Wikipedia. Each sentence is extracted from the entire revision history of English Wikipedia, and the target quality labels were carefully investigated and selected. WikiSQE has about 3.4 M sentences with 153 quality labels. In the experiment with automatic classification using competitive machine learning models, sentences that had problems with citation, syntax/semantics, or propositions were found to be more difficult to detect. In addition, by performing human annotation, we found that the model we developed performed better than the crowdsourced workers. WikiSQE is expected to be a valuable resource for other tasks in NLP.

----

## [1969] Beyond Grounding: Extracting Fine-Grained Event Hierarchies across Modalities

**Authors**: *Hammad A. Ayyubi, Christopher Thomas, Lovish Chum, Rahul Lokesh, Long Chen, Yulei Niu, Xudong Lin, Xuande Feng, Jaywon Koo, Sounak Ray, Shih-Fu Chang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29718](https://doi.org/10.1609/aaai.v38i16.29718)

**Abstract**:

Events describe happenings in our world that are of importance. Naturally, understanding events mentioned in multimedia content and how they are related forms an important way of comprehending our world. Existing literature can infer if events across textual and visual (video) domains are identical (via grounding) and thus, on the same semantic level. However, grounding fails to capture the intricate cross-event relations that exist due to the same events being referred to on many semantic levels. For example, the abstract event of "war'' manifests at a lower semantic level through subevents "tanks firing'' (in video) and airplane "shot'' (in text), leading to a hierarchical, multimodal relationship between the events.


In this paper, we propose the task of extracting event hierarchies from multimodal (video and text) data to capture how the same event manifests itself in different modalities at different semantic levels. This reveals the structure of events and is critical to understanding them. To support research on this task, we introduce the Multimodal Hierarchical Events (MultiHiEve) dataset. Unlike prior video-language datasets, MultiHiEve is composed of news video-article pairs, which makes it rich in event hierarchies. We densely annotate a part of the dataset to construct the test benchmark. We show the limitations of state-of-the-art unimodal and multimodal baselines on this task. Further, we address these limitations via a new weakly supervised model, leveraging only unannotated video-article pairs from MultiHiEve. We perform a thorough evaluation of our proposed method which demonstrates improved performance on this task and highlight opportunities for future research. Data: https://github.com/hayyubi/multihieve

----

## [1970] All Should Be Equal in the Eyes of LMs: Counterfactually Aware Fair Text Generation

**Authors**: *Pragyan Banerjee, Abhinav Java, Surgan Jandial, Simra Shahid, Shaz Furniturewala, Balaji Krishnamurthy, Sumit Bhatia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29719](https://doi.org/10.1609/aaai.v38i16.29719)

**Abstract**:

Fairness in Language Models (LMs) remains a long-standing challenge, given the inherent biases in training data that can be perpetuated by models and affect the downstream tasks. Recent methods employ expensive retraining or attempt debiasing during inference by constraining model outputs to contrast from a reference set of biased templates/exemplars. Regardless, they don’t address the primary goal of fairness to maintain equitability across different demographic groups. In this work, we posit that inferencing LMs to generate unbiased output for one demographic under a context ensues from being aware of outputs for other demographics under the same context. To this end, we propose Counterfactually Aware Fair InferencE (CAFIE), a framework that dynamically compares the model’s understanding of diverse demographics to generate more equitable sentences. We conduct an extensive empirical evaluation using base LMs of varying sizes and across three diverse datasets and found that CAFIE outperforms strong baselines. CAFIE produces fairer text and strikes the best balance between fairness and language modeling capability.

----

## [1971] Graph of Thoughts: Solving Elaborate Problems with Large Language Models

**Authors**: *Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29720](https://doi.org/10.1609/aaai.v38i16.29720)

**Abstract**:

We introduce Graph of Thoughts (GoT): a framework that
advances prompting capabilities in large language models
(LLMs) beyond those offered by paradigms such as 
Chain-of-Thought or Tree of Thoughts (ToT). The key idea and 
primary advantage of GoT is the ability to model the information 
generated by an LLM as an arbitrary graph, where units of 
information ("LLM thoughts") are vertices, and edges correspond
to dependencies between these vertices. This approach enables 
combining arbitrary LLM thoughts into synergistic outcomes, 
distilling the essence of whole networks of thoughts,
or enhancing thoughts using feedback loops. We illustrate
that GoT offers advantages over state of the art on different
tasks, for example increasing the quality of sorting by 62%
over ToT, while simultaneously reducing costs by >31%.
We ensure that GoT is extensible with new thought 
transformations and thus can be used to spearhead new prompting
schemes. This work brings the LLM reasoning closer to human 
thinking or brain mechanisms such as recurrence, both
of which form complex networks

----

## [1972] When Do Program-of-Thought Works for Reasoning?

**Authors**: *Zhen Bi, Ningyu Zhang, Yinuo Jiang, Shumin Deng, Guozhou Zheng, Huajun Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29721](https://doi.org/10.1609/aaai.v38i16.29721)

**Abstract**:

In the realm of embodied artificial intelligence, the reasoning capabilities of Large Language Models (LLMs) play a pivotal role. Although there are effective methods like program-of-thought  prompting for LLMs which uses programming language to tackle complex reasoning tasks, the specific impact of code data on the improvement of reasoning capabilities remains under-explored. To address this gap, we propose complexity-impacted reasoning score CIRS, which combines structural and logical attributes, to measure the correlation between code and reasoning abilities. Specifically, we use the abstract syntax tree to encode the structural information and calculate logical complexity by considering the difficulty and the cyclomatic complexity. Through an empirical analysis, we find not all code data of complexity can be learned or understood by LLMs. Optimal level of complexity is critical to the improvement of reasoning abilities by program-aided prompting. Then we design an auto-synthesizing and stratifying algorithm, and apply it to instruction generation for mathematical reasoning  and code data filtering for code generation tasks. Extensive results demonstrates the effectiveness of our proposed approach.

----

## [1973] Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory

**Authors**: *Aydar Bulatov, Yuri Kuratov, Yermek Kapushev, Mikhail Burtsev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29722](https://doi.org/10.1609/aaai.v38i16.29722)

**Abstract**:

A major limitation for the broader scope of problems solvable by transformers is the quadratic scaling of computational complexity with input size. In this study, we investigate the recurrent memory augmentation of pre-trained transformer models to extend input context length while linearly scaling compute. Our approach demonstrates the capability to store information in memory for sequences of up to an unprecedented two million tokens while maintaining high retrieval accuracy. Experiments with language modeling tasks show perplexity improvement as the number of processed input segments increases. These results underscore the effectiveness of our method, which has significant potential to enhance long-term dependency handling in natural language understanding and generation tasks, as well as enable large-scale context processing for memory-intensive applications.

----

## [1974] MedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models

**Authors**: *Yan Cai, Linlin Wang, Ye Wang, Gerard de Melo, Ya Zhang, Yanfeng Wang, Liang He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29723](https://doi.org/10.1609/aaai.v38i16.29723)

**Abstract**:

The emergence of various medical large language models (LLMs) in the medical domain has highlighted the need for unified evaluation standards, as manual evaluation of LLMs proves to be time-consuming and labor-intensive. To address this issue, we introduce MedBench, a comprehensive benchmark for the Chinese medical domain, comprising 40,041 questions sourced from authentic examination exercises and medical reports of diverse branches of medicine. In particular, this benchmark is composed of four key components: the Chinese Medical Licensing Examination, the Resident Standardization Training Examination, the Doctor In-Charge Qualification Examination, and real-world clinic cases encompassing examinations, diagnoses, and treatments. MedBench replicates the educational progression and clinical practice experiences of doctors in Mainland China, thereby establish- ing itself as a credible benchmark for assessing the mastery of knowledge and reasoning abilities in medical language learning models. We perform extensive experiments and conduct an in-depth analysis from diverse perspectives, which culminate in the following findings: (1) Chinese medical LLMs underperform on this benchmark, highlighting the need for significant advances in clinical knowledge and diagnostic precision. (2) Several general-domain LLMs surprisingly possess considerable medical knowledge. These findings elucidate both the capabilities and limitations of LLMs within the context of MedBench, with the ultimate goal of aiding the medical research community.

----

## [1975] CAR-Transformer: Cross-Attention Reinforcement Transformer for Cross-Lingual Summarization

**Authors**: *Yuang Cai, Yuyu Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29724](https://doi.org/10.1609/aaai.v38i16.29724)

**Abstract**:

Cross-Lingual Summarization (CLS) involves generating a summary for a given document in another language. Most of the existing approaches adopt multi-task training and knowledge distillation, which increases the training cost and improves the performance of CLS tasks intuitively but unexplainably. In this work, we propose Cross-Attention Reinforcement (CAR) module and incorporate the module into the transformer backbone to formulate the CAR-Transformer. The CAR module formulates a pseudo summarization policy parameterized by the cross-attention weights reinforced by the ground-truth monolingual summary without introducing extra model parameters. Our approach demonstrates more consistent improvement across CLS tasks compared to traditional multi-task training methods and outperforms the fine-tuned vanilla mBART by 3.67 and the best-performing multi-task training approach by 1.48 in ROUGE-L F1 score on the WikiLingua Korean-to-English CLS task.

----

## [1976] Compositional Generalization for Multi-Label Text Classification: A Data-Augmentation Approach

**Authors**: *Yuyang Chai, Zhuang Li, Jiahui Liu, Lei Chen, Fei Li, Donghong Ji, Chong Teng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29725](https://doi.org/10.1609/aaai.v38i16.29725)

**Abstract**:

Despite significant advancements in multi-label text classification, the ability of existing models to generalize to novel and seldom-encountered complex concepts, which are compositions of elementary ones, remains underexplored. This research addresses this gap. By creating unique data splits across three benchmarks, we assess the compositional generalization ability of existing multi-label text classification models. Our results show that these models often fail to generalize to compositional concepts encountered infrequently during training, leading to inferior performance on tests with these new combinations. To address this, we introduce a data augmentation method that leverages two innovative text generation models designed to enhance the classification models' capacity for compositional generalization. Our experiments show that this data augmentation approach significantly improves the compositional generalization capabilities of classification models on our benchmarks, with both generation models surpassing other text generation baselines. Our codes available at https://github.com/yychai74/LD-VAE.

----

## [1977] Counterfactual-Enhanced Information Bottleneck for Aspect-Based Sentiment Analysis

**Authors**: *Mingshan Chang, Min Yang, Qingshan Jiang, Ruifeng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29726](https://doi.org/10.1609/aaai.v38i16.29726)

**Abstract**:

Despite having achieved notable success for aspect-based sentiment analysis (ABSA), deep neural networks are susceptible to spurious correlations between input features and output labels, leading to poor robustness. In this paper, we propose a novel Counterfactual-Enhanced Information Bottleneck framework (called CEIB) to reduce spurious correlations for ABSA. CEIB extends the information bottleneck (IB) principle to a factual-counterfactual balancing setting by integrating augmented counterfactual data, with the goal of learning a robust ABSA model. Concretely, we first devise a multi-pattern prompting method, which utilizes the large language model (LLM) to generate high-quality counterfactual samples from the original samples. Then, we employ the information bottleneck principle and separate the mutual information into factual and counterfactual parts. In this way, we can learn effective and robust representations for the ABSA task by balancing the predictive information of these two parts. Extensive experiments on five benchmark ABSA datasets show that our CEIB approach achieves superior prediction performance and robustness over the state-of-the-art baselines. Code and data to reproduce the results in this paper is available at: https://github.com/shesshan/CEIB.

----

## [1978] Visual Instruction Tuning with Polite Flamingo

**Authors**: *Delong Chen, Jianfeng Liu, Wenliang Dai, Baoyuan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29727](https://doi.org/10.1609/aaai.v38i16.29727)

**Abstract**:

Recent research has demonstrated that the multi-task fine-tuning of multi-modal Large Language Models (LLMs) using an assortment of annotated downstream vision-language datasets significantly enhances their performance. Yet, during this process, a side effect, which we termed as the "multi-modal alignment tax", surfaces. This side effect negatively impacts the model's ability to format responses appropriately - for instance, its "politeness" - due to the overly succinct and unformatted nature of raw annotations, resulting in reduced human preference. In this paper, we introduce Polite Flamingo, a multi-modal response rewriter that transforms raw annotations into a more appealing, "polite" format. Polite Flamingo is trained to reconstruct high-quality responses from their automatically distorted counterparts and is subsequently applied to a vast array of vision-language datasets for response rewriting. After rigorous filtering, we generate the PF-1M dataset and further validate its value by fine-tuning a multi-modal LLM with it. Combined with novel methodologies including U-shaped multi-stage tuning and multi-turn augmentation, the resulting model, Clever Flamingo, demonstrates its advantages in both multi-modal understanding and response politeness according to automated and human evaluations. Code and dataset are available at https://github.com/ChenDelong1999/polite-flamingo

----

## [1979] Benchmarking Large Language Models in Retrieval-Augmented Generation

**Authors**: *Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29728](https://doi.org/10.1609/aaai.v38i16.29728)

**Abstract**:

Retrieval-Augmented Generation (RAG) is a promising approach for mitigating the hallucination of large language models (LLMs). However, existing research lacks rigorous evaluation of the impact of retrieval-augmented generation on different large language models, which make it challenging to identify the potential bottlenecks in the capabilities of RAG for different LLMs. In this paper, we systematically investigate the impact of Retrieval-Augmented Generation on large language models. We analyze the performance of different large language models in 4 fundamental abilities required for RAG, including noise robustness, negative rejection, information integration, and counterfactual robustness. To this end, we establish Retrieval-Augmented Generation Benchmark (RGB), a new corpus for RAG evaluation in both English and Chinese. RGB divides the instances within the benchmark into 4 separate testbeds based on the aforementioned fundamental abilities required to resolve the case. Then we evaluate 6 representative LLMs on RGB to diagnose the challenges of current LLMs when applying RAG. Evaluation reveals that while LLMs exhibit a certain degree of noise robustness, they still struggle significantly in terms of negative rejection, information integration, and dealing with false information. The aforementioned assessment outcomes indicate that there is still a considerable journey ahead to effectively apply RAG to LLMs.

----

## [1980] CIDR: A Cooperative Integrated Dynamic Refining Method for Minimal Feature Removal Problem

**Authors**: *Qian Chen, Taolin Zhang, Dongyang Li, Xiaofeng He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29729](https://doi.org/10.1609/aaai.v38i16.29729)

**Abstract**:

The minimal feature removal problem in the post-hoc explanation area aims to identify the minimal feature set (MFS). Prior studies using the greedy algorithm to calculate the minimal feature set lack the exploration of feature interactions under a monotonic assumption which cannot be satisfied in general scenarios. In order to address the above limitations, 
we  propose a Cooperative Integrated Dynamic Refining method (CIDR) to efficiently discover  minimal feature sets. Specifically, we design Cooperative Integrated Gradients (CIG) to detect interactions between features. By incorporating CIG and  characteristics of the minimal feature set, we transform the minimal feature removal problem into a knapsack problem. Additionally, we  devise an auxiliary Minimal Feature Refinement algorithm to determine the  minimal feature set from numerous candidate sets. To the best of our knowledge, our work is the first to address the minimal feature removal problem in the field of natural language processing. Extensive experiments demonstrate that CIDR is capable of tracing representative minimal feature sets with improved interpretability across various models and datasets.

----

## [1981] Is a Large Language Model a Good Annotator for Event Extraction?

**Authors**: *Ruirui Chen, Chengwei Qin, Weifeng Jiang, Dongkyu Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29730](https://doi.org/10.1609/aaai.v38i16.29730)

**Abstract**:

Event extraction is an important task in natural language processing that focuses on mining event-related information from unstructured text. Despite considerable advancements, it is still challenging to achieve satisfactory performance in this task, and issues like data scarcity and imbalance obstruct progress. In this paper, we introduce an innovative approach where we employ Large Language Models (LLMs) as expert annotators for event extraction. We strategically include sample data from the training dataset in the prompt as a reference, ensuring alignment between the data distribution of LLM-generated samples and that of the benchmark dataset. This enables us to craft an augmented dataset that complements existing benchmarks, alleviating the challenges of data imbalance and scarcity and thereby enhancing the performance of fine-tuned models. We conducted extensive experiments to validate the efficacy of our proposed method, and we believe that this approach holds great potential for propelling the development and application of more advanced and reliable event extraction systems in real-world scenarios.

----

## [1982] Modeling Adaptive Inter-Task Feature Interactions via Sentiment-Aware Contrastive Learning for Joint Aspect-Sentiment Prediction

**Authors**: *Wei Chen, Yuxuan Liu, Zhao Zhang, Fuzhen Zhuang, Jiang Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29731](https://doi.org/10.1609/aaai.v38i16.29731)

**Abstract**:

Aspect prediction (AP) and sentiment prediction (SP) are representative applications in fine-grained sentiment anal- ysis. They can be considered as sequential tasks, where AP identifies mentioned aspects in a sentence, and SP infers fine-grained sentiments for these aspects. Recent models perform the aspect-sentiment prediction in a joint man-ner, but heavily rely on the feature interactions of aspect and sentiment. One drawback is that they ignore correlation strength varies between aspect features and sentiment fea- tures across different sentences, and employ a fixed feature interaction strategy may limit effective knowledge transfer across tasks. To tackle this issue, in this paper, we propose an Adaptive Inter-task Feature Interaction framework, AIFI, for joint aspect-sentiment prediction. Specifically, we introduce a novel contrast-based alignment method based on contrastive learning. Our approach considers the AP-specific and SP-specific representations of a given sentence as a positive pair, while representation of another random sentence serves as a negative example. Moreover, we propose an inter-task feature correlation network to predict the contrast strength, which is determined by the temperature coefficient in the InfoNCE loss. This dynamic correlation adjustment enhances model’s ability to capture proper feature interactions more efficiently. Experimental results on three datasets validate the effectiveness of our approach.

----

## [1983] From Coarse to Fine: A Distillation Method for Fine-Grained Emotion-Causal Span Pair Extraction in Conversation

**Authors**: *Xinhao Chen, Chong Yang, Changzhi Sun, Man Lan, Aimin Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29732](https://doi.org/10.1609/aaai.v38i16.29732)

**Abstract**:

We study the problem of extracting emotions and the causes behind these emotions in conversations.
Existing methods either tackle them separately or jointly model them at the coarse-grained level of emotions (fewer emotion categories) and causes (utterance-level causes). 
In this work, we aim to jointly extract more fine-grained emotions and causes.
We construct a fine-grained dataset FG-RECCON, includes 16 fine-grained emotion categories and span-level causes.
To further improve the fine-grained extraction performance, we propose to utilize the casual discourse knowledge in a knowledge distillation way.
Specifically, the teacher model learns to predict causal connective words between utterances, and then guides the student model in identifying both the fine-grained emotion labels and causal spans.
Experimental results demonstrate that our distillation method achieves state-of-the-art performance on both RECCON and FG-RECCON dataset.

----

## [1984] Divergence-Guided Simultaneous Speech Translation

**Authors**: *Xinjie Chen, Kai Fan, Wei Luo, Linlin Zhang, Libo Zhao, Xinggao Liu, Zhongqiang Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29733](https://doi.org/10.1609/aaai.v38i16.29733)

**Abstract**:

To achieve high-quality translation with low latency, a Simultaneous Speech Translation (SimulST) system relies on a policy module to decide whether to translate immediately or wait for additional streaming input, along with a translation model capable of effectively handling partial speech input. Prior research has tackled these components separately, either using ``wait-k'' policies based on fixed-length segments or detected word boundaries, or dynamic policies based on different strategies (e.g., meaningful units), while employing offline models for prefix-to-prefix translation. In this paper, we propose Divergence-Guided Simultaneous Speech Translation (DiG-SST), a tightly integrated approach focusing on both translation quality and latency for streaming input. Specifically, we introduce a simple yet effective prefix-based strategy for training translation models with partial speech input, and develop an adaptive policy that makes read/write decisions for the translation model based on the expected divergence in translation distributions resulting from future input. Our experiments on multiple translation directions of the MuST-C benchmark demonstrate that our approach achieves a better trade-off between translation quality and latency compared to existing methods.

----

## [1985] Benchmarking Large Language Models on Controllable Generation under Diversified Instructions

**Authors**: *Yihan Chen, Benfeng Xu, Quan Wang, Yi Liu, Zhendong Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29734](https://doi.org/10.1609/aaai.v38i16.29734)

**Abstract**:

While large language models (LLMs) have exhibited impressive instruction-following capabilities, it is still unclear whether and to what extent they can respond to explicit constraints that might be entailed in various instructions. As a significant aspect of LLM alignment, it is thus important to formulate such a specialized set of instructions as well as investigate the resulting behavior of LLMs. To address this vacancy, we propose a new benchmark CoDI-Eval to systematically and comprehensively evaluate LLMs' responses to instructions with various constraints. We construct a large collection of constraints-attributed instructions as a test suite focused on both generalization and coverage. Specifically, we advocate an instruction diversification process to synthesize diverse forms of constraint expression and also deliberate the candidate task taxonomy with even finer-grained sub-categories. Finally, we automate the entire evaluation process to facilitate further developments. Different from existing studies on controllable text generation, CoDI-Eval extends the scope to the prevalent instruction-following paradigm for the first time. We provide extensive evaluations of representative LLMs (e.g., ChatGPT, Vicuna) on CoDI-Eval, revealing their limitations in following instructions with specific constraints and there is still a significant gap between open-source and commercial closed-source LLMs. We believe this benchmark will facilitate research into improving the controllability of LLMs' responses to instructions. Our data and code are available at https://github.com/Xt-cyh/CoDI-Eval.

----

## [1986] Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons

**Authors**: *Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29735](https://doi.org/10.1609/aaai.v38i16.29735)

**Abstract**:

Pre-trained language models (PLMs) contain vast amounts of factual knowledge, but how the knowledge is stored in the parameters remains unclear. This paper delves into the complex task of understanding how factual knowledge is stored in multilingual PLMs, and introduces the Architecture-adapted Multilingual Integrated Gradients method, which successfully localizes knowledge neurons more precisely compared to current methods, and is more universal across various architectures and languages. Moreover, we conduct an in-depth exploration on knowledge neurons, leading to the following two important discoveries: (1) The discovery of Language-Independent Knowledge Neurons, which store factual knowledge in a form that transcends language. We design cross-lingual knowledge editing experiments, demonstrating that the PLMs can accomplish this task based on language-independent neurons; (2) The discovery of Degenerate Knowledge Neurons, a novel type of neuron showing that different knowledge neurons can store the same fact. Its property of functional overlap endows the PLMs with a robust mastery of factual knowledge. We design fact-checking experiments, proving that the degenerate knowledge neurons can help the PLMs to detect wrong facts. Experiments corroborate these findings, shedding light on the mechanisms of factual knowledge storage in multilingual PLMs, and contribute valuable insights to the field. The code is available at https://github.com/heng840/AMIG.

----

## [1987] Talk Funny! A Large-Scale Humor Response Dataset with Chain-of-Humor Interpretation

**Authors**: *Yuyan Chen, Yichen Yuan, Panjun Liu, Dayiheng Liu, Qinghao Guan, Mengfei Guo, Haiming Peng, Bang Liu, Zhixu Li, Yanghua Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29736](https://doi.org/10.1609/aaai.v38i16.29736)

**Abstract**:

Humor is a crucial part of human communication. Understanding humor and generating humorous responses in dialogue can provide natural and empathic human-computer interactions.
However, most existing pre-trained language models (PLMs) perform  unsatisfactorily in humor generation.
On one hand, the serious shortage of humor corpus and datasets pose challenges for constructing models that can understand and generate humorous expressions. On the other hand, humor generation relies on rich knowledge and commonsense, which is often tacit and unspoken.
In this paper, we construct the largest Chinese Explainable Humor Response Dataset to date with chain-of-humor and humor mind map annotations, which can be used to comprehensively evaluate as well as improve the humorous response ability of PLMs.
We further design humor-related auxiliary tasks to further enhance PLMs' humorous response performance.
Extensive evaluations demonstrate that our proposed dataset and auxiliary tasks effectively help PLMs to generate humorous responses, laying the groundwork for future humor research.

----

## [1988] Editing Language Model-Based Knowledge Graph Embeddings

**Authors**: *Siyuan Cheng, Ningyu Zhang, Bozhong Tian, Xi Chen, Qingbin Liu, Huajun Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29737](https://doi.org/10.1609/aaai.v38i16.29737)

**Abstract**:

Recently decades have witnessed the empirical success of framing Knowledge Graph (KG) embeddings via language models. However, language model-based KG embeddings are usually deployed as static artifacts, making them difficult to modify post-deployment without re-training after deployment. To address this issue, we propose a new task of editing language model-based KG embeddings in this paper. This task is designed to facilitate rapid, data-efficient updates to KG embeddings without compromising the performance of other aspects. We build four new datasets: E-FB15k237, A-FB15k237, E-WN18RR, and A-WN18RR, and evaluate several knowledge editing baselines demonstrating the limited ability of previous models to handle the proposed challenging task. We further propose a simple yet strong baseline dubbed KGEditor, which utilizes additional parametric layers of the hypernetwork to edit/add facts. Our comprehensive experimental results reveal that KGEditor excels in updating specific facts without impacting the overall performance, even when faced with limited training resources. Code and datasets will be available at https://github.com/AnonymousForPapers/DeltaKG.

----

## [1989] Towards Multi-Intent Spoken Language Understanding via Hierarchical Attention and Optimal Transport

**Authors**: *Xuxin Cheng, Zhihong Zhu, Hongxiang Li, Yaowei Li, Xianwei Zhuang, Yuexian Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29738](https://doi.org/10.1609/aaai.v38i16.29738)

**Abstract**:

Multi-Intent spoken language understanding (SLU) can handle complicated utterances expressing multiple intents, which has attracted increasing attention from researchers. Although existing models have achieved promising performance, most of them still suffer from two leading problems: (1) each intent has its specific scope and the semantic information outside the scope might potentially hinder accurate predictions, i.e. scope barrier; (2) only the guidance from intent to slot is modeled but the guidance from slot to intent is often neglected, i.e. unidirectional guidance. In this paper, we propose a novel Multi-Intent SLU framework termed HAOT, which utilizes hierarchical attention to divide the scopes of each intent and applies optimal transport to achieve the mutual guidance between slot and intent. Experiments demonstrate that our model achieves state-of-the-art performance on two public Multi-Intent SLU datasets, obtaining the 3.4 improvement on MixATIS dataset compared to the previous best models in overall accuracy.

----

## [1990] Cooper: Coordinating Specialized Agents towards a Complex Dialogue Goal

**Authors**: *Yi Cheng, Wenge Liu, Jian Wang, Chak Tou Leong, Yi Ouyang, Wenjie Li, Xian Wu, Yefeng Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29739](https://doi.org/10.1609/aaai.v38i16.29739)

**Abstract**:

In recent years, there has been a growing interest in exploring dialogues with more complex goals, such as negotiation, persuasion, and emotional support, which go beyond traditional service-focused dialogue systems. Apart from the requirement for much more sophisticated strategic reasoning and communication skills, a significant challenge of these tasks lies in the difficulty of objectively measuring the achievement of their goals in a quantifiable way, making it difficult for existing research to directly optimize the dialogue procedure towards them. In our work, we emphasize the multifaceted nature of complex dialogue goals and argue that it is more feasible to accomplish them by comprehensively considering and jointly promoting their different aspects. To this end, we propose a novel dialogue framework, Cooper, which coordinates multiple specialized agents, each dedicated to a specific dialogue goal aspect separately, to approach the complex objective. Through this divide-and-conquer manner, we make complex dialogue goals more approachable and elicit greater intelligence via the collaboration of individual agents. Experiments on persuasion and emotional support dialogues demonstrate the superiority of our method over a set of competitive baselines.  Our codes are available at https://github.com/YiCheng98/Cooper.

----

## [1991] DDDM-VC: Decoupled Denoising Diffusion Models with Disentangled Representation and Prior Mixup for Verified Robust Voice Conversion

**Authors**: *Ha-Yeong Choi, Sang-Hoon Lee, Seong-Whan Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29740](https://doi.org/10.1609/aaai.v38i16.29740)

**Abstract**:

Diffusion-based generative models have recently exhibited powerful generative performance. However, as many attributes exist in the data distribution and owing to several limitations of sharing the model parameters across all levels of the generation process, it remains challenging to control specific styles for each attribute. To address the above problem, we introduce decoupled denoising diffusion models (DDDMs) with disentangled representations, which can enable effective style transfers for each attribute in generative models. In particular, we apply DDDMs for voice conversion (VC) tasks, tackling the intricate challenge of disentangling and individually transferring each speech attributes such as linguistic information, intonation, and timbre. First, we use a self-supervised representation to disentangle the speech representation. Subsequently, the DDDMs are applied to resynthesize the speech from the disentangled representations for style transfer with respect to each attribute. Moreover, we also propose the prior mixup for robust voice style transfer, which uses the converted representation of the mixed style as a prior distribution for the diffusion models. The experimental results reveal that our method outperforms publicly available VC models. Furthermore, we show that our method provides robust generative performance even when using a smaller model size. Audio samples are available at https://hayeong0.github.io/DDDM-VC-demo/.

----

## [1992] How to Protect Copyright Data in Optimization of Large Language Models?

**Authors**: *Timothy Chu, Zhao Song, Chiwun Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29741](https://doi.org/10.1609/aaai.v38i16.29741)

**Abstract**:

Large language models (LLMs) and generative AI have played a transformative role in computer research and applications. Controversy has arisen as to whether these models output copyrighted data, which can occur if the data the models are trained on is copyrighted. LLMs are built on the transformer neural network architecture, which in turn relies on a mathematical computation called Attention that uses the softmax function.

In this paper, we observe that large language model training and optimization can be seen as a softmax regression problem. We then establish a method of efficiently performing softmax regression, in a way that prevents the regression function from generating copyright data. This establishes a theoretical method of training large language models in a way that avoids generating copyright data.

----

## [1993] Unsupervised Layer-Wise Score Aggregation for Textual OOD Detection

**Authors**: *Maxime Darrin, Guillaume Staerman, Eduardo Dadalto Câmara Gomes, Jackie C. K. Cheung, Pablo Piantanida, Pierre Colombo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29742](https://doi.org/10.1609/aaai.v38i16.29742)

**Abstract**:

Out-of-distribution (OOD) detection is a rapidly growing field due to new robustness and security requirements driven by an increased number of AI-based systems. Existing OOD textual detectors often rely on anomaly scores (\textit{e.g.}, Mahalanobis distance) computed on the embedding output of the last layer of the encoder. In this work, we observe that OOD detection performance varies greatly depending on the task and layer output. More importantly, we show that the usual choice (the last layer) is rarely the best one for OOD detection and that far better results can be achieved, provided that an oracle selects the best layer. We propose a data-driven, unsupervised method to leverage this observation to combine layer-wise anomaly scores. In addition, we extend classical textual OOD benchmarks by including classification tasks with a more significant number of classes (up to 150), which reflects more realistic settings. On this augmented benchmark, we show that the proposed post-aggregation methods achieve robust and consistent results comparable to using the best layer according to an oracle while removing manual feature selection altogether.

----

## [1994] Spanning the Spectrum of Hatred Detection: A Persian Multi-Label Hate Speech Dataset with Annotator Rationales

**Authors**: *Zahra Delbari, Nafise Sadat Moosavi, Mohammad Taher Pilehvar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29743](https://doi.org/10.1609/aaai.v38i16.29743)

**Abstract**:

With the alarming rise of hate speech in online communities, the demand for effective NLP models to identify instances of offensive language has reached a critical point. However, the development of such models heavily relies on the availability of annotated datasets, which are scarce, particularly for less-studied languages. To bridge this gap for the Persian language, we present a novel dataset specifically tailored to multi-label hate speech detection.  Our dataset, called Phate, consists of an extensive collection of over seven thousand manually-annotated Persian tweets, offering a rich resource for training and evaluating hate speech detection models on this language. Notably, each annotation in our dataset specifies the targeted group of hate speech and includes a span of the tweet which elucidates the rationale behind the assigned label. The incorporation of these information expands the potential applications of our dataset, facilitating the detection of targeted online harm or allowing the benchmark to serve research on interpretability of hate speech detection models. The dataset, annotation guideline, and all associated codes are accessible at https://github.com/Zahra-D/Phate.

----

## [1995] Enhancing Bilingual Lexicon Induction via Bi-directional Translation Pair Retrieving

**Authors**: *Qiuyu Ding, Hailong Cao, Tiejun Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29744](https://doi.org/10.1609/aaai.v38i16.29744)

**Abstract**:

Most Bilingual Lexicon Induction (BLI) methods retrieve word translation pairs by finding the closest target word for a given source word based on cross-lingual word embeddings (WEs). However, we find that solely retrieving translation from the source-to-target perspective leads to some false positive translation pairs, which significantly harm the precision of BLI. To address this problem, we propose a novel and effective method to improve translation pair retrieval in cross-lingual WEs. Specifically, we consider both source-side and target-side perspectives throughout the retrieval process to alleviate false positive word pairings that emanate from a single perspective. On a benchmark dataset of BLI, our proposed method achieves competitive performance compared to existing state-of-the-art (SOTA) methods. It demonstrates effectiveness and robustness across six experimental languages, including similar language pairs and distant language pairs, under both supervised and unsupervised settings.

----

## [1996] From Retrieval to Generation: A Simple and Unified Generative Model for End-to-End Task-Oriented Dialogue

**Authors**: *Zeyuan Ding, Zhihao Yang, Ling Luo, Yuanyuan Sun, Hongfei Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29745](https://doi.org/10.1609/aaai.v38i16.29745)

**Abstract**:

Retrieving appropriate records from the external knowledge base to generate informative responses is the core capability of end-to-end task-oriented dialogue systems (EToDs). Most of the existing methods additionally train the retrieval model or use the memory network to retrieve the knowledge base, which decouples the knowledge retrieval task from the response generation task, making it difficult to jointly optimize and failing to capture the internal relationship between the two tasks. In this paper, we propose a simple and unified generative model for task-oriented dialogue systems, which recasts the EToDs task as a single sequence generation task and uses maximum likelihood training to train the two tasks in a unified manner. To prevent the generation of non-existent records, we design the prefix trie to constrain the model generation, which ensures consistency between the generated records and the existing records in the knowledge base. Experimental results on three public benchmark datasets demonstrate that our method achieves robust performance on generating system responses and outperforms the baseline systems. To facilitate future research in this area, the code is available at https://github.com/dzy1011/Uni-ToD.

----

## [1997] How to Trade Off the Quantity and Capacity of Teacher Ensemble: Learning Categorical Distribution to Stochastically Employ a Teacher for Distillation

**Authors**: *Zixiang Ding, Guoqing Jiang, Shuai Zhang, Lin Guo, Wei Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29746](https://doi.org/10.1609/aaai.v38i16.29746)

**Abstract**:

We observe two phenomenons with respect to quantity and capacity: 1) more teacher is not always better for multi-teacher knowledge distillation, and 2) stronger teacher is not always better for single-teacher knowledge distillation. To trade off the quantity and capacity of teacher ensemble, in this paper, we propose a new distillation paradigm named Dynamic Knowledge Distillation (DynaKD) that learn an adaptive categorical distribution to stochastically employ a teacher from a teacher ensemble in each step, to transfer knowledge from teacher ensemble into student. DynaKD has three advantages: 1) it can preserve diversity of each teacher via one-to-one distillation manner instead of several-for-one, 2) it can make the best of powerful teacher via those multi-level assistant teachers in ensemble, and 3) it can also dynamically determine the importance of each teacher for various tasks. To verify the effectiveness of the proposed approach, we conduct extensive experiments for BERT compression on GLUE benchmark. Experimental results show that the proposed approach achieves state-of-the-art score compared to previous compression approaches on five out of seven downstream tasks, including pushing MRPC F1 and accuracy to 92.2 (1.4 point absolute improvement), RTE accuracy to 76.2 (2.8 point absolute improvement). Moreover, we conduct also extensive experiments for image classification on CIFAR-100. Similarly, DynaKD achieves also state-of-the-art performance.

----

## [1998] UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding

**Authors**: *Chenpeng Du, Yiwei Guo, Feiyu Shen, Zhijun Liu, Zheng Liang, Xie Chen, Shuai Wang, Hui Zhang, Kai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29747](https://doi.org/10.1609/aaai.v38i16.29747)

**Abstract**:

The utilization of discrete speech tokens, divided into semantic tokens and acoustic tokens, has been proven superior to traditional acoustic feature mel-spectrograms in terms of naturalness and robustness for text-to-speech (TTS) synthesis. Recent popular models, such as VALL-E and SPEAR-TTS, allow zero-shot speaker adaptation through auto-regressive (AR) continuation of acoustic tokens extracted from a short speech prompt. However, these AR models are restricted to generate speech only in a left-to-right direction, making them unsuitable for speech editing where both preceding and following contexts are provided. Furthermore, these models rely on acoustic tokens, which have audio quality limitations imposed by the performance of audio codec models. In this study, we propose a unified context-aware TTS framework called UniCATS, which is capable of both speech continuation and editing. UniCATS comprises two components, an acoustic model CTX-txt2vec and a vocoder CTX-vec2wav. CTX-txt2vec employs contextual VQ-diffusion to predict semantic tokens from the input text, enabling it to incorporate the semantic context and maintain seamless concatenation with the surrounding context. Following that, CTX-vec2wav utilizes contextual vocoding to convert these semantic tokens into waveforms, taking into consideration the acoustic context. Our experimental results demonstrate that CTX-vec2wav outperforms HifiGAN and AudioLM in terms of speech resynthesis from semantic tokens. Moreover, we show that UniCATS achieves state-of-the-art performance in both speech continuation and editing. Audio samples are available at https://cpdu.github.io/unicats.

----

## [1999] DocMSU: A Comprehensive Benchmark for Document-Level Multimodal Sarcasm Understanding

**Authors**: *Hang Du, Guoshun Nan, Sicheng Zhang, Binzhu Xie, Junrui Xu, Hehe Fan, Qimei Cui, Xiaofeng Tao, Xudong Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29748](https://doi.org/10.1609/aaai.v38i16.29748)

**Abstract**:

Multimodal Sarcasm Understanding (MSU) has a wide range of applications in the news field such as public opinion analysis and forgery detection. 
However, existing MSU benchmarks and approaches usually focus on sentence-level MSU. 
In document-level news, sarcasm clues are sparse or small and are often concealed in long text. 
Moreover, compared to sentence-level comments like tweets, which mainly focus on only a few trends or hot topics (e.g., sports events), content in the news is considerably diverse.  
Models created for sentence-level MSU may fail to capture sarcasm clues in document-level news. 
To fill this gap, we present a comprehensive benchmark for Document-level Multimodal Sarcasm Understanding (DocMSU). 
Our dataset contains 102,588 pieces of news with text-image pairs, covering 9 diverse topics such as  health, business, etc.
The proposed large-scale and diverse DocMSU significantly facilitates the research of document-level MSU in real-world scenarios. 
To take on the new challenges posed by DocMSU, we introduce a fine-grained sarcasm comprehension method to properly align the pixel-level image features with word-level textual features in documents. 
Experiments demonstrate the effectiveness of our method, showing that it can serve as a baseline approach to the challenging DocMSU.

----



[Go to the previous page](AAAI-2024-list09.md)

[Go to the next page](AAAI-2024-list11.md)

[Go to the catalog section](README.md)