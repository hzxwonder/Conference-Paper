## [1000] Online Hyperparameter Optimization for Class-Incremental Learning

**Authors**: *Yaoyao Liu, Yingying Li, Bernt Schiele, Qianru Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26070](https://doi.org/10.1609/aaai.v37i7.26070)

**Abstract**:

Class-incremental learning (CIL) aims to train a classification model while the number of classes increases phase-by-phase. An inherent challenge of CIL is the stability-plasticity tradeoff, i.e., CIL models should keep stable to retain old knowledge and keep plastic to absorb new knowledge. However, none of the existing CIL models can achieve the optimal tradeoff in different data-receiving settings—where typically the training-from-half (TFH) setting needs more stability, but the training-from-scratch (TFS) needs more plasticity. To this end, we design an online learning method that can adaptively optimize the tradeoff without knowing the setting as a priori. Specifically, we first introduce the key hyperparameters that influence the tradeoff, e.g., knowledge distillation (KD) loss weights, learning rates, and classifier types. Then, we formulate the hyperparameter optimization process as an online Markov Decision Process (MDP) problem and propose a specific algorithm to solve it. We apply local estimated rewards and a classic bandit algorithm Exp3 to address the issues when applying online MDP methods to the CIL protocol. Our method consistently improves top-performing CIL methods in both TFH and TFS settings, e.g., boosting the average accuracy of TFH and TFS by 2.2 percentage points on ImageNet-Full, compared to the state-of-the-art. Code is provided at https://class-il.mpi-inf.mpg.de/online/

----

## [1001] Hard Sample Aware Network for Contrastive Deep Graph Clustering

**Authors**: *Yue Liu, Xihong Yang, Sihang Zhou, Xinwang Liu, Zhen Wang, Ke Liang, Wenxuan Tu, Liang Li, Jingcan Duan, Cancan Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26071](https://doi.org/10.1609/aaai.v37i7.26071)

**Abstract**:

Contrastive deep graph clustering, which aims to divide nodes into disjoint groups via contrastive mechanisms, is a challenging research spot. Among the recent works, hard sample mining-based algorithms have achieved great attention for their promising performance. However, we find that the existing hard sample mining methods have two problems as follows. 1) In the hardness measurement, the important structural information is overlooked for similarity calculation, degrading the representativeness of the selected hard negative samples. 2) Previous works merely focus on the hard negative sample pairs while neglecting the hard positive sample pairs. Nevertheless, samples within the same cluster but with low similarity should also be carefully learned. To solve the problems, we propose a novel contrastive deep graph clustering method dubbed Hard Sample Aware Network (HSAN) by introducing a comprehensive similarity measure criterion and a general dynamic sample weighing strategy. Concretely, in our algorithm, the similarities between samples are calculated by considering both the attribute embeddings and the structure embeddings, better revealing sample relationships and assisting hardness measurement. Moreover, under the guidance of the carefully collected high-confidence clustering information, our proposed weight modulating function will first recognize the positive and negative samples and then dynamically up-weight the hard sample pairs while down-weighting the easy ones. In this way, our method can mine not only the hard negative samples but also the hard positive sample, thus improving the discriminative capability of the samples further. Extensive experiments and analyses demonstrate the superiority and effectiveness of our proposed method.  The source code of HSAN is shared at https://github.com/yueliu1999/HSAN and a collection (papers, codes and, datasets) of deep graph clustering is shared at https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering on Github.

----

## [1002] Temporal-Frequency Co-training for Time Series Semi-supervised Learning

**Authors**: *Zhen Liu, Qianli Ma, Peitian Ma, Linghao Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26072](https://doi.org/10.1609/aaai.v37i7.26072)

**Abstract**:

Semi-supervised learning (SSL) has been actively studied due to its ability to alleviate the reliance of deep learning models on labeled data. Although existing SSL methods based on pseudo-labeling strategies have made great progress, they rarely consider time-series data's intrinsic properties (e.g., temporal dependence). Learning representations by mining the inherent properties of time series has recently gained much attention. Nonetheless, how to utilize feature representations to design SSL paradigms for time series has not been explored. To this end, we propose a Time Series SSL framework via Temporal-Frequency Co-training (TS-TFC), leveraging the complementary information from two distinct views for unlabeled data learning. In particular, TS-TFC employs time-domain and frequency-domain views to train two deep neural networks simultaneously, and each view's pseudo-labels generated by label propagation in the representation space are adopted to guide the training of the other view's classifier. To enhance the discriminative of representations between categories, we propose a temporal-frequency supervised contrastive learning module, which integrates the learning difficulty of categories to improve the quality of pseudo-labels. Through co-training the pseudo-labels obtained from temporal-frequency representations, the complementary information in the two distinct views is exploited to enable the model to better learn the distribution of categories. Extensive experiments on 106 UCR datasets show that TS-TFC outperforms state-of-the-art methods, demonstrating the effectiveness and robustness of our proposed model.

----

## [1003] Q-functionals for Value-Based Continuous Control

**Authors**: *Samuel Lobel, Sreehari Rammohan, Bowen He, Shangqun Yu, George Konidaris*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26073](https://doi.org/10.1609/aaai.v37i7.26073)

**Abstract**:

We present Q-functionals, an alternative architecture for continuous control deep reinforcement learning. Instead of returning a single value for a state-action pair, our network transforms a state into a function that can be rapidly evaluated in parallel for many actions, allowing us to efficiently choose high-value actions through sampling. This contrasts with the typical architecture of off-policy continuous control, where a policy network is trained for the sole purpose of selecting actions from the Q-function. We represent our action-dependent Q-function as a weighted sum of basis functions (Fourier, Polynomial, etc) over the action space, where the weights are state-dependent and output by the Q-functional network. Fast sampling makes practical a variety of techniques that require Monte-Carlo integration over Q-functions, and enables action-selection strategies besides simple value-maximization. We characterize our framework, describe various implementations of Q-functionals, and demonstrate strong performance on a suite of continuous control tasks.

----

## [1004] A Coreset Learning Reality Check

**Authors**: *Fred Lu, Edward Raff, James Holt*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26074](https://doi.org/10.1609/aaai.v37i7.26074)

**Abstract**:

Subsampling algorithms are a natural approach to reduce data size before fitting models on massive datasets. In recent years, several works have proposed methods for subsampling rows from a data matrix while maintaining relevant information for classification. While these works are supported by theory and limited experiments, to date there has not been a comprehensive evaluation of these methods. In our work, we directly compare multiple methods for logistic regression drawn from the coreset and optimal subsampling literature and discover inconsistencies in their effectiveness. In many cases, methods do not outperform simple uniform subsampling.

----

## [1005] Centerless Multi-View K-means Based on the Adjacency Matrix

**Authors**: *Han Lu, Quanxue Gao, Qianqian Wang, Ming Yang, Wei Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26075](https://doi.org/10.1609/aaai.v37i7.26075)

**Abstract**:

Although K-Means clustering has been widely studied due to its simplicity, these methods still have the following fatal drawbacks. Firstly, they need to initialize the cluster centers, which causes unstable clustering performance. Secondly, they have poor performance on non-Gaussian datasets. Inspired by the affinity matrix, we propose a novel multi-view K-Means based on the adjacency matrix. It maps the affinity matrix to the distance matrix according to the principle that every sample has a small distance from the points in its neighborhood and a large distance from the points outside of the neighborhood. Moreover, this method well exploits the complementary information embedded in different views by minimizing the tensor Schatten p-norm regularize on the third-order tensor which consists of cluster assignment matrices of different views. Additionally, this method avoids initializing cluster centroids to obtain stable performance. And there is no need to compute the means of clusters so that our model is not sensitive to outliers. Experiment on a toy dataset shows the excellent performance on non-Gaussian datasets. And other experiments on several benchmark datasets demonstrate the superiority of our proposed method.

----

## [1006] PINAT: A Permutation INvariance Augmented Transformer for NAS Predictor

**Authors**: *Shun Lu, Yu Hu, Peihao Wang, Yan Han, Jianchao Tan, Jixiang Li, Sen Yang, Ji Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26076](https://doi.org/10.1609/aaai.v37i7.26076)

**Abstract**:

Time-consuming performance evaluation is the bottleneck of traditional Neural Architecture Search (NAS) methods. Predictor-based NAS can speed up performance evaluation by directly predicting performance, rather than training a large number of sub-models and then validating their performance. Most predictor-based NAS approaches use a proxy dataset to train model-based predictors efficiently but suffer from performance degradation and generalization problems. We attribute these problems to the poor abilities of existing predictors to character the sub-models' structure, specifically the topology information extraction and the node feature representation of the input graph data. To address these problems, we propose a Transformer-like NAS predictor PINAT, consisting of a Permutation INvariance Augmentation module serving as both token embedding layer and self-attention head, as well as a Laplacian matrix to be the positional encoding. Our design produces more representative features of the encoded architecture and outperforms state-of-the-art NAS predictors on six search spaces: NAS-Bench-101, NAS-Bench-201, DARTS, ProxylessNAS, PPI, and ModelNet. The code is available at https://github.com/ShunLu91/PINAT.

----

## [1007] Multi-View Domain Adaptive Object Detection on Camera Networks

**Authors**: *Yan Lu, Zhun Zhong, Yuanchao Shu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26077](https://doi.org/10.1609/aaai.v37i7.26077)

**Abstract**:

In this paper, we study a new domain adaptation setting on camera networks, namely Multi-View Domain Adaptive Object Detection (MVDA-OD), in which labeled source data is unavailable in the target adaptation process and target data is captured from multiple overlapping cameras. In such a challenging context, existing methods including adversarial training and self-training fall short due to multi-domain data shift and the lack of source data. To tackle this problem, we propose a novel training framework consisting of two stages. First, we pre-train the backbone using self-supervised learning, in which a multi-view association is developed to construct an effective pretext task. Second, we fine-tune the detection head using robust self-training, where a tracking-based single-view augmentation is introduced to achieve weak-hard consistency learning. By doing so, an object detection model can take advantage of informative samples generated by multi-view association and single-view augmentation to learn discriminative backbones as well as robust detection classifiers. Experiments on two real-world multi-camera datasets demonstrate significant advantages of our approach over the state-of-the-art domain adaptive object detection methods.

----

## [1008] Generative Label Enhancement with Gaussian Mixture and Partial Ranking

**Authors**: *Yunan Lu, Liang He, Fan Min, Weiwei Li, Xiuyi Jia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26078](https://doi.org/10.1609/aaai.v37i7.26078)

**Abstract**:

Label distribution learning (LDL) is an effective learning paradigm for dealing with label ambiguity. When applying LDL, the datasets annotated with label distributions (i.e., the real-valued vectors like the probability distribution) are typically required. Unfortunately, most existing datasets only contain the logical labels, and manual annotating with label distributions is costly. To address this problem, we treat the label distribution as a latent vector and infer its posterior by variational Bayes. Specifically, we propose a generative label enhancement model to encode the process of generating feature vectors and logical label vectors from label distributions in a principled way. In terms of features, we assume that the feature vector is generated by a Gaussian mixture dominated by the label distribution, which captures the one-to-many relationship from the label distribution to the feature vector and thus reduces the feature generation error. In terms of logical labels, we design a probability distribution to generate the logical label vector from a label distribution, which captures partial label ranking in the logical label vector and thus provides a more accurate guidance for inferring the label distribution. Besides, to approximate the posterior of the label distribution, we design a inference model, and derive the variational learning objective. Finally, extensive experiments on real-world datasets validate our proposal.

----

## [1009] Crowd-Level Abnormal Behavior Detection via Multi-Scale Motion Consistency Learning

**Authors**: *Linbo Luo, Yuanjing Li, Haiyan Yin, Shangwei Xie, Ruimin Hu, Wentong Cai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26079](https://doi.org/10.1609/aaai.v37i7.26079)

**Abstract**:

Detecting abnormal crowd motion emerging from complex interactions of individuals is paramount to ensure the safety of crowds. Crowd-level abnormal behaviors (CABs), e.g., counter flow and crowd turbulence, are proven to be the crucial causes of many crowd disasters. In the recent decade, video anomaly detection (VAD) techniques have achieved remarkable success in detecting individual-level abnormal behaviors (e.g., sudden running, fighting and stealing), but research on VAD for CABs is rather limited. Unlike individual-level anomaly, CABs usually do not exhibit salient difference from the normal behaviors when observed locally, and the scale of CABs could vary from one scenario to another. In this paper, we present a systematic study to tackle the important problem of VAD for CABs with a novel crowd motion learning framework, multi-scale motion consistency network (MSMC-Net). MSMC-Net first captures the spatial and temporal crowd motion consistency information in a graph representation. Then, it simultaneously trains multiple feature graphs constructed at different scales to capture rich crowd patterns. An attention network is used to adaptively fuse the multi-scale features for better CAB detection. For the empirical study, we consider three large-scale crowd event datasets, UMN, Hajj and Love Parade. Experimental results show that MSMC-Net could substantially improve the state-of-the-art performance on all the datasets.

----

## [1010] MVCINN: Multi-View Diabetic Retinopathy Detection Using a Deep Cross-Interaction Neural Network

**Authors**: *Xiaoling Luo, Chengliang Liu, Waikeung Wong, Jie Wen, Xiaopeng Jin, Yong Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26080](https://doi.org/10.1609/aaai.v37i7.26080)

**Abstract**:

Diabetic retinopathy (DR) is the main cause of irreversible blindness for working-age adults. The previous models for DR detection have difficulties in clinical application. The main reason is that most of the previous methods only use single-view data, and the single field of view (FOV) only accounts for about 13% of the FOV of the retina, resulting in the loss of most lesion features. To alleviate this problem, we propose a multi-view model for DR detection, which takes full advantage of multi-view images covering almost all of the retinal field. To be specific, we design a Cross-Interaction Self-Attention based Module (CISAM) that interfuses local features extracted from convolutional blocks with long-range global features learned from transformer blocks. Furthermore, considering the pathological association in different views, we use the feature jigsaw to assemble and learn the features of multiple views. Extensive experiments on the latest public multi-view MFIDDR dataset with 34,452 images demonstrate the superiority of our method, which performs favorably against state-of-the-art models. To the best of our knowledge, this work is the first study on the public large-scale multi-view fundus images dataset for DR detection.

----

## [1011] Local Explanations for Reinforcement Learning

**Authors**: *Ronny Luss, Amit Dhurandhar, Miao Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26081](https://doi.org/10.1609/aaai.v37i7.26081)

**Abstract**:

Many works in explainable AI have focused on explaining black-box classification models. Explaining deep reinforcement learning (RL) policies in a manner that could be understood by domain users has received much less attention. In this paper, we propose a novel perspective to understanding RL policies based on identifying important states from automatically learned meta-states. The key conceptual difference between our approach and many previous ones is that we form meta-states based on locality governed by the expert policy dynamics rather than based on similarity of actions, and that we do not assume any particular knowledge of the underlying topology of the state space. Theoretically, we show that our algorithm to find meta-states converges and the objective that selects important states from each meta-state is submodular leading to efficient high quality greedy selection. Experiments on four domains (four rooms, door-key, minipacman, and pong) and a carefully conducted user study illustrate that our perspective leads to better understanding of the policy. We conjecture that this is a result of our meta-states being more intuitive in that the corresponding important states are strong indicators of tractable intermediate goals that are easier for humans to interpret and follow.

----

## [1012] Compositional Prototypical Networks for Few-Shot Classification

**Authors**: *Qiang Lyu, Weiqiang Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26082](https://doi.org/10.1609/aaai.v37i7.26082)

**Abstract**:

It is assumed that pre-training provides the feature extractor with strong class transferability and that high novel class generalization can be achieved by simply reusing the transferable feature extractor. In this work, our motivation is to explicitly learn some fine-grained and transferable meta-knowledge so that feature reusability can be further improved. Concretely, inspired by the fact that humans can use learned concepts or components to help them recognize novel classes, we propose Compositional Prototypical Networks (CPN) to learn a transferable prototype for each human-annotated attribute, which we call a component prototype. We empirically demonstrate that the learned component prototypes have good class transferability and can be reused to construct compositional prototypes for novel classes. Then a learnable weight generator is utilized to adaptively fuse the compositional and visual prototypes. Extensive experiments demonstrate that our method can achieve state-of-the-art results on different datasets and settings. The performance gains are especially remarkable in the 5-way 1-shot setting. The code is available at https://github.com/fikry102/CPN.

----

## [1013] Poisoning with Cerberus: Stealthy and Colluded Backdoor Attack against Federated Learning

**Authors**: *Xiaoting Lyu, Yufei Han, Wei Wang, Jingkai Liu, Bin Wang, Jiqiang Liu, Xiangliang Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26083](https://doi.org/10.1609/aaai.v37i7.26083)

**Abstract**:

Are Federated Learning (FL) systems free from backdoor poisoning with the arsenal of various defense strategies deployed? This is an intriguing problem with significant practical implications regarding the utility of FL services. Despite the recent flourish of poisoning-resilient FL methods, our study shows that carefully tuning the collusion between malicious participants can minimize the trigger-induced bias of the poisoned local model from the poison-free one, which plays the key role in delivering stealthy backdoor attacks and circumventing a wide spectrum of state-of-the-art defense methods in FL. In our work, we instantiate the attack strategy by proposing a distributed backdoor attack method, namely Cerberus Poisoning (CerP). It jointly tunes the backdoor trigger and controls the poisoned model changes on each malicious participant to achieve a stealthy yet successful backdoor attack against a wide spectrum of defensive mechanisms of federated learning techniques. Our extensive study on 3 large-scale benchmark datasets and 13 mainstream defensive mechanisms confirms that Cerberus Poisoning raises a significantly severe threat to the integrity and security of federated learning practices, regardless of the flourish of robust Federated Learning methods.

----

## [1014] OMPQ: Orthogonal Mixed Precision Quantization

**Authors**: *Yuexiao Ma, Taisong Jin, Xiawu Zheng, Yan Wang, Huixia Li, Yongjian Wu, Guannan Jiang, Wei Zhang, Rongrong Ji*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26084](https://doi.org/10.1609/aaai.v37i7.26084)

**Abstract**:

To bridge the ever-increasing gap between deep neural networks' complexity and hardware capability, network quantization has attracted more and more research attention. The latest trend of mixed precision quantization takes advantage of hardware's multiple bit-width arithmetic operations to unleash the full potential of network quantization. However, existing approaches rely heavily on an extremely time-consuming search process and various relaxations when seeking the optimal bit configuration. To address this issue, we propose to optimize a proxy metric of network orthogonality that can be efficiently solved with linear programming, which proves to be highly correlated with quantized model accuracy and bit-width. Our approach significantly reduces the search time and the required data amount by orders of magnitude, but without a compromise on quantization accuracy. Specifically, we achieve 72.08% Top-1 accuracy on ResNet-18 with 6.7Mb parameters, which does not require any searching iterations. Given the high efficiency and low data dependency of our algorithm, we use it for the post-training quantization, which achieves 71.27% Top-1 accuracy on MobileNetV2 with only 1.5Mb parameters.

----

## [1015] Recovering the Graph Underlying Networked Dynamical Systems under Partial Observability: A Deep Learning Approach

**Authors**: *Sergio Machado, Anirudh Sridhar, Paulo Gil, Jorge Henriques, José M. F. Moura, Augusto Santos*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26085](https://doi.org/10.1609/aaai.v37i7.26085)

**Abstract**:

We study the problem of graph structure identification, i.e., of recovering the graph of dependencies among time series. We model these time series data as components of the state of linear stochastic networked dynamical systems. We assume partial observability, where the state evolution of only a subset of nodes comprising the network is observed. We propose a new feature-based paradigm: to each pair of nodes, we compute a feature vector from the observed time series. We prove that these features are linearly separable, i.e., there exists a hyperplane that separates the cluster of features associated with connected pairs of nodes from those of disconnected pairs. This renders the features amenable to train a variety of classifiers to perform causal inference. In particular, we use these features to train Convolutional Neural Networks (CNNs). The resulting causal inference mechanism outperforms state-of-the-art counterparts w.r.t. sample-complexity. The trained CNNs generalize well over structurally distinct networks (dense or sparse) and noise-level profiles. Remarkably, they also generalize well to real-world networks while trained over a synthetic network -- namely, a particular realization of a random graph.

----

## [1016] LIMIP: Lifelong Learning to Solve Mixed Integer Programs

**Authors**: *Sahil Manchanda, Sayan Ranu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26086](https://doi.org/10.1609/aaai.v37i7.26086)

**Abstract**:

Mixed Integer programs (MIPs) are typically solved by the Branch-and-Bound algorithm. Recently, Learning to imitate fast approximations of the expert strong branching heuristic has gained attention due to its success in reducing the running time for solving MIPs. However, existing learning-to-branch methods assume that the entire training data is available in a single session of training. This assumption is often not true, and if the training data is supplied in continual fashion over time, existing techniques suffer from catastrophic forgetting. In this work, we study the hitherto unexplored paradigm of Lifelong Learning to Branch on Mixed Integer Programs. To mitigate catastrophic forgetting, we propose LIMIP, which is powered by the idea of modeling an MIP instance in the form of a bipartite graph, which we map to an embedding space using a bipartite Graph Attention Network. This rich embedding space avoids catastrophic forgetting through the application of knowledge distillation and elastic weight consolidation, wherein we learn the parameters key towards retaining efficacy and are therefore protected from significant drift. We evaluate LIMIP on a series of NP-hard problems and establish that in comparison to existing baselines, LIMIP is up to 50% better when confronted with lifelong learning

----

## [1017] Proximal Stochastic Recursive Momentum Methods for Nonconvex Composite Decentralized Optimization

**Authors**: *Gabriel Mancino-Ball, Shengnan Miao, Yangyang Xu, Jie Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26087](https://doi.org/10.1609/aaai.v37i7.26087)

**Abstract**:

Consider a network of N decentralized computing agents collaboratively solving a nonconvex stochastic composite problem. In this work, we propose a single-loop algorithm, called DEEPSTORM, that achieves optimal sample complexity for this setting. Unlike double-loop algorithms that require a large batch size to compute the (stochastic) gradient once in a while, DEEPSTORM uses a small batch size, creating advantages in occasions such as streaming data and online learning. This is the first method achieving optimal sample complexity for decentralized nonconvex stochastic composite problems, requiring O(1) batch size. We conduct convergence analysis for DEEPSTORM with both constant and diminishing step sizes. Additionally, under proper initialization and a small enough desired solution error, we show that DEEPSTORM with a constant step size achieves a network-independent sample complexity, with an additional linear speed-up with respect to N over centralized methods. All codes are made available at https://github.com/gmancino/DEEPSTORM.

----

## [1018] Online Reinforcement Learning with Uncertain Episode Lengths

**Authors**: *Debmalya Mandal, Goran Radanovic, Jiarui Gan, Adish Singla, Rupak Majumdar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i7.26088](https://doi.org/10.1609/aaai.v37i7.26088)

**Abstract**:

Existing episodic reinforcement algorithms assume that the  length of an episode is fixed across time and known a priori. In this paper, we consider a general framework of  episodic reinforcement learning when the length of each episode is drawn from a distribution. We first establish that this problem is equivalent to online reinforcement learning with general discounting where the learner is trying to optimize the expected discounted sum of rewards over an infinite horizon, but where the discounting function is not necessarily geometric. We show that minimizing regret with this new general discounting is equivalent to minimizing regret with uncertain episode lengths. We then design a reinforcement learning algorithm that minimizes regret with general discounting but acts for the setting with uncertain episode lengths. We instantiate  our general bound for different types of discounting, including geometric and polynomial discounting. We also show that we can obtain similar regret bounds even when the uncertainty over the episode lengths is unknown, by estimating the unknown distribution over time. Finally, we compare our learning algorithms with existing value-iteration based episodic RL algorithms on a grid-world environment.

----

## [1019] Tight Performance Guarantees of Imitator Policies with Continuous Actions

**Authors**: *Davide Maran, Alberto Maria Metelli, Marcello Restelli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26089](https://doi.org/10.1609/aaai.v37i8.26089)

**Abstract**:

Behavioral Cloning (BC) aims at learning a policy that mimics the behavior demonstrated by an expert. The current theoretical understanding of BC is limited to the case of finite actions. In this paper, we study BC with the goal of providing theoretical guarantees on the performance of the imitator policy in the case of continuous actions. We start by deriving a novel bound on the performance gap based on Wasserstein distance, applicable for continuous-action experts, holding under the assumption that the value function is Lipschitz continuous. Since this latter condition is hardy fulfilled in practice, even for Lipschitz Markov Decision Processes and policies, we propose a relaxed setting, proving that value function is always H\"older continuous. This result is of independent interest and allows obtaining in BC a general bound for the performance of the imitator policy. Finally, we analyze noise injection, a common practice in which the expert's action is executed in the environment after the application of a noise kernel. We show that this practice allows deriving stronger performance guarantees, at the price of a bias due to the noise addition.

----

## [1020] Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data

**Authors**: *Andrei Margeloiu, Nikola Simidjievski, Pietro Liò, Mateja Jamnik*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26090](https://doi.org/10.1609/aaai.v37i8.26090)

**Abstract**:

Tabular biomedical data is often high-dimensional but with a very small number of samples. Although recent work showed that well-regularised simple neural networks could outperform more sophisticated architectures on tabular data, they are still prone to overfitting on tiny datasets with many potentially irrelevant features. To combat these issues, we propose Weight Predictor Network with Feature Selection (WPFS) for learning neural networks from high-dimensional and small sample data by reducing the number of learnable parameters and simultaneously performing feature selection. In addition to the classification network, WPFS uses two small auxiliary networks that together output the weights of the first layer of the classification model. We evaluate on nine real-world biomedical datasets and demonstrate that WPFS outperforms other standard as well as more recent methods typically applied to tabular data. Furthermore, we investigate the proposed feature selection mechanism and show that it improves performance while providing useful insights into the learning task.

----

## [1021] Learning Revenue Maximization Using Posted Prices for Stochastic Strategic Patient Buyers

**Authors**: *Eitan-Hai Mashiah, Idan Attias, Yishay Mansour*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26091](https://doi.org/10.1609/aaai.v37i8.26091)

**Abstract**:

We consider a seller faced with buyers which have the ability to delay their decision, which we call patience.
Each buyer's type is composed of value and patience, and it is sampled i.i.d. from a distribution.
The seller, using posted prices, would like to maximize her revenue from selling to the buyer. 
In this paper, we formalize this setting and characterize the resulting Stackelberg equilibrium, where the seller first commits to her strategy, and then the buyers best respond. Following this, we show how to compute both the optimal pure and mixed strategies. 
We then consider a learning setting, where the seller does not have access to the distribution over buyer's types. Our main results are the following. We derive a sample complexity bound for the learning of an approximate optimal pure strategy, by computing the fat-shattering dimension of this setting. Moreover, we provide a general sample complexity bound for the approximate optimal mixed strategy. 
We also consider an online setting and derive a vanishing regret bound with respect to both the optimal pure strategy and the optimal mixed strategy.

----

## [1022] Boundary Graph Neural Networks for 3D Simulations

**Authors**: *Andreas Mayr, Sebastian Lehner, Arno Mayrhofer, Christoph Kloss, Sepp Hochreiter, Johannes Brandstetter*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26092](https://doi.org/10.1609/aaai.v37i8.26092)

**Abstract**:

The abundance of data has given machine learning considerable momentum in natural sciences and engineering, though modeling of physical processes is often difficult. A particularly tough problem is the efficient representation of geometric boundaries. Triangularized geometric boundaries are well understood and ubiquitous in engineering applications. However, it is notoriously difficult to integrate them into machine learning approaches due to their heterogeneity with respect to size and orientation. In this work, we introduce an effective theory to model particle-boundary interactions, which leads to our new Boundary Graph Neural Networks (BGNNs) that dynamically modify graph structures to obey boundary conditions. The new BGNNs are tested on complex 3D granular flow processes of hoppers, rotating drums and mixers, which are all standard components of modern industrial machinery but still have complicated geometry. BGNNs are evaluated in terms of computational efficiency as well as prediction accuracy of particle flows and mixing entropies. BGNNs are able to accurately reproduce 3D granular flows within simulation uncertainties over hundreds of thousands of simulation timesteps. Most notably, in our experiments, particles stay within the geometric objects without using handcrafted conditions or restrictions.

----

## [1023] Diffusion Models Beat GANs on Topology Optimization

**Authors**: *François Mazé, Faez Ahmed*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26093](https://doi.org/10.1609/aaai.v37i8.26093)

**Abstract**:

Structural topology optimization, which aims to find the optimal physical structure that maximizes mechanical performance, is vital in engineering design applications in aerospace, mechanical, and civil engineering. Recently, generative adversarial networks (GANs) have emerged as a popular alternative to traditional iterative topology optimization methods. However, GANs can be challenging to train, have limited generalizability, and often neglect important performance objectives such as mechanical compliance and manufacturability. To address these issues, we propose a new architecture called TopoDiff that uses conditional diffusion models to perform performance-aware and manufacturability-aware topology optimization. Our method introduces a surrogate model-based guidance strategy that actively favors structures with low compliance and good manufacturability. Compared to a state-of-the-art conditional GAN, our approach reduces the average error on physical performance by a factor of eight and produces eleven times fewer infeasible samples. Our work demonstrates the potential of using diffusion models in topology optimization and suggests a general framework for solving engineering optimization problems using external performance with constraint-aware guidance. We provide access to our data, code, and trained models at the following link: https://decode.mit.edu/projects/topodiff/.

----

## [1024] VIDM: Video Implicit Diffusion Models

**Authors**: *Kangfu Mei, Vishal M. Patel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26094](https://doi.org/10.1609/aaai.v37i8.26094)

**Abstract**:

Diffusion models have emerged as a powerful generative method for synthesizing high-quality and diverse set of images. In this paper, we propose a video generation method based on diffusion models, where the effects of motion are modeled in an implicit condition manner, i.e. one can sample plausible video motions according to the latent feature of frames. We improve the quality of the generated videos by proposing multiple strategies such as sampling space truncation, robustness penalty, and positional group normalization. Various experiments are conducted on datasets consisting of videos with different resolutions and different number of frames. Results show that the proposed method outperforms the state-of-the-art generative adversarial network-based methods by a significant margin in terms of FVD scores as well as perceptible visual quality.

----

## [1025] Towards Interpreting and Utilizing Symmetry Property in Adversarial Examples

**Authors**: *Shibin Mei, Chenglong Zhao, Bingbing Ni, Shengchao Yuan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26095](https://doi.org/10.1609/aaai.v37i8.26095)

**Abstract**:

In this paper, we identify symmetry property in adversarial scenario by viewing adversarial attack in a fine-grained manner. A newly designed metric called attack proportion, is thus proposed to count the proportion of the adversarial examples misclassified between classes. We observe that the distribution of attack proportion is unbalanced as each class shows vulnerability to particular classes. Further, some class pairs correlate strongly and have the same degree of attack proportion for each other. We call this intriguing phenomenon symmetry property. We empirically prove this phenomenon is widespread and then analyze the reason behind the existence of symmetry property. This explanation, to some extent, could be utilized to understand robust models, which also inspires us to strengthen adversarial defenses.

----

## [1026] The Unreasonable Effectiveness of Deep Evidential Regression

**Authors**: *Nis Meinert, Jakob Gawlikowski, Alexander Lavin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26096](https://doi.org/10.1609/aaai.v37i8.26096)

**Abstract**:

There is a significant need for principled uncertainty reasoning in machine learning systems as they are increasingly deployed in safety-critical domains.
A new approach with uncertainty-aware regression-based neural networks (NNs), based on learning evidential distributions for aleatoric and epistemic uncertainties, shows promise over traditional deterministic methods and typical Bayesian NNs, notably with the capabilities to disentangle aleatoric and epistemic uncertainties.
Despite some empirical success of Deep Evidential Regression (DER), there are important gaps in the mathematical foundation that raise the question of why the proposed technique seemingly works.
We detail the theoretical shortcomings and analyze the performance on synthetic and real-world data sets, showing that Deep Evidential Regression is a heuristic rather than an exact uncertainty quantification.
We go on to discuss corrections and redefinitions of how aleatoric and epistemic uncertainties should be extracted from NNs.

----

## [1027] HyperJump: Accelerating HyperBand via Risk Modelling

**Authors**: *Pedro Mendes, Maria Casimiro, Paolo Romano, David Garlan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26097](https://doi.org/10.1609/aaai.v37i8.26097)

**Abstract**:

In the literature on hyper-parameter tuning, a number of recent solutions rely on low-fidelity observations (e.g., training with sub-sampled datasets) to identify promising configurations to be tested via high-fidelity observations (e.g., using the full dataset). Among these, HyperBand is arguably one of the most popular solutions, due to its efficiency and theoretically provable robustness. In this work, we introduce HyperJump, a new approach that builds on HyperBand’s robust search strategy and complements it with novel model-based risk analysis techniques that accelerate the search by skipping the evaluation of low risk configurations, i.e., configurations that are likely to be eventually discarded by HyperBand. We evaluate HyperJump on a suite of hyper-parameter optimization problems and show that it provides over one-order of magnitude speed-ups, both in sequential and parallel deployments, on a variety of deep-learning, kernel-based learning and neural architectural search problems when compared to HyperBand and to several state-of-the-art optimizers.

----

## [1028] MHCCL: Masked Hierarchical Cluster-Wise Contrastive Learning for Multivariate Time Series

**Authors**: *Qianwen Meng, Hangwei Qian, Yong Liu, Lizhen Cui, Yonghui Xu, Zhiqi Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26098](https://doi.org/10.1609/aaai.v37i8.26098)

**Abstract**:

Learning semantic-rich representations from raw unlabeled time series data is critical for downstream tasks such as classification and forecasting. Contrastive learning has recently shown its promising representation learning capability in the absence of expert annotations. However, existing contrastive approaches generally treat each instance independently, which leads to false negative pairs that share the same semantics. To tackle this problem, we propose MHCCL, a Masked Hierarchical Cluster-wise Contrastive Learning model, which exploits semantic information obtained from the hierarchical structure consisting of multiple latent partitions for multivariate time series. Motivated by the observation that fine-grained clustering preserves higher purity while coarse-grained one reflects higher-level semantics, we propose a novel downward masking strategy to filter out fake negatives and supplement positives by incorporating the multi-granularity information from the clustering hierarchy. In addition, a novel upward masking strategy is designed in MHCCL to remove outliers of clusters at each partition to refine prototypes, which helps speed up the hierarchical clustering process and improves the clustering quality. We conduct experimental evaluations on seven widely-used multivariate time series datasets. The results demonstrate the superiority of MHCCL over the state-of-the-art approaches for unsupervised time series representation learning.

----

## [1029] Off-Policy Proximal Policy Optimization

**Authors**: *Wenjia Meng, Qian Zheng, Gang Pan, Yilong Yin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26099](https://doi.org/10.1609/aaai.v37i8.26099)

**Abstract**:

Proximal Policy Optimization (PPO) is an important reinforcement learning method, which has achieved great success in sequential decision-making problems. However, PPO faces the issue of sample inefficiency, which is due to the PPO cannot make use of off-policy data. In this paper, we propose an Off-Policy Proximal Policy Optimization method (Off-Policy PPO) that improves the sample efficiency of PPO by utilizing off-policy data. Specifically, we first propose a clipped surrogate objective function that can utilize off-policy data and avoid excessively large policy updates. Next, we theoretically clarify the stability of the optimization process of the proposed surrogate objective by demonstrating the degree of policy update distance is consistent with that in the PPO. We then describe the implementation details of the proposed Off-Policy PPO which iteratively updates policies by optimizing the proposed clipped surrogate objective. Finally, the experimental results on representative continuous control tasks validate that our method outperforms the state-of-the-art methods on most tasks.

----

## [1030] Information-Theoretic Causal Discovery and Intervention Detection over Multiple Environments

**Authors**: *Osman Mian, Michael Kamp, Jilles Vreeken*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26100](https://doi.org/10.1609/aaai.v37i8.26100)

**Abstract**:

Given multiple datasets over a fixed set of random variables, each collected from a different environment, we are interested in discovering the shared underlying causal network and the local interventions per environment, without assuming prior knowledge on which datasets are observational or interventional, and without assuming the shape of the causal dependencies. We formalize this problem using the Algorithmic Model of Causation, instantiate a consistent score via the Minimum Description Length principle, and show under which conditions the network and interventions are identifiable. To efficiently discover causal networks and intervention targets in practice, we introduce the ORION algorithm, which through extensive experiments we show outperforms the state of the art in causal inference over multiple environments.

----

## [1031] AIO-P: Expanding Neural Performance Predictors beyond Image Classification

**Authors**: *Keith G. Mills, Di Niu, Mohammad Salameh, Weichen Qiu, Fred X. Han, Puyuan Liu, Jialin Zhang, Wei Lu, Shangling Jui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26101](https://doi.org/10.1609/aaai.v37i8.26101)

**Abstract**:

Evaluating neural network performance is critical to deep neural network design but a costly procedure. Neural predictors provide an efficient solution by treating architectures as samples and learning to estimate their performance on a given task. However, existing predictors are task-dependent, predominantly estimating neural network performance on image classification benchmarks. They are also search-space dependent; each predictor is designed to make predictions for a specific architecture search space with predefined topologies and set of operations. In this paper, we propose a novel All-in-One Predictor (AIO-P), which aims to pretrain neural predictors on architecture examples from multiple, separate computer vision (CV) task domains and multiple architecture spaces, and then transfer to unseen downstream CV tasks or neural architectures. We describe our proposed techniques for general graph representation, efficient predictor pretraining and knowledge infusion techniques, as well as methods to transfer to downstream tasks/spaces. Extensive experimental results show that AIO-P can achieve Mean Absolute Error (MAE) and Spearman’s Rank Correlation (SRCC) below 1p% and above 0.5, respectively, on a breadth of target downstream CV tasks with or without fine-tuning, outperforming a number of baselines. Moreover, AIO-P can directly transfer to new architectures not seen during training, accurately rank them and serve as an effective performance estimator when paired with an algorithm designed to preserve performance while reducing FLOPs.

----

## [1032] GENNAPE: Towards Generalized Neural Architecture Performance Estimators

**Authors**: *Keith G. Mills, Fred X. Han, Jialin Zhang, Fabian Chudak, Ali Safari Mamaghani, Mohammad Salameh, Wei Lu, Shangling Jui, Di Niu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26102](https://doi.org/10.1609/aaai.v37i8.26102)

**Abstract**:

Predicting neural architecture performance is a challenging task and is crucial to neural architecture design and search. Existing approaches either rely on neural performance predictors which are limited to modeling architectures in a predefined design space involving specific sets of operators and connection rules, and cannot generalize to unseen architectures, or resort to Zero-Cost Proxies which are not always accurate. In this paper, we propose GENNAPE, a Generalized Neural Architecture Performance Estimator, which is pretrained on open neural architecture benchmarks, and aims to generalize to completely unseen architectures through combined innovations in network representation, contrastive pretraining, and a fuzzy clustering-based predictor ensemble. Specifically, GENNAPE represents a given neural network as a Computation Graph (CG) of atomic operations which can model an arbitrary architecture. It first learns a graph encoder via Contrastive Learning to encourage network separation by topological features, and then trains multiple predictor heads, which are soft-aggregated according to the fuzzy membership of a neural network. Experiments show that GENNAPE pretrained on NAS-Bench-101 can achieve superior transferability to 5 different public neural network benchmarks, including NAS-Bench-201, NAS-Bench-301, MobileNet and ResNet families under no or minimum fine-tuning. We further introduce 3 challenging newly labelled neural network benchmarks: HiAML, Inception and Two-Path, which can concentrate in narrow accuracy ranges. Extensive experiments show that GENNAPE can correctly discern high-performance architectures in these families. Finally, when paired with a search algorithm, GENNAPE can find architectures that improve accuracy while reducing FLOPs on three families.

----

## [1033] Adaptive Perturbation-Based Gradient Estimation for Discrete Latent Variable Models

**Authors**: *Pasquale Minervini, Luca Franceschi, Mathias Niepert*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26103](https://doi.org/10.1609/aaai.v37i8.26103)

**Abstract**:

The integration of discrete algorithmic components in deep learning architectures has numerous applications. Recently, Implicit Maximum Likelihood Estimation, a class of gradient estimators for discrete exponential family distributions, was proposed by combining implicit differentiation through perturbation with the path-wise gradient estimator. However, due to the finite difference approximation of the gradients, it is especially sensitive to the choice of the finite difference step size, which needs to be specified by the user. In this work, we present Adaptive IMLE (AIMLE), the first adaptive gradient estimator for complex discrete distributions: it adaptively identifies the target distribution for IMLE by trading off the density of gradient information with the degree of bias in the gradient estimates. We empirically evaluate our estimator on synthetic examples, as well as on Learning to Explain, Discrete Variational Auto-Encoders, and Neural Relational Inference tasks. In our experiments, we show that our adaptive gradient estimator can produce faithful estimates while requiring orders of magnitude fewer samples than other gradient estimators.

----

## [1034] Why Capsule Neural Networks Do Not Scale: Challenging the Dynamic Parse-Tree Assumption

**Authors**: *Matthias Mitterreiter, Marcel Koch, Joachim Giesen, Sören Laue*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26104](https://doi.org/10.1609/aaai.v37i8.26104)

**Abstract**:

Capsule neural networks replace simple, scalar-valued neurons with vector-valued capsules. They are motivated by the pattern recognition system in the human brain, where complex objects are decomposed into a hierarchy of simpler object parts. Such a hierarchy is referred to as a parse-tree. Conceptually, capsule neural networks have been defined to mimic this behavior. The capsule neural network (CapsNet), by Sabour, Frosst, and Hinton, is the first actual implementation of the conceptual idea of capsule neural networks. CapsNets achieved state-of-the-art performance on simple image recognition tasks with fewer parameters and greater robustness to affine transformations than comparable approaches. This sparked extensive follow-up research. However, despite major efforts, no work was able to scale the CapsNet architecture to more reasonable-sized datasets. Here, we provide a reason for this failure and argue that it is most likely not possible to scale CapsNets beyond toy examples. In particular, we show that the concept of a parse-tree, the main idea behind capsule neuronal networks, is not present in CapsNets. We also show theoretically and experimentally that CapsNets suffer from a vanishing gradient problem that results in the starvation of many capsules during training.

----

## [1035] Multiplex Graph Representation Learning via Common and Private Information Mining

**Authors**: *Yujie Mo, Zongqian Wu, Yuhuan Chen, Xiaoshuang Shi, Heng Tao Shen, Xiaofeng Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26105](https://doi.org/10.1609/aaai.v37i8.26105)

**Abstract**:

Self-supervised multiplex graph representation learning (SMGRL) has attracted increasing interest, but previous SMGRL methods still suffer from the following issues: (i) they focus on the common information only (but ignore the private information in graph structures) to lose some essential characteristics related to downstream tasks, and (ii) they ignore the redundant information in node representations of each graph. To solve these issues, this paper proposes a new SMGRL method by jointly mining the common information and the private information in the multiplex graph while minimizing the redundant information within node representations. Specifically, the proposed method investigates the decorrelation losses to extract the common information and minimize the redundant information, while investigating the reconstruction losses to maintain the private information. Comprehensive experimental results verify the superiority of the proposed method, on four public benchmark datasets.

----

## [1036] Fundamentals of Task-Agnostic Data Valuation

**Authors**: *Mohammad Mohammadi Amiri, Frederic Berdoz, Ramesh Raskar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26106](https://doi.org/10.1609/aaai.v37i8.26106)

**Abstract**:

We study valuing the data of a data owner/seller for a data seeker/buyer. Data valuation is often carried out for a specific task assuming a particular utility metric, such as test accuracy on a validation set, that may not exist in practice. In this work, we focus on task-agnostic data valuation without any validation requirements. The data buyer has access to a limited amount of data (which could be publicly available) and seeks more data samples from a data seller. We formulate the problem as estimating the differences in the statistical properties of the data at the seller with respect to the baseline data available at the buyer. We capture these statistical differences through second moment by measuring diversity and relevance of the seller’s data for the buyer; we estimate these measures through queries to the seller without requesting the raw data. We design the queries with the proposed approach so that the seller is blind to the buyer’s raw data and has no knowledge to fabricate responses to the queries to obtain a desired outcome of the diversity and relevance trade-off. We will show through extensive experiments on real tabular and image datasets that the proposed estimates capture the diversity and relevance of the seller’s data for the buyer.

----

## [1037] Exploring the Interaction between Local and Global Latent Configurations for Clustering Single-Cell RNA-Seq: A Unified Perspective

**Authors**: *Nairouz Mrabah, Mohamed Mahmoud Amar, Mohamed Bouguessa, Abdoulaye Banire Diallo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26107](https://doi.org/10.1609/aaai.v37i8.26107)

**Abstract**:

The most recent approaches for clustering single-cell RNA-sequencing data rely on deep auto-encoders. However, three major challenges remain unaddressed. First, current models overlook the impact of the cumulative errors induced by the pseudo-supervised embedding clustering task (Feature Randomness). Second, existing methods neglect the effect of the strong competition between embedding clustering and reconstruction (Feature Drift). Third, the previous deep clustering models regularly fail to consider the topological information of the latent data, even though the local and global latent configurations can bring complementary views to the clustering task. To address these challenges, we propose a novel approach that explores the interaction between local and global latent configurations to progressively adjust the reconstruction and embedding clustering tasks. We elaborate a topological and probabilistic filter to mitigate Feature Randomness and a cell-cell graph structure and content correction mechanism to counteract Feature Drift. The Zero-Inflated Negative Binomial model is also integrated to capture the characteristics of gene expression profiles. We conduct detailed experiments on real-world datasets from multiple representative genome sequencing platforms. Our approach outperforms the state-of-the-art clustering methods in various evaluation metrics.

----

## [1038] Corruption-Tolerant Algorithms for Generalized Linear Models

**Authors**: *Bhaskar Mukhoty, Debojyoti Dey, Purushottam Kar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26108](https://doi.org/10.1609/aaai.v37i8.26108)

**Abstract**:

This paper presents SVAM (Sequential Variance-Altered MLE), a unified framework for learning generalized linear models under adversarial label corruption in training data. SVAM extends to tasks such as least squares regression, logistic regression, and gamma regression, whereas many existing works on learning with label corruptions focus only on least squares regression. SVAM is based on a novel variance reduction technique that may be of independent interest and works by iteratively solving weighted MLEs over variance-altered versions of the GLM objective. SVAM offers provable model recovery guarantees superior to the state-of-the-art for robust regression even when a constant fraction of training labels are adversarially corrupted. SVAM also empirically outperforms several existing problem-specific techniques for robust regression and classification. Code for SVAM is available at https://github.com/purushottamkar/svam/

----

## [1039] Provably Efficient Causal Model-Based Reinforcement Learning for Systematic Generalization

**Authors**: *Mirco Mutti, Riccardo De Santi, Emanuele Rossi, Juan Felipe Calderón, Michael M. Bronstein, Marcello Restelli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26109](https://doi.org/10.1609/aaai.v37i8.26109)

**Abstract**:

In the sequential decision making setting, an agent aims to achieve systematic generalization over a large, possibly infinite, set of environments. Such environments are modeled as discrete Markov decision processes with both states and actions represented through a feature vector. The underlying structure of the environments allows the transition dynamics to be factored into two components: one that is environment-specific and another that is shared. Consider a set of environments that share the laws of motion as an example. In this setting, the agent can take a finite amount of reward-free interactions from a subset of these environments. The agent then must be able to approximately solve any planning task defined over any environment in the original set, relying on the above interactions only. Can we design a provably efficient algorithm that achieves this ambitious goal of systematic generalization? In this paper, we give a partially positive answer to this question. First, we provide a tractable formulation of systematic generalization by employing a causal viewpoint. Then, under specific structural assumptions, we provide a simple learning algorithm that guarantees any desired planning error up to an unavoidable sub-optimality term, while showcasing a polynomial sample complexity.

----

## [1040] Mean Estimation of Truncated Mixtures of Two Gaussians: A Gradient Based Approach

**Authors**: *Sai Ganesh Nagarajan, Gerasimos Palaiopanos, Ioannis Panageas, Tushar Vaidya, Samson Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26110](https://doi.org/10.1609/aaai.v37i8.26110)

**Abstract**:

Even though data is abundant, it is often subjected to some form of censoring or truncation which inherently creates biases. Removing such biases and performing parameter estimation is a classical challenge in Statistics. In this paper, we focus on the problem of estimating the means of a mixture of two balanced d-dimensional Gaussians when the samples are prone to truncation. A recent theoretical study on the performance of the Expectation-Maximization (EM) algorithm for the aforementioned problem showed EM almost surely converges for d=1 and exhibits local convergence for d>1 to the true means. Nevertheless, the EM algorithm for the case of truncated mixture of two Gaussians is not easy to implement as it requires solving a set of nonlinear equations at every iteration which makes the algorithm impractical. In this work, we propose a gradient based variant of the EM algorithm that has global convergence guarantees when d=1 and local convergence for d>1 to the true means. Moreover, the update rule at every iteration is easy to compute which makes the proposed method practical. We also provide numerous experiments to obtain more insights into the effect of truncation on the convergence to the true parameters in high dimensions.

----

## [1041] An Operator Theoretic Approach for Analyzing Sequence Neural Networks

**Authors**: *Ilan Naiman, Omri Azencot*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26111](https://doi.org/10.1609/aaai.v37i8.26111)

**Abstract**:

Analyzing the inner mechanisms of deep neural networks is a fundamental task in machine learning. Existing work provides limited analysis or it depends on local theories, such as fixed-point analysis. In contrast, we propose to analyze trained neural networks using an operator theoretic approach which is rooted in Koopman theory, the Koopman Analysis of Neural Networks (KANN). Key to our method is the Koopman operator, which is a linear object that globally represents the dominant behavior of the network dynamics. The linearity of the Koopman operator facilitates analysis via its eigenvectors and eigenvalues. Our method reveals that the latter eigendecomposition holds semantic information related to the neural network inner workings. For instance,  the eigenvectors highlight positive and negative n-grams in the sentiments analysis task; similarly, the eigenvectors capture the salient features of healthy heart beat signals in the ECG classification problem.

----

## [1042] Do Invariances in Deep Neural Networks Align with Human Perception?

**Authors**: *Vedant Nanda, Ayan Majumdar, Camila Kolling, John P. Dickerson, Krishna P. Gummadi, Bradley C. Love, Adrian Weller*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26112](https://doi.org/10.1609/aaai.v37i8.26112)

**Abstract**:

An evaluation criterion for safe and trustworthy deep learning is how well the invariances captured by representations of deep neural networks (DNNs) are shared with humans. We identify challenges in measuring these invariances. Prior works used gradient-based methods to generate identically represented inputs (IRIs), ie, inputs which have identical representations (on a given layer) of a neural network, and thus capture invariances of a given network. One necessary criterion for a network's invariances to align with human perception is for its IRIs look 'similar' to humans. Prior works, however, have mixed takeaways; some argue that later layers of DNNs do not learn human-like invariances yet others seem to indicate otherwise. We argue that the loss function used to generate IRIs can heavily affect takeaways about invariances of the network and is the primary reason for these conflicting findings. We propose an adversarial regularizer on the IRI generation loss that finds IRIs that make any model appear to have very little shared invariance with humans. Based on this evidence, we argue that there is scope for improving models to have human-like invariances, and further, to have meaningful comparisons between models one should use IRIs generated using the regularizer-free loss. We then conduct an in-depth investigation of how different components (eg architectures, training losses, data augmentations) of the deep learning pipeline contribute to learning models that have good alignment with humans. We find that architectures with residual connections trained using a (self-supervised) contrastive loss with l_p ball adversarial data augmentation tend to learn invariances that are most aligned with humans. Code: github.com/nvedant07/Human-NN-Alignment

----

## [1043] Counterfactual Learning with General Data-Generating Policies

**Authors**: *Yusuke Narita, Kyohei Okumura, Akihiro Shimizu, Kohei Yata*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26113](https://doi.org/10.1609/aaai.v37i8.26113)

**Abstract**:

Off-policy evaluation (OPE) attempts to predict the performance of counterfactual policies using log data from a different policy. We extend its applicability by developing an OPE method for a class of both full support and deficient support logging policies in contextual-bandit settings. This class includes deterministic bandit (such as Upper Confidence Bound) as well as deterministic decision-making based on supervised and unsupervised learning. We prove that our method's prediction converges in probability to the true performance of a counterfactual policy as the sample size increases. We validate our method with experiments on partly and entirely deterministic logging policies. Finally, we apply it to evaluate coupon targeting policies by a major online platform and show how to improve the existing policy.

----

## [1044] Efficient and Accurate Learning of Mixtures of Plackett-Luce Models

**Authors**: *Duc Nguyen, Anderson Y. Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26114](https://doi.org/10.1609/aaai.v37i8.26114)

**Abstract**:

Mixture models of Plackett-Luce (PL), one of the most fundamental ranking models, are an active research area of both theoretical and practical significance. Most previously proposed parameter estimation algorithms instantiate the EM algorithm, often with random initialization. However, such an initialization scheme may not yield a good initial estimate and the algorithms require multiple restarts, incurring a large time complexity. As for the EM procedure, while the E-step can be performed efficiently, maximizing the log-likelihood in the M-step is difficult due to the combinatorial nature of the PL likelihood function. Therefore, previous authors favor algorithms that maximize surrogate likelihood functions. However, the final estimate may deviate from the true maximum likelihood estimate as a consequence. In this paper, we address these known limitations. We propose an initialization algorithm that can provide a provably accurate initial estimate and an EM algorithm that maximizes the true log-likelihood function efficiently. Experiments on both synthetic and real datasets show that our algorithm is competitive in terms of accuracy and speed to baseline algorithms, especially on datasets with a large number of items.

----

## [1045] Behavioral Learning in Security Games: Threat of Multi-Step Manipulative Attacks

**Authors**: *Thanh Hong Nguyen, Arunesh Sinha*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26115](https://doi.org/10.1609/aaai.v37i8.26115)

**Abstract**:

This paper studies the problem of multi-step manipulative attacks in Stackelberg security games, in which a clever attacker attempts to orchestrate its attacks over multiple time steps to mislead the defender's learning of the attacker's behavior. This attack manipulation eventually influences the defender's patrol strategy towards the attacker's benefit. Previous work along this line of research only focuses on one-shot games in which the defender learns the attacker's behavior and then designs a corresponding strategy only once. Our work, on the other hand, investigates the long-term impact of the attacker's manipulation in which current attack and defense choices of players determine the future learning and patrol planning of the defender. This paper has three key contributions. First, we introduce a new multi-step manipulative attack game model that captures the impact of sequential manipulative attacks carried out by the attacker over the entire time horizon. Second, we propose a new algorithm to compute an optimal manipulative attack plan for the attacker, which tackles the challenge of multiple connected optimization components involved in the computation across multiple time steps. Finally, we present extensive experimental results on the impact of such misleading attacks, showing a significant benefit for the attacker and loss for the defender.

----

## [1046] On Instance-Dependent Bounds for Offline Reinforcement Learning with Linear Function Approximation

**Authors**: *Thanh Nguyen-Tang, Ming Yin, Sunil Gupta, Svetha Venkatesh, Raman Arora*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26116](https://doi.org/10.1609/aaai.v37i8.26116)

**Abstract**:

Sample-efficient offline reinforcement learning (RL) with linear function approximation has been studied extensively recently. Much of the prior work has yielded instance-independent rates that hold even for the worst-case realization of problem instances. This work seeks to understand instance-dependent bounds for offline RL with linear function approximation. We present an algorithm called Bootstrapped and Constrained Pessimistic Value Iteration (BCP-VI), which leverages data bootstrapping and constrained optimization on top of pessimism. We show that under a partial data coverage assumption, that of concentrability with respect to an optimal policy, the proposed algorithm yields a fast rate for offline RL when there is a positive gap in the optimal Q-value functions, even if the offline data were collected adaptively. Moreover, when the linear features of the optimal actions in the states reachable by an optimal policy span those reachable by the behavior policy and the optimal actions are unique, offline RL achieves absolute zero sub-optimality error when the number of episodes exceeds a  (finite) instance-dependent threshold. To the best of our knowledge, these are the first results that give a fast rate bound on the sub-optimality and an absolute zero sub-optimality bound for offline RL with linear function approximation from adaptive data with partial coverage. We also provide instance-agnostic and instance-dependent information-theoretical lower bounds to complement our upper bounds.

----

## [1047] Fast Saturating Gate for Learning Long Time Scales with Recurrent Neural Networks

**Authors**: *Kentaro Ohno, Sekitoshi Kanai, Yasutoshi Ida*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26117](https://doi.org/10.1609/aaai.v37i8.26117)

**Abstract**:

Gate functions in recurrent models, such as an LSTM and GRU, play a central role in learning various time scales in modeling time series data by using a bounded activation function. However, it is difficult to train gates to capture extremely long time scales due to gradient vanishing of the bounded function for large inputs, which is known as the saturation problem. We closely analyze the relation between saturation of the gate function and efficiency of the training. We prove that the gradient vanishing of the gate function can be mitigated by accelerating the convergence of the saturating function, i.e., making the output of the function converge to 0 or 1 faster. Based on the analysis results, we propose a gate function called fast gate that has a doubly exponential convergence rate with respect to inputs by simple function composition. We empirically show that our method outperforms previous methods in accuracy and computational efficiency on benchmark tasks involving extremely long time scales.

----

## [1048] Backpropagation-Free Deep Learning with Recursive Local Representation Alignment

**Authors**: *Alexander G. Ororbia II, Ankur Arjun Mali, Daniel Kifer, C. Lee Giles*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26118](https://doi.org/10.1609/aaai.v37i8.26118)

**Abstract**:

Training deep neural networks on large-scale datasets requires significant hardware resources whose costs (even on cloud platforms) put them out of reach of smaller organizations, groups, and individuals. Backpropagation (backprop), the workhorse for training these networks, is an inherently sequential process that is difficult to parallelize. Furthermore, researchers must continually develop various specialized techniques, such as particular weight initializations and enhanced activation functions, to ensure stable parameter optimization. Our goal is to seek an effective, neuro-biologically plausible alternative to backprop that can be used to train deep networks. In this paper, we propose a backprop-free procedure, recursive local representation alignment, for training large-scale architectures. Experiments with residual networks on CIFAR-10 and the large benchmark, ImageNet, show that our algorithm generalizes as well as backprop while converging sooner due to weight updates that are parallelizable and computationally less demanding. This is empirical evidence that a backprop-free algorithm can scale up to larger datasets.

----

## [1049] Bilinear Exponential Family of MDPs: Frequentist Regret Bound with Tractable Exploration & Planning

**Authors**: *Reda Ouhamma, Debabrota Basu, Odalric Maillard*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26119](https://doi.org/10.1609/aaai.v37i8.26119)

**Abstract**:

We study the problem of episodic reinforcement learning in continuous state-action spaces with unknown rewards and transitions. Specifically, we consider the setting where the rewards and transitions are modeled using parametric bilinear exponential families. We propose an algorithm, that a) uses penalized maximum likelihood estimators to learn the unknown parameters, b) injects a calibrated Gaussian noise in the parameter of rewards to ensure exploration, and c) leverages linearity of the bilinear exponential family transitions with respect to an underlying RKHS to perform tractable planning. We provide a frequentist regret upper-bound for our algorithm which, in the case of tabular MDPs, is order-optimal with respect to H and K, where H is the episode length and K is the number of episodes. Our analysis improves the existing bounds for the bilinear exponential family of MDPs by square root of H and removes the handcrafted clipping deployed in existing RLSVI-type algorithms.

----

## [1050] H-TSP: Hierarchically Solving the Large-Scale Traveling Salesman Problem

**Authors**: *Xuanhao Pan, Yan Jin, Yuandong Ding, Mingxiao Feng, Li Zhao, Lei Song, Jiang Bian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26120](https://doi.org/10.1609/aaai.v37i8.26120)

**Abstract**:

We propose an end-to-end learning framework based on hierarchical reinforcement learning, called H-TSP, for addressing the large-scale Traveling Salesman Problem (TSP). The proposed H-TSP constructs a solution of a TSP instance starting from the scratch relying on two components: the upper-level policy chooses a small subset of nodes (up to 200 in our experiment) from all nodes that are to be traversed, while the lower-level policy takes the chosen nodes as input and outputs a tour connecting them to the existing partial route (initially only containing the depot). After jointly training the upper-level and lower-level policies, our approach can directly generate solutions for the given TSP instances without relying on any time-consuming search procedures. To demonstrate effectiveness of the proposed approach, we have conducted extensive experiments on randomly generated TSP instances with different numbers of nodes. We show that H-TSP can achieve comparable results (gap 3.42% vs. 7.32%) as SOTA search-based approaches, and more importantly, we reduce the time consumption up to two orders of magnitude (3.32s vs. 395.85s). To the best of our knowledge, H-TSP is the first end-to-end deep reinforcement learning approach that can scale to TSP instances of up to 10000 nodes. Although there are still gaps to SOTA results with respect to solution quality, we believe that H-TSP will be useful for practical applications, particularly those that are time-sensitive e.g., on-call routing and ride hailing service.

----

## [1051] Ising-Traffic: Using Ising Machine Learning to Predict Traffic Congestion under Uncertainty

**Authors**: *Zhenyu Pan, Anshujit Sharma, Jerry Yao-Chieh Hu, Zhuo Liu, Ang Li, Han Liu, Michael C. Huang, Tong Geng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26121](https://doi.org/10.1609/aaai.v37i8.26121)

**Abstract**:

This paper addresses the challenges in accurate and real-time traffic congestion prediction under uncertainty by proposing Ising-Traffic, a dual-model Ising-based traffic prediction framework that delivers higher accuracy and lower latency than SOTA solutions. While traditional solutions face the dilemma from the trade-off between algorithm complexity and computational efficiency, our Ising-based method breaks away from the trade-off leveraging the Ising model's strong expressivity and the Ising machine's strong computation power. In particular, Ising-Traffic formulates traffic prediction under uncertainty into two Ising models: Reconstruct-Ising and Predict-Ising. Reconstruct-Ising is mapped onto modern Ising machines and handles uncertainty in traffic accurately with negligible latency and energy consumption, while Predict-Ising is mapped onto traditional processors and predicts future congestion precisely with only at most 1.8% computational demands of existing solutions. Our evaluation shows Ising-Traffic delivers on average 98X speedups and 5% accuracy improvement over SOTA.

----

## [1052] FedMDFG: Federated Learning with Multi-Gradient Descent and Fair Guidance

**Authors**: *Zibin Pan, Shuyi Wang, Chi Li, Haijin Wang, Xiaoying Tang, Junhua Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26122](https://doi.org/10.1609/aaai.v37i8.26122)

**Abstract**:

Fairness has been considered as a critical problem in federated learning (FL). In this work, we analyze two direct causes of unfairness in FL - an unfair direction and an improper step size when updating the model. To solve these issues, we introduce an effective way to measure fairness of the model through the cosine similarity, and then propose a federated multiple gradient descent algorithm with fair guidance (FedMDFG) to drive the model fairer. We first convert FL into a multi-objective optimization problem (MOP) and design an advanced multiple gradient descent algorithm to calculate a fair descent direction by adding a fair-driven objective to MOP. A low-communication-cost line search strategy is then designed to find a better step size for the model update. We further show the theoretical analysis on how it can enhance fairness and guarantee the convergence. Finally, extensive experiments in several FL scenarios verify that FedMDFG is robust and outperforms the SOTA FL algorithms in convergence and fairness. The source code is available at https://github.com/zibinpan/FedMDFG.

----

## [1053] Geometric Inductive Biases for Identifiable Unsupervised Learning of Disentangled Representations

**Authors**: *Ziqi Pan, Li Niu, Liqing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26123](https://doi.org/10.1609/aaai.v37i8.26123)

**Abstract**:

The model identifiability is a considerable issue in the unsupervised learning of disentangled representations. The PCA inductive biases revealed recently for unsupervised disentangling in VAE-based models are shown to improve local alignment of latent dimensions with principal components of the data. In this paper, in additional to the PCA inductive biases, we propose novel geometric inductive biases from the manifold perspective for unsupervised disentangling, which induce the model to capture the global geometric properties of the data manifold with guaranteed model identifiability. We also propose a Geometric Disentangling Regularized AutoEncoder (GDRAE) that combines the PCA and the proposed geometric inductive biases in one unified framework. The experimental results show the usefulness of the geometric inductive biases in unsupervised disentangling and the effectiveness of our GDRAE in capturing the geometric inductive biases.

----

## [1054] Isometric Manifold Learning Using Hierarchical Flow

**Authors**: *Ziqi Pan, Jianfu Zhang, Li Niu, Liqing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26124](https://doi.org/10.1609/aaai.v37i8.26124)

**Abstract**:

We propose the Hierarchical Flow (HF) model constrained by isometric regularizations for manifold learning that combines manifold learning goals such as dimensionality reduction, inference, sampling, projection and density estimation into one unified framework. Our proposed HF model is regularized to not only produce embeddings preserving the geometric structure of the manifold, but also project samples onto the manifold in a manner conforming to the rigorous definition of projection. Theoretical guarantees are provided for our HF model to satisfy the two desired properties. In order to detect the real dimensionality of the manifold, we also propose a two-stage dimensionality reduction algorithm, which is a time-efficient algorithm thanks to the hierarchical architecture design of our HF model. Experimental results justify our theoretical analysis, demonstrate the superiority of our dimensionality reduction algorithm in cost of training time, and verify the effect of the aforementioned properties in improving performances on downstream tasks such as anomaly detection.

----

## [1055] Evidential Conditional Neural Processes

**Authors**: *Deep Shankar Pandey, Qi Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26125](https://doi.org/10.1609/aaai.v37i8.26125)

**Abstract**:

The Conditional Neural Process (CNP) family of models offer a promising direction to tackle few-shot problems by achieving better scalability and competitive predictive performance. However, the current CNP models only capture the overall uncertainty for the prediction made on a target data point. They lack a systematic fine-grained quantification on the distinct sources of uncertainty that are essential for model training and decision-making under the few-shot setting. We propose Evidential Conditional Neural Processes (ECNP), which replace the standard Gaussian distribution used by CNP with a much richer hierarchical Bayesian structure through evidential learning to achieve epistemic-aleatoric uncertainty decomposition. The evidential hierarchical structure also leads to a theoretically justified robustness over noisy training tasks. Theoretical analysis on the proposed ECNP establishes the relationship with CNP while offering deeper insights on the roles of the evidential parameters. Extensive experiments conducted on both synthetic and real-world data demonstrate the effectiveness of our proposed model in various few-shot settings.

----

## [1056] Balanced Column-Wise Block Pruning for Maximizing GPU Parallelism

**Authors**: *Cheonjun Park, Mincheol Park, Hyun Jae Oh, Minkyu Kim, Myung Kuk Yoon, Suhyun Kim, Won Woo Ro*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26126](https://doi.org/10.1609/aaai.v37i8.26126)

**Abstract**:

Pruning has been an effective solution to reduce the number of computations and the memory requirement in deep learning.
The pruning unit plays an important role in exploiting the GPU resources efficiently. 
The filter is proposed as a simple pruning unit of structured pruning.
However, since the filter is quite large as pruning unit, the accuracy drop is considerable with a high pruning ratio.
GPU rearranges the weight and input tensors into tiles (blocks) for efficient computation. 
To fully utilize GPU resources, this tile structure should be considered, which is the goal of block pruning. 
However, previous block pruning prunes both row vectors and column vectors. 
Pruning of row vectors in a tile corresponds to filter pruning, and it also interferes with column-wise block pruning of the following layer.
In contrast, column vectors are much smaller than row vectors and can achieve lower accuracy drop.
Additionally, if the pruning ratio for each tile is different,
GPU utilization can be limited by imbalanced workloads by irregular-sized blocks.
The same pruning ratio for the weight tiles processed in parallel enables the actual inference process to fully utilize the resources without idle time.
This paper proposes balanced column-wise block pruning, named BCBP, to satisfy two conditions: the column-wise minimal size of the pruning unit and balanced workloads. 
We demonstrate that BCBP is superior to previous pruning methods through comprehensive experiments.

----

## [1057] Dynamic Structure Pruning for Compressing CNNs

**Authors**: *Jun-Hyung Park, Yeachan Kim, Junho Kim, Joon-Young Choi, SangKeun Lee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26127](https://doi.org/10.1609/aaai.v37i8.26127)

**Abstract**:

Structure pruning is an effective method to compress and accelerate neural networks. While filter and channel pruning are preferable to other structure pruning methods in terms of realistic acceleration and hardware compatibility, pruning methods with a finer granularity, such as intra-channel pruning, are expected to be capable of yielding more compact and computationally efficient networks. Typical intra-channel pruning methods utilize a static and hand-crafted pruning granularity due to a large search space, which leaves room for improvement in their pruning performance. In this work, we introduce a novel structure pruning method, termed as dynamic structure pruning, to identify optimal pruning granularities for intra-channel pruning. In contrast to existing intra-channel pruning methods, the proposed method automatically optimizes dynamic pruning granularities in each layer while training deep neural networks. To achieve this, we propose a differentiable group learning method designed to efficiently learn a pruning granularity based on gradient-based learning of filter groups. The experimental results show that dynamic structure pruning achieves state-of-the-art pruning performance and better realistic acceleration on a GPU compared with channel pruning. In particular, it reduces the FLOPs of ResNet50 by 71.85% without accuracy degradation on the ImageNet dataset. Our code is available at https://github.com/irishev/DSP.

----

## [1058] Scaling Marginalized Importance Sampling to High-Dimensional State-Spaces via State Abstraction

**Authors**: *Brahma S. Pavse, Josiah P. Hanna*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26128](https://doi.org/10.1609/aaai.v37i8.26128)

**Abstract**:

We consider the problem of off-policy evaluation (OPE) in reinforcement learning (RL), where the goal is to estimate the performance of an evaluation policy, pie, using a fixed dataset, D, collected by one or more policies that may be different from pie. Current OPE algorithms may produce poor OPE estimates under policy distribution shift i.e., when the probability of a particular state-action pair occurring under pie is very different from the probability of that same pair occurring in D. In this work, we propose to improve the accuracy of OPE estimators by projecting the high-dimensional state-space into a low-dimensional state-space using concepts from the state abstraction literature. Specifically, we consider marginalized importance sampling (MIS) OPE algorithms which compute state-action distribution correction ratios to produce their OPE estimate. In the original ground state-space, these ratios may have high variance which may lead to high variance OPE. However, we prove that in the lower-dimensional abstract state-space the ratios can have lower variance resulting in lower variance OPE. We then highlight the challenges that arise when estimating the abstract ratios from data, identify sufficient conditions to overcome these issues, and present a minimax optimization problem whose solution yields these abstract ratios. Finally, our empirical evaluation on difficult, high-dimensional state-space OPE tasks shows that the abstract ratios can make MIS OPE estimators achieve lower mean-squared error and more robust to hyperparameter tuning than the ground ratios.

----

## [1059] Conceptual Reinforcement Learning for Language-Conditioned Tasks

**Authors**: *Shaohui Peng, Xing Hu, Rui Zhang, Jiaming Guo, Qi Yi, Ruizhi Chen, Zidong Du, Ling Li, Qi Guo, Yunji Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26129](https://doi.org/10.1609/aaai.v37i8.26129)

**Abstract**:

Despite the broad application of deep reinforcement learning (RL), transferring and adapting the policy to unseen but similar environments is still a significant challenge. Recently, the language-conditioned policy is proposed to facilitate policy transfer through learning the joint representation of observation and text that catches the compact and invariant information across various environments. Existing studies of language-conditioned RL methods often learn the joint representation as a simple latent layer for the given instances (episode-specific observation and text), which inevitably includes noisy or irrelevant information and cause spurious correlations that are dependent on instances, thus hurting generalization performance and training efficiency. To address the above issue, we propose a conceptual reinforcement learning (CRL) framework to learn the concept-like joint representation for language-conditioned policy. The key insight is that concepts are compact and invariant representations in human cognition through extracting similarities from numerous instances in real-world. In CRL, we propose a multi-level attention encoder and two mutual information constraints for learning compact and invariant concepts. Verified in two challenging environments, RTFM and Messenger, CRL significantly improves the training efficiency (up to 70%) and generalization ability (up to 30%) to the new environment dynamics.

----

## [1060] Weighted Policy Constraints for Offline Reinforcement Learning

**Authors**: *Zhiyong Peng, Changlin Han, Yadong Liu, Zongtan Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26130](https://doi.org/10.1609/aaai.v37i8.26130)

**Abstract**:

Offline reinforcement learning (RL) aims to learn policy from the passively collected offline dataset. Applying existing RL methods on the static dataset straightforwardly will raise distribution shift, causing these unconstrained RL methods to fail. To cope with the distribution shift problem, a common practice in offline RL is to constrain the policy explicitly or implicitly close to behavioral policy. However, the available dataset usually contains sub-optimal or inferior actions, constraining the policy near all these actions will make the policy inevitably learn inferior behaviors, limiting the performance of the algorithm. Based on this observation, we propose a weighted policy constraints (wPC) method that only constrains the learned policy to desirable behaviors, making room for policy improvement on other parts. Our algorithm outperforms existing state-of-the-art offline RL algorithms on the D4RL offline gym datasets. Moreover, the proposed algorithm is simple to implement with few hyper-parameters, making the proposed wPC algorithm a robust offline RL method with low computational complexity.

----

## [1061] Latent Autoregressive Source Separation

**Authors**: *Emilian Postolache, Giorgio Mariani, Michele Mancusi, Andrea Santilli, Luca Cosmo, Emanuele Rodolà*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26131](https://doi.org/10.1609/aaai.v37i8.26131)

**Abstract**:

Autoregressive models have achieved impressive results over a wide range of domains in terms of generation quality and downstream task performance. In the continuous domain, a key factor behind this success is the usage of quantized latent spaces (e.g., obtained via VQ-VAE autoencoders), which allow for dimensionality reduction and faster inference times. However, using existing pre-trained models to perform new non-trivial tasks is difficult since it requires additional fine-tuning or extensive training to elicit prompting. This paper introduces LASS as a way to perform vector-quantized Latent Autoregressive Source Separation (i.e., de-mixing an input signal into its constituent sources) without requiring additional gradient-based optimization or modifications of existing models. Our separation method relies on the Bayesian formulation in which the autoregressive models are the priors, and a discrete (non-parametric) likelihood function is constructed by performing frequency counts over latent sums of addend tokens. We test our method on images and audio with several sampling strategies (e.g., ancestral, beam search) showing competitive results with existing approaches in terms of separation quality while offering at the same time significant speedups in terms of inference time and scalability to higher dimensional data.

----

## [1062] Explaining Random Forests Using Bipolar Argumentation and Markov Networks

**Authors**: *Nico Potyka, Xiang Yin, Francesca Toni*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26132](https://doi.org/10.1609/aaai.v37i8.26132)

**Abstract**:

Random forests are decision tree ensembles that can be used 
to solve a variety of machine learning problems. However, as
the number of trees and their individual size can be large,
their decision making process is often incomprehensible.
We show that their decision process can be naturally represented 
as an argumentation problem, which allows creating global explanations 
via argumentative reasoning. We generalize sufficient and necessary 
argumentative explanations using a Markov network encoding, discuss 
the relevance of these explanations and establish relationships to
families of abductive explanations from the literature. As the complexity 
of the explanation problems is high, we present an efficient approximation algorithm with probabilistic approximation guarantees.

----

## [1063] A Model-Agnostic Heuristics for Selective Classification

**Authors**: *Andrea Pugnana, Salvatore Ruggieri*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26133](https://doi.org/10.1609/aaai.v37i8.26133)

**Abstract**:

Selective classification (also known as classification with reject option) conservatively extends a classifier with a selection function to determine whether or not a  prediction should be accepted (i.e., trusted, used, deployed). This is a highly relevant issue in socially sensitive tasks, such as credit scoring.
State-of-the-art approaches rely on Deep Neural Networks (DNNs) that train at the same time both the classifier and the selection function. These approaches are model-specific and computationally expensive. 
We propose a model-agnostic approach, as it can work with any base probabilistic binary classification algorithm, and it can be scalable to large tabular datasets if the base classifier is so. The proposed algorithm, called SCROSS, exploits a cross-fitting strategy and theoretical results for quantile estimation to build the selection function. Experiments on real-world data show that SCROSS improves over existing methods.

----

## [1064] Experimental Observations of the Topology of Convolutional Neural Network Activations

**Authors**: *Emilie Purvine, Davis Brown, Brett A. Jefferson, Cliff A. Joslyn, Brenda Praggastis, Archit Rathore, Madelyn Shapiro, Bei Wang, Youjia Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26134](https://doi.org/10.1609/aaai.v37i8.26134)

**Abstract**:

Topological data analysis (TDA) is a branch of computational mathematics, bridging algebraic topology and data science, that provides compact, noise-robust representations of complex structures. Deep neural networks (DNNs) learn millions of parameters associated with a series of transformations defined by the model architecture resulting in high-dimensional, difficult to interpret internal representations of input data. As DNNs become more ubiquitous across multiple sectors of our society, there is increasing recognition that mathematical methods are needed to aid analysts, researchers, and practitioners in understanding and interpreting how these models' internal representations relate to the final classification. In this paper we apply cutting edge techniques from TDA with the goal of gaining insight towards interpretability of convolutional neural networks used for image classification. We use  two common TDA approaches to explore several methods for modeling hidden layer activations as high-dimensional point clouds, and provide experimental evidence that these point clouds capture valuable structural information about the model's process. First, we demonstrate that a distance metric based on persistent homology can be used to quantify meaningful differences between layers and discuss these distances in the broader context of existing representational similarity metrics for neural network interpretability. Second, we show that a mapper graph can provide semantic insight as to how these models organize hierarchical class knowledge at each layer. These observations demonstrate that TDA is a useful tool to help deep learning practitioners unlock the hidden structures of their models.

----

## [1065] CMVAE: Causal Meta VAE for Unsupervised Meta-Learning

**Authors**: *Guodong Qi, Huimin Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26135](https://doi.org/10.1609/aaai.v37i8.26135)

**Abstract**:

Unsupervised meta-learning aims to learn the meta knowledge from unlabeled data and rapidly adapt to novel tasks. However, existing approaches may be misled by the  context-bias (e.g. background) from the training data. In this paper, we abstract the unsupervised meta-learning problem into a Structural Causal Model (SCM) and point out that such bias arises due to hidden confounders. To eliminate the confounders, we define the priors are conditionally independent, learn the relationships between priors and intervene on them with casual factorization. Furthermore, we propose Causal Meta VAE (CMVAE) that encodes the priors into latent codes in the causal space and learns their relationships simultaneously to achieve the downstream few-shot image classification task. Results on toy datasets and three benchmark datasets demonstrate that our method can remove the context-bias and it outperforms other state-of-the-art unsupervised meta-learning algorithms because of bias-removal. Code is available at https://github.com/GuodongQi/CMVAE.

----

## [1066] Rethinking Data-Free Quantization as a Zero-Sum Game

**Authors**: *Biao Qian, Yang Wang, Richang Hong, Meng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26136](https://doi.org/10.1609/aaai.v37i8.26136)

**Abstract**:

Data-free quantization (DFQ) recovers the performance of quantized network (Q) without accessing the real data, but generates the fake sample via a generator (G) by learning from full-precision network (P) instead. However, such sample generation process is totally independence of Q, specialized as failing to consider the adaptability of the generated samples, i.e., beneficial or adversarial, over the learning process of Q, resulting into non-ignorable performance loss. Building on this, several crucial questions --- how to measure and exploit the sample adaptability to Q under varied bit-width scenarios? how to generate the samples with desirable adaptability to benefit the quantized network? --- impel us to revisit DFQ. In this paper, we answer the above questions from a game-theory perspective to specialize DFQ as a zero-sum game between two players --- a generator and a quantized network, and further propose an Adaptability-aware Sample Generation (AdaSG) method. Technically, AdaSG reformulates DFQ as a dynamic maximization-vs-minimization game process anchored on the sample adaptability. The maximization process aims to generate the sample with desirable adaptability, such sample adaptability is further reduced by the minimization process after calibrating Q for performance recovery. The Balance Gap is defined to guide the stationarity of the game process to maximally benefit Q. The theoretical analysis and empirical studies verify the superiority of AdaSG over the state-of-the-arts. Our code is available at https://github.com/hfutqian/AdaSG.

----

## [1067] Mixture Uniform Distribution Modeling and Asymmetric Mix Distillation for Class Incremental Learning

**Authors**: *Sunyuan Qiang, Jiayi Hou, Jun Wan, Yanyan Liang, Zhen Lei, Du Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26137](https://doi.org/10.1609/aaai.v37i8.26137)

**Abstract**:

Exemplar rehearsal-based methods with knowledge distillation (KD) have been widely used in class incremental learning (CIL) scenarios. However, they still suffer from performance degradation because of severely distribution discrepancy between training and test set caused by the limited storage memory on previous classes. In this paper, we mathematically model the data distribution and the discrepancy at the incremental stages with mixture uniform distribution (MUD). Then, we propose the asymmetric mix distillation method to uniformly minimize the error of each class from distribution discrepancy perspective. Specifically, we firstly promote mixup in CIL scenarios with the incremental mix samplers and incremental mix factor to calibrate the raw training data distribution. Next, mix distillation label augmentation is incorporated into the data distribution to inherit the knowledge information from the previous models. Based on the above augmented data distribution, our trained model effectively alleviates the performance degradation and extensive experimental results validate that our method exhibits superior performance on CIL benchmarks.

----

## [1068] Mutual-Enhanced Incongruity Learning Network for Multi-Modal Sarcasm Detection

**Authors**: *Yang Qiao, Liqiang Jing, Xuemeng Song, Xiaolin Chen, Lei Zhu, Liqiang Nie*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26138](https://doi.org/10.1609/aaai.v37i8.26138)

**Abstract**:

Sarcasm is a sophisticated linguistic phenomenon that is prevalent on today's social media platforms. Multi-modal sarcasm detection aims to identify whether a given sample with multi-modal information (i.e., text and image) is sarcastic. This task's key lies in capturing both inter- and intra-modal incongruities within the same context. Although existing methods have achieved compelling success, they are disturbed by irrelevant information extracted from the whole image and text, or overlooking some important information due to the incomplete input. To address these limitations, we propose a Mutual-enhanced Incongruity Learning Network for multi-modal sarcasm detection, named MILNet. In particular, we design a local semantic-guided incongruity learning module and a global incongruity learning module. Moreover, we introduce a mutual enhancement module to take advantage of the underlying consistency between the two modules to boost the performance. Extensive experiments on a widely-used dataset demonstrate the superiority of our model over cutting-edge methods.

----

## [1069] Training Meta-Surrogate Model for Transferable Adversarial Attack

**Authors**: *Yunxiao Qin, Yuanhao Xiong, Jinfeng Yi, Cho-Jui Hsieh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26139](https://doi.org/10.1609/aaai.v37i8.26139)

**Abstract**:

The problem of adversarial attacks to a black-box model when no queries are allowed has posed a great challenge to the community and has been extensively investigated. In this setting, one simple yet effective method is to transfer the obtained adversarial examples from attacking surrogate models to fool the target model. Previous works have studied what kind of attacks to the surrogate model can generate more transferable adversarial examples, but their performances are still limited due to the mismatches between surrogate models and the target model. In this paper, we tackle this problem from a novel angle---instead of using the original surrogate models, can we obtain a Meta-Surrogate Model (MSM) such that attacks to this model can be easily transferred to other models? We show that this goal can be mathematically formulated as a bi-level optimization problem and design a differentiable attacker to make training feasible. Given one or a set of surrogate models, our method can thus obtain an MSM such that adversarial examples generated on MSM enjoy eximious transferability. Comprehensive experiments on Cifar-10 and ImageNet demonstrate that by attacking the MSM, we can obtain stronger transferable adversarial examples to deceive black-box models including adversarially trained ones, with much higher success rates than existing methods.

----

## [1070] Stochastic Contextual Bandits with Long Horizon Rewards

**Authors**: *Yuzhen Qin, Yingcong Li, Fabio Pasqualetti, Maryam Fazel, Samet Oymak*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26140](https://doi.org/10.1609/aaai.v37i8.26140)

**Abstract**:

The growing interest in complex decision-making and language modeling problems highlights the importance of sample-efficient learning over very long horizons. This work takes a step in this direction by investigating contextual linear bandits where the current reward depends on at most s prior actions and contexts (not necessarily consecutive), up to a time horizon of h. In order to avoid polynomial dependence on h, we propose new algorithms that leverage sparsity to discover the dependence pattern and arm parameters jointly. We consider both the data-poor (T= h) regimes and derive respective regret upper bounds O(d square-root(sT) +min(q, T) and O( square-root(sdT) ),  with sparsity s, feature dimension d,  total time horizon T, and q that is adaptive to the reward dependence pattern. Complementing upper bounds, we also show that learning over a single trajectory brings inherent challenges: While the dependence pattern and arm parameters form a rank-1 matrix, circulant matrices are not isometric over rank-1 manifolds and sample complexity indeed benefits from the sparse reward dependence structure. Our results necessitate a new analysis to address long-range temporal dependencies across data and avoid polynomial dependence on the reward horizon h. Specifically, we utilize connections to the restricted isometry property of circulant matrices formed by dependent sub-Gaussian vectors and establish new guarantees that are also of independent interest.

----

## [1071] Gradient-Variation Bound for Online Convex Optimization with Constraints

**Authors**: *Shuang Qiu, Xiaohan Wei, Mladen Kolar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26141](https://doi.org/10.1609/aaai.v37i8.26141)

**Abstract**:

We study online convex optimization with constraints consisting of multiple functional constraints and a relatively simple constraint set, such as a Euclidean ball. As enforcing the constraints at each time step through projections is computationally challenging in general, we allow decisions to violate the functional constraints but aim to achieve a low regret and cumulative violation of the constraints over a horizon of T time steps. First-order methods achieve an O(sqrt{T}) regret and an O(1) constraint violation, which is the best-known bound under the Slater's condition, but do not take into account the structural information of the problem. Furthermore, the existing algorithms and analysis are limited to Euclidean space. In this paper, we provide an instance-dependent bound for online convex optimization with complex constraints obtained by a novel online primal-dual mirror-prox algorithm. Our instance-dependent regret is quantified by the total gradient variation V_*(T) in the sequence of loss functions. The proposed algorithm works in general normed spaces and simultaneously achieves an O(sqrt{V_*(T)}) regret and an O(1) constraint violation, which is never worse than the best-known (O(sqrt{T}), O(1)) result and improves over previous works that applied mirror-prox-type algorithms for this problem achieving O(T^{2/3}) regret and constraint violation. Finally, our algorithm is computationally efficient, as it only performs mirror descent steps in each iteration instead of solving a general Lagrangian minimization problem.

----

## [1072] Bellman Meets Hawkes: Model-Based Reinforcement Learning via Temporal Point Processes

**Authors**: *Chao Qu, Xiaoyu Tan, Siqiao Xue, Xiaoming Shi, James Zhang, Hongyuan Mei*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26142](https://doi.org/10.1609/aaai.v37i8.26142)

**Abstract**:

We consider a sequential decision making problem where the agent faces the environment characterized by the stochastic discrete events and seeks an optimal intervention policy such that its long-term reward is maximized. This problem exists ubiquitously in social media, finance and health informatics but is rarely investigated by the conventional research in reinforcement learning. To this end, we present a novel framework of the model-based reinforcement learning where the agent's actions and observations  are  asynchronous stochastic discrete events occurring in continuous-time.  We model the dynamics of the environment by Hawkes process with external intervention control term and develop an algorithm to embed such process in the Bellman equation which guides the direction of the value gradient. We demonstrate the superiority of our method  in both synthetic simulator and real-data experiments.

----

## [1073] GLUECons: A Generic Benchmark for Learning under Constraints

**Authors**: *Hossein Rajaby Faghihi, Aliakbar Nafar, Chen Zheng, Roshanak Mirzaee, Yue Zhang, Andrzej Uszok, Alexander Wan, Tanawan Premsri, Dan Roth, Parisa Kordjamshidi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26143](https://doi.org/10.1609/aaai.v37i8.26143)

**Abstract**:

Recent research has shown that integrating domain knowledge into deep learning architectures is effective; It helps reduce the amount of required data, improves the accuracy of the models' decisions, and improves the interpretability of models. However, the research community lacks a convened benchmark for systematically evaluating knowledge integration methods.
In this work, we create a benchmark that is a collection of nine tasks in the domains of natural language processing and computer vision. In all cases, we model external knowledge as constraints, specify the sources of the constraints for each task, and implement various models that use these constraints.
We report the results of these models using a new set of extended evaluation criteria in addition to the task performances for a more in-depth analysis. This effort provides a framework for a more comprehensive and systematic comparison of constraint integration techniques and for identifying related research challenges. It will facilitate further research for alleviating some problems of state-of-the-art neural models.

----

## [1074] Provable Detection of Propagating Sampling Bias in Prediction Models

**Authors**: *Pavan Ravishankar, Qingyu Mo, Edward McFowland III, Daniel B. Neill*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26144](https://doi.org/10.1609/aaai.v37i8.26144)

**Abstract**:

With an increased focus on incorporating fairness in machine learning models, it becomes imperative not only to assess and mitigate bias at each stage of the machine learning pipeline but also to understand the downstream impacts of bias across stages. Here we consider a general, but realistic, scenario in which a predictive model is learned from (potentially biased) training data, and model predictions are assessed post-hoc for fairness by some auditing method. We provide a theoretical analysis of how a specific form of data bias, differential sampling bias, propagates from the data stage to the prediction stage. Unlike prior work, we evaluate the downstream impacts of data biases quantitatively rather than qualitatively and prove theoretical guarantees for detection. Under reasonable assumptions, we quantify how the amount of bias in the model predictions varies as a function of the amount of differential sampling bias in the data, and at what point this bias becomes provably detectable by the auditor. Through experiments on two criminal justice datasets-- the well-known COMPAS dataset and historical data from NYPD's stop and frisk policy-- we demonstrate that the theoretical results hold in practice even when our assumptions are relaxed.

----

## [1075] Diffusing Gaussian Mixtures for Generating Categorical Data

**Authors**: *Florence Regol, Mark Coates*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26145](https://doi.org/10.1609/aaai.v37i8.26145)

**Abstract**:

Learning a categorical distribution comes with its own set of challenges. A successful approach taken by state-of-the-art works is to cast the problem in a continuous domain to take advantage of the impressive performance of the generative models for continuous data. Amongst them are the recently emerging diffusion probabilistic models, which have the observed advantage of generating high-quality samples. Recent advances for categorical generative models have focused on log likelihood improvements. In this work, we propose a generative model for categorical data based on diffusion models with a focus on high-quality sample generation, and propose sampled-based evaluation methods. The efficacy of our method stems from performing diffusion in the continuous domain while having its parameterization informed by the structure of the categorical nature of the target distribution. Our method of evaluation highlights the capabilities and limitations of different generative models for generating categorical data, and includes experiments on synthetic and real-world protein datasets.

----

## [1076] Hypernetworks for Zero-Shot Transfer in Reinforcement Learning

**Authors**: *Sahand Rezaei-Shoshtari, Charlotte Morissette, François Robert Hogan, Gregory Dudek, David Meger*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26146](https://doi.org/10.1609/aaai.v37i8.26146)

**Abstract**:

In this paper, hypernetworks are trained to generate behaviors across a range of unseen task conditions, via a novel TD-based training objective and data from a set of near-optimal RL solutions for training tasks. This work relates to meta RL, contextual RL, and transfer learning, with a particular focus on  zero-shot performance at test time, enabled by knowledge of the task parameters (also known as context). Our technical approach is based upon viewing each RL algorithm as a mapping from the MDP specifics to the near-optimal value function and policy and seek to approximate it with a hypernetwork that can generate near-optimal value functions and policies, given the parameters of the MDP. We show that, under certain conditions, this mapping can be considered as a supervised learning problem. We empirically evaluate the effectiveness of our method for zero-shot transfer to new reward and transition dynamics on a series of continuous control tasks from DeepMind Control Suite. Our method demonstrates significant improvements over baselines from multitask and meta RL approaches.

----

## [1077] Automata Cascades: Expressivity and Sample Complexity

**Authors**: *Alessandro Ronca, Nadezda Alexandrovna Knorozova, Giuseppe De Giacomo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26147](https://doi.org/10.1609/aaai.v37i8.26147)

**Abstract**:

Every automaton can be decomposed into a cascade of basic prime automata. This is the Prime Decomposition Theorem by Krohn and Rhodes. Guided by this theory, we propose automata cascades as a structured, modular, way to describe automata as complex systems made of many components, each implementing a specific functionality. Any automaton can serve as a component; using specific components allows for a fine-grained control of the expressivity of the resulting class of automata; using prime automata as components implies specific expressivity guarantees. Moreover, specifying automata as cascades allows for describing the sample complexity of automata in terms of their components. We show that the sample complexity is linear in the number of components and the maximum complexity of a single component, modulo logarithmic factors. This opens to the possibility of learning automata representing large dynamical systems consisting of many parts interacting with each other. It is in sharp contrast with the established understanding of the sample complexity of automata, described in terms of the overall number of states and input letters, which implies that it is only possible to learn automata where the number of states is linear in the amount of data available. Instead our results show that one can learn automata with a number of states that is exponential in the amount of data available.

----

## [1078] ESPT: A Self-Supervised Episodic Spatial Pretext Task for Improving Few-Shot Learning

**Authors**: *Yi Rong, Xiongbo Lu, Zhaoyang Sun, Yaxiong Chen, Shengwu Xiong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26148](https://doi.org/10.1609/aaai.v37i8.26148)

**Abstract**:

Self-supervised learning (SSL) techniques have recently been integrated into the few-shot learning (FSL) framework and have shown promising results in improving the few-shot image classification performance. However, existing SSL approaches used in FSL typically seek the supervision signals from the global embedding of every single image. Therefore, during the episodic training of FSL, these methods cannot capture and fully utilize the local visual information in image samples and the data structure information of the whole episode, which are beneficial to FSL. To this end, we propose to augment the few-shot learning objective with a novel self-supervised Episodic Spatial Pretext Task (ESPT). Specifically, for each few-shot episode, we generate its corresponding transformed episode by applying a random geometric transformation to all the images in it. Based on these, our ESPT objective is defined as maximizing the local spatial relationship consistency between the original episode and the transformed one. With this definition, the ESPT-augmented FSL objective promotes learning more transferable feature representations that capture the local spatial features of different images and their inter-relational structural information in each input episode, thus enabling the model to generalize better to new categories with only a few samples. Extensive experiments indicate that our ESPT method achieves new state-of-the-art performance for few-shot image classification on three mainstay benchmark datasets. The source code will be available at: https://github.com/Whut-YiRong/ESPT.

----

## [1079] Planning and Learning with Adaptive Lookahead

**Authors**: *Aviv Rosenberg, Assaf Hallak, Shie Mannor, Gal Chechik, Gal Dalal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26149](https://doi.org/10.1609/aaai.v37i8.26149)

**Abstract**:

Some of the most powerful reinforcement learning frameworks use planning for action selection. Interestingly, their planning horizon is either fixed or determined arbitrarily by the state visitation history. Here, we expand beyond the naive fixed horizon and propose a theoretically justified strategy for adaptive selection of the planning horizon as a function of the state-dependent value estimate. We propose two variants for lookahead selection and analyze the trade-off between iteration count and computational complexity per iteration. We then devise a corresponding deep Q-network algorithm with an adaptive tree search horizon. We separate the value estimation per depth to compensate for the off-policy discrepancy between depths. Lastly, we demonstrate the efficacy of our adaptive lookahead method in a maze environment and Atari.

----

## [1080] DisGUIDE: Disagreement-Guided Data-Free Model Extraction

**Authors**: *Jonathan Rosenthal, Eric Enouen, Hung Viet Pham, Lin Tan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26150](https://doi.org/10.1609/aaai.v37i8.26150)

**Abstract**:

Recent model-extraction attacks on Machine Learning as a Service (MLaaS) systems have moved towards data-free approaches, showing the feasibility of stealing models trained with difficult-to-access data. However, these attacks are ineffective or limited due to the low accuracy of extracted models and the high number of queries to the models under attack. The high query cost makes such techniques infeasible for online MLaaS systems that charge per query.
We create a novel approach to get higher accuracy and query efficiency than prior data-free model extraction techniques. Specifically, we introduce a novel generator training scheme that maximizes the disagreement loss between two clone models that attempt to copy the model under attack. This loss, combined with diversity loss and experience replay, enables the generator to produce better instances to train the clone models. Our evaluation on popular datasets CIFAR-10 and CIFAR-100 shows that our approach improves the final model accuracy by up to 3.42% and 18.48% respectively. The average number of queries required to achieve the accuracy of the prior state of the art is reduced by up to 64.95%. We hope this will promote future work on feasible data-free model extraction and defenses against such attacks.

----

## [1081] Overcoming Concept Shift in Domain-Aware Settings through Consolidated Internal Distributions

**Authors**: *Mohammad Rostami, Aram Galstyan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26151](https://doi.org/10.1609/aaai.v37i8.26151)

**Abstract**:

We develop an algorithm to improve the predictive performance of a pre-trained model under \textit{concept shift} without retraining the model from scratch when only unannotated samples of initial concepts are accessible. We model this problem as a domain adaptation problem, where the source domain data is inaccessible during model adaptation. The core idea is based on consolidating the intermediate internal distribution, learned to represent the source domain data, after adapting the model. We provide theoretical analysis and conduct extensive experiments on five benchmark datasets to demonstrate that the proposed method is effective.

----

## [1082] Inferring Patient Zero on Temporal Networks via Graph Neural Networks

**Authors**: *Xiaolei Ru, Jack Murdoch Moore, Xinya Zhang, Yeting Zeng, Gang Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26152](https://doi.org/10.1609/aaai.v37i8.26152)

**Abstract**:

The world is currently seeing frequent local outbreaks of epidemics, such as COVID-19 and Monkeypox. Preventing further propagation of the outbreak requires prompt implementation of control measures, and a critical step is to quickly infer patient zero. This backtracking task is challenging for two reasons. First, due to the sudden emergence of local epidemics, information recording the spreading process is limited. Second, the spreading process has strong randomness. To address these challenges, we tailor a gnn-based model to establish the inverse statistical association between the current and initial state implicitly. This model uses contact topology and the current state of the local population to determine the possibility that each individual could be patient zero. We benchmark our model on data from important epidemiological models on five real temporal networks, showing performance significantly superior to previous methods. We also demonstrate that our method is robust to missing information about contact structure or current state. Further, we find the individuals assigned higher inferred possibility by model are closer to patient zero in terms of core number and the activity sequence recording the times at which the individual had contact with other nodes.

----

## [1083] Accommodating Audio Modality in CLIP for Multimodal Processing

**Authors**: *Ludan Ruan, Anwen Hu, Yuqing Song, Liang Zhang, Sipeng Zheng, Qin Jin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26153](https://doi.org/10.1609/aaai.v37i8.26153)

**Abstract**:

Multimodal processing has attracted much attention lately especially with the success of pre-training. However, the exploration has mainly focused on vision-language pre-training, as introducing more modalities can greatly complicate model design and optimization. In this paper, we extend the state-of-the-art Vision-Language model CLIP to accommodate the audio modality for Vision-Language-Audio multimodal processing. Specifically, we apply inter-modal and intra-modal contrastive learning to explore the correlation between audio and other modalities in addition to the inner characteristics of the audio modality. Moreover, we further design an audio type token to dynamically learn different audio information type for different scenarios, as both verbal and nonverbal heterogeneous information is conveyed in general audios. Our proposed CLIP4VLA model is validated in different downstream tasks including video retrieval and video captioning, and achieves the state-of-the-art performance on the benchmark datasets of MSR-VTT, VATEX, and Audiocaps.The corresponding code and checkpoints will be released at https://github.com/ludanruan/CLIP4VLA.

----

## [1084] Forecasting with Sparse but Informative Variables: A Case Study in Predicting Blood Glucose

**Authors**: *Harry Rubin-Falcone, Joyce M. Lee, Jenna Wiens*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26154](https://doi.org/10.1609/aaai.v37i8.26154)

**Abstract**:

In time-series forecasting, future target values may be affected by both intrinsic and extrinsic effects. When forecasting blood glucose, for example, intrinsic effects can be inferred from the history of the target signal alone (i.e. blood glucose), but accurately modeling the impact of extrinsic effects requires auxiliary signals, like the amount of carbohydrates ingested. Standard forecasting techniques often assume that extrinsic and intrinsic effects vary at similar rates. However, when auxiliary signals are generated at a much lower frequency than the target variable (e.g., blood glucose measurements are made every 5 minutes, while meals occur once every few hours), even well-known extrinsic effects (e.g., carbohydrates increase blood glucose) may prove difficult to learn. To better utilize these sparse but informative variables (SIVs), we introduce a novel encoder/decoder forecasting approach that accurately learns the per-timepoint effect of the SIV, by (i) isolating it from intrinsic effects and (ii) restricting its learned effect based on domain knowledge. On a simulated dataset pertaining to the task of blood glucose forecasting, when the SIV is accurately recorded our approach outperforms baseline approaches in terms of rMSE (13.07 [95% CI: 11.77,14.16] vs. 14.14 [12.69,15.27]). In the presence of a corrupted SIV, the proposed approach can still result in lower error compared to the baseline but the advantage is reduced as noise increases. By isolating their effects and incorporating domain knowledge, our approach makes it possible to better utilize SIVs in forecasting.

----

## [1085] On the Sample Complexity of Representation Learning in Multi-Task Bandits with Global and Local Structure

**Authors**: *Alessio Russo, Alexandre Proutière*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26155](https://doi.org/10.1609/aaai.v37i8.26155)

**Abstract**:

We investigate the sample complexity of learning the optimal arm for multi-task bandit problems. Arms consist of two components: one that is shared across tasks (that we call representation) and one that is task-specific (that we call predictor). 
The objective is to learn the optimal (representation, predictor)-pair for each task, under the assumption that the optimal representation is common to all tasks. Within this framework, efficient learning algorithms should transfer knowledge across tasks. 
We consider the best-arm identification problem with fixed confidence, where, in each round, the learner actively selects both a task, and an arm, and observes the corresponding reward.
We derive instance-specific sample complexity lower bounds, which apply to any algorithm that identifies the best representation, and the best predictor for a task, with prescribed confidence levels.   
We devise an algorithm, OSRL-SC, that can learn the optimal representation, and the optimal predictors, separately, and whose sample complexity approaches the lower bound. Theoretical and numerical results demonstrate that OSRL-SC achieves a better scaling with respect to the number of tasks compared to the classical best-arm identification algorithm.
The code can be found here https://github.com/rssalessio/OSRL-SC.

----

## [1086] Simultaneously Updating All Persistence Values in Reinforcement Learning

**Authors**: *Luca Sabbioni, Luca Al Daire, Lorenzo Bisi, Alberto Maria Metelli, Marcello Restelli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26156](https://doi.org/10.1609/aaai.v37i8.26156)

**Abstract**:

In Reinforcement Learning, the performance of learning agents is highly sensitive to the choice of time discretization. Agents acting at high frequencies have the best control opportunities, along with some drawbacks, such as possible inefficient exploration and vanishing of the action advantages. The repetition of the actions, i.e., action persistence, comes into help, as it allows the agent to visit wider regions of the state space and improve the estimation of the action effects. In this work, we derive a novel operator, the All-Persistence Bellman Operator, which allows an effective use of both the low-persistence experience, by decomposition into sub-transition, and the high-persistence experience, thanks to the introduction of a suitable bootstrap procedure. In this way, we employ transitions collected at any time scale to update simultaneously the action values of the considered persistence set. We prove the contraction property of the All-Persistence Bellman Operator and, based on it, we extend classic Q-learning and DQN. After providing a study on the effects of persistence, we experimentally evaluate our approach in both tabular contexts and more challenging frameworks, including some Atari games.

----

## [1087] Continual Learning with Scaled Gradient Projection

**Authors**: *Gobinda Saha, Kaushik Roy*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26157](https://doi.org/10.1609/aaai.v37i8.26157)

**Abstract**:

In neural networks, continual learning results in gradient interference among sequential tasks, leading to catastrophic forgetting of old tasks while learning new ones. This issue is addressed in recent methods by storing the important gradient spaces for old tasks and updating the model orthogonally during new tasks. However, such restrictive orthogonal gradient updates hamper the learning capability of the new tasks resulting in sub-optimal performance. To improve new learning while minimizing forgetting, in this paper we propose a Scaled Gradient Projection (SGP) method, where we combine the orthogonal gradient projections with scaled gradient steps along the important gradient spaces for the past tasks. The degree of gradient scaling along these spaces depends on the importance of the bases spanning them. We propose an efficient method for computing and accumulating importance of these bases using the singular value decomposition of the input representations for each task. We conduct extensive experiments ranging from continual image classification to reinforcement learning tasks and report better performance with less training overhead than the state-of-the-art approaches.

----

## [1088] Fast Offline Policy Optimization for Large Scale Recommendation

**Authors**: *Otmane Sakhi, David Rohde, Alexandre Gilotte*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26158](https://doi.org/10.1609/aaai.v37i8.26158)

**Abstract**:

Personalised interactive systems such as recommender systems require selecting relevant items from massive catalogs dependent on context. Reward-driven offline optimisation of these systems can be achieved by a relaxation of the discrete problem resulting in policy learning or REINFORCE style learning algorithms. Unfortunately, this relaxation step requires computing a sum over the entire catalogue making the complexity of the evaluation of the gradient (and hence each stochastic gradient descent iterations) linear in the catalogue size. This calculation is untenable in many real world examples such as large catalogue recommender systems, severely limiting the usefulness of this method in practice. In this paper, we derive an approximation of these policy learning algorithms that scale logarithmically with the catalogue size. Our contribution is based upon combining three novel ideas: a new Monte Carlo estimate of the gradient of a policy, the self normalised importance sampling estimator and the use of fast maximum inner product search at training time. Extensive experiments show that our algorithm is an order of magnitude faster than naive approaches yet produces equally good policies.

----

## [1089] Losses over Labels: Weakly Supervised Learning via Direct Loss Construction

**Authors**: *Dylan Sam, J. Zico Kolter*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26159](https://doi.org/10.1609/aaai.v37i8.26159)

**Abstract**:

Owing to the prohibitive costs of generating large amounts of labeled data, programmatic weak supervision is a growing paradigm within machine learning. In this setting, users design heuristics that provide noisy labels for subsets of the data. These weak labels are combined (typically via a graphical model) to form pseudolabels, which are then used to train a downstream model. In this work, we question a foundational premise of the typical weakly supervised learning pipeline: given that the heuristic provides all “label” information, why do we need to generate pseudolabels at all? Instead, we propose to directly transform the heuristics themselves into corresponding loss functions that penalize differences between our model and the heuristic. By constructing losses directly from the heuristics, we can incorporate more information than is used in the standard weakly supervised pipeline, such as how the heuristics make their decisions, which explicitly informs feature selection during training. We call our method Losses over Labels (LoL) as it creates losses directly from heuristics without going through the intermediate step of a label. We show that LoL improves upon existing weak supervision methods on several benchmark text and image classification tasks and further demonstrate that incorporating gradient information leads to better performance on almost every task.

----

## [1090] Representation Learning by Detecting Incorrect Location Embeddings

**Authors**: *Sepehr Sameni, Simon Jenni, Paolo Favaro*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26160](https://doi.org/10.1609/aaai.v37i8.26160)

**Abstract**:

In this paper, we introduce a novel self-supervised learning (SSL) loss for image representation learning. There is a growing belief that generalization in deep neural networks is linked to their ability to discriminate object shapes. Since object shape is related to the location of its parts, we propose to detect those that have been artificially misplaced. We represent object parts with image tokens and train a ViT to detect which token has been combined with an incorrect positional embedding. We then introduce sparsity in the inputs to make the model more robust to occlusions and to speed up the training. We call our method DILEMMA, which stands for Detection of Incorrect Location EMbeddings with MAsked inputs. We apply DILEMMA to MoCoV3, DINO and SimCLR and show an improvement in their performance of respectively 4.41%, 3.97%, and 0.5% under the same training time and with a linear probing transfer on ImageNet-1K. We also show full fine-tuning improvements of MAE combined with our method on ImageNet-100. We evaluate our method via fine-tuning on common SSL benchmarks. Moreover, we show that when downstream tasks are strongly reliant on shape (such as in the YOGA-82 pose dataset), our pre-trained features yield a significant gain over prior work.

----

## [1091] Sparse Coding in a Dual Memory System for Lifelong Learning

**Authors**: *Fahad Sarfraz, Elahe Arani, Bahram Zonooz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26161](https://doi.org/10.1609/aaai.v37i8.26161)

**Abstract**:

Efficient continual learning in humans is enabled by a rich set of neurophysiological mechanisms and interactions between multiple memory systems. The brain efficiently encodes information in non-overlapping sparse codes, which facilitates the learning of new associations faster with controlled interference with previous associations. To mimic sparse coding in DNNs, we enforce activation sparsity along with a dropout mechanism which encourages the model to activate similar units for semantically similar inputs and have less overlap with activation patterns of semantically dissimilar inputs. This provides us with an efficient mechanism for balancing the reusability and interference of features, depending on the similarity of classes across tasks. Furthermore, we employ sparse coding in a multiple-memory replay mechanism. Our method maintains an additional long-term semantic memory that aggregates and consolidates information encoded in the synaptic weights of the working model. Our extensive evaluation and characteristics analysis show that equipped with these biologically inspired mechanisms, the model can further mitigate forgetting. Code available at \url{https://github.com/NeurAI-Lab/SCoMMER}.

----

## [1092] Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Synchronicity

**Authors**: *Pritam Sarkar, Ali Etemad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26162](https://doi.org/10.1609/aaai.v37i8.26162)

**Abstract**:

We present CrissCross, a self-supervised framework for learning audio-visual representations. A novel notion is introduced in our framework whereby in addition to learning the intra-modal and standard 'synchronous' cross-modal relations, CrissCross also learns 'asynchronous' cross-modal relationships. We perform in-depth studies showing that by relaxing the temporal synchronicity between the audio and visual modalities, the network learns strong generalized representations useful for a variety of downstream tasks. To pretrain our proposed solution, we use 3 different datasets with varying sizes, Kinetics-Sound, Kinetics400, and AudioSet. The learned representations are evaluated on a number of downstream tasks namely action recognition, sound classification, and action retrieval. Our experiments show that CrissCross either outperforms or achieves performances on par with the current state-of-the-art self-supervised methods on action recognition and action retrieval with UCF101 and HMDB51, as well as sound classification with ESC50 and DCASE. Moreover, CrissCross outperforms fully-supervised pretraining while pretrained on Kinetics-Sound.

----

## [1093] Dropout Is NOT All You Need to Prevent Gradient Leakage

**Authors**: *Daniel Scheliga, Patrick Maeder, Marco Seeland*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26163](https://doi.org/10.1609/aaai.v37i8.26163)

**Abstract**:

Gradient inversion attacks on federated learning systems reconstruct client training data from exchanged gradient information. To defend against such attacks, a variety of defense mechanisms were proposed. However, they usually lead to an unacceptable trade-off between privacy and model utility. Recent observations suggest that dropout could mitigate gradient leakage and improve model utility if added to neural networks. Unfortunately, this phenomenon has not been systematically researched yet. In this work, we thoroughly analyze the effect of dropout on iterative gradient inversion attacks. We find that state of the art attacks are not able to reconstruct the client data due to the stochasticity induced by dropout during model training. Nonetheless, we argue that dropout does not offer reliable protection if the dropout induced stochasticity is adequately modeled during attack optimization. Consequently, we propose a novel Dropout Inversion Attack (DIA) that jointly optimizes for client data and dropout masks to approximate the stochastic client model. We conduct an extensive systematic evaluation of our attack on four seminal model architectures and three image classification datasets of increasing complexity. We find that our proposed attack bypasses the protection seemingly induced by dropout and reconstructs client data with high fidelity. Our work demonstrates that privacy inducing changes to model architectures alone cannot be assumed to reliably protect from gradient leakage and therefore should be combined with complementary defense mechanisms.

----

## [1094] Exploration via Epistemic Value Estimation

**Authors**: *Simon Schmitt, John Shawe-Taylor, Hado van Hasselt*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26164](https://doi.org/10.1609/aaai.v37i8.26164)

**Abstract**:

How to efficiently explore in reinforcement learning is an open problem. Many exploration algorithms employ the epistemic uncertainty of their own value predictions -- for instance to compute an exploration bonus or upper confidence bound. Unfortunately the required uncertainty is difficult to estimate in general with function approximation.

We propose epistemic value estimation (EVE): a recipe that is compatible with sequential decision making and with neural network function approximators. It equips agents with a tractable posterior over all their parameters from which epistemic value uncertainty can be computed efficiently.

We use the recipe to derive an epistemic Q-Learning agent and observe competitive performance on a series of benchmarks. Experiments confirm that the EVE recipe facilitates efficient exploration in hard exploration tasks.

----

## [1095] Multi-Source Survival Domain Adaptation

**Authors**: *Ammar Shaker, Carolin Lawrence*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26165](https://doi.org/10.1609/aaai.v37i8.26165)

**Abstract**:

Survival analysis is the branch of statistics that studies the relation between the characteristics of living entities and their respective survival times, taking into account the partial information held by censored cases. A good analysis can, for example, determine whether one medical treatment for a group of patients is better than another. With the rise of machine learning, survival analysis can be modeled as learning a function that maps studied patients to their survival times. To succeed with that, there are three crucial issues to be tackled. 
 First, some patient data is censored: we do not know the true survival times for all patients. Second, data is scarce, which led past research to treat different illness types as domains in a multi-task setup. Third, there is the need for adaptation to new or extremely rare illness types, where little or no labels are available. In contrast to previous multi-task setups, we want to investigate how to efficiently adapt to a new survival target domain from multiple survival source domains. 
 For this, we introduce a new survival metric and the corresponding discrepancy measure between survival distributions. These allow us to define domain adaptation for survival analysis while incorporating censored data, which would otherwise have to be dropped. Our experiments on two cancer data sets reveal a superb performance on target domains, a better treatment recommendation, and a weight matrix with a plausible explanation.

----

## [1096] What Do You MEME? Generating Explanations for Visual Semantic Role Labelling in Memes

**Authors**: *Shivam Sharma, Siddhant Agarwal, Tharun Suresh, Preslav Nakov, Md. Shad Akhtar, Tanmoy Chakraborty*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26166](https://doi.org/10.1609/aaai.v37i8.26166)

**Abstract**:

Memes are powerful means for effective communication on social media. Their effortless amalgamation of viral visuals and compelling messages can have far-reaching implications with proper marketing. Previous research on memes has primarily focused on characterizing their affective spectrum and detecting whether the meme's message insinuates any intended harm, such as hate, offense, racism, etc. However, memes often use abstraction, which can be elusive. Here, we introduce a novel task - EXCLAIM, generating explanations for visual semantic role labeling in memes. To this end, we curate ExHVV, a novel dataset that offers natural language explanations of connotative roles for three types of entities - heroes, villains, and victims, encompassing 4,680 entities present in 3K memes. We also benchmark ExHVV with several strong unimodal and multimodal baselines. Moreover, we posit LUMEN, a novel multimodal, multi-task learning framework that endeavors to address EXCLAIM optimally by jointly learning to predict the correct semantic roles and correspondingly to generate suitable natural language explanations. LUMEN distinctly outperforms the best baseline across 18 standard natural language generation evaluation metrics. Our systematic evaluation and analyses demonstrate that characteristic multimodal cues required for adjudicating semantic roles are also helpful for generating suitable explanations.

----

## [1097] Post-hoc Uncertainty Learning Using a Dirichlet Meta-Model

**Authors**: *Maohao Shen, Yuheng Bu, Prasanna Sattigeri, Soumya Ghosh, Subhro Das, Gregory W. Wornell*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26167](https://doi.org/10.1609/aaai.v37i8.26167)

**Abstract**:

It is known that neural networks have the problem of being over-confident when directly using the output label distribution to generate uncertainty measures. Existing methods mainly resolve this issue by retraining the entire model to impose the uncertainty quantification capability so that the learned model can achieve desired performance in accuracy and uncertainty prediction simultaneously. However, training the model from scratch is computationally expensive, and a trade-off might exist between prediction accuracy and uncertainty quantification. To this end, we consider a more practical post-hoc uncertainty learning setting, where a well-trained base model is given, and we focus on the uncertainty quantification task at the second stage of training. We propose a novel Bayesian uncertainty learning approach using the Dirichlet meta-model, which is effective and computationally efficient. Our proposed method requires no additional training data and is flexible enough to quantify different uncertainties and easily adapt to different application settings, including out-of-domain data detection, misclassification detection, and trustworthy transfer learning. Finally, we demonstrate our proposed meta-model approach's flexibility and superior empirical performance on these applications over multiple representative image classification benchmarks.

----

## [1098] Neighbor Contrastive Learning on Learnable Graph Augmentation

**Authors**: *Xiao Shen, Dewang Sun, Shirui Pan, Xi Zhou, Laurence T. Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26168](https://doi.org/10.1609/aaai.v37i8.26168)

**Abstract**:

Recent years, graph contrastive learning (GCL), which aims to learn representations from unlabeled graphs, has made great progress. However, the existing GCL methods mostly adopt human-designed graph augmentations, which are sensitive to various graph datasets. In addition, the contrastive losses originally developed in computer vision have been directly applied to graph data, where the neighboring nodes are regarded as negatives and consequently pushed far apart from the anchor. However, this is contradictory with the homophily assumption of net-works that connected nodes often belong to the same class and should be close to each other. In this work, we propose an end-to-end automatic GCL method, named NCLA to apply neighbor contrastive learning on learnable graph augmentation. Several graph augmented views with adaptive topology are automatically learned by the multi-head graph attention mechanism, which can be compatible with various graph datasets without prior domain knowledge. In addition, a neighbor contrastive loss is devised to allow multiple positives per anchor by taking network topology as the supervised signals. Both augmentations and embeddings are learned end-to-end in the proposed NCLA. Extensive experiments on the benchmark datasets demonstrate that NCLA yields the state-of-the-art node classification performance on self-supervised GCL and even exceeds the supervised ones, when the labels are extremely limited. Our code is released at https://github.com/shenxiaocam/NCLA.

----

## [1099] ProxyBO: Accelerating Neural Architecture Search via Bayesian Optimization with Zero-Cost Proxies

**Authors**: *Yu Shen, Yang Li, Jian Zheng, Wentao Zhang, Peng Yao, Jixiang Li, Sen Yang, Ji Liu, Bin Cui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26169](https://doi.org/10.1609/aaai.v37i8.26169)

**Abstract**:

Designing neural architectures requires immense manual efforts. This has promoted the development of neural architecture search (NAS) to automate the design. While previous NAS methods achieve promising results but run slowly, zero-cost proxies run extremely fast but are less promising. Therefore, it’s of great potential to accelerate NAS via those zero-cost proxies. The existing method has two limitations, which are unforeseeable reliability and one-shot usage. To address the limitations, we present ProxyBO, an efficient Bayesian optimization (BO) framework that utilizes the zero-cost proxies to accelerate neural architecture search. We apply the generalization ability measurement to estimate the fitness of proxies on the task during each iteration and design a novel acquisition function to combine BO with zero-cost proxies based on their dynamic influence. Extensive empirical studies show that ProxyBO consistently outperforms competitive baselines on five tasks from three public benchmarks. Concretely, ProxyBO achieves up to 5.41× and 3.86× speedups over the state-of-the-art approaches REA and BRP-NAS.

----

## [1100] Contrastive Predictive Autoencoders for Dynamic Point Cloud Self-Supervised Learning

**Authors**: *Xiaoxiao Sheng, Zhiqiang Shen, Gang Xiao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26170](https://doi.org/10.1609/aaai.v37i8.26170)

**Abstract**:

We present a new self-supervised paradigm on point cloud sequence understanding. Inspired by the discriminative and generative self-supervised methods, we design two tasks, namely point cloud sequence based Contrastive Prediction and Reconstruction (CPR), to collaboratively learn more comprehensive spatiotemporal representations. Specifically, dense point cloud segments are first input into an encoder to extract embeddings. All but the last ones are then aggregated by a context-aware autoregressor to make predictions for the last target segment. Towards the goal of modeling multi-granularity structures, local and global contrastive learning are performed between predictions and targets. To further improve the generalization of representations, the predictions are also utilized to reconstruct raw point cloud sequences by a decoder, where point cloud colorization is employed to discriminate against different frames. By combining classic contrast and reconstruction paradigms, it makes the learned representations with both global discrimination and local perception. We conduct experiments on four point cloud sequence benchmarks, and report the results on action recognition and gesture recognition under multiple experimental settings. The performances are comparable with supervised methods and show powerful transferability.

----

## [1101] Fixed-Weight Difference Target Propagation

**Authors**: *Tatsukichi Shibuya, Nakamasa Inoue, Rei Kawakami, Ikuro Sato*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26171](https://doi.org/10.1609/aaai.v37i8.26171)

**Abstract**:

Target Propagation (TP) is a biologically more plausible algorithm than the error backpropagation (BP) to train deep networks, and improving practicality of TP is an open issue. TP methods  require the feedforward and feedback networks to form layer-wise autoencoders for propagating the target values generated at the output layer. However, this causes certain drawbacks; e.g., careful hyperparameter tuning is required to synchronize the feedforward and feedback training, and frequent updates of the feedback path are usually required than that of the feedforward path. Learning of the feedforward and feedback networks is sufficient to make TP methods capable of training, but is having these layer-wise autoencoders a necessary condition for TP to work? We answer this question by presenting Fixed-Weight Difference Target Propagation (FW-DTP) that keeps the feedback weights constant during training. We confirmed that this simple method, which naturally resolves the abovementioned problems of TP, can still deliver informative target values to hidden layers for a given task; indeed, FW-DTP consistently achieves higher test performance than a baseline, the Difference Target Propagation (DTP), on four classification datasets. We also present a novel propagation architecture that explains the exact form of the feedback function of DTP to analyze FW-DTP. Our code is available at https://github.com/TatsukichiShibuya/Fixed-Weight-Difference-Target-Propagation.

----

## [1102] Concurrent Multi-Label Prediction in Event Streams

**Authors**: *Xiao Shou, Tian Gao, Dharmashankar Subramanian, Debarun Bhattacharjya, Kristin P. Bennett*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26172](https://doi.org/10.1609/aaai.v37i8.26172)

**Abstract**:

Streams of irregularly occurring events are commonly modeled as a marked temporal point process. Many real-world datasets such as e-commerce transactions and electronic health records often involve events where multiple event types co-occur, e.g. multiple items purchased or multiple diseases diagnosed simultaneously. In this paper, we tackle multi-label prediction in such a problem setting, and propose a novel Transformer-based Conditional Mixture of Bernoulli Network (TCMBN) that leverages neural density estimation to capture complex temporal dependence as well as probabilistic dependence between concurrent event types. We also propose potentially incorporating domain knowledge in the objective by regularizing the predicted probability. To represent probabilistic dependence of concurrent event types graphically, we design a two-step approach that first learns the mixture of Bernoulli network and then solves a least-squares semi-definite constrained program to numerically approximate the sparse precision matrix from a learned covariance matrix. This approach proves to be effective for event prediction while also providing an interpretable and possibly non-stationary structure for insights into event co-occurrence. We demonstrate the superior performance of our approach compared to existing baselines on multiple synthetic and real benchmarks.

----

## [1103] A Generalized Unbiased Risk Estimator for Learning with Augmented Classes

**Authors**: *Senlin Shu, Shuo He, Haobo Wang, Hongxin Wei, Tao Xiang, Lei Feng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26173](https://doi.org/10.1609/aaai.v37i8.26173)

**Abstract**:

In contrast to the standard learning paradigm where all classes can be observed in training data, learning with augmented classes (LAC) tackles the problem where augmented classes unobserved in the training data may emerge in the test phase. Previous research showed that given unlabeled data, an unbiased risk estimator (URE) can be derived, which can be minimized for LAC with theoretical guarantees. However, this URE is only restricted to the specific type of one-versus-rest loss functions for multi-class classification, making it not flexible enough when the loss needs to be changed with the dataset in practice. In this paper, we propose a generalized URE that can be equipped with arbitrary loss functions while maintaining the theoretical guarantees, given unlabeled data for LAC. To alleviate the issue of negative empirical risk commonly encountered by previous studies, we further propose a novel risk-penalty regularization term. Experiments demonstrate the effectiveness of our proposed method.

----

## [1104] Logical Satisfiability of Counterfactuals for Faithful Explanations in NLI

**Authors**: *Suzanna Sia, Anton Belyy, Amjad Almahairi, Madian Khabsa, Luke Zettlemoyer, Lambert Mathias*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26174](https://doi.org/10.1609/aaai.v37i8.26174)

**Abstract**:

Evaluating an explanation's faithfulness is desired for many reasons such as trust, interpretability and diagnosing the sources of model's errors. In this work, which focuses on the NLI task, we introduce the methodology of Faithfulness-through-Counterfactuals, which first generates a counterfactual hypothesis based on the logical predicates expressed in the explanation, and then evaluates if the model's prediction on the counterfactual is consistent with that expressed logic (i.e. if the new formula is \textit{logically satisfiable}). In contrast to existing approaches, this does not require any explanations for training a separate verification model. We first validate the efficacy of automatic counterfactual hypothesis generation, leveraging on the few-shot priming paradigm. Next, we show that our proposed metric distinguishes between human-model agreement and disagreement on new counterfactual input. In addition, we conduct a sensitivity analysis to validate that our metric is sensitive to unfaithful explanations.

----

## [1105] SLIQ: Quantum Image Similarity Networks on Noisy Quantum Computers

**Authors**: *Daniel Silver, Tirthak Patel, Aditya Ranjan, Harshitta Gandhi, William Cutler, Devesh Tiwari*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26175](https://doi.org/10.1609/aaai.v37i8.26175)

**Abstract**:

Exploration into quantum machine learning has grown
tremendously in recent years due to the ability of quantum
computers to speed up classical programs. However, these ef-
forts have yet to solve unsupervised similarity detection tasks
due to the challenge of porting them to run on quantum com-
puters. To overcome this challenge, we propose SLIQ, the
first open-sourced work for resource-efficient quantum sim-
ilarity detection networks, built with practical and effective
quantum learning and variance-reducing algorithms.

----

## [1106] Adaptive Mixing of Auxiliary Losses in Supervised Learning

**Authors**: *Durga Sivasubramanian, Ayush Maheshwari, Prathosh AP, Pradeep Shenoy, Ganesh Ramakrishnan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26176](https://doi.org/10.1609/aaai.v37i8.26176)

**Abstract**:

In many supervised learning scenarios, auxiliary losses are used in order to introduce additional information or constraints into the supervised learning objective. For instance, knowledge distillation aims to mimic outputs of a powerful teacher model; similarly, in rule-based approaches, weak labeling information is provided by labeling functions which may be noisy rule-based approximations to true labels. We tackle the problem of learning to combine these losses in a principled manner. Our proposal, AMAL, uses a bi-level optimization criterion on validation data to learn optimal mixing weights, at an instance-level, over the training data. We describe a meta-learning approach towards solving this bi-level objective, and show how it can be applied to different scenarios in supervised learning. Experiments in a number of knowledge distillation and rule denoising domains show that AMAL provides noticeable gains over competitive baselines in those domains. We empirically analyze our method and share insights into the mechanisms through which it provides performance gains. The code for AMAL is at: https://github.com/durgas16/AMAL.git.

----

## [1107] Securing Secure Aggregation: Mitigating Multi-Round Privacy Leakage in Federated Learning

**Authors**: *Jinhyun So, Ramy E. Ali, Basak Güler, Jiantao Jiao, Amir Salman Avestimehr*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26177](https://doi.org/10.1609/aaai.v37i8.26177)

**Abstract**:

Secure aggregation is a critical component in federated learning (FL), which enables the server to learn the aggregate model of the users without observing their local models. Conventionally, secure aggregation algorithms focus only on ensuring the privacy of individual users in a single training round. We contend that such designs can lead to significant privacy leakages over multiple training rounds, due to partial user selection/participation at each round of FL. In fact, we show that the conventional random user selection strategies in FL lead to leaking users' individual models within number of rounds that is linear in the number of users.
To address this challenge, we introduce a secure aggregation framework, Multi-RoundSecAgg, with multi-round privacy guarantees.
In particular, we introduce a new metric to quantify the privacy guarantees of FL over multiple training rounds, and develop a structured user selection strategy that guarantees the long-term privacy of each user (over any number of training rounds). 
Our framework also carefully accounts for the fairness and the average number of participating users at each round.
Our experiments on MNIST, CIFAR-10 and CIFAR-100 datasets in the IID and the non-IID settings demonstrate the performance improvement over the baselines, both in terms of privacy protection and test accuracy.

----

## [1108] Mixture Manifold Networks: A Computationally Efficient Baseline for Inverse Modeling

**Authors**: *Gregory P. Spell, Simiao Ren, Leslie M. Collins, Jordan M. Malof*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26178](https://doi.org/10.1609/aaai.v37i8.26178)

**Abstract**:

We propose and show the efficacy of a new method to address generic inverse problems. Inverse modeling is the task whereby one seeks to determine the hidden parameters of a natural system that produce a given set of observed measurements. Recent work has shown impressive results using deep learning, but we note that there is a trade-off between model performance and computational time. For some applications, the computational time at inference for the best performing inverse modeling method may be overly prohibitive to its use. In seeking a faster, high-performing model, we present a new method that leverages multiple manifolds as a mixture of backward (e.g., inverse) models in a forward-backward model architecture. These multiple backwards models all share a common forward model, and their training is mitigated by generating training examples from the forward model. The proposed method thus has two innovations: 1) the multiple Manifold Mixture Network (MMN) architecture, and 2) the training procedure involving augmenting backward model training data using the forward model. We demonstrate the advantages of our method by comparing to several baselines on four benchmark inverse problems, and we furthermore provide analysis to motivate its design.

----

## [1109] Sharing Pattern Submodels for Prediction with Missing Values

**Authors**: *Lena Stempfle, Ashkan Panahi, Fredrik D. Johansson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26179](https://doi.org/10.1609/aaai.v37i8.26179)

**Abstract**:

Missing values are unavoidable in many applications of machine learning and present challenges both during training and at test time. When variables are missing in recurring patterns, fitting separate pattern submodels have been proposed as a solution. However, fitting models independently does not make efficient use of all available data. Conversely, fitting a single shared model to the full data set relies on imputation which often leads to biased results when missingness depends on unobserved factors. We propose an alternative approach, called sharing pattern submodels (SPSM), which i) makes predictions that are robust to missing values at test time, ii) maintains or improves the predictive power of pattern submodels, and iii) has a short description, enabling improved interpretability. Parameter sharing is enforced through sparsity-inducing regularization which we prove leads to consistent estimation. Finally, we give conditions for when a sharing model is optimal, even when both missingness and the target outcome depend on unobserved variables. Classification and regression experiments on synthetic and real-world data sets demonstrate that our models achieve a favorable tradeoff between pattern specialization and information sharing.

----

## [1110] Scalable Optimal Multiway-Split Decision Trees with Constraints

**Authors**: *Shivaram Subramanian, Wei Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26180](https://doi.org/10.1609/aaai.v37i8.26180)

**Abstract**:

There has been a surge of interest in learning optimal decision trees using mixed-integer programs (MIP) in recent years, as heuristic-based methods do not guarantee optimality and find it challenging to incorporate constraints that are critical for many practical applications.  However, existing MIP methods that build on an arc-based formulation do not scale well as the number of binary variables is in the order of 2 to the power of the depth of the tree and the size of the dataset. Moreover, they can only handle sample-level constraints and linear metrics. In this paper, we propose a novel path-based MIP formulation where the number of decision variables is independent of dataset size. We present a scalable column generation framework to solve the MIP. Our framework produces a multiway-split tree which is more interpretable than the typical binary-split trees due to its shorter rules. Our framework is more general as it can handle nonlinear metrics such as F1 score, and incorporate a broader class of constraints. We demonstrate its efficacy with extensive experiments. We present results on datasets containing up to 1,008,372  samples while existing MIP-based decision tree models do not scale well on data beyond a few thousand points. We report superior or competitive results compared to the state-of-art MIP-based methods with up to a 24X reduction in runtime.

----

## [1111] REMIT: Reinforced Multi-Interest Transfer for Cross-Domain Recommendation

**Authors**: *Caiqi Sun, Jiewei Gu, Binbin Hu, Xin Dong, Hai Li, Lei Cheng, Linjian Mo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26181](https://doi.org/10.1609/aaai.v37i8.26181)

**Abstract**:

Cold-start problem is one of the most challenging problems for recommender systems. One promising solution to this problem is cross-domain recommendation (CDR) which leverages rich information from an auxiliary source domain to improve the performance of recommender system in the target domain. In particular, the family of embedding and mapping methods for CDR is very effective, which explicitly learn a mapping function from source embeddings to target embeddings to transfer user’s preferences. Recent works usually transfer an overall source embedding by modeling a common or personalized preference bridge for all users. However, a unified user embedding cannot reflect the user’s multiple interests in auxiliary source domain. In this paper, we propose a novel framework called reinforced multi-interest transfer for CDR (REMIT). Specifically, we first construct a heterogeneous information network and employ different meta-path based aggregations to get user’s multiple interests in source domain, then transform different interest embeddings with different meta-generated personalized bridge functions for each user. To better coordinate the transformed user interest embeddings and the item embedding in target domain, we systematically develop a reinforced method to dynamically assign weights to transformed interests for different training instances and optimize the performance of target model. In addition, the REMIT is a general framework that can be applied upon various base models in target domain. Our extensive experimental results on large real-world datasets demonstrate the superior performance and compatibility of REMIT.

----

## [1112] Cooperative and Adversarial Learning: Co-enhancing Discriminability and Transferability in Domain Adaptation

**Authors**: *Hui Sun, Zheng Xie, Xin-Ye Li, Ming Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26182](https://doi.org/10.1609/aaai.v37i8.26182)

**Abstract**:

Discriminability and transferability are two goals of feature learning for domain adaptation (DA), as we aim to find the transferable features from the source domain that are helpful for discriminating the class label in the target domain. Modern DA approaches optimize discriminability and transferability by adopting two separate modules for the two goals upon a feature extractor, but lack fully exploiting their relationship. This paper argues that by letting the discriminative module and transfer module help each other, better DA can be achieved. We propose Cooperative and Adversarial LEarning (CALE) to combine the optimization of discriminability and transferability into a whole, provide one solution for making the discriminative module and transfer module guide each other. Specifically, CALE generates cooperative (easy) examples and adversarial (hard) examples with both discriminative module and transfer module. While the easy examples that contain the module knowledge can be used to enhance each other, the hard ones are used to enhance the robustness of the corresponding goal. Experimental results show the effectiveness of CALE for unifying the learning of discriminability and transferability, as well as its superior performance.

----

## [1113] Fair-CDA: Continuous and Directional Augmentation for Group Fairness

**Authors**: *Rui Sun, Fengwei Zhou, Zhenhua Dong, Chuanlong Xie, Lanqing Hong, Jiawei Li, Rui Zhang, Zhen Li, Zhenguo Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26183](https://doi.org/10.1609/aaai.v37i8.26183)

**Abstract**:

In this work, we propose Fair-CDA, a fine-grained data augmentation strategy for imposing fairness constraints. We use a feature disentanglement method to extract the features highly related to the sensitive attributes. Then we show that group fairness can be achieved by regularizing the models on transition paths of sensitive features between groups. By adjusting the perturbation strength in the direction of the paths, our proposed augmentation is controllable and auditable. To alleviate the accuracy degradation caused by fairness constraints, we further introduce a calibrated model to impute labels for the augmented data. Our proposed method does not assume any data generative model and ensures good generalization for both accuracy and fairness. Experimental results show that Fair-CDA consistently outperforms state-of-the-art methods on widely-used benchmarks, e.g., Adult, CelebA and MovieLens. Especially, Fair-CDA obtains an 86.3% relative improvement for fairness while maintaining the accuracy on the Adult dataset. Moreover, we evaluate Fair-CDA in an online recommendation system to demonstrate the effectiveness of our method in terms of accuracy and fairness.

----

## [1114] Neural Spline Search for Quantile Probabilistic Modeling

**Authors**: *Ruoxi Sun, Chun-Liang Li, Sercan Ö. Arik, Michael W. Dusenberry, Chen-Yu Lee, Tomas Pfister*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26184](https://doi.org/10.1609/aaai.v37i8.26184)

**Abstract**:

Accurate estimation of output quantiles is crucial in many use cases, where it is desired to model the range of possibility. Modeling target distribution at arbitrary quantile levels and at arbitrary input attribute levels are important to offer a comprehensive picture of the data, and requires the quantile function to be expressive enough. The quantile function describing the target distribution using quantile levels is critical for quantile regression. Although various parametric forms for the distributions (that the quantile function specifies) can be adopted, an everlasting problem is selecting the most appropriate one that can properly approximate the data distributions. In this paper, we propose a non-parametric and data-driven approach,
Neural Spline Search (NSS), to represent the observed data distribution without parametric assumptions. NSS is flexible and expressive for modeling data distributions by transforming the inputs with a series of monotonic spline regressions guided by symbolic operators. We demonstrate that NSS outperforms previous methods on synthetic, real-world regression and time-series forecasting tasks.

----

## [1115] Domain Adaptation with Adversarial Training on Penultimate Activations

**Authors**: *Tao Sun, Cheng Lu, Haibin Ling*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26185](https://doi.org/10.1609/aaai.v37i8.26185)

**Abstract**:

Enhancing model prediction confidence on target data is an important objective in Unsupervised Domain Adaptation (UDA). In this paper, we explore adversarial training on penultimate activations, i.e., input features of the final linear classification layer. We show that this strategy is more efficient and better correlated with the objective of boosting prediction confidence than adversarial training on input images or intermediate features, as used in previous works. Furthermore, with activation normalization  commonly used in domain adaptation to reduce domain gap, we derive two variants and systematically analyze the effects of normalization on our adversarial training. This is illustrated both in theory and through empirical analysis on real adaptation tasks. Extensive experiments are conducted on popular UDA benchmarks under both standard setting and source-data free setting. The results validate that our method achieves the best scores against previous arts. Code is available at https://github.com/tsun/APA.

----

## [1116] Fast Convergence in Learning Two-Layer Neural Networks with Separable Data

**Authors**: *Hossein Taheri, Christos Thrampoulidis*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26186](https://doi.org/10.1609/aaai.v37i8.26186)

**Abstract**:

Normalized gradient descent has shown substantial success in speeding up the convergence of  exponentially-tailed loss functions (which includes exponential and logistic losses) on linear classifiers with separable data.  In this paper, we go beyond linear models by studying normalized GD on two-layer neural nets. We prove for exponentially-tailed losses that using normalized GD leads to linear rate of convergence of the training loss to the global optimum. This is made possible by showing certain gradient self-boundedness conditions and a log-Lipschitzness property. We also study generalization of normalized GD for convex objectives via an algorithmic-stability analysis. In particular, we show that normalized GD does not overfit during training by establishing finite-time generalization bounds.

----

## [1117] Federated Learning on Non-IID Graphs via Structural Knowledge Sharing

**Authors**: *Yue Tan, Yixin Liu, Guodong Long, Jing Jiang, Qinghua Lu, Chengqi Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26187](https://doi.org/10.1609/aaai.v37i8.26187)

**Abstract**:

Graph neural networks (GNNs) have shown their superiority in modeling graph data. Owing to the advantages of federated learning, federated graph learning (FGL) enables clients to train strong GNN models in a distributed manner without sharing their private data. A core challenge in federated systems is the non-IID problem, which also widely exists in real-world graph data. For example, local data of clients may come from diverse datasets or even domains, e.g., social networks and molecules, increasing the difficulty for FGL methods to capture commonly shared knowledge and learn a generalized encoder. From real-world graph datasets, we observe that some structural properties are shared by various domains, presenting great potential for sharing structural knowledge in FGL. Inspired by this, we propose FedStar, an FGL framework that extracts and shares the common underlying structure information for inter-graph federated learning tasks. To explicitly extract the structure information rather than encoding them along with the node features, we define structure embeddings and encode them with an independent structure encoder. Then, the structure encoder is shared across clients while the feature-based knowledge is learned in a personalized way, making FedStar capable of capturing more structure-based domain-invariant information and avoiding feature misalignment issues. We perform extensive experiments over both cross-dataset and cross-domain non-IID FGL settings, demonstrating the superiority of FedStar.

----

## [1118] Metric Multi-View Graph Clustering

**Authors**: *Yuze Tan, Yixi Liu, Hongjie Wu, Jiancheng Lv, Shudong Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26188](https://doi.org/10.1609/aaai.v37i8.26188)

**Abstract**:

Graph-based methods have hitherto been used to pursue the coherent patterns of data due to its ease of implementation and efficiency. These methods have been increasingly applied in multi-view learning and achieved promising performance in various clustering tasks. However, despite their noticeable empirical success, existing graph-based multi-view clustering methods may still suffer the suboptimal solution considering that multi-view data can be very complicated in raw feature space. Moreover, existing methods usually adopt the similarity metric by an ad hoc approach, which largely simplifies the relationship among real-world data and results in an inaccurate output. To address these issues, we propose to seamlessly integrates metric learning and graph learning for multi-view clustering. Specifically, we employ a useful metric to depict the inherent structure with linearity-aware of affinity graph representation learned based on the self-expressiveness property. Furthermore, instead of directly utilizing the raw features, we prefer to recover a smooth representation such that the geometric structure of the original data can be retained. We model the above concerns into a unified learning framework, and hence complements each learning subtask in a mutual reinforcement manner. The empirical studies corroborate our theoretical findings, and demonstrate that the proposed method is able to boost the multi-view clustering performance.

----

## [1119] DE-net: Dynamic Text-Guided Image Editing Adversarial Networks

**Authors**: *Ming Tao, Bing-Kun Bao, Hao Tang, Fei Wu, Longhui Wei, Qi Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26189](https://doi.org/10.1609/aaai.v37i8.26189)

**Abstract**:

Text-guided image editing models have shown remarkable results. However, there remain two problems. First, they employ fixed manipulation modules for various editing requirements (e.g., color changing, texture changing, content adding and removing), which results in over-editing or insufficient editing. Second, they do not clearly distinguish between text-required and text-irrelevant parts, which leads to inaccurate editing.
To solve these limitations, we propose:
(i) a Dynamic Editing Block (DEBlock) that composes different editing modules dynamically for various editing requirements.
(ii) a Composition Predictor (Comp-Pred), which predicts the composition weights for DEBlock according to the inference on target texts and source images.
(iii) a Dynamic text-adaptive Convolution Block (DCBlock) that queries source image features to distinguish text-required parts and text-irrelevant parts.
Extensive experiments demonstrate that our DE-Net achieves excellent performance and manipulates source images more correctly and accurately.

----

## [1120] Knowledge Amalgamation for Multi-Label Classification via Label Dependency Transfer

**Authors**: *Jidapa Thadajarassiri, Thomas Hartvigsen, Walter Gerych, Xiangnan Kong, Elke A. Rundensteiner*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26190](https://doi.org/10.1609/aaai.v37i8.26190)

**Abstract**:

Multi-label classification (MLC), which assigns multiple labels to each instance, is crucial to domains from computer vision to text mining. Conventional methods for MLC require huge amounts of labeled data to capture complex dependencies between labels. However, such labeled datasets are expensive, or even impossible, to acquire. Worse yet, these pre-trained MLC models can only be used for the particular label set covered in the training data. Despite this severe limitation, few methods exist for expanding the set of labels predicted by pre-trained models. Instead, we acquire vast amounts of new labeled data and retrain a new model from scratch. Here, we propose combining the knowledge from multiple pre-trained models (teachers) to train a new student model that covers the union of the labels predicted by this set of teachers. This student supports a broader label set than any one of its teachers without using labeled data. We call this new problem knowledge amalgamation for multi-label classification. Our new method, Adaptive KNowledge Transfer (ANT), trains a student by learning from each teacher’s partial knowledge of label dependencies to infer the global dependencies between all labels across the teachers. We show that ANT succeeds in unifying label dependencies among teachers, outperforming five state-of-the-art methods on eight real-world datasets.

----

## [1121] Leveraging Contaminated Datasets to Learn Clean-Data Distribution with Purified Generative Adversarial Networks

**Authors**: *Bowen Tian, Qinliang Su, Jianxing Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26191](https://doi.org/10.1609/aaai.v37i8.26191)

**Abstract**:

Generative adversarial networks (GANs) are known for their strong abilities on capturing the underlying distribution of training instances. Since the seminal work of GAN, many variants of GAN have been proposed. However, existing GANs are almost established on the assumption that the training dataset is clean. But in many real-world applications, this may not hold, that is, the training dataset may be contaminated by a proportion of undesired instances. When training on such datasets, existing GANs will learn a mixture distribution of desired and contaminated instances, rather than the desired distribution of desired data only (target distribution). To learn the target distribution from contaminated datasets, two purified generative adversarial networks (PuriGAN) are developed, in which the discriminators are augmented with the capability to distinguish between target and contaminated instances by leveraging an extra dataset solely composed of contamination instances. We prove that under some mild conditions, the proposed PuriGANs are guaranteed to converge to the distribution of desired instances. Experimental results on several datasets demonstrate that the proposed PuriGANs are able to generate much better images from the desired distribution than comparable baselines when trained on contaminated datasets. In addition, we also demonstrate the usefulness of PuriGAN on downstream applications by applying it to the tasks of semi-supervised anomaly detection on contaminated datasets and PU-learning. Experimental results show that PuriGAN is able to deliver the best performance over comparable baselines on both tasks.

----

## [1122] Heterogeneous Graph Masked Autoencoders

**Authors**: *Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, Nitesh V. Chawla*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26192](https://doi.org/10.1609/aaai.v37i8.26192)

**Abstract**:

Generative self-supervised learning (SSL), especially masked autoencoders, has become one of the most exciting learning paradigms and has shown great potential in handling graph data. However, real-world graphs are always heterogeneous, which poses three critical challenges that existing methods ignore: 1) how to capture complex graph structure? 2) how to incorporate various node attributes? and 3) how to encode different node positions? In light of this, we study the problem of generative SSL on heterogeneous graphs and propose HGMAE, a novel heterogeneous graph masked autoencoder model to address these challenges. HGMAE captures comprehensive graph information via two innovative masking techniques and three unique training strategies. In particular, we first develop metapath masking and adaptive attribute masking with dynamic mask rate to enable effective and stable learning on heterogeneous graphs. We then design several training strategies including metapath-based edge reconstruction to adopt complex structural information, target attribute restoration to incorporate various node attributes, and positional feature prediction to encode node positional information. Extensive experiments demonstrate that HGMAE outperforms both contrastive and generative state-of-the-art baselines on several tasks across multiple datasets. Codes are available at https://github.com/meettyj/HGMAE.

----

## [1123] Unbalanced CO-optimal Transport

**Authors**: *Quang Huy Tran, Hicham Janati, Nicolas Courty, Rémi Flamary, Ievgen Redko, Pinar Demetci, Ritambhara Singh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26193](https://doi.org/10.1609/aaai.v37i8.26193)

**Abstract**:

Optimal transport (OT) compares probability distributions by computing a meaningful alignment between their samples. CO-optimal transport (COOT) takes this comparison further by inferring an alignment between features as well. While this approach leads to better alignments and generalizes both OT and Gromov-Wasserstein distances, we provide a theoretical result showing that it is sensitive to outliers that are omnipresent in real-world data. This prompts us to propose unbalanced COOT for which we provably show its robustness to noise in the compared datasets. To the best of our knowledge, this is the first such result for OT methods in incomparable spaces. With this result in hand, we provide empirical evidence of this robustness for the challenging tasks of heterogeneous domain adaptation with and without varying proportions of classes and simultaneous alignment of samples and features across two single-cell measurements.

----

## [1124] Linear Regularizers Enforce the Strict Saddle Property

**Authors**: *Matthew Ubl, Matthew T. Hale, Kasra Yazdani*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26194](https://doi.org/10.1609/aaai.v37i8.26194)

**Abstract**:

Satisfaction of the strict saddle property has become a standard assumption in non-convex optimization, and it ensures that many first-order optimization
algorithms will almost always escape saddle points. However, functions exist in machine learning that do not satisfy this property, such as the loss function of a neural network with at least two hidden layers. First-order methods such as gradient descent may converge to non-strict saddle points of such functions, and there do not currently exist any first-order methods that reliably escape non-strict saddle points. To address this need, we demonstrate that regularizing a function with a linear term enforces the strict saddle property, and we provide justification for only regularizing locally, i.e., when the norm of the gradient falls below a certain threshold. We analyze bifurcations that may result from this form of regularization, and then we provide a selection rule for regularizers that depends only on the gradient of an objective function. This rule is shown to guarantee that gradient descent will escape the neighborhoods around a broad class of non-strict saddle points, and this behavior is demonstrated on numerical examples of non-strict saddle points common in the optimization literature.

----

## [1125] Policy-Adaptive Estimator Selection for Off-Policy Evaluation

**Authors**: *Takuma Udagawa, Haruka Kiyohara, Yusuke Narita, Yuta Saito, Kei Tateno*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26195](https://doi.org/10.1609/aaai.v37i8.26195)

**Abstract**:

Off-policy evaluation (OPE) aims to accurately evaluate the performance of counterfactual policies using only offline logged data. Although many estimators have been developed, there is no single estimator that dominates the others, because the estimators' accuracy can vary greatly depending on a given OPE task such as the evaluation policy, number of actions, and noise level. Thus, the data-driven estimator selection problem is becoming increasingly important and can have a significant impact on the accuracy of OPE. However, identifying the most accurate estimator using only the logged data is quite challenging because the ground-truth estimation accuracy of estimators is generally unavailable. This paper thus studies this challenging problem of estimator selection for OPE for the first time. In particular, we enable an estimator selection that is adaptive to a given OPE task, by appropriately subsampling available logged data and constructing pseudo policies useful for the underlying estimator selection task. Comprehensive experiments on both synthetic and real-world company data demonstrate that the proposed procedure substantially improves the estimator selection compared to a non-adaptive heuristic. Note that  complete version with technical appendix is available on arXiv: http://arxiv.org/abs/2211.13904.

----

## [1126] A Fair Generative Model Using LeCam Divergence

**Authors**: *Soobin Um, Changho Suh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26196](https://doi.org/10.1609/aaai.v37i8.26196)

**Abstract**:

We explore a fairness-related challenge that arises in generative models. The challenge is that biased training data with imbalanced demographics may yield a high asymmetry in size of generated samples across distinct groups. We focus on practically-relevant scenarios wherein demographic labels are not available and therefore the design of a fair generative model is non-straightforward. In this paper, we propose an optimization framework that regulates the unfairness under such practical settings via one statistical measure, LeCam (LC)-divergence. Specifically to quantify the degree of unfairness, we employ a balanced-yet-small reference dataset and then measure its distance with generated samples using the LC-divergence, which is shown to be particularly instrumental to a small size of the reference dataset. We take a variational optimization approach to implement the LC-based measure. Experiments on benchmark real datasets demonstrate that the proposed framework can significantly improve the fairness performance while maintaining realistic sample quality for a wide range of the reference set size all the way down to 1% relative to training set.

----

## [1127] Efficient Distribution Similarity Identification in Clustered Federated Learning via Principal Angles between Client Data Subspaces

**Authors**: *Saeed Vahidian, Mahdi Morafah, Weijia Wang, Vyacheslav Kungurtsev, Chen Chen, Mubarak Shah, Bill Lin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26197](https://doi.org/10.1609/aaai.v37i8.26197)

**Abstract**:

Clustered federated learning (FL) has been shown to produce promising results by grouping clients into clusters.
This is especially effective in scenarios where separate groups of clients have significant differences in the distributions of their local data. Existing clustered FL algorithms are essentially trying to group together clients with similar distributions so that clients in the same cluster can leverage each other's data to better perform federated learning. However, prior clustered FL algorithms attempt to learn these distribution similarities indirectly during training, which can be quite time consuming as many rounds of federated learning may be required until the formation of clusters is stabilized. In this paper, we propose a new approach to federated learning that directly aims to efficiently identify distribution similarities among clients by analyzing the principal angles between the client data subspaces. Each client applies a truncated singular value decomposition (SVD) step on its local data in a single-shot manner to derive a small set of principal vectors, which provides a signature that succinctly captures the main characteristics of the underlying distribution.
This small set of principal vectors is provided to the server so that the server can directly identify distribution similarities among the clients to form clusters.
This is achieved by comparing the similarities of the principal angles between the client data subspaces spanned by those principal vectors. The approach provides a simple, yet effective clustered FL framework that addresses a broad range of data heterogeneity issues beyond simpler forms of Non-IIDness like label skews. Our clustered FL approach also enables convergence guarantees for non-convex objectives.

----

## [1128] Training-Time Attacks against K-nearest Neighbors

**Authors**: *Ara Vartanian, Will Rosenbaum, Scott Alfeld*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26198](https://doi.org/10.1609/aaai.v37i8.26198)

**Abstract**:

Nearest neighbor-based methods are commonly used for classification tasks and as subroutines of other data-analysis methods. 
An attacker with the capability of inserting their own data points into the training set can manipulate the inferred nearest neighbor structure.
We distill this goal to the task of performing a training-set data insertion attack against k-Nearest Neighbor classification (kNN).
We prove that computing an optimal training-time (a.k.a. poisoning) attack against kNN classification is NP-Hard, even when k = 1 and the attacker can insert only a single data point.
We provide an anytime algorithm to perform such an attack, and a greedy algorithm for general k and attacker budget.
We provide theoretical bounds and empirically demonstrate the effectiveness and practicality of our methods on synthetic and real-world datasets.
Empirically, we find that kNN is vulnerable in practice and that dimensionality reduction is an effective defense.
We conclude with a discussion of open problems illuminated by our analysis.

----

## [1129] Machines of Finite Depth: Towards a Formalization of Neural Networks

**Authors**: *Pietro Vertechi, Mattia G. Bergomi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26199](https://doi.org/10.1609/aaai.v37i8.26199)

**Abstract**:

We provide a unifying framework where artificial neural networks and their architectures can be formally described as particular cases of a general mathematical construction---machines of finite depth. Unlike neural networks, machines have a precise definition, from which several properties follow naturally. Machines of finite depth are modular (they can be combined), efficiently computable, and differentiable. The backward pass of a machine is again a machine and can be computed without overhead using the same procedure as the forward pass. We prove this statement theoretically and practically via a unified implementation that generalizes several classical architectures---dense, convolutional, and recurrent neural networks with a rich shortcut structure---and their respective backpropagation rules.

----

## [1130] Kalman Bayesian Neural Networks for Closed-Form Online Learning

**Authors**: *Philipp Wagner, Xinyang Wu, Marco F. Huber*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26200](https://doi.org/10.1609/aaai.v37i8.26200)

**Abstract**:

Compared to point estimates calculated by standard neural networks, Bayesian neural networks (BNN) provide probability distributions over the output predictions and model parameters, i.e., the weights. Training the weight distribution of a BNN, however, is more involved due to the intractability of the underlying Bayesian inference problem and thus, requires efficient approximations. In this paper, we propose a novel approach for BNN learning via closed-form Bayesian inference. For this purpose, the calculation of the predictive distribution of the output and the update of the weight distribution are treated as Bayesian filtering and smoothing problems, where the weights are modeled as Gaussian random variables. This allows closed-form expressions for training the network's parameters in a sequential/online fashion without gradient descent. We demonstrate our method on several UCI datasets and compare it to the state of the art.

----

## [1131] Auto-Weighted Multi-View Clustering for Large-Scale Data

**Authors**: *Xinhang Wan, Xinwang Liu, Jiyuan Liu, Siwei Wang, Yi Wen, Weixuan Liang, En Zhu, Zhe Liu, Lu Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26201](https://doi.org/10.1609/aaai.v37i8.26201)

**Abstract**:

Multi-view clustering has gained broad attention owing to its capacity to exploit complementary information across multiple data views. Although existing methods demonstrate delightful clustering performance, most of them are of high time complexity and cannot handle large-scale data. Matrix factorization-based models are a representative of solving this problem. However, they assume that the views share a dimension-fixed consensus coefficient matrix and view-specific base matrices, limiting their representability. Moreover, a series of large-scale algorithms that bear one or more hyperparameters are impractical in real-world applications. To address the two issues, we propose an auto-weighted multi-view clustering (AWMVC) algorithm. Specifically, AWMVC first learns coefficient matrices from corresponding base matrices of different dimensions, then fuses them to obtain an optimal consensus matrix. By mapping original features into distinctive low-dimensional spaces, we can attain more comprehensive knowledge, thus obtaining better clustering results. Moreover, we design a six-step alternative optimization algorithm proven to be convergent theoretically. Also, AWMVC shows excellent performance on various benchmark datasets compared with existing ones. The code of AWMVC is publicly available at https://github.com/wanxinhang/AAAI-2023-AWMVC.

----

## [1132] Quantum Multi-Armed Bandits and Stochastic Linear Bandits Enjoy Logarithmic Regrets

**Authors**: *Zongqi Wan, Zhijie Zhang, Tongyang Li, Jialin Zhang, Xiaoming Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26202](https://doi.org/10.1609/aaai.v37i8.26202)

**Abstract**:

Multi-arm bandit (MAB) and stochastic linear bandit (SLB) are important models in reinforcement learning, and it is well-known that classical algorithms for bandits with time horizon T suffer from the regret of at least the square root of T. In this paper, we study MAB and SLB with quantum reward oracles and propose quantum algorithms for both models with the order of the polylog T regrets, exponentially improving the dependence in terms of T. To the best of our knowledge, this is the first provable quantum speedup for regrets of bandit problems and in general exploitation in reinforcement learning. Compared to previous literature on quantum exploration algorithms for MAB and reinforcement learning, our quantum input model is simpler and only assumes quantum oracles for each individual arm.

----

## [1133] FedABC: Targeting Fair Competition in Personalized Federated Learning

**Authors**: *Dui Wang, Li Shen, Yong Luo, Han Hu, Kehua Su, Yonggang Wen, Dacheng Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26203](https://doi.org/10.1609/aaai.v37i8.26203)

**Abstract**:

Federated learning aims to collaboratively train models without accessing their client's local private data. The data may be Non-IID for different clients and thus resulting in poor performance. Recently, personalized federated learning (PFL) has achieved great success in handling Non-IID data by enforcing regularization in local optimization or improving the model aggregation scheme on the server. However, most of the PFL approaches do not take into account the unfair competition issue caused by the imbalanced data distribution and lack of positive samples for some classes in each client. To address this issue, we propose a novel and generic PFL framework termed Federated Averaging via Binary Classification, dubbed FedABC. In particular, we adopt the ``one-vs-all'' training strategy in each client to alleviate the unfair competition between classes by constructing a personalized binary classification problem for each class. This may aggravate the class imbalance challenge and thus a novel personalized binary classification loss that incorporates both the under-sampling and hard sample mining strategies is designed. Extensive experiments are conducted on two popular datasets under different settings, and the results demonstrate that our FedABC can significantly outperform the existing counterparts.

----

## [1134] Spearman Rank Correlation Screening for Ultrahigh-Dimensional Censored Data

**Authors**: *Hongni Wang, Jingxin Yan, Xiaodong Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26204](https://doi.org/10.1609/aaai.v37i8.26204)

**Abstract**:

Herein, we propose a Spearman rank correlation-based screening procedure for ultrahigh-dimensional data with censored response cases. The proposed method is model-free without specifying any regression forms of predictors or response variables and is robust under the unknown monotone transformations of these response variable and predictors. The sure-screening and rank-consistency properties are established under some mild regularity conditions. Simulation studies demonstrate that the new screening method performs well in the presence
of a heavy-tailed distribution, strongly dependent predictors or outliers, and offers superior performance over the existing nonparametric screening procedures. In particular, the new screening method still works well when a response variable is observed under a high censoring rate. An illustrative example is provided.

----

## [1135] Stability-Based Generalization Analysis for Mixtures of Pointwise and Pairwise Learning

**Authors**: *Jiahuan Wang, Jun Chen, Hong Chen, Bin Gu, Weifu Li, Xin Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26205](https://doi.org/10.1609/aaai.v37i8.26205)

**Abstract**:

Recently, some mixture algorithms of pointwise and pairwise learning (PPL) have been formulated by employing the hybrid error metric of “pointwise loss + pairwise loss” and have shown empirical effectiveness on feature selection, ranking and recommendation tasks. However, to the best of our knowledge, the learning theory foundation of PPL has not been touched in the existing works. In this paper, we try to fill this theoretical gap by investigating the generalization properties of PPL. After extending the definitions of algorithmic stability to the PPL setting, we establish the high-probability generalization bounds for uniformly stable PPL algorithms. Moreover, explicit convergence rates of stochastic gradient descent (SGD) and regularized risk minimization (RRM) for PPL are stated by developing the stability analysis technique of pairwise learning. In addition, the refined generalization bounds of PPL are obtained by replacing uniform stability with on-average stability.

----

## [1136] Effective Continual Learning for Text Classification with Lightweight Snapshots

**Authors**: *Jue Wang, Dajie Dong, Lidan Shou, Ke Chen, Gang Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26206](https://doi.org/10.1609/aaai.v37i8.26206)

**Abstract**:

Continual learning is known for suffering from catastrophic forgetting, a phenomenon where previously learned concepts are forgotten upon learning new tasks. A natural remedy is to use trained models for old tasks as ‘teachers’ to regularize the update of the current model to prevent such forgetting. However, this requires storing all past models, which is very space-consuming for large models, e.g. BERT, thus impractical in real-world applications. To tackle this issue, we propose to construct snapshots of seen tasks whose key knowledge is captured in lightweight adapters. During continual learning, we transfer knowledge from past snapshots to the current model through knowledge distillation, allowing the current model to review previously learned knowledge while learning new tasks. We also design representation recalibration to better handle the class-incremental setting. Experiments over various task sequences show that our approach effectively mitigates catastrophic forgetting and outperforms all baselines.

----

## [1137] Optimistic Whittle Index Policy: Online Learning for Restless Bandits

**Authors**: *Kai Wang, Lily Xu, Aparna Taneja, Milind Tambe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26207](https://doi.org/10.1609/aaai.v37i8.26207)

**Abstract**:

Restless multi-armed bandits (RMABs) extend multi-armed bandits to allow for stateful arms, where the state of each arm evolves restlessly with different transitions depending on whether that arm is pulled. Solving RMABs requires information on transition dynamics, which are often unknown upfront. To plan in RMAB settings with unknown transitions, we propose the first online learning algorithm based on the Whittle index policy, using an upper confidence bound (UCB) approach to learn transition dynamics. Specifically, we estimate confidence bounds of the transition probabilities and formulate a bilinear program to compute optimistic Whittle indices using these estimates. Our algorithm, UCWhittle, achieves sublinear O(H \sqrt{T log T}) frequentist regret to solve RMABs with unknown transitions in T episodes with a constant horizon H. Empirically, we demonstrate that UCWhittle leverages the structure of RMABs and the Whittle index policy solution to achieve better performance than existing online learning baselines across three domains, including one constructed from a real-world maternal and childcare dataset.

----

## [1138] AEC-GAN: Adversarial Error Correction GANs for Auto-Regressive Long Time-Series Generation

**Authors**: *Lei Wang, Liang Zeng, Jian Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26208](https://doi.org/10.1609/aaai.v37i8.26208)

**Abstract**:

Large-scale high-quality data is critical for training modern deep neural networks. However, data acquisition can be costly or time-consuming for many time-series applications, thus researchers turn to generative models for generating synthetic time-series data. In particular, recent generative adversarial networks (GANs) have achieved remarkable success in time-series generation. Despite their success, existing GAN models typically generate the sequences in an auto-regressive manner, and we empirically observe that they suffer from severe distribution shifts and bias amplification, especially when generating long sequences. To resolve this problem, we propose Adversarial Error Correction GAN (AEC-GAN), which is capable of dynamically correcting the bias in the past generated data to alleviate the risk of distribution shifts and thus can generate high-quality long sequences. AEC-GAN contains two main innovations: (1) We develop an error correction module to mitigate the bias. In the training phase, we adversarially perturb the realistic time-series data and then optimize this module to reconstruct the original data. In the generation phase, this module can act as an efficient regulator to detect and mitigate the bias. (2) We propose an augmentation method to facilitate GAN's training by introducing adversarial examples. Thus, AEC-GAN can generate high-quality sequences of arbitrary lengths, and the synthetic data can be readily applied to downstream tasks to boost their performance. We conduct extensive experiments on six widely used datasets and three state-of-the-art time-series forecasting models to evaluate the quality of our synthetic time-series data in different lengths and downstream tasks. Both the qualitative and quantitative experimental results demonstrate the superior performance of AEC-GAN over other deep generative models for time-series generation.

----

## [1139] The Implicit Regularization of Momentum Gradient Descent in Overparametrized Models

**Authors**: *Li Wang, Zhiguo Fu, Yingcong Zhou, Zili Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26209](https://doi.org/10.1609/aaai.v37i8.26209)

**Abstract**:

The study of the implicit regularization induced by gradient-based optimization in deep learning is a long-standing pursuit. In the present paper, we  characterize the implicit regularization of momentum gradient descent (MGD) in the continuous-time view, so-called momentum gradient flow (MGF). We show that the components of weight vector are learned for a deep linear neural networks at different evolution rates, and this evolution gap increases with the depth. Firstly, we show that if the depth equals one, the evolution gap between the weight vector components is linear, which is consistent with the performance of ridge. In particular, we establish a tight coupling between MGF and ridge for the least squares regression. In detail, we show that when the regularization parameter of ridge is inversely proportional to the square of the time parameter of MGF, the risk of MGF is no more than 1.54 times that of  ridge, and their relative Bayesian risks are almost indistinguishable. Secondly, if the model becomes deeper, i.e. the depth is greater than or equal to 2, the evolution gap becomes more significant, which implies an implicit bias towards sparse solutions. The numerical experiments strongly support our theoretical results.

----

## [1140] Meta-Reinforcement Learning Based on Self-Supervised Task Representation Learning

**Authors**: *Mingyang Wang, Zhenshan Bing, Xiangtong Yao, Shuai Wang, Kai Huang, Hang Su, Chenguang Yang, Alois Knoll*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26210](https://doi.org/10.1609/aaai.v37i8.26210)

**Abstract**:

Meta-reinforcement learning enables artificial agents to learn from related training tasks and adapt to new tasks efficiently with minimal interaction data. However, most existing research is still limited to narrow task distributions that are parametric and stationary, and does not consider out-of-distribution tasks during the evaluation, thus, restricting its application. In this paper, we propose MoSS, a context-based Meta-reinforcement learning algorithm based on Self-Supervised task representation learning to address this challenge. We extend meta-RL to broad non-parametric task distributions which have never been explored before, and also achieve state-of-the-art results in non-stationary and out-of-distribution tasks. Specifically, MoSS consists of a task inference module and a policy module. We utilize the Gaussian mixture model for task representation to imitate the parametric and non-parametric task variations. Additionally, our online adaptation strategy enables the agent to react at the first sight of a task change, thus being applicable in non-stationary tasks. MoSS also exhibits strong generalization robustness in out-of-distributions tasks which benefits from the reliable and robust task representation. The policy is built on top of an off-policy RL algorithm and the entire network is trained completely off-policy to ensure high sample efficiency. On MuJoCo and Meta-World benchmarks, MoSS outperforms prior works in terms of asymptotic performance, sample efficiency (3-50x faster), adaptation efficiency, and generalization robustness on broad and diverse task distributions.

----

## [1141] Hierarchical Contrastive Learning for Temporal Point Processes

**Authors**: *Qingmei Wang, Minjie Cheng, Shen Yuan, Hongteng Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26211](https://doi.org/10.1609/aaai.v37i8.26211)

**Abstract**:

As an important sequential model, the temporal point process (TPP) plays a central role in real-world sequence modeling and analysis, whose learning is often based on the maximum likelihood estimation (MLE). However, due to imperfect observations, such as incomplete and sparse sequences that are common in practice, the MLE of TPP models often suffers from overfitting and leads to unsatisfactory generalization power. In this work, we develop a novel hierarchical contrastive (HCL) learning method for temporal point processes, which provides a new regularizer of MLE. In principle, our HCL considers the noise contrastive estimation (NCE) problem at the event-level and at the sequence-level jointly. Given a sequence, the event-level NCE maximizes the probability of each observed event given its history while penalizing the conditional probabilities of the unobserved events. At the same time, we generate positive and negative event sequences from the observed sequence and maximize the discrepancy between their likelihoods through the sequence-level NCE. Instead of using time-consuming simulation methods, we generate the positive and negative sequences via a simple but efficient model-guided thinning process. Experimental results show that the MLE method assisted by the HCL regularizer outperforms classic MLE and other contrastive learning methods in learning various TPP models consistently. The code is available at https://github.com/qingmeiwangdaily/HCL_TPP.

----

## [1142] Beyond ADMM: A Unified Client-Variance-Reduced Adaptive Federated Learning Framework

**Authors**: *Shuai Wang, Yanqing Xu, Zhiguo Wang, Tsung-Hui Chang, Tony Q. S. Quek, Defeng Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26212](https://doi.org/10.1609/aaai.v37i8.26212)

**Abstract**:

As a novel distributed learning paradigm, federated learning (FL) faces serious challenges in dealing with massive clients with heterogeneous data distribution and computation and communication resources. Various client-variance-reduction schemes and client sampling strategies have been respectively introduced to improve the robustness of FL. Among others, primal-dual algorithms such as the alternating direction of method multipliers (ADMM) have been found being resilient to data distribution and outperform most of the primal-only FL algorithms. However, the reason behind remains a mystery still. In this paper, we firstly reveal the fact that the federated ADMM is essentially a client-variance-reduced algorithm. While this explains the inherent robustness of federated ADMM, the vanilla version of it lacks the ability to be adaptive to the degree of client heterogeneity. Besides, the global model at the server under client sampling is biased which slows down the practical convergence. To go beyond ADMM, we propose a novel primal-dual FL algorithm, termed FedVRA, that allows one to adaptively control the variance-reduction level and biasness of the global model. In addition, FedVRA unifies several representative FL algorithms in the sense that they are either special instances of FedVRA or are close to it. Extensions of FedVRA to semi/un-supervised learning are also presented. Experiments based on (semi-)supervised image classification tasks demonstrate superiority of FedVRA over the existing schemes in learning scenarios with massive heterogeneous clients and client sampling.

----

## [1143] State-Conditioned Adversarial Subgoal Generation

**Authors**: *Vivienne Huiling Wang, Joni Pajarinen, Tinghuai Wang, Joni-Kristian Kämäräinen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26213](https://doi.org/10.1609/aaai.v37i8.26213)

**Abstract**:

Hierarchical reinforcement learning (HRL) proposes to solve difficult tasks by performing decision-making and control at successively higher levels of temporal abstraction. However, off-policy HRL often suffers from the problem of a non-stationary high-level policy since the low-level policy is constantly changing. In this paper, we propose a novel HRL approach for mitigating the non-stationarity by adversarially enforcing the high-level policy to generate subgoals compatible with the current instantiation of the low-level policy. In practice, the adversarial learning is implemented by training a simple state conditioned discriminator network concurrently with the high-level policy which determines the compatibility level of subgoals. Comparison to state-of-the-art algorithms shows that our approach improves both learning efficiency and performance in challenging continuous control tasks.

----

## [1144] Deep Attentive Model for Knowledge Tracing

**Authors**: *Xinping Wang, Liangyu Chen, Min Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26214](https://doi.org/10.1609/aaai.v37i8.26214)

**Abstract**:

Knowledge Tracing (KT) is a crucial task in the field of online education, since it aims to predict students' performance on exercises based on their learning history. One typical solution for knowledge tracing is to combine the classic models in educational psychology, such as Item Response Theory (IRT) and Cognitive Diagnosis (CD), with Deep Neural Networks (DNN) technologies. In this solution, a student and related exercises are mapped into feature vectors based on the student's performance at the current time step, however, it does not consider the impact of historical behavior sequences, and the relationships between historical sequences and students. In this paper, we develop DAKTN, a novel model which assimilates the historical sequences to tackle this challenge for better knowledge tracing. To be specific, we apply a pooling layer to incorporate the student behavior sequence in the embedding layer. After that, we further design a local activation unit, which can adaptively calculate the representation vectors by taking the relevance of historical sequences into consideration with respect to candidate student and exercises. Through experimental results on three real-world datasets, DAKTN significantly outperforms state-of-the-art baseline models. We also present the reasonableness of DAKTN by ablation testing.

----

## [1145] Correspondence-Free Domain Alignment for Unsupervised Cross-Domain Image Retrieval

**Authors**: *Xu Wang, Dezhong Peng, Ming Yan, Peng Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26215](https://doi.org/10.1609/aaai.v37i8.26215)

**Abstract**:

Cross-domain image retrieval aims at retrieving images across different domains to excavate cross-domain classificatory or correspondence relationships. This paper studies a less-touched problem of cross-domain image retrieval, i.e., unsupervised cross-domain image retrieval, considering the following practical assumptions: (i) no correspondence relationship, and (ii) no category annotations. It is challenging to align and bridge distinct domains without cross-domain correspondence. To tackle the challenge, we present a novel Correspondence-free Domain Alignment (CoDA) method to effectively eliminate the cross-domain gap through In-domain Self-matching Supervision (ISS) and Cross-domain Classifier Alignment (CCA). To be specific, ISS is presented to encapsulate discriminative information into the latent common space by elaborating a novel self-matching supervision mechanism. To alleviate the cross-domain discrepancy, CCA is proposed to align distinct domain-specific classifiers. Thanks to the ISS and CCA, our method could encode the discrimination into the domain-invariant embedding space for unsupervised cross-domain image retrieval. To verify the effectiveness of the proposed method, extensive experiments are conducted on four benchmark datasets compared with six state-of-the-art methods.

----

## [1146] Isolation and Impartial Aggregation: A Paradigm of Incremental Learning without Interference

**Authors**: *Yabin Wang, Zhiheng Ma, Zhiwu Huang, Yaowei Wang, Zhou Su, Xiaopeng Hong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26216](https://doi.org/10.1609/aaai.v37i8.26216)

**Abstract**:

This paper focuses on the prevalent stage interference and stage performance imbalance of incremental learning. To avoid obvious stage learning bottlenecks, we propose a new incremental learning framework, which leverages a series of stage-isolated classifiers to perform the learning task at each stage, without interference from others. To be concrete, to aggregate multiple stage classifiers as a uniform one impartially, we first introduce a temperature-controlled energy metric for indicating the confidence score levels of the stage classifiers. We then propose an anchor-based energy self-normalization strategy to ensure the stage classifiers work at the same energy level. Finally, we design a voting-based inference augmentation strategy for robust inference. The proposed method is rehearsal-free and can work for almost all incremental learning scenarios. We evaluate the proposed method on four large datasets. Extensive results demonstrate the superiority of the proposed method in setting up new state-of-the-art overall performance. Code is available at https://github.com/iamwangyabin/ESN.

----

## [1147] Robust Self-Supervised Multi-Instance Learning with Structure Awareness

**Authors**: *Yejiang Wang, Yuhai Zhao, Zhengkui Wang, Meixia Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26217](https://doi.org/10.1609/aaai.v37i8.26217)

**Abstract**:

Multi-instance learning (MIL) is a supervised learning where each example is a labeled bag with many instances. The typical MIL strategies are to train an instance-level feature extractor followed by aggregating instances features as bag-level representation with labeled information. However, learning such a bag-level representation highly depends on a large number of labeled datasets, which are difficult to get in real-world scenarios. In this paper, we make the first attempt to propose a robust Self-supervised Multi-Instance LEarning architecture with Structure awareness (SMILEs) that learns unsupervised bag representation. Our proposed approach is: 1) permutation invariant to the order of instances in bag; 2) structure-aware to encode the topological structures among the instances; and 3) robust against instances noise or permutation. Specifically, to yield robust MIL model without label information, we augment the multi-instance bag and train the representation encoder to maximize the correspondence between the representations of the same bag in its different augmented forms. Moreover, to capture topological structures from nearby instances in bags, our framework learns optimal graph structures for the bags and these graphs are optimized together with message passing layers and the ordered weighted averaging operator towards contrastive loss. Our main theorem characterizes the permutation invariance of the bag representation. Compared with state-of-the-art supervised MIL baselines, SMILEs achieves average improvement of 4.9%, 4.4% in classification accuracy on 5 benchmark datasets and 20 newsgroups datasets, respectively. In addition, we show that the model is robust to the input corruption.

----

## [1148] Distributed Projection-Free Online Learning for Smooth and Convex Losses

**Authors**: *Yibo Wang, Yuanyu Wan, Shimao Zhang, Lijun Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26218](https://doi.org/10.1609/aaai.v37i8.26218)

**Abstract**:

We investigate the problem of distributed online convex optimization with complicated constraints, in which the projection operation could be the computational bottleneck. To avoid projections, distributed online projection-free methods have been proposed and attain an O(T^{3/4}) regret bound for general convex losses. However, they cannot utilize the smoothness condition, which has been exploited in the centralized setting to improve the regret. In this paper, we propose a new distributed online projection-free method with a tighter regret bound of O(T^{2/3}) for smooth and convex losses. Specifically, we first provide a distributed extension of Follow-the-Perturbed-Leader so that the smoothness can be utilized in the distributed setting. Then, we reduce the computational cost via sampling and blocking techniques. In this way, our method only needs to solve one linear optimization per round on average. Finally, we conduct experiments on benchmark datasets to verify the effectiveness of our proposed method.

----

## [1149] USER: Unsupervised Structural Entropy-Based Robust Graph Neural Network

**Authors**: *Yifei Wang, Yupan Wang, Zeyu Zhang, Song Yang, Kaiqi Zhao, Jiamou Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26219](https://doi.org/10.1609/aaai.v37i8.26219)

**Abstract**:

Unsupervised/self-supervised graph neural networks (GNN) are susceptible to the inherent randomness in the input graph data, which adversely affects the model's performance in downstream tasks. In this paper, we propose USER, an unsupervised and robust version of GNN based on structural entropy, to alleviate the interference of graph perturbations and learn appropriate representations of nodes without label information. To mitigate the effects of undesirable perturbations, we analyze the property of intrinsic connectivity and define the intrinsic connectivity graph. We also identify the rank of the adjacency matrix as a crucial factor in revealing a graph that provides the same embeddings as the intrinsic connectivity graph. To capture such a graph, we introduce structural entropy in the objective function. Extensive experiments conducted on clustering and link prediction tasks under random-perturbation and meta-attack over three datasets show that USER outperforms benchmarks and is robust to heavier perturbations.

----

## [1150] AutoNF: Automated Architecture Optimization of Normalizing Flows with Unconstrained Continuous Relaxation Admitting Optimal Discrete Solution

**Authors**: *Yu Wang, Ján Drgona, Jiaxin Zhang, Karthik Somayaji Nanjangud Suryanarayana, Malachi Schram, Frank Liu, Peng Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26220](https://doi.org/10.1609/aaai.v37i8.26220)

**Abstract**:

Normalizing flows (NF) build upon invertible neural networks and have wide applications in probabilistic modeling. Currently, building a powerful yet computationally efficient flow model relies on empirical fine-tuning over a large design space. While introducing neural architecture search (NAS) to NF is desirable, the invertibility constraint of NF brings new challenges to existing NAS methods whose application is limited to unstructured neural networks. Developing efficient NAS methods specifically for NF remains an open problem. We present AutoNF, the first automated NF architectural optimization framework. First, we present a new mixture distribution formulation that allows efficient differentiable architecture search of flow models without violating the invertibility constraint. Second, under the new formulation, we convert the original NP-hard combinatorial NF architectural optimization problem to an unconstrained continuous relaxation admitting the discrete optimal architectural solution, circumventing the loss of optimality due to binarization in architectural optimization.   We evaluate AutoNF with various density estimation datasets and show its superior performance-cost trade-offs over a set of existing hand-crafted baselines.

----

## [1151] SEnsor Alignment for Multivariate Time-Series Unsupervised Domain Adaptation

**Authors**: *Yucheng Wang, Yuecong Xu, Jianfei Yang, Zhenghua Chen, Min Wu, Xiaoli Li, Lihua Xie*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26221](https://doi.org/10.1609/aaai.v37i8.26221)

**Abstract**:

Unsupervised Domain Adaptation (UDA) methods can reduce label dependency by mitigating the feature discrepancy between labeled samples in a source domain and unlabeled samples in a similar yet shifted target domain. Though achieving good performance, these methods are inapplicable for Multivariate Time-Series (MTS) data. MTS data are collected from multiple sensors, each of which follows various distributions. However, most UDA methods solely focus on aligning global features but cannot consider the distinct distributions of each sensor. To cope with such concerns, a practical domain adaptation scenario is formulated as Multivariate Time-Series Unsupervised Domain Adaptation (MTS-UDA). In this paper, we propose SEnsor Alignment (SEA) for MTS-UDA to reduce the domain discrepancy at both the local and global sensor levels. At the local sensor level, we design the endo-feature alignment to align sensor features and their correlations across domains, whose information represents the features of each sensor and the interactions between sensors. Further, to reduce domain discrepancy at the global sensor level, we design the exo-feature alignment to enforce restrictions on the global sensor features. Meanwhile, MTS also incorporates the essential spatial-temporal dependencies information between sensors, which cannot be transferred by existing UDA methods. Therefore, we model the spatial-temporal information of MTS with a multi-branch self-attention mechanism for simple and effective transfer across domains. Empirical results demonstrate the state-of-the-art performance of our proposed SEA on two public MTS datasets for MTS-UDA. The code is available at
  https://github.com/Frank-Wang-oss/SEA

----

## [1152] Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning

**Authors**: *Yunke Wang, Bo Du, Chang Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26222](https://doi.org/10.1609/aaai.v37i8.26222)

**Abstract**:

Adversarial imitation learning has become a widely used imitation learning framework. The discriminator is often trained by taking expert demonstrations and policy trajectories as examples respectively from two categories (positive vs. negative) and the policy is then expected to produce trajectories that are indistinguishable from the expert demonstrations. But in the real world, the collected expert demonstrations are more likely to be imperfect, where only an unknown fraction of the demonstrations are optimal. Instead of treating imperfect expert demonstrations as absolutely positive or negative, we investigate unlabeled imperfect expert demonstrations as they are. A positive-unlabeled adversarial imitation learning algorithm is developed to dynamically sample expert demonstrations that can well match the trajectories from the constantly optimized agent policy. The trajectories of an initial agent policy could be closer to those non-optimal expert demonstrations, but within the framework of adversarial imitation learning, agent policy will be optimized to cheat the discriminator and produce trajectories that are similar to those optimal expert demonstrations. Theoretical analysis shows that our method learns from the imperfect demonstrations via a self-paced way. Experimental results on MuJoCo and RoboSuite platforms demonstrate the effectiveness of our method from different aspects.

----

## [1153] FedGS: Federated Graph-Based Sampling with Arbitrary Client Availability

**Authors**: *Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Haibing Jin, Peizhen Yang, Siqi Shen, Cheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26223](https://doi.org/10.1609/aaai.v37i8.26223)

**Abstract**:

While federated learning has shown strong results in opti- mizing a machine learning model without direct access to the original data, its performance may be hindered by in- termittent client availability which slows down the conver- gence and biases the final learned model. There are significant challenges to achieve both stable and bias-free training un- der arbitrary client availability. To address these challenges, we propose a framework named Federated Graph-based Sam- pling (FEDGS), to stabilize the global model update and mitigate the long-term bias given arbitrary client availabil- ity simultaneously. First, we model the data correlations of clients with a Data-Distribution-Dependency Graph (3DG) that helps keep the sampled clients data apart from each other, which is theoretically shown to improve the approximation to the optimal model update. Second, constrained by the far- distance in data distribution of the sampled clients, we fur- ther minimize the variance of the numbers of times that the clients are sampled, to mitigate long-term bias. To validate the effectiveness of FEDGS, we conduct experiments on three datasets under a comprehensive set of seven client availability modes. Our experimental results confirm FEDGS’s advantage in both enabling a fair client-sampling scheme and improving the model performance under arbitrary client availability. Our code is available at https://github.com/WwZzz/FedGS.

----

## [1154] Efficient Exploration in Resource-Restricted Reinforcement Learning

**Authors**: *Zhihai Wang, Taoxing Pan, Qi Zhou, Jie Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26224](https://doi.org/10.1609/aaai.v37i8.26224)

**Abstract**:

In many real-world applications of reinforcement learning (RL), performing actions requires consuming certain types of resources that are non-replenishable in each episode. Typical applications include robotic control with limited energy and video games with consumable items. In tasks with non-replenishable resources, we observe that popular RL methods such as soft actor critic suffer from poor sample efficiency. The major reason is that, they tend to exhaust resources fast and thus the subsequent exploration is severely restricted due to the absence of resources. To address this challenge, we first formalize the aforementioned problem as a resource-restricted reinforcement learning, and then propose a novel resource-aware exploration bonus (RAEB) to make reasonable usage of resources. An appealing feature of RAEB is that, it can significantly reduce unnecessary resource-consuming trials while effectively encouraging the agent to explore unvisited states. Experiments demonstrate that the proposed RAEB significantly outperforms state-of-the-art exploration strategies in resource-restricted reinforcement learning environments, improving the sample efficiency by up to an order of magnitude.

----

## [1155] Efficient Explorative Key-Term Selection Strategies for Conversational Contextual Bandits

**Authors**: *Zhiyong Wang, Xutong Liu, Shuai Li, John C. S. Lui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i8.26225](https://doi.org/10.1609/aaai.v37i8.26225)

**Abstract**:

Conversational contextual bandits elicit user preferences by occasionally querying for explicit feedback on key-terms to accelerate learning.  However, there are aspects of existing approaches which limit their performance. First, information gained from key-term-level conversations and arm-level recommendations is not appropriately incorporated to speed up learning. Second, it is important to ask explorative key-terms to quickly elicit the user's potential interests in various domains to accelerate the convergence of user preference estimation, which has never been considered in existing works.  To tackle these issues, we first propose ``ConLinUCB", a general framework for conversational bandits with better information incorporation, combining arm-level and key-term-level feedback to estimate user preference in one step at each time. Based on this framework, we further design two bandit algorithms with explorative key-term selection strategies, ConLinUCB-BS and ConLinUCB-MCR. We prove tighter regret upper bounds of our proposed algorithms. Particularly, ConLinUCB-BS achieves a better regret bound than the previous result. Extensive experiments on synthetic and real-world data show significant advantages of our algorithms in learning accuracy (up to 54% improvement) and computational efficiency (up to 72% improvement), compared to the classic ConUCB algorithm, showing the potential benefit to recommender systems.

----

## [1156] Code-Aware Cross-Program Transfer Hyperparameter Optimization

**Authors**: *Zijia Wang, Xiangyu He, Kehan Chen, Chen Lin, Jinsong Su*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26226](https://doi.org/10.1609/aaai.v37i9.26226)

**Abstract**:

Hyperparameter tuning is an essential task in automatic machine learning and big data management. 
To accelerate tuning, many recent studies focus on augmenting BO, the primary hyperparameter tuning strategy, by transferring information from other tuning tasks.
However, existing studies ignore program similarities in their transfer mechanism, thus they are sub-optimal in cross-program transfer when tuning tasks involve different programs. 
This paper proposes CaTHPO, a code-aware cross-program transfer hyperparameter optimization framework, which makes three improvements. 
(1) It learns code-aware program representation in a self-supervised manner to give an off-the-shelf estimate of program similarities. 
(2) It adjusts the surrogate and AF in BO based on program similarities, thus the hyperparameter search is guided by accumulated information across similar programs. 
(3) It presents a safe controller to dynamically prune undesirable sample points based on tuning experiences of similar programs. 
Extensive experiments on tuning various recommendation models and Spark applications have demonstrated that CatHPO can steadily obtain better and more robust hyperparameter performances within fewer samples than state-of-the-art competitors.

----

## [1157] Predictive Multiplicity in Probabilistic Classification

**Authors**: *Jamelle Watson-Daniels, David C. Parkes, Berk Ustun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26227](https://doi.org/10.1609/aaai.v37i9.26227)

**Abstract**:

Machine learning models are often used to inform real world risk assessment tasks: predicting consumer default risk, predicting whether a person suffers from a serious illness, or predicting a person's risk to appear in court. Given multiple models that perform almost equally well for a  prediction task, to what extent do predictions vary across these models? If predictions are relatively consistent for similar models, then the standard approach of choosing the model that optimizes a penalized loss suffices. But what if predictions vary significantly for similar models? In machine learning, this is referred to as predictive multiplicity i.e. the prevalence of conflicting predictions assigned by near-optimal competing models. In this paper, we present a framework for measuring predictive multiplicity in probabilistic classification (predicting the probability of a positive outcome). We introduce measures that capture the variation in risk estimates over the set of competing models, and develop optimization-based methods to compute these measures efficiently and reliably for convex empirical risk minimization problems. We demonstrate the incidence and prevalence of predictive multiplicity in real-world tasks. Further, we provide insight into how predictive multiplicity arises by analyzing the relationship between predictive multiplicity and data set characteristics (outliers, separability, and  majority-minority structure). Our results emphasize the need to report predictive multiplicity more widely.

----

## [1158] Feature Distribution Fitting with Direction-Driven Weighting for Few-Shot Images Classification

**Authors**: *Xin Wei, Wei Du, Huan Wan, Weidong Min*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26228](https://doi.org/10.1609/aaai.v37i9.26228)

**Abstract**:

Few-shot learning has received increasing attention and witnessed significant advances in recent years. However, most of the few-shot learning methods focus on the optimization of training process, and the learning of metric and sample generating networks. They ignore the importance of learning the ground-truth feature distributions of few-shot classes. This paper proposes a direction-driven weighting method to make the feature distributions of few-shot classes precisely fit the ground-truth distributions. The learned feature distributions can generate an unlimited number of training samples for the few-shot classes to avoid overfitting. Specifically, the proposed method consists of two optimization strategies. The direction-driven strategy is for capturing more complete direction information that can describe the feature distributions. The similarity-weighting strategy is proposed to estimate the impact of different classes in the fitting procedure and assign corresponding weights. Our method outperforms the current state-of-the-art performance by an average of 3% for 1-shot on standard few-shot learning benchmarks like miniImageNet, CIFAR-FS, and CUB. The excellent performance and compelling visualization show that our method can more accurately estimate the ground-truth distributions.

----

## [1159] Learning Instrumental Variable from Data Fusion for Treatment Effect Estimation

**Authors**: *Anpeng Wu, Kun Kuang, Ruoxuan Xiong, Minqing Zhu, Yuxuan Liu, Bo Li, Furui Liu, Zhihua Wang, Fei Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26229](https://doi.org/10.1609/aaai.v37i9.26229)

**Abstract**:

The advent of the big data era brought new opportunities and challenges to draw treatment effect in data fusion, that is, a mixed dataset collected from multiple sources (each source with an independent treatment assignment mechanism). Due to possibly omitted source labels and unmeasured confounders, traditional methods cannot estimate individual treatment assignment probability and infer treatment effect effectively. Therefore, we propose to reconstruct the source label and model it as a Group Instrumental Variable (GIV) to implement IV-based Regression for treatment effect estimation. In this paper, we conceptualize this line of thought and develop a unified framework (Meta-EM) to (1) map the raw data into a representation space to construct Linear Mixed Models for the assigned treatment variable; (2) estimate the distribution differences and model the GIV for the different treatment assignment mechanisms; and (3) adopt an alternating training strategy to iteratively optimize the representations and the joint distribution to model GIV for IV regression. Empirical results demonstrate the advantages of our Meta-EM compared with state-of-the-art methods. The project page with the code and the Supplementary materials is available at https://github.com/causal-machine-learning-lab/meta-em.

----

## [1160] Towards In-Distribution Compatible Out-of-Distribution Detection

**Authors**: *Boxi Wu, Jie Jiang, Haidong Ren, Zifan Du, Wenxiao Wang, Zhifeng Li, Deng Cai, Xiaofei He, Binbin Lin, Wei Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26230](https://doi.org/10.1609/aaai.v37i9.26230)

**Abstract**:

Deep neural network, despite its remarkable capability of discriminating targeted in-distribution samples, shows poor performance on detecting anomalous out-of-distribution data. To address this defect, state-of-the-art solutions choose to train deep networks on an auxiliary dataset of outliers. Various training criteria for these auxiliary outliers are proposed based on heuristic intuitions. However, we find that these intuitively designed outlier training criteria can hurt in-distribution learning and eventually lead to inferior performance. To this end, we identify three causes of the in-distribution incompatibility: contradictory gradient, false likelihood, and distribution shift. Based on our new understandings, we propose a new out-of-distribution detection method by adapting both the top-design of deep models and the loss function. Our method achieves in-distribution compatibility by pursuing less interference with the probabilistic characteristic of in-distribution features. On several benchmarks, our method not only achieves the state-of-the-art out-of-distribution detection performance but also improves the in-distribution accuracy.

----

## [1161] Non-IID Transfer Learning on Graphs

**Authors**: *Jun Wu, Jingrui He, Elizabeth A. Ainsworth*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26231](https://doi.org/10.1609/aaai.v37i9.26231)

**Abstract**:

Transfer learning refers to the transfer of knowledge or information from a relevant source domain to a target domain. However, most existing transfer learning theories and algorithms focus on IID tasks, where the source/target samples are assumed to be independent and identically distributed. Very little effort is devoted to theoretically studying the knowledge transferability on non-IID tasks, e.g., cross-network mining. To bridge the gap, in this paper, we propose rigorous generalization bounds and algorithms for cross-network transfer learning from a source graph to a target graph. The crucial idea is to characterize the cross-network knowledge transferability from the perspective of the Weisfeiler-Lehman graph isomorphism test. To this end, we propose a novel Graph Subtree Discrepancy to measure the graph distribution shift between source and target graphs. Then the generalization error bounds on cross-network transfer learning, including both cross-network node classification and link prediction tasks, can be derived in terms of the source knowledge and the Graph Subtree Discrepancy across domains. This thereby motivates us to propose a generic graph adaptive network (GRADE) to minimize the distribution shift between source and target graphs for cross-network transfer learning. Experimental results verify the effectiveness and efficiency of our GRADE framework on both cross-network node classification and cross-domain recommendation tasks.

----

## [1162] Extracting Low-/High- Frequency Knowledge from Graph Neural Networks and Injecting It into MLPs: An Effective GNN-to-MLP Distillation Framework

**Authors**: *Lirong Wu, Haitao Lin, Yufei Huang, Tianyu Fan, Stan Z. Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26232](https://doi.org/10.1609/aaai.v37i9.26232)

**Abstract**:

Recent years have witnessed the great success of Graph Neural Networks (GNNs) in handling graph-related tasks. However, MLPs remain the primary workhorse for practical industrial applications due to their desirable inference efficiency and scalability. To reduce their gaps, one can directly distill knowledge from a well-designed teacher GNN to a student MLP, which is termed as GNN-to-MLP distillation. However, the process of distillation usually entails a loss of information, and ``which knowledge patterns of GNNs are more likely to be left and distilled into MLPs?" becomes an important question. In this paper, we first factorize the knowledge learned by GNNs into low- and high-frequency components in the spectral domain and then derive their correspondence in the spatial domain. Furthermore, we identified a potential information drowning problem for existing GNN-to-MLP distillation, i.e., the high-frequency knowledge of the pre-trained GNNs may be overwhelmed by the low-frequency knowledge during distillation; we have described in detail what it represents, how it arises, what impact it has, and how to deal with it. In this paper, we propose an efficient Full-Frequency GNN-to-MLP (FF-G2M) distillation framework, which extracts both low-frequency and high-frequency knowledge from GNNs and injects it into MLPs. Extensive experiments show that FF-G2M improves over the vanilla MLPs by 12.6% and outperforms its corresponding teacher GNNs by 2.6% averaged over six graph datasets and three common GNN architectures.

----

## [1163] Symphony in the Latent Space: Provably Integrating High-Dimensional Techniques with Non-linear Machine Learning Models

**Authors**: *Qiong Wu, Jian Li, Zhenming Liu, Yanhua Li, Mihai Cucuringu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26233](https://doi.org/10.1609/aaai.v37i9.26233)

**Abstract**:

This paper revisits building  machine learning algorithms that involve interactions between entities, such as those between financial assets in an actively managed portfolio, or interactions between users in a social network. Our goal is to forecast the future evolution of ensembles of multivariate time series in such applications (e.g., the future return of a financial asset or the future popularity of a Twitter account). Designing ML algorithms for such systems requires addressing the challenges of high-dimensional interactions and non-linearity. Existing approaches usually adopt an ad-hoc approach to integrating high-dimensional techniques into non-linear models and recent studies have shown these approaches have questionable efficacy in time-evolving interacting systems. 


To this end, we propose a novel framework, which we dub as the additive influence model. Under our modeling assumption, we show that it is possible to decouple the learning of high-dimensional interactions from the learning of non-linear feature interactions. To learn the high-dimensional interactions, we leverage kernel-based techniques, with provable guarantees, to embed the entities in a low-dimensional latent space. To learn the non-linear feature-response interactions, we generalize prominent machine learning techniques, including designing a new statistically sound non-parametric method and an ensemble learning algorithm optimized for vector regressions. 
Extensive experiments on two common applications demonstrate that our new algorithms deliver significantly stronger forecasting power compared to standard and recently proposed methods.

----

## [1164] Decentralized Riemannian Algorithm for Nonconvex Minimax Problems

**Authors**: *Xidong Wu, Zhengmian Hu, Heng Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26234](https://doi.org/10.1609/aaai.v37i9.26234)

**Abstract**:

The minimax optimization over Riemannian manifolds (possibly nonconvex constraints) has been actively applied to solve many problems, such as robust dimensionality reduction and deep neural networks with orthogonal weights (Stiefel manifold). Although many optimization algorithms for minimax problems have been developed in the Euclidean setting, it is difficult to convert them into Riemannian cases, and algorithms for nonconvex minimax problems with nonconvex constraints are even rare. On the other hand, to address the big data challenges, decentralized (serverless) training techniques have recently been emerging since they can reduce communications overhead and avoid the bottleneck problem on the server node. Nonetheless, the algorithm for decentralized Riemannian minimax problems has not been studied. In this paper, we study the distributed nonconvex-strongly-concave minimax optimization problem over the Stiefel manifold and propose both deterministic and stochastic minimax methods. The Steifel manifold is a non-convex set. The global function is represented as the finite sum of local functions. For the deterministic setting, we propose DRGDA and prove that our deterministic method achieves a gradient complexity of O( epsilon(-2)) under mild conditions. For the stochastic setting, we propose DRSGDA and prove that our stochastic method achieves a gradient complexity of O( epsilon(-4)). The DRGDA and DRSGDA are the first algorithms for distributed minimax optimization with nonconvex constraints with exact convergence. Extensive experimental results on the Deep Neural Networks (DNNs) training over the Stiefel manifold demonstrate the efficiency of our algorithms.

----

## [1165] Faster Adaptive Federated Learning

**Authors**: *Xidong Wu, Feihu Huang, Zhengmian Hu, Heng Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26235](https://doi.org/10.1609/aaai.v37i9.26235)

**Abstract**:

Federated learning has attracted increasing attention with the emergence of distributed data. While extensive federated learning algorithms have been proposed for the non-convex distributed problem, the federated learning in practice still faces numerous challenges, such as the large training iterations to converge since the sizes of models and datasets keep increasing, and the lack of adaptivity by SGD-based model updates. Meanwhile, the study of adaptive methods in federated learning is scarce and existing works either lack a complete theoretical convergence guarantee or have slow sample complexity. In this paper, we propose an efficient adaptive algorithm (i.e., FAFED) based on the momentum-based variance reduced technique in cross-silo FL. We first explore how to design the adaptive algorithm in the FL setting. By providing a counter-example, we prove that a simple combination of FL and adaptive methods could lead to divergence. More importantly, we provide a convergence analysis for our method and prove that our algorithm is the first adaptive FL algorithm to reach the best-known samples O(epsilon(-3)) and O(epsilon(-2)) communication rounds to find an epsilon-stationary point without large batches. The experimental results on the language modeling task and image classification task with heterogeneous data demonstrate the efficiency of our algorithms.

----

## [1166] Practical Markov Boundary Learning without Strong Assumptions

**Authors**: *Xingyu Wu, Bingbing Jiang, Tianhao Wu, Huanhuan Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26236](https://doi.org/10.1609/aaai.v37i9.26236)

**Abstract**:

Theoretically, the Markov boundary (MB) is the optimal solution for feature selection. However, existing MB learning algorithms often fail to identify some critical features in real-world feature selection tasks, mainly because the strict assumptions of existing algorithms, on either data distribution, variable types, or correctness of criteria, cannot be satisfied in application scenarios. This paper takes further steps toward opening the door to real-world applications for MB. We contribute in particular to a practical MB learning strategy, which can maintain feasibility and effectiveness in real-world data where variables can be numerical or categorical with linear or nonlinear, pairwise or multivariate relationships. Specifically, the equivalence between MB and the minimal conditional covariance operator (CCO) is investigated, which inspires us to design the objective function based on the predictability evaluation of the mapping variables in a reproducing kernel Hilbert space. Based on this, a kernel MB learning algorithm is proposed, where nonlinear multivariate dependence could be considered without extra requirements on data distribution and variable types. Extensive experiments demonstrate the efficacy of these contributions.

----

## [1167] FedNP: Towards Non-IID Federated Learning via Federated Neural Propagation

**Authors**: *Xueyang Wu, Hengguan Huang, Youlong Ding, Hao Wang, Ye Wang, Qian Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26237](https://doi.org/10.1609/aaai.v37i9.26237)

**Abstract**:

Traditional federated learning (FL) algorithms, such as FedAvg, fail to handle non-i.i.d data because they learn a global model by simply averaging biased local models that are trained on non-i.i.d local data, therefore failing to model the global data distribution. 
In this paper, we present a novel Bayesian FL algorithm that successfully handles such a non-i.i.d FL setting by enhancing the local training task with an auxiliary task that explicitly estimates the global data distribution. 
One key challenge in estimating the global data distribution is that the data are partitioned in FL, and therefore the ground-truth global data distribution is inaccessible.
To address this challenge, we propose an expectation-propagation-inspired probabilistic neural network, dubbed federated neural propagation (FedNP), which efficiently estimates the global data distribution given non-i.i.d data partitions. Our algorithm is sampling-free and end-to-end differentiable, can be applied with any conventional FL frameworks and learns richer global data representation.
Experiments on both image classification tasks with synthetic non-i.i.d image data partitions and real-world non-i.i.d speech recognition tasks demonstrate that our framework effectively alleviates the performance deterioration caused by non-i.i.d data.

----

## [1168] MetaZSCIL: A Meta-Learning Approach for Generalized Zero-Shot Class Incremental Learning

**Authors**: *Yanan Wu, Tengfei Liang, Songhe Feng, Yi Jin, Gengyu Lyu, Haojun Fei, Yang Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26238](https://doi.org/10.1609/aaai.v37i9.26238)

**Abstract**:

Generalized zero-shot learning (GZSL) aims to recognize samples whose categories may not have been seen at training. Standard GZSL cannot handle dynamic addition of new seen and unseen classes. In order to address this limitation, some recent attempts have been made to develop continual GZSL methods. However, these methods require end-users to continuously collect and annotate numerous seen class samples, which is unrealistic and hampers the applicability in the real-world. Accordingly, in this paper, we propose a more practical and challenging setting named Generalized Zero-Shot Class Incremental Learning (CI-GZSL). Our setting aims to incrementally learn unseen classes without any training samples, while recognizing all classes previously encountered. We further propose a bi-level meta-learning based method called MetaZSCIL to directly optimize the network to learn how to incrementally learn. Specifically, we sample sequential tasks from seen classes during the offline training to simulate the incremental learning process. For each task, the model is learned using a meta-objective such that it is capable to perform fast adaptation without forgetting. Note that our optimization can be flexibly equipped with most existing generative methods to tackle CI-GZSL. This work introduces a feature generative framework that leverages visual feature distribution alignment to produce replayed samples of previously seen classes to reduce catastrophic forgetting. Extensive experiments conducted on five widely used benchmarks demonstrate the superiority of our proposed method.

----

## [1169] Adversarial Weight Perturbation Improves Generalization in Graph Neural Networks

**Authors**: *Yihan Wu, Aleksandar Bojchevski, Heng Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26239](https://doi.org/10.1609/aaai.v37i9.26239)

**Abstract**:

A lot of theoretical and empirical evidence shows that the flatter local minima tend to improve generalization. Adversarial Weight Perturbation (AWP) is an emerging technique to efficiently and effectively find such minima. In AMP we minimize the loss w.r.t. a bounded worst-case perturbation of the model parameters thereby favoring local minima with a small loss in a neighborhood around them.
The benefits of AWP, and more generally the connections between flatness and generalization, have been extensively studied for i.i.d. data such as images. In this paper, we extensively study this phenomenon for graph data. Along the way, we first derive a generalization bound for non-i.i.d. node classification tasks. Then we identify a vanishing-gradient issue with all existing formulations of AWP and we propose a new Weighted Truncated AWP (WT-AWP) to alleviate this issue. We show that regularizing graph neural networks with WT-AWP consistently improves both natural and robust generalization across many different graph learning tasks and models.

----

## [1170] Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning

**Authors**: *Young Wu, Jeremy McMahan, Xiaojin Zhu, Qiaomin Xie*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26240](https://doi.org/10.1609/aaai.v37i9.26240)

**Abstract**:

In offline multi-agent reinforcement learning (MARL), agents estimate policies from a given dataset. We study reward-poisoning attacks in this setting where an exogenous attacker modifies the rewards in the dataset before the agents see the dataset. The attacker wants to guide each agent into a nefarious target policy while minimizing the Lp norm of the reward modification. Unlike attacks on single-agent RL, we show that the attacker can install the target policy as a Markov Perfect Dominant Strategy Equilibrium (MPDSE), which rational agents are guaranteed to follow. This attack can be significantly cheaper than separate single-agent attacks. We show that the attack works on various MARL agents including uncertainty-aware learners, and we exhibit linear programs to efficiently solve the attack problem. We also study the relationship between the structure of the datasets and the minimal attack cost. Our work paves the way for studying defense in offline MARL.

----

## [1171] Models as Agents: Optimizing Multi-Step Predictions of Interactive Local Models in Model-Based Multi-Agent Reinforcement Learning

**Authors**: *Zifan Wu, Chao Yu, Chen Chen, Jianye Hao, Hankz Hankui Zhuo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26241](https://doi.org/10.1609/aaai.v37i9.26241)

**Abstract**:

Research in model-based reinforcement learning has made significant progress in recent years. Compared to single-agent settings, the exponential dimension growth of the joint state-action space in multi-agent systems dramatically increases the complexity of the environment dynamics, which makes it infeasible to learn an accurate global model and thus necessitates the use of agent-wise local models. However, during multi-step model rollouts, the prediction of one local model can affect the predictions of other local models in the next step. As a result, local prediction errors can be propagated to other localities and eventually give rise to considerably large global errors. Furthermore, since the models are generally used to predict for multiple steps, simply minimizing one-step prediction errors regardless of their long-term effect on other models may further aggravate the propagation of local errors. To this end, we propose Models as AGents (MAG), a multi-agent model optimization framework that reversely treats the local models as multi-step decision making agents and the current policies as the dynamics during the model rollout process. In this way, the local models are able to consider the multi-step mutual affect between each other before making predictions. Theoretically, we show that the objective of MAG is approximately equivalent to maximizing a lower bound of the true environment return. Experiments on the challenging StarCraft II benchmark demonstrate the effectiveness of MAG.

----

## [1172] Differentially Private Learning with Per-Sample Adaptive Clipping

**Authors**: *Tianyu Xia, Shuheng Shen, Su Yao, Xinyi Fu, Ke Xu, Xiaolong Xu, Xing Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26242](https://doi.org/10.1609/aaai.v37i9.26242)

**Abstract**:

Privacy in AI remains a topic that draws attention from researchers and the general public in recent years. As one way to implement privacy-preserving AI, differentially private learning is a framework that enables AI models to use differential privacy (DP). To achieve DP in the learning process, existing algorithms typically limit the magnitude of gradients with a constant clipping, which requires carefully tuned due to its significant impact on model performance. As a solution to this issue, latest works NSGD and Auto-S innovatively propose to use normalization instead of clipping to avoid hyperparameter tuning. However, normalization-based approaches like NSGD and Auto-S rely on a monotonic weight function, which imposes excessive weight on small gradient samples and introduces extra deviation to the update. In this paper, we propose a Differentially Private Per-Sample Adaptive Clipping (DP-PSAC) algorithm based on a non-monotonic adaptive weight function, which guarantees privacy without the typical hyperparameter tuning process of using a constant clipping while significantly reducing the deviation between the update and true batch-averaged gradient. We provide a rigorous theoretical convergence analysis and show that with convergence rate at the same order, the proposed algorithm achieves a lower non-vanishing bound, which is maintained over training iterations, compared with NSGD/Auto-S.  In addition, through extensive experimental evaluation, we show that DP-PSAC outperforms or matches the state-of-the-art methods on multiple main-stream vision and language tasks.

----

## [1173] Zero-Cost Operation Scoring in Differentiable Architecture Search

**Authors**: *Lichuan Xiang, Lukasz Dudziak, Mohamed S. Abdelfattah, Thomas C. P. Chau, Nicholas D. Lane, Hongkai Wen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26243](https://doi.org/10.1609/aaai.v37i9.26243)

**Abstract**:

We formalize and analyze a fundamental component of dif-
ferentiable neural architecture search (NAS): local “opera-
tion scoring” at each operation choice. We view existing
operation scoring functions as inexact proxies for accuracy,
and we find that they perform poorly when analyzed empir-
ically on NAS benchmarks. From this perspective, we intro-
duce a novel perturbation-based zero-cost operation scor-
ing (Zero-Cost-PT) approach, which utilizes zero-cost prox-
ies that were recently studied in multi-trial NAS but de-
grade significantly on larger search spaces, typical for dif-
ferentiable NAS. We conduct a thorough empirical evalu-
ation on a number of NAS benchmarks and large search
spaces, from NAS-Bench-201, NAS-Bench-1Shot1, NAS-
Bench-Macro, to DARTS-like and MobileNet-like spaces,
showing significant improvements in both search time and
accuracy. On the ImageNet classification task on the DARTS
search space, our approach improved accuracy compared to
the best current training-free methods (TE-NAS) while be-
ing over 10× faster (total searching time 25 minutes on a
single GPU), and observed significantly better transferabil-
ity on architectures searched on the CIFAR-10 dataset with
an accuracy increase of 1.8 pp. Our code is available at:
https://github.com/zerocostptnas/zerocost operation score.

----

## [1174] HALOC: Hardware-Aware Automatic Low-Rank Compression for Compact Neural Networks

**Authors**: *Jinqi Xiao, Chengming Zhang, Yu Gong, Miao Yin, Yang Sui, Lizhi Xiang, Dingwen Tao, Bo Yuan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26244](https://doi.org/10.1609/aaai.v37i9.26244)

**Abstract**:

Low-rank compression is an important model compression strategy for obtaining compact neural network models. In general, because the rank values directly determine the model complexity and model accuracy, proper selection of layer-wise rank is very critical and desired. To date, though many low-rank compression approaches, either selecting the ranks in a manual or automatic way, have been proposed, they suffer from costly manual trials or unsatisfied compression performance. In addition, all of the existing works are not designed in a hardware-aware way, limiting the practical performance of the compressed models on real-world hardware platforms.  

To address these challenges, in this paper we propose HALOC, a hardware-aware automatic low-rank compression framework. By interpreting automatic rank selection from an architecture search perspective, we develop an end-to-end solution to determine the suitable layer-wise ranks in a differentiable and hardware-aware way. We further propose design principles and mitigation strategy to efficiently explore the rank space and reduce the potential interference problem.

Experimental results on different datasets and hardware platforms demonstrate the effectiveness of our proposed approach.  On CIFAR-10 dataset, HALOC enables 0.07% and 0.38% accuracy increase over the uncompressed ResNet-20 and VGG-16 models with 72.20% and 86.44% fewer FLOPs, respectively. On ImageNet dataset, HALOC achieves 0.9% higher top-1 accuracy than the original ResNet-18 model with 66.16% fewer FLOPs. HALOC also shows 0.66% higher top-1 accuracy increase than the state-of-the-art automatic low-rank compression solution with fewer computational and memory costs. In addition, HALOC demonstrates the practical speedups on different hardware platforms, verified by the measurement results on desktop GPU, embedded GPU and ASIC accelerator.

----

## [1175] Bayesian Federated Neural Matching That Completes Full Information

**Authors**: *Peng Xiao, Samuel Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26245](https://doi.org/10.1609/aaai.v37i9.26245)

**Abstract**:

Federated learning is a contemporary machine learning paradigm where locally trained models are distilled into a global model. Due to the intrinsic permutation invariance of neural networks, Probabilistic Federated Neural Matching (PFNM) employs a Bayesian nonparametric framework in the generation process of local neurons, and then creates a linear sum assignment formulation in each alternative optimization iteration. But according to our theoretical analysis, the optimization iteration in PFNM omits global information from existing. In this study, we propose a novel approach that overcomes this flaw by introducing a Kullback-Leibler  divergence penalty at each iteration.
The effectiveness of our approach is demonstrated by experiments on both image classification and semantic segmentation tasks.

----

## [1176] CDMA: A Practical Cross-Device Federated Learning Algorithm for General Minimax Problems

**Authors**: *Jiahao Xie, Chao Zhang, Zebang Shen, Weijie Liu, Hui Qian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26246](https://doi.org/10.1609/aaai.v37i9.26246)

**Abstract**:

Minimax problems arise in a wide range of important applications including robust adversarial learning and Generative Adversarial Network (GAN) training. Recently, algorithms for minimax problems in the Federated Learning (FL) paradigm have received considerable interest. Existing federated algorithms for general minimax problems require the full aggregation (i.e., aggregation of local model information from all clients) in each training round. Thus, they are inapplicable to an important setting of FL known as the cross-device setting, which involves numerous unreliable mobile/IoT devices. In this paper, we develop the first practical algorithm named CDMA for general minimax problems in the cross-device FL setting. CDMA is based on a Start-Immediately-With-Enough-Responses mechanism, in which the server first signals a subset of clients to perform local computation and then starts to aggregate the local results reported by clients once it receives responses from enough clients in each round. With this mechanism, CDMA is resilient to the low client availability. In addition, CDMA is incorporated with a lightweight global correction in the local update steps of clients, which mitigates the impact of slow network connections. We establish theoretical guarantees of CDMA under different choices of hyperparameters and conduct experiments on AUC maximization, robust adversarial network training, and GAN training tasks. Theoretical and experimental results demonstrate the efficiency of CDMA.

----

## [1177] Towards Optimal Randomized Strategies in Adversarial Example Game

**Authors**: *Jiahao Xie, Chao Zhang, Weijie Liu, Wensong Bai, Hui Qian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26247](https://doi.org/10.1609/aaai.v37i9.26247)

**Abstract**:

The vulnerability of deep neural network models to adversarial example attacks is a practical challenge in many artificial intelligence applications. A recent line of work shows that the use of randomization in adversarial training is the key to find optimal strategies against adversarial example attacks. However, in a fully randomized setting where both the defender and the attacker can use randomized strategies, there are no efficient algorithm for finding such an optimal strategy. To fill the gap, we propose the first algorithm of its kind, called FRAT, which models the problem with a new infinite-dimensional continuous-time flow on probability distribution spaces. FRAT maintains a lightweight mixture of models for the defender, with flexibility to efficiently update mixing weights and model parameters at each iteration. Furthermore, FRAT utilizes lightweight sampling subroutines to construct a random strategy for the attacker. We prove that the continuous-time limit of FRAT converges to a mixed Nash equilibria in a zero-sum game formed by a defender and an attacker. Experimental results also demonstrate the efficiency of FRAT on CIFAR-10 and CIFAR-100 datasets.

----

## [1178] A Tale of Two Latent Flows: Learning Latent Space Normalizing Flow with Short-Run Langevin Flow for Approximate Inference

**Authors**: *Jianwen Xie, Yaxuan Zhu, Yifei Xu, Dingcheng Li, Ping Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26248](https://doi.org/10.1609/aaai.v37i9.26248)

**Abstract**:

We study a normalizing flow in the latent space of a top-down generator model, in which the normalizing flow model plays the role of the informative prior model of the generator. We propose to jointly learn the latent space normalizing flow prior model and the top-down generator model by a Markov chain Monte Carlo (MCMC)-based maximum likelihood algorithm, where a short-run Langevin sampling from the intractable posterior distribution is performed to infer the latent variables for each observed example, so that the parameters of the normalizing flow prior and the generator can be updated with the inferred latent variables. We show that, under the scenario of non-convergent short-run MCMC, the finite step Langevin dynamics is a flow-like approximate inference model and the learning objective actually follows the perturbation of the maximum likelihood estimation (MLE). We further point out that the learning framework seeks to (i) match the latent space normalizing flow and the aggregated posterior produced by the short-run Langevin flow, and (ii) bias the model from MLE such that the short-run Langevin flow inference is close to the true posterior. Empirical results of extensive experiments validate the effectiveness of the proposed latent space normalizing flow model in the tasks of image generation, image reconstruction, anomaly detection, supervised image inpainting and unsupervised image recovery.

----

## [1179] Semi-supervised Learning with Support Isolation by Small-Paced Self-Training

**Authors**: *Zheng Xie, Hui Sun, Ming Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26249](https://doi.org/10.1609/aaai.v37i9.26249)

**Abstract**:

In this paper, we address a special scenario of semi-supervised learning, where the label missing is caused by a preceding filtering mechanism, i.e., an instance can enter a subsequent process in which its label is revealed if and only if it passes the filtering mechanism. The rejected instances are prohibited to enter the subsequent labeling process due to economical or ethical reasons, making the support of the labeled and unlabeled distributions isolated from each other. In this case, semi-supervised learning approaches which rely on certain coherence of the labeled and unlabeled distribution would suffer from the consequent distribution mismatch, and hence result in poor prediction performance. In this paper, we propose a Small-Paced Self-Training framework, which iteratively discovers labeled and unlabeled instance subspaces with bounded Wasserstein distance. We theoretically prove that such a framework may achieve provably low error on the pseudo labels during learning. Experiments on both benchmark and pneumonia diagnosis tasks show that our method is effective.

----

## [1180] On the Connection between Invariant Learning and Adversarial Training for Out-of-Distribution Generalization

**Authors**: *Shiji Xin, Yifei Wang, Jingtong Su, Yisen Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26250](https://doi.org/10.1609/aaai.v37i9.26250)

**Abstract**:

Despite impressive success in many tasks, deep learning models are shown to rely on spurious features, which will catastrophically fail when generalized to out-of-distribution (OOD) data. Invariant Risk Minimization (IRM) is proposed to alleviate this issue by extracting domain-invariant features for OOD generalization. Nevertheless, recent work shows that IRM is only effective for a certain type of distribution shift (e.g., correlation shift) while it fails for other cases (e.g., diversity shift). Meanwhile, another thread of method, Adversarial Training (AT), has shown better domain transfer performance, suggesting that it has the potential to be an effective candidate for extracting domain-invariant features. This paper investigates this possibility by exploring the similarity between the IRM and AT objectives. Inspired by this connection, we propose Domain-wise Adversarial Training (DAT), an AT-inspired method for alleviating distribution shift by domain-specific perturbations. Extensive experiments show that our proposed DAT can effectively remove domain-varying features and improve OOD generalization under both correlation shift and diversity shift.

----

## [1181] Decentralized Stochastic Multi-Player Multi-Armed Walking Bandits

**Authors**: *Guojun Xiong, Jian Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26251](https://doi.org/10.1609/aaai.v37i9.26251)

**Abstract**:

Multi-player multi-armed bandit is an increasingly relevant decision-making problem, motivated by applications to cognitive radio systems.  Most research for this problem focuses exclusively on the settings that players have full access to all arms and receive no reward when pulling the same arm.  Hence all players solve the same bandit problem with the goal of maximizing their cumulative reward. However, these settings neglect several important factors in many real-world applications, where players have limited access to a dynamic local subset of arms (i.e., an arm could sometimes be ``walking'' and not accessible to the player).  To this end, this paper proposes a multi-player multi-armed walking bandits model, aiming to address aforementioned modeling issues. The goal now is to maximize the reward, however, players can only pull arms from the local subset and only collect a full reward if no other players pull the same arm.  We adopt Upper Confidence Bound (UCB) to deal with the exploration-exploitation tradeoff and employ distributed optimization techniques to properly handle collisions.  By carefully integrating these two techniques, we propose a decentralized algorithm with near-optimal guarantee on the regret, and can be easily implemented to obtain competitive empirical performance.

----

## [1182] Federated Generative Model on Multi-Source Heterogeneous Data in IoT

**Authors**: *Zuobin Xiong, Wei Li, Zhipeng Cai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26252](https://doi.org/10.1609/aaai.v37i9.26252)

**Abstract**:

The study of generative models is a promising branch of deep learning techniques, which has been successfully applied to different scenarios, such as Artificial Intelligence and the Internet of Things. While in most of the existing works, the generative models are realized as a centralized structure, raising the threats of security and privacy and the overburden of communication costs. Rare efforts have been committed to investigating distributed generative models, especially when the training data comes from multiple heterogeneous sources under realistic IoT settings. In this paper, to handle this challenging problem, we design a federated generative model framework that can learn a powerful generator for the hierarchical IoT systems. Particularly, our generative model framework can solve the problem of distributed data generation on multi-source heterogeneous data in two scenarios, i.e., feature related scenario and label related scenario. In addition, in our federated generative models, we develop a synchronous and an asynchronous updating methods to satisfy different application requirements. Extensive experiments on a simulated dataset and multiple real datasets are conducted to evaluate the data generation performance of our proposed generative models through comparison with the state-of-the-arts.

----

## [1183] Contrastive Open Set Recognition

**Authors**: *Baile Xu, Furao Shen, Jian Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26253](https://doi.org/10.1609/aaai.v37i9.26253)

**Abstract**:

In conventional recognition tasks, models are only trained to recognize learned targets, but it is usually difficult to collect training examples of all potential categories. In the testing phase, when models receive test samples from unknown classes, they mistakenly classify the samples into known classes. Open set recognition (OSR) is a more realistic recognition task, which requires the classifier to detect unknown test samples while keeping a high classification accuracy of known classes. In this paper, we study how to improve the OSR performance of deep neural networks from the perspective of representation learning. We employ supervised contrastive learning to improve the quality of feature representations, propose a new supervised contrastive learning method that enables the model to learn from soft training targets, and design an OSR framework on its basis. With the proposed method, we are able to make use of label smoothing and mixup when training deep neural networks contrastively, so as to improve both the robustness of outlier detection in OSR tasks and the accuracy in conventional classification tasks. We validate our method on multiple benchmark datasets and testing scenarios, achieving experimental results that verify the effectiveness of the proposed method.

----

## [1184] Progressive Deep Multi-View Comprehensive Representation Learning

**Authors**: *Cai Xu, Wei Zhao, Jinglong Zhao, Ziyu Guan, Yaming Yang, Long Chen, Xiangyu Song*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26254](https://doi.org/10.1609/aaai.v37i9.26254)

**Abstract**:

Multi-view Comprehensive Representation Learning (MCRL) aims to synthesize information from multiple views to learn comprehensive representations of data items. Prevalent deep MCRL methods typically concatenate synergistic view-specific representations or average aligned view-specific representations in the fusion stage. However, the performance of synergistic fusion methods inevitably degenerate or even fail when partial views are missing in real-world applications; the aligned based fusion methods usually cannot fully exploit the complementarity of multi-view data. To eliminate all these drawbacks, in this work we present a Progressive Deep Multi-view Fusion (PDMF) method. Considering the multi-view comprehensive representation should contain complete information and the view-specific data contain partial information, we deem that it is unstable to directly learn the mapping from partial information to complete information. Hence, PDMF employs a progressive learning strategy, which contains the pre-training and fine-tuning stages. In the pre-training stage, PDMF decodes the auxiliary comprehensive representation to the view-specific data. It also captures the consistency and complementarity by learning the relations between the dimensions of the auxiliary comprehensive representation and all views. In the fine-tuning stage, PDMF learns the mapping from the original data to the comprehensive representation with the help of the auxiliary comprehensive representation and relations. Experiments conducted on a synthetic toy dataset and 4 real-world datasets show that PDMF outperforms state-of-the-art baseline methods. The code is released at https://github.com/winterant/PDMF.

----

## [1185] A Survey on Model Compression and Acceleration for Pretrained Language Models

**Authors**: *Canwen Xu, Julian J. McAuley*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26255](https://doi.org/10.1609/aaai.v37i9.26255)

**Abstract**:

Despite achieving state-of-the-art performance on many NLP tasks, the high energy cost and long inference delay prevent Transformer-based pretrained language models (PLMs) from seeing broader adoption including for edge and mobile computing. Efficient NLP research aims to comprehensively consider computation, time and carbon emission for the entire life-cycle of NLP, including data preparation, model training and inference. In this survey, we focus on the inference stage and review the current state of model compression and acceleration for pretrained language models, including benchmarks, metrics and methodology.

----

## [1186] GraphPrompt: Graph-Based Prompt Templates for Biomedical Synonym Prediction

**Authors**: *Hanwen Xu, Jiayou Zhang, Zhirui Wang, Shizhuo Zhang, Megh Bhalerao, Yucong Liu, Dawei Zhu, Sheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26256](https://doi.org/10.1609/aaai.v37i9.26256)

**Abstract**:

In the expansion of biomedical dataset, the same category may be labeled with different terms, thus being tedious and onerous to curate these terms. Therefore, automatically mapping synonymous terms onto the ontologies is desirable, which we name as biomedical synonym prediction task. Unlike biomedical concept normalization (BCN), no clues from context can be used to enhance synonym prediction, making it essential to extract graph features from ontology. We introduce an expert-curated dataset OBO-syn encompassing 70 different types of concepts and 2 million curated concept-term pairs for evaluating synonym prediction methods. We find BCN methods perform weakly on this task for not making full use of graph information. Therefore, we propose GraphPrompt, a prompt-based learning approach that creates prompt templates according to the graphs. GraphPrompt obtained 37.2% and 28.5% improvement on zero-shot and few-shot settings respectively, indicating the effectiveness of these graph-based prompt templates. We envision that our method GraphPrompt and OBO-syn dataset can be broadly applied to graph-based NLP tasks, and serve as the basis for analyzing diverse and accumulating biomedical data. All the data and codes are avalible at: https://github.com/HanwenXuTHU/GraphPrompt

----

## [1187] Open-Ended Diverse Solution Discovery with Regulated Behavior Patterns for Cross-Domain Adaptation

**Authors**: *Kang Xu, Yan Ma, Bingsheng Wei, Wei Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26257](https://doi.org/10.1609/aaai.v37i9.26257)

**Abstract**:

While Reinforcement Learning can achieve impressive results for complex tasks, the learned policies are generally prone to fail in downstream tasks with even minor model mismatch or unexpected perturbations. Recent works have demonstrated that a policy population with diverse behavior characteristics can generalize to downstream environments with various discrepancies. However, such policies might result in catastrophic damage during the deployment in practical scenarios like real-world systems due to the unrestricted behaviors of trained policies. Furthermore, training diverse policies without regulation of the behavior can result in inadequate feasible policies for extrapolating to a wide range of test conditions with dynamics shifts. In this work, we aim to train diverse policies under the regularization of the behavior patterns. We motivate our paradigm by observing the inverse dynamics in the environment with partial state information and propose Diversity in Regulation (DiR) training diverse policies with regulated behaviors to discover desired patterns that benefit the generalization. Considerable empirical results on various variations of different environments indicate that our method attains improvements over other diversity-driven counterparts.

----

## [1188] Efficient Top-K Feature Selection Using Coordinate Descent Method

**Authors**: *Lei Xu, Rong Wang, Feiping Nie, Xuelong Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26258](https://doi.org/10.1609/aaai.v37i9.26258)

**Abstract**:

Sparse learning based feature selection has been widely investigated in recent years. In this study, we focus on the l2,0-norm based feature selection, which is effective for exact top-k feature selection but challenging to optimize. To solve the general l2,0-norm constrained problems, we novelly develop a parameter-free optimization framework based on the coordinate descend (CD) method, termed CD-LSR. Specifically, we devise a skillful conversion from the original problem to solving one continuous matrix and one discrete selection matrix. Then the nontrivial l2,0-norm constraint can be solved efficiently by solving the selection matrix with CD method. We impose the l2,0-norm on a vanilla least square regression (LSR) model for feature selection and optimize it with CD-LSR. Extensive experiments exhibit the efficiency of CD-LSR, as well as the discrimination ability of l2,0-norm to identify informative features. More importantly, the versatility of CD-LSR facilitates the applications of the l2,0-norm in more sophisticated models. Based on the competitive performance of l2,0-norm on the baseline LSR model, the satisfactory performance of its applications is reasonably expected. The source MATLAB code are available at: https://github.com/solerxl/Code_For_AAAI_2023.

----

## [1189] Label-Specific Feature Augmentation for Long-Tailed Multi-Label Text Classification

**Authors**: *Pengyu Xu, Lin Xiao, Bing Liu, Sijin Lu, Liping Jing, Jian Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26259](https://doi.org/10.1609/aaai.v37i9.26259)

**Abstract**:

Multi-label text classification (MLTC) involves tagging a document with its most relevant subset of labels from a label set. In real applications, labels usually follow a long-tailed distribution, where most labels (called as tail-label) only contain a small number of documents and limit the performance of MLTC. To facilitate this low-resource problem, researchers introduced a simple but effective strategy, data augmentation (DA). However, most existing DA approaches struggle in multi-label settings. The main reason is that the augmented documents for one label may inevitably influence the other co-occurring labels and further exaggerate the long-tailed problem. To mitigate this issue, we propose a new pair-level augmentation framework for MLTC, called Label-Specific Feature Augmentation (LSFA), which merely augments positive feature-label pairs for the tail-labels. LSFA contains two main parts. The first is for label-specific document representation learning in the high-level latent space, the second is for augmenting tail-label features in latent space by transferring the documents second-order statistics (intra-class semantic variations) from head labels to tail labels. At last, we design a new loss function for adjusting classifiers based on augmented datasets. The whole learning procedure can be effectively trained. Comprehensive experiments on benchmark datasets have shown that the proposed LSFA outperforms the state-of-the-art counterparts.

----

## [1190] Neighborhood-Regularized Self-Training for Learning with Few Labels

**Authors**: *Ran Xu, Yue Yu, Hejie Cui, Xuan Kan, Yanqiao Zhu, Joyce C. Ho, Chao Zhang, Carl Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26260](https://doi.org/10.1609/aaai.v37i9.26260)

**Abstract**:

Training deep neural networks (DNNs) with limited supervision has been a popular research topic as it can significantly alleviate the annotation burden. Self-training has been successfully applied in semi-supervised learning tasks, but one drawback of self-training is that it is vulnerable to the label noise from incorrect pseudo labels. Inspired by the fact that samples with similar labels tend to share similar representations, we develop a neighborhood-based sample selection approach to tackle the issue of noisy pseudo labels. We further stabilize self-training via aggregating the predictions from different rounds during sample selection. Experiments on eight tasks show that our proposed method outperforms the strongest self-training baseline with 1.83% and 2.51% performance gain for text and graph datasets on average. Our further analysis demonstrates that our proposed data selection strategy reduces the noise of pseudo labels by 36.8% and saves 57.3% of the time when compared with the best baseline. Our code and appendices will be uploaded to: https://github.com/ritaranx/NeST.

----

## [1191] Resilient Binary Neural Network

**Authors**: *Sheng Xu, Yanjing Li, Teli Ma, Mingbao Lin, Hao Dong, Baochang Zhang, Peng Gao, Jinhu Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26261](https://doi.org/10.1609/aaai.v37i9.26261)

**Abstract**:

Binary neural networks (BNNs) have received ever-increasing popularity for their great capability of reducing storage burden as well as quickening inference time. However, there is a severe performance drop compared with {real-valued} networks, due to its intrinsic frequent weight oscillation during training. In this paper, we introduce a Resilient Binary Neural Network (ReBNN) to mitigate the frequent oscillation for better BNNs' training. We identify that the weight oscillation mainly stems from the non-parametric scaling factor. To address this issue, we propose to parameterize the scaling factor and introduce a weighted reconstruction loss to build an adaptive training objective. For the first time, we show that the weight oscillation  is  controlled by the balanced parameter attached to the reconstruction loss, which provides a theoretical foundation to  parameterize it in back propagation. Based on this, we learn our ReBNN by calculating the balanced parameter based on its maximum magnitude, which can  effectively mitigate the weight oscillation with a resilient training process. Extensive experiments are conducted  upon various network models, such as ResNet and Faster-RCNN for computer vision, as well as BERT for natural language processing. The results demonstrate the overwhelming performance of our ReBNN over prior arts. For example, our ReBNN achieves 66.9% Top-1 accuracy with ResNet-18 backbone on the ImageNet dataset, surpassing existing state-of-the-arts by a significant margin. Our code is open-sourced at https://github.com/SteveTsui/ReBNN.

----

## [1192] Transfer Learning Enhanced DeepONet for Long-Time Prediction of Evolution Equations

**Authors**: *Wuzhe Xu, Yulong Lu, Li Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26262](https://doi.org/10.1609/aaai.v37i9.26262)

**Abstract**:

Deep operator network (DeepONet) has demonstrated great
success in various learning tasks, including learning solution
operators of partial differential equations. In particular, it provides
 an efficient approach to predicting the evolution equations
in a finite time horizon. Nevertheless, the vanilla DeepONet
suffers from the issue of stability degradation in the long-
time prediction. This paper proposes a transfer-learning aided
DeepONet to enhance the stability. Our idea is to use transfer
learning to sequentially update the DeepONets as the surro-
gates for propagators learned in different time frames. The
evolving DeepONets can better track the varying complexities
of the evolution equations, while only need to be updated by
efficient training of a tiny fraction of the operator networks.
Through systematic experiments, we show that the proposed
method not only improves the long-time accuracy of Deep-
ONet while maintaining similar computational cost but also
substantially reduces the sample size of the training set.

----

## [1193] BridgeTower: Building Bridges between Encoders in Vision-Language Representation Learning

**Authors**: *Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26263](https://doi.org/10.1609/aaai.v37i9.26263)

**Abstract**:

Vision-Language (VL) models with the Two-Tower architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. In this paper, we propose BridgeTower, which introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the cross-modal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BridgeTower achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BridgeTower achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BridgeTower achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets. Code and checkpoints are available at https://github.com/microsoft/BridgeTower.

----

## [1194] USDNL: Uncertainty-Based Single Dropout in Noisy Label Learning

**Authors**: *Yuanzhuo Xu, Xiaoguang Niu, Jie Yang, Steve Drew, Jiayu Zhou, Ruizhi Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26264](https://doi.org/10.1609/aaai.v37i9.26264)

**Abstract**:

Deep Neural Networks (DNNs) possess powerful prediction capability thanks to their over-parameterization design, although the large model complexity makes it suffer from noisy supervision. Recent approaches seek to eliminate impacts from noisy labels by excluding data points with large loss values and showing promising performance. However, these approaches usually associate with significant computation overhead and lack of theoretical analysis. In this paper, we adopt a perspective to connect label noise with epistemic uncertainty. We design a simple, efficient, and theoretically provable robust algorithm named USDNL for DNNs with uncertainty-based Dropout. Specifically, we estimate the epistemic uncertainty of the network prediction after early training through single Dropout. The epistemic uncertainty is then combined with cross-entropy loss to select the clean samples during training. Finally, we theoretically show the equivalence of replacing selection loss with single cross-entropy loss. Compared to existing small-loss selection methods, USDNL features its simplicity for practical scenarios by only applying Dropout to a standard network, while still achieving high model accuracy. Extensive empirical results on both synthetic and real-world datasets show that USDNL outperforms other methods. Our code is available at https://github.com/kovelxyz/USDNL.

----

## [1195] Trusted Fine-Grained Image Classification through Hierarchical Evidence Fusion

**Authors**: *Zhikang Xu, Xiaodong Yue, Ying Lv, Wei Liu, Zihao Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26265](https://doi.org/10.1609/aaai.v37i9.26265)

**Abstract**:

Fine-Grained Image Classification (FGIC) aims to classify images into specific subordinate classes of a superclass. Due to insufficient training data and confusing data samples, FGIC may produce uncertain classification results that are untrusted for data applications. In fact, FGIC can be viewed as a hierarchical classification process and the multilayer information facilitates to reduce  uncertainty and improve the reliability of FGIC. In this paper, we adopt the evidence theory to measure uncertainty and confidence in hierarchical classification process and propose a trusted FGIC method through fusing multilayer classification evidence. Comparing with the traditional approaches, the trusted FGIC method not only generates accurate classification results but also reduces the uncertainty of fine-grained classification. Specifically, we construct an evidence extractor at each classification layer to extract multilayer (multi-grained) evidence for image classification. To fuse the extracted multi-grained evidence from coarse to fine, we formulate  evidence fusion with the Dirichlet hyper probability distribution and thereby hierarchically decompose the evidence of coarse-grained classes into fine-grained classes to enhance the classification performances. The ablation experiments validate that the hierarchical evidence fusion can improve the precision and also reduce the uncertainty of fine-grained classification. The comparison with state-of-the-art FGIC methods shows that our proposed method achieves competitive performances.

----

## [1196] Disentangled Representation for Causal Mediation Analysis

**Authors**: *Ziqi Xu, Debo Cheng, Jiuyong Li, Jixue Liu, Lin Liu, Ke Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26266](https://doi.org/10.1609/aaai.v37i9.26266)

**Abstract**:

Estimating direct and indirect causal effects from observational data is crucial to understanding the causal mechanisms and predicting the behaviour under different interventions. Causal mediation analysis is a method that is often used to reveal direct and indirect effects. Deep learning shows promise in mediation analysis, but the current methods only assume latent confounders that affect treatment, mediator and outcome simultaneously, and fail to identify different types of latent confounders (e.g., confounders that only affect the mediator or outcome). Furthermore, current methods are based on the sequential ignorability assumption, which is not feasible for dealing with multiple types of latent confounders. This work aims to circumvent the sequential ignorability assumption and applies the piecemeal deconfounding assumption as an alternative. We propose the Disentangled Mediation Analysis Variational AutoEncoder (DMAVAE), which disentangles the representations of latent confounders into three types to accurately estimate the natural direct effect, natural indirect effect and total effect. Experimental results show that the proposed method outperforms existing methods and has strong generalisation ability. We further apply the method to a real-world dataset to show its potential application.

----

## [1197] Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis

**Authors**: *Han Xuanyuan, Pietro Barbiero, Dobrik Georgiev, Lucie Charlotte Magister, Pietro Liò*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26267](https://doi.org/10.1609/aaai.v37i9.26267)

**Abstract**:

Graph neural networks (GNNs) are highly effective on a variety of graph-related tasks; however, they lack interpretability and transparency. Current explainability approaches are typically local and treat GNNs as black-boxes. They do not look inside the model, inhibiting human trust in the model and explanations. Motivated by the ability of neurons to detect high-level semantic concepts in vision models, we perform a novel analysis on the behaviour of individual GNN neurons to answer questions about GNN interpretability. We propose a novel approach for producing global explanations for GNNs using neuron-level concepts to enable practitioners to have a high-level view of the model. Specifically, (i) to the best of our knowledge, this is the first work which shows that GNN neurons act as concept detectors and have strong alignment with concepts formulated as logical compositions of node degree and neighbourhood properties; (ii) we quantitatively assess the importance of detected concepts, and identify a trade-off between training duration and neuron-level interpretability; (iii) we demonstrate that our global explainability approach has advantages over the current state-of-the-art -- we can disentangle the explanation into individual interpretable concepts backed by logical descriptions, which reduces potential for bias and improves user-friendliness.

----

## [1198] Fast and Accurate Binary Neural Networks Based on Depth-Width Reshaping

**Authors**: *Ping Xue, Yang Lu, Jingfei Chang, Xing Wei, Zhen Wei*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26268](https://doi.org/10.1609/aaai.v37i9.26268)

**Abstract**:

Network binarization (i.e., binary neural networks, BNNs) can efficiently compress deep neural networks and accelerate model inference but cause severe accuracy degradation. Existing BNNs are mainly implemented based on the commonly used full-precision network backbones, and then the accuracy is improved with various techniques. However, there is a question of whether the full-precision network backbone is well adapted to BNNs. We start from the factors of the performance degradation of BNNs and analyze the problems of directly using full-precision network backbones for BNNs: for a given computational budget, the backbone of a BNN may need to be shallower and wider compared to the backbone of a full-precision network. With this in mind, Depth-Width Reshaping (DWR) is proposed to reshape the depth and width of existing full-precision network backbones and further optimize them by incorporating pruning techniques to better fit the BNNs. Extensive experiments demonstrate the analytical result and the effectiveness of the proposed method. Compared with the original backbones, the DWR backbones constructed by the proposed method result in close to O(√s) decrease in activations, while achieving an absolute accuracy increase by up to 1.7% with comparable computational cost. Besides, by using the DWR backbones, existing methods can achieve new state-of-the-art (SOTA) accuracy (e.g., 67.2% on ImageNet with ResNet-18 as the original backbone). We hope this work provides a novel insight into the backbone design of BNNs. The code is available at https://github.com/pingxue-hfut/DWR.

----

## [1199] Learning the Finer Things: Bayesian Structure Learning at the Instantiation Level

**Authors**: *Chase Yakaboski, Eugene Santos Jr.*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26269](https://doi.org/10.1609/aaai.v37i9.26269)

**Abstract**:

Successful machine learning methods require a trade-off between memorization and generalization. Too much memorization and the model cannot generalize to unobserved examples. Too much over-generalization and we risk under-fitting the data. While we commonly measure their performance through cross validation and accuracy metrics, how should these algorithms cope in domains that are extremely under-determined where accuracy is always unsatisfactory? We present a novel probabilistic graphical model structure learning approach that can learn, generalize and explain in these elusive domains by operating at the random variable instantiation level. Using Minimum Description Length (MDL) analysis, we propose a new decomposition of the learning problem over all training exemplars, fusing together minimal entropy inferences to construct a final knowledge base. By leveraging Bayesian Knowledge Bases (BKBs), a framework that operates at the instantiation level and inherently subsumes Bayesian Networks (BNs), we develop both a theoretical MDL score and associated structure learning algorithm that demonstrates significant improvements over learned BNs on 40 benchmark datasets. Further, our algorithm incorporates recent off-the-shelf DAG learning techniques enabling tractable results even on large problems. We then demonstrate the utility of our approach in a significantly under-determined domain by learning gene regulatory networks on breast cancer gene mutational data available from The Cancer Genome Atlas (TCGA).

----



[Go to the previous page](AAAI-2023-list05.md)

[Go to the next page](AAAI-2023-list07.md)

[Go to the catalog section](README.md)