## [1800] Phase-aware Adversarial Defense for Improving Adversarial Robustness

        **Authors**: *Dawei Zhou, Nannan Wang, Heng Yang, Xinbo Gao, Tongliang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23m.html](https://proceedings.mlr.press/v202/zhou23m.html)

        **Abstract**:

        Deep neural networks have been found to be vulnerable to adversarial noise. Recent works show that exploring the impact of adversarial noise on intrinsic components of data can help improve adversarial robustness. However, the pattern closely related to human perception has not been deeply studied. In this paper, inspired by the cognitive science, we investigate the interference of adversarial noise from the perspective of image phase, and find ordinarily-trained models lack enough robustness against phase-level perturbations. Motivated by this, we propose a joint adversarial defense method: a phase-level adversarial training mechanism to enhance the adversarial robustness on the phase pattern; an amplitude-based pre-processing operation to mitigate the adversarial perturbation in the amplitude pattern. Experimental results show that the proposed method can significantly improve the robust accuracy against multiple attacks and even adaptive attacks. In addition, ablation studies demonstrate the effectiveness of our defense strategy.

        ----

        ## [1801] From Relational Pooling to Subgraph GNNs: A Universal Framework for More Expressive Graph Neural Networks

        **Authors**: *Cai Zhou, Xiyuan Wang, Muhan Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23n.html](https://proceedings.mlr.press/v202/zhou23n.html)

        **Abstract**:

        Relational pooling is a framework for building more expressive and permutation-invariant graph neural networks. However, there is limited understanding of the exact enhancement in the expressivity of RP and its connection with the Weisfeiler-Lehman hierarchy. Starting from RP, we propose to explicitly assign labels to nodes as additional features to improve graph isomorphism distinguishing power of message passing neural networks. The method is then extended to higher-dimensional WL, leading to a novel $k,l$-WL algorithm, a more general framework than $k$-WL. We further introduce the subgraph concept into our hierarchy and propose a localized $k,l$-WL framework, incorporating a wide range of existing work, including many subgraph GNNs. Theoretically, we analyze the expressivity of $k,l$-WL w.r.t. $k$ and $l$ and compare it with the traditional $k$-WL. Complexity reduction methods are also systematically discussed to build powerful and practical $k,l$-GNN instances. We theoretically and experimentally prove that our method is universally compatible and capable of improving the expressivity of any base GNN model. Our $k,l$-GNNs achieve superior performance on many synthetic and real-world datasets, which verifies the effectiveness of our framework.

        ----

        ## [1802] Towards Omni-generalizable Neural Methods for Vehicle Routing Problems

        **Authors**: *Jianan Zhou, Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23o.html](https://proceedings.mlr.press/v202/zhou23o.html)

        **Abstract**:

        Learning heuristics for vehicle routing problems (VRPs) has gained much attention due to the less reliance on hand-crafted rules. However, existing methods are typically trained and tested on the same task with a fixed size and distribution (of nodes), and hence suffer from limited generalization performance. This paper studies a challenging yet realistic setting, which considers generalization across both size and distribution in VRPs. We propose a generic meta-learning framework, which enables effective training of an initialized model with the capability of fast adaptation to new tasks during inference. We further develop a simple yet efficient approximation method to reduce the training overhead. Extensive experiments on both synthetic and benchmark instances of the traveling salesman problem (TSP) and capacitated vehicle routing problem (CVRP) demonstrate the effectiveness of our method. The code is available at: https://github.com/RoyalSkye/Omni-VRP.

        ----

        ## [1803] A Three-regime Model of Network Pruning

        **Authors**: *Yefan Zhou, Yaoqing Yang, Arin Chang, Michael W. Mahoney*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23p.html](https://proceedings.mlr.press/v202/zhou23p.html)

        **Abstract**:

        Recent work has highlighted the complex influence training hyperparameters, e.g., the number of training epochs, can have on the prunability of machine learning models. Perhaps surprisingly, a systematic approach to predict precisely how adjusting a specific hyperparameter will affect prunability remains elusive. To address this gap, we introduce a phenomenological model grounded in the statistical mechanics of learning. Our approach uses temperature-like and load-like parameters to model the impact of neural network (NN) training hyperparameters on pruning performance. A key empirical result we identify is a sharp transition phenomenon: depending on the value of a load-like parameter in the pruned model, increasing the value of a temperature-like parameter in the pre-pruned model may either enhance or impair subsequent pruning performance. Based on this transition, we build a three-regime model by taxonomizing the global structure of the pruned NN loss landscape. Our model reveals that the dichotomous effect of high temperature is associated with transitions between distinct types of global structures in the post-pruned model. Based on our results, we present three case-studies: 1) determining whether to increase or decrease a hyperparameter for improved pruning; 2) selecting the best model to prune from a family of models; and 3) tuning the hyperparameter of the Sharpness Aware Minimization method for better pruning performance.

        ----

        ## [1804] Learning to Decouple Complex Systems

        **Authors**: *Zihan Zhou, Tianshu Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23q.html](https://proceedings.mlr.press/v202/zhou23q.html)

        **Abstract**:

        A complex system with cluttered observations may be a coupled mixture of multiple simple sub-systems corresponding to latent entities. Such sub-systems may hold distinct dynamics in the continuous-time domain; therein, complicated interactions between sub-systems also evolve over time. This setting is fairly common in the real world but has been less considered. In this paper, we propose a sequential learning approach under this setting by decoupling a complex system for handling irregularly sampled and cluttered sequential observations. Such decoupling brings about not only subsystems describing the dynamics of each latent entity but also a meta-system capturing the interaction between entities over time. Specifically, we argue that the meta-system evolving within a simplex is governed by projected differential equations (ProjDEs). We further analyze and provide neural-friendly projection operators in the context of Bregman divergence. Experimental results on synthetic and real-world datasets show the advantages of our approach when facing complex and cluttered sequential data compared to the state-of-the-art.

        ----

        ## [1805] ESC: Exploration with Soft Commonsense Constraints for Zero-shot Object Navigation

        **Authors**: *Kaiwen Zhou, Kaizhi Zheng, Connor Pryor, Yilin Shen, Hongxia Jin, Lise Getoor, Xin Eric Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23r.html](https://proceedings.mlr.press/v202/zhou23r.html)

        **Abstract**:

        The ability to accurately locate and navigate to a specific object is a crucial capability for embodied agents that operate in the real world and interact with objects to complete tasks. Such object navigation tasks usually require large-scale training in visual environments with labeled objects, which generalizes poorly to novel objects in unknown environments. In this work, we present a novel zero-shot object navigation method, Exploration with Soft Commonsense constraints (ESC), that transfers commonsense knowledge in pre-trained models to open-world object navigation without any navigation experience nor any other training on the visual environments. First, ESC leverages a pre-trained vision and language model for open-world prompt-based grounding and a pre-trained commonsense language model for room and object reasoning. Then ESC converts commonsense knowledge into navigation actions by modeling it as soft logic predicates for efficient exploration. Extensive experiments on MP3D, HM3D, and RoboTHOR benchmarks show that our ESC method improves significantly over baselines, and achieves new state-of-the-art results for zero-shot object navigation (e.g., 288% relative Success Rate improvement than CoW on MP3D).

        ----

        ## [1806] On Strengthening and Defending Graph Reconstruction Attack with Markov Chain Approximation

        **Authors**: *Zhanke Zhou, Chenyu Zhou, Xuan Li, Jiangchao Yao, Quanming Yao, Bo Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23s.html](https://proceedings.mlr.press/v202/zhou23s.html)

        **Abstract**:

        Although powerful graph neural networks (GNNs) have boosted numerous real-world applications, the potential privacy risk is still underexplored. To close this gap, we perform the first comprehensive study of graph reconstruction attack that aims to reconstruct the adjacency of nodes. We show that a range of factors in GNNs can lead to the surprising leakage of private links. Especially by taking GNNs as a Markov chain and attacking GNNs via a flexible chain approximation, we systematically explore the underneath principles of graph reconstruction attack, and propose two information theory-guided mechanisms: (1) the chain-based attack method with adaptive designs for extracting more private information; (2) the chain-based defense method that sharply reduces the attack fidelity with moderate accuracy loss. Such two objectives disclose a critical belief that to recover better in attack, you must extract more multi-aspect knowledge from the trained GNN; while to learn safer for defense, you must forget more link-sensitive information in training GNNs. Empirically, we achieve state-of-the-art results on six datasets and three common GNNs. The code is publicly available at: https://github.com/tmlr-group/MC-GRA.

        ----

        ## [1807] Sharp Variance-Dependent Bounds in Reinforcement Learning: Best of Both Worlds in Stochastic and Deterministic Environments

        **Authors**: *Runlong Zhou, Zihan Zhang, Simon Shaolei Du*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23t.html](https://proceedings.mlr.press/v202/zhou23t.html)

        **Abstract**:

        We study variance-dependent regret bounds for Markov decision processes (MDPs). Algorithms with variance-dependent regret guarantees can automatically exploit environments with low variance (e.g., enjoying constant regret on deterministic MDPs). The existing algorithms are either variance-independent or suboptimal. We first propose two new environment norms to characterize the fine-grained variance properties of the environment. For model-based methods, we design a variant of the MVP algorithm (Zhang et al., 2021a). We apply new analysis techniques to demonstrate that this algorithm enjoys variance-dependent bounds with respect to the norms we propose. In particular, this bound is simultaneously minimax optimal for both stochastic and deterministic MDPs, the first result of its kind. We further initiate the study on model-free algorithms with variance-dependent regret bounds by designing a reference-function-based algorithm with a novel capped-doubling reference update schedule. Lastly, we also provide lower bounds to complement our upper bounds.

        ----

        ## [1808] Learning Unforeseen Robustness from Out-of-distribution Data Using Equivariant Domain Translator

        **Authors**: *Sicheng Zhu, Bang An, Furong Huang, Sanghyun Hong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23a.html](https://proceedings.mlr.press/v202/zhu23a.html)

        **Abstract**:

        Current approaches for training robust models are typically tailored to scenarios where data variations are accessible in the training set. While shown effective in achieving robustness to these foreseen variations, these approaches are ineffective in learning unforeseen robustness, i.e., robustness to data variations without known characterization or training examples reflecting them. In this work, we learn unforeseen robustness by harnessing the variations in the abundant out-of-distribution data. To overcome the main challenge of using such data, the domain gap, we use a domain translator to bridge it and bound the unforeseen robustness on the target distribution. As implied by our analysis, we propose a two-step algorithm that first trains an equivariant domain translator to map out-of-distribution data to the target distribution while preserving the considered variation, and then regularizes a model’s output consistency on the domain-translated data to improve its robustness. We empirically show the effectiveness of our approach in improving unforeseen and foreseen robustness compared to existing approaches. Additionally, we show that training the equivariant domain translator serves as an effective criterion for source data selection.

        ----

        ## [1809] Markovian Gaussian Process Variational Autoencoders

        **Authors**: *Harrison Zhu, Carles Balsells Rodas, Yingzhen Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23b.html](https://proceedings.mlr.press/v202/zhu23b.html)

        **Abstract**:

        Sequential VAEs have been successfully considered for many high-dimensional time series modelling problems, with many variant models relying on discrete-time mechanisms such as recurrent neural networks (RNNs). On the other hand, continuous-time methods have recently gained attraction, especially in the context of irregularly-sampled time series, where they can better handle the data than discrete-time methods. One such class are Gaussian process variational autoencoders (GPVAEs), where the VAE prior is set as a Gaussian process (GP). However, a major limitation of GPVAEs is that it inherits the cubic computational cost as GPs, making it unattractive to practioners. In this work, we leverage the equivalent discrete state space representation of Markovian GPs to enable linear time GPVAE training via Kalman filtering and smoothing. For our model, Markovian GPVAE (MGPVAE), we show on a variety of high-dimensional temporal and spatiotemporal tasks that our method performs favourably compared to existing approaches whilst being computationally highly scalable.

        ----

        ## [1810] Mixture Proportion Estimation Beyond Irreducibility

        **Authors**: *Yilun Zhu, Aaron Fjeldsted, Darren Holland, George Landon, Azaree Lintereur, Clayton Scott*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23c.html](https://proceedings.mlr.press/v202/zhu23c.html)

        **Abstract**:

        The task of mixture proportion estimation (MPE) is to estimate the weight of a component distribution in a mixture, given observations from both the component and mixture. Previous work on MPE adopts the irreducibility assumption, which ensures identifiablity of the mixture proportion. In this paper, we propose a more general sufficient condition that accommodates several settings of interest where irreducibility does not hold. We further present a resampling-based meta-algorithm that takes any existing MPE algorithm designed to work under irreducibility and adapts it to work under our more general condition. Our approach empirically exhibits improved estimation performance relative to baseline methods and to a recently proposed regrouping-based algorithm.

        ----

        ## [1811] Exploring Model Dynamics for Accumulative Poisoning Discovery

        **Authors**: *Jianing Zhu, Xiawei Guo, Jiangchao Yao, Chao Du, Li He, Shuo Yuan, Tongliang Liu, Liang Wang, Bo Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23d.html](https://proceedings.mlr.press/v202/zhu23d.html)

        **Abstract**:

        Adversarial poisoning attacks pose huge threats to various machine learning applications. Especially, the recent accumulative poisoning attacks show that it is possible to achieve irreparable harm on models via a sequence of imperceptible attacks followed by a trigger batch. Due to the limited data-level discrepancy in real-time data streaming, current defensive methods are indiscriminate in handling the poison and clean samples. In this paper, we dive into the perspective of model dynamics and propose a novel information measure, namely, Memorization Discrepancy, to explore the defense via the model-level information. By implicitly transferring the changes in the data manipulation to that in the model outputs, Memorization Discrepancy can discover the imperceptible poison samples based on their distinct dynamics from the clean samples. We thoroughly explore its properties and propose Discrepancy-aware Sample Correction (DSC) to defend against accumulative poisoning attacks. Extensive experiments comprehensively characterized Memorization Discrepancy and verified its effectiveness. The code is publicly available at: https://github.com/tmlr-group/Memorization-Discrepancy.

        ----

        ## [1812] Decentralized SGD and Average-direction SAM are Asymptotically Equivalent

        **Authors**: *Tongtian Zhu, Fengxiang He, Kaixuan Chen, Mingli Song, Dacheng Tao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23e.html](https://proceedings.mlr.press/v202/zhu23e.html)

        **Abstract**:

        Decentralized stochastic gradient descent (D-SGD) allows collaborative learning on massive devices simultaneously without the control of a central server. However, existing theories claim that decentralization invariably undermines generalization. In this paper, we challenge the conventional belief and present a completely new perspective for understanding decentralized learning. We prove that D-SGD implicitly minimizes the loss function of an average-direction Sharpness-aware minimization (SAM) algorithm under general non-convex non-$\beta$-smooth settings. This surprising asymptotic equivalence reveals an intrinsic regularization-optimization trade-off and three advantages of decentralization: (1) there exists a free uncertainty evaluation mechanism in D-SGD to improve posterior estimation; (2) D-SGD exhibits a gradient smoothing effect; and (3) the sharpness regularization effect of D-SGD does not decrease as total batch size increases, which justifies the potential generalization benefit of D-SGD over centralized SGD (C-SGD) in large-batch scenarios.

        ----

        ## [1813] Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons

        **Authors**: *Banghua Zhu, Michael I. Jordan, Jiantao Jiao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23f.html](https://proceedings.mlr.press/v202/zhu23f.html)

        **Abstract**:

        We provide a theoretical framework for Reinforcement Learning with Human Feedback (RLHF). We show that when the underlying true reward is linear, under both Bradley-Terry-Luce (BTL) model (pairwise comparison) and Plackett-Luce (PL) model ($K$-wise comparison), MLE converges under certain semi-norm for the family of linear reward. On the other hand, when training a policy based on the learned reward model, we show that MLE fails while a pessimistic MLE provides policies with good performance under certain coverage assumption. We also show that under the PL model, both the true MLE and a different MLE which splits the $K$-wise comparison into pairwise comparisons converge, while the true MLE is asymptotically more efficient. Our results validate the empirical success of the existing RLHF algorithms, and provide new insights for algorithm design. Our analysis can also be applied for the problem of online RLHF and inverse reinforcement learning.

        ----

        ## [1814] Unleashing Mask: Explore the Intrinsic Out-of-Distribution Detection Capability

        **Authors**: *Jianing Zhu, Hengzhuang Li, Jiangchao Yao, Tongliang Liu, Jianliang Xu, Bo Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23g.html](https://proceedings.mlr.press/v202/zhu23g.html)

        **Abstract**:

        Out-of-distribution (OOD) detection is an indispensable aspect of secure AI when deploying machine learning models in real-world applications. Previous paradigms either explore better scoring functions or utilize the knowledge of outliers to equip the models with the ability of OOD detection. However, few of them pay attention to the intrinsic OOD detection capability of the given model. In this work, we generally discover the existence of an intermediate stage of a model trained on in-distribution (ID) data having higher OOD detection performance than that of its final stage across different settings, and further identify one critical data-level attribution to be learning with the atypical samples. Based on such insights, we propose a novel method, Unleashing Mask, which aims to restore the OOD discriminative capabilities of the well-trained model with ID data. Our method utilizes a mask to figure out the memorized atypical samples, and then finetune the model or prune it with the introduced mask to forget them. Extensive experiments and analysis demonstrate the effectiveness of our method. The code is available at: https://github.com/tmlr-group/Unleashing-Mask.

        ----

        ## [1815] Benign Overfitting in Deep Neural Networks under Lazy Training

        **Authors**: *Zhenyu Zhu, Fanghui Liu, Grigorios Chrysos, Francesco Locatello, Volkan Cevher*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23h.html](https://proceedings.mlr.press/v202/zhu23h.html)

        **Abstract**:

        This paper focuses on over-parameterized deep neural networks (DNNs) with ReLU activation functions and proves that when the data distribution is well-separated, DNNs can achieve Bayes-optimal test error for classification while obtaining (nearly) zero-training error under the lazy training regime. For this purpose, we unify three interrelated concepts of overparameterization, benign overfitting, and the Lipschitz constant of DNNs. Our results indicate that interpolating with smoother functions leads to better generalization. Furthermore, we investigate the special case where interpolating smooth ground-truth functions is performed by DNNs under the Neural Tangent Kernel (NTK) regime for generalization. Our result demonstrates that the generalization error converges to a constant order that only depends on label noise and initialization noise, which theoretically verifies benign overfitting. Our analysis provides a tight lower bound on the normalized margin under non-smooth activation functions, as well as the minimum eigenvalue of NTK under high-dimensional settings, which has its own interest in learning theory.

        ----

        ## [1816] Interpolation for Robust Learning: Data Augmentation on Wasserstein Geodesics

        **Authors**: *Jiacheng Zhu, Jielin Qiu, Aritra Guha, Zhuolin Yang, XuanLong Nguyen, Bo Li, Ding Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23i.html](https://proceedings.mlr.press/v202/zhu23i.html)

        **Abstract**:

        We propose to study and promote the robustness of a model as per its performance on a continuous geodesic interpolation of subpopulations, e.g., a class of samples in a classification problem. Specifically, (1) we augment the data by finding the worst-case Wasserstein barycenter on the geodesic connecting subpopulation distributions. (2) we regularize the model for smoother performance on the continuous geodesic path connecting subpopulation distributions. (3) Additionally, we provide a theoretical guarantee of robustness improvement and investigate how the geodesic location and the sample size contribute, respectively. Experimental validations of the proposed strategy on four datasets including CIFAR-100 and ImageNet, establish the efficacy of our method, e.g., our method improves the baselines’ certifiable robustness on CIFAR10 upto 7.7%, with 16.8% on empirical robustness on CIFAR-100. Our work provides a new perspective of model robustness through the lens of Wasserstein geodesic-based interpolation with a practical off-the-shelf strategy that can be combined with existing robust training methods.

        ----

        ## [1817] LeadFL: Client Self-Defense against Model Poisoning in Federated Learning

        **Authors**: *Chaoyi Zhu, Stefanie Roos, Lydia Y. Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23j.html](https://proceedings.mlr.press/v202/zhu23j.html)

        **Abstract**:

        Federated Learning is highly susceptible to backdoor and targeted attacks as participants can manipulate their data and models locally without any oversight on whether they follow the correct process. There are a number of server-side defenses that mitigate the attacks by modifying or rejecting local updates submitted by clients. However, we find that bursty adversarial patterns with a high variance in the number of malicious clients can circumvent the existing defenses. We propose a client-self defense, LeadFL, that is combined with existing server-side defenses to thwart backdoor and targeted attacks. The core idea of LeadFL is a novel regularization term in local model training such that the Hessian matrix of local gradients is nullified. We provide the convergence analysis of LeadFL and its robustness guarantee in terms of certified radius. Our empirical evaluation shows that LeadFL is able to mitigate bursty adversarial patterns for both iid and non-iid data distributions. It frequently reduces the backdoor accuracy from more than 75% for state-of-the-art defenses to less than 10% while its impact on the main task accuracy is always less than for other client-side defenses.

        ----

        ## [1818] XTab: Cross-table Pretraining for Tabular Transformers

        **Authors**: *Bingzhao Zhu, Xingjian Shi, Nick Erickson, Mu Li, George Karypis, Mahsa Shoaran*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23k.html](https://proceedings.mlr.press/v202/zhu23k.html)

        **Abstract**:

        The success of self-supervised learning in computer vision and natural language processing has motivated pretraining methods on tabular data. However, most existing tabular self-supervised learning models fail to leverage information across multiple data tables and cannot generalize to new tables. In this work, we introduce XTab, a framework for cross-table pretraining of tabular transformers on datasets from various domains. We address the challenge of inconsistent column types and quantities among tables by utilizing independent featurizers and using federated learning to pretrain the shared component. Tested on 84 tabular prediction tasks from the OpenML-AutoML Benchmark (AMLB), we show that (1) XTab consistently boosts the generalizability, learning speed, and performance of multiple tabular transformers, (2) by pretraining FT-Transformer via XTab, we achieve superior performance than other state-of-the-art tabular deep learning models on various tasks such as regression, binary, and multiclass classification.

        ----

        ## [1819] Provable Multi-instance Deep AUC Maximization with Stochastic Pooling

        **Authors**: *Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23l.html](https://proceedings.mlr.press/v202/zhu23l.html)

        **Abstract**:

        This paper considers a novel application of deep AUC maximization (DAM) for multi-instance learning (MIL), in which a single class label is assigned to a bag of instances (e.g., multiple 2D slices of a CT scan for a patient). We address a neglected yet non-negligible computational challenge of MIL in the context of DAM, i.e., bag size is too large to be loaded into GPU memory for backpropagation, which is required by the standard pooling methods of MIL. To tackle this challenge, we propose variance-reduced stochastic pooling methods in the spirit of stochastic optimization by formulating the loss function over the pooled prediction as a multi-level compositional function. By synthesizing techniques from stochastic compositional optimization and non-convex min-max optimization, we propose a unified and provable muli-instance DAM (MIDAM) algorithm with stochastic smoothed-max pooling or stochastic attention-based pooling, which only samples a few instances for each bag to compute a stochastic gradient estimator and to update the model parameter. We establish a similar convergence rate of the proposed MIDAM algorithm as the state-of-the-art DAM algorithms. Our extensive experiments on conventional MIL datasets and medical datasets demonstrate the superiority of our MIDAM algorithm. The method is open-sourced at https://libauc.org/.

        ----

        ## [1820] Surrogate Model Extension (SME): A Fast and Accurate Weight Update Attack on Federated Learning

        **Authors**: *Junyi Zhu, Ruicong Yao, Matthew B. Blaschko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23m.html](https://proceedings.mlr.press/v202/zhu23m.html)

        **Abstract**:

        In Federated Learning (FL) and many other distributed training frameworks, collaborators can hold their private data locally and only share the network weights trained with the local data after multiple iterations. Gradient inversion is a family of privacy attacks that recovers data from its generated gradients. Seemingly, FL can provide a degree of protection against gradient inversion attacks on weight updates, since the gradient of a single step is concealed by the accumulation of gradients over multiple local iterations. In this work, we propose a principled way to extend gradient inversion attacks to weight updates in FL, thereby better exposing weaknesses in the presumed privacy protection inherent in FL. In particular, we propose a surrogate model method based on the characteristic of two-dimensional gradient flow and low-rank property of local updates. Our method largely boosts the ability of gradient inversion attacks on weight updates containing many iterations and achieves state-of-the-art (SOTA) performance. Additionally, our method runs up to $100\times$ faster than the SOTA baseline in the common FL scenario. Our work re-evaluates and highlights the privacy risk of sharing network weights. Our code is available at https://github.com/JunyiZhu-AI/surrogate_model_extension.

        ----

        ## [1821] Weak Proxies are Sufficient and Preferable for Fairness with Missing Sensitive Attributes

        **Authors**: *Zhaowei Zhu, Yuanshun Yao, Jiankai Sun, Hang Li, Yang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23n.html](https://proceedings.mlr.press/v202/zhu23n.html)

        **Abstract**:

        Evaluating fairness can be challenging in practice because the sensitive attributes of data are often inaccessible due to privacy constraints. The go-to approach that the industry frequently adopts is using off-the-shelf proxy models to predict the missing sensitive attributes, e.g. Meta (Alao et al., 2021) and Twitter (Belli et al., 2022). Despite its popularity, there are three important questions unanswered: (1) Is directly using proxies efficacious in measuring fairness? (2) If not, is it possible to accurately evaluate fairness using proxies only? (3) Given the ethical controversy over infer-ring user private information, is it possible to only use weak (i.e. inaccurate) proxies in order to protect privacy? Our theoretical analyses show that directly using proxy models can give a false sense of (un)fairness. Second, we develop an algorithm that is able to measure fairness (provably) accurately with only three properly identified proxies. Third, we show that our algorithm allows the use of only weak proxies (e.g. with only 68.85% accuracy on COMPAS), adding an extra layer of protection on user privacy. Experiments validate our theoretical analyses and show our algorithm can effectively measure and mitigate bias. Our results imply a set of practical guidelines for prac-titioners on how to use proxies properly. Code is available at https://github.com/UCSC-REAL/fair-eval.

        ----

        ## [1822] Label Distributionally Robust Losses for Multi-class Classification: Consistency, Robustness and Adaptivity

        **Authors**: *Dixian Zhu, Yiming Ying, Tianbao Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhu23o.html](https://proceedings.mlr.press/v202/zhu23o.html)

        **Abstract**:

        We study a family of loss functions named label-distributionally robust (LDR) losses for multi-class classification that are formulated from distributionally robust optimization (DRO) perspective, where the uncertainty in the given label information are modeled and captured by taking the worse case of distributional weights. The benefits of this perspective are several fold: (i) it provides a unified framework to explain the classical cross-entropy (CE) loss and SVM loss and their variants, (ii) it includes a special family corresponding to the temperature-scaled CE loss, which is widely adopted but poorly understood; (iii) it allows us to achieve adaptivity to the uncertainty degree of label information at an instance level. Our contributions include: (1) we study both consistency and robustness by establishing top-$k$ ($\forall k\geq 1$) consistency of LDR losses for multi-class classification, and a negative result that a top-$1$ consistent and symmetric robust loss cannot achieve top-$k$ consistency simultaneously for all $k\geq 2$; (2) we propose a new adaptive LDR loss that automatically adapts the individualized temperature parameter to the noise degree of class label of each instance; (3) we demonstrate stable and competitive performance for the proposed adaptive LDR loss on 7 benchmark datasets under 6 noisy label and 1 clean settings against 13 loss functions, and on one real-world noisy dataset. The method is open-sourced at https://github.com/Optimization-AI/ICML2023_LDR.

        ----

        ## [1823] Likelihood Adjusted Semidefinite Programs for Clustering Heterogeneous Data

        **Authors**: *Yubo Zhuang, Xiaohui Chen, Yun Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhuang23a.html](https://proceedings.mlr.press/v202/zhuang23a.html)

        **Abstract**:

        Clustering is a widely deployed unsupervised learning tool. Model-based clustering is a flexible framework to tackle data heterogeneity when the clusters have different shapes. Likelihood-based inference for mixture distributions often involves non-convex and high-dimensional objective functions, imposing difficult computational and statistical challenges. The classic expectation-maximization (EM) algorithm is a computationally thrifty iterative method that maximizes a surrogate function minorizing the log-likelihood of observed data in each iteration, which however suffers from bad local maxima even in the special case of the standard Gaussian mixture model with common isotropic covariance matrices. On the other hand, recent studies reveal that the unique global solution of a semidefinite programming (SDP) relaxed $K$-means achieves the information-theoretically sharp threshold for perfectly recovering the cluster labels under the standard Gaussian mixture model. In this paper, we extend the SDP approach to a general setting by integrating cluster labels as model parameters and propose an iterative likelihood adjusted SDP (iLA-SDP) method that directly maximizes the exact observed likelihood in the presence of data heterogeneity. By lifting the cluster assignment to group-specific membership matrices, iLA-SDP avoids centroids estimation – a key feature that allows exact recovery under well-separateness of centroids without being trapped by their adversarial configurations. Thus iLA-SDP is less sensitive than EM to initialization and more stable on high-dimensional data. Our numeric experiments demonstrate that iLA-SDP can achieve lower mis-clustering errors over several widely used clustering methods including $K$-means, SDP and EM algorithms.

        ----

        ## [1824] Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?

        **Authors**: *Juliusz Krysztof Ziomek, Haitham Bou-Ammar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ziomek23a.html](https://proceedings.mlr.press/v202/ziomek23a.html)

        **Abstract**:

        Learning decompositions of expensive-to-evaluate black-box functions promises to scale Bayesian optimisation (BO) to high-dimensional problems. However, the success of these techniques depends on finding proper decompositions that accurately represent the black-box. While previous works learn those decompositions based on data, we investigate data-independent decomposition sampling rules in this paper. We find that data-driven learners of decompositions can be easily misled towards local decompositions that do not hold globally across the search space. Then, we formally show that a random tree-based decomposition sampler exhibits favourable theoretical guarantees that effectively trade off maximal information gain and functional mismatch between the actual black-box and its surrogate as provided by the decomposition. Those results motivate the development of the random decomposition upper-confidence bound algorithm (RDUCB) that is straightforward to implement - (almost) plug-and-play - and, surprisingly, yields significant empirical gains compared to the previous state-of-the-art on a comprehensive set of benchmarks. We also confirm the plug-and-play nature of our modelling component by integrating our method with HEBO, showing improved practical gains in the highest dimensional tasks from Bayesmark problem suite.

        ----

        ## [1825] Revisiting Bellman Errors for Offline Model Selection

        **Authors**: *Joshua P. Zitovsky, Daniel de Marchi, Rishabh Agarwal, Michael Rene Kosorok*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zitovsky23a.html](https://proceedings.mlr.press/v202/zitovsky23a.html)

        **Abstract**:

        Offline model selection (OMS), that is, choosing the best policy from a set of many policies given only logged data, is crucial for applying offline RL in real-world settings. One idea that has been extensively explored is to select policies based on the mean squared Bellman error (MSBE) of the associated Q-functions. However, previous work has struggled to obtain adequate OMS performance with Bellman errors, leading many researchers to abandon the idea. To this end, we elucidate why previous work has seen pessimistic results with Bellman errors and identify conditions under which OMS algorithms based on Bellman errors will perform well. Moreover, we develop a new estimator of the MSBE that is more accurate than prior methods. Our estimator obtains impressive OMS performance on diverse discrete control tasks, including Atari games.

        ----

        ## [1826] spred: Solving L1 Penalty with SGD

        **Authors**: *Ziyin Liu, Zihao Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ziyin23a.html](https://proceedings.mlr.press/v202/ziyin23a.html)

        **Abstract**:

        We propose to minimize a generic differentiable objective with $L_1$ constraint using a simple reparametrization and straightforward stochastic gradient descent. Our proposal is the direct generalization of previous ideas that the $L_1$ penalty may be equivalent to a differentiable reparametrization with weight decay. We prove that the proposed method, spred, is an exact differentiable solver of $L_1$ and that the reparametrization trick is completely “benign" for a generic nonconvex function. Practically, we demonstrate the usefulness of the method in (1) training sparse neural networks to perform gene selection tasks, which involves finding relevant features in a very high dimensional space, and (2) neural network compression task, to which previous attempts at applying the $L_1$-penalty have been unsuccessful. Conceptually, our result bridges the gap between the sparsity in deep learning and conventional statistical learning.

        ----

        ## [1827] The Benefits of Mixup for Feature Learning

        **Authors**: *Difan Zou, Yuan Cao, Yuanzhi Li, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zou23a.html](https://proceedings.mlr.press/v202/zou23a.html)

        **Abstract**:

        Mixup, a simple data augmentation method that randomly mixes two data points via linear interpolation, has been extensively applied in various deep learning applications to gain better generalization. However, its theoretical explanation remains largely unclear. In this work, we aim to seek a fundamental understanding of the benefits of Mixup. We first show that Mixup using different linear interpolation parameters for features and labels can still achieve similar performance as standard Mixup. This indicates that the intuitive linearity explanation in Zhang et al., (2018) may not fully explain the success of Mixup. Then, we perform a theoretical study of Mixup from the feature learning perspective. We consider a feature-noise data model and show that Mixup training can effectively learn the rare features (appearing in a small fraction of data) from its mixture with the common features (appearing in a large fraction of data). In contrast, standard training can only learn the common features but fails to learn the rare features, thus suffering from bad generalization performance. Moreover, our theoretical analysis also shows that the benefits of Mixup for feature learning are mostly gained in the early training phase, based on which we propose to apply early stopping in Mixup. Experimental results verify our theoretical findings and demonstrate the effectiveness of the early-stopped Mixup training.

        ----

        

[Go to the previous page](ICML-2023-list09.md)

[Go to the catalog section](README.md)