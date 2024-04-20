## [0] Harnessing Hard Mixed Samples with Decoupled Regularizer

**Authors**: *Zicheng Liu, Siyuan Li, Ge Wang, Lirong Wu, Cheng Tan, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5c47c1b7adf19e8dc633812a4acf6d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5c47c1b7adf19e8dc633812a4acf6d2-Abstract-Conference.html)

**Abstract**:

Mixup is an efficient data augmentation approach that improves the generalization of neural networks by smoothing the decision boundary with mixed data. Recently, dynamic mixup methods have improved previous \textit{static} policies effectively (e.g., linear interpolation) by maximizing target-related salient regions in mixed samples, but excessive additional time costs are not acceptable. These additional computational overheads mainly come from optimizing the mixed samples according to the mixed labels. However, we found that the extra optimizing step may be redundant because label-mismatched mixed samples are informative hard mixed samples for deep models to localize discriminative features. In this paper, we thus are not trying to propose a more complicated dynamic mixup policy but rather an efficient mixup objective function with decoupled regularizer, named decoupled mixup (DM). The primary effect is that DM can adaptively utilize those hard mixed samples to mine discriminative features without losing the original smoothness of mixup. As a result, DM enables static mixup methods to achieve comparable or even exceed the performance of dynamic methods without any extra computation. This also leads to an interesting objective design problem for mixup training that we need to focus on both smoothing the decision boundaries and identifying discriminative features. Extensive experiments on supervised and semi-supervised learning benchmarks across seven datasets validate the effectiveness of DM.

----

## [0] The Utility of "Even if" Semifactual Explanation to Optimise Positive Outcomes

**Authors**: *Eoin M. Kenny, Weipeng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e146ca55a2b18be41942cfa677123d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e146ca55a2b18be41942cfa677123d-Abstract-Conference.html)

**Abstract**:

When users receive either a positive or negative outcome from an automated system, Explainable AI (XAI) has almost exclusively focused on how to mutate negative outcomes into positive ones by crossing a decision boundary using counterfactuals (e.g., "If you earn 2k more, we will accept your loan application"). Here, we instead focus on positive outcomes, and take the novel step of using XAI to optimise them (e.g., "Even if you wish to half your down-payment, we will still accept your loan application"). Explanations such as these that employ "even if..." reasoning, and do not cross a decision boundary, are known as semifactuals. To instantiate semifactuals in this context, we introduce the concept of Gain (i.e., how much a user stands to benefit from the explanation), and consider the first causal formalisation of semifactuals. Tests on benchmark datasets show our algorithms are better at maximising gain compared to prior work, and that causality is important in the process. Most importantly however, a user study supports our main hypothesis by showing people find semifactual explanations more useful than counterfactuals when they receive the positive outcome of a loan acceptance.

----

## [0] VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models

**Authors**: *Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html)

**Abstract**:

Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLATTACK to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multi-modal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multi-modal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level.  We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLATTACK framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models.

----

## [0] Mode Connectivity in Auction Design

**Authors**: *Christoph Hertrich, Yixin Tao, László A. Végh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html)

**Abstract**:

Optimal auction design is a fundamental problem in algorithmic game theory. This problem is notoriously difficult already in very simple settings. Recent work in differentiable economics showed that neural networks can efficiently learn known optimal auction mechanisms and discover interesting new ones. In an attempt to theoretically justify their empirical success, we focus on one of the first such networks, RochetNet, and a generalized version for affine maximizer auctions. We prove that they satisfy mode connectivity, i.e., locally optimal solutions are connected by a simple, piecewise linear path such that every solution on the path is almost as good as one of the two local optima. Mode connectivity has been recently investigated as an intriguing empirical and theoretically justifiable property of neural networks used for prediction problems. Our results give the first such analysis in the context of differentiable economics, where neural networks are used directly for solving non-convex optimization problems.

----

## [0] Katakomba: Tools and Benchmarks for Data-Driven NetHack

**Authors**: *Vladislav Kurenkov, Alexander Nikulin, Denis Tarasov, Sergey Kolesnikov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5f596699d8d4637532f955c7f2860f4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5f596699d8d4637532f955c7f2860f4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

NetHack is known as the frontier of reinforcement learning research where learning-based methods still need to catch up to rule-based solutions. One of the promising directions for a breakthrough is using pre-collected datasets similar to recent developments in robotics, recommender systems, and more under the umbrella of offline reinforcement learning (ORL). Recently, a large-scale NetHack dataset was released; while it was a necessary step forward, it has yet to gain wide adoption in the ORL community. In this work, we argue that there are three major obstacles for adoption: tool-wise, implementation-wise, and benchmark-wise. To address them, we develop an open-source library that provides workflow fundamentals familiar to the ORL community: pre-defined D4RL-style tasks, uncluttered baseline implementations, and reliable evaluation tools with accompanying configs and logs synced to the cloud.

----

## [0] Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model

**Authors**: *Peter Súkeník, Marco Mondelli, Christoph H. Lampert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html)

**Abstract**:

Neural collapse (NC) refers to the surprising structure of the last layer of deep neural networks in the terminal phase of gradient descent training. Recently, an increasing amount of experimental evidence has pointed to the propagation of NC to earlier layers of neural networks. However, while the NC in the last layer is well studied theoretically, much less is known about its multi-layered counterpart - deep neural collapse (DNC). In particular, existing work focuses either on linear layers or only on the last two layers at the price of an extra assumption. Our work fills this gap by generalizing the established analytical framework for NC - the unconstrained features model - to multiple non-linear layers. Our key technical contribution is to show that, in a deep unconstrained features model, the unique global optimum for binary classification exhibits all the properties typical of DNC. This explains the existing experimental evidence of DNC. We also empirically show that (i) by optimizing deep unconstrained features models via gradient descent, the resulting solution agrees well with our theory, and (ii) trained networks recover the unconstrained features suitable for the occurrence of DNC, thus supporting the validity of this modeling principle.

----

## [0] IEBins: Iterative Elastic Bins for Monocular Depth Estimation

**Authors**: *Shuwei Shao, Zhongcai Pei, Xingming Wu, Zhong Liu, Weihai Chen, Zhengguo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html)

**Abstract**:

Monocular depth estimation (MDE) is a fundamental topic of geometric computer vision and a core technique for many downstream applications. Recently, several methods reframe the MDE as a classification-regression problem where a linear combination of probabilistic distribution and bin centers is used to predict depth. In this paper, we propose a novel concept of iterative elastic bins (IEBins) for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. Furthermore, we develop a dedicated framework composed of a feature extractor and an iterative optimizer that has powerful temporal context modeling capabilities benefiting from the GRU-based architecture. Extensive experiments on the KITTI, NYU-Depth-v2 and SUN RGB-D datasets demonstrate that the proposed method surpasses prior state-of-the-art competitors. The source code is publicly available at https://github.com/ShuweiShao/IEBins.

----

## [0] Fine-Tuning Language Models with Just Forward Passes

**Authors**: *Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html)

**Abstract**:

Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zerothorder optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same budget. We conduct comprehensive experiments across model types (masked and autoregressive LMs), model scales (up to 66B), and downstream tasks (classification, multiple-choice, and generation). Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear probing; (2) MeZO achieves comparable performance to fine-tuning with backpropagation across multiple tasks, with up to 12× memory reduction and up to 2× GPU-hour reduction in our implementation; (3) MeZO is compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix tuning; (4) MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1). We support our empirical findings with theoretical insights, highlighting how adequate pre-training and task prompts enable MeZO to fine-tune huge models, despite classical ZO analyses suggesting otherwise.

----

## [0] HEDNet: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds

**Authors**: *Gang Zhang, Junnan Chen, Guohuan Gao, Jianmin Li, Xiaolin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a64e641fa00a7eb9500cb7e1835d0495-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a64e641fa00a7eb9500cb7e1835d0495-Abstract-Conference.html)

**Abstract**:

3D object detection in point clouds is important for autonomous driving systems. A primary challenge in 3D object detection stems from the sparse distribution of points within the 3D scene. Existing high-performance methods typically employ 3D sparse convolutional neural networks with small kernels to extract features. To reduce computational costs, these methods resort to submanifold sparse convolutions, which prevent the information exchange among spatially disconnected features. Some recent approaches have attempted to address this problem by introducing large-kernel convolutions or self-attention mechanisms, but they either achieve limited accuracy improvements or incur excessive computational costs. We propose HEDNet, a hierarchical encoder-decoder network for 3D object detection, which leverages encoder-decoder blocks to capture long-range dependencies among features in the spatial space, particularly for large and distant objects. We conducted extensive experiments on the Waymo Open and nuScenes datasets. HEDNet achieved superior detection accuracy on both datasets than previous state-of-the-art methods with competitive efficiency. The code is available at https://github.com/zhanggang001/HEDNet.

----

## [0] FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning

**Authors**: *Jinyuan Jia, Zhuowen Yuan, Dinuka Sahabandu, Luyao Niu, Arezoo Rajabi, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) provides a distributed training paradigm where multiple clients can jointly train a global model without sharing their local data. However, recent studies have shown that FL offers an additional surface for backdoor attacks. For instance, an attacker can compromise a subset of clients and thus corrupt the global model to misclassify an input with a backdoor trigger as the adversarial target. Existing defenses for FL against backdoor attacks usually detect and exclude the corrupted information from the compromised clients based on a static attacker model. However, such defenses are inadequate against dynamic attackers who strategically adapt their attack strategies. To bridge this gap, we model the strategic interactions between the defender and dynamic attackers as a minimax game. Based on the analysis of the game, we design an interactive defense mechanism FedGame. We prove that under mild assumptions, the global model trained with FedGame under backdoor attacks is close to that trained without attacks. Empirically, we compare FedGame with multiple state-of-the-art baselines on several benchmark datasets under various attacks. We show that FedGame can effectively defend against strategic attackers and achieves significantly higher robustness than baselines. Our code is available at: https://github.com/AI-secure/FedGame.

----

## [0] Ensemble-based Deep Reinforcement Learning for Vehicle Routing Problems under Distribution Shift

**Authors**: *Yuan Jiang, Zhiguang Cao, Yaoxin Wu, Wen Song, Jie Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a68120d2eb2f53f7d9e71547591aef11-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a68120d2eb2f53f7d9e71547591aef11-Abstract-Conference.html)

**Abstract**:

While performing favourably on the independent and identically distributed (i.i.d.) instances, most of the existing neural methods for vehicle routing problems (VRPs) struggle to generalize in the presence of a distribution shift. To tackle this issue, we propose an ensemble-based deep reinforcement learning method for VRPs, which learns a group of diverse sub-policies to cope with various instance distributions. In particular, to prevent convergence of the parameters to the same one, we enforce diversity across sub-policies by leveraging Bootstrap with random initialization. Moreover, we also explicitly pursue inequality between sub-policies by exploiting regularization terms during training to further enhance diversity. Experimental results show that our method is able to outperform the state-of-the-art neural baselines on randomly generated instances of various distributions, and also generalizes favourably on the benchmark instances from TSPLib and CVRPLib, which confirmed the effectiveness of the whole method and the respective designs.

----

## [0] Module-wise Training of Neural Networks via the Minimizing Movement Scheme

**Authors**: *Skander Karkar, Ibrahim Ayed, Emmanuel de Bézenac, Patrick Gallinari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6a1e4c756d700d9aedcc1896a7e6fb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6a1e4c756d700d9aedcc1896a7e6fb0-Abstract-Conference.html)

**Abstract**:

Greedy layer-wise or module-wise training of neural networks is compelling in constrained and on-device settings where memory is limited, as it circumvents a number of problems of end-to-end back-propagation. However, it suffers from a stagnation problem, whereby early layers overfit and deeper layers stop increasing the test accuracy after a certain depth. We propose to solve this issue by introducing a simple module-wise regularization inspired by the minimizing movement scheme for gradient flows in distribution space. We call the method TRGL for Transport Regularized Greedy Learning and study it theoretically, proving that it leads to greedy modules that are regular and that progressively solve the task. Experimentally, we show improved accuracy of module-wise training of various architectures such as ResNets, Transformers and VGG, when our regularization is added, superior to that of other module-wise training methods and often to end-to-end training, with as much as 60% less memory usage.

----

## [0] POMDP Planning for Object Search in Partially Unknown Environment

**Authors**: *Yongbo Chen, Hanna Kurniawati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)

**Abstract**:

Efficiently searching for target objects in complex environments that contain various types of furniture, such as shelves, tables, and beds, is crucial for mobile robots, but it poses significant challenges due to various factors such as localization errors, limited field of view, and visual occlusion. To address this problem, we propose a Partially Observable Markov Decision Process (POMDP) formulation with a growing state space for object search in a 3D region. We solve this POMDP by carefully designing a perception module and developing a planning algorithm, called Growing Partially Observable Monte-Carlo Planning (GPOMCP), based on online Monte-Carlo tree search and belief tree reuse with a novel upper confidence bound. We have demonstrated that belief tree reuse is reasonable and achieves good performance when the belief differences are limited. Additionally, we introduce a guessed target object with an updating grid world to guide the search in the information-less and reward-less cases, like the absence of any detected objects. We tested our approach using Gazebo simulations on four scenarios of target finding in a realistic indoor living environment with the Fetch robot simulator. Compared to the baseline approaches, which are based on POMCP, our results indicate that our approach enables the robot to find the target object with a higher success rate faster while using the same computational requirements.

----

## [0] On the Statistical Consistency of Risk-Sensitive Bayesian Decision-Making

**Authors**: *Prateek Jaiswal, Harsha Honnappa, Vinayak A. Rao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html)

**Abstract**:

We study data-driven decision-making problems in the Bayesian framework, where the expectation in the Bayes risk is replaced by a risk-sensitive entropic risk measure with respect to the posterior distribution. We focus on problems where calculating the posterior distribution is intractable, a typical situation in modern applications with large datasets and complex data generating models. We leverage a dual representation of the entropic risk measure to introduce a novel risk-sensitive variational Bayesian (RSVB) framework for jointly computing a risk-sensitive posterior approximation and the corresponding decision rule. Our general framework includes \textit{loss-calibrated} VB (Lacoste-Julien et al. [2011] ) as a special case. We also study the impact of these computational approximations on the predictive performance of the inferred decision rules. We compute the convergence rates of the RSVB approximate posterior and the corresponding optimal value. We illustrate our theoretical findings in parametric and nonparametric settings with the help of three examples.

----

## [0] Does Graph Distillation See Like Vision Dataset Counterpart?

**Authors**: *Beining Yang, Kai Wang, Qingyun Sun, Cheng Ji, Xingcheng Fu, Hao Tang, Yang You, Jianxin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6efa49c54bedf4411f1bcd32f15937a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6efa49c54bedf4411f1bcd32f15937a-Abstract-Conference.html)

**Abstract**:

Training on large-scale graphs has achieved remarkable results in graph representation learning, but its cost and storage have attracted increasing concerns. Existing graph condensation methods primarily focus on optimizing the feature matrices of condensed graphs while overlooking the impact of the structure information from the original graphs. To investigate the impact of the structure information, we conduct analysis from the spectral domain and empirically identify substantial Laplacian Energy Distribution (LED) shifts in previous works. Such shifts lead to poor performance in cross-architecture generalization and specific tasks, including anomaly detection and link prediction. In this paper, we propose a novel Structure-broadcasting Graph Dataset Distillation (\textbf{SGDD}) scheme for broadcasting the original structure information to the generation of the synthetic one, which explicitly prevents overlooking the original structure information. Theoretically, the synthetic graphs by SGDD are expected to have smaller LED shifts than previous works, leading to superior performance in both cross-architecture settings and specific tasks.We validate the proposed SGDD~across 9 datasets and achieve state-of-the-art results on all of them: for example, on YelpChi dataset, our approach maintains 98.6\% test accuracy of training on the original graph dataset with 1,000 times saving on the scale of the graph. Moreover, we empirically evaluate there exist 17.6\% $\sim$ 31.4\% reductions in LED shift crossing 9 datasets. Extensive experiments and analysis verify the effectiveness and necessity of the proposed designs. The code will be made public.

----

## [0] Online Learning under Adversarial Nonlinear Constraints

**Authors**: *Pavel Kolev, Georg Martius, Michael Muehlebach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6f2763089c0bd8f56006c42f09ee24c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6f2763089c0bd8f56006c42f09ee24c-Abstract-Conference.html)

**Abstract**:

In many applications, learning systems are required to process continuous non-stationary data streams.We study this problem in an online learning framework and propose an algorithm that can deal with adversarial time-varying and nonlinear constraints.As we show in our work, the algorithm called Constraint Violation Velocity Projection (CVV-Pro) achieves $\sqrt{T}$ regret and converges to the feasible set at a rate of $1/\sqrt{T}$, despite the fact that the feasible set is slowly time-varying and a priori unknown to the learner. CVV-Pro only relies on local sparse linear approximations of the feasible set and therefore avoids optimizing over the entire set at each iteration, which is in sharp contrast to projected gradients or Frank-Wolfe methods. We also empirically evaluate our algorithm on two-player games, where the players are subjected to a shared constraint.

----

## [0] Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents

**Authors**: *Woojun Kim, Yongjae Shin, Jongeui Park, Youngchul Sung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6f6a5c517b2b92f3d309786af64086c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6f6a5c517b2b92f3d309786af64086c-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (RL) has achieved remarkable success in solving complex tasks through its integration with deep neural networks (DNNs) as function approximators. However, the reliance on DNNs has introduced a new challenge called primacy bias, whereby these function approximators tend to prioritize early experiences, leading to overfitting. To alleviate this bias, a reset method has been proposed, which involves periodic resets of a portion or the entirety of a deep RL agent while preserving the replay buffer. However, the use of this method can result in performance collapses after executing the reset, raising concerns from the perspective of safe RL and regret minimization. In this paper, we propose a novel reset-based method that leverages deep ensemble learning to address the limitations of the vanilla reset method and enhance sample efficiency. The effectiveness of the proposed method is validated through various experiments including those in the domain of safe RL. Numerical results demonstrate its potential for real-world applications requiring high sample efficiency and safety considerations.

----

## [0] Im-Promptu: In-Context Composition from Image Prompts

**Authors**: *Bhishma Dedhia, Michael Chang, Jake Snell, Tom Griffiths, Niraj K. Jha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html)

**Abstract**:

Large language models are few-shot learners that can solve diverse tasks from a handful of demonstrations. This implicit understanding of tasks suggests that the attention mechanisms over word tokens may play a role in analogical reasoning. In this work, we investigate whether analogical reasoning can enable in-context composition over composable elements of visual stimuli. First, we introduce a suite of three benchmarks to test the generalization properties of a visual in-context learner. We formalize the notion of an analogy-based in-context learner and use it to design a meta-learning framework called Im-Promptu. Whereas the requisite token granularity for language is well established, the appropriate compositional granularity for enabling in-context generalization in visual stimuli is usually unspecified. To this end, we use Im-Promptu to train multiple agents with different levels of compositionality, including vector representations, patch representations, and object slots. Our experiments reveal tradeoffs between extrapolation abilities and the degree of compositionality, with non-compositional representations extending learned composition rules to unseen domains but performing poorly on combinatorial tasks. Patch-based representations require patches to contain entire objects for robust extrapolation. At the same time, object-centric tokenizers coupled with a cross-attention module generate consistent and high-fidelity solutions, with these inductive biases being particularly crucial for compositional generalization. Lastly, we demonstrate a use case of Im-Promptu as an intuitive programming interface for image generation.

----

## [0] Sequential Predictive Two-Sample and Independence Testing

**Authors**: *Aleksandr Podkopaev, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a72b207734d6112f6b47447e46be40e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a72b207734d6112f6b47447e46be40e9-Abstract-Conference.html)

**Abstract**:

We study the problems of sequential nonparametric two-sample and independence testing. Sequential tests process data online and allow using observed data to decide whether to stop and reject the null hypothesis or to collect more data, while maintaining type I error control. We build upon the principle of (nonparametric) testing by betting, where a gambler places bets on future observations and their wealth measures evidence against the null hypothesis. While recently developed kernel-based betting strategies often work well on simple distributions, selecting a suitable kernel for high-dimensional or structured data, such as images, is often nontrivial. To address this drawback, we design prediction-based betting strategies that rely on the following fact: if a sequentially updated predictor starts to consistently determine (a) which distribution an instance is drawn from, or (b) whether an instance is drawn from the joint distribution or the product of the marginal distributions (the latter produced by external randomization), it provides evidence against the two-sample or independence nulls respectively. We empirically demonstrate the superiority of our tests over kernel-based approaches under structured settings. Our tests can be applied beyond the case of independent and identically distributed data, remaining valid and powerful even when the data distribution drifts over time.

----

## [0] Towards Symmetry-Aware Generation of Periodic Materials

**Authors**: *Youzhi Luo, Chengkai Liu, Shuiwang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html)

**Abstract**:

We consider the problem of generating periodic materials with deep models. While symmetry-aware molecule generation has been studied extensively, periodic materials possess different symmetries, which have not been completely captured by existing methods.In this work, we propose SyMat, a novel material generation approach that can capture physical symmetries of periodic material structures. SyMat generates atom types and lattices of materials through generating atom type sets, lattice lengths and lattice angles with a variational auto-encoder model. In addition, SyMat employs a score-based diffusion model to generate atom coordinates of materials, in which a novel symmetry-aware probabilistic model is used in the coordinate diffusion process. We show that SyMat is theoretically invariant to all symmetry transformations on materials and demonstrate that SyMat achieves promising performance on random generation and property optimization tasks. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

----

## [0] DICES Dataset: Diversity in Conversational AI Evaluation for Safety

**Authors**: *Lora Aroyo, Alex S. Taylor, Mark Díaz, Christopher Homan, Alicia Parrish, Gregory Serapio-García, Vinodkumar Prabhakaran, Ding Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning approaches often require training and evaluation datasets with a clear separation between positive and negative examples. This requirement overly simplifies the natural subjectivity present in many tasks, and obscures the inherent diversity in human perceptions and opinions about many content items. Preserving the variance in content and diversity in human perceptions in datasets is often quite expensive and laborious. This is especially troubling when building safety datasets for conversational AI systems, as safety is socio-culturally situated in this context. To demonstrate this crucial aspect of conversational AI safety, and to facilitate in-depth model performance analyses, we introduce the DICES (Diversity In Conversational AI Evaluation for Safety) dataset that contains fine-grained demographics information about raters, high replication of ratings per item to ensure statistical power for analyses, and encodes rater votes as distributions across different demographics to allow for in-depth explorations of different aggregation strategies. The DICES dataset enables the observation and measurement of variance, ambiguity, and diversity in the context of safety for conversational AI. We further describe a set of metrics that show how rater diversity influences safety perception across different geographic regions, ethnicity groups, age groups, and genders. The goal of the DICES dataset is to be used as a shared resource and benchmark that respects diverse perspectives during safety evaluation of conversational AI systems.

----

## [0] SALSA VERDE: a machine learning attack on LWE with sparse small secrets

**Authors**: *Cathy Yuanchen Li, Emily Wenger, Zeyuan Allen-Zhu, François Charton, Kristin E. Lauter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html)

**Abstract**:

Learning with Errors (LWE) is a hard math problem used in post-quantum cryptography. Homomorphic Encryption (HE) schemes rely on the hardness of the LWE problem for their security, and two LWE-based cryptosystems were recently standardized by NIST for digital signatures and key exchange (KEM).  Thus, it is critical to continue assessing the security of LWE and specific parameter choices. For example, HE uses secrets with small entries, and the HE community has considered standardizing small sparse secrets to improve efficiency and functionality.  However, prior work, SALSA and PICANTE, showed that ML attacks can recover sparse binary secrets. Building on these, we propose VERDE, an improved ML attack that can recover sparse binary, ternary, and narrow Gaussian secrets. Using improved preprocessing and secret recovery techniques, VERDE can attack LWE with larger dimensions ($n=512$) and smaller moduli ($\log_2 q=12$ for $n=256$), using less time and power. We propose novel architectures for scaling. Finally, we develop a theory that explains the success of ML LWE attacks.

----

## [0] Designing Robust Transformers using Robust Kernel Density Estimation

**Authors**: *Xing Han, Tongzheng Ren, Tan Nguyen, Khai Nguyen, Joydeep Ghosh, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html)

**Abstract**:

Transformer-based architectures have recently exhibited remarkable successes across different domains beyond just powering large language models. However, existing approaches typically focus on predictive accuracy and computational cost, largely ignoring certain other practical issues such as robustness to contaminated samples. In this paper, by re-interpreting the self-attention mechanism as a non-parametric kernel density estimator, we adapt classical robust kernel density estimation methods to develop novel classes of transformers that are resistant to adversarial attacks and data contamination. We first propose methods that down-weight outliers in RKHS when computing the self-attention operations. We empirically show that these methods produce improved performance over existing state-of-the-art methods, particularly on image data under adversarial attacks. Then we leverage the median-of-means principle to obtain another efficient approach that results in noticeably enhanced performance and robustness on language modeling and time series classification tasks. Our methods can be combined with existing transformers to augment their robust properties, thus promising to impact a wide variety of applications.

----

## [0] Benchmarking Distribution Shift in Tabular Data with TableShift

**Authors**: *Josh Gardner, Zoran Popovic, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Robustness to distribution shift has become a growing concern for text and image models as they transition from research subjects to deployment in the real world. However, high-quality benchmarks for distribution shift in tabular machine learning tasks are still lacking despite the widespread real-world use of tabular data and differences in the models used for tabular data in comparison to text and images. As a consequence, the robustness of tabular models to distribution shift is poorly understood. To address this issue, we introduce TableShift, a distribution shift benchmark for tabular data. TableShift contains 15 binary classification tasks in total, each with an associated shift, and includes a diverse set of data sources, prediction targets, and distribution shifts. The benchmark covers domains including finance, education, public policy, healthcare, and civic participation, and is accessible using only a few lines of Python code via the TableShift API. We conduct a large-scale study comparing several state-of-the-art tabular data models alongside robust learning and domain generalization methods on the benchmark tasks. Our study demonstrates (1) a linear trend between in-distribution (ID) and out-of-distribution (OOD) accuracy; (2) domain robustness methods can reduce shift gaps but at the cost of reduced ID accuracy; (3) a strong relationship between shift gap (difference between ID and OOD performance) and shifts in the label distribution. The benchmark data, Python package, model implementations, and more information about TableShift are available at https://github.com/mlfoundations/tableshift and https://tableshift.org .

----

## [0] Weakly Supervised 3D Open-vocabulary Segmentation

**Authors**: *Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El-Saddik, Christian Theobalt, Eric P. Xing, Shijian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a76b693f36916a5ed84d6e5b39a0dc03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a76b693f36916a5ed84d6e5b39a0dc03-Abstract-Conference.html)

**Abstract**:

Open-vocabulary segmentation of 3D scenes is a fundamental function of human perception and thus a crucial objective in computer vision research. However, this task is heavily impeded by the lack of large-scale and diverse 3D open-vocabulary segmentation datasets for training robust and generalizable models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation models helps but it compromises the open-vocabulary feature as the 2D models are mostly finetuned with close-vocabulary datasets. We tackle the challenges in 3D open-vocabulary segmentation by exploiting pre-trained foundation models CLIP and DINO in a weakly supervised manner. Specifically, given only the open-vocabulary text descriptions of the objects in a scene, we distill the open-vocabulary multimodal knowledge and object reasoning capability of CLIP and DINO into a neural radiance field (NeRF), which effectively lifts 2D features into view-consistent 3D segmentation. A notable aspect of our approach is that it does not require any manual segmentation annotations for either the foundation models or the distillation process. Extensive experiments show that our method even outperforms fully supervised models trained with segmentation annotations in certain scenes, suggesting that 3D open-vocabulary segmentation can be effectively learned from 2D images and text-image pairs. Code is available at https://github.com/Kunhao-Liu/3D-OVS.

----

## [0] Better Private Linear Regression Through Better Private Feature Selection

**Authors**: *Travis Dick, Jennifer Gillenwater, Matthew Joseph*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a79699db176ed0efc04a9da171e52112-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a79699db176ed0efc04a9da171e52112-Abstract-Conference.html)

**Abstract**:

Existing work on differentially private linear regression typically assumes that end users can precisely set data bounds or algorithmic hyperparameters. End users often struggle to meet these requirements without directly examining the data (and violating privacy). Recent work has attempted to develop solutions that shift these burdens from users to algorithms, but they struggle to provide utility as the feature dimension grows. This work extends these algorithms to higher-dimensional problems by introducing a differentially private feature selection method based on Kendall rank correlation. We prove a utility guarantee for the setting where features are normally distributed and conduct experiments across 25 datasets. We find that adding this private feature selection step before regression significantly broadens the applicability of ``plug-and-play'' private linear regression algorithms at little additional cost to privacy, computation, or decision-making by the end user.

----

## [0] Geometric Neural Diffusion Processes

**Authors**: *Emile Mathieu, Vincent Dutordoir, Michael Hutchinson, Valentin De Bortoli, Yee Whye Teh, Richard E. Turner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)

**Abstract**:

Denoising diffusion models have proven to be a flexible and effective paradigm for generative modelling.Their recent extension to infinite dimensional Euclidean spaces has allowed for the modelling of stochastic processes.However, many problems in the natural sciences incorporate symmetries and involve data living in non-Euclidean spaces.In this work, we extend the framework of diffusion models to incorporate a series of geometric priors in infinite-dimension modelling.We do so by a) constructing a noising process which admits, as limiting distribution, a geometric Gaussian process that transforms under the symmetry group of interest, and b) approximating the score with a neural network that is equivariant w.r.t. this group.We show that with these conditions, the generative functional model admits the same symmetry.We demonstrate scalability and capacity of the model, using a novel Langevin-based conditional sampler, to fit complex scalar and vector fields, with Euclidean and spherical codomain, on synthetic and real-world weather data.

----

## [0] Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations

**Authors**: *Yiheng Lin, James A. Preiss, Emile Anand, Yingying Li, Yisong Yue, Adam Wierman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7a7180fe7f82ff98eee0827c5e9c141-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7a7180fe7f82ff98eee0827c5e9c141-Abstract-Conference.html)

**Abstract**:

We study online adaptive policy selection in systems with time-varying costs and dynamics. We develop the Gradient-based Adaptive Policy Selection (GAPS) algorithm together with a general analytical framework for online policy selection via online optimization. Under our proposed notion of contractive policy classes, we show that GAPS approximates the behavior of an ideal online gradient descent algorithm on the policy parameters while requiring less information and computation. When convexity holds, our algorithm is the first to achieve optimal policy regret. When convexity does not hold, we provide the first local regret bound for online policy selection. Our numerical experiments show that GAPS can adapt to changing environments more quickly than existing benchmarks.

----

## [0] IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

**Authors**: *Pascal Leroy, Pablo G. Morato, Jonathan Pisane, Athanasios Kolios, Damien Ernst*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7a7c0c92f195cce85f99768621ac6c0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7a7c0c92f195cce85f99768621ac6c0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce IMP-MARL, an open-source suite of multi-agent reinforcement learning (MARL) environments for large-scale Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.In IMP, a multi-component engineering system is subject to a risk of failure due to its components' damage condition.Specifically, each agent plans inspections and repairs for a specific system component, aiming to minimise maintenance costs while cooperating to minimise system failure risk.With IMP-MARL, we release several environments including one related to offshore wind structural systems, in an effort to meet today's needs to improve management strategies to support sustainable and reliable energy systems.Supported by IMP practical engineering environments featuring up to 100 agents, we conduct a benchmark campaign, where the scalability and performance of state-of-the-art cooperative MARL methods are compared against expert-based heuristic policies. The results reveal that centralised training with decentralised execution methods scale better with the number of agents than fully centralised or decentralised RL approaches, while also outperforming expert-based heuristic policies in most IMP environments.Based on our findings, we additionally outline remaining cooperation and scalability challenges that future MARL methods should still address.Through IMP-MARL, we encourage the implementation of new environments and the further development of MARL methods.

----

## [0] Learning Multi-agent Behaviors from Distributed and Streaming Demonstrations

**Authors**: *Shicheng Liu, Minghui Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7affe50ab177b9a7f0a05f07a9ca205-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7affe50ab177b9a7f0a05f07a9ca205-Abstract-Conference.html)

**Abstract**:

This paper considers the problem of inferring the behaviors of multiple interacting experts by estimating their reward functions and constraints where the distributed demonstrated trajectories are sequentially revealed to a group of learners. We formulate the problem as a distributed online bi-level optimization problem where the outer-level problem is to estimate the reward functions and the inner-level problem is to learn the constraints and corresponding policies. We propose a novel ``multi-agent behavior inference from distributed and streaming demonstrations" (MA-BIRDS) algorithm that allows the learners to solve the outer-level and inner-level problems in a single loop through intermittent communications. We formally guarantee that the distributed learners achieve consensus on reward functions, constraints, and policies, the average local regret (over $N$ online iterations) decreases at the rate of $O(1/N^{1-\eta_1}+1/N^{1-\eta_2}+1/N)$, and the cumulative constraint violation increases sub-linearly at the rate of $O(N^{\eta_2}+1)$ where $\eta_1,\eta_2\in (1/2,1)$.

----

## [0] CAPro: Webly Supervised Learning with Cross-modality Aligned Prototypes

**Authors**: *Yulei Qin, Xingyu Chen, Yunhang Shen, Chaoyou Fu, Yun Gu, Ke Li, Xing Sun, Rongrong Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7e0d77325db843fd5baf1298163e89a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7e0d77325db843fd5baf1298163e89a-Abstract-Conference.html)

**Abstract**:

Webly supervised learning has attracted increasing attention for its effectiveness in exploring publicly accessible data at scale without manual annotation. However, most existing methods of learning with web datasets are faced with challenges from label noise, and they have limited assumptions on clean samples under various noise. For instance, web images retrieved with queries of ”tiger cat“ (a cat species) and ”drumstick“ (a musical instrument) are almost dominated by images of tigers and chickens, which exacerbates the challenge of fine-grained visual concept learning. In this case, exploiting both web images and their associated texts is a requisite solution to combat real-world noise. In this paper, we propose Cross-modality Aligned Prototypes (CAPro), a unified prototypical contrastive learning framework to learn visual representations with correct semantics. For one thing, we leverage textual prototypes, which stem from the distinct concept definition of classes, to select clean images by text matching and thus disambiguate the formation of visual prototypes. For another, to handle missing and mismatched noisy texts, we resort to the visual feature space to complete and enhance individual texts and thereafter improve text matching. Such semantically aligned visual prototypes are further polished up with high-quality samples, and engaged in both cluster regularization and noise removal. Besides, we propose collective bootstrapping to encourage smoother and wiser label reference from appearance-similar instances in a manner of dictionary look-up. Extensive experiments on WebVision1k and NUS-WIDE (Web) demonstrate that CAPro well handles realistic noise under both single-label and multi-label scenarios. CAPro achieves new state-of-the-art performance and exhibits robustness to open-set recognition. Codes are available at https://github.com/yuleiqin/capro.

----

## [0] Diversify & Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement

**Authors**: *Daesol Cho, Seungjae Lee, H. Jin Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) often faces the challenges of uninformed search problems where the agent should explore without access to the domain knowledge such as characteristics of the environment or external rewards. To tackle these challenges, this work proposes a new approach for curriculum RL called $\textbf{D}$iversify for $\textbf{D}$isagreement \& $\textbf{C}$onquer ($\textbf{D2C}$). Unlike previous curriculum learning methods, D2C requires only a few examples of desired outcomes and works in any environment, regardless of its geometry or the distribution of the desired outcome examples. The proposed method performs diversification of the goal-conditional classifiers to identify similarities between visited and desired outcome states and ensures that the classifiers disagree on states from out-of-distribution, which enables quantifying the unexplored region and designing an arbitrary goal-conditioned intrinsic reward signal in a simple and intuitive way. The proposed method then employs bipartite matching to define a curriculum learning objective that produces a sequence of well-adjusted intermediate goals, which enable the agent to automatically explore and conquer the unexplored region. We present experimental results demonstrating that D2C outperforms prior curriculum RL methods in both quantitative and qualitative aspects, even with the arbitrarily distributed desired outcome examples.

----

## [0] Distribution-Free Model-Agnostic Regression Calibration via Nonparametric Methods

**Authors**: *Shang Liu, Zhongze Cai, Xiaocheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a81dc87f7b3b7ab8489d5bb48c4a8d92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a81dc87f7b3b7ab8489d5bb48c4a8d92-Abstract-Conference.html)

**Abstract**:

In this paper, we consider the uncertainty quantification problem for regression models. Specifically, we consider an individual calibration objective for characterizing the quantiles of the prediction model. While such an objective is well-motivated from downstream tasks such as newsvendor cost, the existing methods have been largely heuristic and lack of statistical guarantee in terms of individual calibration. We show via simple examples that the existing methods focusing on population-level calibration guarantees such as average calibration or sharpness can lead to harmful and unexpected results. We propose simple nonparametric calibration methods that are agnostic of the underlying prediction model and enjoy both computational efficiency and statistical consistency. Our approach enables a better understanding of the possibility of individual calibration, and we establish matching upper and lower bounds for the calibration error of our proposed methods. Technically, our analysis combines the nonparametric analysis with a covering number argument for parametric analysis, which advances the existing theoretical analyses in the literature of nonparametric density estimation and quantile bandit problems. Importantly, the nonparametric perspective sheds new theoretical insights into regression calibration in terms of the curse of dimensionality and reconciles the existing results on the impossibility of individual calibration. To our knowledge, we make the first effort to reach both individual calibration and finite-sample guarantee with minimal assumptions in terms of conformal prediction. Numerical experiments show the advantage of such a simple approach under various metrics, and also under covariates shift. We hope our work provides a simple benchmark and a starting point of theoretical ground for future research on regression calibration.

----

## [0] Synthetic-to-Real Pose Estimation with Geometric Reconstruction

**Authors**: *Qiuxia Lin, Kerui Gu, Linlin Yang, Angela Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html)

**Abstract**:

Pose estimation is remarkably successful under supervised learning, but obtaining annotations, especially for new deployments, is costly and time-consuming. This work tackles adapting models trained on synthetic data to real-world target domains with only unlabelled data. A common approach is model fine-tuning with pseudo-labels from the target domain; yet many pseudo-labelling strategies cannot provide sufficient high-quality pose labels. This work proposes a reconstruction-based strategy as a complement to pseudo-labelling for synthetic-to-real domain adaptation. We generate the driving image by geometrically transforming a base image according to the predicted keypoints and enforce a reconstruction loss to refine the predictions. It provides a novel solution to effectively correct confident yet inaccurate keypoint locations through image reconstruction in domain adaptation. Our approach outperforms the previous state-of-the-arts by 8% for PCK on four large-scale hand and human real-world datasets. In particular, we excel on endpoints such as fingertips and head, with 7.2% and 29.9% improvements in PCK.

----

## [0] Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies

**Authors**: *Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, Yonghong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html)

**Abstract**:

Vanilla spiking neurons in Spiking Neural Networks (SNNs) use charge-fire-reset neuronal dynamics, which can only be simulated serially and can hardly learn long-time dependencies. We find that when removing reset, the neuronal dynamics can be reformulated in a non-iterative form and parallelized. By rewriting neuronal dynamics without reset to a general formulation, we propose the Parallel Spiking Neuron (PSN), which generates hidden states that are independent of their predecessors, resulting in parallelizable neuronal dynamics and extremely high simulation speed. The weights of inputs in the PSN are fully connected, which maximizes the utilization of temporal information. To avoid the use of future inputs for step-by-step inference, the weights of the PSN can be masked, resulting in the masked PSN. By sharing weights across time-steps based on the masked PSN, the sliding PSN is proposed to handle sequences of varying lengths. We evaluate the PSN family on simulation speed and temporal/static data classification, and the results show the overwhelming advantage of the PSN family in efficiency and accuracy. To the best of our knowledge, this is the first study about parallelizing spiking neurons and can be a cornerstone for the spiking deep learning research. Our codes are available at https://github.com/fangwei123456/Parallel-Spiking-Neuron.

----

## [0] Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment

**Authors**: *Zihui Xue, Kristen Grauman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html)

**Abstract**:

The egocentric and exocentric viewpoints of a human activity look dramatically different, yet invariant representations to link them are essential for many potential applications in robotics and augmented reality.  Prior work is limited to learning view-invariant features from paired synchronized viewpoints.  We relax that strong data assumption and propose to learn fine-grained action features that are invariant to the viewpoints by aligning egocentric and exocentric videos in time, even when not captured simultaneously or in the same environment. To this end, we propose AE2, a self-supervised embedding approach with two key designs: (1) an object-centric encoder that explicitly focuses on regions corresponding to hands and active objects; (2) a contrastive-based alignment objective that leverages temporally reversed frames as negative samples. For evaluation, we establish a benchmark for fine-grained video understanding in the ego-exo context, comprising four datasets---including an ego tennis forehand dataset we collected, along with dense per-frame labels we annotated for each dataset. On the four datasets, our AE2 method strongly outperforms prior work in a variety of fine-grained downstream tasks, both in regular and cross-view settings.

----

## [0] Training Private Models That Know What They Don't Know

**Authors**: *Stephan Rabanser, Anvith Thudi, Abhradeep Guha Thakurta, Krishnamurthy Dvijotham, Nicolas Papernot*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8526465a91166fbb90aaa8452b21eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8526465a91166fbb90aaa8452b21eda-Abstract-Conference.html)

**Abstract**:

Training reliable deep learning models which avoid making overconfident but incorrect predictions is a longstanding challenge. This challenge is further exacerbated when learning has to be differentially private: protection provided to sensitive data comes at the price of injecting additional randomness into the learning process. In this work, we conduct a thorough empirical investigation of selective classifiers---that can abstain under uncertainty---under a differential privacy constraint. We find that some popular selective prediction approaches are ineffective in a differentially private setting because they increase the risk of privacy leakage. At the same time, we identify that a recent approach that only uses checkpoints produced by an off-the-shelf private learning algorithm stands out as particularly suitable under DP. Further, we show that differential privacy does not just harm utility but also degrades selective classification performance. To analyze this effect across privacy levels, we propose a novel evaluation mechanism which isolates selective prediction performance across model utility levels at full coverage. Our experimental results show that recovering the performance level attainable by non-private models is possible but comes at a considerable coverage cost as the privacy budget decreases.

----

## [0] Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors**: *Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)

**Abstract**:

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper, we leverage a mapping between reward functions and optimal policies to show that this constrained reward maximization problem can be optimized exactly with a single stage of policy training, essentially solving a classification problem on the human preference data. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for fitting a reward model, sampling from the LM during fine-tuning, or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds RLHF's ability to control sentiment of generations and improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

----

## [0] Diffusion Representation for Asymmetric Kernels via Magnetic Transform

**Authors**: *Mingzhen He, Fan He, Ruikai Yang, Xiaolin Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html)

**Abstract**:

As a nonlinear dimension reduction technique, the diffusion map (DM) has been widely used. In DM, kernels play an important role for capturing the nonlinear relationship of data. However, only symmetric kernels can be used now, which prevents the use of DM in directed graphs, trophic networks, and other real-world scenarios where the intrinsic and extrinsic geometries in data are asymmetric. A promising technique is the magnetic transform which  converts an asymmetric matrix to a Hermitian one. However, we are facing essential problems, including how diffusion distance could be preserved and how divergence could be avoided during diffusion process. Via theoretical proof, we successfully establish a diffusion representation framework with the magnetic transform, named MagDM. The effectiveness and robustness for dealing data endowed with asymmetric proximity are demonstrated on three synthetic datasets and two trophic networks.

----

## [0] Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

**Authors**: *Pritam Sarkar, Ahmad Beirami, Ali Etemad*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a86d17b6cd70366d56ab48d2a05a4df1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a86d17b6cd70366d56ab48d2a05a4df1-Abstract-Conference.html)

**Abstract**:

Video self-supervised learning (VSSL) has made significant progress in recent years. However, the exact behavior and dynamics of these models under different forms of distribution shift are not yet known. In this paper, we comprehensively study the behavior of six popular self-supervised methods (v-SimCLR, v-MoCo, v-BYOL, v-SimSiam, v-DINO, v-MAE) in response to various forms of natural distribution shift, i.e., (i) context shift, (ii) viewpoint shift, (iii) actor shift, (iv) source shift, (v) generalizability to unknown classes (zero-shot), and (vi) open-set recognition. To perform this extensive study, we carefully craft a test bed consisting of 17 in-distribution and out-of-distribution benchmark pairs using available public datasets and a series of evaluation protocols to stress-test the different methods under the intended shifts. Our study uncovers a series of intriguing findings and interesting behaviors of VSSL methods. For instance, we observe that while video models generally struggle with context shifts, v-MAE and supervised learning exhibit more robustness. Moreover, our study shows that v-MAE is a strong temporal learner, whereas contrastive methods, v-SimCLR and v-MoCo, exhibit strong performances against viewpoint shifts. When studying the notion of open-set recognition, we notice a trade-off between closed-set and open-set recognition performance if the pretrained VSSL encoders are used without finetuning. We hope that our work will contribute to the development of robust video representation learning frameworks for various real-world scenarios. The project page and code are available at: https://pritamqu.github.io/OOD-VSSL.

----

## [0] M2SODAI: Multi-Modal Maritime Object Detection Dataset With RGB and Hyperspectral Image Sensors

**Authors**: *Jonggyu Jang, Sangwoo Oh, Youjin Kim, Dongmin Seo, Youngchol Choi, Hyun Jong Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8757b889350a3782b384a3ec0dfbae9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8757b889350a3782b384a3ec0dfbae9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Object detection in aerial images is a growing area of research, with maritime object detection being a particularly important task for reliable surveillance, monitoring, and active rescuing. Notwithstanding astonishing advances of computer visiontechnologies, detecting ships and floating matters in these images are challenging due to factors such as object distance. What makes it worse is pervasive sea surface effects such as sunlight reflection, wind, and waves. Hyperspectral image (HSI) sensors, providing more than 100 channels in wavelengths of visible and near-infrared, can extract intrinsic information of materials from a few pixels of HSIs.The advent of HSI sensors motivates us to leverage HSIs to circumvent false positives due to the sea surface effects.Unfortunately, there are few public HSI datasets due to the high cost and labor involved in collecting them, hindering object detection research based on HSIs. We have collected and annotated a new dataset called ``Multi-Modal Ship and flOating matter Detection in Aerial Images (M$^{2}$SODAI),'', which includes synchronized image pairs of RGB and HSI data, along with bounding box labels for nearly 6,000 instances per category. We also propose a new multi-modal extension of the feature pyramid network called DoubleFPN.Extensive experiments on our benchmark demonstrate that fusion of RGB and HSI data can enhance mAP, especially in the presence of the sea surface effects.

----

## [0] Projection-Free Methods for Solving Nonconvex-Concave Saddle Point Problems

**Authors**: *Morteza Boroun, Erfan Yazdandoost Hamedani, Afrooz Jalilzadeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a899a801fab59f14777fcc08842b6fc5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a899a801fab59f14777fcc08842b6fc5-Abstract-Conference.html)

**Abstract**:

In this paper, we investigate a class of constrained saddle point (SP) problems where the objective function is nonconvex-concave and smooth. This class of problems has wide applicability in machine learning, including robust multi-class classification and dictionary learning. Several projection-based primal-dual methods have been developed to tackle this problem; however, the availability of methods with projection-free oracles remains limited. To address this gap, we propose efficient single-loop projection-free methods reliant on first-order information. In particular, using regularization and nested approximation techniques, we propose a primal-dual conditional gradient method that solely employs linear minimization oracles to handle constraints. Assuming that the constraint set in the maximization is strongly convex, our method achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-6})$ iterations. When the projection onto the constraint set of maximization is easy to compute, we propose a one-sided projection-free method that achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-4})$ iterations. Moreover, we present improved iteration complexities of our methods under a strong concavity assumption. To the best of our knowledge, our proposed algorithms are among the first projection-free methods with convergence guarantees for solving nonconvex-concave SP problems.

----

## [0] Better Correlation and Robustness: A Distribution-Balanced Self-Supervised Learning Framework for Automatic Dialogue Evaluation

**Authors**: *Peiwen Yuan, Xinglin Wang, Jiayi Shi, Bin Sun, Yiwei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8b148559549ce33261e79b4400e0d77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8b148559549ce33261e79b4400e0d77-Abstract-Conference.html)

**Abstract**:

Turn-level dialogue evaluation models (TDEMs), using self-supervised learning (SSL) framework, have achieved state-of-the-art performance in open-domain dialogue evaluation. However, these models inevitably face two potential problems. First, they have low correlations with humans on medium coherence samples as the SSL framework often brings training data with unbalanced coherence distribution. Second, the SSL framework leads TDEM to nonuniform score distribution. There is a danger that the nonuniform score distribution will weaken the robustness of TDEM through our theoretical analysis. To tackle these problems, we propose Better Correlation and Robustness (BCR), a distribution-balanced self-supervised learning framework for TDEM. Given a dialogue dataset, BCR offers an effective training set reconstructing method to provide coherence-balanced training signals and further facilitate balanced evaluating abilities of TDEM. To get a uniform score distribution, a novel loss function is proposed, which can adjust adaptively according to the uniformity of score distribution estimated by kernel density estimation. Comprehensive experiments on 17 benchmark datasets show that vanilla BERT-base using BCR outperforms SOTA methods significantly by 11.3% on average. BCR also demonstrates strong generalization ability as it can lead multiple SOTA methods to attain better correlation and robustness.

----

## [0] Learning Topology-Agnostic EEG Representations with Geometry-Aware Modeling

**Authors**: *Ke Yi, Yansen Wang, Kan Ren, Dongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8c893712cb7858e49631fb03c941f8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8c893712cb7858e49631fb03c941f8d-Abstract-Conference.html)

**Abstract**:

Large-scale pre-training has shown great potential to enhance models on downstream tasks in vision and language. Developing similar techniques for scalp electroencephalogram (EEG) is suitable since unlabelled data is plentiful. Meanwhile, various sampling channel selections and inherent structural and spatial information bring challenges and avenues to improve existing pre-training strategies further. In order to break boundaries between different EEG resources and facilitate cross-dataset EEG pre-training, we propose to map all kinds of channel selections to a unified topology. We further introduce MMM, a pre-training framework with Multi-dimensional position encoding, Multi-level channel hierarchy, and Multi-stage pre-training strategy built on the unified topology to obtain topology-agnostic representations. Experiments demonstrate that our approach yields impressive improvements over previous state-of-the-art techniques on emotional recognition benchmark datasets.

----

## [0] Correlation Aware Sparsified Mean Estimation Using Random Projection

**Authors**: *Shuli Jiang, Pranay Sharma, Gauri Joshi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8e21789027e92739f89df92cc172bcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8e21789027e92739f89df92cc172bcf-Abstract-Conference.html)

**Abstract**:

We study the problem of communication-efficient distributed vector mean estimation, which is a commonly used subroutine in distributed optimization and Federated Learning (FL). Rand-$k$ sparsification is a commonly used technique to reduce communication cost, where each client sends $k < d$ of its coordinates to the server. However, Rand-$k$ is agnostic to any correlations, that might exist between clients in practical scenarios. The recently proposed Rand-$k$-Spatial estimator leverages the cross-client correlation information at the server to improve Rand-$k$'s performance. Yet, the performance of Rand-$k$-Spatial is suboptimal, and improving mean estimation is key to a faster convergence in distributed optimization. We propose the Rand-Proj-Spatial estimator with a more flexible encoding-decoding procedure, which generalizes the encoding of Rand-$k$ by projecting the client vectors to a random $k$-dimensional subspace. We utilize Subsampled Randomized Hadamard Transform (SRHT) as the projection matrix, and show that Rand-Proj-Spatial with SRHT outperforms Rand-$k$-Spatial, using the correlation information more efficiently. Furthermore, we propose an approach to incorporate varying degrees of correlation, and suggest a practical variant of Rand-Proj-Spatial when the correlation information is not available to the server. Finally, experiments on real-world distributed optimization tasks showcase the superior performance of Rand-Proj-Spatial compared to Rand-$k$-Spatial and other more sophisticated sparsification techniques.

----

## [0] Accelerating Value Iteration with Anchoring

**Authors**: *Jongmin Lee, Ernest Ryu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html)

**Abstract**:

Value Iteration (VI) is foundational to the theory and practice of modern reinforcement learning, and it is known to converge at a $\mathcal{O}(\gamma^k)$-rate. Surprisingly, however, the optimal rate for the VI setup was not known, and finding a general acceleration mechanism has been an open problem. In this paper, we present the first accelerated VI for both the Bellman consistency and optimality operators. Our method, called Anc-VI, is based on an \emph{anchoring} mechanism (distinct from Nesterov's acceleration), and it reduces the Bellman error faster than standard VI. In particular, Anc-VI exhibits a $\mathcal{O}(1/k)$-rate for $\gamma\approx 1$ or even $\gamma=1$, while standard VI has rate $\mathcal{O}(1)$ for $\gamma\ge 1-1/k$, where $k$ is the iteration count. We also provide a complexity lower bound matching the upper bound up to a constant factor of $4$, thereby establishing optimality of the accelerated rate of Anc-VI. Finally, we show that the anchoring mechanism provides the same benefit in the approximate VI and Gauss--Seidel VI setups as well.

----

## [0] Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion

**Authors**: *Yang Liu, Feng Wang, Naiyan Wang, Zhaoxiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f7f12b29d9b8c227785f6b529f63b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f7f12b29d9b8c227785f6b529f63b7-Abstract-Conference.html)

**Abstract**:

Radar is ubiquitous in autonomous driving systems due to its low cost and good adaptability to bad weather. Nevertheless, the radar detection performance is usually inferior because its point cloud is sparse and not accurate due to the poor azimuth and elevation resolution. Moreover, point cloud generation algorithms already drop weak signals to reduce the false targets which may be suboptimal for the use of deep fusion. In this paper, we propose a novel method named EchoFusion to skip the existing radar signal processing pipeline and then incorporate the radar raw data with other sensors. Specifically, we first generate the Bird's Eye View (BEV) queries and then take corresponding spectrum features from radar to fuse with other sensors. By this approach, our method could utilize both rich and lossless distance and speed clues from radar echoes and rich semantic clues from images, making our method surpass all existing methods on the RADIal dataset, and approach the performance of LiDAR. The code will be released on https://github.com/tusen-ai/EchoFusion.

----

## [0] D4: Improving LLM Pretraining via Document De-Duplication and Diversification

**Authors**: *Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Over recent years, an increasing amount of compute and data has been poured into training large language models (LLMs), usually by doing one-pass learning on as many tokens as possible randomly selected from large-scale web corpora. While training on ever-larger portions of the internet leads to consistent performance improvements, the size of these improvements diminishes with scale, and there has been little work exploring the effect of data selection on pre-training and downstream performance beyond simple de-duplication methods such as MinHash. Here, we show that careful data selection (on top of de-duplicated data) via pre-trained model embeddings can speed up training (20% efficiency gains) and improves average downstream accuracy on 16 NLP tasks (up to 2%) at the 6.7B model scale. Furthermore, we show that repeating data intelligently consistently outperforms baseline training (while repeating random data performs worse than baseline training). Our results indicate that clever data selection can significantly improve LLM pre-training, calls into question the common practice of training for a single epoch on as much data as possible, and demonstrates a path to keep improving our models past the limits of randomly sampling web data.

----

## [0] Effective Bayesian Heteroscedastic Regression with Deep Neural Networks

**Authors**: *Alexander Immer, Emanuele Palumbo, Alexander Marx, Julia E. Vogt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html)

**Abstract**:

Flexibly quantifying both irreducible aleatoric and model-dependent epistemic uncertainties plays an important role for complex regression problems. While deep neural networks in principle can provide this flexibility and learn heteroscedastic aleatoric uncertainties through non-linear functions, recent works highlight that maximizing the log likelihood objective parameterized by mean and variance can lead to compromised mean fits since the gradient are scaled by the predictive variance, and propose adjustments in line with this premise. We instead propose to use the natural parametrization of the Gaussian, which has been shown to be more stable for heteroscedastic regression based on non-linear feature maps and Gaussian processes. Further, we emphasize the significance of principled regularization of the network parameters and prediction. We therefore propose an efficient Laplace approximation for heteroscedastic neural networks that allows automatic regularization through empirical Bayes and provides epistemic uncertainties, both of which improve generalization.We showcase on a range of regression problems—including a new heteroscedastic image regression benchmark—that our methods are scalable, improve over previous approaches for heteroscedastic regression, and provide epistemic uncertainty without requiring hyperparameter tuning.

----

## [0] Multi-task learning with summary statistics

**Authors**: *Parker Knight, Rui Duan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a924b7178e5975dfed1de235f0b72973-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a924b7178e5975dfed1de235f0b72973-Abstract-Conference.html)

**Abstract**:

Multi-task learning has emerged as a powerful machine learning paradigm for integrating data from multiple sources, leveraging similarities between tasks to improve overall model performance. However, the application of multi-task learning to real-world settings is hindered by data-sharing constraints, especially in healthcare settings. To address this challenge, we propose a flexible multi-task learning framework utilizing summary statistics from various sources. Additionally, we present an adaptive parameter selection approach based on a variant of Lepski's method, allowing for data-driven tuning parameter selection when only summary statistics are accessible. Our systematic non-asymptotic analysis characterizes the performance of the proposed methods under various regimes of the source datasets' sample complexity and overlap.  We demonstrate our theoretical findings and the performance of the method  through extensive simulations. This work offers a more flexible tool for training related models across various domains, with  practical implications in genetic risk prediction and many other fields.

----

## [0] Estimating Noise Correlations Across Continuous Conditions With Wishart Processes

**Authors**: *Amin Nejatbakhsh, Isabel Garon, Alex Williams*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html)

**Abstract**:

The signaling capacity of a neural population depends on the scale and orientation of its covariance across trials. Estimating this "noise" covariance is challenging and is thought to require a large number of stereotyped trials. New approaches are therefore needed to interrogate the structure of neural noise across rich, naturalistic behaviors and sensory experiences, with few trials per condition. Here, we exploit the fact that conditions are smoothly parameterized in many experiments and leverage Wishart process models to pool statistical power from trials in neighboring conditions. We demonstrate that these models perform favorably on experimental data from the mouse visual cortex and monkey motor cortex relative to standard covariance estimators. Moreover, they produce smooth estimates of covariance as a function of stimulus parameters, enabling estimates of noise correlations in entirely unseen conditions as well as continuous estimates of Fisher information—a commonly used measure of signal fidelity. Together, our results suggest that Wishart processes are broadly applicable tools for quantification and uncertainty estimation of noise correlations in trial-limited regimes, paving the way toward understanding the role of noise in complex neural computations and behavior.

----

## [0] Toward Understanding Generative Data Augmentation

**Authors**: *Chenyu Zheng, Guoqiang Wu, Chongxuan Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html)

**Abstract**:

Generative data augmentation, which scales datasets by obtaining fake labeled examples from a trained conditional generative model, boosts classification performance in various learning tasks including (semi-)supervised learning, few-shot learning, and adversarially robust learning. However, little work has theoretically investigated the effect of generative data augmentation. To fill this gap, we establish a general stability bound in this not independently and identicallydistributed (non-i.i.d.) setting, where the learned distribution is dependent on the original train set and generally not the same as the true distribution. Our theoretical result includes the divergence between the learned distribution and the true distribution. It shows that generative data augmentation can enjoy a faster learning rate when the order of divergence term is $o(\max\left( \log(m)\beta_m, 1 / \sqrt{m})\right)$, where $m$ is the train set size and $\beta_m$ is the corresponding stability constant. We further specify the learning setup to the Gaussian mixture model and generative adversarial nets. We prove that in both cases, though generative data augmentation does not enjoy a faster learning rate, it can improve the learning guarantees at a constant level when the train set is small, which is significant when the awful overfitting occurs. Simulation results on the Gaussian mixture model and empirical results on generative adversarial nets support our theoretical conclusions.

----

## [0] TOA: Task-oriented Active VQA

**Authors**: *Xiaoying Xing, Mingfu Liang, Ying Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a95cc4f370bcc418e7b57d6512e28f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a95cc4f370bcc418e7b57d6512e28f52-Abstract-Conference.html)

**Abstract**:

Knowledge-based visual question answering (VQA) requires external knowledge to answer the question about an image. Early methods explicitly retrieve knowledge from external knowledge bases, which often introduce noisy information. Recently large language models like GPT-3 have shown encouraging performance as implicit knowledge source and revealed planning abilities. However, current large language models can not effectively understand image inputs, thus it remains an open problem to extract the image information and input to large language models. Prior works have used image captioning and object descriptions to represent the image. However, they may either drop the essential visual information to answer the question correctly or involve irrelevant objects to the task-of-interest. To address this problem, we propose to let large language models make an initial hypothesis according to their knowledge, then actively collect the visual evidence required to verify the hypothesis. In this way, the model can attend to the essential visual information in a task-oriented manner. We leverage several vision modules from the perspectives of spatial attention (i.e., Where to look) and attribute attention (i.e., What to look), which is similar to human cognition. The experiments show that our proposed method outperforms the baselines on open-ended knowledge-based VQA datasets and presents clear reasoning procedure with better interpretability.

----

## [0] Universal Gradient Descent Ascent Method for Nonconvex-Nonconcave Minimax Optimization

**Authors**: *Taoli Zheng, Linglingzhi Zhu, Anthony Man-Cho So, Jose H. Blanchet, Jiajin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a961dea42c23c3c0d01b79918701fb6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a961dea42c23c3c0d01b79918701fb6e-Abstract-Conference.html)

**Abstract**:

Nonconvex-nonconcave minimax optimization has received intense attention over the last decade due to its broad applications in machine learning. Most existing algorithms rely on one-sided information, such as the convexity (resp. concavity) of the primal (resp. dual) functions, or other specific structures, such as the Polyak-Łojasiewicz (PŁ) and Kurdyka-Łojasiewicz (KŁ) conditions. However, verifying these regularity conditions is challenging in practice. To meet this challenge, we propose a novel universally applicable single-loop algorithm, the doubly smoothed gradient descent ascent method (DS-GDA), which naturally balances the primal and dual updates. That is, DS-GDA with the same hyperparameters is able to uniformly solve nonconvex-concave, convex-nonconcave, and nonconvex-nonconcave problems with one-sided KŁ properties, achieving convergence with $\mathcal{O}(\epsilon^{-4})$ complexity. Sharper (even optimal) iteration complexity can be obtained when the KŁ exponent is known. Specifically, under the one-sided KŁ condition with exponent $\theta\in(0,1)$, DS-GDA converges with an iteration complexity of $\mathcal{O}(\epsilon^{-2\max\\{2\theta,1\\}})$. They all match the corresponding best results in the literature. Moreover, we show that DS-GDA is practically applicable to general nonconvex-nonconcave problems even without any regularity conditions, such as the PŁ condition, KŁ condition, or weak Minty variational inequalities condition.  For various challenging nonconvex-nonconcave examples in the literature, including *Forsaken*, *Bilinearly-coupled minimax*, *Sixth-order polynomial*, and *PolarGame*, the proposed DS-GDA can all get rid of limit cycles. To the best of our knowledge, this is the first first-order algorithm to achieve convergence on all of these formidable problems.

----

## [0] On Evaluating Adversarial Robustness of Large Vision-Language Models

**Authors**: *Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, Min Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html)

**Abstract**:

Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, multimodal generation exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulnerable modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where adversaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a surprisingly high success rate for generating targeted responses. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: https://yunqing-me.github.io/AttackVLM/.

----

## [0] Generator Born from Classifier

**Authors**: *Runpeng Yu, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97f0218b49bc17ea3f121a0e724f028-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97f0218b49bc17ea3f121a0e724f028-Abstract-Conference.html)

**Abstract**:

In this paper, we make a bold attempt toward an ambitious task: given a pre-trained classifier, we aim to reconstruct an image generator, without relying on any data samples. From a black-box perspective, this challenge seems intractable, since it inevitably involves identifying the inverse function for a classifier, which is, by nature, an information extraction process. As such, we resort to leveraging the knowledge encapsulated within the parameters of the neural network. Grounded on the theory of Maximum-Margin Bias of gradient descent, we propose a novel learning paradigm, in which the generator is trained to ensure that the convergence conditions of the network parameters are satisfied over the generated distribution of the samples. Empirical validation from various image generation tasks substantiates the efficacy of our strategy.

----

## [0] Scattering Vision Transformer: Spectral Mixing Matters

**Authors**: *Badri N. Patro, Vijay Srinivas Agneeswaran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97f8072e51a785434b2da3e9cbf5aae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97f8072e51a785434b2da3e9cbf5aae-Abstract-Conference.html)

**Abstract**:

Vision transformers have gained significant attention and achieved state-of-the-art performance in various computer vision tasks, including image classification, instance segmentation, and object detection. However, challenges remain in addressing attention complexity and effectively capturing fine-grained information within images. Existing solutions often resort to down-sampling operations, such as pooling, to reduce computational cost. Unfortunately, such operations are non-invertible and can result in information loss. In this paper, we present a novel approach called Scattering Vision Transformer (SVT) to tackle these challenges. SVT incorporates a spectrally scattering network that enables the capture of intricate image details. SVT overcomes the invertibility issue associated with down-sampling operations by separating low-frequency and high-frequency components. Furthermore, SVT introduces a unique spectral gating network utilizing Einstein multiplication for token and channel mixing, effectively reducing complexity. We show that SVT achieves state-of-the-art performance on the ImageNet dataset with a significant reduction in a number of parameters and FLOPS. SVT shows 2\% improvement over LiTv2 and iFormer. SVT-H-S reaches 84.2\% top-1 accuracy, while SVT-H-B reaches 85.2\% (state-of-art for base versions) and SVT-H-L reaches 85.7\% (again state-of-art for large versions). SVT also shows comparable results in other vision tasks such as instance segmentation. SVT also outperforms other transformers in transfer learning on standard datasets such as CIFAR10, CIFAR100, Oxford Flower, and Stanford Car datasets.  The project page is available on this webpage.\url{https://badripatro.github.io/svt/}.

----

## [0] Task-aware world model learning with meta weighting via bi-level optimization

**Authors**: *Huining Yuan, Hongkun Dou, Xingyu Jiang, Yue Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html)

**Abstract**:

Aligning the world model with the environment for the agentâ€™s specific task is crucial in model-based reinforcement learning. While value-equivalent models may achieve better task awareness than maximum-likelihood models, they sacrifice a large amount of semantic information and face implementation issues. To combine the benefits of both types of models, we propose Task-aware Environment Modeling Pipeline with bi-level Optimization (TEMPO), a bi-level model learning framework that introduces an additional level of optimization on top of a maximum-likelihood model by incorporating a meta weighter network that weights each training sample. The meta weighter in the upper level learns to generate novel sample weights by minimizing a proposed task-aware model loss. The model in the lower level focuses on important samples while maintaining rich semantic information in state representations. We evaluate TEMPO on a variety of continuous and discrete control tasks from the DeepMind Control Suite and Atari video games. Our results demonstrate that TEMPO achieves state-of-the-art performance regarding asymptotic performance, training stability, and convergence speed.

----

## [0] LEPARD: Learning Explicit Part Discovery for 3D Articulated Shape Reconstruction

**Authors**: *Di Liu, Anastasis Stathopoulos, Qilong Zhangli, Yunhe Gao, Dimitris N. Metaxas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a99f50fb024a56d15f057a1830ed0a00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a99f50fb024a56d15f057a1830ed0a00-Abstract-Conference.html)

**Abstract**:

Reconstructing the 3D articulated shape of an animal from a single in-the-wild image is a challenging task. We propose LEPARD, a learning-based framework that discovers semantically meaningful 3D parts and reconstructs 3D shapes in a part-based manner. This is advantageous as 3D parts are robust to pose variations due to articulations and their shape is typically simpler than the overall shape of the object. In our framework, the parts are explicitly represented as parameterized primitive surfaces with global and local deformations in 3D that deform to match the image evidence. We propose a kinematics-inspired optimization to guide each transformation of the primitive deformation given 2D evidence. Similar to recent approaches, LEPARD is only trained using off-the-shelf deep features from DINO and does not require any form of 2D or 3D annotations. Experiments on 3D animal shape reconstruction, demonstrate significant improvement over existing alternatives in terms of both the overall reconstruction performance as well as the ability to discover semantically meaningful and consistent parts.

----

## [0] Curriculum Learning With Infant Egocentric Videos

**Authors**: *Saber Sheybani, Himanshu Hansaria, Justin Wood, Linda B. Smith, Zoran Tiganj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html)

**Abstract**:

Infants possess a remarkable ability to rapidly learn and process visual inputs. As an infant's mobility increases, so does the variety and dynamics of their visual inputs. Is this change in the properties of the visual inputs beneficial or even critical for the proper development of the visual system? To address this question, we used video recordings from infants wearing head-mounted cameras to train a variety of self-supervised learning models. Critically, we separated the infant data by age group and evaluated the importance of training with a curriculum aligned with developmental order. We found that initiating learning with the data from the youngest age group provided the strongest learning signal and led to the best learning outcomes in terms of downstream task performance. We then showed that the benefits of the data from the youngest age group are due to the slowness and simplicity of the visual experience. The results provide strong empirical evidence for the importance of the properties of the early infant experience and developmental progression in training. More broadly, our approach and findings take a noteworthy step towards reverse engineering the learning mechanisms in newborn brains using image-computable models from artificial intelligence.

----

## [0] MarioGPT: Open-Ended Text2Level Generation through Large Language Models

**Authors**: *Shyam Sudhakaran, Miguel González Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, Sebastian Risi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9bbeb2858dfbdbd4c19814e5d80ec60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9bbeb2858dfbdbd4c19814e5d80ec60-Abstract-Conference.html)

**Abstract**:

Procedural Content Generation (PCG) is a technique to generate complex and diverse environments in an automated way. However, while generating content with PCG methods is often straightforward, generating meaningful content that reflects specific  intentions and constraints remains challenging. Furthermore, many PCG algorithms lack the ability to generate content in an open-ended manner. Recently, Large Language Models (LLMs) have  shown to be incredibly effective in many diverse domains. These trained LLMs can be fine-tuned, re-using information and accelerating training for new tasks. Here, we introduce MarioGPT, a fine-tuned GPT2 model trained to generate tile-based game levels, in our case Super Mario Bros levels. MarioGPT can not only generate diverse levels, but can be  text-prompted for controllable level generation, addressing one of the key challenges of current PCG techniques.  As far as we know, MarioGPT is the first text-to-level model and combined with novelty search it enables the  generation of diverse levels with varying play-style dynamics (i.e. player paths) and the open-ended discovery of an increasingly diverse range of content. Code available at https://github.com/shyamsn97/mario-gpt.

----

## [0] Is Learning in Games Good for the Learners?

**Authors**: *William Brown, Jon Schneider, Kiran Vodrahalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9ea92ef18aae17627d133534209e640-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9ea92ef18aae17627d133534209e640-Abstract-Conference.html)

**Abstract**:

We consider a number of questions related to tradeoffs between reward and regret in repeated gameplay between two agents. To facilitate this,  we introduce a notion of generalized equilibrium which allows for asymmetric regret constraints, and yields polytopes of feasible values for each agent and pair of regret constraints, where we show that any such equilibrium is reachable by a pair of algorithms which maintain their regret guarantees against arbitrary opponents. As a central example, we highlight the case one agent is no-swap and the other's regret is unconstrained. We show that this captures an extension of Stackelberg equilibria with a matching optimal value, and that there exists a wide class of games where a player can significantly increase their utility by deviating from a no-swap-regret algorithm against a no-swap learner (in fact, almost any game without pure Nash equilibria is of this form). Additionally, we make use of generalized equilibria to consider tradeoffs in terms of the opponent's algorithm choice. We give a tight characterization for the maximal reward obtainable against some no-regret learner, yet we also show a class of games in which this is bounded away from the value obtainable against the class of common "mean-based" no-regret algorithms. Finally, we consider the question of learning reward-optimal strategies via repeated play with a no-regret agent when the game is initially unknown. Again we show tradeoffs depending on the opponent's learning algorithm: the Stackelberg strategy is learnable in exponential time with any no-regret agent (and in polynomial time with any no-adaptive-regret agent) for any game where it is learnable via queries, and there are games where it is learnable in polynomial time against any no-swap-regret agent but requires exponential time against a mean-based no-regret agent.

----

## [0] The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit

**Authors**: *Lorenzo Noci, Chuning Li, Mufan Bill Li, Bobby He, Thomas Hofmann, Chris J. Maddison, Dan Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html)

**Abstract**:

In deep learning theory, the covariance matrix of the representations serves as aproxy to examine the network’s trainability. Motivated by the success of Transform-ers, we study the covariance matrix of a modified Softmax-based attention modelwith skip connections in the proportional limit of infinite-depth-and-width. Weshow that at initialization the limiting distribution can be described by a stochasticdifferential equation (SDE) indexed by the depth-to-width ratio. To achieve awell-defined stochastic limit, the Transformer’s attention mechanism is modifiedby centering the Softmax output at identity, and scaling the Softmax logits by awidth-dependent temperature parameter. We examine the stability of the networkthrough the corresponding SDE, showing how the scale of both the drift and diffu-sion can be elegantly controlled with the aid of residual connections. The existenceof a stable SDE implies that the covariance structure is well-behaved, even for verylarge depth and width, thus preventing the notorious issues of rank degeneracyin deep attention models. Finally, we show, through simulations, that the SDEprovides a surprisingly good description of the corresponding finite-size model.We coin the name shaped Transformer for these architectural modifications.

----

## [0] Decompose Novel into Known: Part Concept Learning For 3D Novel Class Discovery

**Authors**: *Tingyu Weng, Jun Xiao, Haiyong Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa31eee8f2351176ddd4d14646d4a950-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa31eee8f2351176ddd4d14646d4a950-Abstract-Conference.html)

**Abstract**:

In this work, we address 3D novel class discovery (NCD) that discovers novel classes from an unlabeled dataset by leveraging the knowledge of disjoint known classes. The key challenge of 3D NCD is that learned features by known class recognition are heavily biased and hinder generalization to novel classes. Since geometric parts are more generalizable across different classes, we propose to decompose novel into known parts, coined DNIK, to mitigate the above problems. DNIK learns a part concept bank encoding rich part geometric patterns from known classes so that novel 3D shapes can be represented as part concept compositions to facilitate cross-category generalization. Moreover, we formulate three constraints on part concepts to ensure diverse part concepts without collapsing. A part relation encoding module (PRE) is also developed to leverage part-wise spatial relations for better recognition. We construct three 3D NCD tasks for evaluation and extensive experiments show that our method achieves significantly superior results than SOTA baselines (+11.7%, +14.1%, and +16.3% improvements on average for three tasks, respectively). Code and data will be released.

----

## [0] Long Sequence Hopfield Memory

**Authors**: *Hamza Tahir Chaudhry, Jacob A. Zavatone-Veth, Dmitry Krotov, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa32ebcdd2ce1bed4ef7f456fc8fa5c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa32ebcdd2ce1bed4ef7f456fc8fa5c1-Abstract-Conference.html)

**Abstract**:

Sequence memory is an essential attribute of natural and artificial intelligence that enables agents to encode, store, and retrieve complex sequences of stimuli and actions. Computational models of sequence memory have been proposed where recurrent Hopfield-like neural networks are trained with temporally asymmetric Hebbian rules. However, these networks suffer from limited sequence capacity (maximal length of the stored sequence) due to interference between the memories. Inspired by recent work on Dense Associative Memories, we expand the sequence capacity of these models by introducing a nonlinear interaction term, enhancing separation between the patterns. We derive novel scaling laws for sequence capacity with respect to network size, significantly outperforming existing scaling laws for models based on traditional Hopfield networks, and verify these theoretical results with numerical simulation. Moreover, we introduce a generalized pseudoinverse rule to recall sequences of highly correlated patterns. Finally, we extend this model to store sequences with variable timing between states' transitions and describe a biologically-plausible implementation, with connections to motor neuroscience.

----

## [0] Provably Safe Reinforcement Learning with Step-wise Violation Constraints

**Authors**: *Nuoya Xiong, Yihan Du, Longbo Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa3e67220ca4cd50010165c950fc8056-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa3e67220ca4cd50010165c950fc8056-Abstract-Conference.html)

**Abstract**:

We investigate a novel safe reinforcement learning problem with step-wise violation constraints. Our problem differs from existing works in that we focus on stricter step-wise violation constraints and do not assume the existence of safe actions, making our formulation more suitable for safety-critical applications that need to ensure safety in all decision steps but may not always possess safe actions, e.g., robot control and autonomous driving.We propose an efficient algorithm SUCBVI, which guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ or gap-dependent $\widetilde{\mathcal{O}}(S/\mathcal{C}_{\mathrm{gap}} + S^2AH^2)$ step-wise violation and $\widetilde{\mathcal{O}}(\sqrt{H^3SAT})$ regret. Lower bounds are provided to validate the optimality in both violation and  regret performance with respect to the number of states $S$ and the total number of steps $T$. Moreover, we further study an innovative safe reward-free exploration problem with step-wise violation constraints. For this problem, we design algorithm SRF-UCRL to find a near-optimal safe policy, which achieves nearly state-of-the-art  sample complexity $\widetilde{\mathcal{O}}((\frac{S^2AH^2}{\varepsilon}+\frac{H^4SA}{\varepsilon^2})(\log(\frac{1}{\delta})+S))$, and guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ violation during exploration.  Experimental results demonstrate the  superiority of our algorithms in safety performance and corroborate our theoretical results.

----

## [0] Human spatiotemporal pattern learning as probabilistic program synthesis

**Authors**: *Tracey Mills, Josh Tenenbaum, Samuel Cheyette*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5c083f9d387c49514eb5c4dc2dc16b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5c083f9d387c49514eb5c4dc2dc16b-Abstract-Conference.html)

**Abstract**:

People are adept at learning a wide variety of structured patterns from small amounts of data, presenting a conundrum from the standpoint of the bias-variance tradeoff: what kinds of representations and algorithms support the joint flexibility and data-paucity of human learning? One possibility is that people "learn by programming": inducing probabilistic models to fit observed data. Here, we experimentally test human learning in the domain of structured 2-dimensional patterns, using a task in which participants repeatedly predicted where a dot would move based on its previous trajectory. We evaluate human performance against standard parametric and non-parametric time-series models, as well as two Bayesian program synthesis models whose hypotheses vary in their degree of structure: a compositional Gaussian Process model and a structured "Language of Thought" (LoT) model. We find that signatures of human pattern learning are best explained by the LoT model, supporting the idea that the flexibility and data-efficiency of human structure learning can be understood as probabilistic inference over an expressive space of programs.

----

## [0] Fair Allocation of Indivisible Chores: Beyond Additive Costs

**Authors**: *Bo Li, Fangxiao Wang, Yu Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5d22c77b380e2261332bb641b3c2e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5d22c77b380e2261332bb641b3c2e3-Abstract-Conference.html)

**Abstract**:

We study the maximin share (MMS) fair allocation of $m$ indivisible tasks to $n$ agents who have costs for completing the assigned tasks.It is known that exact MMS fairness cannot be guaranteed, and so far the best-known approximation for additive cost functions is $\frac{13}{11}$ by Huang and Segal-Halevi [EC, 2023]; however, beyond additivity, very little is known. In this work, we first prove that no algorithm can ensure better than $\min\{n,\frac{\log m}{\log \log m}\}$-approximation if the cost functions are submodular. This result also shows a sharp contrast with the allocation of goods where constant approximations exist as shown by Barman and Krishnamurthy [TEAC, 2020] and Ghodsi et al. [AIJ, 2022]. We then prove that for subadditive costs, there always exists an allocation that is $\min\{n,\lceil\log m\rceil\}$-approximation, and thus the approximation ratio is asymptotically tight.Besides multiplicative approximation, we also consider the ordinal relaxation, 1-out-of-$d$ MMS, which was recently proposed by Hosseini et al. [JAIR and AAMAS, 2022]. Our impossibility result implies that for any $d\ge 2$, a 1-out-of-$d$ MMS allocation may not exist.Due to these hardness results for general subadditive costs, we turn to studying two specific subadditive costs, namely, bin packing and job scheduling. For both settings, we show that constant approximate allocations exist for both multiplicative and ordinal relaxations of MMS.

----

## [0] Robust Second-Order Nonconvex Optimization and Its Application to Low Rank Matrix Sensing

**Authors**: *Shuyao Li, Yu Cheng, Ilias Diakonikolas, Jelena Diakonikolas, Rong Ge, Stephen J. Wright*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5f224975a67914067519faddeacba3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5f224975a67914067519faddeacba3-Abstract-Conference.html)

**Abstract**:

Finding an approximate second-order stationary point (SOSP) is a well-studied and fundamental problem in stochastic nonconvex optimization with many applications in machine learning.However, this problem is poorly understood in the presence of outliers, limiting the use of existing nonconvex algorithms in adversarial settings.In this paper, we study the problem of finding SOSPs in the strong contamination model, where a constant fraction of datapoints are arbitrarily corrupted.We introduce a general framework for efficiently finding an approximate SOSP with \emph{dimension-independent} accuracy guarantees, using $\widetilde{O}({D^2}/{\epsilon})$ samples where $D$ is the ambient dimension and $\epsilon$ is the fraction of corrupted datapoints.As a concrete application of our framework, we apply it to the problem of low rank matrix sensing, developing efficient and provably robust algorithms that can tolerate corruptions in both the sensing matrices and the measurements.In addition, we establish a Statistical Query lower bound providing evidence that the quadratic dependence on $D$ in the sample complexity is necessary for computationally efficient algorithms.

----

## [0] Incentivized Communication for Federated Bandits

**Authors**: *Zhepei Wei, Chuanhao Li, Haifeng Xu, Hongning Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa61d142c0081a8259a6372a3bb0af2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa61d142c0081a8259a6372a3bb0af2b-Abstract-Conference.html)

**Abstract**:

Most existing works on federated bandits take it for granted that all clients are altruistic about sharing their data with the server for the collective good whenever needed. Despite their compelling theoretical guarantee on performance and communication efficiency, this assumption is overly idealistic and oftentimes violated in practice, especially when the algorithm is operated over self-interested clients, who are reluctant to share data without explicit benefits. Negligence of such self-interested behaviors can significantly affect the learning efficiency and even the practical operability of federated bandit learning. In light of this, we aim to spark new insights into this under-explored research area by formally introducing an incentivized communication problem for federated bandits, where the server shall motivate clients to share data by providing incentives. Without loss of generality, we instantiate this bandit problem with the contextual linear setting and propose the first incentivized communication protocol, namely, Inc-FedUCB, that achieves near-optimal regret with provable communication and incentive cost guarantees. Extensive empirical experiments on both synthetic and real-world datasets further validate the effectiveness of the proposed method across various environments.

----

## [0] Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand

**Authors**: *Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, Bo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa6287ca31ae1474ea802342d0c8ba63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa6287ca31ae1474ea802342d0c8ba63-Abstract-Conference.html)

**Abstract**:

The prosperity of deep neural networks (DNNs) is largely benefited from open-source datasets, based on which users can evaluate and improve their methods. In this paper, we revisit backdoor-based dataset ownership verification (DOV), which is currently the only feasible approach to protect the copyright of open-source datasets. We reveal that these methods are fundamentally harmful given that they could introduce malicious misclassification behaviors to watermarked DNNs by the adversaries. In this paper, we design DOV from another perspective by making watermarked models (trained on the protected dataset) correctly classify some `hard' samples that will be misclassified by the benign model. Our method is inspired by the generalization property of DNNs, where we find a \emph{hardly-generalized domain} for the original dataset (as its \emph{domain watermark}). It can be easily learned with the protected dataset containing modified samples. Specifically, we formulate the domain generation as a bi-level optimization and propose to optimize a set of visually-indistinguishable clean-label modified data with similar effects to domain-watermarked samples from the hardly-generalized domain to ensure watermark stealthiness. We also design a hypothesis-test-guided ownership verification via our domain watermark and provide the theoretical analyses of our method. Extensive experiments on three benchmark datasets are conducted, which verify the effectiveness of our method and its resistance to potential adaptive methods.

----

## [0] DISCO-10M: A Large-Scale Music Dataset

**Authors**: *Luca A. Lanzendörfer, Florian Grötschla, Emil Funke, Roger Wattenhofer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa7ef4c0f4aaabf376088a1a74e09d4c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa7ef4c0f4aaabf376088a1a74e09d4c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Music datasets play a crucial role in advancing research in machine learning for music. However, existing music datasets suffer from limited size, accessibility, and lack of audio resources. To address these shortcomings, we present DISCO-10M, a novel and extensive music dataset that surpasses the largest previously available music dataset by an order of magnitude. To ensure high-quality data, we implement a multi-stage filtering process. This process incorporates similarities based on textual descriptions and audio embeddings. Moreover, we provide precomputed CLAP embeddings alongside DISCO-10M, facilitating direct application on various downstream tasks. These embeddings enable efficient exploration of machine learning applications on the provided data. With DISCO-10M, we aim to democratize and facilitate new research to help advance the development of novel machine learning models for music: https://huggingface.co/DISCOX

----

## [0] Guide Your Agent with Adaptive Multimodal Rewards

**Authors**: *Changyeon Kim, Younggyo Seo, Hao Liu, Lisa Lee, Jinwoo Shin, Honglak Lee, Kimin Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html)

**Abstract**:

Developing an agent capable of adapting to unseen environments remains a difficult challenge in imitation learning. This work presents Adaptive Return-conditioned Policy (ARP), an efficient framework designed to enhance the agent's generalization ability using natural language task descriptions and pre-trained multimodal encoders. Our key idea is to calculate a similarity between visual observations and natural language instructions in the pre-trained multimodal embedding space (such as CLIP) and use it as a reward signal. We then train a return-conditioned policy using expert demonstrations labeled with multimodal rewards. Because the multimodal rewards provide adaptive signals at each timestep, our ARP effectively mitigates the goal misgeneralization. This results in superior generalization performances even when faced with unseen text instructions, compared to existing text-conditioned policies. To improve the quality of rewards, we also introduce a fine-tuning method for pre-trained multimodal encoders, further enhancing the performance. Video demonstrations and source code are available on the project website: \url{https://sites.google.com/view/2023arp}.

----

## [0] Fine-Grained Theoretical Analysis of Federated Zeroth-Order Optimization

**Authors**: *Jun Chen, Hong Chen, Bin Gu, Hao Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aaa973f65b98c96e5f850d706464a3c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aaa973f65b98c96e5f850d706464a3c4-Abstract-Conference.html)

**Abstract**:

Federated zeroth-order optimization (FedZO) algorithm enjoys the advantages of both zeroth-order optimization and federated learning, and has shown exceptional performance on black-box attack and softmax regression tasks. However, there is no generalization analysis for FedZO, and its analysis on computing convergence rate is slower than the corresponding first-order optimization setting. This paper aims to establish systematic theoretical assessments of FedZO by developing the analysis technique of on-average model stability. We establish the first generalization error bound of FedZO under the Lipschitz continuity and smoothness conditions. Then, refined generalization and optimization bounds are provided by replacing bounded gradient with heavy-tailed gradient noise and utilizing the second-order Taylor expansion for gradient approximation. With the help of a new error decomposition strategy, our theoretical analysis is also extended to the asynchronous case. For FedZO, our fine-grained analysis fills the theoretical gap on the generalization guarantees and polishes the convergence characterization of the computing algorithm.

----

## [0] Sparse Deep Learning for Time Series Data: Theory and Applications

**Authors**: *Mingxuan Zhang, Yan Sun, Faming Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aaa9c20f0a217a1aef6fa5d97f310292-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aaa9c20f0a217a1aef6fa5d97f310292-Abstract-Conference.html)

**Abstract**:

Sparse deep learning has become a popular technique for improving the performance of deep neural networks in areas such as uncertainty quantification, variable selection, and large-scale network compression. However, most existing research has focused on problems where the observations are independent and identically distributed (i.i.d.), and there has been little work on the problems where the observations are dependent, such as time series data and sequential data in natural language processing. This paper aims to address this gap by studying the theory for sparse deep learning with dependent data. We show that sparse recurrent neural networks (RNNs) can be consistently estimated, and their predictions are asymptotically normally distributed under appropriate assumptions, enabling the prediction uncertainty to be correctly quantified. Our numerical results show that sparse deep learning outperforms state-of-the-art methods, such as conformal predictions, in prediction uncertainty quantification for time series data. Furthermore, our results indicate that the proposed method can consistently identify the autoregressive order for time series data and outperform existing methods in large-scale model compression. Our proposed method has important practical implications in fields such as finance, healthcare, and energy, where both accurate point estimates and prediction uncertainty quantification are of concern.

----

## [0] Imbalanced Mixed Linear Regression

**Authors**: *Pini Zilber, Boaz Nadler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aad615d33ba5071045656ba24d800c7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aad615d33ba5071045656ba24d800c7b-Abstract-Conference.html)

**Abstract**:

We consider the problem of mixed linear regression (MLR), where each observed sample belongs to one of $K$ unknown linear models. In practical applications, the mixture of the $K$ models may be imbalanced with a significantly different number of samples from each model. Unfortunately, most MLR methods do not perform well in such settings. Motivated by this practical challenge, in this work we propose Mix-IRLS, a novel, simple and fast algorithm for MLR with excellent performance on both balanced and imbalanced mixtures.In contrast to popular approaches that recover the $K$ models simultaneously, Mix-IRLS does it sequentially using tools from robust regression. Empirically, beyond imbalanced mixtures, Mix-IRLS succeeds in a broad range of additional settings where other methods fail, including small sample sizes, presence of outliers, and an unknown number of models $K$. Furthermore, Mix-IRLS outperforms competing methods on several real-world datasets, in some cases by a large margin. We complement our empirical results by deriving a recovery guarantee for Mix-IRLS, which highlights its advantage on imbalanced mixtures.

----

## [0] GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference

**Authors**: *Ziang Li, Mengda Yang, Yaxin Liu, Juan Wang, Hongxin Hu, Wenzhe Yi, Xiaoyang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)

**Abstract**:

Split Inference (SI) is an emerging deep learning paradigm that addresses computational constraints on edge devices and preserves data privacy through collaborative edge-cloud approaches. However, SI is vulnerable to Data Reconstruction Attacks (DRA), which aim to reconstruct users' private prediction instances. Existing attack methods suffer from various limitations. Optimization-based DRAs do not leverage public data effectively, while Learning-based DRAs depend heavily on auxiliary data quantity and distribution similarity. Consequently, these approaches yield unsatisfactory attack results and are sensitive to defense mechanisms. To overcome these challenges, we propose a GAN-based LAtent Space Search attack (GLASS) that harnesses abundant prior knowledge from public data using advanced StyleGAN technologies. Additionally, we introduce GLASS++ to enhance reconstruction stability. Our approach represents the first GAN-based DRA against SI, and extensive evaluation across different split points and adversary setups demonstrates its state-of-the-art performance. Moreover, we thoroughly examine seven defense mechanisms, highlighting our method's capability to reveal private information even in the presence of these defenses.

----

## [0] Random-Access Infinite Context Length for Transformers

**Authors**: *Amirkeivan Mohtashami, Martin Jaggi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab05dc8bf36a9f66edbff6992ec86f56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab05dc8bf36a9f66edbff6992ec86f56-Abstract-Conference.html)

**Abstract**:

While Transformers have shown remarkable success in natural language processing, their attention mechanism's large memory requirements have limited their ability to handle longer contexts. Prior approaches, such as recurrent memory or retrieval-based augmentation, have either compromised the random-access flexibility of attention (i.e., the capability to select any token in the entire context) or relied on separate mechanisms for relevant context retrieval, which may not be compatible with the model's attention. In this paper, we present a novel approach that allows access to the complete context while retaining random-access flexibility, closely resembling running attention on the entire context. Our method uses a landmark token to represent each block of the input and trains the attention to use it for selecting relevant blocks, enabling retrieval of blocks directly through the attention mechanism instead of by relying on a separate mechanism. Our approach seamlessly integrates with specialized data structures and the system's memory hierarchy, enabling processing of arbitrarily long context lengths. We demonstrate that our method can obtain comparable performance with Transformer-XL while significantly reducing the number of retrieved tokens in each step. Finally, we show that fine-tuning LLaMA 7B with our method successfully extends its context length capacity to over 32k tokens, allowing for inference at the context lengths of GPT-4. We release the implementation of landmark attention and the code to reproduce our experiments at https://github.com/epfml/landmark-attention/.

----

## [0] Egocentric Planning for Scalable Embodied Task Achievement

**Authors**: *Xiaotian Liu, Héctor Palacios, Christian Muise*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab0b1be09c317cb068aecfa7fa86a7e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab0b1be09c317cb068aecfa7fa86a7e3-Abstract-Conference.html)

**Abstract**:

Embodied agents face significant challenges when tasked with performing actions in diverse environments, particularly in generalizing across object types and executing suitable actions to accomplish tasks. Furthermore, agents should exhibit robustness, minimizing the execution of illegal actions. In this work, we present Egocentric Planning, an innovative approach that combines symbolic planning and Object-oriented POMDPs to solve tasks in complex environments, harnessing existing models for visual perception and natural language processing. We evaluated our approach in ALFRED, a simulated environment designed for domestic tasks, and demonstrated its high scalability, achieving an impressive 36.07\% unseen success rate in the ALFRED benchmark and winning the ALFRED challenge at CVPR Embodied AI workshop. Our method requires reliable perception and the specification or learning of a symbolic description of the preconditions and effects of the agent's actions, as well as what object types reveal information about others. It can naturally scale to solve new tasks beyond ALFRED, as long as they can be solved using the available skills. This work offers a solid baseline for studying end-to-end and hybrid methods that aim to generalize to new tasks, including recent approaches relying on LLMs, but often struggle to scale to long sequences of actions or produce robust plans for novel tasks.

----

## [0] Removing Hidden Confounding in Recommendation: A Unified Multi-Task Learning Approach

**Authors**: *Haoxuan Li, Kunhan Wu, Chunyuan Zheng, Yanghao Xiao, Hao Wang, Zhi Geng, Fuli Feng, Xiangnan He, Peng Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab3f114401f0523ca1cc09de0621f400-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab3f114401f0523ca1cc09de0621f400-Abstract-Conference.html)

**Abstract**:

In recommender systems, the collected data used for training is always subject to selection bias, which poses a great challenge for unbiased learning. Previous studies proposed various debiasing methods based on observed user and item features, but ignored the effect of hidden confounding. To address this problem, recent works suggest the use of sensitivity analysis for worst-case control of the unknown true propensity, but only valid when the true propensity is near to the nominal propensity within a finite bound. In this paper, we first perform theoretical analysis to reveal the possible failure of previous approaches, including propensity-based, multi-task learning, and bi-level optimization methods, in achieving unbiased learning when hidden confounding is present. Then, we propose a unified multi-task learning approach to remove hidden confounding, which uses a few unbiased ratings to calibrate the learned nominal propensities and nominal error imputations from biased data. We conduct extensive experiments on three publicly available benchmark datasets containing a fully exposed large-scale industrial dataset, validating the effectiveness of the proposed methods in removing hidden confounding.

----

## [0] Generative Category-level Object Pose Estimation via Diffusion Models

**Authors**: *Jiyao Zhang, Mingdong Wu, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab59d149fc0c2c9039d3e3049f7914b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab59d149fc0c2c9039d3e3049f7914b1-Abstract-Conference.html)

**Abstract**:

Object pose estimation plays a vital role in embodied AI and computer vision, enabling intelligent agents to comprehend and interact with their surroundings. Despite the practicality of category-level pose estimation, current approaches encounter challenges with partially observed point clouds, known as the multihypothesis issue. In this study, we propose a novel solution by reframing categorylevel object pose estimation as conditional generative modeling, departing from traditional point-to-point regression. Leveraging score-based diffusion models, we estimate object poses by sampling candidates from the diffusion model and aggregating them through a two-step process: filtering out outliers via likelihood estimation and subsequently mean-pooling the remaining candidates. To avoid the costly integration process when estimating the likelihood, we introduce an alternative method that distils an energy-based model from the original score-based model, enabling end-to-end likelihood estimation. Our approach achieves state-of-the-art performance on the REAL275 dataset, surpassing 50% and 60% on strict 5 ◦ 2cm and 5 ◦ 5cm metrics, respectively. Furthermore, our method demonstrates strong generalization to novel categories without the need for fine-tuning and can readily adapt to object pose tracking tasks, yielding comparable results to the current state-of-the-art baselines. Our checkpoints and demonstrations can be found at https://sites.google.com/view/genpose.

----

## [0] On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective

**Authors**: *Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab5a2bf4385bee44f3919060b184605b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab5a2bf4385bee44f3919060b184605b-Abstract-Conference.html)

**Abstract**:

Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating saliency maps, and counterfactual explanations. However, saliency maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the saliency maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties:They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the saliency map for such models, we prove this gradient encodes  both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another.  Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the saliency map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.

----

## [0] DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models

**Authors**: *Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html)

**Abstract**:

Current deep networks are very data-hungry and benefit from training on large-scale datasets, which are often time-consuming to collect and annotate. By contrast, synthetic data can be generated infinitely using generative models such as DALL-E and diffusion models, with minimal effort and cost. In this paper, we present DatasetDM, a generic dataset generation model that can produce diverse syntheticimages and the corresponding high-quality perception annotations (e.g., segmentation masks, and depth). Our method builds upon the pre-trained diffusion model and extends text-guided image synthesis to perception data generation. We show that the rich latent code of the diffusion model can be effectively decoded as accurate perception annotations using a decoder module. Training the decoder only needs less than 1% (around 100 images) of manually labeled images, enabling the generation of an infinitely large annotated dataset. Then these synthetic data can be used for training various perception models on downstream tasks. To showcase the power of the proposed approach, we generate datasets with rich dense pixel-wise labels for a wide range of downstream tasks, including semantic15segmentation, instance segmentation, and depth estimation. Notably, it achieves 1) state-of-the-art results on semantic segmentation and instance segmentation; 2) significantly more efficient and robust in domain generalization than the real data; 3) state-of-the-art results in zero-shot segmentation setting; and 4) flexibility for efficient application and novel task composition (e.g., image editing)

----

## [0] No-Regret Online Prediction with Strategic Experts

**Authors**: *Omid Sadeghi, Maryam Fazel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab9f9cfe97da3665e08f50ade9f8c4d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab9f9cfe97da3665e08f50ade9f8c4d6-Abstract-Conference.html)

**Abstract**:

We study a generalization of the online binary prediction with expert advice framework where at each round, the learner is allowed to pick $m\geq 1$ experts from a pool of $K$ experts and the overall utility is a modular or submodular function of the chosen experts. We focus on the setting in which experts act strategically and aim to maximize their influence on the algorithm's predictions by potentially misreporting their beliefs about the events. Among others, this setting finds applications in forecasting competitions where the learner seeks not only to make predictions by aggregating different forecasters but also to rank them according to their relative performance. Our goal is to design algorithms that satisfy the following two requirements: 1) \emph{Incentive-compatible}: Incentivize the experts to report their beliefs truthfully, and 2) \emph{No-regret}: Achieve sublinear regret with respect to the true beliefs of the best fixed set of $m$ experts in hindsight. Prior works have studied this framework when $m=1$ and provided incentive-compatible no-regret algorithms for the problem. We first show that a simple reduction of our problem to the $m=1$ setting is neither efficient nor effective. Then, we provide algorithms that utilize the specific structure of the utility functions to achieve the two desired goals.

----

## [0] Learning Unseen Modality Interaction

**Authors**: *Yunhua Zhang, Hazel Doughty, Cees Snoek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abb4847bbd60f38b1b7649d26c7a0067-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abb4847bbd60f38b1b7649d26c7a0067-Abstract-Conference.html)

**Abstract**:

Multimodal learning assumes all modality combinations of interest are available during training to learn cross-modal correspondences. In this paper, we challenge this modality-complete assumption for multimodal learning and instead strive for generalization to unseen modality combinations during inference. We pose the problem of unseen modality interaction and introduce a first solution. It exploits a module that projects the multidimensional features of different modalities into a common space with rich information preserved. This allows the information to be accumulated with a simple summation operation across available modalities. To reduce overfitting to less discriminative modality combinations during training, we further improve the model learning with pseudo-supervision indicating the reliability of a modalityâ€™s prediction. We demonstrate that our approach is effective for diverse tasks and modalities by evaluating it for multimodal video classification, robot state regression, and multimedia retrieval. Project website: https://xiaobai1217.github.io/Unseen-Modality-Interaction/.

----

## [0] Autonomous Capability Assessment of Sequential Decision-Making Systems in Stochastic Settings

**Authors**: *Pulkit Verma, Rushang Karia, Siddharth Srivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abbb7f20cdffdd3bb7d98447f60b0b0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abbb7f20cdffdd3bb7d98447f60b0b0c-Abstract-Conference.html)

**Abstract**:

It is essential for users to understand what their AI systems can and can't do in order to use them safely. However, the problem of enabling users to assess AI systems with sequential decision-making (SDM) capabilities is relatively understudied. This paper presents a new approach for modeling the capabilities of black-box AI systems that can plan and act, along with the possible effects and requirements for executing those capabilities in stochastic settings. We present an active-learning approach that can effectively interact with a black-box SDM system and learn an interpretable probabilistic model describing its capabilities. Theoretical analysis of the approach identifies the conditions under which the learning process is guaranteed to converge to the correct model of the agent; empirical evaluations on different agents and simulated scenarios show that this approach is few-shot generalizable and can effectively describe the capabilities of arbitrary black-box SDM agents in a sample-efficient manner.

----

## [0] Model-Free Active Exploration in Reinforcement Learning

**Authors**: *Alessio Russo, Alexandre Proutière*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abbbb25cddb2c2cd08714e6bfa2f0634-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abbbb25cddb2c2cd08714e6bfa2f0634-Abstract-Conference.html)

**Abstract**:

We study the problem of exploration in Reinforcement Learning and present a novel model-free solution. We adopt an information-theoretical viewpoint and start from the  instance-specific lower bound of the number of samples that have to be collected to identify a nearly-optimal policy. Deriving this lower bound along with the optimal exploration strategy entails solving an intricate optimization problem and requires a model of the system. In turn, most existing sample optimal exploration algorithms rely on estimating the model. We derive an approximation of the instance-specific lower bound that only involves quantities that can be inferred using model-free approaches. Leveraging this approximation, we devise an ensemble-based model-free exploration strategy  applicable to both tabular and continuous Markov decision processes. Numerical results demonstrate that our strategy is able to identify efficient policies faster than state-of-the-art exploration approaches.

----

## [0] Universality laws for Gaussian mixtures in generalized linear models

**Authors**: *Yatin Dandi, Ludovic Stephan, Florent Krzakala, Bruno Loureiro, Lenka Zdeborová*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abccb8a90b30d45b948360ba41f5a20f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abccb8a90b30d45b948360ba41f5a20f-Abstract-Conference.html)

**Abstract**:

A recent line of work in high-dimensional statistics working under the Gaussian mixture hypothesis has led to a number of results in the context of empirical risk minimization, Bayesian uncertainty quantification, separation of kernel methods and neural networks, ensembling and fluctuation of random features. We provide rigorous proofs for the applicability of these results to a general class of datasets $(\mathbf{x_i},y_i, {i=1,\dots,n})$ containing independent samples from a mixture distribution $\sum_{c\in\mathcal{C}} \rho_{c}P_{c}^{\mathbf{x}}$. Specifically, we consider the hypothesis class of generalized linear models $\hat{y} = F(\mathbf{\Theta}^{\top}\mathbf{x})$ and investigate the asymptotic joint statistics of a family of generalized linear estimators $(\mathbf{\Theta}^{(1)}, \dots, \mathbf{\Theta}^{(M)})$, obtained either from (a) minimizing an empirical risk $\hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y})$ or (b) sampling from the associated Gibbs measure $\exp(-\beta n \hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y}))$. Our main contribution is to characterize under which conditions the asymptotic joint statistics of this family depends (on a weak sense) only on the means and covariances of the class conditional features distribution $P_{c}^{\mathbf{x}}$. This allows us to prove the universality of different quantities of interest, including training, generalization errors, as well as the geometrical properties and correlations of the estimators.

----

## [0] ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers

**Authors**: *Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, Lei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abe1eb21ceb046209c96a0f5e7544ccc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abe1eb21ceb046209c96a0f5e7544ccc-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) excel at implementing code from functionality descriptions but struggle with algorithmic problems that require not only implementation but also identification of the suitable algorithm. Moreover, LLM-generated programs lack guaranteed correctness and require human verification. To address these challenges, we propose ALGO, a framework that synthesizes Algorithmic programs with LLM-Generated Oracles to guide the generation and verify their correctness. ALGO first generates a reference oracle by prompting an LLM to exhaustively enumerate all the combinations of relevant variables. This oracle is then utilized to guide an arbitrary search strategy in exploring the algorithm space and to verify the synthesized algorithms. Our study shows that the LLM-generatedoracles are correct for 88% of the cases. With the oracles as verifiers, ALGO can be integrated with any existing code generation model in a model-agnostic manner to enhance its performance. Experiments show that when equipped with ALGO, we achieve an 8× better one-submission pass rate over the Codex model and a 2.6× better one-submission pass rate over CodeT, the current state-of-the-art model on CodeContests. We can also get 1.3× better pass rate over the ChatGPT Code Interpreter on unseen problems. The problem set we used for testing, the prompts we used, the verifier and solution programs, and the test cases generated by ALGOare available at https://github.com/zkx06111/ALGO.

----

## [0] Private Everlasting Prediction

**Authors**: *Moni Naor, Kobbi Nissim, Uri Stemmer, Chao Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html)

**Abstract**:

A private learner is trained on a sample of labeled points and generates a hypothesis that can be used for predicting the labels of newly sampled points while protecting the privacy of the training set [Kasiviswannathan et al., FOCS 2008]. Past research uncovered that private learners may need to exhibit significantly higher sample complexity than non-private learners as is the case of learning of one-dimensional threshold functions [Bun et al., FOCS 2015, Alon et al., STOC 2019].We explore prediction as an alternative to learning. A predictor answers a stream of classification queries instead of outputting a hypothesis. Earlier work has considered a private prediction model with a single classification query [Dwork and Feldman, COLT 2018]. We observe that when answering a stream of queries, a predictor must modify the hypothesis it uses over time, and in a manner that  cannot rely solely on the training set.We introduce {\em private everlasting prediction} taking into account the privacy of both the training set {\em and} the (adaptively chosen) queries made to the predictor. We then present a generic construction of private everlasting predictors in the PAC model.The sample complexity of the initial training sample in our construction is quadratic (up to polylog factors) in the VC dimension of the concept class. Our construction allows prediction for all concept classes with finite VC dimension, and in particular threshold functions over infinite domains, for which (traditional) private learning is known to be impossible.

----

## [0] A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation

**Authors**: *Thomas Fel, Victor Boutin, Louis Béthune, Rémi Cadène, Mazda Moayeri, Léo Andéol, Mathieu Chalvidal, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html)

**Abstract**:

In recent years, concept-based approaches have emerged as some of the most promising explainability methods to help us interpret the decisions of Artificial Neural Networks (ANNs). These methods seek to discover intelligible visual ``concepts'' buried within the complex patterns of ANN activations in two key steps: (1) concept extraction followed by (2) importance estimation. While these two steps are shared across methods, they all differ in their specific implementations. Here, we introduce a unifying theoretical framework that recast the first step -- concept extraction problem -- as a special case of dictionary learning, and we formalize the second step -- concept importance estimation -- as a more general form of attribution method.This framework offers several advantages as it allows us: (i) to propose new evaluation metrics for comparing different concept extraction approaches; (ii) to leverage modern attribution methods and evaluation metrics to extend and systematically evaluate state-of-the-art concept-based approaches and importance estimation techniques; (iii)  to derive theoretical guarantees regarding the optimality of such methods. We further leverage our framework to try to tackle a crucial question in explainability: how to efficiently identify clusters of data points that are classified based on a similar shared strategy.To illustrate these findings and to highlight the main strategies of a model, we introduce a visual representation called the strategic cluster graph. Finally, we present Lens, a dedicated website that offers a complete compilation of these visualizations for all classes of the ImageNet dataset.

----

## [0] Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

**Authors**: *Fenja Falta, Christoph Großbröhmer, Alessa Hering, Alexander Bigalke, Mattias P. Heinrich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A popular benchmark for intra-patient lung registration is provided by the DIR-LAB COPDgene dataset consisting of large-motion in- and expiratory breath-hold CT pairs. This dataset alone, however, does not provide enough samples to properly train state-of-the-art deep learning methods. Other public datasets often also provide only small sample sizes or include primarily small motions between scans that do not translate well to larger deformations. For point-based geometric registration, the PVT1010 dataset provides a large number of vessel point clouds without any correspondences and a labeled test set corresponding to the COPDgene cases. However, the absence of correspondences for supervision complicates training, and a fair comparison with image-based algorithms is infeasible, since CT scans for the training data are not publicly available.We here provide a combined benchmark for image- and point-based registration approaches. We curated a total of 248 public multi-centric in- and expiratory lung CT scans from 124 patients, which show large motion between scans, processed them to ensure sufficient homogeneity between the data and generated vessel point clouds that are well distributed even deeper inside the lungs. For supervised training, we provide vein and artery segmentations of the vessels and multiple thousand image-derived keypoint correspondences for each pair. For validation, we provide multiple scan pairs with manual landmark annotations. Finally, as first baselines on our new benchmark, we evaluate several image and point cloud registration methods on the dataset.

----

## [0] An Iterative Self-Learning Framework for Medical Domain Generalization

**Authors**: *Zhenbang Wu, Huaxiu Yao, David M. Liebovitz, Jimeng Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac0035c349f3fe8af6a93fe44697b5bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac0035c349f3fe8af6a93fe44697b5bd-Abstract-Conference.html)

**Abstract**:

Deep learning models have been widely used to assist doctors with clinical decision-making. However, these models often encounter a significant performance drop when applied to data that differs from the distribution they were trained on. This challenge is known as the domain shift problem. Existing domain generalization algorithms attempt to address this problem by assuming the availability of domain IDs and training a single model to handle all domains. However, in healthcare settings, patients can be classified into numerous latent domains, where the actual domain categorizations are unknown. Furthermore, each patient domain exhibits distinct clinical characteristics, making it sub-optimal to train a single model for all domains. To overcome these limitations, we propose SLGD, a self-learning framework that iteratively discovers decoupled domains and trains personalized classifiers for each decoupled domain. We evaluate the generalizability of SLGD across spatial and temporal data distribution shifts on two real-world public EHR datasets: eICU and MIMIC-IV. Our results show that SLGD achieves up to 11% improvement in the AUPRC score over the best baseline.

----

## [0] A benchmark of categorical encoders for binary classification

**Authors**: *Federico Matteucci, Vadim Arzamasov, Klemens Böhm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Categorical encoders transform categorical features into numerical representations that are indispensable for a wide range of machine learning models.Existing encoder benchmark studies lack generalizability because of their limited choice of (1) encoders, (2) experimental factors, and (3) datasets. Additionally, inconsistencies arise from the adoption of varying aggregation strategies.This paper is the most comprehensive benchmark of categorical encoders to date, including an extensive evaluation of 32 configurations of encoders from diverse families, with 36 combinations of experimental factors, and on 50 datasets.The study shows the profound influence of dataset selection, experimental factors, and aggregation strategies on the benchmark's conclusions~---~aspects disregarded in previous encoder benchmarks.Our code is available at \url{https://github.com/DrCohomology/EncoderBenchmarking}.

----

## [0] Structure of universal formulas

**Authors**: *Dmitry Yarotsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac04e54e0a2d1927d60709019e4e7870-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac04e54e0a2d1927d60709019e4e7870-Abstract-Conference.html)

**Abstract**:

By universal formulas we understand parameterized analytic expressions that have a fixed complexity, but nevertheless can approximate any continuous function on a compact set. There exist various examples of such formulas, including some in the form of neural networks. In this paper we analyze the essential structural elements of these highly expressive models. We introduce a hierarchy of expressiveness classes connecting the global approximability property to the weaker property of infinite VC dimension, and prove a series of classification results for several increasingly complex functional families. In particular, we introduce a general family of polynomially-exponentially-algebraic functions that, as we prove, is subject to polynomial constraints. As a consequence, we show that fixed-size neural networks with not more than one layer of neurons having transcendental activations (e.g., sine or standard sigmoid) cannot in general approximate functions on arbitrary finite sets. On the other hand, we give examples of functional families, including two-hidden-layer neural networks, that approximate functions on arbitrary finite sets, but fail to do that  on the whole domain of definition.

----

## [0] Model-enhanced Vector Index

**Authors**: *Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang, Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding, Xupeng Miao, Haonan Wang, Bochen Pang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Qi Zhang, Fan Yang, Xing Xie, Mao Yang, Bin Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html)

**Abstract**:

Embedding-based retrieval methods construct vector indices to search for document representations that are most similar to the query representations. They are widely used in document retrieval due to low latency and decent recall performance. Recent research indicates that deep retrieval solutions offer better model quality, but are hindered by unacceptable serving latency and the inability to support document updates. In this paper, we aim to enhance the vector index with end-to-end deep generative models, leveraging the differentiable advantages of deep retrieval models while maintaining desirable serving efficiency. We propose Model-enhanced Vector Index (MEVI), a differentiable model-enhanced index empowered by a twin-tower representation model. MEVI leverages a Residual Quantization (RQ) codebook to bridge the sequence-to-sequence deep retrieval and embedding-based models. To substantially reduce the inference time, instead of decoding the unique document ids in long sequential steps, we first generate some semantic virtual cluster ids of candidate documents in a small number of steps, and then leverage the well-adapted embedding vectors to further perform a fine-grained search for the relevant documents in the candidate virtual clusters. We empirically show that our model achieves better performance on the commonly used academic benchmarks MSMARCO Passage and Natural Questions, with comparable serving latency to dense retrieval solutions.

----

## [0] Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models

**Authors**: *Tianxiang Gao, Xiaokai Huo, Hailiang Liu, Hongyang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac24656b0b5f543b202f748d62041637-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac24656b0b5f543b202f748d62041637-Abstract-Conference.html)

**Abstract**:

Neural networks with wide layers have attracted significant attention due to their equivalence to Gaussian processes, enabling perfect fitting of training data while maintaining generalization performance, known as benign overfitting. However, existing results mainly focus on shallow or finite-depth networks, necessitating a comprehensive analysis of wide neural networks with infinite-depth layers, such as neural ordinary differential equations (ODEs) and deep equilibrium models (DEQs). In this paper, we specifically investigate the deep equilibrium model (DEQ), an infinite-depth neural network with shared weight matrices across layers. Our analysis reveals that as the width of DEQ layers approaches infinity, it converges to a Gaussian process, establishing what is known as the Neural Network and Gaussian Process (NNGP) correspondence. Remarkably, this convergence holds even when the limits of depth and width are interchanged, which is not observed in typical infinite-depth Multilayer Perceptron (MLP) networks. Furthermore, we demonstrate that the associated Gaussian vector remains non-degenerate for any pairwise distinct input data, ensuring a strictly positive smallest eigenvalue of the corresponding kernel matrix using the NNGP kernel. These findings serve as fundamental elements for studying the training and generalization of DEQs, laying the groundwork for future research in this area.

----

## [0] Tree Variational Autoencoders

**Authors**: *Laura Manduchi, Moritz Vandenhirtz, Alain Ryser, Julia E. Vogt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac58b418745b3e5f10c80110c963969f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac58b418745b3e5f10c80110c963969f-Abstract-Conference.html)

**Abstract**:

We propose Tree Variational Autoencoder (TreeVAE), a new generative hierarchical clustering model  that learns a flexible tree-based posterior distribution over latent variables. TreeVAE hierarchically divides samples according to their intrinsic characteristics, shedding light on hidden structures in the data. It adapts its architecture to discover the optimal tree for encoding dependencies between latent variables. The proposed tree-based generative architecture enables lightweight conditional inference and improves generative performance by utilizing specialized leaf decoders.   We show that TreeVAE uncovers underlying clusters in the data and finds meaningful hierarchical relations between the different groups on a variety of datasets, including real-world imaging data.   We present empirically that TreeVAE provides a more competitive log-likelihood lower bound than the sequential counterparts.   Finally, due to its generative nature, TreeVAE is able to generate new samples from the discovered clusters via conditional sampling.

----

## [0] Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes

**Authors**: *Yihong Sun, Bharath Hariharan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac5c594dedf66affb098c39a3bcfdb3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac5c594dedf66affb098c39a3bcfdb3d-Abstract-Conference.html)

**Abstract**:

Unsupervised monocular depth estimation techniques have demonstrated encouraging results but typically assume that the scene is static. These techniques suffer when trained on dynamical scenes, where apparent object motion can equally be explained by hypothesizing the object's independent motion, or by altering its depth. This ambiguity causes depth estimators to predict erroneous depth for moving objects. To resolve this issue, we introduce Dynamo-Depth, an unifying approach that disambiguates dynamical motion by jointly learning monocular depth, 3D independent flow field, and motion segmentation from unlabeled monocular videos. Specifically, we offer our key insight that a good initial estimation of motion segmentation is sufficient for jointly learning depth and independent motion despite the fundamental underlying ambiguity. Our proposed method achieves state-of-the-art performance on monocular depth estimation on Waymo Open and nuScenes Dataset with significant improvement in the depth of moving objects. Code and additional results are available at https://dynamo-depth.github.io.

----

## [0] LIMA: Less Is More for Alignment

**Authors**: *Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)

**Abstract**:

Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history.Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data.In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43\% of cases; this statistic is as high as 58\% when compared to Bard and 65\% versus DaVinci003, which was trained with human feedback.Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.

----



[Go to the previous page](NIPS-2023-list2300.md)

[Go to the next page](NIPS-2023-list2302.md)

[Go to the catalog section](README.md)