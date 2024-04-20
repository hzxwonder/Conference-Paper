## [800] LOG: Active Model Adaptation for Label-Efficient OOD Generalization

        **Authors**: *Jie-Jing Shao, Lan-Zhe Guo, Xiaowen Yang, Yu-Feng Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4757094e8ccc17e3e25b40efaf06c746-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4757094e8ccc17e3e25b40efaf06c746-Abstract-Conference.html)

        **Abstract**:

        This work discusses how to achieve worst-case Out-Of-Distribution (OOD) generalization for a variety of distributions based on a relatively small labeling cost. The problem has broad applications, especially in non-i.i.d. open-world scenarios. Previous studies either rely on a large amount of labeling cost or lack of guarantees about the worst-case generalization. In this work, we show for the first time that active model adaptation could achieve both good performance and robustness based on the invariant risk minimization principle. We propose \textsc{Log}, an interactive model adaptation framework, with two sub-modules: active sample selection and causal invariant learning. Specifically, we formulate the active selection as a mixture distribution separation problem and present an unbiased estimator, which could find the samples that violate the current invariant relationship, with a provable guarantee. The theoretical analysis supports that both sub-modules contribute to generalization. A large number of experimental results confirm the promising performance of the new algorithm.

        ----

        ## [801] PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds

        **Authors**: *Aoran Xiao, Jiaxing Huang, Dayan Guan, Kaiwen Cui, Shijian Lu, Ling Shao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/475b85eb74d201bead9927807e713e95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/475b85eb74d201bead9927807e713e95-Abstract-Conference.html)

        **Abstract**:

        LiDAR point clouds, which are usually scanned by rotating LiDAR sensors continuously, capture precise geometry of the surrounding environment and are crucial to many autonomous detection and navigation tasks. Though many 3D deep architectures have been developed, efficient collection and annotation of large amounts of point clouds remain one major challenge in the analytics and understanding of point cloud data. This paper presents PolarMix, a point cloud augmentation technique that is simple and generic but can mitigate the data constraint effectively across various perception tasks and scenarios. PolarMix enriches point cloud distributions and preserves point cloud fidelity via two cross-scan augmentation strategies that cut, edit, and mix point clouds along the scanning direction. The first is scene-level swapping which exchanges point cloud sectors of two LiDAR scans that are cut along the LiDAR scanning direction. The second is instance-level rotation and paste which crops point instances from one LiDAR scan, rotates them by multiple angles (to create multiple copies), and paste the rotated point instances into other scans. Extensive experiments show that PolarMix achieves superior performance consistently across different perception tasks and scenarios. In addition, it can work as a plug-and-play for various 3D deep architectures and also performs well for unsupervised domain adaptation.

        ----

        ## [802] Learning from Distributed Users in Contextual Linear Bandits Without Sharing the Context

        **Authors**: *Osama A. Hanna, Lin Yang, Christina Fragouli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html)

        **Abstract**:

        Contextual linear bandits is a rich and theoretically important model that has many practical applications. Recently, this setup gained a lot of interest in applications over wireless where communication constraints can be a performance bottleneck, especially when the contexts come from a large $d$-dimensional space. In this paper, we consider the distributed contextual linear bandit learning problem, where the agents who observe the contexts and take actions are geographically separated from the learner who performs the learning while not seeing the contexts. We assume that contexts are generated from a distribution and propose a method that uses $\approx 5d$ bits per context for the case of unknown context distribution and $0$ bits per context if the context distribution is known, while achieving nearly the same regret bound as if the contexts were directly observable. The former bound improves upon existing bounds by a $\log(T)$ factor, where $T$ is the length of the horizon, while the latter achieves information theoretical tightness.

        ----

        ## [803] Provably Feedback-Efficient Reinforcement Learning via Active Reward Learning

        **Authors**: *Dingwen Kong, Lin Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/476c289f685e27936aa089e9d53a4213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/476c289f685e27936aa089e9d53a4213-Abstract-Conference.html)

        **Abstract**:

        An appropriate reward function is of paramount importance in specifying a task in reinforcement learning (RL). Yet, it is known to be extremely challenging in practice to design a correct reward function for even simple tasks. Human-in-the-loop (HiL) RL allows humans to communicate complex goals to the RL agent by providing various types of feedback. However, despite achieving great empirical successes, HiL RL usually requires \emph{too much} feedback from a human teacher and also suffers from insufficient theoretical understanding. In this paper, we focus on addressing this issue from a theoretical perspective, aiming to provide provably feedback-efficient algorithmic frameworks that take human-in-the-loop to specify rewards of given tasks. We provide an \emph{active-learning}-based RL algorithm that first explores the environment without specifying a reward function and then asks a human teacher for only a few queries about the rewards of a task at some state-action pairs. After that, the algorithm guarantees to provide a nearly optimal policy for the task with high probability. We show that, even with the presence of random noise in the feedback, the algorithm only takes $\tilde{O}(H{\dim_{R}^2})$ queries on the reward function to provide an $\epsilon$-optimal policy for any $\epsilon > 0$. Here $H$ is the horizon of the RL environment, and $\dim_{R}$ specifies the complexity of the function class representing the reward function. In contrast, standard RL algorithms require to query the reward function for at least $\Omega(\operatorname{poly}(d, 1/\epsilon))$ state-action pairs where $d$ depends on the complexity of the environmental transition.

        ----

        ## [804] Recurrent Memory Transformer

        **Authors**: *Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/47e288629a6996a17ce50b90a056a0e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/47e288629a6996a17ce50b90a056a0e1-Abstract-Conference.html)

        **Abstract**:

        Transformer-based models show their effectiveness across multiple domains and tasks. The self-attention allows to combine information from all sequence elements into context-aware representations. However, global and local information has to be stored mostly in the same element-wise representations. Moreover, the length of an input sequence is limited by quadratic computational complexity of self-attention.  In this work, we propose and study a memory-augmented segment-level recurrent Transformer (RMT). Memory allows to store and process local and global information as well as to pass information between segments of the long sequence with the help of recurrence.  We implement a memory mechanism with no changes to Transformer model by adding special memory tokens to the input or output sequence. Then the model is trained to control both memory operations and sequence representations processing.  Results of experiments show that RMT performs on par with the Transformer-XL on language modeling for smaller memory sizes and outperforms it for tasks that require longer sequence processing. We show that adding memory tokens to Tr-XL is able to improve its performance. This makes Recurrent Memory Transformer a promising architecture for applications that require learning of long-term dependencies and general purpose in memory processing, such as algorithmic tasks and reasoning.

        ----

        ## [805] Hierarchical Lattice Layer for Partially Monotone Neural Networks

        **Authors**: *Hiroki Yanagisawa, Kohei Miyaguchi, Takayuki Katsuki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/47ed62021460f2e9bba7be3e74260090-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/47ed62021460f2e9bba7be3e74260090-Abstract-Conference.html)

        **Abstract**:

        Partially monotone regression is a regression analysis in which the target values are monotonically increasing with respect to a subset of input features.   The TensorFlow Lattice library is one of the standard machine learning libraries for partially monotone regression.  It consists of several neural network layers, and its core component is the lattice layer.  One of the problems of the lattice layer is that it requires the projected gradient descent algorithm with many constraints to train it.  Another problem is that it cannot receive a high-dimensional input vector due to the memory consumption.   We propose a novel neural network layer, the hierarchical lattice layer (HLL), as an extension of the lattice layer so that we can use a standard stochastic gradient descent algorithm to train HLL while satisfying monotonicity constraints and so that it can receive a high-dimensional input vector.  Our experiments demonstrate that HLL did not sacrifice its prediction performance on real datasets compared with the lattice layer.

        ----

        ## [806] Class-Dependent Label-Noise Learning with Cycle-Consistency Regularization

        **Authors**: *De Cheng, Yixiong Ning, Nannan Wang, Xinbo Gao, Heng Yang, Yuxuan Du, Bo Han, Tongliang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/47f75e809409709c6d226ab5ca0c9703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/47f75e809409709c6d226ab5ca0c9703-Abstract-Conference.html)

        **Abstract**:

        In label-noise learning, estimating the transition matrix plays an important role in building statistically consistent classifier. Current state-of-the-art consistent estimator for the transition matrix has been developed under the newly proposed sufficiently scattered assumption, through incorporating the minimum volume constraint of the transition matrix T into label-noise learning. To compute the volume of  T, it heavily relies on the estimated  noisy class posterior. However, the estimation error of the noisy class posterior could usually be large as deep learning methods tend to easily overfit the noisy labels. Then, directly minimizing the volume of such obtained T could lead the transition matrix to be poorly estimated.  Therefore, how to reduce the side-effects of the inaccurate noisy class posterior has become the bottleneck of such method. In this paper, we creatively propose to estimate the transition matrix under the forward-backward cycle-consistency regularization, of which we have greatly reduced the dependency of estimating the transition matrix T on the noisy class posterior. We show that the cycle-consistency regularization helps to minimize the volume of the transition matrix T indirectly without exploiting the estimated noisy class posterior, which could further encourage the estimated transition matrix T to converge to its optimal solution. Extensive experimental results consistently justify the effectiveness of the proposed method, on reducing the estimation error of the transition matrix and greatly boosting the classification performance.

        ----

        ## [807] SeqPATE: Differentially Private Text Generation via Knowledge Distillation

        **Authors**: *Zhiliang Tian, Yingxiu Zhao, Ziyue Huang, Yu-Xiang Wang, Nevin L. Zhang, He He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/480045ad846b44bf31441c1f1d9dd768-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/480045ad846b44bf31441c1f1d9dd768-Abstract-Conference.html)

        **Abstract**:

        Protecting the privacy of user data is crucial for text generation models, which can leak sensitive information during generation. Differentially private (DP) learning methods provide guarantees against identifying the existence of a training sample from model outputs. PATE is a recent DP learning algorithm that achieves high utility with strong privacy protection on training samples. However, text generation models output tokens sequentially in a large output space; the classic PATE algorithm is not customized for this setting. Furthermore, PATE works well to protect sample-level privacy, but is not designed to protect phrases in samples. In this paper, we propose SeqPATE, an extension of PATE to text generation that protects the privacy of individual training samples and sensitive phrases in training data. To adapt PATE to text generation, we generate pseudo-contexts and reduce the sequence generation problem to a next-word prediction problem. To handle the large output space, we propose a candidate filtering strategy to dynamically reduce the output space, and refine the teacher aggregation of PATE to avoid low agreement due to voting for a large number of candidates. To further reduce privacy losses, we use knowledge distillation to reduce the number of teacher queries. The experiments verify the effectiveness of SeqPATE in protecting both training samples and sensitive phrases.

        ----

        ## [808] Bayesian Persuasion for Algorithmic Recourse

        **Authors**: *Keegan Harris, Valerie Chen, Joon Sik Kim, Ameet Talwalkar, Hoda Heidari, Zhiwei Steven Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/480150047ecb2187a3a8b8dccfd8f2de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/480150047ecb2187a3a8b8dccfd8f2de-Abstract-Conference.html)

        **Abstract**:

        When subjected to automated decision-making, decision subjects may strategically modify their observable features in ways they believe will maximize their chances of receiving a favorable decision. In many practical situations, the underlying assessment rule is deliberately kept secret to avoid gaming and maintain competitive advantage. The resulting opacity forces the decision subjects to rely on incomplete information when making strategic feature modifications. We capture such settings as a game of Bayesian persuasion, in which the decision maker offers a form of recourse to the decision subject by providing them with an action recommendation (or signal) to incentivize them to modify their features in desirable ways. We show that when using persuasion, the decision maker and decision subject are never worse off in expectation, while the decision maker can be significantly better off. While the decision makerâ€™s problem of finding the optimal Bayesian incentive compatible (BIC) signaling policy takes the form of optimization over infinitely many variables, we show that this optimization can be cast as a linear program over finitely-many regions of the space of possible assessment rules. While this reformulation simplifies the problem dramatically, solving the linear program requires reasoning about exponentially-many variables, even in relatively simple cases. Motivated by this observation, we provide a polynomial-time approximation scheme that recovers a near-optimal signaling policy. Finally, our numerical simulations on semi-synthetic data empirically demonstrate the benefits of using persuasion in the algorithmic recourse setting.

        ----

        ## [809] Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees

        **Authors**: *Jonathan Brophy, Daniel Lowd*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48088756ec0ce6ba362bddc7ebeb3915-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48088756ec0ce6ba362bddc7ebeb3915-Abstract-Conference.html)

        **Abstract**:

        Gradient-boosted regression trees (GBRTs) are hugely popular for solving tabular regression problems, but provide no estimate of uncertainty. We propose Instance-Based Uncertainty estimation for Gradient-boosted regression trees (IBUG), a simple method for extending any GBRT point predictor to produce probabilistic predictions. IBUG computes a non-parametric distribution around a prediction using the $k$-nearest training instances, where distance is measured with a tree-ensemble kernel. The runtime of IBUG depends on the number of training examples at each leaf in the ensemble, and can be improved by sampling trees or training instances. Empirically, we find that IBUG achieves similar or better performance than the previous state-of-the-art across 22 benchmark regression datasets. We also find that IBUG can achieve improved probabilistic performance by using different base GBRT models, and can more flexibly model the posterior distribution of a prediction than competing methods. We also find that previous methods suffer from poor probabilistic calibration on some datasets, which can be mitigated using a scalar factor tuned on the validation data. Source code is available at https://github.com/jjbrophy47/ibug.

        ----

        ## [810] Transcormer: Transformer for Sentence Scoring with Sliding Language Modeling

        **Authors**: *Kaitao Song, Yichong Leng, Xu Tan, Yicheng Zou, Tao Qin, Dongsheng Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/486ff0b164cf92b0255fe39863bcf99e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/486ff0b164cf92b0255fe39863bcf99e-Abstract-Conference.html)

        **Abstract**:

        Sentence scoring aims at measuring the likelihood score of a sentence and is widely used in many natural language processing scenarios, like reranking, which is to select the best sentence from multiple candidates. Previous works on sentence scoring mainly adopted either causal language modeling (CLM) like GPT or masked language modeling (MLM) like BERT, which have some limitations: 1) CLM only utilizes unidirectional information for the probability estimation of a sentence without considering bidirectional context, which affects the scoring quality; 2) MLM can only estimate the probability of partial tokens at a time and thus requires multiple forward passes to estimate the probability of the whole sentence, which incurs large computation and time cost. In this paper, we propose \textit{Transcormer} -- a Transformer model with a novel \textit{sliding language modeling} (SLM) for sentence scoring. Specifically, our SLM adopts a triple-stream self-attention mechanism to estimate the probability of all tokens in a sentence with bidirectional context and only requires a single forward pass. SLM can avoid the limitations of CLM (only unidirectional context) and MLM (multiple forward passes) and inherit their advantages, and thus achieve high effectiveness and efficiency in scoring. Experimental results on multiple tasks demonstrate that our method achieves better performance than other language modelings.

        ----

        ## [811] Models Out of Line: A Fourier Lens on Distribution Shift Robustness

        **Authors**: *Sara Fridovich-Keil, Brian R. Bartoldson, James Diffenderfer, Bhavya Kailkhura, Timo Bremer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48736dba3b8d933fabbfdb4f22a7be71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48736dba3b8d933fabbfdb4f22a7be71-Abstract-Conference.html)

        **Abstract**:

        Improving the accuracy of deep neural networks on out-of-distribution (OOD) data is critical to an acceptance of deep learning in real world applications. It has been observed that accuracies on in-distribution (ID) versus OOD data follow a linear trend and models that outperform this baseline are exceptionally rare (and referred to as ``effectively robust‚Äù). Recently, some promising approaches have been developed to improve OOD robustness: model pruning, data augmentation, and ensembling or zero-shot evaluating large pretrained models. However, there still is no clear understanding of the conditions on OOD data and model properties that are required to observe effective robustness. We approach this issue by conducting a comprehensive empirical study of diverse approaches that are known to impact OOD robustness on a broad range of natural and synthetic distribution shifts of CIFAR-10 and ImageNet. In particular, we view the "effective robustness puzzle" through a Fourier lens and ask how spectral properties of both models and OOD data correlate with OOD robustness. We find this Fourier lens offers some insight into why certain robust models, particularly those from the CLIP family, achieve OOD robustness. However, our analysis also makes clear that no known metric is consistently the best explanation of OOD robustness. Thus, to aid future research into the OOD puzzle, we address the gap in publicly-available models with effective robustness by introducing a set of pretrained CIFAR-10 models---$RobustNets$---with varying levels of OOD robustness.

        ----

        ## [812] Deep Learning Methods for Proximal Inference via Maximum Moment Restriction

        **Authors**: *Benjamin Kompa, David R. Bellamy, Thomas Kolokotrones, James M. Robins, Andrew Beam*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/487c9d6ef55e73aa9dfd4b48fe3713a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/487c9d6ef55e73aa9dfd4b48fe3713a6-Abstract-Conference.html)

        **Abstract**:

        The No Unmeasured Confounding Assumption is widely used to identify causal effects in observational studies. Recent work on proximal inference has provided alternative identification results that succeed even in the presence of unobserved confounders, provided that one has measured a sufficiently rich set of proxy variables, satisfying specific structural conditions. However, proximal inference requires solving an ill-posed integral equation. Previous approaches have used a variety of machine learning techniques to estimate a solution to this integral equation, commonly referred to as the bridge function. However, prior work has often been limited by relying on pre-specified kernel functions, which are not data adaptive and struggle to scale to large datasets. In this work, we introduce a flexible and scalable  method based on a deep neural network to estimate causal effects in the presence of unmeasured confounding using proximal inference. Our method achieves state of the art performance on two well-established proximal inference benchmarks. Finally, we provide theoretical consistency guarantees for our method.

        ----

        ## [813] Nest Your Adaptive Algorithm for Parameter-Agnostic Nonconvex Minimax Optimization

        **Authors**: *Junchi Yang, Xiang Li, Niao He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/488b8db9ec118c3d750c34d1812a5a3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/488b8db9ec118c3d750c34d1812a5a3a-Abstract-Conference.html)

        **Abstract**:

        Adaptive algorithms like AdaGrad and AMSGrad are successful in nonconvex optimization owing to their parameter-agnostic ability â€“ requiring no a priori knowledge about problem-specific parameters nor tuning of learning rates. However, when it comes to nonconvex minimax optimization, direct extensions of such adaptive optimizers without proper time-scale separation may fail to work in practice. We provide such an example proving that the simple combination of Gradient Descent Ascent (GDA) with adaptive stepsizes can diverge if the primal-dual stepsize ratio is not carefully chosen; hence, a fortiori, such adaptive extensions are not parameter-agnostic. To address the issue, we formally introduce a Nested Adaptive framework, NeAda for short, that carries an inner loop for adaptively maximizing the dual variable with controllable stopping criteria and an outer loop for adaptively minimizing the primal variable. Such mechanism can be equipped with off-the-shelf adaptive optimizers and automatically balance the progress in the primal and dual variables. Theoretically, for nonconvex-strongly-concave minimax problems, we show that NeAda with AdaGrad stepsizes can achieve the near-optimal $\widetilde{O}(\epsilon^{-2})$ and $\widetilde{O}(\epsilon^{-4})$ gradient complexities respectively in the deterministic and stochastic settings, without prior information on the problem's smoothness and strong concavity parameters. To the best of our knowledge, this is the first algorithm that simultaneously achieves near-optimal convergence rates and parameter-agnostic adaptation in the nonconvex minimax setting. Numerically, we further illustrate the robustness of the NeAda family with experiments on simple test functions and a real-world application.

        ----

        ## [814] Brownian Noise Reduction: Maximizing Privacy Subject to Accuracy Constraints

        **Authors**: *Justin Whitehouse, Aaditya Ramdas, Zhiwei Steven Wu, Ryan M. Rogers*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48aaa5ea741ae8430bd58e25917d267d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48aaa5ea741ae8430bd58e25917d267d-Abstract-Conference.html)

        **Abstract**:

        There is a disconnect between how researchers and practitioners handle privacy-utility tradeoffs. Researchers primarily operate from a privacy first perspective, setting strict privacy requirements and minimizing risk subject to these constraints. Practitioners often desire an accuracy first perspective, possibly satisfied with the greatest privacy they can get subject to obtaining sufficiently small error. Ligett et al. have introduced a `"noise reduction" algorithm to address the latter perspective. The authors show that by adding correlated Laplace noise and progressively reducing it on demand, it is possible to produce a sequence of increasingly accurate estimates of a private parameter and only pay a privacy cost for the least noisy iterate released. In this work, we generalize noise reduction to the setting of Gaussian noise, introducing the Brownian mechanism. The Brownian mechanism works by first adding Gaussian noise of high variance corresponding to the final point of a simulated Brownian motion. Then, at the practitioner's discretion, noise is gradually decreased by tracing back along the Brownian path to an earlier time. Our mechanism is more naturally applicable to the common setting of bounded $\ell_2$-sensitivity, empirically outperforms existing work on common statistical tasks, and provides customizable control of privacy loss over the entire interaction with the practitioner. We complement our Brownian mechanism with ReducedAboveThreshold, a generalization of the classical AboveThreshold algorithm that provides adaptive privacy guarantees. Overall, our results demonstrate that one can meet utility constraints while still maintaining strong levels of privacy.

        ----

        ## [815] Are AlphaZero-like Agents Robust to Adversarial Perturbations?

        **Authors**: *Li-Cheng Lan, Huan Zhang, Ti-Rong Wu, Meng-Yu Tsai, I-Chen Wu, Cho-Jui Hsieh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48adb34f7ee39177c4c23a8e4253a492-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48adb34f7ee39177c4c23a8e4253a492-Abstract-Conference.html)

        **Abstract**:

        The success of AlphaZero (AZ) has demonstrated that neural-network-based Go AIs can surpass human performance by a large margin. Given that the state space of Go is extremely large and a human player can play the game from any legal state, we ask whether adversarial states exist for Go AIs that may lead them to play surprisingly wrong actions.In this paper, we first extend the concept of adversarial examples to the game of Go: we generate perturbed states that are ``semantically'' equivalent to the original state by adding meaningless moves to the game, and an adversarial state is a perturbed state leading to an undoubtedly inferior action that is obvious even for Go beginners. However, searching the adversarial state is challenging due to the large, discrete, and non-differentiable search space. To tackle this challenge, we develop the first adversarial attack on Go AIs that can efficiently search for adversarial states by strategically reducing the search space. This method can also be extended to other board games such as NoGo. Experimentally, we show that the actions taken by both Policy-Value neural network (PV-NN) and Monte Carlo tree search (MCTS) can be misled by adding one or two meaningless stones; for example, on 58\% of the AlphaGo Zero self-play games, our method can make the widely used KataGo agent with 50 simulations of MCTS plays a losing action by adding two meaningless stones. We additionally evaluated the adversarial examples found by our algorithm with amateur human Go players, and 90\% of examples indeed lead the Go agent to play an obviously inferior action. Ourcode is available at \url{https://PaperCode.cc/GoAttack}.

        ----

        ## [816] AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos

        **Authors**: *Yanze Wu, Xintao Wang, Gen Li, Ying Shan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48cca987b3af66e1a607abd4820b330d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48cca987b3af66e1a607abd4820b330d-Abstract-Conference.html)

        **Abstract**:

        This paper studies the problem of real-world video super-resolution (VSR) for animation videos, and reveals three key improvements for practical animation VSR. First, recent real-world super-resolution approaches typically rely on degradation simulation using basic operators without any learning capability, such as blur, noise, and compression. In this work, we propose to learn such basic operators from real low-quality animation videos, and incorporate the learned ones into the degradation generation pipeline. Such neural-network-based basic operators could help to better capture the distribution of real degradations. Second, a large-scale high-quality animation video dataset, AVC, is built to facilitate comprehensive training and evaluations for animation VSR. Third, we further investigate an efficient multi-scale network structure. It takes advantage of the efficiency of unidirectional recurrent networks and the effectiveness of sliding-window-based methods. Thanks to the above delicate designs, our method, AnimeSR, is capable of restoring real-world low-quality animation videos effectively and efficiently, achieving superior performance to previous state-of-the-art methods.

        ----

        ## [817] On the Spectral Bias of Convolutional Neural Tangent and Gaussian Process Kernels

        **Authors**: *Amnon Geifman, Meirav Galun, David Jacobs, Ronen Basri*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/48fd58527b29c5c0ef2cae43065636e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/48fd58527b29c5c0ef2cae43065636e6-Abstract-Conference.html)

        **Abstract**:

        We study the properties of various over-parameterized convolutional neural architectures through their respective Gaussian Process and Neural Tangent kernels. We prove that, with normalized multi-channel input and ReLU activation, the eigenfunctions of these kernels with the uniform measure are formed by products of spherical harmonics, defined over the channels of the different pixels. We next use hierarchical factorizable kernels to bound their respective eigenvalues. We show that the eigenvalues decay polynomially, quantify the rate of decay, and derive measures that reflect the composition of hierarchical features in these networks. Our theory provides a concrete quantitative characterization of the role of locality and hierarchy in the inductive bias of over-parameterized convolutional network architectures.

        ----

        ## [818] Fairness Transferability Subject to Bounded Distribution Shift

        **Authors**: *Yatong Chen, Reilly Raab, Jialu Wang, Yang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4937610670be26d651ecdb4f2206d95f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4937610670be26d651ecdb4f2206d95f-Abstract-Conference.html)

        **Abstract**:

        Given an algorithmic predictor that is "fair"' on some source distribution, will it still be fair on an unknown target distribution that differs from the source within some bound? In this paper, we study the transferability of statistical group fairness for machine learning predictors (i.e., classifiers or regressors subject to bounded distribution shift. Such shifts may be introduced by initial training data uncertainties, user adaptation to a deployed predictor, dynamic environments, or the use of pre-trained models in new settings. Herein, we develop a bound that characterizes such transferability, flagging potentially inappropriate deployments of machine learning for socially consequential tasks. We first develop a framework for bounding violations of statistical fairness subject to distribution shift, formulating a generic upper bound for transferred fairness violations as our primary result.  We then develop bounds for specific worked examples, focusing on two commonly used fairness definitions (i.e., demographic parity and equalized odds) and two classes of distribution shift (i.e., covariate shift and label shift). Finally, we compare our theoretical bounds to deterministic models of distribution shift and against real-world data, finding that we are able to estimate fairness violation bounds in practice, even when simplifying assumptions are only approximately satisfied.

        ----

        ## [819] Improving Self-Supervised Learning by Characterizing Idealized Representations

        **Authors**: *Yann Dubois, Stefano Ermon, Tatsunori B. Hashimoto, Percy Liang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/494f876fad056843f310ad647274dd99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/494f876fad056843f310ad647274dd99-Abstract-Conference.html)

        **Abstract**:

        Despite the empirical successes of self-supervised learning (SSL) methods, it is unclear what characteristics of their representations lead to high downstream accuracies. In this work, we characterize properties that SSL representations should ideally satisfy. Specifically, we prove necessary and sufficient conditions such that for any task invariant to given data augmentations, probes (e.g., linear or MLP) trained on that representation attain perfect accuracy. These requirements lead to a unifying conceptual framework for improving existing SSL methods and deriving new ones. For contrastive learning, our framework prescribes simple but significant improvements to previous methods such as using asymmetric projection heads. For non-contrastive learning, we use our framework to derive a simple and novel objective. Our resulting SSL algorithms outperform baselines on standard benchmarks, including SwAV+multicrops on linear probing of ImageNet.

        ----

        ## [820] On the difficulty of learning chaotic dynamics with RNNs

        **Authors**: *Jonas M. Mikhaeil, Zahra Monfared, Daniel Durstewitz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/495e55f361708bedbab5d81f92048dcd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/495e55f361708bedbab5d81f92048dcd-Abstract-Conference.html)

        **Abstract**:

        Recurrent neural networks (RNNs) are wide-spread machine learning tools for modeling sequential and time series data. They are notoriously hard to train because their loss gradients backpropagated in time tend to saturate or diverge during training. This is known as the exploding and vanishing gradient problem. Previous solutions to this issue either built on rather complicated, purpose-engineered architectures with gated memory buffers, or - more recently - imposed constraints that ensure convergence to a fixed point or restrict (the eigenspectrum of) the recurrence matrix. Such constraints, however, convey severe limitations on the expressivity of the RNN. Essential intrinsic dynamics such as multistability or chaos are disabled. This is inherently at disaccord with the chaotic nature of many, if not most, time series encountered in nature and society. It is particularly problematic in scientific applications where one aims to reconstruct the underlying dynamical system. Here we offer a comprehensive theoretical treatment of this problem by relating the loss gradients during RNN training to the Lyapunov spectrum of RNN-generated orbits. We mathematically prove that RNNs producing stable equilibrium or cyclic behavior have bounded gradients, whereas the gradients of RNNs with chaotic dynamics always diverge. Based on these analyses and insights we suggest ways of how to optimize the training process on chaotic data according to the system's Lyapunov spectrum, regardless of the employed RNN architecture.

        ----

        ## [821] SKFlow: Learning Optical Flow with Super Kernels

        **Authors**: *Shangkun Sun, Yuanqi Chen, Yu Zhu, Guodong Guo, Ge Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4990dad2c1696224de42573d0222554a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4990dad2c1696224de42573d0222554a-Abstract-Conference.html)

        **Abstract**:

        Optical flow estimation is a classical yet challenging task in computer vision. One of the essential factors in accurately predicting optical flow is to alleviate occlusions between frames. However, it is still a thorny problem for current top-performing optical flow estimation methods due to insufficient local evidence to model occluded areas. In this paper, we propose the Super Kernel Flow Network (SKFlow), a CNN architecture to ameliorate the impacts of occlusions on optical flow estimation. SKFlow benefits from the super kernels which bring enlarged receptive fields to complement the absent matching information and recover the occluded motions. We present efficient super kernel designs by utilizing conical connections and hybrid depth-wise convolutions. Extensive experiments demonstrate the effectiveness of SKFlow on multiple benchmarks, especially in the occluded areas. Without pre-trained backbones on ImageNet and with a modest increase in computation, SKFlow achieves compelling performance and ranks $\textbf{1st}$ among currently published methods on the Sintel benchmark. On the challenging Sintel clean and final passes (test), SKFlow surpasses the best-published result in the unmatched areas ($7.96$ and $12.50$) by $9.09\%$ and $7.92\%$. The code is available at https://github.com/littlespray/SKFlow.

        ----

        ## [822] Mingling Foresight with Imagination: Model-Based Cooperative Multi-Agent Reinforcement Learning

        **Authors**: *Zhiwei Xu, Dapeng Li, Bin Zhang, Yuan Zhan, Yunpeng Bai, Guoliang Fan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html)

        **Abstract**:

        Recently, model-based agents have achieved better performance than model-free ones using the same computational budget and training time in single-agent environments. However, due to the complexity of multi-agent systems, it is tough to learn the model of the environment. The significant compounding error may hinder the learning process when model-based methods are applied to multi-agent tasks. This paper proposes an implicit model-based multi-agent reinforcement learning method based on value decomposition methods. Under this method, agents can interact with the learned virtual environment and evaluate the current state value according to imagined future states in the latent space, making agents have the foresight. Our approach can be applied to any multi-agent value decomposition method. The experimental results show that our method improves the sample efficiency in different partially observable Markov decision process domains.

        ----

        ## [823] End-to-end Stochastic Optimization with Energy-based Model

        **Authors**: *Lingkai Kong, Jiaming Cui, Yuchen Zhuang, Rui Feng, B. Aditya Prakash, Chao Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/49cf35ff2298c10452db99d08036805b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/49cf35ff2298c10452db99d08036805b-Abstract-Conference.html)

        **Abstract**:

        Decision-focused learning (DFL) was recently proposed for stochastic optimization problems that involve unknown parameters. By integrating predictive modeling with an implicitly differentiable optimization layer, DFL has shown superior performance to the standard two-stage predict-then-optimize pipeline. However, most existing DFL methods are only applicable to convex problems or a subset of nonconvex problems that can be easily relaxed to convex ones. Further, they can be inefficient in training due to the requirement of solving and differentiating through the optimization problem in every training iteration. We propose SO-EBM, a general and efficient DFL method for stochastic optimization using energy-based models. Instead of relying on KKT conditions to induce an implicit optimization layer, SO-EBM explicitly parameterizes the original optimization problem using a differentiable optimization layer based on energy functions. To better approximate the optimization landscape, we propose a coupled training objective that uses a maximum likelihood loss to capture the optimum location and a distribution-based regularizer to capture the overall energy landscape. Finally, we propose an efficient training procedure for SO-EBM with a self-normalized importance sampler based on a Gaussian mixture proposal. We evaluate SO-EBM in three applications: power scheduling, COVID-19 resource allocation, and non-convex adversarial security game, demonstrating the effectiveness and efficiency of SO-EBM.

        ----

        ## [824] Learning low-dimensional generalizable natural features from retina using a U-net

        **Authors**: *Siwei Wang, Benjamin Hoshal, Elizabeth de Laittre, Thierry Mora, Michael Berry, Stephanie E. Palmer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/49d608425f1bee2864e034a9e9e1ec9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/49d608425f1bee2864e034a9e9e1ec9e-Abstract-Conference.html)

        **Abstract**:

        Much of sensory neuroscience focuses on sensory features that are chosen by the experimenter because they are thought to be behaviorally relevant to the organism. However, it is not generally known what these features are in complex, natural scenes. This work focuses on using the retinal encoding of natural movies to determine the presumably behaviorally-relevant features that the brain represents. It is prohibitive to parameterize a natural movie and its respective retinal encoding fully. We use time within a natural movie as a proxy for the whole suite of features evolving across the scene. We then use a task-agnostic deep architecture, an encoder-decoder, to model the retinal encoding process and characterize its representation of ``time in the natural scene'' in a compressed latent space. In our end-to-end training, an encoder learns a compressed latent representation from a large population of salamander retinal ganglion cells responding to natural movies, while a decoder samples from this compressed latent space to generate the appropriate movie frame. By comparing latent representations of retinal activity from three movies, we find that the retina performs transfer learning to encode time: the precise, low-dimensional representation of time learned from one movie can be used to represent time in a different movie, with up to 17ms resolution. We then show that static textures and velocity features of a natural movie are synergistic. The retina simultaneously encodes both to establishes a generalizable, low-dimensional representation of time in the natural scene.

        ----

        ## [825] Split-kl and PAC-Bayes-split-kl Inequalities for Ternary Random Variables

        **Authors**: *Yi-Shan Wu, Yevgeny Seldin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/49ffa271264808cf500ea528ed8ec9b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/49ffa271264808cf500ea528ed8ec9b3-Abstract-Conference.html)

        **Abstract**:

        We present a new concentration of measure inequality for sums of independent bounded random variables, which we name a split-kl inequality. The inequality combines the combinatorial power of the kl inequality with ability to exploit low variance. While for Bernoulli random variables the kl inequality is tighter than the Empirical Bernstein, for random variables taking values inside a bounded interval and having low variance the Empirical Bernstein inequality is tighter than the kl. The proposed split-kl inequality yields the best of both worlds. We discuss an application of the split-kl inequality to bounding excess losses. We also derive a PAC-Bayes-split-kl inequality and use a synthetic example and several UCI datasets to compare it with the PAC-Bayes-kl, PAC-Bayes Empirical Bernstein, PAC-Bayes Unexpected Bernstein, and PAC-Bayes Empirical Bennett inequalities.

        ----

        ## [826] Wasserstein $K$-means for clustering probability distributions

        **Authors**: *Yubo Zhuang, Xiaohui Chen, Yun Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html)

        **Abstract**:

        Clustering is an important exploratory data analysis technique to group objects based on their similarity. The widely used $K$-means clustering method relies on some notion of distance to partition data into a fewer number of groups. In the Euclidean space, centroid-based and distance-based formulations of the $K$-means are equivalent. In modern machine learning applications, data often arise as probability distributions and a natural generalization to handle measure-valued data is to use the optimal transport metric. Due to non-negative Alexandrov curvature of the Wasserstein space, barycenters suffer from regularity and non-robustness issues. The peculiar behaviors of Wasserstein barycenters may make the centroid-based formulation fail to represent the within-cluster data points, while the more direct distance-based $K$-means approach and its semidefinite program (SDP) relaxation are capable of recovering the true cluster labels. In the special case of clustering Gaussian distributions, we show that the SDP relaxed Wasserstein $K$-means can achieve exact recovery given the clusters are well-separated under the $2$-Wasserstein metric. Our simulation and real data examples also demonstrate that distance-based $K$-means can achieve better classification performance over the standard centroid-based $K$-means for clustering probability distributions and images.

        ----

        ## [827] Learning Long-Term Crop Management Strategies with CyclesGym

        **Authors**: *Matteo Turchetta, Luca Corinzia, Scott Sussex, Amanda Burton, Juan Herrera, Ioannis Athanasiadis, Joachim M. Buhmann, Andreas Krause*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a22ceafe2dd6e0d32df1f7c0a69ab68-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a22ceafe2dd6e0d32df1f7c0a69ab68-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        To improve the sustainability and resilience of modern food systems, designing improved crop management strategies is crucial. The increasing abundance of data on agricultural systems suggests that future strategies could benefit from adapting to environmental conditions, but how to design these adaptive policies poses a new frontier. A natural technique for learning policies in these kinds of sequential decision-making problems is reinforcement learning (RL). To obtain the large number of samples required to learn effective RL policies, existing work has used mechanistic crop growth models (CGMs) as simulators. These solutions focus on single-year, single-crop simulations for learning strategies for a single agricultural management practice. However, to learn sustainable long-term policies we must be able to train in multi-year environments, with multiple crops, and consider a wider array of management techniques. We introduce CYCLESGYM, an RL environment based on the multi-year, multi-crop CGM Cycles. CYCLESGYM allows for long-term planning in agroecosystems, provides modular state space and reward constructors and weather generators, and allows for complex actions. For RL researchers, this is a novel benchmark to investigate issues arising in real-world applications. For agronomists, we demonstrate the potential of RL as a powerful optimization tool for agricultural systems management in multi-year case studies on nitrogen (N) fertilization and crop planning scenarios.

        ----

        ## [828] Identification, Amplification and Measurement: A bridge to Gaussian Differential Privacy

        **Authors**: *Yi Liu, Ke Sun, Bei Jiang, Linglong Kong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a29e8bc94b4c5d21d58a4fffdff800b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a29e8bc94b4c5d21d58a4fffdff800b-Abstract-Conference.html)

        **Abstract**:

        Gaussian differential privacy (GDP) is a single-parameter family of privacy notions that provides coherent guarantees to avoid the exposure of sensitive individual information. Despite the extra interpretability and tighter bounds under composition GDP provides, many widely used mechanisms (e.g., the Laplace mechanism) inherently provide GDP guarantees but often fail to take advantage of this new framework because their privacy guarantees were derived under a different background. In this paper, we study the asymptotic properties of privacy profiles and develop a simple criterion to identify algorithms with GDP properties. We propose an efficient method for GDP algorithms to narrow down possible values of an optimal privacy measurement, $\mu$ with an arbitrarily small and quantifiable margin of error. For non GDP algorithms, we provide a post-processing procedure that can amplify existing privacy guarantees to meet the GDP condition. As applications, we compare two single-parameter families of privacy notions, $\epsilon$-DP, and $\mu$-GDP, and show that all $\epsilon$-DP algorithms are intrinsically also GDP. Lastly, we show that the combination of our measurement process and the composition theorem of GDP is a powerful and convenient tool to handle compositions compared to the traditional standard and advanced composition theorems.

        ----

        ## [829] MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields

        **Authors**: *Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, Gábor Csányi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html)

        **Abstract**:

        Creating fast and accurate force fields is a long-standing challenge in computational chemistry and materials science. Recently, Equivariant Message Passing Neural Networks (MPNNs) have emerged as a powerful tool for building machine learning interatomic potentials, outperforming other approaches in terms of accuracy. However, they suffer from high computational cost and poor scalability. Moreover, most MPNNs only pass two-body messages leading to an intricate relationship between the number of layers and the expressivity of the features. This work introduces MACE, a new equivariant MPNN model that uses higher order messages, and demonstrates that this leads to an improved learning law. We show that by using four-body messages, the required number of message passing iterations reduces to just one, resulting in a fast and highly parallelizable model, reaching or exceeding state of the art accuracy on the rMD17 and 3BPA benchmark tasks. Our implementation is available at https://github.com/ACEsuit/mace.

        ----

        ## [830] Follow-the-Perturbed-Leader for Adversarial Markov Decision Processes with Bandit Feedback

        **Authors**: *Yan Dai, Haipeng Luo, Liyu Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a5c76c63f83ea45fb136d62db6c7104-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a5c76c63f83ea45fb136d62db6c7104-Abstract-Conference.html)

        **Abstract**:

        We consider regret minimization for Adversarial Markov Decision Processes (AMDPs), where the loss functions are changing over time and adversarially chosen, and the learner only observes the losses for the visited state-action pairs (i.e., bandit feedback). While there has been a surge of studies on this problem using Online-Mirror-Descent (OMD) methods, very little is known about the Follow-the-Perturbed-Leader (FTPL) methods, which are usually computationally more efficient and also easier to implement since it only requires solving an offline planning problem. Motivated by this, we take a closer look at FTPL for learning AMDPs, starting from the standard episodic finite-horizon setting. We find some unique and intriguing difficulties in the analysis and propose a workaround to eventually show that FTPL is also able to achieve near-optimal regret bounds in this case. More importantly, we then find two significant applications: First, the analysis of FTPL turns out to be readily generalizable to delayed bandit feedback with order-optimal regret, while OMD methods exhibit extra difficulties (Jin et al., 2022). Second, using FTPL, we also develop the first no-regret algorithm for learning communicating AMDPs in the infinite-horizon setting with bandit feedback and stochastic transitions. Our algorithm is efficient assuming access to an offline planning oracle, while even for the easier full-information setting, the only existing algorithm (Chandrasekaran and Tewari, 2021) is computationally inefficient.

        ----

        ## [831] Improving Multi-Task Generalization via Regularizing Spurious Correlation

        **Authors**: *Ziniu Hu, Zhe Zhao, Xinyang Yi, Tiansheng Yao, Lichan Hong, Yizhou Sun, Ed H. Chi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4a9eaf6dff3fdac9ab1aaf4c0fe2d563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4a9eaf6dff3fdac9ab1aaf4c0fe2d563-Abstract-Conference.html)

        **Abstract**:

        Multi-Task Learning (MTL) is a powerful learning paradigm to improve generalization performance via knowledge sharing. However, existing studies find that MTL could sometimes hurt generalization, especially when two tasks are less correlated. One possible reason that hurts generalization is spurious correlation, i.e., some knowledge is spurious and not causally related to task labels, but the model could mistakenly utilize them and thus fail when such correlation changes. In MTL setup, there exist several unique challenges of spurious correlation. First, the risk of having non-causal knowledge is higher, as the shared MTL model needs to encode all knowledge from different tasks, and causal knowledge for one task could be potentially spurious to the other. Second, the confounder between task labels brings in a different type of spurious correlation to MTL. Given such label-label confounders, we theoretically and empirically show that MTL is prone to taking non-causal knowledge from other tasks. To solve this problem, we propose Multi-Task Causal Representation Learning (MT-CRL) framework. MT-CRL aims to represent multi-task knowledge via disentangled neural modules, and learn which module is causally related to each task via MTL-specific invariant regularization. Experiments show that MT-CRL could enhance MTL model's performance by 5.5% on average over Multi-MNIST, MovieLens, Taskonomy, CityScape, and NYUv2, and show it could indeed alleviate spurious correlation problem.

        ----

        ## [832] Estimating and Explaining Model Performance When Both Covariates and Labels Shift

        **Authors**: *Lingjiao Chen, Matei Zaharia, James Y. Zou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4aa13186c795a52ba88f5b822f4b77eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4aa13186c795a52ba88f5b822f4b77eb-Abstract-Conference.html)

        **Abstract**:

        Deployed machine learning (ML) models often encounter new user data that differs from their training data. Therefore, estimating how well a given model might perform on the new data is an important step toward reliable ML applications. This is very challenging, however, as the data distribution can change in flexible ways, and we may not have any labels on the new data, which is often the case in monitoring settings. In this paper, we propose a new distribution shift model, Sparse Joint Shift (SJS), which considers the joint shift of both labels and a few features. This unifies and generalizes several existing shift models including label shift and sparse covariate shift, where only marginal feature or label distribution shifts are considered. We describe mathematical conditions under which SJS is identifiable. We further propose SEES, an algorithmic framework to characterize the distribution shift under SJS and to estimate a modelâ€™s performance on new data without any labels. We conduct extensive experiments on several real-world datasets with various ML models. Across different datasets and distribution shifts, SEES achieves significant (up to an order of magnitude) shift estimation error improvements over existing approaches.

        ----

        ## [833] Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems

        **Authors**: *Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong, Shujie Li, Lei Chen, Min Yang, Chenyun Yu, Bo Hu, Zang Li, Yu Xu, Xiaohu Qie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ad4fc1528374422dd7a69dea9e72948-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ad4fc1528374422dd7a69dea9e72948-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Existing benchmark datasets for recommender systems (RS)  either are created  at a small scale or involve very limited forms of user feedback. RS models evaluated on such datasets often lack practical values for large-scale real-world applications. In this paper, we describe Tenrec, a novel and publicly available data collection for RS that records various user feedback from four different recommendation scenarios. To be specific, Tenrec has the following five characteristics: (1) it is large-scale, containing around 5 million users and 140 million interactions; (2) it has not only positive user feedback, but also true  negative feedback (vs. one-class recommendation); (3) it contains overlapped users and items across four different scenarios; (4) it contains various types of  user positive feedback, in forms of clicking, liking, sharing, and following, etc; (5) it contains additional features beyond the user IDs and item IDs. We verify Tenrec on ten diverse  recommendation  tasks by running several classical baseline models per task. Tenrec has the potential to become a  useful benchmark dataset for a majority of popular recommendation tasks.  Our source codes and datasets will be included  in supplementary materials.

        ----

        ## [834] Joint Entropy Search For Maximally-Informed Bayesian Optimization

        **Authors**: *Carl Hvarfner, Frank Hutter, Luigi Nardi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b03821747e89ce803b2dac590f6a39b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b03821747e89ce803b2dac590f6a39b-Abstract-Conference.html)

        **Abstract**:

        Information-theoretic Bayesian optimization techniques have become popular for optimizing expensive-to-evaluate black-box functions due to their non-myopic qualities. Entropy Search and Predictive Entropy Search both consider the entropy over the optimum in the input space, while the recent Max-value Entropy Search considers the entropy over the optimal value in the output space. We propose Joint Entropy Search (JES), a novel information-theoretic acquisition function that considers an entirely new quantity, namely the entropy over the joint optimal probability density over both input and output space. To incorporate this information, we consider the reduction in entropy from conditioning on fantasized optimal input/output pairs. The resulting approach primarily relies on standard GP machinery and  removes complex approximations typically associated with information-theoretic methods. With minimal computational overhead, JES shows superior decision-making, and yields state-of-the-art performance for information-theoretic approaches across a wide suite of tasks. As a light-weight approach with superior results, JES provides a new go-to acquisition function for Bayesian optimization.

        ----

        ## [835] Provable General Function Class Representation Learning in Multitask Bandits and MDP

        **Authors**: *Rui Lu, Andrew Zhao, Simon S. Du, Gao Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b121e627d3c5683f312ad168988f3f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b121e627d3c5683f312ad168988f3f0-Abstract-Conference.html)

        **Abstract**:

        While multitask representation learning has become a popular approach in reinforcement learning (RL) to boost the sample efficiency, the theoretical understanding of why and how it works is still limited. Most previous analytical works could only assume that the representation function is already known to the agent or from linear function class, since analyzing general function class representation encounters non-trivial technical obstacles such as generalization guarantee, formulation of confidence bound in abstract function space, etc. However, linear-case analysis heavily relies on the particularity of linear function class, while real-world practice usually adopts general non-linear representation functions like neural networks. This significantly reduces its applicability. In this work, we extend the analysis to general function class representations. Specifically, we consider an agent playing $M$ contextual bandits (or MDPs) concurrently and extracting a shared representation function $\phi$ from a specific function class $\Phi$ using our proposed Generalized Functional Upper Confidence Bound algorithm (GFUCB). We theoretically validate the benefit of multitask representation learning within general function class for bandits and linear MDP for the first time. Lastly, we conduct experiments to demonstrate the effectiveness of our algorithm with neural net representation.

        ----

        ## [836] Weighted Mutual Learning with Diversity-Driven Model Compression

        **Authors**: *Miao Zhang, Li Wang, David Campos, Wei Huang, Chenjuan Guo, Bin Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html)

        **Abstract**:

        Online distillation attracts attention from the community as it simplifies the traditional two-stage knowledge distillation process into a single stage. Online distillation collaboratively trains a group of peer models, which are treated as students, and all students gain extra knowledge from each other. However, memory consumption and diversity among peers are two key challenges to the scalability and quality of online distillation. To address the two challenges, this paper presents a framework called Weighted Mutual Learning with Diversity-Driven Model Compression (WML) for online distillation. First, at the base of a hierarchical structure where peers share different parts, we leverage the structured network pruning to generate diversified peer models and reduce the memory requirements. Second, rather than taking the average of peers, this paper, for the first time, leverages a bi-level formulation to estimate the relative importance of peers with a close-form, to further boost the effectiveness of the distillation from each other. Extensive experiments show the generalization of the proposed framework, which outperforms existing online distillation methods on a variety of deep neural networks. More interesting, as a byproduct, \WML produces a series of pruned models under different model sizes in a single run, which also achieves competitive results compared with existing channel pruning methods.

        ----

        ## [837] S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

        **Authors**: *Daesol Cho, Dongseok Shim, H. Jin Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b32c2943a02331792877cc6b5205f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b32c2943a02331792877cc6b5205f49-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (Offline RL) suffers from the innate distributional shift as it cannot interact with the physical environment during training. To alleviate such limitation, state-based offline RL leverages a learned dynamics model from the logged experience and augments the predicted state transition to extend the data distribution. For exploiting such benefit also on the image-based RL, we firstly propose a generative model, S2P (State2Pixel), which synthesizes the raw pixel of the agent from its corresponding state. It enables bridging the gap between the state and the image domain in RL algorithms, and virtually exploring unseen image distribution via model-based transition in the state space. Through experiments, we confirm that our S2P-based image synthesis not only improves the image-based offline RL performance but also shows powerful generalization capability on unseen tasks.

        ----

        ## [838] Neural Collapse with Normalized Features: A Geometric Analysis over the Riemannian Manifold

        **Authors**: *Can Yaras, Peng Wang, Zhihui Zhu, Laura Balzano, Qing Qu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b3cc0d1c897ebcf71aca92a4a26ac83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b3cc0d1c897ebcf71aca92a4a26ac83-Abstract-Conference.html)

        **Abstract**:

        When training overparameterized deep networks for classification tasks, it has been widely observed that the learned features exhibit a so-called "neural collapse'" phenomenon. More specifically, for the output features of the penultimate layer, for each class the within-class features converge to their means, and the means of different classes exhibit a certain tight frame structure, which is also aligned with the last layer's classifier. As feature normalization in the last layer becomes a common practice in modern representation learning, in this work we theoretically justify the neural collapse phenomenon under normalized features. Based on an unconstrained feature model, we simplify the empirical loss function in a multi-class classification task into a nonconvex optimization problem over the Riemannian manifold by constraining all features and classifiers over the sphere. In this context, we analyze the nonconvex landscape of the Riemannian optimization problem over the product of spheres, showing a benign global landscape in the sense that the only global minimizers are the neural collapse solutions while all other critical points are strict saddle points with negative curvature. Experimental results on practical deep networks corroborate our theory and demonstrate that better representations can be learned faster via feature normalization. Code for our experiments can be found at https://github.com/cjyaras/normalized-neural-collapse.

        ----

        ## [839] Conformalized Fairness via Quantile Regression

        **Authors**: *Meichen Liu, Lei Ding, Dengdeng Yu, Wulong Liu, Linglong Kong, Bei Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b52b3c50110fc10f6a1a86055682ea2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b52b3c50110fc10f6a1a86055682ea2-Abstract-Conference.html)

        **Abstract**:

        Algorithmic fairness has received increased attention in socially sensitive domains. While rich literature on mean fairness has been established, research on quantile fairness remains sparse but vital. To fulfill great needs and advocate the significance of quantile fairness, we propose a novel framework to learn a real-valued quantile function under the fairness requirement of Demographic Parity with respect to sensitive attributes, such as race or gender, and thereby derive a reliable fair prediction interval. Using optimal transport and functional synchronization techniques, we establish theoretical guarantees of distribution-free coverage and exact fairness for the induced prediction interval constructed by fair quantiles. A hands-on pipeline is provided to incorporate flexible quantile regressions with an efficient fairness adjustment post-processing algorithm. We demonstrate the superior empirical performance of this approach on several benchmark datasets. Our results show the modelâ€™s ability to uncover the mechanism underlying the fairness-accuracy trade-off in a wide range of societal and medical applications.

        ----

        ## [840] Efficient Methods for Non-stationary Online Learning

        **Authors**: *Peng Zhao, Yan-Feng Xie, Lijun Zhang, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b70484ebef62484e0c8cdd269e482fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b70484ebef62484e0c8cdd269e482fd-Abstract-Conference.html)

        **Abstract**:

        Non-stationary online learning has drawn much attention in recent years. In particular, \emph{dynamic regret} and \emph{adaptive regret} are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity--those methods typically maintain $O(\log T)$ base-learners simultaneously for a $T$-round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from $O(\log T)$ to $1$.  Moreover, our obtained algorithms require only one gradient query and one function evaluation at each round. Our technique hinges on the reduction mechanism developed in parameter-free online learning and requires non-trivial twists on non-stationary online methods. Empirical studies verify our theoretical findings.

        ----

        ## [841] Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces

        **Authors**: *Leonard Papenmeier, Luigi Nardi, Matthias Poloczek*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b7439a4ab0b8e4bcb4e2412c6a10a58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b7439a4ab0b8e4bcb4e2412c6a10a58-Abstract-Conference.html)

        **Abstract**:

        Recent advances have extended the scope of Bayesian optimization (BO) to expensive-to-evaluate black-box functions with dozens of dimensions, aspiring to unlock impactful applications, for example, in the life sciences, neural architecture search, and robotics. However, a closer examination reveals that the state-of-the-art methods for high-dimensional Bayesian optimization (HDBO) suffer from degrading performance as the number of dimensions increases, or even risk failure if certain unverifiable assumptions are not met. This paper proposes BAxUS that leverages a novel family of nested random subspaces to adapt the space it optimizes over to the problem. This ensures high performance while removing the risk of failure, which we assert via theoretical guarantees. A comprehensive evaluation demonstrates that BAxUS achieves better results than the state-of-the-art methods for a broad set of applications.

        ----

        ## [842] ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection

        **Authors**: *Huiping Zhuang, Zhenyu Weng, Hongxin Wei, Renchunzi Xie, Kar-Ann Toh, Zhiping Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html)

        **Abstract**:

        Class-incremental learning (CIL) learns a classification model with training data of different classes arising progressively. Existing CIL either suffers from serious accuracy loss due to catastrophic forgetting, or invades data privacy by revisiting used exemplars. Inspired by learning of linear problems, we propose an analytic class-incremental learning (ACIL) with absolute memorization of past knowledge  while avoiding breaching of data privacy (i.e., without storing historical data). The absolute memorization is demonstrated in the sense that the CIL using ACIL given present data would give identical results to that from its joint-learning counterpart that consumes both present and historical samples. This equality is theoretically validated. The data privacy is ensured by showing that no historical data are involved during the learning process. Empirical validations demonstrate ACIL's competitive accuracy performance with near-identical results for various incremental task settings (e.g., 5-50 phases). This also allows ACIL to outperform the state-of-the-art methods for large-phase scenarios (e.g., 25 and 50 phases).

        ----

        ## [843] MsSVT: Mixed-scale Sparse Voxel Transformer for 3D Object Detection on Point Clouds

        **Authors**: *Shaocong Dong, Lihe Ding, Haiyang Wang, Tingfa Xu, Xinli Xu, Jie Wang, Ziyang Bian, Ying Wang, Jianan Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4bad7c27534efca029ca0d366c47c0e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4bad7c27534efca029ca0d366c47c0e3-Abstract-Conference.html)

        **Abstract**:

        3D object detection from the LiDAR point cloud is fundamental to autonomous driving. Large-scale outdoor scenes usually feature significant variance in instance scales, thus requiring features rich in long-range and fine-grained information to support accurate detection. Recent detectors leverage the power of window-based transformers to model long-range dependencies but tend to blur out fine-grained details. To mitigate this gap, we present a novel Mixed-scale Sparse Voxel Transformer, named MsSVT, which can well capture both types of information simultaneously by the divide-and-conquer philosophy. Specifically, MsSVT explicitly divides attention heads into multiple groups, each in charge of attending to information within a particular range. All groups' output is merged to obtain the final mixed-scale features. Moreover, we provide a novel chessboard sampling strategy to reduce the computational complexity of applying a window-based transformer in 3D voxel space. To improve efficiency, we also implement the voxel sampling and gathering operations sparsely with a hash map. Endowed by the powerful capability and high efficiency of modeling mixed-scale information, our single-stage detector built on top of MsSVT surprisingly outperforms state-of-the-art two-stage detectors on Waymo. Our project page: https://github.com/dscdyc/MsSVT.

        ----

        ## [844] Deterministic Langevin Monte Carlo with Normalizing Flows for Bayesian Inference

        **Authors**: *Richard D. P. Grumitt, Biwei Dai, Uros Seljak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4bbdef62653d8088717640e7660a1ebb-Abstract-Conference.html)

        **Abstract**:

        We propose a general purpose Bayesian inference algorithm for expensive likelihoods, replacing the stochastic term in the Langevin equation with a deterministic density gradient term. The particle density is evaluated from the current particle positions using a Normalizing Flow (NF), which is differentiable and has good generalization properties in high dimensions. We take advantage of NF preconditioning and NF based Metropolis-Hastings updates for a faster convergence. We show on various examples that the method is competitive against state of the art sampling methods.

        ----

        ## [845] Evolution of Neural Tangent Kernels under Benign and Adversarial Training

        **Authors**: *Noel Loo, Ramin M. Hasani, Alexander Amini, Daniela Rus*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4bc4e9ecd5ae4a75048dc216a770cba1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4bc4e9ecd5ae4a75048dc216a770cba1-Abstract-Conference.html)

        **Abstract**:

        Two key challenges facing modern deep learning is mitigating deep networks vulnerability to adversarial attacks, and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen and the underlying feature map is fixed. In finite-widths however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work which studies adversarial robustness of NTK during training.  In this work, we perform an empirical study of the evolution of the NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with $76.1\%$ robust accuracy under PGD attacks with $\varepsilon = 4/255$ on CIFAR-10.

        ----

        ## [846] Zero-Sum Stochastic Stackelberg Games

        **Authors**: *Denizalp Goktas, Sadie Zhao, Amy Greenwald*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4beaed6a33716fcfe7b5250d10520eb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4beaed6a33716fcfe7b5250d10520eb9-Abstract-Conference.html)

        **Abstract**:

        Zero-sum stochastic games have found important applications in a variety of fields, from machine learning to economics. Work on this model has primarily focused on the computation of Nash equilibrium due to its effectiveness in solving adversarial board and video games. Unfortunately, a Nash equilibrium is not guaranteed to exist in zero-sum stochastic games when the payoffs at each state are not convex-concave in the players' actions. A Stackelberg equilibrium, however, is guaranteed to exist. Consequently, in this paper, we study zero-sum stochastic Stackelberg games. Going beyond known existence results for (non-stationary) Stackelberg equilibria, we prove the existence of recursive (i.e., Markov perfect) Stackelberg equilibria (recSE) in these games, provide necessary and sufficient conditions for a policy profile to be a recSE, and show that recSE can be computed in (weakly) polynomial time via value iteration. Finally, we show that zero-sum stochastic Stackelberg games can model the problem of pricing and allocating goods across agents and time. More specifically, we propose a zero-sum stochastic Stackelberg game whose recSE correspond to the recursive competitive equilibria of a large class of stochastic Fisher markets. We close with a series of experiments that showcase how our methodology can be used to solve the consumption-savings problem in stochastic Fisher markets.

        ----

        ## [847] Evaluating Out-of-Distribution Performance on Document Image Classifiers

        **Authors**: *Stefan Larson, Gordon Lim, Yutong Ai, David Kuang, Kevin Leach*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The ability of a document classifier to handle inputs that are drawn from a distribution different from the training distribution is crucial for robust deployment and generalizability. The RVL-CDIP corpus is the de facto standard benchmark for document classification, yet to our knowledge all studies that use this corpus do not include evaluation on out-of-distribution documents. In this paper, we curate and release a new out-of-distribution benchmark for evaluating out-of-distribution performance for document classifiers. Our new out-of-distribution benchmark consists of two types of documents: those that are not part of any of the 16 in-domain RVL-CDIP categories (RVL-CDIP-O), and those that are one of the 16 in-domain categories yet are drawn from a distribution different from that of the original RVL-CDIP dataset (RVL-CDIP-N). While prior work on document classification for in-domain RVL-CDIP documents reports high accuracy scores, we find that these models exhibit accuracy drops of between roughly 15-30% on our new out-of-domain RVL-CDIP-N benchmark, and further struggle to distinguish between in-domain RVL-CDIP-N and out-of-domain RVL-CDIP-O inputs. Our new benchmark provides researchers with a valuable new resource for analyzing out-of-distribution performance on document classifiers.

        ----

        ## [848] Distributed Learning of Conditional Quantiles in the Reproducing Kernel Hilbert Space

        **Authors**: *Heng Lian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4c12e97f2e05304a451e18c9c945036f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4c12e97f2e05304a451e18c9c945036f-Abstract-Conference.html)

        **Abstract**:

        We study distributed learning of nonparametric conditional quantiles with Tikhonov regularization in a reproducing kernel Hilbert space (RKHS). Although distributed parametric quantile regression has been investigated in several existing works, the current nonparametric quantile setting poses different challenges and is still unexplored. The difficulty lies in the illusive explicit bias-variance decomposition in the quantile RKHS setting as in the regularized least squares regression. For the simple divide-and-conquer approach that partitions the data set into multiple parts and then takes an arithmetic average of the individual outputs, we establish the risk bounds using a novel second-order empirical process for quantile risk.

        ----

        ## [849] Spatial Mixture-of-Experts

        **Authors**: *Nikoli Dryden, Torsten Hoefler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html)

        **Abstract**:

        Many data have an underlying dependence on spatial location; it may be weather on the Earth, a simulation on a mesh, or a registered image. Yet this feature is rarely taken advantage of, and violates common assumptions made by many neural network layers, such as translation equivariance. Further, many works that do incorporate locality fail to capture fine-grained structure. To address this, we introduce the Spatial Mixture-of-Experts (SMoE) layer, a sparsely-gated layer that learns spatial structure in the input domain and routes experts at a fine-grained level to utilize it. We also develop new techniques to train SMoEs, including a self-supervised routing loss and damping expert errors. Finally, we show strong results for SMoEs on numerous tasks, and set new state-of-the-art results for medium-range weather prediction and post-processing ensemble weather forecasts.

        ----

        ## [850] Amortized Mixing Coupling Processes for Clustering

        **Authors**: *Huafeng Liu, Liping Jing*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4c91cf13f8827a1b46656439e32ff74b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4c91cf13f8827a1b46656439e32ff74b-Abstract-Conference.html)

        **Abstract**:

        Considering the ever-increasing scale of data, which may contain tens of thousands of data points or complicated latent structures, the issue of scalability and algorithmic efficiency becomes of vital importance for clustering. In this paper, we propose cluster-wise amortized mixing coupling processes (AMCP), which is able to achieve efficient amortized clustering in a well-defined non-parametric Bayesian posterior. Specifically, AMCP learns clusters sequentially with the aid of the proposed intra-cluster mixing (IntraCM) and inter-cluster coupling (InterCC) strategies, which investigate the relationship between data points and reference distribution in a linear optimal transport mixing view, and coupling the unassigned set and assigned set to generate new cluster. IntraCM and InterCC avoid pairwise calculation of distances between clusters and reduce the computational complexity from quadratic to linear in the current number of clusters. Furthermore, cluster-wise sequential process is able to improve the quick adaptation ability for the next cluster generation. In this case, AMCP simultaneously learns what makes a cluster, how to group data points into clusters, and how to adaptively control the number of clusters. To illustrate the superiority of the proposed method, we perform experiments on both synthetic data and real-world data in terms of clustering performance and computational efficiency. The source code is available at https://github.com/HuafengHK/AMCP.

        ----

        ## [851] Hilbert Distillation for Cross-Dimensionality Networks

        **Authors**: *Dian Qin, Haishuai Wang, Zhe Liu, Hongjia Xu, Sheng Zhou, Jiajun Bu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4c9477b9e2c7ec0ad3f4f15077aaf85a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4c9477b9e2c7ec0ad3f4f15077aaf85a-Abstract-Conference.html)

        **Abstract**:

        3D convolutional neural networks have revealed superior performance in processing volumetric data such as video and medical imaging. However, the competitive performance by leveraging 3D networks results in huge computational costs, which are far beyond that of 2D networks. In this paper, we propose a novel Hilbert curve-based cross-dimensionality distillation approach that facilitates the knowledge of 3D networks to improve the performance of 2D networks. The proposed Hilbert Distillation (HD) method preserves the structural information via the Hilbert curve, which maps high-dimensional (>=2) representations to one-dimensional continuous space-filling curves. Since the distilled 2D networks are supervised by the curves converted from dimensionally heterogeneous 3D features, the 2D networks are given an informative view in terms of learning structural information embedded in well-trained high-dimensional representations. We further propose a Variable-length Hilbert Distillation (VHD) method to dynamically shorten the walking stride of the Hilbert curve in activation feature areas and lengthen the stride in context feature areas, forcing the 2D networks to pay more attention to learning from activation features. The proposed algorithm outperforms the current state-of-the-art distillation techniques adapted to cross-dimensionality distillation on two classification tasks. Moreover, the distilled 2D networks by the proposed method achieve competitive performance with the original 3D networks, indicating the lightweight distilled 2D networks could potentially be the substitution of cumbersome 3D networks in the real-world scenario.

        ----

        ## [852] Provably Efficient Offline Multi-agent Reinforcement Learning via Strategy-wise Bonus

        **Authors**: *Qiwen Cui, Simon S. Du*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4cca5640267b416cef4f00630aef93a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4cca5640267b416cef4f00630aef93a2-Abstract-Conference.html)

        **Abstract**:

        This paper considers offline multi-agent reinforcement learning. We propose the strategy-wise concentration principle which directly builds a confidence interval for the joint strategy, in contrast to the point-wise concentration principle which builds a confidence interval for each point in the joint action space. For two-player zero-sum Markov games, by exploiting the convexity of the strategy-wise bonus, we propose a computationally efficient algorithm whose sample complexity enjoys a better dependency on the number of actions than the prior methods based on the point-wise bonus. Furthermore, for offline multi-agent general-sum Markov games,  based on the strategy-wise bonus and a novel surrogate function, we give the first algorithm whose sample complexity only scales $\sum_{i=1}^m A_i$ where $A_i$ is the action size of the $i$-th player and $m$ is the number of players. In sharp contrast, the sample complexity of methods based on the point-wise bonus would scale with the size of the joint action space $\Pi_{i=1}^m A_i$ due to the curse of multiagents. Lastly, all of our algorithms can naturally take a pre-specified strategy class $\Pi$ as input and output a strategy that is close to the best strategy in $\Pi$. In this setting, the sample complexity only scales with $\log |\Pi|$ instead of $\sum_{i=1}^m A_i$.

        ----

        ## [853] A Best-of-Both-Worlds Algorithm for Bandits with Delayed Feedback

        **Authors**: *Saeed Masoudian, Julian Zimmert, Yevgeny Seldin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4cd1d0d9892e4136fe86c97b89f77c6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4cd1d0d9892e4136fe86c97b89f77c6b-Abstract-Conference.html)

        **Abstract**:

        We present a modified tuning of the algorithm of  Zimmert and Seldin [2020] for adversarial multiarmed bandits with delayed feedback, which in addition to the minimax optimal adversarial regret guarantee shown by Zimmert and Seldin [2020] simultaneously achieves a near-optimal regret guarantee in the stochastic setting with fixed delays. Specifically, the adversarial regret guarantee is $\mathcal{O}(\sqrt{TK} + \sqrt{dT\log K})$, where $T$ is the time horizon, $K$ is the number of arms, and $d$ is the fixed delay, whereas the stochastic regret guarantee is $\mathcal{O}\left(\sum_{i \neq i^*}(\frac{1}{\Delta_i} \log(T) + \frac{d}{\Delta_{i}}) + d K^{1/3}\log K\right)$, where $\Delta_i$ are the suboptimality gaps. We also present an extension of the algorithm to the case of arbitrary delays, which is based on an oracle knowledge of the maximal delay $d_{max}$ and achieves $\mathcal{O}(\sqrt{TK} + \sqrt{D\log K} + d_{max}K^{1/3} \log K)$ regret in the adversarial regime, where $D$ is the total delay, and $\mathcal{O}\left(\sum_{i \neq i^*}(\frac{1}{\Delta_i} \log(T) + \frac{\sigma_{max}}{\Delta_{i}}) + d_{max}K^{1/3}\log K\right)$ regret in the stochastic regime, where $\sigma_{max}$ is the maximal number of outstanding observations. Finally, we present a lower bound that matches regret upper bound achieved by the skipping technique of  Zimmert and Seldin [2020] in the adversarial setting.

        ----

        ## [854] LIFT: Language-Interfaced Fine-Tuning for Non-language Machine Learning Tasks

        **Authors**: *Tuan Dinh, Yuchen Zeng, Ruisu Zhang, Ziqian Lin, Michael Gira, Shashank Rajput, Jy-yong Sohn, Dimitris S. Papailiopoulos, Kangwook Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ce7fe1d2730f53cb3857032952cd1b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ce7fe1d2730f53cb3857032952cd1b8-Abstract-Conference.html)

        **Abstract**:

        Fine-tuning pretrained language models (LMs) without making any architectural changes has become a norm for learning various language downstream tasks. However, for non-language downstream tasks, a common practice is to employ task-specific designs for input, output layers, and loss functions. For instance, it is possible to fine-tune an LM into an MNIST classifier by replacing the word embedding layer with an image patch embedding layer, the word token output layer with a 10-way output layer, and the word prediction loss with a 10-way classification loss, respectively. A natural question arises: Can LM fine-tuning solve non-language downstream tasks without changing the model architecture or loss function? To answer this, we propose Language-Interfaced Fine-Tuning (LIFT) and study its efficacy and limitations by conducting an extensive empirical study on a suite of non-language classification and regression tasks. LIFT does not make any changes to the model architecture or loss function, and it solely relies on the natural language interface, enabling "no-code machine learning with LMs."  We find that LIFT performs comparably well across a wide range of low-dimensional classification and regression tasks, matching the performances of the best baselines in many cases, especially for the classification tasks. We also report experimental results on the fundamental properties of LIFT, including inductive bias, robustness, and sample complexity. We also analyze the effect of pretraining on LIFT and a few properties/techniques specific to LIFT, e.g., context-aware learning via appropriate prompting, calibrated predictions, data generation, and two-stage fine-tuning. Our code is available at https://github.com/UW-Madison-Lee-Lab/LanguageInterfacedFineTuning.

        ----

        ## [855] Capturing Failures of Large Language Models via Human Cognitive Biases

        **Authors**: *Erik Jones, Jacob Steinhardt*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d13b2d99519c5415661dad44ab7edcd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d13b2d99519c5415661dad44ab7edcd-Abstract-Conference.html)

        **Abstract**:

        Large language models generate complex, open-ended outputs: instead of outputting a class label they write summaries, generate dialogue, or produce working code. In order to asses the reliability of these open-ended generation systems, we aim to identify qualitative categories of erroneous behavior, beyond identifying individual errors. To hypothesize and test for such qualitative errors, we draw inspiration from human cognitive biases---systematic patterns of deviation from rational judgement. Specifically, we use cognitive biases as motivation to (i) generate hypotheses for problems that models may have, and (ii) develop experiments that elicit these problems. Using code generation as a case study, we find that OpenAIâ€™s Codex errs predictably based on how the input prompt is framed, adjusts outputs towards anchors, and is biased towards outputs that mimic frequent training examples. We then use our framework to elicit high-impact errors such as incorrectly deleting files. Our results indicate that experimental methodology from cognitive science can help characterize how machine learning systems behave.

        ----

        ## [856] Template based Graph Neural Network with Optimal Transport Distances

        **Authors**: *Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d3525bc60ba1adc72336c0392d3d902-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d3525bc60ba1adc72336c0392d3d902-Abstract-Conference.html)

        **Abstract**:

        Current Graph Neural Networks (GNN) architectures generally rely on two important components: node features embedding through message passing, and aggregation with a specialized form of pooling. The structural (or topological) information is implicitly taken into account in these two steps. We propose in this work a novel point of view, which places distances to some learnable graph templates at the core of the graph representation. This distance embedding is constructed thanks to an optimal transport distance: the Fused Gromov-Wasserstein (FGW) distance, which encodes simultaneously feature and structure dissimilarities by solving a soft graph-matching problem. We postulate that the vector of FGW distances to a set of template graphs has a strong discriminative power, which is then fed to a non-linear classifier for final predictions. Distance embedding can be seen as a new layer, and can leverage on existing message passing techniques to promote sensible feature representations. Interestingly enough, in our work the optimal set of template graphs is also learnt in  an end-to-end fashion by differentiating through this layer. After describing the corresponding learning procedure, we empirically validate our claim on several synthetic and real life graph classification datasets, where our method is competitive or surpasses kernel and GNN state-of-the-art approaches. We complete our experiments by an ablation study and a sensitivity analysis to parameters.

        ----

        ## [857] Knowledge Distillation Improves Graph Structure Augmentation for Graph Neural Networks

        **Authors**: *Lirong Wu, Haitao Lin, Yufei Huang, Stan Z. Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d4a3b6a34332d80349137bcc98164a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d4a3b6a34332d80349137bcc98164a5-Abstract-Conference.html)

        **Abstract**:

        Graph (structure) augmentation aims to perturb the graph structure through heuristic or probabilistic rules, enabling the nodes to capture richer contextual information and thus improving generalization performance. While there have been a few graph structure augmentation methods proposed recently, none of them are aware of a potential negative augmentation problem, which may be caused by overly severe distribution shifts between the original and augmented graphs. In this paper, we take an important graph property, namely graph homophily, to analyze the distribution shifts between the two graphs and thus measure the severity of an augmentation algorithm suffering from negative augmentation. To tackle this problem, we propose a novel Knowledge Distillation for Graph Augmentation (KDGA) framework, which helps to reduce the potential negative effects of distribution shifts, i.e., negative augmentation problem. Specifically, KDGA extracts the knowledge of any GNN teacher model trained on the augmented graphs and injects it into a partially parameter-shared student model that is tested on the original graph. As a simple but efficient framework, KDGA is applicable to a variety of existing graph augmentation methods and can significantly improve the performance of various GNN architectures. For three popular graph augmentation methods, namely GAUG, MH-Aug, and GraphAug, the experimental results show that the learned student models outperform their vanilla implementations by an average accuracy of 4.6% (GAUG), 4.2% (MH-Aug), and 4.6% (GraphAug) on eight graph datasets.

        ----

        ## [858] Learning Invariant Graph Representations for Out-of-Distribution Generalization

        **Authors**: *Haoyang Li, Ziwei Zhang, Xin Wang, Wenwu Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d4e0ab9d8ff180bf5b95c258842d16e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d4e0ab9d8ff180bf5b95c258842d16e-Abstract-Conference.html)

        **Abstract**:

        Graph representation learning has shown effectiveness when testing and training graph data come from the same distribution, but most existing approaches fail to generalize under distribution shifts. Invariant learning, backed by the invariance principle from causality, can achieve guaranteed generalization under distribution shifts in theory and has shown great successes in practice. However, invariant learning for graphs under distribution shifts remains unexplored and challenging. To solve this problem, we propose Graph Invariant Learning (GIL) model capable of learning generalized graph representations under distribution shifts. Our proposed method can capture the invariant relationships between predictive graph structural information and labels in a mixture of latent environments through jointly optimizing three tailored modules. Specifically, we first design a GNN-based subgraph generator to identify invariant subgraphs. Then we use the variant subgraphs, i.e., complements of invariant subgraphs, to infer the latent environment labels. We further propose an invariant learning module to learn graph representations that can generalize to unknown test graphs. Theoretical justifications for our proposed method are also provided. Extensive experiments on both synthetic and real-world datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts for the graph classification task.

        ----

        ## [859] A Quadrature Rule combining Control Variates and Adaptive Importance Sampling

        **Authors**: *Rémi Leluc, François Portier, Johan Segers, Aigerim Zhuman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d4e8614a37f0aff841ba87ed1a898c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d4e8614a37f0aff841ba87ed1a898c1-Abstract-Conference.html)

        **Abstract**:

        Driven by several successful applications such as in stochastic gradient descent or in Bayesian computation, control variates have become a major tool for Monte Carlo integration. However, standard methods do not allow the distribution of the particles to evolve during the algorithm, as is the case in  sequential simulation methods. Within the standard adaptive importance sampling framework, a simple weighted least squares approach is proposed to improve the procedure with control variates. The procedure takes the form of a quadrature rule with adapted quadrature weights to reflect the information brought in by the control variates. The quadrature points and weights do not depend on the integrand, a computational advantage in case of multiple integrands. Moreover, the target density needs to be known only up to a multiplicative constant. Our main result is a non-asymptotic bound on the probabilistic error of the procedure. The bound proves that for improving the estimate's accuracy, the benefits from adaptive importance sampling and control variates can be combined. The good behavior of the method is illustrated empirically on synthetic examples and real-world data for Bayesian linear regression.

        ----

        ## [860] GAL: Gradient Assisted Learning for Decentralized Multi-Organization Collaborations

        **Authors**: *Enmao Diao, Jie Ding, Vahid Tarokh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d6938f94ab47d32128c239a4bfedae0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d6938f94ab47d32128c239a4bfedae0-Abstract-Conference.html)

        **Abstract**:

        Collaborations among multiple organizations, such as financial institutions, medical centers, and retail markets in decentralized settings are crucial to providing improved service and performance. However, the underlying organizations may have little interest in sharing their local data, models, and objective functions. These requirements have created new challenges for multi-organization collaboration. In this work, we propose Gradient Assisted Learning (GAL), a new method for multiple organizations to assist each other in supervised learning tasks without sharing local data, models, and objective functions. In this framework, all participants collaboratively optimize the aggregate of local loss functions, and each participant autonomously builds its own model by iteratively fitting the gradients of the overarching objective function. We also provide asymptotic convergence analysis and practical case studies of GAL. Experimental studies demonstrate that GAL can achieve performance close to centralized learning when all data, models, and objective functions are fully disclosed.

        ----

        ## [861] Direct Advantage Estimation

        **Authors**: *Hsiao-Ru Pan, Nico Gürtler, Alexander Neitz, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html)

        **Abstract**:

        The predominant approach in reinforcement learning is to assign credit to actions based on the expected return. However, we show that the return may depend on the policy in a way which could lead to excessive variance in value estimation and slow down learning. Instead, we show that the advantage function can be interpreted as causal effects and shares similar properties with causal representations. Based on this insight, we propose Direct Advantage Estimation (DAE), a novel method that can model the advantage function and estimate it directly from on-policy data while simultaneously minimizing the variance of the return without requiring the (action-)value function. We also relate our method to Temporal Difference methods by showing how value functions can be seamlessly integrated into DAE. The proposed method is easy to implement and can be readily adapted by modern actor-critic methods. We evaluate DAE empirically on three discrete control domains and show that it can outperform generalized advantage estimation (GAE), a strong baseline for advantage estimation, on a majority of the environments when applied to policy optimization.

        ----

        ## [862] Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning

        **Authors**: *Hua Wei, Jingxiao Chen, Xiyang Ji, Hongyang Qin, Minwen Deng, Siqin Li, Liang Wang, Weinan Zhang, Yong Yu, Liu Lin, Lanxiao Huang, Deheng Ye, Qiang Fu, Wei Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        This paper introduces Honor of Kings Arena, a reinforcement learning (RL) environment based on the Honor of Kings, one of the worldâ€™s most popular games at present. Compared to other environments studied in most previous work, ours presents new generalization challenges for competitive reinforcement learning. It is a multi-agent problem with one agent competing against its opponent; and it requires the generalization ability as it has diverse targets to control and diverse opponents to compete with. We describe the observation, action, and reward specifications for the Honor of Kings domain and provide an open-source Python-based interface for communicating with the game engine. We provide twenty target heroes with a variety of tasks in Honor of Kings Arena and present initial baseline results for RL-based methods with feasible computing resources.  Finally, we showcase the generalization challenges imposed by Honor of Kings Arena and possible remedies to the challenges. All of the software, including the environment-class, are publicly available.

        ----

        ## [863] On the Symmetries of Deep Learning Models and their Internal Representations

        **Authors**: *Charles Godfrey, Davis Brown, Tegan Emerson, Henry Kvinge*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4df3510ad02a86d69dc32388d91606f8-Abstract-Conference.html)

        **Abstract**:

        Symmetry has been a fundamental tool in the exploration of a broad range of complex systems. In machine learning, symmetry has been explored in both models and data. In this paper we seek to connect the symmetries arising from the architecture of a family of models with the symmetries of that family’s internal representation of data. We do this by calculating a set of fundamental symmetry groups, which we call the intertwiner groups of the model. Each of these arises from a particular nonlinear layer of the model and different nonlinearities result in different symmetry groups. These groups change the weights of a model in such a way that the underlying function that the model represents remains constant but the internal representations of data inside the model may change. We connect intertwiner groups to a model’s internal representations of data through a range of experiments that probe similarities between hidden states across models with the same architecture. Our work suggests that the symmetries of a network are propagated into the symmetries in that network’s representation of data, providing us with a better understanding of how architecture affects the learning and prediction process. Finally, we speculate that for ReLU networks, the intertwiner groups may provide a justification for the common practice of concentrating model interpretability exploration on the activation basis in hidden layers rather than arbitrary linear combinations thereof.

        ----

        ## [864] TabNAS: Rejection Sampling for Neural Architecture Search on Tabular Datasets

        **Authors**: *Chengrun Yang, Gabriel Bender, Hanxiao Liu, Pieter-Jan Kindermans, Madeleine Udell, Yifeng Lu, Quoc V. Le, Da Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e392aa9bc70ed731d3c9c32810f92fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e392aa9bc70ed731d3c9c32810f92fb-Abstract-Conference.html)

        **Abstract**:

        The best neural architecture for a given machine learning problem depends on many factors: not only the complexity and structure of the dataset, but also on resource constraints including latency, compute, energy consumption, etc. Neural architecture search (NAS) for tabular datasets is an important but under-explored problem. Previous NAS algorithms designed for image search spaces incorporate resource constraints directly into the reinforcement learning (RL) rewards. However, for NAS on tabular datasets, this protocol often discovers suboptimal architectures. This paper develops TabNAS, a new and more effective approach to handle resource constraints in tabular NAS using an RL controller motivated by the idea of rejection sampling. TabNAS immediately discards any architecture that violates the resource constraints without training or learning from that architecture. TabNAS uses a Monte-Carlo-based correction to the RL policy gradient update to account for this extra filtering step. Results on several tabular datasets demonstrate the superiority of TabNAS over previous reward-shaping methods: it finds better models that obey the constraints.

        ----

        ## [865] Embed and Emulate: Learning to estimate parameters of dynamical systems with uncertainty quantification

        **Authors**: *Ruoxi Jiang, Rebecca Willett*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e580cdd54fe38ca9a5b8ea6fe99bb44-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e580cdd54fe38ca9a5b8ea6fe99bb44-Abstract-Conference.html)

        **Abstract**:

        This paper explores learning emulators for parameter estimation with uncertainty estimation of high-dimensional dynamical systems. We assume access to a computationally complex simulator that inputs a candidate parameter and outputs a corresponding multi-channel time series. Our task is to accurately estimate a range of likely values of the underlying parameters. Standard iterative approaches necessitate running the simulator many times, which is computationally prohibitive. This paper describes a novel framework for learning feature embeddings of observed dynamics jointly with an emulator that can replace high-cost simulators. Leveraging a contrastive learning approach, our method exploits intrinsic data properties within and across parameter and trajectory domains. On a coupled 396-dimensional multiscale Lorenz 96 system, our method significantly outperforms a typical parameter estimation method based on predefined metrics and a classical numerical simulator, and with only 1.19% of the baseline's computation time. Ablation studies highlight the potential of explicitly designing learned emulators for parameter estimation by leveraging contrastive learning.

        ----

        ## [866] A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval

        **Authors**: *Hao Li, Jingkuan Song, Lianli Gao, Pengpeng Zeng, Haonan Zhang, Gongfu Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e786a87e7ae249de2b1aeaf5d8fde82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e786a87e7ae249de2b1aeaf5d8fde82-Abstract-Conference.html)

        **Abstract**:

        Cross-modal retrieval aims to build correspondence between multiple modalities by learning a common representation space. Typically, an image can match multiple texts semantically and vice versa, which significantly increases the difficulty of this task. To address this problem, probabilistic embedding is proposed to quantify these many-to-many relationships. However, existing datasets (e.g., MS-COCO) and metrics (e.g., Recall@K) cannot fully represent these diversity correspondences due to non-exhaustive annotations. Based on this observation, we utilize semantic correlation computed by CIDEr to find the potential correspondences. Then we present an effective metric, named Average Semantic Precision (ASP), which can measure the ranking precision of semantic correlation for retrieval sets. Additionally, we introduce a novel and concise objective, coined Differentiable ASP Approximation (DAA). Concretely, DAA can optimize ASP directly by making the ranking function of ASP differentiable through a sigmoid function. To verify the effectiveness of our approach, extensive experiments are conducted on MS-COCO, CUB Captions, and Flickr30K, which are commonly used in cross-modal retrieval. The results show that our approach obtains superior performance over the state-of-the-art approaches on all metrics. The code and trained models are released at https://github.com/leolee99/2022-NeurIPS-DAA.

        ----

        ## [867] Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attack

        **Authors**: *Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e81308aa2eb8e2e4eccf122d4827af7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e81308aa2eb8e2e4eccf122d4827af7-Abstract-Conference.html)

        **Abstract**:

        A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they ofteneither drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.

        ----

        ## [868] Dict-TTS: Learning to Pronounce with Prior Dictionary Knowledge for Text-to-Speech

        **Authors**: *Ziyue Jiang, Su Zhe, Zhou Zhao, Qian Yang, Yi Ren, Jinglin Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html)

        **Abstract**:

        Polyphone disambiguation aims to capture accurate pronunciation knowledge from natural text sequences for reliable Text-to-speech (TTS) systems. However, previous approaches require substantial annotated training data and additional efforts from language experts, making it difficult to extend high-quality neural TTS systems to out-of-domain daily conversations and countless languages worldwide. This paper tackles the polyphone disambiguation problem from a concise and novel perspective: we propose Dict-TTS, a semantic-aware generative text-to-speech model with an online website dictionary (the existing prior information in the natural language). Specifically, we design a semantics-to-pronunciation attention (S2PA) module to match the semantic patterns between the input text sequence and the prior semantics in the dictionary and obtain the corresponding pronunciations; The S2PA module can be easily trained with the end-to-end TTS model without any annotated phoneme labels. Experimental results in three languages show that our model outperforms several strong baseline models in terms of pronunciation accuracy and improves the prosody modeling of TTS systems. Further extensive analyses demonstrate that each design in Dict-TTS is effective. The code is available at https://github.com/Zain-Jiang/Dict-TTS.

        ----

        ## [869] AZ-whiteness test: a test for signal uncorrelation on spatio-temporal graphs

        **Authors**: *Daniele Zambon, Cesare Alippi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html)

        **Abstract**:

        We present the first whiteness hypothesis test for graphs, i.e., a whiteness test for multivariate time series associated with the nodes of a dynamic graph; as such, the test represents an important model assessment tool for graph deep learning, e.g., in forecasting setups. The statistical test aims at detecting existing serial dependencies among close-in-time observations, as well as spatial dependencies among neighboring observations given the underlying graph. The proposed AZ-test can be intended as a spatio-temporal extension of traditional tests designed for system identification to graph signals. The AZ-test is versatile, allowing the underlying graph to be dynamic, changing in topology and set of nodes over time, and weighted, thus accounting for connections of different strength, as it is the case in many application scenarios like sensor and transportation networks. The asymptotic distribution of the designed test can be derived under the null hypothesis without assuming identically distributed data. We show the effectiveness of the test on both synthetic and real-world problems, and illustrate how it can be employed to assess the quality of spatio-temporal forecasting models by analyzing the prediction residuals appended to the graph stream.

        ----

        ## [870] ViSioNS: Visual Search in Natural Scenes Benchmark

        **Authors**: *Fermín Travi, Gonzalo Ruarte, Gastón Bujia, Juan E. Kamienkowski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ea14e6090343523ddcd5d3ca449695f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ea14e6090343523ddcd5d3ca449695f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Visual search is an essential part of almost any everyday human interaction with the visual environment. Nowadays, several algorithms are able to predict gaze positions during simple observation, but few models attempt to simulate human behavior during visual search in natural scenes. Furthermore, these models vary widely in their design and exhibit differences in the datasets and metrics with which they were evaluated. Thus, there is a need for a reference point, on which each model can be tested and from where potential improvements can be derived. In this study, we select publicly available state-of-the-art visual search models and datasets in natural scenes, and provide a common framework for their evaluation. To this end, we apply a unified format and criteria, bridging the gaps between them, and we estimate the models’ efficiency and similarity with humans using a specific set of metrics. This integration has allowed us to enhance the Ideal Bayesian Searcher by combining it with a neural network-based visual search model, which enables it to generalize to other datasets. The present work sheds light on the limitations of current models and how integrating different approaches with a unified criteria can lead to better algorithms. Moreover, it moves forward on bringing forth a solution for the urgent need for benchmarking data and metrics to support the development of more general human visual search computational models. All of the code used here, including metrics, plots, and visual search models, alongside the preprocessed datasets, are available at $\url{https://github.com/FerminT/VisualSearchBenchmark}$.

        ----

        ## [871] Value Function Decomposition for Iterative Design of Reinforcement Learning Agents

        **Authors**: *James MacGlashan, Evan Archer, Alisa Devlic, Takuma Seno, Craig Sherstan, Peter R. Wurman, Peter Stone*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4eb2c0adafbe71269f3a772c130f9e53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4eb2c0adafbe71269f3a772c130f9e53-Abstract-Conference.html)

        **Abstract**:

        Designing reinforcement learning (RL) agents is typically a difficult process that requires numerous design iterations. Learning can fail for a multitude of reasons and standard RL methods provide too few tools to provide insight into the exact cause. In this paper, we show how to integrate \textit{value decomposition} into a broad class of actor-critic algorithms and use it to assist in the iterative agent-design process. Value decomposition separates a reward function into distinct components and learns value estimates for each. These value estimates provide insight into an agent's learning and decision-making process and enable new training methods to mitigate common problems. As a demonstration, we introduce SAC-D, a variant of soft actor-critic (SAC) adapted for value decomposition. SAC-D maintains similar performance to SAC, while learning a larger set of value predictions. We also introduce decomposition-based tools that exploit this information, including a new reward \textit{influence} metric, which measures each reward component's effect on agent decision-making. Using these tools, we provide several demonstrations of decomposition's use in identifying and addressing problems in the design of both environments and agents. Value decomposition is broadly applicable and easy to incorporate into existing algorithms and workflows, making it a powerful tool in an RL practitioner's toolbox.

        ----

        ## [872] HandMeThat: Human-Robot Communication in Physical and Social Environments

        **Authors**: *Yanming Wan, Jiayuan Mao, Josh Tenenbaum*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4eb33c53ed5b14ce9028309431f565cc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4eb33c53ed5b14ce9028309431f565cc-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce HandMeThat, a benchmark for a holistic evaluation of instruction understanding and following in physical and social environments. While previous datasets primarily focused on language grounding and planning, HandMeThat considers the resolution of human instructions with ambiguities based on the physical (object states and relations) and social (human actions and goals) information. HandMeThat contains 10,000 episodes of human-robot interactions. In each episode, the robot first observes a trajectory of human actions towards her internal goal. Next, the robot receives a human instruction and should take actions to accomplish the subgoal set through the instruction. In this paper, we present a textual interface for our benchmark, where the robot interacts with a virtual environment through textual commands. We evaluate several baseline models on HandMeThat, and show that both offline and online reinforcement learning algorithms perform poorly on HandMeThat, suggesting significant room for future work on physical and social human-robot communications and interactions.

        ----

        ## [873] Task-Agnostic Graph Explanations

        **Authors**: *Yaochen Xie, Sumeet Katariya, Xianfeng Tang, Edward W. Huang, Nikhil Rao, Karthik Subbian, Shuiwang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4eb7f0abf16d08e50ed42beb1e22e782-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4eb7f0abf16d08e50ed42beb1e22e782-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have emerged as powerful tools to encode graph-structured data. Due to their broad applications, there is an increasing need to develop tools to explain how GNNs make decisions given graph-structured data. Existing learning-based GNN explanation approaches are task-specific in training and hence suffer from crucial drawbacks. Specifically, they are incapable of producing explanations for a multitask prediction model with a single explainer. They are also unable to provide explanations in cases where the GNN is trained in a self-supervised manner, and the resulting representations are used in future downstream tasks. To address these limitations, we propose a Task-Agnostic GNN Explainer (TAGE) that is independent of downstream models and trained under self-supervision with no knowledge of downstream tasks. TAGE enables the explanation of GNN embedding models with unseen downstream tasks and allows efficient explanation of multitask models. Our extensive experiments show that TAGE can significantly speed up the explanation efficiency by using the same model to explain predictions for multiple downstream tasks while achieving explanation quality as good as or even better than current state-of-the-art GNN explanation approaches.

        ----

        ## [874] Embrace the Gap: VAEs Perform Independent Mechanism Analysis

        **Authors**: *Patrik Reizinger, Luigi Gresele, Jack Brady, Julius von Kügelgen, Dominik Zietlow, Bernhard Schölkopf, Georg Martius, Wieland Brendel, Michel Besserve*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4eb91efe090f72f7cf42c69aab03fe85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4eb91efe090f72f7cf42c69aab03fe85-Abstract-Conference.html)

        **Abstract**:

        Variational autoencoders (VAEs) are a popular framework for modeling complex data distributions; they can be efficiently trained via variational inference by maximizing the evidence lower bound (ELBO), at the expense of a gap to the exact (log-)marginal likelihood. While VAEs are commonly used for representation learning, it is unclear why ELBO maximization would yield useful representations, since unregularized maximum likelihood estimation cannot invert the data-generating process. Yet, VAEs often succeed at this task. We seek to elucidate this apparent paradox by studying nonlinear VAEs in the limit of near-deterministic decoders. We first prove that, in this regime, the optimal encoder approximately inverts the decoder---a commonly used but unproven conjecture---which we refer to as self-consistency. Leveraging self-consistency, we show that the ELBO converges to a regularized log-likelihood. This allows VAEs to perform what has recently been termed independent mechanism analysis (IMA): it adds an inductive bias towards decoders with column-orthogonal Jacobians, which helps recovering the true latent factors. The gap between ELBO and log-likelihood is therefore welcome, since it bears unanticipated benefits for nonlinear representation learning. In experiments on synthetic and image data, we show that VAEs uncover the true latent factors when the data generating process satisfies the IMA assumption.

        ----

        ## [875] Dataset Inference for Self-Supervised Models

        **Authors**: *Adam Dziedzic, Haonan Duan, Muhammad Ahmad Kaleem, Nikita Dhawan, Jonas Guan, Yannis Cattan, Franziska Boenisch, Nicolas Papernot*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ebf0617b32da2cd083c3b17c7285cce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ebf0617b32da2cd083c3b17c7285cce-Abstract-Conference.html)

        **Abstract**:

        Self-supervised models are increasingly prevalent in machine learning (ML) since they reduce the need for expensively labeled data. Because of their versatility in downstream applications, they are increasingly used as a service exposed via public APIs. At the same time, these encoder models are particularly vulnerable to model stealing attacks due to the high dimensionality of vector representations they output. Yet, encoders remain undefended: existing mitigation strategies for stealing attacks focus on supervised learning. We introduce a new dataset inference defense, which uses the private training set of the victim encoder model to attribute its ownership in the event of stealing. The intuition is that the log-likelihood of an encoder's output representations is higher on the victim's training data than on test data if it is stolen from the victim, but not if it is independently trained. We compute this log-likelihood using density estimation models. As part of our evaluation, we also propose measuring the fidelity of stolen encoders and quantifying the effectiveness of the theft detection without involving downstream tasks; instead, we leverage mutual information and distance measurements. Our extensive empirical results in the vision domain demonstrate that dataset inference is a promising direction for defending self-supervised models against model stealing.

        ----

        ## [876] Statistically Meaningful Approximation: a Case Study on Approximating Turing Machines with Transformers

        **Authors**: *Colin Wei, Yining Chen, Tengyu Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ebf1d74f53ece08512a23309d58df89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ebf1d74f53ece08512a23309d58df89-Abstract-Conference.html)

        **Abstract**:

        A common lens to theoretically study neural net architectures is to analyze the functions they can approximate. However, the constructions from approximation theory often have unrealistic aspects, for example, reliance on infinite precision to memorize target function values. To address this issue, we propose a formal definition of statistically meaningful approximation which requires the approximating network to exhibit good statistical learnability. We present case studies on statistically meaningful approximation for two classes of functions: boolean circuits and Turing machines. We show that overparameterized feedforward neural nets can statistically meaningfully approximate boolean circuits with sample complexity depending only polynomially on the circuit size, not the size of the approximating network. In addition, we show that transformers can statistically meaningfully approximate Turing machines with computation time bounded by T, requiring sample complexity polynomial in the alphabet size, state space size, and log(T). Our analysis introduces new tools for generalization bounds that provide much tighter sample complexity guarantees than the typical VC-dimension or norm-based bounds, which may be of independent interest.

        ----

        ## [877] Improved Feature Distillation via Projector Ensemble

        **Authors**: *Yudong Chen, Sen Wang, Jiajun Liu, Xuwei Xu, Frank de Hoog, Zi Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ec0b6648bdf487a2f1c815924339022-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ec0b6648bdf487a2f1c815924339022-Abstract-Conference.html)

        **Abstract**:

        In knowledge distillation, previous feature distillation methods mainly focus on the design of loss functions and the selection of the distilled layers, while the effect of the feature projector between the student and the teacher remains under-explored. In this paper, we first discuss a plausible mechanism of the projector with empirical evidence and then propose a new feature distillation method based on a projector ensemble for further performance improvement. We observe that the student network benefits from a projector even if the feature dimensions of the student and the teacher are the same. Training a student backbone without a projector can be considered as a multi-task learning process, namely achieving discriminative feature extraction for classification and feature matching between the student and the teacher for distillation at the same time. We hypothesize and empirically verify that without a projector, the student network tends to overfit the teacher's feature distributions despite having different architecture and weights initialization. This leads to degradation on the quality of the student's deep features that are eventually used in classification. Adding a projector, on the other hand, disentangles the two learning tasks and helps the student network to focus better on the main feature extraction task while still being able to utilize teacher features as a guidance through the projector. Motivated by the positive effect of the projector in feature distillation, we propose an ensemble of projectors to further improve the quality of student features. Experimental results on different datasets with a series of teacher-student pairs illustrate the effectiveness of the proposed method. Code is available at https://github.com/chenyd7/PEFD.

        ----

        ## [878] Does GNN Pretraining Help Molecular Representation?

        **Authors**: *Ruoxi Sun, Hanjun Dai, Adams Wei Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)

        **Abstract**:

        Extracting informative representations of molecules using Graph neural networks (GNNs) is crucial in AI-driven drug discovery. Recently, the graph research community has been trying to replicate the success of self-supervised pretraining in natural language processing, with several successes claimed. However, we find the benefit brought by self-supervised pretraining on small molecular data can be negligible in many cases. We conduct thorough ablation studies on the key components of GNN pretraining, including pretraining objectives, data splitting methods, input features, pretraining dataset scales, and GNN architectures, to see how they affect the accuracy of the downstream tasks. Our first important finding is, self-supervised graph pretraining do not always have statistically significant advantages over non-pretraining methods in many settings. Secondly, although noticeable improvement can be observed with additional supervised pretraining, the improvement may diminish with richer features or more balanced data splits. Thirdly, hyper-parameters could have larger impacts on accuracy of downstream tasks than the choice of pretraining tasks, especially when the scales of downstream tasks are small. Finally, we provide our conjectures where the complexity of some pretraining methods on small molecules might be insufficient, followed by empirical evidences on different pretraining datasets.

        ----

        ## [879] ULNeF: Untangled Layered Neural Fields for Mix-and-Match Virtual Try-On

        **Authors**: *Igor Santesteban, Miguel A. Otaduy, Nils Thuerey, Dan Casas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ee3ac2cd119023c79b0d21c4a464dc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ee3ac2cd119023c79b0d21c4a464dc7-Abstract-Conference.html)

        **Abstract**:

        Recent advances in neural models have shown great results for virtual try-on (VTO) problems, where a 3D representation of a garment is deformed to fit a target body shape. However, current solutions are limited to a single garment layer, and cannot address the combinatorial complexity of mixing different garments. Motivated by this limitation, we investigate the use of neural fields for mix-and-match VTO, and identify and solve a fundamental challenge that existing neural-field methods cannot address: the interaction between layered neural fields. To this end, we propose a neural model that untangles layered neural fields to represent collision-free garment surfaces. The key ingredient is a neural untangling projection operator that works directly on the layered neural fields, not on explicit surface representations. Algorithms to resolve object-object interaction are inherently limited by the use of explicit geometric representations, and we show how methods that work directly on neural implicit representations could bring a change of paradigm and open the door to radically different approaches.

        ----

        ## [880] Introspective Learning : A Two-Stage approach for Inference in Neural Networks

        **Authors**: *Mohit Prabhushankar, Ghassan AlRegib*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4eef032250ac525903063cd760cb0480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4eef032250ac525903063cd760cb0480-Abstract-Conference.html)

        **Abstract**:

        In this paper, we advocate for two stages in a neural network's decision making process. The first is the existing feed-forward inference framework where patterns in given data are sensed and associated with previously learned patterns. The second stage is a slower reflection stage where we ask the network to reflect on its feed-forward decision by considering and evaluating all available choices. Together, we term the two stages as introspective learning. We use gradients of trained neural networks as a measurement of this reflection. A simple three-layered Multi Layer Perceptron is used as the second stage that predicts based on all extracted gradient features. We perceptually visualize the post-hoc explanations from both stages to provide a visual grounding to introspection. For the application of recognition, we show that an introspective network is 4% more robust and 42% less prone to calibration errors when generalizing to noisy data. We also illustrate the value of introspective networks in downstream tasks that require generalizability and calibration including active learning, out-of-distribution detection, and uncertainty estimation. Finally, we ground the proposed machine introspection to human introspection for the application of image quality assessment.

        ----

        ## [881] Bayesian Active Learning with Fully Bayesian Gaussian Processes

        **Authors**: *Christoffer Riis, Francisco Antunes, Frederik Boe Hüttel, Carlos Lima Azevedo, Francisco Pereira*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f1fba885f266d87653900fd3045e8af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f1fba885f266d87653900fd3045e8af-Abstract-Conference.html)

        **Abstract**:

        The bias-variance trade-off is a well-known problem in machine learning that only gets more pronounced the less available data there is. In active learning, where labeled data is scarce or difficult to obtain, neglecting this trade-off can cause inefficient and non-optimal querying, leading to unnecessary data labeling. In this paper, we focus on active learning with Gaussian Processes (GPs). For the GP, the bias-variance trade-off is made by optimization of the two hyperparameters: the length scale and noise-term. Considering that the optimal mode of the joint posterior of the hyperparameters is equivalent to the optimal bias-variance trade-off, we approximate this joint posterior and utilize it to design two new acquisition functions. The first one is a Bayesian variant of Query-by-Committee (B-QBC), and the second is an extension that explicitly minimizes the predictive variance through a Query by Mixture of Gaussian Processes (QB-MGP) formulation. Across six simulators, we empirically show that B-QBC, on average, achieves the best marginal likelihood, whereas QB-MGP achieves the best predictive performance. We show that incorporating the bias-variance trade-off in the acquisition functions mitigates unnecessary and expensive data labeling.

        ----

        ## [882] E-MAPP: Efficient Multi-Agent Reinforcement Learning with Parallel Program Guidance

        **Authors**: *Can Chang, Ni Mu, Jiajun Wu, Ling Pan, Huazhe Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f2accafe6fa355624f3ee42207cc7b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f2accafe6fa355624f3ee42207cc7b8-Abstract-Conference.html)

        **Abstract**:

        A critical challenge in multi-agent reinforcement learning(MARL) is for multiple agents to efficiently accomplish complex, long-horizon tasks. The agents often have difficulties in cooperating on common goals, dividing complex tasks, and planning through several stages to make progress. We propose to address these challenges by guiding agents with programs designed for parallelization, since programs as a representation contain rich structural and semantic information, and are widely used as abstractions for long-horizon tasks. Specifically, we introduce Efficient Multi-Agent Reinforcement Learning with Parallel Program Guidance(E-MAPP), a novel framework that leverages parallel programs to guide multiple agents to efficiently accomplish goals that require planning over $10+$ stages. E-MAPP integrates the structural information from a parallel program, promotes the cooperative behaviors grounded in program semantics, and improves the time efficiency via a task allocator. We conduct extensive experiments on a series of challenging, long-horizon cooperative tasks in the Overcooked environment. Results show that E-MAPP outperforms strong baselines in terms of the completion rate, time efficiency, and zero-shot generalization ability by a large margin.

        ----

        ## [883] In Defense of the Unitary Scalarization for Deep Multi-Task Learning

        **Authors**: *Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, Pawan Kumar Mudigonda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f301ae934f396086bfefd1139039dbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f301ae934f396086bfefd1139039dbd-Abstract-Conference.html)

        **Abstract**:

        Recent multi-task learning research argues against unitary scalarization, where training simply minimizes the sum of the task losses. Several ad-hoc multi-task optimization algorithms have instead been proposed, inspired by various hypotheses about what makes multi-task settings difficult.  The majority of these optimizers require per-task gradients, and introduce significant memory, runtime, and implementation overhead. We show that unitary scalarization, coupled with standard regularization and stabilization techniques from single-task learning, matches or improves upon the performance of complex multi-task optimizers in popular supervised and reinforcement learning settings. We then present an analysis suggesting that many specialized multi-task optimizers can be partly interpreted as forms of regularization, potentially explaining our surprising results. We believe our results call for a critical reevaluation of recent research in the area.

        ----

        ## [884] Flexible Neural Image Compression via Code Editing

        **Authors**: *Chenjian Gao, Tongda Xu, Dailan He, Yan Wang, Hongwei Qin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f3820576130a8f796ddbf204c841487-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f3820576130a8f796ddbf204c841487-Abstract-Conference.html)

        **Abstract**:

        Neural image compression (NIC) has outperformed traditional image codecs in rate-distortion (R-D) performance. However, it usually requires a dedicated encoder-decoder pair for each point on R-D curve, which greatly hinders its practical deployment. While some recent works have enabled bitrate control via conditional coding, they impose strong prior during training and provide limited flexibility. In this paper we propose Code Editing, a highly flexible coding method for NIC based on semi-amortized inference and adaptive quantization. Our work is a new paradigm for variable bitrate NIC, and experimental results show that our method surpasses existing variable-rate methods. Furthermore, our approach is so flexible that it can also achieves ROI coding and multi-distortion trade-off with a single decoder. Our approach is compatible to all NIC methods with differentiable decoder NIC, and it can be even directly adopted on existing pre-trained models.

        ----

        ## [885] Homomorphic Matrix Completion

        **Authors**: *Xiao-Yang Liu, Zechu (Steven) Li, Xiaodong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f550cb7b30b59553e50cd08a9dbf068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f550cb7b30b59553e50cd08a9dbf068-Abstract-Conference.html)

        **Abstract**:

        In recommendation systems, global positioning, system identification and mobile social networks, it is a fundamental routine that a server completes a low-rank matrix from an observed subset of its entries. However, sending data to a cloud server raises up the data privacy concern due to eavesdropping attacks and the single-point failure problem, e.g., the Netflix prize contest was canceled after a privacy lawsuit. In this paper, we propose a homomorphic matrix completion algorithm for privacy-preserving data completion. First, we formulate a \textit{homomorphic matrix completion} problem where a server performs matrix completion on cyphertexts, and propose an encryption scheme that is fast and easy to implement. Secondly, we prove that the proposed scheme satisfies the \textit{homomorphism property} that decrypting the recovered matrix on cyphertexts will obtain the target complete matrix in plaintext. Thirdly, we prove that the proposed scheme satisfies an $(\epsilon, \delta)$-differential privacy property. While with similar level of privacy guarantee, we reduce the best-known error bound $O(\sqrt[10]{n_1^3n_2})$ to EXACT recovery at a price of more samples. Finally, on numerical data and real-world data, we show that both homomorphic nuclear-norm minimization and alternating minimization algorithms achieve accurate recoveries on cyphertexts, verifying the homomorphism property.

        ----

        ## [886] Spending Thinking Time Wisely: Accelerating MCTS with Virtual Expansions

        **Authors**: *Weirui Ye, Pieter Abbeel, Yang Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f5aeaee95e528a0ec5040bfa2fe9303-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f5aeaee95e528a0ec5040bfa2fe9303-Abstract-Conference.html)

        **Abstract**:

        One of the most important AI research questions is to trade off computation versus performance since ``perfect rationality" exists in theory but is impossible to achieve in practice. Recently, Monte-Carlo tree search (MCTS) has attracted considerable attention due to the significant performance improvement in various challenging domains. However, the expensive time cost during search severely restricts its scope for applications. This paper proposes the Virtual MCTS (V-MCTS), a variant of MCTS that spends more search time on harder states and less search time on simpler states adaptively. We give theoretical bounds of the proposed method and evaluate the performance and computations on $9 \times 9$ Go board games and Atari games. Experiments show that our method can achieve comparable performances to the original search algorithm while requiring less than $50\%$ search time on average. We believe that this approach is a viable alternative for tasks under limited time and resources. The code is available at \url{https://github.com/YeWR/V-MCTS.git}.

        ----

        ## [887] Intrinsic dimensionality estimation using Normalizing Flows

        **Authors**: *Christian Horvat, Jean-Pascal Pfister*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f918fa3a7c38b2d9b8b484bcc433334-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f918fa3a7c38b2d9b8b484bcc433334-Abstract-Conference.html)

        **Abstract**:

        How many degrees of freedom are there in a dataset consisting of $M$ samples embedded in $\mathbb{R}^D$? This number, formally known as \textsl{intrinsic dimensionality}, can be estimated using nearest neighbor statistics. However, nearest neighbor statistics do not scale to large datasets as their complexity scales quadratically in $M$, $\mathcal{O}(M^2)$. Additionally, methods based on nearest neighbor statistics perform poorly on datasets embedded in high dimensions where $D\gg 1$. In this paper, we propose a novel method to estimate the intrinsic dimensionality using Normalizing Flows that scale to large datasets and high dimensions. The method is based on some simple back-of-the-envelope calculations predicting how the singular values of the flow's Jacobian change when inflating the dataset with different noise magnitudes. Singular values associated with directions normal to the manifold evolve differently than singular values associated with directions tangent to the manifold. We test our method on various datasets, including 64x64 RGB images, where we achieve state-of-the-art results.

        ----

        ## [888] Moment Distributionally Robust Tree Structured Prediction

        **Authors**: *Yeshu Li, Danyal Saeed, Xinhua Zhang, Brian D. Ziebart, Kevin Gimpel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f92d2f498b88f1bd43732312272967a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f92d2f498b88f1bd43732312272967a-Abstract-Conference.html)

        **Abstract**:

        Structured prediction of tree-shaped objects is heavily studied under the name of syntactic dependency parsing. Current practice based on maximum likelihood or margin is either agnostic to or inconsistent with the evaluation loss. Risk minimization alleviates the discrepancy between training and test objectives but typically induces a non-convex problem. These approaches adopt explicit regularization to combat overfitting without probabilistic interpretation. We propose a moment-based distributionally robust optimization approach for tree structured prediction, where the worst-case expected loss over a set of distributions within bounded moment divergence from the empirical distribution is minimized. We develop efficient algorithms for arborescences and other variants of trees. We derive Fisher consistency, convergence rates and generalization bounds for our proposed method. We evaluate its empirical effectiveness on dependency parsing benchmarks.

        ----

        ## [889] Minimax Optimal Fixed-Budget Best Arm Identification in Linear Bandits

        **Authors**: *Junwen Yang, Vincent Y. F. Tan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4f9342b74c3bb63f6e030d8263082ab6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4f9342b74c3bb63f6e030d8263082ab6-Abstract-Conference.html)

        **Abstract**:

        We study the problem of best arm identification in linear bandits in the fixed-budget setting. By leveraging properties of the G-optimal design and incorporating it into the arm allocation rule, we design a parameter-free algorithm, Optimal Design-based Linear Best Arm Identification (OD-LinBAI). We provide a theoretical analysis of the failure probability of OD-LinBAI. Instead of all the optimality gaps, the performance of OD-LinBAI depends only on the gaps of the top $d$ arms, where $d$ is the effective dimension of the linear bandit instance. Complementarily, we present a minimax lower bound for this problem. The upper and lower bounds show that OD-LinBAI is minimax optimal up to constant multiplicative factors in the exponent, which is a significant theoretical improvement over existing methods (e.g., BayesGap, Peace, LinearExploration and GSE), and settles the question of ascertaining the difficulty of learning the best arm in the fixed-budget setting. Finally, numerical experiments  demonstrate considerable empirical improvements over existing algorithms on a variety of real and synthetic datasets.

        ----

        ## [890] Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction

        **Authors**: *Muralidhar Andoorveedu, Zhanda Zhu, Bojian Zheng, Gennady Pekhimenko*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4fc81f4cd2715d995018e0799262176b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4fc81f4cd2715d995018e0799262176b-Abstract-Conference.html)

        **Abstract**:

        Training deep learning models can be computationally expensive. Prior works have shown that increasing the batch size can potentially lead to better overall throughput. However, the batch size is frequently limited by the accelerator memory capacity due to the activations/feature maps stored for the training backward pass, as larger batch sizes require larger feature maps to be stored. Transformer-based models, which have recently seen a surge in popularity due to their good performance and applicability to a variety of tasks, have a similar problem. To remedy this issue, we propose Tempo, a new approach to efficiently use accelerator (e.g., GPU) memory resources for training Transformer-based models. Our approach provides drop-in replacements for the GELU, LayerNorm, and Attention layers, reducing the memory usage and ultimately leading to more efficient training. We implement Tempo and evaluate the throughput, memory usage, and accuracy/loss on the BERT Large pre-training task. We demonstrate that Tempo enables up to 2Ã— higher batch sizes and 16% higher training throughput over the state-of-the-art baseline. We also evaluate Tempo on GPT2 and RoBERTa models, showing 19% and 26% speedup over the baseline.

        ----

        ## [891] I2DFormer: Learning Image to Document Attention for Zero-Shot Image Classification

        **Authors**: *Muhammad Ferjad Naeem, Yongqin Xian, Luc Van Gool, Federico Tombari*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4fca3029c9ead4551937ed6987502e5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4fca3029c9ead4551937ed6987502e5f-Abstract-Conference.html)

        **Abstract**:

        Despite the tremendous progress in zero-shot learning (ZSL), the majority of existing methods still rely on human-annotated attributes, which are difficult to annotate and scale. An unsupervised alternative is to represent each class using the word embedding associated with its semantic class name. However, word embeddings extracted from pre-trained language models do not necessarily capture visual similarities, resulting in poor zero-shot performance.  In this work, we argue that online textual documents e.g., Wikipedia, contain rich visual descriptions about object classes, therefore can be used as powerful unsupervised side information for ZSL. To this end, we propose I2DFormer, a novel transformer-based ZSL framework that jointly learns to encode images and documents by aligning both modalities in a shared embedding space. In order to distill discriminative visual words from noisy documents, we introduce a new cross-modal attention module that learns fine-grained interactions between image patches and document words. Consequently, our I2DFormer not only learns highly discriminative document embeddings that capture visual similarities but also gains the ability to localize visually relevant words in image regions. Quantitatively, we demonstrate that our I2DFormer significantly outperforms previous unsupervised semantic embeddings under both zero-shot and generalized zero-shot learning settings on three public datasets. Qualitatively, we show that our method leads to highly interpretable results where document words can be grounded in the image regions.

        ----

        ## [892] Benchmarking Heterogeneous Treatment Effect Models through the Lens of Interpretability

        **Authors**: *Jonathan Crabbé, Alicia Curth, Ioana Bica, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4fd7b4ed13f78b9ba7afcd9d01615896-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4fd7b4ed13f78b9ba7afcd9d01615896-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Estimating personalized effects of treatments is a complex, yet pervasive problem. To tackle it, recent developments in the machine learning (ML) literature on heterogeneous treatment effect estimation gave rise to many sophisticated, but opaque, tools: due to their flexibility, modularity and ability to learn constrained representations, neural networks in particular have become central to this literature. Unfortunately, the assets of such black boxes come at a cost: models typically involve countless nontrivial operations, making it difficult to understand what they have learned. Yet, understanding these models can be crucial -- in a medical context, for example, discovered knowledge on treatment effect heterogeneity could inform treatment prescription in clinical practice. In this work, we therefore use post-hoc feature importance methods to identify features that influence the model's predictions. This allows us to evaluate treatment effect estimators along a new and important dimension that has been overlooked in previous work: We construct a benchmarking environment to empirically investigate the ability of personalized treatment effect models to identify predictive covariates -- covariates that determine differential responses to treatment. Our benchmarking environment then enables us to provide new insight into the strengths and weaknesses of different types of treatment effects models as we modulate different challenges specific to treatment effect estimation -- e.g. the ratio of prognostic to predictive information, the possible nonlinearity of potential outcomes and the presence and type of confounding.

        ----

        ## [893] AD-DROP: Attribution-Driven Dropout for Robust Language Model Fine-Tuning

        **Authors**: *Tao Yang, Jinghao Deng, Xiaojun Quan, Qifan Wang, Shaoliang Nie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4fdf8d49476a8001c91f9e9e90530e13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4fdf8d49476a8001c91f9e9e90530e13-Abstract-Conference.html)

        **Abstract**:

        Fine-tuning large pre-trained language models on downstream tasks is apt to suffer from overfitting when limited training data is available. While dropout proves to be an effective antidote by randomly dropping a proportion of units, existing research has not examined its effect on the self-attention mechanism. In this paper, we investigate this problem through self-attention attribution and find that dropping attention positions with low attribution scores can accelerate training and increase the risk of overfitting. Motivated by this observation, we propose Attribution-Driven Dropout (AD-DROP), which randomly discards some high-attribution positions to encourage the model to make predictions by relying more on low-attribution positions to reduce overfitting. We also develop a cross-tuning strategy to alternate fine-tuning and AD-DROP to avoid dropping high-attribution positions excessively. Extensive experiments on various benchmarks show that AD-DROP yields consistent improvements over baselines. Analysis further confirms that AD-DROP serves as a strategic regularizer to prevent overfitting during fine-tuning.

        ----

        ## [894] Reinforced Genetic Algorithm for Structure-based Drug Design

        **Authors**: *Tianfan Fu, Wenhao Gao, Connor W. Coley, Jimeng Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4fe1859112230a032c7143a9adc3be78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4fe1859112230a032c7143a9adc3be78-Abstract-Conference.html)

        **Abstract**:

        Structure-based drug design (SBDD) aims to discover drug candidates by finding molecules (ligands) that bind tightly to a disease-related protein (targets), which is the primary approach to computer-aided drug discovery. Recently, applying deep generative models for three-dimensional (3D) molecular design conditioned on protein pockets to solve SBDD has attracted much attention, but their formulation as probabilistic modeling often leads to unsatisfactory optimization performance. On the other hand, traditional combinatorial optimization methods such as genetic algorithms (GA) have demonstrated state-of-the-art performance in various molecular optimization tasks. However, they do not utilize protein target structure to inform design steps but rely on a random-walk-like exploration, which leads to unstable performance and no knowledge transfer between different tasks despite the similar binding physics. To achieve a more stable and efficient SBDD, we propose Reinforced Genetic Algorithm (RGA) that uses neural models to prioritize the profitable design steps and suppress random-walk behavior. The neural models take the 3D structure of the targets and ligands as inputs and are pre-trained using native complex structures to utilize the knowledge of the shared binding physics from different targets and then fine-tuned during optimization. We conduct thorough empirical studies on optimizing binding affinity to various disease targets and show that RGA outperforms the baselines in terms of docking scores and is more robust to random initializations. The ablation study also indicates that the training on different targets helps improve the performance by leveraging the shared underlying physics of the binding processes. The code is available at https://github.com/futianfan/reinforced-genetic-algorithm.

        ----

        ## [895] A Variational Edge Partition Model for Supervised Graph Representation Learning

        **Authors**: *Yilin He, Chaojie Wang, Hao Zhang, Bo Chen, Mingyuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4ffbfbcf4ad7304d57158b046525e46c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4ffbfbcf4ad7304d57158b046525e46c-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs), which propagate the node features through the edges and learn how to transform the aggregated features under label supervision, have achieved great success in supervised feature extraction for both node-level and graph-level  classification tasks. However, GNNs typically treat the graph structure as given and ignore how the edges are formed. This paper introduces a graph generative process to model how the observed edges are generated by aggregating the node interactions over a set of overlapping node communities, each of which contributes to the edges via a logical OR mechanism. Based on this generative model, we partition each edge into the summation of multiple community-specific weighted edges and use them to define community-specific GNNs. A variational inference framework is proposed to jointly learn a GNN-based inference network  that partitions the edges into different communities, these community-specific GNNs, and a GNN-based predictor that combines community-specific GNNs for the end classification task. Extensive evaluations on real-world graph datasets have verified the effectiveness of the proposed method in learning discriminative representations for both node-level and graph-level classification tasks.

        ----

        ## [896] Learning Optimal Flows for Non-Equilibrium Importance Sampling

        **Authors**: *Yu Cao, Eric Vanden-Eijnden*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5000f096bed9360a060d835c2a1703bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5000f096bed9360a060d835c2a1703bb-Abstract-Conference.html)

        **Abstract**:

        Many applications in computational sciences and statistical inference require the computation of expectations with respect to complex high-dimensional distributions with unknown normalization constants, as well as the estimation of these constants. Here we develop a method to perform these calculations based on generating samples from a simple base distribution, transporting them by the flow generated by a velocity field, and performing averages along these flowlines. This non-equilibrium importance sampling (NEIS) strategy is straightforward to implement and can be used for calculations with arbitrary target distributions. On the theory side, we discuss how to tailor the velocity field to the target and establish general conditions under which the proposed estimator is a perfect estimator with zero-variance. We also draw connections between NEIS and approaches based on mapping a base distribution onto a target via a transport map. On the computational side, we show how to use deep learning to represent the velocity field by a neural network and train it towards the zero variance optimum. These results are illustrated numerically on benchmark examples (with dimension up to $10$), where after training the velocity field, the variance of the NEIS estimator is reduced by up to $6$ orders of magnitude than that of a vanilla estimator. We also compare the performances of NEIS with those of Neal's annealed importance sampling (AIS).

        ----

        ## [897] Decision Trees with Short Explainable Rules

        **Authors**: *Victor Feitosa Souza, Ferdinando Cicalese, Eduardo Sany Laber, Marco Molinaro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/500637d931d4feb99d5cce84af1f53ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/500637d931d4feb99d5cce84af1f53ba-Abstract-Conference.html)

        **Abstract**:

        Decision trees are widely used in many settings where interpretable models are preferred or required. As confirmed by recent empirical studies,  the interpretability/explanability of a decision tree critically depends on some of its structural parameters, like size and the  average/maximum depth of its leaves. There is indeed a vast literature on the design and analysis of decision tree algorithms that aim at optimizing these parameters.This paper contributes to this important line of research: we propose as a novel criterion of measuring the interpretability of a decision tree, the sparsity of the set of attributes that are (on average) required to explain the classification of the examples. We give a tight characterization of the best possible guarantees achievable by a decision tree built to optimize both our newmeasure (which we call the {\em explanation size})  and the more classical measures of worst-case and average depth. In particular, we give an algorithm that guarantees $O(\ln n )$-approximation (hence optimal if $P \neq NP$) for the minimization of both the average/worst-case explanation size and the average/worst-case depth. In addition to our theoretical contributions, experiments with 20 real datasets show that our algorithm has accuracy competitive with CART while producing trees that allow for much simpler explanations.

        ----

        ## [898] NAS-Bench-360: Benchmarking Neural Architecture Search on Diverse Tasks

        **Authors**: *Renbo Tu, Nicholas Roberts, Mikhail Khodak, Junhong Shen, Frederic Sala, Ameet Talwalkar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/506630e4a43bb9d64a49f98b9ba934e9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/506630e4a43bb9d64a49f98b9ba934e9-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Most existing neural architecture search (NAS) benchmarks and algorithms prioritize well-studied tasks, e.g. image classification on CIFAR or ImageNet. This makes the performance of NAS approaches in more diverse areas poorly understood. In this paper, we present NAS-Bench-360, a benchmark suite to evaluate methods on domains beyond those traditionally studied in architecture search, and use it to address the following question: do state-of-the-art NAS methods perform well on diverse tasks? To construct the benchmark, we curate ten tasks spanning a diverse array of application domains, dataset sizes, problem dimensionalities, and learning objectives. Each task is carefully chosen to interoperate with modern CNN-based search methods while possibly being far-afield from its original development domain. To speed up and reduce the cost of NAS research, for two of the tasks we release the precomputed performance of 15,625 architectures comprising a standard CNN search space. Experimentally, we show the need for more robust NAS evaluation of the kind NAS-Bench-360 enables by showing that several modern NAS procedures perform inconsistently across the ten tasks, with many catastrophically poor results. We also demonstrate how NAS-Bench-360 and its associated precomputed results will enable future scientific discoveries by testing whether several recent hypotheses promoted in the NAS literature hold on diverse tasks. NAS-Bench-360 is hosted at https://nb360.ml.cmu.edu.

        ----

        ## [899] Sparse Fourier Backpropagation in Cryo-EM Reconstruction

        **Authors**: *Dari Kimanius, Kiarash Jamali, Sjors H. W. Scheres*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50729453d56ecf6a8b7be78998776472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50729453d56ecf6a8b7be78998776472-Abstract-Conference.html)

        **Abstract**:

        Electron cryo-microscopy (cryo-EM) is a powerful method for investigating the structures of protein molecules, with important implications for understanding the molecular processes of life and drug development. In this technique, many noisy, two-dimensional projection images of protein molecules in unknown poses are combined into one or more three-dimensional reconstructions. The presence of multiple structural states in the data represents a major bottleneck in existing processing pipelines, often requiring expert user supervision. Variational auto-encoders (VAEs) have recently been proposed as an attractive means for learning the data manifold of data sets with a large number of different states. These methods are based on a coordinate-based approach, similar to Neural Radiance Fields (NeRF), to make volumetric reconstructions from 2D image data in Fourier-space. Although NeRF is a powerful method for real-space reconstruction, many of the benefits of the method do not transfer to Fourier-space, e.g. inductive bias for spatial locality. We present an approach where the VAE reconstruction is expressed on a volumetric grid, and demonstrate how this model can be trained efficiently through a novel backpropagation method that exploits the sparsity of the projection operation in Fourier-space. We achieve improved results on a simulated data set and at least equivalent results on an experimental data set when compared to the coordinate-based approach, while also substantially lowering computational cost. Our approach is computationally more efficient, especially in inference, enabling interactive analysis of the latent space by the user.

        ----

        ## [900] Modular Flows: Differential Molecular Generation

        **Authors**: *Yogesh Verma, Samuel Kaski, Markus Heinonen, Vikas K. Garg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/509f7977030f3550300f541ec228c3fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/509f7977030f3550300f541ec228c3fc-Abstract-Conference.html)

        **Abstract**:

        Generating new molecules is fundamental to advancing critical applications such as drug discovery and material synthesis. Flows can generate molecules effectively by inverting the encoding process, however, existing flow models either require artifactual dequantization or specific node/edge orderings, lack desiderata such as permutation invariance, or induce discrepancy between encoding and decoding steps that necessitates post hoc validity correction. Inspired by graph PDEs, we circumvent these issues with novel continuous normalizing E(3)-equivariant flows, based on a system of coupled node ODEs, that repeatedly reconcile locally toward globally aligned densities. Our models can be cast as message passing temporal networks, and result in superlative density estimation and  molecular generation. In particular, our generated samples achieve state of the art on both the standard QM9 and ZINC250K benchmarks.

        ----

        ## [901] Anonymous Bandits for Multi-User Systems

        **Authors**: *Hossein Esfandiari, Vahab Mirrokni, Jon Schneider*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50a057e9fe79ffa3f4120fb6fb88071a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50a057e9fe79ffa3f4120fb6fb88071a-Abstract-Conference.html)

        **Abstract**:

        In this work, we present and study a new framework for online learning in systems with multiple users that provide user anonymity. Specifically, we extend the notion of bandits to obey the standard $k$-anonymity constraint by requiring each observation to be an aggregation of rewards for at least $k$ users. This provides a simple yet effective framework where one can learn a clustering of users in an online fashion without observing any user's individual decision. We initiate the study of anonymous bandits and provide the first sublinear regret algorithms and lower bounds for this setting.

        ----

        ## [902] Unsupervised Object Detection Pretraining with Joint Object Priors Generation and Detector Learning

        **Authors**: *Yizhou Wang, Meilin Chen, Shixiang Tang, Feng Zhu, Haiyang Yang, Lei Bai, Rui Zhao, Yunfeng Yan, Donglian Qi, Wanli Ouyang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50ca96a1a9ebe0b5e5688a504feb6107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50ca96a1a9ebe0b5e5688a504feb6107-Abstract-Conference.html)

        **Abstract**:

        Unsupervised pretraining methods for object detection aim to learn object discrimination and localization ability from large amounts of images. Typically, recent works design pretext tasks that supervise the detector to predict the defined object priors. They normally leverage heuristic methods to produce object priors, \emph{e.g.,} selective search, which separates the prior generation and detector learning and leads to sub-optimal solutions. In this work, we propose a novel object detection pretraining framework that could generate object priors and learn detectors jointly by generating accurate object priors from the model itself. Specifically, region priors are extracted by attention maps from the encoder, which highlights foregrounds. Instance priors are the selected high-quality output bounding boxes of the detection decoder. By assuming objects as instances in the foreground, we can generate object priors with both region and instance priors. Moreover, our object priors are jointly refined along with the detector optimization. With better object priors as supervision, the model could achieve better detection capability, which in turn promotes the object priors generation. Our method improves the competitive approaches by \textbf{+1.3 AP}, \textbf{+1.7 AP} in 1\% and 10\% COCO low-data regimes object detection.

        ----

        ## [903] Invariance Learning in Deep Neural Networks with Differentiable Laplace Approximations

        **Authors**: *Alexander Immer, Tycho F. A. van der Ouderaa, Gunnar Rätsch, Vincent Fortuin, Mark van der Wilk*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50d005f92a6c5c9646db4b761da676ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50d005f92a6c5c9646db4b761da676ba-Abstract-Conference.html)

        **Abstract**:

        Data augmentation is commonly applied to improve performance of deep learning by enforcing the knowledge that certain transformations on the input preserve the output. Currently, the data augmentation parameters are chosen by human effort and costly cross-validation, which makes it cumbersome to apply to new datasets. We develop a convenient gradient-based method for selecting the data augmentation without validation data during training of a deep neural network. Our approach relies on phrasing data augmentation as an invariance in the prior distribution on the functions of a neural network, which allows us to learn it using Bayesian model selection. This has been shown to work in Gaussian processes, but not yet for deep neural networks. We propose a differentiable Kronecker-factored Laplace approximation to the marginal likelihood as our objective, which can be optimised without human supervision or validation data. We show that our method can successfully recover invariances present in the data, and that this improves generalisation and data efficiency on image datasets.

        ----

        ## [904] QueryPose: Sparse Multi-Person Pose Regression via Spatial-Aware Part-Level Query

        **Authors**: *Yabo Xiao, Kai Su, Xiaojuan Wang, Dongdong Yu, Lei Jin, Mingshu He, Zehuan Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50d277e84b2bcbaadcd84548a87e8cc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50d277e84b2bcbaadcd84548a87e8cc4-Abstract-Conference.html)

        **Abstract**:

        We propose a sparse end-to-end multi-person pose regression framework, termed QueryPose, which can directly predict multi-person keypoint sequences from the input image. The existing end-to-end methods rely on dense representations to preserve the spatial detail and structure for precise keypoint localization. However, the dense paradigm introduces complex and redundant post-processes during inference. In our framework, each human instance is encoded by several learnable spatial-aware part-level queries associated with an instance-level query. First, we propose the Spatial Part Embedding Generation Module (SPEGM) that considers the local spatial attention mechanism to generate several spatial-sensitive part embeddings, which contain spatial details and structural information for enhancing the part-level queries. Second, we introduce the Selective Iteration Module (SIM) to adaptively update the sparse part-level queries via the generated spatial-sensitive part embeddings stage-by-stage. Based on the two proposed modules, the part-level queries are able to fully encode the spatial details and structural information for precise keypoint regression. With the bipartite matching, QueryPose avoids the hand-designed post-processes. Without bells and whistles, QueryPose surpasses the existing dense end-to-end methods with 73.6 AP on MS COCO mini-val set and 72.7 AP on CrowdPose test set. Code is available at https://github.com/buptxyb666/QueryPose.

        ----

        ## [905] EAGER: Asking and Answering Questions for Automatic Reward Shaping in Language-guided RL

        **Authors**: *Thomas Carta, Pierre-Yves Oudeyer, Olivier Sigaud, Sylvain Lamprier*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50eb39ab717507cccbe2b8590de32030-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/50eb39ab717507cccbe2b8590de32030-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) in long horizon and sparse reward tasks is notoriously difficult and requires a lot of training steps. A standard solution to speed up the process is to leverage additional reward signals, shaping it to better guide the learning process.In the context of language-conditioned RL, the abstraction and generalisation properties of the language input provide opportunities for more efficient ways of shaping the reward.In this paper, we leverage this idea and propose an automated reward shaping method where the agent extracts auxiliary objectives from the general language goal. These auxiliary objectives use a question generation (QG) and a question answering (QA) system: they consist of questions leading the agent to try to reconstruct partial information about the global goal using its own trajectory.When it succeeds, it receives an intrinsic reward proportional to its confidence in its answer. This incentivizes the agent to generate trajectories which unambiguously explain various aspects of the general language goal.Our experimental study using various BabyAI environments shows that this approach, which does not require engineer intervention to design the auxiliary objectives, improves sample efficiency by effectively directing the exploration.

        ----

        ## [906] OpenFilter: A Framework to Democratize Research Access to Social Media AR Filters

        **Authors**: *Piera Riccio, Bill Psomas, Francesco Galati, Francisco Escolano, Thomas Hofmann, Nuria Oliver*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/50fd4a244de17f856709036edda9854e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/50fd4a244de17f856709036edda9854e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Augmented Reality or AR filters on selfies have become very popular on social media platforms for a variety of applications, including marketing, entertainment and aesthetics. Given the wide adoption of AR face filters and the importance of faces in our social structures and relations, there is increased interest by the scientific community to analyze the impact of such filters from a psychological, artistic and sociological perspective. However, there are few quantitative analyses in this area mainly due to a lack of publicly available datasets of facial images with applied AR filters. The proprietary, close nature of most social media platforms does not allow users, scientists and practitioners to access the code and the details of the available AR face filters. Scraping faces from these platforms to collect data is ethically unacceptable and should, therefore, be avoided in research. In this paper, we present OpenFilter, a flexible framework to apply AR filters available in social media platforms on existing large collections of human faces. Moreover, we share FairBeauty and B-LFW, two beautified versions of the publicly available FairFace and LFW datasets and we outline insights derived from the analysis of these beautified datasets.

        ----

        ## [907] Improving Policy Learning via Language Dynamics Distillation

        **Authors**: *Victor Zhong, Jesse Mu, Luke Zettlemoyer, Edward Grefenstette, Tim Rocktäschel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/51053d7b8473df7d5a2165b2a8ee9629-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/51053d7b8473df7d5a2165b2a8ee9629-Abstract-Conference.html)

        **Abstract**:

        Recent work has shown that augmenting environments with language descriptions improves policy learning. However, for environments with complex language abstractions, learning how to ground language to observations is difficult due to sparse, delayed rewards. We propose Language Dynamics Distillation (LDD), which pretrains a model to predict environment dynamics given demonstrations with language descriptions, and then fine-tunes these language-aware pretrained representations via reinforcement learning (RL). In this way, the model is trained to both maximize expected reward and retain knowledge about how language relates to environment dynamics. On SILG, a benchmark of five tasks with language descriptions that evaluate distinct generalization challenges on unseen environments (NetHack, ALFWorld, RTFM, Messenger, and Touchdown), LDD outperforms tabula-rasa RL, VAE pretraining, and methods that learn from unlabeled demonstrations in inverse RL and reward shaping with pretrained experts. In our analyses, we show that language descriptions in demonstrations improve sample-efficiency and generalization across environments, and that dynamics modeling with expert demonstrations is more effective than with non-experts.

        ----

        ## [908] Learning Energy Networks with Generalized Fenchel-Young Losses

        **Authors**: *Mathieu Blondel, Felipe Llinares-López, Robert Dadashi, Léonard Hussenot, Matthieu Geist*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/510cfd9945f8bde6f0cf9b27ff1f8a76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/510cfd9945f8bde6f0cf9b27ff1f8a76-Abstract-Conference.html)

        **Abstract**:

        Energy-based models, a.k.a. energy networks, perform inference by optimizing an energy function, typically parametrized by a neural network. This allows one to capture potentially complex relationships between inputs andoutputs.To learn the parameters of the energy function, the solution to thatoptimization problem is typically fed into a loss function.The key challenge for training energy networks lies in computing loss gradients,as this typically requires argmin/argmax differentiation.In this paper, building upon a generalized notion of conjugate function,which replaces the usual bilinear pairing with a general energy function,we propose generalized Fenchel-Young losses, a natural loss construction forlearning energy networks. Our losses enjoy many desirable properties and theirgradients can be computed efficiently without argmin/argmax differentiation.We also prove the calibration of their excess risk in the case of linear-concaveenergies. We demonstrate our losses on multilabel classification and imitation learning tasks.

        ----

        ## [909] Interaction-Grounded Learning with Action-Inclusive Feedback

        **Authors**: *Tengyang Xie, Akanksha Saran, Dylan J. Foster, Lekan P. Molu, Ida Momennejad, Nan Jiang, Paul Mineiro, John Langford*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/512b6bc067a6c6fa6a6ff8e5f6445e10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/512b6bc067a6c6fa6a6ff8e5f6445e10-Abstract-Conference.html)

        **Abstract**:

        Consider the problem setting of Interaction-Grounded Learning (IGL), in which a learner's goal is to optimally interact with the environment with no explicit reward to ground its policies. The agent observes a context vector, takes an action, and receives a feedback vector, using this information to effectively optimize a policy with respect to a latent reward function. Prior analyzed approaches fail when the feedback vector contains the action, which significantly limits IGLâ€™s success in many potential scenarios such as Brain-computer interface (BCI) or Human-computer interface (HCI) applications. We address this by creating an algorithm and analysis which allows IGL to work even when the feedback vector contains the action, encoded in any fashion. We provide theoretical guarantees and large-scale experiments based on supervised datasets to demonstrate the effectiveness of the new approach.

        ----

        ## [910] Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels

        **Authors**: *Mohamad Amin Mohamadi, Wonho Bae, Danica J. Sutherland*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5132940b1bced8a7b28e9695d49d435a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5132940b1bced8a7b28e9695d49d435a-Abstract-Conference.html)

        **Abstract**:

        We propose a new method for approximating active learning acquisition strategies that are based on retraining with hypothetically-labeled candidate data points. Although this is usually infeasible with deep networks, we use the neural tangent kernel to approximate the result of retraining, and prove that this approximation works asymptotically even in an active learning setup -- approximating look-ahead'' selection criteria with far less computation required. This also enables us to conduct sequential active learning, i.e.\ updating the model in a streaming regime, without needing to retrain the model with SGD after adding each new data point. Moreover, our querying strategy, which better understands how the model's predictions will change by adding new data points in comparison to the standard (myopic'') criteria, beats other look-ahead strategies by large margins, and achieves equal or better performance compared to state-of-the-art methods on several benchmark datasets in pool-based active learning.

        ----

        ## [911] The Neural Testbed: Evaluating Joint Predictions

        **Authors**: *Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Xiuyuan Lu, Morteza Ibrahimi, Dieterich Lawson, Botao Hao, Brendan O'Donoghue, Benjamin Van Roy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5141f6bc105d30edbae48f1d2e0b1e66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5141f6bc105d30edbae48f1d2e0b1e66-Abstract-Conference.html)

        **Abstract**:

        Predictive distributions quantify uncertainties ignored by point estimates. This paper introduces The Neural Testbed: an open source benchmark for controlled and principled evaluation of agents that generate such predictions. Crucially, the testbed assesses agents not only on the quality of their marginal predictions per input, but also on their joint predictions across many inputs. We evaluate a range of agents using a simple neural network data generating process.Our results indicate that some popular Bayesian deep learning agents do not fare well with joint predictions, even when they can produce accurate marginal predictions. We also show that the quality of joint predictions drives performance in downstream decision tasks. We find these results are robust across choice a wide range of generative models, and highlight the practical importance of joint predictions to the community.

        ----

        ## [912] Learning Dense Object Descriptors from Multiple Views for Low-shot Category Generalization

        **Authors**: *Stefan Stojanov, Anh Thai, Zixuan Huang, James M. Rehg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/517a0884c56008f8bf9d5912ca771d71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/517a0884c56008f8bf9d5912ca771d71-Abstract-Conference.html)

        **Abstract**:

        A hallmark of the deep learning era for computer vision is the successful use of large-scale labeled datasets to train feature representations. This has been done for tasks ranging from object recognition and semantic segmentation to optical flow estimation and novel view synthesis of 3D scenes. In this work, we aim to learn dense discriminative object representations for low-shot category recognition without requiring any category labels. To this end, we propose Deep Object Patch Encodings (DOPE), which can be trained from multiple views of object instances without any category or semantic object part labels. To train DOPE, we assume access to sparse depths, foreground masks and known cameras, to obtain pixel-level correspondences between views of an object, and use this to formulate a self-supervised learning task to learn discriminative object patches. We find that DOPE can directly be used for low-shot classification of novel categories using local-part matching, and is competitive with and outperforms supervised and self-supervised learning baselines.

        ----

        ## [913] Monte Carlo Tree Descent for Black-Box Optimization

        **Authors**: *Yaoguang Zhai, Sicun Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5185aa776fd64ae3b4c6dae1af1066b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5185aa776fd64ae3b4c6dae1af1066b1-Abstract-Conference.html)

        **Abstract**:

        The key to Black-Box Optimization is to efficiently search through input regions with potentially widely-varying numerical properties, to achieve low-regret descent and fast progress toward the optima. Monte Carlo Tree Search (MCTS) methods have recently been introduced to improve Bayesian optimization by computing better partitioning of the search space that balances exploration and exploitation. Extending this promising framework, we study how to further integrate sample-based descent for faster optimization.  We design novel ways of expanding Monte Carlo search trees, with new descent methods at vertices that incorporate stochastic search and Gaussian Processes. We propose the corresponding rules for balancing progress and uncertainty, branch selection, tree expansion, and backpropagation. The designed search process puts more emphasis on sampling for faster descent and uses localized Gaussian Processes as auxiliary metrics for both exploitation and exploration. We show empirically that the proposed algorithms can outperform state-of-the-art methods on many challenging benchmark problems.

        ----

        ## [914] Teacher Forcing Recovers Reward Functions for Text Generation

        **Authors**: *Yongchang Hao, Yuxin Liu, Lili Mou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/51ae7d9db3423ae96cd6afeb01529819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/51ae7d9db3423ae96cd6afeb01529819-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) has been widely used in text generation to alleviate the exposure bias issue or to utilize non-parallel datasets. The reward function plays an important role in making RL training successful. However, previous reward functions are typically task-specific and sparse, restricting the use of RL. In our work, we propose a task-agnostic approach that derives a step-wise reward function directly from a model trained with teacher forcing. We additionally propose a simple modification to stabilize the RL training on non-parallel datasets with our induced reward function. Empirical results show that our method outperforms self-training and reward regression methods on several text generation tasks, confirming the effectiveness of our reward function.

        ----

        ## [915] Masked Autoencoding for Scalable and Generalizable Decision Making

        **Authors**: *Fangchen Liu, Hao Liu, Aditya Grover, Pieter Abbeel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/51fda94414996902ddaaa35561b97294-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/51fda94414996902ddaaa35561b97294-Abstract-Conference.html)

        **Abstract**:

        We are interested in learning scalable agents for reinforcement learning that can learn from large-scale, diverse sequential data similar to current large vision and language models. To this end, this paper presents masked decision prediction (MaskDP), a simple and scalable self-supervised pretraining method for reinforcement learning (RL) and behavioral cloning (BC). In our MaskDP approach, we employ a masked autoencoder (MAE) to state-action trajectories, wherein we randomly mask state and action tokens and reconstruct the missing data. By doing so, the model is required to infer masked out states and actions and extract information about dynamics. We find that masking different proportions of the input sequence significantly helps with learning a better model that generalizes well to multiple downstream tasks. In our empirical study we Ô¨Ånd that a MaskDP model gains the capability of zero-shot transfer to new BC tasks, such as single and multiple goal reaching, and it can zero-shot infer skills from a few example transitions. In addition, MaskDP transfers well to offline RL and shows promising scaling behavior w.r.t. to model size. It is amenable to data efficient finetuning, achieving competitive results with prior methods based on autoregressive pretraining.

        ----

        ## [916] Distributional Reward Estimation for Effective Multi-agent Deep Reinforcement Learning

        **Authors**: *Jifeng Hu, Yanchao Sun, Hechang Chen, Sili Huang, Haiyin Piao, Yi Chang, Lichao Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html)

        **Abstract**:

        Multi-agent reinforcement learning has drawn increasing attention in practice, e.g., robotics and automatic driving, as it can explore optimal policies using samples generated by interacting with the environment. However, high reward uncertainty still remains a problem when we want to train a satisfactory model, because obtaining high-quality reward feedback is usually expensive and even infeasible. To handle this issue, previous methods mainly focus on passive reward correction. At the same time, recent active reward estimation methods have proven to be a recipe for reducing the effect of reward uncertainty. In this paper, we propose a novel Distributional Reward Estimation framework for effective Multi-Agent Reinforcement Learning (DRE-MARL). Our main idea is to design the multi-action-branch reward estimation and policy-weighted reward aggregation for stabilized training. Specifically, we design the multi-action-branch reward estimation to model reward distributions on all action branches. Then we utilize reward aggregation to obtain stable updating signals during training. Our intuition is that consideration of all possible consequences of actions could be useful for learning policies. The superiority of the DRE-MARL is demonstrated using benchmark multi-agent scenarios, compared with the SOTA baselines in terms of both effectiveness and robustness.

        ----

        ## [917] Sampling from Log-Concave Distributions with Infinity-Distance Guarantees

        **Authors**: *Oren Mangoubi, Nisheeth K. Vishnoi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/520b7f40c79813ff1ec5ce41ecbea8a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/520b7f40c79813ff1ec5ce41ecbea8a1-Abstract-Conference.html)

        **Abstract**:

        For a $d$-dimensional log-concave distribution $\pi(\theta) \propto e^{-f(\theta)}$ constrained to a convex body $K$, the problem of outputting samples from a distribution $\nu$ which is $\varepsilon$-close in infinity-distance $\sup_{\theta \in K} |\log  \frac{\nu(\theta)}{\pi(\theta)}|$ to $\pi$ arises in differentially private optimization. While sampling within total-variation distance $\varepsilon$ of $\pi$ can be done by algorithms whose runtime depends polylogarithmically on $\frac{1}{\varepsilon}$, prior algorithms for sampling in $\varepsilon$ infinity distance have runtime bounds that depend polynomially on $\frac{1}{\varepsilon}$. We bridge this gap by presenting an algorithm that outputs a point  $\varepsilon$-close to $\pi$ in infinity distance that requires at most  $\mathrm{poly}(\log \frac{1}{\varepsilon}, d)$ calls to a membership oracle for $K$ and evaluation oracle for $f$, when $f$ is Lipschitz. Our approach departs from prior works that construct  Markov chains on a $\frac{1}{\varepsilon^2}$-discretization of $K$ to achieve a sample with $\varepsilon$ infinity-distance error, and present a method to directly convert continuous samples from $K$ with total-variation bounds to samples with infinity bounds. This approach also allows us to obtain an improvement on the dimension $d$ in the running time for the problem of sampling from a log-concave distribution on polytopes $K$ with infinity distance $\varepsilon$, by plugging in TV-distance running time bounds for the Dikin Walk Markov chain.

        ----

        ## [918] ELASTIC: Numerical Reasoning with Adaptive Symbolic Compiler

        **Authors**: *Jiaxin Zhang, Yashar Moshfeghi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/522ef98b1e52f5918e5abc868651175d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/522ef98b1e52f5918e5abc868651175d-Abstract-Conference.html)

        **Abstract**:

        Numerical reasoning over text is a challenging task of Artificial Intelligence (AI), requiring reading comprehension and numerical reasoning abilities. Previous approaches use numerical reasoning programs to represent the reasoning process. However, most works do not separate the generation of operators and operands, which are key components of a numerical reasoning program, thus limiting their ability to generate such programs for complicated tasks. In this paper, we introduce the numEricaL reASoning with adapTive symbolIc Compiler (ELASTIC) model, which is constituted of the RoBERTa as the Encoder and a Compiler with four modules: Reasoning Manager, Operator Generator, Operands Generator, and Memory Register. ELASTIC is robust when conducting complicated reasoning. Also, it is domain agnostic by supporting the expansion of diverse operators without caring about the number of operands it contains. Experiments show that ELASTIC achieves 68.96 and 65.21 of execution accuracy and program accuracy on the FinQA dataset and 83.00 program accuracy on the MathQA dataset, outperforming previous state-of-the-art models significantly.

        ----

        ## [919] Training Spiking Neural Networks with Local Tandem Learning

        **Authors**: *Qu Yang, Jibin Wu, Malu Zhang, Yansong Chua, Xinchao Wang, Haizhou Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/523caec7832a47fb19b8471dbfeec471-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/523caec7832a47fb19b8471dbfeec471-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks (SNNs) are shown to be more biologically plausible and energy efficient over their predecessors. However, there is a lack of an efficient and generalized training method for deep SNNs, especially for deployment on analog computing substrates. In this paper, we put forward a generalized learning rule, termed Local Tandem Learning (LTL). The LTL rule follows the teacher-student learning approach by mimicking the intermediate feature representations of a pre-trained ANN. By decoupling the learning of network layers and leveraging highly informative supervisor signals, we demonstrate rapid network convergence within five training epochs on the CIFAR-10 dataset while having low computational complexity. Our experimental results have also shown that the SNNs thus trained can achieve comparable accuracies to their teacher ANNs on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets. Moreover, the proposed LTL rule is hardware friendly. It can be easily implemented on-chip to perform fast parameter calibration and provide robustness against the notorious device non-ideality issues. It, therefore, opens up a myriad of opportunities for training and deployment of SNN on ultra-low-power mixed-signal neuromorphic computing chips.

        ----

        ## [920] FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting

        **Authors**: *Tian Zhou, Ziqing Ma, Xue Wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html)

        **Abstract**:

        Recent studies have shown that deep learning models such as RNNs and Transformers have brought significant performance gains for long-term forecasting of time series because they effectively utilize historical information. We found, however, that there is still great room for improvement in how to preserve historical information in neural networks while avoiding overfitting to noise present in the history. Addressing this allows better utilization of the capabilities of deep learning models. To this end, we design a Frequency improved Legendre Memory model, or FiLM: it applies Legendre polynomial projections to approximate historical information, uses Fourier projection to remove noise, and adds a low-rank approximation to speed up computation. Our empirical studies show that the proposed FiLM significantly improves the accuracy of state-of-the-art models in multivariate and univariate long-term forecasting by (19.2%, 22.6%), respectively. We also demonstrate that the representation module developed in this work can be used as a general plugin to improve the long-term prediction performance of other deep learning modules. Code is available at  https://github.com/tianzhou2011/FiLM/.

        ----

        ## [921] Differentially Private Linear Sketches: Efficient Implementations and Applications

        **Authors**: *Fuheng Zhao, Dan Qiao, Rachel Redberg, Divyakant Agrawal, Amr El Abbadi, Yu-Xiang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/525338e0d98401a62950bc7c454eb83d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/525338e0d98401a62950bc7c454eb83d-Abstract-Conference.html)

        **Abstract**:

        Linear sketches have been widely adopted to process fast data streams, and they can be used to accurately answer frequency estimation, approximate top K items, and summarize data distributions. When data are sensitive, it is desirable to provide privacy guarantees for linear sketches to preserve private information while delivering useful results with theoretical bounds. We show that linear sketches can ensure privacy and maintain their unique properties with a small amount of noise added at initialization. From the differentially private linear sketches, we showcase that the state-of-the-art quantile sketch in the turnstile model can also be private and maintain high performance. Experiments further demonstrate that our proposed differentially private sketches are quantitatively and qualitatively similar to noise-free sketches with high utilization on synthetic and real datasets.

        ----

        ## [922] Coordinates Are NOT Lonely - Codebook Prior Helps Implicit Neural 3D representations

        **Authors**: *Fukun Yin, Wen Liu, Zilong Huang, Pei Cheng, Tao Chen, Gang Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/525d24400247f884c3419b0b7b1c4829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/525d24400247f884c3419b0b7b1c4829-Abstract-Conference.html)

        **Abstract**:

        Implicit neural 3D representation has achieved impressive results in surface or scene reconstruction and novel view synthesis, which typically uses the coordinate-based multi-layer perceptrons (MLPs) to learn a continuous scene representation. However, existing approaches, such as Neural Radiance Field (NeRF) and its variants, usually require dense input views (i.e. 50-150) to obtain decent results. To relive the over-dependence on massive calibrated images and enrich the coordinate-based feature representation, we explore injecting the prior information into the coordinate-based network and introduce a novel coordinate-based model, CoCo-INR, for implicit neural 3D representation. The cores of our method are two attention modules: codebook attention and coordinate attention. The former extracts the useful prototypes containing rich geometry and appearance information from the prior codebook, and the latter propagates such prior information into each coordinate and enriches its feature representation for a scene or object surface. With the help of the prior information, our method can render 3D views with more photo-realistic appearance and geometries than the current methods using fewer calibrated images available. Experiments on various scene reconstruction datasets, including DTU and BlendedMVS, and the full 3D head reconstruction dataset, H3DS, demonstrate the robustness under fewer input views and fine detail-preserving capability of our proposed method.

        ----

        ## [923] Scalable Neural Video Representations with Learnable Positional Features

        **Authors**: *Subin Kim, Sihyun Yu, Jaeho Lee, Jinwoo Shin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5297e56ac65ba2bfa70ee9fc4818c042-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5297e56ac65ba2bfa70ee9fc4818c042-Abstract-Conference.html)

        **Abstract**:

        Succinct representation of complex signals using coordinate-based neural representations (CNRs) has seen great progress, and several recent efforts focus on extending them for handling videos. Here, the main challenge is how to (a) alleviate a compute-inefficiency in training CNRs to (b) achieve high-quality video encoding while (c) maintaining the parameter-efficiency. To meet all requirements (a), (b), and (c) simultaneously, we propose neural video representations with learnable positional features (NVP), a novel CNR by introducing "learnable positional features" that effectively amortize a video as latent codes. Specifically, we first present a CNR architecture based on designing 2D latent keyframes to learn the common video contents across each spatio-temporal axis, which dramatically improves all of those three requirements. Then, we propose to utilize existing powerful image and video codecs as a compute-/memory-efficient compression procedure of latent codes. We demonstrate the superiority of NVP on the popular UVG benchmark; compared with prior arts, NVP not only trains 2 times faster (less than 5 minutes) but also exceeds their encoding quality as 34.07$\rightarrow$34.57 (measured with the PSNR metric), even using $>$8 times fewer parameters. We also show intriguing properties of NVP, e.g., video inpainting, video frame interpolation, etc.

        ----

        ## [924] Data Augmentation MCMC for Bayesian Inference from Privatized Data

        **Authors**: *Nianqiao Ju, Jordan Awan, Ruobin Gong, Vinayak Rao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/529d0f9b0fb7c8d4b7d52221faee48d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/529d0f9b0fb7c8d4b7d52221faee48d6-Abstract-Conference.html)

        **Abstract**:

        Differentially private mechanisms protect privacy by introducing additional randomness into the data. Restricting access to only the privatized data makes it challenging to perform valid statistical inference on parameters underlying the confidential data. Specifically, the likelihood function of the privatized data requires integrating over the large space of confidential databases and is typically intractable. For Bayesian analysis, this results in a posterior distribution that is doubly intractable, rendering traditional MCMC techniques inapplicable. We propose an MCMC framework to perform Bayesian inference from the privatized data, which is applicable to a wide range of statistical models and privacy mechanisms. Our MCMC algorithm augments the model parameters with the unobserved confidential data, and alternately updates each one. For the potentially challenging step of updating the confidential data, we propose a generic approach that exploits the privacy guarantee of the mechanism to ensure efficiency. We give results on the computational complexity, acceptance rate, and mixing properties of our MCMC. We illustrate the efficacy and applicability of our methods on a na√Øve-Bayes log-linear model and on a linear regression model.

        ----

        ## [925] Neural Circuit Architectural Priors for Embodied Control

        **Authors**: *Nikhil X. Bhattasali, Anthony M. Zador, Tatiana A. Engel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/52e431bd7689d98426300cb103bb0ee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/52e431bd7689d98426300cb103bb0ee3-Abstract-Conference.html)

        **Abstract**:

        Artificial neural networks for motor control usually adopt generic architectures like fully connected MLPs. While general, these tabula rasa architectures rely on large amounts of experience to learn, are not easily transferable to new bodies, and have internal dynamics that are difficult to interpret. In nature, animals are born with highly structured connectivity in their nervous systems shaped by evolution; this innate circuitry acts synergistically with learning mechanisms to provide inductive biases that enable most animals to function well soon after birth and learn efficiently. Convolutional networks inspired by visual circuitry have encoded useful biases for vision. However, it is unknown the extent to which ANN architectures inspired by neural circuitry can yield useful biases for other AI domains. In this work, we ask what advantages biologically inspired ANN architecture can provide in the domain of motor control. Specifically, we translate C. elegans locomotion circuits into an ANN model controlling a simulated Swimmer agent. On a locomotion task, our architecture achieves good initial performance and asymptotic performance comparable with MLPs, while dramatically improving data efficiency and requiring orders of magnitude fewer parameters. Our architecture is interpretable and transfers to new body designs. An ablation analysis shows that constrained excitation/inhibition is crucial for learning, while weight initialization contributes to good initial performance. Our work demonstrates several advantages of biologically inspired ANN architecture and encourages future work in more complex embodied control.

        ----

        ## [926] Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization

        **Authors**: *Samuel Daulton, Xingchen Wan, David Eriksson, Maximilian Balandat, Michael A. Osborne, Eytan Bakshy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/531230cfac80c65017ad0f85d3031edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/531230cfac80c65017ad0f85d3031edc-Abstract-Conference.html)

        **Abstract**:

        Optimizing expensive-to-evaluate black-box functions of discrete (and potentially continuous) design parameters is a ubiquitous problem in scientific and engineering applications. Bayesian optimization (BO) is a popular, sample-efficient method that leverages a probabilistic surrogate model and  an acquisition function (AF) to select promising designs to evaluate. However, maximizing the AF over mixed or high-cardinality discrete search spaces is challenging standard gradient-based methods cannot be used directly or evaluating the AF at every point in the search space would be computationally prohibitive. To address this issue, we propose using probabilistic reparameterization (PR). Instead of directly optimizing the AF over the search space containing discrete parameters, we instead maximize the expectation of the AF over a probability distribution defined by continuous parameters. We prove that under suitable reparameterizations, the BO policy that maximizes the probabilistic objective is the same as that which maximizes the AF, and therefore, PR enjoys the same regret bounds as the original BO policy using the underlying AF. Moreover, our approach provably converges to a stationary point of the probabilistic objective under gradient ascent using scalable, unbiased estimators of both the probabilistic objective and its gradient. Therefore, as the number of starting points and gradient steps increase, our approach will recover of a maximizer of the AF (an often-neglected requisite for commonly used BO regret bounds). We validate our approach empirically and demonstrate state-of-the-art optimization performance on a wide range of real-world applications. PR is complementary to (and benefits) recent work and naturally generalizes to settings with multiple objectives and black-box constraints.

        ----

        ## [927] Learning to Constrain Policy Optimization with Virtual Trust Region

        **Authors**: *Thai Hung Le, Thommen Karimpanal George, Majid Abdolshah, Dung Nguyen, Kien Do, Sunil Gupta, Svetha Venkatesh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/531998dc1fc858b5857a90b74d96ecab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/531998dc1fc858b5857a90b74d96ecab-Abstract-Conference.html)

        **Abstract**:

        We introduce a constrained optimization method for policy gradient reinforcement learning, which uses two trust regions to regulate each policy update. In addition to using the proximity of one single old policy as the first trust region as done by prior works, we propose forming a second trust region by constructing another virtual policy that represents a wide range of past policies. We then enforce the new policy to stay closer to the virtual policy, which is beneficial if the old policy performs poorly. We propose a mechanism to automatically build the virtual policy from a memory buffer of past policies, providing a new capability for dynamically selecting appropriate trust regions during the optimization process. Our proposed method, dubbed Memory-Constrained Policy Optimization (MCPO), is examined in diverse environments, including robotic locomotion control, navigation with sparse rewards and Atari games, consistently demonstrating competitive performance against recent on-policy constrained policy gradient methods.

        ----

        ## [928] Verification and search algorithms for causal DAGs

        **Authors**: *Davin Choo, Kirankumar Shiragur, Arnab Bhattacharyya*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5340b0c0b76dc0115f5cc91c20c1251d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5340b0c0b76dc0115f5cc91c20c1251d-Abstract-Conference.html)

        **Abstract**:

        We study two problems related to recovering causal graphs from interventional data: (i) $\textit{verification}$, where the task is to check if a purported causal graph is correct, and (ii) $\textit{search}$, where the task is to recover the correct causal graph. For both, we wish to minimize the number of interventions performed. For the first problem, we give a characterization of a minimal sized set of atomic interventions that is necessary and sufficient to check the correctness of a claimed causal graph. Our characterization uses the notion of $\textit{covered edges}$, which enables us to obtain simple proofs and also easily reason about earlier known results. We also generalize our results to the settings of bounded size interventions and node-dependent interventional costs. For all the above settings, we provide the first known provable algorithms for efficiently computing (near)-optimal verifying sets on general graphs. For the second problem, we give a simple adaptive algorithm based on graph separators that produces an atomic intervention set which fully orients any essential graph while using $\mathcal{O}(\log n)$ times the optimal number of interventions needed to $\textit{verify}$ (verifying size) the underlying DAG on $n$ vertices. This approximation is tight as $\textit{any}$ search algorithm on an essential line graph has worst case approximation ratio of $\Omega(\log n)$ with respect to the verifying size. With bounded size interventions, each of size $\leq k$, our algorithm gives an $\mathcal{O}(\log n \cdot \log k)$ factor approximation. Our result is the first known algorithm that gives a non-trivial approximation guarantee to the verifying size on general unweighted graphs and with bounded size interventions.

        ----

        ## [929] Causally motivated multi-shortcut identification and removal

        **Authors**: *Jiayun Zheng, Maggie Makar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/536d643875321d6c3282ee8c7ea5eb6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/536d643875321d6c3282ee8c7ea5eb6a-Abstract-Conference.html)

        **Abstract**:

        For predictive models to provide reliable guidance in decision making processes, they are often required to be accurate and robust to distribution shifts. Shortcut learning--where a model relies on spurious correlations or shortcuts to predict the target label--undermines the robustness property, leading to models with poor out-of-distribution accuracy despite good in-distribution performance. Existing work on shortcut learning either assumes that the set of possible shortcuts is known a priori or is discoverable using interpretability methods such as saliency maps, which might not always be true. Instead, we propose a two step approach to (1) efficiently identify relevant shortcuts, and (2) leverage the identified shortcuts to build models that are robust to distribution shifts. Our approach relies on having access to a (possibly) high dimensional set of auxiliary labels at training time, some of which correspond to possible shortcuts. We show both theoretically and empirically that our approach is able to identify a sufficient set of shortcuts leading to more efficient predictors in finite samples.

        ----

        ## [930] Avalon: A Benchmark for RL Generalization Using Procedurally Generated Worlds

        **Authors**: *Joshua Albrecht, Abraham J. Fetterman, Bryden Fogelman, Ellie Kitanidis, Bartosz Wróblewski, Nicole Seo, Michael Rosenthal, Maksis Knutins, Zack Polizzi, James Simon, Kanjun Qiu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/539f1f7dd156cfe1222b0be83f247d35-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/539f1f7dd156cfe1222b0be83f247d35-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Despite impressive successes, deep reinforcement learning (RL) systems still fall short of human performance on generalization to new tasks and environments that differ from their training. As a benchmark tailored for studying RL generalization, we introduce Avalon, a set of tasks in which embodied agents in highly diverse procedural 3D worlds must survive by navigating terrain, hunting or gathering food, and avoiding hazards. Avalon is unique among existing RL benchmarks in that the reward function, world dynamics, and action space are the same for every task, with tasks differentiated solely by altering the environment; its 20 tasks, ranging in complexity from eat and throw to hunt and navigate, each create worlds in which the agent must perform specific skills in order to survive. This setup enables investigations of generalization within tasks, between tasks, and to compositional tasks that require combining skills learned from previous tasks. Avalon includes a highly efficient simulator, a library of baselines, and a benchmark with scoring metrics evaluated against hundreds of hours of human performance, all of which are open-source and publicly available. We find that standard RL baselines make progress on most tasks but are still far from human performance, suggesting Avalon is challenging enough to advance the quest for generalizable RL.

        ----

        ## [931] Learning Equivariant Segmentation with Instance-Unique Querying

        **Authors**: *Wenguan Wang, James Liang, Dongfang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/53a525a5f8910609263ffd130ef370b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/53a525a5f8910609263ffd130ef370b8-Abstract-Conference.html)

        **Abstract**:

        Prevalent state-of-the-art instance segmentation methods fall into a query-based scheme, in which instance masks are derived by querying the image feature using a set of instance-aware embeddings. In this work, we devise a new training framework that boosts query-based models through discriminative query embedding learning. It explores two essential properties, namely dataset-level uniqueness and transformation equivariance, of the relation between queries and instances. First, our algorithm uses the queries to retrieve the corresponding instances from the whole training dataset, instead of only searching within individual scenes. As querying instances across scenes is more challenging, the segmenters are forced to learn more discriminative queries for effective instance separation. Second, our algorithm encourages both image (instance) representations and queries to be equivariant against geometric transformations, leading to more robust, instance-query matching. On top of four famous, query-based models (i.e., CondInst, SOLOv2, SOTR, and Mask2Former), our training algorithm provides significant performance gains (e.g., +1.6 â€“ 3.2 AP) on COCO dataset. In addition, our algorithm promotes the performance of SOLOv2 by 2.7 AP, on LVISv1 dataset.

        ----

        ## [932] NeuPhysics: Editable Neural Geometry and Physics from Monocular Videos

        **Authors**: *Yi-Ling Qiao, Alexander Gao, Ming C. Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html)

        **Abstract**:

        We present a method for learning 3D geometry and physics parameters of a dynamic scene from only a monocular RGB video input. To decouple the learning of underlying scene geometry from dynamic motion, we represent the scene as a time-invariant signed distance function (SDF) which serves as a reference frame, along with a time-conditioned deformation field. We further bridge this neural geometry representation with a differentiable physics simulator by designing a two-way conversion between the neural field and its corresponding hexahedral mesh, enabling us to estimate physics parameters from the source video by minimizing a cycle consistency loss. Our method also allows a user to interactively edit 3D objects from the source video by modifying the recovered hexahedral mesh, and propagating the operation back to the neural field representation. Experiments show that our method achieves superior mesh and video reconstruction of dynamic scenes compared to competing Neural Field approaches, and we provide extensive examples which demonstrate its ability to extract useful 3D representations from videos captured with consumer-grade cameras.

        ----

        ## [933] Measuring Data Reconstruction Defenses in Collaborative Inference Systems

        **Authors**: *Mengda Yang, Ziang Li, Juan Wang, Hongxin Hu, Ao Ren, Xiaoyang Xu, Wenzhe Yi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/53f1c3ec5df814b5aabe9ae88a29bb49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/53f1c3ec5df814b5aabe9ae88a29bb49-Abstract-Conference.html)

        **Abstract**:

        The collaborative inference systems are designed to speed up the prediction processes in edge-cloud scenarios, where the local devices and the cloud system work together to run a complex deep-learning model. However, those edge-cloud collaborative inference systems are vulnerable to emerging reconstruction attacks, where malicious cloud service providers are able to recover the edge-side usersâ€™ private data. To defend against such attacks, several defense countermeasures have been recently introduced. Unfortunately, little is known about the robustness of those defense countermeasures. In this paper, we take the first step towards measuring the robustness of those state-of-the-art defenses with respect to reconstruction attacks. Specifically, we show that the latent privacy features are still retained in the obfuscated representations. Motivated by such an observation, we design a technology called Sensitive Feature Distillation (SFD) to restore sensitive information from the protected feature representations. Our experiments show that SFD can break through defense mechanisms in model partitioning scenarios, demonstrating the inadequacy of existing defense mechanisms as a privacy-preserving technique against reconstruction attacks. We hope our findings inspire further work in improving the robustness of defense mechanisms against reconstruction attacks for collaborative inference systems.

        ----

        ## [934] ZARTS: On Zero-order Optimization for Neural Architecture Search

        **Authors**: *Xiaoxing Wang, Wenxuan Guo, Jianlin Su, Xiaokang Yang, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/53f2c82c6b165a963b353194113ee71e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/53f2c82c6b165a963b353194113ee71e-Abstract-Conference.html)

        **Abstract**:

        Differentiable architecture search (DARTS) has been a popular one-shot paradigm for NAS due to its high efficiency. It introduces trainable architecture parameters to represent the importance of candidate operations and proposes first/second-order approximation to estimate their gradients, making it possible to solve NAS by gradient descent algorithm. However, our in-depth empirical results show that the approximation often distorts the loss landscape, leading to the biased objective to optimize and, in turn, inaccurate gradient estimation for architecture parameters. This work turns to zero-order optimization and proposes a novel NAS scheme, called ZARTS, to search without enforcing the above approximation. Specifically, three representative zero-order optimization methods are introduced: RS, MGS, and GLD, among which MGS performs best by balancing the accuracy and speed. Moreover, we explore the connections between RS/MGS and gradient descent algorithm and show that our ZARTS can be seen as a robust gradient-free counterpart to DARTS. Extensive experiments on multiple datasets and search spaces show the remarkable performance of our method. In particular, results on 12 benchmarks verify the outstanding robustness of ZARTS, where the performance of DARTS collapses due to its known instability issue. Also, we search on the search space of DARTS to compare with peer methods, and our discovered architecture achieves 97.54\% accuracy on CIFAR-10 and 75.7\% top-1 accuracy on ImageNet. Finally, we combine our ZARTS with three orthogonal variants of DARTS for faster search speed and better performance.  Source code will be made publicly available at:  \url{https://github.com/vicFigure/ZARTS}.

        ----

        ## [935] Make Some Noise: Reliable and Efficient Single-Step Adversarial Training

        **Authors**: *Pau de Jorge Aranda, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5434a6b40f8f65488e722bc33d796c8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5434a6b40f8f65488e722bc33d796c8b-Abstract-Conference.html)

        **Abstract**:

        Recently, Wong et al. (2020) showed that adversarial training with single-step FGSM leads to a characteristic failure mode named catastrophic overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. Experimentally they showed that simply adding a random perturbation prior to FGSM (RS-FGSM) could prevent CO. However,  Andriushchenko & Flammarion (2020) observed that RS-FGSM still leads to CO for larger perturbations, and proposed a computationally expensive regularizer (GradAlign) to avoid it. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with \textit{not clipping} is highly effective in avoiding CO for large perturbation radii. We then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous state of-the-art GradAlign while achieving 3$\times$ speed-up.

        ----

        ## [936] Structural Pruning via Latency-Saliency Knapsack

        **Authors**: *Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu, José M. Álvarez*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5434be94e82c54327bb9dcaf7fca52b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5434be94e82c54327bb9dcaf7fca52b6-Abstract-Conference.html)

        **Abstract**:

        Structural pruning can simplify network architecture and improve inference speed. We propose Hardware-Aware Latency Pruning (HALP) that formulates structural pruning as a global resource allocation optimization problem, aiming at maximizing the accuracy while constraining latency under a predefined budget on targeting device. For filter importance ranking, HALP leverages latency lookup table to track latency reduction potential and global saliency score to gauge accuracy drop. Both metrics can be evaluated very efficiently during pruning, allowing us to reformulate global structural pruning under a reward maximization problem given target constraint. This makes the problem solvable via our augmented knapsack solver, enabling HALP to surpass prior work in pruning efficacy and accuracy-efficiency trade-off. We examine HALP on both classification and detection tasks, over varying networks, on ImageNet and VOC datasets, on different platforms. In particular, for ResNet-50/-101 pruning on ImageNet, HALP improves network throughput by $1.60\times$/$1.90\times$ with $+0.3\%$/$-0.2\%$ top-1 accuracy changes, respectively. For SSD pruning on VOC, HALP improves throughput by $1.94\times$ with only a $0.56$ mAP drop. HALP consistently outperforms prior art, sometimes by large margins. Project page at \url{https://halp-neurips.github.io/}.

        ----

        ## [937] Risk Bounds of Multi-Pass SGD for Least Squares in the Interpolation Regime

        **Authors**: *Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, Sham M. Kakade*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/543924fdf260ba990f2ef84f940f3db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/543924fdf260ba990f2ef84f940f3db2-Abstract-Conference.html)

        **Abstract**:

        Stochastic gradient descent (SGD) has achieved great success due to its superior performance in both optimization and generalization. Most of existing generalization analyses are made for single-pass SGD, which is a less practical variant compared to the commonly-used multi-pass SGD. Besides, theoretical analyses for multi-pass SGD often concern a worst-case instance in a class of problems, which may be pessimistic to explain the superior generalization ability for some particular problem instance. The goal of this paper is to provide an instance-dependent excess risk bound of multi-pass SGD for least squares in the interpolation regime, which is expressed as a function of the iteration number, stepsize, and data covariance. We show that the excess risk of SGD can be exactly decomposed into the excess risk of GD and a positive fluctuation error, suggesting that SGD always performs worse, instance-wisely, than GD, in generalization. On the other hand, we show that although SGD needs more iterations than GD to achieve the same level of excess risk, it saves the number of stochastic gradient evaluations, and therefore is preferable in terms of computational time.

        ----

        ## [938] Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal

        **Authors**: *Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/544696ef4847c903376ed6ec58f3a703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/544696ef4847c903376ed6ec58f3a703-Abstract-Conference.html)

        **Abstract**:

        Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.

        ----

        ## [939] EfficientFormer: Vision Transformers at MobileNet Speed

        **Authors**: *Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Abstract-Conference.html)

        **Abstract**:

        Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm. Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves $79.2\%$ top-1 accuracy on ImageNet-1K with only $1.6$ ms inference latency on iPhone 12 (compiled with CoreML), which runs as fast as MobileNetV2$\times 1.4$ ($1.6$ ms, $74.7\%$ top-1), and our largest model, EfficientFormer-L7, obtains $83.3\%$ accuracy with only $7.0$ ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance.

        ----

        ## [940] Holomorphic Equilibrium Propagation Computes Exact Gradients Through Finite Size Oscillations

        **Authors**: *Axel Laborieux, Friedemann Zenke*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/545a114e655f9d25ba0d56ea9a01fc6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/545a114e655f9d25ba0d56ea9a01fc6e-Abstract-Conference.html)

        **Abstract**:

        Equilibrium propagation (EP) is an alternative to backpropagation (BP) that allows the training of deep neural networks with local learning rules. It thus provides a compelling framework for training neuromorphic systems and understanding learning in neurobiology. However, EP requires infinitesimal teaching signals, thereby limiting its applicability to noisy physical systems. Moreover, the algorithm requires separate temporal phases and has not been applied to large-scale problems. Here we address these issues by extending EP to holomorphic networks. We show analytically that this extension naturally leads to exact gradients for finite-amplitude teaching signals. Importantly, the gradient can be computed as the first Fourier coefficient from finite neuronal activity oscillations in continuous time without requiring separate phases. Further, we demonstrate in numerical simulations that our approach permits robust estimation of gradients in the presence of noise and that deeper models benefit from the finite teaching signals. Finally, we establish the first benchmark for EP on the ImageNet $32 \times 32$ dataset and show that it matches the performance of an equivalent network trained with BP. Our work provides analytical insights that enable scaling EP to large-scale problems and establishes a formal framework for how oscillations could support learning in biological and neuromorphic systems.

        ----

        ## [941] Learning Substructure Invariance for Out-of-Distribution Molecular Representations

        **Authors**: *Nianzu Yang, Kaipeng Zeng, Qitian Wu, Xiaosong Jia, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/547108084f0c2af39b956f8eadb75d1b-Abstract-Conference.html)

        **Abstract**:

        Molecule representation learning (MRL) has been extensively studied and current methods have shown promising power for various tasks, e.g., molecular property prediction and target  identification. However, a common hypothesis of existing methods is that either the model development or experimental evaluation is mostly based on i.i.d. data across training and testing. Such a hypothesis can be violated in real-world applications where testing molecules could come from new environments, bringing about serious performance degradation or unexpected prediction. We propose a new representation learning framework entitled MoleOOD to enhance the robustness of MRL models against such distribution shifts, motivated by an observation that the (bio)chemical properties of molecules are usually invariantly associated with certain privileged molecular substructures across different environments (e.g., scaffolds, sizes, etc.). Specifically, We introduce an environment inference model to identify the latent factors that impact data generation from different distributions in a fully data-driven manner. We also propose a new learning objective to guide the molecule encoder to leverage environment-invariant substructures that more stably relate with the labels across environments. Extensive experiments on ten real-world datasets demonstrate that our model has a stronger generalization ability than existing methods under various out-of-distribution (OOD) settings, despite the absence of manual specifications of environments. Particularly, our method achieves up to 5.9\% and 3.9\% improvement over the strongest baselines on OGB and DrugOOD benchmarks in terms of ROC-AUC, respectively. Our source code is publicly available at \url{https://github.com/yangnianzu0515/MoleOOD}.

        ----

        ## [942] The Dollar Street Dataset: Images Representing the Geographic and Socioeconomic Diversity of the World

        **Authors**: *William Gaviria Rojas, Sudnya Frederick Diamos, Keertan Kini, David Kanter, Vijay Janapa Reddi, Cody Coleman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5474d9d43c0519aa176276ff2c1ca528-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/5474d9d43c0519aa176276ff2c1ca528-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        It is crucial that image datasets for computer vision are representative and contain accurate demographic information to ensure their robustness and fairness, especially for smaller subpopulations. To address this issue, we present Dollar Street - a supervised dataset that contains 38,479 images of everyday household items from homes around the world. This dataset was manually curated and fully labeled, including tags for objects (e.g. “toilet,” “toothbrush,” “stove”) and demographic data such as region, country and home monthly income. This dataset includes images from homes with no internet access and incomes as low as \$26.99 per month, visually capturing valuable socioeconomic diversity of traditionally under-represented populations. All images and data are licensed under CC-BY, permitting their use in academic and commercial work. Moreover, we show that this dataset can improve the performance of classification tasks for images of household items from lower income homes, addressing a critical need for datasets that combat bias.

        ----

        ## [943] LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning

        **Authors**: *Yi-Lin Sung, Jaemin Cho, Mohit Bansal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html)

        **Abstract**:

        Fine-tuning large pre-trained models on downstream tasks has been adopted in a variety of domains recently. However, it is costly to update the entire parameter set of large pre-trained models. Although recently proposed parameter-efficient transfer learning (PETL) techniques allow updating a small subset of parameters (e.g. only using 2% of parameters) inside a pre-trained backbone network for a new task, they only reduce the training memory requirement by up to 30%. This is because the gradient computation for the trainable parameters still requires back-propagation through the large pre-trained backbone model. To address this, we propose Ladder Side-Tuning (LST), a new PETL technique that can reduce training memory requirements by more substantial amounts. Unlike existing parameter-efficient methods that insert additional parameters inside backbone networks, we train a ladder side network, a small and separate network that takes intermediate activations as input via shortcut connections (ladders) from backbone networks and makes predictions. LST has significantly lower memory requirements than previous methods, because it does not require back-propagation through the backbone network, but instead only through the side network and ladder connections. We evaluate our method with various models (T5 and CLIP-T5) on both natural language processing (GLUE) and vision-and-language (VQA, GQA, NLVR2, MSCOCO) tasks. LST saves 69% of the memory costs to fine-tune the whole network, while other methods only save 26% of that in similar parameter usages (hence, 2.7x more memory savings). Moreover, LST achieves higher accuracy than Adapter and LoRA in a low-memory regime. To further show the advantage of this better memory efficiency, we also apply LST to larger T5 models (T5-large, T5-3B), attaining better GLUE performance than full fine-tuning and other PETL methods. The trend also holds in the experiments on vision-and-language tasks, where LST achieves similar accuracy to other PETL methods when training a similar number of parameters while also having 2.7x more memory savings. Our code is available at: https://github.com/ylsung/Ladder-Side-Tuning.

        ----

        ## [944] CGLB: Benchmark Tasks for Continual Graph Learning

        **Authors**: *Xikun Zhang, Dongjin Song, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/548a41b9cac6f50dccf7e63e9e1b1b9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/548a41b9cac6f50dccf7e63e9e1b1b9b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Continual learning on graph data, which aims to accommodate new tasks over newly emerged graph data while maintaining the model performance over existing tasks, is attracting increasing attention from the community. Unlike continual learning on Euclidean data ($\textit{e.g.}$, images, texts, etc.) that has established benchmarks and unified experimental settings, benchmark tasks are rare for Continual Graph Learning (CGL). Moreover, due to the variety of graph data and its complex topological structures, existing works adopt different protocols to configure datasets and experimental settings. This creates a great obstacle to compare different techniques and thus hinders the development of CGL. To this end, we systematically study the task configurations in different application scenarios and develop a comprehensive Continual Graph Learning Benchmark (CGLB) curated from different public datasets. Specifically, CGLB contains both node-level and graph-level continual graph learning tasks under task-incremental (currently widely adopted) and class-incremental (more practical, challenging, yet underexplored) settings, as well as a toolkit for training, evaluating, and visualizing different CGL methods. Within CGLB, we also systematically explain the difference among these task configurations by comparing them to classical continual learning settings. Finally, we comprehensively compare state-of-the-art baselines on CGLB to investigate their effectiveness. Given CGLB and the developed toolkit, the barrier to exploring CGL has been greatly lowered and researchers can focus more on the model development without worrying about tedious work on pre-processing of datasets or encountering unseen pitfalls. The benchmark and the toolkit are available through https://github.com/QueuQ/CGLB.

        ----

        ## [945] Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning

        **Authors**: *Zhecheng Yuan, Zhengrong Xue, Bo Yuan, Xueqian Wang, Yi Wu, Yang Gao, Huazhe Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/548a482d4496ce109cddfbeae5defa7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/548a482d4496ce109cddfbeae5defa7d-Abstract-Conference.html)

        **Abstract**:

        Learning generalizable policies that can adapt to unseen environments remains challenging in visual Reinforcement Learning (RL). Existing approaches try to acquire a robust representation via diversifying the appearances of in-domain observations for better generalization. Limited by the specific observations of the environment, these methods ignore the possibility of exploring diverse real-world image datasets. In this paper, we investigate how a visual RL agent would benefit from the off-the-shelf visual representations. Surprisingly, we find that the early layers in an ImageNet pre-trained ResNet model could provide rather generalizable representations for visual RL. Hence, we propose Pre-trained Image Encoder for Generalizable visual reinforcement learning (PIE-G), a simple yet effective framework that can generalize to the unseen visual scenarios in a zero-shot manner. Extensive experiments are conducted on DMControl Generalization Benchmark, DMControl Manipulation Tasks, Drawer World, and CARLA to verify the effectiveness of PIE-G. Empirical evidence suggests PIE-G improves sample efficiency and significantly outperforms previous state-of-the-art methods in terms of generalization performance. In particular, PIE-G boasts a 55% generalization performance gain on average in the challenging video background setting. Project Page: https://sites.google.com/view/pie-g/home.

        ----

        ## [946] Amortized Inference for Heterogeneous Reconstruction in Cryo-EM

        **Authors**: *Axel Levy, Gordon Wetzstein, Julien N. P. Martel, Frédéric Poitevin, Ellen D. Zhong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54b8b4e0b4ba4aad112e84f32e3b5dbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54b8b4e0b4ba4aad112e84f32e3b5dbb-Abstract-Conference.html)

        **Abstract**:

        Cryo-electron microscopy (cryo-EM) is an imaging modality that provides unique insights into the dynamics of proteins and other building blocks of life. The algorithmic challenge of jointly estimating the poses, 3D structure, and conformational heterogeneity of a biomolecule from millions of noisy and randomly oriented 2D projections in a computationally efficient manner, however, remains unsolved. Our method, cryoFIRE, performs ab initio heterogeneous reconstruction with unknown poses in an amortized framework, thereby avoiding the computationally expensive step of pose search while enabling the analysis of conformational heterogeneity. Poses and conformation are jointly estimated by an encoder while a physics-based decoder aggregates the images into an implicit neural representation of the conformational space. We show that our method can provide one order of magnitude speedup on datasets containing millions of images, without any loss of accuracy. We validate that the joint estimation of poses and conformations can be amortized over the size of the dataset. For the first time, we prove that an amortized method can extract interpretable dynamic information from experimental datasets.

        ----

        ## [947] RKHS-SHAP: Shapley Values for Kernel Methods

        **Authors**: *Siu Lun Chau, Robert Hu, Javier González, Dino Sejdinovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54bb63eaec676b87a2278a22b1bd02a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54bb63eaec676b87a2278a22b1bd02a2-Abstract-Conference.html)

        **Abstract**:

        Feature attribution for kernel methods is often heuristic and not individualised for each prediction. To address this, we turn to the concept of Shapley values (SV), a coalition game theoretical framework that has previously been applied to different machine learning model interpretation tasks, such as linear models, tree ensembles and deep networks. By analysing SVs from a functional perspective, we propose RKHS-SHAP, an attribution method for kernel machines that can efficiently compute both Interventional and Observational Shapley values using kernel mean embeddings of distributions. We show theoretically that our method is robust with respect to local perturbations - a key yet often overlooked desideratum for consistent model interpretation. Further, we propose Shapley regulariser, applicable to a general empirical risk minimisation framework, allowing learning while controlling the level of specific feature's contributions to the model. We demonstrate that the Shapley regulariser enables learning which is robust to covariate shift of a given feature and fair learning which controls the SVs of sensitive features.

        ----

        ## [948] Single-Stage Visual Relationship Learning using Conditional Queries

        **Authors**: *Alakh Desai, Tz-Ying Wu, Subarna Tripathi, Nuno Vasconcelos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54d2d38a56a74387d5916ee40e462295-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54d2d38a56a74387d5916ee40e462295-Abstract-Conference.html)

        **Abstract**:

        Research in scene graph generation (SGG) usually considers two-stage models, that is, detecting a set of entities, followed by combining them and labeling all possible relationships. While showing promising results, the pipeline structure induces large parameter and computation overhead, and typically hinders end-to-end optimizations. To address this, recent research attempts to train single-stage models that are more computationally efficient. With the advent of DETR, a set-based detection model, one-stage models attempt to predict a set of subject-predicate-object triplets directly in a single shot. However, SGG is inherently a multi-task learning problem that requires modeling entity and predicate distributions simultaneously. In this paper, we propose Transformers with conditional queries for SGG, namely, TraCQ with a new formulation for SGG that avoids the multi-task learning problem and the combinatorial entity pair distribution. We employ a DETR-based encoder-decoder design and leverage conditional queries to significantly reduce the entity label space as well, which leads to 20% fewer parameters compared to state-of-the-art one-stage models. Experimental results show that TraCQ not only outperforms existing single-stage scene graph generation methods, it also beats state-of-the-art two-stage methods on the Visual Genome dataset, yet is capable of end-to-end training and faster inference.

        ----

        ## [949] Unlabelled Sample Compression Schemes for Intersection-Closed Classes and Extremal Classes

        **Authors**: *Joachim Hyam Rubinstein, Benjamin I. P. Rubinstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html)

        **Abstract**:

        The sample compressibility of concept classes plays an important role in learning theory, as a sufficient condition for PAC learnability, and more recently as an avenue for robust generalisation in adaptive data analysis. Whether compression schemes of size $O(d)$ must necessarily exist for all classes of VC dimension $d$ is unknown, but conjectured to be true by Warmuth. Recently Chalopin, Chepoi, Moran, and Warmuth (2018) gave a beautiful unlabelled sample compression scheme of size VC dimension for all maximum classes: classes that meet the Sauer-Shelah-Perles Lemma with equality. They also offered a counterexample to compression schemes based on a promising approach known as corner peeling. In this paper we simplify and extend their proof technique to deal with so-called extremal classes of VC dimension $d$ which contain maximum classes of VC dimension $d-1$. A criterion is given which would imply that all extremal classes admit unlabelled compression schemes of size $d$. We also prove that all intersection-closed classes with VC dimension $d$ admit unlabelled compression schemes of size at most $11d$.

        ----

        ## [950] VCT: A Video Compression Transformer

        **Authors**: *Fabian Mentzer, George Toderici, David Minnen, Sergi Caelles, Sung Jin Hwang, Mario Lucic, Eirikur Agustsson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html)

        **Abstract**:

        We show how transformers can be used to vastly simplify neural video compression. Previous methods have been relying on an increasing number of architectural biases and priors, including motion prediction and warping operations, resulting in complex models. Instead, we independently map input frames to representations and use a transformer to model their dependencies, letting it predict the distribution of future representations given the past. The resulting video compression transformer outperforms previous methods on standard video compression data sets. Experiments on synthetic data show that our model learns to handle complex motion patterns such as panning, blurring and fading purely from data. Our approach is easy to implement, and we release code to facilitate future research.

        ----

        ## [951] Amortized Inference for Causal Structure Learning

        **Authors**: *Lars Lorch, Scott Sussex, Jonas Rothfuss, Andreas Krause, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/54f7125dee9b8b3dc798bb9a082b09e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/54f7125dee9b8b3dc798bb9a082b09e2-Abstract-Conference.html)

        **Abstract**:

        Inferring causal structure poses a combinatorial search problem that typically involves evaluating structures with a score or independence test. The resulting search is costly, and designing suitable scores or tests that capture prior knowledge is difficult. In this work, we propose to amortize causal structure learning. Rather than searching over structures, we train a variational inference model to directly predict the causal structure from observational or interventional data. This allows our inference model to acquire domain-specific inductive biases for causal discovery solely from data generated by a simulator, bypassing both the hand-engineering of suitable score functions and the search over graphs. The architecture of our inference model emulates permutation invariances that are crucial for statistical efficiency in structure learning, which facilitates generalization to significantly larger problem instances than seen during training. On synthetic data and semisynthetic gene expression data, our models exhibit robust generalization capabilities when subject to substantial distribution shifts and significantly outperform existing algorithms, especially in the challenging genomics domain. Our code and models are publicly available at: https://github.com/larslorch/avici

        ----

        ## [952] Emergent Graphical Conventions in a Visual Communication Game

        **Authors**: *Shuwen Qiu, Sirui Xie, Lifeng Fan, Tao Gao, Jungseock Joo, Song-Chun Zhu, Yixin Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/550ff553efc2c58410f277c667d12786-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/550ff553efc2c58410f277c667d12786-Abstract-Conference.html)

        **Abstract**:

        Humans communicate with graphical sketches apart from symbolic languages. Primarily focusing on the latter, recent studies of emergent communication overlook the sketches; they do not account for the evolution process through which symbolic sign systems emerge in the trade-off between iconicity and symbolicity. In this work, we take the very first step to model and simulate this process via two neural agents playing a visual communication game; the sender communicates with the receiver by sketching on a canvas. We devise a novel reinforcement learning method such that agents are evolved jointly towards successful communication and abstract graphical conventions. To inspect the emerged conventions, we define three key properties -- iconicity, symbolicity, and semanticity -- and design evaluation methods accordingly. Our experimental results under different controls are consistent with the observation in studies of human graphical conventions. Of note, we find that evolved sketches can preserve the continuum of semantics under proper environmental pressures. More interestingly, co-evolved agents can switch between conventionalized and iconic communication based on their familiarity with referents. We hope the present research can pave the path for studying emergent communication with the modality of sketches.

        ----

        ## [953] Repairing Neural Networks by Leaving the Right Past Behind

        **Authors**: *Ryutaro Tanno, Melanie F. Pradier, Aditya V. Nori, Yingzhen Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/552260cfb5e292e511eaa780806ac984-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/552260cfb5e292e511eaa780806ac984-Abstract-Conference.html)

        **Abstract**:

        Prediction failures of machine learning models often arise from deficiencies in training data, such as incorrect labels, outliers, and selection biases. However, such data points that are responsible for a given failure mode are generally not known a priori, let alone a mechanism for repairing the failure. This work draws on the Bayesian view of continual learning, and develops a generic framework for both, identifying training examples which have given rise to the target failure, and fixing the model through erasing information about them. This framework naturally allows leveraging recent advances in continual learning to this new problem of model repairment, while subsuming the existing works on influence functions and data deletion as specific instances. Experimentally, the proposed approach outperforms the baselines for both identification of detrimental training data and fixing model failures in a generalisable manner.

        ----

        ## [954] Selective compression learning of latent representations for variable-rate image compression

        **Authors**: *Jooyoung Lee, Seyoon Jeong, Munchurl Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5526c73e3ff4f2a34009e13d15f52fcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5526c73e3ff4f2a34009e13d15f52fcb-Abstract-Conference.html)

        **Abstract**:

        Recently, many neural network-based image compression methods have shown promising results superior to the existing tool-based conventional codecs. However, most of them are often trained as separate models for different target bit rates, thus increasing the model complexity. Therefore, several studies have been conducted for learned compression that supports variable rates with single models, but they require additional network modules, layers, or inputs that often lead to complexity overhead, or do not provide sufficient coding efficiency. In this paper, we firstly propose a selective compression method that partially encodes the latent representations in a fully generalized manner for deep learning-based variable-rate image compression. The proposed method adaptively determines essential representation elements for compression of different target quality levels. For this, we first generate a 3D importance map as the nature of input content to represent the underlying importance of the representation elements. The 3D importance map is then adjusted for different target quality levels using importance adjustment curves. The adjusted 3D importance map is finally converted into a 3D binary mask to determine the essential representation elements for compression. The proposed method can be easily integrated with the existing compression models with a negligible amount of overhead increase. Our method can also enable continuously variable-rate compression via simple interpolation of the importance adjustment curves among different quality levels. The extensive experimental results show that the proposed method can achieve comparable compression efficiency as those of the separately trained reference compression models and can reduce decoding time owing to the selective compression.

        ----

        ## [955] Multi-LexSum: Real-world Summaries of Civil Rights Lawsuits at Multiple Granularities

        **Authors**: *Zejiang Shen, Kyle Lo, Lauren Yu, Nathan Dahlberg, Margo Schlanger, Doug Downey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/552ef803bef9368c29e53c167de34b55-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/552ef803bef9368c29e53c167de34b55-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        With the advent of large language models, methods for abstractive summarization have made great strides, creating potential for use in applications to aid knowledge workers processing unwieldy document collections. One such setting is the Civil Rights Litigation Clearinghouse (CRLC, https://clearinghouse.net), which posts information about large-scale civil rights lawsuits, serving lawyers, scholars, and the general public. Today, summarization in the CRLC requires extensive training of lawyers and law students who spend hours per case understanding multiple relevant documents in order to produce high-quality summaries of key events and outcomes. Motivated by this ongoing real-world summarization effort, we introduce Multi-LexSum, a collection of 9,280 expert-authored summaries drawn from ongoing CRLC writing. Multi-LexSum presents a challenging multi-document summarization task given the length of the source documents, often exceeding two hundred pages per case. Furthermore, Multi-LexSum is distinct from other datasets in its multiple target summaries, each at a different granularity (ranging from one-sentence "extreme" summaries to multi-paragraph narrations of over five hundred words). We present extensive analysis demonstrating that despite the high-quality summaries in the training data (adhering to strict content and style guidelines), state-of-the-art summarization models perform poorly on this task. We release Multi-LexSum for further summarization research and to facilitate the development of applications to assist in the CRLC's mission at https://multilexsum.github.io.

        ----

        ## [956] Increasing Confidence in Adversarial Robustness Evaluations

        **Authors**: *Roland S. Zimmermann, Wieland Brendel, Florian Tramèr, Nicholas Carlini*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5545d9bcefb7d03d5ad39a905d14fbe3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5545d9bcefb7d03d5ad39a905d14fbe3-Abstract-Conference.html)

        **Abstract**:

        Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks and, thus, weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network. For eleven out of thirteen previously-published defenses, the original evaluation of the defense fails our test, while stronger attacks that break these defenses pass it. We hope that attack unit tests - such as ours - will be a major component in future robustness evaluations and increase confidence in an empirical field that is currently riddled with skepticism.

        ----

        ## [957] Local Bayesian optimization via maximizing probability of descent

        **Authors**: *Quan Nguyen, Kaiwen Wu, Jacob R. Gardner, Roman Garnett*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/555479a201da27c97aaeed842d16ca49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/555479a201da27c97aaeed842d16ca49-Abstract-Conference.html)

        **Abstract**:

        Local optimization presents a promising approach to expensive, high-dimensional black-box optimization by sidestepping the need to globally explore the search space. For objective functions whose gradient cannot be evaluated directly, Bayesian optimization offers one solution -- we construct a probabilistic model of the objective, design a policy to learn about the gradient at the current location, and use the resulting information to navigate the objective landscape. Previous work has realized this scheme by minimizing the variance in the estimate of the gradient, then moving in the direction of the expected gradient. In this paper, we re-examine and refine this approach. We demonstrate that, surprisingly, the expected value of the gradient is not always the direction maximizing the probability of descent, and in fact, these directions may be nearly orthogonal. This observation then inspires an elegant optimization scheme seeking to maximize the probability of descent while moving in the direction of most-probable descent. Experiments on both synthetic and real-world objectives show that our method outperforms previous realizations of this optimization scheme and is competitive against other, significantly more complicated baselines.

        ----

        ## [958] Staircase Attention for Recurrent Processing of Sequences

        **Authors**: *Da Ju, Stephen Roller, Sainbayar Sukhbaatar, Jason Weston*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5565ab682d6c7f8d9da34ba0919974b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5565ab682d6c7f8d9da34ba0919974b0-Abstract-Conference.html)

        **Abstract**:

        Attention mechanisms have become a standard tool for sequence modeling tasks, in particular by stacking self-attention layers over the entire input sequence as in the Transformer architecture. In this work we introduce a novel attention procedure called staircase attention that, unlike self-attention, operates across the sequence (in time) recurrently processing the input by adding another step of processing. A step in the staircase comprises of backward tokens (encoding the sequence so far seen) and forward tokens (ingesting a new part of the sequence). Thus our model can trade off performance and compute, by increasing the amount of recurrence through time and depth. Staircase attention is shown to be able to solve tasks that involve tracking that conventional Transformers cannot, due to this recurrence. Further, it is shown to provide improved modeling power for the same size model (number of parameters) compared to self-attentive Transformers on large language modeling and dialogue tasks, yielding significant perplexity gains.

        ----

        ## [959] Causal Inference with Non-IID Data using Linear Graphical Models

        **Authors**: *Chi Zhang, Karthika Mohan, Judea Pearl*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5573c63e8a89e32086e5c71cf0cc8fe4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5573c63e8a89e32086e5c71cf0cc8fe4-Abstract-Conference.html)

        **Abstract**:

        Traditional causal inference techniques assume data are independent and identically distributed (IID) and thus  ignores interactions among units. However, a unit’s treatment may affect another unit's outcome (interference), a unit’s treatment may be correlated with another unit’s outcome, or a unit’s treatment and outcome may be spuriously correlated through another unit. To capture such nuances, we model the data generating process using causal graphs and conduct a systematic analysis of the bias caused by different types of interactions when computing causal effects. We derive theorems to detect and quantify the interaction bias, and derive conditions under which it is safe to ignore interactions. Put differently, we present conditions under which causal effects can be computed with negligible bias by assuming that samples are IID. Furthermore, we develop a method to eliminate bias in cases where blindly assuming IID is expected to yield a significantly biased estimate. Finally, we test the coverage and performance of our methods through simulations.

        ----

        ## [960] Data-Driven Offline Decision-Making via Invariant Representation Learning

        **Authors**: *Han Qi, Yi Su, Aviral Kumar, Sergey Levine*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/559726fdfb19005e368be4ce3d40e3e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/559726fdfb19005e368be4ce3d40e3e5-Abstract-Conference.html)

        **Abstract**:

        The goal in offline data-driven decision-making is synthesize decisions that optimize a black-box utility function, using a previously-collected static dataset, with no active interaction. These problems appear in many forms: offline reinforcement learning (RL), where we must produce actions that optimize the long-term reward, bandits from logged data, where the goal is to determine the correct arm, and offline model-based optimization (MBO) problems, where we must find the optimal design provided access to only a static dataset. A key challenge in all these settings is distributional shift: when we optimize with respect to the input into a model trained from offline data, it is easy to produce an out-of-distribution (OOD) input that appears erroneously good. In contrast to prior approaches that utilize pessimism or conservatism to tackle this problem, in this paper, we formulate offline data-driven decision-making as domain adaptation, where the goal is to make accurate predictions for the value of optimized decisions (“target domain”), when training only on the dataset (“source domain”). This perspective leads to invariant objective models (IOM), our approach for addressing distributional shift by enforcing invariance between the learned representations of the training dataset and optimized decisions. In IOM, if the optimized decisions are too different from the training dataset, the representation will be forced to lose much of the information that distinguishes good designs from bad ones, making all choices seem mediocre. Critically, when the optimizer is aware of this representational tradeoff, it should choose not to stray too far from the training distribution, leading to a natural trade-off between distributional shift and learning performance.

        ----

        ## [961] Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection

        **Authors**: *Yiming Li, Yang Bai, Yong Jiang, Yong Yang, Shu-Tao Xia, Bo Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/55bfedfd31489e5ae83c9ce8eec7b0e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/55bfedfd31489e5ae83c9ce8eec7b0e1-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) have demonstrated their superiority in practice. Arguably, the rapid development of DNNs is largely benefited from high-quality (open-sourced) datasets, based on which researchers and developers can easily evaluate and improve their learning methods. Since the data collection is usually time-consuming or even expensive, how to protect their copyrights is of great significance and worth further exploration. In this paper, we revisit dataset ownership verification. We find that existing verification methods introduced new security risks in DNNs trained on the protected dataset, due to the targeted nature of poison-only backdoor watermarks. To alleviate this problem, in this work, we explore the untargeted backdoor watermarking scheme, where the abnormal model behaviors are not deterministic. Specifically, we introduce two dispersibilities and prove their correlation, based on which we design the untargeted backdoor watermark under both poisoned-label and clean-label settings. We also discuss how to use the proposed untargeted backdoor watermark for dataset ownership verification. Experiments on benchmark datasets verify the effectiveness of our methods and their resistance to existing backdoor defenses.

        ----

        ## [962] Learning Symmetric Rules with SATNet

        **Authors**: *Sangho Lim, Eun-Gyeol Oh, Hongseok Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5642b9811a9ac5281be1cc84c275f251-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5642b9811a9ac5281be1cc84c275f251-Abstract-Conference.html)

        **Abstract**:

        SATNet is a differentiable constraint solver with a custom backpropagation algorithm, which can be used as a layer in a deep-learning system. It is a promising proposal for bridging deep learning and logical reasoning. In fact, SATNet has been successfully applied to learn, among others, the rules of a complex logical puzzle, such as Sudoku, just from input and output pairs where inputs are given as images. In this paper, we show how to improve the learning of SATNet by exploiting symmetries in the target rules of a given but unknown logical puzzle or more generally a logical formula. We present SymSATNet, a variant of SATNet that translates the given symmetries of the target rules to a condition on the parameters of SATNet and requires that the parameters should have a particular parametric form that guarantees the condition. The requirement dramatically reduces the number of parameters to learn for the rules with enough symmetries, and makes the parameter learning of SymSATNet much easier than that of SATNet. We also describe a technique for automatically discovering symmetries of the target rules from examples. Our experiments with Sudoku and Rubik's cube show the substantial improvement of SymSATNet over the baseline SATNet.

        ----

        ## [963] The Privacy Onion Effect: Memorization is Relative

        **Authors**: *Nicholas Carlini, Matthew Jagielski, Chiyuan Zhang, Nicolas Papernot, Andreas Terzis, Florian Tramèr*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/564b5f8289ba846ebc498417e834c253-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/564b5f8289ba846ebc498417e834c253-Abstract-Conference.html)

        **Abstract**:

        Machine learning models trained on private datasets have been shown to leak their private data. Recent work has found that the average data point is rarely leaked---it is often the outlier samples that are subject to memorization and, consequently, leakage. We demonstrate and analyze an Onion Effect of memorization: removing the "layer" of outlier points that are most vulnerable to a privacy attack exposes a new layer of previously-safe points to the same attack. We perform several experiments that are consistent with this hypothesis. For example, we show that for membership inference attacks, when the layer of easiest-to-attack examples is removed, another layer below becomes easy-to-attack. The existence of this effect has various consequences. For example, it suggests that proposals to defend against memorization without training with rigorous privacy guarantees are unlikely to be effective. Further, it suggests that privacy-enhancing technologies such as machine unlearning could actually harm the privacy of other users.

        ----

        ## [964] Langevin Autoencoders for Learning Deep Latent Variable Models

        **Authors**: *Shohei Taniguchi, Yusuke Iwasawa, Wataru Kumagai, Yutaka Matsuo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html)

        **Abstract**:

        Markov chain Monte Carlo (MCMC), such as Langevin dynamics, is valid for approximating intractable distributions. However, its usage is limited in the context of deep latent variable models owing to costly datapoint-wise sampling iterations and slow convergence. This paper proposes the amortized Langevin dynamics (ALD), wherein datapoint-wise MCMC iterations are entirely replaced with updates of an encoder that maps observations into latent variables. This amortization enables efficient posterior sampling without datapoint-wise iterations. Despite its efficiency, we prove that ALD is valid as an MCMC algorithm, whose Markov chain has the target posterior as a stationary distribution under mild assumptions. Based on the ALD, we also present a new deep latent variable model named the Langevin autoencoder (LAE). Interestingly, the LAE can be implemented by slightly modifying the traditional autoencoder. Using multiple synthetic datasets, we first validate that ALD can properly obtain samples from target posteriors. We also evaluate the LAE on the image generation task, and show that our LAE can outperform existing methods based on variational inference, such as the variational autoencoder, and other MCMC-based methods in terms of the test likelihood.

        ----

        ## [965] Envy-free Policy Teaching to Multiple Agents

        **Authors**: *Jiarui Gan, Rupak Majumdar, Adish Singla, Goran Radanovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5692c7dbc4abcaa50f9ce609819212e5-Abstract-Conference.html)

        **Abstract**:

        We study envy-free policy teaching. A number of agents independently explore a common Markov decision process (MDP), but each with their own reward function and discounting rate. A teacher wants to teach a target policy to this diverse group of agents, by means of modifying the agents' reward functions: providing additional bonuses to certain actions, or penalizing them. When personalized reward modification programs are used, an important question is how to design the programs so that the agents think they are treated fairly. We adopt the notion of envy-freeness (EF) from the literature on fair division to formalize this problem and investigate several fundamental questions about the existence of EF solutions in our setting, the computation of cost-minimizing solutions, as well as the price of fairness (PoF), which measures the increase of cost due to the consideration of fairness. We show that 1) an EF solution may not exist if penalties are not allowed in the modifications, but otherwise always exists. 2) Computing a cost-minimizing EF solution can be formulated as convex optimization and hence solved efficiently. 3) The PoF increases but at most quadratically with the geometric sum of the discount factor, and at most linearly with the size of the MDP and the number of agents involved; we present tight asymptotic bounds on the PoF. These results indicate that fairness can be incorporated in multi-agent teaching without significant computational or PoF burdens.

        ----

        ## [966] Provably Efficient Model-Free Constrained RL with Linear Function Approximation

        **Authors**: *Arnob Ghosh, Xingyu Zhou, Ness B. Shroff*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/56b8f22d895c45f60eaac9580152afd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/56b8f22d895c45f60eaac9580152afd9-Abstract-Conference.html)

        **Abstract**:

        We study the constrained reinforcement learning problem, in which an agent aims to maximize the expected cumulative reward subject to a constraint on the expected total value of a utility function.  In contrast to existing model-based approaches or model-free methods accompanied with a `simulatorâ€™, we aim to develop the first \emph{model-free}, \emph{simulator-free} algorithm that achieves a sublinear regret and a sublinear constraint violation even in \emph{large-scale} systems. To this end, we consider the episodic constrained Markov decision processes with linear function approximation, where the transition dynamics and the reward function can be represented as a linear function of some known feature mapping. We show that $\tilde{\mathcal{O}}(\sqrt{d^3H^3T})$ regret and  $\tilde{\mathcal{O}}(\sqrt{d^3H^3T})$ constraint violation bounds can be achieved, where $d$ is the dimension of the feature mapping, $H$ is the length of the episode, and $T$ is the total number of steps. Our bounds are attained without explicitly estimating the unknown transition model or requiring a simulator, and they depend on the state space only through the dimension of the feature mapping. Hence our bounds hold even when the number of states goes to infinity. Our main results are achieved via novel adaptations of the standard LSVI-UCB algorithms. In particular, we first introduce primal-dual optimization into the LSVI-UCB algorithm to balance between regret and constraint violation. More importantly, we replace the standard greedy selection with respect to the state-action function with a soft-max policy. This turns out to be key in establishing uniform concentration (a critical step for provably efficient model-free exploration) for the constrained case via its approximation-smoothness trade-off. Finally, we also show that one can achieve an even zero constraint violation for large enough $T$ by trading the regret a little bit but still maintaining the same order with respect to $T$.

        ----

        ## [967] Towards Optimal Communication Complexity in Distributed Non-Convex Optimization

        **Authors**: *Kumar Kshitij Patel, Lingxiao Wang, Blake E. Woodworth, Brian Bullins, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/56bd21259e28ebdc4d7e1503733bf421-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/56bd21259e28ebdc4d7e1503733bf421-Abstract-Conference.html)

        **Abstract**:

        We study the problem of distributed stochastic non-convex optimization with intermittent communication. We consider the full participation setting where $M$ machines work in parallel over $R$ communication rounds and the partial participation setting where $M$ machines are sampled independently every round from some meta-distribution over machines. We propose and analyze a new algorithm that improves existing methods by requiring fewer and lighter variance reduction operations. We also present lower bounds, showing our algorithm is either $\textit{optimal}$ or $\textit{almost optimal}$ in most settings. Numerical experiments demonstrate the superior performance of our algorithm.

        ----

        ## [968] Eliciting Thinking Hierarchy without a Prior

        **Authors**: *Yuqing Kong, Yunqi Li, Yubo Zhang, Zhihuan Huang, Jinzhao Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/56d7585405a534b3af91905650ce7f9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/56d7585405a534b3af91905650ce7f9e-Abstract-Conference.html)

        **Abstract**:

        When we use the wisdom of the crowds, we usually rank the answers according to their popularity, especially when we cannot verify the answers. However, this can be very dangerous when the majority make systematic mistakes. A fundamental question arises: can we build a hierarchy among the answers without any prior where the higher-ranking answers, which may not be supported by the majority, are from more sophisticated people? To address the question, we propose 1) a novel model to describe people's thinking hierarchy; 2) two algorithms to learn the thinking hierarchy without any prior; 3) a novel open-response based crowdsourcing approach based on the above theoretic framework. In addition to theoretic justifications, we conduct four empirical crowdsourcing studies and show that a) the accuracy of the top-ranking answers learned by our approach is much higher than that of plurality voting (In one question, the plurality answer is supported by 74 respondents but the correct answer is only supported by 3 respondents. Our approach ranks the correct answer the highest without any prior); b) our model has a high goodness-of-fit, especially for the questions where our top-ranking answer is correct. To the best of our knowledge, we are the first to propose a thinking hierarchy model with empirical validations in the general problem-solving scenarios; and the first to propose a practical open-response-based crowdsourcing approach that beats plurality voting without any prior.

        ----

        ## [969] Fair and Efficient Allocations Without Obvious Manipulations

        **Authors**: *Alexandros Psomas, Paritosh Verma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57250222014c35949476f3f272c322d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57250222014c35949476f3f272c322d2-Abstract-Conference.html)

        **Abstract**:

        We consider the fundamental problem of allocating a set of indivisible goods among strategic agents with additive valuation functions. It is well known that, in the absence of monetary transfers, Pareto efficient and truthful rules are dictatorial, while there is no deterministic truthful mechanism that allocates all items and achieves envy-freeness up to one item (EF1), even for the case of two agents. In this paper, we investigate the interplay of fairness and efficiency under a relaxation of truthfulness called non-obvious manipulability (NOM), recently proposed by~\citep{troyan2020obvious}. We show that this relaxation allows us to bypass the aforementioned negative results in a very strong sense. Specifically, we prove that there are deterministic and EF1 algorithms that are not obviously manipulable, and the algorithm that maximizes utilitarian social welfare (the sum of agents' utilities), which is Pareto efficient but not dictatorial, is not obviously manipulable for $n \geq 3$ agents (but obviously manipulable for $n=2$ agents). At the same time, maximizing the egalitarian social welfare (the minimum of agents' utilities) or the Nash social welfare (the product of agents' utilities) is obviously manipulable for any number of agents and items. Our main result is an approximation preserving black-box reduction from the problem of designing EF1 and NOM mechanisms to the problem of designing EF1 algorithms. En route, we prove an interesting structural result about EF1 allocations, as well as new ``best-of-both-worlds'' results (for the problem without incentives), that might be of independent interest.

        ----

        ## [970] Bridge the Gap Between Architecture Spaces via A Cross-Domain Predictor

        **Authors**: *Yuqiao Liu, Yehui Tang, Zeqiong Lv, Yunhe Wang, Yanan Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/572aaddf9ff774f7c1cf3d0c81c7185b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/572aaddf9ff774f7c1cf3d0c81c7185b-Abstract-Conference.html)

        **Abstract**:

        Neural Architecture Search (NAS) can automatically design promising neural architectures without artificial experience. Though it achieves great success, prohibitively high search cost is required to find a high-performance architecture, which blocks its practical implementation. Neural predictor can directly evaluate the performance of neural networks based on their architectures and thereby save much budget. However, existing neural predictors require substantial annotated architectures trained from scratch, which still consume many computational resources. To solve this issue, we propose a Cross-Domain Predictor (CDP), which is trained based on the existing NAS benchmark datasets (e.g., NAS-Bench-101), but can be used to find high-performance architectures in large-scale search spaces. Particularly, we propose a progressive subspace adaptation strategy to address the domain discrepancy between the source architecture space and the target space. Considering the large difference between two architecture spaces, an assistant space is developed to smooth the transfer process. Compared with existing NAS methods, the proposed CDP is much more efficient. For example, CDP only requires the search cost of 0.1 GPU Days to find architectures with 76.9% top-1 accuracy on ImageNet and 97.51% on CIFAR-10.

        ----

        ## [971] Shield Decentralization for Safe Multi-Agent Reinforcement Learning

        **Authors**: *Daniel Melcer, Christopher Amato, Stavros Tripakis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57444e14ecd9e2c8f603b4f012ce3811-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57444e14ecd9e2c8f603b4f012ce3811-Abstract-Conference.html)

        **Abstract**:

        Learning safe solutions is an important but challenging problem in multi-agent reinforcement learning (MARL). Shielded reinforcement learning is one approach for preventing agents from choosing unsafe actions. Current shielded reinforcement learning methods for MARL make strong assumptions about communication and full observability. In this work, we extend the formalization of the shielded reinforcement learning problem to a decentralized multi-agent setting. We then present an algorithm for decomposition of a centralized shield, allowing shields to be used in such decentralized, communication-free environments. Our results show that agents equipped with decentralized shields perform comparably to agents with centralized shields in several tasks, allowing shielding to be used in environments with decentralized training and execution for the first time.

        ----

        ## [972] Domain Generalization without Excess Empirical Risk

        **Authors**: *Ozan Sener, Vladlen Koltun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57568e093cbe0a222de0334b36e83cf5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57568e093cbe0a222de0334b36e83cf5-Abstract-Conference.html)

        **Abstract**:

        Given data from diverse sets of distinct distributions, domain generalization aims to learn models that generalize to unseen distributions. A common approach is designing a data-driven surrogate penalty to capture generalization and minimize the empirical risk jointly with the penalty. We argue that a significant failure mode of this recipe is an excess risk due to an erroneous penalty or hardness in joint optimization. We present an approach that eliminates this problem. Instead of jointly minimizing empirical risk with the penalty, we minimize the penalty under the constraint of optimality of the empirical risk. This change guarantees that the domain generalization penalty cannot impair optimization of the empirical risk, \ie, in-distribution performance. To solve the proposed optimization problem, we demonstrate an exciting connection to rate-distortion theory and utilize its tools to design an efficient method. Our approach can be applied to any penalty-based domain generalization method, and we demonstrate its effectiveness by applying it to three examplar methods from the literature, showing significant improvements.

        ----

        ## [973] Star Temporal Classification: Sequence Modeling with Partially Labeled Data

        **Authors**: *Vineel Pratap, Awni Hannun, Gabriel Synnaeve, Ronan Collobert*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html)

        **Abstract**:

        We develop an algorithm which can learn from partially labeled and unsegmented sequential data. Most sequential loss functions, such as Connectionist Temporal Classification (CTC), break down when many labels are missing. We address this problem with Star Temporal Classification (STC) which uses a special star token to allow alignments which include all possible tokens whenever a token could be missing. We express STC as the composition of weighted finite-state transducers (WFSTs) and use GTN (a framework for automatic differentiation with WFSTs) to compute gradients. We perform extensive experiments on automatic speech recognition. These experiments show that STC can close the performance gap with supervised baseline to about 1% WER when up to 70% of the labels are missing. We also perform experiments in handwriting recognition to show that our method easily applies to other temporal classification tasks.

        ----

        ## [974] Signal Processing for Implicit Neural Representations

        **Authors**: *Dejia Xu, Peihao Wang, Yifan Jiang, Zhiwen Fan, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html)

        **Abstract**:

        Implicit Neural Representations (INRs) encoding continuous multi-media data via multi-layer perceptrons has shown undebatable promise in various computer vision tasks. Despite many successful applications, editing and processing an INR remains intractable as signals are represented by latent parameters of a neural network. Existing works manipulate such continuous representations via processing on their discretized instance, which breaks down the compactness and continuous nature of INR. In this work, we present a pilot study on the question: how to directly modify an INR without explicit decoding? We answer this question by proposing an implicit neural signal processing network, dubbed INSP-Net, via differential operators on INR. Our key insight is that spatial gradients of neural networks can be computed analytically and are invariant to translation, while mathematically we show that any continuous convolution filter can be uniformly approximated by a linear combination of high-order differential operators. With these two knobs, INSP-Net instantiates the signal processing operator as a weighted composition of computational graphs corresponding to the high-order derivatives of INRs, where the weighting parameters can be data-driven learned. Based on our proposed INSP-Net, we further build the first Convolutional Neural Network (CNN) that implicitly runs on INRs, named INSP-ConvNet. Our experiments validate the expressiveness of INSP-Net and INSP-ConvNet in fitting low-level image and geometry processing kernels (e.g. blurring, deblurring, denoising, inpainting, and smoothening) as well as for high-level tasks on implicit fields such as image classification.

        ----

        ## [975] Fault-Aware Neural Code Rankers

        **Authors**: *Jeevana Priya Inala, Chenglong Wang, Mei Yang, Andrés Codas, Mark Encarnación, Shuvendu K. Lahiri, Madanlal Musuvathi, Jianfeng Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5762c579d09811b7639be2389b3d07be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5762c579d09811b7639be2389b3d07be-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) have demonstrated an impressive ability to generate code for various programming tasks. In many instances, LLMs can generate a correct program for a task when given numerous trials. Consequently, a recent trend is to do large scale sampling of programs using a model and then filtering/ranking the programs based on the program execution on a small number of known unit tests to select one candidate solution. However, these approaches assume that the unit tests are given and assume the ability to safely execute the generated programs (which can do arbitrary dangerous operations such as file manipulations). Both of the above assumptions are impractical in real-world software development. In this paper, we propose CodeRanker, a neural ranker that can predict the correctness of a sampled program without executing it. Our CodeRanker is fault-aware i.e., it is trained to predict different kinds of execution information such as predicting the exact compile/runtime error type (e.g., an IndexError or a TypeError). We show that CodeRanker can significantly increase the pass@1 accuracy of various code generation models (including Codex, GPT-Neo, GPT-J) on APPS, HumanEval and MBPP datasets.

        ----

        ## [976] Prompt Certified Machine Unlearning with Randomized Gradient Smoothing and Quantization

        **Authors**: *Zijie Zhang, Yang Zhou, Xin Zhao, Tianshi Che, Lingjuan Lyu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5771d9f214b75be6ff20f63bba315644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5771d9f214b75be6ff20f63bba315644-Abstract-Conference.html)

        **Abstract**:

        The right to be forgotten calls for efficient machine unlearning techniques that make trained machine learning models forget a cohort of data. The combination of training and unlearning operations in traditional machine unlearning methods often leads to the expensive computational cost on large-scale data. This paper presents a prompt certified machine unlearning algorithm, PCMU, which executes one-time operation of simultaneous training and unlearning in advance for a series of machine unlearning requests, without the knowledge of the removed/forgotten data. First, we establish a connection between randomized smoothing for certified robustness on classification and randomized smoothing for certified machine unlearning on gradient quantization. Second, we propose a prompt certified machine unlearning model based on randomized data smoothing and gradient quantization. We theoretically derive the certified radius R regarding the data change before and after data removals and the certified budget of data removals about R. Last but not least, we present another practical framework of randomized gradient smoothing and quantization, due to the dilemma of producing high confidence certificates in the first framework. We theoretically demonstrate the certified radius R' regarding the gradient change, the correlation between two types of certified radii, and the certified budget of data removals about R'.

        ----

        ## [977] What Makes a "Good" Data Augmentation in Knowledge Distillation - A Statistical Perspective

        **Authors**: *Huan Wang, Suhas Lohit, Michael N. Jones, Yun Fu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57b53238ff22bc0dc62de08f53eb5de2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57b53238ff22bc0dc62de08f53eb5de2-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation (KD) is a general neural network training approach that uses a teacher model to guide the student model. Existing works mainly study KD from the network output side (e.g., trying to design a better KD loss function), while few have attempted to understand it from the input side. Especially, its interplay with data augmentation (DA) has not been well understood. In this paper, we ask: Why do some DA schemes (e.g., CutMix) inherently perform much better than others in KD? What makes a "good" DA in KD? Our investigation from a statistical perspective suggests that a good DA scheme should reduce the covariance of the teacher-student cross-entropy. A practical metric, the stddev of teacherâ€™s mean probability (T. stddev), is further presented and well justified empirically. Besides the theoretical understanding, we also introduce a new entropy-based data-mixing DA scheme, CutMixPick, to further enhance CutMix. Extensive empirical studies support our claims and demonstrate how we can harvest considerable performance gains simply by using a better DA scheme in knowledge distillation. Code: https://github.com/MingSun-Tse/Good-DA-in-KD.

        ----

        ## [978] Supervising the Multi-Fidelity Race of Hyperparameter Configurations

        **Authors**: *Martin Wistuba, Arlind Kadra, Josif Grabocka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57b694fef23ae7b9308eb4d46342595d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57b694fef23ae7b9308eb4d46342595d-Abstract-Conference.html)

        **Abstract**:

        Multi-fidelity (gray-box) hyperparameter optimization techniques (HPO) have recently emerged as a promising direction for tuning Deep Learning methods. However, existing methods suffer from a sub-optimal allocation of the HPO budget to the hyperparameter configurations. In this work, we introduce DyHPO, a Bayesian Optimization method that learns to decide which hyperparameter configuration to train further in a dynamic race among all feasible configurations. We propose a new deep kernel for Gaussian Processes that embeds the learning curve dynamics, and an acquisition function that incorporates multi-budget information. We demonstrate the significant superiority of DyHPO against state-of-the-art hyperparameter optimization methods through large-scale experiments comprising 50 datasets (Tabular, Image, NLP) and diverse architectures (MLP, CNN/NAS, RNN).

        ----

        ## [979] Streaming Radiance Fields for 3D Video Synthesis

        **Authors**: *Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Ping Tan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57c2cc952f388f6185db98f441351c96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57c2cc952f388f6185db98f441351c96-Abstract-Conference.html)

        **Abstract**:

        We present an explicit-grid based method for efficiently reconstructing streaming radiance fields for novel view synthesis of real world dynamic scenes. Instead of training a single model that combines all the frames, we formulate the dynamic modeling problem with an incremental learning paradigm in which per-frame model difference is trained to complement the adaption of a base model on the current frame. By exploiting the simple yet effective tuning strategy with narrow bands, the proposed method realizes a feasible framework for handling video sequences on-the-fly with high training efficiency. The storage overhead induced by using explicit grid representations can be significantly reduced through the use of model difference based compression. We also introduce an efficient strategy to further accelerate model optimization for each frame. Experiments on challenging video sequences demonstrate that our approach is capable of achieving a training speed of 15 seconds per-frame with competitive rendering quality, which attains $1000 \times$ speedup over the state-of-the-art implicit methods.

        ----

        ## [980] Byzantine-tolerant federated Gaussian process regression for streaming data

        **Authors**: *Xu Zhang, Zhenyuan Yuan, Minghui Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57c56985d9afe89bf78a8264c91071aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57c56985d9afe89bf78a8264c91071aa-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider Byzantine-tolerant federated learning for streaming data using Gaussian process regression (GPR). In particular, a cloud and a group of agents aim to collaboratively learn a latent function where some agents are subject to Byzantine attacks. We develop a Byzantine-tolerant federated GPR algorithm, which includes three modules: agent-based local GPR, cloud-based aggregated GPR and agent-based fused GPR. We derive the upper bounds on prediction error between the mean from the cloud-based aggregated GPR and the target function provided that Byzantine agents are less than one quarter of all the agents. We also characterize the lower and upper bounds of the predictive variance. Experiments on a synthetic dataset and two real-world datasets are conducted to evaluate the proposed algorithm.

        ----

        ## [981] Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence

        **Authors**: *Sunghwan Hong, Jisu Nam, Seokju Cho, Susung Hong, Sangryul Jeon, Dongbo Min, Seungryong Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57c5a7c83b056d74bc97b7db36bd3649-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57c5a7c83b056d74bc97b7db36bd3649-Abstract-Conference.html)

        **Abstract**:

        Existing pipelines of semantic correspondence commonly include extracting high-level semantic features for the invariance against intra-class variations and background clutters. This architecture, however, inevitably results in a low-resolution matching field that additionally requires an ad-hoc interpolation process as a post-processing for converting it into a high-resolution one, certainly limiting the overall performance of matching results. To overcome this, inspired by recent success of implicit neural representation, we present a novel method for semantic correspondence, called Neural Matching Field (NeMF). However, complicacy and high-dimensionality of a 4D matching field are the major hindrances, which we propose a cost embedding network to process a coarse cost volume to use as a guidance for establishing high-precision matching field through the following fully-connected network. Nevertheless, learning a high-dimensional matching field remains challenging mainly due to computational complexity, since a na\"ive exhaustive inference would require querying from all pixels in the 4D space to infer pixel-wise correspondences. To overcome this, we propose adequate training and inference procedures, which in the training phase, we randomly sample matching candidates and in the inference phase, we iteratively performs PatchMatch-based inference and coordinate optimization at test time. With these combined, competitive results are attained on several standard benchmarks for semantic correspondence. Code and pre-trained weights are available at~\url{https://ku-cvlab.github.io/NeMF/}.

        ----

        ## [982] Consistency of Constrained Spectral Clustering under Graph Induced Fair Planted Partitions

        **Authors**: *Shubham Gupta, Ambedkar Dukkipati*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57d7e7e1593ad1ab6818c258fa5654ce-Abstract-Conference.html)

        **Abstract**:

        Spectral clustering is popular among practitioners and theoreticians alike. While performance guarantees for spectral clustering are well understood, recent studies have focused on enforcing "fairness" in clusters, requiring them to be "balanced" with respect to a categorical sensitive node attribute (e.g. the race distribution in clusters must match the race distribution in the population). In this paper, we consider a setting where sensitive attributes indirectly manifest in an auxiliary representation graph rather than being directly observed. This graph specifies node pairs that can represent each other with respect to sensitive attributes and is observed in addition to the usual similarity graph. Our goal is to find clusters in the similarity graph while respecting a new individual-level fairness constraint encoded by the representation graph. We develop variants of unnormalized and normalized spectral clustering for this task and analyze their performance under a fair planted partition model induced by the representation graph. This model uses both the cluster membership of the nodes and the structure of the representation graph to generate random similarity graphs. To the best of our knowledge, these are the first consistency results for constrained spectral clustering under an individual-level fairness constraint. Numerical results corroborate our theoretical findings.

        ----

        ## [983] Local Spatiotemporal Representation Learning for Longitudinally-consistent Neuroimage Analysis

        **Authors**: *Mengwei Ren, Neel Dey, Martin Styner, Kelly N. Botteron, Guido Gerig*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57da66da25d0ce77e0129b246f358851-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57da66da25d0ce77e0129b246f358851-Abstract-Conference.html)

        **Abstract**:

        Recent self-supervised advances in medical computer vision exploit the global and local anatomical self-similarity for pretraining prior to downstream tasks such as segmentation. However, current methods assume i.i.d. image acquisition, which is invalid in clinical study designs where follow-up longitudinal scans track subject-specific temporal changes. Further, existing self-supervised methods for medically-relevant image-to-image architectures exploit only spatial or temporal self-similarity and do so via a loss applied only at a single image-scale, with naive multi-scale spatiotemporal extensions collapsing to degenerate solutions. To these ends, this paper makes two contributions: (1) It presents a local and multi-scale spatiotemporal representation learning method for image-to-image architectures trained on longitudinal images. It exploits the spatiotemporal self-similarity of learned multi-scale intra-subject image features for pretraining and develops several feature-wise regularizations that avoid degenerate representations; (2) During finetuning, it proposes a surprisingly simple self-supervised segmentation consistency regularization to exploit intra-subject correlation. Benchmarked across various segmentation tasks, the proposed framework outperforms both well-tuned randomly-initialized baselines and current self-supervised techniques designed for both i.i.d. and longitudinal datasets. These improvements are demonstrated across both longitudinal neurodegenerative adult MRI and developing infant brain MRI and yield both higher performance and longitudinal consistency.

        ----

        ## [984] Overparameterization from Computational Constraints

        **Authors**: *Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57e48ac3aa4d107979bf5c6ebc9fe99d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57e48ac3aa4d107979bf5c6ebc9fe99d-Abstract-Conference.html)

        **Abstract**:

        Overparameterized models with millions of parameters have been hugely successful. In this work, we ask:  can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.

        ----

        ## [985] A Unifying Framework of Off-Policy General Value Function Evaluation

        **Authors**: *Tengyu Xu, Zhuoran Yang, Zhaoran Wang, Yingbin Liang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57ef0373c890b30407eadfe6e06c8c84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57ef0373c890b30407eadfe6e06c8c84-Abstract-Conference.html)

        **Abstract**:

        General Value Function (GVF) is a powerful tool to represent both the {\em predictive} and {\em retrospective} knowledge in reinforcement learning (RL). In practice, often multiple interrelated GVFs need to be evaluated jointly with pre-collected off-policy samples. In the literature, the gradient temporal difference (GTD) learning method has been adopted to evaluate GVFs in the off-policy setting, but such an approach may suffer from a large estimation error even if the function approximation class is sufficiently expressive. Moreover, none of the previous work have formally established the convergence guarantee to the ground truth GVFs under the function approximation settings. In this paper, we address both issues through the lens of a class of GVFs with causal filtering, which cover a wide range of RL applications such as reward variance, value gradient, cost in anomaly detection, stationary distribution gradient, etc. We propose a new algorithm called GenTD for off-policy GVFs evaluation and show that GenTD learns multiple interrelated multi-dimensional GVFs as efficiently as a single canonical scalar value function. We further show that unlike GTD, the learned GVFs by GenTD are guaranteed to converge to the ground truth GVFs as long as the function approximation power is sufficiently large. To our best knowledge, GenTD is the first off-policy GVF evaluation algorithm that has global optimality guarantee.

        ----

        ## [986] Anchor-Changing Regularized Natural Policy Gradient for Multi-Objective Reinforcement Learning

        **Authors**: *Ruida Zhou, Tao Liu, Dileep Kalathil, P. R. Kumar, Chao Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/57fbe68cb318cad62c4ae4c91c83cba3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/57fbe68cb318cad62c4ae4c91c83cba3-Abstract-Conference.html)

        **Abstract**:

        We study policy optimization for Markov decision processes (MDPs) with multiple reward value functions, which are to be jointly optimized according to given criteria such as proportional fairness (smooth concave scalarization), hard constraints (constrained MDP), and max-min trade-off. We propose an Anchor-changing Regularized Natural Policy Gradient (ARNPG) framework, which can systematically incorporate ideas from well-performing first-order methods into the design of policy optimization algorithms for multi-objective MDP problems. Theoretically, the designed algorithms based on the ARNPG framework achieve $\tilde{O}(1/T)$ global convergence with exact gradients. Empirically, the ARNPG-guided algorithms also demonstrate superior performance compared to some existing policy gradient-based approaches in both exact gradients and sample-based scenarios.

        ----

        ## [987] Do Current Multi-Task Optimization Methods in Deep Learning Even Help?

        **Authors**: *Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, Orhan Firat*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract-Conference.html)

        **Abstract**:

        Recent research has proposed a series of specialized optimization algorithms for deep multi-task models. It is often claimed that these multi-task optimization (MTO) methods yield solutions that are superior to the ones found by simply optimizing a weighted average of the task losses. In this paper, we perform large-scale experiments on a variety of language and vision tasks to examine the empirical validity of these claims. We show that, despite the added design and computational complexity of these algorithms, MTO methods do not yield any performance improvements beyond what is achievable via traditional optimization approaches. We highlight alternative strategies that consistently yield improvements to the performance profile and point out common training pitfalls that might cause suboptimal results. Finally, we outline challenges in reliably evaluating the performance of MTO algorithms and discuss potential solutions.

        ----

        ## [988] TAP-Vid: A Benchmark for Tracking Any Point in a Video

        **Authors**: *Carl Doersch, Ankush Gupta, Larisa Markeeva, Adrià Recasens, Lucas Smaira, Yusuf Aytar, João Carreira, Andrew Zisserman, Yi Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/58168e8a92994655d6da3939e7cc0918-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Generic motion understanding from video involves not only tracking objects, but also perceiving how their surfaces deform and move. This information is useful to make inferences about 3D shape, physical properties and object interactions. While the problem of tracking arbitrary physical points on surfaces over longer video clips has received some attention, no dataset or benchmark for evaluation existed, until now.  In this paper, we first formalize the problem, naming it tracking any point (TAP). We introduce a companion benchmark,TAP-Vid, which is composed of both real-world videos with accurate human annotations of point tracks, and synthetic videos with perfect ground-truth point tracks. Central to the construction of our benchmark is a novel semi-automatic crowdsourced pipeline which uses optical flow estimates to compensate for easier, short-term motion like camera shake, allowing annotators to focus on harder sections of the video. We validate our pipeline on synthetic data and propose a simple end-to-end point tracking model, TAP-Net, showing that it outperforms all prior methods on our benchmark when trained on synthetic data.

        ----

        ## [989] Alignment-guided Temporal Attention for Video Action Recognition

        **Authors**: *Yizhou Zhao, Zhenyang Li, Xun Guo, Yan Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5820ad65b1c27411417ae8b59433e580-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5820ad65b1c27411417ae8b59433e580-Abstract-Conference.html)

        **Abstract**:

        Temporal modeling is crucial for various video learning tasks. Most recent approaches employ either factorized (2D+1D) or joint (3D) spatial-temporal operations to extract temporal contexts from the input frames. While the former is more efficient in computation, the latter often obtains better performance. In this paper, we attribute this to a dilemma between the sufficiency and the efficiency of interactions among various positions in different frames. These interactions affect the extraction of task-relevant information shared among frames. To resolve this issue, we prove that frame-by-frame alignments have the potential to increase the mutual information between frame representations, thereby including more task-relevant information to boost effectiveness. Then we propose Alignment-guided Temporal Attention (ATA) to extend 1-dimensional temporal attention with parameter-free patch-level alignments between neighboring frames. It can act as a general plug-in for image backbones to conduct the action recognition task without any model-specific design. Extensive experiments on multiple benchmarks demonstrate the superiority and generality of our module.

        ----

        ## [990] Meta Reinforcement Learning with Finite Training Tasks - a Density Estimation Approach

        **Authors**: *Zohar Rimon, Aviv Tamar, Gilad Adler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5833b4daf5b076dd1cdb362b163dff0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5833b4daf5b076dd1cdb362b163dff0c-Abstract-Conference.html)

        **Abstract**:

        In meta reinforcement learning (meta RL), an agent learns from a set of training tasks how to quickly solve a new task, drawn from the same task distribution. The optimal meta RL policy, a.k.a.~the Bayes-optimal behavior, is well defined, and guarantees optimal reward in expectation, taken with respect to the task distribution. The question we explore in this work is how many training tasks are required to guarantee approximately optimal behavior with high probability. Recent work provided the first such PAC analysis for a model-free setting, where a history-dependent policy was learned from the training tasks. In this work, we propose a different approach: directly learn the task distribution, using density estimation techniques, and then train a policy on the learned task distribution. We show that our approach leads to bounds that depend on the dimension of the task distribution. In particular, in settings where the task distribution lies in a low-dimensional manifold, we extend our analysis to use dimensionality reduction techniques and account for such structure, obtaining significantly better bounds than previous work, which strictly depend on the number of states and actions. The key of our approach is the regularization implied by the kernel density estimation method. We further demonstrate that this regularization is useful in practice, when `plugged in' the state-of-the-art VariBAD meta RL algorithm.

        ----

        ## [991] TotalSelfScan: Learning Full-body Avatars from Self-Portrait Videos of Faces, Hands, and Bodies

        **Authors**: *Junting Dong, Qi Fang, Yudong Guo, Sida Peng, Qing Shuai, Xiaowei Zhou, Hujun Bao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/589c5bd0aa4322e37813e8e41ddf8034-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/589c5bd0aa4322e37813e8e41ddf8034-Abstract-Conference.html)

        **Abstract**:

        Recent advances in implicit neural representations make it possible to reconstruct a human-body model from a monocular self-rotation video. While previous works present impressive results of human body reconstruction, the quality of  reconstructed face and hands are relatively low. The main reason is that the image region occupied by these parts is very small compared to the body. To solve this problem, we propose a new approach named TotalSelfScan, which reconstructs the full-body model from several monocular self-rotation videos that focus on the face, hands, and body, respectively. Compared to recording a single video, this setting has almost no additional cost but provides more details of essential parts. To learn the full-body model, instead of encoding the whole body in a single network, we propose a multi-part representation to model separate parts and then fuse the part-specific observations into a single unified human model. Once learned, the full-body model enables rendering photorealistic free-viewpoint videos under novel human poses. Experiments show that TotalSelfScan can significantly improve the reconstruction and rendering quality on the face and hands compared to the existing methods. The code is available at \url{https://zju3dv.github.io/TotalSelfScan}.

        ----

        ## [992] DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning

        **Authors**: *Seungjae Lee, Jigang Kim, Inkyu Jang, H. Jin Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58b286aea34a91a3d33e58af0586fa40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58b286aea34a91a3d33e58af0586fa40-Abstract-Conference.html)

        **Abstract**:

        Hierarchical Reinforcement Learning (HRL) has made notable progress in complex control tasks by leveraging temporal abstraction. However, previous HRL algorithms often suffer from serious data inefficiency as environments get large. The extended components, $i.e.$, goal space and length of episodes, impose a burden on either one or both high-level and low-level policies since both levels share the total horizon of the episode. In this paper, we present a method of Decoupling Horizons Using a Graph in Hierarchical Reinforcement Learning (DHRL) which can alleviate this problem by decoupling the horizons of high-level and low-level policies and bridging the gap between the length of both horizons using a graph. DHRL provides a freely stretchable high-level action interval, which facilitates longer temporal abstraction and faster training in complex tasks. Our method outperforms state-of-the-art HRL algorithms in typical HRL environments. Moreover, DHRL achieves long and complex locomotion and manipulation tasks.

        ----

        ## [993] A Classification of $G$-invariant Shallow Neural Networks

        **Authors**: *Devanshu Agrawal, James Ostrowski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58b9a640af6d69781e90969d936e87ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58b9a640af6d69781e90969d936e87ce-Abstract-Conference.html)

        **Abstract**:

        When trying to fit a deep neural network (DNN) to a $G$-invariant target function with $G$ a group, it only makes sense to constrain the DNN to be $G$-invariant as well. However, there can be many different ways to do this, thus raising the problem of ``$G$-invariant neural architecture design'': What is the optimal $G$-invariant architecture for a given problem? Before we can consider the optimization problem itself, we must understand the search space, the architectures in it, and how they relate to one another. In this paper, we take a first step towards this goal; we prove a theorem that gives a classification of all $G$-invariant single-hidden-layer or ``shallow'' neural network ($G$-SNN) architectures with ReLU activation for any finite orthogonal group $G$, and we prove a second theorem that characterizes the inclusion maps or ``network morphisms'' between the architectures that can be leveraged during neural architecture search (NAS). The proof is based on a correspondence of every $G$-SNN to a signed permutation representation of $G$ acting on the hidden neurons; the classification is equivalently given in terms of the first cohomology classes of $G$, thus admitting a topological interpretation. The $G$-SNN architectures corresponding to nontrivial cohomology classes have, to our knowledge, never been explicitly identified in the literature previously. Using a code implementation, we enumerate the $G$-SNN architectures for some example groups $G$ and visualize their structure. Finally, we prove that architectures corresponding to inequivalent cohomology classes coincide in function space only when their weight matrices are zero, and we discuss the implications of this for NAS.

        ----

        ## [994] A Conditional Randomization Test for Sparse Logistic Regression in High-Dimension

        **Authors**: *Binh T. Nguyen, Bertrand Thirion, Sylvain Arlot*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58be158bf831a706b1a66cffbc401cac-Abstract-Conference.html)

        **Abstract**:

        Identifying the relevant variables for a classification model with correct confidence levels is a central but difficult task in high-dimension. Despite the core role of sparse logistic regression in statistics and machine learning, it still lacks a good solution for accurate inference in the regime where the number of features $p$ is as large as or larger than the number of samples $n$. Here we tackle this problem by improving the Conditional Randomization Test (CRT). The original CRT algorithm shows promise as a way to output p-values while making few assumptions on the distribution of the test statistics. As it comes with a prohibitive computational cost even in mildly high-dimensional problems, faster solutions based on distillation have been proposed. Yet, they rely on unrealistic hypotheses and result in low-power solutions. To improve this, we propose \emph{CRT-logit}, an algorithm that combines a variable-distillation step and a decorrelation step that takes into account the geometry of $\ell_1$-penalized logistic regression problem. We provide a theoretical analysis of this procedure, and demonstrate its effectiveness on simulations, along with experiments on large-scale brain-imaging and genomics datasets.

        ----

        ## [995] Biologically-Plausible Determinant Maximization Neural Networks for Blind Separation of Correlated Sources

        **Authors**: *Bariscan Bozkurt, Cengiz Pehlevan, Alper T. Erdogan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58cb483be90d31f9afea3a9e992a2abe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58cb483be90d31f9afea3a9e992a2abe-Abstract-Conference.html)

        **Abstract**:

        Extraction of latent sources of complex stimuli is critical for making sense of the world. While the brain solves this blind source separation (BSS) problem continuously, its algorithms remain unknown. Previous work on biologically-plausible BSS algorithms assumed that observed signals are linear mixtures of statistically independent or uncorrelated sources, limiting the domain of applicability of these algorithms. To overcome this limitation, we propose novel biologically-plausible neural networks for the blind separation of potentially dependent/correlated sources. Differing from previous work, we assume some general geometric, not statistical, conditions on the source vectors allowing separation of potentially dependent/correlated sources. Concretely, we assume that the source vectors are sufficiently scattered in their domains which can be described by certain polytopes. Then, we consider recovery of these sources by the Det-Max criterion, which maximizes the determinant of the output correlation matrix to enforce a similar spread for the source estimates. Starting from this normative principle, and using a weighted similarity matching approach that enables arbitrary linear transformations adaptable by local learning rules, we derive two-layer biologically-plausible neural network algorithms that can separate mixtures into sources coming from a variety of source domains. We demonstrate that our algorithms outperform other biologically-plausible BSS algorithms on correlated source separation problems.

        ----

        ## [996] Generalized One-shot Domain Adaptation of Generative Adversarial Networks

        **Authors**: *Zicheng Zhang, Yinglu Liu, Congying Han, Tiande Guo, Ting Yao, Tao Mei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58ce6a4b9c16d11975f11e4a23871041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58ce6a4b9c16d11975f11e4a23871041-Abstract-Conference.html)

        **Abstract**:

        The adaptation of a Generative Adversarial Network (GAN) aims to transfer a pre-trained GAN to a target domain with limited training data. In this paper, we focus on the one-shot case, which is more challenging and rarely explored in previous works. We consider that the adaptation from a source domain to a target domain can be decoupled into two parts: the transfer of global style like texture and color, and the emergence of new entities that do not belong to the source domain. While previous works mainly focus on style transfer, we propose a novel and concise framework to address the \textit{generalized one-shot adaptation} task for both style and entity transfer, in which a reference image and its binary entity mask are provided. Our core idea is to constrain the gap between the internal distributions of the reference and syntheses by sliced Wasserstein distance. To better achieve it, style fixation is used at first to roughly obtain the exemplary style, and an auxiliary network is introduced to the generator to disentangle entity and style transfer. Besides, to realize cross-domain correspondence, we propose the variational Laplacian regularization to constrain the smoothness of the adapted generator. Both quantitative and qualitative experiments demonstrate the effectiveness of our method in various scenarios. Code is available at \url{https://github.com/zhangzc21/Generalized-One-shot-GAN-adaptation}.

        ----

        ## [997] Representing Spatial Trajectories as Distributions

        **Authors**: *Dídac Surís, Carl Vondrick*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/58f4a9ca9031ff197cb4a61b456574bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/58f4a9ca9031ff197cb4a61b456574bf-Abstract-Conference.html)

        **Abstract**:

        We introduce a representation learning framework for spatial trajectories. We represent partial observations of trajectories as probability distributions in a learned latent space, which characterize the uncertainty about unobserved parts of the trajectory. Our framework allows us to obtain samples from a trajectory for any continuous point in timeâ€”both interpolating and extrapolating. Our flexible approach supports directly modifying specific attributes of a trajectory, such as its pace, as well as combining different partial observations into single representations. Experiments show our method's superiority over baselines in prediction tasks.

        ----

        ## [998] EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations

        **Authors**: *Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard E. L. Higgins, Sanja Fidler, David Fouhey, Dima Damen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/590a7ebe0da1f262c80d0188f5c4c222-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/590a7ebe0da1f262c80d0188f5c4c222-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce VISOR, a new dataset of pixel annotations and a benchmark suite for segmenting hands and active objects in egocentric video. VISOR annotates videos from EPIC-KITCHENS, which comes with a new set of challenges not encountered in current video segmentation datasets. Specifically, we need to ensure both short- and long-term consistency of pixel-level annotations as objects undergo transformative interactions, e.g. an onion is peeled, diced and cooked - where we aim to obtain accurate pixel-level annotations of the peel, onion pieces, chopping board, knife, pan, as well as the acting hands. VISOR introduces an annotation pipeline, AI-powered in parts, for scalability and quality. In total, we publicly release 272K manual semantic masks of 257 object classes, 9.9M interpolated dense masks, 67K hand-object relations, covering 36 hours of 179 untrimmed videos. Along with the annotations, we introduce three challenges in video object segmentation, interaction understanding and long-term reasoning.For data, code and leaderboards: http://epic-kitchens.github.io/VISOR

        ----

        ## [999] Inverse Design for Fluid-Structure Interactions using Graph Network Simulators

        **Authors**: *Kelsey R. Allen, Tatiana Lopez-Guevara, Kimberly L. Stachenfeld, Alvaro Sanchez-Gonzalez, Peter W. Battaglia, Jessica B. Hamrick, Tobias Pfaff*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/59593615e358d52295578e0d8e94ec4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/59593615e358d52295578e0d8e94ec4a-Abstract-Conference.html)

        **Abstract**:

        Designing physical artifacts that serve a purpose---such as tools and other functional structures---is central to engineering as well as everyday human behavior. Though automating design using machine learning has tremendous promise, existing methods are often limited by the task-dependent distributions they were exposed to during training. Here we showcase a task-agnostic approach to inverse design, by combining general-purpose graph network simulators with gradient-based design optimization. This constitutes a simple, fast, and reusable approach that solves high-dimensional problems with complex physical dynamics, including designing surfaces and tools to manipulate fluid flows and optimizing the shape of an airfoil to minimize drag. This framework produces high-quality designs by propagating gradients through trajectories of hundreds of steps, even when using models that were pre-trained for single-step predictions on data substantially different from the design tasks. In our fluid manipulation tasks, the resulting designs outperformed those found by sampling-based optimization techniques. In airfoil design, they matched the quality of those obtained with a specialized solver. Our results suggest that despite some remaining challenges, machine learning-based simulators are maturing to the point where they can support general-purpose design optimization across a variety of fluid-structure interaction domains.

        ----

        

[Go to the previous page](NIPS-2022-list04.md)

[Go to the next page](NIPS-2022-list06.md)

[Go to the catalog section](README.md)