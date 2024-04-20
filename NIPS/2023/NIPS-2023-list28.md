## [0] DAW: Exploring the Better Weighting Function for Semi-supervised Semantic Segmentation

**Authors**: *Rui Sun, Huayu Mai, Tianzhu Zhang, Feng Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c28ef8449dc21c90696c80ce47b3b5cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c28ef8449dc21c90696c80ce47b3b5cc-Abstract-Conference.html)

**Abstract**:

The critical challenge of semi-supervised semantic segmentation lies in how to fully exploit a large volume of unlabeled data to improve the modelâ€™s generalization performance for robust segmentation.  Existing methods tend to employ certain criteria (weighting function) to select pixel-level pseudo labels. However, the trade-off exists between inaccurate yet utilized pseudo-labels, and correct yet discarded pseudo-labels in these methods when handling pseudo-labels without thoughtful consideration of the weighting function, hindering the generalization ability of the model. In this paper, we systematically analyze the trade-off in previous methods when dealing with pseudo-labels. We formally define the trade-off between inaccurate yet utilized pseudo-labels, and correct yet discarded pseudo-labels by explicitly modeling the confidence distribution of correct and inaccurate pseudo-labels, equipped with a unified weighting function. To this end, we propose Distribution-Aware Weighting (DAW) to strive to minimize the negative equivalence impact raised by the trade-off. We find an interesting fact that the optimal solution for the weighting function is a hard step function, with the jump point located at the intersection of the two confidence distributions. Besides, we devise distribution alignment to mitigate the issue of the discrepancy between the prediction distributions of labeled and unlabeled data. Extensive experimental results on multiple benchmarks including mitochondria segmentation demonstrate that DAW performs favorably against state-of-the-art methods.

----

## [0] Feature Dropout: Revisiting the Role of Augmentations in Contrastive Learning

**Authors**: *Alex Tamkin, Margalit Glasgow, Xiluo He, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c290d4373c495b2cad0625d6288260f0-Abstract-Conference.html)

**Abstract**:

What role do augmentations play in contrastive learning? Recent work suggests that good augmentations are label-preserving with respect to a specific downstream task. We complicate this picture by showing that label-destroying augmentations can be useful in the foundation model setting, where the goal is to learn diverse, general-purpose representations for multiple downstream tasks. We perform contrastive learning experiments on a range of image and audio datasets with multiple downstream tasks (e.g. for digits superimposed on photographs, predicting the class of one vs. the other). We find that Viewmaker Networks, a recently proposed model for learning augmentations for contrastive learning, produce label-destroying augmentations that stochastically destroy features needed for different downstream tasks. These augmentations are interpretable (e.g. altering shapes, digits, or letters added to images) and surprisingly often result in better performance compared to expert-designed augmentations, despite not preserving label information. To support our empirical results, we theoretically analyze a simple contrastive learning setting with a linear model. In this setting, label-destroying augmentations are crucial for preventing one set of features from suppressing the learning of features useful for another downstream task. Our results highlight the need for analyzing the interaction between multiple downstream tasks when trying to explain the success of foundation models.

----

## [0] On the Exploitability of Instruction Tuning

**Authors**: *Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c2a8060fd22744b38177d9e428a052e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c2a8060fd22744b38177d9e428a052e0-Abstract-Conference.html)

**Abstract**:

Instruction tuning is an effective technique to align large language models (LLMs) with human intent. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs.

----

## [0] Residual Q-Learning: Offline and Online Policy Customization without Value

**Authors**: *Chenran Li, Chen Tang, Haruki Nishimura, Jean Mercat, Masayoshi Tomizuka, Wei Zhan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c2e4cebba2fdb3dac7d2022421062765-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c2e4cebba2fdb3dac7d2022421062765-Abstract-Conference.html)

**Abstract**:

Imitation Learning (IL) is a widely used framework for learning imitative behavior from demonstrations. It is especially appealing for solving complex real-world tasks where handcrafting reward function is difficult, or when the goal is to mimic human expert behavior. However, the learned imitative policy can only follow the behavior in the demonstration. When applying the imitative policy, we may need to customize the policy behavior to meet different requirements coming from diverse downstream tasks. Meanwhile, we still want the customized policy to maintain its imitative nature. To this end, we formulate a new problem setting called policy customization. It defines the learning task as training a policy that inherits the characteristics of the prior policy while satisfying some additional requirements imposed by a target downstream task. We propose a novel and principled approach to interpret and determine the trade-off between the two task objectives. Specifically, we formulate the customization problem as a Markov Decision Process (MDP) with a reward function that combines 1) the inherent reward of the demonstration; and 2) the add-on reward specified by the downstream task. We propose a novel framework, Residual Q-learning, which can solve the formulated MDP by leveraging the prior policy without knowing the inherent reward or value function of the prior policy. We derive a family of residual Q-learning algorithms that can realize offline and online policy customization, and show that the proposed algorithms can effectively accomplish policy customization tasks in various environments. Demo videos and code are available on our website: https://sites.google.com/view/residualq-learning.

----

## [0] LICO: Explainable Models with Language-Image COnsistency

**Authors**: *Yiming Lei, Zilong Li, Yangyang Li, Junping Zhang, Hongming Shan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c2eac51b6353a4441e8b7426f8e8db78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c2eac51b6353a4441e8b7426f8e8db78-Abstract-Conference.html)

**Abstract**:

Interpreting the decisions of deep learning models has been actively studied since the explosion of deep neural networks. One of the most convincing interpretation approaches is salience-based visual interpretation, such as Grad-CAM, where the generation of attention maps depends merely on categorical labels. Although existing interpretation methods can provide explainable decision clues, they often yield partial correspondence between image and saliency maps due to the limited discriminative information from one-hot labels. This paper develops a Language-Image COnsistency model for explainable image classification, termed LICO,  by correlating learnable linguistic prompts with corresponding visual features in a coarse-to-fine manner. Specifically, we first establish a coarse global manifold structure alignment by minimizing the distance between the distributions of image and language features. We then achieve fine-grained saliency maps by applying optimal transport (OT) theory to assign local feature maps with class-specific prompts. Extensive experimental results on eight benchmark datasets demonstrate that the proposed LICO achieves a significant improvement in generating more explainable attention maps in conjunction with existing interpretation methods such as Grad-CAM. Remarkably, LICO improves the classification performance of existing  models without introducing any computational overhead during inference.

----

## [0] Solving Inverse Physics Problems with Score Matching

**Authors**: *Benjamin J. Holzschuh, Simona Vegetti, Nils Thuerey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c2f2230abc7ccf669f403be881d3ffb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c2f2230abc7ccf669f403be881d3ffb7-Abstract-Conference.html)

**Abstract**:

We propose to solve inverse problems involving the temporal evolution of physics systems by leveraging recent advances from diffusion models. Our method moves the system's current state backward in time step by step by combining an approximate inverse physics simulator and a learned correction function. A central insight of our work is that training the learned correction with a single-step loss is equivalent to a score matching objective, while recursively predicting longer parts of the trajectory during training relates to maximum likelihood training of a corresponding probability flow.We highlight the advantages of our algorithm compared to standard denoising score matching and implicit score matching, as well as fully learned baselines for a wide range of inverse physics problems. The resulting inverse solver has excellent accuracy and temporal stability and, in contrast to other learned inverse solvers, allows for sampling the posterior of the solutions. Code and experiments are available at https://github.com/tum-pbs/SMDP.

----

## [0] Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples

**Authors**: *Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3532dd633e600e9f6db57aa7ae0c858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3532dd633e600e9f6db57aa7ae0c858-Abstract-Conference.html)

**Abstract**:

Mixup refers to interpolation-based data augmentation, originally motivated as a way to go beyond empirical risk minimization (ERM). Its extensions mostly focus on the definition of interpolation and the space (input or feature) where it takes place, while the augmentation process itself is less studied. In most methods, the number of generated examples is limited to the mini-batch size and the number of examples being interpolated is limited to two (pairs), in the input space.We make progress in this direction by introducing MultiMix, which generates an arbitrarily large number of interpolated examples beyond the mini-batch size and interpolates the entire mini-batch in the embedding space. Effectively, we sample on the entire convex hull of the mini-batch rather than along linear segments between pairs of examples.On sequence data, we further extend to Dense MultiMix. We densely interpolate features and target labels at each spatial location and also apply the loss densely. To mitigate the lack of dense labels, we inherit labels from examples and weight interpolation factors by attention as a measure of confidence.Overall, we increase the number of loss terms per mini-batch by orders of magnitude at little additional cost. This is only possible because of interpolating in the embedding space. We empirically show that our solutions yield significant improvement over state-of-the-art mixup methods on four different benchmarks, despite interpolation being only linear. By analyzing the embedding space, we show that the classes are more tightly clustered and uniformly spread over the embedding space, thereby explaining the improved behavior.

----

## [0] Approximation-Generalization Trade-offs under (Approximate) Group Equivariance

**Authors**: *Mircea Petrache, Shubhendu Trivedi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c35f8e2fc6d81f195009a1d2ae5f6ae9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c35f8e2fc6d81f195009a1d2ae5f6ae9-Abstract-Conference.html)

**Abstract**:

The explicit incorporation of task-specific inductive biases through symmetry has emerged as a general design precept in the development of high-performance machine learning models. For example, group equivariant neural networks have demonstrated impressive performance across various domains and applications such as protein and drug design. A prevalent intuition about such models is that the integration of relevant symmetry results in enhanced generalization. Moreover, it is posited that when the data and/or the model exhibits only approximate or partial symmetry, the optimal or best-performing model is one where the model symmetry aligns with the data symmetry. In this paper, we conduct a formal unified investigation of these intuitions. To begin, we present quantitative bounds that demonstrate how models capturing task-specific symmetries lead to improved generalization. Utilizing this quantification, we examine the more general question of dealing with approximate/partial symmetries. We establish, for a given symmetry group, a quantitative comparison between the approximate equivariance of the model and that of the data distribution, precisely connecting model equivariance error and data equivariance error. Our result delineates the conditions under which the model equivariance error is optimal, thereby yielding the best-performing model for the given task and data.

----

## [0] Equivariant Neural Operator Learning with Graphon Convolution

**Authors**: *Chaoran Cheng, Jian Peng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c362fbc0d182c6b4b8dadb90177239e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c362fbc0d182c6b4b8dadb90177239e4-Abstract-Conference.html)

**Abstract**:

We propose a general architecture that combines the coefficient learning scheme with a residual operator layer for learning mappings between continuous functions in the 3D Euclidean space. Our proposed model is guaranteed to achieve SE(3)-equivariance by design. From the graph spectrum view, our method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, our model can effectively capture the geometric information while preserving equivariance. Through extensive experiments on large-scale electron density datasets, we observed that our model significantly outperformed the current state-of-the-art architectures. Multiple ablation studies were also carried out to demonstrate the effectiveness of the proposed architecture.

----

## [0] Reinforcement Learning with Simple Sequence Priors

**Authors**: *Tankred Saanum, Noémi Élteto, Peter Dayan, Marcel Binz, Eric Schulz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3909e3abe8ebdb20c42a42ce0bc907d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3909e3abe8ebdb20c42a42ce0bc907d-Abstract-Conference.html)

**Abstract**:

In reinforcement learning (RL), simplicity is typically quantified on an action-by-action basis -- but this timescale ignores temporal regularities, like repetitions, often present in sequential strategies. We therefore propose an RL algorithm that learns to solve tasks with sequences of actions that are compressible. We explore two possible sources of simple action sequences: Sequences that can be learned by autoregressive models, and sequences that are compressible with off-the-shelf data compression algorithms. Distilling these preferences into sequence priors, we derive a novel information-theoretic objective that incentivizes agents to learn policies that maximize rewards while conforming to these priors. We show that the resulting RL algorithm leads to faster learning, and attains higher returns than state-of-the-art model-free approaches in a series of continuous control tasks from the DeepMind Control Suite. These priors also produce a powerful information-regularized agent that is robust to noisy observations and can perform open-loop control.

----

## [0] Fed-FA: Theoretically Modeling Client Data Divergence for Federated Language Backdoor Defense

**Authors**: *Zhiyuan Zhang, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c39578c86423df5f9e8834ce1cd456e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c39578c86423df5f9e8834ce1cd456e4-Abstract-Conference.html)

**Abstract**:

Federated learning algorithms enable neural network models to be trained across multiple decentralized edge devices without sharing private data. However, they are susceptible to backdoor attacks launched by malicious clients. Existing robust federated aggregation algorithms heuristically detect and exclude suspicious clients based on their parameter distances, but they are ineffective on Natural Language Processing (NLP) tasks. The main reason is that, although text backdoor patterns are obvious at the underlying dataset level, they are usually hidden at the parameter level, since injecting backdoors into texts with discrete feature space has less impact on the statistics of the model parameters. To settle this issue, we propose to identify backdoor clients by explicitly modeling the data divergence among clients in federated NLP systems. Through theoretical analysis, we derive the f-divergence indicator to estimate the client data divergence with aggregation updates and Hessians. Furthermore, we devise a dataset synthesization method with a Hessian reassignment mechanism guided by the diffusion theory to address the key challenge of inaccessible datasets in calculating clients' data Hessians.We then present the novel Federated F-Divergence-Based Aggregation~(\textbf{Fed-FA}) algorithm, which leverages the f-divergence indicator to detect and discard suspicious clients. Extensive empirical results show that Fed-FA outperforms all the parameter distance-based methods in defending against backdoor attacks among various natural language backdoor attack scenarios.

----

## [0] One Less Reason for Filter Pruning: Gaining Free Adversarial Robustness with Structured Grouped Kernel Pruning

**Authors**: *Shaochen (Henry) Zhong, Zaichuan You, Jiamu Zhang, Sebastian Zhao, Zachary LeClaire, Zirui Liu, Daochen Zha, Vipin Chaudhary, Shuai Xu, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3aba4234afd1c8116d879ba183f4835-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3aba4234afd1c8116d879ba183f4835-Abstract-Conference.html)

**Abstract**:

Densely structured pruning methods utilizing simple pruning heuristics can deliver immediate compression and acceleration benefits with acceptable benign performances. However, empirical findings indicate such naively pruned networks are extremely fragile under simple adversarial attacks. Naturally, we would be interested in knowing if such a phenomenon also holds for carefully designed modern structured pruning methods. If so, then to what extent is the severity? And what kind of remedies are available? Unfortunately, both the questions and the solution remain largely unaddressed: no prior art is able to provide a thorough investigation on the adversarial performance of modern structured pruning methods (spoiler: it is not good), yet the few works that attempt to provide mitigation often do so at various extra costs with only to-be-desired performance.In this work, we answer both questions by fairly and comprehensively investigating the adversarial performance of 10+ popular structured pruning methods. Solution-wise, we take advantage of Grouped Kernel Pruning (GKP)'s recent success in pushing densely structured pruning freedom to a more fine-grained level. By mixing up kernel smoothness — a classic robustness-related kernel-level metric — into a modified GKP procedure, we present a one-shot-post-train-weight-dependent GKP method capable of advancing SOTA performance on both the benign and adversarial scale, while requiring no extra (in fact, often less) cost than a standard pruning procedure. Please refer to our GitHub repository for code implementation, tool sharing, and model checkpoints.

----

## [0] Survival Instinct in Offline Reinforcement Learning

**Authors**: *Anqi Li, Dipendra Misra, Andrey Kolobov, Ching-An Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3e969ea20542a6a11e6caeac736a0b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3e969ea20542a6a11e6caeac736a0b9-Abstract-Conference.html)

**Abstract**:

We present a novel observation about the behavior of offline reinforcement learning (RL) algorithms: on many benchmark datasets, offline RL can produce well-performing and safe policies even when trained with "wrong" reward labels, such as those that are zero everywhere or are negatives of the true rewards. This phenomenon cannot be easily explained by offline RL's return maximization objective. Moreover, it gives offline RL a degree of robustness that is uncharacteristic of its online RL counterparts, which are known to be sensitive to reward design. We demonstrate that this surprising robustness property is attributable to an interplay between the notion of pessimism in offline RL algorithms and certain implicit biases in common data collection practices. As we prove in this work, pessimism endows the agent with a survival instinct, i.e., an incentive to stay within the data support in the long term, while the limited and biased data coverage further constrains the set of survival policies. Formally, given a reward class -- which may not even contain the true reward -- we identify conditions on the training data distribution that enable offline RL to learn a near-optimal and safe policy from any reward within the class. We argue that the survival instinct should be taken into account when interpreting results from existing offline RL benchmarks and when creating future ones. Our empirical and theoretical results suggest a new paradigm for offline RL, whereby an agent is "nudged" to learn a desirable behavior with imperfect reward but purposely biased data coverage. Please visit our website https://survival-instinct.github.io for accompanied code and videos.

----

## [0] Recurrent Hypernetworks are Surprisingly Strong in Meta-RL

**Authors**: *Jacob Beck, Risto Vuorio, Zheng Xiong, Shimon Whiteson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3fa3a7d50b34732c6d08f6f66380d75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3fa3a7d50b34732c6d08f6f66380d75-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (RL) is notoriously impractical to deploy due to sample inefficiency. Meta-RL directly addresses this sample inefficiency by learning to perform few-shot learning when a distribution of related tasks is available for meta-training. While many specialized meta-RL methods have been proposed, recent work suggests that end-to-end learning in conjunction with an off-the-shelf sequential model, such as a recurrent network, is a surprisingly strong baseline. However, such claims have been controversial due to limited supporting evidence, particularly in the face of prior work establishing precisely the opposite. In this paper, we conduct an empirical investigation. While we likewise find that a recurrent network can achieve strong performance, we demonstrate that the use of hypernetworks is crucial to maximizing their potential. Surprisingly, when combined with hypernetworks, the recurrent baselines that are far simpler than existing specialized methods actually achieve the strongest performance of all methods evaluated. We provide code at https://github.com/jacooba/hyper.

----

## [0] The Target-Charging Technique for Privacy Analysis across Interactive Computations

**Authors**: *Edith Cohen, Xin Lyu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c3fe2a07ec47b89c50e89706d2e23358-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c3fe2a07ec47b89c50e89706d2e23358-Abstract-Conference.html)

**Abstract**:

We propose the \emph{Target Charging Technique} (TCT), a unified privacy analysis framework for interactive settings where a sensitive dataset is accessed multiple times using differentially private algorithms. Unlike traditional composition, where privacy guarantees deteriorate quickly with the number of accesses, TCT allows computations that don't hit a specified \emph{target}, often the vast majority, to be essentially free (while incurring instead a small overhead on those that do hit their targets). TCT generalizes tools such as the sparse vector technique and top-k selection from private candidates and extends their remarkable privacy enhancement benefits from noisy Lipschitz functions to general private algorithms.

----

## [0] EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras

**Authors**: *Guangrong Zhao, Yurun Yang, Jingwei Liu, Ning Chen, Yiran Shen, Hongkai Wen, Guohao Lan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c41b5d8c1ba15b2aa83e4fa1541f02c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c41b5d8c1ba15b2aa83e4fa1541f02c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In this paper, we present EV-Eye, a first-of-its-kind large scale multimodal eye tracking dataset aimed at inspiring research on high-frequency eye/gaze tracking. EV-Eye utilizes an emerging bio-inspired event camera to capture independent pixel-level intensity changes induced by eye movements, achieving sub-microsecond latency. Our dataset was curated over a two-week period and collected from 48 participants encompassing diverse genders and age groups. It comprises over 1.5 million near-eye grayscale images and 2.7 billion event samples generated by two DAVIS346 event cameras. Additionally, the dataset contains 675 thousands scene images and 2.7 million gaze references captured by Tobii Pro Glasses 3 eye tracker for cross-modality validation. Compared with existing event-based high-frequency eye tracking datasets, our dataset is significantly larger in size, and the gaze references involve more natural eye movement patterns, i.e., fixation, saccade and smooth pursuit. Alongside the event data, we also present a hybrid eye tracking method as benchmark, which leverages both the near-eye grayscale images and event data for robust and high-frequency eye tracking. We show that our method achieves higher accuracy for both pupil and gaze estimation tasks compared to the existing solution.

----

## [0] Diffusion Schrödinger Bridge Matching

**Authors**: *Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract-Conference.html)

**Abstract**:

Solving transport problems, i.e. finding a map transporting one given distribution to another, has numerous applications in machine learning. Novel mass transport methods motivated by generative modeling have recently been proposed, e.g. Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) implement such a transport through a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE). However, while it is desirable in many applications to approximate the deterministic dynamic Optimal Transport (OT) map which admits attractive properties, DDMs and FMMs are not guaranteed to provide transports close to the OT map. In contrast, Schrödinger bridges (SBs) compute stochastic dynamic mappings which recover entropy-regularized versions of OT. Unfortunately, existing numerical methods approximating SBs either scale poorly with dimension or accumulate errors across iterations. In this work, we introduce Iterative Markovian Fitting (IMF), a new methodology for solving SB problems, and Diffusion Schrödinger Bridge Matching (DSBM), a novel numerical algorithm for computing IMF iterates. DSBM significantly improves over previous SB numerics and recovers as special/limiting cases various recent transport methods. We demonstrate the performance of DSBM on a variety of problems.

----

## [0] Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction

**Authors**: *Ruoyu Li, Qing Li, Yu Zhang, Dan Zhao, Yong Jiang, Yong Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c43b987f23fd5ea840df2b2be426315c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c43b987f23fd5ea840df2b2be426315c-Abstract-Conference.html)

**Abstract**:

Many security applications require unsupervised anomaly detection, as malicious data are extremely rare and often only unlabeled normal data are available for training (i.e., zero-positive). However, security operators are concerned about the high stakes of trusting black-box models due to their lack of interpretability. In this paper, we propose a post-hoc method to globally explain a black-box unsupervised anomaly detection model via rule extraction.First, we propose the concept of distribution decomposition rules that decompose the complex distribution of normal data into multiple compositional distributions. To find such rules, we design an unsupervised Interior Clustering Tree that incorporates the model prediction into the splitting criteria. Then, we propose the Compositional Boundary Exploration (CBE) algorithm to obtain the boundary inference rules that estimate the decision boundary of the original model on each compositional distribution. By merging these two types of rules into a rule set, we can present the inferential process of the unsupervised black-box model in a human-understandable way, and build a surrogate rule-based model for online deployment at the same time. We conduct comprehensive experiments on the explanation of four distinct unsupervised anomaly detection models on various real-world datasets. The evaluation shows that our method outperforms existing methods in terms of diverse metrics including fidelity, correctness and robustness.

----

## [0] Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning

**Authors**: *Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, Sergey Levine*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html)

**Abstract**:

A compelling use case of offline reinforcement learning (RL) is to obtain a policy initialization from existing datasets followed by fast online fine-tuning with limited interaction. However, existing offline RL methods tend to behave poorly during fine-tuning. In this paper, we devise an approach for learning an effective initialization from offline data that also enables fast online fine-tuning capabilities. Our approach, calibrated Q-learning (Cal-QL), accomplishes this by learning a conservative value function initialization that underestimates the value of the learned policy from offline data, while also being calibrated, in the sense that the learned Q-values are at a reasonable scale. We refer to this property as calibration, and define it formally as providing a lower bound on the true value function of the learned policy and an upper bound on the value of some other (suboptimal) reference policy, which may simply be the behavior policy. We show that offline RL algorithms that learn such calibrated value functions lead to effective online fine-tuning, enabling us to take the benefits of offline initializations in online fine-tuning. In practice, Cal-QL can be implemented on top of the conservative Q learning (CQL) for offline RL within a one-line code change. Empirically, Cal-QL outperforms state-of-the-art methods on 9/11 fine-tuning benchmark tasks that we study in this paper. Code and video are available at https://nakamotoo.github.io/Cal-QL

----

## [0] PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning

**Authors**: *Hojoon Lee, Hanseul Cho, Hyunseung Kim, Daehoon Gwak, Joonkee Kim, Jaegul Choo, Se-Young Yun, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c464fc4516aca4e68f2a14e67c6f0402-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c464fc4516aca4e68f2a14e67c6f0402-Abstract-Conference.html)

**Abstract**:

In Reinforcement Learning (RL), enhancing sample efficiency is crucial, particularly in scenarios when data acquisition is costly and risky. In principle, off-policy RL algorithms can improve sample efficiency by allowing multiple updates per environment interaction. However, these multiple updates often lead the model to overfit to earlier interactions, which is referred to as the loss of plasticity. Our study investigates the underlying causes of this phenomenon by dividing plasticity into two aspects. Input plasticity, which denotes the model's adaptability to changing input data, and label plasticity, which denotes the model's adaptability to evolving input-output relationships. Synthetic experiments on the CIFAR-10 dataset reveal that finding smoother minima of loss landscape enhances input plasticity, whereas refined gradient propagation improves label plasticity. Leveraging these findings, we introduce the PLASTIC algorithm, which harmoniously combines techniques to address both concerns. With minimal architectural modifications, PLASTIC achieves competitive performance on benchmarks including Atari-100k and Deepmind Control Suite. This result emphasizes the importance of preserving the model's plasticity to elevate the sample efficiency in RL. The code is available at https://github.com/dojeon-ai/plastic.

----

## [0] Metropolis Sampling for Constrained Diffusion Models

**Authors**: *Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, Valentin De Bortoli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c47bfcc8e2eccdc540fad1e25f13aa4d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c47bfcc8e2eccdc540fad1e25f13aa4d-Abstract-Conference.html)

**Abstract**:

Denoising diffusion models have recently emerged as the predominant paradigm for generative modelling on image domains. In addition, their extension to Riemannian manifolds has facilitated a range of applications across the natural sciences. While many of these problems stand to benefit from the ability to specify arbitrary, domain-informed constraints, this setting is not covered by the existing (Riemannian) diffusion model methodology. Recent work has attempted to address this issue by constructing novel noising processes based on the reflected Brownian motion and logarithmic barrier methods. However, the associated samplers are either computationally burdensome or only apply to convex subsets of Euclidean space. In this paper, we introduce an alternative, simple noising scheme based on Metropolis sampling that affords substantial gains in computational efficiency and empirical performance compared to the earlier samplers. Of independent interest, we prove that this new process corresponds to a valid discretisation of the reflected Brownian motion. We demonstrate the scalability and flexibility of our approach on a range of problem settings with convex and non-convex constraints, including applications from geospatial modelling, robotics and protein design.

----

## [0] ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction

**Authors**: *Yixun Liang, Hao He, Yingcong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c47ec10bc135be5c3663ba344d29a6a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c47ec10bc135be5c3663ba344d29a6a5-Abstract-Conference.html)

**Abstract**:

Generalizable neural surface reconstruction techniques have attracted great attention in recent years. However, they encounter limitations of low confidence depth distribution and inaccurate surface reasoning due to the oversimplified volume rendering process employed. In this paper, we present Reconstruction TRansformer (ReTR), a novel framework that leverages the transformer architecture to redesign the rendering process, enabling complex render interaction modeling. It introduces a learnable $\textit{meta-ray token}$ and utilizes the cross-attention mechanism to simulate the interaction of rendering process with sampled points and render the observed color. Meanwhile, by operating within a high-dimensional feature space rather than the color space, ReTR mitigates sensitivity to projected colors in source views. Such improvements result in accurate surface assessment with high confidence. We demonstrate the effectiveness of our approach on various datasets, showcasing how our method outperforms the current state-of-the-art approaches in terms of reconstruction quality and generalization ability. $\textit{Our code is available at }$ https://github.com/YixunLiang/ReTR.

----

## [0] FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation

**Authors**: *Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, Lu Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c481049f7410f38e788f67c171c64ad5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c481049f7410f38e788f67c171c64ad5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recently, open-domain text-to-video (T2V) generation models have made remarkable progress. However, the promising results are mainly shown by the qualitative cases of generated videos, while the quantitative evaluation of T2V models still faces two critical problems. Firstly, existing studies lack fine-grained evaluation of T2V models on different categories of text prompts. Although some benchmarks have categorized the prompts, their categorization either only focuses on a single aspect or fails to consider the temporal information in video generation. Secondly, it is unclear whether the automatic evaluation metrics are consistent with human standards. To address these problems, we propose FETV, a benchmark for Fine-grained Evaluation of Text-to-Video generation. FETV is multi-aspect, categorizing the prompts based on three orthogonal aspects: the major content, the attributes to control and the prompt complexity. FETV is also temporal-aware, which introduces several temporal categories tailored for video generation. Based on FETV, we conduct comprehensive manual evaluations of four representative T2V models, revealing their pros and cons on different categories of prompts from different aspects. We also extend FETV as a testbed to evaluate the reliability of automatic T2V metrics. The multi-aspect categorization of FETV enables fine-grained analysis of the metrics' reliability in different scenarios. We find that existing automatic metrics (e.g., CLIPScore and FVD) correlate poorly with human evaluation. To address this problem, we explore several solutions to improve CLIPScore and FVD, and develop two automatic metrics that exhibit significant higher correlation with humans than existing metrics. Benchmark page: https://github.com/llyx97/FETV.

----

## [0] Stability Guarantees for Feature Attributions with Multiplicative Smoothing

**Authors**: *Anton Xue, Rajeev Alur, Eric Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4889bd7f7ce643003746526da2c2fc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4889bd7f7ce643003746526da2c2fc4-Abstract-Conference.html)

**Abstract**:

Explanation methods for machine learning models tend not to provide any formal guarantees and may not reflect the underlying decision-making process.In this work, we analyze stability as a property for reliable feature attribution methods. We prove that relaxed variants of stability are guaranteed if the model is sufficiently Lipschitz with respect to the masking of features. We develop a smoothing method called Multiplicative Smoothing (MuS) to achieve such a model.We show that MuS overcomes the theoretical limitations of standard smoothing techniques and can be integrated with any classifier and feature attribution method.We evaluate MuS on vision and language models with various feature attribution methods, such as LIME and SHAP, and demonstrate that MuS endows feature attributions with non-trivial stability guarantees.

----

## [0] Pruning vs Quantization: Which is Better?

**Authors**: *Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, Tijmen Blankevoort*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c48bc80aa5d3cbbdd712d1cc107b8319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c48bc80aa5d3cbbdd712d1cc107b8319-Abstract-Conference.html)

**Abstract**:

Neural network pruning and quantization techniques are almost as old as neural networks themselves. However, to date, only ad-hoc comparisons between the two have been published. In this paper, we set out to answer the question of which is better: neural network quantization or pruning? By answering this question, we hope to inform design decisions made on neural network hardware going forward. We provide an extensive comparison between the two techniques for compressing deep neural networks. First, we give an analytical comparison of expected quantization and pruning error for general data distributions.Then, we provide lower and upper bounds for the per-layer pruning and quantization error in trained networks and compare these to empirical error after optimization.Finally, we provide an extensive experimental comparison for training 8 large-scale models trained on 3 tasks and provide insights into the representations learned during fine-tuning with quantization and pruning in the loop.Our results show that in most cases quantization outperforms pruning. Only in some scenarios with a very high compression ratio, compression might be beneficial from an accuracy standpoint.

----

## [0] EvoFed: Leveraging Evolutionary Strategies for Communication-Efficient Federated Learning

**Authors**: *Mohammad Mahdi Rahimi, Hasnain Irshad Bhatti, Younghyun Park, Humaira Kousar, Do-Yeon Kim, Jaekyun Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c48fe446e651cd49fb58a6833e015103-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c48fe446e651cd49fb58a6833e015103-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is a decentralized machine learning paradigm that enables collaborative model training across dispersed nodes without having to force individual nodes to share data.However, its broad adoption is hindered by the high communication costs of transmitting a large number of model parameters. This paper presents EvoFed, a novel approach that integrates Evolutionary Strategies (ES) with FL to address these challenges.EvoFed employs a concept of `fitness-based information sharingâ€™, deviating significantly from the conventional model-based FL. Rather than exchanging the actual updated model parameters, each node transmits a distance-based similarity measure between the locally updated model and each member of the noise-perturbed model population. Each node, as well as the server, generates an identical population set of perturbed models in a completely synchronized fashion using the same random seeds. With properly chosen noise variance and population size, perturbed models can be combined to closely reflect the actual model updated using the local dataset, allowing the transmitted similarity measures (or fitness values) to carry nearly the complete information about the model parameters.As the population size is typically much smaller than the number of model parameters, the savings in communication load is large. The server aggregates these fitness values and is able to update the global model. This global fitness vector is then disseminated back to the nodes, each of which applies the same update to be synchronized to the global model. Our analysis shows that EvoFed converges, and our experimental results validate that at the cost of increased local processing loads, EvoFed achieves performance comparable to FedAvg while reducing overall communication requirements drastically in various practical settings.

----

## [0] UUKG: Unified Urban Knowledge Graph Dataset for Urban Spatiotemporal Prediction

**Authors**: *Yansong Ning, Hao Liu, Hao Wang, Zhenyu Zeng, Hui Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4a30a4dd840cfeff30ba4d2661ff097-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4a30a4dd840cfeff30ba4d2661ff097-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Accurate Urban SpatioTemporal Prediction (USTP) is of great importance to the development and operation of the smart city. As an emerging building block, multi-sourced urban data are usually integrated as urban knowledge graphs (UrbanKGs) to provide critical knowledge for urban spatiotemporal prediction models. However, existing UrbanKGs are often tailored for specific downstream prediction tasks and are not publicly available, which limits the potential advancement. This paper presents UUKG, the unified urban knowledge graph dataset for knowledge-enhanced urban spatiotemporal predictions. Specifically, we first construct UrbanKGs consisting of millions of triplets for two metropolises by connecting heterogeneous urban entities such as administrative boroughs, POIs, and road segments. Moreover, we conduct qualitative and quantitative analysis on constructed UrbanKGs and uncover diverse high-order structural patterns, such as hierarchies and cycles, that can be leveraged to benefit downstream USTP tasks. To validate and facilitate the use of UrbanKGs, we implement and evaluate 15 KG embedding methods on the KG completion task and integrate the learned KG embeddings into 9 spatiotemporal models for five different USTP tasks. The extensive experimental results not only provide benchmarks of knowledge-enhanced USTP models under different task settings but also highlight the potential of state-of-the-art high-order structure-aware UrbanKG embedding methods. We hope the proposed UUKG fosters research on urban knowledge graphs and broad smart city applications. The dataset and source code are available at https://github.com/usail-hkust/UUKG/.

----

## [0] StateMask: Explaining Deep Reinforcement Learning through State Mask

**Authors**: *Zelei Cheng, Xian Wu, Jiahao Yu, Wenhai Sun, Wenbo Guo, Xinyu Xing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4bf73386022473a652a18941e9ea6f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4bf73386022473a652a18941e9ea6f8-Abstract-Conference.html)

**Abstract**:

Despite the promising performance of deep reinforcement learning (DRL) agents in many challenging scenarios, the black-box nature of these agents greatly limits their applications in critical domains. Prior research has proposed several explanation techniques to understand the deep learning-based policies in RL. Most existing methods explain why an agent takes individual actions rather than pinpointing the critical steps to its final reward. To fill this gap, we propose StateMask, a novel method to identify the states most critical to the agent's final reward. The high-level idea of StateMask is to learn a mask net that blinds a target agent and forces it to take random actions at some steps without compromising the agent's performance. Through careful design, we can theoretically ensure that the masked agent performs similarly to the original agent. We evaluate StateMask in various popular RL environments and show its superiority over existing explainers in explanation fidelity. We also show that StateMask  has better utilities, such as launching adversarial attacks and patching policy errors.

----

## [0] Faster Margin Maximization Rates for Generic Optimization Methods

**Authors**: *Guanghui Wang, Zihao Hu, Vidya Muthukumar, Jacob D. Abernethy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4cfdc27b46659e70a142ac249485a49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4cfdc27b46659e70a142ac249485a49-Abstract-Conference.html)

**Abstract**:

First-order optimization methods tend to inherently favor certain solutions over others when minimizing a given training objective with multiple local optima. This phenomenon, known as \emph{implicit bias}, plays a critical role in understanding the generalization capabilities of optimization algorithms. Recent research has revealed that gradient-descent-based methods exhibit an implicit bias for the $\ell_2$-maximal margin classifier in the context of separable binary classification. In contrast, generic optimization methods, such as mirror descent and steepest descent, have been shown to converge to maximal margin classifiers defined by alternative geometries. However, while gradient-descent-based algorithms demonstrate fast implicit bias rates, the implicit bias rates of generic optimization methods have been relatively slow. To address this limitation, in this paper, we present a series of state-of-the-art implicit bias rates for mirror descent and steepest descent algorithms. Our primary technique involves transforming a generic optimization algorithm into an online learning dynamic that solves a regularized bilinear game, providing a unified framework for analyzing the implicit bias of various optimization methods. The accelerated rates are derived leveraging the regret bounds of online learning algorithms within this game framework.

----

## [0] Managing Temporal Resolution in Continuous Value Estimation: A Fundamental Trade-off

**Authors**: *Zichen Vincent Zhang, Johannes Kirschner, Junxi Zhang, Francesco Zanini, Alex Ayoub, Masood Dehghan, Dale Schuurmans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4d66eae503694424123b93ac0fbaf17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4d66eae503694424123b93ac0fbaf17-Abstract-Conference.html)

**Abstract**:

A default assumption in reinforcement learning (RL) and optimal control is that observations arrive at discrete time points on a fixed clock cycle. Yet, many applications involve continuous-time systems where the time discretization, in principle, can be managed. The impact of time discretization on RL methods has not been fully characterized in existing theory, but a more detailed analysis of its effect could reveal opportunities for improving data-efficiency. We address this gap by analyzing Monte-Carlo policy evaluation for LQR systems and uncover a fundamental trade-off between approximation and statistical error in value estimation. Importantly, these two errors behave differently to time discretization, leading to an optimal choice of temporal resolution for a given data budget. These findings show that managing the temporal resolution can provably improve policy evaluation efficiency in LQR systems with finite data. Empirically, we demonstrate the trade-off in numerical simulations of LQR instances and standard RL benchmarks for non-linear continuous control.

----

## [0] Federated Linear Bandits with Finite Adversarial Actions

**Authors**: *Li Fan, Ruida Zhou, Chao Tian, Cong Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c4e380fb74dec9da9c7212e834657aa9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c4e380fb74dec9da9c7212e834657aa9-Abstract-Conference.html)

**Abstract**:

We study a federated linear bandits model, where $M$ clients communicate with a central server to solve a linear contextual bandits problem with finite adversarial action sets that may be different across clients. To address the unique challenges of **adversarial finite** action sets, we propose the FedSupLinUCB algorithm, which extends the principles of SupLinUCB and OFUL algorithms in linear contextual bandits. We prove that FedSupLinUCB achieves a total regret of $\tilde{O}(\sqrt{d T})$, where $T$ is the total number of arm pulls from all clients, and $d$ is the ambient dimension of the linear model. This matches the minimax lower bound and thus is order-optimal (up to polylog terms). We study both asynchronous and synchronous cases and show that the communication cost can be controlled as $O(d M^2 \log(d)\log(T))$ and $O(\sqrt{d^3 M^3} \log(d))$, respectively. The FedSupLinUCB design is further extended to two scenarios: (1) variance-adaptive, where a total regret of $\tilde{O} (\sqrt{d \sum \nolimits_{t=1}^{T} \sigma_t^2})$ can be achieved with $\sigma_t^2$ being the noise variance of round $t$; and (2) adversarial corruption, where a total regret of $\tilde{O}(\sqrt{dT} + d C_p)$ can be achieved with $C_p$ being the total corruption budget. Experiment results corroborate the theoretical analysis and demonstrate the effectiveness of \alg on both synthetic and real-world datasets.

----

## [0] BERT Lost Patience Won't Be Robust to Adversarial Slowdown

**Authors**: *Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c50a537060022ba5fc3d6a856625b664-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c50a537060022ba5fc3d6a856625b664-Abstract-Conference.html)

**Abstract**:

In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE

----

## [0] RECKONING: Reasoning through Dynamic Knowledge Encoding

**Authors**: *Zeming Chen, Gail Weiss, Eric Mitchell, Asli Celikyilmaz, Antoine Bosselut*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c518f504ad5894ccb264a9890f0f5544-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c518f504ad5894ccb264a9890f0f5544-Abstract-Conference.html)

**Abstract**:

Recent studies on transformer-based language models show that they can answer questions by reasoning over knowledge provided as part of the context (i.e., in-context reasoning). However, since the available knowledge is often not filtered for a particular question, in-context reasoning can be sensitive to distractor facts, additional content that is irrelevant to a question but that may be relevant for a different question (i.e., not necessarily random noise). In these situations, the model fails todistinguish the necessary knowledge to answer the question, leading to spurious reasoning and degraded performance. This reasoning failure contrasts with the model’s apparent ability to distinguish its contextual knowledge from all the knowledge it has memorized during pre-training. Following this observation, we propose teaching the model to reason more robustly by folding the provided contextual knowledge into the model’s parameters before presenting it with a question. Our method, RECKONING, is a bi-level learning algorithm that teaches language models to reason by updating their parametric knowledge through back-propagation, allowing them to answer questions using the updated parameters. During training, the inner loop rapidly adapts a copy of the model weights to encode contextual knowledge into its parameters. In the outer loop, the model learns to use the updated weights to reproduce and answer reasoning questions about the memorized knowledge. Our experiments on three diverse multi-hop reasoning datasets show that RECKONING’s performance improves over the in-context reasoning baseline (by up to 4.5%). We also find that compared to in-context reasoning, RECKONING generalizes better to longer reasoning chains unseen during training, is more robust to distractors in the context, and is computationally more efficient when multiple questions are asked about the same knowledge.

----

## [0] Regularity as Intrinsic Reward for Free Play

**Authors**: *Cansu Sancaktar, Justus H. Piater, Georg Martius*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c529dba08a146ea8d6cf715ae8930cbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c529dba08a146ea8d6cf715ae8930cbe-Abstract-Conference.html)

**Abstract**:

We propose regularity as a novel reward signal for intrinsically-motivated reinforcement learning. Taking inspiration from child development, we postulate that striving for structure and order helps guide exploration towards a subspace of tasks that are not favored by naive uncertainty-based intrinsic rewards. Our generalized formulation of Regularity as Intrinsic Reward (RaIR) allows us to operationalize it within model-based reinforcement learning. In a synthetic environment, we showcase the plethora of structured patterns that can emerge from pursuing this regularity objective. We also demonstrate the strength of our method in a multi-object robotic manipulation environment. We incorporate RaIR into free play and use it to complement the modelâ€™s epistemic uncertainty as an intrinsic reward. Doing so, we witness the autonomous construction of towers and other regular structures during free play, which leads to a substantial improvement in zero-shot downstream task performance on assembly tasks.

----

## [0] Guiding Large Language Models via Directional Stimulus Prompting

**Authors**: *Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html)

**Abstract**:

We introduce Directional Stimulus Prompting, a novel framework for guiding black-box large language models (LLMs) towards specific desired outputs. Instead of directly adjusting LLMs, our method employs a small tunable policy model (e.g., T5) to generate an auxiliary directional stimulus prompt for each input instance. These directional stimulus prompts act as nuanced, instance-specific hints and clues to guide LLMs in generating desired outcomes, such as including specific keywords in the generated summary. Our approach sidesteps the challenges of direct LLM tuning by optimizing the policy model to explore directional stimulus prompts that align LLMs with desired behaviors. The policy model can be optimized through 1) supervised fine-tuning using labeled data and 2) reinforcement learning from offline or online rewards based on the LLM's output. We evaluate our method across various tasks, including summarization, dialogue response generation, and chain-of-thought reasoning. Our experiments indicate a consistent improvement in the performance of LLMs such as ChatGPT, Codex, and InstructGPT on these supervised tasks with minimal labeled data. Remarkably, by utilizing merely 80 dialogues from the MultiWOZ dataset, our approach boosts ChatGPT's performance by a relative 41.4%, achieving or exceeding the performance of some fully supervised state-of-the-art models. Moreover, the instance-specific chain-of-thought prompt generated through our method enhances InstructGPT's reasoning accuracy, outperforming both generalized human-crafted prompts and those generated through automatic prompt engineering. The code and data are publicly available at https://github.com/Leezekun/Directional-Stimulus-Prompting.

----

## [0] Distributionally Robust Ensemble of Lottery Tickets Towards Calibrated Sparse Network Training

**Authors**: *Hitesh Sapkota, Dingrong Wang, Zhiqiang Tao, Qi Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c5cf13bfd3762821ef7607e63ee90075-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c5cf13bfd3762821ef7607e63ee90075-Abstract-Conference.html)

**Abstract**:

The recently developed sparse network training methods, such as Lottery Ticket Hypothesis (LTH) and its variants, have shown impressive learning capacity by finding sparse sub-networks from a dense one. While these methods could largely sparsify deep networks, they generally focus more on realizing comparable accuracy to dense counterparts yet neglect network calibration. However, how to achieve calibrated network predictions lies at the core of improving model reliability, especially when it comes to addressing the overconfident issue and out-of-distribution cases. In this study, we propose a novel Distributionally Robust Optimization (DRO) framework to achieve an ensemble of lottery tickets towards calibrated network sparsification. Specifically, the proposed DRO ensemble aims to learn multiple diverse and complementary sparse sub-networks (tickets) with the guidance of uncertainty sets, which encourage tickets to gradually capture different data distributions from easy to hard and naturally complement each other. We theoretically justify the strong calibration performance by showing how the proposed robust training process guarantees to lower the confidence of incorrect predictions. Extensive experimental results on several benchmarks show that our proposed lottery ticket ensemble leads to a clear calibration improvement without sacrificing accuracy and burdening inference costs. Furthermore, experiments on OOD datasets demonstrate the robustness of our approach in the open-set environment.

----

## [0] A Recurrent Neural Circuit Mechanism of Temporal-scaling Equivariant Representation

**Authors**: *Junfeng Zuo, Xiao Liu, Ying Nian Wu, Si Wu, Wenhao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c5e44243e16c9d61d3897ba1095f5f6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c5e44243e16c9d61d3897ba1095f5f6c-Abstract-Conference.html)

**Abstract**:

Time perception is critical in our daily life. An important feature of time perception is temporal scaling (TS): the ability to generate temporal sequences (e.g., motor actions) at different speeds. However, it is largely unknown about the math principle underlying temporal scaling in recurrent circuits in the brain. To shed insight, the present study investigates the temporal scaling from the Lie group point of view. We propose a canonical nonlinear recurrent circuit dynamics, modeled as a continuous attractor network, whose neuronal population responses embed a temporal sequence that is TS equivariant. Furthermore, we found the TS group operators can be explicitly represented by a control input fed into the recurrent circuit, where the input gain determines the temporal scaling factor (group parameter), and the spatial offset between the control input and network state emerges the generator.  The neuronal responses in the recurrent circuit are also consistent with experimental findings. We illustrated that the recurrent circuit can drive a feedforward circuit to generate complex temporal sequences with different time scales, even in the case of negative time scaling (''time reversal'').  Our work for the first time analytically links the abstract temporal scaling group and concrete neural circuit dynamics.

----

## [0] Sample Complexity of Goal-Conditioned Hierarchical Reinforcement Learning

**Authors**: *Arnaud Robert, Ciara Pike-Burke, Aldo A. Faisal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c5ed2c8acda8c3716b1b6f9c6c713aaa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c5ed2c8acda8c3716b1b6f9c6c713aaa-Abstract-Conference.html)

**Abstract**:

Hierarchical Reinforcement Learning (HRL) algorithms can perform planning at multiple levels of abstraction. Empirical results have shown that state or temporal abstractions might significantly improve the sample efficiency of algorithms. Yet, we still do not have a complete understanding of the basis of those efficiency gains nor any theoretically grounded design rules. In this paper, we derive a lower bound on the sample complexity for the considered class of goal-conditioned HRL algorithms. The proposed lower bound empowers us to quantify the benefits of hierarchical decomposition and leads to the design of a simple Q-learning-type algorithm that leverages hierarchical decompositions. We empirically validate our theoretical findings by investigating the sample complexity of the proposed hierarchical algorithm on a spectrum of tasks (hierarchical $n$-rooms, Gymnasium's Taxi). The hierarchical $n$-rooms tasks were designed to allow us to dial their complexity over multiple orders of magnitude. Our theory and algorithmic findings provide a step towards answering the foundational question of quantifying the improvement hierarchical decomposition offers over monolithic solutions in reinforcement learning.

----

## [0] MiliPoint: A Point Cloud Dataset for mmWave Radar

**Authors**: *Han Cui, Shu Zhong, Jiacheng Wu, Zichao Shen, Naim Dahnoun, Yiren Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c60468eca9cd0b0083f0ff9d0aeb171a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c60468eca9cd0b0083f0ff9d0aeb171a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Millimetre-wave (mmWave) radar has emerged as an attractive and cost-effective alternative for human activity sensing compared to traditional camera-based systems. mmWave radars are also non-intrusive, providing better protection for user privacy. However, as a Radio Frequency based technology, mmWave radars rely on capturing reflected signals from objects, making them more prone to noise compared to cameras. This raises an intriguing question for the deep learning community: Can we develop more effective point set-based deep learning methods for such attractive sensors?   To answer this question, our work, termed MiliPoint, delves into this idea by providing a large-scale, open dataset for the community to explore how mmWave radars can be utilised for human activity recognition. Moreover, MiliPoint stands out as it is larger in size than existing datasets, has more diverse human actions represented, and encompasses all three key tasks in human activity recognition. We have also established a range of point-based deep neural networks such as DGCNN, PointNet++ and PointTransformer, on MiliPoint, which can serve to set the ground baseline for further development.

----

## [0] NAR-Former V2: Rethinking Transformer for Universal Neural Network Representation Learning

**Authors**: *Yun Yi, Haokui Zhang, Rong Xiao, Nannan Wang, Xiaoyu Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c60bd92a01804b7df0540ed7ca2f7c05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c60bd92a01804b7df0540ed7ca2f7c05-Abstract-Conference.html)

**Abstract**:

As more deep learning models are being applied in real-world applications, there is a growing need for modeling and learning the representations of neural networks themselves. An effective representation can be used to predict target attributes of networks without the need for actual training and deployment procedures, facilitating efficient network design and deployment. Recently, inspired by the success of Transformer, some Transformer-based representation learning frameworks have been proposed and achieved promising performance in handling cell-structured models. However, graph neural network (GNN) based approaches still dominate the field of learning representation for the entire network. In this paper, we revisit the Transformer and compare it with GNN to analyze their different architectural characteristics. We then propose a modified Transformer-based universal neural network representation learning model NAR-Former V2. It can learn efficient representations from both cell-structured networks and entire networks. Specifically, we first take the network as a graph and design a straightforward tokenizer to encode the network into a sequence. Then, we incorporate the inductive representation learning capability of GNN into Transformer, enabling Transformer to generalize better when encountering unseen architecture. Additionally, we introduce a series of simple yet effective modifications to enhance the ability of the Transformer in learning representation from graph structures. In encoding entire networks and then predicting the latency, our proposed method surpasses the GNN-based method NNLP by a significant margin on the NNLQP dataset. Furthermore, regarding accuracy prediction on the cell-structured NASBench101 and NASBench201 datasets, our method achieves highly comparable performance to other state-of-the-art methods. The code is available at https://github.com/yuny220/NAR-Former-V2.

----

## [0] Sensitivity in Translation Averaging

**Authors**: *Lalit Manam, Venu Madhav Govindu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c62fe1daeb10814d33e5a33ba466ecaf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c62fe1daeb10814d33e5a33ba466ecaf-Abstract-Conference.html)

**Abstract**:

In 3D computer vision, translation averaging solves for absolute translations given a set of pairwise relative translation directions. While there has been much work on robustness to outliers and studies on the uniqueness of the solution, this paper deals with a distinctly different problem of sensitivity in translation averaging under uncertainty. We first analyze sensitivity in estimating scales corresponding to relative directions under small perturbations of the relative directions. Then, we formally define the conditioning of the translation averaging problem, which assesses the reliability of estimated translations based solely on the input directions. We give a sufficient criterion to ensure that the problem is well-conditioned. Subsequently, we provide an efficient algorithm to identify and remove combinations of directions which make the problem ill-conditioned while ensuring uniqueness of the solution. We demonstrate the utility of such analysis in global structure-from-motion pipelines for obtaining 3D reconstructions, which reveals the benefits of filtering the ill-conditioned set of directions in translation averaging in terms of reduced translation errors, a higher number of 3D points triangulated and faster convergence of bundle adjustment.

----

## [0] Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness and Smoothness

**Authors**: *Chung-En Tsai, Ying-Ting Lin, Yen-Huan Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html)

**Abstract**:

This work introduces the first small-loss and gradual-variation regret bounds for online portfolio selection, marking the first instances of data-dependent bounds for online convex optimization with non-Lipschitz, non-smooth losses. The algorithms we propose exhibit sublinear regret rates in the worst cases and achieve logarithmic regrets when the data is "easy," with per-round time almost linear in the number of investment alternatives. The regret bounds are derived using novel smoothness characterizations of the logarithmic loss, a local norm-based analysis of following the regularized leader (FTRL) with self-concordant regularizers, which are not necessarily barriers, and an implicit variant of optimistic FTRL with the log-barrier.

----

## [0] Outlier-Robust Wasserstein DRO

**Authors**: *Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c67b138497305835e76fdedd48dd4e59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c67b138497305835e76fdedd48dd4e59-Abstract-Conference.html)

**Abstract**:

Distributionally robust optimization (DRO) is an effective approach for data-driven decision-making in the presence of uncertainty. Geometric uncertainty due to~sampling or localized perturbations of data points is captured by Wasserstein DRO (WDRO), which seeks to learn a model that performs uniformly well over a Wasserstein ball centered around the observed data distribution. However, WDRO fails to account for non-geometric perturbations such as adversarial outliers, which can greatly distort the Wasserstein distance measurement and impede the learned model. We address this gap by proposing a novel outlier-robust WDRO framework for decision-making under both geometric (Wasserstein) perturbations and non-geometric (total variation (TV)) contamination that allows an $\varepsilon$-fraction of data to be arbitrarily corrupted. We design an uncertainty set using a certain robust Wasserstein ball that accounts for both perturbation types and derive minimax optimal excess risk bounds for this procedure that explicitly capture the Wasserstein and TV risks. We prove a strong duality result that enables tractable convex reformulations and efficient computation of our outlier-robust WDRO problem. When the loss function depends only on low-dimensional features of the data, we eliminate certain dimension dependencies from the risk bounds that are unavoidable in the general setting. Finally, we present experiments validating our theory on standard regression and classification tasks.

----

## [0] Certified Minimax Unlearning with Generalization Rates and Deletion Capacity

**Authors**: *Jiaqi Liu, Jian Lou, Zhan Qin, Kui Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c69465280855cfe25d566e359da140c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c69465280855cfe25d566e359da140c1-Abstract-Conference.html)

**Abstract**:

We study the problem of $(\epsilon,\delta)$-certified machine unlearning for minimax models. Most of the existing works focus on unlearning from standard statistical learning models that have a single variable and their unlearning steps hinge on the direct Hessian-based conventional Newton update. We develop a new $(\epsilon,\delta)$-certified machine unlearning algorithm for minimax models. It proposes a minimax unlearning step consisting of a total Hessian-based complete Newton update and the Gaussian mechanism borrowed from differential privacy. To obtain the unlearning certification, our method injects calibrated Gaussian noises by carefully analyzing the ''sensitivity'' of the minimax unlearning step (i.e., the closeness between the minimax unlearning variables and the retraining-from-scratch variables). We derive the generalization rates in terms of population strong and weak primal-dual risk for three different cases of loss functions, i.e., (strongly-)convex-(strongly-)concave losses. We also provide the deletion capacity to guarantee that a desired population risk can be maintained as long as the number of deleted samples does not exceed the derived amount. With training samples $n$ and model dimension $d$, it yields the order $\mathcal O(n/d^{1/4})$, which shows a strict gap over the baseline method of differentially private minimax learning that has $\mathcal O(n/d^{1/2})$. In addition, our rates of generalization and deletion capacity match the state-of-the-art rates derived previously for standard statistical learning models.

----

## [0] An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations

**Authors**: *Haoran Yang, Xiangyu Zhao, Yicong Li, Hongxu Chen, Guandong Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6af791af7ef0f3e02bccef011211ca5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6af791af7ef0f3e02bccef011211ca5-Abstract-Conference.html)

**Abstract**:

Graph contrastive learning (GCL) has emerged as a potent technology for numerous graph learning tasks. It has been successfully applied to real-world recommender systems, where the contrastive loss and the downstream recommendation objectives are always combined to form the overall objective function. Such a strategy is inconsistent with the original GCL paradigm, where graph embeddings are pre-trained without involving downstream training objectives. In this paper, we innovatively propose a prompt-enhanced framework for GCL-based recommender systems, namely CPTPP, which can fully leverage the advantages of the original GCL protocol through prompt tuning. Specifically, we first summarise user profiles in graph recommender systems to automatically generate personalized user prompts. These prompts will then be combined with pre-trained user embeddings to conduct prompt-tuning in downstream tasks, thereby narrowing the distinct targets between pre-training and downstream tasks. Extensive experiments on three benchmark datasets validate the effectiveness of CPTPP against state-of-the-art baselines. A further visualization experiment demonstrates that user embeddings generated by CPTPP have a more uniform distribution, indicating a better capacity to model the diversity of user preferences.The implementation code is available online to ease reproducibility: https://anonymous.4open.science/r/CPTPP-F8F4

----

## [0] Can Language Models Teach? Teacher Explanations Improve Student Performance via Personalization

**Authors**: *Swarnadeep Saha, Peter Hase, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6afe9a5d1e1068796d32613ddca1ab7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6afe9a5d1e1068796d32613ddca1ab7-Abstract-Conference.html)

**Abstract**:

A hallmark property of explainable AI models is the ability to teach other agents, communicating knowledge of how to perform a task. While Large Language Models (LLMs) perform complex reasoning by generating explanations for their predictions, it is unclear whether they also make good teachers for weaker agents. To address this, we consider a student-teacher framework between two LLM agents and study if, when, and how the teacher should intervene with natural language explanations to improve the student’s performance. Since communication is expensive, we define a budget such that the teacher only communicates explanations for a fraction of the data, after which the student should perform well on its own. We decompose the teaching problem along four axes: (1) if teacher’s test time in- tervention improve student predictions, (2) when it is worth explaining a data point, (3) how the teacher should personalize explanations to better teach the student, and (4) if teacher explanations also improve student performance on future unexplained data. We first show that teacher LLMs can indeed intervene on student reasoning to improve their performance. Next, inspired by the Theory of Mind abilities of effective teachers, we propose building two few-shot mental models of the student. The first model defines an Intervention Function that simulates the utility of an intervention, allowing the teacher to intervene when this utility is the highest and improving student performance at lower budgets. The second model enables the teacher to personalize explanations for a particular student and outperform unpersonalized teachers. We also demonstrate that in multi-turn interactions, teacher explanations generalize and learning from explained data improves student performance on future unexplained data. Finally, we also verify that misaligned teachers can lower student performance to random chance by intentionally misleading them.

----

## [0] Finding Local Minima Efficiently in Decentralized Optimization

**Authors**: *Wenhan Xian, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6b84d35d783cf289bb0cd7c7b897ea6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6b84d35d783cf289bb0cd7c7b897ea6-Abstract-Conference.html)

**Abstract**:

In this paper we study the second-order optimality of decentralized stochastic algorithm that escapes saddle point efficiently for nonconvex optimization problems. We propose a new pure gradient-based decentralized stochastic algorithm PEDESTAL with a novel convergence analysis framework to address the technical challenges unique to the decentralized stochastic setting. Our method is the first decentralized stochastic algorithm to achieve second-order optimality with non-asymptotic analysis. We provide theoretical guarantees with the gradient complexity of $\tilde{O} (\epsilon^{-3})$ to find $O(\epsilon, \sqrt{\epsilon})$-second-order stationary point, which matches state-of-the-art results of centralized counterparts or decentralized methods to find first-order stationary point. We also conduct two decentralized tasks in our experiments, a matrix sensing task with synthetic data and a matrix factorization task with a real-world dataset to validate the performance of our method.

----

## [0] Clifford Group Equivariant Neural Networks

**Authors**: *David Ruhe, Johannes Brandstetter, Patrick Forré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6e0125e14ea3d1a3de3c33fd2d49fc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6e0125e14ea3d1a3de3c33fd2d49fc4-Abstract-Conference.html)

**Abstract**:

We introduce Clifford Group Equivariant Neural Networks: a novel approach for constructing $\mathrm{O}(n)$- and $\mathrm{E}(n)$-equivariant models. We identify and study the *Clifford group*: a subgroup inside the Clifford algebra tailored to achieve several favorable properties. Primarily, the group's action forms an orthogonal automorphism that extends beyond the typical vector space to the entire Clifford algebra while respecting the multivector grading. This leads to several non-equivalent subrepresentations corresponding to the multivector decomposition. Furthermore, we prove that the action respects not just the vector space structure of the Clifford algebra but also its multiplicative structure, i.e., the geometric product. These findings imply that every polynomial in multivectors, including their grade projections, constitutes an equivariant map with respect to the Clifford group, allowing us to parameterize equivariant neural network layers. An advantage worth mentioning is that we obtain expressive layers that can elegantly generalize to inner-product spaces of any dimension. We demonstrate, notably from a single core implementation, state-of-the-art performance on several distinct tasks, including a three-dimensional $n$-body experiment, a four-dimensional Lorentz-equivariant high-energy physics experiment, and a five-dimensional convex hull experiment.

----

## [0] C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models

**Authors**: *Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, Junxian He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6ec1844bec96d6d32ae95ae694e23d8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6ec1844bec96d6d32ae95ae694e23d8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

New NLP benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present C-Eval, the first comprehensive Chinese evaluation suite designed to assess advanced knowledge and reasoning abilities of foundation models in a Chinese context. C-Eval comprises multiple-choice questions across four difficulty levels: middle school, high school, college, and professional. The questions span 52 diverse disciplines, ranging from humanities to science and engineering. C-Eval is accompanied by C-Eval Hard, a subset of very challenging subjects in C-Eval that requires advanced reasoning abilities to solve. We conduct a comprehensive evaluation of the most advanced LLMs on C-Eval, including both English- and Chinese-oriented models. Results indicate that only GPT-4 could achieve an average accuracy of over 60%, suggesting that there is still significant room for improvement for current LLMs. We anticipate C-Eval will help analyze important strengths and shortcomings of foundation models, and foster their development and growth for Chinese users.

----

## [0] NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF

**Authors**: *Stefan Lionar, Xiangyu Xu, Min Lin, Gim Hee Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c6f1e44be16e87887b7b894d59ba7f29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c6f1e44be16e87887b7b894d59ba7f29-Abstract-Conference.html)

**Abstract**:

Remarkable progress has been made in 3D reconstruction from single-view RGB-D inputs. MCC is the current state-of-the-art method in this field, which achieves unprecedented success by combining vision Transformers with large-scale training. However, we identified two key limitations of MCC: 1) The Transformer decoder is inefficient in handling large number of query points; 2) The 3D representation struggles to recover high-fidelity details. In this paper, we propose a new approach called NU-MCC that addresses these limitations. NU-MCC includes two key innovations: a Neighborhood decoder and a Repulsive Unsigned Distance Function (Repulsive UDF). First, our Neighborhood decoder introduces center points as an efficient proxy of input visual features, allowing each query point to only attend to a small neighborhood. This design not only results in much faster inference speed but also enables the exploitation of finer-scale visual features for improved recovery of 3D textures. Second, our Repulsive UDF is a novel alternative to the occupancy field used in MCC, significantly improving the quality of 3D object reconstruction. Compared to standard UDFs that suffer from holes in results, our proposed Repulsive UDF can achieve more complete surface reconstruction. Experimental results demonstrate that NU-MCC is able to learn a strong 3D representation, significantly advancing the state of the art in single-view 3D reconstruction. Particularly, it outperforms MCC by 9.7% in terms of the F1-score on the CO3D-v2 dataset with more than 5x faster running speed.

----

## [0] Convergence analysis of ODE models for accelerated first-order methods via positive semidefinite kernels

**Authors**: *Jungbin Kim, Insoon Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c70741145c2c4f1d0c2e91b98729a49a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c70741145c2c4f1d0c2e91b98729a49a-Abstract-Conference.html)

**Abstract**:

We propose a novel methodology that systematically analyzes ordinary differential equation (ODE) models for first-order optimization methods by converting the task of proving convergence rates into verifying the positive semidefiniteness of specific Hilbert-Schmidt integral operators. Our approach is based on the performance estimation problems (PEP) introduced by Drori and Teboulle. Unlike previous works on PEP, which rely on finite-dimensional linear algebra, we use tools from functional analysis. Using the proposed method, we establish convergence rates of various accelerated gradient flow models, some of which are new. As an immediate consequence of our framework, we show a correspondence between minimizing function values and minimizing gradient norms.

----

## [0] Curvature Filtrations for Graph Generative Model Evaluation

**Authors**: *Joshua Southern, Jeremy Wayland, Michael Bronstein, Bastian Rieck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c710d6b4507e70c6332bee871b8d1ca5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c710d6b4507e70c6332bee871b8d1ca5-Abstract-Conference.html)

**Abstract**:

Graph generative model evaluation necessitates understanding differences between graphs on the distributional level. This entails being able to harness salient attributes of graphs in an efficient manner. Curvature constitutes one such property of graphs, and has recently started to prove useful in characterising graphs. Its expressive properties, stability, and practical utility in model evaluation remain largely unexplored, however. We combine graph curvature descriptors with emerging methods from topological data analysis to obtain robust, expressive descriptors for evaluating graph generative models.

----

## [0] DiffUTE: Universal Text Editing Diffusion Model

**Authors**: *Haoxing Chen, Zhuoer Xu, Zhangxuan Gu, Jun Lan, Xing Zheng, Yaohui Li, Changhua Meng, Huijia Zhu, Weiqiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html)

**Abstract**:

Diffusion model based language-guided image editing has achieved great success recently. However, existing state-of-the-art diffusion models struggle with rendering correct text and text style during generation. To tackle this problem, we propose a universal self-supervised text editing diffusion model (DiffUTE), which aims to replace or modify words in the source image with another one while maintaining its realistic appearance. Specifically, we build our model on a diffusion model and carefully modify the network structure to enable the model for drawing multilingual characters with the help of glyph and position information. Moreover, we design a self-supervised learning framework to leverage large amounts of web data to improve the representation ability of the model. Experimental results show that our method achieves an impressive performance and enables controllable editing on in-the-wild images with high fidelity. Our code will be avaliable in \url{https://github.com/chenhaoxing/DiffUTE}.

----

## [0] Sampling weights of deep neural networks

**Authors**: *Erik Lien Bolager, Iryna Burak, Chinmay Datar, Qing Sun, Felix Dietrich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7201deff8d507a8fe2e86d34094e154-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7201deff8d507a8fe2e86d34094e154-Abstract-Conference.html)

**Abstract**:

We introduce a probability distribution, combined with an efficient sampling algorithm, for weights and biases of fully-connected neural networks. In a supervised learning context, no iterative optimization or gradient computations of internal network parameters are needed to obtain a trained network. The sampling is based on the idea of random feature models. However, instead of a data-agnostic distribution, e.g., a normal distribution, we use both the input and the output training data to sample shallow and deep networks. We prove that sampled networks are universal approximators. For Barron functions, we show that the $L^2$-approximation error of sampled shallow networks decreases with the square root of the number of neurons. Our sampling scheme is invariant to rigid body transformations and scaling of the input data, which implies many popular pre-processing techniques are not required. In numerical experiments, we demonstrate that sampled networks achieve accuracy comparable to iteratively trained ones, but can be constructed orders of magnitude faster. Our test cases involve a classification benchmark from OpenML, sampling of neural operators to represent maps in function spaces, and transfer learning using well-known architectures.

----

## [0] Fast Attention Requires Bounded Entries

**Authors**: *Josh Alman, Zhao Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html)

**Abstract**:

In modern machine learning, inner product attention computation is a fundamental task for training large language models such as Transformer, GPT-1, BERT, GPT-2, GPT-3 and ChatGPT. Formally, in this problem, one is given as input three matrices $Q, K, V \in [-B,B]^{n \times d}$, and the goal is to construct the matrix $\mathrm{Att}(Q,K,V) := \mathrm{diag}(A {\bf 1}_n)^{-1} A V \in \mathbb{R}^{n \times d}$, where $A = \exp(QK^\top/d)$ is the `attention matrix', and $\exp$ is applied entry-wise. Straightforward methods for this problem explicitly compute the $n \times n$ attention matrix $A$, and hence require time $\Omega(n^2)$ even when $d = n^{o(1)}$ is small. In this paper, we investigate whether faster algorithms are possible by \emph{implicitly} making use of the matrix $A$. We present two results, showing that there is a sharp transition at $B = \Theta(\sqrt{\log n})$.$\bullet$ If $d = O(\log n)$ and $B = o(\sqrt{\log n})$, there is an $n^{1+o(1)}$ time algorithm to approximate $\mathrm{Att}(Q,K,V)$ up to $1/\mathrm{poly}(n)$ additive error.$\bullet$ If $d = O(\log n)$ and $B = \Theta (\sqrt{\log n})$, assuming the Strong Exponential Time Hypothesis from fine-grained complexity theory, it is impossible to approximate $\mathrm{Att}(Q,K,V)$ up to $1/\mathrm{poly}(n)$ additive error in truly subquadratic time $n^{2 - \Omega(1)}$.This gives a theoretical explanation for the phenomenon observed in practice that attention computation is much more efficient when the input matrices have smaller entries.

----

## [0] Open Compound Domain Adaptation with Object Style Compensation for Semantic Segmentation

**Authors**: *Tingliang Feng, Hao Shi, Xueyang Liu, Wei Feng, Liang Wan, Yanlin Zhou, Di Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c74a3a6f44a44b204e26b1a6d7fe4a66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c74a3a6f44a44b204e26b1a6d7fe4a66-Abstract-Conference.html)

**Abstract**:

Many methods of semantic image segmentation have borrowed the success of open compound domain adaptation. They minimize the style gap between the images of source and target domains, more easily predicting the accurate pseudo annotations for target domain's images that train segmentation network. The existing methods globally adapt the scene style of the images, whereas the object styles of different categories or instances are adapted improperly. This paper proposes the Object Style Compensation, where we construct the Object-Level Discrepancy Memory with multiple sets of discrepancy features. The discrepancy features in a set capture the style changes of the same category's object instances adapted from target to source domains. We learn the discrepancy features from the images of source and target domains, storing the discrepancy features in memory. With this memory, we select appropriate discrepancy features for compensating the style information of the object instances of various categories, adapting the object styles to a unified style of source domain. Our method enables a more accurate computation of the pseudo annotations for target domain's images, thus yielding state-of-the-art results on different datasets.

----

## [0] Going beyond persistent homology using persistent homology

**Authors**: *Johanna Immonen, Amauri H. Souza, Vikas Garg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c78f81a878a72566422f37279bca0fd0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c78f81a878a72566422f37279bca0fd0-Abstract-Conference.html)

**Abstract**:

Representational limits of message-passing graph neural networks (MP-GNNs), e.g., in terms of the Weisfeiler-Leman (WL) test for isomorphism, are well understood. Augmenting these graph models with topological features via persistent homology (PH) has gained prominence, but identifying the class of attributed graphs that PH can recognize remains open.  We introduce a novel concept of color-separating sets to provide a complete resolution to this important problem. Specifically, we establish the necessary and sufficient conditions for distinguishing graphs based on the persistence of their connected components, obtained from filter functions on vertex and edge colors. Our constructions expose the limits of vertex- and edge-level PH, proving that neither category subsumes the other. Leveraging these theoretical insights, we propose RePHINE for learning topological features on graphs. RePHINE efficiently combines vertex- and edge-level PH, achieving a scheme that is provably more powerful than both. Integrating RePHINE into MP-GNNs boosts their expressive power, resulting in gains over standard PH on several benchmarks for graph classification.

----

## [0] Explore to Generalize in Zero-Shot RL

**Authors**: *Ev Zisselman, Itai Lavie, Daniel Soudry, Aviv Tamar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c793577b644268259b1416464a6cdb8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c793577b644268259b1416464a6cdb8c-Abstract-Conference.html)

**Abstract**:

We study zero-shot generalization in reinforcement learning - optimizing a policy on a set of training tasks to perform well on a similar but unseen test task. To mitigate overfitting, previous work explored different notions of invariance to the task. However, on problems such as the ProcGen Maze, an adequate solution that is invariant to the task visualization does not exist, and therefore invariance-based approaches fail. Our insight is that learning a policy that effectively $\textit{explores}$ the domain is harder to memorize than a policy that maximizes reward for a specific task, and therefore we expect such learned behavior to generalize well; we indeed demonstrate this empirically on several domains that are difficult for invariance-based approaches. Our $\textit{Explore to Generalize}$ algorithm (ExpGen) builds on this insight: we train an additional ensemble of agents that optimize reward. At test time, either the ensemble agrees on an action, and we generalize well, or we take exploratory actions, which generalize well and drive us to a novel part of the state space, where the ensemble may potentially agree again. We show that our approach is the state-of-the-art on tasks of the ProcGen challenge that have thus far eluded effective generalization, yielding a success rate of 83% on the Maze task and 74% on Heist with $200$ training levels. ExpGen can also be combined with an invariance based approach to gain the best of both worlds, setting new state-of-the-art results on ProcGen.Code available at [https://github.com/EvZissel/expgen](https://github.com/EvZissel/expgen).

----

## [0] CoLLAT: On Adding Fine-grained Audio Understanding to Language Models using Token-Level Locked-Language Tuning

**Authors**: *Dadallage A. R. Silva, Spencer Whitehead, Christopher T. Lengerich, Hugh Leather*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html)

**Abstract**:

Humans can easily understand various audio concepts, but conventional audio classification models fail due to their inability to predict unseen classes during training. To address this challenge, recent literature has explored contrastive language-audio pretraining to learn an audio understanding model using natural language supervision from a pretrained language model. However, despite their reasonable zero-shot performance in audio understanding, these models typically fail to achieve optimal performance while preserving the text understanding capabilities of the pretrained language model. They also perform poorly when comprehending audio clips with multiple audio concepts. To bridge these gaps, we propose $CoLLAT$: $Co$ntrastive $L$ocked $L$anguage and $A$udio $T$uning. This is a framework to effectively learn an audio understanding model with a locked language model, which is learned using a novel pretraining objective for audio-to-text grounding to yield fine-grained audio understanding. Our extensive experiments, which include several downstream applications such as audio classification, cross-modal retrieval, and audio-guided image generation, demonstrate that $CoLLAT$ yields state-of-the-art performance for audio understanding. Additionally, it unlocks audio guidance to applications built on top of pretrained language models.

----

## [0] Abide by the law and follow the flow: conservation laws for gradient flows

**Authors**: *Sibylle Marcotte, Rémi Gribonval, Gabriel Peyré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7bee9b76be21146fd592fc2b46614d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7bee9b76be21146fd592fc2b46614d5-Abstract-Conference.html)

**Abstract**:

Understanding the geometric properties of gradient descent dynamics is a key ingredient in deciphering the recent success of very large machine learning models. A striking observation is that trained over-parameterized models retain some properties of the optimization initialization. This "implicit bias" is believed to be responsible for some favorable properties of the trained models and could explain their good generalization properties. The purpose of this article is threefold. First, we rigorously expose the definition and basic properties of "conservation laws", that define quantities conserved during gradient flows of a given model (e.g. of a ReLU network with a given architecture) with any training data and any loss. Then we explain how to find the maximal number of independent conservation lawsby performing finite-dimensional algebraic manipulations on the Lie algebra generated by the Jacobian of the model. Finally, we provide algorithms to: a) compute a family of polynomial laws; b) compute the maximal number of (not necessarily polynomial) independent conservation laws. We provide showcase examples that we fully work out theoretically. Besides, applying the two algorithms confirms for a number of ReLU network architectures that all known laws are recovered by the algorithm, and that there are no other independent laws. Such computational tools pave the way to understanding desirable properties of optimization initialization in large machine learning models.

----

## [0] Breadcrumbs to the Goal: Supervised Goal Selection from Human-in-the-Loop Feedback

**Authors**: *Marcel Torne Villasevil, Max Balsells, Zihan Wang, Samedh Desai, Tao Chen, Pulkit Agrawal, Abhishek Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7c7cf10082e454b9662a686ce6f1b6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7c7cf10082e454b9662a686ce6f1b6f-Abstract-Conference.html)

**Abstract**:

Exploration and reward specification are fundamental and intertwined challenges for reinforcement learning. Solving sequential decision making tasks with a non-trivial element of exploration requires either specifying carefully designed reward functions or relying on indiscriminate, novelty seeking exploration bonuses. Human supervisors can provide effective guidance in the loop to direct the exploration process, but prior methods to leverage this guidance require constant synchronous high-quality human feedback, which is expensive and impractical to obtain. In this work, we propose a technique - Human Guided Exploration (HUGE), that is able to leverage low-quality feedback from non-expert users, which is infrequent, asynchronous and noisy, to guide exploration for reinforcement learning, without requiring careful reward specification. The key idea is to separate the challenges of directed exploration and policy learning - human feedback is used to direct exploration, while self-supervised policy learning is used to independently learn unbiased behaviors from the collected data. We show that this procedure can leverage noisy, asynchronous human feedback to learn tasks with no hand-crafted reward design or exploration bonuses. We show that HUGE is able to learn a variety of challenging multi-stage robotic navigation and manipulation tasks in simulation using crowdsourced feedback from non-expert users. Moreover, this paradigm can be scaled to learning directly on real-world robots.

----

## [0] Embroid: Unsupervised Prediction Smoothing Can Improve Few-Shot Classification

**Authors**: *Neel Guha, Mayee F. Chen, Kush Bhatia, Azalia Mirhoseini, Frederic Sala, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7f35864fef057d6fa315afa0275b3ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7f35864fef057d6fa315afa0275b3ad-Abstract-Conference.html)

**Abstract**:

Recent work has shown that language models' (LMs) prompt-based learning capabilities make them well suited for automating data labeling in domains where manual annotation is expensive. The challenge is that while writing an initial prompt is cheap, improving a prompt is costly---practitioners often require significant labeled data in order to evaluate the impact of prompt modifications. Our work asks whether it is possible to improve prompt-based learning without additional labeled data. We approach this problem by attempting to modify the predictions of a prompt, rather than the prompt itself. Our intuition is that accurate predictions should also be consistent: samples which are similar under some feature representation should receive the same prompt prediction. We propose Embroid, a method which computes multiple representations of a dataset under different embedding functions, and uses the consistency between the LM predictions for neighboring samples to identify mispredictions. Embroid then uses these neighborhoods to create additional predictions for each sample, and combines these predictions with a simple latent variable graphical model in order to generate a final corrected prediction. In addition to providing a theoretical analysis of Embroid, we conduct a rigorous empirical evaluation across six different LMs and up to 95 different tasks. We find that (1) Embroid substantially improves performance over original prompts (e.g., by an average of 7.3 points on GPT-JT), (2) also realizes improvements for more sophisticated prompting strategies (e.g., chain-of-thought), and (3) can be specialized to domains like law through the embedding functions.

----

## [0] Perceptual Kalman Filters: Online State Estimation under a Perfect Perceptual-Quality Constraint

**Authors**: *Dror Freirich, Tomer Michaeli, Ron Meir*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c7f43ada17acc234f568dc66da527418-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c7f43ada17acc234f568dc66da527418-Abstract-Conference.html)

**Abstract**:

Many practical settings call for the reconstruction of temporal signals from corrupted or missing data. Classic examples include decoding, tracking, signal enhancement and denoising. Since the reconstructed signals are ultimately viewed by humans, it is desirable to achieve reconstructions that are pleasing to human perception.Mathematically, perfect perceptual-quality is achieved when the distribution of restored signals is the same as that of natural signals, a requirement which has been heavily researched in static estimation settings (i.e. when a whole signal is processed at once). Here, we study the problem of optimal causal  filtering under a perfect perceptual-quality constraint, which is a task of fundamentally different nature.  Specifically, we analyze a Gaussian Markov signal observed through a linear noisy transformation. In the absence of perceptual constraints, the Kalman filter is known to be optimal in the MSE sense for this setting. Here, we show that adding the perfect perceptual quality constraint (i.e. the requirement of temporal consistency), introduces a fundamental dilemma whereby the filter may have to ``knowingly'' ignore new information revealed by the observations in order to conform to its past decisions. This often comes at the cost of a significant increase in the MSE (beyond that encountered in static settings). Our analysis goes beyond the classic innovation process of the Kalman filter, and introduces the novel concept of an unutilized information process. Using this tool, we present a recursive formula for perceptual filters, and demonstrate the qualitative effects of perfect perceptual-quality estimation on a video reconstruction problem.

----

## [0] SEENN: Towards Temporal Spiking Early Exit Neural Networks

**Authors**: *Yuhang Li, Tamar Geller, Youngeun Kim, Priyadarshini Panda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c801e68207da477bbc44182b9fac1129-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c801e68207da477bbc44182b9fac1129-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) have recently become more popular as a biologically plausible substitute for traditional Artificial Neural Networks (ANNs). SNNs are cost-efficient and deployment-friendly because they process input in both spatial and temporal manner using binary spikes. However, we observe that the information capacity in SNNs is affected by the number of timesteps, leading to an accuracy-efficiency tradeoff. In this work, we study a fine-grained adjustment of the number of timesteps in SNNs. Specifically, we treat the number of timesteps as a variable conditioned on different input samples to reduce redundant timesteps for certain data. We call our method Spiking Early-Exit Neural Networks (SEENNs). To determine the appropriate number of timesteps, we propose SEENN-I which uses a confidence score thresholding to filter out the uncertain predictions, and SEENN-II which determines the number of timesteps by reinforcement learning. Moreover, we demonstrate that SEENN is compatible with both the directly trained SNN and the ANN-SNN conversion. By dynamically adjusting the number of timesteps, our SEENN achieves a remarkable reduction in the average number of timesteps during inference. For example, our SEENN-II ResNet-19 can achieve 96.1\% accuracy with an average of 1.08 timesteps on the CIFAR-10 test dataset. Code is shared at https://github.com/Intelligent-Computing-Lab-Yale/SEENN.

----

## [0] Distributionally Robust Skeleton Learning of Discrete Bayesian Networks

**Authors**: *Yeshu Li, Brian D. Ziebart*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c80addda8bcd95339921cba7581ac7bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c80addda8bcd95339921cba7581ac7bd-Abstract-Conference.html)

**Abstract**:

We consider the problem of learning the exact skeleton of general discrete Bayesian networks from potentially corrupted data. Building on distributionally robust optimization and a regression approach, we propose to optimize the most adverse risk over a family of distributions within bounded Wasserstein distance or KL divergence to the empirical distribution. The worst-case risk accounts for the effect of outliers. The proposed approach applies for general categorical random variables without assuming faithfulness, an ordinal relationship or a specific form of conditional distribution. We present efficient algorithms and show the proposed methods are closely related to the standard regularized regression approach. Under mild assumptions, we derive non-asymptotic guarantees for successful structure learning with logarithmic sample complexities for bounded-degree graphs. Numerical study on synthetic and real datasets validates the effectiveness of our method.

----

## [0] Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification

**Authors**: *Kanishk Jain, Shyamgopal Karthik, Vineet Gandhi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c81690e2cfe63aede8519ad448f56d71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c81690e2cfe63aede8519ad448f56d71-Abstract-Conference.html)

**Abstract**:

We investigate the problem of reducing mistake severity for fine-grained classification. Fine-grained classification can be challenging, mainly due to the requirement of knowledge or domain expertise for accurate annotation. However, humans are particularly adept at performing coarse classification as it requires relatively low levels of expertise. To this end, we present a novel approach for Post-Hoc Correction called Hierarchical Ensembles (HiE) that utilizes label hierarchy to improve the performance of fine-grained classification at test-time using the coarse-grained predictions. By only requiring the parents of leaf nodes, our method significantly reduces avg. mistake severity while improving top-1 accuracy on the iNaturalist-19 and tieredImageNet-H datasets, achieving a new state-of-the-art on both benchmarks. We also investigate the efficacy of our approach in the semi-supervised setting. Our approach brings notable gains in top-1 accuracy while significantly decreasing the severity of mistakes as training data decreases for the fine-grained classes. The simplicity and post-hoc nature of HiE renders it practical to be used with any off-the-shelf trained model to improve its predictions further.

----

## [0] Robust Matrix Sensing in the Semi-Random Model

**Authors**: *Xing Gao, Yu Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c836d71b4702d9046b14ce1228c4c11b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c836d71b4702d9046b14ce1228c4c11b-Abstract-Conference.html)

**Abstract**:

Low-rank matrix recovery is a fundamental problem in machine learning with numerous applications. In practice, the problem can be solved by convex optimization namely nuclear norm minimization, or by non-convex optimization as it is well-known that for low-rank matrix problems like matrix sensing and matrix completion, all local optima of the natural non-convex objectives are also globally optimal under certain ideal assumptions.In this paper, we study new approaches for matrix sensing in a semi-random model where an adversary can add any number of arbitrary sensing matrices. More precisely, the problem is to recover a low-rank matrix $X^\star$ from linear measurements $b_i = \langle A_i, X^\star \rangle$, where an unknown subset of the sensing matrices satisfies the Restricted Isometry Property (RIP) and the rest of the $A_i$'s are chosen adversarially.It is known that in the semi-random model, existing non-convex objectives can have bad local optima. To fix this, we present a descent-style algorithm that provably recovers the ground-truth matrix $X^\star$. For the closely-related problem of semi-random matrix completion, prior work [CG18] showed that all bad local optima can be eliminated by reweighting the input data. However, the analogous approach for matrix sensing requires reweighting a set of matrices to satisfy RIP, which is a condition that is NP-hard to check. Instead, we build on the framework proposed in [KLL$^+$23] for semi-random sparse linear regression, where the algorithm in each iteration reweights the input based on the current solution, and then takes a weighted gradient step that is guaranteed to work well locally. Our analysis crucially exploits the connection between sparsity in vector problems and low-rankness in matrix problems, which may have other applications in obtaining robust algorithms for sparse and low-rank problems.

----

## [0] Implicit variance regularization in non-contrastive SSL

**Authors**: *Manu Srinath Halvagal, Axel Laborieux, Friedemann Zenke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c837ab3eebe77bffac634939f22ac458-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c837ab3eebe77bffac634939f22ac458-Abstract-Conference.html)

**Abstract**:

Non-contrastive SSL methods like BYOL and SimSiam rely on asymmetric predictor networks to avoid representational collapse without negative samples. Yet, how predictor networks facilitate stable learning is not fully understood. While previous theoretical analyses assumed Euclidean losses, most practical implementations rely on cosine similarity. To gain further theoretical insight into non-contrastive SSL, we analytically study learning dynamics in conjunction with Euclidean and cosine similarity in the eigenspace of closed-form linear predictor networks. We show that both avoid collapse through implicit variance regularization albeit through different dynamical mechanisms. Moreover, we find that the eigenvalues act as effective learning rate multipliers and propose a family of isotropic loss functions (IsoLoss) that equalize convergence rates across eigenmodes. Empirically, IsoLoss speeds up the initial learning dynamics and increases robustness, thereby allowing us to dispense with the EMA target network typically used with non-contrastive methods. Our analysis sheds light on the variance regularization mechanisms of non-contrastive SSL and lays the theoretical grounds for crafting novel loss functions that shape the learning dynamics of the predictor's spectrum.

----

## [0] ATMAN: Understanding Transformer Predictions Through Memory Efficient Attention Manipulation

**Authors**: *Björn Deiseroth, Mayukh Deb, Samuel Weinbach, Manuel Brack, Patrick Schramowski, Kristian Kersting*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c83bc020a020cdeb966ed10804619664-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c83bc020a020cdeb966ed10804619664-Abstract-Conference.html)

**Abstract**:

Generative transformer models have become increasingly complex, with large numbers of parameters and the ability to process multiple input modalities. Current methods for explaining their predictions are resource-intensive. Most crucially, they require prohibitively large amounts of additional memory, since they rely on backpropagation which allocates almost twice as much GPU memory as the forward pass. This makes it difficult, if not impossible, to use explanations in production. We present AtMan that provides explanations of generative transformer models at almost no extra cost. Specifically, AtMan is a modality-agnostic perturbation method that manipulates the attention mechanisms of transformers to produce relevance maps for the input with respect to the output prediction. Instead of using backpropagation, AtMan applies a parallelizable token-based search method relying on cosine similarity neighborhood in the embedding space. Our exhaustive experiments on text and image-text benchmarks demonstrate that AtMan outperforms current state-of-the-art gradient-based methods on several metrics while being computationally efficient. As such, AtMan is suitable for use in large model inference deployments.

----

## [0] Wasserstein Quantum Monte Carlo: A Novel Approach for Solving the Quantum Many-Body Schrödinger Equation

**Authors**: *Kirill Neklyudov, Jannes Nys, Luca Thiede, Juan Carrasquilla, Qiang Liu, Max Welling, Alireza Makhzani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c8450235f227f136242f774b2799581f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c8450235f227f136242f774b2799581f-Abstract-Conference.html)

**Abstract**:

Solving the quantum many-body Schrödinger equation is a fundamental and challenging problem in the fields of quantum physics, quantum chemistry, and material sciences. One of the common computational approaches to this problem is Quantum Variational Monte Carlo (QVMC), in which ground-state solutions are obtained by minimizing the energy of the system within a restricted family of parameterized wave functions. Deep learning methods partially address the limitations of traditional QVMC by representing a rich family of wave functions in terms of neural networks. However, the optimization objective in QVMC remains notoriously hard to minimize and requires second-order optimization methods such as natural gradient. In this paper, we first reformulate energy functional minimization in the space of Born distributions corresponding to particle-permutation (anti-)symmetric wave functions, rather than the space of wave functions. We then interpret QVMC as the Fisher--Rao gradient flow in this distributional space, followed by a projection step onto the variational manifold. This perspective provides us with a principled framework to derive new QMC algorithms, by endowing the distributional space with better metrics, and following the projected gradient flow induced by those metrics. More specifically, we propose "Wasserstein Quantum Monte Carlo" (WQMC), which uses the gradient flow induced by the Wasserstein metric, rather than the Fisher--Rao metric, and corresponds to transporting the probability mass, rather than teleporting it. We demonstrate empirically that the dynamics of WQMC results in faster convergence to the ground state of molecular systems.

----

## [0] Textually Pretrained Speech Language Models

**Authors**: *Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis Conneau, Felix Kreuk, Jade Copet, Alexandre Défossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, Yossi Adi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c859b99b5d717c9035e79d43dfd69435-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c859b99b5d717c9035e79d43dfd69435-Abstract-Conference.html)

**Abstract**:

Speech language models (SpeechLMs) process and generate acoustic data only, without textual supervision. In this work, we propose TWIST, a method for training SpeechLMs using a warm-start from a pretrained textual language models. We show using both automatic and human evaluations that TWIST outperforms a cold-start SpeechLM across the board. We empirically analyze the effect of different model design choices such as the speech tokenizer, the pretrained textual model, and the dataset size. We find that model and dataset scale both play an important role in constructing better-performing SpeechLMs. Based on our observations, we present the largest (to the best of our knowledge) SpeechLM both in terms of number of parameters and training data. We additionally introduce two spoken versions of the StoryCloze textual benchmark to further improve model evaluation and advance future research in the field. We make speech samples, code and models publicly available.

----

## [0] Riemannian Residual Neural Networks

**Authors**: *Isay Katsman, Eric Chen, Sidhanth Holalkere, Anna Asch, Aaron Lou, Ser Nam Lim, Christopher De Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c868aa7437dc9b29e674cd2e25689021-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c868aa7437dc9b29e674cd2e25689021-Abstract-Conference.html)

**Abstract**:

Recent methods in geometric deep learning have introduced various neural networks to operate over data that lie on Riemannian manifolds. Such networks are often necessary to learn well over graphs with a hierarchical structure or to learn over manifold-valued data encountered in the natural sciences. These networks are often inspired by and directly generalize standard Euclidean neural networks. However, extending Euclidean networks is difficult and has only been done for a select few manifolds. In this work, we examine the residual neural network (ResNet) and show how to extend this construction to general Riemannian manifolds in a geometrically principled manner. Originally introduced to help solve the vanishing gradient problem, ResNets have become ubiquitous in machine learning due to their beneficial learning properties, excellent empirical results, and easy-to-incorporate nature when building varied neural networks. We find that our Riemannian ResNets mirror these desirable properties: when compared to existing manifold neural networks designed to learn over hyperbolic space and the manifold of symmetric positive definite matrices, we outperform both kinds of networks in terms of relevant testing metrics and training dynamics.

----

## [0] Aligning Gradient and Hessian for Neural Signed Distance Function

**Authors**: *Ruian Wang, Zixiong Wang, Yunxiao Zhang, Shuang-Min Chen, Shiqing Xin, Changhe Tu, Wenping Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c87bd5843849884e9430f1693b018d71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c87bd5843849884e9430f1693b018d71-Abstract-Conference.html)

**Abstract**:

The Signed Distance Function (SDF), as an implicit surface representation, provides a crucial method for reconstructing a watertight surface from unorganized point clouds. The SDF has a fundamental relationship with the principles of surface vector calculus. Given a smooth surface, there exists a thin-shell space in which the SDF is differentiable everywhere such that the gradient of the SDF is an eigenvector of its Hessian matrix, with a corresponding eigenvalue of zero. In this paper, we introduce a method to directly learn the SDF from point clouds in the absence of normals. Our motivation is grounded in a fundamental observation: aligning the gradient and the Hessian of the SDF provides a more efficient mechanism to govern gradient directions. This, in turn, ensures that gradient changes more accurately reflect the true underlying variations in shape. Extensive experimental results demonstrate its ability to accurately recover the underlying shape while effectively suppressing the presence of ghost geometry.

----

## [0] Achieving Cross Modal Generalization with Multimodal Unified Representation

**Authors**: *Yan Xia, Hai Huang, Jieming Zhu, Zhou Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c89f09849eb5af489abb122394ff0f0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c89f09849eb5af489abb122394ff0f0b-Abstract-Conference.html)

**Abstract**:

This paper introduces a novel task called Cross Modal Generalization (CMG), which addresses the challenge of learning a unified discrete representation from paired multimodal data during pre-training. Then in downstream tasks, the model can achieve zero-shot generalization ability in other modalities when only one modal is labeled. Existing approaches in multimodal representation learning focus more on coarse-grained alignment or rely on the assumption that  information from different modalities is completely aligned, which is impractical in real-world scenarios. To overcome this limitation, we propose \textbf{Uni-Code}, which contains two key contributions: the Dual Cross-modal Information Disentangling (DCID) module and the Multi-Modal Exponential Moving Average (MM-EMA). These methods facilitate bidirectional supervision between modalities and align semantically equivalent information in a shared discrete latent space, enabling fine-grained unified representation of multimodal sequences. During pre-training, we investigate various modality combinations, including audio-visual, audio-text, and the tri-modal combination of audio-visual-text. Extensive experiments on various downstream tasks, i.e., cross-modal event classification, localization, cross-modal retrieval, query-based video segmentation, and cross-dataset event localization, demonstrate the effectiveness of our proposed methods. The code is available at https://github.com/haihuangcode/CMG.

----

## [0] Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training

**Authors**: *Yefan Zhou, Tianyu Pang, Keqin Liu, Charles H. Martin, Michael W. Mahoney, Yaoqing Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c8a4dd7d9e13583d714ce8580da7bbc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c8a4dd7d9e13583d714ce8580da7bbc7-Abstract-Conference.html)

**Abstract**:

Regularization in modern machine learning is crucial, and it can take various forms in algorithmic design: training set, model family, error function, regularization terms, and optimizations. In particular, the learning rate, which can be interpreted as a temperature-like parameter within the statistical mechanics of learning, plays a crucial role in neural network training. Indeed, many widely adopted training strategies basically just define the decay of the learning rate over time. This process can be interpreted as decreasing a temperature, using either a global learning rate (for the entire model) or a learning rate that varies for each parameter. This paper proposes TempBalance, a straightforward yet effective layer-wise learning rate method. TempBalance is based on Heavy-Tailed Self-Regularization (HT-SR) Theory, an approach which characterizes the implicit self-regularization of different layers in trained models. We demonstrate the efficacy of using HT-SR-motivated metrics to guide the scheduling and balancing of temperature across all network layers during model training, resulting in improved performance during testing. We implement TempBalance on CIFAR10, CIFAR100, SVHN, and TinyImageNet datasets using ResNets, VGGs and WideResNets with various depths and widths. Our results show that TempBalance significantly outperforms ordinary SGD and carefully-tuned spectral norm regularization. We also show that TempBalance outperforms a number of state-of-the-art optimizers and learning rate schedulers.

----

## [0] Gaussian Process Probes (GPP) for Uncertainty-Aware Probing

**Authors**: *Zi Wang, Alexander Ku, Jason Baldridge, Tom Griffiths, Been Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c8b100b376a7b338c84801b699935098-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c8b100b376a7b338c84801b699935098-Abstract-Conference.html)

**Abstract**:

Understanding which concepts models can and cannot represent has been fundamental to many tasks: from effective and responsible use of models to detecting out of distribution data. We introduce Gaussian process probes (GPP), a unified and simple framework for probing and measuring uncertainty about concepts represented by models. As a Bayesian extension of linear probing methods, GPP asks what kind of distribution over classifiers (of concepts) is induced by the model. This distribution can be used to measure both what the model represents and how confident the probe is about what the model represents.  GPP can be applied to any pre-trained  model with vector representations of inputs (e.g., activations). It does not require access to training data, gradients, or the architecture. We validate GPP on datasets containing both synthetic and real images. Our experiments show it can (1) probe a model's representations of concepts even with a very small number of examples, (2) accurately measure both epistemic uncertainty (how confident the probe is) and aleatory uncertainty (how fuzzy the concepts are to the model), and (3) detect out of distribution data using those uncertainty measures as well as classic methods do. By using Gaussian processes to expand what probing can offer, GPP provides a data-efficient, versatile and uncertainty-aware tool for understanding and evaluating the capabilities of machine learning models.

----

## [0] Inferring Hybrid Neural Fluid Fields from Videos

**Authors**: *Hong-Xing Yu, Yang Zheng, Yuan Gao, Yitong Deng, Bo Zhu, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c8e1620b29d546c2999a9339ab29aa82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c8e1620b29d546c2999a9339ab29aa82-Abstract-Conference.html)

**Abstract**:

We study recovering fluid density and velocity from sparse multiview videos. Existing neural dynamic reconstruction methods predominantly rely on optical flows; therefore, they cannot accurately estimate the density and uncover the underlying velocity due to the inherent visual ambiguities of fluid velocity, as fluids are often shapeless and lack stable visual features. The challenge is further pronounced by the turbulent nature of fluid flows, which calls for properly designed fluid velocity representations. To address these challenges, we propose hybrid neural fluid fields (HyFluid), a neural approach to jointly infer fluid density and velocity fields. Specifically, to deal with visual ambiguities of fluid velocity, we introduce a set of physics-based losses that enforce inferring a physically plausible velocity field, which is divergence-free and drives the transport of density. To deal with the turbulent nature of fluid velocity, we design a hybrid neural velocity representation that includes a base neural velocity field that captures most irrotational energy and a vortex particle-based velocity that models residual turbulent velocity. We show that our method enables recovering vortical flow details. Our approach opens up possibilities for various learning and reconstruction applications centered around 3D incompressible flow, including fluid re-simulation and editing, future prediction, and neural dynamic scene composition. Project website: https://kovenyu.com/HyFluid/

----

## [0] MeGraph: Capturing Long-Range Interactions by Alternating Local and Hierarchical Aggregation on Multi-Scaled Graph Hierarchy

**Authors**: *Honghua Dong, Jiawei Xu, Yu Yang, Rui Zhao, Shiwen Wu, Chun Yuan, Xiu Li, Chris J. Maddison, Lei Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9034f4f90fbfad5b80f47fe3dd6cf51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9034f4f90fbfad5b80f47fe3dd6cf51-Abstract-Conference.html)

**Abstract**:

Graph neural networks, which typically exchange information between local neighbors, often struggle to capture long-range interactions (LRIs) within the graph. Building a graph hierarchy via graph pooling methods is a promising approach to address this challenge; however, hierarchical information propagation cannot entirely take over the role of local information aggregation. To balance locality and hierarchy, we integrate the local and hierarchical structures, represented by intra- and inter-graphs respectively, of a multi-scale graph hierarchy into a single mega graph. Our proposed MeGraph model consists of multiple layers alternating between local and hierarchical information aggregation on the mega graph. Each layer first performs local-aware message-passing on graphs of varied scales via the intra-graph edges, then fuses information across the entire hierarchy along the bidirectional pathways formed by inter-graph edges. By repeating this fusion process, local and hierarchical information could intertwine and complement each other. To evaluate our model, we establish a new Graph Theory Benchmark designed to assess LRI capture ability, in which MeGraph demonstrates dominant performance. Furthermore, MeGraph exhibits superior or equivalent performance to state-of-the-art models on the Long Range Graph Benchmark. The experimental results on commonly adopted real-world datasets further demonstrate the broad applicability of MeGraph.

----

## [0] Double and Single Descent in Causal Inference with an Application to High-Dimensional Synthetic Control

**Authors**: *Jann Spiess, Guido Imbens, Amar Venugopal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c904c5d43d8a01177063977bd67bf6fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c904c5d43d8a01177063977bd67bf6fc-Abstract-Conference.html)

**Abstract**:

Motivated by a recent literature on the double-descent phenomenon in machine learning, we consider highly over-parameterized models in causal inference, including synthetic control with many control units. In such models, there may be so many free parameters that the model fits the training data perfectly. We first investigate high-dimensional linear regression for imputing wage data and estimating average treatment effects, where we find that models with many more covariates than sample size can outperform simple ones. We then document the performance of high-dimensional synthetic control estimators with many control units. We find that adding control units can help improve imputation performance even beyond the point where the pre-treatment fit is perfect. We provide a unified theoretical perspective on the performance of these high-dimensional models. Specifically, we show that more complex models can be interpreted as model-averaging estimators over simpler ones, which we link to an improvement in average performance. This perspective yields concrete insights into the use of synthetic control when control units are many relative to the number of pre-treatment periods.

----

## [0] IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers

**Authors**: *Zhenglin Huang, Xiaoan Bao, Na Zhang, Qingqi Zhang, Xiao Tu, Biao Wu, Xi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c917d8b9e01427f3184d80ade22f4d1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c917d8b9e01427f3184d80ade22f4d1f-Abstract-Conference.html)

**Abstract**:

Data augmentation has been proven effective for training high-accuracy convolutional neural network classifiers by preventing overfitting. However, building deep neural networks in real-world scenarios requires not only high accuracy on clean data but also robustness when data distributions shift. While prior methods have proposed that there is a trade-off between accuracy and robustness, we propose IPMix, a simple data augmentation approach to improve robustness without hurting clean accuracy. IPMix integrates three levels of data augmentation (image-level, patch-level, and pixel-level) into a coherent and label-preserving technique to increase the diversity of training data with limited computational overhead. To further improve the robustness, IPMix introduces structural complexity at different levels to generate more diverse images and adopts the random mixing method for multi-scale information fusion. Experiments demonstrate that IPMix outperforms state-of-the-art corruption robustness on CIFAR-C and ImageNet-C. In addition, we show that IPMix also significantly improves the other safety measures, including robustness to adversarial perturbations, calibration, prediction consistency, and anomaly detection, achieving state-of-the-art or comparable results on several benchmarks, including ImageNet-R, ImageNet-A, and ImageNet-O.

----

## [0] Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning

**Authors**: *Seungyong Moon, Junyoung Yeom, Bumsoo Park, Hyun Oh Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c919a2b5ec1de69f2629f9119676e336-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c919a2b5ec1de69f2629f9119676e336-Abstract-Conference.html)

**Abstract**:

Discovering achievements with a hierarchical structure in procedurally generated environments presents a significant challenge.This requires an agent to possess a broad range of abilities, including generalization and long-term reasoning. Many prior methods have been built upon model-based or hierarchical approaches, with the belief that an explicit module for long-term planning would be advantageous for learning hierarchical dependencies. However, these methods demand an excessive number of environment interactions or large model sizes, limiting their practicality. In this work, we demonstrate that proximal policy optimization (PPO), a simple yet versatile model-free algorithm, outperforms previous methods when optimized with recent implementation practices. Moreover, we find that the PPO agent can predict the next achievement to be unlocked to some extent, albeit with limited confidence. Based on this observation, we introduce a novel contrastive learning method, called achievement distillation, which strengthens the agent's ability to predict the next achievement. Our method exhibits a strong capacity for discovering hierarchical achievements and shows state-of-the-art performance on the challenging Crafter environment in a sample-efficient manner while utilizing fewer model parameters.

----

## [0] VisoGender: A dataset for benchmarking gender bias in image-text pronoun resolution

**Authors**: *Siobhan Mackenzie Hall, Fernanda Gonçalves Abrantes, Hanwen Zhu, Grace Sodunke, Aleksandar Shtedritski, Hannah Rose Kirk*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c93f26b1381b17693055a611a513f1e9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/c93f26b1381b17693055a611a513f1e9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce VisoGender, a novel dataset for benchmarking gender bias in vision-language models. We focus on occupation-related biases within a hegemonic system of binary gender, inspired by Winograd and Winogender schemas, where each image is associated with a caption containing a pronoun relationship of subjects and objects in the scene. VisoGender is balanced by gender representation in professional roles, supporting bias evaluation in two ways: i) resolution bias, where we evaluate the difference between pronoun resolution accuracies for image subjects with gender presentations perceived as masculine versus feminine by human annotators and ii) retrieval bias, where we compare ratios of professionals perceived to have masculine and feminine gender presentations retrieved for a gender-neutral search query. We benchmark several state-of-the-art vision-language models and find that they demonstrate bias in resolving binary gender in complex scenes. While the direction and magnitude of gender bias depends on the task and the model being evaluated, captioning models are generally less biased than Vision-Language Encoders.

----

## [0] Concept Distillation: Leveraging Human-Centered Explanations for Model Improvement

**Authors**: *Avani Gupta, Saurabh Saini, P. J. Narayanan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9450295fd667740a39a68148fc17f6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9450295fd667740a39a68148fc17f6e-Abstract-Conference.html)

**Abstract**:

Humans use abstract concepts for understanding instead of hard features. Recent interpretability research has focused on human-centered concept explanations of neural networks. Concept Activation Vectors (CAVs) estimate a model's sensitivity and possible biases to a given concept. We extend CAVs from post-hoc analysis to ante-hoc training to reduce model bias through fine-tuning using an additional Concept Loss. Concepts are defined on the final layer of the network in the past. We generalize it to intermediate layers, including the last convolution layer. We also introduce Concept Distillation, a method to define rich and effective concepts using a pre-trained knowledgeable model as the teacher. Our method can sensitize or desensitize a model towards concepts. We show applications of concept-sensitive training to debias several classification problems. We also show a way to induce prior knowledge into a reconstruction problem. We show that concept-sensitive training can improve model interpretability, reduce biases, and induce prior knowledge.

----

## [0] Mitigating the Effect of Incidental Correlations on Part-based Learning

**Authors**: *Gaurav Bhatt, Deepayan Das, Leonid Sigal, Vineeth N. Balasubramanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9493f7cb0d1ec4ae5fc6e0c1a5aca63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9493f7cb0d1ec4ae5fc6e0c1a5aca63-Abstract-Conference.html)

**Abstract**:

Intelligent systems possess a crucial characteristic of breaking complicated problems into smaller reusable components or parts and adjusting to new tasks using these part representations. However, current part-learners encounter difficulties in dealing with incidental correlations resulting from the limited observations of objects that may appear only in specific arrangements or with specific backgrounds. These incidental correlations may have a detrimental impact on the generalization and interpretability of learned part representations. This study asserts that part-based representations could be more interpretable and generalize better with limited data, employing two innovative regularization methods. The first regularization separates foreground and background information's generative process via a unique mixture-of-parts formulation. Structural constraints are imposed on the parts using a weakly-supervised loss, guaranteeing that the mixture-of-parts for foreground and background entails soft, object-agnostic masks. The second regularization assumes the form of a distillation loss, ensuring the invariance of the learned parts to the incidental background correlations. Furthermore, we incorporate sparse and orthogonal constraints to facilitate learning high-quality part representations.By reducing the impact of incidental background correlations on the learned parts, we exhibit state-of-the-art (SoTA) performance on few-shot learning tasks on benchmark datasets, including MiniImagenet, TieredImageNet, and FC100. We also demonstrate that the part-based representations acquired through our approach generalize better than existing techniques, even under domain shifts of the background and common data corruption on the ImageNet-9 dataset.

----

## [0] Towards In-context Scene Understanding

**Authors**: *Ivana Balazevic, David Steiner, Nikhil Parthasarathy, Relja Arandjelovic, Olivier J. Hénaff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c94a632545000531f0b47000e9caa5b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c94a632545000531f0b47000e9caa5b6-Abstract-Conference.html)

**Abstract**:

In-context learning––the ability to configure a model's behavior with different prompts––has revolutionized the field of natural language processing, alleviating the need for task-specific models and paving the way for generalist models capable of assisting with any query. Computer vision, in contrast, has largely stayed in the former regime: specialized decoders and finetuning protocols are generally required to perform dense tasks such as semantic segmentation and depth estimation. In this work we explore a simple mechanism for in-context learning of such scene understanding tasks: nearest neighbor retrieval from a prompt of annotated features. We propose a new pretraining protocol––leveraging attention within and across images––which yields representations particularly useful in this regime. The resulting Hummingbird model, suitably prompted, performs various scene understanding tasks without modification while approaching the performance of specialists that have been finetuned for each task. Moreover, Hummingbird can be configured to perform new tasks much more efficiently than finetuned models, raising the possibility of scene understanding in the interactive assistant regime.

----

## [0] Prediction and Control in Continual Reinforcement Learning

**Authors**: *Nishanth Anand, Doina Precup*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c94bbbef466ab1b2cfa100e41413b3a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c94bbbef466ab1b2cfa100e41413b3a8-Abstract-Conference.html)

**Abstract**:

Temporal difference (TD) learning is often used to update the estimate of the value function which is used by RL agents to extract useful policies. In this paper, we focus on value function estimation in continual reinforcement learning. We propose to decompose the value function into two components which update at different timescales: a permanent value function, which holds general knowledge that persists over time, and a transient value function, which allows quick adaptation to new situations. We establish theoretical results showing that our approach is well suited for continual learning and draw connections to the complementary learning systems (CLS) theory from neuroscience. Empirically, this approach improves performance significantly on both prediction and control problems.

----

## [0] EDGI: Equivariant Diffusion for Planning with Embodied Agents

**Authors**: *Johann Brehmer, Joey Bose, Pim de Haan, Taco S. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c95c049637c5c549c2a08e8d6dcbca4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c95c049637c5c549c2a08e8d6dcbca4b-Abstract-Conference.html)

**Abstract**:

Embodied agents operate in a structured world, often solving tasks with spatial, temporal, and permutation symmetries. Most algorithms for planning and model-based reinforcement learning (MBRL) do not take this rich geometric structure into account, leading to sample inefficiency and poor generalization. We introduce the Equivariant Diffuser for Generating Interactions (EDGI), an algorithm for MBRL and planning that is equivariant with respect to the product of the spatial symmetry group SE(3), the discrete-time translation group ℤ, and the object permutation group Sₙ. EDGI follows the Diffuser framework by Janner et al. (2022) in treating both learning a world model and planning in it as a conditional generative modeling problem, training a diffusion model on an offline trajectory dataset. We introduce a new SE(3) × ℤ × Sₙ-equivariant diffusion model that supports multiple representations. We integrate this model in a planning loop, where conditioning and classifier guidance let us softly break the symmetry for specific tasks as needed. On object manipulation and navigation tasks, EDGI is substantially more sample efficient and generalizes better across the symmetry group than non-equivariant models.

----

## [0] Topological RANSAC for instance verification and retrieval without fine-tuning

**Authors**: *Guoyuan An, Ju-hyeong Seon, Inkyu An, Yuchi Huo, Sung-eui Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c972859a984a21658432d7320c7df385-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c972859a984a21658432d7320c7df385-Abstract-Conference.html)

**Abstract**:

This paper presents an innovative approach to enhancing explainable image retrieval, particularly in situations where a fine-tuning set is unavailable. The widely-used SPatial verification (SP) method, despite its efficacy, relies on a spatial model and the hypothesis-testing strategy for instance recognition, leading to inherent limitations, including the assumption of planar structures and neglect of topological relations among features. To address these shortcomings, we introduce a pioneering technique that replaces the spatial model with a topological one within the RANSAC process. We propose bio-inspired saccade and fovea functions to verify the topological consistency among features, effectively circumventing the issues associated with SP's spatial model. Our experimental results demonstrate that our method significantly outperforms SP, achieving state-of-the-art performance in non-fine-tuning retrieval. Furthermore, our approach can enhance performance when used in conjunction with fine-tuned features. Importantly, our method retains high explainability and is lightweight, offering a practical and adaptable solution for a variety of real-world applications.

----

## [0] An Alternating Optimization Method for Bilevel Problems under the Polyak-Łojasiewicz Condition

**Authors**: *Quan Xiao, Songtao Lu, Tianyi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c981fd12b1d5703f19bd8289da9fc996-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c981fd12b1d5703f19bd8289da9fc996-Abstract-Conference.html)

**Abstract**:

Bilevel optimization has recently regained interest owing to its applications in emerging machine learning fields such as hyperparameter optimization, meta-learning, and reinforcement learning. Recent results have shown that simple alternating (implicit) gradient-based algorithms can match the convergence rate of single-level gradient descent (GD) when addressing bilevel problems with a strongly convex lower-level objective. However, it remains unclear whether this result can be generalized to bilevel problems beyond this basic setting. In this paper, we first introduce a stationary metric for the considered bilevel problems, which generalizes the existing metric, for a nonconvex lower-level objective that satisfies the Polyak-Łojasiewicz (PL) condition. We then propose a Generalized ALternating mEthod for bilevel opTimization (GALET) tailored to BLO with convex PL LL problem and establish that GALET achieves an $\epsilon$-stationary point for the considered problem within $\tilde{\cal O}(\epsilon^{-1})$ iterations, which matches the iteration complexity of GD for single-level smooth nonconvex problems.

----

## [0] Sub-optimality of the Naive Mean Field approximation for proportional high-dimensional Linear Regression

**Authors**: *Jiaze Qiu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9a7214961b9bd0ee93755bfa0abcea7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9a7214961b9bd0ee93755bfa0abcea7-Abstract-Conference.html)

**Abstract**:

The Na√Øve Mean Field (NMF) approximation is widely employed in modern Machine Learning due to the huge computational gains it bestows on the statistician. Despite its popularity in practice, theoretical guarantees for high-dimensional problems are only available under strong structural assumptions (e.g. sparsity). Moreover, existing theory often does not explain empirical observations noted in the existing literature.  In this paper, we take a step towards addressing these problems by deriving sharp asymptotic characterizations for the NMF approximation in high-dimensional linear regression. Our results apply to a wide class of natural priors and allow for model mismatch (i.e. the underlying statistical model can be different from the fitted model).  We work under an iid Gaussian design and the proportional asymptotic regime, where the number of features and number of observations grow at a proportional rate. As a consequence of our asymptotic characterization, we establish two concrete corollaries: (a) we establish the inaccuracy of the NMF approximation for the log-normalizing constant in this regime, and (b) we provide theoretical results backing the empirical observation that the NMF approximation can be overconfident in terms of uncertainty quantification.Our results utilize recent advances in the theory of Gaussian comparison inequalities. To the best of our knowledge, this is the first application of these ideas to the analysis of Bayesian variational inference problems. Our theoretical results are corroborated by numerical experiments. Lastly, we believe our results can be generalized to non-Gaussian designs and provide empirical evidence to support it.

----

## [0] The Gain from Ordering in Online Learning

**Authors**: *Vasilis Kontonis, Mingchen Ma, Christos Tzamos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9b1fe9c41f1eeec3a659154d575a282-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9b1fe9c41f1eeec3a659154d575a282-Abstract-Conference.html)

**Abstract**:

We study fixed-design online learning where the learner is allowed to choose the order of the datapoints in order to minimize their regret (aka self-directed online learning).  We focus on the fundamental task of online linear regression: the learner is given a dataset $X$ with $n$ examples in $d$ dimensions and at step $t$  they select a point $x_t \in X$, predict a value $\widetilde y_t$,  and suffer loss $(\widetilde y_t - w^\ast \cdot x_t)^2$.  The goal is to design algorithms that order the examples and achieve better regret  than random- or worst-order online algorithms.For an arbitrary dataset $X$, we show that, under the Exponential Time Hypothesis, no efficient algorithm can approximate the optimal (best-order)  regret  within a factor of $d^{1/\poly(\log \log d)}$.We then show that, for structured datasets, we can bypass the above hardness result and achieve nearly optimal regret. When the examples of $X$ are drawn i.i.d.\ from the uniform distribution on the sphere, we present an algorithm based on the greedy heuristic of  selecting ``easiest'' examples first that achieves a $\log d$-approximation of the optimal regret.

----

## [0] Variational Annealing on Graphs for Combinatorial Optimization

**Authors**: *Sebastian Sanokowski, Wilhelm Berghammer, Sepp Hochreiter, Sebastian Lehner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9c54ac0dd5e942b99b2b51c297544fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9c54ac0dd5e942b99b2b51c297544fd-Abstract-Conference.html)

**Abstract**:

Several recent unsupervised learning methods use probabilistic approaches to solve combinatorial optimization (CO) problems based on the assumption of statistically independent solution variables. We demonstrate that this assumption imposes performance limitations in particular on difficult problem instances. Our results corroborate that an autoregressive approach which captures statistical dependencies among solution variables yields superior performance on many popular CO problems. We introduce Subgraph Tokenization in which the configuration of a set of solution variables is represented by a single token. This tokenization technique alleviates the drawback of the long sequential sampling procedure which is inherent to autoregressive methods without sacrificing expressivity. Importantly, we theoretically motivate an annealed entropy regularization and show empirically that it is essential for efficient and stable learning.

----

## [0] Chasing Fairness Under Distribution Shift: A Model Weight Perturbation Approach

**Authors**: *Zhimeng Stephen Jiang, Xiaotian Han, Hongye Jin, Guanchu Wang, Rui Chen, Na Zou, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9cd2d12abe92f30b1442557bdbe8f5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9cd2d12abe92f30b1442557bdbe8f5a-Abstract-Conference.html)

**Abstract**:

Fairness in machine learning has attracted increasing attention in recent years. The fairness methods improving algorithmic fairness for in-distribution data may not perform well under distribution shifts. In this paper, we first theoretically demonstrate the inherent connection between distribution shift,  data perturbation, and model weight perturbation.Subsequently, we analyze the sufficient conditions to guarantee fairness (i.e., low demographic parity) for the target dataset, including fairness for the source dataset, and low prediction difference between the source and target datasets for each sensitive attribute group. Motivated by these sufficient conditions, we propose robust fairness regularization (RFR) by considering the worst case within the model weight perturbation ball for each sensitive attribute group. We evaluate the effectiveness of our proposed RFR algorithm on synthetic and real distribution shifts across various datasets. Experimental results demonstrate that RFR achieves better fairness-accuracy trade-off performance compared with several baselines. The source code is available at \url{https://github.com/zhimengj0326/RFR_NeurIPS23}.

----

## [0] Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow Shrink Trees

**Authors**: *Bryan Andrews, Joseph D. Ramsey, Ruben Sanchez-Romero, Jazmin Camchong, Erich Kummerfeld*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html)

**Abstract**:

Learning graphical conditional independence structures is an important machine learning problem and a cornerstone of causal discovery. However, the accuracy and execution time of learning algorithms generally struggle to scale to problems with hundreds of highly connected variables---for instance, recovering brain networks from fMRI data. We introduce the best order score search (BOSS) and grow-shrink trees (GSTs) for learning directed acyclic graphs (DAGs) in this paradigm. BOSS greedily searches over permutations of variables, using GSTs to construct and score DAGs from permutations. GSTs efficiently cache scores to eliminate redundant calculations. BOSS achieves state-of-the-art performance in accuracy and execution time, comparing favorably to a variety of combinatorial and gradient-based learning algorithms under a broad range of conditions. To demonstrate its practicality, we apply BOSS to two sets of resting-state fMRI data: simulated data with pseudo-empirical noise distributions derived from randomized empirical fMRI cortical signals and clinical data from 3T fMRI scans processed into cortical parcels. BOSS is available for use within the TETRAD project which includes Python and R wrappers.

----

## [0] Bayesian Active Causal Discovery with Multi-Fidelity Experiments

**Authors**: *Zeyu Zhang, Chaozhuo Li, Xu Chen, Xing Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9d9659d1d960b53e8121469ef1f2df5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9d9659d1d960b53e8121469ef1f2df5-Abstract-Conference.html)

**Abstract**:

This paper studies the problem of active causal discovery when the experiments can be done based on multi-fidelity oracles, where higher fidelity experiments are more precise and expensive, while the lower ones are cheaper but less accurate. In this paper, we formally define the task of multi-fidelity active causal discovery, and design a probabilistic model for solving this problem. In specific, we first introduce a mutual-information based acquisition function to determine which variable should be intervened at which fidelity, and then a cascading model is proposed to capture the correlations between different fidelity oracles. Beyond the above basic framework, we also extend it to the batch intervention scenario. We find that the theoretical foundations behind the widely used and efficient greedy method do not hold in our problem. To solve this problem, we introduce a new concept called $\epsilon$-submodular, and design a constraint based fidelity model to theoretically validate the greedy method. We conduct extensive experiments to demonstrate the effectiveness of our model.

----

## [0] Meta-Learning with Neural Bandit Scheduler

**Authors**: *Yunzhe Qi, Yikun Ban, Tianxin Wei, Jiaru Zou, Huaxiu Yao, Jingrui He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9e6ac15e689e06139d7b39e1667b165-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9e6ac15e689e06139d7b39e1667b165-Abstract-Conference.html)

**Abstract**:

Meta-learning has been proven an effective learning paradigm for training machine learning models with good generalization ability. Apart from the common practice of uniformly sampling the meta-training tasks, existing methods working on task scheduling strategies are mainly based on pre-defined sampling protocols or the assumed task-model correlations, and greedily make scheduling decisions, which can lead to sub-optimal performance bottlenecks of the meta-model. In this paper, we propose a novel task scheduling framework under Contextual Bandits settings, named BASS, which directly optimizes the task scheduling strategy based on the status of the meta-model. By balancing the exploitation and exploration in meta-learning task scheduling, BASS can help tackle the challenge of limited knowledge about the task distribution during the early stage of meta-training, while simultaneously exploring potential benefits for forthcoming meta-training iterations through an adaptive exploration strategy. Theoretical analysis and extensive experiments are presented to show the effectiveness of our proposed framework.

----

## [0] ClusterFomer: Clustering As A Universal Visual Learner

**Authors**: *James Liang, Yiming Cui, Qifan Wang, Tong Geng, Wenguan Wang, Dongfang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/c9ef471a579197c4ed99df2aa542ce97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/c9ef471a579197c4ed99df2aa542ce97-Abstract-Conference.html)

**Abstract**:

This paper presents ClusterFormer, a universal vision model that is based on the Clustering paradigm with TransFormer. It comprises two novel designs: 1) recurrent cross-attention clustering, which reformulates the cross-attention mechanism in Transformer and enables recursive updates of cluster centers to facilitate strong representation learning; and 2) feature dispatching, which uses the updated cluster centers to redistribute image features through similarity-based metrics, resulting in a transparent pipeline. This elegant design streamlines an explainable and transferable workflow, capable of tackling heterogeneous vision tasks (i.e., image classification, object detection, and image segmentation) with varying levels of clustering granularity (i.e., image-, box-, and pixel-level). Empirical results demonstrate that ClusterFormer outperforms various well-known specialized architectures, achieving 83.41% top-1 acc. over ImageNet-1K for image classification, 54.2% and 47.0% mAP over MS COCO for object detection and instance segmentation, 52.4% mIoU over ADE20K for semantic segmentation, and 55.8% PQ over COCO Panoptic for panoptic segmentation. This work aims to initiate a paradigm shift in universal visual understanding and to benefit the broader field.

----

## [0] Spike-driven Transformer

**Authors**: *Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, Guoqi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with four unique properties: (1) Event-driven, no calculation is triggered when the input of Transformer is zero; (2) Binary spike communication, all matrix multiplications associated with the spike matrix can be transformed into sparse additions; (3) Self-attention with linear complexity at both token and channel dimensions; (4) The operations between spike-form Query, Key, and Value are mask and addition. Together, there are only sparse addition operations in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to $87.2\times$ lower computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve 77.1\% top-1 accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field.

----

## [0] Dis-inhibitory neuronal circuits can control the sign of synaptic plasticity

**Authors**: *Julian Rossbroich, Friedemann Zenke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca22641c182b3b9608634edb4d09bc33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca22641c182b3b9608634edb4d09bc33-Abstract-Conference.html)

**Abstract**:

How neuronal circuits achieve credit assignment remains a central unsolved question in systems neuroscience. Various studies have suggested plausible solutions for back-propagating error signals through multi-layer networks. These purely functionally motivated models assume distinct neuronal compartments to represent local error signals that determine the sign of synaptic plasticity. However, this explicit error modulation is inconsistent with phenomenological plasticity models in which the sign depends primarily on postsynaptic activity. Here we show how a plausible microcircuit model and Hebbian learning rule derived within an adaptive control theory framework can resolve this discrepancy. Assuming errors are encoded in top-down dis-inhibitory synaptic afferents, we show that error-modulated learning emerges naturally at the circuit level when recurrent inhibition explicitly influences Hebbian plasticity. The same learning rule accounts for experimentally observed plasticity in the absence of inhibition and performs comparably to back-propagation of error (BP) on several non-linearly separable benchmarks. Our findings bridge the gap between functional and experimentally observed plasticity rules and make concrete predictions on inhibitory modulation of excitatory plasticity.

----

## [0] Accelerated Zeroth-order Method for Non-Smooth Stochastic Convex Optimization Problem with Infinite Variance

**Authors**: *Nikita Kornilov, Ohad Shamir, Aleksandr V. Lobanov, Darina Dvinskikh, Alexander V. Gasnikov, Innokentiy Shibaev, Eduard Gorbunov, Samuel Horváth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca24eb48806df3af49e5ac59d8a46f67-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca24eb48806df3af49e5ac59d8a46f67-Abstract-Conference.html)

**Abstract**:

In this paper, we consider non-smooth stochastic convex optimization with two function evaluations per round under infinite noise variance. In the classical setting when noise has finite variance, an optimal algorithm, built upon the batched accelerated gradient method, was proposed in (Gasnikov et. al., 2022). This optimality is defined in terms of iteration and oracle complexity, as well as the maximal admissible level of adversarial noise. However, the assumption of finite variance is burdensome and it might not hold in many practical scenarios. To address this, we demonstrate how to adapt a refined clipped version of the accelerated gradient (Stochastic Similar Triangles) method from (Sadiev et al., 2023) for a two-point zero-order oracle. This adaptation entails extending the batching technique to accommodate infinite variance — a non-trivial task that stands as a distinct contribution of this paper.

----



[Go to the previous page](NIPS-2023-list2700.md)

[Go to the next page](NIPS-2023-list2702.md)

[Go to the catalog section](README.md)