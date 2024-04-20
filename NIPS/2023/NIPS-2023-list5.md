## [0] Adaptive Topological Feature via Persistent Homology: Filtration Learning for Point Clouds

**Authors**: *Naoki Nishikawa, Yuichi Ike, Kenji Yamanishi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d49235669869ab737c1da9d64b7c769-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d49235669869ab737c1da9d64b7c769-Abstract-Conference.html)

**Abstract**:

Machine learning for point clouds has been attracting much attention, with many applications in various fields, such as shape recognition and material science. For enhancing the accuracy of such machine learning methods, it is often effective to incorporate global topological features, which are typically extracted by persistent homology. In the calculation of persistent homology for a point cloud, we choose a filtration for the point cloud, an increasing sequence of spaces. Since the performance of machine learning methods combined with persistent homology is highly affected by the choice of a filtration, we need to tune it depending on data and tasks. In this paper, we propose a framework that learns a filtration adaptively with the use of neural networks. In order to make the resulting persistent homology isometry-invariant, we develop a neural network architecture with such invariance. Additionally, we show a theoretical result on a finite-dimensional approximation of filtration functions, which justifies the proposed network architecture. Experimental results demonstrated the efficacy of our framework in several classification tasks.

----

## [0] Accurate Interpolation for Scattered Data through Hierarchical Residual Refinement

**Authors**: *Shizhe Ding, Boyang Xia, Dongbo Bu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5a92867cf463fad136cfa23395840b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5a92867cf463fad136cfa23395840b-Abstract-Conference.html)

**Abstract**:

Accurate interpolation algorithms are highly desired in various theoretical and engineering scenarios. Unlike the traditional numerical algorithms that have exact zero-residual constraints on observed points, the neural network-based interpolation methods exhibit non-zero residuals at these points. These residuals, which provide observations of an underlying residual function, can guide predicting interpolation functions, but have not been exploited by the existing approaches. To fill this gap, we propose Hierarchical INTerpolation Network (HINT), which utilizes the residuals on observed points to guide target function estimation in a hierarchical fashion. HINT consists of several sequentially arranged lightweight interpolation blocks. The first interpolation block estimates the main component of the target function, while subsequent blocks predict the residual components using observed points residuals of the preceding blocks. The main component and residual components are accumulated to form the final interpolation results. Furthermore, under the assumption that finer residual prediction requires a more focused attention range on observed points, we utilize hierarchical local constraints in correlation modeling between observed and target points. Extensive experiments demonstrate that HINT outperforms existing interpolation algorithms significantly in terms of interpolation accuracy across a wide variety of datasets, which underscores its potential for practical scenarios.

----

## [0] Learning Universal Policies via Text-Guided Video Generation

**Authors**: *Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html)

**Abstract**:

A goal of artificial intelligence is to construct an agent that can solve a wide variety of tasks. Recent progress in text-guided image synthesis has yielded models with an impressive ability to generate complex novel images, exhibiting combinatorial generalization across domains. Motivated by this success, we investigate whether such tools can be used to construct more general-purpose agents. Specifically, we cast the sequential decision making problem as a text-conditioned video generation problem, where, given a text-encoded specification of a desired goal, a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video. By leveraging text as the underlying goal specification, we are able to naturally and combinatorially generalize to novel goals. The proposed policy-as-video formulation can further represent environments with different state and action spaces in a unified space of images, which, for example, enables learning and generalization across a variety of robot manipulation tasks. Finally, by leveraging pretrained language embeddings and widely available videos from the internet, the approach enables knowledge transfer through predicting highly realistic video plans for real robots.

----

## [0] Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming

**Authors**: *Jacobus G. M. van der Linden, Mathijs de Weerdt, Emir Demirovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5fce9627e15c84db572a66e029b1fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5fce9627e15c84db572a66e029b1fc-Abstract-Conference.html)

**Abstract**:

Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue.Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees.We explore this relationship in detail and show the necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints.Experiments on five application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.

----

## [0] Polyhedron Attention Module: Learning Adaptive-order Interactions

**Authors**: *Tan Zhu, Fei Dou, Xinyu Wang, Jin Lu, Jinbo Bi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d83ad88759cef8192451543e5d59bf6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d83ad88759cef8192451543e5d59bf6-Abstract-Conference.html)

**Abstract**:

Learning feature interactions can be the key for multivariate predictive modeling. ReLU-activated neural networks create piecewise linear prediction models, and other nonlinear activation functions lead to models with only high-order feature interactions. Recent methods incorporate candidate polynomial terms of fixed orders into deep learning, which is subject to the issue of combinatorial explosion, or learn the orders that are difficult to adapt to different regions of the feature space. We propose a Polyhedron Attention Module (PAM) to create piecewise polynomial models where the input space is split into polyhedrons which define the different pieces and on each piece the hyperplanes that define the polyhedron boundary multiply to form the interactive terms, resulting in interactions of adaptive order to each piece. PAM is interpretable to identify important interactions in predicting a target. Theoretic analysis shows that PAM has stronger expression capability than ReLU-activated networks. Extensive experimental results demonstrate the superior classification performance of PAM on massive datasets of the click-through rate prediction and PAM can learn meaningful interaction effects in a medical problem.

----

## [0] Faster Query Times for Fully Dynamic k-Center Clustering with Outliers

**Authors**: *Leyla Biabani, Annika Hennes, Morteza Monemizadeh, Melanie Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d8e261c241aa72f9b4a02af7f52587e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d8e261c241aa72f9b4a02af7f52587e-Abstract-Conference.html)

**Abstract**:

Given a point set $P\subseteq M$ from a metric space $(M,d)$ and numbers $k, z \in N$, the *metric $k$-center problem with $z$ outliers* is to find a set $C^\ast\subseteq P$ of $k$ points such that the maximum distance of all but at most $z$ outlier points of $P$ to their nearest center in ${C}^\ast$ is minimized. We consider this problem in the fully dynamic model, i.e., under insertions and deletions of points, for the case that the metric space has a bounded doubling dimension $dim$. We utilize a hierarchical data structure to maintain the points and their neighborhoods, which enables us to efficiently find the clusters. In particular, our data structure can be queried at any time to generate a $(3+\varepsilon)$-approximate solution for input values of $k$ and $z$ in worst-case query time $\varepsilon^{-O(dim)}k \log{n} \log\log{\Delta}$, where $\Delta$ is the ratio between the maximum and minimum distance between two points in $P$. Moreover, it allows insertion/deletion of a point in worst-case update time $\varepsilon^{-O(dim)}\log{n}\log{\Delta}$. Our result achieves a significantly faster query time with respect to $k$ and $z$ than the current state-of-the-art by Pellizzoni, Pietracaprina, and Pucci, which uses $\varepsilon^{-O(dim)}(k+z)^2\log{\Delta}$ query time to obtain a $(3+\varepsilon)$-approximation.

----

## [0] Natural Language Instruction-following with Task-related Language Development and Translation

**Authors**: *Jing-Cheng Pang, Xinyu Yang, Si-Hang Yang, Xiong-Hui Chen, Yang Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dc2fe8d9ae956616f86bab3ce5edc59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dc2fe8d9ae956616f86bab3ce5edc59-Abstract-Conference.html)

**Abstract**:

Natural language-conditioned reinforcement learning (RL) enables agents to follow human instructions. Previous approaches generally implemented language-conditioned RL by providing the policy with human instructions in natural language (NL) and training the policy to follow instructions. In this is outside-in approach, the policy must comprehend the NL and manage the task simultaneously. However, the unbounded NL examples often bring much extra complexity for solving concrete RL tasks, which can distract policy learning from completing the task. To ease the learning burden of the policy, we investigate an inside-out scheme for natural language-conditioned RL by developing a task language (TL) that is task-related and easily understood by the policy, thus reducing the policy learning burden. Besides, we employ a translator to translate natural language into the TL, which is used in RL to achieve efficient policy training. We implement this scheme as TALAR (TAsk Language with predicAte Representation) that learns multiple predicates to model object relationships as the TL. Experiments indicate that TALAR not only better comprehends NL instructions but also leads to a better instruction-following policy that significantly improves the success rate over baselines and adapts to unseen expressions of NL instruction. Besides, the TL is also an effective sub-task abstraction compatible with hierarchical RL.

----

## [0] Convergence of Actor-Critic with Multi-Layer Neural Networks

**Authors**: *Haoxing Tian, Alex Olshevsky, Yannis Paschalidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html)

**Abstract**:

The early theory of actor-critic methods considered convergence using linear function approximators for the policy and value functions. Recent work has established convergence using neural network approximators with a single hidden layer. In this work we are taking the natural next step and establish convergence using deep neural networks with an arbitrary number of hidden layers, thus closing a gap between theory and practice. We show that actor-critic updates projected on a ball around the initial condition will converge to a neighborhood where the average of the squared gradients is $\tilde{O} \left( 1/\sqrt{m} \right) + O \left( \epsilon \right)$, with $m$ being the width of the neural network and $\epsilon$ the approximation quality of the best critic neural network over the projected set.

----

## [0] Percentile Criterion Optimization in Offline Reinforcement Learning

**Authors**: *Cyrus Cousins, Elita A. Lobo, Marek Petrik, Yair Zick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dec73169509c223220744b2c9b2df37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dec73169509c223220744b2c9b2df37-Abstract-Conference.html)

**Abstract**:

In reinforcement learning, robust policies for high-stakes decision-making problems with limited data are usually computed by optimizing the percentile criterion. The percentile criterion is optimized by constructing an uncertainty set that contains the true model with high probability and optimizing the policy for the worst model in the set. Since the percentile criterion is non-convex, constructing these sets itself is challenging. Existing works use Bayesian credible regions as uncertainty sets, but they are often unnecessarily large and result in learning overly conservative policies. To overcome these shortcomings, we propose a novel Value-at-Risk based dynamic programming algorithm to optimize the percentile criterion without explicitly constructing any uncertainty sets. Our theoretical and empirical results show that our algorithm implicitly constructs much smaller uncertainty sets and learns less-conservative robust policies.

----

## [0] TextDiffuser: Diffusion Models as Text Painters

**Authors**: *Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html)

**Abstract**:

Diffusion models have gained increasing attention for their impressive generation abilities but currently struggle with rendering accurate and coherent text. To address this issue, we introduce TextDiffuser, focusing on generating images with visually appealing text that is coherent with backgrounds. TextDiffuser consists of two stages: first, a Transformer model generates the layout of keywords extracted from text prompts, and then diffusion models generate images conditioned on the text prompt and the generated layout. Additionally, we contribute the first large-scale text images dataset with OCR annotations, MARIO-10M, containing 10 million image-text pairs with text recognition, detection, and character-level segmentation annotations. We further collect the MARIO-Eval benchmark to serve as a comprehensive tool for evaluating text rendering quality. Through experiments and user studies, we demonstrate that TextDiffuser is flexible and controllable to create high-quality text images using text prompts alone or together with text template images, and conduct text inpainting to reconstruct incomplete images with text. We will make the code, model and dataset publicly available.

----

## [0] Object-centric Learning with Cyclic Walks between Parts and Whole

**Authors**: *Ziyu Wang, Mike Zheng Shou, Mengmi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html)

**Abstract**:

Learning object-centric representations from complex natural environments enables both humans and machines with reasoning abilities from low-level perceptual features. To capture compositional entities of the scene, we proposed cyclic walks between perceptual features extracted from vision transformers and object entities. First, a slot-attention module interfaces with these perceptual features and produces a finite set of slot representations. These slots can bind to any object entities in the scene via inter-slot competitions for attention. Next, we establish entity-feature correspondence with cyclic walks along high transition probability based on the pairwise similarity between perceptual features (aka "parts") and slot-binded object representations (aka "whole"). The whole is greater than its parts and the parts constitute the whole. The part-whole interactions form cycle consistencies, as supervisory signals, to train the slot-attention module. Our rigorous experiments on \textit{seven} image datasets in \textit{three} \textit{unsupervised} tasks demonstrate that the networks trained with our cyclic walks can disentangle foregrounds and backgrounds, discover objects, and segment semantic objects in complex scenes. In contrast to object-centric models attached with a decoder for the pixel-level or feature-level reconstructions, our cyclic walks provide strong learning signals, avoiding computation overheads and enhancing memory efficiency. Our source code and data are available at: \href{https://github.com/ZhangLab-DeepNeuroCogLab/Parts-Whole-Object-Centric-Learning/}{link}.

----

## [0] Experiment Planning with Function Approximation

**Authors**: *Aldo Pacchiano, Jonathan Lee, Emma Brunskill*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e0d9f30c100129259f66660403fb1e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e0d9f30c100129259f66660403fb1e2-Abstract-Conference.html)

**Abstract**:

We study the problem of experiment planning with function approximation in contextual bandit problems. In settings where there is a significant overhead to deploying adaptive algorithms---for example, when the execution of the data collection policies is required to be distributed, or a human in the loop is needed to implement these policies---producing in advance a set of policies for data collection is paramount. We study the setting where a large dataset of contexts but not rewards is available and may be used by the learner to design an effective data collection strategy. Although when rewards are linear this problem has been well studied, results are still missing for more complex reward models. In this work we propose two experiment planning strategies compatible with function approximation. The first is an eluder planning and sampling procedure that can recover optimality guarantees depending on the eluder dimension of the reward function class. For the second, we show that a uniform sampler achieves competitive optimality rates in the setting where the number of actions is small. We finalize our results introducing a statistical gap fleshing out the fundamental differences between planning and adaptive learning and provide results for planning with model selection.

----

## [0] White-Box Transformers via Sparse Rate Reduction

**Authors**: *Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Benjamin D. Haeffele, Yi Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e118ba9ee76c20df728b42a35fb4704-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e118ba9ee76c20df728b42a35fb4704-Abstract-Conference.html)

**Abstract**:

In this paper, we contend that the objective of representation learning is  to compress and transform the distribution of the data, say sets of tokens, towards a mixture of low-dimensional Gaussian distributions supported on incoherent subspaces. The quality of the final representation can be measured by a unified objective function called sparse rate reduction. From this perspective, popular deep networks such as transformers can be naturally viewed as realizing iterative schemes to optimize this objective incrementally. Particularly, we show that the standard transformer block can be derived from alternating optimization on complementary parts of this objective: the multi-head self-attention operator can be viewed as a gradient descent step to compress the token sets by minimizing their lossy coding rate, and the subsequent multi-layer perceptron can be viewed as attempting to sparsify the representation of the tokens. This leads to a family of white-box transformer-like deep network architectures which are mathematically fully interpretable. Despite their simplicity, experiments show that these networks indeed learn to optimize the designed objective: they compress and sparsify representations of large-scale real-world vision datasets such as ImageNet, and achieve performance very close to thoroughly engineered transformers such as ViT. Code is at https://github.com/Ma-Lab-Berkeley/CRATE.

----

## [0] Task-Robust Pre-Training for Worst-Case Downstream Adaptation

**Authors**: *Jianghui Wang, Yang Chen, Xingyu Xie, Cong Fang, Zhouchen Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e4322fddd833f83c855660ac65e428d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e4322fddd833f83c855660ac65e428d-Abstract-Conference.html)

**Abstract**:

Pre-training has achieved remarkable success when transferred to downstream tasks. In machine learning, we care about not only the good performance of a model but also its behavior under reasonable shifts of condition. The same philosophy holds when pre-training a foundation model. However,  the foundation model may not uniformly behave well for a series of related downstream tasks. This happens, for example, when conducting mask recovery regression where the recovery ability or the training instances diverge like pattern features are extracted dominantly on pre-training, but semantic features are also required on a downstream task. This paper considers pre-training a model that guarantees a uniformly good performance over the downstream tasks. We call this goal as downstream-task robustness.Our method first separates the upstream task into several representative ones and applies a simple minimax loss for pre-training. We then design an efficient algorithm to solve the minimax lossand prove its convergence in the convex setting. In the experiments, we show both on large-scale natural language processing and computer vision datasets our method increases the metrics on worse-case downstream tasks. Additionally, some theoretical explanations for why our loss is beneficial are provided. Specifically, we show fewer samples are inherently required for the most challenging downstream task in some cases.

----

## [0] Inconsistency, Instability, and Generalization Gap of Deep Neural Network Training

**Authors**: *Rie Johnson, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e58b1bf9f218fcd19e4539e982752a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e58b1bf9f218fcd19e4539e982752a5-Abstract-Conference.html)

**Abstract**:

As deep neural networks are highly expressive, it is important to find solutions with small generalization gap (the difference between the performance on the training data and unseen data).  Focusing on the stochastic nature of training, we first present a theoretical analysis in which the bound of generalization gap depends on what we call inconsistency and instability of model outputs, which can be estimated on unlabeled data.  Our empirical study based on this analysis shows that instability and inconsistency are strongly predictive of generalization gap in various settings.  In particular, our finding indicates that inconsistency is a more reliable indicator of generalization gap than the sharpness of the loss landscape.  Furthermore, we show that algorithmic reduction of inconsistency leads to superior performance.  The results also provide a theoretical basis for existing methods such as co-distillation and ensemble.

----

## [0] Neural approximation of Wasserstein distance via a universal architecture for symmetric and factorwise group invariant functions

**Authors**: *Samantha Chen, Yusu Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e5f58d98523298cba093f658cfdf2d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e5f58d98523298cba093f658cfdf2d6-Abstract-Conference.html)

**Abstract**:

Learning distance functions between complex objects, such as the Wasserstein distance to compare point sets, is a common goal in machine learning applications. However, functions on such complex objects (e.g., point sets and graphs) are often required to be invariant to a wide variety of group actions e.g. permutation or rigid transformation. Therefore, continuous and symmetric *product* functions (such as distance functions) on such complex objects must also be invariant to the *product* of such group actions. We call these functions symmetric and factor-wise group invariant functions (or SGFI functions} in short).In this paper, we first present a general neural network architecture for approximating SFGI functions. The main contribution of this paper combines this general NN with a sketching idea in order to develop a specific and efficient neural network which can approximate the $p$-th Wasserstein distance between point sets.Very importantly, the required model complexity is *independent* of the sizes of input point sets. On the theoretical front, to the best of our knowledge, this is the first result showing that there exists a neural network with the capacity to approximate Wasserstein distance with bounded model complexity. Our work provides an interesting integration of sketching ideas for geometric problems with universal approximation of symmetric functions. On the empirical front, we present a range of results showing that our newly proposed neural network architecture performs comparatively or better than other models (including a SOTA Siamese Autoencoder based approach). In particular, our NN generalizes significantly better and trains much faster than the SOTA Siamese AE.Finally, this line of investigation could be useful in exploring effective neural network design for solving a broad range of geometric optimization problems (e.g., $k$-means in a metric space).

----

## [0] Revisiting the Evaluation of Image Synthesis with GANs

**Authors**: *Mengping Yang, Ceyuan Yang, Yichi Zhang, Qingyan Bai, Yujun Shen, Bo Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e5fa672b2c35744cbcfacb2e77f1cb0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e5fa672b2c35744cbcfacb2e77f1cb0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A good metric, which promises a reliable comparison between solutions, is essential for any well-defined task. Unlike most vision tasks that have per-sample ground-truth, image synthesis tasks target generating unseen data and hence are usually evaluated through a distributional distance between one set of real samples and another set of generated samples. This study presents an empirical investigation into the evaluation of synthesis performance, with generative adversarial networks (GANs) as a representative of generative models. In particular, we make in-depth analyses of various factors, including how to represent a data point in the representation space, how to calculate a fair distance using selected samples, and how many instances to use from each set. Extensive experiments conducted on multiple datasets and settings reveal several important findings. Firstly, a group of models that include both CNN-based and ViT-based architectures serve as reliable and robust feature extractors for measurement evaluation. Secondly, Centered Kernel Alignment (CKA) provides a better comparison across various extractors and hierarchical layers in one model. Finally, CKA is more sample-efficient and enjoys better agreement with human judgment in characterizing the similarity between two internal data correlations. These findings contribute to the development of a new measurement system, which enables a consistent and reliable re-evaluation of current state-of-the-art generative models.

----

## [0] What Do Deep Saliency Models Learn about Visual Attention?

**Authors**: *Shi Chen, Ming Jiang, Qi Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e680f115a22d60cbc228a0c6dae5936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e680f115a22d60cbc228a0c6dae5936-Abstract-Conference.html)

**Abstract**:

In recent years, deep saliency models have made significant progress in predicting human visual attention. However, the mechanisms behind their success remain largely unexplained due to the opaque nature of deep neural networks. In this paper, we present a novel analytic framework that sheds light on the implicit features learned by saliency models and provides principled interpretation and quantification of their contributions to saliency prediction. Our approach decomposes these implicit features into interpretable bases that are explicitly aligned with semantic attributes and reformulates saliency prediction as a weighted combination of probability maps connecting the bases and saliency. By applying our framework, we conduct extensive analyses from various perspectives, including the positive and negative weights of semantics, the impact of training data and architectural designs, the progressive influences of fine-tuning, and common error patterns of state-of-the-art deep saliency models. Additionally, we demonstrate the effectiveness of our framework by exploring visual attention characteristics in various application scenarios, such as the atypical attention of people with autism spectrum disorder, attention to emotion-eliciting stimuli, and attention evolution over time. Our code is publicly available at \url{https://github.com/szzexpoi/saliency_analysis}.

----

## [0] Three Iterations of (d - 1)-WL Test Distinguish Non Isometric Clouds of d-dimensional Points

**Authors**: *Valentino Delle Rose, Alexander Kozachinskiy, Cristobal Rojas, Mircea Petrache, Pablo Barceló*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e6cf8f77bd8e907f53babcd7664c710-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e6cf8f77bd8e907f53babcd7664c710-Abstract-Conference.html)

**Abstract**:

The Weisfeiler-Lehman (WL) test is a fundamental iterative algorithm for checking the isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of Euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud. Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.

----

## [0] Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving

**Authors**: *Sepidehsadat (Sepid) Hossieni, Mohammad Amin Shabani, Saghar Irandoust, Yasutaka Furukawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html)

**Abstract**:

This paper presents an end-to-end neural architecture based on Diffusion Models for spatial puzzle solving, particularly jigsaw puzzle and room arrangement tasks.In the latter task, for instance, the proposed system ``PuzzleFusion'' takes a set of room layouts as polygonal curves in the top-down view and aligns the room layout pieces by estimating their 2D translations and rotations, akin to solving the jigsaw puzzle of room layouts. A surprising discovery of the paper is that the simple use of a Diffusion Model effectively solves these challenging spatial puzzle tasks as a conditional generation process. To enable learning of an end-to-end neural system, the paper introduces new datasets with ground-truth arrangements: 1) 2D Voronoi Jigsaw Dataset, a synthetic one where pieces are generated by voronoi diagram of 2D pointset; and 2) MagicPlan Dataset, a real one from a production pipeline by MagicPlan, where pieces are room layouts constructed by augmented reality App by real-estate consumers.The qualitative and quantitative evaluations demonstrate that the proposed approach outperforms the competing methods by significant margins in all three spatial puzzle tasks. We have provided code and data in https://sepidsh.github.io/puzzlefusion.

----

## [0] Visual Instruction Inversion: Image Editing via Image Prompting

**Authors**: *Thao Nguyen, Yuheng Li, Utkarsh Ojha, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e75f7539cbde5de895fab238ff42519-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e75f7539cbde5de895fab238ff42519-Abstract-Conference.html)

**Abstract**:

Text-conditioned image editing has emerged as a powerful tool for editing images.However, in many situations, language can be ambiguous and ineffective in describing specific image edits.When faced with such challenges, visual prompts can be a more informative and intuitive way to convey ideas.We present a method for image editing via visual prompting.Given pairs of example that represent the "before" and "after" images of an edit, our goal is to learn a text-based editing direction that can be used to perform the same edit on new images.We leverage the rich, pretrained editing capabilities of text-to-image diffusion models by inverting visual prompts into editing instructions.Our results show that with just one example pair, we can achieve competitive results compared to state-of-the-art text-conditioned image editing frameworks.

----

## [0] Algorithm Selection for Deep Active Learning with Imbalanced Datasets

**Authors**: *Jifan Zhang, Shuai Shao, Saurabh Verma, Robert D. Nowak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e77af93008ee6cd248a31723ce357d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e77af93008ee6cd248a31723ce357d8-Abstract-Conference.html)

**Abstract**:

Label efficiency has become an increasingly important objective in deep learning applications. Active learning aims to reduce the number of labeled examples needed to train deep networks, but the empirical performance of active learning algorithms can vary dramatically across datasets and applications. It is difficult to know in advance which active learning strategy will perform well or best in a given application. To address this, we propose the first adaptive algorithm selection strategy for deep active learning. For any unlabeled dataset, our (meta) algorithm TAILOR (Thompson ActIve Learning algORithm selection) iteratively and adaptively chooses among a set of candidate active learning algorithms. TAILOR uses novel reward functions aimed at gathering class-balanced examples. Extensive experiments in multi-class and multi-label applications demonstrate TAILOR's effectiveness in achieving accuracy comparable or better than that of the best of the candidate algorithms. Our implementation of TAILOR is open-sourced at https://github.com/jifanz/TAILOR.

----

## [0] Federated Compositional Deep AUC Maximization

**Authors**: *Xinwen Zhang, Yihan Zhang, Tianbao Yang, Richard Souvenir, Hongchang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e7b192fc8b3acb93749c5accfa60e0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e7b192fc8b3acb93749c5accfa60e0c-Abstract-Conference.html)

**Abstract**:

Federated learning has attracted increasing attention due to the promise of balancing privacy and large-scale learning; numerous approaches have been proposed. However, most existing approaches focus on problems with balanced data, and prediction performance is far from satisfactory for many real-world applications where the number of samples in different classes is highly imbalanced. To address this challenging problem, we developed a novel federated learning method for imbalanced data by directly optimizing the area under curve (AUC) score. In particular, we formulate the AUC maximization problem as a federated compositional minimax optimization problem, develop a local stochastic compositional gradient descent ascent with momentum algorithm, and provide bounds on the computational and communication complexities of our algorithm. To the best of our knowledge, this is the first work to achieve such favorable theoretical results. Finally, extensive experimental results confirm the efficacy of our method.

----

## [0] On Learning Latent Models with Multi-Instance Weak Supervision

**Authors**: *Kaifu Wang, Efthymia Tsamoura, Dan Roth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e83498c3eafe109a44b12979c2c73db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e83498c3eafe109a44b12979c2c73db-Abstract-Conference.html)

**Abstract**:

We consider a weakly supervised learning scenario where the supervision signal is generated by a transition function $\sigma$ of labels associated with multiple input instances. We formulate this problem as *multi-instance Partial Label Learning (multi-instance PLL)*, which is an extension to the standard PLL problem. Our problem is met in different fields, including latent structural learning and neuro-symbolic integration. Despite the existence of many learning techniques, limited theoretical analysis has been dedicated to this problem. In this paper, we provide the first theoretical study of multi-instance PLL with possibly an unknown transition $\sigma$. Our main contributions are as follows: First, we proposed a necessary and sufficient condition for the learnability of the problem. This condition nontrivially generalizes and relaxes the existing *small ambiguity degree* in PLL literature since we allow the transition to be deterministic. Second, we derived Rademacher-style error bounds based on the top-$k$ surrogate loss that is widely used in the neuro-symbolic literature. Furthermore, we conclude with empirical experiments for learning with an unknown transition. The empirical results align with our theoretical findings; however, they also expose the issue of scalability in the weak supervision literature.

----

## [0] Degraded Polygons Raise Fundamental Questions of Neural Network Perception

**Authors**: *Leonard Tang, Dan Ley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ec408df112bc9b186d7b8fe0ada902a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ec408df112bc9b186d7b8fe0ada902a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deeplearning vision models suffer in a variety of settings that humans capably handle. Inlight of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images underdegradation, first introduced over 30 years ago in the Recognition-by-Componentstheory of human vision. Specifically, we study the performance and behavior ofneural networks on the seemingly simple task of classifying regular polygons atvarying orders of degradation along their perimeters. To this end, we implement theAutomated Shape Recoverability Testfor rapidly generating large-scale datasetsof perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity ofneural networks to recognize and recover such degraded shapes when initializedwith different priors. Ultimately, we find that neural networksâ€™ behavior on thissimple task conflicts with human behavior, raising a fundamental question of therobustness and learning capabilities of modern computer vision models

----

## [0] Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks

**Authors**: *Blake Bordelon, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html)

**Abstract**:

We analyze the dynamics of finite width effects in wide but finite feature learning neural networks. Starting from a dynamical mean field theory  description of infinite width deep neural network kernel and prediction dynamics, we provide a characterization of the $\mathcal{O}(1/\sqrt{\text{width}})$ fluctuations of the DMFT order parameters over random initializations of the network weights. Our results, while perturbative in width, unlike prior analyses, are non-perturbative in the strength of feature learning. In the lazy limit of network training, all kernels are random but static in time and the prediction variance has a universal form. However, in the rich, feature learning regime, the fluctuations of the kernels and predictions are dynamically coupled with a variance that can be computed self-consistently. In two layer networks, we show how feature learning can dynamically reduce the variance of the final tangent kernel and final network predictions. We also show how initialization variance can slow down online learning in wide but finite networks. In deeper networks, kernel variance can dramatically accumulate through subsequent layers at large feature learning strengths, but feature learning continues to improve the signal-to-noise ratio of the feature kernels. In discrete time, we demonstrate that large learning rate phenomena such as edge of stability effects can be well captured by infinite width dynamics and that initialization variance can decrease dynamically. For CNNs trained on CIFAR-10, we empirically find significant corrections to both the bias and variance of network dynamics due to finite width.

----

## [0] TART: A plug-and-play Transformer module for task-agnostic reasoning

**Authors**: *Kush Bhatia, Avanika Narayan, Christopher De Sa, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ece70d2259b8e9510e2d4ca8754cecf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ece70d2259b8e9510e2d4ca8754cecf-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) exhibit in-context learning abilities which enable the same model to perform several tasks without any task-specific training. In contrast, traditional adaptation approaches, such as fine-tuning, modify the underlying models for each specific task. In-context learning, however, consistently underperforms task-specific tuning approaches even when presented with the same examples. While most existing approaches (e.g., prompt engineering) focus on the LLM's learned representations to patch this performance gap, our experiments actually reveal that LLM representations contain sufficient information to make good predictions. As such, we focus on the LLM's reasoning abilities and demonstrate that this performance gap exists due to their inability to perform simple probabilistic reasoning tasks. This raises an intriguing question: Are LLMs actually capable of learning how to reason in a task-agnostic manner? We answer this in the affirmative and, as a proof of concept, propose TART which generically improves an LLM's reasoning abilities using a synthetically trained reasoning module. TART trains this Transformer-based reasoning module in a task-agnostic manner using only synthetic logistic regression tasks and composes it with an arbitrary real-world pre-trained model without any additional training. With a single inference module, TART improves performance across different model families (GPT-Neo, Pythia, Bloom), model sizes (100M - 6B), tasks (14 NLP classification tasks), and even across different modalities (audio and vision). On the RAFT Benchmark, TART improves GPT-Neo (125M)'s performance such that it outperforms Bloom (176B), and is within $4$% of GPT-3.

----

## [0] Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment

**Authors**: *Carsten T. Lüth, Till J. Bungert, Lukas Klein, Paul Jaeger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract-Conference.html)

**Abstract**:

Active Learning (AL) aims to reduce the labeling burden by interactively selecting the most informative samples from a pool of unlabeled data. While there has been extensive research on improving AL query methods in recent years, some studies have questioned the effectiveness of AL compared to emerging paradigms such as semi-supervised (Semi-SL) and self-supervised learning (Self-SL), or a simple optimization of classifier configurations. Thus, today’s AL literature presents an inconsistent and contradictory landscape, leaving practitioners uncertain about whether and how to use AL in their tasks. In this work, we make the case that this inconsistency arises from a lack of systematic and realistic evaluation of AL methods. Specifically, we identify five key pitfalls in the current literature that reflect the delicate considerations required for AL evaluation. Further, we present an evaluation framework that overcomes these pitfalls and thus enables meaningful statements about the performance of AL methods. To demonstrate the relevance of our protocol, we present a large-scale empirical study and benchmark for image classification spanning various data sets, query methods, AL settings, and training paradigms. Our findings clarify the inconsistent picture in the literature and enable us to give hands-on recommendations for practitioners. The benchmark is hosted at https://github.com/IML-DKFZ/realistic-al.

----

## [0] Semantic HELM: A Human-Readable Memory for Reinforcement Learning

**Authors**: *Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1eeacdf8770e6dd5164cdeec8bcfa8cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1eeacdf8770e6dd5164cdeec8bcfa8cc-Abstract-Conference.html)

**Abstract**:

Reinforcement learning agents deployed in the real world often have to cope with partially observable environments. Therefore, most agents employ memory mechanisms to approximate the state of the environment. Recently, there have been impressive success stories in mastering partially observable environments, mostly in the realm of computer games like Dota 2, StarCraft II, or MineCraft. However, existing methods lack interpretability in the sense that it is not comprehensible for humans what the agent stores in its memory.In this regard, we propose a novel memory mechanism that represents past events in human language.Our method uses CLIP to associate visual inputs with language tokens. Then we feed these tokens to a pretrained language model that serves the agent as memory and provides it with a coherent and human-readable representation of the past.We train our memory mechanism on a set of partially observable environments and find that it excels on tasks that require a memory component, while mostly attaining performance on-par with strong baselines on tasks that do not. On a challenging continuous recognition task, where memorizing the past is crucial, our memory mechanism converges two orders of magnitude faster than prior methods.Since our memory mechanism is human-readable, we can peek at an agent's memory and check whether crucial pieces of information have been stored.This significantly enhances troubleshooting and paves the way toward more interpretable agents.

----

## [0] Empowering Convolutional Neural Nets with MetaSin Activation

**Authors**: *Farnood Salehi, Tunç Aydin, André Gaillard, Guglielmo Camporese, Yuxuan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f05584d537c92c8271699f207677475-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f05584d537c92c8271699f207677475-Abstract-Conference.html)

**Abstract**:

ReLU networks have remained the default choice for models in the area of image prediction despite their well-established spectral bias towards learning low frequencies faster, and consequently their difficulty of reproducing high frequency visual details. As an alternative, sin networks showed promising results in learning implicit representations of visual data. However training these networks in practically relevant settings proved to be difficult, requiring careful initialization, dealing with issues due to inconsistent gradients, and a degeneracy in local minima. In this work, we instead propose replacing a baseline network’s existing activations with a novel ensemble function with trainable parameters. The proposed MetaSin activation can be trained reliably without requiring intricate initialization schemes, and results in consistently lower test loss compared to alternatives. We demonstrate our method in the areas of Monte-Carlo denoising and image resampling where we set new state-of-the-art through a knowledge distillation based training procedure. We present ablations on hyper-parameter settings, comparisons with alternative activation function formulations, and discuss the use of our method in other domains, such as image classification.

----

## [0] When Does Confidence-Based Cascade Deferral Suffice?

**Authors**: *Wittawat Jitkrittum, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, Sanjiv Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html)

**Abstract**:

Cascades are a classical strategy to enable inference cost to vary adaptively across samples, wherein a sequence of classifiers are invoked in turn.  A deferral rule determines whether to invoke the next classifier in the sequence, or to terminate prediction.  One simple deferral rule employs the confidence of the current classifier, e.g., based on the maximum predicted softmax probability.  Despite being oblivious to the structure of the cascade --- e.g., not modelling the errors of downstream models --- such confidence-based deferral often works remarkably well in practice.  In this paper, we seek to better understand the conditions under which confidence-based deferral may fail, and when alternate deferral strategies can perform better.  We first present a theoretical characterisation of the optimal deferral rule, which precisely characterises settings under which confidence-based deferral may suffer.  We then study post-hoc deferral mechanisms, and demonstrate they can significantly improve upon confidence-based deferral in settings where (i)  downstream models are specialists that only work well on a subset of inputs, (ii) samples are subject to label noise, and (iii) there is distribution shift between the train and test set.

----

## [0] DeWave: Discrete Encoding of EEG Waves for EEG to Text Translation

**Authors**: *Yiqun Duan, Charles Chau, Zhen Wang, Yu-Kai Wang, Chin-Teng Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html)

**Abstract**:

The translation of brain dynamics into natural language is pivotal for brain-computer interfaces (BCIs), a field that has seen substantial growth in recent years. With the swift advancement of large language models, such as ChatGPT, the need to bridge the gap between the brain and languages becomes increasingly pressing. Current methods, however, require eye-tracking fixations or event markers to segment brain dynamics into word-level features, which can restrict the practical application of these systems. These event markers may not be readily available or could be challenging to acquire during real-time inference, and the sequence of eye fixations may not align with the order of spoken words. To tackle these issues, we introduce a novel framework, DeWave, that integrates discrete encoding sequences into open-vocabulary EEG-to-text translation tasks. DeWave uses a quantized variational encoder to derive discrete codex encoding and align it with pre-trained language models. This discrete codex representation brings forth two advantages: 1) it alleviates the order mismatch between eye fixations and spoken words by introducing text-EEG contrastive alignment training, and 2) it minimizes the interference caused by individual differences in EEG waves through an invariant discrete codex. Our model surpasses the previous baseline (40.1 and 31.7) by 3.06% and 6.34\%, respectively, achieving 41.35 BLEU-1 and 33.71 Rouge-F on the ZuCo Dataset. Furthermore, this work is the first to facilitate the translation of entire EEG signal periods without the need for word-level order markers (e.g., eye fixations), scoring 20.5 BLEU-1 and 29.5 Rouge-1 on the ZuCo Dataset, respectively.

----

## [0] SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data

**Authors**: *Bang An, Xun Zhou, Yongjian Zhong, Tianbao Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f3cbee17170c3ffff3e413d2df54f6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f3cbee17170c3ffff3e413d2df54f6b-Abstract-Conference.html)

**Abstract**:

The problem of urban event ranking aims at predicting the top-$k$ most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. Due to the common assumption that items are independent. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named SpatialRank. SpatialRank features  adaptive graph convolution layers that dynamically learn the spatiotemporal dependencies across locations from data. In addition, the model optimizes through surrogates a hybrid NDCG loss with a spatial component to better rank neighboring spatial locations. We design an importance-sampling with a spatial filtering algorithm to effectively evaluate the loss during training. Comprehensive experiments on three real-world datasets demonstrate that SpatialRank can effectively identify the top riskiest locations of crimes and traffic accidents and outperform state-of-art methods in terms of NDCG by up to 12.7%.

----

## [0] An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions

**Authors**: *Mohammad Jalali, Cheuk Ting Li, Farzan Farnia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)

**Abstract**:

The evaluation of generative models has received significant attention in the machine learning community.  When applied to a multi-modal distribution which is common among image datasets, an intuitive evaluation criterion is the number of modes captured by the generative model. While several scores have been proposed to evaluate the quality and diversity of a model's generated data, the correspondence between existing scores and the number of modes in the distribution is unclear. In this work, we propose an information-theoretic diversity evaluation method for multi-modal underlying distributions. We utilize the R\'enyi Kernel Entropy (RKE) as an evaluation score based on quantum information theory to measure the number of modes in generated samples. To interpret the proposed evaluation method, we show that the RKE score can output the number of modes of a mixture of sub-Gaussian components. We also prove estimation error bounds for estimating the RKE score from limited data, suggesting a fast convergence of the empirical RKE score to the score for the underlying data distribution. Utilizing the RKE score, we conduct an extensive evaluation of state-of-the-art generative models over standard image datasets. The numerical results indicate that while the recent algorithms for training generative models manage to improve the mode-based diversity over the earlier architectures, they remain incapable of capturing the full diversity of real data. Our empirical results provide a ranking of widely-used generative models based on the RKE score of their generated samples.

----

## [0] A Cross-Moment Approach for Causal Effect Estimation

**Authors**: *Yaroslav Kivva, Saber Salehkaleybar, Negar Kiyavash*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f6100363156cced8633f4e89dd8ceb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f6100363156cced8633f4e89dd8ceb1-Abstract-Conference.html)

**Abstract**:

We consider the problem of estimating the causal effect of a treatment on an outcome in  linear structural causal models (SCM) with latent confounders when we have access to a single proxy variable.Several methods (such as difference-in-difference (DiD) estimator or negative outcome control) have been proposed in this setting in the literature. However, these approaches require either restrictive assumptions on the data generating model or having access to at least two proxy variables.We propose a method to estimate the causal effect using cross moments between the treatment, the outcome, and the proxy variable. In particular, we show that the causal effect can be identified with simple arithmetic operations on the cross moments if the latent confounder in linear SCM is non-Gaussian.In this setting, DiD estimator provides an unbiased estimate only in the special case where the latent confounder has exactly the same direct causal effects on the outcomes in the pre-treatment and post-treatment phases. This translates to the common trend assumption in DiD, which we effectively relax.Additionally, we provide an impossibility result that shows the causal effect cannot be identified if the observational distribution over the treatment, the outcome, and the proxy is jointly Gaussian. Our experiments on both synthetic and real-world datasets showcase the effectivenessof the proposed approach in estimating the causal effect.

----

## [0] Combining Behaviors with the Successor Features Keyboard

**Authors**: *Wilka Carvalho, Andre Saraiva, Angelos Filos, Andrew K. Lampinen, Loic Matthey, Richard L. Lewis, Honglak Lee, Satinder Singh, Danilo Jimenez Rezende, Daniel Zoran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f69928210578f4cf5b538a8c8806798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f69928210578f4cf5b538a8c8806798-Abstract-Conference.html)

**Abstract**:

The Option Keyboard (OK) was recently proposed as a method for transferring behavioral knowledge across tasks. OK transfers knowledge by adaptively combining subsets of known behaviors using Successor Features (SFs) and Generalized Policy Improvement (GPI).However, it relies on hand-designed state-features and task encodings which are cumbersome to design for every new environment.In this work, we propose the "Successor Features Keyboard" (SFK), which enables transfer with discovered state-features and task encodings.To enable discovery, we propose the "Categorical Successor Feature Approximator" (CSFA), a novel learning algorithm for estimating SFs while jointly discovering state-features and task encodings.With SFK and CSFA, we achieve the first demonstration of transfer with SFs in a challenging 3D environment where all the necessary representations are discovered.We first compare CSFA against other methods for approximating SFs and show that only CSFA discovers representations compatible with SF&GPI at this scale.We then compare SFK against transfer learning baselines and show that it transfers most quickly to long-horizon tasks.

----

## [0] Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

**Authors**: *Chenyu You, Weicheng Dai, Yifei Min, Fenglin Liu, David A. Clifton, S. Kevin Zhou, Lawrence H. Staib, James S. Duncan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html)

**Abstract**:

For medical image segmentation, contrastive learning is the dominant practice to improve the quality of visual representations by contrasting semantically similar and dissimilar pairs of samples. This is enabled by the observation that without accessing ground truth labels, negative examples with truly dissimilar anatomical features, if sampled, can significantly improve the performance. In reality, however, these samples may come from similar anatomical features and the models may struggle to distinguish the minority tail-class samples, making the tail classes more prone to misclassification, both of which typically lead to model collapse. In this paper, we propose $\texttt{ARCO}$, a semi-supervised contrastive learning (CL) framework with stratified group theory for medical image segmentation. In particular, we first propose building $\texttt{ARCO}$ through the concept of variance-reduced estimation, and show that certain variance-reduction techniques are particularly beneficial in pixel/voxel-level segmentation tasks with extremely limited labels. Furthermore, we theoretically prove these sampling techniques are universal in variance reduction. Finally, we experimentally validate our approaches on eight benchmarks, i.e., five 2D/3D medical and three semantic segmentation datasets, with different label settings, and our methods consistently outperform state-of-the-art semi-supervised methods. Additionally, we augment the CL frameworks with these sampling techniques and demonstrate significant gains over previous methods. We believe our work is an important step towards semi-supervised medical image segmentation by quantifying the limitation of current self-supervision objectives for accomplishing such challenging safety-critical tasks.

----

## [0] Boosting Verification of Deep Reinforcement Learning via Piece-Wise Linear Decision Neural Networks

**Authors**: *Jiaxu Tian, Dapeng Zhi, Si Liu, Peixin Wang, Cheng Chen, Min Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f96b24df4b06f5d68389845a9a13ed9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f96b24df4b06f5d68389845a9a13ed9-Abstract-Conference.html)

**Abstract**:

Formally verifying deep reinforcement learning (DRL) systems suffers from both inaccurate verification results and limited scalability. The major obstacle lies in the large overestimation introduced inherently during training and then transforming the inexplicable decision-making models, i.e., deep neural networks (DNNs), into easy-to-verify models. In this paper, we propose an inverse transform-then-train approach, which first encodes a DNN into an equivalent set of efficiently and tightly verifiable linear control policies and then optimizes them via reinforcement learning.  We accompany our inverse approach with a novel neural network model called piece-wise linear decision neural networks (PLDNNs), which are compatible with most existing DRL training algorithms with comparable performance against conventional DNNs. Our extensive experiments show that, compared to DNN-based  DRL systems, PLDNN-based systems can be more efficiently and tightly verified with up to $438$ times speedup and a significant reduction in overestimation.  In particular, even a complex $12$-dimensional DRL system is efficiently verified with up to 7 times deeper computation steps.

----

## [0] Star-Shaped Denoising Diffusion Probabilistic Models

**Authors**: *Andrey Okhotin, Dmitry Molchanov, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, Dmitry P. Vetrov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1fcefa894924bb1688041b7a26fb8aea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1fcefa894924bb1688041b7a26fb8aea-Abstract-Conference.html)

**Abstract**:

Denoising Diffusion Probabilistic Models (DDPMs) provide the foundation for the recent breakthroughs in generative modeling.Their Markovian structure makes it difficult to define DDPMs with distributions other than Gaussian or discrete.In this paper, we introduce Star-Shaped DDPM (SS-DDPM).Its star-shaped diffusion process allows us to bypass the need to define the transition probabilities or compute posteriors.We establish duality between star-shaped and specific Markovian diffusions for the exponential family of distributions and derive efficient algorithms for training and sampling from SS-DDPMs.In the case of Gaussian distributions, SS-DDPM is equivalent to DDPM.However, SS-DDPMs provide a simple recipe for designing diffusion models with distributions such as Beta, von Misesâ€“Fisher, Dirichlet, Wishart and others, which can be especially useful when data lies on a constrained manifold.We evaluate the model in different settings and find it competitive even on image data, where Beta SS-DDPM achieves results comparable to a Gaussian DDPM.Our implementation is available at https://github.com/andrey-okhotin/star-shaped

----

## [0] An Adaptive Algorithm for Learning with Unknown Distribution Drift

**Authors**: *Alessio Mazzetto, Eli Upfal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1fe6f635fe265292aba3987b5123ae3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1fe6f635fe265292aba3987b5123ae3d-Abstract-Conference.html)

**Abstract**:

We develop and analyze a general technique for learning with an unknown distribution drift. Given a sequence of independent observations from the last $T$ steps of a drifting distribution, our algorithm agnostically learns a family of functions with respect to the current distribution at time $T$. Unlike previous work, our technique does not require prior knowledge about the magnitude of the drift. Instead, the algorithm adapts to the sample data. Without explicitly estimating the drift, the algorithm learns a family of functions with almost the same error as a learning algorithm that knows the magnitude of the drift in advance. Furthermore, since our algorithm adapts to the data, it can guarantee a better learning error than an algorithm that relies on loose bounds on the drift. We demonstrate the application of our technique in two fundamental learning scenarios: binary classification and linear regression.

----

## [0] QLoRA: Efficient Finetuning of Quantized LLMs

**Authors**: *Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)

**Abstract**:

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information-theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small, high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations, showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training.

----

## [0] EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning

**Authors**: *Ping Guo, Xiangpeng Wei, Yue Hu, Baosong Yang, Dayiheng Liu, Fei Huang, Jun Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/201408406e0c5cf7626c4baeae6eaadd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/201408406e0c5cf7626c4baeae6eaadd-Abstract-Conference.html)

**Abstract**:

Expressing universal semantics common to all languages is helpful to understand the meanings of complex and culture-specific sentences. The research theme underlying this scenario focuses on learning universal representations across languages with the usage of massive parallel corpora. However, due to the sparsity and scarcity of parallel data, there is still a big challenge in learning authentic ``universals'' for any two languages. In this paper, we propose Emma-X: an EM-like Multilingual pre-training Algorithm, to learn Cross-lingual universals with the aid of excessive multilingual non-parallel data. Emma-X unifies the cross-lingual representation learning task and an extra semantic relation prediction task within an EM framework. Both the extra semantic classifier and the cross-lingual sentence encoder approximate the semantic relation of two sentences, and supervise each other until convergence. To evaluate Emma-X, we conduct experiments on xrete, a newly introduced benchmark containing 12 widely studied cross-lingual tasks that fully depend on sentence-level representations. Results reveal that Emma-X achieves state-of-the-art performance. Further geometric analysis of the built representation space with three requirements demonstrates the superiority of Emma-X over advanced models.

----

## [0] Tester-Learners for Halfspaces: Universal Algorithms

**Authors**: *Aravind Gollakota, Adam R. Klivans, Konstantinos Stavropoulos, Arsen Vasilyan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/204d9a9a4816a45909010587ffc3204b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/204d9a9a4816a45909010587ffc3204b-Abstract-Conference.html)

**Abstract**:

We give the first tester-learner for halfspaces that succeeds universally over a wide class of structured distributions. Our universal tester-learner runs in fully polynomial time and has the following guarantee: the learner achieves error $O(\mathrm{opt}) + \epsilon$ on any labeled distribution that the tester accepts, and moreover, the tester accepts whenever the marginal is any distribution that satisfies a Poincare inequality. In contrast to prior work on testable learning, our tester is not tailored to any single target distribution but rather succeeds for an entire target class of distributions. The class of Poincare distributions includes all strongly log-concave distributions, and, assuming the Kannan--Lovasz--Simonovits (KLS) conjecture, includes all log-concave distributions. In the special case where the label noise is known to be Massart, our tester-learner achieves error $\mathrm{opt} + \epsilon$ while accepting all log-concave distributions unconditionally (without assuming KLS).Our tests rely on checking hypercontractivity of the unknown distribution using a sum-of-squares (SOS) program, and crucially make use of the fact that Poincare distributions are certifiably hypercontractive in the SOS framework.

----

## [0] Revisit the Power of Vanilla Knowledge Distillation: from Small Scale to Large Scale

**Authors**: *Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/204f828ba287fdecf41dd002e9a07d8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/204f828ba287fdecf41dd002e9a07d8c-Abstract-Conference.html)

**Abstract**:

The tremendous success of large models trained on extensive datasets demonstrates that scale is a key ingredient in achieving superior results. Therefore, the reflection on the rationality of designing knowledge distillation (KD) approaches for limited-capacity architectures solely based on small-scale datasets is now deemed imperative. In this paper, we identify the small data pitfall that presents in previous KD methods, which results in the underestimation of the power of vanilla KD framework on large-scale datasets such as ImageNet-1K. Specifically, we show that employing stronger data augmentation techniques and using larger datasets can directly decrease the gap between vanilla KD and other meticulously designed KD variants. This highlights the necessity of designing and evaluating KD approaches in the context of practical scenarios, casting off the limitations of small-scale datasets. Our investigation of the vanilla KD and its variants in more complex schemes, including stronger training strategies and different model capacities, demonstrates that vanilla KD is elegantly simple but astonishingly effective in large-scale scenarios. Without bells and whistles, we obtain state-of-the-art ResNet-50, ViT-S, and ConvNeXtV2-T models for ImageNet, which achieve 83.1%, 84.3%, and 85.0% top-1 accuracy, respectively. PyTorch code and checkpoints can be found at https://github.com/Hao840/vanillaKD.

----

## [0] Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context

**Authors**: *Julian Tanke, Oh-Hun Kwon, Felix B. Mueller, Andreas Doering, Jürgen Gall*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Forecasting human motion of multiple persons is very challenging. It requires to model the interactions between humans and the interactions with objects and the environment. For example, a person might want to make a coffee, but if the coffee machine is already occupied the person will haveto wait. These complex relations between scene geometry and persons ariseconstantly in our daily lives, and models that wish to accurately forecasthuman behavior will have to take them into consideration. To facilitate research in this direction, we propose Humans in Kitchens, alarge-scale multi-person human motion dataset with annotated 3D human poses, scene geometry and activities per person and frame.Our dataset consists of over 7.3h recorded data of up to 16 persons at the same time in four kitchen scenes, with more than 4M annotated human poses, represented by a parametric 3D body model. In addition, dynamic scene geometry and objects like chair or cupboard are annotated per frame. As first benchmarks, we propose two protocols for short-term and long-term human motion forecasting.

----

## [0] LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings

**Authors**: *Ningyi Liao, Siqiang Luo, Xiang Li, Jieming Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/206191b9b7349e2743d98d855dec9e58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/206191b9b7349e2743d98d855dec9e58-Abstract-Conference.html)

**Abstract**:

Heterophilous Graph Neural Network (GNN) is a family of GNNs that specializes in learning graphs under heterophily, where connected nodes tend to have different labels. Most existing heterophilous models incorporate iterative non-local computations to capture node relationships. However, these approaches have limited application to large-scale graphs due to their high computational costs and challenges in adopting minibatch schemes. In this work, we study the scalability issues of heterophilous GNN and propose a scalable model, LD2, which simplifies the learning process by decoupling graph propagation and generating expressive embeddings prior to training. Theoretical analysis demonstrates that LD2 achieves optimal time complexity in training, as well as a memory footprint that remains independent of the graph scale. We conduct extensive experiments to showcase that our model is capable of lightweight minibatch training on large-scale heterophilous graphs, with up to $15\times$ speed improvement and efficient memory utilization, while maintaining comparable or better performance than the baselines.

----

## [0] On Single-Index Models beyond Gaussian Data

**Authors**: *Aaron Zweig, Loucas Pillaud-Vivien, Joan Bruna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2063a00c435aafbcc58c16ce1e522139-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2063a00c435aafbcc58c16ce1e522139-Abstract-Conference.html)

**Abstract**:

Sparse high-dimensional functions have arisen as a rich framework to study the behavior of gradient-descent methods using shallow neural networks, and showcasing its ability to perform feature learning beyond linear models. Amongst those functions, the simplest are single-index models $f(x) = \phi( x \cdot \theta^*)$, where the labels are generated by an arbitrary non-linear link function $\phi$ of an unknown one-dimensional projection $\theta^*$ of the input data. By focusing on Gaussian data, several recent works have built a remarkable picture, where the so-called information exponent (related to the regularity of the link function) controls the required sample complexity. In essence, these tools exploit the stability and spherical symmetry of Gaussian distributions.In this work, we explore extensions of this picture beyond the Gaussian setting, where both stability or symmetry might be violated. Focusing on the planted setting where $\phi$ is known, our main results establish that Stochastic Gradient Descent recovers the unknown direction $\theta^*$ with constant probability in the high-dimensional regime, under mild assumptions that significantly extend ~[Yehudai and Shamir,20].

----

## [0] Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

**Authors**: *Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, Liang Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2082273791021571c410f41d565d0b45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2082273791021571c410f41d565d0b45-Abstract-Conference.html)

**Abstract**:

Hand-crafted image quality metrics, such as PSNR and SSIM, are commonly used to evaluate model privacy risk under reconstruction attacks. Under these metrics, reconstructed images that are determined to resemble the original one generally indicate more privacy leakage. Images determined as overall dissimilar, on the other hand, indicate higher robustness against attack. However, there is no guarantee that these metrics well reflect human opinions, which offers trustworthy judgement for model privacy leakage.  In this paper, we comprehensively study the faithfulness of these hand-crafted metrics to human perception of privacy information from the reconstructed images. On 5 datasets ranging from natural images, faces, to fine-grained classes, we use 4 existing attack methods to reconstruct images from many different classification models and, for each reconstructed image, we ask multiple human annotators to assess whether this image is recognizable. Our studies reveal that the hand-crafted metrics only have a weak correlation with the human evaluation of privacy leakage and that even these metrics themselves often contradict each other. These observations suggest risks of current metrics  in the community. To address this potential risk, we propose a learning-based measure called SemSim to evaluate the Semantic Similarity between the original and reconstructed images. SemSim is trained with a standard triplet loss, using an original image as an anchor, one of its recognizable reconstructed images as a positive sample, and an unrecognizable one as a negative. By training on human annotations, SemSim exhibits a greater reflection of privacy leakage on the semantic level. We show that SemSim has a significantly higher correlation with human judgment compared with existing metrics. Moreover, this strong correlation generalizes to unseen datasets, models and attack methods. We envision this work as a milestone for image quality evaluation closer to the human level. The project webpage can be accessed at https://sites.google.com/view/semsim.

----

## [0] Graph Denoising Diffusion for Inverse Protein Folding

**Authors**: *Kai Yi, Bingxin Zhou, Yiqing Shen, Pietro Lió, Yuguang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20888d00c5df685de2c09790040e0327-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20888d00c5df685de2c09790040e0327-Abstract-Conference.html)

**Abstract**:

Inverse protein folding is challenging due to its inherent one-to-many mapping characteristic, where numerous possible amino acid sequences can fold into a single, identical protein backbone. This task involves not only identifying viable sequences but also representing the sheer diversity of potential solutions. However, existing discriminative models, such as transformer-based auto-regressive models, struggle to encapsulate the diverse range of plausible solutions. In contrast, diffusion probabilistic models, as an emerging genre of generative approaches, offer the potential to generate a diverse set of sequence candidates for determined protein backbones. We propose a novel graph denoising diffusion model for inverse protein folding, where a given protein backbone guides the diffusion process on the corresponding amino acid residue types. The model infers the joint distribution of amino acids conditioned on the nodes' physiochemical properties and local environment. Moreover, we utilize amino acid replacement matrices for the diffusion forward process, encoding the biologically-meaningful prior knowledge of amino acids from their spatial and sequential neighbors as well as themselves, which reduces the sampling space of the generative process. Our model achieves state-of-the-art performance over a set of popular baseline methods in sequence recovery and exhibits great potential in generating diverse protein sequences for a determined protein backbone structure.

----

## [0] Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation

**Authors**: *Chaofan Ma, Yuhuan Yang, Chen Ju, Fei Zhang, Ya Zhang, Yanfeng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2093ed77c549eda95bd6f7212b735b43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2093ed77c549eda95bd6f7212b735b43-Abstract-Conference.html)

**Abstract**:

Open-vocabulary semantic segmentation is a challenging task that requires segmenting novel object categories at inference time. Recent works explore vision-language pre-training to handle this task, but suffer from unrealistic assumptions in practical scenarios, i.e., low-quality textual category names.For example, this paradigm assumes that new textual categories will be accurately and completely provided, and exist in lexicons during pre-training.However, exceptions often happen when meet with ambiguity for brief or incomplete names, new words that are not present in the pre-trained lexicons, and difficult-to-describe categories for users.To address these issues, this work proposes a novel attribute decomposition-aggregation framework, AttrSeg, inspired by human cognition in understanding new concepts. Specifically, in the decomposition stage, we decouple class names into diverse attribute descriptions to complement semantic contexts from multiple perspectives.Two attribute construction strategies are designed: using large language models for common categories, and involving manually labelling for human-invented categories. In the aggregation stage, we group diverse attributes into an integrated global description, to form a discriminative classifier that distinguishes the target object from others. One hierarchical aggregation architecture is further proposed to achieve multi-level aggregation, leveraging the meticulously designed clustering module.The final result is obtained by computing the similarity between aggregated attributes and images embedding.To evaluate the effectiveness, we annotate three datasets with attribute descriptions, and conduct extensive experiments and ablation studies. The results show the superior performance of attribute decomposition-aggregation.We refer readers to the latest arXiv version at https://arxiv.org/abs/2309.00096.

----

## [0] Stable and low-precision training for large-scale vision-language models

**Authors**: *Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract-Conference.html)

**Abstract**:

We introduce new methods for 1) accelerating and 2) stabilizing training for large language-vision models. 1) For acceleration, we introduce SwitchBack, a linear layer for int8 quantized training which provides a speed-up of 13-25% while matching the performance of bfloat16 training within 0.1 percentage points for the 1B parameter CLIP ViT-Huge---the largest int8 training to date. Our main focus is int8 as GPU support for float8 is rare, though we also analyze float8 training through simulation. While SwitchBack proves effective for float8, we show that standard techniques are also successful if the network is trained and initialized so that large feature magnitudes are discouraged, which we accomplish via layer-scale initialized with zeros. 2) For stability, we analyze loss spikes and find they consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator. As a result, we recommend an AdamW-Adafactor hybrid which avoids loss spikes when training a CLIP ViT-Huge model and outperforms gradient clipping at the scales we test.

----

## [0] Recommender Systems with Generative Retrieval

**Authors**: *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Mahesh Sathiamoorthy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)

**Abstract**:

Modern recommender systems perform large-scale retrieval by embedding queries and item candidates in the same unified space, followed by approximate nearest neighbor search to select top candidates given a query embedding. In this paper, we propose a novel generative retrieval approach, where the retrieval model autoregressively decodes the identifiers of the target candidates. To that end, we create semantically meaningful tuple of codewords to serve as a Semantic ID for each item. Given Semantic IDs for items in a user session, a Transformer-based sequence-to-sequence model is trained to predict the Semantic ID of the next item that the user will interact with. We show that recommender systems trained with the proposed paradigm significantly outperform the current SOTA models on various datasets. In addition, we show that incorporating Semantic IDs into the sequence-to-sequence model enhances its ability to generalize, as evidenced by the improved retrieval performance observed for items with no prior interaction history.

----

## [0] Active Vision Reinforcement Learning under Limited Visual Observability

**Authors**: *Jinghuan Shang, Michael S. Ryoo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20e6b4dd2b1f82bc599c593882f67f75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20e6b4dd2b1f82bc599c593882f67f75-Abstract-Conference.html)

**Abstract**:

In this work, we investigate Active Vision Reinforcement Learning (ActiveVision-RL), where an embodied agent simultaneously learns action policy for the task while also controlling its visual observations in partially observable environments. We denote the former as motor policy and the latter as sensory policy. For example, humans solve real world tasks by hand manipulation (motor policy) together with eye movements (sensory policy). ActiveVision-RL poses challenges on coordinating two policies given their mutual influence. We propose SUGARL, Sensorimotor Understanding Guided Active Reinforcement Learning, a framework that models motor and sensory policies separately, but jointly learns them using with an intrinsic sensorimotor reward. This learnable reward is assigned by sensorimotor reward module, incentivizes the sensory policy to select observations that are optimal to infer its own motor action, inspired by the sensorimotor stage of humans. Through a series of experiments, we show the effectiveness of our method across a range of observability conditions and its adaptability to existed RL algorithms. The sensory policies learned through our method are observed to exhibit effective active vision strategies.

----

## [0] Optimization of Inter-group criteria for clustering with minimum size constraints

**Authors**: *Eduardo Sany Laber, Lucas Murtinho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20f814ecdaa8c76131e21683447e347b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20f814ecdaa8c76131e21683447e347b-Abstract-Conference.html)

**Abstract**:

Internal measures that are used to assess the quality of a clustering usually take into account intra-group and/or inter-group criteria.There are many papers in the literature that propose algorithms with provable approximation guarantees for optimizing the former. However,  the optimization of inter-group criteria is much less understood.Here, we contribute to the state-of-the-art of this literature by devising algorithms with provable guarantees for the maximization of two natural inter-group criteria, namely the minimum spacing and the minimum spanning tree spacing. The former is the minimum distance between points in different groups while the latter captures separability through the cost of the minimum spanning tree that connects all groups. We obtain results for both the unrestricted case, in which no constraint on the clusters is imposed, and for the constrained case where each group is required to have a minimum number of points. Our constraint is motivated by the fact that the popular Single-Linkage, which optimizes both criteria in the unrestricted case, produces clustering with many tiny groups.To complement our work, we present an empirical study with 10 real datasets that provides evidence that our methods work very well in practical settings.

----

## [0] CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation

**Authors**: *Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, Joyce Chai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21293a43d0321c5602dd893be2c2332b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21293a43d0321c5602dd893be2c2332b-Abstract-Conference.html)

**Abstract**:

Diffusion models (DMs) have enabled breakthroughs in image synthesis tasks but lack an intuitive interface for consistent image-to-image (I2I) translation. Various methods have been explored to address this issue, including mask-based methods, attention-based methods, and image-conditioning. However, it remains a critical challenge to enable unpaired I2I translation with pre-trained DMs while maintaining satisfying consistency. This paper introduces Cyclenet, a novel but simple method that incorporates cycle consistency into DMs to regularize image manipulation. We validate Cyclenet on unpaired I2I tasks of different granularities. Besides the scene and object level translation, we additionally contribute a multi-domain I2I translation dataset to study the physical state changes of objects. Our empirical studies show that Cyclenet is superior in translation consistency and quality, and can generate high-quality images for out-of-domain distributions with a simple change of the textual prompt. Cyclenet is a practical framework, which is robust even with very limited training data (around 2k) and requires minimal computational resources (1 GPU) to train. Project homepage: https://cyclenetweb.github.io/

----

## [0] Bandit Social Learning under Myopic Behavior

**Authors**: *Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Aleksandrs Slivkins*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/212b143b5a5d6b88feb0fb1441b9756e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/212b143b5a5d6b88feb0fb1441b9756e-Abstract-Conference.html)

**Abstract**:

We study social learning dynamics motivated by reviews on online platforms. Theagents collectively follow a simple multi-armed bandit protocol, but each agentacts myopically, without regards to exploration. We allow a wide range of myopicbehaviors that are consistent with (parameterized) confidence intervals for the armsâ€™expected rewards. We derive stark exploration failures for any such behavior, andprovide matching positive results. As a special case, we obtain the first generalresults on failure of the greedy algorithm in bandits, thus providing a theoreticalfoundation for why bandit algorithms should explore.

----

## [0] Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

**Authors**: *Rainer Engelken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/214ce905bf2072535e34b3cf873cbbc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/214ce905bf2072535e34b3cf873cbbc8-Abstract-Conference.html)

**Abstract**:

Training recurrent neural networks (RNNs) remains a challenge due to the instability of gradients across long time horizons, which can lead to exploding and vanishing gradients. Recent research has linked these problems to the values of Lyapunov exponents for the forward-dynamics, which describe the growth or shrinkage of infinitesimal perturbations. Here, we propose gradient flossing, a novel approach to tackling gradient instability by pushing Lyapunov exponents of the forward dynamics toward zero during learning. We achieve this by regularizing Lyapunov exponents through backpropagation using differentiable linear algebra. This enables us to "floss" the gradients, stabilizing them and thus improving network training. We show that gradient flossing controls not only the gradient norm but also the condition number of the long-term Jacobian, facilitating multidimensional error feedback propagation. We find that applying gradient flossing before training enhances both the success rate and convergence speed for tasks involving long time horizons.For challenging tasks, we show that gradient flossing during training can further increase the time horizon that can be bridged by backpropagation through time. Moreover, we demonstrate the effectiveness of our approach on various RNN architectures and tasks of variable temporal complexity. Additionally, we provide a simple implementation of our gradient flossing algorithm that can be used in practice. Our results indicate that gradient flossing via regularizing Lyapunov exponents can significantly enhance the effectiveness of RNN training and mitigate the exploding and vanishing gradients problem.

----

## [0] How to Data in Datathons

**Authors**: *Carlos Mougan, Richard Plant, Clare Teng, Marya Bazzi, Alvaro Cabrejas Egea, Ryan Sze-Yin Chan, David Salvador Jasin, Martin Stoffel, Kirstie J. Whitaker, Jules Manser*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/215a55741fbe4baad173468f93336a7d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/215a55741fbe4baad173468f93336a7d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The rise of datathons, also known as data or data science hackathons, has provided a platform to collaborate, learn, and innovate quickly. Despite their significant potential benefits, organizations often struggle to effectively work with data due to a lack of clear guidelines and best practices for potential issues that might arise. Drawing on our own experiences and insights from organizing +80 datathon challenges with +60 partnership organizations since 2016, we provide a guide that serves as a resource for organizers to navigate the data-related complexities of datathons. We apply our proposed framework to 10 case studies.

----

## [0] Convolution Monge Mapping Normalization for learning on sleep data

**Authors**: *Théo Gnassounou, Rémi Flamary, Alexandre Gramfort*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html)

**Abstract**:

In many machine learning applications on signals and biomedical data, especially electroencephalogram (EEG), one major challenge is the variability of the data across subjects, sessions, and hardware devices. In this work, we propose a new method called Convolutional Monge Mapping Normalization ($\texttt{CMMN}$), which consists in filtering the signals in order to adapt their power spectrum density (PSD) to a Wasserstein barycenter estimated on training data. $\texttt{CMMN}$ relies on novel closed-form solutions for optimal transport mappings and barycenters and provides individual test time adaptation to new data without needing to retrain a prediction model. Numerical experiments on sleep EEG data show that $\texttt{CMMN}$ leads to significant and consistent performance gains independent from the neural network architecture when adapting between subjects, sessions, and even datasets collected with different hardware. Notably our performance gain is on par with much more numerically intensive Domain Adaptation (DA) methods and can be used in conjunction with those for even better performances.

----

## [0] Online learning of long-range dependencies

**Authors**: *Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, João Sacramento*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2184d8450c8a641f9a10c49279087c97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2184d8450c8a641f9a10c49279087c97-Abstract-Conference.html)

**Abstract**:

Online learning holds the promise of enabling efficient long-term credit assignment in recurrent neural networks. However, current algorithms fall short of offline backpropagation by either not being scalable or failing to learn long-range dependencies. Here we present a high-performance online learning algorithm that merely doubles the memory and computational requirements of a single inference pass. We achieve this by leveraging independent recurrent modules in multi-layer networks, an architectural motif that has recently been shown to be particularly powerful. Experiments on synthetic memory problems and on the challenging long-range arena benchmark suite reveal that our algorithm performs competitively, establishing a new standard for what can be achieved through online learning. This ability to learn long-range dependencies offers a new perspective on learning in the brain and opens a promising avenue in neuromorphic computing.

----

## [0] Fast Rank-1 Lattice Targeted Sampling for Black-box Optimization

**Authors**: *Yueming Lyu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/218d0323ce235090b43a1166159ee328-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/218d0323ce235090b43a1166159ee328-Abstract-Conference.html)

**Abstract**:

Black-box optimization has gained great attention for its success in recent applications.    However, scaling up to high-dimensional problems with good query efficiency remains challenging. This paper proposes a novel Rank-1 Lattice Targeted Sampling (RLTS) technique to address this issue. Our RLTS benefits from random rank-1 lattice Quasi-Monte Carlo, which enables us to perform fast local exact Gaussian processes (GP)  training and inference with $O(n \log n)$ complexity w.r.t. $n$ batch samples. Furthermore, we developed a fast coordinate searching method with $O(n \log n)$ time complexity for fast targeted sampling. The fast computation enables us to plug our RLTS into the sampling phase of stochastic optimization methods. This improves the query efficiency while  scaling up to higher dimensional problems than Bayesian optimization. Moreover,  to construct rank-1 lattices efficiently, we proposed a closed-form construction.  Extensive experiments on challenging benchmark test functions and black-box prompt fine-tuning for large language models demonstrate the query efficiency of our RLTS technique.

----

## [0] DreamHuman: Animatable 3D Avatars from Text

**Authors**: *Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Fieraru, Cristian Sminchisescu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21912f7057935149fa58408ee8cb460e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21912f7057935149fa58408ee8cb460e-Abstract-Conference.html)

**Abstract**:

We present \emph{DreamHuman}, a method to generate realistic animatable 3D human avatar models entirely from textual descriptions. Recent text-to-3D methods have made considerable strides in generation, but are still lacking in important aspects. Control and often spatial resolution remain limited, existing methods produce fixed rather than 3D human models that can be placed in different poses (i.e. re-posable or animatable), and anthropometric consistency for complex structures like people remains a challenge. \emph{DreamHuman} connects large text-to-image synthesis models, neural radiance fields, and statistical human body models in a novel optimization framework. This makes it possible to generate dynamic 3D human avatars with high-quality textures and learnt per-instance rigid and non rigid geometric deformations. We demonstrate that our method is capable to generate a wide variety of animatable, realistic 3D human models from text. These have diverse appearance, clothing, skin tones and body shapes, and outperform both generic text-to-3D approaches and previous text-based 3D avatar generators in visual fidelity.

----

## [0] Rank-1 Matrix Completion with Gradient Descent and Small Random Initialization

**Authors**: *Daesung Kim, Hye Won Chung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21c426323068204f4199c490d730e88e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21c426323068204f4199c490d730e88e-Abstract-Conference.html)

**Abstract**:

The nonconvex formulation of the matrix completion problem has received significant attention in recent years due to its affordable complexity compared to the convex formulation. Gradient Descent (GD) is a simple yet efficient baseline algorithm for solving nonconvex optimization problems. The success of GD has been witnessed in many different problems in both theory and practice when it is combined with random initialization. However, previous works on matrix completion require either careful initialization or regularizers to prove the convergence of GD. In this paper, we study the rank-1 symmetric matrix completion and prove that GD converges to the ground truth when small random initialization is used. We show that in a logarithmic number of iterations, the trajectory enters the region where local convergence occurs. We provide an upper bound on the initialization size that is sufficient to guarantee the convergence, and show that a larger initialization can be used as more samples are available. We observe that the implicit regularization effect of GD plays a critical role in the analysis, and for the entire trajectory, it prevents each entry from becoming much larger than the others.

----

## [0] No-Regret Learning in Dynamic Competition with Reference Effects Under Logit Demand

**Authors**: *Mengzi Amy Guo, Donghao Ying, Javad Lavaei, Zuo-Jun Max Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21c9fd36a6d94a491bf330a0ba0e5f6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21c9fd36a6d94a491bf330a0ba0e5f6e-Abstract-Conference.html)

**Abstract**:

This work is dedicated to the algorithm design in a competitive framework, with the primary goal of learning a stable equilibrium. We consider the dynamic price competition between two firms operating within an opaque marketplace, where each firm lacks information about its competitor. The demand follows the multinomial logit (MNL) choice model, which depends on the consumers' observed price and their reference price, and consecutive periods in the repeated games are connected by reference price updates. We use the notion of stationary Nash equilibrium (SNE), defined as the fixed point of the equilibrium pricing policy for the single-period game, to simultaneously capture the long-run market equilibrium and stability. We propose the online projected gradient ascent algorithm (OPGA), where the firms adjust prices using the first-order derivatives of their log-revenues that can be obtained from the market feedback mechanism. Despite the absence of typical properties required for the convergence of online games, such as strong monotonicity and variational stability, we demonstrate that under diminishing step-sizes, the price and reference price paths generated by OPGA converge to the unique SNE, thereby achieving the no-regret learning and a stable market. Moreover, with appropriate step-sizes, we prove that this convergence exhibits a rate of $\mathcal{O}(1/t)$.

----

## [0] VoxDet: Voxel Learning for Novel Instance Detection

**Authors**: *Bowen Li, Jiashun Wang, Yaoyu Hu, Chen Wang, Sebastian A. Scherer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21f1c5bbf2519321c1bee9bfa9edcd46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21f1c5bbf2519321c1bee9bfa9edcd46-Abstract-Conference.html)

**Abstract**:

Detecting unseen instances based on multi-view templates is a challenging problem due to its open-world nature. Traditional methodologies, which primarily rely on $2 \mathrm{D}$ representations and matching techniques, are often inadequate in handling pose variations and occlusions. To solve this, we introduce VoxDet, a pioneer 3D geometry-aware framework that fully utilizes the strong 3D voxel representation and reliable voxel matching mechanism. VoxDet first ingeniously proposes template voxel aggregation (TVA) module, effectively transforming multi-view 2D images into 3D voxel features. By leveraging associated camera poses, these features are aggregated into a compact 3D template voxel. In novel instance detection, this voxel representation demonstrates heightened resilience to occlusion and pose variations. We also discover that a $3 \mathrm{D}$ reconstruction objective helps to pre-train the 2D-3D mapping in TVA. Second, to quickly align with the template voxel, VoxDet incorporates a Query Voxel Matching (QVM) module. The 2D queries are first converted into their voxel representation with the learned 2D-3D mapping. We find that since the 3D voxel representations encode the geometry, we can first estimate the relative rotation and then compare the aligned voxels, leading to improved accuracy and efficiency. In addition to method, we also introduce the first instance detection benchmark, RoboTools, where 20 unique instances are video-recorded with camera extrinsic. RoboTools also provides 24 challenging cluttered scenarios with more than $9 \mathrm{k}$ box annotations. Exhaustive experiments are conducted on the demanding LineMod-Occlusion, YCB-video, and RoboTools benchmarks, where VoxDet outperforms various $2 \mathrm{D}$ baselines remarkably with faster speed. To the best of our knowledge, VoxDet is the first to incorporate implicit 3D knowledge for 2D novel instance detection tasks.

----

## [0] Evaluating and Inducing Personality in Pre-trained Language Models

**Authors**: *Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, Yixin Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html)

**Abstract**:

Standardized and quantified evaluation of machine behaviors is a crux of understanding LLMs. In this study, we draw inspiration from psychometric studies by leveraging human personality theory as a tool for studying machine behaviors. Originating as a philosophical quest for human behaviors, the study of personality delves into how individuals differ in thinking, feeling, and behaving. Toward building and understanding human-like social machines, we are motivated to ask: Can we assess machine behaviors by leveraging human psychometric tests in a **principled** and **quantitative** manner? If so, can we induce a specific personality in LLMs? To answer these questions, we introduce the Machine Personality Inventory (MPI) tool for studying machine behaviors; MPI follows standardizedpersonality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By systematically evaluating LLMs with MPI, we provide the first piece of evidence demonstrating the efficacy of MPI in studying LLMs behaviors. We further devise a Personality Prompting (P$^2$) method to induce LLMs with specific personalities in a **controllable** way, capable of producing diverse and verifiable behaviors. We hope this work sheds light on future studies by adopting personality as the essential indicator for various downstream tasks, and could further motivate research into equally intriguing human-like machine behaviors.

----

## [0] On Measuring Fairness in Generative Models

**Authors**: *Christopher T. H. Teo, Milad Abdollahzadeh, Ngai-Man Cheung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/220165f9c7f51163b73c8c7fff578b4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/220165f9c7f51163b73c8c7fff578b4e-Abstract-Conference.html)

**Abstract**:

Recently, there has been increased interest in fair generative models. In this work,we conduct, for the first time, an in-depth study on fairness measurement, acritical component in gauging progress on fair generative models. We make threecontributions. First, we conduct a study that reveals that the existing fairnessmeasurement framework has considerable measurement errors, even when highlyaccurate sensitive attribute (SA) classifiers are used. These findings cast doubtson previously reported fairness improvements. Second, to address this issue,we propose CLassifier Error-Aware Measurement (CLEAM), a new frameworkwhich uses a statistical model to account for inaccuracies in SA classifiers. Ourproposed CLEAM reduces measurement errors significantly, e.g., 4.98%â†’0.62%for StyleGAN2 w.r.t. Gender. Additionally, CLEAM achieves this with minimaladditional overhead. Third, we utilize CLEAM to measure fairness in importanttext-to-image generator and GANs, revealing considerable biases in these modelsthat raise concerns about their applications. Code and more resources: https://sutd-visual-computing-group.github.io/CLEAM/.

----

## [0] IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against Unauthorized Data Usage in Diffusion-Based Generative AI

**Authors**: *Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/222dda29587fbc2979ca99fd5ed00735-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/222dda29587fbc2979ca99fd5ed00735-Abstract-Conference.html)

**Abstract**:

Diffusion-based image generation models,  such as Stable Diffusion or DALLÂ·E 2, are able to learn from given images and generate high-quality samples following the guidance from prompts. For instance, they can be used to create artistic images that mimic the style of an artist based on his/her original artworks or to maliciously edit the original images for fake content. However, such ability also brings serious ethical issues without proper authorization from the owner of the original images. In response, several attempts have been made to protect the original images from such unauthorized data usage by adding imperceptible perturbations, which are designed to mislead the diffusion model and make it unable to properly generate new samples. In this work, we introduce a perturbation purification platform, named IMPRESS, to evaluate the effectiveness of imperceptible perturbations as a protective measure.IMPRESS is based on the key observation that imperceptible perturbations could lead to a perceptible inconsistency between the original image and the diffusion-reconstructed image, which can be used to devise a new optimization strategy for purifying the image, which may weaken the protection of the original image from unauthorized data usage (e.g., style mimicking, malicious editing).The proposed IMPRESS platform offers a comprehensive evaluation of several contemporary protection methods, and can be used as an evaluation platform for future protection methods.

----

## [0] Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

**Authors**: *Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2232e8fee69b150005ac420bfa83d705-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2232e8fee69b150005ac420bfa83d705-Abstract-Conference.html)

**Abstract**:

Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that RoCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, RoCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, RoCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.

----

## [0] Block-Coordinate Methods and Restarting for Solving Extensive-Form Games

**Authors**: *Darshan Chakrabarti, Jelena Diakonikolas, Christian Kroer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2280faacd674566a5eace1bd1098f507-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2280faacd674566a5eace1bd1098f507-Abstract-Conference.html)

**Abstract**:

Coordinate descent methods are popular in machine learning and optimization for their simple sparse updates and excellent practical performance. In the context of large-scale sequential game solving, these same properties would be attractive, but until now no such methods were known, because the strategy spaces do not satisfy the typical separable block structure exploited by such methods.We present the first cyclic coordinate-descent-like method for the polytope of sequence-form strategies, which form the strategy spaces for the players in an extensive-form game (EFG). Our method exploits the recursive structure of the proximal update induced by what are known as dilated regularizers, in order to allow for a pseudo block-wise update.We show that our method enjoys a O(1/T) convergence rate to a two-player zero-sum Nash equilibrium, while avoiding the worst-case polynomial scaling with the number of blocks common to cyclic methods. We empirically show that our algorithm usually performs better than other state-of-the-art first-order methods (i.e., mirror prox), and occasionally can even beat CFR$^+$, a state-of-the-art algorithm for numerical equilibrium computation in zero-sum EFGs. We then introduce a restarting heuristic for EFG solving. We show empirically that restarting can lead to speedups, sometimes huge, both for our cyclic method, as well as for existing methods such as mirror prox and predictive CFR$^+$.

----

## [0] ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction

**Authors**: *Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, Huiqi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html)

**Abstract**:

Most advanced unsupervised anomaly detection (UAD) methods rely on modeling feature representations of frozen encoder networks pre-trained on large-scale datasets, e.g. ImageNet. However, the features extracted from the encoders that are borrowed from natural image domains coincide little with the features required in the target UAD domain, such as industrial inspection and medical imaging. In this paper, we propose a novel epistemic UAD method, namely ReContrast, which optimizes the entire network to reduce biases towards the pre-trained image domain and orients the network in the target domain. We start with a feature reconstruction approach that detects anomalies from errors. Essentially, the elements of contrastive learning are elegantly embedded in feature reconstruction to prevent the network from training instability, pattern collapse, and identical shortcut, while simultaneously optimizing both the encoder and decoder on the target domain. To demonstrate our transfer ability on various image domains, we conduct extensive experiments across two popular industrial defect detection benchmarks and three medical image UAD tasks, which shows our superiority over current state-of-the-art methods.

----

## [0] Transferable Adversarial Robustness for Categorical Data via Universal Robust Embeddings

**Authors**: *Klim Kireev, Maksym Andriushchenko, Carmela Troncoso, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/22a25fc3da528794d52664dacc7bd470-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/22a25fc3da528794d52664dacc7bd470-Abstract-Conference.html)

**Abstract**:

Research on adversarial robustness is primarily focused on image and text data. Yet, many scenarios in which lack of robustness can result in serious risks, such as fraud detection, medical diagnosis, or recommender systems often do not rely on images or text but instead on tabular data. Adversarial robustness in tabular data poses two serious challenges. First, tabular datasets often contain categorical features, and therefore cannot be tackled directly with existing optimization procedures. Second, in the tabular domain, algorithms that are not based on deep networks are widely used and offer great performance, but algorithms to enhance robustness are tailored to neural networks (e.g. adversarial training).In this paper, we tackle both challenges. We present a method that allows us to train adversarially robust deep networks for tabular data and to transfer this robustness to other classifiers via universal robust embeddings tailored to categorical data. These embeddings, created using a bilevel alternating minimization framework, can be transferred to boosted trees or random forests making them robust without the need for adversarial training while preserving their high accuracy on tabular data. We show that our methods outperform existing techniques within a practical threat model suitable for tabular data.

----

## [0] Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection

**Authors**: *Chengsen Wang, Zirui Zhuang, Qi Qi, Jingyu Wang, Xingyu Wang, Haifeng Sun, Jianxin Liao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html)

**Abstract**:

Many unsupervised methods have recently been proposed for multivariate time series anomaly detection. However, existing works mainly focus on stable data yet often omit the drift generated from non-stationary environments, which may lead to numerous false alarms. We propose **D**ynamic **D**ecomposition with **D**iffusion **R**econstruction (D$^3$R), a novel anomaly detection network for real-world unstable data to fill the gap. D$^3$R tackles the drift via decomposition and reconstruction. In the decomposition procedure, we utilize data-time mix-attention to dynamically decompose long-period multivariate time series, overcoming the limitation of the local sliding window. The information bottleneck is critical yet difficult to determine in the reconstruction procedure. To avoid retraining once the bottleneck changes, we control it externally by noise diffusion and directly reconstruct the polluted data. The whole model can be trained end-to-end. Extensive experiments on various real-world datasets demonstrate that D$^3$R significantly outperforms existing methods, with a 11% average relative improvement over the previous SOTA models.

----

## [0] MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

**Authors**: *Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma, Zhiguang Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recently, Meta-Black-Box Optimization with Reinforcement Learning (MetaBBO-RL) has showcased the power of leveraging RL at the meta-level to mitigate manual fine-tuning of low-level black-box optimizers. However, this field is hindered by the lack of a unified benchmark. To fill this gap, we introduce MetaBox, the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. Our MetaBox is open-source and accessible at: https://github.com/GMC-DRL/MetaBox.

----

## [0] Improving Adversarial Robustness via Information Bottleneck Distillation

**Authors**: *Huafeng Kuang, Hong Liu, Yongjian Wu, Shin'ichi Satoh, Rongrong Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html)

**Abstract**:

Previous studies have shown that optimizing the information bottleneck can significantly improve the robustness of deep neural networks. Our study closely examines the information bottleneck principle and proposes an Information Bottleneck Distillation approach. This specially designed, robust distillation technique utilizes prior knowledge obtained from a robust pre-trained model to boost information bottlenecks.  Specifically, we propose two distillation strategies that align with the two optimization processes of the information bottleneck. Firstly, we use a robust soft-label distillation method to increase the mutual information between latent features and output prediction. Secondly, we introduce an adaptive feature distillation method that automatically transfers relevant knowledge from the teacher model to the student model, thereby reducing the mutual information between the input and latent features. We conduct extensive experiments to evaluate our approach's robustness against state-of-the-art adversarial attackers such as PGD-attack and AutoAttack. Our experimental results demonstrate the effectiveness of our approach in significantly improving adversarial robustness. Our code is available at https://github.com/SkyKuang/IBD.

----

## [0] Reading Relevant Feature from Global Representation Memory for Visual Object Tracking

**Authors**: *Xinyu Zhou, Pinxue Guo, Lingyi Hong, Jinglun Li, Wei Zhang, Weifeng Ge, Wenqiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2349293cb1bf2ce36d5c566f660f957e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2349293cb1bf2ce36d5c566f660f957e-Abstract-Conference.html)

**Abstract**:

Reference features from a template or historical frames are crucial for visual object tracking. Prior works utilize all features from a fixed template or memory for visual object tracking. However, due to the dynamic nature of videos, the required reference historical information for different search regions at different time steps is also inconsistent. Therefore, using all features in the template and memory can lead to redundancy and impair tracking performance. To alleviate this issue, we propose a novel tracking paradigm, consisting of a relevance attention mechanism and a global representation memory, which can adaptively assist the search region in selecting the most relevant historical information from reference features. Specifically, the proposed relevance attention mechanism in this work differs from previous approaches in that it can dynamically choose and build the optimal global representation memory for the current frame by accessing cross-frame information globally. Moreover, it can flexibly read the relevant historical information from the constructed memory to reduce redundancy and counteract the negative effects of harmful information. Extensive experiments validate the effectiveness of the proposed method, achieving competitive performance on five challenging datasets with 71 FPS.

----

## [0] Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks

**Authors**: *Eshaan Nichani, Alex Damian, Jason D. Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/236b6a814a1d2c0ff504ca7bf380f7ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/236b6a814a1d2c0ff504ca7bf380f7ff-Abstract-Conference.html)

**Abstract**:

One of the central questions in the theory of deep learning is to understand how neural networks learn hierarchical features. The ability of deep networks to extract salient features is crucial to both their outstanding generalization ability and the modern deep learning paradigm of pretraining and finetuneing. However, this feature learning process remains poorly understood from a theoretical perspective, with existing analyses largely restricted to two-layer networks. In this work we show that three-layer neural networks have provably richer feature learning capabilities than two-layer networks. We analyze the features learned by a three-layer network trained with layer-wise gradient descent, and present a general purpose theorem which upper bounds the sample complexity and width needed to achieve low test error when the target has specific hierarchical structure. We instantiate our framework in specific statistical learning settings -- single-index models and functions of quadratic features -- and show that in the latter setting three-layer networks obtain a sample complexity improvement over all existing guarantees for two-layer networks. Crucially, this sample complexity improvement relies on the ability of three-layer networks to efficiently learn nonlinear features. We then establish a concrete optimization-based depth separation by constructing a function which is efficiently learnable via gradient descent on a three-layer network, yet cannot be learned efficiently by a two-layer network. Our work makes progress towards understanding the provable benefit of three-layer neural networks over two-layer networks in the feature learning regime.

----

## [0] Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training

**Authors**: *Tiansheng Huang, Sihao Hu, Ka Ho Chow, Fatih Ilhan, Selim F. Tekin, Ling Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2376f25ef1725a9e3516ee3c86a59f46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2376f25ef1725a9e3516ee3c86a59f46-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is vulnerable to backdoor attacks due to its distributed computing nature. Existing defense solution usually requires larger amount of computation in either the training or testing phase, which limits their practicality in the resource-constrain scenarios. A more practical defense, i.e., neural network (NN) pruning based defense has been proposed in centralized backdoor setting. However, our empirical study shows that traditional pruning-based solution suffers \textit{poison-coupling} effect in FL, which significantly degrades the defense performance.This paper presents Lockdown, an isolated subspace training method to mitigate the poison-coupling effect. Lockdown follows three key procedures. First, it modifies the training protocol by isolating the training subspaces for different clients. Second, it utilizes randomness in initializing isolated subspacess, and performs subspace pruning and subspace recovery to segregate the subspaces between malicious and benign clients. Third, it introduces quorum consensus to cure the global model by purging malicious/dummy parameters. Empirical results show that Lockdown achieves \textit{superior} and \textit{consistent} defense performance compared to existing representative approaches against backdoor attacks. Another value-added property of Lockdown is the communication-efficiency and model complexity reduction, which are both critical for resource-constrain FL scenario. Our code is available at \url{https://github.com/git-disl/Lockdown}.

----

## [0] Robust Lipschitz Bandits to Adversarial Corruptions

**Authors**: *Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html)

**Abstract**:

Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

----

## [0] Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily

**Authors**: *Langzhang Liang, Xiangjing Hu, Zenglin Xu, Zixing Song, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23aa2163dea287441ebebc1295d5b3fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23aa2163dea287441ebebc1295d5b3fc-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have been shown to achieve remarkable performance on node classification tasks by exploiting both graph structures and node features. The majority of existing GNNs rely on the implicit homophily assumption. Recent studies have demonstrated that GNNs may struggle to model heterophilous graphs where nodes with different labels are more likely connected. To address this issue, we propose a generic GNN applicable to both homophilous and heterophilous graphs, namely Low-Rank Graph Neural Network (LRGNN). Our analysis demonstrates that a signed graph's global label relationship matrix has a low rank. This insight inspires us to predict the label relationship matrix by solving a robust low-rank matrix approximation problem, as prior research has proven that low-rank approximation could achieve perfect recovery under certain conditions. The experimental results reveal that the solution bears a strong resemblance to the label relationship matrix, presenting two advantages for graph modeling: a block diagonal structure and varying distributions of within-class and between-class entries.

----

## [0] Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts

**Authors**: *Ruth Dannenfelser, Jeffrey Zhong, Ran Zhang, Vicky Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23e3d86c9a19d0caf2ec997e73dfcfbd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/23e3d86c9a19d0caf2ec997e73dfcfbd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Many of the most commonly explored natural language processing (NLP) information extraction tasks can be thought of as evaluations of declarative knowledge, or fact-based information extraction. Procedural knowledge extraction, i.e., breaking down a described process into a series of steps, has received much less attention, perhaps in part due to the lack of structured datasets that capture the knowledge extraction process from end-to-end. To address this unmet need, we present FlaMBé (Flow annotations for Multiverse Biological entities), a collection of expert-curated datasets across a series of complementary tasks that capture procedural knowledge in biomedical texts. This dataset is inspired by the observation that one ubiquitous source of procedural knowledge that is described as unstructured text is within academic papers describing their methodology. The workflows annotated in FlaMBé are from texts in the burgeoning field of single cell research, a research area that has become notorious for the number of software tools and complexity of workflows used. Additionally, FlaMBé provides, to our knowledge, the largest manually curated named entity recognition (NER) and disambiguation (NED) datasets for tissue/cell type, a fundamental biological entity that is critical for knowledge extraction in the biomedical research domain. Beyond providing a valuable dataset to enable further development of NLP models for procedural knowledge extraction, automating the process of workflow mining also has important implications for advancing reproducibility in biomedical research.

----

## [0] RRHF: Rank Responses to Align Language Models with Human Feedback

**Authors**: *Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html)

**Abstract**:

Reinforcement Learning from Human Feedback (RLHF) facilitates the alignment of large language models with human preferences, significantly enhancing the quality of interactions between humans and models. InstructGPT implements RLHF through several stages, including Supervised Fine-Tuning (SFT), reward model training, and Proximal Policy Optimization (PPO). However, PPO is sensitive to hyperparameters and requires multiple models in its standard implementation, making it hard to train and scale up to larger parameter counts.In contrast, we propose a novel learning paradigm called RRHF, which scores sampled responses from different sources via a logarithm of conditional probabilities and learns to align these probabilities with human preferences through ranking loss.RRHF can leverage sampled responses from various sources including the model responses from itself, other large language model responses, and human expert responses to learn to rank them.RRHF only needs 1 to 2 models during tuning and can efficiently align language models with human preferences robustly without complex hyperparameter tuning. Additionally, RRHF can be considered an extension of SFT and reward model training while being simpler than PPO in terms of coding, model counts, and hyperparameters. We evaluate RRHF on the Helpful and Harmless dataset, demonstrating comparable alignment performance with PPO by reward model score and human labeling.Extensive experiments show that the performance of RRHF is highly related to sampling quality which suggests RRHF is a best-of-$n$ learner.

----

## [0] Sparsity-Preserving Differentially Private Training of Large Embedding Models

**Authors**: *Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chiyuan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23ff02034404b65776080cbf7148addd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23ff02034404b65776080cbf7148addd-Abstract-Conference.html)

**Abstract**:

As the use of large embedding models in recommendation systems and language applications increases, concerns over user data privacy have also risen.  DP-SGD, a training algorithm that combines differential privacy with stochastic gradient descent, has been the workhorse in protecting user privacy without compromising model accuracy by much. However, applying DP-SGD naively to embedding models can destroy gradient sparsity, leading to reduced training efficiency. To address this issue, we present two new algorithms, DP-FEST and DP-AdaFEST, that preserve gradient sparsity during the private training of large embedding models.  Our algorithms achieve substantial reductions ($10^6 \times$) in gradient size, while maintaining comparable levels of accuracy, on benchmark real-world datasets.

----

## [0] Bayesian target optimisation for high-precision holographic optogenetics

**Authors**: *Marcus Triplett, Marta Gajowa, Hillel Adesnik, Liam Paninski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html)

**Abstract**:

Two-photon optogenetics has transformed our ability to probe the structure and function of neural circuits. However, achieving precise optogenetic control of neural ensemble activity has remained fundamentally constrained by the problem of off-target stimulation (OTS): the inadvertent activation of nearby non-target neurons due to imperfect confinement of light onto target neurons. Here we propose a novel computational approach to this problem called Bayesian target optimisation. Our approach uses nonparametric Bayesian inference to model neural responses to optogenetic stimulation, and then optimises the laser powers and optical target locations needed to achieve a desired activity pattern with minimal OTS. We validate our approach in simulations and using data from in vitro experiments, showing that Bayesian target optimisation considerably reduces OTS across all conditions we test. Together, these results establish our ability to overcome OTS, enabling optogenetic stimulation with substantially improved precision.

----

## [0] From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models

**Authors**: *Roy Uziel, Or Dinari, Oren Freifeld*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/240cc9ac4789351653d13cfcba4ee85c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/240cc9ac4789351653d13cfcba4ee85c-Abstract-Conference.html)

**Abstract**:

In the task of semi-supervised video object segmentation, the input is the binary mask of an object in the first frame, and the desired output consists of the corresponding masks of that object in the subsequent frames. Existing leading solutions have two main drawbacks: 1) an expensive and typically-supervised training on videos; 2) a large memory footprint during inference. Here we present a training-free solution, with a low-memory footprint, that yields state-of-the-art results. The proposed method combines pre-trained deep learning-based features (trained on still images) with more classical methods for streaming-data clustering. Designed to adapt to temporal concept drifts and generalize to diverse video content without relying on annotated images or videos, the method eliminates the need for additional training or fine-tuning, ensuring fast inference and immediate applicability to new videos. Concretely, we represent an object via a dynamic ensemble of temporally- and spatially-coherent mixtures over a representation built from pre-trained ViT features and positional embeddings. A convolutional conditional random field further improves spatial coherence and helps reject outliers. We demonstrate the efficacy of the method on key benchmarks: the DAVIS-2017 and YouTube-VOS 2018 validation datasets. Moreover, by the virtue of the low-memory footprint of the compact cluster-based representation, the method scales gracefully to high-resolution ViT features. Our code is available at https://github.com/BGU-CS-VIL/Training-Free-VOS

----

## [0] How hard are computer vision datasets? Calibrating dataset difficulty to viewing time

**Authors**: *David Mayo, Jesse Cummings, Xinyu Lin, Dan Gutfreund, Boris Katz, Andrei Barbu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2432e9646556f3a98dc78c1f4a10481b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2432e9646556f3a98dc78c1f4a10481b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Humans outperform object recognizers despite the fact that models perform well on current datasets, including those explicitly designed to challenge machines with debiased images or distribution shift. This problem persists, in part, because we have no guidance on the absolute difficulty of an image or dataset making it hard to objectively assess progress toward human-level performance, to cover the range of human abilities, and to increase the challenge posed by a dataset. We develop a dataset difficulty metric MVT, Minimum Viewing Time, that addresses these three problems. Subjects view an image that flashes on screen and then classify the object in the image. Images that require brief flashes to recognize are easy, those which require seconds of viewing are hard. We compute the ImageNet and ObjectNet image difficulty distribution, which we find significantly undersamples hard images. Nearly 90% of current benchmark performance is derived from images that are easy for humans. Rather than hoping that we will make harder datasets, we can for the first time objectively guide dataset difficulty during development. We can also subset recognition performance as a function of difficulty: model performance drops precipitously while human performance remains stable. Difficulty provides a new lens through which to view model performance, one which uncovers new scaling laws: vision-language models stand out as being the most robust and human-like while all other techniques scale poorly. We release tools to automatically compute MVT, along with image sets which are tagged by difficulty. Objective image difficulty has practical applications – one can measure how hard a test set is before deploying a real-world system – and scientific applications such as discovering the neural correlates of image difficulty and enabling new object recognition techniques that eliminate the benchmark-vs- real-world performance gap.

----

## [0] What Knowledge Gets Distilled in Knowledge Distillation?

**Authors**: *Utkarsh Ojha, Yuheng Li, Anirudh Sundara Rajan, Yingyu Liang, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html)

**Abstract**:

Knowledge distillation aims to transfer useful information from a teacher network to a student network, with the primary goal of improving the student's performance for the task at hand. Over the years, there has a been a deluge of novel techniques and use cases of knowledge distillation. Yet, despite the various improvements, there seems to be a glaring gap in the community's fundamental understanding of the process. Specifically, what is the knowledge that gets distilled in knowledge distillation?  In other words, in what ways does the student become similar to the teacher? Does it start to localize objects in the same way? Does it get fooled by the same adversarial samples? Does its data invariance properties become similar? Our work presents a comprehensive study to try to answer these questions. We show that existing methods can indeed indirectly distill these properties beyond improving task performance. We further study why knowledge distillation might work this way, and show that our findings have practical implications as well.

----

## [0] Kernelized Cumulants: Beyond Kernel Mean Embeddings

**Authors**: *Patric Bonnier, Harald Oberhauser, Zoltán Szabó*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/243697ace81f57daef8737ff2c5cffd3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/243697ace81f57daef8737ff2c5cffd3-Abstract-Conference.html)

**Abstract**:

In $\mathbb{R}^d$, it is well-known that cumulants provide an alternative to moments that can achieve the same goals with numerous benefits such as lower variance estimators. In this paper we extend cumulants to reproducing kernel Hilbert spaces (RKHS) using tools from tensor algebras and show that they are computationally tractable by a kernel trick. These kernelized cumulants provide a new set of all-purpose statistics; the classical maximum mean discrepancy and Hilbert-Schmidt independence criterion arise as the degree one objects in our general construction. We argue both theoretically and empirically (on synthetic, environmental, and traffic data analysis) that going beyond degree one has several advantages and can be achieved with the same computational complexity and minimal overhead in our experiments.

----

## [0] Contrastive Training of Complex-Valued Autoencoders for Object Discovery

**Authors**: *Aleksandar Stanic, Anand Gopalakrishnan, Kazuki Irie, Jürgen Schmidhuber*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2439ec22091b9d6cfbebf3284b40116e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2439ec22091b9d6cfbebf3284b40116e-Abstract-Conference.html)

**Abstract**:

Current state-of-the-art object-centric models use slots and attention-based routing for binding. However, this class of models has several conceptual limitations: the number of slots is hardwired; all slots have equal capacity; training has high computational cost; there are no object-level relational factors within slots. Synchrony-based models in principle can address these limitations by using complex-valued activations which store binding information in their phase components. However, working examples of such synchrony-based models have been developed only very recently, and are still limited to toy grayscale datasets and simultaneous storage of less than three objects in practice. Here we introduce architectural modifications and a novel contrastive learning method that greatly improve the state-of-the-art synchrony-based model. For the first time, we obtain a class of synchrony-based models capable of discovering objects in an unsupervised manner in multi-object color datasets and simultaneously representing more than three objects.

----

## [0] Non-adversarial training of Neural SDEs with signature kernel scores

**Authors**: *Zacharia Issa, Blanka Horvath, Maud Lemercier, Cristopher Salvi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html)

**Abstract**:

Neural SDEs are continuous-time generative models for sequential data. State-of-the-art performance for irregular time series generation has been previously obtained by training these models adversarially as GANs. However, as typical for GAN architectures, training is notoriously unstable, often suffers from mode collapse, and requires specialised techniques such as weight clipping and gradient penalty to mitigate these issues. In this paper, we introduce a novel class of scoring rules on pathspace based on signature kernels and use them as objective for training Neural SDEs non-adversarially. By showing strict properness of such kernel scores and consistency of the corresponding estimators, we provide existence and uniqueness guarantees for the minimiser. With this formulation, evaluating the generator-discriminator pair amounts to solving a system of linear path-dependent PDEs which allows for memory-efficient adjoint-based backpropagation. Moreover, because the proposed kernel scores are well-defined for paths with values in infinite dimensional spaces of functions, our framework can be easily extended to generate spatiotemporal data. Our procedure significantly outperforms alternative ways of training Neural SDEs on a variety of tasks including the simulation of rough volatility models, the conditional probabilistic forecasts of real-world forex pairs where the conditioning variable is an observed past trajectory, and the mesh-free generation of limit order book dynamics.

----

## [0] Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

**Authors**: *Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2468f84a13ff8bb6767a67518fb596eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2468f84a13ff8bb6767a67518fb596eb-Abstract-Conference.html)

**Abstract**:

Text-to-Image diffusion models have made tremendous progress over the past two years, enabling the generation of highly realistic images based on open-domain text descriptions. However, despite their success, text descriptions often struggle to adequately convey detailed controls, even when composed of long and complex texts. Moreover, recent studies have also shown that these models face challenges in understanding such complex texts and generating the corresponding images. Therefore, there is a growing need to enable more control modes beyond text description. In this paper, we introduce Uni-ControlNet, a unified framework that allows for the simultaneous utilization of different local controls (e.g., edge maps, depth map, segmentation masks) and global controls (e.g., CLIP image embeddings) in a flexible and composable manner within one single model. Unlike existing methods, Uni-ControlNet only requires the fine-tuning of two additional adapters upon frozen pre-trained text-to-image diffusion models, eliminating the huge cost of training from scratch. Moreover, thanks to some dedicated adapter designs, Uni-ControlNet only necessitates a constant number (i.e., 2) of adapters, regardless of the number of local or global controls used. This not only reduces the fine-tuning costs and model size, making it more suitable for real-world deployment, but also facilitate composability of different conditions. Through both quantitative and qualitative comparisons, Uni-ControlNet demonstrates its superiority over existing methods in terms of controllability, generation quality and composability. Code is available at https://github.com/ShihaoZhaoZSH/Uni-ControlNet.

----

## [0] Loss Decoupling for Task-Agnostic Continual Learning

**Authors**: *Yan-Shuo Liang, Wu-Jun Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/249f73e01f0a2bb6c8d971b565f159a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/249f73e01f0a2bb6c8d971b565f159a7-Abstract-Conference.html)

**Abstract**:

Continual learning requires the model to learn multiple tasks in a sequential order. To perform continual learning, the model must possess the abilities to maintain performance on old tasks (stability) and adapt itself to learn new tasks (plasticity). Task-agnostic problem in continual learning is a challenging problem, in which task identities are not available in the inference stage and hence the model must learn to distinguish all the classes in all the tasks. In task-agnostic problem, the model needs to learn two new objectives for learning a new task, including distinguishing new classes from old classes and distinguishing between different new classes. For task-agnostic problem, replay-based methods are commonly used. These methods update the model with both saved old samples and new samples for continual learning. Most existing replay-based methods mix the two objectives in task-agnostic problem together, inhibiting the models from achieving a good trade-off between stability and plasticity. In this paper, we propose a simple yet effective method, called loss decoupling (LODE), for task-agnostic continual learning. LODE separates the two objectives for the new task by decoupling the loss of the new task. As a result, LODE can assign different weights for different objectives, which provides a way to obtain a better trade-off between stability and plasticity than those methods with coupled loss. Experiments show that LODE can outperform existing state-of-the-art replay-based methods on multiple continual learning datasets.

----

## [0] Federated Learning via Meta-Variational Dropout

**Authors**: *Insu Jeon, Minui Hong, Junhyeog Yun, Gunhee Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) aims to train a global inference model from remotely distributed clients, gaining popularity due to its benefit of improving data privacy. However, traditional FL often faces challenges in practical applications, including model overfitting and divergent local models due to limited and non-IID data among clients. To address these issues, we introduce a novel Bayesian meta-learning approach called meta-variational dropout (MetaVD). MetaVD learns to predict client-dependent dropout rates via a shared hypernetwork, enabling effective model personalization of FL algorithms in limited non-IID data settings. We also emphasize the posterior adaptation view of meta-learning and the posterior aggregation view of Bayesian FL via the conditional dropout posterior. We conducted extensive experiments on various sparse and non-IID FL datasets. MetaVD demonstrated excellent classification accuracy and uncertainty calibration performance, especially for out-of-distribution (OOD) clients. MetaVD compresses the local model parameters needed for each client, mitigating model overfitting and reducing communication costs. Code is available at https://github.com/insujeon/MetaVD.

----

## [0] Horospherical Decision Boundaries for Large Margin Classification in Hyperbolic Space

**Authors**: *Xiran Fan, Chun-Hao Yang, Baba C. Vemuri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24cb8b08f3cb2f59671e33faac4790e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24cb8b08f3cb2f59671e33faac4790e6-Abstract-Conference.html)

**Abstract**:

Hyperbolic spaces have been quite popular in the recent past for representing hierarchically organized data. Further, several classification algorithms for data in these spaces have been proposed in the literature. These algorithms mainly use either hyperplanes or geodesics for decision boundaries in a large margin classifiers setting leading to a non-convex optimization problem. In this paper, we propose a novel large margin classifier based on horospherical decision boundaries that leads to a geodesically convex optimization problem that can be optimized using any Riemannian gradient descent technique guaranteeing a globally optimal solution. We present several experiments depicting the competitive performance of our classifier in comparison to SOTA.

----

## [0] MIM4DD: Mutual Information Maximization for Dataset Distillation

**Authors**: *Yuzhang Shang, Zhihang Yuan, Yan Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24d36eee157559e0d2549455fba28f6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24d36eee157559e0d2549455fba28f6a-Abstract-Conference.html)

**Abstract**:

Dataset distillation (DD) aims to synthesize a small dataset whose test performance is comparable to a full dataset using the same model. State-of-the-art (SoTA) methods optimize synthetic datasets primarily by matching heuristic indicators extracted from two networks: one from real data and one from synthetic data (see Fig.1, Left), such as gradients and training trajectories. DD is essentially a compression problem that emphasizes on maximizing the preservation of information contained in the data. We argue that well-defined metrics which measure the amount of shared information between variables in information theory are necessary for success measurement, but are never considered by previous works. Thus, we introduce mutual information (MI) as the metric to quantify the shared information between the synthetic and the real datasets, and devise MIM4DD numerically maximizing the MI via a newly designed optimizable objective within a contrastive learning framework to update the synthetic dataset. Specifically, we designate the samples in different datasets who share the same labels as positive pairs, and vice versa negative pairs. Then we respectively pull and push those samples in positive and negative pairs into contrastive space via minimizing NCE loss. As a result, the targeted MI can be transformed into a lower bound represented by feature maps of samples, which is numerically feasible. Experiment results show that MIM4DD can be implemented as an add-on module to existing SoTA DD methods.

----

## [0] Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks

**Authors**: *Woojin Cho, Kookjin Lee, Donsub Rim, Noseong Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html)

**Abstract**:

In various engineering and applied science applications, repetitive numerical simulations of partial differential equations (PDEs) for varying input parameters are often required (e.g., aircraft shape optimization over many design parameters) and solvers are required to perform rapid execution. In this study, we suggest a path that potentially opens up a possibility for physics-informed neural networks (PINNs), emerging deep-learning-based solvers, to be considered as one such solver. Although PINNs have pioneered a proper integration of deep-learning and scientific computing, they require repetitive time-consuming training of neural networks, which is not suitable for many-query scenarios. To address this issue, we propose a lightweight low-rank PINNs containing only hundreds of model parameters and an associated hypernetwork-based meta-learning algorithm, which allows efficient approximation of solutions of PDEs for varying ranges of PDE input parameters. Moreover, we show that the proposed method is effective in overcoming a challenging issue, known as "failure modes" of PINNs.

----

## [0] Solving a Class of Non-Convex Minimax Optimization in Federated Learning

**Authors**: *Xidong Wu, Jianhui Sun, Zhengmian Hu, Aidong Zhang, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/254009e8d528f98764a060e877a1b01c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/254009e8d528f98764a060e877a1b01c-Abstract-Conference.html)

**Abstract**:

The minimax problems arise throughout machine learning applications, ranging from adversarial training and policy evaluation in reinforcement learning to AUROC maximization. To address the large-scale distributed data challenges across multiple clients with communication-efficient distributed training, federated learning (FL) is gaining popularity. Many optimization algorithms for minimax problems have been developed in the centralized setting (\emph{i.e.}, single-machine). Nonetheless, the algorithm for minimax problems under FL is still underexplored. In this paper, we study a class of federated nonconvex minimax optimization problems. We propose FL algorithms (FedSGDA+ and FedSGDA-M) and reduce existing complexity results for the most common minimax problems. For nonconvex-concave problems, we propose FedSGDA+ and reduce the communication complexity to $O(\varepsilon^{-6})$. Under nonconvex-strongly-concave and nonconvex-PL minimax settings, we prove that FedSGDA-M has the best-known sample complexity of $O(\kappa^{3} N^{-1}\varepsilon^{-3})$ and the best-known communication complexity of $O(\kappa^{2}\varepsilon^{-2})$. FedSGDA-M is the first algorithm to match the best sample complexity $O(\varepsilon^{-3})$ achieved by the single-machine method under the nonconvex-strongly-concave setting. Extensive experimental results on fair classification and AUROC maximization show the efficiency of our algorithms.

----

## [0] End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes

**Authors**: *Alexandre Maraval, Matthieu Zimmer, Antoine Grosnit, Haitham Bou-Ammar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html)

**Abstract**:

Meta-Bayesian optimisation (meta-BO) aims to improve the sample efficiency of Bayesian optimisation by leveraging data from related tasks. While previous methods successfully meta-learn either a surrogate model or an acquisition function independently, joint training of both components remains an open challenge. This paper proposes the first end-to-end differentiable meta-BO framework that generalises neural processes to learn acquisition functions via transformer architectures. We enable this end-to-end framework with reinforcement learning (RL) to tackle the lack of labelled acquisition data. Early on, we notice that training transformer-based neural processes from scratch with RL is challenging due to insufficient supervision, especially when rewards are sparse. We formalise this claim with a combinatorial analysis showing that the widely used notion of regret as a reward signal exhibits a logarithmic sparsity pattern in trajectory lengths. To tackle this problem, we augment the RL objective with an auxiliary task that guides part of the architecture to learn a valid probabilistic model as an inductive bias. We demonstrate that our method achieves state-of-the-art regret results against various baselines in experiments on standard hyperparameter optimisation tasks and also outperforms others in the real-world problems of mixed-integer programming tuning, antibody design, and logic synthesis for electronic design automation.

----

## [0] Contextual Bandits and Imitation Learning with Preference-Based Active Queries

**Authors**: *Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2567c95fd41459a98a73ba893775d22a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2567c95fd41459a98a73ba893775d22a-Abstract-Conference.html)

**Abstract**:

We consider the problem of contextual bandits and imitation learning, where the learner lacks direct knowledge of the executed action's reward. Instead, the learner can actively request the expert at each round to compare two actions and receive noisy preference feedback. The learner's objective is two-fold: to minimize regret associated with the executed actions, while simultaneously, minimizing the number of comparison queries made to the expert. In this paper, we assume that the learner has access to a function class that can represent the expert's preference model under appropriate link functions and present an algorithm that leverages an online regression oracle with respect to this function class. For the contextual bandit setting, our algorithm achieves a regret bound that combines the best of both worlds, scaling as $O(\min\\{\sqrt{T}, d/\Delta\\})$, where $T$ represents the number of interactions, $d$ represents the eluder dimension of the function class, and $\Delta$ represents the minimum preference of the optimal action over any suboptimal action under all contexts. Our algorithm does not require the knowledge of $\Delta$, and the obtained regret bound is comparable to what can be achieved in the standard contextual bandits setting where the learner observes reward signals at each round. Additionally, our algorithm makes only $O(\min\\{T, d^2/\Delta^2\\})$ queries to the expert. We then extend our algorithm to the imitation learning setting, where the agent engages with an unknown environment in episodes of length $H$, and provide similar guarantees regarding regret and query complexity. Interestingly, with preference-based feedback, our imitation learning algorithm can learn a policy outperforming a sub-optimal expert, matching the result from interactive imitation learning algorithms [Ross and Bagnell, 2014] that require access to the expert's actions and also reward signals.

----

## [0] Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding

**Authors**: *George Ma, Yifei Wang, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/257b3a7438b1f3709e91a86adf2fdc0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/257b3a7438b1f3709e91a86adf2fdc0a-Abstract-Conference.html)

**Abstract**:

Spectral embedding is a powerful graph embedding technique that has received a lot of attention recently due to its effectiveness on Graph Transformers. However, from a theoretical perspective, the universal expressive power of spectral embedding comes at the price of losing two important invariance properties of graphs, sign and basis invariance, which also limits its effectiveness on graph data. To remedy this issue, many previous methods developed costly approaches to learn new invariants and suffer from high computation complexity. In this work, we explore a minimal approach that resolves the ambiguity issues by directly finding canonical directions for the eigenvectors, named Laplacian Canonization (LC). As a pure pre-processing method, LC is light-weighted and can be applied to any existing GNNs. We provide a thorough investigation, from theory to algorithm, on this approach, and discover an efficient algorithm named Maximal Axis Projection (MAP) that works for both sign and basis invariance and successfully canonizes more than 90\% of all eigenvectors. Experiments on real-world benchmark datasets like ZINC, MOLTOX21, and MOLPCBA show that MAP consistently outperforms existing methods while bringing minimal computation overhead. Code is available at https://github.com/PKU-ML/LaplacianCanonization.

----



[Go to the previous page](NIPS-2023-list400.md)

[Go to the next page](NIPS-2023-list402.md)

[Go to the catalog section](README.md)