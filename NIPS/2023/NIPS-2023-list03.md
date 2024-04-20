## [400] Adaptive Topological Feature via Persistent Homology: Filtration Learning for Point Clouds

        **Authors**: *Naoki Nishikawa, Yuichi Ike, Kenji Yamanishi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d49235669869ab737c1da9d64b7c769-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d49235669869ab737c1da9d64b7c769-Abstract-Conference.html)

        **Abstract**:

        Machine learning for point clouds has been attracting much attention, with many applications in various fields, such as shape recognition and material science. For enhancing the accuracy of such machine learning methods, it is often effective to incorporate global topological features, which are typically extracted by persistent homology. In the calculation of persistent homology for a point cloud, we choose a filtration for the point cloud, an increasing sequence of spaces. Since the performance of machine learning methods combined with persistent homology is highly affected by the choice of a filtration, we need to tune it depending on data and tasks. In this paper, we propose a framework that learns a filtration adaptively with the use of neural networks. In order to make the resulting persistent homology isometry-invariant, we develop a neural network architecture with such invariance. Additionally, we show a theoretical result on a finite-dimensional approximation of filtration functions, which justifies the proposed network architecture. Experimental results demonstrated the efficacy of our framework in several classification tasks.

        ----

        ## [401] Accurate Interpolation for Scattered Data through Hierarchical Residual Refinement

        **Authors**: *Shizhe Ding, Boyang Xia, Dongbo Bu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5a92867cf463fad136cfa23395840b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5a92867cf463fad136cfa23395840b-Abstract-Conference.html)

        **Abstract**:

        Accurate interpolation algorithms are highly desired in various theoretical and engineering scenarios. Unlike the traditional numerical algorithms that have exact zero-residual constraints on observed points, the neural network-based interpolation methods exhibit non-zero residuals at these points. These residuals, which provide observations of an underlying residual function, can guide predicting interpolation functions, but have not been exploited by the existing approaches. To fill this gap, we propose Hierarchical INTerpolation Network (HINT), which utilizes the residuals on observed points to guide target function estimation in a hierarchical fashion. HINT consists of several sequentially arranged lightweight interpolation blocks. The first interpolation block estimates the main component of the target function, while subsequent blocks predict the residual components using observed points residuals of the preceding blocks. The main component and residual components are accumulated to form the final interpolation results. Furthermore, under the assumption that finer residual prediction requires a more focused attention range on observed points, we utilize hierarchical local constraints in correlation modeling between observed and target points. Extensive experiments demonstrate that HINT outperforms existing interpolation algorithms significantly in terms of interpolation accuracy across a wide variety of datasets, which underscores its potential for practical scenarios.

        ----

        ## [402] Learning Universal Policies via Text-Guided Video Generation

        **Authors**: *Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, Pieter Abbeel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html)

        **Abstract**:

        A goal of artificial intelligence is to construct an agent that can solve a wide variety of tasks. Recent progress in text-guided image synthesis has yielded models with an impressive ability to generate complex novel images, exhibiting combinatorial generalization across domains. Motivated by this success, we investigate whether such tools can be used to construct more general-purpose agents. Specifically, we cast the sequential decision making problem as a text-conditioned video generation problem, where, given a text-encoded specification of a desired goal, a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video. By leveraging text as the underlying goal specification, we are able to naturally and combinatorially generalize to novel goals. The proposed policy-as-video formulation can further represent environments with different state and action spaces in a unified space of images, which, for example, enables learning and generalization across a variety of robot manipulation tasks. Finally, by leveraging pretrained language embeddings and widely available videos from the internet, the approach enables knowledge transfer through predicting highly realistic video plans for real robots.

        ----

        ## [403] Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming

        **Authors**: *Jacobus G. M. van der Linden, Mathijs de Weerdt, Emir Demirovic*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d5fce9627e15c84db572a66e029b1fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d5fce9627e15c84db572a66e029b1fc-Abstract-Conference.html)

        **Abstract**:

        Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue.Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees.We explore this relationship in detail and show the necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints.Experiments on five application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.

        ----

        ## [404] Polyhedron Attention Module: Learning Adaptive-order Interactions

        **Authors**: *Tan Zhu, Fei Dou, Xinyu Wang, Jin Lu, Jinbo Bi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d83ad88759cef8192451543e5d59bf6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d83ad88759cef8192451543e5d59bf6-Abstract-Conference.html)

        **Abstract**:

        Learning feature interactions can be the key for multivariate predictive modeling. ReLU-activated neural networks create piecewise linear prediction models, and other nonlinear activation functions lead to models with only high-order feature interactions. Recent methods incorporate candidate polynomial terms of fixed orders into deep learning, which is subject to the issue of combinatorial explosion, or learn the orders that are difficult to adapt to different regions of the feature space. We propose a Polyhedron Attention Module (PAM) to create piecewise polynomial models where the input space is split into polyhedrons which define the different pieces and on each piece the hyperplanes that define the polyhedron boundary multiply to form the interactive terms, resulting in interactions of adaptive order to each piece. PAM is interpretable to identify important interactions in predicting a target. Theoretic analysis shows that PAM has stronger expression capability than ReLU-activated networks. Extensive experimental results demonstrate the superior classification performance of PAM on massive datasets of the click-through rate prediction and PAM can learn meaningful interaction effects in a medical problem.

        ----

        ## [405] Faster Query Times for Fully Dynamic k-Center Clustering with Outliers

        **Authors**: *Leyla Biabani, Annika Hennes, Morteza Monemizadeh, Melanie Schmidt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d8e261c241aa72f9b4a02af7f52587e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d8e261c241aa72f9b4a02af7f52587e-Abstract-Conference.html)

        **Abstract**:

        Given a point set $P\subseteq M$ from a metric space $(M,d)$ and numbers $k, z \in N$, the *metric $k$-center problem with $z$ outliers* is to find a set $C^\ast\subseteq P$ of $k$ points such that the maximum distance of all but at most $z$ outlier points of $P$ to their nearest center in ${C}^\ast$ is minimized. We consider this problem in the fully dynamic model, i.e., under insertions and deletions of points, for the case that the metric space has a bounded doubling dimension $dim$. We utilize a hierarchical data structure to maintain the points and their neighborhoods, which enables us to efficiently find the clusters. In particular, our data structure can be queried at any time to generate a $(3+\varepsilon)$-approximate solution for input values of $k$ and $z$ in worst-case query time $\varepsilon^{-O(dim)}k \log{n} \log\log{\Delta}$, where $\Delta$ is the ratio between the maximum and minimum distance between two points in $P$. Moreover, it allows insertion/deletion of a point in worst-case update time $\varepsilon^{-O(dim)}\log{n}\log{\Delta}$. Our result achieves a significantly faster query time with respect to $k$ and $z$ than the current state-of-the-art by Pellizzoni, Pietracaprina, and Pucci, which uses $\varepsilon^{-O(dim)}(k+z)^2\log{\Delta}$ query time to obtain a $(3+\varepsilon)$-approximation.

        ----

        ## [406] Natural Language Instruction-following with Task-related Language Development and Translation

        **Authors**: *Jing-Cheng Pang, Xinyu Yang, Si-Hang Yang, Xiong-Hui Chen, Yang Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dc2fe8d9ae956616f86bab3ce5edc59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dc2fe8d9ae956616f86bab3ce5edc59-Abstract-Conference.html)

        **Abstract**:

        Natural language-conditioned reinforcement learning (RL) enables agents to follow human instructions. Previous approaches generally implemented language-conditioned RL by providing the policy with human instructions in natural language (NL) and training the policy to follow instructions. In this is outside-in approach, the policy must comprehend the NL and manage the task simultaneously. However, the unbounded NL examples often bring much extra complexity for solving concrete RL tasks, which can distract policy learning from completing the task. To ease the learning burden of the policy, we investigate an inside-out scheme for natural language-conditioned RL by developing a task language (TL) that is task-related and easily understood by the policy, thus reducing the policy learning burden. Besides, we employ a translator to translate natural language into the TL, which is used in RL to achieve efficient policy training. We implement this scheme as TALAR (TAsk Language with predicAte Representation) that learns multiple predicates to model object relationships as the TL. Experiments indicate that TALAR not only better comprehends NL instructions but also leads to a better instruction-following policy that significantly improves the success rate over baselines and adapts to unseen expressions of NL instruction. Besides, the TL is also an effective sub-task abstraction compatible with hierarchical RL.

        ----

        ## [407] Convergence of Actor-Critic with Multi-Layer Neural Networks

        **Authors**: *Haoxing Tian, Alex Olshevsky, Yannis Paschalidis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html)

        **Abstract**:

        The early theory of actor-critic methods considered convergence using linear function approximators for the policy and value functions. Recent work has established convergence using neural network approximators with a single hidden layer. In this work we are taking the natural next step and establish convergence using deep neural networks with an arbitrary number of hidden layers, thus closing a gap between theory and practice. We show that actor-critic updates projected on a ball around the initial condition will converge to a neighborhood where the average of the squared gradients is $\tilde{O} \left( 1/\sqrt{m} \right) + O \left( \epsilon \right)$, with $m$ being the width of the neural network and $\epsilon$ the approximation quality of the best critic neural network over the projected set.

        ----

        ## [408] Percentile Criterion Optimization in Offline Reinforcement Learning

        **Authors**: *Cyrus Cousins, Elita A. Lobo, Marek Petrik, Yair Zick*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1dec73169509c223220744b2c9b2df37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1dec73169509c223220744b2c9b2df37-Abstract-Conference.html)

        **Abstract**:

        In reinforcement learning, robust policies for high-stakes decision-making problems with limited data are usually computed by optimizing the percentile criterion. The percentile criterion is optimized by constructing an uncertainty set that contains the true model with high probability and optimizing the policy for the worst model in the set. Since the percentile criterion is non-convex, constructing these sets itself is challenging. Existing works use Bayesian credible regions as uncertainty sets, but they are often unnecessarily large and result in learning overly conservative policies. To overcome these shortcomings, we propose a novel Value-at-Risk based dynamic programming algorithm to optimize the percentile criterion without explicitly constructing any uncertainty sets. Our theoretical and empirical results show that our algorithm implicitly constructs much smaller uncertainty sets and learns less-conservative robust policies.

        ----

        ## [409] TextDiffuser: Diffusion Models as Text Painters

        **Authors**: *Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1df4afb0b4ebf492a41218ce16b6d8df-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have gained increasing attention for their impressive generation abilities but currently struggle with rendering accurate and coherent text. To address this issue, we introduce TextDiffuser, focusing on generating images with visually appealing text that is coherent with backgrounds. TextDiffuser consists of two stages: first, a Transformer model generates the layout of keywords extracted from text prompts, and then diffusion models generate images conditioned on the text prompt and the generated layout. Additionally, we contribute the first large-scale text images dataset with OCR annotations, MARIO-10M, containing 10 million image-text pairs with text recognition, detection, and character-level segmentation annotations. We further collect the MARIO-Eval benchmark to serve as a comprehensive tool for evaluating text rendering quality. Through experiments and user studies, we demonstrate that TextDiffuser is flexible and controllable to create high-quality text images using text prompts alone or together with text template images, and conduct text inpainting to reconstruct incomplete images with text. We will make the code, model and dataset publicly available.

        ----

        ## [410] Object-centric Learning with Cyclic Walks between Parts and Whole

        **Authors**: *Ziyu Wang, Mike Zheng Shou, Mengmi Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html)

        **Abstract**:

        Learning object-centric representations from complex natural environments enables both humans and machines with reasoning abilities from low-level perceptual features. To capture compositional entities of the scene, we proposed cyclic walks between perceptual features extracted from vision transformers and object entities. First, a slot-attention module interfaces with these perceptual features and produces a finite set of slot representations. These slots can bind to any object entities in the scene via inter-slot competitions for attention. Next, we establish entity-feature correspondence with cyclic walks along high transition probability based on the pairwise similarity between perceptual features (aka "parts") and slot-binded object representations (aka "whole"). The whole is greater than its parts and the parts constitute the whole. The part-whole interactions form cycle consistencies, as supervisory signals, to train the slot-attention module. Our rigorous experiments on \textit{seven} image datasets in \textit{three} \textit{unsupervised} tasks demonstrate that the networks trained with our cyclic walks can disentangle foregrounds and backgrounds, discover objects, and segment semantic objects in complex scenes. In contrast to object-centric models attached with a decoder for the pixel-level or feature-level reconstructions, our cyclic walks provide strong learning signals, avoiding computation overheads and enhancing memory efficiency. Our source code and data are available at: \href{https://github.com/ZhangLab-DeepNeuroCogLab/Parts-Whole-Object-Centric-Learning/}{link}.

        ----

        ## [411] Experiment Planning with Function Approximation

        **Authors**: *Aldo Pacchiano, Jonathan Lee, Emma Brunskill*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e0d9f30c100129259f66660403fb1e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e0d9f30c100129259f66660403fb1e2-Abstract-Conference.html)

        **Abstract**:

        We study the problem of experiment planning with function approximation in contextual bandit problems. In settings where there is a significant overhead to deploying adaptive algorithms---for example, when the execution of the data collection policies is required to be distributed, or a human in the loop is needed to implement these policies---producing in advance a set of policies for data collection is paramount. We study the setting where a large dataset of contexts but not rewards is available and may be used by the learner to design an effective data collection strategy. Although when rewards are linear this problem has been well studied, results are still missing for more complex reward models. In this work we propose two experiment planning strategies compatible with function approximation. The first is an eluder planning and sampling procedure that can recover optimality guarantees depending on the eluder dimension of the reward function class. For the second, we show that a uniform sampler achieves competitive optimality rates in the setting where the number of actions is small. We finalize our results introducing a statistical gap fleshing out the fundamental differences between planning and adaptive learning and provide results for planning with model selection.

        ----

        ## [412] White-Box Transformers via Sparse Rate Reduction

        **Authors**: *Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Benjamin D. Haeffele, Yi Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e118ba9ee76c20df728b42a35fb4704-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e118ba9ee76c20df728b42a35fb4704-Abstract-Conference.html)

        **Abstract**:

        In this paper, we contend that the objective of representation learning is  to compress and transform the distribution of the data, say sets of tokens, towards a mixture of low-dimensional Gaussian distributions supported on incoherent subspaces. The quality of the final representation can be measured by a unified objective function called sparse rate reduction. From this perspective, popular deep networks such as transformers can be naturally viewed as realizing iterative schemes to optimize this objective incrementally. Particularly, we show that the standard transformer block can be derived from alternating optimization on complementary parts of this objective: the multi-head self-attention operator can be viewed as a gradient descent step to compress the token sets by minimizing their lossy coding rate, and the subsequent multi-layer perceptron can be viewed as attempting to sparsify the representation of the tokens. This leads to a family of white-box transformer-like deep network architectures which are mathematically fully interpretable. Despite their simplicity, experiments show that these networks indeed learn to optimize the designed objective: they compress and sparsify representations of large-scale real-world vision datasets such as ImageNet, and achieve performance very close to thoroughly engineered transformers such as ViT. Code is at https://github.com/Ma-Lab-Berkeley/CRATE.

        ----

        ## [413] Task-Robust Pre-Training for Worst-Case Downstream Adaptation

        **Authors**: *Jianghui Wang, Yang Chen, Xingyu Xie, Cong Fang, Zhouchen Lin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e4322fddd833f83c855660ac65e428d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e4322fddd833f83c855660ac65e428d-Abstract-Conference.html)

        **Abstract**:

        Pre-training has achieved remarkable success when transferred to downstream tasks. In machine learning, we care about not only the good performance of a model but also its behavior under reasonable shifts of condition. The same philosophy holds when pre-training a foundation model. However,  the foundation model may not uniformly behave well for a series of related downstream tasks. This happens, for example, when conducting mask recovery regression where the recovery ability or the training instances diverge like pattern features are extracted dominantly on pre-training, but semantic features are also required on a downstream task. This paper considers pre-training a model that guarantees a uniformly good performance over the downstream tasks. We call this goal as downstream-task robustness.Our method first separates the upstream task into several representative ones and applies a simple minimax loss for pre-training. We then design an efficient algorithm to solve the minimax lossand prove its convergence in the convex setting. In the experiments, we show both on large-scale natural language processing and computer vision datasets our method increases the metrics on worse-case downstream tasks. Additionally, some theoretical explanations for why our loss is beneficial are provided. Specifically, we show fewer samples are inherently required for the most challenging downstream task in some cases.

        ----

        ## [414] Inconsistency, Instability, and Generalization Gap of Deep Neural Network Training

        **Authors**: *Rie Johnson, Tong Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e58b1bf9f218fcd19e4539e982752a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e58b1bf9f218fcd19e4539e982752a5-Abstract-Conference.html)

        **Abstract**:

        As deep neural networks are highly expressive, it is important to find solutions with small generalization gap (the difference between the performance on the training data and unseen data).  Focusing on the stochastic nature of training, we first present a theoretical analysis in which the bound of generalization gap depends on what we call inconsistency and instability of model outputs, which can be estimated on unlabeled data.  Our empirical study based on this analysis shows that instability and inconsistency are strongly predictive of generalization gap in various settings.  In particular, our finding indicates that inconsistency is a more reliable indicator of generalization gap than the sharpness of the loss landscape.  Furthermore, we show that algorithmic reduction of inconsistency leads to superior performance.  The results also provide a theoretical basis for existing methods such as co-distillation and ensemble.

        ----

        ## [415] Neural approximation of Wasserstein distance via a universal architecture for symmetric and factorwise group invariant functions

        **Authors**: *Samantha Chen, Yusu Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e5f58d98523298cba093f658cfdf2d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e5f58d98523298cba093f658cfdf2d6-Abstract-Conference.html)

        **Abstract**:

        Learning distance functions between complex objects, such as the Wasserstein distance to compare point sets, is a common goal in machine learning applications. However, functions on such complex objects (e.g., point sets and graphs) are often required to be invariant to a wide variety of group actions e.g. permutation or rigid transformation. Therefore, continuous and symmetric *product* functions (such as distance functions) on such complex objects must also be invariant to the *product* of such group actions. We call these functions symmetric and factor-wise group invariant functions (or SGFI functions} in short).In this paper, we first present a general neural network architecture for approximating SFGI functions. The main contribution of this paper combines this general NN with a sketching idea in order to develop a specific and efficient neural network which can approximate the $p$-th Wasserstein distance between point sets.Very importantly, the required model complexity is *independent* of the sizes of input point sets. On the theoretical front, to the best of our knowledge, this is the first result showing that there exists a neural network with the capacity to approximate Wasserstein distance with bounded model complexity. Our work provides an interesting integration of sketching ideas for geometric problems with universal approximation of symmetric functions. On the empirical front, we present a range of results showing that our newly proposed neural network architecture performs comparatively or better than other models (including a SOTA Siamese Autoencoder based approach). In particular, our NN generalizes significantly better and trains much faster than the SOTA Siamese AE.Finally, this line of investigation could be useful in exploring effective neural network design for solving a broad range of geometric optimization problems (e.g., $k$-means in a metric space).

        ----

        ## [416] Revisiting the Evaluation of Image Synthesis with GANs

        **Authors**: *Mengping Yang, Ceyuan Yang, Yichi Zhang, Qingyan Bai, Yujun Shen, Bo Dai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e5fa672b2c35744cbcfacb2e77f1cb0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e5fa672b2c35744cbcfacb2e77f1cb0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        A good metric, which promises a reliable comparison between solutions, is essential for any well-defined task. Unlike most vision tasks that have per-sample ground-truth, image synthesis tasks target generating unseen data and hence are usually evaluated through a distributional distance between one set of real samples and another set of generated samples. This study presents an empirical investigation into the evaluation of synthesis performance, with generative adversarial networks (GANs) as a representative of generative models. In particular, we make in-depth analyses of various factors, including how to represent a data point in the representation space, how to calculate a fair distance using selected samples, and how many instances to use from each set. Extensive experiments conducted on multiple datasets and settings reveal several important findings. Firstly, a group of models that include both CNN-based and ViT-based architectures serve as reliable and robust feature extractors for measurement evaluation. Secondly, Centered Kernel Alignment (CKA) provides a better comparison across various extractors and hierarchical layers in one model. Finally, CKA is more sample-efficient and enjoys better agreement with human judgment in characterizing the similarity between two internal data correlations. These findings contribute to the development of a new measurement system, which enables a consistent and reliable re-evaluation of current state-of-the-art generative models.

        ----

        ## [417] What Do Deep Saliency Models Learn about Visual Attention?

        **Authors**: *Shi Chen, Ming Jiang, Qi Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e680f115a22d60cbc228a0c6dae5936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e680f115a22d60cbc228a0c6dae5936-Abstract-Conference.html)

        **Abstract**:

        In recent years, deep saliency models have made significant progress in predicting human visual attention. However, the mechanisms behind their success remain largely unexplained due to the opaque nature of deep neural networks. In this paper, we present a novel analytic framework that sheds light on the implicit features learned by saliency models and provides principled interpretation and quantification of their contributions to saliency prediction. Our approach decomposes these implicit features into interpretable bases that are explicitly aligned with semantic attributes and reformulates saliency prediction as a weighted combination of probability maps connecting the bases and saliency. By applying our framework, we conduct extensive analyses from various perspectives, including the positive and negative weights of semantics, the impact of training data and architectural designs, the progressive influences of fine-tuning, and common error patterns of state-of-the-art deep saliency models. Additionally, we demonstrate the effectiveness of our framework by exploring visual attention characteristics in various application scenarios, such as the atypical attention of people with autism spectrum disorder, attention to emotion-eliciting stimuli, and attention evolution over time. Our code is publicly available at \url{https://github.com/szzexpoi/saliency_analysis}.

        ----

        ## [418] Three Iterations of (d - 1)-WL Test Distinguish Non Isometric Clouds of d-dimensional Points

        **Authors**: *Valentino Delle Rose, Alexander Kozachinskiy, Cristobal Rojas, Mircea Petrache, Pablo Barceló*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e6cf8f77bd8e907f53babcd7664c710-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e6cf8f77bd8e907f53babcd7664c710-Abstract-Conference.html)

        **Abstract**:

        The Weisfeiler-Lehman (WL) test is a fundamental iterative algorithm for checking the isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of Euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud. Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.

        ----

        ## [419] Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving

        **Authors**: *Sepidehsadat (Sepid) Hossieni, Mohammad Amin Shabani, Saghar Irandoust, Yasutaka Furukawa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e70ac91ad26ba5b24cf11b12a1f90fe-Abstract-Conference.html)

        **Abstract**:

        This paper presents an end-to-end neural architecture based on Diffusion Models for spatial puzzle solving, particularly jigsaw puzzle and room arrangement tasks.In the latter task, for instance, the proposed system ``PuzzleFusion'' takes a set of room layouts as polygonal curves in the top-down view and aligns the room layout pieces by estimating their 2D translations and rotations, akin to solving the jigsaw puzzle of room layouts. A surprising discovery of the paper is that the simple use of a Diffusion Model effectively solves these challenging spatial puzzle tasks as a conditional generation process. To enable learning of an end-to-end neural system, the paper introduces new datasets with ground-truth arrangements: 1) 2D Voronoi Jigsaw Dataset, a synthetic one where pieces are generated by voronoi diagram of 2D pointset; and 2) MagicPlan Dataset, a real one from a production pipeline by MagicPlan, where pieces are room layouts constructed by augmented reality App by real-estate consumers.The qualitative and quantitative evaluations demonstrate that the proposed approach outperforms the competing methods by significant margins in all three spatial puzzle tasks. We have provided code and data in https://sepidsh.github.io/puzzlefusion.

        ----

        ## [420] Visual Instruction Inversion: Image Editing via Image Prompting

        **Authors**: *Thao Nguyen, Yuheng Li, Utkarsh Ojha, Yong Jae Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e75f7539cbde5de895fab238ff42519-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e75f7539cbde5de895fab238ff42519-Abstract-Conference.html)

        **Abstract**:

        Text-conditioned image editing has emerged as a powerful tool for editing images.However, in many situations, language can be ambiguous and ineffective in describing specific image edits.When faced with such challenges, visual prompts can be a more informative and intuitive way to convey ideas.We present a method for image editing via visual prompting.Given pairs of example that represent the "before" and "after" images of an edit, our goal is to learn a text-based editing direction that can be used to perform the same edit on new images.We leverage the rich, pretrained editing capabilities of text-to-image diffusion models by inverting visual prompts into editing instructions.Our results show that with just one example pair, we can achieve competitive results compared to state-of-the-art text-conditioned image editing frameworks.

        ----

        ## [421] Algorithm Selection for Deep Active Learning with Imbalanced Datasets

        **Authors**: *Jifan Zhang, Shuai Shao, Saurabh Verma, Robert D. Nowak*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e77af93008ee6cd248a31723ce357d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e77af93008ee6cd248a31723ce357d8-Abstract-Conference.html)

        **Abstract**:

        Label efficiency has become an increasingly important objective in deep learning applications. Active learning aims to reduce the number of labeled examples needed to train deep networks, but the empirical performance of active learning algorithms can vary dramatically across datasets and applications. It is difficult to know in advance which active learning strategy will perform well or best in a given application. To address this, we propose the first adaptive algorithm selection strategy for deep active learning. For any unlabeled dataset, our (meta) algorithm TAILOR (Thompson ActIve Learning algORithm selection) iteratively and adaptively chooses among a set of candidate active learning algorithms. TAILOR uses novel reward functions aimed at gathering class-balanced examples. Extensive experiments in multi-class and multi-label applications demonstrate TAILOR's effectiveness in achieving accuracy comparable or better than that of the best of the candidate algorithms. Our implementation of TAILOR is open-sourced at https://github.com/jifanz/TAILOR.

        ----

        ## [422] Federated Compositional Deep AUC Maximization

        **Authors**: *Xinwen Zhang, Yihan Zhang, Tianbao Yang, Richard Souvenir, Hongchang Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e7b192fc8b3acb93749c5accfa60e0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e7b192fc8b3acb93749c5accfa60e0c-Abstract-Conference.html)

        **Abstract**:

        Federated learning has attracted increasing attention due to the promise of balancing privacy and large-scale learning; numerous approaches have been proposed. However, most existing approaches focus on problems with balanced data, and prediction performance is far from satisfactory for many real-world applications where the number of samples in different classes is highly imbalanced. To address this challenging problem, we developed a novel federated learning method for imbalanced data by directly optimizing the area under curve (AUC) score. In particular, we formulate the AUC maximization problem as a federated compositional minimax optimization problem, develop a local stochastic compositional gradient descent ascent with momentum algorithm, and provide bounds on the computational and communication complexities of our algorithm. To the best of our knowledge, this is the first work to achieve such favorable theoretical results. Finally, extensive experimental results confirm the efficacy of our method.

        ----

        ## [423] On Learning Latent Models with Multi-Instance Weak Supervision

        **Authors**: *Kaifu Wang, Efthymia Tsamoura, Dan Roth*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1e83498c3eafe109a44b12979c2c73db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1e83498c3eafe109a44b12979c2c73db-Abstract-Conference.html)

        **Abstract**:

        We consider a weakly supervised learning scenario where the supervision signal is generated by a transition function $\sigma$ of labels associated with multiple input instances. We formulate this problem as *multi-instance Partial Label Learning (multi-instance PLL)*, which is an extension to the standard PLL problem. Our problem is met in different fields, including latent structural learning and neuro-symbolic integration. Despite the existence of many learning techniques, limited theoretical analysis has been dedicated to this problem. In this paper, we provide the first theoretical study of multi-instance PLL with possibly an unknown transition $\sigma$. Our main contributions are as follows: First, we proposed a necessary and sufficient condition for the learnability of the problem. This condition nontrivially generalizes and relaxes the existing *small ambiguity degree* in PLL literature since we allow the transition to be deterministic. Second, we derived Rademacher-style error bounds based on the top-$k$ surrogate loss that is widely used in the neuro-symbolic literature. Furthermore, we conclude with empirical experiments for learning with an unknown transition. The empirical results align with our theoretical findings; however, they also expose the issue of scalability in the weak supervision literature.

        ----

        ## [424] Degraded Polygons Raise Fundamental Questions of Neural Network Perception

        **Authors**: *Leonard Tang, Dan Ley*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ec408df112bc9b186d7b8fe0ada902a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ec408df112bc9b186d7b8fe0ada902a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deeplearning vision models suffer in a variety of settings that humans capably handle. Inlight of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images underdegradation, first introduced over 30 years ago in the Recognition-by-Componentstheory of human vision. Specifically, we study the performance and behavior ofneural networks on the seemingly simple task of classifying regular polygons atvarying orders of degradation along their perimeters. To this end, we implement theAutomated Shape Recoverability Testfor rapidly generating large-scale datasetsof perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity ofneural networks to recognize and recover such degraded shapes when initializedwith different priors. Ultimately, we find that neural networksâ€™ behavior on thissimple task conflicts with human behavior, raising a fundamental question of therobustness and learning capabilities of modern computer vision models

        ----

        ## [425] Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks

        **Authors**: *Blake Bordelon, Cengiz Pehlevan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html)

        **Abstract**:

        We analyze the dynamics of finite width effects in wide but finite feature learning neural networks. Starting from a dynamical mean field theory  description of infinite width deep neural network kernel and prediction dynamics, we provide a characterization of the $\mathcal{O}(1/\sqrt{\text{width}})$ fluctuations of the DMFT order parameters over random initializations of the network weights. Our results, while perturbative in width, unlike prior analyses, are non-perturbative in the strength of feature learning. In the lazy limit of network training, all kernels are random but static in time and the prediction variance has a universal form. However, in the rich, feature learning regime, the fluctuations of the kernels and predictions are dynamically coupled with a variance that can be computed self-consistently. In two layer networks, we show how feature learning can dynamically reduce the variance of the final tangent kernel and final network predictions. We also show how initialization variance can slow down online learning in wide but finite networks. In deeper networks, kernel variance can dramatically accumulate through subsequent layers at large feature learning strengths, but feature learning continues to improve the signal-to-noise ratio of the feature kernels. In discrete time, we demonstrate that large learning rate phenomena such as edge of stability effects can be well captured by infinite width dynamics and that initialization variance can decrease dynamically. For CNNs trained on CIFAR-10, we empirically find significant corrections to both the bias and variance of network dynamics due to finite width.

        ----

        ## [426] TART: A plug-and-play Transformer module for task-agnostic reasoning

        **Authors**: *Kush Bhatia, Avanika Narayan, Christopher De Sa, Christopher Ré*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ece70d2259b8e9510e2d4ca8754cecf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ece70d2259b8e9510e2d4ca8754cecf-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) exhibit in-context learning abilities which enable the same model to perform several tasks without any task-specific training. In contrast, traditional adaptation approaches, such as fine-tuning, modify the underlying models for each specific task. In-context learning, however, consistently underperforms task-specific tuning approaches even when presented with the same examples. While most existing approaches (e.g., prompt engineering) focus on the LLM's learned representations to patch this performance gap, our experiments actually reveal that LLM representations contain sufficient information to make good predictions. As such, we focus on the LLM's reasoning abilities and demonstrate that this performance gap exists due to their inability to perform simple probabilistic reasoning tasks. This raises an intriguing question: Are LLMs actually capable of learning how to reason in a task-agnostic manner? We answer this in the affirmative and, as a proof of concept, propose TART which generically improves an LLM's reasoning abilities using a synthetically trained reasoning module. TART trains this Transformer-based reasoning module in a task-agnostic manner using only synthetic logistic regression tasks and composes it with an arbitrary real-world pre-trained model without any additional training. With a single inference module, TART improves performance across different model families (GPT-Neo, Pythia, Bloom), model sizes (100M - 6B), tasks (14 NLP classification tasks), and even across different modalities (audio and vision). On the RAFT Benchmark, TART improves GPT-Neo (125M)'s performance such that it outperforms Bloom (176B), and is within $4$% of GPT-3.

        ----

        ## [427] Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment

        **Authors**: *Carsten T. Lüth, Till J. Bungert, Lukas Klein, Paul Jaeger*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract-Conference.html)

        **Abstract**:

        Active Learning (AL) aims to reduce the labeling burden by interactively selecting the most informative samples from a pool of unlabeled data. While there has been extensive research on improving AL query methods in recent years, some studies have questioned the effectiveness of AL compared to emerging paradigms such as semi-supervised (Semi-SL) and self-supervised learning (Self-SL), or a simple optimization of classifier configurations. Thus, today’s AL literature presents an inconsistent and contradictory landscape, leaving practitioners uncertain about whether and how to use AL in their tasks. In this work, we make the case that this inconsistency arises from a lack of systematic and realistic evaluation of AL methods. Specifically, we identify five key pitfalls in the current literature that reflect the delicate considerations required for AL evaluation. Further, we present an evaluation framework that overcomes these pitfalls and thus enables meaningful statements about the performance of AL methods. To demonstrate the relevance of our protocol, we present a large-scale empirical study and benchmark for image classification spanning various data sets, query methods, AL settings, and training paradigms. Our findings clarify the inconsistent picture in the literature and enable us to give hands-on recommendations for practitioners. The benchmark is hosted at https://github.com/IML-DKFZ/realistic-al.

        ----

        ## [428] Semantic HELM: A Human-Readable Memory for Reinforcement Learning

        **Authors**: *Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1eeacdf8770e6dd5164cdeec8bcfa8cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1eeacdf8770e6dd5164cdeec8bcfa8cc-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning agents deployed in the real world often have to cope with partially observable environments. Therefore, most agents employ memory mechanisms to approximate the state of the environment. Recently, there have been impressive success stories in mastering partially observable environments, mostly in the realm of computer games like Dota 2, StarCraft II, or MineCraft. However, existing methods lack interpretability in the sense that it is not comprehensible for humans what the agent stores in its memory.In this regard, we propose a novel memory mechanism that represents past events in human language.Our method uses CLIP to associate visual inputs with language tokens. Then we feed these tokens to a pretrained language model that serves the agent as memory and provides it with a coherent and human-readable representation of the past.We train our memory mechanism on a set of partially observable environments and find that it excels on tasks that require a memory component, while mostly attaining performance on-par with strong baselines on tasks that do not. On a challenging continuous recognition task, where memorizing the past is crucial, our memory mechanism converges two orders of magnitude faster than prior methods.Since our memory mechanism is human-readable, we can peek at an agent's memory and check whether crucial pieces of information have been stored.This significantly enhances troubleshooting and paves the way toward more interpretable agents.

        ----

        ## [429] Empowering Convolutional Neural Nets with MetaSin Activation

        **Authors**: *Farnood Salehi, Tunç Aydin, André Gaillard, Guglielmo Camporese, Yuxuan Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f05584d537c92c8271699f207677475-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f05584d537c92c8271699f207677475-Abstract-Conference.html)

        **Abstract**:

        ReLU networks have remained the default choice for models in the area of image prediction despite their well-established spectral bias towards learning low frequencies faster, and consequently their difficulty of reproducing high frequency visual details. As an alternative, sin networks showed promising results in learning implicit representations of visual data. However training these networks in practically relevant settings proved to be difficult, requiring careful initialization, dealing with issues due to inconsistent gradients, and a degeneracy in local minima. In this work, we instead propose replacing a baseline network’s existing activations with a novel ensemble function with trainable parameters. The proposed MetaSin activation can be trained reliably without requiring intricate initialization schemes, and results in consistently lower test loss compared to alternatives. We demonstrate our method in the areas of Monte-Carlo denoising and image resampling where we set new state-of-the-art through a knowledge distillation based training procedure. We present ablations on hyper-parameter settings, comparisons with alternative activation function formulations, and discuss the use of our method in other domains, such as image classification.

        ----

        ## [430] When Does Confidence-Based Cascade Deferral Suffice?

        **Authors**: *Wittawat Jitkrittum, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, Sanjiv Kumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html)

        **Abstract**:

        Cascades are a classical strategy to enable inference cost to vary adaptively across samples, wherein a sequence of classifiers are invoked in turn.  A deferral rule determines whether to invoke the next classifier in the sequence, or to terminate prediction.  One simple deferral rule employs the confidence of the current classifier, e.g., based on the maximum predicted softmax probability.  Despite being oblivious to the structure of the cascade --- e.g., not modelling the errors of downstream models --- such confidence-based deferral often works remarkably well in practice.  In this paper, we seek to better understand the conditions under which confidence-based deferral may fail, and when alternate deferral strategies can perform better.  We first present a theoretical characterisation of the optimal deferral rule, which precisely characterises settings under which confidence-based deferral may suffer.  We then study post-hoc deferral mechanisms, and demonstrate they can significantly improve upon confidence-based deferral in settings where (i)  downstream models are specialists that only work well on a subset of inputs, (ii) samples are subject to label noise, and (iii) there is distribution shift between the train and test set.

        ----

        ## [431] DeWave: Discrete Encoding of EEG Waves for EEG to Text Translation

        **Authors**: *Yiqun Duan, Charles Chau, Zhen Wang, Yu-Kai Wang, Chin-Teng Lin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html)

        **Abstract**:

        The translation of brain dynamics into natural language is pivotal for brain-computer interfaces (BCIs), a field that has seen substantial growth in recent years. With the swift advancement of large language models, such as ChatGPT, the need to bridge the gap between the brain and languages becomes increasingly pressing. Current methods, however, require eye-tracking fixations or event markers to segment brain dynamics into word-level features, which can restrict the practical application of these systems. These event markers may not be readily available or could be challenging to acquire during real-time inference, and the sequence of eye fixations may not align with the order of spoken words. To tackle these issues, we introduce a novel framework, DeWave, that integrates discrete encoding sequences into open-vocabulary EEG-to-text translation tasks. DeWave uses a quantized variational encoder to derive discrete codex encoding and align it with pre-trained language models. This discrete codex representation brings forth two advantages: 1) it alleviates the order mismatch between eye fixations and spoken words by introducing text-EEG contrastive alignment training, and 2) it minimizes the interference caused by individual differences in EEG waves through an invariant discrete codex. Our model surpasses the previous baseline (40.1 and 31.7) by 3.06% and 6.34\%, respectively, achieving 41.35 BLEU-1 and 33.71 Rouge-F on the ZuCo Dataset. Furthermore, this work is the first to facilitate the translation of entire EEG signal periods without the need for word-level order markers (e.g., eye fixations), scoring 20.5 BLEU-1 and 29.5 Rouge-1 on the ZuCo Dataset, respectively.

        ----

        ## [432] SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data

        **Authors**: *Bang An, Xun Zhou, Yongjian Zhong, Tianbao Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f3cbee17170c3ffff3e413d2df54f6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f3cbee17170c3ffff3e413d2df54f6b-Abstract-Conference.html)

        **Abstract**:

        The problem of urban event ranking aims at predicting the top-$k$ most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. Due to the common assumption that items are independent. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named SpatialRank. SpatialRank features  adaptive graph convolution layers that dynamically learn the spatiotemporal dependencies across locations from data. In addition, the model optimizes through surrogates a hybrid NDCG loss with a spatial component to better rank neighboring spatial locations. We design an importance-sampling with a spatial filtering algorithm to effectively evaluate the loss during training. Comprehensive experiments on three real-world datasets demonstrate that SpatialRank can effectively identify the top riskiest locations of crimes and traffic accidents and outperform state-of-art methods in terms of NDCG by up to 12.7%.

        ----

        ## [433] An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions

        **Authors**: *Mohammad Jalali, Cheuk Ting Li, Farzan Farnia*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f5c5cd01b864d53cc5fa0a3472e152e-Abstract-Conference.html)

        **Abstract**:

        The evaluation of generative models has received significant attention in the machine learning community.  When applied to a multi-modal distribution which is common among image datasets, an intuitive evaluation criterion is the number of modes captured by the generative model. While several scores have been proposed to evaluate the quality and diversity of a model's generated data, the correspondence between existing scores and the number of modes in the distribution is unclear. In this work, we propose an information-theoretic diversity evaluation method for multi-modal underlying distributions. We utilize the R\'enyi Kernel Entropy (RKE) as an evaluation score based on quantum information theory to measure the number of modes in generated samples. To interpret the proposed evaluation method, we show that the RKE score can output the number of modes of a mixture of sub-Gaussian components. We also prove estimation error bounds for estimating the RKE score from limited data, suggesting a fast convergence of the empirical RKE score to the score for the underlying data distribution. Utilizing the RKE score, we conduct an extensive evaluation of state-of-the-art generative models over standard image datasets. The numerical results indicate that while the recent algorithms for training generative models manage to improve the mode-based diversity over the earlier architectures, they remain incapable of capturing the full diversity of real data. Our empirical results provide a ranking of widely-used generative models based on the RKE score of their generated samples.

        ----

        ## [434] A Cross-Moment Approach for Causal Effect Estimation

        **Authors**: *Yaroslav Kivva, Saber Salehkaleybar, Negar Kiyavash*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f6100363156cced8633f4e89dd8ceb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f6100363156cced8633f4e89dd8ceb1-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of estimating the causal effect of a treatment on an outcome in  linear structural causal models (SCM) with latent confounders when we have access to a single proxy variable.Several methods (such as difference-in-difference (DiD) estimator or negative outcome control) have been proposed in this setting in the literature. However, these approaches require either restrictive assumptions on the data generating model or having access to at least two proxy variables.We propose a method to estimate the causal effect using cross moments between the treatment, the outcome, and the proxy variable. In particular, we show that the causal effect can be identified with simple arithmetic operations on the cross moments if the latent confounder in linear SCM is non-Gaussian.In this setting, DiD estimator provides an unbiased estimate only in the special case where the latent confounder has exactly the same direct causal effects on the outcomes in the pre-treatment and post-treatment phases. This translates to the common trend assumption in DiD, which we effectively relax.Additionally, we provide an impossibility result that shows the causal effect cannot be identified if the observational distribution over the treatment, the outcome, and the proxy is jointly Gaussian. Our experiments on both synthetic and real-world datasets showcase the effectivenessof the proposed approach in estimating the causal effect.

        ----

        ## [435] Combining Behaviors with the Successor Features Keyboard

        **Authors**: *Wilka Carvalho, Andre Saraiva, Angelos Filos, Andrew K. Lampinen, Loic Matthey, Richard L. Lewis, Honglak Lee, Satinder Singh, Danilo Jimenez Rezende, Daniel Zoran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f69928210578f4cf5b538a8c8806798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f69928210578f4cf5b538a8c8806798-Abstract-Conference.html)

        **Abstract**:

        The Option Keyboard (OK) was recently proposed as a method for transferring behavioral knowledge across tasks. OK transfers knowledge by adaptively combining subsets of known behaviors using Successor Features (SFs) and Generalized Policy Improvement (GPI).However, it relies on hand-designed state-features and task encodings which are cumbersome to design for every new environment.In this work, we propose the "Successor Features Keyboard" (SFK), which enables transfer with discovered state-features and task encodings.To enable discovery, we propose the "Categorical Successor Feature Approximator" (CSFA), a novel learning algorithm for estimating SFs while jointly discovering state-features and task encodings.With SFK and CSFA, we achieve the first demonstration of transfer with SFs in a challenging 3D environment where all the necessary representations are discovered.We first compare CSFA against other methods for approximating SFs and show that only CSFA discovers representations compatible with SF&GPI at this scale.We then compare SFK against transfer learning baselines and show that it transfers most quickly to long-horizon tasks.

        ----

        ## [436] Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

        **Authors**: *Chenyu You, Weicheng Dai, Yifei Min, Fenglin Liu, David A. Clifton, S. Kevin Zhou, Lawrence H. Staib, James S. Duncan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f7e6d5c84b0ed286d0e69b7d2c79b47-Abstract-Conference.html)

        **Abstract**:

        For medical image segmentation, contrastive learning is the dominant practice to improve the quality of visual representations by contrasting semantically similar and dissimilar pairs of samples. This is enabled by the observation that without accessing ground truth labels, negative examples with truly dissimilar anatomical features, if sampled, can significantly improve the performance. In reality, however, these samples may come from similar anatomical features and the models may struggle to distinguish the minority tail-class samples, making the tail classes more prone to misclassification, both of which typically lead to model collapse. In this paper, we propose $\texttt{ARCO}$, a semi-supervised contrastive learning (CL) framework with stratified group theory for medical image segmentation. In particular, we first propose building $\texttt{ARCO}$ through the concept of variance-reduced estimation, and show that certain variance-reduction techniques are particularly beneficial in pixel/voxel-level segmentation tasks with extremely limited labels. Furthermore, we theoretically prove these sampling techniques are universal in variance reduction. Finally, we experimentally validate our approaches on eight benchmarks, i.e., five 2D/3D medical and three semantic segmentation datasets, with different label settings, and our methods consistently outperform state-of-the-art semi-supervised methods. Additionally, we augment the CL frameworks with these sampling techniques and demonstrate significant gains over previous methods. We believe our work is an important step towards semi-supervised medical image segmentation by quantifying the limitation of current self-supervision objectives for accomplishing such challenging safety-critical tasks.

        ----

        ## [437] Boosting Verification of Deep Reinforcement Learning via Piece-Wise Linear Decision Neural Networks

        **Authors**: *Jiaxu Tian, Dapeng Zhi, Si Liu, Peixin Wang, Cheng Chen, Min Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1f96b24df4b06f5d68389845a9a13ed9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1f96b24df4b06f5d68389845a9a13ed9-Abstract-Conference.html)

        **Abstract**:

        Formally verifying deep reinforcement learning (DRL) systems suffers from both inaccurate verification results and limited scalability. The major obstacle lies in the large overestimation introduced inherently during training and then transforming the inexplicable decision-making models, i.e., deep neural networks (DNNs), into easy-to-verify models. In this paper, we propose an inverse transform-then-train approach, which first encodes a DNN into an equivalent set of efficiently and tightly verifiable linear control policies and then optimizes them via reinforcement learning.  We accompany our inverse approach with a novel neural network model called piece-wise linear decision neural networks (PLDNNs), which are compatible with most existing DRL training algorithms with comparable performance against conventional DNNs. Our extensive experiments show that, compared to DNN-based  DRL systems, PLDNN-based systems can be more efficiently and tightly verified with up to $438$ times speedup and a significant reduction in overestimation.  In particular, even a complex $12$-dimensional DRL system is efficiently verified with up to 7 times deeper computation steps.

        ----

        ## [438] Star-Shaped Denoising Diffusion Probabilistic Models

        **Authors**: *Andrey Okhotin, Dmitry Molchanov, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, Dmitry P. Vetrov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1fcefa894924bb1688041b7a26fb8aea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1fcefa894924bb1688041b7a26fb8aea-Abstract-Conference.html)

        **Abstract**:

        Denoising Diffusion Probabilistic Models (DDPMs) provide the foundation for the recent breakthroughs in generative modeling.Their Markovian structure makes it difficult to define DDPMs with distributions other than Gaussian or discrete.In this paper, we introduce Star-Shaped DDPM (SS-DDPM).Its star-shaped diffusion process allows us to bypass the need to define the transition probabilities or compute posteriors.We establish duality between star-shaped and specific Markovian diffusions for the exponential family of distributions and derive efficient algorithms for training and sampling from SS-DDPMs.In the case of Gaussian distributions, SS-DDPM is equivalent to DDPM.However, SS-DDPMs provide a simple recipe for designing diffusion models with distributions such as Beta, von Misesâ€“Fisher, Dirichlet, Wishart and others, which can be especially useful when data lies on a constrained manifold.We evaluate the model in different settings and find it competitive even on image data, where Beta SS-DDPM achieves results comparable to a Gaussian DDPM.Our implementation is available at https://github.com/andrey-okhotin/star-shaped

        ----

        ## [439] An Adaptive Algorithm for Learning with Unknown Distribution Drift

        **Authors**: *Alessio Mazzetto, Eli Upfal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1fe6f635fe265292aba3987b5123ae3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1fe6f635fe265292aba3987b5123ae3d-Abstract-Conference.html)

        **Abstract**:

        We develop and analyze a general technique for learning with an unknown distribution drift. Given a sequence of independent observations from the last $T$ steps of a drifting distribution, our algorithm agnostically learns a family of functions with respect to the current distribution at time $T$. Unlike previous work, our technique does not require prior knowledge about the magnitude of the drift. Instead, the algorithm adapts to the sample data. Without explicitly estimating the drift, the algorithm learns a family of functions with almost the same error as a learning algorithm that knows the magnitude of the drift in advance. Furthermore, since our algorithm adapts to the data, it can guarantee a better learning error than an algorithm that relies on loose bounds on the drift. We demonstrate the application of our technique in two fundamental learning scenarios: binary classification and linear regression.

        ----

        ## [440] QLoRA: Efficient Finetuning of Quantized LLMs

        **Authors**: *Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)

        **Abstract**:

        We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information-theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small, high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations, showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training.

        ----

        ## [441] EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning

        **Authors**: *Ping Guo, Xiangpeng Wei, Yue Hu, Baosong Yang, Dayiheng Liu, Fei Huang, Jun Xie*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/201408406e0c5cf7626c4baeae6eaadd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/201408406e0c5cf7626c4baeae6eaadd-Abstract-Conference.html)

        **Abstract**:

        Expressing universal semantics common to all languages is helpful to understand the meanings of complex and culture-specific sentences. The research theme underlying this scenario focuses on learning universal representations across languages with the usage of massive parallel corpora. However, due to the sparsity and scarcity of parallel data, there is still a big challenge in learning authentic ``universals'' for any two languages. In this paper, we propose Emma-X: an EM-like Multilingual pre-training Algorithm, to learn Cross-lingual universals with the aid of excessive multilingual non-parallel data. Emma-X unifies the cross-lingual representation learning task and an extra semantic relation prediction task within an EM framework. Both the extra semantic classifier and the cross-lingual sentence encoder approximate the semantic relation of two sentences, and supervise each other until convergence. To evaluate Emma-X, we conduct experiments on xrete, a newly introduced benchmark containing 12 widely studied cross-lingual tasks that fully depend on sentence-level representations. Results reveal that Emma-X achieves state-of-the-art performance. Further geometric analysis of the built representation space with three requirements demonstrates the superiority of Emma-X over advanced models.

        ----

        ## [442] Tester-Learners for Halfspaces: Universal Algorithms

        **Authors**: *Aravind Gollakota, Adam R. Klivans, Konstantinos Stavropoulos, Arsen Vasilyan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/204d9a9a4816a45909010587ffc3204b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/204d9a9a4816a45909010587ffc3204b-Abstract-Conference.html)

        **Abstract**:

        We give the first tester-learner for halfspaces that succeeds universally over a wide class of structured distributions. Our universal tester-learner runs in fully polynomial time and has the following guarantee: the learner achieves error $O(\mathrm{opt}) + \epsilon$ on any labeled distribution that the tester accepts, and moreover, the tester accepts whenever the marginal is any distribution that satisfies a Poincare inequality. In contrast to prior work on testable learning, our tester is not tailored to any single target distribution but rather succeeds for an entire target class of distributions. The class of Poincare distributions includes all strongly log-concave distributions, and, assuming the Kannan--Lovasz--Simonovits (KLS) conjecture, includes all log-concave distributions. In the special case where the label noise is known to be Massart, our tester-learner achieves error $\mathrm{opt} + \epsilon$ while accepting all log-concave distributions unconditionally (without assuming KLS).Our tests rely on checking hypercontractivity of the unknown distribution using a sum-of-squares (SOS) program, and crucially make use of the fact that Poincare distributions are certifiably hypercontractive in the SOS framework.

        ----

        ## [443] Revisit the Power of Vanilla Knowledge Distillation: from Small Scale to Large Scale

        **Authors**: *Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/204f828ba287fdecf41dd002e9a07d8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/204f828ba287fdecf41dd002e9a07d8c-Abstract-Conference.html)

        **Abstract**:

        The tremendous success of large models trained on extensive datasets demonstrates that scale is a key ingredient in achieving superior results. Therefore, the reflection on the rationality of designing knowledge distillation (KD) approaches for limited-capacity architectures solely based on small-scale datasets is now deemed imperative. In this paper, we identify the small data pitfall that presents in previous KD methods, which results in the underestimation of the power of vanilla KD framework on large-scale datasets such as ImageNet-1K. Specifically, we show that employing stronger data augmentation techniques and using larger datasets can directly decrease the gap between vanilla KD and other meticulously designed KD variants. This highlights the necessity of designing and evaluating KD approaches in the context of practical scenarios, casting off the limitations of small-scale datasets. Our investigation of the vanilla KD and its variants in more complex schemes, including stronger training strategies and different model capacities, demonstrates that vanilla KD is elegantly simple but astonishingly effective in large-scale scenarios. Without bells and whistles, we obtain state-of-the-art ResNet-50, ViT-S, and ConvNeXtV2-T models for ImageNet, which achieve 83.1%, 84.3%, and 85.0% top-1 accuracy, respectively. PyTorch code and checkpoints can be found at https://github.com/Hao840/vanillaKD.

        ----

        ## [444] Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context

        **Authors**: *Julian Tanke, Oh-Hun Kwon, Felix B. Mueller, Andreas Doering, Jürgen Gall*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2052b3e0617ecb2ce9474a6feaf422b3-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Forecasting human motion of multiple persons is very challenging. It requires to model the interactions between humans and the interactions with objects and the environment. For example, a person might want to make a coffee, but if the coffee machine is already occupied the person will haveto wait. These complex relations between scene geometry and persons ariseconstantly in our daily lives, and models that wish to accurately forecasthuman behavior will have to take them into consideration. To facilitate research in this direction, we propose Humans in Kitchens, alarge-scale multi-person human motion dataset with annotated 3D human poses, scene geometry and activities per person and frame.Our dataset consists of over 7.3h recorded data of up to 16 persons at the same time in four kitchen scenes, with more than 4M annotated human poses, represented by a parametric 3D body model. In addition, dynamic scene geometry and objects like chair or cupboard are annotated per frame. As first benchmarks, we propose two protocols for short-term and long-term human motion forecasting.

        ----

        ## [445] LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings

        **Authors**: *Ningyi Liao, Siqiang Luo, Xiang Li, Jieming Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/206191b9b7349e2743d98d855dec9e58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/206191b9b7349e2743d98d855dec9e58-Abstract-Conference.html)

        **Abstract**:

        Heterophilous Graph Neural Network (GNN) is a family of GNNs that specializes in learning graphs under heterophily, where connected nodes tend to have different labels. Most existing heterophilous models incorporate iterative non-local computations to capture node relationships. However, these approaches have limited application to large-scale graphs due to their high computational costs and challenges in adopting minibatch schemes. In this work, we study the scalability issues of heterophilous GNN and propose a scalable model, LD2, which simplifies the learning process by decoupling graph propagation and generating expressive embeddings prior to training. Theoretical analysis demonstrates that LD2 achieves optimal time complexity in training, as well as a memory footprint that remains independent of the graph scale. We conduct extensive experiments to showcase that our model is capable of lightweight minibatch training on large-scale heterophilous graphs, with up to $15\times$ speed improvement and efficient memory utilization, while maintaining comparable or better performance than the baselines.

        ----

        ## [446] On Single-Index Models beyond Gaussian Data

        **Authors**: *Aaron Zweig, Loucas Pillaud-Vivien, Joan Bruna*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2063a00c435aafbcc58c16ce1e522139-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2063a00c435aafbcc58c16ce1e522139-Abstract-Conference.html)

        **Abstract**:

        Sparse high-dimensional functions have arisen as a rich framework to study the behavior of gradient-descent methods using shallow neural networks, and showcasing its ability to perform feature learning beyond linear models. Amongst those functions, the simplest are single-index models $f(x) = \phi( x \cdot \theta^*)$, where the labels are generated by an arbitrary non-linear link function $\phi$ of an unknown one-dimensional projection $\theta^*$ of the input data. By focusing on Gaussian data, several recent works have built a remarkable picture, where the so-called information exponent (related to the regularity of the link function) controls the required sample complexity. In essence, these tools exploit the stability and spherical symmetry of Gaussian distributions.In this work, we explore extensions of this picture beyond the Gaussian setting, where both stability or symmetry might be violated. Focusing on the planted setting where $\phi$ is known, our main results establish that Stochastic Gradient Descent recovers the unknown direction $\theta^*$ with constant probability in the high-dimensional regime, under mild assumptions that significantly extend ~[Yehudai and Shamir,20].

        ----

        ## [447] Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

        **Authors**: *Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, Liang Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2082273791021571c410f41d565d0b45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2082273791021571c410f41d565d0b45-Abstract-Conference.html)

        **Abstract**:

        Hand-crafted image quality metrics, such as PSNR and SSIM, are commonly used to evaluate model privacy risk under reconstruction attacks. Under these metrics, reconstructed images that are determined to resemble the original one generally indicate more privacy leakage. Images determined as overall dissimilar, on the other hand, indicate higher robustness against attack. However, there is no guarantee that these metrics well reflect human opinions, which offers trustworthy judgement for model privacy leakage.  In this paper, we comprehensively study the faithfulness of these hand-crafted metrics to human perception of privacy information from the reconstructed images. On 5 datasets ranging from natural images, faces, to fine-grained classes, we use 4 existing attack methods to reconstruct images from many different classification models and, for each reconstructed image, we ask multiple human annotators to assess whether this image is recognizable. Our studies reveal that the hand-crafted metrics only have a weak correlation with the human evaluation of privacy leakage and that even these metrics themselves often contradict each other. These observations suggest risks of current metrics  in the community. To address this potential risk, we propose a learning-based measure called SemSim to evaluate the Semantic Similarity between the original and reconstructed images. SemSim is trained with a standard triplet loss, using an original image as an anchor, one of its recognizable reconstructed images as a positive sample, and an unrecognizable one as a negative. By training on human annotations, SemSim exhibits a greater reflection of privacy leakage on the semantic level. We show that SemSim has a significantly higher correlation with human judgment compared with existing metrics. Moreover, this strong correlation generalizes to unseen datasets, models and attack methods. We envision this work as a milestone for image quality evaluation closer to the human level. The project webpage can be accessed at https://sites.google.com/view/semsim.

        ----

        ## [448] Graph Denoising Diffusion for Inverse Protein Folding

        **Authors**: *Kai Yi, Bingxin Zhou, Yiqing Shen, Pietro Lió, Yuguang Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20888d00c5df685de2c09790040e0327-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20888d00c5df685de2c09790040e0327-Abstract-Conference.html)

        **Abstract**:

        Inverse protein folding is challenging due to its inherent one-to-many mapping characteristic, where numerous possible amino acid sequences can fold into a single, identical protein backbone. This task involves not only identifying viable sequences but also representing the sheer diversity of potential solutions. However, existing discriminative models, such as transformer-based auto-regressive models, struggle to encapsulate the diverse range of plausible solutions. In contrast, diffusion probabilistic models, as an emerging genre of generative approaches, offer the potential to generate a diverse set of sequence candidates for determined protein backbones. We propose a novel graph denoising diffusion model for inverse protein folding, where a given protein backbone guides the diffusion process on the corresponding amino acid residue types. The model infers the joint distribution of amino acids conditioned on the nodes' physiochemical properties and local environment. Moreover, we utilize amino acid replacement matrices for the diffusion forward process, encoding the biologically-meaningful prior knowledge of amino acids from their spatial and sequential neighbors as well as themselves, which reduces the sampling space of the generative process. Our model achieves state-of-the-art performance over a set of popular baseline methods in sequence recovery and exhibits great potential in generating diverse protein sequences for a determined protein backbone structure.

        ----

        ## [449] Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation

        **Authors**: *Chaofan Ma, Yuhuan Yang, Chen Ju, Fei Zhang, Ya Zhang, Yanfeng Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2093ed77c549eda95bd6f7212b735b43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2093ed77c549eda95bd6f7212b735b43-Abstract-Conference.html)

        **Abstract**:

        Open-vocabulary semantic segmentation is a challenging task that requires segmenting novel object categories at inference time. Recent works explore vision-language pre-training to handle this task, but suffer from unrealistic assumptions in practical scenarios, i.e., low-quality textual category names.For example, this paradigm assumes that new textual categories will be accurately and completely provided, and exist in lexicons during pre-training.However, exceptions often happen when meet with ambiguity for brief or incomplete names, new words that are not present in the pre-trained lexicons, and difficult-to-describe categories for users.To address these issues, this work proposes a novel attribute decomposition-aggregation framework, AttrSeg, inspired by human cognition in understanding new concepts. Specifically, in the decomposition stage, we decouple class names into diverse attribute descriptions to complement semantic contexts from multiple perspectives.Two attribute construction strategies are designed: using large language models for common categories, and involving manually labelling for human-invented categories. In the aggregation stage, we group diverse attributes into an integrated global description, to form a discriminative classifier that distinguishes the target object from others. One hierarchical aggregation architecture is further proposed to achieve multi-level aggregation, leveraging the meticulously designed clustering module.The final result is obtained by computing the similarity between aggregated attributes and images embedding.To evaluate the effectiveness, we annotate three datasets with attribute descriptions, and conduct extensive experiments and ablation studies. The results show the superior performance of attribute decomposition-aggregation.We refer readers to the latest arXiv version at https://arxiv.org/abs/2309.00096.

        ----

        ## [450] Stable and low-precision training for large-scale vision-language models

        **Authors**: *Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, Ludwig Schmidt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract-Conference.html)

        **Abstract**:

        We introduce new methods for 1) accelerating and 2) stabilizing training for large language-vision models. 1) For acceleration, we introduce SwitchBack, a linear layer for int8 quantized training which provides a speed-up of 13-25% while matching the performance of bfloat16 training within 0.1 percentage points for the 1B parameter CLIP ViT-Huge---the largest int8 training to date. Our main focus is int8 as GPU support for float8 is rare, though we also analyze float8 training through simulation. While SwitchBack proves effective for float8, we show that standard techniques are also successful if the network is trained and initialized so that large feature magnitudes are discouraged, which we accomplish via layer-scale initialized with zeros. 2) For stability, we analyze loss spikes and find they consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator. As a result, we recommend an AdamW-Adafactor hybrid which avoids loss spikes when training a CLIP ViT-Huge model and outperforms gradient clipping at the scales we test.

        ----

        ## [451] Recommender Systems with Generative Retrieval

        **Authors**: *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Mahesh Sathiamoorthy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)

        **Abstract**:

        Modern recommender systems perform large-scale retrieval by embedding queries and item candidates in the same unified space, followed by approximate nearest neighbor search to select top candidates given a query embedding. In this paper, we propose a novel generative retrieval approach, where the retrieval model autoregressively decodes the identifiers of the target candidates. To that end, we create semantically meaningful tuple of codewords to serve as a Semantic ID for each item. Given Semantic IDs for items in a user session, a Transformer-based sequence-to-sequence model is trained to predict the Semantic ID of the next item that the user will interact with. We show that recommender systems trained with the proposed paradigm significantly outperform the current SOTA models on various datasets. In addition, we show that incorporating Semantic IDs into the sequence-to-sequence model enhances its ability to generalize, as evidenced by the improved retrieval performance observed for items with no prior interaction history.

        ----

        ## [452] Active Vision Reinforcement Learning under Limited Visual Observability

        **Authors**: *Jinghuan Shang, Michael S. Ryoo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20e6b4dd2b1f82bc599c593882f67f75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20e6b4dd2b1f82bc599c593882f67f75-Abstract-Conference.html)

        **Abstract**:

        In this work, we investigate Active Vision Reinforcement Learning (ActiveVision-RL), where an embodied agent simultaneously learns action policy for the task while also controlling its visual observations in partially observable environments. We denote the former as motor policy and the latter as sensory policy. For example, humans solve real world tasks by hand manipulation (motor policy) together with eye movements (sensory policy). ActiveVision-RL poses challenges on coordinating two policies given their mutual influence. We propose SUGARL, Sensorimotor Understanding Guided Active Reinforcement Learning, a framework that models motor and sensory policies separately, but jointly learns them using with an intrinsic sensorimotor reward. This learnable reward is assigned by sensorimotor reward module, incentivizes the sensory policy to select observations that are optimal to infer its own motor action, inspired by the sensorimotor stage of humans. Through a series of experiments, we show the effectiveness of our method across a range of observability conditions and its adaptability to existed RL algorithms. The sensory policies learned through our method are observed to exhibit effective active vision strategies.

        ----

        ## [453] Optimization of Inter-group criteria for clustering with minimum size constraints

        **Authors**: *Eduardo Sany Laber, Lucas Murtinho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/20f814ecdaa8c76131e21683447e347b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/20f814ecdaa8c76131e21683447e347b-Abstract-Conference.html)

        **Abstract**:

        Internal measures that are used to assess the quality of a clustering usually take into account intra-group and/or inter-group criteria.There are many papers in the literature that propose algorithms with provable approximation guarantees for optimizing the former. However,  the optimization of inter-group criteria is much less understood.Here, we contribute to the state-of-the-art of this literature by devising algorithms with provable guarantees for the maximization of two natural inter-group criteria, namely the minimum spacing and the minimum spanning tree spacing. The former is the minimum distance between points in different groups while the latter captures separability through the cost of the minimum spanning tree that connects all groups. We obtain results for both the unrestricted case, in which no constraint on the clusters is imposed, and for the constrained case where each group is required to have a minimum number of points. Our constraint is motivated by the fact that the popular Single-Linkage, which optimizes both criteria in the unrestricted case, produces clustering with many tiny groups.To complement our work, we present an empirical study with 10 real datasets that provides evidence that our methods work very well in practical settings.

        ----

        ## [454] CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation

        **Authors**: *Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, Joyce Chai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21293a43d0321c5602dd893be2c2332b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21293a43d0321c5602dd893be2c2332b-Abstract-Conference.html)

        **Abstract**:

        Diffusion models (DMs) have enabled breakthroughs in image synthesis tasks but lack an intuitive interface for consistent image-to-image (I2I) translation. Various methods have been explored to address this issue, including mask-based methods, attention-based methods, and image-conditioning. However, it remains a critical challenge to enable unpaired I2I translation with pre-trained DMs while maintaining satisfying consistency. This paper introduces Cyclenet, a novel but simple method that incorporates cycle consistency into DMs to regularize image manipulation. We validate Cyclenet on unpaired I2I tasks of different granularities. Besides the scene and object level translation, we additionally contribute a multi-domain I2I translation dataset to study the physical state changes of objects. Our empirical studies show that Cyclenet is superior in translation consistency and quality, and can generate high-quality images for out-of-domain distributions with a simple change of the textual prompt. Cyclenet is a practical framework, which is robust even with very limited training data (around 2k) and requires minimal computational resources (1 GPU) to train. Project homepage: https://cyclenetweb.github.io/

        ----

        ## [455] Bandit Social Learning under Myopic Behavior

        **Authors**: *Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Aleksandrs Slivkins*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/212b143b5a5d6b88feb0fb1441b9756e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/212b143b5a5d6b88feb0fb1441b9756e-Abstract-Conference.html)

        **Abstract**:

        We study social learning dynamics motivated by reviews on online platforms. Theagents collectively follow a simple multi-armed bandit protocol, but each agentacts myopically, without regards to exploration. We allow a wide range of myopicbehaviors that are consistent with (parameterized) confidence intervals for the armsâ€™expected rewards. We derive stark exploration failures for any such behavior, andprovide matching positive results. As a special case, we obtain the first generalresults on failure of the greedy algorithm in bandits, thus providing a theoreticalfoundation for why bandit algorithms should explore.

        ----

        ## [456] Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

        **Authors**: *Rainer Engelken*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/214ce905bf2072535e34b3cf873cbbc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/214ce905bf2072535e34b3cf873cbbc8-Abstract-Conference.html)

        **Abstract**:

        Training recurrent neural networks (RNNs) remains a challenge due to the instability of gradients across long time horizons, which can lead to exploding and vanishing gradients. Recent research has linked these problems to the values of Lyapunov exponents for the forward-dynamics, which describe the growth or shrinkage of infinitesimal perturbations. Here, we propose gradient flossing, a novel approach to tackling gradient instability by pushing Lyapunov exponents of the forward dynamics toward zero during learning. We achieve this by regularizing Lyapunov exponents through backpropagation using differentiable linear algebra. This enables us to "floss" the gradients, stabilizing them and thus improving network training. We show that gradient flossing controls not only the gradient norm but also the condition number of the long-term Jacobian, facilitating multidimensional error feedback propagation. We find that applying gradient flossing before training enhances both the success rate and convergence speed for tasks involving long time horizons.For challenging tasks, we show that gradient flossing during training can further increase the time horizon that can be bridged by backpropagation through time. Moreover, we demonstrate the effectiveness of our approach on various RNN architectures and tasks of variable temporal complexity. Additionally, we provide a simple implementation of our gradient flossing algorithm that can be used in practice. Our results indicate that gradient flossing via regularizing Lyapunov exponents can significantly enhance the effectiveness of RNN training and mitigate the exploding and vanishing gradients problem.

        ----

        ## [457] How to Data in Datathons

        **Authors**: *Carlos Mougan, Richard Plant, Clare Teng, Marya Bazzi, Alvaro Cabrejas Egea, Ryan Sze-Yin Chan, David Salvador Jasin, Martin Stoffel, Kirstie J. Whitaker, Jules Manser*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/215a55741fbe4baad173468f93336a7d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/215a55741fbe4baad173468f93336a7d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The rise of datathons, also known as data or data science hackathons, has provided a platform to collaborate, learn, and innovate quickly. Despite their significant potential benefits, organizations often struggle to effectively work with data due to a lack of clear guidelines and best practices for potential issues that might arise. Drawing on our own experiences and insights from organizing +80 datathon challenges with +60 partnership organizations since 2016, we provide a guide that serves as a resource for organizers to navigate the data-related complexities of datathons. We apply our proposed framework to 10 case studies.

        ----

        ## [458] Convolution Monge Mapping Normalization for learning on sleep data

        **Authors**: *Théo Gnassounou, Rémi Flamary, Alexandre Gramfort*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html)

        **Abstract**:

        In many machine learning applications on signals and biomedical data, especially electroencephalogram (EEG), one major challenge is the variability of the data across subjects, sessions, and hardware devices. In this work, we propose a new method called Convolutional Monge Mapping Normalization ($\texttt{CMMN}$), which consists in filtering the signals in order to adapt their power spectrum density (PSD) to a Wasserstein barycenter estimated on training data. $\texttt{CMMN}$ relies on novel closed-form solutions for optimal transport mappings and barycenters and provides individual test time adaptation to new data without needing to retrain a prediction model. Numerical experiments on sleep EEG data show that $\texttt{CMMN}$ leads to significant and consistent performance gains independent from the neural network architecture when adapting between subjects, sessions, and even datasets collected with different hardware. Notably our performance gain is on par with much more numerically intensive Domain Adaptation (DA) methods and can be used in conjunction with those for even better performances.

        ----

        ## [459] Online learning of long-range dependencies

        **Authors**: *Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, João Sacramento*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2184d8450c8a641f9a10c49279087c97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2184d8450c8a641f9a10c49279087c97-Abstract-Conference.html)

        **Abstract**:

        Online learning holds the promise of enabling efficient long-term credit assignment in recurrent neural networks. However, current algorithms fall short of offline backpropagation by either not being scalable or failing to learn long-range dependencies. Here we present a high-performance online learning algorithm that merely doubles the memory and computational requirements of a single inference pass. We achieve this by leveraging independent recurrent modules in multi-layer networks, an architectural motif that has recently been shown to be particularly powerful. Experiments on synthetic memory problems and on the challenging long-range arena benchmark suite reveal that our algorithm performs competitively, establishing a new standard for what can be achieved through online learning. This ability to learn long-range dependencies offers a new perspective on learning in the brain and opens a promising avenue in neuromorphic computing.

        ----

        ## [460] Fast Rank-1 Lattice Targeted Sampling for Black-box Optimization

        **Authors**: *Yueming Lyu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/218d0323ce235090b43a1166159ee328-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/218d0323ce235090b43a1166159ee328-Abstract-Conference.html)

        **Abstract**:

        Black-box optimization has gained great attention for its success in recent applications.    However, scaling up to high-dimensional problems with good query efficiency remains challenging. This paper proposes a novel Rank-1 Lattice Targeted Sampling (RLTS) technique to address this issue. Our RLTS benefits from random rank-1 lattice Quasi-Monte Carlo, which enables us to perform fast local exact Gaussian processes (GP)  training and inference with $O(n \log n)$ complexity w.r.t. $n$ batch samples. Furthermore, we developed a fast coordinate searching method with $O(n \log n)$ time complexity for fast targeted sampling. The fast computation enables us to plug our RLTS into the sampling phase of stochastic optimization methods. This improves the query efficiency while  scaling up to higher dimensional problems than Bayesian optimization. Moreover,  to construct rank-1 lattices efficiently, we proposed a closed-form construction.  Extensive experiments on challenging benchmark test functions and black-box prompt fine-tuning for large language models demonstrate the query efficiency of our RLTS technique.

        ----

        ## [461] DreamHuman: Animatable 3D Avatars from Text

        **Authors**: *Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Fieraru, Cristian Sminchisescu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21912f7057935149fa58408ee8cb460e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21912f7057935149fa58408ee8cb460e-Abstract-Conference.html)

        **Abstract**:

        We present \emph{DreamHuman}, a method to generate realistic animatable 3D human avatar models entirely from textual descriptions. Recent text-to-3D methods have made considerable strides in generation, but are still lacking in important aspects. Control and often spatial resolution remain limited, existing methods produce fixed rather than 3D human models that can be placed in different poses (i.e. re-posable or animatable), and anthropometric consistency for complex structures like people remains a challenge. \emph{DreamHuman} connects large text-to-image synthesis models, neural radiance fields, and statistical human body models in a novel optimization framework. This makes it possible to generate dynamic 3D human avatars with high-quality textures and learnt per-instance rigid and non rigid geometric deformations. We demonstrate that our method is capable to generate a wide variety of animatable, realistic 3D human models from text. These have diverse appearance, clothing, skin tones and body shapes, and outperform both generic text-to-3D approaches and previous text-based 3D avatar generators in visual fidelity.

        ----

        ## [462] Rank-1 Matrix Completion with Gradient Descent and Small Random Initialization

        **Authors**: *Daesung Kim, Hye Won Chung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21c426323068204f4199c490d730e88e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21c426323068204f4199c490d730e88e-Abstract-Conference.html)

        **Abstract**:

        The nonconvex formulation of the matrix completion problem has received significant attention in recent years due to its affordable complexity compared to the convex formulation. Gradient Descent (GD) is a simple yet efficient baseline algorithm for solving nonconvex optimization problems. The success of GD has been witnessed in many different problems in both theory and practice when it is combined with random initialization. However, previous works on matrix completion require either careful initialization or regularizers to prove the convergence of GD. In this paper, we study the rank-1 symmetric matrix completion and prove that GD converges to the ground truth when small random initialization is used. We show that in a logarithmic number of iterations, the trajectory enters the region where local convergence occurs. We provide an upper bound on the initialization size that is sufficient to guarantee the convergence, and show that a larger initialization can be used as more samples are available. We observe that the implicit regularization effect of GD plays a critical role in the analysis, and for the entire trajectory, it prevents each entry from becoming much larger than the others.

        ----

        ## [463] No-Regret Learning in Dynamic Competition with Reference Effects Under Logit Demand

        **Authors**: *Mengzi Amy Guo, Donghao Ying, Javad Lavaei, Zuo-Jun Max Shen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21c9fd36a6d94a491bf330a0ba0e5f6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21c9fd36a6d94a491bf330a0ba0e5f6e-Abstract-Conference.html)

        **Abstract**:

        This work is dedicated to the algorithm design in a competitive framework, with the primary goal of learning a stable equilibrium. We consider the dynamic price competition between two firms operating within an opaque marketplace, where each firm lacks information about its competitor. The demand follows the multinomial logit (MNL) choice model, which depends on the consumers' observed price and their reference price, and consecutive periods in the repeated games are connected by reference price updates. We use the notion of stationary Nash equilibrium (SNE), defined as the fixed point of the equilibrium pricing policy for the single-period game, to simultaneously capture the long-run market equilibrium and stability. We propose the online projected gradient ascent algorithm (OPGA), where the firms adjust prices using the first-order derivatives of their log-revenues that can be obtained from the market feedback mechanism. Despite the absence of typical properties required for the convergence of online games, such as strong monotonicity and variational stability, we demonstrate that under diminishing step-sizes, the price and reference price paths generated by OPGA converge to the unique SNE, thereby achieving the no-regret learning and a stable market. Moreover, with appropriate step-sizes, we prove that this convergence exhibits a rate of $\mathcal{O}(1/t)$.

        ----

        ## [464] VoxDet: Voxel Learning for Novel Instance Detection

        **Authors**: *Bowen Li, Jiashun Wang, Yaoyu Hu, Chen Wang, Sebastian A. Scherer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21f1c5bbf2519321c1bee9bfa9edcd46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21f1c5bbf2519321c1bee9bfa9edcd46-Abstract-Conference.html)

        **Abstract**:

        Detecting unseen instances based on multi-view templates is a challenging problem due to its open-world nature. Traditional methodologies, which primarily rely on $2 \mathrm{D}$ representations and matching techniques, are often inadequate in handling pose variations and occlusions. To solve this, we introduce VoxDet, a pioneer 3D geometry-aware framework that fully utilizes the strong 3D voxel representation and reliable voxel matching mechanism. VoxDet first ingeniously proposes template voxel aggregation (TVA) module, effectively transforming multi-view 2D images into 3D voxel features. By leveraging associated camera poses, these features are aggregated into a compact 3D template voxel. In novel instance detection, this voxel representation demonstrates heightened resilience to occlusion and pose variations. We also discover that a $3 \mathrm{D}$ reconstruction objective helps to pre-train the 2D-3D mapping in TVA. Second, to quickly align with the template voxel, VoxDet incorporates a Query Voxel Matching (QVM) module. The 2D queries are first converted into their voxel representation with the learned 2D-3D mapping. We find that since the 3D voxel representations encode the geometry, we can first estimate the relative rotation and then compare the aligned voxels, leading to improved accuracy and efficiency. In addition to method, we also introduce the first instance detection benchmark, RoboTools, where 20 unique instances are video-recorded with camera extrinsic. RoboTools also provides 24 challenging cluttered scenarios with more than $9 \mathrm{k}$ box annotations. Exhaustive experiments are conducted on the demanding LineMod-Occlusion, YCB-video, and RoboTools benchmarks, where VoxDet outperforms various $2 \mathrm{D}$ baselines remarkably with faster speed. To the best of our knowledge, VoxDet is the first to incorporate implicit 3D knowledge for 2D novel instance detection tasks.

        ----

        ## [465] Evaluating and Inducing Personality in Pre-trained Language Models

        **Authors**: *Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, Yixin Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html)

        **Abstract**:

        Standardized and quantified evaluation of machine behaviors is a crux of understanding LLMs. In this study, we draw inspiration from psychometric studies by leveraging human personality theory as a tool for studying machine behaviors. Originating as a philosophical quest for human behaviors, the study of personality delves into how individuals differ in thinking, feeling, and behaving. Toward building and understanding human-like social machines, we are motivated to ask: Can we assess machine behaviors by leveraging human psychometric tests in a **principled** and **quantitative** manner? If so, can we induce a specific personality in LLMs? To answer these questions, we introduce the Machine Personality Inventory (MPI) tool for studying machine behaviors; MPI follows standardizedpersonality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By systematically evaluating LLMs with MPI, we provide the first piece of evidence demonstrating the efficacy of MPI in studying LLMs behaviors. We further devise a Personality Prompting (P$^2$) method to induce LLMs with specific personalities in a **controllable** way, capable of producing diverse and verifiable behaviors. We hope this work sheds light on future studies by adopting personality as the essential indicator for various downstream tasks, and could further motivate research into equally intriguing human-like machine behaviors.

        ----

        ## [466] On Measuring Fairness in Generative Models

        **Authors**: *Christopher T. H. Teo, Milad Abdollahzadeh, Ngai-Man Cheung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/220165f9c7f51163b73c8c7fff578b4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/220165f9c7f51163b73c8c7fff578b4e-Abstract-Conference.html)

        **Abstract**:

        Recently, there has been increased interest in fair generative models. In this work,we conduct, for the first time, an in-depth study on fairness measurement, acritical component in gauging progress on fair generative models. We make threecontributions. First, we conduct a study that reveals that the existing fairnessmeasurement framework has considerable measurement errors, even when highlyaccurate sensitive attribute (SA) classifiers are used. These findings cast doubtson previously reported fairness improvements. Second, to address this issue,we propose CLassifier Error-Aware Measurement (CLEAM), a new frameworkwhich uses a statistical model to account for inaccuracies in SA classifiers. Ourproposed CLEAM reduces measurement errors significantly, e.g., 4.98%â†’0.62%for StyleGAN2 w.r.t. Gender. Additionally, CLEAM achieves this with minimaladditional overhead. Third, we utilize CLEAM to measure fairness in importanttext-to-image generator and GANs, revealing considerable biases in these modelsthat raise concerns about their applications. Code and more resources: https://sutd-visual-computing-group.github.io/CLEAM/.

        ----

        ## [467] IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against Unauthorized Data Usage in Diffusion-Based Generative AI

        **Authors**: *Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/222dda29587fbc2979ca99fd5ed00735-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/222dda29587fbc2979ca99fd5ed00735-Abstract-Conference.html)

        **Abstract**:

        Diffusion-based image generation models,  such as Stable Diffusion or DALLÂ·E 2, are able to learn from given images and generate high-quality samples following the guidance from prompts. For instance, they can be used to create artistic images that mimic the style of an artist based on his/her original artworks or to maliciously edit the original images for fake content. However, such ability also brings serious ethical issues without proper authorization from the owner of the original images. In response, several attempts have been made to protect the original images from such unauthorized data usage by adding imperceptible perturbations, which are designed to mislead the diffusion model and make it unable to properly generate new samples. In this work, we introduce a perturbation purification platform, named IMPRESS, to evaluate the effectiveness of imperceptible perturbations as a protective measure.IMPRESS is based on the key observation that imperceptible perturbations could lead to a perceptible inconsistency between the original image and the diffusion-reconstructed image, which can be used to devise a new optimization strategy for purifying the image, which may weaken the protection of the original image from unauthorized data usage (e.g., style mimicking, malicious editing).The proposed IMPRESS platform offers a comprehensive evaluation of several contemporary protection methods, and can be used as an evaluation platform for future protection methods.

        ----

        ## [468] Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

        **Authors**: *Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2232e8fee69b150005ac420bfa83d705-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2232e8fee69b150005ac420bfa83d705-Abstract-Conference.html)

        **Abstract**:

        Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that RoCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, RoCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, RoCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.

        ----

        ## [469] Block-Coordinate Methods and Restarting for Solving Extensive-Form Games

        **Authors**: *Darshan Chakrabarti, Jelena Diakonikolas, Christian Kroer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2280faacd674566a5eace1bd1098f507-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2280faacd674566a5eace1bd1098f507-Abstract-Conference.html)

        **Abstract**:

        Coordinate descent methods are popular in machine learning and optimization for their simple sparse updates and excellent practical performance. In the context of large-scale sequential game solving, these same properties would be attractive, but until now no such methods were known, because the strategy spaces do not satisfy the typical separable block structure exploited by such methods.We present the first cyclic coordinate-descent-like method for the polytope of sequence-form strategies, which form the strategy spaces for the players in an extensive-form game (EFG). Our method exploits the recursive structure of the proximal update induced by what are known as dilated regularizers, in order to allow for a pseudo block-wise update.We show that our method enjoys a O(1/T) convergence rate to a two-player zero-sum Nash equilibrium, while avoiding the worst-case polynomial scaling with the number of blocks common to cyclic methods. We empirically show that our algorithm usually performs better than other state-of-the-art first-order methods (i.e., mirror prox), and occasionally can even beat CFR$^+$, a state-of-the-art algorithm for numerical equilibrium computation in zero-sum EFGs. We then introduce a restarting heuristic for EFG solving. We show empirically that restarting can lead to speedups, sometimes huge, both for our cyclic method, as well as for existing methods such as mirror prox and predictive CFR$^+$.

        ----

        ## [470] ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction

        **Authors**: *Jia Guo, Shuai Lu, Lize Jia, Weihang Zhang, Huiqi Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html)

        **Abstract**:

        Most advanced unsupervised anomaly detection (UAD) methods rely on modeling feature representations of frozen encoder networks pre-trained on large-scale datasets, e.g. ImageNet. However, the features extracted from the encoders that are borrowed from natural image domains coincide little with the features required in the target UAD domain, such as industrial inspection and medical imaging. In this paper, we propose a novel epistemic UAD method, namely ReContrast, which optimizes the entire network to reduce biases towards the pre-trained image domain and orients the network in the target domain. We start with a feature reconstruction approach that detects anomalies from errors. Essentially, the elements of contrastive learning are elegantly embedded in feature reconstruction to prevent the network from training instability, pattern collapse, and identical shortcut, while simultaneously optimizing both the encoder and decoder on the target domain. To demonstrate our transfer ability on various image domains, we conduct extensive experiments across two popular industrial defect detection benchmarks and three medical image UAD tasks, which shows our superiority over current state-of-the-art methods.

        ----

        ## [471] Transferable Adversarial Robustness for Categorical Data via Universal Robust Embeddings

        **Authors**: *Klim Kireev, Maksym Andriushchenko, Carmela Troncoso, Nicolas Flammarion*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/22a25fc3da528794d52664dacc7bd470-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/22a25fc3da528794d52664dacc7bd470-Abstract-Conference.html)

        **Abstract**:

        Research on adversarial robustness is primarily focused on image and text data. Yet, many scenarios in which lack of robustness can result in serious risks, such as fraud detection, medical diagnosis, or recommender systems often do not rely on images or text but instead on tabular data. Adversarial robustness in tabular data poses two serious challenges. First, tabular datasets often contain categorical features, and therefore cannot be tackled directly with existing optimization procedures. Second, in the tabular domain, algorithms that are not based on deep networks are widely used and offer great performance, but algorithms to enhance robustness are tailored to neural networks (e.g. adversarial training).In this paper, we tackle both challenges. We present a method that allows us to train adversarially robust deep networks for tabular data and to transfer this robustness to other classifiers via universal robust embeddings tailored to categorical data. These embeddings, created using a bilevel alternating minimization framework, can be transferred to boosted trees or random forests making them robust without the need for adversarial training while preserving their high accuracy on tabular data. We show that our methods outperform existing techniques within a practical threat model suitable for tabular data.

        ----

        ## [472] Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection

        **Authors**: *Chengsen Wang, Zirui Zhuang, Qi Qi, Jingyu Wang, Xingyu Wang, Haifeng Sun, Jianxin Liao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html)

        **Abstract**:

        Many unsupervised methods have recently been proposed for multivariate time series anomaly detection. However, existing works mainly focus on stable data yet often omit the drift generated from non-stationary environments, which may lead to numerous false alarms. We propose **D**ynamic **D**ecomposition with **D**iffusion **R**econstruction (D$^3$R), a novel anomaly detection network for real-world unstable data to fill the gap. D$^3$R tackles the drift via decomposition and reconstruction. In the decomposition procedure, we utilize data-time mix-attention to dynamically decompose long-period multivariate time series, overcoming the limitation of the local sliding window. The information bottleneck is critical yet difficult to determine in the reconstruction procedure. To avoid retraining once the bottleneck changes, we control it externally by noise diffusion and directly reconstruct the polluted data. The whole model can be trained end-to-end. Extensive experiments on various real-world datasets demonstrate that D$^3$R significantly outperforms existing methods, with a 11% average relative improvement over the previous SOTA models.

        ----

        ## [473] MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

        **Authors**: *Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma, Zhiguang Cao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recently, Meta-Black-Box Optimization with Reinforcement Learning (MetaBBO-RL) has showcased the power of leveraging RL at the meta-level to mitigate manual fine-tuning of low-level black-box optimizers. However, this field is hindered by the lack of a unified benchmark. To fill this gap, we introduce MetaBox, the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. Our MetaBox is open-source and accessible at: https://github.com/GMC-DRL/MetaBox.

        ----

        ## [474] Improving Adversarial Robustness via Information Bottleneck Distillation

        **Authors**: *Huafeng Kuang, Hong Liu, Yongjian Wu, Shin'ichi Satoh, Rongrong Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html)

        **Abstract**:

        Previous studies have shown that optimizing the information bottleneck can significantly improve the robustness of deep neural networks. Our study closely examines the information bottleneck principle and proposes an Information Bottleneck Distillation approach. This specially designed, robust distillation technique utilizes prior knowledge obtained from a robust pre-trained model to boost information bottlenecks.  Specifically, we propose two distillation strategies that align with the two optimization processes of the information bottleneck. Firstly, we use a robust soft-label distillation method to increase the mutual information between latent features and output prediction. Secondly, we introduce an adaptive feature distillation method that automatically transfers relevant knowledge from the teacher model to the student model, thereby reducing the mutual information between the input and latent features. We conduct extensive experiments to evaluate our approach's robustness against state-of-the-art adversarial attackers such as PGD-attack and AutoAttack. Our experimental results demonstrate the effectiveness of our approach in significantly improving adversarial robustness. Our code is available at https://github.com/SkyKuang/IBD.

        ----

        ## [475] Reading Relevant Feature from Global Representation Memory for Visual Object Tracking

        **Authors**: *Xinyu Zhou, Pinxue Guo, Lingyi Hong, Jinglun Li, Wei Zhang, Weifeng Ge, Wenqiang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2349293cb1bf2ce36d5c566f660f957e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2349293cb1bf2ce36d5c566f660f957e-Abstract-Conference.html)

        **Abstract**:

        Reference features from a template or historical frames are crucial for visual object tracking. Prior works utilize all features from a fixed template or memory for visual object tracking. However, due to the dynamic nature of videos, the required reference historical information for different search regions at different time steps is also inconsistent. Therefore, using all features in the template and memory can lead to redundancy and impair tracking performance. To alleviate this issue, we propose a novel tracking paradigm, consisting of a relevance attention mechanism and a global representation memory, which can adaptively assist the search region in selecting the most relevant historical information from reference features. Specifically, the proposed relevance attention mechanism in this work differs from previous approaches in that it can dynamically choose and build the optimal global representation memory for the current frame by accessing cross-frame information globally. Moreover, it can flexibly read the relevant historical information from the constructed memory to reduce redundancy and counteract the negative effects of harmful information. Extensive experiments validate the effectiveness of the proposed method, achieving competitive performance on five challenging datasets with 71 FPS.

        ----

        ## [476] Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks

        **Authors**: *Eshaan Nichani, Alex Damian, Jason D. Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/236b6a814a1d2c0ff504ca7bf380f7ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/236b6a814a1d2c0ff504ca7bf380f7ff-Abstract-Conference.html)

        **Abstract**:

        One of the central questions in the theory of deep learning is to understand how neural networks learn hierarchical features. The ability of deep networks to extract salient features is crucial to both their outstanding generalization ability and the modern deep learning paradigm of pretraining and finetuneing. However, this feature learning process remains poorly understood from a theoretical perspective, with existing analyses largely restricted to two-layer networks. In this work we show that three-layer neural networks have provably richer feature learning capabilities than two-layer networks. We analyze the features learned by a three-layer network trained with layer-wise gradient descent, and present a general purpose theorem which upper bounds the sample complexity and width needed to achieve low test error when the target has specific hierarchical structure. We instantiate our framework in specific statistical learning settings -- single-index models and functions of quadratic features -- and show that in the latter setting three-layer networks obtain a sample complexity improvement over all existing guarantees for two-layer networks. Crucially, this sample complexity improvement relies on the ability of three-layer networks to efficiently learn nonlinear features. We then establish a concrete optimization-based depth separation by constructing a function which is efficiently learnable via gradient descent on a three-layer network, yet cannot be learned efficiently by a two-layer network. Our work makes progress towards understanding the provable benefit of three-layer neural networks over two-layer networks in the feature learning regime.

        ----

        ## [477] Lockdown: Backdoor Defense for Federated Learning with Isolated Subspace Training

        **Authors**: *Tiansheng Huang, Sihao Hu, Ka Ho Chow, Fatih Ilhan, Selim F. Tekin, Ling Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2376f25ef1725a9e3516ee3c86a59f46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2376f25ef1725a9e3516ee3c86a59f46-Abstract-Conference.html)

        **Abstract**:

        Federated learning (FL) is vulnerable to backdoor attacks due to its distributed computing nature. Existing defense solution usually requires larger amount of computation in either the training or testing phase, which limits their practicality in the resource-constrain scenarios. A more practical defense, i.e., neural network (NN) pruning based defense has been proposed in centralized backdoor setting. However, our empirical study shows that traditional pruning-based solution suffers \textit{poison-coupling} effect in FL, which significantly degrades the defense performance.This paper presents Lockdown, an isolated subspace training method to mitigate the poison-coupling effect. Lockdown follows three key procedures. First, it modifies the training protocol by isolating the training subspaces for different clients. Second, it utilizes randomness in initializing isolated subspacess, and performs subspace pruning and subspace recovery to segregate the subspaces between malicious and benign clients. Third, it introduces quorum consensus to cure the global model by purging malicious/dummy parameters. Empirical results show that Lockdown achieves \textit{superior} and \textit{consistent} defense performance compared to existing representative approaches against backdoor attacks. Another value-added property of Lockdown is the communication-efficiency and model complexity reduction, which are both critical for resource-constrain FL scenario. Our code is available at \url{https://github.com/git-disl/Lockdown}.

        ----

        ## [478] Robust Lipschitz Bandits to Adversarial Corruptions

        **Authors**: *Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/238f3b98bbe998b4f2234443907fe663-Abstract-Conference.html)

        **Abstract**:

        Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

        ----

        ## [479] Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily

        **Authors**: *Langzhang Liang, Xiangjing Hu, Zenglin Xu, Zixing Song, Irwin King*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23aa2163dea287441ebebc1295d5b3fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23aa2163dea287441ebebc1295d5b3fc-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have been shown to achieve remarkable performance on node classification tasks by exploiting both graph structures and node features. The majority of existing GNNs rely on the implicit homophily assumption. Recent studies have demonstrated that GNNs may struggle to model heterophilous graphs where nodes with different labels are more likely connected. To address this issue, we propose a generic GNN applicable to both homophilous and heterophilous graphs, namely Low-Rank Graph Neural Network (LRGNN). Our analysis demonstrates that a signed graph's global label relationship matrix has a low rank. This insight inspires us to predict the label relationship matrix by solving a robust low-rank matrix approximation problem, as prior research has proven that low-rank approximation could achieve perfect recovery under certain conditions. The experimental results reveal that the solution bears a strong resemblance to the label relationship matrix, presenting two advantages for graph modeling: a block diagonal structure and varying distributions of within-class and between-class entries.

        ----

        ## [480] Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts

        **Authors**: *Ruth Dannenfelser, Jeffrey Zhong, Ran Zhang, Vicky Yao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23e3d86c9a19d0caf2ec997e73dfcfbd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/23e3d86c9a19d0caf2ec997e73dfcfbd-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Many of the most commonly explored natural language processing (NLP) information extraction tasks can be thought of as evaluations of declarative knowledge, or fact-based information extraction. Procedural knowledge extraction, i.e., breaking down a described process into a series of steps, has received much less attention, perhaps in part due to the lack of structured datasets that capture the knowledge extraction process from end-to-end. To address this unmet need, we present FlaMBé (Flow annotations for Multiverse Biological entities), a collection of expert-curated datasets across a series of complementary tasks that capture procedural knowledge in biomedical texts. This dataset is inspired by the observation that one ubiquitous source of procedural knowledge that is described as unstructured text is within academic papers describing their methodology. The workflows annotated in FlaMBé are from texts in the burgeoning field of single cell research, a research area that has become notorious for the number of software tools and complexity of workflows used. Additionally, FlaMBé provides, to our knowledge, the largest manually curated named entity recognition (NER) and disambiguation (NED) datasets for tissue/cell type, a fundamental biological entity that is critical for knowledge extraction in the biomedical research domain. Beyond providing a valuable dataset to enable further development of NLP models for procedural knowledge extraction, automating the process of workflow mining also has important implications for advancing reproducibility in biomedical research.

        ----

        ## [481] RRHF: Rank Responses to Align Language Models with Human Feedback

        **Authors**: *Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html)

        **Abstract**:

        Reinforcement Learning from Human Feedback (RLHF) facilitates the alignment of large language models with human preferences, significantly enhancing the quality of interactions between humans and models. InstructGPT implements RLHF through several stages, including Supervised Fine-Tuning (SFT), reward model training, and Proximal Policy Optimization (PPO). However, PPO is sensitive to hyperparameters and requires multiple models in its standard implementation, making it hard to train and scale up to larger parameter counts.In contrast, we propose a novel learning paradigm called RRHF, which scores sampled responses from different sources via a logarithm of conditional probabilities and learns to align these probabilities with human preferences through ranking loss.RRHF can leverage sampled responses from various sources including the model responses from itself, other large language model responses, and human expert responses to learn to rank them.RRHF only needs 1 to 2 models during tuning and can efficiently align language models with human preferences robustly without complex hyperparameter tuning. Additionally, RRHF can be considered an extension of SFT and reward model training while being simpler than PPO in terms of coding, model counts, and hyperparameters. We evaluate RRHF on the Helpful and Harmless dataset, demonstrating comparable alignment performance with PPO by reward model score and human labeling.Extensive experiments show that the performance of RRHF is highly related to sampling quality which suggests RRHF is a best-of-$n$ learner.

        ----

        ## [482] Sparsity-Preserving Differentially Private Training of Large Embedding Models

        **Authors**: *Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chiyuan Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/23ff02034404b65776080cbf7148addd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/23ff02034404b65776080cbf7148addd-Abstract-Conference.html)

        **Abstract**:

        As the use of large embedding models in recommendation systems and language applications increases, concerns over user data privacy have also risen.  DP-SGD, a training algorithm that combines differential privacy with stochastic gradient descent, has been the workhorse in protecting user privacy without compromising model accuracy by much. However, applying DP-SGD naively to embedding models can destroy gradient sparsity, leading to reduced training efficiency. To address this issue, we present two new algorithms, DP-FEST and DP-AdaFEST, that preserve gradient sparsity during the private training of large embedding models.  Our algorithms achieve substantial reductions ($10^6 \times$) in gradient size, while maintaining comparable levels of accuracy, on benchmark real-world datasets.

        ----

        ## [483] Bayesian target optimisation for high-precision holographic optogenetics

        **Authors**: *Marcus Triplett, Marta Gajowa, Hillel Adesnik, Liam Paninski*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html)

        **Abstract**:

        Two-photon optogenetics has transformed our ability to probe the structure and function of neural circuits. However, achieving precise optogenetic control of neural ensemble activity has remained fundamentally constrained by the problem of off-target stimulation (OTS): the inadvertent activation of nearby non-target neurons due to imperfect confinement of light onto target neurons. Here we propose a novel computational approach to this problem called Bayesian target optimisation. Our approach uses nonparametric Bayesian inference to model neural responses to optogenetic stimulation, and then optimises the laser powers and optical target locations needed to achieve a desired activity pattern with minimal OTS. We validate our approach in simulations and using data from in vitro experiments, showing that Bayesian target optimisation considerably reduces OTS across all conditions we test. Together, these results establish our ability to overcome OTS, enabling optogenetic stimulation with substantially improved precision.

        ----

        ## [484] From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models

        **Authors**: *Roy Uziel, Or Dinari, Oren Freifeld*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/240cc9ac4789351653d13cfcba4ee85c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/240cc9ac4789351653d13cfcba4ee85c-Abstract-Conference.html)

        **Abstract**:

        In the task of semi-supervised video object segmentation, the input is the binary mask of an object in the first frame, and the desired output consists of the corresponding masks of that object in the subsequent frames. Existing leading solutions have two main drawbacks: 1) an expensive and typically-supervised training on videos; 2) a large memory footprint during inference. Here we present a training-free solution, with a low-memory footprint, that yields state-of-the-art results. The proposed method combines pre-trained deep learning-based features (trained on still images) with more classical methods for streaming-data clustering. Designed to adapt to temporal concept drifts and generalize to diverse video content without relying on annotated images or videos, the method eliminates the need for additional training or fine-tuning, ensuring fast inference and immediate applicability to new videos. Concretely, we represent an object via a dynamic ensemble of temporally- and spatially-coherent mixtures over a representation built from pre-trained ViT features and positional embeddings. A convolutional conditional random field further improves spatial coherence and helps reject outliers. We demonstrate the efficacy of the method on key benchmarks: the DAVIS-2017 and YouTube-VOS 2018 validation datasets. Moreover, by the virtue of the low-memory footprint of the compact cluster-based representation, the method scales gracefully to high-resolution ViT features. Our code is available at https://github.com/BGU-CS-VIL/Training-Free-VOS

        ----

        ## [485] How hard are computer vision datasets? Calibrating dataset difficulty to viewing time

        **Authors**: *David Mayo, Jesse Cummings, Xinyu Lin, Dan Gutfreund, Boris Katz, Andrei Barbu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2432e9646556f3a98dc78c1f4a10481b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2432e9646556f3a98dc78c1f4a10481b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Humans outperform object recognizers despite the fact that models perform well on current datasets, including those explicitly designed to challenge machines with debiased images or distribution shift. This problem persists, in part, because we have no guidance on the absolute difficulty of an image or dataset making it hard to objectively assess progress toward human-level performance, to cover the range of human abilities, and to increase the challenge posed by a dataset. We develop a dataset difficulty metric MVT, Minimum Viewing Time, that addresses these three problems. Subjects view an image that flashes on screen and then classify the object in the image. Images that require brief flashes to recognize are easy, those which require seconds of viewing are hard. We compute the ImageNet and ObjectNet image difficulty distribution, which we find significantly undersamples hard images. Nearly 90% of current benchmark performance is derived from images that are easy for humans. Rather than hoping that we will make harder datasets, we can for the first time objectively guide dataset difficulty during development. We can also subset recognition performance as a function of difficulty: model performance drops precipitously while human performance remains stable. Difficulty provides a new lens through which to view model performance, one which uncovers new scaling laws: vision-language models stand out as being the most robust and human-like while all other techniques scale poorly. We release tools to automatically compute MVT, along with image sets which are tagged by difficulty. Objective image difficulty has practical applications – one can measure how hard a test set is before deploying a real-world system – and scientific applications such as discovering the neural correlates of image difficulty and enabling new object recognition techniques that eliminate the benchmark-vs- real-world performance gap.

        ----

        ## [486] What Knowledge Gets Distilled in Knowledge Distillation?

        **Authors**: *Utkarsh Ojha, Yuheng Li, Anirudh Sundara Rajan, Yingyu Liang, Yong Jae Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation aims to transfer useful information from a teacher network to a student network, with the primary goal of improving the student's performance for the task at hand. Over the years, there has a been a deluge of novel techniques and use cases of knowledge distillation. Yet, despite the various improvements, there seems to be a glaring gap in the community's fundamental understanding of the process. Specifically, what is the knowledge that gets distilled in knowledge distillation?  In other words, in what ways does the student become similar to the teacher? Does it start to localize objects in the same way? Does it get fooled by the same adversarial samples? Does its data invariance properties become similar? Our work presents a comprehensive study to try to answer these questions. We show that existing methods can indeed indirectly distill these properties beyond improving task performance. We further study why knowledge distillation might work this way, and show that our findings have practical implications as well.

        ----

        ## [487] Kernelized Cumulants: Beyond Kernel Mean Embeddings

        **Authors**: *Patric Bonnier, Harald Oberhauser, Zoltán Szabó*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/243697ace81f57daef8737ff2c5cffd3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/243697ace81f57daef8737ff2c5cffd3-Abstract-Conference.html)

        **Abstract**:

        In $\mathbb{R}^d$, it is well-known that cumulants provide an alternative to moments that can achieve the same goals with numerous benefits such as lower variance estimators. In this paper we extend cumulants to reproducing kernel Hilbert spaces (RKHS) using tools from tensor algebras and show that they are computationally tractable by a kernel trick. These kernelized cumulants provide a new set of all-purpose statistics; the classical maximum mean discrepancy and Hilbert-Schmidt independence criterion arise as the degree one objects in our general construction. We argue both theoretically and empirically (on synthetic, environmental, and traffic data analysis) that going beyond degree one has several advantages and can be achieved with the same computational complexity and minimal overhead in our experiments.

        ----

        ## [488] Contrastive Training of Complex-Valued Autoencoders for Object Discovery

        **Authors**: *Aleksandar Stanic, Anand Gopalakrishnan, Kazuki Irie, Jürgen Schmidhuber*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2439ec22091b9d6cfbebf3284b40116e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2439ec22091b9d6cfbebf3284b40116e-Abstract-Conference.html)

        **Abstract**:

        Current state-of-the-art object-centric models use slots and attention-based routing for binding. However, this class of models has several conceptual limitations: the number of slots is hardwired; all slots have equal capacity; training has high computational cost; there are no object-level relational factors within slots. Synchrony-based models in principle can address these limitations by using complex-valued activations which store binding information in their phase components. However, working examples of such synchrony-based models have been developed only very recently, and are still limited to toy grayscale datasets and simultaneous storage of less than three objects in practice. Here we introduce architectural modifications and a novel contrastive learning method that greatly improve the state-of-the-art synchrony-based model. For the first time, we obtain a class of synchrony-based models capable of discovering objects in an unsupervised manner in multi-object color datasets and simultaneously representing more than three objects.

        ----

        ## [489] Non-adversarial training of Neural SDEs with signature kernel scores

        **Authors**: *Zacharia Issa, Blanka Horvath, Maud Lemercier, Cristopher Salvi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html)

        **Abstract**:

        Neural SDEs are continuous-time generative models for sequential data. State-of-the-art performance for irregular time series generation has been previously obtained by training these models adversarially as GANs. However, as typical for GAN architectures, training is notoriously unstable, often suffers from mode collapse, and requires specialised techniques such as weight clipping and gradient penalty to mitigate these issues. In this paper, we introduce a novel class of scoring rules on pathspace based on signature kernels and use them as objective for training Neural SDEs non-adversarially. By showing strict properness of such kernel scores and consistency of the corresponding estimators, we provide existence and uniqueness guarantees for the minimiser. With this formulation, evaluating the generator-discriminator pair amounts to solving a system of linear path-dependent PDEs which allows for memory-efficient adjoint-based backpropagation. Moreover, because the proposed kernel scores are well-defined for paths with values in infinite dimensional spaces of functions, our framework can be easily extended to generate spatiotemporal data. Our procedure significantly outperforms alternative ways of training Neural SDEs on a variety of tasks including the simulation of rough volatility models, the conditional probabilistic forecasts of real-world forex pairs where the conditioning variable is an observed past trajectory, and the mesh-free generation of limit order book dynamics.

        ----

        ## [490] Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

        **Authors**: *Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2468f84a13ff8bb6767a67518fb596eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2468f84a13ff8bb6767a67518fb596eb-Abstract-Conference.html)

        **Abstract**:

        Text-to-Image diffusion models have made tremendous progress over the past two years, enabling the generation of highly realistic images based on open-domain text descriptions. However, despite their success, text descriptions often struggle to adequately convey detailed controls, even when composed of long and complex texts. Moreover, recent studies have also shown that these models face challenges in understanding such complex texts and generating the corresponding images. Therefore, there is a growing need to enable more control modes beyond text description. In this paper, we introduce Uni-ControlNet, a unified framework that allows for the simultaneous utilization of different local controls (e.g., edge maps, depth map, segmentation masks) and global controls (e.g., CLIP image embeddings) in a flexible and composable manner within one single model. Unlike existing methods, Uni-ControlNet only requires the fine-tuning of two additional adapters upon frozen pre-trained text-to-image diffusion models, eliminating the huge cost of training from scratch. Moreover, thanks to some dedicated adapter designs, Uni-ControlNet only necessitates a constant number (i.e., 2) of adapters, regardless of the number of local or global controls used. This not only reduces the fine-tuning costs and model size, making it more suitable for real-world deployment, but also facilitate composability of different conditions. Through both quantitative and qualitative comparisons, Uni-ControlNet demonstrates its superiority over existing methods in terms of controllability, generation quality and composability. Code is available at https://github.com/ShihaoZhaoZSH/Uni-ControlNet.

        ----

        ## [491] Loss Decoupling for Task-Agnostic Continual Learning

        **Authors**: *Yan-Shuo Liang, Wu-Jun Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/249f73e01f0a2bb6c8d971b565f159a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/249f73e01f0a2bb6c8d971b565f159a7-Abstract-Conference.html)

        **Abstract**:

        Continual learning requires the model to learn multiple tasks in a sequential order. To perform continual learning, the model must possess the abilities to maintain performance on old tasks (stability) and adapt itself to learn new tasks (plasticity). Task-agnostic problem in continual learning is a challenging problem, in which task identities are not available in the inference stage and hence the model must learn to distinguish all the classes in all the tasks. In task-agnostic problem, the model needs to learn two new objectives for learning a new task, including distinguishing new classes from old classes and distinguishing between different new classes. For task-agnostic problem, replay-based methods are commonly used. These methods update the model with both saved old samples and new samples for continual learning. Most existing replay-based methods mix the two objectives in task-agnostic problem together, inhibiting the models from achieving a good trade-off between stability and plasticity. In this paper, we propose a simple yet effective method, called loss decoupling (LODE), for task-agnostic continual learning. LODE separates the two objectives for the new task by decoupling the loss of the new task. As a result, LODE can assign different weights for different objectives, which provides a way to obtain a better trade-off between stability and plasticity than those methods with coupled loss. Experiments show that LODE can outperform existing state-of-the-art replay-based methods on multiple continual learning datasets.

        ----

        ## [492] Federated Learning via Meta-Variational Dropout

        **Authors**: *Insu Jeon, Minui Hong, Junhyeog Yun, Gunhee Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24a8d40f6656e542f3fd43bac678e71b-Abstract-Conference.html)

        **Abstract**:

        Federated Learning (FL) aims to train a global inference model from remotely distributed clients, gaining popularity due to its benefit of improving data privacy. However, traditional FL often faces challenges in practical applications, including model overfitting and divergent local models due to limited and non-IID data among clients. To address these issues, we introduce a novel Bayesian meta-learning approach called meta-variational dropout (MetaVD). MetaVD learns to predict client-dependent dropout rates via a shared hypernetwork, enabling effective model personalization of FL algorithms in limited non-IID data settings. We also emphasize the posterior adaptation view of meta-learning and the posterior aggregation view of Bayesian FL via the conditional dropout posterior. We conducted extensive experiments on various sparse and non-IID FL datasets. MetaVD demonstrated excellent classification accuracy and uncertainty calibration performance, especially for out-of-distribution (OOD) clients. MetaVD compresses the local model parameters needed for each client, mitigating model overfitting and reducing communication costs. Code is available at https://github.com/insujeon/MetaVD.

        ----

        ## [493] Horospherical Decision Boundaries for Large Margin Classification in Hyperbolic Space

        **Authors**: *Xiran Fan, Chun-Hao Yang, Baba C. Vemuri*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24cb8b08f3cb2f59671e33faac4790e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24cb8b08f3cb2f59671e33faac4790e6-Abstract-Conference.html)

        **Abstract**:

        Hyperbolic spaces have been quite popular in the recent past for representing hierarchically organized data. Further, several classification algorithms for data in these spaces have been proposed in the literature. These algorithms mainly use either hyperplanes or geodesics for decision boundaries in a large margin classifiers setting leading to a non-convex optimization problem. In this paper, we propose a novel large margin classifier based on horospherical decision boundaries that leads to a geodesically convex optimization problem that can be optimized using any Riemannian gradient descent technique guaranteeing a globally optimal solution. We present several experiments depicting the competitive performance of our classifier in comparison to SOTA.

        ----

        ## [494] MIM4DD: Mutual Information Maximization for Dataset Distillation

        **Authors**: *Yuzhang Shang, Zhihang Yuan, Yan Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24d36eee157559e0d2549455fba28f6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24d36eee157559e0d2549455fba28f6a-Abstract-Conference.html)

        **Abstract**:

        Dataset distillation (DD) aims to synthesize a small dataset whose test performance is comparable to a full dataset using the same model. State-of-the-art (SoTA) methods optimize synthetic datasets primarily by matching heuristic indicators extracted from two networks: one from real data and one from synthetic data (see Fig.1, Left), such as gradients and training trajectories. DD is essentially a compression problem that emphasizes on maximizing the preservation of information contained in the data. We argue that well-defined metrics which measure the amount of shared information between variables in information theory are necessary for success measurement, but are never considered by previous works. Thus, we introduce mutual information (MI) as the metric to quantify the shared information between the synthetic and the real datasets, and devise MIM4DD numerically maximizing the MI via a newly designed optimizable objective within a contrastive learning framework to update the synthetic dataset. Specifically, we designate the samples in different datasets who share the same labels as positive pairs, and vice versa negative pairs. Then we respectively pull and push those samples in positive and negative pairs into contrastive space via minimizing NCE loss. As a result, the targeted MI can be transformed into a lower bound represented by feature maps of samples, which is numerically feasible. Experiment results show that MIM4DD can be implemented as an add-on module to existing SoTA DD methods.

        ----

        ## [495] Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks

        **Authors**: *Woojin Cho, Kookjin Lee, Donsub Rim, Noseong Park*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html)

        **Abstract**:

        In various engineering and applied science applications, repetitive numerical simulations of partial differential equations (PDEs) for varying input parameters are often required (e.g., aircraft shape optimization over many design parameters) and solvers are required to perform rapid execution. In this study, we suggest a path that potentially opens up a possibility for physics-informed neural networks (PINNs), emerging deep-learning-based solvers, to be considered as one such solver. Although PINNs have pioneered a proper integration of deep-learning and scientific computing, they require repetitive time-consuming training of neural networks, which is not suitable for many-query scenarios. To address this issue, we propose a lightweight low-rank PINNs containing only hundreds of model parameters and an associated hypernetwork-based meta-learning algorithm, which allows efficient approximation of solutions of PDEs for varying ranges of PDE input parameters. Moreover, we show that the proposed method is effective in overcoming a challenging issue, known as "failure modes" of PINNs.

        ----

        ## [496] Solving a Class of Non-Convex Minimax Optimization in Federated Learning

        **Authors**: *Xidong Wu, Jianhui Sun, Zhengmian Hu, Aidong Zhang, Heng Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/254009e8d528f98764a060e877a1b01c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/254009e8d528f98764a060e877a1b01c-Abstract-Conference.html)

        **Abstract**:

        The minimax problems arise throughout machine learning applications, ranging from adversarial training and policy evaluation in reinforcement learning to AUROC maximization. To address the large-scale distributed data challenges across multiple clients with communication-efficient distributed training, federated learning (FL) is gaining popularity. Many optimization algorithms for minimax problems have been developed in the centralized setting (\emph{i.e.}, single-machine). Nonetheless, the algorithm for minimax problems under FL is still underexplored. In this paper, we study a class of federated nonconvex minimax optimization problems. We propose FL algorithms (FedSGDA+ and FedSGDA-M) and reduce existing complexity results for the most common minimax problems. For nonconvex-concave problems, we propose FedSGDA+ and reduce the communication complexity to $O(\varepsilon^{-6})$. Under nonconvex-strongly-concave and nonconvex-PL minimax settings, we prove that FedSGDA-M has the best-known sample complexity of $O(\kappa^{3} N^{-1}\varepsilon^{-3})$ and the best-known communication complexity of $O(\kappa^{2}\varepsilon^{-2})$. FedSGDA-M is the first algorithm to match the best sample complexity $O(\varepsilon^{-3})$ achieved by the single-machine method under the nonconvex-strongly-concave setting. Extensive experimental results on fair classification and AUROC maximization show the efficiency of our algorithms.

        ----

        ## [497] End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes

        **Authors**: *Alexandre Maraval, Matthieu Zimmer, Antoine Grosnit, Haitham Bou-Ammar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html)

        **Abstract**:

        Meta-Bayesian optimisation (meta-BO) aims to improve the sample efficiency of Bayesian optimisation by leveraging data from related tasks. While previous methods successfully meta-learn either a surrogate model or an acquisition function independently, joint training of both components remains an open challenge. This paper proposes the first end-to-end differentiable meta-BO framework that generalises neural processes to learn acquisition functions via transformer architectures. We enable this end-to-end framework with reinforcement learning (RL) to tackle the lack of labelled acquisition data. Early on, we notice that training transformer-based neural processes from scratch with RL is challenging due to insufficient supervision, especially when rewards are sparse. We formalise this claim with a combinatorial analysis showing that the widely used notion of regret as a reward signal exhibits a logarithmic sparsity pattern in trajectory lengths. To tackle this problem, we augment the RL objective with an auxiliary task that guides part of the architecture to learn a valid probabilistic model as an inductive bias. We demonstrate that our method achieves state-of-the-art regret results against various baselines in experiments on standard hyperparameter optimisation tasks and also outperforms others in the real-world problems of mixed-integer programming tuning, antibody design, and logic synthesis for electronic design automation.

        ----

        ## [498] Contextual Bandits and Imitation Learning with Preference-Based Active Queries

        **Authors**: *Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2567c95fd41459a98a73ba893775d22a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2567c95fd41459a98a73ba893775d22a-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of contextual bandits and imitation learning, where the learner lacks direct knowledge of the executed action's reward. Instead, the learner can actively request the expert at each round to compare two actions and receive noisy preference feedback. The learner's objective is two-fold: to minimize regret associated with the executed actions, while simultaneously, minimizing the number of comparison queries made to the expert. In this paper, we assume that the learner has access to a function class that can represent the expert's preference model under appropriate link functions and present an algorithm that leverages an online regression oracle with respect to this function class. For the contextual bandit setting, our algorithm achieves a regret bound that combines the best of both worlds, scaling as $O(\min\\{\sqrt{T}, d/\Delta\\})$, where $T$ represents the number of interactions, $d$ represents the eluder dimension of the function class, and $\Delta$ represents the minimum preference of the optimal action over any suboptimal action under all contexts. Our algorithm does not require the knowledge of $\Delta$, and the obtained regret bound is comparable to what can be achieved in the standard contextual bandits setting where the learner observes reward signals at each round. Additionally, our algorithm makes only $O(\min\\{T, d^2/\Delta^2\\})$ queries to the expert. We then extend our algorithm to the imitation learning setting, where the agent engages with an unknown environment in episodes of length $H$, and provide similar guarantees regarding regret and query complexity. Interestingly, with preference-based feedback, our imitation learning algorithm can learn a policy outperforming a sub-optimal expert, matching the result from interactive imitation learning algorithms [Ross and Bagnell, 2014] that require access to the expert's actions and also reward signals.

        ----

        ## [499] Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding

        **Authors**: *George Ma, Yifei Wang, Yisen Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/257b3a7438b1f3709e91a86adf2fdc0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/257b3a7438b1f3709e91a86adf2fdc0a-Abstract-Conference.html)

        **Abstract**:

        Spectral embedding is a powerful graph embedding technique that has received a lot of attention recently due to its effectiveness on Graph Transformers. However, from a theoretical perspective, the universal expressive power of spectral embedding comes at the price of losing two important invariance properties of graphs, sign and basis invariance, which also limits its effectiveness on graph data. To remedy this issue, many previous methods developed costly approaches to learn new invariants and suffer from high computation complexity. In this work, we explore a minimal approach that resolves the ambiguity issues by directly finding canonical directions for the eigenvectors, named Laplacian Canonization (LC). As a pure pre-processing method, LC is light-weighted and can be applied to any existing GNNs. We provide a thorough investigation, from theory to algorithm, on this approach, and discover an efficient algorithm named Maximal Axis Projection (MAP) that works for both sign and basis invariance and successfully canonizes more than 90\% of all eigenvectors. Experiments on real-world benchmark datasets like ZINC, MOLTOX21, and MOLPCBA show that MAP consistently outperforms existing methods while bringing minimal computation overhead. Code is available at https://github.com/PKU-ML/LaplacianCanonization.

        ----

        ## [500] Localized Symbolic Knowledge Distillation for Visual Commonsense Models

        **Authors**: *Jae Sung Park, Jack Hessel, Khyathi Chandu, Paul Pu Liang, Ximing Lu, Peter West, Youngjae Yu, Qiuyuan Huang, Jianfeng Gao, Ali Farhadi, Yejin Choi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html)

        **Abstract**:

        Instruction following vision-language (VL) models offer a flexibleinterface that supports a broad range of multimodal tasks in a zero-shot fashion.However, interfaces that operate on full images do not directly enable the user toâ€œpoint to" and access specific regions within images. This capability is importantnot only to support reference-grounded VL benchmarks, but also, for practicalapplications that require precise within-image reasoning. We build LocalizedVisual Commonsense model which allows users to specify (multiple) regions-as-input. We train our model by sampling localized commonsense knowledgefrom a large language model (LLM): specifically, we prompt a LLM to collectcommonsense knowledge given a global literal image description and a localliteral region description automatically generated by a set of VL models. Thispipeline is scalable and fully automatic, as no aligned or human-authored imageand text pairs are required. With a separately trained critic model that selectshigh quality examples, we find that training on the localized commonsense corpusexpanded solely from images can successfully distill existing VL models to supporta reference-as-input interface. Empirical results and human evaluations in zero-shotsettings demonstrate that our distillation method results in more precise VL modelsof reasoning compared to a baseline of passing a generated referring expression.

        ----

        ## [501] SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation

        **Authors**: *Mengcheng Lan, Xinjiang Wang, Yiping Ke, Jiaxing Xu, Litong Feng, Wayne Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/25823c8eadef751dbd09a0ab9f463b59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/25823c8eadef751dbd09a0ab9f463b59-Abstract-Conference.html)

        **Abstract**:

        Unsupervised semantic segmentation is a challenging task that segments images into semantic groups without manual annotation. Prior works have primarily focused on leveraging prior knowledge of semantic consistency or priori concepts from self-supervised learning methods, which often overlook the coherence property of image segments. In this paper, we demonstrate that the smoothness prior, asserting that close features in a metric space share the same semantics, can significantly simplify segmentation by casting unsupervised semantic segmentation as an energy minimization problem. Under this paradigm, we propose a novel approach called SmooSeg that harnesses self-supervised learning methods to model the closeness relationships among observations as smoothness signals. To effectively discover coherent semantic segments, we introduce a novel smoothness loss that promotes piecewise smoothness within segments while preserving discontinuities across different segments. Additionally, to further enhance segmentation quality, we design an asymmetric teacher-student style predictor that generates smoothly updated pseudo labels, facilitating an optimal fit between observations and labeling outputs. Thanks to the rich supervision cues of the smoothness prior, our SmooSeg significantly outperforms STEGO in terms of pixel accuracy on three datasets: COCOStuff (+14.9\%), Cityscapes (+13.0\%), and Potsdam-3 (+5.7\%).

        ----

        ## [502] Fast Trainable Projection for Robust Fine-tuning

        **Authors**: *Junjiao Tian, Yen-Cheng Liu, James Seale Smith, Zsolt Kira*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/259e59fe23ebd09252647fed42949182-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/259e59fe23ebd09252647fed42949182-Abstract-Conference.html)

        **Abstract**:

        Robust fine-tuning aims to achieve competitive in-distribution (ID) performance while maintaining the out-of-distribution (OOD) robustness of a pre-trained model when transferring it to a downstream task. Recently, projected gradient descent has been successfully used in robust fine-tuning by constraining the deviation from the initialization of the fine-tuned model explicitly through projection. However, algorithmically, two limitations prevent this method from being adopted more widely, scalability and efficiency.  In this paper, we propose a new projection-based fine-tuning algorithm, Fast Trainable Projection (FTP) for computationally efficient learning of per-layer projection constraints, resulting in an average 35% speedup on our benchmarks compared to prior works. FTP can be combined with existing optimizers such as AdamW, and be used in a plug-and-play fashion. Finally, we show that FTP is a special instance of hyper-optimizers that tune the hyper-parameters of optimizers in a learnable manner through nested differentiation. Empirically, we show superior robustness on OOD datasets, including domain shifts and natural corruptions, across four different vision tasks with five different pre-trained models. Additionally, we demonstrate that FTP is broadly applicable and beneficial to other learning scenarios such as low-label and continual learning settings thanks to its easy adaptability. The code will be available at https://github.com/GT-RIPL/FTP.git.

        ----

        ## [503] Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation

        **Authors**: *Shengpu Tang, Jenna Wiens*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/25b15618c98ff0c4655df0c5a277e1c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/25b15618c98ff0c4655df0c5a277e1c6-Abstract-Conference.html)

        **Abstract**:

        In applying reinforcement learning (RL) to high-stakes domains, quantitative and qualitative evaluation using observational data can help practitioners understand the generalization performance of new policies. However, this type of off-policy evaluation (OPE) is inherently limited since offline data may not reflect the distribution shifts resulting from the application of new policies. On the other hand, online evaluation by collecting rollouts according to the new policy is often infeasible, as deploying new policies in these domains can be unsafe. In this work, we propose a semi-offline evaluation framework as an intermediate step between offline and online evaluation, where human users provide annotations of unobserved counterfactual trajectories. While tempting to simply augment existing data with such annotations, we show that this naive approach can lead to biased results. Instead, we design a new family of OPE estimators based on importance sampling (IS) and a novel weighting scheme that incorporate counterfactual annotations without introducing additional bias. We analyze the theoretical properties of our approach, showing its potential to reduce both bias and variance compared to standard IS estimators. Our analyses reveal important practical considerations for handling biased, noisy, or missing annotations. In a series of proof-of-concept experiments involving bandits and a healthcare-inspired simulator, we demonstrate that our approach outperforms purely offline IS estimators and is robust to imperfect annotations. Our framework, combined with principled human-centered design of annotation solicitation, can enable the application of RL in high-stakes domains.

        ----

        ## [504] Are GATs Out of Balance?

        **Authors**: *Nimrah Mustafa, Aleksandar Bojchevski, Rebekka Burkholz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/25d463c05b414125f598cdf8022b3b46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/25d463c05b414125f598cdf8022b3b46-Abstract-Conference.html)

        **Abstract**:

        While the expressive power and computational capabilities of graph neural networks (GNNs) have been theoretically studied, their optimization and learning dynamics, in general, remain largely unexplored. Our study undertakes the Graph Attention Network (GAT), a popular GNN architecture in which a node's neighborhood aggregation is weighted by parameterized attention coefficients. We derive a conservation law of GAT gradient flow dynamics, which explains why a high portion of parameters in GATs with standard initialization struggle to change during training. This effect is amplified in deeper GATs, which perform significantly worse than their shallow counterparts. To alleviate this problem, we devise an initialization scheme that balances the GAT network. Our approach i) allows more effective propagation of gradients and in turn enables trainability of deeper networks, and ii) attains a considerable speedup in training and convergence time in comparison to the standard initialization. Our main theorem serves as a stepping stone to studying the learning dynamics of positive homogeneous models with attention mechanisms.

        ----

        ## [505] SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation

        **Authors**: *Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Wang Yanjun, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang, Ziwei Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2614947a25d7c435bcd56c51958ddcb1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2614947a25d7c435bcd56c51958ddcb1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Expressive human pose and shape estimation (EHPS) unifies body, hands, and face motion capture with numerous applications. Despite encouraging progress, current state-of-the-art methods still depend largely on a confined set of training datasets. In this work, we investigate scaling up EHPS towards the first generalist foundation model (dubbed SMPLer-X), with up to ViT-Huge as the backbone and training with up to 4.5M instances from diverse data sources. With big data and the large model, SMPLer-X exhibits strong performance across diverse test benchmarks and excellent transferability to even unseen environments. 1) For the data scaling, we perform a systematic investigation on 32 EHPS datasets, including a wide range of scenarios that a model trained on any single dataset cannot handle. More importantly, capitalizing on insights obtained from the extensive benchmarking process, we optimize our training scheme and select datasets that lead to a significant leap in EHPS capabilities. 2) For the model scaling, we take advantage of vision transformers to study the scaling law of model sizes in EHPS. Moreover, our finetuning strategy turn SMPLer-X into specialist models, allowing them to achieve further performance boosts. Notably, our foundation model SMPLer-X consistently delivers state-of-the-art results on seven benchmarks such as AGORA (107.2 mm NMVE), UBody (57.4 mm PVE), EgoBody (63.6 mm PVE), and EHF (62.3 mm PVE without finetuning).

        ----

        ## [506] Fast Asymptotically Optimal Algorithms for Non-Parametric Stochastic Bandits

        **Authors**: *Dorian Baudry, Fabien Pesquerel, Rémy Degenne, Odalric-Ambrym Maillard*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/26300457961c3e056ea61c9d3ebec2a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/26300457961c3e056ea61c9d3ebec2a4-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of regret minimization in non-parametric stochastic bandits. When the rewards are known to be bounded from above, there exists asymptotically optimal algorithms, with asymptotic regret depending on an infimum of Kullback-Leibler divergences (KL). These algorithms are computationally expensive and require storing all past rewards, thus simpler but non-optimal algorithms are often used instead. We introduce several methods to approximate the infimum KL which reduce drastically the computational and memory costs of existing optimal algorithms, while keeping their regret guaranties. We apply our findings to design new variants of the MED and IMED algorithms, and demonstrate their interest with extensive numerical simulations.

        ----

        ## [507] SHAP-IQ: Unified Approximation of any-order Shapley Interactions

        **Authors**: *Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, Barbara Hammer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/264f2e10479c9370972847e96107db7f-Abstract-Conference.html)

        **Abstract**:

        Predominately in explainable artificial intelligence (XAI) research, the Shapley value (SV) is applied to determine feature attributions for any black box model. Shapley interaction indices extend the SV to define any-order feature interactions. Defining a unique Shapley interaction index is an open research question and, so far, three definitions have been proposed, which differ by their choice of axioms. Moreover, each definition requires a specific approximation technique. Here, we propose SHAPley Interaction Quantification (SHAP-IQ), an efficient sampling-based approximator to compute Shapley interactions for arbitrary cardinal interaction indices (CII), i.e. interaction indices that satisfy the linearity, symmetry and dummy axiom. SHAP-IQ is based on a novel representation and, in contrast to existing methods, we provide theoretical guarantees for its approximation quality, as well as estimates for the variance of the point estimates. For the special case of SV, our approach reveals a novel representation of the SV and corresponds to Unbiased KernelSHAP with a greatly simplified calculation. We illustrate the computational efficiency and effectiveness by explaining language, image classification and high-dimensional synthetic models.

        ----

        ## [508] Towards Last-layer Retraining for Group Robustness with Fewer Annotations

        **Authors**: *Tyler LaBonte, Vidya Muthukumar, Abhishek Kumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/265bee74aee86df77e8e36d25e786ab5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/265bee74aee86df77e8e36d25e786ab5-Abstract-Conference.html)

        **Abstract**:

        Empirical risk minimization (ERM) of neural networks is prone to over-reliance on spurious correlations and poor generalization on minority groups. The recent deep feature reweighting (DFR) technique achieves state-of-the-art group robustness via simple last-layer retraining, but it requires held-out group and class annotations to construct a group-balanced reweighting dataset. In this work, we examine this impractical requirement and find that last-layer retraining can be surprisingly effective with no group annotations (other than for model selection) and only a handful of class annotations. We first show that last-layer retraining can greatly improve worst-group accuracy even when the reweighting dataset has only a small proportion of worst-group data. This implies a "free lunch" where holding out a subset of training data to retrain the last layer can substantially outperform ERM on the entire dataset with no additional data, annotations, or computation for training. To further improve group robustness, we introduce a lightweight method called selective last-layer finetuning (SELF), which constructs the reweighting dataset using misclassifications or disagreements. Our experiments present the first evidence that model disagreement upsamples worst-group data, enabling SELF to nearly match DFR on four well-established benchmarks across vision and language tasks with no group annotations and less than 3% of the held-out class annotations.

        ----

        ## [509] Analysis of Variance of Multiple Causal Networks

        **Authors**: *Zhongli Jiang, Dabao Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/26c233f48fb05bbd52a520e4bb9e3760-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/26c233f48fb05bbd52a520e4bb9e3760-Abstract-Conference.html)

        **Abstract**:

        Constructing a directed cyclic graph (DCG) is challenged by both algorithmic difficulty and computational burden. Comparing multiple DCGs is even more difficult, compounded by the need to identify dynamic causalities across graphs. We propose to unify multiple DCGs with a single structural model and develop a limited-information-based method to simultaneously construct multiple networks and infer their disparities, which can be visualized by appropriate correspondence analysis. The algorithm provides DCGs with robust non-asymptotic theoretical properties. It is designed with two sequential stages, each of which involves parallel computation tasks that are scalable to the network complexity. Taking advantage of high-performance clusters, our method makes it possible to evaluate the statistical significance of DCGs using the bootstrap method. We demonstrated the effectiveness of our method by applying it to synthetic and real datasets.

        ----

        ## [510] Revisiting the Minimalist Approach to Offline Reinforcement Learning

        **Authors**: *Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, Sergey Kolesnikov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/26cce1e512793f2072fd27c391e04652-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/26cce1e512793f2072fd27c391e04652-Abstract-Conference.html)

        **Abstract**:

        Recent years have witnessed significant advancements in offline reinforcement learning (RL), resulting in the development of numerous algorithms with varying degrees of complexity. While these algorithms have led to noteworthy improvements, many incorporate seemingly minor design choices that impact their effectiveness beyond core algorithmic advances. However, the effect of these design choices on established baselines remains understudied. In this work, we aim to bridge this gap by conducting a retrospective analysis of recent works in offline RL and propose ReBRAC, a minimalistic algorithm that integrates such design elements built on top of the TD3+BC method. We evaluate ReBRAC on 51 datasets with both proprioceptive and visual state spaces using D4RL and V-D4RL benchmarks, demonstrating its state-of-the-art performance among ensemble-free methods in both offline and offline-to-online settings. To further illustrate the efficacy of these design choices, we perform a large-scale ablation study and hyperparameter sensitivity analysis on the scale of thousands of experiments.

        ----

        ## [511] Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift

        **Authors**: *Saurabh Garg, Amrith Setlur, Zachary C. Lipton, Sivaraman Balakrishnan, Virginia Smith, Aditi Raghunathan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/26f96550613971371c5d07f37f0e06c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/26f96550613971371c5d07f37f0e06c0-Abstract-Conference.html)

        **Abstract**:

        Self-training and contrastive learning have emerged as leading techniques for incorporating unlabeled data, both under distribution shift (unsupervised domain adaptation) and when it is absent (semi-supervised learning). However, despite the popularity and compatibility of these techniques, their efficacy in combination remains surprisingly unexplored. In this paper, we first undertake a systematic empirical investigation of this combination, finding (i) that in domain adaptation settings, self-training and contrastive learning offer significant complementary gains; and (ii) that in semi-supervised learning settings, surprisingly, the benefits are not synergistic. Across eight distribution shift datasets (e.g., BREEDs, WILDS), we demonstrate that the combined method obtains 3--8\% higher accuracy than either approach independently. Finally, we theoretically analyze these techniques in a simplified model of distribution shift demonstrating scenarios under which the features produced by contrastive learning can yield a good initialization for self-training to further amplify gains and achieve optimal performance, even when either method alone would fail.

        ----

        ## [512] Low Tensor Rank Learning of Neural Dynamics

        **Authors**: *Arthur Pellegrino, N. Alex Cayco-Gajic, Angus Chadwick*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27030ad2ec1d8f2c3847a64e382c30ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27030ad2ec1d8f2c3847a64e382c30ca-Abstract-Conference.html)

        **Abstract**:

        Learning relies on coordinated synaptic changes in recurrently connected populations of neurons. Therefore, understanding the collective evolution of synaptic connectivity over learning is a key challenge in neuroscience and machine learning. In particular, recent work has shown that the weight matrices of task-trained RNNs are typically low rank, but how this low rank structure unfolds over learning is unknown. To address this, we investigate the rank of the 3-tensor formed by the weight matrices throughout learning. By fitting RNNs of varying rank to large-scale neural recordings during a motor learning task, we find that the inferred weights are low-tensor-rank and therefore evolve over a fixed low-dimensional subspace throughout the entire course of learning. We next validate the observation of low-tensor-rank learning on an RNN trained to solve the same task. Finally, we present a set of mathematical results bounding the matrix and tensor ranks of gradient descent learning dynamics which show that low-tensor-rank weights emerge naturally in RNNs trained to solve low-dimensional tasks. Taken together, our findings provide insight on the evolution of population connectivity over learning in both biological and artificial neural networks, and enable reverse engineering of learning-induced changes in recurrent dynamics from large-scale neural recordings.

        ----

        ## [513] MonoUNI: A Unified Vehicle and Infrastructure-side Monocular 3D Object Detection Network with Sufficient Depth Clues

        **Authors**: *Jinrang Jia, Zhenjia Li, Yifeng Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2703a0e3c2b33506295a77762338cf24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2703a0e3c2b33506295a77762338cf24-Abstract-Conference.html)

        **Abstract**:

        Monocular 3D detection of vehicle and infrastructure sides are two important topics in autonomous driving. Due to diverse sensor installations and focal lengths, researchers are faced with the challenge of constructing algorithms for the two topics based on different prior knowledge. In this paper, by taking into account the diversity of pitch angles and focal lengths, we propose a unified optimization target named normalized depth, which realizes the unification of 3D detection problems for the two sides. Furthermore, to enhance the accuracy of monocular 3D detection, 3D normalized cube depth of obstacle is developed to promote the learning of depth information. We posit that the richness of depth clues is a pivotal factor impacting the detection performance on both the vehicle and infrastructure sides. A richer set of depth clues facilitates the model to learn better spatial knowledge, and the 3D normalized cube depth offers sufficient depth clues. Extensive experiments demonstrate the effectiveness of our approach. Without introducing any extra information, our method, named MonoUNI, achieves state-of-the-art performance on five widely used monocular 3D detection benchmarks, including Rope3D and DAIR-V2X-I for the infrastructure side, KITTI and Waymo for the vehicle side, and nuScenes for the cross-dataset evaluation.

        ----

        ## [514] Active Reasoning in an Open-World Environment

        **Authors**: *Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2712b17bb58ea5b2b65c45857b024744-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2712b17bb58ea5b2b65c45857b024744-Abstract-Conference.html)

        **Abstract**:

        Recent advances in vision-language learning have achieved notable success on complete-information question-answering datasets through the integration of extensive world knowledge. Yet, most models operate passively, responding to questions based on pre-stored knowledge. In stark contrast, humans possess the ability to actively explore, accumulate, and reason using both newfound and existing information to tackle incomplete-information questions. In response to this gap, we introduce Conan, an interactive open-world environment devised for the assessment of active reasoning. Conan facilitates active exploration and promotes multi-round abductive inference, reminiscent of rich, open-world settings like Minecraft. Diverging from previous works that lean primarily on single-round deduction via instruction following, Conan compels agents to actively interact with their surroundings, amalgamating new evidence with prior knowledge to elucidate events from incomplete observations. Our analysis on \bench underscores the shortcomings of contemporary state-of-the-art models in active exploration and understanding complex scenarios. Additionally, we explore Abduction from Deduction, where agents harness Bayesian rules to recast the challenge of abduction as a deductive process. Through Conan, we aim to galvanize advancements in active reasoning and set the stage for the next generation of artificial intelligence agents adept at dynamically engaging in environments.

        ----

        ## [515] 2Direction: Theoretically Faster Distributed Training with Bidirectional Communication Compression

        **Authors**: *Alexander Tyurin, Peter Richtárik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2717ad172c5495837582d70a8519abfb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2717ad172c5495837582d70a8519abfb-Abstract-Conference.html)

        **Abstract**:

        We consider distributed convex optimization problems in the regime when the communication between the server and the workers is expensive in both uplink and downlink directions. We develop a new and provably accelerated method, which we call 2Direction, based on fast bidirectional compressed communication and a new bespoke error-feedback mechanism which may be of independent interest. Indeed, we find that the EF and EF21-P mechanisms (Seide et al., 2014; Gruntkowska et al., 2023) that have considerable success in the design of efficient non-accelerated methods are not appropriate for accelerated methods. In particular, we prove that 2Direction improves the previous state-of-the-art communication complexity $\widetilde{\Theta}\left(K \times \left(\frac{L}{\alpha \mu} + \frac{L_{\max} \omega}{n \mu} + \omega\right)\right)$ (Gruntkowska et al., 2023) to $\widetilde{\Theta}(K \times (\sqrt{\frac{L (\omega + 1)}{\alpha \mu}} + \sqrt{\frac{L_{\max} \omega^2}{n \mu}} + \frac{1}{\alpha} + \omega))$ in the $\mu$--strongly-convex setting, where $L$ and $L_{\max}$ are smoothness constants, $n$ is \# of workers, $\omega$ and $\alpha$ are compression errors of the Rand$K$ and Top$K$ sparsifiers (as examples), $K$ is \# of coordinates/bits that the server and workers send to each other. Moreover, our method is the first that improves upon the communication complexity of the vanilla accelerated gradient descent method (AGD). We obtain similar improvements in the general convex regime as well. Finally, our theoretical findings are corroborated by experimental evidence.

        ----

        ## [516] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

        **Authors**: *Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, Karthik Narasimhan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html)

        **Abstract**:

        Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.Our experiments show that ToT significantly enhances language modelsâ€™ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4\% of tasks, our method achieved a success rate of 74\%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.

        ----

        ## [517] What functions can Graph Neural Networks compute on random graphs? The role of Positional Encoding

        **Authors**: *Nicolas Keriven, Samuel Vaiter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/271ec4d1a9ff5e6b81a6e21d38b1ba96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/271ec4d1a9ff5e6b81a6e21d38b1ba96-Abstract-Conference.html)

        **Abstract**:

        We aim to deepen the theoretical understanding of Graph Neural Networks (GNNs) on large graphs, with a focus on their expressive power.Existing analyses relate this notion to the graph isomorphism problem, which is mostly relevant for graphs of small sizes, or studied graph classification or regression tasks, while prediction tasks on \emph{nodes} are far more relevant on large graphs. Recently, several works showed that, on very general random graphs models, GNNs converge to certains functions as the number of nodes grows.In this paper, we provide a more complete and intuitive description of the function space generated by equivariant GNNs for node-tasks, through general notions of convergence that encompass several previous examples. We emphasize the role of input node features, and study the impact of \emph{node Positional Encodings} (PEs), a recent line of work that has been shown to yield state-of-the-art results in practice. Through the study of several examples of PEs on large random graphs, we extend previously known universality results to significantly more general models. Our theoretical results hint at some normalization tricks, which is shown numerically to have a positive impact on GNN generalization on synthetic and real data. Our proofs contain new concentration inequalities of independent interest.

        ----

        ## [518] High-dimensional Asymptotics of Denoising Autoencoders

        **Authors**: *Hugo Cui, Lenka Zdeborová*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2722a0ccf6acfe3d144fdbb0dedd80b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2722a0ccf6acfe3d144fdbb0dedd80b5-Abstract-Conference.html)

        **Abstract**:

        We address the problem of denoising data from a Gaussian mixture using a two-layer non-linear autoencoder with tied weights and a skip connection. We consider the high-dimensional limit where the number of training samples and the input dimension jointly tend to infinity while the number of hidden units remains bounded. We provide closed-form expressions for the denoising mean-squared test error. Building on this result, we quantitatively characterize the advantage of the considered architecture over the autoencoder without the skip connection that relates closely to principal component analysis. We further show that our results capture accurately the learning curves on a range of real datasets.

        ----

        ## [519] Learning to Reason and Memorize with Self-Notes

        **Authors**: *Jack Lanchantin, Shubham Toshniwal, Jason Weston, Arthur Szlam, Sainbayar Sukhbaatar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/274d0146144643ee2459a602123c60ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/274d0146144643ee2459a602123c60ff-Abstract-Conference.html)

        **Abstract**:

        Large language models have been shown to struggle with multi-step reasoning, and do not retain previous reasoning steps for future use. We propose a simple method for solving both of these problems by allowing the model to take Self-Notes. Unlike recent chain-of-thought or scratchpad approaches, the model can deviate from the input context at any time to explicitly think and write down its thoughts. This allows the model to perform reasoning on the fly as it reads the context and even integrate previous reasoning steps, thus enhancing its memory with useful information and enabling multi-step reasoning. Experiments across a wide variety of tasks demonstrate that our method can outperform chain-of-thought and scratchpad methods by taking Self-Notes that interleave the input text.

        ----

        ## [520] What can a Single Attention Layer Learn? A Study Through the Random Features Lens

        **Authors**: *Hengyu Fu, Tianyu Guo, Yu Bai, Song Mei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/274db6bf1b01d8b4f07feaeb8c46f474-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/274db6bf1b01d8b4f07feaeb8c46f474-Abstract-Conference.html)

        **Abstract**:

        Attention layers---which map a sequence of inputs to a sequence of outputs---are core building blocks of the Transformer architecture which has achieved significant breakthroughs in modern artificial intelligence. This paper presents a rigorous theoretical study on the learning and generalization of a single multi-head attention layer, with a sequence of key vectors and a separate query vector as input. We consider the random feature setting where the attention layer has a large number of heads, with randomly sampled frozen query and key matrices, and trainable value matrices. We show that such a random-feature attention layer can express a broad class of target functions that are permutation invariant to the key vectors.  We further provide quantitative excess risk bounds for learning these target functions from finite samples, using random feature attention with finitely many heads.Our results feature several implications unique to the attention structure compared with existing random features theory for neural networks, such as (1) Advantages in the sample complexity over standard two-layer random-feature networks;  (2) Concrete and natural classes of functions that can be learned efficiently by a random-feature attention layer; and (3) The effect of the sampling distribution of the query-key weight matrix (the product of the query and key matrix), where Gaussian random weights with a non-zero mean result in better sample complexities over the zero-mean counterpart for learning certain natural target functions. Experiments on simulated data corroborate our theoretical findings and further illustrate the interplay between the sample size and the complexity of the target function.

        ----

        ## [521] Let the Flows Tell: Solving Graph Combinatorial Problems with GFlowNets

        **Authors**: *Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron C. Courville, Yoshua Bengio, Ling Pan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract-Conference.html)

        **Abstract**:

        Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space.On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates.In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment.Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quality solutions.Our implementation is open-sourced at https://github.com/zdhNarsil/GFlowNet-CombOpt.

        ----

        ## [522] Debiasing Scores and Prompts of 2D Diffusion for View-consistent Text-to-3D Generation

        **Authors**: *Susung Hong, Donghoon Ahn, Seungryong Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27725882a88f202e07319abbb3be7693-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27725882a88f202e07319abbb3be7693-Abstract-Conference.html)

        **Abstract**:

        Existing score-distilling text-to-3D generation techniques, despite their considerable promise, often encounter the view inconsistency problem. One of the most notable issues is the Janus problem, where the most canonical view of an object (\textit{e.g}., face or head) appears in other views. In this work, we explore existing frameworks for score-distilling text-to-3D generation and identify the main causes of the view inconsistency problem---the embedded bias of 2D diffusion models. Based on these findings, we propose two approaches to debias the score-distillation frameworks for view-consistent text-to-3D generation. Our first approach, called score debiasing, involves cutting off the score estimated by 2D diffusion models and gradually increasing the truncation value throughout the optimization process. Our second approach, called prompt debiasing, identifies conflicting words between user prompts and view prompts using a language model, and adjusts the discrepancy between view prompts and the viewing direction of an object. Our experimental results show that our methods improve the realism of the generated 3D objects by significantly reducing artifacts and achieve a good trade-off between faithfulness to the 2D diffusion models and 3D consistency with little overhead. Our project page is available at~\url{https://susunghong.github.io/Debiased-Score-Distillation-Sampling/}.

        ----

        ## [523] On the Learnability of Multilabel Ranking

        **Authors**: *Vinod Raman, Unique Subedi, Ambuj Tewari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2786baf8091ee8ecb060580239967ba0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2786baf8091ee8ecb060580239967ba0-Abstract-Conference.html)

        **Abstract**:

        Multilabel ranking is a central task in machine learning. However, the most fundamental question of learnability in a multilabel ranking setting with relevance-score feedback remains unanswered. In this work, we characterize the learnability of multilabel ranking problems in both batch and online settings for a large family of ranking losses. Along the way, we give two equivalence classes of ranking losses based on learnability that capture most losses used in practice.

        ----

        ## [524] MAG-GNN: Reinforcement Learning Boosted Graph Neural Network

        **Authors**: *Lecheng Kong, Jiarui Feng, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2788b4cdf421e03650868cc4184bfed8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2788b4cdf421e03650868cc4184bfed8-Abstract-Conference.html)

        **Abstract**:

        While Graph Neural Networks (GNNs) recently became powerful tools in graph learning tasks, considerable efforts have been spent on improving GNNs' structural encoding ability. A particular line of work proposed subgraph GNNs that use subgraph information to improve GNNs' expressivity and achieved great success. However, such effectivity sacrifices the efficiency of GNNs by enumerating all possible subgraphs. In this paper, we analyze the necessity of complete subgraph enumeration and show that a model can achieve a comparable level of expressivity by considering a small subset of the subgraphs. We then formulate the identification of the optimal subset as a combinatorial optimization problem and propose Magnetic Graph Neural Network (MAG-GNN), a reinforcement learning (RL) boosted GNN, to solve the problem. Starting with a candidate subgraph set, MAG-GNN employs an RL agent to iteratively update the subgraphs to locate the most expressive set for prediction. This reduces the exponential complexity of subgraph enumeration to the constant complexity of a subgraph search algorithm while keeping good expressivity. We conduct extensive experiments on many datasets, showing that MAG-GNN achieves competitive performance to state-of-the-art methods and even outperforms many subgraph GNNs. We also demonstrate that MAG-GNN effectively reduces the running time of subgraph GNNs.

        ----

        ## [525] RanPAC: Random Projections and Pre-trained Models for Continual Learning

        **Authors**: *Mark D. McDonnell, Dong Gong, Amin Parvaneh, Ehsan Abbasnejad, Anton van den Hengel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html)

        **Abstract**:

        Continual learning (CL) aims to incrementally learn different tasks (such as classification) in a non-stationary data stream without forgetting old ones. Most CL works focus on tackling catastrophic forgetting under a learning-from-scratch paradigm. However, with the increasing prominence of foundation models, pre-trained models equipped with informative representations have become available for various downstream requirements. Several CL methods based on pre-trained models have been explored, either utilizing pre-extracted features directly (which makes bridging distribution gaps challenging) or incorporating adaptors (which may be subject to forgetting). In this paper, we propose a concise and effective approach for CL with pre-trained models. Given that forgetting occurs during parameter updating, we contemplate an alternative approach that exploits training-free random projectors and class-prototype accumulation, which thus bypasses the issue. Specifically, we inject a frozen Random Projection layer with nonlinear activation between the pre-trained model's feature representations and output head, which captures interactions between features with expanded dimensionality, providing enhanced linear separability for class-prototype-based CL. We also demonstrate the importance of decorrelating the class-prototypes to reduce the distribution disparity when using pre-trained representations. These techniques prove to be effective and circumvent the problem of forgetting for both class- and domain-incremental continual learning. Compared to previous methods applied to pre-trained ViT-B/16 models, we reduce final error rates by between 20% and 62% on seven class-incremental benchmark datasets, despite not using any rehearsal memory. We conclude that the full potential of pre-trained models for simple, effective, and fast continual learning has not hitherto been fully tapped. Code is available at https://github.com/RanPAC/RanPAC.

        ----

        ## [526] FGPrompt: Fine-grained Goal Prompting for Image-goal Navigation

        **Authors**: *Xinyu Sun, Peihao Chen, Jugang Fan, Jian Chen, Thomas H. Li, Mingkui Tan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27c4e15d9af120d7fef04432c7db577f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27c4e15d9af120d7fef04432c7db577f-Abstract-Conference.html)

        **Abstract**:

        Learning to navigate to an image-specified goal is an important but challenging task for autonomous systems like household robots. The agent is required to well understand and reason the location of the navigation goal from a picture shot in the goal position. Existing methods try to solve this problem by learning a navigation policy, which captures semantic features of the goal image and observation image independently and lastly fuses them for predicting a sequence of navigation actions. However, these methods suffer from two major limitations. 1) They may miss detailed information in the goal image, and thus fail to reason the goal location. 2) More critically, it is hard to focus on the goal-relevant regions in the observation image, because they attempt to understand observation without goal conditioning. In this paper, we aim to overcome these limitations by designing a Fine-grained Goal Prompting (\sexyname) method for image-goal navigation. In particular, we leverage fine-grained and high-resolution feature maps in the goal image as prompts to perform conditioned embedding, which preserves detailed information in the goal image and guides the observation encoder to pay attention to goal-relevant regions. Compared with existing methods on the image-goal navigation benchmark, our method brings significant performance improvement on 3 benchmark datasets (\textit{i.e.,} Gibson, MP3D, and HM3D). Especially on Gibson, we surpass the state-of-the-art success rate by 8\% with only 1/50 model size.

        ----

        ## [527] Inner-Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction

        **Authors**: *Yukun Qiu, Guo-Hao Xu, Wei-Shi Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27c852e9d6c76890ca633f111c556a4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27c852e9d6c76890ca633f111c556a4f-Abstract-Conference.html)

        **Abstract**:

        Monocular 3D scene reconstruction aims to reconstruct the 3D structure of scenes based on posed images. Recent volumetric-based methods directly predict the truncated signed distance function (TSDF) volume and have achieved promising results. The memory cost of volumetric-based methods will grow cubically as the volume size increases, so a coarse-to-fine strategy is necessary for saving memory. Specifically, the coarse-to-fine strategy distinguishes surface voxels from non-surface voxels, and only potential surface voxels are considered in the succeeding procedure. However, the non-surface voxels have various features, and in particular, the voxels on the inner side of the surface are quite different from those on the outer side since there exists an intrinsic gap between them. Therefore, grouping inner-surface and outer-surface voxels into the same class will force the classifier to spend its capacity to bridge the gap. By contrast, it is relatively easy for the classifier to distinguish inner-surface and outer-surface voxels due to the intrinsic gap. Inspired by this, we propose the inner-outer aware reconstruction (IOAR) model. IOAR explores a new coarse-to-fine strategy to classify outer-surface, inner-surface and surface voxels. In addition, IOAR separates occupancy branches from TSDF branches to avoid mutual interference between them. Since our model can better classify the surface, outer-surface and inner-surface voxels, it can predict more precise meshes than existing methods. Experiment results on ScanNet, ICL-NUIM and TUM-RGBD datasets demonstrate the effectiveness and generalization of our model. The code is available at https://github.com/YorkQiu/InnerOuterAwareReconstruction.

        ----

        ## [528] Sheaf Hypergraph Networks

        **Authors**: *Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, Pietro Lió*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27f243af2887d7f248f518d9b967a882-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27f243af2887d7f248f518d9b967a882-Abstract-Conference.html)

        **Abstract**:

        Higher-order relations are widespread in nature, with numerous phenomena involving complex interactions that extend beyond simple pairwise connections. As a result, advancements in higher-order processing can accelerate the growth of various fields requiring structured data. Current approaches typically represent these interactions using hypergraphs.We enhance this representation by introducing cellular sheaves for hypergraphs, a mathematical construction that adds extra structure to the conventional hypergraph while maintaining their local, higher-order connectivity. Drawing inspiration from existing Laplacians in the literature, we develop two unique formulations of sheaf hypergraph Laplacians: linear and non-linear. Our theoretical analysis demonstrates that incorporating sheaves into the hypergraph Laplacian provides a more expressive inductive bias than standard hypergraph diffusion, creating a powerful instrument for effectively modelling complex data structures.We employ these sheaf hypergraph Laplacians to design two categories of models: Sheaf Hypergraph Neural Networks and Sheaf Hypergraph Convolutional Networks. These models generalize classical Hypergraph Networks often found in the literature. Through extensive experimentation, we show that this generalization significantly improves performance, achieving top results on multiple benchmark datasets for hypergraph node classification.

        ----

        ## [529] f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences

        **Authors**: *Siddhant Agarwal, Ishan Durugkar, Peter Stone, Amy Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/27f4d95417bb722201597bf4d67cbacc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/27f4d95417bb722201597bf4d67cbacc-Abstract-Conference.html)

        **Abstract**:

        Goal-Conditioned Reinforcement Learning (RL) problems often have access to sparse rewards where the agent receives a reward signal only when it has achieved the goal, making policy optimization a difficult problem.  Several works augment this sparse reward with a learned dense reward function, but this can lead to sub-optimal policies if the reward is misaligned.   Moreover, recent works have demonstrated that effective shaping rewards for a particular problem can depend on the underlying learning algorithm.   This paper introduces a novel way to encourage exploration called  $f$-Policy Gradients, or $f$-PG. $f$-PG minimizes the f-divergence between the agent's state visitation distribution and the goal, which we show can lead to an optimal policy. We derive gradients for various f-divergences to optimize this objective. Our learning paradigm provides dense learning signals for exploration in sparse reward settings. We further introduce an entropy-regularized policy optimization objective, that we call $state$-MaxEnt RL (or $s$-MaxEnt RL) as a special case of our objective. We show that several metric-based shaping rewards like L2 can be used with $s$-MaxEnt RL, providing a common ground to study such metric-based shaping rewards with efficient exploration. We find that $f$-PG has better performance compared to standard policy gradient methods on a challenging gridworld as well as the Point Maze and FetchReach environments. More information on our website https://agarwalsiddhant10.github.io/projects/fpg.html.

        ----

        ## [530] Counterfactually Fair Representation

        **Authors**: *Zhiqun Zuo, Mahdi Khalili, Xueru Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2828ee0c871f78a98ed2a198a166a439-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2828ee0c871f78a98ed2a198a166a439-Abstract-Conference.html)

        **Abstract**:

        The use of machine learning models in high-stake applications (e.g., healthcare, lending, college admission) has raised growing concerns due to potential biases against protected social groups. Various fairness notions and methods have been proposed to mitigate such biases. In this work, we focus on Counterfactual Fairness (CF), a fairness notion that is dependent on an underlying causal graph and first proposed by Kusner $\textit{et al.}$; it requires that the outcome an individual perceives is the same in the real world as it would be in a "counterfactual" world, in which the individual belongs to another social group. Learning fair models satisfying CF can be challenging. It was shown in (Kusner $\textit{et al.}$) that a sufficient condition for satisfying CF is to $\textbf{not}$ use features that are descendants of sensitive attributes in the causal graph. This implies a simple method that learns CF models only using non-descendants of sensitive attributes while eliminating all descendants. Although several subsequent works proposed methods that use all features for training CF models, there is no theoretical guarantee that they can satisfy CF. In contrast, this work proposes a new algorithm that trains models using all the available features. We theoretically and empirically show that models trained with this method can satisfy CF.

        ----

        ## [531] Truncating Trajectories in Monte Carlo Policy Evaluation: an Adaptive Approach

        **Authors**: *Riccardo Poiani, Nicole Nobili, Alberto Maria Metelli, Marcello Restelli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html)

        **Abstract**:

        Policy evaluation via Monte Carlo (MC) simulation is at the core of many MC Reinforcement Learning (RL) algorithms (e.g., policy gradient methods). In this context, the designer of the learning system specifies an interaction budget that the agent usually spends by collecting trajectories of fixed length within a simulator. However, is this data collection strategy the best option? To answer this question, in this paper, we consider as quality index the variance of an unbiased policy return estimator that uses trajectories of different lengths, i.e., truncated. We first derive a closed-form expression of this variance that clearly shows the sub-optimality of the fixed-length trajectory schedule. Furthermore, it suggests that adaptive data collection strategies that spend the available budget sequentially might be able to allocate a larger portion of transitions in timesteps in which more accurate sampling is required to reduce the variance of the final estimate. Building on these findings, we present an adaptive algorithm called Robust and Iterative Data collection strategy Optimization (RIDO). The main intuition behind RIDO is to split the available interaction budget into mini-batches. At each round, the agent determines the most convenient schedule of trajectories that minimizes an empirical and robust estimate of the estimator's variance. After discussing the theoretical properties of our method, we conclude by assessing its performance across multiple domains. Our results show that RIDO can adapt its trajectory schedule toward timesteps where more sampling is required to increase the quality of the final estimation.

        ----

        ## [532] DP-Mix: Mixup-based Data Augmentation for Differentially Private Learning

        **Authors**: *Wenxuan Bao, Francesco Pittaluga, Vijay Kumar B. G, Vincent Bindschaedler*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28484cee66f27fa070796b631cc5242d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28484cee66f27fa070796b631cc5242d-Abstract-Conference.html)

        **Abstract**:

        Data augmentation techniques, such as image transformations and combinations, are highly effective at improving the generalization of computer vision models, especially when training data is limited. However, such techniques are fundamentally incompatible with differentially private learning approaches, due to the latter’s built-in assumption that each training image’s contribution to the learned model is bounded. In this paper, we investigate why naive applications of multi-sample data augmentation techniques, such as mixup, fail to achieve good performance and propose two novel data augmentation techniques specifically designed for the constraints of differentially private learning. Our first technique, DP-MixSelf, achieves SoTA classification performance across a range of datasets and settings by performing mixup on self-augmented data. Our second technique, DP-MixDiff, further improves performance by incorporating synthetic data from a pre-trained diffusion model into the mixup process. We open-source the code at https://github.com/wenxuan-Bao/DP-Mix.

        ----

        ## [533] Point Cloud Completion with Pretrained Text-to-Image Diffusion Models

        **Authors**: *Yoni Kasten, Ohad Rahamim, Gal Chechik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/284afdc2309f9667d2d4fb9290235b0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/284afdc2309f9667d2d4fb9290235b0c-Abstract-Conference.html)

        **Abstract**:

        Point cloud data collected in real-world applications are often incomplete. This is because they are observed from partial viewpoints, which capture only a specific perspective or angle, or due to occlusion and low resolution. Existing completion approaches rely on datasets of specific predefined objects to guide the completion of incomplete, and possibly noisy, point clouds. However, these approaches perform poorly with Out-Of-Distribution (OOD) objects, which are either absent from the dataset or poorly represented. In recent years, the field of text-guided image generation has made significant progress, leading to major breakthroughs in text guided shape generation. We describe an approach called SDS-Complete that uses a pre-trained text-to-image diffusion model and leverages the text semantic of a given incomplete point cloud of an object, to obtain a complete surface representation. SDS-Complete can complete a variety of objects at test time optimization without the need for an expensive collection of 3D information. We evaluate SDS-Complete on incomplete scanned objects, captured by real-world depth sensors and LiDAR scanners, and demonstrate that is effective in handling objects which are typically absent from common datasets.

        ----

        ## [534] Eliciting User Preferences for Personalized Multi-Objective Decision Making through Comparative Feedback

        **Authors**: *Han Shao, Lee Cohen, Avrim Blum, Yishay Mansour, Aadirupa Saha, Matthew R. Walter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/286e7ab0ce6a68282394c92361c27b57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/286e7ab0ce6a68282394c92361c27b57-Abstract-Conference.html)

        **Abstract**:

        In this work, we propose a multi-objective decision making framework that accommodates different user preferences over objectives, where preferences are learned via policy comparisons. Our model consists of a known Markov decision process with a vector-valued reward function, with each user having an unknown preference vector that expresses the relative importance of each objective. The goal is to efficiently compute a near-optimal policy for a given user. We consider two user feedback models. We first address the case where a user is provided with two policies and returns their preferred policy as feedback. We then move to a different user feedback model, where a user is instead provided with two small weighted sets of representative trajectories and selects the preferred one. In both cases, we suggest an algorithm that finds a nearly optimal policy for the user using a number of comparison queries that scales quasilinearly in the number of objectives.

        ----

        ## [535] Deep Recurrent Optimal Stopping

        **Authors**: *Niranjan Damera Venkata, Chiranjib Bhattacharyya*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28795419a644f41ede3fa058b13fc622-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28795419a644f41ede3fa058b13fc622-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) have recently emerged as a powerful paradigm for solving Markovian optimal stopping problems. However, a ready extension of DNN-based methods to non-Markovian settings requires significant state and parameter space expansion, manifesting the curse of dimensionality. Further, efficient state-space transformations permitting Markovian approximations, such as those afforded by recurrent neural networks (RNNs), are either structurally infeasible or are confounded by the curse of non-Markovianity. Considering these issues, we introduce, for the first time, an optimal stopping policy gradient algorithm (OSPG) that can leverage RNNs effectively in non-Markovian settings by implicitly optimizing value functions without recursion, mitigating the curse of non-Markovianity. The OSPG algorithm is derived from an inference procedure on a novel Bayesian network representation of discrete-time non-Markovian optimal stopping trajectories and, as a consequence, yields an offline policy gradient algorithm that eliminates expensive Monte Carlo policy rollouts.

        ----

        ## [536] A Partially-Supervised Reinforcement Learning Framework for Visual Active Search

        **Authors**: *Anindya Sarkar, Nathan Jacobs, Yevgeniy Vorobeychik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/288b63aa98084366c4536ba0574a0f22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/288b63aa98084366c4536ba0574a0f22-Abstract-Conference.html)

        **Abstract**:

        Visual active search (VAS) has been proposed as a  modeling framework in which visual cues are used to guide exploration, with the goal of identifying regions of interest in a large geospatial area. Its potential applications include identifying hot spots of rare wildlife poaching activity, search-and-rescue scenarios, identifying illegal trafficking of weapons, drugs, or people, and many others. State of the art approaches to VAS include applications of deep reinforcement learning (DRL), which yield end-to-end search policies, and traditional active search, which combines predictions with custom algorithmic approaches. While the DRL framework has been shown to greatly outperform traditional active search in such domains, its end-to-end nature does not make full use of supervised information attained either during training, or during actual search, a significant limitation if search tasks differ significantly from those in the training distribution. We propose an approach that combines the strength of both DRL and conventional active search approaches by decomposing the search policy into a prediction module, which produces a geospatial distribution of regions of interest based on task embedding and search history, and a search module, which takes the predictions and search history as input and outputs the search distribution. In addition, we develop a novel meta-learning approach for jointly learning the resulting combined policy that can make effective use of supervised information obtained both at training and decision time. Our extensive experiments demonstrate that the proposed representation and meta-learning frameworks significantly outperform state of the art in visual active search on several problem domains.

        ----

        ## [537] Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors

        **Authors**: *Yong Liu, Chenyu Li, Jianmin Wang, Mingsheng Long*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html)

        **Abstract**:

        Real-world time series are characterized by intrinsic non-stationarity that poses a principal challenge for deep forecasting models. While previous models suffer from complicated series variations induced by changing temporal distribution, we tackle non-stationary time series with modern Koopman theory that fundamentally considers the underlying time-variant dynamics. Inspired by Koopman theory of portraying complex dynamical systems, we disentangle time-variant and time-invariant components from intricate non-stationary series by Fourier Filter and design Koopman Predictor to advance respective dynamics forward. Technically, we propose Koopa as a novel Koopman forecaster composed of stackable blocks that learn hierarchical dynamics. Koopa seeks measurement functions for Koopman embedding and utilizes Koopman operators as linear portraits of implicit transition. To cope with time-variant dynamics that exhibits strong locality, Koopa calculates context-aware operators in the temporal neighborhood and is able to utilize incoming ground truth to scale up forecast horizon. Besides, by integrating Koopman Predictors into deep residual structure, we ravel out the binding reconstruction loss in previous Koopman forecasters and achieve end-to-end forecasting objective optimization. Compared with the state-of-the-art model, Koopa achieves competitive performance while saving 77.3% training time and 76.0% memory.

        ----

        ## [538] Bridging Discrete and Backpropagation: Straight-Through and Beyond

        **Authors**: *Liyuan Liu, Chengyu Dong, Xiaodong Liu, Bin Yu, Jianfeng Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html)

        **Abstract**:

        Backpropagation, the cornerstone of deep learning, is limited to computing gradients for continuous variables. This limitation poses challenges for problems involving discrete latent variables. To address this issue, we propose a novel approach to approximate the gradient of parameters involved in generating discrete latent variables. First, we examine the widely used Straight-Through (ST) heuristic and demonstrate that it works as a first-order approximation of the gradient. Guided by our findings, we propose ReinMax, which achieves second-order accuracy by integrating Heunâ€™s method, a second-order numerical method for solving ODEs. ReinMax does not require Hessian or other second-order derivatives, thus having negligible computation overheads. Extensive experimental results on various tasks demonstrate the superiority of ReinMax over the state of the art.

        ----

        ## [539] Distributed Inference and Fine-tuning of Large Language Models Over The Internet

        **Authors**: *Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, Colin A. Raffel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28bf1419b9a1f908c15f6195f58cb865-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28bf1419b9a1f908c15f6195f58cb865-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) are useful in many NLP tasks and become more capable with size, with the best open-source models having over 50 billion parameters. However, using these 50B+ models requires high-end hardware, making them inaccessible to most researchers. In this work, we investigate methods for cost-efficient inference and fine-tuning of LLMs, comparing local and distributed strategies. We observe that a large enough model (50B+) can run efficiently even on geodistributed devices in a consumer-grade network. This could allow running LLM efficiently by pooling together idle compute resources of multiple research groups and volunteers. We address two open problems: (1) how to perform inference and fine-tuning reliably if any device can disconnect abruptly and (2) how to partition LLMs between devices with uneven hardware, joining and leaving at will. In order to do that, we develop special fault-tolerant inference algorithms and load-balancing protocols that automatically assign devices to maximize the total system throughput. We showcase these algorithms in Petals â€” a decentralized system that runs Llama 2 (70B) and BLOOM (176B) over the Internet up to $10\times$ faster than offloading for interactive generation. We evaluate the performance of our system in simulated conditions and a real-world setup spanning two continents.

        ----

        ## [540] Contrast, Attend and Diffuse to Decode High-Resolution Images from Brain Activities

        **Authors**: *Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, Marie-Francine Moens*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28dad4a70f748a2980998d3ed0f1b8d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28dad4a70f748a2980998d3ed0f1b8d2-Abstract-Conference.html)

        **Abstract**:

        Decoding visual stimuli from neural responses recorded by functional Magnetic Resonance Imaging (fMRI) presents an intriguing intersection between cognitive neuroscience and machine learning, promising advancements in understanding human visual perception. However, the task is challenging due to the noisy nature of fMRI signals and the intricate pattern of brain visual representations. To mitigate these challenges, we introduce a two-phase fMRI representation learning framework. The first phase pre-trains an fMRI feature learner with a proposed Double-contrastive Mask Auto-encoder to learn denoised representations. The second phase tunes the feature learner to attend to neural activation patterns most informative for visual reconstruction with guidance from an image auto-encoder. The optimized fMRI feature learner then conditions a latent diffusion model to reconstruct image stimuli from brain activities. Experimental results demonstrate our model's superiority in generating high-resolution and semantically accurate images, substantially exceeding previous state-of-the-art methods by 39.34% in the 50-way-top-1 semantic classification accuracy. The code implementations is available at https://github.com/soinx0629/visdecneurips/.

        ----

        ## [541] Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision

        **Authors**: *Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Josh Tenenbaum, Frédo Durand, Bill Freeman, Vincent Sitzmann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28e4ee96c94e31b2d040b4521d2b299e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28e4ee96c94e31b2d040b4521d2b299e-Abstract-Conference.html)

        **Abstract**:

        Denoising diffusion models are a powerful type of generative models used to capture complex distributions of real-world signals. However, their applicability is limited to scenarios where training samples are readily available, which is not always the case in real-world applications. For example, in inverse graphics, the goal is to generate samples from a distribution of 3D scenes that align with a given image, but ground-truth 3D scenes are unavailable and only 2D images are accessible. To address this limitation, we propose a novel class of denoising diffusion probabilistic models that learn to sample from distributions of signals that are never directly observed. Instead, these signals are measured indirectly through a known differentiable forward model, which produces partial observations of the unknown signal. Our approach involves integrating the forward model directly into the denoising process. A key contribution of our work is the integration of a differentiable forward model into the denoising process. This integration effectively connects the generative modeling of observations with the generative modeling of the underlying signals, allowing for end-to-end training of a conditional generative model over signals. During inference, our approach enables sampling from the distribution of underlying signals that are consistent with a given partial observation. We demonstrate the effectiveness of our method on three challenging computer vision tasks. For instance, in the context of inverse graphics, our model enables direct sampling from the distribution of 3D scenes that align with a single 2D input image.

        ----

        ## [542] Computational Guarantees for Doubly Entropic Wasserstein Barycenters

        **Authors**: *Tomas Vaskevicius, Lénaïc Chizat*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/28fa97d12d6e3877f1c10c605d2cffa0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/28fa97d12d6e3877f1c10c605d2cffa0-Abstract-Conference.html)

        **Abstract**:

        We study the computation of doubly regularized Wasserstein barycenters, a recently introduced family of entropic barycenters governed by inner and outer regularization strengths. Previous research has demonstrated that various regularization parameter choices unify several notions of entropy-penalized barycenters while also revealing new ones, including a special case of debiased barycenters.  In this paper, we propose and analyze an algorithm for computing doubly regularized Wasserstein barycenters. Our procedure builds on damped Sinkhorn iterations followed by exact maximization/minimization steps and guarantees convergence for any choice of regularization parameters. An inexact variant of our algorithm, implementable using approximate Monte Carlo sampling, offers the first non-asymptotic convergence guarantees for approximating Wasserstein barycenters between discrete point clouds in the free-support/grid-free setting.

        ----

        ## [543] WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting

        **Authors**: *Yuxin Jia, Youfang Lin, Xinyan Hao, Yan Lin, Shengnan Guo, Huaiyu Wan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)

        **Abstract**:

        Capturing semantic information is crucial for accurate long-range time series forecasting, which involves modeling global and local correlations, as well as discovering long- and short-term repetitive patterns. Previous works have partially addressed these issues separately, but have not been able to address all of them simultaneously. Meanwhile, their time and memory complexities are still not sufficiently low for long-range forecasting. To address the challenge of capturing different types of semantic information, we propose a novel Water-wave Information Transmission (WIT) framework. This framework captures both long- and short-term repetitive patterns through bi-granular information transmission. It also models global and local correlations by recursively fusing and selecting information using Horizontal Vertical Gated Selective Unit (HVGSU). In addition, to improve the computing efficiency, we propose a generic Recurrent Acceleration Network (RAN) which reduces the time complexity to $\mathcal{O}(\sqrt{L})$ while maintaining the memory complexity at $\mathcal{O}(L)$. Our proposed method, called Water-wave Information Transmission and Recurrent Acceleration Network (WITRAN), outperforms the state-of-the-art methods by 5.80% and 14.28% on long-range and ultra-long-range time series forecasting tasks respectively, as demonstrated by experiments on four benchmark datasets. The code is available at: https://github.com/Water2sea/WITRAN.

        ----

        ## [544] Spatio-Angular Convolutions for Super-resolution in Diffusion MRI

        **Authors**: *Matthew Lyon, Paul A. Armitage, Mauricio A. Álvarez*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/294de0fa7149adcb88aa3119c239c63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/294de0fa7149adcb88aa3119c239c63e-Abstract-Conference.html)

        **Abstract**:

        Diffusion MRI (dMRI) is a widely used imaging modality, but requires long scanning times to acquire high resolution datasets. By leveraging the unique geometry present within this domain, we present a novel approach to dMRI angular super-resolution that extends upon the parametric continuous convolution (PCConv) framework. We introduce several additions to the operation including a Fourier feature mapping, 'global' co-ordinates, and domain specific context. Using this framework, we build a fully parametric continuous convolution network (PCCNN) and compare against existing models. We demonstrate the PCCNN performs competitively while using significantly fewer parameters. Moreover, we show that this formulation generalises well to clinically relevant downstream analyses such as fixel-based analysis, and neurite orientation dispersion and density imaging.

        ----

        ## [545] Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning

        **Authors**: *Changsheng Lv, Shuai Zhang, Yapeng Tian, Mengshi Qi, Huadong Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29571f8fda54fe93631c41aad4215abc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29571f8fda54fe93631c41aad4215abc-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose a Disentangled Counterfactual Learning (DCL) approach for physical audiovisual commonsense reasoning. The task aims to infer objects’ physics commonsense based on both video and audio input, with the main challenge is how to imitate the reasoning ability of humans. Most of the current methods fail to take full advantage of different characteristics in multi-modal data, and lacking causal reasoning ability in models impedes the progress of implicit physical knowledge inferring. To address these issues, our proposed DCL method decouples videos into static (time-invariant) and dynamic (time-varying) factors in the latent space by the disentangled sequential encoder, which adopts a variational autoencoder (VAE) to maximize the mutual information with a contrastive loss function. Furthermore, we introduce a counterfactual learning module to augment the model’s reasoning ability by modeling physical knowledge relationships among different objects under counterfactual intervention. Our proposed method is a plug-and-play module that can be incorporated into any baseline. In experiments, we show that our proposed method improves baseline methods and achieves state-of-the-art performance. Our source code is available at https://github.com/Andy20178/DCL.

        ----

        ## [546] Protein Design with Guided Discrete Diffusion

        **Authors**: *Nate Gruver, Samuel Stanton, Nathan C. Frey, Tim G. J. Rudner, Isidro Hötzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, Andrew Gordon Wilson*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29591f355702c3f4436991335784b503-Abstract-Conference.html)

        **Abstract**:

        A popular approach to protein design is to combine a generative model with a discriminative model for conditional sampling. The generative model samples plausible sequences while the discriminative model guides a search for sequences with high fitness. Given its broad success in conditional sampling, classifier-guided diffusion modeling is a promising foundation for protein design, leading many to develop guided diffusion models for structure with inverse folding to recover sequences. In this work, we propose diffusioN Optimized Sampling (NOS), a guidance method for discrete diffusion models that follows gradients in the hidden states of the denoising network. NOS makes it possible to perform design directly in sequence space, circumventing significant limitations of structure-based methods, including scarce data and challenging inverse design. Moreover, we use NOS to generalize LaMBO, a Bayesian optimization procedure for sequence design that facilitates multiple objectives and edit-based constraints. The resulting method, LaMBO-2, enables discrete diffusions and  stronger performance with limited edits through a novel application of saliency maps. We apply LaMBO-2 to a real-world protein design task, optimizing antibodies for higher expression yield and binding affinity to several therapeutic targets under locality and developability constraints, attaining a 99\% expression rate and 40\% binding rate in exploratory in vitro experiments.

        ----

        ## [547] Adaptive whitening with fast gain modulation and slow synaptic plasticity

        **Authors**: *Lyndon R. Duong, Eero P. Simoncelli, Dmitri B. Chklovskii, David Lipshutz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/297fe652867e4897e9f1fe1cd715de19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/297fe652867e4897e9f1fe1cd715de19-Abstract-Conference.html)

        **Abstract**:

        Neurons in early sensory areas rapidly adapt to changing sensory statistics, both by normalizing the variance of their individual responses and by reducing correlations between their responses. Together, these transformations may be viewed as an adaptive form of statistical whitening. Existing mechanistic models of adaptive whitening exclusively use either synaptic plasticity or gain modulation as the biological substrate for adaptation; however, on their own, each of these models has significant limitations. In this work, we unify these approaches in a normative multi-timescale mechanistic model that adaptively whitens its responses with complementary computational roles for synaptic plasticity and gain modulation. Gains are modified on a fast timescale to adapt to the current statistical context, whereas synapses are modified on a slow timescale to match structural properties of the input statistics that are invariant across contexts. Our model is derived from a novel multi-timescale whitening objective that factorizes the inverse whitening matrix into basis vectors, which correspond to synaptic weights, and a diagonal matrix, which corresponds to neuronal gains. We test our model on synthetic and natural datasets and find that the synapses learn optimal configurations over long timescales that enable adaptive whitening on short timescales using gain modulation.

        ----

        ## [548] Tanh Works Better with Asymmetry

        **Authors**: *Dongjin Kim, Woojeong Kim, Suhyun Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/298281b9e89197195eb461e68ad20136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/298281b9e89197195eb461e68ad20136-Abstract-Conference.html)

        **Abstract**:

        Batch Normalization is commonly located in front of activation functions, as proposed by the original paper. Swapping the order, i.e., using Batch Normalization after activation functions, has also been attempted, but its performance is generally not much different from the conventional order when ReLU or a similar activation function is used. However, in the case of bounded activation functions like Tanh, we discovered that the swapped order achieves considerably better performance than the conventional order on various benchmarks and architectures. This paper reports this remarkable phenomenon and closely examines what contributes to this performance improvement. By looking at the output distributions of individual activation functions, not the whole layers, we found that many of them are asymmetrically saturated. The experiments designed to induce a different degree of asymmetric saturation support the hypothesis that asymmetric saturation helps improve performance. In addition, Batch Normalization after bounded activation functions relocates the asymmetrically saturated output of activation functions near zero, enabling the swapped model to have high sparsity, further improving performance. Extensive experiments with Tanh, LeCun Tanh, and Softsign show that the swapped models achieve improved performance with a high degree of asymmetric saturation.  Finally, based on this investigation, we test a Tanh function shifted to be asymmetric. This shifted Tanh function that is manipulated to have consistent asymmetry shows even higher accuracy than the original Tanh used in the swapped order, confirming the asymmetry's importance. The code is available at https://github.com/hipros/tanhworksbetterwithasymmetry.

        ----

        ## [549] Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning

        **Authors**: *Yihang Yao, Zuxin Liu, Zhepeng Cen, Jiacheng Zhu, Wenhao Yu, Tingnan Zhang, Ding Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29906cbd165b78991da2c4dbabc2a04b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29906cbd165b78991da2c4dbabc2a04b-Abstract-Conference.html)

        **Abstract**:

        Safe reinforcement learning (RL) focuses on training reward-maximizing agents subject to pre-defined safety constraints. Yet, learning versatile safe policies that can adapt to varying safety constraint requirements during deployment without retraining remains a largely unexplored and challenging area. In this work, we formulate the versatile safe RL problem and consider two primary requirements: training efficiency and zero-shot adaptation capability. To address them, we introduce the Conditioned Constrained Policy Optimization (CCPO) framework, consisting of two key modules: (1) Versatile Value Estimation (VVE) for approximating value functions under unseen threshold conditions, and (2) Conditioned Variational Inference (CVI) for encoding arbitrary constraint thresholds during policy optimization. Our extensive experiments demonstrate that CCPO outperforms the baselines in terms of safety and task performance while preserving zero-shot adaptation capabilities to different constraint thresholds data-efficiently. This makes our approach suitable for real-world dynamic applications.

        ----

        ## [550] Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

        **Authors**: *Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alexander J. Smola, Xu Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29962c2c9daf1fbd92530a7c958dfc2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29962c2c9daf1fbd92530a7c958dfc2b-Abstract-Conference.html)

        **Abstract**:

        This work proposes POMP, a prompt pre-training method for vision-language models. Being memory and computation efficient, POMP enables the learned prompt to condense semantic information for a rich set of visual concepts with over twenty-thousand classes. Once pre-trained, the prompt with a strong transferable ability can be directly plugged into a variety of visual recognition tasks including image classification, semantic segmentation, and object detection, to boost recognition performances in a zero-shot manner. Empirical evaluation shows that POMP achieves state-of-the-art performances on 21 datasets, e.g., 67.0% average accuracy on 10 classification datasets (+3.1% compared to CoOp) and 84.4 hIoU on open-vocabulary Pascal VOC segmentation (+6.9 compared to ZSSeg).

        ----

        ## [551] Composing Parameter-Efficient Modules with Arithmetic Operation

        **Authors**: *Jinghan Zhang, Shiqi Chen, Junteng Liu, Junxian He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html)

        **Abstract**:

        As an efficient alternative to conventional full fine-tuning, parameter-efficient fine-tuning (PEFT) is becoming the prevailing method to adapt pretrained language models. In PEFT, a lightweight module is learned on each dataset while the underlying pretrained language model remains unchanged, resulting in multiple compact modules representing diverse skills when applied to various domains and tasks. In this paper, we propose to compose these parameter-efficient modules through linear arithmetic operations in the weight space, thereby integrating different module capabilities. Specifically, we first define an addition and negation operator for the module, and then further compose these two basic operators to perform flexible arithmetic. Our approach requires no additional training and enables highly flexible module composition.  We apply different arithmetic operations to compose the parameter-efficient modules for (1) distribution generalization, (2) multi-tasking, (3) detoxifying, and (4) domain transfer. Additionally, we extend our approach to detoxify Alpaca-LoRA, the latest instruction-tuned large language model based on LLaMA. Empirical results demonstrate that our approach produces new and effective parameter-efficient modules that significantly outperform existing ones across all settings.

        ----

        ## [552] UltraRE: Enhancing RecEraser for Recommendation Unlearning via Error Decomposition

        **Authors**: *Yuyuan Li, Chaochao Chen, Yizhao Zhang, Weiming Liu, Lingjuan Lyu, Xiaolin Zheng, Dan Meng, Jun Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29a0ea49a103a233b17c0705cdeccb66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29a0ea49a103a233b17c0705cdeccb66-Abstract-Conference.html)

        **Abstract**:

        With growing concerns regarding privacy in machine learning models, regulations have committed to granting individuals the right to be forgotten while mandating companies to develop non-discriminatory machine learning systems, thereby fueling the study of the machine unlearning problem. Our attention is directed toward a practical unlearning scenario, i.e., recommendation unlearning. As the state-of-the-art framework, i.e., RecEraser, naturally achieves full unlearning completeness, our objective is to enhance it in terms of model utility and unlearning efficiency. In this paper, we rethink RecEraser from an ensemble-based perspective and focus on its three potential losses, i.e., redundancy, relevance, and combination. Under the theoretical guidance of the above three losses, we propose a new framework named UltraRE, which simplifies and powers RecEraser for recommendation tasks. Specifically, for redundancy loss, we incorporate transport weights in the clustering algorithm to optimize the equilibrium between collaboration and balance while enhancing efficiency; for relevance loss, we ensure that sub-models reach convergence on their respective group data; for combination loss, we simplify the combination estimator without compromising its efficacy. Extensive experiments on three real-world datasets demonstrate the effectiveness of UltraRE.

        ----

        ## [553] WCLD: Curated Large Dataset of Criminal Cases from Wisconsin Circuit Courts

        **Authors**: *Elliott Ash, Naman Goel, Nianyun Li, Claudia Marangon, Peiyao Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29c80c549ed67ddd7259559c1bb07c1b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/29c80c549ed67ddd7259559c1bb07c1b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Machine learning based decision-support tools in criminal justice systems are subjects of intense discussions and academic research. There are important open questions about the utility and fairness of such tools. Academic researchers often rely on a few small datasets that are not sufficient to empirically study various real-world aspects of these questions. In this paper, we contribute WCLD, a curated large dataset of 1.5 million criminal cases from circuit courts in the U.S. state of Wisconsin. We used reliable public data from 1970 to 2020 to curate attributes like prior criminal counts and recidivism outcomes. The dataset contains large number of samples from five racial groups, in addition to information like sex and age (at judgment and first offense). Other attributes in this dataset include neighborhood characteristics obtained from census data, detailed types of offense, charge severity, case decisions, sentence lengths, year of filing etc. We also provide pseudo-identifiers for judge, county and zipcode. The dataset will not only enable researchers to more rigorously study algorithmic fairness in the context of criminal justice, but also relate algorithmic challenges with various systemic issues. We also discuss in detail the process of constructing the dataset and provide a datasheet. The WCLD dataset is available at https://clezdata.github.io/wcld/.

        ----

        ## [554] Weitzman's Rule for Pandora's Box with Correlations

        **Authors**: *Evangelia Gergatsouli, Christos Tzamos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29d319f7c1513c9ecd81d3a6e9632a6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29d319f7c1513c9ecd81d3a6e9632a6e-Abstract-Conference.html)

        **Abstract**:

        Pandora’s Box is a central problem in decision making under uncertainty that can model various real life scenarios. In this problem we are given n boxes, each with a fixed opening cost, and an unknown value drawn from a known distribution, only revealed if we pay the opening cost. Our goal is to find a strategy for opening boxes to minimize the sum of the value selected and the opening cost paid.In this work we revisit Pandora’s Box when the value distributions are correlated, first studied in [CGT+20]. We show that the optimal algorithm for the independent case, given by Weitzman’s rule, directly works for the correlated case. In fact, our algorithm results in significantly improved approximation guarantees compared to the previous work, while also being substantially simpler. We also show how to implement the rule given only sample access to the correlated distribution of values. Specifically, we find that a number of samples that is polynomial in the number of boxes is sufficient for the algorithm to work.

        ----

        ## [555] Compositional Sculpting of Iterative Generative Processes

        **Authors**: *Timur Garipov, Sebastiaan De Peuter, Ge Yang, Vikas K. Garg, Samuel Kaski, Tommi S. Jaakkola*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29dd9e016b7b2f15ceb0ea93dbf1fa53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29dd9e016b7b2f15ceb0ea93dbf1fa53-Abstract-Conference.html)

        **Abstract**:

        High training costs of generative models and the need to fine-tune them for specific tasks have created a strong interest in model reuse and composition.A key challenge in composing iterative generative processes, such as GFlowNets and diffusion models, is that to realize the desired target distribution, all steps of the generative process need to be coordinated, and satisfy delicate balance conditions.In this work, we propose Compositional Sculpting: a general approach for defining compositions of iterative generative processes. We then introduce a method for sampling from these compositions built on classifier guidance.We showcase ways to accomplish compositional sculpting in both GFlowNets and diffusion models. We highlight two binary operations $\\unicode{x2014}$ the $\\textit{harmonic mean}\\unicode{x00A0}(p_1 \\otimes p_2$) and the $\\textit{contrast}\\unicode{x00A0}(p_1 \\,\\unicode{x25D1}\\,\\, p_2$) between pairs, and the generalization of these operations to multiple component distributions.We offer empirical results on image and molecular generation tasks. Project codebase: https://github.com/timgaripov/compositional-sculpting.

        ----

        ## [556] Face Reconstruction from Facial Templates by Learning Latent Space of a Generator Network

        **Authors**: *Hatef Otroshi-Shahreza, Sébastien Marcel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29e4b51d45dc8f534260adc45b587363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29e4b51d45dc8f534260adc45b587363-Abstract-Conference.html)

        **Abstract**:

        In this paper, we focus on the template inversion attack against face recognition systems and propose a new method to reconstruct face images from facial templates. Within a generative adversarial network (GAN)-based framework, we learn a mapping from facial templates to the intermediate latent space of a pre-trained face generation network, from which we can generate high-resolution realistic reconstructed face images. We show that our proposed method can be applied in whitebox and blackbox attacks against face recognition systems. Furthermore, we evaluate the transferability of our attack when the adversary uses the reconstructed face image to impersonate the underlying subject in an attack against another face recognition system. Considering the adversary's knowledge and the target face recognition system, we define five different attacks and evaluate the vulnerability of state-of-the-art face recognition systems. Our experiments show that our proposed method achieves high success attack rates in whitebox and blackbox scenarios. Furthermore, the reconstructed face images are transferable and can be used to enter target face recognition systems with a different feature extractor model. We also explore important areas in the reconstructed face images that can fool the target face recognition system.

        ----

        ## [557] Triangulation Residual Loss for Data-efficient 3D Pose Estimation

        **Authors**: *Jiachen Zhao, Tao Yu, Liang An, Yipeng Huang, Fang Deng, Qionghai Dai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29e8437db7b549160ce03d336ff66f65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29e8437db7b549160ce03d336ff66f65-Abstract-Conference.html)

        **Abstract**:

        This paper presents Triangulation Residual loss (TR loss) for multiview 3D pose estimation in a data-efficient manner. Existing 3D supervised models usually require large-scale 3D annotated datasets, but the amount of existing data is still insufficient to train supervised models to achieve ideal performance, especially for animal pose estimation. To employ unlabeled multiview data for training, previous epipolar-based consistency provides a self-supervised loss that considers only the local consistency in pairwise views, resulting in limited performance and heavy calculations. In contrast, TR loss enables self-supervision with global multiview geometric consistency. Starting from initial 2D keypoint estimates, the TR loss can fine-tune the corresponding 2D detector without 3D supervision by simply minimizing the smallest singular value of the triangulation matrix in an end-to-end fashion. Our method achieves the state-of-the-art 25.8mm MPJPE and competitive 28.7mm MPJPE with only 5\% 2D labeled training data on the Human3.6M dataset. Experiments on animals such as mice demonstrate our TR loss's data-efficient training ability.

        ----

        ## [558] A Long N-step Surrogate Stage Reward for Deep Reinforcement Learning

        **Authors**: *Junmin Zhong, Ruofan Wu, Jennie Si*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29ef811e72b2b97cf18dd5d866b0f472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29ef811e72b2b97cf18dd5d866b0f472-Abstract-Conference.html)

        **Abstract**:

        We introduce a new stage reward estimator  named the long $N$-step surrogate stage (LNSS) reward for deep reinforcement learning (RL). It aims at mitigating the high variance problem, which has shown impeding successful convergence of learning, hurting task performance, and hindering applications of deep RL in continuous control problems. In this paper we show that LNSS, which utilizes a long reward trajectory of  rewards of future steps, provides consistent performance improvement measured by average reward, convergence speed, learning success rate,and variance reduction in $Q$ values and rewards.  Our evaluations are based on a variety of environments in DeepMind Control Suite and OpenAI Gym  by using  LNSS in baseline deep RL algorithms such as DDPG, D4PG, and TD3. We show  that LNSS reward has enabled good results that have been challenging to obtain by deep RL previously. Our analysis also shows that  LNSS exponentially reduces the upper bound on the variances of $Q$ values from respective single-step methods.

        ----

        ## [559] CODA: Generalizing to Open and Unseen Domains with Compaction and Disambiguation

        **Authors**: *Chaoqi Chen, Luyao Tang, Yue Huang, Xiaoguang Han, Yizhou Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29f3514801f3f327d808799f5ac122ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29f3514801f3f327d808799f5ac122ba-Abstract-Conference.html)

        **Abstract**:

        The generalization capability of machine learning systems degenerates notably when the test distribution drifts from the training distribution. Recently, Domain Generalization (DG) has been gaining momentum in enabling machine learning models to generalize to unseen domains. However, most DG methods assume that training and test data share an identical label space, ignoring the potential unseen categories in many real-world applications. In this paper, we delve into a more general but difficult problem termed Open Test-Time DG (OTDG), where both domain shift and open class may occur on the unseen test data. We propose Compaction and Disambiguation (CODA), a novel two-stage framework for learning compact representations and adapting to open classes in the wild. To meaningfully regularize the model's decision boundary, CODA introduces virtual unknown classes and optimizes a new training objective to insert unknowns into the latent space by compacting the embedding space of source known classes. To adapt target samples to the source model, we then disambiguate the decision boundaries between known and unknown classes with a test-time training objective, mitigating the adaptivity gap and catastrophic forgetting challenges. Experiments reveal that CODA can significantly outperform the previous best method on standard DG datasets and harmonize the classification accuracy between known and unknown classes.

        ----

        ## [560] Scale-Space Hypernetworks for Efficient Biomedical Image Analysis

        **Authors**: *Jose Javier Gonzalez Ortiz, John V. Guttag, Adrian V. Dalca*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29f421fbdcc82aeb349d784d3aaccdb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29f421fbdcc82aeb349d784d3aaccdb3-Abstract-Conference.html)

        **Abstract**:

        Convolutional Neural Networks (CNNs) are the predominant model used for a variety of medical image analysis tasks. At inference time, these models are computationally intensive, especially with volumetric data.In principle, it is possible to trade accuracy for computational efficiency by manipulating the rescaling factor in the downsample and upsample layers of CNN architectures.However, properly exploring the accuracy-efficiency trade-off is prohibitively expensive with existing models.To address this, we introduce Scale-Space HyperNetworks (SSHN), a method that learns a spectrum of CNNs with varying internal rescaling factors.A single SSHN characterizes an entire Pareto accuracy-efficiency curve of models that match, and occasionally surpass, the outcomes of training many separate networks with fixed rescaling factors.We demonstrate the proposed approach in several medical image analysis applications, comparing SSHN against strategies with both fixed and dynamic rescaling factors.We find that SSHN consistently provides a better accuracy-efficiency trade-off at a fraction of the training cost. Trained SSHNs enable the user to quickly choose a rescaling factor that appropriately balances accuracy and computational efficiency for their particular needs at inference.

        ----

        ## [561] Follow-ups Also Matter: Improving Contextual Bandits via Post-serving Contexts

        **Authors**: *Chaoqi Wang, Ziyu Ye, Zhe Feng, Ashwinkumar Badanidiyuru Varadaraja, Haifeng Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/29f47df77b7e536ebd0fe5e0cc964a32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/29f47df77b7e536ebd0fe5e0cc964a32-Abstract-Conference.html)

        **Abstract**:

        Standard contextual bandit problem assumes that all the relevant contexts are observed before the algorithm chooses an arm. This modeling paradigm, while useful, often falls short when dealing with problems in which additional valuable contexts can be observed after arm selection. For example, content recommendation platforms like Youtube, Instagram, Tiktok receive much additional features about a user's reward after the user clicks a content (e.g., how long the user stayed, what is the user's watch speed, etc.). To improve online learning efficiency in these applications,  we  study a novel contextual bandit problem with post-serving contexts and design a new algorithm, poLinUCB,  that achieves tight regret under standard assumptions. Core to our technical proof is a robustified and generalized version of the well-known Elliptical Potential Lemma (EPL), which can accommodate  noise in data. Such robustification is necessary for tackling our problem, though we believe it could also be of general interest.Extensive empirical tests on both synthetic and real-world datasets  demonstrate the significant benefit of utilitzing post-serving contexts as well as the superior performance of  our   algorithm over the state-of-the-art approaches.

        ----

        ## [562] Offline Minimax Soft-Q-learning Under Realizability and Partial Coverage

        **Authors**: *Masatoshi Uehara, Nathan Kallus, Jason D. Lee, Wen Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a095b46705d7e6f81fc50270fe770c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a095b46705d7e6f81fc50270fe770c2-Abstract-Conference.html)

        **Abstract**:

        We consider offline reinforcement learning (RL) where we only have only access to offline data. In contrast to numerous offline RL algorithms that necessitate the uniform coverage of the offline data over state and action space, we propose value-based algorithms with PAC guarantees under partial coverage, specifically, coverage of offline data against a single policy, and realizability of soft Q-function (a.k.a., entropy-regularized Q-function) and another function, which is defined as a solution to a saddle point of certain minimax optimization problem). Furthermore, we show the analogous result for Q-functions instead of soft Q-functions. To attain these guarantees, we use novel algorithms with minimax loss functions to accurately estimate soft Q-functions and Q-functions with -convergence guarantees measured on the offline data. We introduce these loss functions by casting the estimation problems into nonlinear convex optimization problems and taking the Lagrange functions.

        ----

        ## [563] Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption

        **Authors**: *Yige Hong, Qiaomin Xie, Yudong Chen, Weina Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a0babff3ddd4ba12062219ec161ce86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a0babff3ddd4ba12062219ec161ce86-Abstract-Conference.html)

        **Abstract**:

        We study the infinite-horizon restless bandit problem with the average reward criterion, in both discrete-time and continuous-time settings.A fundamental goal is to efficiently compute policies that achieve a diminishing optimality gap as the number of arms, $N$, grows large. Existing results on asymptotic optimality all rely on the uniform global attractor property (UGAP), a complex and challenging-to-verify assumption. In this paper, we propose a general, simulation-based framework, Follow-the-Virtual-Advice, that converts any single-armed policy into a policy for the original $N$-armed problem. This is done by simulating the single-armed policy on each arm and carefully steering the real state towards the simulated state. Our framework can be instantiated to produce a policy with an $O(1/\sqrt{N})$ optimality gap. In the discrete-time setting, our result holds under a simpler synchronization assumption, which covers some problem instances that violate UGAP. More notably, in the continuous-time setting, we do not require \emph{any} additional assumptions beyond the standard unichain condition. In both settings, our work is the first asymptotic optimality result that does not require UGAP.

        ----

        ## [564] Smooth Flipping Probability for Differential Private Sign Random Projection Methods

        **Authors**: *Ping Li, Xiaoyun Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a128987fe57b27fa0c7a0b748b0fa1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a128987fe57b27fa0c7a0b748b0fa1e-Abstract-Conference.html)

        **Abstract**:

        We develop a series of differential privacy (DP) algorithms from a family of random projection (RP) and sign random projection (SignRP) methods. We first show how to improve the previous DP-RP approach using the ``optimal Gaussian mechanism''. Then, we propose a series of DP-SignRP algorithms that leverage the robustness of the ``sign flipping probability'' of random projections. That is, given $x = \sum_{i=1}^p u_i w_{i}$ where $u$ is a $p$-dimensional data vector and $w$ is a symmetric random vector,  $sign(x)$ only has a fairly small probability to be flipped if there is a small modification on data $u$, depending on the specific distribution of $w$. This robustness leads to our novel design of ``smooth flipping probability'' for SignRP-type algorithms with better utility than using the standard randomized response mechanism. Retrieval and classification experiments demonstrate that,  among the presented DP-RP  algorithms,  \textbf{DP-SignOPORP} (where OPORP  is an improvement over the celebrated  count-sketch algorithms), performs the best in general.In the industrial practice, DP methods were not very popular for machine learning or search, largely because the performance typically would  drop substantially if DP is applied. Since our proposed new DP algorithms have significantly improved the  performance, it is anticipated that our work will motivate a wide adoption of DP in practice. Finally, we   stress that, since our  methods are applied to the original data (i.e., feature vectors), the privacy of  downstream tasks is naturally protected.

        ----

        ## [565] Idempotent Learned Image Compression with Right-Inverse

        **Authors**: *Yanghao Li, Tongda Xu, Yan Wang, Jingjing Liu, Ya-Qin Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of idempotent learned image compression (LIC).The idempotence of codec refers to the stability of codec to re-compression.To achieve idempotence, previous codecs adopt invertible transforms such as DCT and normalizing flow.In this paper, we first identify that invertibility of transform is sufficient but not necessary for idempotence. Instead, it can be relaxed into right-invertibility. And such relaxation allows wider family of transforms.Based on this identification, we implement an idempotent codec using our proposed blocked convolution and null-space enhancement.Empirical results show that we achieve state-of-the-art rate-distortion performance among idempotent codecs. Furthermore, our codec can be extended into near-idempotent codec by relaxing the right-invertibility. And this near-idempotent codec has significantly less quality decay after $50$ rounds of re-compression compared with other near-idempotent codecs.

        ----

        ## [566] A Simple Yet Effective Strategy to Robustify the Meta Learning Paradigm

        **Authors**: *Qi Wang, Yiqin Lv, Yang-He Feng, Zheng Xie, Jincai Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a28bea6298d106eed091ac403d8c22b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a28bea6298d106eed091ac403d8c22b-Abstract-Conference.html)

        **Abstract**:

        Meta learning is a promising paradigm to enable skill transfer across tasks.Most previous methods employ the empirical risk minimization principle in optimization.However, the resulting worst fast adaptation to a subset of tasks can be catastrophic in risk-sensitive scenarios.To robustify fast adaptation, this paper optimizes meta learning pipelines from a distributionally robust perspective and meta trains models with the measure of tail task risk.We take the two-stage strategy as heuristics to solve the robust meta learning problem, controlling the worst fast adaptation cases at a certain probabilistic level. Experimental results show that our simple method can improve the robustness of meta learning to task distributions and reduce the conditional expectation of the worst fast adaptation risk.

        ----

        ## [567] Stabilized Neural Differential Equations for Learning Dynamics with Explicit Constraints

        **Authors**: *Alistair White, Niki Kilbertus, Maximilian Gelbrecht, Niklas Boers*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a4179ef39846557e99f6bfac580ea2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a4179ef39846557e99f6bfac580ea2e-Abstract-Conference.html)

        **Abstract**:

        Many successful methods to learn dynamical systems from data have recently been introduced. However, ensuring that the inferred dynamics preserve known constraints, such as conservation laws or restrictions on the allowed system states, remains challenging. We propose stabilized neural differential equations (SNDEs), a method to enforce arbitrary manifold constraints for neural differential equations. Our approach is based on a stabilization term that, when added to the original dynamics, renders the constraint manifold provably asymptotically stable. Due to its simplicity, our method is compatible with all common neural differential equation (NDE) models and broadly applicable. In extensive empirical evaluations, we demonstrate that SNDEs outperform existing methods while broadening the types of constraints that can be incorporated into NDE training.

        ----

        ## [568] On the Importance of Exploration for Generalization in Reinforcement Learning

        **Authors**: *Yiding Jiang, J. Zico Kolter, Roberta Raileanu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a4310c4fd24bd336aa2f64f93cb5d39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a4310c4fd24bd336aa2f64f93cb5d39-Abstract-Conference.html)

        **Abstract**:

        Existing approaches for improving generalization in deep reinforcement learning (RL) have mostly focused on representation learning, neglecting RL-specific aspects such as exploration. We hypothesize that the agent's exploration strategy plays a key role in its ability to generalize to new environments.Through a series of experiments in a tabular contextual MDP, we show that exploration is helpful not only for efficiently finding the optimal policy for the training environments but also for acquiring knowledge that helps decision making in unseen environments. Based on these observations, we propose EDE: Exploration via Distributional Ensemble, a method that encourages the exploration of states with high epistemic uncertainty through an ensemble of Q-value distributions. The proposed algorithm is the first value-based approach to achieve strong performance on both Procgen and Crafter, two benchmarks for generalization in RL with high-dimensional observations. The open-sourced implementation can be found at https://github.com/facebookresearch/ede.

        ----

        ## [569] Uniform Convergence with Square-Root Lipschitz Loss

        **Authors**: *Lijia Zhou, Zhen Dai, Frederic Koehler, Nati Srebro*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a50f08293e5f635655e8bec8f013d99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a50f08293e5f635655e8bec8f013d99-Abstract-Conference.html)

        **Abstract**:

        We establish generic uniform convergence guarantees for Gaussian data in terms of the Radamacher complexity of the hypothesis class and the Lipschitz constant of the square root of the scalar loss function.  We show how these guarantees substantially generalize previous results based on smoothness (Lipschitz constant of the derivative), and allow us to handle the broader class of square-root-Lipschtz losses, which includes also non-smooth loss functions appropriate for studying phase retrieval and ReLU regression, as well as rederive and better understand “optimistic rate” and interpolation learning guarantees.

        ----

        ## [570] A Fractional Graph Laplacian Approach to Oversmoothing

        **Authors**: *Sohir Maskey, Raffaele Paolino, Aras Bacho, Gitta Kutyniok*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a514213ba899f2911723a38be8d4096-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a514213ba899f2911723a38be8d4096-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs) have shown state-of-the-art performances in various applications. However, GNNs often struggle to capture long-range dependencies in graphs due to oversmoothing. In this paper, we generalize the concept of oversmoothing from undirected to directed graphs. To this aim, we extend the notion of Dirichlet energy by considering a directed symmetrically normalized Laplacian. As vanilla graph convolutional networks are prone to oversmooth, we adopt a neural graph ODE framework. Specifically, we propose fractional graph Laplacian neural ODEs, which describe non-local dynamics. We prove that our approach allows propagating information  between distant nodes while maintaining a low probability of long-distance jumps. Moreover, we show that our method is more flexible with respect to the convergence of the graph’s Dirichlet energy, thereby mitigating oversmoothing. We conduct extensive experiments on synthetic and real-world graphs, both  directed and undirected, demonstrating our method’s versatility across diverse graph homophily levels.  Ourcode is available at https://github.com/RPaolino/fLode

        ----

        ## [571] On the Convergence and Sample Complexity Analysis of Deep Q-Networks with ε-Greedy Exploration

        **Authors**: *Shuai Zhang, Hongkang Li, Meng Wang, Miao Liu, Pin-Yu Chen, Songtao Lu, Sijia Liu, Keerthiram Murugesan, Subhajit Chaudhury*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a91de02871011d0090e662ffd6f2328-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a91de02871011d0090e662ffd6f2328-Abstract-Conference.html)

        **Abstract**:

        This paper provides a theoretical understanding of deep Q-Network (DQN) with the $\varepsilon$-greedy exploration in deep reinforcement learning.Despite the tremendous empirical achievement of the DQN, its theoretical characterization remains underexplored.First, the exploration strategy is either impractical or ignored in the existing analysis.  Second, in contrast to conventional Q-learning algorithms, the DQN employs the target network and experience replay to acquire an unbiased estimation of the mean-square Bellman error (MSBE) utilized in training  the Q-network. However,the existing theoretical analysis of DQNs lacks convergence analysis or bypasses the technical challenges by deploying a significantly overparameterized neural network, which is not computationally efficient. This paper provides the first theoretical convergence and sample complexity analysis of the  practical setting of DQNs with $\epsilon$-greedy policy. We prove an iterative procedure with decaying $\epsilon$ converges to the optimal Q-value function geometrically. Moreover, a higher level of $\epsilon$ values enlarges the region of convergence but slows down the convergence, while the opposite holds for a lower level of $\epsilon$ values. Experiments justify our established theoretical insights on DQNs.

        ----

        ## [572] Joint Data-Task Generation for Auxiliary Learning

        **Authors**: *Hong Chen, Xin Wang, Yuwei Zhou, Yijian Qin, Chaoyu Guan, Wenwu Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a91fb5a4c03e0b6d889e1c52f775480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a91fb5a4c03e0b6d889e1c52f775480-Abstract-Conference.html)

        **Abstract**:

        Current auxiliary learning methods mainly adopt the methodology of reweighing losses for the manually collected auxiliary data and tasks. However, these methods heavily rely on domain knowledge during data collection, which may be hardly available in reality. Therefore, current methods will become less effective and even do harm to the primary task when unhelpful auxiliary data and tasks are employed. To tackle the problem, we propose a joint data-task generation framework for auxiliary learning (DTG-AuxL), which can bring benefits to the primary task by generating the new auxiliary data and task in a joint manner. The proposed DTG-AuxL framework contains a joint generator and a bi-level optimization strategy. Specifically, the joint generator contains a feature generator and a label generator, which are designed to be applicable  and expressive for various auxiliary learning scenarios. The bi-level optimization strategy optimizes the joint generator and the task learning model, where the joint generator is effectively optimized in the upper level via the implicit gradient from the primary loss and the explicit gradient of our proposed instance regularization, while the task learning model is optimized in the lower level by the generated data and task. Extensive experiments show that our proposed DTG-AuxL framework consistently outperforms existing methods in various auxiliary learning scenarios, particularly when the manually collected auxiliary data and tasks are unhelpful.

        ----

        ## [573] On Proper Learnability between Average- and Worst-case Robustness

        **Authors**: *Vinod Raman, Unique Subedi, Ambuj Tewari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a952768bb85041f95ed06a5b60cf4d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a952768bb85041f95ed06a5b60cf4d5-Abstract-Conference.html)

        **Abstract**:

        Recently, Montasser at al. (2019) showed that finite VC dimension is not sufficient for proper adversarially robust PAC learning. In light of this hardness, there is a growing effort to study what type of relaxations to the adversarially robust PAC learning setup can enable proper learnability. In this work, we initiate the study of proper learning under relaxations of the worst-case robust loss. We give a family of robust loss relaxations under which VC classes are properly PAC learnable with sample complexity close to what one would require in the standard PAC learning setup. On the other hand, we show that for an existing and natural relaxation of the worst-case robust loss, finite VC dimension is not sufficient for proper learning. Lastly, we give new generalization guarantees for the adversarially robust empirical risk minimizer.

        ----

        ## [574] Distributional Policy Evaluation: a Maximum Entropy approach to Representation Learning

        **Authors**: *Riccardo Zamboni, Alberto Maria Metelli, Marcello Restelli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2a98af4fea6a24b73af7b588ca95f755-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2a98af4fea6a24b73af7b588ca95f755-Abstract-Conference.html)

        **Abstract**:

        The Maximum Entropy (Max-Ent) framework has been effectively employed in a variety of Reinforcement Learning (RL) tasks. In this paper, we first propose a novel Max-Ent framework for policy evaluation in a distributional RL setting, named Distributional Maximum Entropy Policy Evaluation (D-Max-Ent PE). We derive a generalization-error bound that depends on the complexity of the representation employed, showing that this framework can explicitly take into account the features used to represent the state space while evaluating a policy. Then, we exploit these favorable properties to drive the representation learning of the state space in a Structural Risk Minimization fashion. We employ state-aggregation functions as feature functions and we specialize the D-Max-Ent approach into an algorithm, named D-Max-Ent Progressive Factorization, which constructs a progressively finer-grained representation of the state space by balancing the trade-off between preserving information (bias) and reducing the effective number of states, i.e., the complexity of the representation space (variance). Finally, we report the results of some illustrative numerical simulations, showing that the proposed algorithm matches the expected theoretical behavior and highlighting the relationship between aggregations and sample regimes.

        ----

        ## [575] Thin and deep Gaussian processes

        **Authors**: *Daniel Augusto de Souza, Alexander Nikitin, St John, Magnus Ross, Mauricio A. Álvarez, Marc Peter Deisenroth, João Paulo Pordeus Gomes, Diego Mesquita, César C. Lincoln Mattos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aa212d6f40c1cb19b777e83db00ec6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aa212d6f40c1cb19b777e83db00ec6a-Abstract-Conference.html)

        **Abstract**:

        Gaussian processes (GPs) can provide a principled approach to uncertainty quantification with easy-to-interpret kernel hyperparameters, such as the lengthscale, which controls the correlation distance of function values.However, selecting an appropriate kernel can be challenging.Deep GPs avoid manual kernel engineering by successively parameterizing kernels with GP layers, allowing them to learn low-dimensional embeddings of the inputs that explain the output data.Following the architecture of deep neural networks, the most common deep GPs warp the input space layer-by-layer but lose all the interpretability of shallow GPs. An alternative construction is to successively parameterize the lengthscale of a kernel, improving the interpretability but ultimately giving away the notion of learning lower-dimensional embeddings. Unfortunately, both methods are susceptible to particular pathologies which may hinder fitting and limit their interpretability.This work proposes a novel synthesis of both previous approaches: {Thin and Deep GP} (TDGP). Each TDGP layer defines locally linear transformations of the original input data maintaining the concept of latent embeddings while also retaining the interpretation of lengthscales of a kernel. Moreover, unlike the prior solutions, TDGP induces non-pathological manifolds that admit learning lower-dimensional representations.We show with theoretical and experimental results that i) TDGP is, unlike previous models, tailored to specifically discover lower-dimensional manifolds in the input data, ii) TDGP behaves well when increasing the number of layers, and iii) TDGP performs well in standard benchmark datasets.

        ----

        ## [576] Human-like Few-Shot Learning via Bayesian Reasoning over Natural Language

        **Authors**: *Kevin Ellis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aa9b18b9ab37b0ab1fdaae46fb781d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aa9b18b9ab37b0ab1fdaae46fb781d4-Abstract-Conference.html)

        **Abstract**:

        A core tension in models of concept learning is that the model must carefully balance the tractability of inference against the expressivity of the hypothesis class. Humans, however, can efficiently learn a broad range of concepts. We introduce a model of inductive learning that seeks to be human-like in that sense.It implements a Bayesian reasoning process where a language model first proposes candidate hypotheses expressed in natural language, which are then re-weighed by a prior and a likelihood.By estimating the prior from human data, we can predict human judgments on learning problems involving numbers and sets, spanning concepts that are generative, discriminative, propositional, and higher-order.

        ----

        ## [577] CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion

        **Authors**: *Anders Vestergaard Nørskov, Alexander Neergaard Zahid, Morten Mørup*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aab54135bd206ef6d4949ce17528d98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aab54135bd206ef6d4949ce17528d98-Abstract-Conference.html)

        **Abstract**:

        Electroencephalography (EEG) is a prominent non-invasive neuroimaging technique providing insights into brain function. Unfortunately, EEG data exhibit a high degree of noise and variability across subjects hampering generalizable signal extraction. Therefore, a key aim in EEG analysis is to extract the underlying neural activation (content) as well as to account for the individual subject variability (style). We hypothesize that the ability to convert EEG signals between tasks and subjects requires the extraction of latent representations accounting for content and style. Inspired by recent advancements in voice conversion technologies, we propose a novel contrastive split-latent permutation autoencoder (CSLP-AE) framework that directly optimizes for EEG conversion. Importantly, the latent representations are guided using contrastive learning to promote the latent splits to explicitly represent subject (style) and task (content). We contrast CSLP-AE to conventional supervised, unsupervised (AE), and self-supervised (contrastive learning) training and find that the proposed approach provides favorable generalizable characterizations of subject and task. Importantly, the procedure also enables zero-shot conversion between unseen subjects. While the present work only considers conversion of EEG, the proposed CSLP-AE provides a general framework for signal conversion and extraction of content (task activation) and style (subject variability) components of general interest for the modeling and analysis of biological signals.

        ----

        ## [578] Delegated Classification

        **Authors**: *Eden Saig, Inbal Talgam-Cohen, Nir Rosenfeld*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aab664e0d1656e8b56c74f868e1ea69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aab664e0d1656e8b56c74f868e1ea69-Abstract-Conference.html)

        **Abstract**:

        When machine learning is outsourced to a rational agent, conflicts of interest might arise and severely impact predictive performance. In this work, we propose a theoretical framework for incentive-aware delegation of machine learning tasks. We model delegation as a principal-agent game, in which accurate learning can be incentivized by the principal using performance-based contracts. Adapting the economic theory of contract design to this setting, we define budget-optimal contracts and prove they take a simple threshold form under reasonable assumptions. In the binary-action case, the optimality of such contracts is shown to be equivalent to the classic Neyman-Pearson lemma, establishing a formal connection between contract design and statistical hypothesis testing. Empirically, we demonstrate that budget-optimal contracts can be constructed using small-scale data, leveraging recent advances in the study of learning curves and scaling laws. Performance and economic outcomes are evaluated using synthetic and real-world classification tasks.

        ----

        ## [579] PTQD: Accurate Post-Training Quantization for Diffusion Models

        **Authors**: *Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aab8a76c7e761b66eccaca0927787de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aab8a76c7e761b66eccaca0927787de-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have recently dominated image synthesis and other related generative tasks. However, the iterative denoising process is expensive in computations at inference time, making diffusion models less practical for low-latency and scalable real-world applications. Post-training quantization of diffusion models can significantly reduce the model size and accelerate the sampling process without requiring any re-training. Nonetheless, applying existing post-training quantization methods directly to low-bit diffusion models can significantly impair the quality of generated samples. Specifically, for each denoising step, quantization noise leads to deviations in the estimated mean and mismatches with the predetermined variance schedule. Moreover, as the sampling process proceeds, the quantization noise may accumulate, resulting in a low signal-to-noise ratio (SNR) during the later denoising steps. To address these challenges, we propose a unified formulation for the quantization noise and diffusion perturbed noise in the quantized denoising process. Specifically, we first disentangle the quantization noise into its correlated and residual uncorrelated parts regarding its full-precision counterpart. The correlated part can be easily corrected by estimating the correlation coefficient. For the uncorrelated part, we subtract the bias from the quantized results to correct the mean deviation and calibrate the denoising variance schedule to absorb the excess variance resulting from quantization. Moreover, we introduce a mixed-precision scheme for selecting the optimal bitwidth for each denoising step, which prioritizes lower bitwidths to expedite early denoising steps, while ensuring that higher bitwidths maintain a high signal-to-noise ratio (SNR) in the later steps. Extensive experiments demonstrate that our method outperforms previous post-training quantized diffusion models in generating high-quality samples, with only a $0.06$ increase in FID score compared to full-precision LDM-4 on ImageNet $256\times256$, while saving $19.9\times$ bit operations. Code is available at [https://github.com/ziplab/PTQD](https://github.com/ziplab/PTQD).

        ----

        ## [580] Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery

        **Authors**: *Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark E. Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)

        **Abstract**:

        Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.

        ----

        ## [581] Doubly Constrained Fair Clustering

        **Authors**: *John P. Dickerson, Seyed A. Esmaeili, Jamie H. Morgenstern, Claire Jie Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ab87e2179b8ea209b52463802d62560-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ab87e2179b8ea209b52463802d62560-Abstract-Conference.html)

        **Abstract**:

        The remarkable attention which fair clustering has received in the last few years has resulted in a significant number of different notions of fairness. Despite the fact that these notions are well-justified, they are often motivated and studied in a disjoint manner where one fairness desideratum is considered exclusively in isolation from the others. This leaves the understanding of the relations between different fairness notions as an important open problem in fair clustering. In this paper, we take the first step in this direction. Specifically, we consider the two most prominent demographic representation fairness notions in clustering: (1) Group Fairness ($\textbf{GF}$), where the different demographic groups are supposed to have close to population-level representation in each cluster and (2) Diversity in Center Selection ($\textbf{DS}$), where the selected centers are supposed to have close to population-level representation of each group. We show that given a constant approximation algorithm for one constraint ($\textbf{GF}$ or $\textbf{DS}$ only) we can obtain a constant approximation solution that satisfies both constraints simultaneously. Interestingly, we prove that any given solution that satisfies the $\textbf{GF}$ constraint can always be post-processed at a bounded degradation to the clustering cost to additionally satisfy the $\textbf{DS}$ constraint while the same statement is not true given a solution that satisfies $\textbf{DS}$ instead. Furthermore, we show that both $\textbf{GF}$ and $\textbf{DS}$ are incompatible (having an empty feasibility set in the worst case) with a collection of other distance-based fairness notions. Finally, we carry experiments to validate our theoretical findings.

        ----

        ## [582] ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting

        **Authors**: *Zongsheng Yue, Jianyi Wang, Chen Change Loy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ac2eac5098dba08208807b65c5851cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ac2eac5098dba08208807b65c5851cc-Abstract-Conference.html)

        **Abstract**:

        Diffusion-based image super-resolution (SR) methods are mainly limited by the low inference speed due to the requirements of hundreds or even thousands of sampling steps. Existing acceleration sampling techniques inevitably sacrifice performance to some extent, leading to over-blurry SR results. To address this issue, we propose a novel and efficient diffusion model for SR that significantly reduces the number of diffusion steps, thereby eliminating the need for post-acceleration during inference and its associated performance deterioration. Our method constructs a Markov chain that transfers between the high-resolution image and the low-resolution image by shifting the residual between them, substantially improving the transition efficiency. Additionally, an elaborate noise schedule is developed to flexibly control the shifting speed and the noise strength during the diffusion process. Extensive experiments demonstrate that the proposed method obtains superior or at least comparable performance to current state-of-the-art methods on both synthetic and real-world datasets, \textit{\textbf{even only with 20 sampling steps}}. Our code and model will be made publicly.

        ----

        ## [583] WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding

        **Authors**: *Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ac879d1865475a7abc8dfc7a9c15c27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ac879d1865475a7abc8dfc7a9c15c27-Abstract-Conference.html)

        **Abstract**:

        Graphs are widely used to model interconnected entities and improve downstream predictions in various real-world applications. However, real-world graphs nowadays are often associated with complex attributes on multiple types of nodes and even links that are hard to model uniformly, while the widely used graph neural networks (GNNs) often require sufficient training toward specific downstream predictions to achieve strong performance. In this work, we take a fundamentally different approach than GNNs, to simultaneously achieve deep joint modeling of complex attributes and flexible structures of real-world graphs and obtain unsupervised generic graph representations that are not limited to specific downstream predictions. Our framework, built on a natural integration of language models (LMs) and random walks (RWs), is straightforward, powerful and data-efficient. Specifically, we first perform attributed RWs on the graph and design an automated program to compose roughly meaningful textual sequences directly from the attributed RWs; then we fine-tune an LM using the RW-based textual sequences and extract embedding vectors from the LM, which encapsulates both attribute semantics and graph structures. In our experiments, we evaluate the learned node embeddings towards different downstream prediction tasks on multiple real-world attributed graph datasets and observe significant improvements over a comprehensive set of state-of-the-art unsupervised node embedding methods. We believe this work opens a door for more sophisticated technical designs and empirical evaluations toward the leverage of LMs for the modeling of real-world graphs.

        ----

        ## [584] Generalizing Nonlinear ICA Beyond Structural Sparsity

        **Authors**: *Yujia Zheng, Kun Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2aebc17b683792a17dd4a24fcb038ba6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2aebc17b683792a17dd4a24fcb038ba6-Abstract-Conference.html)

        **Abstract**:

        Nonlinear independent component analysis (ICA) aims to uncover the true latent sources from their observable nonlinear mixtures. Despite its significance, the identifiability of nonlinear ICA is known to be impossible without additional assumptions. Recent advances have proposed conditions on the connective structure from sources to observed variables, known as Structural Sparsity, to achieve identifiability in an unsupervised manner. However, the sparsity constraint may not hold universally for all sources in practice. Furthermore, the assumptions of bijectivity of the mixing process and independence among all sources, which arise from the setting of ICA, may also be violated in many real-world scenarios. To address these limitations and generalize nonlinear ICA, we propose a set of new identifiability results in the general settings of undercompleteness, partial sparsity and source dependence, and flexible grouping structures. Specifically, we prove identifiability when there are more observed variables than sources (undercomplete), and when certain sparsity and/or source independence assumptions are not met for some changing sources. Moreover, we show that even in cases with flexible grouping structures (e.g., part of the sources can be divided into irreducible independent groups with various sizes), appropriate identifiability results can also be established. Theoretical claims are supported empirically on both synthetic and real-world datasets.

        ----

        ## [585] Towards Characterizing the First-order Query Complexity of Learning (Approximate) Nash Equilibria in Zero-sum Matrix Games

        **Authors**: *Hédi Hadiji, Sarah Sachs, Tim van Erven, Wouter M. Koolen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2af57f909a99113db071672da236a5f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2af57f909a99113db071672da236a5f2-Abstract-Conference.html)

        **Abstract**:

        In the first-order query model for zero-sum $K\times K$ matrix games, players observe the expected pay-offs for all their possible actions under the randomized action played by their opponent. This classical model has received renewed interest after the discovery by Rakhlin and Sridharan that $\epsilon$-approximate Nash equilibria can be computed efficiently from $O(\frac{\ln K}{\epsilon})$ instead of $O(\frac{\ln K}{\epsilon^2})$ queries. Surprisingly, the optimal number of such queries, as a function of both $\epsilon$ and $K$, is not known. We make progress on this question on two fronts. First, we fully characterise the query complexity of learning exact equilibria ($\epsilon=0$), by showing that they require a number of queries that is linear in $K$, which means that it is essentially as hard as querying the whole matrix, which can also be done with $K$ queries. Second, for $\epsilon > 0$, the current query complexity upper bound stands at $O(\min(\frac{\ln(K)}{\epsilon} , K))$. We argue that, unfortunately, obtaining a matching lower bound is not possible with existing techniques: we prove that no lower bound can be derived by constructing hard matrices whose entries take values in a known countable set, because such matrices can be fully identified by a single query. This rules out, for instance, reducing to an optimization problem over the hypercube by encoding it as a binary payoff matrix. We then introduce a new technique for lower bounds, which allows us to obtain lower bounds of order $\tilde\Omega(\log(\frac{1}{K\epsilon})$ for any $\epsilon \leq 1 / (cK^4)$, where $c$ is a constant independent of $K$. We further discuss possible future directions to improve on our techniques in order to close the gap with the upper bounds.

        ----

        ## [586] Fast Conditional Mixing of MCMC Algorithms for Non-log-concave Distributions

        **Authors**: *Xiang Cheng, Bohan Wang, Jingzhao Zhang, Yusong Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b00b3331bd0f5fbfdd966ac06338f6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b00b3331bd0f5fbfdd966ac06338f6d-Abstract-Conference.html)

        **Abstract**:

        MCMC algorithms offer empirically efficient tools for sampling from a target distribution $\pi(x) \propto \exp(-V(x))$. However, on the theory side, MCMC algorithms suffer from slow mixing rate when $\pi(x)$ is non-log-concave. Our work examines this gap and shows that when Poincar\'e-style inequality holds on a subset $\mathcal{X}$ of the state space, the conditional distribution of MCMC iterates over $\mathcal{X}$ mixes fast to the true conditional distribution. This fast mixing guarantee can hold in cases when global mixing is provably slow. We formalize the statement and quantify the conditional mixing rate. We further show that conditional mixing can have interesting implications for sampling from mixtures of Gaussians, parameter estimation for Gaussian mixture models, and Gibbs-sampling with well-connected local minima.

        ----

        ## [587] How to Select Which Active Learning Strategy is Best Suited for Your Specific Problem and Budget

        **Authors**: *Guy Hacohen, Daphna Weinshall*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b09bb02b90584e2be94ff3ae09289bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b09bb02b90584e2be94ff3ae09289bc-Abstract-Conference.html)

        **Abstract**:

        In the domain of Active Learning (AL), a learner actively selects which unlabeled examples to seek labels from an oracle, while operating within predefined budget constraints. Importantly, it has been recently shown that distinct query strategies are better suited for different conditions and budgetary constraints. In practice, the determination of the most appropriate AL strategy for a given situation remains an open problem. To tackle this challenge, we propose a practical derivative-based method that dynamically identifies the best strategy for a given budget. Intuitive motivation for our approach is provided by the theoretical analysis of a simplified scenario. We then introduce a method to dynamically select an AL strategy, which takes into account the unique characteristics of the problem and the available budget. Empirical results showcase the effectiveness of our approach across diverse budgets and computer vision tasks.

        ----

        ## [588] Aligning Synthetic Medical Images with Clinical Knowledge using Human Feedback

        **Authors**: *Shenghuan Sun, Gregory M. Goldgof, Atul J. Butte, Ahmed M. Alaa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b1d1e5affe5fdb70372cd90dd8afd49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b1d1e5affe5fdb70372cd90dd8afd49-Abstract-Conference.html)

        **Abstract**:

        Generative models capable of precisely capturing nuanced clinical features in medical images hold great promise for facilitating clinical data sharing, enhancing rare disease datasets, and efficiently synthesizing (annotated) medical images at scale. Despite their potential, assessing the quality of synthetic medical images remains a challenge. While modern generative models can synthesize visually-realistic medical images, the clinical plausibility of these images may be called into question. Domain-agnostic scores, such as FID score, precision, and recall, cannot incorporate clinical knowledge and are, therefore, not suitable for assessing clinical sensibility. Additionally, there are numerous unpredictable ways in which generative models may fail to synthesize clinically plausible images, making it challenging to anticipate potential failures and design automated scores for their detection. To address these challenges, this paper introduces a pathologist-in-the-loop framework for generating clinically-plausible synthetic medical images. Our framework comprises three steps: (1) pretraining a conditional diffusion model to generate medical images conditioned on a clinical concept, (2) expert pathologist evaluation of the generated images to assess whether they satisfy clinical desiderata, and (3) training a reward model that predicts human feedback on new samples, which we use to incorporate expert knowledge into the finetuning objective of the diffusion model. Our results show that human feedback significantly improves the quality of synthetic images in terms of fidelity, diversity, utility in downstream applications, and plausibility as evaluated by experts. We also demonstrate that human feedback can teach the model new clinical concepts not annotated in the original training data. Our results demonstrate the value of incorporating human feedback in clinical applications where generative models may struggle to capture extensive domain knowledge from raw data alone.

        ----

        ## [589] Interpretable Graph Networks Formulate Universal Algebra Conjectures

        **Authors**: *Francesco Giannini, Stefano Fioravanti, Oguzhan Keskin, Alisia Maria Lupidi, Lucie Charlotte Magister, Pietro Lió, Pietro Barbiero*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b2011a7d5396faf5899863d896a3c24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b2011a7d5396faf5899863d896a3c24-Abstract-Conference.html)

        **Abstract**:

        The rise of Artificial Intelligence (AI) recently empowered researchers to investigate hard mathematical problems which eluded traditional approaches for decades. Yet, the use of AI in Universal Algebra (UA)---one of the fields laying the foundations of modern mathematics---is still completely unexplored. This work proposes the first use of AI to investigate UA's conjectures with an equivalent equational and topological characterization. While topological representations would enable the analysis of such properties using graph neural networks, the limited transparency and brittle explainability of these models hinder their straightforward use to empirically validate existing conjectures or to formulate new ones. To bridge these gaps, we propose a general algorithm generating AI-ready datasets based on UA's conjectures, and introduce a novel neural layer to build fully interpretable graph networks. The results of our experiments demonstrate that interpretable graph networks: (i) enhance interpretability without sacrificing task accuracy, (ii) strongly generalize when predicting universal algebra's properties, (iii) generate simple explanations that empirically validate existing conjectures, and (iv) identify subgraphs suggesting the formulation of novel conjectures.

        ----

        ## [590] GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph

        **Authors**: *Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html)

        **Abstract**:

        Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms the previous adapter-based methods.

        ----

        ## [591] FaceComposer: A Unified Model for Versatile Facial Content Creation

        **Authors**: *Jiayu Wang, Kang Zhao, Yifeng Ma, Shiwei Zhang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b4caf39e645680f826ae0a9e7ae9402-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b4caf39e645680f826ae0a9e7ae9402-Abstract-Conference.html)

        **Abstract**:

        This work presents FaceComposer, a unified generative model that accomplishes a variety of facial content creation tasks, including text-conditioned face synthesis, text-guided face editing, face animation etc. Based on the latent diffusion framework, FaceComposer follows the paradigm of compositional generation and employs diverse face-specific conditions, e.g., Identity Feature and Projected Normalized Coordinate Code, to release the model creativity at all possible. To support text control and animation, we clean up some existing face image datasets and collect around 500 hours of talking-face videos, forming a high-quality large-scale multi-modal face database. A temporal self-attention module is incorporated into the U-Net structure, which allows learning the denoising process on the mixture of images and videos. Extensive experiments suggest that our approach not only achieves comparable or even better performance than state-of-the-arts on each single task, but also facilitates some combined tasks with one-time forward, demonstrating its potential in serving as a foundation generative model in face domain. We further develop an interface such that users can enjoy our one-step service to create, edit, and animate their own characters. Code, dataset, model, and interface will be made publicly available.

        ----

        ## [592] A Unified Solution for Privacy and Communication Efficiency in Vertical Federated Learning

        **Authors**: *Ganyu Wang, Bin Gu, Qingsong Zhang, Xiang Li, Boyu Wang, Charles X. Ling*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b5af479527167d4af78847a9b9b645f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b5af479527167d4af78847a9b9b645f-Abstract-Conference.html)

        **Abstract**:

        Vertical Federated Learning (VFL) is a collaborative machine learning paradigm that enables multiple participants to jointly train a model on their private data without sharing it.To make VFL practical, privacy security and communication efficiency should both be satisfied. Recent research has shown that Zero-Order Optimization (ZOO) in VFL can effectively conceal the internal information of the model without adding costly privacy protective add-ons, making it a promising approach for privacy and efficiency.However, there are still two key problems that have yet to be resolved. First, the convergence rate of ZOO-based VFL is significantly slower compared to gradient-based VFL, resulting in low efficiency in model training and more communication round, which hinders its application on large neural networks. Second, although ZOO-based VFL has demonstrated resistance to state-of-the-art (SOTA) attacks, its privacy guarantee lacks a theoretical explanation.To address these challenges, we propose a novel cascaded hybrid optimization approach that employs a zeroth-order (ZO) gradient on the most critical output layer of the clients, with other parts utilizing the first-order (FO) gradient. This approach preserves the privacy protection of ZOO while significantly enhancing convergence.Moreover, we theoretically prove that applying ZOO to the VFL is equivalent to adding Gaussian Mechanism to the gradient information, which offers an implicit differential privacy guarantee. Experimental results demonstrate that our proposed framework achieves similar utility as the Gaussian mechanism under the same privacy budget, while also having significantly lower communication costs compared with SOTA communication-efficient VFL frameworks.

        ----

        ## [593] Optimization and Bayes: A Trade-off for Overparameterized Neural Networks

        **Authors**: *Zhengmian Hu, Heng Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b950a297fc888c95bfeb587ef000d70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b950a297fc888c95bfeb587ef000d70-Abstract-Conference.html)

        **Abstract**:

        This paper proposes a novel algorithm, Transformative Bayesian Learning (TansBL), which bridges the gap between empirical risk minimization (ERM) and Bayesian learning for neural networks. We compare ERM, which uses gradient descent to optimize, and Bayesian learning with importance sampling for their generalization and computational complexity. We derive the first algorithm-dependent PAC-Bayesian generalization bound for infinitely wide networks based on an exact KL divergence between the trained posterior distribution obtained by infinitesimal step size gradient descent and a Gaussian prior. Moreover, we show how to transform gradient-based optimization into importance sampling by incorporating a weight. While Bayesian learning has better generalization, it suffers from low sampling efficiency. Optimization methods, on the other hand, have good sampling efficiency but poor generalization. Our proposed algorithm TansBL enables a trade-off between generalization and sampling efficiency.

        ----

        ## [594] Understanding Social Reasoning in Language Models with Language Models

        **Authors**: *Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, Noah D. Goodman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2b9efb085d3829a2aadffab63ba206de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2b9efb085d3829a2aadffab63ba206de-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        As Large Language Models (LLMs) become increasingly integrated into our everyday lives, understanding their ability to comprehend human mental states becomes critical for ensuring effective interactions. However, despite the recent attempts to assess the Theory-of-Mind (ToM) reasoning capabilities of LLMs, the degree to which these models can align with human ToM remains a nuanced topic of exploration. This is primarily due to two distinct challenges: (1) the presence of inconsistent results from previous evaluations, and (2) concerns surrounding the validity of existing evaluation methodologies. To address these challenges, we present a novel framework for procedurally generating evaluations with LLMs by populating causal templates. Using our framework, we create a new social reasoning benchmark (BigToM) for LLMs which consists of 25 controls and 5,000 model-written evaluations. We find that human participants rate the quality of our benchmark higher than previous crowd-sourced evaluations and comparable to expert-written evaluations. Using BigToM, we evaluate the social reasoning capabilities of a variety of LLMs and compare model performances with human performance. Our results suggest that GPT4 has ToM capabilities that mirror human inference patterns, though less reliable, while other LLMs struggle.

        ----

        ## [595] Reproducibility in Multiple Instance Learning: A Case For Algorithmic Unit Tests

        **Authors**: *Edward Raff, James Holt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2bab8865fa4511e445767e3750b2b5ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2bab8865fa4511e445767e3750b2b5ac-Abstract-Conference.html)

        **Abstract**:

        Multiple Instance Learning (MIL) is a sub-domain of classification problems with positive and negative labels and a "bag" of inputs, where the label is positive if and only if a positive element is contained within the bag, and otherwise is negative. Training in this context requires associating the bag-wide label to instance-level information, and implicitly contains a causal assumption and asymmetry to the task (i.e., you can't swap the labels without changing the semantics). MIL problems occur in healthcare (one malignant cell indicates cancer), cyber security (one malicious executable makes an infected computer), and many other tasks. In this work, we examine five of the most prominent deep-MIL models and find that none of them respects the standard MIL assumption. They are able to learn anti-correlated instances, i.e., defaulting to "positive" labels until seeing a negative counter-example, which should not be possible for a correct MIL model. We suspect that enhancements and other works derived from these models will share the same issue. In any context in which these models are being used, this creates the potential for learning incorrect models, which creates risk of operational failure.  We identify and demonstrate this problem via a proposed ``algorithmic unit test'', where we create synthetic datasets that can be solved by a MIL respecting model, and which clearly reveal learning that violates MIL assumptions. The five evaluated methods each fail one or more of these tests. This provides a model-agnostic way to identify violations of modeling assumptions, which we hope will be useful for future development and evaluation of MIL models.

        ----

        ## [596] Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference

        **Authors**: *Basile Confavreux, Poornima Ramesh, Pedro J. Gonçalves, Jakob H. Macke, Tim P. Vogels*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html)

        **Abstract**:

        There is substantial experimental evidence that learning and memory-related behaviours rely on local synaptic changes, but the search for distinct plasticity rules has been driven by human intuition, with limited success for multiple, co-active plasticity rules in biological networks. More recently, automated meta-learning approaches have been used in simplified settings, such as rate networks and small feed-forward spiking networks. Here, we develop a simulation-based inference (SBI) method for sequentially filtering plasticity rules through an increasingly fine mesh of constraints that can be modified on-the-fly. This method, filter SBI, allows us to infer entire families of complex and co-active plasticity rules in spiking networks. We first consider flexibly parameterized doublet (Hebbian) rules, and find that the set of inferred rules contains solutions that extend and refine -and also reject- predictions from mean-field theory. Next, we expand the search space of plasticity rules by modelling them as multi-layer perceptrons that combine several plasticity-relevant factors, such as weight, voltage, triplets and co-dependency. Out of the millions of possible rules, we identify thousands of unique rule combinations that satisfy biological constraints like plausible activity and weight dynamics. The resulting rules can be used as a starting point for further investigations into specific network computations, and already suggest refinements and predictions for classical experimental approaches on plasticity. This flexible approach for principled exploration of complex plasticity rules in large recurrent spiking networks presents the most advanced search tool to date for enabling robust predictions and deep insights into the plasticity mechanisms underlying brain function.

        ----

        ## [597] Joint Training of Deep Ensembles Fails Due to Learner Collusion

        **Authors**: *Alan Jeffares, Tennison Liu, Jonathan Crabbé, Mihaela van der Schaar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html)

        **Abstract**:

        Ensembles of machine learning models have been well established as a powerful method of improving performance over a single model. Traditionally, ensembling algorithms train their base learners independently or sequentially with the goal of optimizing their joint performance. In the case of deep ensembles of neural networks, we are provided with the opportunity to directly optimize the true objective: the joint performance of the ensemble as a whole. Surprisingly, however, directly minimizing the loss of the ensemble appears to rarely be applied in practice. Instead, most previous research trains individual models independently with ensembling performed post hoc. In this work, we show that this is for good reason - joint optimization of ensemble loss results in degenerate behavior. We approach this problem by decomposing the ensemble objective into the strength of the base learners and the diversity between them. We discover that joint optimization results in a phenomenon in which base learners collude to artificially inflate their apparent diversity. This pseudo-diversity fails to generalize beyond the training data, causing a larger generalization gap. We proceed to comprehensively demonstrate the practical implications of this effect on a range of standard machine learning tasks and architectures by smoothly interpolating between independent training and joint optimization.

        ----

        ## [598] Flexible Attention-Based Multi-Policy Fusion for Efficient Deep Reinforcement Learning

        **Authors**: *Zih-Yun Chiu, Yi-Lin Tuan, William Yang Wang, Michael C. Yip*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c23b3c72127e15fedc276722faee927-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c23b3c72127e15fedc276722faee927-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) agents have long sought to approach the efficiency of human learning. Humans are great observers who can learn by aggregating external knowledge from various sources, including observations from others' policies of attempting a task. Prior studies in RL have incorporated external knowledge policies to help agents improve sample efficiency. However, it remains non-trivial to perform arbitrary combinations and replacements of those policies, an essential feature for generalization and transferability. In this work, we present Knowledge-Grounded RL (KGRL), an RL paradigm fusing multiple knowledge policies and aiming for human-like efficiency and flexibility. We propose a new actor architecture for KGRL, Knowledge-Inclusive Attention Network (KIAN), which allows free knowledge rearrangement due to embedding-based attentive action prediction. KIAN also addresses entropy imbalance, a problem arising in maximum entropy KGRL that hinders an agent from efficiently exploring the environment, through a new design of policy distributions. The experimental results demonstrate that KIAN outperforms alternative methods incorporating external knowledge policies and achieves efficient and flexible learning. Our implementation is available at https://github.com/Pascalson/KGRL.git .

        ----

        ## [599] Balanced Training for Sparse GANs

        **Authors**: *Yite Wang, Jing Wu, Naira Hovakimyan, Ruoyu Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c28efa5a86dca4b603a36c08f49f240-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c28efa5a86dca4b603a36c08f49f240-Abstract-Conference.html)

        **Abstract**:

        Over the past few years, there has been growing interest in developing larger and deeper neural networks, including deep generative models like generative adversarial networks (GANs). However, GANs typically come with high computational complexity,  leading researchers to explore methods for reducing the training and inference costs.  One such approach gaining popularity in supervised learning is dynamic sparse training (DST), which maintains good performance while enjoying excellent training efficiency.  Despite its potential benefits, applying DST to GANs presents challenges due to the adversarial nature of the training process. In this paper, we propose a novel metric called the balance ratio (BR) to study the balance between the sparse generator and discriminator. We also introduce a new method called balanced dynamic sparse training (ADAPT), which seeks to control the BR during GAN training to achieve a good trade-off between performance and computational cost. Our proposed method shows promising results on multiple datasets, demonstrating its effectiveness.

        ----

        

[Go to the previous page](NIPS-2023-list02.md)

[Go to the next page](NIPS-2023-list04.md)

[Go to the catalog section](README.md)