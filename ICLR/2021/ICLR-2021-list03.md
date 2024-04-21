## [400] Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth

**Authors**: *Thao Nguyen, Maithra Raghu, Simon Kornblith*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KJNcAkY8tY4](https://openreview.net/forum?id=KJNcAkY8tY4)

**Abstract**:

A key factor in the success of deep neural networks is the ability to scale models to improve performance by varying the architecture depth and width. This simple property of neural network design has resulted in highly effective architectures for a variety of tasks. Nevertheless, there is limited understanding of effects of depth and width on the learned representations. In this paper, we study this fundamental question. We begin by investigating how varying depth and width affects model hidden representations, finding a characteristic block structure in the hidden representations of larger capacity (wider or deeper) models. We demonstrate that this block structure arises when model capacity is large relative to the size of the training set, and is indicative of the underlying layers preserving and propagating the dominant principal component of their representations. This discovery has important ramifications for features learned by different models, namely, representations outside the block structure are often similar across architectures with varying widths and depths, but the block structure is unique to each model. We analyze the output predictions of different model architectures, finding that even when the overall accuracy is similar, wide and deep models exhibit distinctive error patterns and variations across classes.

----

## [401] Learning to Set Waypoints for Audio-Visual Navigation

**Authors**: *Changan Chen, Sagnik Majumder, Ziad Al-Halah, Ruohan Gao, Santhosh Kumar Ramakrishnan, Kristen Grauman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cR91FAodFMe](https://openreview.net/forum?id=cR91FAodFMe)

**Abstract**:

In audio-visual navigation, an agent intelligently travels through a complex, unmapped 3D environment using both sights and sounds to find a sound source (e.g., a phone ringing in another room). Existing models learn to act at a fixed granularity of agent motion and rely on simple recurrent aggregations of the audio observations. We introduce a reinforcement learning approach to audio-visual navigation with two key novel elements: 1) waypoints that are dynamically set and learned end-to-end within the navigation policy, and 2) an acoustic memory that provides a structured, spatially grounded record of what the agent has heard as it moves. Both new ideas capitalize on the synergy of audio and visual data for revealing the geometry of an unmapped space. We demonstrate our approach on two challenging datasets of real-world 3D scenes, Replica and Matterport3D. Our model improves the state of the art by a substantial margin, and our experiments reveal that learning the links between sights, sounds, and space is essential for audio-visual navigation.

----

## [402] Semi-supervised Keypoint Localization

**Authors**: *Olga Moskvyak, Frédéric Maire, Feras Dayoub, Mahsa Baktashmotlagh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yFJ67zTeI2](https://openreview.net/forum?id=yFJ67zTeI2)

**Abstract**:

Knowledge about the locations of keypoints of an object in an image can assist in fine-grained classification and identification tasks, particularly for the case of objects that exhibit large variations in poses that greatly influence their visual appearance, such as wild animals. However, supervised training of a keypoint detection network requires annotating a large image dataset for each animal species, which is a labor-intensive task. To reduce the need for labeled data, we propose to learn simultaneously keypoint heatmaps and pose invariant keypoint representations in a semi-supervised manner using a small set of labeled images along with a larger set of unlabeled images. Keypoint representations are learnt with a semantic keypoint consistency constraint that forces the keypoint detection network to learn similar features for the same keypoint across the dataset. Pose invariance is achieved by making keypoint representations for the image and its augmented copies closer together in feature space. Our semi-supervised approach significantly outperforms previous methods on several benchmarks for human and animal body landmark localization.

----

## [403] Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective

**Authors**: *Wuyang Chen, Xinyu Gong, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Cnon5ezMHtu](https://openreview.net/forum?id=Cnon5ezMHtu)

**Abstract**:

Neural Architecture Search (NAS) has been explosively studied to automate the discovery of top-performer neural networks. Current works require heavy training of supernet or intensive architecture evaluations, thus suffering from heavy resource consumption and often incurring search bias due to truncated training or approximations. Can we select the best neural architectures without involving any training and eliminate a drastic portion of the search cost? 

We provide an affirmative answer, by proposing a novel framework called \textit{training-free neural architecture search} ($\textbf{TE-NAS}$). TE-NAS ranks architectures by analyzing the spectrum of the neural tangent kernel (NTK), and the number of linear regions in the input space. Both are motivated by recent theory advances in deep networks, and can be computed without any training. We show that: (1) these two measurements imply the $\textit{trainability}$ and $\textit{expressivity}$ of a neural network; and (2) they strongly correlate with the network's actual test accuracy. Further on, we design a pruning-based NAS mechanism to achieve a more flexible and superior trade-off between the trainability and expressivity during the search. In NAS-Bench-201 and DARTS search spaces, TE-NAS completes high-quality search but only costs $\textbf{0.5}$ and $\textbf{4}$ GPU hours with one 1080Ti on CIFAR-10 and ImageNet, respectively. We hope our work to inspire more attempts in bridging between the theoretic findings of deep networks and practical impacts in real NAS applications.

----

## [404] Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers

**Authors**: *Benjamin Eysenbach, Shreyas Chaudhari, Swapnil Asawa, Sergey Levine, Ruslan Salakhutdinov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eqBwg3AcIAK](https://openreview.net/forum?id=eqBwg3AcIAK)

**Abstract**:

We propose a simple, practical, and intuitive approach for domain adaptation in reinforcement learning. Our approach stems from the idea that the agent's experience in the source domain should look similar to its experience in the target domain. Building off of a probabilistic view of RL, we achieve this goal by compensating for the difference in dynamics by modifying the reward function. This modified reward function is simple to estimate by learning auxiliary classifiers that distinguish source-domain transitions from target-domain transitions. Intuitively, the agent is penalized for transitions that would indicate that the agent is interacting with the source domain, rather than the target domain. Formally, we prove that applying our method in the source domain is guaranteed to obtain a near-optimal policy for the target domain, provided that the source and target domains satisfy a lightweight assumption. Our approach is applicable to domains with continuous states and actions and does not require learning an explicit model of the dynamics. On discrete and continuous control tasks, we illustrate the mechanics of our approach and demonstrate its scalability to high-dimensional~tasks.

----

## [405] Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning

**Authors**: *Wonyong Jeong, Jaehong Yoon, Eunho Yang, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ce6CFXBh30h](https://openreview.net/forum?id=ce6CFXBh30h)

**Abstract**:

While existing federated learning approaches mostly require that clients have fully-labeled data to train on, in realistic settings, data obtained at the client-side often comes without any accompanying labels. Such deficiency of labels may result from either high labeling cost, or difficulty of annotation due to the requirement of expert knowledge. Thus the private data at each client may be either partly labeled, or completely unlabeled with labeled data being available only at the server, which leads us to a new practical federated learning problem, namely Federated Semi-Supervised Learning (FSSL). In this work, we study two essential scenarios of FSSL based on the location of the labeled data. The first scenario considers a conventional case where clients have both labeled and unlabeled data (labels-at-client), and the second scenario considers a more challenging case, where the labeled data is only available at the server (labels-at-server). We then propose a novel method to tackle the problems, which we refer to as Federated Matching (FedMatch). FedMatch improves upon naive combinations of federated learning and semi-supervised learning approaches with a new inter-client consistency loss and decomposition of the parameters for disjoint learning on labeled and unlabeled data. Through extensive experimental validation of our method in the two different scenarios, we show that our method outperforms both local semi-supervised learning and baselines which naively combine federated learning with semi-supervised learning.

----

## [406] Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates

**Authors**: *Zengyi Qin, Kaiqing Zhang, Yuxiao Chen, Jingkai Chen, Chuchu Fan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=P6_q1BRxY8Q](https://openreview.net/forum?id=P6_q1BRxY8Q)

**Abstract**:

We study the multi-agent safe control problem where agents should avoid collisions to static obstacles and collisions with each other while reaching their goals. Our core idea is to learn the multi-agent control policy jointly with  learning the control barrier functions as safety certificates. We propose a new joint-learning framework that can be implemented in a decentralized fashion, which can adapt to an arbitrarily large number of agents. Building upon this framework, we further improve the scalability by  incorporating neural network architectures  that are invariant to the quantity and permutation of neighboring agents. In addition, we propose a new spontaneous policy refinement method to further enforce the certificate condition during testing. We provide extensive experiments to demonstrate that our method significantly outperforms other leading multi-agent control approaches in terms of maintaining safety and completing original tasks. Our approach also shows substantial generalization capability in that the control policy can be trained with 8 agents in one scenario, while being used on other scenarios with up to 1024 agents in complex multi-agent environments and dynamics. Videos and source code can be found at https://realm.mit.edu/blog/learning-safe-multi-agent-control-decentralized-neural-barrier-certificates.

----

## [407] Direction Matters: On the Implicit Bias of Stochastic Gradient Descent with Moderate Learning Rate

**Authors**: *Jingfeng Wu, Difan Zou, Vladimir Braverman, Quanquan Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3X64RLgzY6O](https://openreview.net/forum?id=3X64RLgzY6O)

**Abstract**:

Understanding the algorithmic bias of stochastic gradient descent (SGD) is one of the key challenges in modern machine learning and deep learning theory. Most of the existing works, however, focus on very small or even infinitesimal learning rate regime, and fail to cover practical scenarios where the learning rate is moderate and annealing. In this paper, we make an initial attempt to characterize the particular regularization effect of SGD in the moderate learning rate regime by studying its behavior for optimizing an overparameterized linear regression problem. In this case, SGD and GD are known to converge to the unique minimum-norm solution; however, with the moderate and annealing learning rate, we show that they exhibit different directional bias: SGD converges along the large eigenvalue directions of the data matrix, while GD goes after the small eigenvalue directions. Furthermore, we show that such directional bias does matter when early stopping is adopted, where the SGD output is nearly optimal but the GD output is suboptimal. Finally, our theory explains several folk arts in practice used for SGD hyperparameter tuning, such as (1) linearly scaling the initial learning rate with batch size; and (2) overrunning SGD with high learning rate even when the loss stops decreasing.

----

## [408] Fast And Slow Learning Of Recurrent Independent Mechanisms

**Authors**: *Kanika Madan, Nan Rosemary Ke, Anirudh Goyal, Bernhard Schölkopf, Yoshua Bengio*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Lc28QAB4ypz](https://openreview.net/forum?id=Lc28QAB4ypz)

**Abstract**:

Decomposing knowledge into interchangeable pieces promises a generalization advantage when there are changes in distribution. A learning agent interacting with its environment is likely to be faced with situations requiring novel combinations of existing pieces of knowledge. We hypothesize that such a decomposition of knowledge is particularly relevant for being able to generalize in a systematic way to out-of-distribution changes. To study these ideas, we propose a particular training framework in which we assume that the pieces of knowledge an agent needs and its reward function are stationary and can be re-used across tasks. An attention mechanism dynamically selects which modules can be adapted to the current task, and the parameters of the \textit{selected} modules are allowed to change quickly as the learner is confronted with variations in what it experiences, while the parameters of the attention mechanisms act as stable, slowly changing, meta-parameters. We focus on pieces of knowledge captured by an ensemble of  modules sparsely communicating with each other via a bottleneck of attention. We find that meta-learning the  modular aspects of the proposed system greatly helps in achieving faster adaptation in a reinforcement learning setup involving navigation in a partially observed grid world with image-level input.  We also find that reversing the role of parameters and meta-parameters does not work nearly as well, suggesting a particular role for fast adaptation of the dynamically selected modules.

----

## [409] Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples

**Authors**: *Ziang Yan, Yiwen Guo, Jian Liang, Changshui Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pzpytjk3Xb2](https://openreview.net/forum?id=pzpytjk3Xb2)

**Abstract**:

To craft black-box adversarial examples, adversaries need to query the victim model and take proper advantage of its feedback. Existing black-box attacks generally suffer from high query complexity, especially when only the top-1 decision (i.e., the hard-label prediction) of the victim model is available.  In this paper, we propose a novel hard-label black-box attack named Policy-Driven Attack, to reduce the query complexity. Our core idea is to learn promising search directions of the adversarial examples using a well-designed policy network in a novel reinforcement learning formulation, in which the queries become more sensible. Experimental results demonstrate that our method can significantly reduce the query complexity in comparison with existing state-of-the-art hard-label black-box attacks on various image classification benchmark datasets. Code and models for reproducing our results are available at https://github.com/ZiangYan/pda.pytorch

----

## [410] A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks

**Authors**: *Nikunj Saunshi, Sadhika Malladi, Sanjeev Arora*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vVjIW3sEc1s](https://openreview.net/forum?id=vVjIW3sEc1s)

**Abstract**:

Autoregressive language models, pretrained using large text corpora to do well on next word prediction, have been successful at solving many downstream tasks, even with zero-shot usage. However, there is little theoretical understanding of this success. This paper initiates a mathematical study of this phenomenon for the downstream task of text classification by considering the following questions: (1) What is the intuitive connection between the pretraining task of next word prediction and text classification? (2) How can we mathematically formalize this connection and quantify the benefit of language modeling? For (1), we hypothesize, and verify empirically, that classification tasks of interest can be reformulated as sentence completion tasks, thus making language modeling a meaningful pretraining task. With a mathematical formalization of this hypothesis, we make progress towards (2) and show that language models that are $\epsilon$-optimal in cross-entropy (log-perplexity) learn features that can linearly solve such classification tasks with $\mathcal{O}(\sqrt{\epsilon})$ error, thus demonstrating that doing well on language modeling can be beneficial for downstream tasks. We experimentally verify various assumptions and theoretical findings, and also use insights from the analysis to design a new objective function that performs well on some classification tasks.

----

## [411] Representation Learning for Sequence Data with Deep Autoencoding Predictive Components

**Authors**: *Junwen Bai, Weiran Wang, Yingbo Zhou, Caiming Xiong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Naqw7EHIfrv](https://openreview.net/forum?id=Naqw7EHIfrv)

**Abstract**:

We propose Deep Autoencoding Predictive Components (DAPC) -- a self-supervised representation learning method for sequence data, based on the intuition that useful representations of sequence data should exhibit a simple structure in the latent space. We encourage this latent structure by maximizing an estimate of \emph{predictive information} of latent feature sequences, which is the mutual information between the past and future windows at each time step. In contrast to the mutual information lower bound commonly used by contrastive learning, the estimate of predictive information we adopt is exact under a Gaussian assumption. Additionally, it can be computed without negative sampling. To reduce the degeneracy of the latent space extracted by powerful encoders and keep useful information from the inputs, we regularize predictive information learning with a challenging masked reconstruction loss. We demonstrate that our method recovers the latent space of noisy dynamical systems, extracts predictive features for forecasting tasks, and improves automatic speech recognition when used to pretrain the encoder on large amounts of unlabeled data.

----

## [412] A unifying view on implicit bias in training linear neural networks

**Authors**: *Chulhee Yun, Shankar Krishnan, Hossein Mobahi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZsZM-4iMQkH](https://openreview.net/forum?id=ZsZM-4iMQkH)

**Abstract**:

We study the implicit bias of gradient flow (i.e., gradient descent with infinitesimal step size) on linear neural network training. We propose a tensor formulation of neural networks that includes fully-connected, diagonal, and convolutional networks as special cases, and investigate the linear version of the formulation called linear tensor networks. With this formulation, we can characterize the convergence direction of the network parameters as singular vectors of a tensor defined by the network. For $L$-layer linear tensor networks that are orthogonally decomposable, we show that gradient flow on separable classification finds a stationary point of the $\ell_{2/L}$ max-margin problem in a "transformed" input space defined by the network. For underdetermined regression, we prove that gradient flow finds a global minimum which minimizes a norm-like function that interpolates between weighted $\ell_1$ and $\ell_2$ norms in the transformed input space. Our theorems subsume existing results in the literature while removing standard convergence assumptions. We also provide experiments that corroborate our analysis.

----

## [413] What Makes Instance Discrimination Good for Transfer Learning?

**Authors**: *Nanxuan Zhao, Zhirong Wu, Rynson W. H. Lau, Stephen Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tC6iW2UUbJf](https://openreview.net/forum?id=tC6iW2UUbJf)

**Abstract**:

Contrastive visual pretraining based on the instance discrimination pretext task has made significant progress. Notably, recent work on unsupervised pretraining has shown to surpass the supervised counterpart for finetuning downstream applications such as object detection and segmentation.   It comes as a surprise that image annotations would be better left unused for transfer learning.  In this work, we investigate the following problems: What makes instance discrimination pretraining good for transfer learning? What knowledge is actually learned and transferred from these models?  From this understanding of instance discrimination, how can we better exploit human annotation labels for pretraining? Our findings are threefold. First, what truly matters for the transfer is low-level and mid-level representations, not high-level representations.  Second, the intra-category invariance enforced by the traditional supervised model weakens transferability by increasing task misalignment. Finally, supervised pretraining can be strengthened by following an exemplar-based approach without explicit constraints among the instances within the same category.

----

## [414] Learning Accurate Entropy Model with Global Reference for Image Compression

**Authors**: *Yichen Qian, Zhiyu Tan, Xiuyu Sun, Ming Lin, Dongyang Li, Zhenhong Sun, Hao Li, Rong Jin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cTbIjyrUVwJ](https://openreview.net/forum?id=cTbIjyrUVwJ)

**Abstract**:

In recent deep image compression neural networks, the entropy model plays a critical role in estimating the prior distribution of deep image encodings. Existing methods combine hyperprior with local context in the entropy estimation function. This greatly limits their performance due to the absence of a global vision. In this work, we propose a novel Global Reference Model for image compression to effectively leverage both the local and the global context information, leading to an enhanced compression rate. The proposed method scans decoded latents and then finds the most relevant latent to assist the distribution estimating of the current latent. A by-product of this work is the innovation of a mean-shifting GDN module that further improves the performance. Experimental results demonstrate that the proposed model outperforms the rate-distortion performance of most of the state-of-the-art methods in the industry.

----

## [415] Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search

**Authors**: *Peidong Liu, Gengwei Zhang, Bochao Wang, Hang Xu, Xiaodan Liang, Yong Jiang, Zhenguo Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5jzlpHvvRk](https://openreview.net/forum?id=5jzlpHvvRk)

**Abstract**:

Designing proper loss functions for vision tasks has been a long-standing research direction to advance the capability of existing models. For object detection, the well-established classification and regression loss functions have been carefully designed by considering diverse learning challenges (e.g. class imbalance, hard negative samples, and scale variances). Inspired by the recent progress in network architecture search, it is interesting to explore the possibility of discovering new loss function formulations via directly searching the primitive operation combinations. So that the learned losses not only fit for diverse object detection challenges to alleviate huge human efforts, but also have better alignment with evaluation metric and good mathematical convergence property. Beyond the previous auto-loss works on face recognition and image classification, our work makes the first attempt to discover new loss functions for the challenging object detection from primitive operation levels and finds the searched losses are insightful. We propose an effective convergence-simulation driven evolutionary search algorithm, called CSE-Autoloss, for speeding up the search progress by regularizing the mathematical rationality of loss candidates via two progressive convergence simulation modules: convergence property verification and model optimization simulation. CSE-Autoloss involves the search space (i.e. 21 mathematical operators, 3 constant-type inputs, and 3 variable-type inputs) that cover a wide range of the possible variants of existing losses and discovers best-searched loss function combination within a short time (around 1.5 wall-clock days with 20x speedup in comparison to the vanilla evolutionary algorithm). We conduct extensive evaluations of loss function search on popular detectors and validate the good generalization capability of searched losses across diverse architectures and various datasets. Our experiments show that the best-discovered loss function combinations outperform default combinations (Cross-entropy/Focal loss for classification and L1 loss for regression) by 1.1% and 0.8% in terms of mAP for two-stage and one-stage detectors on COCO respectively. Our searched losses are available at https://github.com/PerdonLiu/CSE-Autoloss.

----

## [416] Effective Abstract Reasoning with Dual-Contrast Network

**Authors**: *Tao Zhuo, Mohan S. Kankanhalli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ldxlzGYWDmW](https://openreview.net/forum?id=ldxlzGYWDmW)

**Abstract**:

As a step towards improving the abstract reasoning capability of machines, we aim to solve Raven’s Progressive Matrices (RPM) with neural networks, since solving RPM puzzles is highly correlated with human intelligence. Unlike previous methods that use auxiliary annotations or assume hidden rules to produce appropriate feature representation, we only use the ground truth answer of each question for model learning,  aiming for an intelligent agent to have a strong learning capability with a small amount of supervision.  Based on the RPM problem formulation,  the correct answer filled into the missing entry of the third row/column has  to  best  satisfy  the  same  rules  shared  between  the  first  two  rows/columns.Thus  we  design  a  simple  yet  effective  Dual-Contrast  Network  (DCNet)  to  exploit the inherent structure of RPM puzzles.  Specifically, a rule contrast module is  designed  to  compare  the  latent  rules  between  the  filled  row/column  and  the first two rows/columns; a choice contrast module is designed to increase the relative differences between candidate choices.  Experimental results on the RAVEN and  PGM  datasets  show  that  DCNet  outperforms  the  state-of-the-art  methods by a large margin of 5.77%.   Further experiments on few training samples and model generalization also show the effectiveness of DCNet.  Code is available at https://github.com/visiontao/dcnet.

----

## [417] Do not Let Privacy Overbill Utility: Gradient Embedding Perturbation for Private Learning

**Authors**: *Da Yu, Huishuai Zhang, Wei Chen, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7aogOj_VYO0](https://openreview.net/forum?id=7aogOj_VYO0)

**Abstract**:

The privacy leakage of the model about the training data can be bounded in the differential privacy mechanism. However, for meaningful privacy parameters, a differentially private model degrades the utility drastically when the model comprises a large number of trainable parameters.  In this paper, we propose an algorithm  \emph{Gradient Embedding Perturbation (GEP)} towards training differentially private deep models with decent accuracy. Specifically, in each gradient descent step, GEP first projects individual private gradient into a non-sensitive anchor subspace, producing a low-dimensional gradient embedding and a small-norm residual gradient. Then, GEP perturbs the low-dimensional embedding and the residual gradient separately according to the privacy budget. Such a decomposition permits a small perturbation variance, which greatly helps to break the dimensional barrier of private learning. With GEP, we achieve decent accuracy with low computational cost and modest privacy guarantee for deep models.  Especially, with privacy bound $\epsilon=8$, we achieve $74.9\%$ test accuracy on CIFAR10 and $95.1\%$ test accuracy on  SVHN, significantly improving over existing results.

----

## [418] Set Prediction without Imposing Structure as Conditional Density Estimation

**Authors**: *David W. Zhang, Gertjan J. Burghouts, Cees G. M. Snoek*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=04ArenGOz3](https://openreview.net/forum?id=04ArenGOz3)

**Abstract**:

Set prediction is about learning to predict a collection of unordered variables with unknown interrelations. Training such models with set losses imposes the structure of a metric space over sets. We focus on stochastic and underdefined cases, where an incorrectly chosen loss function leads to implausible predictions. Example tasks include conditional point-cloud reconstruction and predicting future states of molecules. In this paper we propose an alternative to training via set losses, by viewing learning as conditional density estimation. Our learning framework fits deep energy-based models and approximates the intractable likelihood with gradient-guided sampling. Furthermore, we propose a stochastically augmented prediction algorithm that enables multiple predictions, reflecting the possible variations in the target set. We empirically demonstrate on a variety of datasets the capability to learn multi-modal densities and produce different plausible predictions. Our approach is competitive with previous set prediction models on standard benchmarks. More importantly, it extends the family of addressable tasks beyond those that have unambiguous predictions.

----

## [419] Clustering-friendly Representation Learning via Instance Discrimination and Feature Decorrelation

**Authors**: *Yaling Tao, Kentaro Takagi, Kouta Nakata*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=e12NDM7wkEY](https://openreview.net/forum?id=e12NDM7wkEY)

**Abstract**:

Clustering is one of the most fundamental tasks in machine learning. Recently, deep clustering has become a major trend in clustering techniques. Representation learning often plays an important role in the effectiveness of deep clustering, and thus can be a principal cause of performance degradation. In this paper, we propose a clustering-friendly representation learning method using instance discrimination and feature decorrelation. Our deep-learning-based representation learning method is motivated by the properties of classical spectral clustering. Instance discrimination learns similarities among data and feature decorrelation removes redundant correlation among features. We utilize an instance discrimination method in which learning individual instance classes leads to learning similarity among instances. Through detailed experiments and examination, we show that the approach can be adapted to learning a latent space for clustering. We design novel softmax-formulated decorrelation constraints for learning. In evaluations of image clustering using CIFAR-10 and ImageNet-10, our method achieves accuracy of 81.5% and 95.4%, respectively. We also show that the softmax-formulated constraints are compatible with various neural networks.

----

## [420] Language-Agnostic Representation Learning of Source Code from Structure and Context

**Authors**: *Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Xh5eMZVONGF](https://openreview.net/forum?id=Xh5eMZVONGF)

**Abstract**:

Source code (Context) and its parsed abstract syntax tree (AST; Structure) are two complementary representations of the same computer program. Traditionally, designers of machine learning models have relied predominantly either on Structure or Context. We propose a new model, which jointly learns on Context and Structure of source code. In contrast to previous approaches, our model uses only language-agnostic features, i.e., source code and features that can be computed directly from the AST. Besides obtaining state-of-the-art on monolingual code summarization on all five programming languages considered in this work, we propose the first multilingual code summarization model. We show that jointly training on non-parallel data from multiple programming languages improves results on all individual languages, where the strongest gains are on low-resource languages. Remarkably, multilingual training only from Context does not lead to the same improvements, highlighting the benefits of combining Structure and Context for representation learning on code.

----

## [421] Training GANs with Stronger Augmentations via Contrastive Discriminator

**Authors**: *Jongheon Jeong, Jinwoo Shin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eo6U4CAwVmg](https://openreview.net/forum?id=eo6U4CAwVmg)

**Abstract**:

Recent works in Generative Adversarial Networks (GANs) are actively revisiting various data augmentation techniques as an effective way to prevent discriminator overfitting. It is still unclear, however, that which augmentations could actually improve GANs, and in particular, how to apply a wider range of augmentations in training. In this paper, we propose a novel way to address these questions by incorporating a recent contrastive representation learning scheme into the GAN discriminator, coined ContraD. This "fusion" enables the discriminators to work with much stronger augmentations without increasing their training instability, thereby preventing the discriminator overfitting issue in GANs more effectively. Even better, we observe that the contrastive learning itself also benefits from our GAN training, i.e., by maintaining discriminative features between real and fake samples, suggesting a strong coherence between the two worlds: good contrastive representations are also good for GAN discriminators, and vice versa. Our experimental results show that GANs with ContraD consistently improve FID and IS compared to other recent techniques incorporating data augmentations, still maintaining highly discriminative features in the discriminator in terms of the linear evaluation. Finally, as a byproduct, we also show that our GANs trained in an unsupervised manner (without labels) can induce many conditional generative models via a simple latent sampling, leveraging the learned features of ContraD. Code is available at https://github.com/jh-jeong/ContraD.

----

## [422] Influence Functions in Deep Learning Are Fragile

**Authors**: *Samyadeep Basu, Phillip Pope, Soheil Feizi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xHKVVHGDOEk](https://openreview.net/forum?id=xHKVVHGDOEk)

**Abstract**:

Influence functions approximate the effect of training samples in test-time predictions and have a wide variety of applications in machine learning interpretability and uncertainty estimation. A commonly-used (first-order) influence function can be implemented efficiently as a post-hoc method requiring access only to the gradients and Hessian of the model. For linear models, influence functions are well-defined due to the convexity of the underlying loss function and are generally accurate even across difficult settings where model changes are fairly large such as estimating group influences. Influence functions, however, are not well-understood in the context of deep learning with non-convex loss functions.  In this paper, we provide a comprehensive and large-scale empirical study of successes and failures of influence functions in neural network models trained on datasets such as Iris, MNIST, CIFAR-10 and ImageNet. Through our extensive experiments, we show that the network architecture, its depth and width, as well as the extent of model parameterization and regularization techniques have strong effects in the accuracy of influence functions. In particular, we find that (i) influence estimates are fairly accurate for shallow networks, while for deeper networks the estimates are often erroneous; (ii) for certain network architectures and datasets, training with weight-decay regularization is important to get high-quality influence estimates; and (iii) the accuracy of influence estimates can vary significantly depending on the examined test points. These results suggest that in general influence functions in deep learning are fragile and call for developing improved influence estimation methods to mitigate these issues in non-convex setups.

----

## [423] Separation and Concentration in Deep Networks

**Authors**: *John Zarka, Florentin Guth, Stéphane Mallat*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8HhkbjrWLdE](https://openreview.net/forum?id=8HhkbjrWLdE)

**Abstract**:

Numerical experiments demonstrate that deep neural network classifiers progressively separate class distributions around their mean, achieving linear separability on the training set, and increasing the Fisher discriminant ratio. We explain this mechanism with two types of operators. We prove that a rectifier without biases applied to sign-invariant tight frames can separate class means and increase Fisher ratios. On the opposite, a soft-thresholding on tight frames can reduce within-class variabilities while preserving class means. Variance reduction bounds are proved for Gaussian mixture models. For image classification, we show that separation of class means can be achieved with rectified wavelet tight frames that are not learned. It defines a scattering transform. Learning  $1 \times 1$ convolutional tight frames along scattering channels and applying a soft-thresholding reduces within-class variabilities. The resulting scattering network reaches the classification accuracy of ResNet-18 on CIFAR-10 and ImageNet, with fewer layers and no learned biases.

----

## [424] Colorization Transformer

**Authors**: *Manoj Kumar, Dirk Weissenborn, Nal Kalchbrenner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5NA1PinlGFu](https://openreview.net/forum?id=5NA1PinlGFu)

**Abstract**:

We present the Colorization Transformer, a novel approach for diverse high fidelity image colorization based on self-attention. Given a grayscale image, the colorization proceeds in three steps. We first use a conditional autoregressive transformer to produce a low resolution coarse coloring of the grayscale image. Our architecture adopts conditional transformer layers to effectively condition grayscale input. Two subsequent fully parallel networks upsample the coarse colored low resolution image into a finely colored high resolution image. Sampling from the Colorization Transformer produces diverse colorings whose fidelity outperforms the previous state-of-the-art on colorising ImageNet based on FID results and based on a human evaluation in a Mechanical Turk test. Remarkably, in  more than 60\% of cases human evaluators prefer the highest rated among three generated colorings over the ground truth. The code and pre-trained checkpoints for Colorization Transformer are publicly available  at https://github.com/google-research/google-research/tree/master/coltran

----

## [425] Autoregressive Dynamics Models for Offline Policy Evaluation and Optimization

**Authors**: *Michael R. Zhang, Thomas Paine, Ofir Nachum, Cosmin Paduraru, George Tucker, Ziyu Wang, Mohammad Norouzi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kmqjgSNXby](https://openreview.net/forum?id=kmqjgSNXby)

**Abstract**:

Standard dynamics models for continuous control make use of feedforward computation to predict the conditional distribution of next state and reward given current state and action using a multivariate Gaussian with a diagonal covariance structure. This modeling choice assumes that different dimensions of the next state and reward are conditionally independent given the current state and action and may be driven by the fact that fully observable physics-based simulation environments entail deterministic transition dynamics. In this paper, we challenge this conditional independence assumption and propose a family of expressive autoregressive dynamics models that generate different dimensions of the next state and reward sequentially conditioned on previous dimensions. We demonstrate that autoregressive dynamics models indeed outperform standard feedforward models in log-likelihood on heldout transitions. Furthermore, we compare different model-based and model-free off-policy evaluation (OPE) methods on RL Unplugged, a suite of offline MuJoCo datasets, and find that autoregressive dynamics models consistently outperform all baselines, achieving a new state-of-the-art. Finally, we show that autoregressive dynamics models are useful for offline policy optimization by serving as a way to enrich the replay buffer through data augmentation and improving performance using model-based planning.

----

## [426] FedBN: Federated Learning on Non-IID Features via Local Batch Normalization

**Authors**: *Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6YEQUn0QICG](https://openreview.net/forum?id=6YEQUn0QICG)

**Abstract**:

The emerging paradigm of federated learning (FL) strives to enable collaborative training of deep models on the network edge without centrally aggregating raw data and hence improving data privacy. In most cases, the assumption of independent and identically distributed samples across local clients does not hold for federated learning setups. Under this setting, neural network training performance may vary significantly according to the data distribution and even hurt training convergence. Most of the previous work has focused on a difference in the distribution of labels or client shifts. Unlike those settings, we address an important problem of FL, e.g., different scanners/sensors in medical imaging, different scenery distribution in autonomous driving (highway vs. city), where local clients store examples with different distributions compared to other clients, which we denote as feature shift non-iid. In this work, we propose an effective method that uses local batch normalization to alleviate the feature shift before averaging models. The resulting scheme, called FedBN, outperforms both classical FedAvg, as well as the state-of-the-art for non-iid data (FedProx) on our extensive experiments. These empirical results are supported by a convergence analysis that shows in a simplified setting that FedBN has a faster convergence rate than FedAvg. Code is available at https://github.com/med-air/FedBN.

----

## [427] Learning Robust State Abstractions for Hidden-Parameter Block MDPs

**Authors**: *Amy Zhang, Shagun Sodhani, Khimya Khetarpal, Joelle Pineau*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fmOOI2a3tQP](https://openreview.net/forum?id=fmOOI2a3tQP)

**Abstract**:

Many control tasks exhibit similar dynamics that can be modeled as having common latent structure. Hidden-Parameter Markov Decision Processes (HiP-MDPs) explicitly model this structure to improve sample efficiency in multi-task settings.
However, this setting makes strong assumptions on the observability of the state that limit its application in real-world scenarios with rich observation spaces.  In this work, we leverage ideas of common structure from the HiP-MDP setting, and extend it to enable robust state abstractions inspired by Block MDPs. We  derive instantiations of this new framework for  both multi-task reinforcement learning (MTRL) and  meta-reinforcement learning (Meta-RL) settings. Further, we provide transfer and generalization bounds based on task and state similarity, along with sample complexity bounds that depend on the aggregate number of samples across tasks, rather than the number of tasks, a significant improvement over prior work. To further demonstrate efficacy of the proposed method, we empirically compare and show improvement over multi-task and meta-reinforcement learning baselines.

----

## [428] Meta-Learning with Neural Tangent Kernels

**Authors**: *Yufan Zhou, Zhenyi Wang, Jiayi Xian, Changyou Chen, Jinhui Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ti87Pv5Oc8](https://openreview.net/forum?id=Ti87Pv5Oc8)

**Abstract**:

Model Agnostic Meta-Learning (MAML) has emerged as a standard framework for meta-learning, where a meta-model is learned with the ability of fast adapting to new tasks. However, as a double-looped optimization problem, MAML needs to differentiate through the whole inner-loop optimization path for every outer-loop training step, which may lead to both computational inefficiency and sub-optimal solutions. In this paper, we generalize MAML to allow meta-learning to be defined in function spaces, and propose the first meta-learning paradigm in the Reproducing Kernel Hilbert Space (RKHS) induced by the meta-model's Neural Tangent Kernel (NTK). Within this paradigm, we introduce two meta-learning algorithms in the RKHS, which no longer need a sub-optimal iterative inner-loop adaptation as in the MAML framework. We achieve this goal by 1) replacing the adaptation with a fast-adaptive regularizer in the RKHS; and 2) solving the adaptation analytically based on the NTK theory. Extensive experimental studies demonstrate advantages of our paradigm in both efficiency and quality of solutions compared to related meta-learning algorithms. Another interesting feature of our proposed methods is that they are demonstrated to be more robust to adversarial attacks and out-of-distribution adaptation than popular baselines, as demonstrated in our experiments.

----

## [429] Continual learning in recurrent neural networks

**Authors**: *Benjamin Ehret, Christian Henning, Maria R. Cervera, Alexander Meulemans, Johannes von Oswald, Benjamin F. Grewe*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8xeBUgD8u9](https://openreview.net/forum?id=8xeBUgD8u9)

**Abstract**:

While a diverse collection of continual learning (CL) methods has been proposed to prevent catastrophic forgetting, a thorough investigation of their effectiveness for processing sequential data with recurrent neural networks (RNNs) is lacking. Here, we provide the first comprehensive evaluation of established CL methods on a variety of sequential data benchmarks. Specifically, we shed light on the particularities that arise when applying weight-importance methods, such as elastic weight consolidation, to RNNs. In contrast to feedforward networks, RNNs iteratively reuse a shared set of weights and require working memory to process input samples. We show that the performance of weight-importance methods is not directly affected by the length of the processed sequences, but rather by high working memory requirements, which lead to an increased need for stability at the cost of decreased plasticity for learning subsequent tasks. We additionally provide theoretical arguments supporting this interpretation by studying linear RNNs. Our study shows that established CL methods can be successfully ported to the recurrent case, and that a recent regularization approach based on hypernetworks outperforms weight-importance methods, thus emerging as a promising candidate for CL in RNNs. Overall, we provide insights on the differences between CL in feedforward networks and RNNs, while guiding towards effective solutions to tackle CL on sequential data.

----

## [430] A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention

**Authors**: *Grégoire Mialon, Dexiong Chen, Alexandre d'Aspremont, Julien Mairal*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZK6vTvb84s](https://openreview.net/forum?id=ZK6vTvb84s)

**Abstract**:

We address the problem of learning on sets of features, motivated by the need of performing pooling operations in long biological sequences of varying sizes, with long-range dependencies, and possibly few labeled data. To address this challenging task, we introduce a parametrized representation of fixed size, which  embeds and then aggregates elements from a given input set according to the optimal transport plan between the set and a trainable reference. Our approach scales to large datasets and allows end-to-end training of the reference, while also providing a simple unsupervised learning mechanism with small computational cost. Our aggregation technique admits two useful interpretations: it may be seen as a mechanism related to attention layers in neural networks, or it may be seen as a scalable surrogate of a classical optimal transport-based kernel. We experimentally demonstrate the effectiveness of our approach on biological sequences, achieving state-of-the-art results for protein fold recognition and detection of chromatin profiles tasks, and, as a proof of concept, we show promising results for processing natural language sequences. We provide an open-source implementation of our embedding that can be used alone or as a module in larger learning models at https://github.com/claying/OTK.

----

## [431] Learning "What-if" Explanations for Sequential Decision-Making

**Authors**: *Ioana Bica, Daniel Jarrett, Alihan Hüyük, Mihaela van der Schaar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=h0de3QWtGG](https://openreview.net/forum?id=h0de3QWtGG)

**Abstract**:

Building interpretable parameterizations of real-world decision-making on the basis of demonstrated behavior--i.e. trajectories of observations and actions made by an expert maximizing some unknown reward function--is essential for introspecting and auditing policies in different institutions. In this paper, we propose learning explanations of expert decisions by modeling their reward function in terms of preferences with respect to ``"what if'' outcomes: Given the current history of observations, what would happen if we took a particular action? To learn these cost-benefit tradeoffs associated with the expert's actions, we integrate counterfactual reasoning into batch inverse reinforcement learning. This offers a principled way of defining reward functions and explaining expert behavior, and also satisfies the constraints of real-world decision-making---where active experimentation is often impossible (e.g. in healthcare). Additionally, by estimating the effects of different actions, counterfactuals readily tackle the off-policy nature of policy evaluation in the batch setting, and can naturally accommodate settings where the expert policies depend on histories of observations rather than just current states. Through illustrative experiments in both real and simulated medical environments, we highlight the effectiveness of our batch, counterfactual inverse reinforcement learning approach in recovering accurate and interpretable descriptions of behavior.

----

## [432] Improving Transformation Invariance in Contrastive Representation Learning

**Authors**: *Adam Foster, Rattana Pukdee, Tom Rainforth*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NomEDgIEBwE](https://openreview.net/forum?id=NomEDgIEBwE)

**Abstract**:

We propose methods to strengthen the invariance properties of representations obtained by contrastive learning. While existing approaches implicitly induce a degree of invariance as representations are learned, we look to more directly enforce invariance in the encoding process. To this end, we first introduce a training objective for contrastive learning that uses a novel regularizer to control how the representation changes under transformation. We show that representations trained with this objective perform better on downstream tasks and are more robust to the introduction of nuisance transformations at test time. Second, we propose a change to how test time representations are generated by introducing a feature averaging approach that combines encodings from multiple transformations of the original input, finding that this leads to across the board performance gains. Finally, we introduce the novel Spirograph dataset to explore our ideas in the context of a differentiable generative process with multiple downstream tasks, showing that our techniques for learning invariance are highly beneficial.

----

## [433] Shapley explainability on the data manifold

**Authors**: *Christopher Frye, Damien de Mijolla, Tom Begley, Laurence Cowton, Megan Stanley, Ilya Feige*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OPyWRrcjVQw](https://openreview.net/forum?id=OPyWRrcjVQw)

**Abstract**:

Explainability in AI is crucial for model development, compliance with regulation, and providing operational nuance to predictions. The Shapley framework for explainability attributes a model’s predictions to its input features in a mathematically principled and model-agnostic way. However, general implementations of Shapley explainability make an untenable assumption: that the model’s features are uncorrelated. In this work, we demonstrate unambiguous drawbacks of this assumption and develop two solutions to Shapley explainability that respect the data manifold. One solution, based on generative modelling, provides flexible access to data imputations; the other directly learns the Shapley value-function, providing performance and stability at the cost of flexibility. While “off-manifold” Shapley values can (i) give rise to incorrect explanations, (ii) hide implicit model dependence on sensitive attributes, and (iii) lead to unintelligible explanations in higher-dimensional data, on-manifold explainability overcomes these problems.

----

## [434] Noise or Signal: The Role of Image Backgrounds in Object Recognition

**Authors**: *Kai Yuanqing Xiao, Logan Engstrom, Andrew Ilyas, Aleksander Madry*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gl3D-xY7wLq](https://openreview.net/forum?id=gl3D-xY7wLq)

**Abstract**:

We assess the tendency of state-of-the-art object recognition models to depend on signals from image backgrounds. We create a toolkit for disentangling foreground and background signal on ImageNet images, and find that (a) models can achieve non-trivial accuracy by relying on the background alone, (b) models often misclassify images even in the presence of correctly classified foregrounds--up to 88% of the time with adversarially chosen backgrounds, and (c) more accurate models tend to depend on backgrounds less. Our analysis of backgrounds brings us closer to understanding which correlations machine learning models use, and how they determine models' out of distribution performance.

----

## [435] Enjoy Your Editing: Controllable GANs for Image Editing via Latent Space Navigation

**Authors**: *Peiye Zhuang, Oluwasanmi Koyejo, Alexander G. Schwing*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HOFxeCutxZR](https://openreview.net/forum?id=HOFxeCutxZR)

**Abstract**:

Controllable semantic image editing enables a user to change entire image attributes with a few clicks, e.g., gradually making a summer scene look like it was taken in winter. Classic approaches for this task use a Generative Adversarial Net (GAN) to learn a latent space and suitable latent-space transformations. However, current approaches often suffer from attribute edits that are entangled, global image identity changes, and diminished photo-realism. To address these concerns, we learn multiple attribute transformations simultaneously, integrate attribute regression into the training of transformation functions, and apply a content loss and an adversarial loss that encourages the maintenance of image identity and photo-realism. We propose quantitative evaluation strategies for measuring controllable editing performance, unlike prior work, which primarily focuses on qualitative evaluation. Our model permits better control for both single- and multiple-attribute editing while preserving image identity and realism during transformation. We provide empirical results for both natural and synthetic images, highlighting that our model achieves state-of-the-art performance for targeted image manipulation.

----

## [436] Perceptual Adversarial Robustness: Defense Against Unseen Threat Models

**Authors**: *Cassidy Laidlaw, Sahil Singla, Soheil Feizi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dFwBosAcJkN](https://openreview.net/forum?id=dFwBosAcJkN)

**Abstract**:

A key challenge in adversarial robustness is the lack of a precise mathematical characterization of human perception, used in the definition of adversarial attacks that are imperceptible to human eyes. Most current attacks and defenses try to get around this issue by considering restrictive adversarial threat models such as those bounded by $L_2$ or $L_\infty$ distance, spatial perturbations, etc. However, models that are robust against any of these restrictive threat models are still fragile against other threat models, i.e. they have poor generalization to unforeseen attacks. Moreover, even if a model is robust against the union of several restrictive threat models, it is still susceptible to other imperceptible adversarial examples that are not contained in any of the constituent threat models. To resolve these issues, we propose adversarial training against the set of all imperceptible adversarial examples. Since this set is intractable to compute without a human in the loop, we approximate it using deep neural networks. We call this threat model the neural perceptual threat model (NPTM); it includes adversarial examples with a bounded neural perceptual distance (a neural network-based approximation of the true perceptual distance) to natural images. Through an extensive perceptual study, we show that the neural perceptual distance correlates well with human judgements of perceptibility of adversarial examples, validating our threat model.

Under the NPTM, we develop novel perceptual adversarial attacks and defenses. Because the NPTM is very broad, we find that Perceptual Adversarial Training (PAT) against a perceptual attack gives robustness against many other types of adversarial attacks. We test PAT on CIFAR-10 and ImageNet-100 against five diverse adversarial attacks: $L_2$, $L_\infty$, spatial, recoloring, and JPEG. We find that PAT achieves state-of-the-art robustness against the union of these five attacks—more than doubling the accuracy over the next best model—without training against any of them. That is, PAT generalizes well to unforeseen perturbation types. This is vital in sensitive applications where a particular threat model cannot be assumed, and to the best of our knowledge, PAT is the first adversarial training defense with this property.

Code and data are available at https://github.com/cassidylaidlaw/perceptual-advex

----

## [437] Zero-Cost Proxies for Lightweight NAS

**Authors**: *Mohamed S. Abdelfattah, Abhinav Mehrotra, Lukasz Dudziak, Nicholas Donald Lane*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0cmMMy8J5q](https://openreview.net/forum?id=0cmMMy8J5q)

**Abstract**:

Neural Architecture Search (NAS) is quickly becoming the standard methodology to design neural network models. However, NAS is typically compute-intensive because multiple models need to be evaluated before choosing the best one. To reduce the computational power and time needed, a proxy task is often used for evaluating each model instead of full training. In this paper, we evaluate conventional reduced-training proxies and quantify how well they preserve ranking between neural network models during search when compared with the rankings produced by final trained accuracy. We propose a series of zero-cost proxies, based on recent pruning literature, that use just a single minibatch of training data to compute a model's score. Our zero-cost proxies use 3 orders of magnitude less computation but can match and even outperform conventional proxies. For example, Spearman's rank correlation coefficient between final validation accuracy and our best zero-cost proxy on NAS-Bench-201 is 0.82, compared to 0.61 for EcoNAS (a recently proposed reduced-training proxy). Finally, we use these zero-cost proxies to enhance existing NAS search algorithms such as random search, reinforcement learning, evolutionary search and predictor-based search. For all search methodologies and across three different NAS datasets, we are able to significantly improve sample efficiency, and thereby decrease computation, by using our zero-cost proxies. For example on NAS-Bench-101, we achieved the same accuracy 4$\times$ quicker than the best previous result. Our code is made public at: https://github.com/mohsaied/zero-cost-nas.

----

## [438] Usable Information and Evolution of Optimal Representations During Training

**Authors**: *Michael Kleinman, Alessandro Achille, Daksh Idnani, Jonathan C. Kao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=p8agn6bmTbr](https://openreview.net/forum?id=p8agn6bmTbr)

**Abstract**:

We introduce a notion of usable information contained in the representation learned by a deep network, and use it to study how optimal representations for the task emerge during training. We show that the implicit regularization coming from training with Stochastic Gradient Descent with a high learning-rate and small batch size plays an important role in learning minimal sufficient representations for the task. In the process of arriving at a minimal sufficient representation, we find that the content of the representation changes dynamically during training. In particular, we find that semantically meaningful but ultimately irrelevant information is encoded in the early transient dynamics of training, before being later discarded. In addition, we evaluate how perturbing the initial part of training impacts the learning dynamics and the resulting representations. We show these effects on both perceptual decision-making tasks inspired by neuroscience literature, as well as on standard image classification tasks.

----

## [439] Exploring the Uncertainty Properties of Neural Networks' Implicit Priors in the Infinite-Width Limit

**Authors**: *Ben Adlam, Jaehoon Lee, Lechao Xiao, Jeffrey Pennington, Jasper Snoek*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MjvduJCsE4](https://openreview.net/forum?id=MjvduJCsE4)

**Abstract**:

Modern deep learning models have achieved great success in predictive accuracy for many data modalities. However, their application to many real-world tasks is restricted by poor uncertainty estimates, such as overconfidence on out-of-distribution (OOD) data and ungraceful failing under distributional shift. Previous benchmarks have found that ensembles of neural networks (NNs) are typically the best calibrated models on OOD data. Inspired by this, we leverage recent theoretical advances that characterize the function-space prior of an infinitely-wide NN as a Gaussian process, termed the neural network Gaussian process (NNGP). We use the NNGP with a softmax link function to build a probabilistic model for multi-class classification and marginalize over the latent Gaussian outputs to sample from the posterior. This gives us a better understanding of the implicit prior NNs place on function space and allows a direct comparison of the calibration of the NNGP and its finite-width analogue. We also examine the calibration of previous approaches to classification with the NNGP, which treat classification problems as regression to the one-hot labels. In this case the Bayesian posterior is exact, and we compare several heuristics to generate a categorical distribution over classes. We find these methods are well calibrated under distributional shift. Finally, we consider an infinite-width final layer in conjunction with a pre-trained embedding. This replicates the important practical use case of transfer learning and allows scaling to significantly larger datasets. As well as achieving competitive predictive accuracy, this approach is better calibrated than its finite width analogue.

----

## [440] On the geometry of generalization and memorization in deep neural networks

**Authors**: *Cory Stephenson, Suchismita Padhy, Abhinav Ganesh, Yue Hui, Hanlin Tang, SueYeon Chung*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=V8jrrnwGbuc](https://openreview.net/forum?id=V8jrrnwGbuc)

**Abstract**:

Understanding how large neural networks avoid memorizing training data is key to explaining their high generalization performance. To examine the structure of when and where memorization occurs in a deep network, we use a recently developed replica-based mean field theoretic geometric analysis method. We find that all layers preferentially learn from examples which share features, and link this behavior to generalization performance. Memorization predominately occurs in the deeper layers, due to decreasing object manifolds’ radius and dimension, whereas early layers are minimally affected. This predicts that generalization can be restored by reverting the final few layer weights to earlier epochs before significant memorization occurred, which is confirmed by the experiments. Additionally, by studying generalization under different model sizes, we reveal the connection between the double descent phenomenon and the underlying model geometry. Finally, analytical analysis shows that networks avoid memorization early in training because close to initialization, the gradient contribution from permuted examples are small. These findings provide quantitative evidence for the structure of memorization across layers of a deep neural network, the drivers for such structure, and its connection to manifold geometric properties.

----

## [441] Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks

**Authors**: *Alexander Levine, Soheil Feizi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YUGG2tFuPM](https://openreview.net/forum?id=YUGG2tFuPM)

**Abstract**:

Adversarial poisoning attacks distort training data in order to corrupt the test-time behavior of a classifier. A provable defense provides a certificate for each test sample, which is a lower bound on the magnitude of any adversarial distortion of the training set that can corrupt the test sample's classification.
We propose two novel provable defenses against poisoning attacks: (i) Deep Partition Aggregation (DPA), a certified defense against a general poisoning threat model, defined as the insertion or deletion of a bounded number of samples to the training set --- by implication, this threat model also includes arbitrary distortions to a bounded number of images and/or labels; and (ii) Semi-Supervised DPA (SS-DPA), a certified defense against label-flipping poisoning attacks. DPA is an ensemble method where base models are trained on partitions of the training set determined by a hash function. DPA is related to both subset aggregation, a well-studied ensemble method in classical machine learning, as well as to randomized smoothing, a popular provable defense against evasion (inference) attacks. Our defense against label-flipping poison attacks, SS-DPA, uses a semi-supervised learning algorithm as its base classifier model: each base classifier is trained using the entire unlabeled training set in addition to the labels for a partition. SS-DPA significantly outperforms the existing certified defense for label-flipping attacks (Rosenfeld et al., 2020) on both MNIST and CIFAR-10: provably tolerating, for at least half of test images, over 600 label flips (vs. < 200 label flips) on MNIST and over 300 label flips (vs. 175 label flips) on CIFAR-10. Against general poisoning attacks where no prior certified defenses exists, DPA can certify $\geq$ 50% of test images against over 500 poison image insertions on MNIST, and nine insertions on CIFAR-10. These results establish new state-of-the-art provable defenses against general and label-flipping poison attacks. Code is available at https://github.com/alevine0/DPA

----

## [442] DC3: A learning method for optimization with hard constraints

**Authors**: *Priya L. Donti, David Rolnick, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=V1ZHVxJ6dSS](https://openreview.net/forum?id=V1ZHVxJ6dSS)

**Abstract**:

Large optimization problems with hard constraints arise in many settings, yet classical solvers are often prohibitively slow, motivating the use of deep networks as cheap "approximate solvers." Unfortunately, naive deep learning approaches typically cannot enforce the hard constraints of such problems, leading to infeasible solutions. In this work, we present Deep Constraint Completion and Correction (DC3), an algorithm to address this challenge. Specifically, this method enforces feasibility via a differentiable procedure, which implicitly completes partial solutions to satisfy equality constraints and unrolls gradient-based corrections to satisfy inequality constraints. We demonstrate the effectiveness of DC3 in both synthetic optimization tasks and the real-world setting of AC optimal power flow, where hard constraints encode the physics of the electrical grid. In both cases, DC3 achieves near-optimal objective values while preserving feasibility.

----

## [443] Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study

**Authors**: *Zhiqiang Shen, Zechun Liu, Dejia Xu, Zitian Chen, Kwang-Ting Cheng, Marios Savvides*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PObuuGVrGaZ](https://openreview.net/forum?id=PObuuGVrGaZ)

**Abstract**:

This work aims to empirically clarify a recently discovered perspective that label smoothing is incompatible with knowledge distillation. We begin by introducing the motivation behind on how this incompatibility is raised, i.e., label smoothing erases relative information between teacher logits. We provide a novel connection on how label smoothing affects distributions of semantically similar and dissimilar classes. Then we propose a metric to quantitatively measure the degree of erased information in sample's representation. After that, we study its one-sidedness and imperfection of the incompatibility view through massive analyses, visualizations and comprehensive experiments on Image Classification, Binary Networks, and Neural Machine Translation. Finally, we broadly discuss several circumstances wherein label smoothing will indeed lose its effectiveness.

----

## [444] Shape-Texture Debiased Neural Network Training

**Authors**: *Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan L. Yuille, Cihang Xie*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Db4yerZTYkz](https://openreview.net/forum?id=Db4yerZTYkz)

**Abstract**:

Shape and texture are two prominent and complementary cues for recognizing objects. Nonetheless, Convolutional Neural Networks are often biased towards either texture or shape, depending on the training dataset. Our ablation shows that such bias degenerates model performance. Motivated by this observation, we develop a simple algorithm for shape-texture debiased learning. To prevent models from exclusively attending on a single cue in representation learning, we augment training data with images with conflicting shape and texture information (eg, an image of chimpanzee shape but with lemon texture) and, most importantly, provide the corresponding supervisions from shape and texture simultaneously. 

Experiments show that our method successfully improves model performance on several image recognition benchmarks and adversarial robustness. For example, by training on ImageNet, it helps ResNet-152 achieve substantial improvements on ImageNet (+1.2%), ImageNet-A  (+5.2%), ImageNet-C (+8.3%) and Stylized-ImageNet (+11.1%), and on defending against FGSM adversarial attacker on ImageNet (+14.4%). Our method also claims to be compatible with other advanced data augmentation strategies, eg, Mixup, and CutMix. The code is available here: https://github.com/LiYingwei/ShapeTextureDebiasedTraining.

----

## [445] Using latent space regression to analyze and leverage compositionality in GANs

**Authors**: *Lucy Chai, Jonas Wulff, Phillip Isola*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sjuuTm4vj0](https://openreview.net/forum?id=sjuuTm4vj0)

**Abstract**:

In recent years, Generative Adversarial Networks have become ubiquitous in both research and public perception, but how GANs convert an unstructured latent code to a high quality output is still an open question. In this work, we investigate regression into the latent space as a probe to understand the compositional properties of GANs.  We find that combining the regressor and a pretrained generator provides a strong image prior, allowing us to create composite images from a collage of random image parts at inference time while maintaining global consistency. To compare compositional properties across different generators, we measure the trade-offs between reconstruction of the unrealistic input and image quality of the regenerated samples. We find that the regression approach enables more localized editing of individual image parts compared to direct editing in the latent space, and we conduct experiments to quantify this independence effect. Our method is agnostic to the semantics of edits, and does not require labels or predefined concepts during training. Beyond image composition, our method extends to a number of related applications, such as image inpainting or example-based image editing, which we demonstrate on several GANs and datasets, and
because it uses only a single forward pass, it can operate in real-time. Code is available on our project page: https://chail.github.io/latent-composition/.

----

## [446] Blending MPC & Value Function Approximation for Efficient Reinforcement Learning

**Authors**: *Mohak Bhardwaj, Sanjiban Choudhury, Byron Boots*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RqCC_00Bg7V](https://openreview.net/forum?id=RqCC_00Bg7V)

**Abstract**:

Model-Predictive Control (MPC) is a powerful tool for controlling complex, real-world systems that uses a model to make predictions about future behavior. For each state encountered, MPC solves an online optimization problem to choose a control action that will minimize future cost. This is a surprisingly effective strategy, but real-time performance requirements warrant the use of simple models. If the model is not sufficiently accurate, then the resulting controller can be biased, limiting performance. We present a framework for improving on MPC with model-free reinforcement learning (RL). The key insight is to view MPC as constructing a series of local Q-function approximations. We show that by using a parameter $\lambda$, similar to the trace decay parameter in TD($\lambda$), we can systematically trade-off learned value estimates against the local Q-function approximations. We present a theoretical analysis that shows how error from inaccurate models in MPC and value function estimation in RL can be balanced. We further propose an algorithm that changes $\lambda$ over time to reduce the dependence on MPC as our estimates of the value function improve, and test the efficacy our approach on challenging high-dimensional manipulation tasks with biased models in simulation. We demonstrate that our approach can obtain performance comparable with MPC with access to true dynamics even under severe model bias and is more sample efficient as compared to model-free RL.

----

## [447] Model Patching: Closing the Subgroup Performance Gap with Data Augmentation

**Authors**: *Karan Goel, Albert Gu, Yixuan Li, Christopher Ré*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9YlaeLfuhJF](https://openreview.net/forum?id=9YlaeLfuhJF)

**Abstract**:

Classifiers in machine learning are often brittle when deployed. Particularly concerning are models with inconsistent performance on specific subgroups of a class, e.g., exhibiting disparities in skin cancer classification in the presence or absence of a spurious bandage. To mitigate these performance differences, we introduce model patching, a two-stage framework for improving robustness that encourages the model to be invariant to subgroup differences, and focus on class information shared by subgroups. Model patching first models subgroup features within a class and learns semantic transformations between them, and then trains a classifier with data augmentations that deliberately manipulate subgroup features. We instantiate model patching with CAMEL, which (1) uses a CycleGAN to learn the intra-class, inter-subgroup augmentations, and (2) balances subgroup performance using a theoretically-motivated subgroup consistency regularizer, accompanied by a new robust objective. We demonstrate CAMEL’s effectiveness on 3 benchmark datasets, with reductions in robust error of up to 33% relative to the best baseline. Lastly, CAMEL successfully patches a model that fails due to spurious features on a real-world skin cancer dataset.

----

## [448] Non-asymptotic Confidence Intervals of Off-policy Evaluation: Primal and Dual Bounds

**Authors**: *Yihao Feng, Ziyang Tang, Na Zhang, Qiang Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dKg5D1Z1Lm](https://openreview.net/forum?id=dKg5D1Z1Lm)

**Abstract**:

Off-policy evaluation (OPE) is the task of estimating the expected reward of a given policy based on offline data previously collected under different policies. Therefore, OPE is a key step in applying reinforcement learning to real-world domains such as medical treatment, where interactive data collection is expensive or even unsafe. As the observed data tends to be noisy and limited, it is essential to provide rigorous  uncertainty quantification, not just a point estimation, when applying OPE to make high stakes decisions. This work considers the problem of constructing non-asymptotic confidence intervals in infinite-horizon  off-policy evaluation, which remains a challenging open question. We develop a practical algorithm through a primal-dual optimization-based approach, which leverages the kernel Bellman loss (KBL) of Feng et al. 2019 and a new  martingale concentration inequality of KBL applicable to time-dependent data with unknown mixing conditions. Our algorithm makes  minimum assumptions on the data and the function class of the Q-function,  and works for the behavior-agnostic settings where the data is collected under a mix of arbitrary unknown behavior policies.  We present empirical results that clearly demonstrate the advantages of our approach over existing methods.

----

## [449] Linear Mode Connectivity in Multitask and Continual Learning

**Authors**: *Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Dilan Görür, Razvan Pascanu, Hassan Ghasemzadeh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Fmg_fQYUejf](https://openreview.net/forum?id=Fmg_fQYUejf)

**Abstract**:

Continual (sequential) training and multitask (simultaneous) training are often attempting to solve the same overall objective: to find a solution that performs well on all considered tasks. The main difference is in the training regimes, where continual learning can only have access to one task at a time, which for neural networks typically leads to catastrophic forgetting. That is, the solution found for a subsequent task does not perform well on the previous ones anymore. 
    However, the relationship between the different minima that the two training regimes arrive at is not well understood. What sets them apart? Is there a local structure that could explain the difference in performance achieved by the two different schemes? 
    Motivated by recent work showing that different minima of the same task are typically connected by very simple curves of low error, we investigate whether multitask and continual solutions are similarly connected. We empirically find that indeed such connectivity can be reliably achieved and, more interestingly, it can be done by a linear path, conditioned on having the same initialization for both. We thoroughly analyze this observation and discuss its significance for the continual learning process.
    Furthermore, we exploit this finding to propose an effective algorithm that constrains the sequentially learned minima to behave as the multitask solution.  We show that our method outperforms several state of the art continual learning algorithms on various vision benchmarks.

----

## [450] Robust and Generalizable Visual Representation Learning via Random Convolutions

**Authors**: *Zhenlin Xu, Deyi Liu, Junlin Yang, Colin Raffel, Marc Niethammer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BVSM0x3EDK6](https://openreview.net/forum?id=BVSM0x3EDK6)

**Abstract**:

While successful for various computer vision tasks, deep neural networks have shown to be vulnerable to texture style shifts and small perturbations to which humans are robust. In this work, we show that the robustness of neural networks can be greatly improved through the use of random convolutions as data augmentation. Random convolutions are approximately shape-preserving and may distort local textures. Intuitively, randomized convolutions create an infinite number of new domains with similar global shapes but random local texture. Therefore, we explore using outputs of multi-scale random convolutions as new images or mixing them with the original images during training. When applying a network trained with our approach to unseen domains, our method consistently improves the performance on domain generalization benchmarks and is scalable to ImageNet. In particular, in the challenging scenario of generalizing to the sketch domain in PACS and to ImageNet-Sketch, our method outperforms state-of-art methods by a large margin. More interestingly, our method can benefit downstream tasks by providing a more robust pretrained visual representation.

----

## [451] Intrinsic-Extrinsic Convolution and Pooling for Learning on 3D Protein Structures

**Authors**: *Pedro Hermosilla, Marco Schäfer, Matej Lang, Gloria Fackelmann, Pere-Pau Vázquez, Barbora Kozlíková, Michael Krone, Tobias Ritschel, Timo Ropinski*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=l0mSUROpwY](https://openreview.net/forum?id=l0mSUROpwY)

**Abstract**:

Proteins perform a large variety of functions in living organisms and thus play a key role in biology. However, commonly used algorithms in protein representation learning were not specifically designed for protein data, and are therefore not able to capture all relevant structural levels of a protein during learning. To fill this gap, we propose two new learning operators, specifically designed to process protein structures. First, we introduce a novel convolution operator that considers the primary, secondary, and tertiary structure of a protein by using $n$-D convolutions defined on both the Euclidean distance, as well as multiple geodesic distances between the atoms in a multi-graph. Second, we introduce a set of hierarchical pooling operators that enable multi-scale protein analysis. We further evaluate the accuracy of our algorithms on common downstream tasks, where we outperform state-of-the-art protein learning algorithms.

----

## [452] Variational State-Space Models for Localisation and Dense 3D Mapping in 6 DoF

**Authors**: *Atanas Mirchev, Baris Kayalibay, Patrick van der Smagt, Justin Bayer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XAS3uKeFWj](https://openreview.net/forum?id=XAS3uKeFWj)

**Abstract**:

We solve the problem of 6-DoF localisation and 3D dense reconstruction in spatial environments as approximate Bayesian inference in a deep state-space model. Our approach leverages both learning and domain knowledge from multiple-view geometry and rigid-body dynamics. This results in an expressive predictive model of the world, often missing in current state-of-the-art visual SLAM solutions. The combination of variational inference, neural networks and a differentiable raycaster ensures that our model is amenable to end-to-end gradient-based optimisation. We evaluate our approach on realistic unmanned aerial vehicle flight data, nearing the performance of state-of-the-art visual-inertial odometry systems. We demonstrate the applicability of the model to generative prediction and planning.

----

## [453] AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights

**Authors**: *Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han, Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Iz3zU3M316D](https://openreview.net/forum?id=Iz3zU3M316D)

**Abstract**:

Normalization techniques, such as batch normalization (BN), are a boon for modern deep learning. They let weights converge more quickly with often better generalization performances. It has been argued that the normalization-induced scale invariance among the weights provides an advantageous ground for gradient descent (GD) optimizers: the effective step sizes are automatically reduced over time, stabilizing the overall training procedure. It is often overlooked, however, that the additional introduction of momentum in GD optimizers results in a far more rapid reduction in effective step sizes for scale-invariant weights, a phenomenon that has not yet been studied and may have caused unwanted side effects in the current practice. This is a crucial issue because arguably the vast majority of modern deep neural networks consist of (1) momentum-based GD (e.g. SGD or Adam) and (2) scale-invariant parameters (e.g. more than 90% of the weights in ResNet are scale-invariant due to BN). In this paper, we verify that the widely-adopted combination of the two ingredients lead to the premature decay of effective step sizes and sub-optimal model performances. We propose a simple and effective remedy, SGDP and AdamP: get rid of the radial component, or the norm-increasing direction, at each optimizer step. Because of the scale invariance, this modification only alters the effective step sizes without changing the effective update directions, thus enjoying the original convergence properties of GD optimizers. Given the ubiquity of momentum GD and scale invariance in machine learning, we have evaluated our methods against the baselines on 13 benchmarks. They range from vision tasks like classification (e.g. ImageNet), retrieval (e.g. CUB and SOP), and detection (e.g. COCO) to language modelling (e.g. WikiText) and audio classification (e.g. DCASE) tasks. We verify that our solution brings about uniform gains in performances in those benchmarks. Source code is available at https://github.com/clovaai/adamp

----

## [454] MiCE: Mixture of Contrastive Experts for Unsupervised Image Clustering

**Authors**: *Tsung Wei Tsai, Chongxuan Li, Jun Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gV3wdEOGy_V](https://openreview.net/forum?id=gV3wdEOGy_V)

**Abstract**:

We present Mixture of Contrastive Experts (MiCE), a unified probabilistic clustering framework that simultaneously exploits the discriminative representations learned by contrastive learning and the semantic structures captured by a latent mixture model. Motivated by the mixture of experts, MiCE employs a gating function to partition an unlabeled dataset into subsets according to the latent semantics and multiple experts to discriminate distinct subsets of instances assigned to them in a contrastive learning manner. To solve the nontrivial inference and learning problems caused by the latent variables, we further develop a scalable variant of the Expectation-Maximization (EM) algorithm for MiCE and provide proof of the convergence. Empirically, we evaluate the clustering performance of MiCE on four widely adopted natural image datasets. MiCE achieves significantly better results than various previous methods and a strong contrastive learning baseline.

----

## [455] HalentNet: Multimodal Trajectory Forecasting with Hallucinative Intents

**Authors**: *Deyao Zhu, Mohamed Zahran, Li Erran Li, Mohamed Elhoseiny*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9GBZBPn0Jx](https://openreview.net/forum?id=9GBZBPn0Jx)

**Abstract**:

Motion forecasting is essential for making intelligent decisions in robotic navigation. As a result, the multi-agent behavioral prediction has become a core component of modern human-robot interaction applications such as autonomous driving. Due to various intentions and interactions among agents, agent trajectories can have multiple possible futures. Hence, the motion forecasting model's ability to cover possible modes becomes essential to enable accurate prediction. Towards this goal, we introduce HalentNet to better model the future motion distribution in addition to a traditional trajectory regression learning objective by incorporating generative augmentation losses. We model intents with unsupervised discrete random variables whose training is guided by a collaboration between two key signals: A discriminative loss that encourages intents' diversity and a hallucinative loss that explores intent transitions (i.e., mixed intents) and encourages their smoothness. This regulates the neural network behavior to be more accurately predictive on uncertain scenarios due to the active yet careful exploration of possible future agent behavior. Our model's learned representation leads to better and more semantically meaningful coverage of the trajectory distribution. Our experiments show that our method can improve over the state-of-the-art trajectory forecasting benchmarks, including vehicles and pedestrians, for about 20% on average FDE and 50% on road boundary violation rate when predicting 6 seconds future. We also conducted human experiments to show that our predicted trajectories received 39.6% more votes than the runner-up approach and 32.2% more votes than our variant without hallucinative mixed intent loss. The code will be released soon.

----

## [456] Model-based micro-data reinforcement learning: what are the crucial model properties and which model to choose?

**Authors**: *Balázs Kégl, Gabriel Hurtado, Albert Thomas*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=p5uylG94S68](https://openreview.net/forum?id=p5uylG94S68)

**Abstract**:

We contribute to micro-data model-based reinforcement learning (MBRL) by rigorously comparing popular generative models using a fixed (random shooting) control agent. We find that on an environment that requires multimodal posterior predictives, mixture density nets outperform all other models by a large margin. When multimodality is not required, our surprising finding is that we do not need probabilistic posterior predictives: deterministic models are on par, in fact they consistently (although non-significantly) outperform their probabilistic counterparts. We also found that heteroscedasticity at training time, perhaps acting as a regularizer, improves predictions at longer horizons. At the methodological side, we design metrics and an experimental protocol which can be used to evaluate the various models, predicting their asymptotic performance when using them on the control problem. Using this framework, we improve the state-of-the-art sample complexity of MBRL on Acrobot by two to four folds, using an aggressive training schedule which is outside of the hyperparameter interval usually considered.

----

## [457] Private Image Reconstruction from System Side Channels Using Generative Models

**Authors**: *Yuanyuan Yuan, Shuai Wang, Junping Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=y06VOYLcQXa](https://openreview.net/forum?id=y06VOYLcQXa)

**Abstract**:

System side channels denote effects imposed on the underlying system and hardware when running a program, such as its accessed CPU cache lines. Side channel analysis (SCA) allows attackers to infer program secrets based on observed side channel signals. Given the ever-growing adoption of machine learning as a service (MLaaS), image analysis software on cloud platforms has been exploited by reconstructing private user images from system side channels. Nevertheless, to date, SCA is still highly challenging, requiring technical knowledge of victim software's internal operations. For existing SCA attacks, comprehending such internal operations requires heavyweight program analysis or manual efforts.

This research proposes an attack framework to reconstruct private user images processed by media software via system side channels. The framework forms an effective workflow by incorporating convolutional networks, variational autoencoders, and generative adversarial networks. Our evaluation of two popular side channels shows that the reconstructed images consistently match user inputs, making privacy leakage attacks more practical. We also show surprising results that even one-bit data read/write pattern side channels, which are deemed minimally informative, can be used to reconstruct quality images using our framework.

----

## [458] Contextual Transformation Networks for Online Continual Learning

**Authors**: *Quang Pham, Chenghao Liu, Doyen Sahoo, Steven C. H. Hoi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zx_uX-BO7CH](https://openreview.net/forum?id=zx_uX-BO7CH)

**Abstract**:

Continual learning methods with fixed architectures rely on a single network to learn models that can perform well on all tasks.
As a result, they often only accommodate common features of those tasks but neglect each task's specific features. On the other hand, dynamic architecture methods can have a separate network for each task, but they are too expensive to train and not scalable in practice, especially in online settings.
To address this problem, we propose a novel online continual learning method named ``Contextual Transformation Networks” (CTN) to efficiently model the \emph{task-specific features} while enjoying neglectable complexity overhead compared to other fixed architecture methods. 
Moreover, inspired by the Complementary Learning Systems (CLS) theory, we propose a novel dual memory design and an objective to train CTN that can address both catastrophic forgetting and knowledge transfer simultaneously. 
Our extensive experiments show that CTN is competitive with a large scale dynamic architecture network and consistently outperforms other fixed architecture methods under the same standard backbone. Our implementation can be found at \url{https://github.com/phquang/Contextual-Transformation-Network}.

----

## [459] A Unified Approach to Interpreting and Boosting Adversarial Transferability

**Authors**: *Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=X76iqnUbBjz](https://openreview.net/forum?id=X76iqnUbBjz)

**Abstract**:

In this paper, we use the interaction inside adversarial perturbations to explain and boost the adversarial transferability. We discover and prove the negative correlation between the adversarial transferability and the interaction inside adversarial perturbations. The negative correlation is further verified through different DNNs with various inputs. Moreover, this negative correlation can be regarded as a unified perspective to understand current transferability-boosting methods. To this end, we prove that some classic methods of enhancing the transferability essentially decease interactions inside adversarial perturbations. Based on this, we propose to directly penalize interactions during the attacking process, which significantly improves the adversarial transferability. We will release the code when the paper is accepted.

----

## [460] The inductive bias of ReLU networks on orthogonally separable data

**Authors**: *Mary Phuong, Christoph H. Lampert*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=krz7T0xU9Z_](https://openreview.net/forum?id=krz7T0xU9Z_)

**Abstract**:

We study the inductive bias of two-layer ReLU networks trained by gradient flow. We identify a class of easy-to-learn (`orthogonally separable') datasets, and characterise the solution that ReLU networks trained on such datasets converge to. Irrespective of network width, the solution turns out to be a combination of two max-margin classifiers: one corresponding to the positive data subset and one corresponding to the negative data subset.
The proof is based on the recently introduced concept of extremal sectors, for which we prove a number of properties in the context of orthogonal separability. In particular, we prove stationarity of activation patterns from some time $T$ onwards, which enables a reduction of the ReLU network to an ensemble of linear subnetworks.

----

## [461] A statistical theory of cold posteriors in deep neural networks

**Authors**: *Laurence Aitchison*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Rd138pWXMvG](https://openreview.net/forum?id=Rd138pWXMvG)

**Abstract**:

To get Bayesian neural networks to perform comparably to standard neural networks it is usually necessary to artificially reduce uncertainty using a tempered or cold posterior. This is extremely concerning: if the prior is accurate, Bayes inference/decision theory is optimal, and any artificial changes to the posterior should harm performance. While this suggests that the prior may be at fault, here we argue that in fact, BNNs for image classification use the wrong likelihood. In particular, standard image benchmark datasets such as CIFAR-10 are carefully curated. We develop a generative model describing curation which gives a principled Bayesian account of cold posteriors, because the likelihood under this new generative model closely matches the tempered likelihoods used in past work.

----

## [462] IOT: Instance-wise Layer Reordering for Transformer Structures

**Authors**: *Jinhua Zhu, Lijun Wu, Yingce Xia, Shufang Xie, Tao Qin, Wengang Zhou, Houqiang Li, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ipUPfYxWZvM](https://openreview.net/forum?id=ipUPfYxWZvM)

**Abstract**:

With sequentially stacked self-attention, (optional) encoder-decoder attention, and feed-forward layers, Transformer achieves big success in natural language processing (NLP), and many variants have been proposed. Currently, almost all these models assume that the \emph{layer order} is fixed and kept the same across data samples. We observe that different data samples actually favor different orders of the layers. Based on this observation, in this work, we break the assumption of the fixed layer order in Transformer and introduce instance-wise layer reordering into model structure. Our Instance-wise Ordered Transformer (IOT) can model variant functions by reordered layers, which enables each sample to select the better one to improve the model performance under the constraint of almost same number of parameters. To achieve this, we introduce a light predictor with negligible parameter and inference cost to decide the most capable and favorable layer order for any input sequence. Experiments on $3$ tasks (neural machine translation, abstractive summarization, and code generation) and $9$ datasets demonstrate consistent improvements of our method. We further show that our method can also be applied to other architectures beyond Transformer. Our code is released at Github\footnote{\url{https://github.com/instance-wise-ordered-transformer/IOT}}.

----

## [463] Counterfactual Generative Networks

**Authors**: *Axel Sauer, Andreas Geiger*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BXewfAYMmJw](https://openreview.net/forum?id=BXewfAYMmJw)

**Abstract**:

Neural networks are prone to learning shortcuts -- they often model simple correlations, ignoring more complex ones that potentially generalize better. Prior works on image classification show that instead of learning a connection to object shape, deep classifiers tend to exploit spurious correlations with low-level texture or the background for solving the classification task. In this work, we take a step towards more robust and interpretable classifiers that explicitly expose the task's causal structure. Building on current advances in deep generative modeling, we propose to decompose the image generation process into independent causal mechanisms that we train without direct supervision. By exploiting appropriate inductive biases, these mechanisms disentangle object shape, object texture, and background; hence, they allow for generating counterfactual images. We demonstrate the ability of our model to generate such images on MNIST and ImageNet. Further, we show that the counterfactual images can improve out-of-distribution robustness with a marginal drop in performance on the original classification task, despite being synthetic. Lastly, our generative model can be trained efficiently on a single GPU, exploiting common pre-trained models as inductive biases.

----

## [464] Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data

**Authors**: *Jonathan Pilault, Amine Elhattami, Christopher J. Pal*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=de11dbHzAMF](https://openreview.net/forum?id=de11dbHzAMF)

**Abstract**:

Multi-Task Learning (MTL) networks have emerged as a promising method for transferring learned knowledge across different tasks. However, MTL must deal with challenges such as: overfitting to low resource tasks, catastrophic forgetting, and negative task transfer, or learning interference. Often, in Natural Language Processing (NLP), a separate model per task is needed to obtain the best performance. However, many fine-tuning approaches are both parameter inefficient, i.e., potentially involving one new model per task, and highly susceptible to losing knowledge acquired during pretraining. We propose a novel Transformer based Hypernetwork Adapter consisting of a new conditional attention mechanism as well as a set of task-conditioned modules that facilitate weight sharing. Through this construction, we achieve more efficient parameter sharing and mitigate forgetting by keeping half of the weights of a pretrained model fixed. We also use a new multi-task data sampling strategy to mitigate the negative effects of data imbalance across tasks. Using this approach, we are able to surpass single task fine-tuning methods while being parameter and data efficient (using around 66% of the data). Compared to other BERT Large methods on GLUE, our 8-task model surpasses other Adapter methods by 2.8% and our 24-task model outperforms by 0.7-1.0% models that use MTL and single task fine-tuning. We show that a larger variant of our single multi-task model approach performs competitively across 26 NLP tasks and yields state-of-the-art results on a number of test and development sets.

----

## [465] Towards Impartial Multi-task Learning

**Authors**: *Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IMPnRXEWpvr](https://openreview.net/forum?id=IMPnRXEWpvr)

**Abstract**:

Multi-task learning (MTL) has been widely used in representation learning. However, naively training all tasks simultaneously may lead to the partial training issue, where specific tasks are trained more adequately than others. In this paper, we propose to learn multiple tasks impartially. Specifically, for the task-shared parameters, we optimize the scaling factors via a closed-form solution, such that the aggregated gradient (sum of raw gradients weighted by the scaling factors) has equal projections onto individual tasks. For the task-specific parameters, we dynamically weigh the task losses so that all of them are kept at a comparable scale. Further, we find the above gradient balance and loss balance are complementary and thus propose a hybrid balance method to further improve the performance. Our impartial multi-task learning (IMTL) can be end-to-end trained without any heuristic hyper-parameter tuning, and is general to be applied on all kinds of losses without any distribution assumption. Moreover, our IMTL can converge to similar results even when the task losses are designed to have different scales, and thus it is scale-invariant. We extensively evaluate our IMTL on the standard MTL benchmarks including Cityscapes, NYUv2 and CelebA. It outperforms existing loss weighting methods under the same experimental settings.

----

## [466] Theoretical bounds on estimation error for meta-learning

**Authors**: *James Lucas, Mengye Ren, Irene Raissa Kameni, Toniann Pitassi, Richard S. Zemel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=SZ3wtsXfzQR](https://openreview.net/forum?id=SZ3wtsXfzQR)

**Abstract**:

Machine learning models have traditionally been developed under the assumption that the training and test distributions match exactly. However, recent success in few-shot learning and related problems are encouraging signs that these models can be adapted to more realistic settings where train and test distributions differ. Unfortunately, there is severely limited theoretical support for these algorithms and little is known about the difficulty of these problems. In this work, we provide novel information-theoretic lower-bounds on minimax rates of convergence for algorithms that are trained on data from multiple sources and tested on novel data. Our bounds depend intuitively on the information shared between sources of data, and characterize the difficulty of learning in this setting for arbitrary algorithms. We demonstrate these bounds on a hierarchical Bayesian model of meta-learning, computing both upper and lower bounds on parameter estimation via maximum-a-posteriori inference.

----

## [467] Domain-Robust Visual Imitation Learning with Mutual Information Constraints

**Authors**: *Edoardo Cetin, Oya Çeliktutan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QubpWYfdNry](https://openreview.net/forum?id=QubpWYfdNry)

**Abstract**:

Human beings are able to understand objectives and learn by simply observing others perform a task. Imitation learning methods aim to replicate such capabilities, however, they generally depend on access to a full set of optimal states and actions taken with the agent's actuators and from the agent's point of view. In this paper, we introduce a new algorithm - called Disentangling Generative Adversarial Imitation Learning (DisentanGAIL) - with the purpose of bypassing such constraints. Our algorithm enables autonomous agents to learn directly from high dimensional observations of an expert performing a task, by making use of adversarial learning with a latent representation inside the discriminator network. Such latent representation is regularized through mutual information constraints to incentivize learning only features that encode information about the completion levels of the task being demonstrated. This allows to obtain a shared feature space to successfully perform imitation while disregarding the differences between the expert's and the agent's domains. Empirically, our algorithm is able to efficiently imitate in a diverse range of control problems including balancing, manipulation and locomotive tasks, while being robust to various domain differences in terms of both environment appearance and agent embodiment.

----

## [468] Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding

**Authors**: *Sana Tonekaboni, Danny Eytan, Anna Goldenberg*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8qDwejCuCN](https://openreview.net/forum?id=8qDwejCuCN)

**Abstract**:

Time series are often complex and rich in information but sparsely labeled and therefore challenging to model. In this paper, we propose a self-supervised framework for learning robust and generalizable representations for time series. Our approach, called Temporal Neighborhood Coding (TNC), takes advantage of the local smoothness of a signal's generative process to define neighborhoods in time with stationary properties. Using a debiased contrastive objective, our framework learns time series representations by ensuring that in the encoding space, the distribution of signals from within a neighborhood is distinguishable from the distribution of non-neighboring signals. Our motivation stems from the medical field, where the ability to model the dynamic nature of time series data is especially valuable for identifying, tracking, and predicting the underlying patients' latent states in settings where labeling data is practically impossible. We compare our method to recently developed unsupervised representation learning approaches and demonstrate superior performance on clustering and classification tasks for multiple datasets.

----

## [469] Enforcing robust control guarantees within neural network policies

**Authors**: *Priya L. Donti, Melrose Roderick, Mahyar Fazlyab, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5lhWG3Hj2By](https://openreview.net/forum?id=5lhWG3Hj2By)

**Abstract**:

When designing controllers for safety-critical systems, practitioners often face a challenging tradeoff between robustness and performance. While robust control methods provide rigorous guarantees on system stability under certain worst-case disturbances, they often yield simple controllers that perform poorly in the average (non-worst) case. In contrast, nonlinear control methods trained using deep learning have achieved state-of-the-art performance on many control tasks, but often lack robustness guarantees. In this paper, we propose a technique that combines the strengths of these two approaches: constructing a generic nonlinear control policy class, parameterized by neural networks, that nonetheless enforces the same provable robustness criteria as robust control. Specifically, our approach entails integrating custom convex-optimization-based projection layers into a neural network-based policy. We demonstrate the power of this approach on several domains, improving in average-case performance over existing robust control methods and in worst-case stability over (non-robust) deep RL methods.

----

## [470] Active Contrastive Learning of Audio-Visual Video Representations

**Authors**: *Shuang Ma, Zhaoyang Zeng, Daniel McDuff, Yale Song*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OMizHuea_HB](https://openreview.net/forum?id=OMizHuea_HB)

**Abstract**:

Contrastive learning has been shown to produce generalizable representations of audio and visual data by maximizing the lower bound on the mutual information (MI) between different views of an instance. However, obtaining a tight lower bound requires a sample size exponential in MI and thus a large set of negative samples. We can incorporate more samples by building a large queue-based dictionary, but there are theoretical limits to performance improvements even with a large number of negative samples. We hypothesize that random negative sampling leads to a highly redundant dictionary that results in suboptimal representations for downstream tasks. In this paper, we propose an active contrastive learning approach that builds an actively sampled dictionary with diverse and informative items, which improves the quality of negative samples and improves performances on tasks where there is high mutual information in the data, e.g., video classification. Our model achieves state-of-the-art performance on challenging audio and visual downstream benchmarks including UCF101, HMDB51 and ESC50.

----

## [471] Parameter Efficient Multimodal Transformers for Video Representation Learning

**Authors**: *Sangho Lee, Youngjae Yu, Gunhee Kim, Thomas M. Breuel, Jan Kautz, Yale Song*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6UdQLhqJyFD](https://openreview.net/forum?id=6UdQLhqJyFD)

**Abstract**:

The recent success of Transformers in the language domain has motivated adapting it to a multimodal setting, where a new visual model is trained in tandem with an already pretrained language model. However, due to the excessive memory requirements from Transformers, existing work typically fixes the language model and train only the vision module, which limits its ability to learn cross-modal information in an end-to-end manner. In this work, we focus on reducing the parameters of multimodal Transformers in the context of audio-visual video representation learning. We alleviate the high memory requirement by sharing the parameters of Transformers across layers and modalities; we decompose the Transformer into modality-specific and modality-shared parts so that the model learns the dynamics of each modality both individually and together, and propose a novel parameter sharing scheme based on low-rank approximation. We show that our approach reduces parameters of the Transformers up to 97%, allowing us to train our model end-to-end from scratch. We also propose a negative sampling approach based on an instance similarity measured on the CNN embedding space that our model learns together with the Transformers. To demonstrate our approach, we pretrain our model on 30-second clips (480 frames) from Kinetics-700 and transfer it to audio-visual classification tasks.

----

## [472] Robust Pruning at Initialization

**Authors**: *Soufiane Hayou, Jean-Francois Ton, Arnaud Doucet, Yee Whye Teh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vXj_ucZQ4hA](https://openreview.net/forum?id=vXj_ucZQ4hA)

**Abstract**:

Overparameterized Neural Networks (NN) display state-of-the-art performance. However, there is a growing need for smaller, energy-efficient, neural networks to be able to use machine learning applications on devices with limited computational resources. A popular approach consists of using pruning techniques. While these techniques have traditionally focused on pruning pre-trained NN (LeCun et al.,1990; Hassibi et al., 1993), recent work by Lee et al. (2018) has shown promising results when pruning at initialization. However, for Deep NNs, such procedures remain unsatisfactory as the resulting pruned networks can be difficult to train and, for instance, they do not prevent one layer from being fully pruned. In this paper, we provide a comprehensive theoretical analysis of Magnitude and Gradient based pruning at initialization and training of sparse architectures.  This allows us to propose novel principled approaches which we validate experimentally on a variety of NN architectures.

----

## [473] Efficient Wasserstein Natural Gradients for Reinforcement Learning

**Authors**: *Ted Moskovitz, Michael Arbel, Ferenc Huszar, Arthur Gretton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OHgnfSrn2jv](https://openreview.net/forum?id=OHgnfSrn2jv)

**Abstract**:

A novel optimization approach is proposed for application to policy gradient methods and evolution strategies for reinforcement learning (RL). The procedure uses a computationally efficient \emph{Wasserstein natural gradient} (WNG) descent that takes advantage of the geometry induced by a Wasserstein penalty to speed optimization. This method follows the recent theme in RL of including divergence penalties in the objective to establish trust regions. Experiments on challenging tasks demonstrate improvements in both computational cost and performance over advanced baselines.

----

## [474] Probing BERT in Hyperbolic Spaces

**Authors**: *Boli Chen, Yao Fu, Guangwei Xu, Pengjun Xie, Chuanqi Tan, Mosha Chen, Liping Jing*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=17VnwXYZyhH](https://openreview.net/forum?id=17VnwXYZyhH)

**Abstract**:

Recently, a variety of probing tasks are proposed to discover linguistic properties learned in contextualized word embeddings. Many of these works implicitly assume these embeddings lay in certain metric spaces, typically the Euclidean space. This work considers a family of geometrically special spaces, the hyperbolic spaces, that exhibit better inductive biases for hierarchical structures and may better reveal linguistic hierarchies encoded in contextualized representations. We introduce a $\textit{Poincaré probe}$, a structural probe projecting these embeddings into a Poincaré subspace with explicitly defined hierarchies. We focus on two probing objectives: (a) dependency trees where the hierarchy is defined as head-dependent structures; (b) lexical sentiments where the hierarchy is defined as the polarity of words (positivity and negativity). We argue that a key desideratum of a probe is its sensitivity to the existence of linguistic structures. We apply our probes on BERT, a typical contextualized embedding model. In a syntactic subspace, our probe better recovers tree structures than Euclidean probes, revealing the possibility that the geometry of BERT syntax may not necessarily be Euclidean. In a sentiment subspace, we reveal two possible meta-embeddings for positive and negative sentiments and show how lexically-controlled contextualization would change the geometric localization of embeddings. We demonstrate the findings with our Poincaré probe via extensive experiments and visualization. Our results can be reproduced at https://github.com/FranxYao/PoincareProbe

----

## [475] On Fast Adversarial Robustness Adaptation in Model-Agnostic Meta-Learning

**Authors**: *Ren Wang, Kaidi Xu, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Chuang Gan, Meng Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=o81ZyBCojoA](https://openreview.net/forum?id=o81ZyBCojoA)

**Abstract**:

Model-agnostic meta-learning (MAML) has emerged as one of the most successful meta-learning techniques in few-shot learning. It enables us to learn a $\textit{meta-initialization}$ of model parameters (that we call $\textit{meta-model}$) to rapidly adapt to new tasks using a small amount of labeled training data. Despite the generalization power of the meta-model, it remains elusive that how $\textit{adversarial robustness}$ can be maintained by MAML in few-shot learning. In addition to generalization, robustness is also desired for a meta-model to defend adversarial examples (attacks). Toward promoting adversarial robustness in MAML, we first study $\textit{when}$ a robustness-promoting regularization should be incorporated, given the fact that MAML adopts a bi-level (fine-tuning vs. meta-update) learning procedure. We show that robustifying the meta-update stage is sufficient to make robustness adapted to the task-specific fine-tuning stage even if the latter uses a standard training protocol. We also make additional justification on the acquired robustness adaptation by peering into the interpretability of neurons' activation maps. Furthermore, we investigate $\textit{how}$ robust regularization can $\textit{efficiently}$ be designed in MAML. We propose a general but easily-optimized robustness-regularized meta-learning framework, which allows the use of unlabeled data augmentation, fast adversarial attack generation, and computationally-light fine-tuning. In particular, we for the first time show that the auxiliary contrastive learning task can enhance the adversarial robustness of MAML. Finally, extensive experiments are conducted to demonstrate the effectiveness of our proposed methods in robust few-shot learning.

----

## [476] Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics

**Authors**: *Vinay Venkatesh Ramasesh, Ethan Dyer, Maithra Raghu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LhY8QdUGSuw](https://openreview.net/forum?id=LhY8QdUGSuw)

**Abstract**:

Catastrophic forgetting is a recurring challenge to developing versatile deep learning models. Despite its ubiquity, there is limited understanding of its connections to neural network (hidden) representations and task semantics. In this paper, we address this important knowledge gap. Through quantitative analysis of neural representations, we find that deeper layers are disproportionately responsible for forgetting, with sequential training resulting in an erasure of earlier task representational subspaces. Methods to mitigate forgetting stabilize these deeper layers, but show diversity on precise effects, with some increasing feature reuse while others store task representations orthogonally, preventing interference. These insights also enable the development of an analytic argument and empirical picture relating forgetting to task semantic similarity, where we find that maximal forgetting occurs for task sequences with intermediate similarity.

----

## [477] Trusted Multi-View Classification

**Authors**: *Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OOsR8BzCnl5](https://openreview.net/forum?id=OOsR8BzCnl5)

**Abstract**:

Multi-view classification (MVC) generally focuses on improving classification accuracy by using information from different views, typically integrating them into a unified comprehensive representation for downstream tasks. However, it is also crucial to dynamically assess the quality of a view for different samples in order to provide reliable uncertainty estimations, which indicate whether predictions can be trusted. To this end, we propose a novel multi-view classification method, termed trusted multi-view classification, which provides a new paradigm for multi-view learning by dynamically integrating different views at an evidence level. The algorithm jointly utilizes multiple views to promote both classification reliability (uncertainty estimation during testing) and robustness (out-of-distribution-awareness during training) by integrating evidence from each view. To achieve this, the Dirichlet distribution is used to model the distribution of the class probabilities, parameterized with evidence from different views and integrated with the Dempster-Shafer theory. The unified learning framework induces accurate uncertainty and accordingly endows the model with both reliability and robustness for out-of-distribution samples. Extensive experimental results validate the effectiveness of the proposed model in accuracy, reliability and robustness.

----

## [478] i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning

**Authors**: *Kibok Lee, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, Honglak Lee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=T6AxtOaWydQ](https://openreview.net/forum?id=T6AxtOaWydQ)

**Abstract**:

Contrastive representation learning has shown to be effective to learn representations from unlabeled data. However, much progress has been made in vision domains relying on data augmentations carefully designed using domain knowledge. In this work, we propose i-Mix, a simple yet effective domain-agnostic regularization strategy for improving contrastive representation learning. We cast contrastive learning as training a non-parametric classifier by assigning a unique virtual class to each data in a batch. Then, data instances are mixed in both the input and virtual label spaces, providing more augmented data during training. In experiments, we demonstrate that i-Mix consistently improves the quality of learned representations across domains, including image, speech, and tabular data. Furthermore, we confirm its regularization effect via extensive ablation studies across model and dataset sizes. The code is available at https://github.com/kibok90/imix.

----

## [479] Initialization and Regularization of Factorized Neural Layers

**Authors**: *Mikhail Khodak, Neil A. Tenenholtz, Lester Mackey, Nicolò Fusi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KTlJT1nof6d](https://openreview.net/forum?id=KTlJT1nof6d)

**Abstract**:

Factorized layers—operations parameterized by products of two or more matrices—occur in a variety of deep learning contexts, including compressed model training, certain types of knowledge distillation, and multi-head self-attention architectures. We study how to initialize and regularize deep nets containing such layers, examining two simple, understudied schemes, spectral initialization and Frobenius decay, for improving their performance. The guiding insight is to design optimization routines for these networks that are as close as possible to that of their well-tuned, non-decomposed counterparts; we back this intuition with an analysis of how the initialization and regularization schemes impact training with gradient descent, drawing on modern attempts to understand the interplay of weight-decay and batch-normalization. Empirically, we highlight the benefits of spectral initialization and Frobenius decay across a variety of settings. In model compression, we show that they enable low-rank methods to significantly outperform both unstructured sparsity and tensor methods on the task of training low-memory residual networks; analogs of the schemes also improve the performance of tensor decomposition techniques. For knowledge distillation, Frobenius decay enables a simple, overcomplete baseline that yields a compact model from over-parameterized training without requiring retraining with or pruning a teacher network. Finally, we show how both schemes applied to multi-head attention lead to improved performance on both translation and unsupervised pre-training.

----

## [480] Learning to Generate 3D Shapes with Generative Cellular Automata

**Authors**: *Dongsu Zhang, Changwoon Choi, Jeonghwan Kim, Young Min Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rABUmU3ulQh](https://openreview.net/forum?id=rABUmU3ulQh)

**Abstract**:

In this work, we present a probabilistic 3D generative model, named Generative Cellular Automata, which is able to produce diverse and high quality shapes. We formulate the shape generation process as sampling from the transition kernel of a Markov chain, where the sampling chain eventually evolves to the full shape of the learned distribution. The transition kernel employs the local update rules of cellular automata, effectively reducing the search space in a high-resolution 3D grid space by exploiting the connectivity and sparsity of 3D shapes. Our progressive generation only focuses on the sparse set of occupied voxels and their neighborhood, thus enables the utilization of an expressive sparse convolutional network. We propose an effective training scheme to obtain the local homogeneous rule of generative cellular automata with sequences that are slightly different from the sampling chain but converge to the full shapes in the training data. Extensive experiments on probabilistic shape completion and shape generation demonstrate that our method achieves competitive performance against recent methods.

----

## [481] Self-Supervised Learning of Compressed Video Representations

**Authors**: *Youngjae Yu, Sangho Lee, Gunhee Kim, Yale Song*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jMPcEkJpdD](https://openreview.net/forum?id=jMPcEkJpdD)

**Abstract**:

Self-supervised learning of video representations has received great attention. Existing methods typically require frames to be decoded before being processed, which increases compute and storage requirements and ultimately hinders large-scale training. In this work, we propose an efficient self-supervised approach to learn video representations by eliminating the expensive decoding step. We use a three-stream video architecture that encodes I-frames and P-frames of a compressed video. Unlike existing approaches that encode I-frames and P-frames individually, we propose to jointly encode them by establishing bidirectional dynamic connections across streams. To enable self-supervised learning, we propose two pretext tasks that leverage the multimodal nature (RGB, motion vector, residuals) and the internal GOP structure of compressed videos. The first task asks our network to predict zeroth-order motion statistics in a spatio-temporal pyramid; the second task asks correspondence types between I-frames and P-frames after applying temporal transformations. We show that our approach achieves competitive performance on compressed video recognition both in supervised and self-supervised regimes.

----

## [482] Factorizing Declarative and Procedural Knowledge in Structured, Dynamical Environments

**Authors**: *Anirudh Goyal, Alex Lamb, Phanideep Gampa, Philippe Beaudoin, Charles Blundell, Sergey Levine, Yoshua Bengio, Michael Curtis Mozer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VVdmjgu7pKM](https://openreview.net/forum?id=VVdmjgu7pKM)

**Abstract**:

Modeling a structured, dynamic environment like a video game requires keeping track of the objects and their states (declarative knowledge) as well as predicting how objects behave (procedural knowledge). Black-box models with a monolithic hidden state often fail to apply procedural knowledge consistently and uniformly, i.e., they lack systematicity. For example, in a video game, correct prediction of one enemy's trajectory does not ensure correct prediction of another's. We address this issue via an architecture that factorizes declarative and procedural knowledge and that imposes modularity within each form of knowledge. The architecture consists of active modules called object files that maintain the state of a single object and invoke passive external knowledge sources called schemata that prescribe state updates. To use a video game as an illustration, two enemies of the same type will share schemata but will have separate object files to encode their distinct state (e.g., health, position). We propose to use attention to determine which object files to update, the selection of schemata, and the propagation of information between object files. The resulting architecture is a drop-in replacement conforming to the same input-output interface as normal recurrent networks (e.g., LSTM, GRU) yet achieves substantially better generalization on environments that have multiple object tokens of the same type, including a challenging intuitive physics benchmark.

----

## [483] Cut out the annotator, keep the cutout: better segmentation with weak supervision

**Authors**: *Sarah M. Hooper, Michael Wornow, Ying Hang Seah, Peter Kellman, Hui Xue, Frederic Sala, Curtis P. Langlotz, Christopher Ré*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bjkX6Kzb5H](https://openreview.net/forum?id=bjkX6Kzb5H)

**Abstract**:

Constructing large, labeled training datasets for segmentation models is an expensive and labor-intensive process. This is a common challenge in machine learning, addressed by methods that require few or no labeled data points such as few-shot learning (FSL) and weakly-supervised learning (WS). Such techniques, however, have limitations when applied to image segmentation---FSL methods often produce noisy results and are strongly dependent on which few datapoints are labeled, while WS models struggle to fully exploit rich image information. We propose a framework that fuses FSL and WS for segmentation tasks, enabling users to train high-performing segmentation networks with very few hand-labeled training points. We use FSL models as weak sources in a WS framework, requiring a very small set of reference labeled images, and introduce a new WS model that focuses on key areas---areas with contention among noisy labels---of the image to fuse these weak sources. Empirically, we evaluate our proposed approach over seven well-motivated segmentation tasks. We show that our methods can achieve within 1.4 Dice points compared to fully supervised networks while only requiring five hand-labeled training points. Compared to existing FSL methods, our approach improves performance by a mean 3.6 Dice points over the next-best method.

----

## [484] FastSpeech 2: Fast and High-Quality End-to-End Text to Speech

**Authors**: *Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=piLPYqxtWuA](https://openreview.net/forum?id=piLPYqxtWuA)

**Abstract**:

Non-autoregressive text to speech (TTS) models such as FastSpeech can synthesize speech significantly faster than previous autoregressive models with comparable quality. The training of FastSpeech model relies on an autoregressive teacher model for duration prediction (to provide more information as input) and knowledge distillation (to simplify the data distribution in output), which can ease the one-to-many mapping problem (i.e., multiple speech variations correspond to the same text) in TTS. However, FastSpeech has several disadvantages: 1) the teacher-student distillation pipeline is complicated and time-consuming, 2) the duration extracted from the teacher model is not accurate enough, and the target mel-spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality. In this paper, we propose FastSpeech 2, which addresses the issues in FastSpeech and better solves the one-to-many mapping problem in TTS by 1) directly training the model with ground-truth target instead of the simplified output from teacher, and 2) introducing more variation information of speech (e.g., pitch, energy and more accurate duration) as conditional inputs. Specifically, we extract duration, pitch and energy from speech waveform and directly take them as conditional inputs in training and use predicted values in inference. We further design FastSpeech 2s, which is the first attempt to directly generate speech waveform from text in parallel, enjoying the benefit of fully end-to-end inference. Experimental results show that 1) FastSpeech 2 achieves a 3x training speed-up over FastSpeech, and FastSpeech 2s enjoys even faster inference speed; 2) FastSpeech 2 and 2s outperform FastSpeech in voice quality, and FastSpeech 2 can even surpass autoregressive models. Audio samples are available at https://speechresearch.github.io/fastspeech2/.

----

## [485] On Learning Universal Representations Across Languages

**Authors**: *Xiangpeng Wei, Rongxiang Weng, Yue Hu, Luxi Xing, Heng Yu, Weihua Luo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Uu1Nw-eeTxJ](https://openreview.net/forum?id=Uu1Nw-eeTxJ)

**Abstract**:

Recent studies have demonstrated the overwhelming advantage of cross-lingual pre-trained models (PTMs), such as multilingual BERT and XLM, on cross-lingual NLP tasks. However, existing approaches essentially capture the co-occurrence among tokens through involving the masked language model (MLM) objective with token-level cross entropy. In this work, we extend these approaches to learn sentence-level representations and show the effectiveness on cross-lingual understanding and generation. Specifically, we propose a Hierarchical Contrastive Learning (HiCTL) method to (1) learn universal representations for parallel sentences distributed in one or multiple languages and (2) distinguish the semantically-related words from a shared cross-lingual vocabulary for each sentence. We conduct evaluations on two challenging cross-lingual tasks, XTREME and machine translation. Experimental results show that the HiCTL outperforms the state-of-the-art XLM-R by an absolute gain of 4.2% accuracy on the XTREME benchmark as well as achieves substantial improvements on both of the high resource and low-resource English$\rightarrow$X translation tasks over strong baselines.

----

## [486] Effective Distributed Learning with Random Features: Improved Bounds and Algorithms

**Authors**: *Yong Liu, Jiankun Liu, Shuqiang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jxdXSW9Doc](https://openreview.net/forum?id=jxdXSW9Doc)

**Abstract**:

In this paper, we study the statistical properties of distributed kernel ridge regression together with random features (DKRR-RF), and obtain optimal generalization bounds under the basic setting, which can substantially relax the restriction on the number of local machines in the existing state-of-art bounds. Specifically, we first show that the simple combination of divide-and-conquer technique and random features can achieve the same statistical accuracy as the exact KRR in expectation requiring only $\mathcal{O}(|\mathcal{D}|)$ memory and $\mathcal{O}(|\mathcal{D}|^{1.5})$ time. Then, beyond the generalization bounds in expectation that demonstrate the average information for multiple trails, we derive generalization bounds in probability to capture the learning performance for a single trail. Finally, we propose an effective communication strategy to further improve the performance of DKRR-RF, and validate the theoretical bounds via numerical experiments.

----

## [487] Multi-Class Uncertainty Calibration via Mutual Information Maximization-based Binning

**Authors**: *Kanil Patel, William H. Beluch, Bin Yang, Michael Pfeiffer, Dan Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AICNpd8ke-m](https://openreview.net/forum?id=AICNpd8ke-m)

**Abstract**:

Post-hoc multi-class calibration is a common approach for providing high-quality confidence estimates of deep neural network predictions. Recent work has shown that widely used scaling methods underestimate their calibration error, while alternative Histogram Binning (HB) methods often fail to preserve classification accuracy. When classes have small prior probabilities, HB also faces the issue of severe sample-inefficiency after the conversion into K one-vs-rest class-wise calibration problems. The goal of this paper is to resolve the identified issues of HB in order to provide calibrated confidence estimates using only a small holdout calibration dataset for bin optimization while preserving multi-class ranking accuracy. From an information-theoretic perspective, we derive the I-Max concept for binning, which maximizes the mutual information between labels and quantized logits. This concept mitigates potential loss in ranking performance due to lossy quantization, and by disentangling the optimization of bin edges and representatives allows simultaneous improvement of ranking and calibration performance. To improve the sample efficiency and estimates from a small calibration set, we propose a shared class-wise (sCW) calibration strategy, sharing one calibrator among similar classes (e.g., with similar class priors) so that the training sets of their class-wise calibration problems can be merged to train the single calibrator. The combination of sCW and I-Max binning outperforms the state of the art calibration methods on various evaluation metrics across different benchmark datasets and models, using a small calibration set (e.g., 1k samples for ImageNet).

----

## [488] Neural ODE Processes

**Authors**: *Alexander Norcliffe, Cristian Bodnar, Ben Day, Jacob Moss, Pietro Liò*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=27acGyyI1BY](https://openreview.net/forum?id=27acGyyI1BY)

**Abstract**:

Neural Ordinary Differential Equations (NODEs) use a neural network to model the instantaneous rate of change in the state of a system. However, despite their apparent suitability for dynamics-governed time-series, NODEs present a few disadvantages. First, they are unable to adapt to incoming data-points, a fundamental requirement for real-time applications imposed by the natural direction of time. Second, time-series are often composed of a sparse set of measurements that could be explained by many possible underlying dynamics. NODEs do not capture this uncertainty. In contrast, Neural Processes (NPs) are a new class of stochastic processes providing uncertainty estimation and fast data-adaptation, but lack an explicit treatment of the flow of time. To address these problems, we introduce Neural ODE Processes (NDPs), a new class of stochastic processes determined by a distribution over Neural ODEs. By maintaining an adaptive data-dependent distribution over the underlying ODE, we show that our model can successfully capture the dynamics of low-dimensional systems from just a few data-points. At the same time, we demonstrate that NDPs scale up to challenging high-dimensional time-series with unknown latent dynamics such as rotating MNIST digits.

----

## [489] Conformation-Guided Molecular Representation with Hamiltonian Neural Networks

**Authors**: *Ziyao Li, Shuwen Yang, Guojie Song, Lingsheng Cai*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=q-cnWaaoUTH](https://openreview.net/forum?id=q-cnWaaoUTH)

**Abstract**:

Well-designed molecular representations (fingerprints) are vital to combine medical chemistry and deep learning. Whereas incorporating 3D geometry of molecules (i.e. conformations) in their representations seems beneficial, current 3D algorithms are still in infancy. In this paper, we propose a novel molecular representation algorithm which preserves 3D conformations of molecules with a Molecular Hamiltonian Network (HamNet). In HamNet, implicit positions and momentums of atoms in a molecule interact in the Hamiltonian Engine following the discretized Hamiltonian equations. These implicit coordinations are supervised with real conformations with translation- & rotation-invariant losses, and further used as inputs to the Fingerprint Generator, a message-passing neural network. Experiments show that the Hamiltonian Engine can well preserve molecular conformations, and that the fingerprints generated by HamNet achieve state-of-the-art performances on MoleculeNet, a standard molecular machine learning benchmark.

----

## [490] An Unsupervised Deep Learning Approach for Real-World Image Denoising

**Authors**: *Dihan Zheng, Sia Huat Tan, Xiaowen Zhang, Zuoqiang Shi, Kaisheng Ma, Chenglong Bao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tIjRAiFmU3y](https://openreview.net/forum?id=tIjRAiFmU3y)

**Abstract**:

Designing an unsupervised image denoising approach in practical applications is a challenging task due to the complicated data acquisition process. In the real-world case, the noise distribution is so complex that the simplified additive white Gaussian (AWGN) assumption rarely holds, which significantly deteriorates the Gaussian denoisers' performance. To address this problem, we apply a deep neural network that maps the noisy image into a latent space in which the AWGN assumption holds, and thus any existing Gaussian denoiser is applicable. More specifically, the proposed neural network consists of the encoder-decoder structure and approximates the likelihood term in the Bayesian framework. Together with a Gaussian denoiser, the neural network can be trained with the input image itself and does not require any pre-training in other datasets. Extensive experiments on real-world noisy image datasets have shown that the combination of neural networks and Gaussian denoisers improves the performance of the original Gaussian denoisers by a large margin. In particular, the neural network+BM3D method significantly outperforms other unsupervised denoising approaches and is competitive with supervised networks such as DnCNN, FFDNet, and CBDNet.

----

## [491] Uncertainty in Gradient Boosting via Ensembles

**Authors**: *Andrey Malinin, Liudmila Prokhorenkova, Aleksei Ustimenko*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1Jv6b0Zq3qi](https://openreview.net/forum?id=1Jv6b0Zq3qi)

**Abstract**:

For many practical, high-risk applications, it is essential to quantify uncertainty in a model's predictions to avoid costly mistakes. While predictive uncertainty is widely studied for neural networks, the topic seems to be under-explored for models based on gradient boosting. However, gradient boosting often achieves state-of-the-art results on tabular data. This work examines a probabilistic ensemble-based framework for deriving uncertainty estimates in the predictions of gradient boosting classification and regression models. We conducted experiments on a range of synthetic and real datasets and investigated the applicability of ensemble approaches to gradient boosting models that are themselves ensembles of decision trees. Our analysis shows that ensembles of gradient boosting models successfully detect anomalous inputs while having limited ability to improve the predicted total uncertainty. Importantly, we also propose a concept of a virtual ensemble to get the benefits of an ensemble via only one gradient boosting model, which significantly reduces complexity.

----

## [492] Lossless Compression of Structured Convolutional Models via Lifting

**Authors**: *Gustav Sourek, Filip Zelezný, Ondrej Kuzelka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=oxnp2q-PGL4](https://openreview.net/forum?id=oxnp2q-PGL4)

**Abstract**:

Lifting is an efficient technique to scale up graphical models generalized to relational domains by exploiting the underlying symmetries. Concurrently, neural models are continuously expanding from grid-like tensor data into structured representations, such as various attributed graphs and relational databases. To address the irregular structure of the data, the models typically extrapolate on the idea of convolution, effectively introducing parameter sharing in their, dynamically unfolded, computation graphs. The computation graphs themselves then reflect the symmetries of the underlying data, similarly to the lifted graphical models. Inspired by lifting, we introduce a simple and efficient technique to detect the symmetries and compress the neural models without loss of any information. We demonstrate through experiments that such compression can lead to significant speedups of structured convolutional models, such as various Graph Neural Networks, across various tasks, such as molecule classification and knowledge-base completion.

----

## [493] Neural networks with late-phase weights

**Authors**: *Johannes von Oswald, Seijin Kobayashi, João Sacramento, Alexander Meulemans, Christian Henning, Benjamin F. Grewe*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=C0qJUx5dxFb](https://openreview.net/forum?id=C0qJUx5dxFb)

**Abstract**:

The largely successful method of training neural networks is to learn their weights using some variant of stochastic gradient descent (SGD). Here, we show that the solutions found by SGD can be further improved by ensembling a subset of the weights in late stages of learning. At the end of learning, we obtain back a single model by taking a spatial average in weight space. To avoid incurring increased computational costs, we investigate a family of low-dimensional late-phase weight models which interact multiplicatively with the remaining parameters. Our results show that augmenting standard models with late-phase weights improves generalization in established benchmarks such as CIFAR-10/100, ImageNet and enwik8. These findings are complemented with a theoretical analysis of a noisy quadratic problem which provides a simplified picture of the late phases of neural network learning.

----

## [494] Disambiguating Symbolic Expressions in Informal Documents

**Authors**: *Dennis Müller, Cezary Kaliszyk*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=K5j7D81ABvt](https://openreview.net/forum?id=K5j7D81ABvt)

**Abstract**:

We propose the task of \emph{disambiguating} symbolic expressions in informal STEM documents in the form of \LaTeX files -- that is, determining their precise semantics and abstract syntax tree -- as a neural machine translation task. We discuss the distinct challenges involved and present a dataset with roughly 33,000 entries. We evaluated several baseline models on this dataset, which failed to yield even syntactically valid \LaTeX before overfitting. Consequently, we describe a methodology using a \emph{transformer} language model pre-trained on sources obtained from \url{arxiv.org}, which yields promising results despite the small size of the dataset. We evaluate our model using a plurality of dedicated techniques, taking syntax and semantics of symbolic expressions into account.

----

## [495] Learning Parametrised Graph Shift Operators

**Authors**: *George Dasoulas, Johannes F. Lutzeyer, Michalis Vazirgiannis*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0OlrLvrsHwQ](https://openreview.net/forum?id=0OlrLvrsHwQ)

**Abstract**:

In many domains data is currently represented as graphs and therefore, the graph representation of this data becomes increasingly important in machine learning. Network data is, implicitly or explicitly, always represented using a graph shift operator (GSO) with the most common choices being the adjacency, Laplacian matrices and their normalisations. In this paper, a novel parametrised GSO (PGSO) is proposed, where specific parameter values result in the most commonly used GSOs and message-passing operators in graph neural network (GNN) frameworks. The PGSO is suggested as a replacement of the standard GSOs that are used in state-of-the-art GNN architectures and the optimisation of the PGSO parameters is seamlessly included in the model training. It is proved that the PGSO has real eigenvalues and a set of real eigenvectors independent of the parameter values and spectral bounds on the PGSO are derived. PGSO parameters are shown to adapt to the sparsity of the graph structure in a study on stochastic blockmodel networks, where they are found to automatically replicate the GSO regularisation found in the literature. On several real-world datasets the accuracy of state-of-the-art GNN architectures is improved by the inclusion of the PGSO in both node- and graph-classification tasks.

----

## [496] Efficient Conformal Prediction via Cascaded Inference with Expanded Admission

**Authors**: *Adam Fisch, Tal Schuster, Tommi S. Jaakkola, Regina Barzilay*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tnSo6VRLmT](https://openreview.net/forum?id=tnSo6VRLmT)

**Abstract**:

In this paper, we present a novel approach for conformal prediction (CP), in which we aim to identify a set of promising prediction candidates---in place of a single prediction. This set is guaranteed to contain a correct answer with high probability, and is well-suited for many open-ended classification tasks. In the standard CP paradigm, the predicted set can often be unusably large and also costly to obtain. This is particularly pervasive in settings where the correct answer is not unique, and the number of total possible answers is high. We first expand the CP correctness criterion to allow for additional, inferred "admissible" answers, which can substantially reduce the size of the predicted set while still providing valid performance guarantees. Second, we amortize costs by conformalizing prediction cascades, in which we aggressively prune implausible labels early on by using progressively stronger classifiers---again, while still providing valid performance guarantees. We demonstrate the empirical effectiveness of our approach for multiple applications in natural language processing and computational chemistry for drug discovery.

----

## [497] GANs Can Play Lottery Tickets Too

**Authors**: *Xuxi Chen, Zhenyu Zhang, Yongduo Sui, Tianlong Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1AoMhc_9jER](https://openreview.net/forum?id=1AoMhc_9jER)

**Abstract**:

Deep generative adversarial networks (GANs) have gained growing popularity in numerous scenarios, while usually suffer from high parameter complexities for resource-constrained real-world applications. However, the compression of GANs has less been explored. A few works show that heuristically applying compression techniques normally leads to unsatisfactory results, due to the notorious training instability of GANs. In parallel, the lottery ticket hypothesis shows prevailing success on discriminative models, in locating sparse matching subnetworks capable of training in isolation to full model performance. In this work, we for the first time study the existence of such trainable matching subnetworks in deep GANs. For a range of GANs, we certainly find matching subnetworks at $67\%$-$74\%$ sparsity. We observe that with or without pruning discriminator has a minor effect on the existence and quality of matching subnetworks, while the initialization weights used in the discriminator plays a significant role. We then show the powerful transferability of these subnetworks to unseen tasks. Furthermore, extensive experimental results demonstrate that our found subnetworks substantially outperform previous state-of-the-art GAN compression approaches in both image generation (e.g. SNGAN) and image-to-image translation GANs (e.g. CycleGAN). Codes available at https://github.com/VITA-Group/GAN-LTH.

----

## [498] ResNet After All: Neural ODEs and Their Numerical Solution

**Authors**: *Katharina Ott, Prateek Katiyar, Philipp Hennig, Michael Tiemann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HxzSxSxLOJZ](https://openreview.net/forum?id=HxzSxSxLOJZ)

**Abstract**:

A key appeal of the recently proposed Neural Ordinary Differential Equation (ODE) framework is that it seems to provide a continuous-time extension of discrete residual neural networks. 
As we show herein, though, trained Neural ODE models actually depend on the specific numerical method used during training.
If the trained model is supposed to be a flow generated from an ODE, it should be possible to choose another numerical solver with equal or smaller numerical error without loss of performance.
We observe that if training relies on a solver with overly coarse discretization, then testing with another solver of equal or smaller numerical error results in a sharp drop in accuracy. 
In such cases, the combination of vector field and numerical method cannot be interpreted as a flow generated from an ODE, which arguably poses a fatal breakdown of the Neural ODE concept.
We observe, however, that there exists a critical step size beyond which the training yields a valid ODE vector field. 
We propose a method that monitors the behavior of the ODE solver during training to adapt its step size, aiming to ensure a valid ODE without unnecessarily increasing computational cost.
We verify this adaption algorithm on a common bench mark dataset as well as a synthetic dataset.

----

## [499] Semantic Re-tuning with Contrastive Tension

**Authors**: *Fredrik Carlsson, Amaru Cuba Gyllensten, Evangelia Gogoulou, Erik Ylipää Hellqvist, Magnus Sahlgren*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ov_sMNau-PF](https://openreview.net/forum?id=Ov_sMNau-PF)

**Abstract**:

Extracting semantically useful natural language sentence representations from pre-trained deep neural networks such as Transformers remains a challenge. We first demonstrate that pre-training objectives impose a significant task bias onto the final layers of models with a layer-wise survey of the Semantic Textual Similarity (STS) correlations for multiple common Transformer language models. We then propose a new self-supervised method called Contrastive Tension (CT) to counter such biases. CT frames the training objective as a noise-contrastive task between the final layer representations of two independent models, in turn making the final layer representations suitable for feature extraction. Results from multiple common unsupervised and supervised STS tasks indicate that CT outperforms previous State Of The Art (SOTA), and when combining CT with supervised data we improve upon previous SOTA results with large margins.

----

## [500] Property Controllable Variational Autoencoder via Invertible Mutual Dependence

**Authors**: *Xiaojie Guo, Yuanqi Du, Liang Zhao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tYxG_OMs9WE](https://openreview.net/forum?id=tYxG_OMs9WE)

**Abstract**:

Deep generative models have made important progress towards modeling complex, high dimensional data via learning latent representations. Their usefulness is nevertheless often limited by a lack of control over the generative process or a poor understanding of the latent representation. To overcome these issues, attention is now focused on discovering latent variables correlated to the data properties and ways to manipulate these properties. This paper presents the new Property controllable VAE (PCVAE), where a new Bayesian model is proposed to inductively bias the latent representation using explicit data properties via novel group-wise and property-wise disentanglement. Each data property corresponds seamlessly to a latent variable, by innovatively enforcing invertible mutual dependence between them. This allows us to move along the learned latent dimensions to control specific properties of the generated data with great precision. Quantitative and qualitative evaluations confirm that the PCVAE outperforms the existing models by up to 28% in capturing and 65% in manipulating the desired properties.

----

## [501] Latent Convergent Cross Mapping

**Authors**: *Edward De Brouwer, Adam Arany, Jaak Simm, Yves Moreau*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4TSiOTkKe5P](https://openreview.net/forum?id=4TSiOTkKe5P)

**Abstract**:

Discovering causal structures of temporal processes is a major tool of scientific inquiry because it helps us better understand and explain the mechanisms driving a phenomenon of interest, thereby facilitating analysis, reasoning, and synthesis for such systems. 
However, accurately inferring causal structures within a phenomenon based on observational data only is still an open problem. Indeed, this type of data usually consists in short time series with missing or noisy values for which causal inference is increasingly difficult. In this work, we propose a method to uncover causal relations in chaotic dynamical systems from short, noisy and sporadic time series (that is, incomplete observations at infrequent and irregular intervals) where the classical convergent cross mapping (CCM) fails. Our method works by learning a Neural ODE latent process modeling the state-space dynamics of the time series and by checking the existence of a continuous map between the resulting processes. We provide theoretical analysis and show empirically that Latent-CCM can reliably uncover the true causal pattern, unlike traditional methods.

----

## [502] Adaptive Universal Generalized PageRank Graph Neural Network

**Authors**: *Eli Chien, Jianhao Peng, Pan Li, Olgica Milenkovic*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=n6jl7fLxrP](https://openreview.net/forum?id=n6jl7fLxrP)

**Abstract**:

In many important graph data processing applications the acquired information includes both node features and observations of the graph topology. Graph neural networks (GNNs) are designed to exploit both sources of evidence but they do not optimally trade-off their utility and integrate them in a manner that is also universal. Here, universality refers to independence on homophily or heterophily graph assumptions. We address these issues by introducing a new Generalized PageRank (GPR) GNN architecture that adaptively learns the GPR weights so as to jointly optimize node feature and topological information extraction, regardless of the extent to which the node labels are homophilic or heterophilic. Learned GPR weights automatically adjust to the node label pattern, irrelevant on the type of initialization, and thereby guarantee excellent learning performance for label patterns that are usually hard to handle. Furthermore, they allow one to avoid feature over-smoothing, a process which renders feature information nondiscriminative, without requiring the network to be shallow. Our accompanying theoretical analysis of the GPR-GNN method is facilitated by novel synthetic benchmark datasets generated by the so-called contextual stochastic block model. We also compare the performance of our GNN architecture with that of several state-of-the-art GNNs on the problem of node-classification, using well-known benchmark homophilic and heterophilic datasets. The results demonstrate that GPR-GNN offers significant performance improvement compared to existing techniques on both synthetic and benchmark data.

----

## [503] Neural Learning of One-of-Many Solutions for Combinatorial Problems in Structured Output Spaces

**Authors**: *Yatin Nandwani, Deepanshu Jindal, Mausam, Parag Singla*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ATp1nW2FuZL](https://openreview.net/forum?id=ATp1nW2FuZL)

**Abstract**:

Recent research has proposed neural architectures for solving combinatorial problems in structured output spaces. In many such problems, there may exist multiple solutions for a given input, e.g. a partially filled Sudoku puzzle may have many completions satisfying all constraints. Further, we are often interested in finding any "one" of the possible solutions, without any preference between them. Existing approaches completely ignore this solution multiplicity. In this paper, we argue that being oblivious to the presence of multiple solutions can severely hamper their training ability. Our contribution is two-fold. First, we formally define the task of learning one-of-many solutions for combinatorial problems in structured output spaces, which is applicable for solving several problems of interest such as N-Queens, and Sudoku. Second, we present a generic learning framework that adapts an existing prediction network for a combinatorial problem to handle solution multiplicity. Our framework uses a selection module, whose goal is to dynamically determine, for every input, the solution that is most effective for training the network parameters in any given learning iteration. We propose an RL based approach to jointly train the selection module with the prediction network. Experiments on three different domains, and using two different prediction networks,  demonstrate that our framework significantly improves the accuracy in our setting, obtaining up to 21 pt gain over the baselines.

----

## [504] My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control

**Authors**: *Vitaly Kurin, Maximilian Igl, Tim Rocktäschel, Wendelin Boehmer, Shimon Whiteson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=N3zUDGN5lO](https://openreview.net/forum?id=N3zUDGN5lO)

**Abstract**:

Multitask Reinforcement Learning is a promising way to obtain models with better performance, generalisation, data efficiency, and robustness. Most existing work is limited to compatible settings, where the state and action space dimensions are the same across tasks. Graph Neural Networks (GNN) are one way to address incompatible environments, because they can process graphs of arbitrary size. They also allow practitioners to inject biases encoded in the structure of the input graph. Existing work in graph-based continuous control uses the physical morphology of the agent to construct the input graph, i.e., encoding limb features as node labels and using edges to connect the nodes if their corresponded limbs are physically connected.
In this work, we present a series of ablations on existing methods that show that morphological information encoded in the graph does not improve their performance. Motivated by the hypothesis that any benefits GNNs extract from the graph structure are outweighed by difficulties they create for message passing, we also propose Amorpheus, a transformer-based approach. Further results show that, while Amorpheus ignores the morphological information that GNNs encode, it nonetheless substantially outperforms GNN-based methods.

----

## [505] FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning

**Authors**: *Hong-You Chen, Wei-Lun Chao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dgtpE6gKjHn](https://openreview.net/forum?id=dgtpE6gKjHn)

**Abstract**:

Federated learning aims to collaboratively train a strong global model by accessing users' locally trained models but not their own data. A crucial step is therefore to aggregate local models into a global model, which has been shown challenging when users have non-i.i.d. data. In this paper, we propose a novel aggregation algorithm named FedBE, which takes a Bayesian inference perspective by sampling higher-quality global models and combining them via Bayesian model Ensemble, leading to much robust aggregation. We show that an effective model distribution can be constructed by simply fitting a Gaussian or Dirichlet distribution to the local models. Our empirical studies validate FedBE's superior performance, especially when users' data are not i.i.d. and when the neural networks go deeper. Moreover, FedBE is compatible with recent efforts in regularizing users' model training, making it an easily applicable module: you only need to replace the aggregation method but leave other parts of your federated learning algorithm intact.

----

## [506] MALI: A memory efficient and reverse accurate integrator for Neural ODEs

**Authors**: *Juntang Zhuang, Nicha C. Dvornek, Sekhar Tatikonda, James S. Duncan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=blfSjHeFM_e](https://openreview.net/forum?id=blfSjHeFM_e)

**Abstract**:

Neural ordinary differential equations (Neural ODEs) are a new family of deep-learning models with continuous depth. However, the numerical estimation of the gradient in the continuous case is not well solved: existing implementations of the adjoint method suffer from inaccuracy in reverse-time trajectory, while the naive method and the adaptive checkpoint adjoint method (ACA) have a memory cost that grows with integration time. In this project, based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost $w.r.t$ integration time similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation). We validate MALI in various tasks: on image recognition tasks, to our knowledge, MALI is the first to enable feasible training of a Neural ODE on ImageNet and outperform a well-tuned ResNet, while existing methods fail due to either heavy memory burden or inaccuracy; for time series modeling, MALI significantly outperforms the adjoint method; and for continuous generative models, MALI achieves new state-of-the-art performance. We provide a pypi package: https://jzkay12.github.io/TorchDiffEqPack

----

## [507] Reducing the Computational Cost of Deep Generative Models with Binary Neural Networks

**Authors**: *Thomas Bird, Friso H. Kingma, David Barber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sTeoJiB4uR](https://openreview.net/forum?id=sTeoJiB4uR)

**Abstract**:

Deep generative models provide a powerful set of tools to understand real-world data. But as these models improve, they increase in size and complexity, so their computational cost in memory and execution time grows. Using binary weights in neural networks is one method which has shown promise in reducing this cost. However, whether binary neural networks can be used in generative models is an open problem. In this work we show, for the first time, that we can successfully train generative models which utilize binary neural networks. This reduces the computational cost of the models massively. We develop a new class of binary weight normalization, and provide insights for architecture designs of these binarized generative models. We demonstrate that two state-of-the-art deep generative models, the ResNet VAE and Flow++ models, can be binarized effectively using these techniques. We train binary models that achieve loss values close to those of the regular models but are 90%-94% smaller in size, and also allow significant speed-ups in execution time.

----

## [508] In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness

**Authors**: *Sang Michael Xie, Ananya Kumar, Robbie Jones, Fereshte Khani, Tengyu Ma, Percy Liang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jznizqvr15J](https://openreview.net/forum?id=jznizqvr15J)

**Abstract**:

Consider a prediction setting with few in-distribution labeled examples and many unlabeled examples both in- and out-of-distribution (OOD). The goal is to learn a model which performs well both in-distribution and OOD. In these settings, auxiliary information is often cheaply available for every input. How should we best leverage this auxiliary information for the prediction task? Empirically across three image and time-series datasets, and theoretically in a multi-task linear regression setting, we show that (i) using auxiliary information as input features improves in-distribution error but can hurt OOD error; but (ii) using auxiliary information as outputs of auxiliary pre-training tasks improves OOD error. To get the best of both worlds, we introduce In-N-Out, which first trains a model with auxiliary inputs and uses it to pseudolabel all the in-distribution inputs, then pre-trains a model on OOD auxiliary outputs and fine-tunes this model with the pseudolabels (self-training). We show both theoretically and empirically that In-N-Out outperforms auxiliary inputs or outputs alone on both in-distribution and OOD error.

----

## [509] Incremental few-shot learning via vector quantization in deep embedded space

**Authors**: *Kuilin Chen, Chi-Guhn Lee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3SV-ZePhnZM](https://openreview.net/forum?id=3SV-ZePhnZM)

**Abstract**:

The capability of incrementally learning new tasks without forgetting old ones is a challenging problem due to catastrophic forgetting. This challenge becomes greater when novel tasks contain very few labelled training samples. Currently, most methods are dedicated to class-incremental learning and rely on sufficient training data to learn additional weights for newly added classes. Those methods cannot be easily extended to incremental regression tasks and could suffer from severe overfitting when learning few-shot novel tasks. In this study, we propose a nonparametric method in deep embedded space to tackle incremental few-shot learning problems. The knowledge about the learned tasks are compressed into a small number of quantized reference vectors. The proposed method learns new tasks sequentially by adding more reference vectors to the model using few-shot samples in each novel task. For classification problems, we employ the nearest neighbor scheme to make classification on sparsely available data and incorporate intra-class variation, less forgetting regularization and calibration of reference vectors to mitigate catastrophic forgetting. In addition, the proposed learning vector quantization (LVQ) in deep embedded space can be customized as a kernel smoother to handle incremental few-shot regression tasks. Experimental results demonstrate that the proposed method outperforms other state-of-the-art methods in incremental learning.

----

## [510] Contrastive Syn-to-Real Generalization

**Authors**: *Wuyang Chen, Zhiding Yu, Shalini De Mello, Sifei Liu, José M. Álvarez, Zhangyang Wang, Anima Anandkumar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=F8whUO8HNbP](https://openreview.net/forum?id=F8whUO8HNbP)

**Abstract**:

Training on synthetic data can be beneficial for label or data-scarce scenarios. However, synthetically trained models often suffer from poor generalization in real domains due to domain gaps. In this work, we make a key observation that the diversity of the learned feature embeddings plays an important role in the generalization performance. To this end, we propose contrastive synthetic-to-real generalization (CSG), a novel framework that leverage the pre-trained ImageNet knowledge to prevent overfitting to the synthetic domain, while promoting the diversity of feature embeddings as an inductive bias to improve generalization. In addition, we enhance the proposed CSG framework with attentional pooling (A-pool) to let the model focus on semantically important regions and further improve its generalization. We demonstrate the effectiveness of CSG on various synthetic training tasks, exhibiting state-of-the-art performance on zero-shot domain generalization.

----

## [511] Remembering for the Right Reasons: Explanations Reduce Catastrophic Forgetting

**Authors**: *Sayna Ebrahimi, Suzanne Petryk, Akash Gokul, William Gan, Joseph E. Gonzalez, Marcus Rohrbach, Trevor Darrell*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tHgJoMfy6nI](https://openreview.net/forum?id=tHgJoMfy6nI)

**Abstract**:

The goal of continual learning (CL) is to learn a sequence of tasks without suffering from the phenomenon of catastrophic forgetting. Previous work has shown that leveraging memory in the form of a replay buffer can reduce performance degradation on prior tasks. We hypothesize that forgetting can be further reduced when the model is encouraged to remember the \textit{evidence} for previously made decisions. As a first step towards exploring this hypothesis, we propose a simple novel training paradigm, called Remembering for the Right Reasons (RRR), that additionally stores visual model explanations for each example in the buffer and ensures the model has ``the right reasons'' for its predictions by encouraging its explanations to remain consistent with those used to make decisions at training time. Without this constraint, there is a drift in explanations and increase in forgetting as conventional continual learning algorithms learn new tasks. We demonstrate how RRR can be easily added to any memory or regularization-based approach and results in reduced forgetting, and more importantly, improved model explanations. We have evaluated our approach in the standard and few-shot settings and observed a consistent improvement across various CL approaches using different architectures and techniques to generate model explanations and demonstrated our approach showing a promising connection between explainability and continual learning. Our code is available at \url{https://github.com/SaynaEbrahimi/Remembering-for-the-Right-Reasons}.

----

## [512] Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online

**Authors**: *Yangchen Pan, Kirby Banman, Martha White*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zElset1Klrp](https://openreview.net/forum?id=zElset1Klrp)

**Abstract**:

Recent work has shown that sparse representations---where only a small percentage of units are active---can significantly reduce interference. Those works, however, relied on relatively complex regularization or meta-learning approaches, that have only been used offline in a pre-training phase. In this work, we pursue a direction that achieves sparsity by design, rather than by learning. Specifically, we design an activation function that produces sparse representations deterministically by construction, and so is more amenable to online training. The idea relies on the simple approach of binning, but overcomes the two key limitations of binning: zero gradients for the flat regions almost everywhere,  and lost precision---reduced discrimination---due to coarse aggregation. We introduce a Fuzzy Tiling Activation (FTA) that provides non-negligible gradients and produces overlap between bins that improves discrimination. We first show that FTA is robust under covariate shift in a synthetic online supervised learning problem, where we can vary the level of correlation and drift. Then we move to the deep reinforcement learning setting and investigate both value-based and policy gradient algorithms that use neural networks with FTAs, in classic discrete control and Mujoco continuous control environments. We show that algorithms equipped with FTAs are able to learn a stable policy faster without needing target networks on most domains.

----

## [513] High-Capacity Expert Binary Networks

**Authors**: *Adrian Bulat, Brais Martínez, Georgios Tzimiropoulos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MxaY4FzOTa](https://openreview.net/forum?id=MxaY4FzOTa)

**Abstract**:

Network binarization is a promising hardware-aware direction for creating efficient deep models. Despite its memory and computational advantages, reducing the accuracy gap between binary models and their real-valued counterparts remains an unsolved challenging research problem. To this end, we make the following 3 contributions: (a) To increase model capacity, we propose Expert Binary Convolution, which, for the first time, tailors conditional computing to binary networks by learning to select one data-specific expert binary filter at a time conditioned on input features. (b) To increase representation capacity, we propose to address the inherent information bottleneck in binary networks by introducing an efficient width expansion mechanism which keeps the binary operations within the same budget. (c) To improve network design, we propose a principled binary network growth mechanism that unveils a set of network topologies of favorable properties. Overall, our method improves upon prior work, with no increase in computational cost, by $\sim6 \%$, reaching a groundbreaking $\sim 71\%$ on ImageNet classification. Code will be made available $\href{https://www.adrianbulat.com/binary-networks}{here}$.

----

## [514] Learning What To Do by Simulating the Past

**Authors**: *David Lindner, Rohin Shah, Pieter Abbeel, Anca D. Dragan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kBVJ2NtiY-](https://openreview.net/forum?id=kBVJ2NtiY-)

**Abstract**:

Since reward functions are hard to specify, recent work has focused on learning policies from human feedback. However, such approaches are impeded by the expense of acquiring such feedback. Recent work proposed that agents have access to a source of information that is effectively free: in any environment that humans have acted in, the state will already be optimized for human preferences, and thus an agent can extract information about what humans want from the state. Such learning is possible in principle, but requires simulating all possible past trajectories that could have led to the observed state. This is feasible in gridworlds, but how do we scale it to complex tasks? In this work, we show that by combining a learned feature encoder with learned inverse models, we can enable agents to simulate human actions backwards in time to infer what they must have done. The resulting algorithm is able to reproduce a specific skill in MuJoCo environments given a single state sampled from the optimal policy for that skill.

----

## [515] Progressive Skeletonization: Trimming more fat from a network at initialization

**Authors**: *Pau de Jorge, Amartya Sanyal, Harkirat S. Behl, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9GsFOUyUPi](https://openreview.net/forum?id=9GsFOUyUPi)

**Abstract**:

Recent studies have shown that skeletonization (pruning parameters) of networks at initialization provides all the practical benefits of sparsity both at inference and training time, while only marginally degrading their performance. However, we observe that beyond a certain level of sparsity (approx 95%), these approaches fail to preserve the network performance, and to our surprise, in many cases perform even worse than trivial random pruning. To this end, we propose an objective to find a skeletonized network with maximum foresight connection sensitivity (FORCE) whereby the trainability, in terms of connection sensitivity, of a pruned network is taken into consideration. We then propose two approximate procedures to maximize our objective (1) Iterative SNIP: allows parameters that were unimportant at earlier stages of skeletonization to become important at later stages; and (2) FORCE: iterative process that allows exploration by allowing already pruned parameters to resurrect at later stages of skeletonization. Empirical analysis on a large suite of experiments show that our approach, while providing at least as good performance as other recent approaches on moderate pruning levels, provide remarkably improved performance on high pruning levels (could remove up to 99.5% parameters while keeping the networks trainable).

----

## [516] Filtered Inner Product Projection for Crosslingual Embedding Alignment

**Authors**: *Vin Sachidananda, Ziyi Yang, Chenguang Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=A2gNouoXE7](https://openreview.net/forum?id=A2gNouoXE7)

**Abstract**:

Due to widespread interest in machine translation and transfer learning, there are numerous algorithms for mapping multiple embeddings to a shared representation space. Recently, these algorithms have been studied in the setting of bilingual lexicon induction where one seeks to align the embeddings of a source and a target language such that translated word pairs lie close to one another in a common representation space. In this paper, we propose a method, Filtered Inner Product Projection (FIPP), for mapping embeddings to a common representation space. As semantic shifts are pervasive across languages and domains, FIPP first identifies the common geometric structure in both embeddings and then, only on the common structure, aligns the Gram matrices of these embeddings. FIPP is applicable even when the source and target embeddings are of differing dimensionalities. Additionally, FIPP provides computational benefits in ease of implementation and is faster to compute than current approaches. Following the baselines in Glavas et al. 2019, we evaluate FIPP both in the context of bilingual lexicon induction and downstream language tasks. We show that FIPP outperforms existing methods on the XLING BLI dataset for most language pairs while also providing robust performance across downstream tasks.

----

## [517] Learning Manifold Patch-Based Representations of Man-Made Shapes

**Authors**: *Dmitriy Smirnov, Mikhail Bessmeltsev, Justin Solomon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Gu5WqN9J3Fn](https://openreview.net/forum?id=Gu5WqN9J3Fn)

**Abstract**:

Choosing the right representation for geometry is crucial for making 3D models compatible with existing applications. Focusing on piecewise-smooth man-made shapes, we propose a new representation that is usable in conventional CAD modeling pipelines and can also be learned by deep neural networks. We demonstrate its benefits by applying it to the task of sketch-based modeling. Given a raster image, our system infers a set of parametric surfaces that realize the input in 3D. To capture piecewise smooth geometry, we learn a special shape representation: a deformable parametric template composed of Coons patches. Naively training such a system, however, is hampered by non-manifold artifacts in the parametric shapes and by a lack of data. To address this, we introduce loss functions that bias the network to output non-self-intersecting shapes and implement them as part of a fully self-supervised system, automatically generating both shape templates and synthetic training data. We develop a testbed for sketch-based modeling, demonstrate shape interpolation, and provide comparison to related work.

----

## [518] Aligning AI With Shared Human Values

**Authors**: *Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, Jacob Steinhardt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dNy_RKzJacY](https://openreview.net/forum?id=dNy_RKzJacY)

**Abstract**:

We show how to assess a language model's knowledge of basic concepts of morality. We introduce the ETHICS dataset, a new benchmark that spans concepts in justice, well-being, duties, virtues, and commonsense morality. Models predict widespread moral judgments about diverse text scenarios. This requires connecting physical and social world knowledge to value judgements, a capability that may enable us to steer chatbot outputs or eventually regularize open-ended reinforcement learning agents. With the ETHICS dataset, we find that current language models have a promising but incomplete ability to predict basic human ethical judgements. Our work shows that progress can be made on machine ethics today, and it provides a steppingstone toward AI that is aligned with human values.

----

## [519] Kanerva++: Extending the Kanerva Machine With Differentiable, Locally Block Allocated Latent Memory

**Authors**: *Jason Ramapuram, Yan Wu, Alexandros Kalousis*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QoWatN-b8T](https://openreview.net/forum?id=QoWatN-b8T)

**Abstract**:

Episodic and semantic memory are critical components of the human memory model. The theory of complementary learning systems (McClelland et al., 1995) suggests that the compressed representation produced by a serial event (episodic memory) is later restructured to build a more generalized form of reusable knowledge (semantic memory). In this work, we develop a new principled Bayesian memory allocation scheme that bridges the gap between episodic and semantic memory via a hierarchical latent variable model. We take inspiration from traditional heap allocation and extend the idea of locally contiguous memory to the Kanerva Machine, enabling a novel differentiable block allocated latent memory. In contrast to the Kanerva Machine, we simplify the process of memory writing by treating it as a fully feed forward deterministic process, relying on the stochasticity of the read key distribution to disperse information within the memory. We demonstrate that this allocation scheme improves performance in memory conditional image generation, resulting in new state-of-the-art conditional likelihood values on binarized MNIST (≤41.58 nats/image) , binarized Omniglot (≤66.24 nats/image), as well as presenting competitive performance on CIFAR10, DMLab Mazes, Celeb-A and ImageNet32×32.

----

## [520] Measuring Massive Multitask Language Understanding

**Authors**: *Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=d7KBjmI3GmQ](https://openreview.net/forum?id=d7KBjmI3GmQ)

**Abstract**:

We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.

----

## [521] Towards Robust Neural Networks via Close-loop Control

**Authors**: *Zhuotong Chen, Qianxiao Li, Zheng Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=2AL06y9cDE-](https://openreview.net/forum?id=2AL06y9cDE-)

**Abstract**:

Despite their success in massive engineering applications, deep neural networks are vulnerable to various perturbations due to their black-box nature. Recent study has shown that a deep neural network can misclassify the data even if the input data is perturbed by an imperceptible amount. In this paper, we address the robustness issue of neural networks by a novel close-loop control method from the perspective of dynamic systems. Instead of modifying the parameters in a fixed neural network architecture, a close-loop control process is added to generate control signals adaptively for the perturbed or corrupted data. We connect the robustness of neural networks with optimal control using the geometrical information of underlying data to design the control objective. The detailed analysis shows how the embedding manifolds of state trajectory affect error estimation of the proposed method. Our approach can simultaneously maintain the performance on clean data and improve the robustness against many types of data perturbations. It can also further improve the performance of robustly trained neural networks against different perturbations. To the best of our knowledge, this is the first work that improves the robustness of neural networks with close-loop control.

----

## [522] Statistical inference for individual fairness

**Authors**: *Subha Maity, Songkai Xue, Mikhail Yurochkin, Yuekai Sun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=z9k8BWL-_2u](https://openreview.net/forum?id=z9k8BWL-_2u)

**Abstract**:

As we rely on machine learning (ML) models to make more consequential decisions, the issue of ML models perpetuating unwanted social biases has come to the fore of the public's and the research community's attention. In this paper, we focus on the problem of detecting violations of individual fairness in ML models. We formalize the problem as measuring the susceptibility of ML models against a form of adversarial attack and develop a suite of inference tools for the adversarial loss. The tools allow practitioners to assess the individual fairness of ML models in a statistically-principled way: form confidence intervals for the adversarial loss and test hypotheses of model fairness with (asymptotic) non-coverage/Type I error rate control. We demonstrate the utility of our tools in a real-world case study.

----

## [523] HyperGrid Transformers: Towards A Single Model for Multiple Tasks

**Authors**: *Yi Tay, Zhe Zhao, Dara Bahri, Donald Metzler, Da-Cheng Juan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hiq1rHO8pNT](https://openreview.net/forum?id=hiq1rHO8pNT)

**Abstract**:

Achieving state-of-the-art performance on natural language understanding tasks typically relies on fine-tuning a fresh model for every task. Consequently, this approach leads to a higher overall parameter cost, along with higher technical maintenance for serving multiple models. Learning a single multi-task model that is able to do well for all the tasks has been a challenging and yet attractive proposition. In this paper, we propose HyperGrid Transformers, a new Transformer architecture that leverages task-conditioned hyper networks for controlling its feed-forward layers. Specifically, we propose a decomposable hypernetwork that learns grid-wise projections that help to specialize regions in weight matrices for different tasks. In order to construct the proposed hypernetwork, our method learns the interactions and composition between a global (task-agnostic) state and a local task-specific state. We conduct an extensive set of experiments on GLUE/SuperGLUE. On the SuperGLUE test set, we match the performance of the state-of-the-art while being $16$ times more parameter efficient. Our method helps bridge the gap between fine-tuning and multi-task learning approaches.

----

## [524] Greedy-GQ with Variance Reduction: Finite-time Analysis and Improved Complexity

**Authors**: *Shaocong Ma, Ziyi Chen, Yi Zhou, Shaofeng Zou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6t_dLShIUyZ](https://openreview.net/forum?id=6t_dLShIUyZ)

**Abstract**:

Greedy-GQ is a value-based reinforcement learning (RL) algorithm for optimal control. Recently, the finite-time analysis of Greedy-GQ has been developed under linear function approximation and Markovian sampling, and the algorithm is shown to achieve an $\epsilon$-stationary point with a sample complexity in the order of $\mathcal{O}(\epsilon^{-3})$. Such a high sample complexity is due to the large variance induced by the Markovian samples. In this paper, we propose a variance-reduced Greedy-GQ (VR-Greedy-GQ) algorithm for off-policy optimal control. In particular, the algorithm applies the SVRG-based variance reduction scheme to reduce the stochastic variance of the two time-scale updates. We study the finite-time convergence of VR-Greedy-GQ under linear function approximation and Markovian sampling and show that the algorithm achieves a much smaller bias and variance error than the original Greedy-GQ. In particular, we prove that VR-Greedy-GQ achieves an improved sample complexity that is in the order of $\mathcal{O}(\epsilon^{-2})$. We further compare the performance of VR-Greedy-GQ with that of Greedy-GQ in various RL experiments to corroborate our theoretical findings.

----

## [525] On InstaHide, Phase Retrieval, and Sparse Matrix Factorization

**Authors**: *Sitan Chen, Xiaoxiao Li, Zhao Song, Danyang Zhuo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AhElGnhU2BV](https://openreview.net/forum?id=AhElGnhU2BV)

**Abstract**:

In this work, we examine the security of InstaHide, a scheme recently proposed by \cite{hsla20} for preserving the security of private datasets in the context of distributed learning. To generate a synthetic training example to be shared among the distributed learners, InstaHide takes a convex combination of private feature vectors and randomly flips the sign of each entry of the resulting vector with probability 1/2. A salient question is whether this scheme is secure in any provable sense, perhaps under a plausible complexity-theoretic assumption. 

The answer to this turns out to be quite subtle and closely related to the average-case complexity of a multi-task, missing-data version of the classic problem of phase retrieval that is interesting in its own right. Motivated by this connection, under the standard distributional assumption that the public/private feature vectors are isotropic Gaussian, we design an algorithm that can actually recover a private vector using only the public vectors and a sequence of synthetic vectors generated by InstaHide.

----

## [526] VA-RED2: Video Adaptive Redundancy Reduction

**Authors**: *Bowen Pan, Rameswar Panda, Camilo Luciano Fosco, Chung-Ching Lin, Alex J. Andonian, Yue Meng, Kate Saenko, Aude Oliva, Rogério Feris*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=g21u6nlbPzn](https://openreview.net/forum?id=g21u6nlbPzn)

**Abstract**:

Performing inference on deep learning models for videos remains a challenge due to the large amount of computational resources required to achieve robust recognition. An inherent property of real-world videos is the high correlation of information across frames which can translate into redundancy in either temporal or spatial feature maps of the models, or both. The type of redundant features depends on the dynamics and type of events in the video: static videos have more temporal redundancy while videos focusing on objects tend to have more channel redundancy. Here we present a redundancy reduction framework, termed VA-RED$^2$, which is input-dependent. Specifically, our VA-RED$^2$ framework uses an input-dependent policy to decide how many features need to be computed for temporal and channel dimensions. To keep the capacity of the original model, after fully computing the necessary features, we reconstruct the remaining redundant features from those using cheap linear operations. We learn the adaptive policy jointly with the network weights in a differentiable way with a shared-weight mechanism, making it highly efficient. Extensive experiments on multiple video datasets and different visual tasks show that our framework achieves $20\% - 40\%$ reduction in computation (FLOPs) when compared to state-of-the-art methods without any performance loss. Project page: http://people.csail.mit.edu/bpan/va-red/.

----

## [527] SEDONA: Search for Decoupled Neural Networks toward Greedy Block-wise Learning

**Authors**: *Myeongjang Pyeon, Jihwan Moon, Taeyoung Hahn, Gunhee Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XLfdzwNKzch](https://openreview.net/forum?id=XLfdzwNKzch)

**Abstract**:

Backward locking and update locking are well-known sources of inefficiency in backpropagation that prevent from concurrently updating layers. Several works have recently suggested using local error signals to train network blocks asynchronously to overcome these limitations. However, they often require numerous iterations of trial-and-error to find the best configuration for local training, including how to decouple network blocks and which auxiliary networks to use for each block. In this work, we propose a differentiable search algorithm named SEDONA to automate this process. Experimental results show that our algorithm can consistently discover transferable decoupled architectures for VGG and ResNet variants, and significantly outperforms the ones trained with end-to-end backpropagation and other state-of-the-art greedy-leaning methods in CIFAR-10, Tiny-ImageNet and ImageNet.

----

## [528] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning

**Authors**: *Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew J. Hausknecht*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0IOX0YcCdTn](https://openreview.net/forum?id=0IOX0YcCdTn)

**Abstract**:

Given a simple request like Put a washed apple in the kitchen fridge, humans can reason in purely abstract terms by imagining action sequences and scoring their likelihood of success, prototypicality, and efficiency, all without moving a muscle. Once we see the kitchen in question, we can update our abstract plans to fit the scene. Embodied agents require the same abilities, but existing work does not yet provide the infrastructure necessary for both reasoning abstractly and executing concretely. We address this limitation by introducing ALFWorld, a simulator that enables agents to learn abstract, text-based policies in TextWorld (Côté et al., 2018) and then execute goals from the ALFRED benchmark (Shridhar et al., 2020) in a rich visual environment. ALFWorld enables the creation of a new BUTLER agent whose abstract knowledge, learned in TextWorld, corresponds directly to concrete, visually grounded actions. In turn, as we demonstrate empirically, this fosters better agent generalization than training only in the visually grounded environment. BUTLER’s simple, modular design factors the problem to allow researchers to focus on models for improving every piece of the pipeline (language understanding, planning, navigation, and visual scene understanding).

----

## [529] Learning Task Decomposition with Ordered Memory Policy Network

**Authors**: *Yuchen Lu, Yikang Shen, Siyuan Zhou, Aaron C. Courville, Joshua B. Tenenbaum, Chuang Gan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vcopnwZ7bC](https://openreview.net/forum?id=vcopnwZ7bC)

**Abstract**:

Many complex real-world tasks are composed of several levels of subtasks. Humans leverage these hierarchical structures to accelerate the learning process and achieve better generalization. In this work, we study the inductive bias and propose Ordered Memory Policy Network (OMPN) to discover subtask hierarchy by learning from demonstration. The discovered subtask hierarchy could be used to perform task decomposition, recovering the subtask boundaries in an unstructured demonstration. Experiments on Craft and Dial demonstrate that our model can achieve higher task decomposition performance under both unsupervised and weakly supervised settings, comparing with strong baselines. OMPN can also be directly applied to partially observable environments and still achieve higher task decomposition performance. Our visualization further confirms that the subtask hierarchy can emerge in our model 1.

----

## [530] Adversarially-Trained Deep Nets Transfer Better: Illustration on Image Classification

**Authors**: *Francisco Utrera, Evan Kravitz, N. Benjamin Erichson, Rajiv Khanna, Michael W. Mahoney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ijJZbomCJIm](https://openreview.net/forum?id=ijJZbomCJIm)

**Abstract**:

Transfer learning has emerged as a powerful methodology for adapting pre-trained deep neural networks on image recognition tasks to new domains. This process consists of taking a neural network pre-trained on a large feature-rich source dataset, freezing the early layers that encode essential generic image properties, and then fine-tuning the last few layers in order to capture specific information related to the target situation. This approach is particularly useful when only limited or weakly labeled data are available for the new task. In this work, we demonstrate that adversarially-trained models transfer better than non-adversarially-trained models, especially if only limited data are available for the new domain task. Further, we observe that adversarial training biases the learnt representations to retaining shapes, as opposed to textures, which impacts the transferability of the source models. Finally, through the lens of influence functions, we discover that transferred adversarially-trained models contain more human-identifiable semantic information, which explains -- at least partly -- why adversarially-trained models transfer better.

----

## [531] UMEC: Unified model and embedding compression for efficient recommendation systems

**Authors**: *Jiayi Shen, Haotao Wang, Shupeng Gui, Jianchao Tan, Zhangyang Wang, Ji Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BM---bH_RSh](https://openreview.net/forum?id=BM---bH_RSh)

**Abstract**:

The recommendation system (RS) plays an important role in the content recommendation and retrieval scenarios. The core part of the system is the Ranking neural network, which is usually a bottleneck of whole system performance during online inference.  In this work, we propose a unified model and embedding compression (UMEC) framework to hammer an efficient neural network-based recommendation system.  Our framework jointly learns input feature selection and neural network compression together, and solve them as an end-to-end resource-constrained optimization problem using ADMM.  Our method outperforms other baselines in terms of neural network Flops, sparse embedding feature size and the number of sparse embedding features.  We evaluate our method on the public benchmark of DLRM, trained over the Kaggle Criteo dataset. The codes can be found at https://github.com/VITA-Group/UMEC.

----

## [532] Exploring Balanced Feature Spaces for Representation Learning

**Authors**: *Bingyi Kang, Yu Li, Sa Xie, Zehuan Yuan, Jiashi Feng*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OqtLIabPTit](https://openreview.net/forum?id=OqtLIabPTit)

**Abstract**:

Existing self-supervised learning (SSL) methods are mostly applied for training representation models from artificially balanced datasets (e.g., ImageNet). It is unclear how well they will perform in the practical scenarios where datasets are often imbalanced w.r.t. the classes. Motivated by this question, we conduct a series of studies on the performance of self-supervised contrastive learning and supervised learning methods over multiple datasets where training instance distributions vary from a balanced one to a long-tailed one. Our findings are quite intriguing. Different from supervised methods with large performance drop, the self-supervised contrastive learning methods perform stably well even when the datasets are heavily imbalanced. This motivates us to explore the balanced feature spaces learned by contrastive learning, where the feature representations present similar linear separability w.r.t. all the classes. Our further experiments reveal that a representation model generating a balanced feature space can generalize better than that yielding an imbalanced one across multiple settings. Inspired by these insights, we develop a novel representation learning method, called $k$-positive contrastive learning. It effectively combines strengths of the supervised method and the contrastive learning method to learn representations that are both discriminative and balanced. Extensive experiments demonstrate its superiority on multiple recognition tasks. Remarkably, it achieves new state-of-the-art on challenging long-tailed recognition benchmarks. Code and models will be released.

----

## [533] Calibration of Neural Networks using Splines

**Authors**: *Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu, Richard Hartley*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eQe8DEWNN2W](https://openreview.net/forum?id=eQe8DEWNN2W)

**Abstract**:

Calibrating neural networks is of utmost importance when employing them in safety-critical applications where the downstream decision making depends on the predicted probabilities. Measuring calibration error amounts to comparing two empirical distributions. In this work, we introduce a binning-free calibration measure inspired by the classical Kolmogorov-Smirnov (KS) statistical test in which the main idea is to compare the respective cumulative probability distributions. From this, by approximating the empirical cumulative distribution using a differentiable function via splines, we obtain a recalibration function, which maps the network outputs to actual (calibrated) class assignment probabilities. The spline-fitting is performed using a held-out calibration set and the obtained recalibration function is evaluated on an unseen test set. We tested our method against existing calibration approaches on various image classification datasets and our spline-based recalibration approach consistently outperforms existing methods on KS error as well as other commonly used calibration measures. Code is available online at https://github.com/kartikgupta-at-anu/spline-calibration.

----

## [534] Improving Relational Regularized Autoencoders with Spherical Sliced Fused Gromov Wasserstein

**Authors**: *Khai Nguyen, Son Nguyen, Nhat Ho, Tung Pham, Hung Bui*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=DiQD7FWL233](https://openreview.net/forum?id=DiQD7FWL233)

**Abstract**:

Relational regularized autoencoder (RAE) is a framework to learn the distribution of data by minimizing a reconstruction loss together with a relational regularization on the prior of latent space. A recent attempt to reduce the inner discrepancy between the prior and aggregated posterior distributions is to incorporate sliced fused Gromov-Wasserstein (SFG) between these distributions. That approach has a weakness since it treats every slicing direction similarly, meanwhile several directions are not useful for the discriminative task. To improve the discrepancy and consequently the relational regularization, we propose a new relational discrepancy, named spherical sliced fused Gromov Wasserstein (SSFG), that can find an important area of projections characterized by a von Mises-Fisher distribution. Then, we introduce two variants of SSFG to improve its performance. The first variant, named mixture spherical sliced fused Gromov Wasserstein (MSSFG), replaces the vMF distribution by a mixture of von Mises-Fisher distributions to capture multiple important areas of directions that are far from each other. The second variant, named power spherical sliced fused Gromov Wasserstein (PSSFG), replaces the vMF distribution by a power spherical distribution to improve the sampling time of the vMF distribution in high dimension settings. We then apply the new discrepancies to the RAE framework to achieve its new variants. Finally, we conduct extensive experiments to show that the new autoencoders have favorable performance in learning latent manifold structure, image generation, and reconstruction.

----

## [535] Rethinking Positional Encoding in Language Pre-training

**Authors**: *Guolin Ke, Di He, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=09-528y2Fgf](https://openreview.net/forum?id=09-528y2Fgf)

**Abstract**:

In this work, we investigate the positional encoding methods used in language pre-training (e.g., BERT) and identify several problems in the existing formulations. First, we show that in the absolute positional encoding, the addition operation applied on positional embeddings and word embeddings brings mixed correlations between the two heterogeneous information resources. It may bring unnecessary randomness in the attention and further limit the expressiveness of the model.  Second, we question whether treating the position of the symbol \texttt{[CLS]} the same as other words is a reasonable design, considering its special role (the representation of the entire sentence) in the downstream tasks. Motivated from above analysis, we propose a new positional encoding method called \textbf{T}ransformer with \textbf{U}ntied \textbf{P}ositional \textbf{E}ncoding (TUPE). In the self-attention module, TUPE computes the word contextual correlation and positional correlation separately with different parameterizations and then adds them together. This design removes the mixed and noisy correlations over heterogeneous embeddings and offers more expressiveness by using different projection matrices. Furthermore, TUPE unties the \texttt{[CLS]} symbol from other positions, making it easier to capture information from all positions. Extensive experiments and ablation studies on GLUE benchmark demonstrate the effectiveness of the proposed method. Codes and models are released at \url{https://github.com/guolinke/TUPE}.

----

## [536] Discovering Non-monotonic Autoregressive Orderings with Variational Inference

**Authors**: *Xuanlin Li, Brandon Trabucco, Dong Huk Park, Michael Luo, Sheng Shen, Trevor Darrell, Yang Gao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jP1vTH3inC](https://openreview.net/forum?id=jP1vTH3inC)

**Abstract**:

The predominant approach for language modeling is to encode a sequence of tokens from left to right, but this eliminates a source of information: the order by which the sequence was naturally generated. One strategy to recover this information is to decode both the content and ordering of tokens. Some prior work supervises content and ordering with hand-designed loss functions to encourage specific orders or bootstraps from a predefined ordering. These approaches require domain-specific insight. Other prior work searches over valid insertion operations that lead to ground truth sequences during training, which has high time complexity and cannot be efficiently parallelized. We address these limitations with an unsupervised learner that can be trained in a fully-parallelizable manner to discover high-quality autoregressive orders in a data driven way without a domain-specific prior. The learner is a neural network that performs variational inference with the autoregressive ordering as a latent variable. Since the corresponding variational lower bound is not differentiable, we develop a practical algorithm for end-to-end optimization using policy gradients. Strong empirical results with our solution on sequence modeling tasks suggest that our algorithm is capable of discovering various autoregressive orders for different sequences that are competitive with or even better than fixed orders.

----

## [537] Differentiable Trust Region Layers for Deep Reinforcement Learning

**Authors**: *Fabian Otto, Philipp Becker, Ngo Anh Vien, Hanna Carolin Maria Ziesche, Gerhard Neumann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qYZD-AO1Vn](https://openreview.net/forum?id=qYZD-AO1Vn)

**Abstract**:

Trust region methods are a popular tool in reinforcement learning as they yield robust policy updates in continuous and discrete action spaces. However, enforcing such trust regions in deep reinforcement learning is difficult. Hence, many approaches, such as Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO), are based on approximations. Due to those approximations, they violate the constraints or fail to find the optimal solution within the trust region. Moreover, they are difficult to implement, often lack sufficient exploration, and have been shown to depend on seemingly unrelated implementation choices. In this work, we propose differentiable neural network layers to enforce trust regions for deep Gaussian policies via closed-form projections. Unlike existing methods, those layers formalize trust regions for each state individually and can complement existing reinforcement learning algorithms. We derive trust region projections based on the Kullback-Leibler divergence, the Wasserstein L2 distance, and the Frobenius norm for Gaussian distributions. We empirically demonstrate that those projection layers achieve similar or better results than existing methods while being almost agnostic to specific implementation choices. The code is available at https://git.io/Jthb0.

----

## [538] SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization

**Authors**: *A. F. M. Shahab Uddin, Mst. Sirazam Monira, Wheemyung Shin, TaeChoong Chung, Sung-Ho Bae*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-M0QkvBGTTq](https://openreview.net/forum?id=-M0QkvBGTTq)

**Abstract**:

Advanced data augmentation strategies have widely been studied to improve the generalization ability of deep learning models. Regional dropout is one of the popular solutions that guides the model to focus on less discriminative parts by randomly removing image regions, resulting in improved regularization. However, such information removal is undesirable. On the other hand, recent strategies suggest to randomly cut and mix patches and their labels among training images, to enjoy the advantages of regional dropout without having any pointless pixel in the augmented images. We argue that such random selection strategies of the patches may not necessarily represent sufficient information about the corresponding object and thereby mixing the labels according to that uninformative patch enables the model to learn unexpected feature representation. Therefore, we propose SaliencyMix that carefully selects a representative image patch with the help of a saliency map and mixes this indicative patch with the target image, thus leading the model to learn more appropriate feature representation. SaliencyMix achieves the best known top-1 error of $21.26\%$ and $20.09\%$ for ResNet-50 and ResNet-101 architectures on ImageNet classification, respectively, and also improves the model robustness against adversarial perturbations. Furthermore, models that are trained with SaliencyMix, help to improve the object detection performance.  Source code is available at \url{https://github.com/SaliencyMix/SaliencyMix}.

----

## [539] Task-Agnostic Morphology Evolution

**Authors**: *Donald Joseph Hejna III, Pieter Abbeel, Lerrel Pinto*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CGQ6ENUMX6](https://openreview.net/forum?id=CGQ6ENUMX6)

**Abstract**:

Deep reinforcement learning primarily focuses on learning behavior, usually overlooking the fact that an agent's function is largely determined by form. So, how should one go about finding a morphology fit for solving tasks in a given environment? Current approaches that co-adapt morphology and behavior use a specific task's reward as a signal for morphology optimization. However, this often requires expensive policy optimization and results in task-dependent morphologies that are not built to generalize. In this work, we propose a new approach, Task-Agnostic Morphology Evolution (TAME), to alleviate both of these issues. Without any task or reward specification, TAME evolves morphologies by only applying randomly sampled action primitives on a population of agents. This is accomplished using an information-theoretic objective that efficiently ranks agents by their ability to reach diverse states in the environment and the causality of their actions. Finally, we empirically demonstrate that across 2D, 3D, and manipulation environments TAME can evolve morphologies that match the multi-task performance of those learned with task supervised algorithms. Our code and videos can be found at https://sites.google.com/view/task-agnostic-evolution .

----

## [540] Learning Associative Inference Using Fast Weight Memory

**Authors**: *Imanol Schlag, Tsendsuren Munkhdalai, Jürgen Schmidhuber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TuK6agbdt27](https://openreview.net/forum?id=TuK6agbdt27)

**Abstract**:

Humans can quickly associate stimuli to solve problems in novel contexts. Our novel neural network model learns state representations of facts that can be composed to perform such associative inference. To this end, we augment the LSTM model with an associative memory, dubbed \textit{Fast Weight Memory} (FWM). Through differentiable operations at every step of a given input sequence, the LSTM \textit{updates and maintains} compositional associations stored in the rapidly changing FWM weights. Our model is trained end-to-end by gradient descent and yields excellent performance on compositional language reasoning problems, meta-reinforcement-learning for POMDPs, and small-scale word-level language modelling.

----

## [541] Boost then Convolve: Gradient Boosting Meets Graph Neural Networks

**Authors**: *Sergei Ivanov, Liudmila Prokhorenkova*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ebS5NUfoMKL](https://openreview.net/forum?id=ebS5NUfoMKL)

**Abstract**:

Graph neural networks (GNNs) are powerful models that have been successful in various graph representation learning tasks. Whereas gradient boosted decision trees (GBDT) often outperform other machine learning methods when faced with heterogeneous tabular data. But what approach should be used for graphs with tabular node features? Previous GNN models have mostly focused on networks with homogeneous sparse features and, as we show, are suboptimal in the heterogeneous setting. In this work, we propose a novel architecture that trains GBDT and GNN jointly to get the best of both worlds: the GBDT model deals with heterogeneous features, while GNN accounts for the graph structure. Our model benefits from end-to-end optimization by allowing new trees to fit the gradient updates of GNN. With an extensive experimental comparison to the leading GBDT and GNN models, we demonstrate a significant increase in performance on a variety of graphs with tabular features. The code is available: https://github.com/nd7141/bgnn.

----

## [542] Degree-Quant: Quantization-Aware Training for Graph Neural Networks

**Authors**: *Shyam Anil Tailor, Javier Fernández-Marqués, Nicholas Donald Lane*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NSBrFgJAHg](https://openreview.net/forum?id=NSBrFgJAHg)

**Abstract**:

Graph neural networks (GNNs) have demonstrated strong performance on a wide variety of tasks due to their ability to model non-uniform structured data. Despite their promise, there exists little research exploring methods to make them more efficient at inference time. In this work, we explore the viability of training quantized GNNs, enabling the usage of low precision integer arithmetic during inference. For GNNs seemingly unimportant choices in quantization implementation cause dramatic changes in performance. We identify the sources of error that uniquely arise when attempting to quantize GNNs, and propose an architecturally-agnostic and stable method, Degree-Quant, to improve performance over existing quantization-aware training baselines commonly used on other architectures, such as CNNs. We validate our method on six datasets and show, unlike previous quantization attempts, that models generalize to unseen graphs. Models trained with Degree-Quant for INT8 quantization perform as well as FP32 models in most cases; for INT4 models, we obtain up to 26% gains over the baselines. Our work enables up to 4.7x speedups on CPU when using INT8 arithmetic.

----

## [543] Network Pruning That Matters: A Case Study on Retraining Variants

**Authors**: *Duong H. Le, Binh-Son Hua*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Cb54AMqHQFP](https://openreview.net/forum?id=Cb54AMqHQFP)

**Abstract**:

Network pruning is an effective method to reduce the computational expense of over-parameterized neural networks for deployment on low-resource systems. Recent state-of-the-art techniques for retraining pruned networks such as weight rewinding and learning rate rewinding have been shown to outperform the traditional fine-tuning technique in recovering the lost accuracy (Renda et al., 2020), but so far it is unclear what accounts for such performance. In this work, we conduct extensive experiments to verify and analyze the uncanny effectiveness of learning rate rewinding. We find that the reason behind the success of learning rate rewinding is the usage of a large learning rate. Similar phenomenon can be observed in other learning rate schedules that involve large learning rates, e.g., the 1-cycle learning rate schedule (Smith et al., 2019). By leveraging the right learning rate schedule in retraining, we demonstrate a counter-intuitive phenomenon in that randomly pruned networks could even achieve better performance than methodically pruned networks (fine-tuned with the conventional approach). Our results emphasize the cruciality of the learning rate schedule in pruned network retraining - a detail often overlooked by practitioners during the implementation of network pruning.

----

## [544] Auto Seg-Loss: Searching Metric Surrogates for Semantic Segmentation

**Authors**: *Hao Li, Chenxin Tao, Xizhou Zhu, Xiaogang Wang, Gao Huang, Jifeng Dai*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MJAqnaC2vO1](https://openreview.net/forum?id=MJAqnaC2vO1)

**Abstract**:

Designing proper loss functions is essential in training deep networks. Especially in the field of semantic segmentation, various evaluation metrics have been proposed for diverse scenarios. Despite the success of the widely adopted cross-entropy loss and its variants, the mis-alignment between the loss functions and evaluation metrics degrades the network performance. Meanwhile, manually designing loss functions for each specific metric requires expertise and significant manpower. In this paper, we propose to automate the design of metric-specific loss functions by searching differentiable surrogate losses for each metric. We substitute the non-differentiable operations in the metrics with parameterized functions, and conduct parameter search to optimize the shape of loss surfaces. Two constraints are introduced to regularize the search space and make the search efficient. Extensive experiments on PASCAL VOC and Cityscapes demonstrate that the searched surrogate losses outperform the manually designed loss functions consistently. The searched losses can generalize well to other datasets and networks. Code shall be released at https://github.com/fundamentalvision/Auto-Seg-Loss.

----

## [545] Differentiable Segmentation of Sequences

**Authors**: *Erik Scharwächter, Jonathan Lennartz, Emmanuel Müller*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4T489T4yav](https://openreview.net/forum?id=4T489T4yav)

**Abstract**:

Segmented models are widely used to describe non-stationary sequential data with discrete change points. Their estimation usually requires solving a mixed discrete-continuous optimization problem, where the segmentation is the discrete part and all other model parameters are continuous. A number of estimation algorithms have been developed that are highly specialized for their specific model assumptions. The dependence on non-standard algorithms makes it hard to integrate segmented models in state-of-the-art deep learning architectures that critically depend on gradient-based optimization techniques. In this work, we formulate a relaxed variant of segmented models that enables joint estimation of all model parameters, including the segmentation, with gradient descent. We build on recent advances in learning continuous warping functions and propose a novel family of warping functions based on the two-sided power (TSP) distribution. TSP-based warping functions are differentiable, have simple closed-form expressions, and can represent segmentation functions exactly. Our formulation includes the important class of segmented generalized linear models as a special case, which makes it highly versatile. We use our approach to model the spread of COVID-19 with Poisson regression, apply it on a change point detection task, and learn classification models with concept drift. The experiments show that our approach effectively learns all these tasks with standard algorithms for gradient descent.

----

## [546] Grounding Physical Concepts of Objects and Events Through Dynamic Visual Reasoning

**Authors**: *Zhenfang Chen, Jiayuan Mao, Jiajun Wu, Kwan-Yee Kenneth Wong, Joshua B. Tenenbaum, Chuang Gan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bhCDO_cEGCz](https://openreview.net/forum?id=bhCDO_cEGCz)

**Abstract**:

We study the problem of dynamic visual reasoning on raw videos. This is a challenging problem; currently, state-of-the-art models often require dense supervision on physical object properties and events from simulation, which are impractical to obtain in real life. In this paper, we present the Dynamic Concept Learner (DCL), a unified framework that grounds physical objects and events from video and language. DCL first adopts a trajectory extractor to track each object over time and to represent it as a latent, object-centric feature vector. Building upon this object-centric representation, DCL learns to approximate the dynamic interaction among objects using graph networks. DCL further incorporates a semantic parser to parse question into semantic programs and, finally, a program executor to run the program to answer the question, levering the learned dynamics model. After training, DCL can detect and associate objects across the frames, ground visual properties and physical events, understand the causal relationship between events, make future and counterfactual predictions, and leverage these extracted presentations for answering queries. DCL achieves state-of-the-art performance on CLEVRER, a challenging causal video reasoning dataset, even without using ground-truth attributes and collision labels from simulations for training. We further test DCL on a newly proposed video-retrieval and event localization dataset derived from CLEVRER, showing its strong generalization capacity.

----

## [547] Learning Deep Features in Instrumental Variable Regression

**Authors**: *Liyuan Xu, Yutian Chen, Siddarth Srinivasan, Nando de Freitas, Arnaud Doucet, Arthur Gretton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sy4Kg_ZQmS7](https://openreview.net/forum?id=sy4Kg_ZQmS7)

**Abstract**:

Instrumental variable (IV) regression is a standard strategy for learning causal relationships between confounded treatment and outcome variables from observational data by using an instrumental variable, which affects the outcome only through the treatment. In classical IV regression, learning proceeds in two stages: stage 1 performs linear regression from the instrument to the treatment; and stage 2 performs linear regression from the treatment to the outcome, conditioned on the instrument. We propose a novel method, deep feature instrumental variable regression (DFIV), to address the case where relations between instruments, treatments, and outcomes may be nonlinear. In this case, deep neural nets are trained to define informative nonlinear features on the instruments and treatments. We propose an alternating training regime for these features to ensure good end-to-end performance when composing stages 1 and 2, thus obtaining highly flexible feature maps in a computationally efficient manner.
DFIV outperforms recent state-of-the-art methods on challenging IV benchmarks, including settings involving high dimensional image data. DFIV also exhibits competitive performance in off-policy policy evaluation for reinforcement learning, which can be understood as an IV regression task.

----

## [548] Online Adversarial Purification based on Self-supervised Learning

**Authors**: *Changhao Shi, Chester Holtz, Gal Mishne*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_i3ASPp12WS](https://openreview.net/forum?id=_i3ASPp12WS)

**Abstract**:

Deep neural networks are known to be vulnerable to adversarial examples, where a perturbation in the input space leads to an amplified shift in the latent network representation. In this paper, we combine canonical supervised learning with self-supervised representation learning, and present Self-supervised Online Adversarial Purification (SOAP), a novel defense strategy that uses a self-supervised loss to purify adversarial examples at test-time. Our approach leverages the label-independent nature of self-supervised signals and counters the adversarial perturbation with respect to the self-supervised tasks. SOAP yields competitive robust accuracy against state-of-the-art adversarial training and purification methods, with considerably less training complexity. In addition, our approach is robust even when adversaries are given the knowledge of the purification defense strategy. To the best of our knowledge, our paper is the first that generalizes the idea of using self-supervised signals to perform online test-time purification.

----

## [549] Graph Information Bottleneck for Subgraph Recognition

**Authors**: *Junchi Yu, Tingyang Xu, Yu Rong, Yatao Bian, Junzhou Huang, Ran He*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bM4Iqfg8M2k](https://openreview.net/forum?id=bM4Iqfg8M2k)

**Abstract**:

Given the input graph and its label/property, several key problems  of graph learning, such as finding interpretable subgraphs, graph denoising and graph compression,  can be  attributed to the fundamental problem of recognizing a subgraph of the original one.  This subgraph shall be as informative as possible, yet contains less redundant and noisy structure. This problem setting is closely related to the well-known information bottleneck (IB) principle, which, however, has less been studied for the irregular graph data and graph neural networks (GNNs). In this paper, we propose a framework of Graph Information Bottleneck (GIB) for the subgraph recognition problem in deep graph learning. Under this framework, one can recognize the maximally informative yet compressive subgraph, named IB-subgraph.  However, the GIB objective is notoriously hard to optimize, mostly due to the intractability of the mutual information of irregular graph data and the unstable optimization process. In order to tackle these challenges, we propose:  i) a GIB objective based-on a mutual information estimator for the irregular graph data; ii) a bi-level optimization scheme to maximize the GIB objective; iii) a connectivity loss to stabilize the optimization process. We evaluate the properties of the IB-subgraph in three application scenarios: improvement of graph classification, graph interpretation and graph denoising. Extensive experiments demonstrate that the information-theoretic  IB-subgraph  enjoys superior graph properties.

----

## [550] In Search of Lost Domain Generalization

**Authors**: *Ishaan Gulrajani, David Lopez-Paz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lQdXeXDoWtI](https://openreview.net/forum?id=lQdXeXDoWtI)

**Abstract**:

The goal of domain generalization algorithms is to predict well on distributions different from those seen during training.
While a myriad of domain generalization algorithms exist, inconsistencies in experimental conditions---datasets, network architectures, and model selection criteria---render fair comparisons difficult.
The goal of this paper is to understand how useful domain generalization algorithms are in realistic settings.
As a first step, we realize that model selection is non-trivial for domain generalization tasks, and we argue that algorithms without a model selection criterion remain incomplete.
Next we implement DomainBed, a testbed for domain generalization including seven benchmarks, fourteen algorithms, and three model selection criteria.
When conducting extensive experiments using DomainBed we find that when carefully implemented and tuned, ERM outperforms the state-of-the-art in terms of average performance.
Furthermore, no algorithm included in DomainBed outperforms ERM by more than one point when evaluated under the same experimental conditions.
We hope that the release of DomainBed, alongside contributions from fellow researchers, will streamline reproducible and rigorous advances in domain generalization.

----

## [551] Robust Curriculum Learning: from clean label detection to noisy label self-correction

**Authors**: *Tianyi Zhou, Shengjie Wang, Jeff A. Bilmes*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lmTWnm3coJJ](https://openreview.net/forum?id=lmTWnm3coJJ)

**Abstract**:

Neural network training can easily overfit noisy labels resulting in poor generalization performance. Existing methods address this problem by (1) filtering out the noisy data and only using the clean data for training or (2) relabeling the noisy data by the model during training or by another model trained only on a clean dataset. However, the former does not leverage the features' information of wrongly-labeled data, while the latter may produce wrong pseudo-labels for some data and introduce extra noises. In this paper, we propose a smooth transition and interplay between these two strategies as a curriculum that selects training samples dynamically. In particular, we start with learning from clean data and then gradually move to learn noisy-labeled data with pseudo labels produced by a time-ensemble of the model and data augmentations. Instead of using the instantaneous loss computed at the current step, our data selection is based on the dynamics of both the loss and output consistency for each sample across historical steps and different data augmentations, resulting in more precise detection of both clean labels and correct pseudo labels. On multiple benchmarks of noisy labels, we show that our curriculum learning strategy can significantly improve the test accuracy without any auxiliary model or extra clean data.

----

## [552] Cross-Attentional Audio-Visual Fusion for Weakly-Supervised Action Localization

**Authors**: *Jun-Tae Lee, Mihir Jain, Hyoungwoo Park, Sungrack Yun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hWr3e3r-oH5](https://openreview.net/forum?id=hWr3e3r-oH5)

**Abstract**:

Temporally localizing actions in videos is one of the key components for video understanding. Learning from weakly-labeled data is seen as a potential solution towards avoiding expensive frame-level annotations. Different from other works which only depend on visual-modality, we propose to learn richer audiovisual representation for weakly-supervised action localization. First, we propose a multi-stage cross-attention mechanism to collaboratively fuse audio and visual features, which preserves the intra-modal characteristics. Second, to model both foreground and background frames, we construct an open-max classifier that treats the background class as an open-set. Third, for precise action localization, we design consistency losses to enforce temporal continuity for the action class prediction, and also help with foreground-prediction reliability. Extensive experiments on two publicly available video-datasets (AVE and ActivityNet1.2) show that the proposed method effectively fuses audio and visual modalities, and achieves the state-of-the-art results for weakly-supervised action localization.

----

## [553] CoCon: A Self-Supervised Approach for Controlled Text Generation

**Authors**: *Alvin Chan, Yew-Soon Ong, Bill Pung, Aston Zhang, Jie Fu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VD_ozqvBy4W](https://openreview.net/forum?id=VD_ozqvBy4W)

**Abstract**:

Pretrained Transformer-based language models (LMs) display remarkable natural language generation capabilities. With their immense potential, controlling text generation of such LMs is getting attention. While there are studies that seek to control high-level attributes (such as sentiment and topic) of generated text, there is still a lack of more precise control over its content at the word- and phrase-level. Here, we propose Content-Conditioner (CoCon) to control an LM's output text with a content input, at a fine-grained level. In our self-supervised approach, the CoCon block learns to help the LM complete a partially-observed text sequence by conditioning with content inputs that are withheld from the LM. Through experiments, we show that CoCon can naturally incorporate target content into generated texts and control high-level text attributes in a zero-shot manner.

----

## [554] Group Equivariant Generative Adversarial Networks

**Authors**: *Neel Dey, Antong Chen, Soheil Ghafurian*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rgFNuJHHXv](https://openreview.net/forum?id=rgFNuJHHXv)

**Abstract**:

Recent improvements in generative adversarial visual synthesis incorporate real and fake image transformation in a self-supervised setting, leading to increased stability and perceptual fidelity. However, these approaches typically involve image augmentations via additional regularizers in the GAN objective and thus spend valuable network capacity towards approximating transformation equivariance instead of their desired task. In this work, we explicitly incorporate inductive symmetry priors into the network architectures via group-equivariant convolutional networks. Group-convolutions have higher expressive power with fewer samples and lead to better gradient feedback between generator and discriminator. We show that group-equivariance integrates seamlessly with recent techniques for GAN training across regularizers, architectures, and loss functions. We demonstrate the utility of our methods for conditional synthesis by improving generation in the limited data regime across symmetric imaging datasets and even find benefits for natural images with preferred orientation.

----

## [555] What they do when in doubt: a study of inductive biases in seq2seq learners

**Authors**: *Eugene Kharitonov, Rahma Chaabouni*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YmA86Zo-P_t](https://openreview.net/forum?id=YmA86Zo-P_t)

**Abstract**:

Sequence-to-sequence (seq2seq) learners are widely used, but we still have only limited knowledge about what inductive biases shape the way they generalize. We address that by investigating how popular seq2seq learners generalize in tasks that have high ambiguity in the training data. We use four new tasks  to study learners' preferences for memorization, arithmetic, hierarchical, and compositional reasoning. Further, we connect to Solomonoff's theory of induction and propose to use description length as a principled and sensitive measure of inductive biases. In our experimental study, we find that LSTM-based learners can learn to perform counting, addition, and multiplication by a constant from a single training example. Furthermore, Transformer and LSTM-based learners show a bias toward the hierarchical induction over the linear one, while CNN-based learners prefer the opposite. The latter also show a bias toward a compositional generalization over memorization. Finally, across all our experiments, description length proved to be a sensitive measure of inductive biases.

----

## [556] A teacher-student framework to distill future trajectories

**Authors**: *Alexander Neitz, Giambattista Parascandolo, Bernhard Schölkopf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ECuvULjFQia](https://openreview.net/forum?id=ECuvULjFQia)

**Abstract**:

By learning to predict trajectories of dynamical systems, model-based methods can make extensive use of all observations from past experience. However, due to partial observability, stochasticity, compounding errors, and irrelevant dynamics, training to predict observations explicitly often results in poor models. Model-free techniques try to side-step the problem by learning to predict values directly. While breaking the explicit dependency on future observations can result in strong performance, this usually comes at the cost of low sample efficiency, as the abundant information about the dynamics contained in future observations goes unused. Here we take a step back from both approaches: Instead of hand-designing how trajectories should be incorporated, a teacher network learns to interpret the trajectories and to provide target activations which guide a student model that can only observe the present. The teacher is trained with meta-gradients to maximize the student's performance on a validation set. We show that our approach performs well on tasks that are difficult for model-free and model-based methods, and we study the role of every component through ablation studies.

----

## [557] Learning a Latent Search Space for Routing Problems using Variational Autoencoders

**Authors**: *André Hottung, Bhanu Bhandari, Kevin Tierney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=90JprVrJBO](https://openreview.net/forum?id=90JprVrJBO)

**Abstract**:

Methods for automatically learning to solve routing problems are rapidly improving in performance. While most of these methods excel at generating solutions quickly, they are unable to effectively utilize longer run times because they lack a sophisticated search component. We present a learning-based optimization approach that allows a guided search in the distribution of high-quality solutions for a problem instance. More precisely, our method uses a conditional variational autoencoder that learns to map points in a continuous (latent) search space to high-quality, instance-specific routing problem solutions. The learned space can then be searched by any unconstrained continuous optimization method. We show that even using a standard differential evolution search strategy our approach is able to outperform existing purely machine learning based approaches.

----

## [558] Universal approximation power of deep residual neural networks via nonlinear control theory

**Authors**: *Paulo Tabuada, Bahman Gharesifard*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-IXhmY16R3M](https://openreview.net/forum?id=-IXhmY16R3M)

**Abstract**:

In this paper, we explain the universal approximation capabilities of deep residual neural networks through geometric nonlinear control. Inspired by recent work establishing links between residual networks and control systems, we provide a general sufficient condition for a residual network to have the power of universal approximation by asking the activation function, or one of its derivatives, to satisfy a quadratic differential equation. Many activation functions used in practice satisfy this assumption, exactly or approximately, and we show this property to be sufficient for an adequately deep neural network with $n+1$ neurons per
layer to approximate arbitrarily well, on a compact set and with respect to the supremum norm, any continuous function from $\mathbb{R}^n$ to $\mathbb{R}^n$. We further show this result to hold for very simple architectures for which the weights only need to assume two values. The first key technical contribution consists of relating the universal approximation problem to controllability of an ensemble of control systems corresponding to a residual network and to leverage classical Lie algebraic techniques to characterize controllability. The second technical contribution is to identify monotonicity as the bridge between controllability of finite ensembles and uniform approximability on compact sets.

----

## [559] On the Universality of Rotation Equivariant Point Cloud Networks

**Authors**: *Nadav Dym, Haggai Maron*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6NFBvWlRXaG](https://openreview.net/forum?id=6NFBvWlRXaG)

**Abstract**:

Learning functions on point clouds has applications in many fields, including computer vision, computer graphics, physics, and chemistry. Recently, there has been a growing interest in neural architectures that are invariant or equivariant to all three shape-preserving transformations of point clouds: translation, rotation, and permutation. In this paper, we present a first study of the approximation power of these architectures. We first derive two sufficient conditions for an equivariant architecture to have the universal approximation property, based on a novel characterization of the space of equivariant polynomials. We then use these conditions to show that two recently suggested models, Tensor field Networks and SE3-Transformers, are universal, and for devising two other novel universal architectures.

----

## [560] CT-Net: Channel Tensorization Network for Video Classification

**Authors**: *Kunchang Li, Xianhang Li, Yali Wang, Jun Wang, Yu Qiao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UoaQUQREMOs](https://openreview.net/forum?id=UoaQUQREMOs)

**Abstract**:

3D convolution is powerful for video classification but often computationally expensive, recent studies mainly focus on decomposing it on spatial-temporal and/or channel dimensions.  Unfortunately,  most approaches fail to achieve a preferable balance between convolutional efficiency and feature-interaction sufficiency.  For this reason,  we propose a concise and novel Channel Tensorization Network (CT-Net),  by treating the channel dimension of input feature as a multiplication of K sub-dimensions. On one hand,  it naturally factorizes convolution in a multiple dimension way,  leading to a light computation burden.  On the other hand, it can effectively enhance feature interaction from different channels,  and progressively enlarge the 3D receptive field of such interaction to boost classification accuracy.  Furthermore, we equip our CT-Module with a Tensor Excitation (TE) mechanism. It can learn to exploit spatial, temporal and channel attention in a high-dimensional manner, to improve the cooperative power of all the feature dimensions in our CT-Module. Finally, we flexibly adapt ResNet as our CT-Net. Extensive experiments are conducted on several challenging video benchmarks, e.g., Kinetics-400, Something-Something V1 and V2. Our CT-Net outperforms a number of recent SOTA approaches, in terms of accuracy and/or efficiency.

----

## [561] Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units

**Authors**: *Jonathan Cornford, Damjan Kalajdzievski, Marco Leite, Amélie Lamarquette, Dimitri Michael Kullmann, Blake Aaron Richards*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eU776ZYxEpz](https://openreview.net/forum?id=eU776ZYxEpz)

**Abstract**:

The units in artificial neural networks (ANNs) can be thought of as abstractions of biological neurons, and ANNs are increasingly used in neuroscience research. However, there are many important differences between ANN units and real neurons. One of the most notable is the absence of Dale's principle, which ensures that biological neurons are either exclusively excitatory or inhibitory. Dale's principle is typically left out of ANNs because its inclusion impairs learning. This is problematic, because one of the great advantages of ANNs for neuroscience research is their ability to learn complicated, realistic tasks. Here, by taking inspiration from feedforward inhibitory interneurons in the brain we show that we can develop ANNs with separate populations of excitatory and inhibitory units that learn just as well as standard ANNs. We call these networks Dale's ANNs (DANNs). We present two insights that enable DANNs to learn well: (1) DANNs are related to normalization schemes, and can be initialized such that the inhibition centres and standardizes the excitatory activity, (2) updates to inhibitory neuron parameters should be scaled using corrections based on the Fisher Information matrix. These results demonstrate how ANNs that respect Dale's principle can be built without sacrificing learning performance, which is important for future work using ANNs as models of the brain. The results may also have interesting implications for how inhibitory plasticity in the real brain operates.

----

## [562] Uncertainty Estimation in Autoregressive Structured Prediction

**Authors**: *Andrey Malinin, Mark J. F. Gales*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jN5y-zb5Q7m](https://openreview.net/forum?id=jN5y-zb5Q7m)

**Abstract**:

Uncertainty estimation is important for ensuring safety and robustness of AI systems.  While most research in the area has focused on un-structured prediction tasks, limited work has investigated general uncertainty estimation approaches for structured prediction. Thus, this work aims to investigate uncertainty estimation for structured prediction tasks within a single unified and interpretable probabilistic ensemble-based framework.  We consider: uncertainty estimation for sequence data at the token-level and complete sequence-level; interpretations for, and applications of, various measures of uncertainty; and discuss both the theoretical and practical challenges associated with obtaining them. This work also provides baselines for token-level and sequence-level error detection, and sequence-level out-of-domain input detection on the WMT’14 English-French and WMT’17 English-German translation and LibriSpeech speech recognition datasets.

----

## [563] Transformer protein language models are unsupervised structure learners

**Authors**: *Roshan Rao, Joshua Meier, Tom Sercu, Sergey Ovchinnikov, Alexander Rives*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fylclEqgvgd](https://openreview.net/forum?id=fylclEqgvgd)

**Abstract**:

Unsupervised contact prediction is central to uncovering physical, structural, and functional constraints for protein structure determination and design. For decades, the predominant approach has been to infer evolutionary constraints from a set of related sequences. In the past year, protein language models have emerged as a potential alternative, but performance has fallen short of state-of-the-art approaches in bioinformatics. In this paper we demonstrate that Transformer attention maps learn contacts from the unsupervised language modeling objective. We find the highest capacity models that have been trained to date already outperform a state-of-the-art unsupervised contact prediction pipeline, suggesting these pipelines can be replaced with a single forward pass of an end-to-end model.

----

## [564] ANOCE: Analysis of Causal Effects with Multiple Mediators via Constrained Structural Learning

**Authors**: *Hengrui Cai, Rui Song, Wenbin Lu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7I12hXRi8F](https://openreview.net/forum?id=7I12hXRi8F)

**Abstract**:

In the era of causal revolution, identifying the causal effect of an exposure on the outcome of interest is an important problem in many areas, such as epidemics, medicine, genetics, and economics. Under a general causal graph, the exposure may have a direct effect on the outcome and also an indirect effect regulated by a set of mediators. An analysis of causal effects that interprets the causal mechanism contributed through mediators is hence challenging but on demand. To the best of our knowledge, there are no feasible algorithms that give an exact decomposition of the indirect effect on the level of individual mediators, due to common interaction among mediators in the complex graph. In this paper, we establish a new statistical framework to comprehensively characterize causal effects with multiple mediators, namely, ANalysis Of Causal Effects (ANOCE), with a newly introduced definition of the mediator effect, under the linear structure equation model. We further propose a constrained causal structure learning method by incorporating a novel identification constraint that specifies the temporal causal relationship of variables. The proposed algorithm is applied to investigate the causal effects of 2020 Hubei lockdowns on reducing the spread of the coronavirus in Chinese major cities out of Hubei.

----

## [565] Plan-Based Relaxed Reward Shaping for Goal-Directed Tasks

**Authors**: *Ingmar Schubert, Ozgur S. Oguz, Marc Toussaint*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=w2Z2OwVNeK](https://openreview.net/forum?id=w2Z2OwVNeK)

**Abstract**:

In high-dimensional state spaces, the usefulness of Reinforcement Learning (RL) is limited by the problem of exploration. This issue has been addressed using potential-based reward shaping (PB-RS) previously. In the present work, we introduce Final-Volume-Preserving Reward Shaping (FV-RS). FV-RS relaxes the strict optimality guarantees of PB-RS to a guarantee of preserved long-term behavior. Being less restrictive, FV-RS allows for reward shaping functions that are even better suited for improving the sample efficiency of RL algorithms. In particular, we consider settings in which the agent has access to an approximate plan. Here, we use examples of simulated robotic manipulation tasks to demonstrate that plan-based FV-RS can indeed significantly improve the sample efficiency of RL over plan-based PB-RS.

----

## [566] CcGAN: Continuous Conditional Generative Adversarial Networks for Image Generation

**Authors**: *Xin Ding, Yongwei Wang, Zuheng Xu, William J. Welch, Z. Jane Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PrzjugOsDeE](https://openreview.net/forum?id=PrzjugOsDeE)

**Abstract**:

This work proposes the continuous conditional generative adversarial network (CcGAN), the first generative model for image generation conditional on continuous, scalar conditions (termed regression labels). Existing conditional GANs (cGANs) are mainly designed for categorical conditions (e.g., class labels); conditioning on a continuous label is mathematically distinct and raises two fundamental problems: (P1) Since there may be very few (even zero) real images for some regression labels, minimizing existing empirical versions of cGAN losses (a.k.a. empirical cGAN losses) often fails in practice; (P2) Since regression labels are scalar and infinitely many, conventional label input methods (e.g., combining a hidden map of the generator/discriminator with a one-hot encoded label) are not applicable. The proposed CcGAN solves the above problems, respectively, by (S1) reformulating existing empirical cGAN losses to be appropriate for the continuous scenario; and (S2) proposing a novel method to incorporate regression labels into the generator and the discriminator. The reformulation in (S1) leads to two novel empirical discriminator losses, termed the hard vicinal discriminator loss (HVDL) and the soft vicinal discriminator loss (SVDL) respectively, and a novel empirical generator loss. The error bounds of a discriminator trained with HVDL and SVDL are derived under mild assumptions in this work. A new benchmark dataset, RC-49, is also proposed for generative image modeling conditional on regression labels. Our experiments on the Circular 2-D Gaussians, RC-49, and UTKFace datasets show that CcGAN is able to generate diverse, high-quality samples from the image distribution conditional on a given regression label. Moreover, in these experiments, CcGAN substantially outperforms cGAN both visually and quantitatively.

----

## [567] Single-Photon Image Classification

**Authors**: *Thomas Fischbacher, Luciano Sbaiz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CHLhSw9pSw8](https://openreview.net/forum?id=CHLhSw9pSw8)

**Abstract**:

Quantum Computing based Machine Learning mainly focuses on quantum computing hardware that is experimentally challenging to realize due to requiring quantum gates that operate at very low temperature. We demonstrate the existence of a "quantum computing toy model" that illustrates key aspects of quantum information processing while being experimentally accessible with room temperature optics. Pondering the question of the theoretical classification accuracy performance limit for MNIST (respectively "Fashion-MNIST") classifiers, subject to the constraint that a decision has to be made after detection of the very first photon that passed through an image-filter, we show that a machine learning system that is permitted to use quantum interference on the photon's state can substantially outperform any machine learning system that can not.  Specifically, we prove that a "classical" MNIST (respectively "Fashion-MNIST") classifier cannot achieve an accuracy of better than $21.28\%$ (respectively $18.28\%$ for "Fashion-MNIST") if it must make a decision after seeing a single photon falling on one of the $28\times 28$ image pixels of a detector array.  We further demonstrate that a classifier that is permitted to employ quantum interference by optically transforming the photon state prior to detection can achieve a classification accuracy of at least $41.27\%$ for MNIST (respectively $36.14\%$ for "Fashion-MNIST"). We show in detail how to train the corresponding quantum state transformation with TensorFlow and also explain how this example can serve as a teaching tool for the measurement process in quantum mechanics.

----

## [568] Self-supervised Adversarial Robustness for the Low-label, High-data Regime

**Authors**: *Sven Gowal, Po-Sen Huang, Aäron van den Oord, Timothy A. Mann, Pushmeet Kohli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bgQek2O63w](https://openreview.net/forum?id=bgQek2O63w)

**Abstract**:

Recent work discovered that training models to be invariant to adversarial perturbations requires substantially larger datasets than those required for standard classification. Perhaps more surprisingly, these larger datasets can be "mostly" unlabeled. Pseudo-labeling, a technique simultaneously pioneered by four separate and simultaneous works in 2019, has been proposed as a competitive alternative to labeled data for training adversarially robust models. However, when the amount of labeled data decreases, the performance of pseudo-labeling catastrophically drops, thus questioning the theoretical insights put forward by Uesato et al. (2019), which suggest that the sample complexity for learning an adversarially robust model from unlabeled data should match the fully supervised case. We introduce Bootstrap Your Own Robust Latents (BYORL), a self-supervised learning technique based on BYOL for training adversarially robust models. Our method enables us to train robust representations without any labels (reconciling practice with theory). Most notably, this robust representation can be leveraged by a linear classifier to train adversarially robust models, even when the linear classifier is not trained adversarially. We evaluate BYORL and pseudo-labeling on CIFAR-10 and ImageNet and demonstrate that BYORL achieves significantly higher robustness (i.e., models resulting from BYORL are up to two times more accurate). Experiments on CIFAR-10 against $\ell_2$ and $\ell_\infty$ norm-bounded perturbations demonstrate that BYORL achieves near state-of-the-art robustness with as little as 500 labeled examples. We also note that against $\ell_2$ norm-bounded perturbations of size $\epsilon = 128/255$, BYORL surpasses the known state-of-the-art with an accuracy under attack of 77.61% (against 72.91% for the prior art).

----

## [569] Uncertainty-aware Active Learning for Optimal Bayesian Classifier

**Authors**: *Guang Zhao, Edward R. Dougherty, Byung-Jun Yoon, Francis J. Alexander, Xiaoning Qian*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Mu2ZxFctAI](https://openreview.net/forum?id=Mu2ZxFctAI)

**Abstract**:

For pool-based active learning, in each iteration a candidate training sample is chosen for labeling by optimizing an acquisition function. In Bayesian classification, expected Loss Reduction~(ELR) methods maximize the expected reduction in the classification error given a new labeled candidate based on a one-step-look-ahead strategy. ELR is the optimal strategy with a single query; however, since such myopic strategies cannot identify the long-term effect of a query on the classification error, ELR may get stuck before reaching the optimal classifier.  In this paper, inspired by the mean objective cost of uncertainty (MOCU), a metric quantifying the uncertainty directly affecting the classification error, we propose an acquisition function based on a weighted form of MOCU. Similar to ELR, the proposed method focuses on the reduction of the uncertainty that pertains to the classification error. But unlike any other existing scheme, it provides the critical advantage that the resulting Bayesian active learning algorithm guarantees convergence to the optimal classifier of the true model. We demonstrate its performance with both synthetic and real-world datasets.

----

## [570] Latent Skill Planning for Exploration and Transfer

**Authors**: *Kevin Xie, Homanga Bharadhwaj, Danijar Hafner, Animesh Garg, Florian Shkurti*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jXe91kq3jAq](https://openreview.net/forum?id=jXe91kq3jAq)

**Abstract**:

To quickly solve new tasks in complex environments, intelligent agents need to build up reusable knowledge. For example, a learned world model captures knowledge about the environment that applies to new tasks. Similarly, skills capture general behaviors that can apply to new tasks. In this paper, we investigate how these two approaches can be integrated into a single reinforcement learning agent. Specifically, we leverage the idea of partial amortization for fast adaptation at test time. For this, actions are produced by a policy that is learned over time while the skills it conditions on are chosen using online planning. We demonstrate the benefits of our design decisions across a suite of challenging locomotion tasks and demonstrate improved sample efficiency in single tasks as well as in transfer from one task to another, as compared to competitive baselines. Videos are available at: https://sites.google.com/view/latent-skill-planning/

----

## [571] Learning continuous-time PDEs from sparse data with graph neural networks

**Authors**: *Valerii Iakovlev, Markus Heinonen, Harri Lähdesmäki*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aUX5Plaq7Oy](https://openreview.net/forum?id=aUX5Plaq7Oy)

**Abstract**:

The behavior of many dynamical systems follow complex, yet still unknown partial differential equations (PDEs). While several machine learning methods have been proposed to learn PDEs directly from data, previous methods are limited to discrete-time approximations or make the limiting assumption of the observations arriving at regular grids. We propose a general continuous-time differential model for dynamical systems whose governing equations are parameterized by message passing graph neural networks. The model admits arbitrary space and time discretizations, which removes constraints on the locations of observation points and time intervals between the observations. The model is trained with continuous-time adjoint method enabling efficient neural PDE inference. We demonstrate the model's ability to work with unstructured grids, arbitrary time steps, and noisy observations. We compare our method with existing approaches on several well-known physical systems that involve first and higher-order PDEs with state-of-the-art predictive performance.

----

## [572] Characterizing signal propagation to close the performance gap in unnormalized ResNets

**Authors**: *Andrew Brock, Soham De, Samuel L. Smith*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IX3Nnir2omJ](https://openreview.net/forum?id=IX3Nnir2omJ)

**Abstract**:

Batch Normalization is a key component in almost all state-of-the-art image classifiers, but it also introduces practical challenges: it breaks the independence between training examples within a batch, can incur compute and memory overhead, and often results in unexpected bugs. Building on recent theoretical analyses of deep ResNets at initialization, we propose a simple set of analysis tools to characterize signal propagation on the forward pass, and leverage these tools to design highly performant ResNets without activation normalization layers.  Crucial to our success is an adapted version of the recently proposed Weight Standardization.  Our analysis tools show how this technique preserves the signal in ReLU networks by ensuring that the per-channel activation means do not grow with depth. Across a range of FLOP budgets, our networks attain performance competitive with state-of-the-art EfficientNets on ImageNet.

----

## [573] Robust Overfitting may be mitigated by properly learned smoothening

**Authors**: *Tianlong Chen, Zhenyu Zhang, Sijia Liu, Shiyu Chang, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qZzy5urZw9](https://openreview.net/forum?id=qZzy5urZw9)

**Abstract**:

A recent study (Rice et al.,  2020) revealed overfitting to be a dominant phenomenon in adversarially robust training of deep networks, and that appropriate early-stopping of adversarial training (AT) could match the performance gains of most recent algorithmic improvements. This intriguing problem of robust overfitting motivates us to seek more remedies. As a pilot study, this paper investigates two empirical means to inject more learned smoothening during AT: one leveraging knowledge distillation and self-training to smooth the logits, the other performing stochastic weight averaging (Izmailov et al., 2018) to smooth the weights. Despite the embarrassing simplicity, the two approaches are surprisingly effective and hassle-free in mitigating robust overfitting. Experiments demonstrate that by plugging in them to AT, we can simultaneously boost the standard accuracy by $3.72\%\sim6.68\%$ and robust accuracy by $0.22\%\sim2 .03\%$, across multiple datasets (STL-10, SVHN, CIFAR-10, CIFAR-100, and Tiny ImageNet), perturbation types ($\ell_{\infty}$ and $\ell_2$), and robustified methods (PGD, TRADES, and FSGM), establishing the new state-of-the-art bar in AT. We present systematic visualizations and analyses to dive into their possible working mechanisms. We also carefully exclude the possibility of gradient masking by evaluating our models' robustness against transfer attacks. Codes are available at https://github.com/VITA-Group/Alleviate-Robust-Overfitting.

----

## [574] Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning

**Authors**: *Tianlong Chen, Zhenyu Zhang, Sijia Liu, Shiyu Chang, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LXMSvPmsm0g](https://openreview.net/forum?id=LXMSvPmsm0g)

**Abstract**:

The lottery ticket hypothesis states that a highly sparsified sub-network can be trained in isolation, given the appropriate weight initialization. This paper extends that hypothesis from one-shot task learning, and demonstrates for the first time that such extremely compact and independently trainable sub-networks can be also identified in the lifelong learning scenario, which we call lifelong tickets. We show that the resulting lifelong ticket can further be leveraged to improve the performance of learning over continual tasks. However, it is highly non-trivial to conduct network pruning in the lifelong setting. Two critical roadblocks arise: i) As many tasks now arrive sequentially, finding tickets in a greedy weight pruning fashion will inevitably suffer from the intrinsic bias, that the earlier emerging tasks impact more; ii) As lifelong learning is consistently challenged by catastrophic forgetting, the compact network capacity of tickets might amplify the risk of forgetting. In view of those, we introduce two pruning options, e.g., top-down and bottom-up, for finding lifelong tickets. Compared to the top-down pruning that extends vanilla (iterative) pruning over sequential tasks, we show that the bottom-up one, which can dynamically shrink and (re-)expand model capacity, effectively avoids the undesirable excessive pruning in the early stage. We additionally introduce lottery teaching that further overcomes forgetting via knowledge distillation aided by external unlabeled data. Unifying those ingredients, we demonstrate the existence of very competitive lifelong tickets, e.g., achieving 3-8% of the dense model size with even higher accuracy, compared to strong class-incremental learning baselines on CIFAR-10/CIFAR-100/Tiny-ImageNet datasets. Codes available at https://github.com/VITA-Group/Lifelong-Learning-LTH.

----

## [575] Symmetry-Aware Actor-Critic for 3D Molecular Design

**Authors**: *Gregor N. C. Simm, Robert Pinsler, Gábor Csányi, José Miguel Hernández-Lobato*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jEYKjPE1xYN](https://openreview.net/forum?id=jEYKjPE1xYN)

**Abstract**:

Automating molecular design using deep reinforcement learning (RL) has the potential to greatly accelerate the search for novel materials. Despite recent progress on leveraging graph representations to design molecules, such methods are fundamentally limited by the lack of three-dimensional (3D) information. In light of this, we propose a novel actor-critic architecture for 3D molecular design that can generate molecular structures unattainable with previous approaches. This is achieved by exploiting the symmetries of the design process through a rotationally covariant state-action representation based on a spherical harmonics series expansion. We demonstrate the benefits of our approach on several 3D molecular design tasks, where we find that building in such symmetries significantly improves generalization and the quality of generated molecules.

----

## [576] PseudoSeg: Designing Pseudo Labels for Semantic Segmentation

**Authors**: *Yuliang Zou, Zizhao Zhang, Han Zhang, Chun-Liang Li, Xiao Bian, Jia-Bin Huang, Tomas Pfister*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-TwO99rbVRu](https://openreview.net/forum?id=-TwO99rbVRu)

**Abstract**:

Recent advances in semi-supervised learning (SSL) demonstrate that a combination of consistency regularization and pseudo-labeling can effectively improve image classification accuracy in the low-data regime. Compared to classification, semantic segmentation tasks require much more intensive labeling costs. Thus, these tasks greatly benefit from data-efficient training methods. However, structured outputs in segmentation render particular difficulties (e.g., designing pseudo-labeling and augmentation) to apply existing SSL strategies. To address this problem, we present a simple and novel re-design of pseudo-labeling to generate well-calibrated structured pseudo labels for training with unlabeled or weakly-labeled data. Our proposed pseudo-labeling strategy is network structure agnostic to apply in a one-stage consistency training framework. We demonstrate the effectiveness of the proposed pseudo-labeling strategy in both low-data and high-data regimes. Extensive experiments have validated that pseudo labels generated from wisely fusing diverse sources and strong data augmentation are crucial to consistency training for segmentation. The source code will be released.

----

## [577] NAS-Bench-ASR: Reproducible Neural Architecture Search for Speech Recognition

**Authors**: *Abhinav Mehrotra, Alberto Gil C. P. Ramos, Sourav Bhattacharya, Lukasz Dudziak, Ravichander Vipperla, Thomas Chau, Mohamed S. Abdelfattah, Samin Ishtiaq, Nicholas Donald Lane*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CU0APx9LMaL](https://openreview.net/forum?id=CU0APx9LMaL)

**Abstract**:

Powered by innovations in novel architecture design, noise tolerance techniques and increasing model capacity, Automatic Speech Recognition (ASR) has made giant strides in reducing word-error-rate over the past decade. ASR models are often trained with tens of thousand hours of high quality speech data to produce state-of-the-art (SOTA) results. Industry-scale ASR model training thus remains computationally heavy and time-consuming, and consequently has attracted little attention in adopting automatic techniques. On the other hand, Neural Architecture Search (NAS) has gained a lot of interest in the recent years thanks to its successes in discovering efficient architectures, often outperforming handcrafted alternatives. However, by changing the standard training process into a bi-level optimisation problem, NAS approaches often require significantly more time and computational power compared to single-model training, and at the same time increase complexity of the overall process. As a result, NAS has been predominately applied to problems which do not require as extensive training as ASR, and even then reproducibility of NAS algorithms is often problematic. Lately, a number of benchmark datasets has been introduced to address reproducibility issues by pro- viding NAS researchers with information about performance of different models obtained through exhaustive evaluation. However, these datasets focus mainly on computer vision and NLP tasks and thus suffer from limited coverage of application domains. In order to increase diversity in the existing NAS benchmarks, and at the same time provide systematic study of the effects of architectural choices for ASR, we release NAS-Bench-ASR – the first NAS benchmark for ASR models. The dataset consists of 8, 242 unique models trained on the TIMIT audio dataset for three different target epochs, and each starting from three different initializations. The dataset also includes runtime measurements of all the models on a diverse set of hardware platforms. Lastly, we show that identified good cell structures in our search space for TIMIT transfer well to a much larger LibriSpeech dataset.

----

## [578] Scaling the Convex Barrier with Active Sets

**Authors**: *Alessandro De Palma, Harkirat S. Behl, Rudy Bunel, Philip H. S. Torr, M. Pawan Kumar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uQfOy7LrlTR](https://openreview.net/forum?id=uQfOy7LrlTR)

**Abstract**:

Tight and efficient neural network bounding is of critical importance for the scaling of neural network verification systems. A number of efficient specialised dual solvers for neural network bounds have been presented recently, but they are often too loose to verify more challenging properties. This lack of tightness is linked to the weakness of the employed relaxation, which is usually a linear program of size linear in the number of neurons. While a tighter linear relaxation for piecewise linear activations exists, it comes at the cost of exponentially many constraints and thus currently lacks an efficient customised solver. We alleviate this deficiency via a novel dual algorithm that realises the full potential of the new relaxation by operating on a small active set of dual variables. Our method recovers the strengths of the new relaxation in the dual space: tightness and a linear separation oracle. At the same time, it shares the benefits of previous dual approaches for weaker relaxations: massive parallelism, GPU implementation, low cost per iteration and valid bounds at any time. As a consequence, we obtain better bounds than off-the-shelf solvers in only a fraction of their running time and recover the speed-accuracy trade-offs of looser dual solvers if the computational budget is small. We demonstrate that this results in significant formal verification speed-ups.

----

## [579] Local Convergence Analysis of Gradient Descent Ascent with Finite Timescale Separation

**Authors**: *Tanner Fiez, Lillian J. Ratliff*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AWOSz_mMAPx](https://openreview.net/forum?id=AWOSz_mMAPx)

**Abstract**:

We study the role that a finite timescale separation parameter $\tau$ has on gradient descent-ascent in non-convex, non-concave zero-sum games where the learning rate of player 1 is denoted by $\gamma_1$ and the learning rate of player 2 is defined to be $\gamma_2=\tau\gamma_1$. We provide a non-asymptotic construction of the finite timescale separation parameter $\tau^{\ast}$ such that gradient descent-ascent locally converges to  $x^{\ast}$ for all $\tau \in (\tau^{\ast}, \infty)$ if and only if it is a strict local minmax equilibrium. Moreover, we provide explicit local convergence rates given the finite timescale separation. The convergence results we present are complemented by a non-convergence result: given a critical point $x^{\ast}$ that is not a strict local minmax equilibrium, we present a non-asymptotic construction of a finite timescale separation $\tau_{0}$ such that gradient descent-ascent with timescale separation $\tau\in (\tau_0, \infty)$ does not converge to $x^{\ast}$. Finally, we extend the results to gradient penalty regularization methods for generative adversarial networks and empirically demonstrate on CIFAR-10 and CelebA the significant impact timescale separation has on training performance.

----

## [580] Activation-level uncertainty in deep neural networks

**Authors**: *Pablo Morales-Alvarez, Daniel Hernández-Lobato, Rafael Molina, José Miguel Hernández-Lobato*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UvBPbpvHRj-](https://openreview.net/forum?id=UvBPbpvHRj-)

**Abstract**:

Current approaches for uncertainty estimation in deep learning often produce too confident results. Bayesian Neural Networks (BNNs) model uncertainty in the space of weights, which is usually high-dimensional and limits the quality of variational approximations. The more recent functional BNNs (fBNNs) address this only partially because, although the prior is specified in the space of functions, the posterior approximation is still defined in terms of stochastic weights. In this work we propose to move uncertainty from the weights (which are deterministic) to the activation function. Specifically, the activations are modelled with simple 1D Gaussian Processes (GP), for which a triangular kernel inspired by the ReLu non-linearity is explored. Our experiments show that activation-level stochasticity provides more reliable uncertainty estimates than BNN and fBNN, whereas it performs competitively in standard prediction tasks. We also study the connection with deep GPs, both theoretically and empirically. More precisely, we show that activation-level uncertainty requires fewer inducing points and is better suited for deep architectures.

----

## [581] Efficient Continual Learning with Modular Networks and Task-Driven Priors

**Authors**: *Tom Veniat, Ludovic Denoyer, Marc'Aurelio Ranzato*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EKV158tSfwv](https://openreview.net/forum?id=EKV158tSfwv)

**Abstract**:

Existing literature in Continual Learning (CL) has focused on overcoming catastrophic forgetting, the inability of the learner to recall how to perform tasks observed in the past. 
There are however other desirable properties of a CL system, such as the ability to transfer knowledge from previous tasks and to scale memory and compute sub-linearly with the number of tasks. Since most current benchmarks focus only on forgetting using short streams of tasks, we first propose a new suite of benchmarks to probe CL algorithms across these new axes. 
Finally, we introduce a new modular architecture, whose modules represent atomic skills that can be composed to perform a certain task. Learning a task reduces to figuring out which past modules to re-use, and which new modules to instantiate to solve the current task. Our learning algorithm leverages a task-driven prior over the exponential search space of all possible ways to combine modules, enabling efficient learning on long streams of tasks. 
Our experiments show that this modular architecture and learning algorithm perform competitively on widely used CL benchmarks while yielding superior performance on the more challenging benchmarks we introduce in this work. The Benchmark is publicly available at https://github.com/facebookresearch/CTrLBenchmark.

----

## [582] No Cost Likelihood Manipulation at Test Time for Making Better Mistakes in Deep Networks

**Authors**: *Shyamgopal Karthik, Ameya Prabhu, Puneet K. Dokania, Vineet Gandhi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=193sEnKY1ij](https://openreview.net/forum?id=193sEnKY1ij)

**Abstract**:

There has been increasing interest in building deep hierarchy-aware classifiers that aim to quantify and reduce the severity of mistakes, and not just reduce the number of errors. The idea is to exploit the label hierarchy (e.g., the WordNet ontology) and consider graph distances as a proxy for mistake severity. Surprisingly, on examining mistake-severity distributions of the top-1 prediction, we find that current state-of-the-art hierarchy-aware deep classifiers do not always show practical improvement over the standard cross-entropy baseline in making better mistakes. The reason for the reduction in average mistake-severity can be attributed to the increase in low-severity mistakes, which may also explain the noticeable drop in their accuracy. To this end, we use the classical Conditional Risk Minimization (CRM) framework for hierarchy-aware classification. Given a cost matrix and a reliable estimate of likelihoods (obtained from a trained network), CRM simply amends mistakes at inference time; it needs no extra hyperparameters and requires adding just a few lines of code to the standard cross-entropy baseline. It significantly outperforms the state-of-the-art and consistently obtains large reductions in the average hierarchical distance of top-$k$ predictions across datasets, with very little loss in accuracy. CRM, because of its simplicity, can be used with any off-the-shelf trained model that provides reliable likelihood estimates.

----

## [583] Ringing ReLUs: Harmonic Distortion Analysis of Nonlinear Feedforward Networks

**Authors**: *Christian H. X. Ali Mehmeti-Göpel, David Hartmann, Michael Wand*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TaYhv-q1Xit](https://openreview.net/forum?id=TaYhv-q1Xit)

**Abstract**:

In this paper, we apply harmonic distortion analysis to understand the effect of nonlinearities in the spectral domain. Each nonlinear layer creates higher-frequency harmonics, which we call "blueshift", whose magnitude increases with network depth, thereby increasing the “roughness” of the output landscape. Unlike differential models (such as vanishing gradients, sharpness), this provides a more global view of how network architectures behave across larger areas of their parameter domain. For example, the model predicts that residual connections are able to counter the effect by dampening corresponding higher frequency modes. We empirically verify the connection between blueshift and architectural choices, and provide evidence for a connection with trainability.

----

## [584] Distance-Based Regularisation of Deep Networks for Fine-Tuning

**Authors**: *Henry Gouk, Timothy M. Hospedales, Massimiliano Pontil*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IFqrg1p5Bc](https://openreview.net/forum?id=IFqrg1p5Bc)

**Abstract**:

We investigate approaches to regularisation during fine-tuning of deep neural networks. First we provide a neural network generalisation bound based on Rademacher complexity that uses the distance the weights have moved from their initial values. This bound has no direct dependence on the number of weights and compares favourably to other bounds when applied to convolutional networks. Our bound is highly relevant for fine-tuning, because providing a network with a good initialisation based on transfer learning means that learning can modify the weights less, and hence achieve tighter generalisation. Inspired by this, we develop a simple yet effective fine-tuning algorithm that constrains the hypothesis class to a small sphere centred on the initial pre-trained weights, thus obtaining provably better generalisation performance than conventional transfer learning. Empirical evaluation shows that our algorithm works well, corroborating our theoretical results. It outperforms both state of the art fine-tuning competitors, and penalty-based alternatives that we show do not directly constrain the radius of the search space.

----

## [585] Genetic Soft Updates for Policy Evolution in Deep Reinforcement Learning

**Authors**: *Enrico Marchesini, Davide Corsi, Alessandro Farinelli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TGFO0DbD_pk](https://openreview.net/forum?id=TGFO0DbD_pk)

**Abstract**:

The combination of Evolutionary Algorithms (EAs) and Deep Reinforcement Learning (DRL) has been recently proposed to merge the benefits of both solutions. Existing mixed approaches, however, have been successfully applied only to actor-critic methods and present significant overhead. We address these issues by introducing a novel mixed framework that exploits a periodical genetic evaluation to soft update the weights of a DRL agent. The resulting approach is applicable with any DRL method and, in a worst-case scenario, it does not exhibit detrimental behaviours. Experiments in robotic applications and continuous control benchmarks demonstrate the versatility of our approach that significantly outperforms prior DRL, EAs, and mixed approaches. Finally, we employ formal verification to confirm the policy improvement, mitigating the inefficient exploration and hyper-parameter sensitivity of DRL.ment, mitigating the inefficient exploration and hyper-parameter sensitivity of DRL.

----

## [586] Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis

**Authors**: *Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1Fqg133qRaI](https://openreview.net/forum?id=1Fqg133qRaI)

**Abstract**:

Training Generative Adversarial Networks (GAN) on high-fidelity images usually requires large-scale GPU-clusters and a vast number of training images. In this paper, we study the few-shot image synthesis task for GAN with minimum computing cost. We propose a light-weight GAN structure that gains superior quality on 1024^2 resolution. Notably, the model converges from scratch with just a few hours of training on a single RTX-2080 GPU, and has a consistent performance, even with less than 100 training samples. Two technique designs constitute our work, a skip-layer channel-wise excitation module and a self-supervised discriminator trained as a feature-encoder. With thirteen datasets covering a wide variety of image domains (The datasets and code are available at https://github.com/odegeasslbc/FastGAN-pytorch), we show our model's superior performance compared to the state-of-the-art StyleGAN2, when data and computing budget are limited.

----

## [587] IsarStep: a Benchmark for High-level Mathematical Reasoning

**Authors**: *Wenda Li, Lei Yu, Yuhuai Wu, Lawrence C. Paulson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Pzj6fzU6wkj](https://openreview.net/forum?id=Pzj6fzU6wkj)

**Abstract**:

A well-defined benchmark is essential for measuring and accelerating research progress of machine learning models. In this paper, we present a benchmark for high-level mathematical reasoning and study the reasoning capabilities of neural sequence-to-sequence models. We build a non-synthetic dataset from the largest repository of proofs written by human experts in a theorem prover. The dataset has a broad coverage of undergraduate and research-level mathematical and computer science theorems. In our defined task, a model is required to fill in a missing intermediate proposition given surrounding proofs. This task provides a starting point for the long-term goal of having machines generate human-readable proofs automatically. Our experiments and analysis reveal that while the task is challenging, neural models can capture non-trivial mathematical reasoning. We further design a hierarchical transformer that outperforms the transformer baseline.

----

## [588] Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network

**Authors**: *James Diffenderfer, Bhavya Kailkhura*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=U_mat0b9iv](https://openreview.net/forum?id=U_mat0b9iv)

**Abstract**:

Recently, Frankle & Carbin (2019) demonstrated that randomly-initialized dense networks contain subnetworks that once found can be trained to reach test accuracy comparable to the trained dense network. However, finding these high performing trainable subnetworks is expensive, requiring iterative process of training and pruning weights. In this paper, we propose (and prove) a stronger Multi-Prize Lottery Ticket Hypothesis:

A sufficiently over-parameterized neural network with random weights contains several subnetworks (winning tickets) that (a) have comparable accuracy to a dense target network with learned weights (prize 1), (b) do not require any further training to achieve prize 1 (prize 2), and (c) is robust to extreme forms of quantization (i.e., binary weights and/or activation) (prize 3).

This provides a new paradigm for learning compact yet highly accurate binary neural networks simply by pruning and quantizing randomly weighted full precision neural networks. We also propose an algorithm for finding multi-prize tickets (MPTs) and test it by performing a series of experiments on CIFAR-10 and ImageNet datasets. Empirical results indicate that as models grow deeper and wider, multi-prize tickets start to reach similar (and sometimes even higher) test accuracy compared to their significantly larger and full-precision counterparts that have been weight-trained. Without ever updating the weight values, our MPTs-1/32 not only set new binary weight network state-of-the-art (SOTA) Top-1 accuracy -- 94.8% on CIFAR-10 and 74.03% on ImageNet -- but also outperform their full-precision counterparts by 1.78% and 0.76%, respectively. Further, our MPT-1/1 achieves SOTA Top-1 accuracy (91.9%) for binary neural networks on CIFAR-10. Code and pre-trained models are available at: https://github.com/chrundle/biprop.

----

## [589] Average-case Acceleration for Bilinear Games and Normal Matrices

**Authors**: *Carles Domingo-Enrich, Fabian Pedregosa, Damien Scieur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=H0syOoy3Ash](https://openreview.net/forum?id=H0syOoy3Ash)

**Abstract**:

Advances in generative modeling and adversarial learning have given rise to renewed interest in smooth games. However, the absence of symmetry in the matrix of second derivatives poses challenges that are not present in the classical minimization framework. While a rich theory of average-case analysis has been developed for minimization problems, little is known in the context of smooth games. In this work we take a first step towards closing this gap by developing average-case optimal first-order methods for a subset of smooth games. 
We make the following three main contributions. First, we show that for zero-sum bilinear games the average-case optimal method is the optimal method for the minimization of the Hamiltonian. Second, we provide an explicit expression for the optimal method corresponding to normal matrices, potentially non-symmetric. Finally, we specialize it to matrices with eigenvalues located in a disk and show a provable speed-up compared to worst-case optimal algorithms. We illustrate our findings through benchmarks with a varying degree of mismatch with our assumptions.

----

## [590] Economic Hyperparameter Optimization with Blended Search Strategy

**Authors**: *Chi Wang, Qingyun Wu, Silu Huang, Amin Saied*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VbLH04pRA3](https://openreview.net/forum?id=VbLH04pRA3)

**Abstract**:

We study the problem of using low cost to search for hyperparameter configurations in a large search space with heterogeneous evaluation cost and model quality.
We propose a blended search strategy to combine the strengths of global and local search, and prioritize them on the fly with the goal of minimizing the total cost spent in finding good configurations. Our approach demonstrates robust performance for tuning both tree-based models and deep neural networks on a large AutoML benchmark, as well as superior performance in model quality, time, and resource consumption for a production transformer-based NLP model fine-tuning task.

----

## [591] BSQ: Exploring Bit-Level Sparsity for Mixed-Precision Neural Network Quantization

**Authors**: *Huanrui Yang, Lin Duan, Yiran Chen, Hai Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TiXl51SCNw8](https://openreview.net/forum?id=TiXl51SCNw8)

**Abstract**:

Mixed-precision quantization can potentially achieve the optimal tradeoff between performance and compression rate of deep neural networks, and thus, have been widely investigated. However, it lacks a systematic method to determine the exact quantization scheme. Previous methods either examine only a small manually-designed search space or utilize a cumbersome neural architecture search to explore the vast search space. These approaches cannot lead to an optimal quantization scheme efficiently. This work proposes bit-level sparsity quantization (BSQ) to tackle the mixed-precision quantization from a new angle of inducing bit-level sparsity. We consider each bit of quantized weights as an independent trainable variable and introduce a differentiable bit-sparsity regularizer. BSQ can induce all-zero bits across a group of weight elements and realize the dynamic precision reduction, leading to a mixed-precision quantization scheme of the original model. Our method enables the exploration of the full mixed-precision space with a single gradient-based optimization process, with only one hyperparameter to tradeoff the performance and compression. BSQ achieves both higher accuracy and higher bit reduction on various model architectures on the CIFAR-10 and ImageNet datasets comparing to previous methods.

----

## [592] AutoLRS: Automatic Learning-Rate Schedule by Bayesian Optimization on the Fly

**Authors**: *Yuchen Jin, Tianyi Zhou, Liangyu Zhao, Yibo Zhu, Chuanxiong Guo, Marco Canini, Arvind Krishnamurthy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=SlrqM9_lyju](https://openreview.net/forum?id=SlrqM9_lyju)

**Abstract**:

The learning rate (LR) schedule is one of the most important hyper-parameters needing careful tuning in training DNNs. However, it is also one of the least automated parts of machine learning systems and usually costs significant manual effort and computing. Though there are pre-defined LR schedules and optimizers with adaptive LR, they introduce new hyperparameters that need to be tuned separately for different tasks/datasets. In this paper, we consider the question: Can we automatically tune the LR over the course of training without human involvement? We propose an efficient method, AutoLRS, which automatically optimizes the LR for each training stage by modeling training dynamics. AutoLRS aims to find an LR that minimizes the validation loss, every $\tau$ steps. We formulate it as black-box optimization and solve it by Bayesian optimization (BO). However, collecting training instances for BO requires a system to evaluate each LR queried by BO's acquisition function for $\tau$ steps, which is prohibitively expensive in practice. Instead, we apply each candidate LR for only $\tau'\ll\tau$ steps and train an exponential model to predict the validation loss after $\tau$ steps. This mutual-training process between BO and the exponential model allows us to bound the number of training steps invested in the BO search. We demonstrate the advantages and the generality of AutoLRS through extensive experiments of training DNNs from diverse domains and using different optimizers. The LR schedules auto-generated by AutoLRS leads to a speedup of $1.22\times$, $1.43\times$, and $1.5\times$ when training ResNet-50, Transformer, and BERT, respectively, compared to the LR schedules in their original papers, and an average speedup of $1.31\times$ over state-of-the-art highly tuned LR schedules.

----

## [593] BERTology Meets Biology: Interpreting Attention in Protein Language Models

**Authors**: *Jesse Vig, Ali Madani, Lav R. Varshney, Caiming Xiong, Richard Socher, Nazneen Fatema Rajani*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YWtLZvLmud7](https://openreview.net/forum?id=YWtLZvLmud7)

**Abstract**:

Transformer architectures have proven to learn useful representations for protein classification and generation tasks. However, these representations present challenges in interpretability. In this work, we demonstrate a set of methods for analyzing protein Transformer models through the lens of attention. We show that attention: (1) captures the folding structure of proteins, connecting amino acids that are far apart in the underlying sequence, but spatially close in the three-dimensional structure, (2) targets binding sites, a key functional component of proteins, and (3) focuses on progressively more complex biophysical properties with increasing layer depth. We find this behavior to be consistent across three Transformer architectures (BERT, ALBERT, XLNet) and two distinct protein datasets. We also present a three-dimensional visualization of the interaction between attention and protein structure. Code for visualization and analysis is available at https://github.com/salesforce/provis.

----

## [594] Learning Task-General Representations with Generative Neuro-Symbolic Modeling

**Authors**: *Reuben Feinman, Brenden M. Lake*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qzBUIzq5XR2](https://openreview.net/forum?id=qzBUIzq5XR2)

**Abstract**:

People can learn rich, general-purpose conceptual representations from only raw perceptual inputs. Current machine learning approaches fall well short of these human standards, although different modeling traditions often have complementary strengths. Symbolic models can capture the compositional and causal knowledge that enables flexible generalization, but they struggle to learn from raw inputs, relying on strong abstractions and simplifying assumptions. Neural network models can learn directly from raw data, but they struggle to capture compositional and causal structure and typically must retrain to tackle new tasks. We bring together these two traditions to learn generative models of concepts that capture rich compositional and causal structure, while learning from raw data. We develop a generative neuro-symbolic (GNS) model of handwritten character concepts that uses the control flow of a probabilistic program, coupled with symbolic stroke primitives and a symbolic image renderer, to represent the causal and compositional processes by which characters are formed. The distributions of parts (strokes), and correlations between parts, are modeled with neural network subroutines, allowing the model to learn directly from raw data and express nonparametric statistical relationships. We apply our model to the Omniglot challenge of human-level concept learning, using a background set of alphabets to learn an expressive prior distribution over character drawings. In a subsequent evaluation, our GNS model uses probabilistic inference to learn rich conceptual representations from a single training image that generalize to 4 unique tasks, succeeding where previous work has fallen short.

----

## [595] Zero-shot Synthesis with Group-Supervised Learning

**Authors**: *Yunhao Ge, Sami Abu-El-Haija, Gan Xin, Laurent Itti*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8wqCDnBmnrT](https://openreview.net/forum?id=8wqCDnBmnrT)

**Abstract**:

Visual cognition of primates is superior to that of artificial neural networks in its ability to “envision” a visual object, even a newly-introduced one, in different attributes including pose, position, color, texture, etc.  To aid neural networks to envision objects with different attributes,  we propose a family of objective functions, expressed on groups of examples, as a novel learning framework that we term Group-Supervised Learning (GSL). GSL allows us to decompose inputs into a disentangled representation with swappable components, that can be recombined to synthesize new samples.  For instance, images of red boats & blue cars can be decomposed and recombined to synthesize novel images of red cars.   We propose an implementation based on auto-encoder, termed group-supervised zero-shot synthesis network (GZS-Net) trained with our learning framework, that can produce a high-quality red car even if no such example is witnessed during training. We test our model and learning framework on existing benchmarks, in addition to a new dataset that we open-source. We qualitatively and quantitatively demonstrate that GZS-Net trained with GSL outperforms state-of-the-art methods

----

## [596] Selective Classification Can Magnify Disparities Across Groups

**Authors**: *Erik Jones, Shiori Sagawa, Pang Wei Koh, Ananya Kumar, Percy Liang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=N0M_4BkQ05i](https://openreview.net/forum?id=N0M_4BkQ05i)

**Abstract**:

Selective classification, in which models can abstain on uncertain predictions, is a natural approach to improving accuracy in settings where errors are costly but abstentions are manageable. In this paper, we find that while selective classification can improve average accuracies, it can simultaneously magnify existing accuracy disparities between various groups within a population, especially in the presence of spurious correlations. We observe this behavior consistently across five vision and NLP datasets. Surprisingly, increasing abstentions can even decrease accuracies on some groups. To better understand this phenomenon, we study the margin distribution, which captures the model’s confidences over all predictions. For symmetric margin distributions, we prove that whether selective classification monotonically improves or worsens accuracy is fully determined by the accuracy at full coverage (i.e., without any abstentions) and whether the distribution satisfies a property we call left-log-concavity. Our analysis also shows that selective classification tends to magnify full-coverage accuracy disparities. Motivated by our analysis, we train distributionally-robust models that achieve similar full-coverage accuracies across groups and show that selective classification uniformly improves each group on these models. Altogether, our results suggest that selective classification should be used with care and underscore the importance of training models to perform equally well across groups at full coverage.

----

## [597] Better Fine-Tuning by Reducing Representational Collapse

**Authors**: *Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, Sonal Gupta*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OQ08SN70M1V](https://openreview.net/forum?id=OQ08SN70M1V)

**Abstract**:

Although widely adopted, existing approaches for fine-tuning pre-trained language models have been shown to be unstable across hyper-parameter settings, motivating recent work on trust region methods. In this paper, we present a simplified and efficient method rooted in trust region theory that replaces previously used adversarial objectives with parametric noise (sampling from either a normal or uniform distribution), thereby discouraging representation change during fine-tuning when possible without hurting performance. We also introduce a new analysis to motivate the use of trust region methods more generally, by studying representational collapse; the degradation of generalizable representations from pre-trained models as they are fine-tuned for a specific end task. Extensive experiments show that our fine-tuning method matches or exceeds the performance of previous trust region methods on a range of understanding and generation tasks (including DailyMail/CNN, Gigaword, Reddit TIFU, and the GLUE benchmark), while also being much faster. We also show that it is less prone to representation collapse; the pre-trained models maintain more generalizable representations every time they are fine-tuned.

----

## [598] Training independent subnetworks for robust prediction

**Authors**: *Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai, Dustin Tran*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OGg9XnKxFAH](https://openreview.net/forum?id=OGg9XnKxFAH)

**Abstract**:

Recent approaches to efficiently ensemble neural networks have shown that strong robustness and uncertainty performance  can be achieved with a negligible gain in parameters over the original network. However, these methods still require multiple forward passes for prediction, leading to a significant runtime cost. In this work, we show a surprising result:
the benefits of using multiple predictions can be achieved 'for free' under a single model's forward pass. In particular, we show that, using a multi-input multi-output (MIMO) configuration, one can utilize a single model's capacity to train multiple subnetworks that independently learn the task at hand. By ensembling the predictions made by the subnetworks, we improve model robustness without increasing compute. We observe a significant improvement in negative log-likelihood, accuracy, and calibration error on CIFAR10, CIFAR100,  ImageNet, and their out-of-distribution variants compared to previous methods.

----

## [599] Meta-Learning of Structured Task Distributions in Humans and Machines

**Authors**: *Sreejan Kumar, Ishita Dasgupta, Jonathan D. Cohen, Nathaniel D. Daw, Thomas L. Griffiths*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=--gvHfE3Xf5](https://openreview.net/forum?id=--gvHfE3Xf5)

**Abstract**:

In recent years, meta-learning, in which a model is trained on a family of tasks (i.e. a task distribution), has emerged as an approach to training neural networks to perform tasks that were previously assumed to require structured representations, making strides toward closing the gap between humans and machines. However, we argue that evaluating meta-learning remains a challenge, and can miss whether meta-learning actually uses the structure embedded within the tasks. These meta-learners might therefore still be significantly different from humans learners. To demonstrate this difference, we first define a new meta-reinforcement learning task in which a structured task distribution is generated using a compositional grammar. We then introduce a novel approach to constructing a "null task distribution" with the same statistical complexity as this structured task distribution but without the explicit rule-based structure used to generate the structured task. We train a standard meta-learning agent, a recurrent network trained with model-free reinforcement learning, and compare it with human performance across the two task distributions. We find a double dissociation in which humans do better in the structured task distribution whereas agents do better in the null task distribution -- despite comparable statistical complexity. This work highlights that multiple strategies can achieve reasonable meta-test performance, and that careful construction of control task distributions is a valuable way to understand which strategies meta-learners acquire, and how they might differ from humans.

----



[Go to the previous page](ICLR-2021-list02.md)

[Go to the next page](ICLR-2021-list04.md)

[Go to the catalog section](README.md)