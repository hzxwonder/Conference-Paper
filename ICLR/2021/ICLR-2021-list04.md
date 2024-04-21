## [600] BiPointNet: Binary Neural Network for Point Clouds

**Authors**: *Haotong Qin, Zhongang Cai, Mingyuan Zhang, Yifu Ding, Haiyu Zhao, Shuai Yi, Xianglong Liu, Hao Su*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9QLRCVysdlO](https://openreview.net/forum?id=9QLRCVysdlO)

**Abstract**:

To alleviate the resource constraint for real-time point cloud applications that run on edge devices, in this paper we present BiPointNet, the first model binarization approach for efficient deep learning on point clouds. We discover that the immense performance drop of binarized models for point clouds mainly stems from two challenges: aggregation-induced feature homogenization that leads to a degradation of information entropy, and scale distortion that hinders optimization and invalidates scale-sensitive structures. With theoretical justifications and in-depth analysis, our BiPointNet introduces Entropy-Maximizing Aggregation (EMA) to modulate the distribution before aggregation for the maximum information entropy, and Layer-wise Scale Recovery (LSR) to efficiently restore feature representation capacity. Extensive experiments show that BiPointNet outperforms existing binarization methods by convincing margins, at the level even comparable with the full precision counterpart. We highlight that our techniques are generic, guaranteeing significant improvements on various fundamental tasks and mainstream backbones. Moreover, BiPointNet gives an impressive 14.7× speedup and 18.9× storage saving on real-world resource-constrained devices.

----

## [601] Benchmarks for Deep Off-Policy Evaluation

**Authors**: *Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang, Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, Tom Le Paine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kWSeGEeHvF8](https://openreview.net/forum?id=kWSeGEeHvF8)

**Abstract**:

Off-policy evaluation (OPE) holds the promise of being able to leverage large, offline datasets for both evaluating and selecting complex policies for decision making. The ability to learn offline is particularly important in many real-world domains, such as in healthcare, recommender systems, or robotics, where online data collection is an expensive and potentially dangerous process. Being able to accurately evaluate and select high-performing policies without requiring online interaction could yield significant benefits in safety, time, and cost for these applications. While many OPE methods have been proposed in recent years, comparing results between papers is difficult because currently there is a lack of a comprehensive and unified benchmark, and measuring algorithmic progress has been challenging due to the lack of difficult evaluation tasks. In order to address this gap, we present a collection of policies that in conjunction with existing offline datasets can be used for benchmarking off-policy evaluation. Our tasks include a range of challenging high-dimensional continuous control problems, with wide selections of datasets and policies for performing policy selection. The goal of our benchmark is to provide a standardized measure of progress that is motivated from a set of principles designed to challenge and test the limits of existing OPE methods. We perform an evaluation of state-of-the-art algorithms and provide open-source access to our data and code to foster future research in this area.

----

## [602] Planning from Pixels using Inverse Dynamics Models

**Authors**: *Keiran Paster, Sheila A. McIlraith, Jimmy Ba*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=V6BjBgku7Ro](https://openreview.net/forum?id=V6BjBgku7Ro)

**Abstract**:

Learning dynamics models in high-dimensional observation spaces can be challenging for model-based RL agents. We propose a novel way to learn models in a latent space by learning to predict sequences of future actions conditioned on task completion. These models track task-relevant environment dynamics over a distribution of tasks, while simultaneously serving as an effective heuristic for planning with sparse rewards. We evaluate our method on challenging visual goal completion tasks and show a substantial increase in performance compared to prior model-free approaches.

----

## [603] Understanding the effects of data parallelism and sparsity on neural network training

**Authors**: *Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr, Martin Jaggi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rsogjAnYs4z](https://openreview.net/forum?id=rsogjAnYs4z)

**Abstract**:

We study two factors in neural network training: data parallelism and sparsity; here, data parallelism means processing training data in parallel using distributed systems (or equivalently increasing batch size), so that training can be accelerated; for sparsity, we refer to pruning parameters in a neural network model, so as to reduce computational and memory cost. Despite their promising benefits, however, understanding of their effects on neural network training remains elusive. In this work, we first measure these effects rigorously by conducting extensive experiments while tuning all metaparameters involved in the optimization. As a result, we find across various workloads of data set, network model, and optimization algorithm that there exists a general scaling trend between batch size and number of training steps to convergence for the effect of data parallelism, and further, difficulty of training under sparsity. Then, we develop a theoretical analysis based on the convergence properties of stochastic gradient methods and smoothness of the optimization landscape, which illustrates the observed phenomena precisely and generally, establishing a better account of the effects of data parallelism and sparsity on neural network training.

----

## [604] NOVAS: Non-convex Optimization via Adaptive Stochastic Search for End-to-end Learning and Control

**Authors**: *Ioannis Exarchos, Marcus Aloysius Pereira, Ziyi Wang, Evangelos A. Theodorou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Iw4ZGwenbXf](https://openreview.net/forum?id=Iw4ZGwenbXf)

**Abstract**:

In this work we propose the use of adaptive stochastic search as a building block for general, non-convex optimization operations within deep neural network architectures. Specifically, for an objective function located at some layer in the network and parameterized by some network parameters, we employ adaptive stochastic search to perform optimization over its output. This operation is differentiable and does not obstruct the passing of gradients during backpropagation, thus enabling us to incorporate it as a component in end-to-end learning. We study the proposed optimization module's properties and benchmark it against two existing alternatives on a synthetic energy-based structured prediction task, and further showcase its use in stochastic optimal control applications.

----

## [605] MoVie: Revisiting Modulated Convolutions for Visual Counting and Beyond

**Authors**: *Duy-Kien Nguyen, Vedanuj Goswami, Xinlei Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8e6BrwU6AjQ](https://openreview.net/forum?id=8e6BrwU6AjQ)

**Abstract**:

This paper focuses on visual counting, which aims to predict the number of occurrences given a natural image and a query (e.g. a question or a category). Unlike most prior works that use explicit, symbolic models which can be computationally expensive and limited in generalization, we propose a simple and effective alternative by revisiting modulated convolutions that fuse the query and the image locally. Following the design of residual bottleneck, we call our method MoVie, short for Modulated conVolutional bottlenecks. Notably, MoVie reasons implicitly and holistically and only needs a single forward-pass during inference. Nevertheless, MoVie showcases strong performance for counting: 1) advancing the state-of-the-art on counting-specific VQA tasks while being more efficient; 2) outperforming prior-art on difficult benchmarks like COCO for common object counting; 3) helped us secure the first place of 2020 VQA challenge when integrated as a module for ‘number’ related questions in generic VQA models. Finally, we show evidence that modulated convolutions such as MoVie can serve as a general mechanism for reasoning tasks beyond counting.

----

## [606] NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation

**Authors**: *Angtian Wang, Adam Kortylewski, Alan L. Yuille*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pmj131uIL9H](https://openreview.net/forum?id=pmj131uIL9H)

**Abstract**:

3D pose estimation is a challenging but important task in computer vision. In this work, we show that standard deep learning approaches to 3D pose estimation are not robust to partial occlusion. Inspired by the robustness of generative vision models to partial occlusion, we propose to integrate deep neural networks with 3D generative representations of objects into a unified neural architecture that we term NeMo. In particular, NeMo learns a generative model of neural feature activations at each vertex on a dense 3D mesh. Using differentiable rendering we estimate the 3D object pose by minimizing the reconstruction error between NeMo and the feature representation of the target image. To avoid local optima in the reconstruction loss, we train the feature extractor to maximize the distance between the individual feature representations on the mesh using contrastive learning. Our extensive experiments on PASCAL3D+, occluded-PASCAL3D+ and ObjectNet3D show that NeMo is much more robust to partial occlusion compared to standard deep networks, while retaining competitive performance on non-occluded data. Interestingly, our experiments also show that NeMo performs reasonably well even when the mesh representation only crudely approximates the true object geometry with a cuboid, hence revealing that the detailed 3D geometry is not needed for accurate 3D pose estimation.

----

## [607] On Graph Neural Networks versus Graph-Augmented MLPs

**Authors**: *Lei Chen, Zhengdao Chen, Joan Bruna*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tiqI7w64JG2](https://openreview.net/forum?id=tiqI7w64JG2)

**Abstract**:

From the perspectives of expressive power and learning, this work compares multi-layer Graph Neural Networks (GNNs) with a simplified alternative that we call Graph-Augmented Multi-Layer Perceptrons (GA-MLPs), which first augments node features with certain multi-hop operators on the graph and then applies learnable node-wise functions. From the perspective of graph isomorphism testing, we show both theoretically and numerically that GA-MLPs with suitable operators can distinguish almost all non-isomorphic graphs, just like the Weisfeiler-Lehman (WL) test and GNNs. However, by viewing them as node-level functions and examining the equivalence classes they induce on rooted graphs, we prove a separation in expressive power between GA-MLPs and GNNs that grows exponentially in depth. In particular, unlike GNNs, GA-MLPs are unable to count the number of attributed walks. We also demonstrate via community detection experiments that GA-MLPs can be limited by their choice of operator family, whereas GNNs have higher flexibility in learning.

----

## [608] Dual-mode ASR: Unify and Improve Streaming ASR with Full-context Modeling

**Authors**: *Jiahui Yu, Wei Han, Anmol Gulati, Chung-Cheng Chiu, Bo Li, Tara N. Sainath, Yonghui Wu, Ruoming Pang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Pz_dcqfcKW8](https://openreview.net/forum?id=Pz_dcqfcKW8)

**Abstract**:

Streaming automatic speech recognition (ASR) aims to emit each hypothesized word as quickly and accurately as possible, while full-context ASR waits for the completion of a full speech utterance before emitting completed hypotheses. In this work, we propose a unified framework, Dual-mode ASR, to train a single end-to-end ASR model with shared weights for both streaming and full-context speech recognition. We show that the latency and accuracy of streaming ASR significantly benefit from weight sharing and joint training of full-context ASR, especially with inplace knowledge distillation during the training. The Dual-mode ASR framework can be applied to recent state-of-the-art convolution-based and transformer-based ASR networks. We present extensive experiments with two state-of-the-art ASR networks, ContextNet and Conformer, on two datasets, a widely used public dataset LibriSpeech and a large-scale dataset MultiDomain. Experiments and ablation studies demonstrate that Dual-mode ASR not only simplifies the workflow of training and deploying streaming and full-context ASR models, but also significantly improves both emission latency and recognition accuracy of streaming ASR. With Dual-mode ASR, we achieve new state-of-the-art streaming ASR results on both LibriSpeech and MultiDomain in terms of accuracy and latency.

----

## [609] Deep Learning meets Projective Clustering

**Authors**: *Alaa Maalouf, Harry Lang, Daniela Rus, Dan Feldman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EQfpYwF3-b](https://openreview.net/forum?id=EQfpYwF3-b)

**Abstract**:

A common approach for compressing Natural Language Processing (NLP) networks is to encode the embedding layer as a matrix $A\in\mathbb{R}^{n\times d}$, compute its rank-$j$ approximation $A_j$ via SVD (Singular Value Decomposition), and then factor $A_j$ into a pair of matrices that correspond to smaller fully-connected layers to replace the original embedding layer. Geometrically, the rows of $A$ represent points in $\mathbb{R}^d$, and the rows of $A_j$ represent their projections onto the $j$-dimensional subspace that minimizes the sum of squared distances (``errors'') to the points. 
In practice, these rows of $A$ may be spread around $k>1$ subspaces, so factoring $A$ based on a single subspace may lead to large errors that turn into large drops in accuracy.

Inspired by \emph{projective clustering} from computational geometry,  we suggest replacing this subspace by a set of $k$ subspaces, each of dimension $j$, that minimizes the sum of squared distances over every point (row in $A$) to its \emph{closest} subspace. Based on this approach, we provide a novel architecture that replaces the original embedding layer by a set of $k$ small layers that operate in parallel and are then recombined with a single fully-connected layer. 

Extensive experimental results on the GLUE benchmark yield networks that are both more accurate and smaller compared to the standard matrix factorization (SVD). For example, we further compress DistilBERT by reducing the size of the embedding layer by $40\%$ while incurring only a $0.5\%$ average drop in accuracy over all nine GLUE tasks, compared to a $2.8\%$ drop using the existing SVD approach.
On RoBERTa we achieve $43\%$ compression of the embedding layer with less than a $0.8\%$ average drop in accuracy as compared to a $3\%$ drop previously.

----

## [610] Reinforcement Learning with Random Delays

**Authors**: *Yann Bouteiller, Simon Ramstedt, Giovanni Beltrame, Christopher J. Pal, Jonathan Binas*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QFYnKlBJYR](https://openreview.net/forum?id=QFYnKlBJYR)

**Abstract**:

Action and observation delays commonly occur in many Reinforcement Learning applications, such as remote control scenarios. We study the anatomy of randomly delayed environments, and show that partially resampling trajectory fragments in hindsight allows for off-policy multi-step value estimation. We apply this principle to derive Delay-Correcting Actor-Critic (DCAC), an algorithm based on Soft Actor-Critic with significantly better performance in environments with delays. This is shown theoretically and also demonstrated practically on a delay-augmented version of the MuJoCo continuous control benchmark.

----

## [611] Isotropy in the Contextual Embedding Space: Clusters and Manifolds

**Authors**: *Xingyu Cai, Jiaji Huang, Yuchen Bian, Kenneth Church*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xYGNO86OWDH](https://openreview.net/forum?id=xYGNO86OWDH)

**Abstract**:

The geometric properties of contextual embedding spaces for deep language models such as BERT and ERNIE, have attracted considerable attention in recent years. Investigations on the contextual embeddings demonstrate a strong anisotropic space such that most of the vectors fall within a narrow cone, leading to high cosine similarities.  It is surprising that these LMs are as successful as they are, given that most of their embedding vectors are as similar to one another as they are. In this paper, we argue that the isotropy indeed exists in the space, from a different but more constructive perspective. We identify isolated clusters and low dimensional manifolds in the contextual embedding space, and introduce tools to both qualitatively and quantitatively analyze them. We hope the study in this paper could provide insights towards a better understanding of the deep language models.

----

## [612] Spatio-Temporal Graph Scattering Transform

**Authors**: *Chao Pan, Siheng Chen, Antonio Ortega*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CF-ZIuSMXRz](https://openreview.net/forum?id=CF-ZIuSMXRz)

**Abstract**:

Although spatio-temporal graph neural networks have achieved great empirical success in handling multiple correlated time series, they may be impractical in some real-world scenarios due to a lack of sufficient high-quality training data. Furthermore, spatio-temporal graph neural networks lack theoretical interpretation. To address these issues, we put forth a novel mathematically designed framework to analyze spatio-temporal data. Our proposed spatio-temporal graph scattering transform (ST-GST) extends traditional scattering transform to the spatio-temporal domain. It performs iterative applications of spatio-temporal graph wavelets and  nonlinear activation functions, which can be viewed as a forward pass of spatio-temporal graph convolutional networks without training. Since all the filter coefficients in ST-GST are mathematically designed, it is promising for the real-world scenarios with limited training data, and also allows for a theoretical analysis, which shows that  the proposed ST-GST is stable to small perturbations of input signals and structures. Finally, our experiments show that i) ST-GST outperforms spatio-temporal graph convolutional networks by an increase of 35% in accuracy for MSR Action3D dataset; ii) it is  better and computationally more efficient to design the transform based on separable  spatio-temporal graphs than the joint ones; and iii) nonlinearity in ST-GST is critical to empirical performance.

----

## [613] Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization

**Authors**: *Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, Shixiang Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3hGNqpI4WS](https://openreview.net/forum?id=3hGNqpI4WS)

**Abstract**:

Most reinforcement learning (RL) algorithms assume online access to the environment, in which one may readily interleave updates to the policy with experience collection using that policy. However, in many real-world applications such as health, education, dialogue agents, and robotics, the cost or potential risk of deploying a new data-collection policy is high, to the point that it can become prohibitive to update the data-collection policy more than a few times during learning. With this view, we propose a novel concept of deployment efficiency, measuring the number of distinct data-collection policies that are used during policy learning. We observe that naïvely applying existing model-free offline RL algorithms recursively does not lead to a practical deployment-efficient and sample-efficient algorithm. We propose a novel model-based algorithm, Behavior-Regularized Model-ENsemble (BREMEN), that not only performs better than or comparably as the state-of-the-art dynamic-programming-based and concurrently-proposed model-based offline approaches on existing benchmarks, but can also effectively optimize a policy offline using 10-20 times fewer data than prior works. Furthermore, the recursive application of BREMEN achieves impressive deployment efficiency while maintaining the same or better sample efficiency, learning successful policies from scratch on simulated robotic environments with only 5-10 deployments, compared to typical values of hundreds to millions in standard RL baselines.

----

## [614] gradSim: Differentiable simulation for system identification and visuomotor control

**Authors**: *J. Krishna Murthy, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine, Jérôme Parent-Lévesque, Kevin Xie, Kenny Erleben, Liam Paull, Florian Shkurti, Derek Nowrouzezahrai, Sanja Fidler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=c_E8kFWfhp0](https://openreview.net/forum?id=c_E8kFWfhp0)

**Abstract**:

In this paper, we tackle the problem of estimating object physical properties such as mass, friction, and elasticity directly from video sequences. Such a system identification problem is fundamentally ill-posed due to the loss of information during image formation. Current best solutions to the problem require precise 3D labels which are labor intensive to gather, and infeasible to create for many systems such as deformable solids or cloth. In this work we present gradSim, a framework that overcomes the dependence on 3D supervision by combining differentiable multiphysics simulation and differentiable rendering to jointly model the evolution of scene dynamics and image formation. This unique combination enables backpropagation from pixels in a video sequence through to the underlying physical attributes that generated them. Furthermore, our unified computation graph across dynamics and rendering engines enables the learning of challenging visuomotor control tasks, without relying on state-based (3D) supervision, while obtaining performance competitive to/better than techniques that require precise 3D labels.

----

## [615] Evaluations and Methods for Explanation through Robustness Analysis

**Authors**: *Cheng-Yu Hsieh, Chih-Kuan Yeh, Xuanqing Liu, Pradeep Kumar Ravikumar, Seungyeon Kim, Sanjiv Kumar, Cho-Jui Hsieh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4dXmpCDGNp7](https://openreview.net/forum?id=4dXmpCDGNp7)

**Abstract**:

Feature based explanations, that provide importance of each feature towards the model prediction, is arguably one of the most intuitive ways to explain a model. In this paper, we establish a novel set of evaluation criteria for such feature based explanations by robustness analysis. In contrast to existing evaluations which require us to specify some way to "remove" features that could inevitably introduces biases and artifacts, we make use of the subtler notion of smaller adversarial perturbations. By optimizing towards our proposed evaluation criteria, we obtain new explanations that are loosely necessary and sufficient for a prediction. We further extend the explanation to extract the set of features that would move the current prediction to a target class by adopting targeted adversarial attack for the robustness analysis. Through experiments across multiple domains and a user study, we validate the usefulness of our evaluation criteria and our derived explanations.

----

## [616] RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs

**Authors**: *Meng Qu, Junkun Chen, Louis-Pascal A. C. Xhonneux, Yoshua Bengio, Jian Tang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tGZu6DlbreV](https://openreview.net/forum?id=tGZu6DlbreV)

**Abstract**:

This paper studies learning logic rules for reasoning on knowledge graphs. Logic rules provide interpretable explanations when used for prediction as well as being able to generalize to other tasks, and hence are critical to learn. Existing methods either suffer from the problem of searching in a large search space (e.g., neural logic programming) or ineffective optimization due to sparse rewards (e.g., techniques based on reinforcement learning). To address these limitations, this paper proposes a probabilistic model called RNNLogic. RNNLogic treats logic rules as a latent variable, and simultaneously trains a rule generator as well as a reasoning predictor with logic rules. We develop an EM-based algorithm for optimization. In each iteration, the reasoning predictor is updated to explore some generated logic rules for reasoning. Then in the E-step, we select a set of high-quality rules from all generated rules with both the rule generator and reasoning predictor via posterior inference; and in the M-step, the rule generator is updated with the rules selected in the E-step. Experiments on four datasets prove the effectiveness of RNNLogic.

----

## [617] Can a Fruit Fly Learn Word Embeddings?

**Authors**: *Yuchen Liang, Chaitanya K. Ryali, Benjamin Hoover, Leopold Grinberg, Saket Navlakha, Mohammed J. Zaki, Dmitry Krotov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xfmSoxdxFCG](https://openreview.net/forum?id=xfmSoxdxFCG)

**Abstract**:

The mushroom body of the fruit fly brain is one of the best studied systems in neuroscience. At its core it consists of a population of Kenyon cells, which receive inputs from multiple sensory modalities. These cells are inhibited by the anterior paired lateral neuron, thus creating a sparse high dimensional representation of the inputs. In this work we study a mathematical formalization of this network motif and apply it to learning the correlational structure between words and their context in a corpus of unstructured text, a common natural language processing (NLP) task. We show that this network can learn semantic representations of words and can generate both static and context-dependent word embeddings. Unlike conventional methods (e.g., BERT, GloVe) that use dense representations for word embedding, our algorithm encodes semantic meaning of words and their context in the form of sparse binary hash codes. The quality of the learned representations is evaluated on word similarity analysis, word-sense disambiguation, and document classification. It is shown that not only can the fruit fly network motif achieve performance comparable to existing methods in NLP, but, additionally, it uses only a fraction of the computational resources (shorter training time and smaller memory footprint).

----

## [618] Neural representation and generation for RNA secondary structures

**Authors**: *Zichao Yan, William L. Hamilton, Mathieu Blanchette*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=snOgiCYZgJ7](https://openreview.net/forum?id=snOgiCYZgJ7)

**Abstract**:

Our work is concerned with the generation and targeted design of RNA, a type of genetic macromolecule that can adopt complex structures which influence their cellular activities and functions. The design of large scale and complex biological structures spurs dedicated graph-based deep generative modeling techniques, which represents a key but underappreciated aspect of computational drug discovery. In this work, we investigate the principles behind representing and generating different RNA structural modalities, and propose a flexible framework to jointly embed and generate these molecular structures along with their sequence in a meaningful latent space. Equipped with a deep understanding of RNA molecular structures, our most sophisticated encoding and decoding methods operate on the molecular graph as well as the junction tree hierarchy, integrating strong inductive bias about RNA structural regularity and folding mechanism such that high structural validity, stability and diversity of generated RNAs are achieved. Also, we seek to adequately organize the latent space of RNA molecular embeddings with regard to the interaction with proteins, and targeted optimization is used to navigate in this latent space to search for desired novel RNA molecules.

----

## [619] WaNet - Imperceptible Warping-based Backdoor Attack

**Authors**: *Tuan Anh Nguyen, Anh Tuan Tran*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eEn8KTtJOx](https://openreview.net/forum?id=eEn8KTtJOx)

**Abstract**:

With the thriving of deep learning and the widespread practice of using pre-trained networks, backdoor attacks have become an increasing security threat drawing many research interests in recent years. A third-party model can be poisoned in training to work well in normal conditions but behave maliciously when a trigger pattern appears. However, the existing backdoor attacks are all built on noise perturbation triggers, making them noticeable to humans. In this paper, we instead propose using warping-based triggers. The proposed backdoor outperforms the previous methods in a human inspection test by a wide margin, proving its stealthiness. To make such models undetectable by machine defenders, we propose a novel training mode, called the ``noise mode. The trained networks successfully attack and bypass the state-ofthe art defense methods on standard classification datasets, including MNIST, CIFAR-10, GTSRB, and CelebA. Behavior analyses show that our backdoors are transparent to network inspection, further proving this novel attack mechanism's efficiency.

----

## [620] LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition

**Authors**: *Valeriia Cherepanova, Micah Goldblum, Harrison Foley, Shiyuan Duan, John P. Dickerson, Gavin Taylor, Tom Goldstein*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hJmtwocEqzc](https://openreview.net/forum?id=hJmtwocEqzc)

**Abstract**:

Facial recognition systems are increasingly deployed by private corporations, government agencies, and contractors for consumer services and mass surveillance programs alike.  These systems are typically built by scraping social media profiles for user images.  Adversarial perturbations have been proposed for bypassing facial recognition systems.  However, existing methods fail on full-scale systems and commercial APIs.  We develop our own adversarial filter that accounts for the entire image processing pipeline and is demonstrably effective against industrial-grade pipelines that include face detection and large scale databases.  Additionally, we release an easy-to-use webtool that significantly degrades the accuracy of Amazon Rekognition and the Microsoft Azure Face Recognition API, reducing the accuracy of each to below 1%.

----

## [621] Learning from others' mistakes: Avoiding dataset biases without modeling them

**Authors**: *Victor Sanh, Thomas Wolf, Yonatan Belinkov, Alexander M. Rush*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Hf3qXoiNkR](https://openreview.net/forum?id=Hf3qXoiNkR)

**Abstract**:

State-of-the-art natural language processing (NLP) models often learn to model dataset biases and surface form correlations instead of features that target the intended underlying task. Previous work has demonstrated effective methods to circumvent these issues when knowledge of the bias is available. We consider cases where the bias issues may not be explicitly identified, and show a method for training models that learn to ignore these problematic correlations. Our approach relies on the observation that models with limited capacity primarily learn to exploit biases in the dataset. We can leverage the errors of such limited capacity models to train a more robust model in a product of experts, thus bypassing the need to hand-craft a biased model. We show the effectiveness of this method to retain improvements in out-of-distribution settings even if no particular bias is targeted by the biased model.

----

## [622] Prototypical Contrastive Learning of Unsupervised Representations

**Authors**: *Junnan Li, Pan Zhou, Caiming Xiong, Steven C. H. Hoi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KmykpuSrjcq](https://openreview.net/forum?id=KmykpuSrjcq)

**Abstract**:

This paper presents Prototypical Contrastive Learning (PCL), an unsupervised representation learning method that bridges contrastive learning with clustering. PCL not only learns low-level features for the task of instance discrimination, but more importantly, it implicitly encodes semantic structures of the data into the learned embedding space. Specifically, we introduce prototypes as latent variables to help find the maximum-likelihood estimation of the network parameters in an Expectation-Maximization framework. We iteratively perform E-step as finding the distribution of prototypes via clustering and M-step as optimizing the network via contrastive learning. We propose ProtoNCE loss, a generalized version of the InfoNCE loss for contrastive learning, which encourages representations to be closer to their assigned prototypes. PCL outperforms state-of-the-art instance-wise contrastive learning methods on multiple benchmarks with substantial improvement in low-resource transfer learning. Code and pretrained models are available at https://github.com/salesforce/PCL.

----

## [623] Extreme Memorization via Scale of Initialization

**Authors**: *Harsh Mehta, Ashok Cutkosky, Behnam Neyshabur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Z4R1vxLbRLO](https://openreview.net/forum?id=Z4R1vxLbRLO)

**Abstract**:

We construct an experimental setup in which changing the scale of initialization strongly impacts the implicit regularization induced by SGD, interpolating from good generalization performance to completely memorizing the training set while making little progress on the test set. Moreover, we find that the extent and manner in which generalization ability is affected depends on the activation and loss function used, with sin activation being the most extreme. In the case of the homogeneous ReLU activation, we show that this behavior can be attributed to the loss function. Our empirical investigation reveals that increasing the scale of initialization correlates with misalignment of representations and gradients across examples in the same class. This insight allows us to device an alignment measure over gradients and representations which can capture this phenomenon. We demonstrate that our alignment measure correlates with generalization of deep models trained on image classification tasks.

----

## [624] Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling

**Authors**: *Benedikt Boecking, Willie Neiswanger, Eric P. Xing, Artur Dubrawski*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IDFQI9OY6K](https://openreview.net/forum?id=IDFQI9OY6K)

**Abstract**:

Obtaining large annotated datasets is critical for training successful machine learning models and it is often a bottleneck in practice. Weak supervision offers a promising alternative for producing labeled datasets without ground truth annotations by generating probabilistic labels using multiple noisy heuristics. This process can scale to large datasets and has demonstrated state of the art performance in diverse domains such as healthcare and e-commerce. One practical issue with learning from user-generated heuristics is that their creation requires creativity, foresight, and domain expertise from those who hand-craft them, a process which can be tedious and subjective. We develop the first framework for interactive weak supervision in which a method proposes heuristics and learns from user feedback given on each proposed heuristic. Our experiments demonstrate that only a small number of feedback iterations are needed to train models that achieve highly competitive test set performance without access to ground truth training labels. We conduct user studies, which show that users are able to effectively provide feedback on heuristics and that test set results track the performance of simulated oracles.

----

## [625] Adaptive Procedural Task Generation for Hard-Exploration Problems

**Authors**: *Kuan Fang, Yuke Zhu, Silvio Savarese, Li Fei-Fei*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8xLkv08d70T](https://openreview.net/forum?id=8xLkv08d70T)

**Abstract**:

We introduce Adaptive Procedural Task Generation (APT-Gen), an approach to progressively generate a sequence of tasks as curricula to facilitate reinforcement learning in hard-exploration problems. At the heart of our approach, a task generator learns to create tasks from a parameterized task space via a black-box procedural generation module. To enable curriculum learning in the absence of a direct indicator of learning progress, we propose to train the task generator by balancing the agent's performance in the generated tasks and the similarity to the target tasks. Through adversarial training, the task similarity is adaptively estimated by a task discriminator defined on the agent's experiences, allowing the generated tasks to approximate target tasks of unknown parameterization or outside of the predefined task space. Our experiments on the grid world and robotic manipulation task domains show that APT-Gen achieves substantially better performance than various existing baselines by generating suitable tasks of rich variations.

----

## [626] Multi-timescale Representation Learning in LSTM Language Models

**Authors**: *Shivangi Mahto, Vy Ai Vo, Javier S. Turek, Alexander Huth*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9ITXiTrAoT](https://openreview.net/forum?id=9ITXiTrAoT)

**Abstract**:

Language models must capture statistical dependencies between words at timescales ranging from very short to very long. Earlier work has demonstrated that dependencies in natural language tend to decay with distance between words according to a power law. However, it is unclear how this knowledge can be used for analyzing or designing neural network language models. In this work, we derived a theory for how the memory gating mechanism in long short-term memory (LSTM) language models can capture power law decay. We found that unit timescales within an LSTM, which are determined by the forget gate bias, should follow an Inverse Gamma distribution. Experiments then showed that LSTM language models trained on natural English text learn to approximate this theoretical distribution. Further, we found that explicitly imposing the theoretical distribution upon the model during training yielded better language model perplexity overall, with particular improvements for predicting low-frequency (rare) words. Moreover, the explicit multi-timescale model selectively routes information about different types of words through units with different timescales, potentially improving model interpretability. These results demonstrate the importance of careful, theoretically-motivated analysis of memory and timescale in language models.

----

## [627] Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation

**Authors**: *Jungo Kasai, Nikolaos Pappas, Hao Peng, James Cross, Noah A. Smith*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KpfasTaLUpq](https://openreview.net/forum?id=KpfasTaLUpq)

**Abstract**:

Much recent effort has been invested in non-autoregressive neural machine translation, which appears to be an efficient alternative to state-of-the-art autoregressive machine translation on modern GPUs.  In contrast to the latter, where generation is sequential, the former allows generation to be parallelized across target token positions. Some of the latest non-autoregressive models have achieved impressive translation quality-speed tradeoffs compared to autoregressive baselines. In this work, we reexamine this tradeoff and argue that autoregressive baselines can be substantially sped up without loss in accuracy. Specifically, we study autoregressive models with encoders and decoders of varied depths. Our extensive experiments show that given a sufficiently deep encoder, a single-layer autoregressive decoder can substantially outperform strong non-autoregressive models with comparable inference speed. We show that the speed disadvantage for autoregressive baselines compared to non-autoregressive methods has been overestimated in three aspects: suboptimal layer allocation, insufficient speed measurement, and lack of knowledge distillation. Our results establish a new protocol for future research toward fast, accurate machine translation. Our code is available at https://github.com/jungokasai/deep-shallow.

----

## [628] Learning Better Structured Representations Using Low-rank Adaptive Label Smoothing

**Authors**: *Asish Ghoshal, Xilun Chen, Sonal Gupta, Luke Zettlemoyer, Yashar Mehdad*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5NsEIflpbSv](https://openreview.net/forum?id=5NsEIflpbSv)

**Abstract**:

Training with soft targets instead of hard targets has been shown to improve performance and calibration of deep neural networks. Label smoothing is a popular way of computing soft targets, where one-hot encoding of a class is smoothed with a uniform distribution. Owing to its simplicity, label smoothing has found wide-spread use for training deep neural networks on a wide variety of tasks, ranging from image and text classification to machine translation and semantic parsing. Complementing recent empirical justification for label smoothing, we obtain PAC-Bayesian generalization bounds for label smoothing and show that the generalization error depends on the choice of the noise (smoothing) distribution. Then we propose low-rank adaptive label smoothing (LORAS): a simple yet novel method for training with learned soft targets that generalizes label smoothing and adapts to the latent structure of the label space in structured prediction tasks. Specifically, we evaluate our method on semantic parsing tasks and show that training with appropriately smoothed soft targets can significantly improve accuracy and model calibration, especially in low-resource settings. Used in conjunction with pre-trained sequence-to-sequence models, our method achieves state of the art performance on four semantic parsing data sets. LORAS can be used with any model, improves performance and implicit model calibration  without increasing the number of model parameters, and can be scaled to problems with large label spaces containing tens of thousands of labels.

----

## [629] Predicting Inductive Biases of Pre-Trained Models

**Authors**: *Charles Lovering, Rohan Jha, Tal Linzen, Ellie Pavlick*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mNtmhaDkAr](https://openreview.net/forum?id=mNtmhaDkAr)

**Abstract**:

Most current NLP systems are based on a pre-train-then-fine-tune paradigm, in which a large neural network is first trained in a self-supervised way designed to encourage the network to extract broadly-useful linguistic features, and then fine-tuned for a specific task of interest. Recent work attempts to understand why this recipe works and explain when it fails. Currently, such analyses have produced two sets of apparently-contradictory results. Work that analyzes the representations that result from pre-training (via "probing classifiers") finds evidence that rich features of linguistic structure can be decoded with high accuracy, but work that analyzes model behavior after fine-tuning (via "challenge sets") indicates that decisions are often not based on such structure but rather on spurious heuristics specific to the training set. In this work, we test the hypothesis that the extent to which a feature influences a model's decisions can be predicted using a combination of two factors: The feature's "extractability" after pre-training (measured using information-theoretic probing techniques), and the "evidence" available during fine-tuning (defined as the feature's co-occurrence rate with the label). In experiments with both synthetic and natural language data, we find strong evidence (statistically significant correlations) supporting this hypothesis.

----

## [630] Molecule Optimization by Explainable Evolution

**Authors**: *Binghong Chen, Tianzhe Wang, Chengtao Li, Hanjun Dai, Le Song*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jHefDGsorp5](https://openreview.net/forum?id=jHefDGsorp5)

**Abstract**:

Optimizing molecules for desired properties is a fundamental yet challenging task in chemistry, material science, and drug discovery. This paper develops a novel algorithm for optimizing molecular properties via an Expectation-Maximization (EM) like explainable evolutionary process. The algorithm is designed to mimic human experts in the process of searching for desirable molecules and alternate between two stages: the first stage on explainable local search which identifies rationales, i.e., critical subgraph patterns accounting for desired molecular properties, and the second stage on molecule completion which explores the larger space of molecules containing good rationales. We test our approach against various baselines on a real-world multi-property optimization task where each method is given the same number of queries to the property oracle. We show that our evolution-by-explanation algorithm is 79% better than the best baseline in terms of a generic metric combining aspects such as success rate, novelty, and diversity. Human expert evaluation on optimized molecules shows that 60% of top molecules obtained from our methods are deemed successful.

----

## [631] Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies

**Authors**: *Paul Pu Liang, Manzil Zaheer, Yuan Wang, Amr Ahmed*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Vd7lCMvtLqg](https://openreview.net/forum?id=Vd7lCMvtLqg)

**Abstract**:

Learning continuous representations of discrete objects such as text, users, movies, and URLs lies at the heart of many applications including language and user modeling. When using discrete objects as input to neural networks, we often ignore the underlying structures (e.g., natural groupings and similarities) and embed the objects independently into individual vectors. As a result, existing methods do not scale to large vocabulary sizes. In this paper, we design a simple and efficient embedding algorithm that learns a small set of anchor embeddings and a sparse transformation matrix. We call our method Anchor & Transform (ANT) as the embeddings of discrete objects are a sparse linear combination of the anchors, weighted according to the transformation matrix. ANT is scalable, flexible, and end-to-end trainable. We further provide a statistical interpretation of our algorithm as a Bayesian nonparametric prior for embeddings that encourages sparsity and leverages natural groupings among objects. By deriving an approximate inference algorithm based on Small Variance Asymptotics, we obtain a natural extension that automatically learns the optimal number of anchors instead of having to tune it as a hyperparameter. On text classification, language modeling, and movie recommendation benchmarks, we show that ANT is particularly suitable for large vocabulary sizes and demonstrates stronger performance with fewer parameters (up to 40x compression) as compared to existing compression baselines.

----

## [632] PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences

**Authors**: *Hehe Fan, Xin Yu, Yuhang Ding, Yi Yang, Mohan S. Kankanhalli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O3bqkf_Puys](https://openreview.net/forum?id=O3bqkf_Puys)

**Abstract**:

Point cloud sequences are irregular and unordered in the spatial dimension while exhibiting regularities and order in the temporal dimension. Therefore, existing grid based convolutions for conventional video processing cannot be directly applied to spatio-temporal modeling of raw point cloud sequences. In this paper, we propose a point spatio-temporal (PST) convolution to achieve informative representations of point cloud sequences. The proposed PST convolution first disentangles space and time in point cloud sequences.  Then, a spatial convolution is employed to capture the local structure of points in the 3D space, and a temporal convolution is used to model the dynamics of the spatial regions along the time dimension.  Furthermore, we incorporate the proposed PST convolution into a deep network, namely PSTNet, to extract features of point cloud sequences in a hierarchical manner.  Extensive experiments on widely-used 3D action recognition and 4D semantic segmentation datasets demonstrate the effectiveness of PSTNet to model point cloud sequences.

----

## [633] Group Equivariant Conditional Neural Processes

**Authors**: *Makoto Kawano, Wataru Kumagai, Akiyoshi Sannai, Yusuke Iwasawa, Yutaka Matsuo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=e8W-hsu_q5](https://openreview.net/forum?id=e8W-hsu_q5)

**Abstract**:

We present the group equivariant conditional neural process (EquivCNP), a meta-learning method with permutation invariance in a data set as in conventional conditional neural processes (CNPs), and it also has transformation equivariance in data space. Incorporating group equivariance, such as rotation and scaling equivariance, provides a way to consider the symmetry of real-world data. We give a decomposition theorem for permutation-invariant and group-equivariant maps, which leads us to construct EquivCNPs with an infinite-dimensional latent space to handle group symmetries. In this paper, we build architecture using Lie group convolutional layers for practical implementation. We show that EquivCNP with translation equivariance achieves comparable performance to conventional CNPs in a 1D regression task. Moreover, we demonstrate that incorporating an appropriate Lie group equivariance, EquivCNP is capable of zero-shot generalization for an image-completion task by selecting an appropriate Lie group equivariance.

----

## [634] When does preconditioning help or hurt generalization?

**Authors**: *Shun-ichi Amari, Jimmy Ba, Roger Baker Grosse, Xuechen Li, Atsushi Nitanda, Taiji Suzuki, Denny Wu, Ji Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=S724o4_WB3](https://openreview.net/forum?id=S724o4_WB3)

**Abstract**:

While second order optimizers such as natural gradient descent (NGD) often speed up optimization, their effect on generalization has been called into question. This work presents a more nuanced view on how the \textit{implicit bias} of optimizers affects the comparison of generalization properties. 
We provide an exact asymptotic bias-variance decomposition of the generalization error of preconditioned ridgeless regression in the overparameterized regime, and consider the inverse population Fisher information matrix (used in NGD) as a particular example. We determine the optimal preconditioner $\boldsymbol{P}$ for both the bias and variance, and find that the relative generalization performance of different optimizers depends on label noise and ``shape'' of the signal (true parameters): when the labels are noisy, the model is misspecified, or the signal is misaligned with the features, NGD can achieve lower risk; conversely, GD generalizes better under clean labels, a well-specified model, or aligned signal. 
Based on this analysis, we discuss several approaches to manage the bias-variance tradeoff, and the potential benefit of interpolating between first- and second-order updates. We then extend our analysis to regression in the reproducing kernel Hilbert space and demonstrate that preconditioning can lead to more efficient decrease in the population risk. Lastly, we empirically compare the generalization error of first- and second-order optimizers in neural network experiments, and observe robust trends matching our theoretical analysis.

----

## [635] Learning Reasoning Paths over Semantic Graphs for Video-grounded Dialogues

**Authors**: *Hung Le, Nancy F. Chen, Steven C. H. Hoi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hPWj1qduVw8](https://openreview.net/forum?id=hPWj1qduVw8)

**Abstract**:

Compared to traditional visual question answering, video-grounded dialogues require additional reasoning over dialogue context to answer questions in a multi-turn setting. Previous approaches to video-grounded dialogues mostly use dialogue context as a simple text input without modelling the inherent information flows at the turn level. In this paper, we propose a novel framework of Reasoning Paths in Dialogue Context (PDC). PDC model discovers information flows among dialogue turns through a semantic graph constructed based on lexical components in each question and answer. PDC model then learns to predict reasoning paths over this semantic graph. Our path prediction model predicts a path from the current turn through past dialogue turns that contain additional visual cues to answer the current question. Our reasoning model sequentially processes both visual and textual information through this reasoning path and the propagated features are used to generate the answer. Our experimental results demonstrate the effectiveness of our method and provide additional insights on how models use semantic dependencies in a dialogue context to retrieve visual cues.

----

## [636] Prototypical Representation Learning for Relation Extraction

**Authors**: *Ning Ding, Xiaobin Wang, Yao Fu, Guangwei Xu, Rui Wang, Pengjun Xie, Ying Shen, Fei Huang, Haitao Zheng, Rui Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aCgLmfhIy_f](https://openreview.net/forum?id=aCgLmfhIy_f)

**Abstract**:

Recognizing relations between entities is a pivotal task of relational learning.  
Learning relation representations from distantly-labeled datasets is difficult because of the abundant label noise and complicated expressions in human language.  
This paper aims to learn predictive, interpretable, and robust relation representations from distantly-labeled data that are effective in different settings, including supervised, distantly supervised, and few-shot learning. 
Instead of solely relying on the supervision from noisy labels, we propose to learn prototypes for each relation from contextual information to best explore the intrinsic semantics of relations. 
Prototypes are representations in the feature space abstracting the essential semantics of relations between entities in sentences.
We learn prototypes based on objectives with clear geometric interpretation, where the prototypes are unit vectors uniformly dispersed in a unit ball, and statement embeddings are centered at the end of their corresponding prototype vectors on the surface of the ball. 
This approach allows us to learn meaningful, interpretable prototypes for the final classification.
Results on several relation learning tasks show that our model significantly outperforms the previous state-of-the-art models.
We further demonstrate the robustness of the encoder and the interpretability of prototypes with extensive experiments.

----

## [637] Layer-adaptive Sparsity for the Magnitude-based Pruning

**Authors**: *Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, Jinwoo Shin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=H6ATjJ0TKdf](https://openreview.net/forum?id=H6ATjJ0TKdf)

**Abstract**:

Recent discoveries on neural network pruning reveal that, with a carefully chosen layerwise sparsity, a simple magnitude-based pruning achieves state-of-the-art tradeoff between sparsity and performance. However, without a clear consensus on ``how to choose,'' the layerwise sparsities are mostly selected algorithm-by-algorithm, often resorting to handcrafted heuristics or an extensive hyperparameter search. To fill this gap, we propose a novel importance score for global pruning, coined layer-adaptive magnitude-based pruning (LAMP) score; the score is a rescaled version of weight magnitude that incorporates the model-level $\ell_2$ distortion incurred by pruning, and does not require any hyperparameter tuning or heavy computation.
Under various image classification setups, LAMP consistently outperforms popular existing schemes for layerwise sparsity selection.
Furthermore, we observe that LAMP continues to outperform baselines even in weight-rewinding setups, while the connectivity-oriented layerwise sparsity (the strongest baseline overall) performs worse than a simple global magnitude-based pruning in this case. Code: https://github.com/jaeho-lee/layer-adaptive-sparsity

----

## [638] Refining Deep Generative Models via Discriminator Gradient Flow

**Authors**: *Abdul Fatir Ansari, Ming Liang Ang, Harold Soh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Zbc-ue9p_rE](https://openreview.net/forum?id=Zbc-ue9p_rE)

**Abstract**:

Deep generative modeling has seen impressive advances in recent years, to the point where it is now commonplace to see simulated samples (e.g., images) that closely resemble real-world data. However, generation quality is generally inconsistent for any given model and can vary dramatically between samples. We introduce Discriminator Gradient $f$low (DG$f$low), a new technique that improves generated samples via the gradient flow of entropy-regularized $f$-divergences between the real and the generated data distributions. The gradient flow takes the form of a non-linear Fokker-Plank equation, which can be easily simulated by sampling from the equivalent McKean-Vlasov process. By refining inferior samples, our technique avoids wasteful sample rejection used by previous methods (DRS & MH-GAN). Compared to existing works that focus on specific GAN variants, we show our refinement approach can be applied to GANs with vector-valued critics and even other deep generative models such as VAEs and Normalizing Flows. Empirical results on multiple synthetic, image, and text datasets demonstrate that DG$f$low leads to significant improvement in the quality of generated samples for a variety of generative models, outperforming the state-of-the-art Discriminator Optimal Transport (DOT) and Discriminator Driven Latent Sampling (DDLS) methods.

----

## [639] Explaining the Efficacy of Counterfactually Augmented Data

**Authors**: *Divyansh Kaushik, Amrith Setlur, Eduard H. Hovy, Zachary Chase Lipton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HHiiQKWsOcV](https://openreview.net/forum?id=HHiiQKWsOcV)

**Abstract**:

In attempts to produce machine learning models less reliant on spurious patterns in NLP datasets, researchers have recently proposed curating counterfactually augmented data (CAD) via a human-in-the-loop process in which given some documents and their (initial) labels, humans must revise the text to make a counterfactual label applicable. Importantly, edits that are not necessary to flip the applicable label are prohibited. Models trained on the augmented (original and revised) data appear, empirically, to rely less on semantically irrelevant words and to generalize better out of domain. While this work draws loosely on causal thinking, the underlying causal model (even at an abstract level) and the principles underlying the observed out-of-domain improvements remain unclear. In this paper, we introduce a toy analog based on linear Gaussian models, observing interesting relationships between causal models, measurement noise, out-of-domain generalization, and reliance on spurious signals. Our analysis provides some insights that help to explain the efficacy of CAD. Moreover, we develop the hypothesis that while adding noise to causal features should degrade both in-domain and out-of-domain performance, adding noise to non-causal features should lead to relative improvements in out-of-domain performance. This idea inspires a speculative test for determining whether a feature attribution technique has identified the causal spans. If adding noise (e.g., by random word flips) to the highlighted spans degrades both in-domain and out-of-domain performance on a battery of challenge datasets, but adding noise to the complement gives improvements out-of-domain, this suggests we have identified causal spans. Thus, we present a large scale empirical study comparing spans edited to create CAD to those selected by attention and saliency maps. Across numerous challenge domains and models, we find that the hypothesized phenomenon is pronounced for CAD.

----

## [640] Lipschitz Recurrent Neural Networks

**Authors**: *N. Benjamin Erichson, Omri Azencot, Alejandro F. Queiruga, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-N7PBXqOUJZ](https://openreview.net/forum?id=-N7PBXqOUJZ)

**Abstract**:

Viewing recurrent neural networks (RNNs) as continuous-time dynamical systems, we propose a recurrent unit that describes the hidden state's evolution with two parts: a well-understood linear component plus a Lipschitz nonlinearity. This particular functional form facilitates stability analysis of the long-term behavior of the recurrent unit using tools from nonlinear systems theory. In turn, this enables architectural design decisions before experimentation. Sufficient conditions for global stability of the recurrent unit are obtained, motivating a novel scheme for constructing hidden-to-hidden matrices. Our experiments demonstrate that the Lipschitz RNN can outperform existing recurrent units on a range of benchmark tasks, including computer vision, language modeling and speech prediction tasks. Finally, through Hessian-based analysis we demonstrate that our Lipschitz recurrent unit is more robust with respect to input and parameter perturbations as compared to other continuous-time RNNs.

----

## [641] Learning Hyperbolic Representations of Topological Features

**Authors**: *Panagiotis Kyriakis, Iordanis Fostiropoulos, Paul Bogdan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yqPnIRhHtZv](https://openreview.net/forum?id=yqPnIRhHtZv)

**Abstract**:

Learning task-specific representations of persistence diagrams is an important problem in topological data analysis and machine learning. However, current state of the art methods are restricted in terms of their expressivity as they are focused on Euclidean representations. Persistence diagrams often contain features of infinite persistence (i.e., essential features) and Euclidean spaces shrink their importance relative to non-essential features because they cannot assign infinite distance to finite points. To deal with this issue, we propose a method to learn representations of persistence diagrams on hyperbolic spaces, more specifically on the Poincare ball. By representing features of infinite persistence infinitesimally close to the boundary of the ball, their distance to non-essential features approaches infinity, thereby their relative importance is preserved. This is achieved without utilizing extremely high values for the learnable parameters, thus the representation can be fed into downstream optimization methods and trained efficiently in an end-to-end fashion. We present experimental results on graph and image classification tasks and show that the performance of our method is on par with or exceeds the performance of other state of the art methods.

----

## [642] Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech

**Authors**: *Yoonhyung Lee, Joongbo Shin, Kyomin Jung*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=o3iritJHLfO](https://openreview.net/forum?id=o3iritJHLfO)

**Abstract**:

Although early text-to-speech (TTS) models such as Tacotron 2 have succeeded in generating human-like speech, their autoregressive architectures have several limitations: (1) They require a lot of time to generate a mel-spectrogram consisting of hundreds of steps. (2) The autoregressive speech generation shows a lack of robustness due to its error propagation property. In this paper, we propose a novel non-autoregressive TTS model called BVAE-TTS, which eliminates the architectural limitations and generates a mel-spectrogram in parallel. BVAE-TTS adopts a bidirectional-inference variational autoencoder (BVAE) that learns hierarchical latent representations using both bottom-up and top-down paths to increase its expressiveness. To apply BVAE to TTS, we design our model to utilize text information via an attention mechanism. By using attention maps that BVAE-TTS generates, we train a duration predictor so that the model uses the predicted duration of each phoneme at inference. In experiments conducted on LJSpeech dataset, we show that our model generates a mel-spectrogram 27 times faster than Tacotron 2 with similar speech quality. Furthermore, our BVAE-TTS outperforms Glow-TTS, which is one of the state-of-the-art non-autoregressive TTS models, in terms of both speech quality and inference speed while having 58% fewer parameters.

----

## [643] Risk-Averse Offline Reinforcement Learning

**Authors**: *Núria Armengol Urpí, Sebastian Curi, Andreas Krause*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TBIzh9b5eaz](https://openreview.net/forum?id=TBIzh9b5eaz)

**Abstract**:

Training Reinforcement Learning (RL) agents in high-stakes applications might be too prohibitive due to the risk associated to exploration. Thus, the agent can only use data previously collected by safe policies. While previous work considers optimizing the average performance using offline data, we focus on optimizing a risk-averse criteria, namely the CVaR. In particular, we present the Offline Risk-Averse Actor-Critic (O-RAAC), a model-free RL algorithm that is able to learn risk-averse policies in a fully offline setting. We show that O-RAAC learns policies with higher CVaR than risk-neutral approaches in different robot control tasks. Furthermore, considering risk-averse criteria guarantees distributional robustness of the average performance with respect to particular distribution shifts. We demonstrate empirically that in the presence of natural distribution-shifts, O-RAAC learns policies with good average performance.

----

## [644] Group Equivariant Stand-Alone Self-Attention For Vision

**Authors**: *David W. Romero, Jean-Baptiste Cordonnier*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JkfYjnOEo6M](https://openreview.net/forum?id=JkfYjnOEo6M)

**Abstract**:

We provide a general self-attention formulation to impose group equivariance to arbitrary symmetry groups. This is achieved by defining positional encodings that are invariant to the action of the group considered. Since the group acts on the positional encoding directly, group equivariant self-attention networks (GSA-Nets) are steerable by nature. Our experiments on vision benchmarks demonstrate consistent improvements of GSA-Nets over non-equivariant self-attention networks.

----

## [645] A Better Alternative to Error Feedback for Communication-Efficient Distributed Learning

**Authors**: *Samuel Horváth, Peter Richtárik*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vYVI1CHPaQg](https://openreview.net/forum?id=vYVI1CHPaQg)

**Abstract**:

Modern large-scale machine learning applications require stochastic optimization algorithms to be implemented on distributed computing systems. A key bottleneck of such systems is the communication overhead for exchanging information across the workers, such as stochastic gradients. Among the many techniques proposed to remedy this issue, one of the most successful is the framework of compressed communication with error feedback (EF). EF remains the only known technique that can deal with the error induced by contractive compressors which are not unbiased, such as Top-$K$ or PowerSGD.  In this paper, we propose a new and theoretically and practically better alternative to EF for dealing with contractive compressors. In particular, we propose a construction which can transform any contractive compressor into an induced unbiased compressor. Following this transformation, existing methods able to work with unbiased compressors can be applied. We show that our approach leads to vast improvements over EF, including reduced memory requirements, better communication complexity guarantees and fewer assumptions. We further extend our results to federated learning with partial participation following an arbitrary distribution over the nodes and demonstrate the benefits thereof. We perform several numerical experiments which validate our theoretical findings.

----

## [646] Neural Delay Differential Equations

**Authors**: *Qunxi Zhu, Yao Guo, Wei Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Q1jmmQz72M2](https://openreview.net/forum?id=Q1jmmQz72M2)

**Abstract**:

Neural Ordinary Differential Equations (NODEs), a framework of continuous-depth neural networks, have been widely applied, showing exceptional efficacy in coping with some representative datasets.  Recently, an augmented framework has been successfully developed for conquering some limitations emergent in application of the original framework.  Here we propose a new class of continuous-depth neural networks with delay, named as Neural Delay Differential Equations (NDDEs),  and, for computing the corresponding gradients, we use the adjoint sensitivity method to obtain the delayed dynamics of the adjoint. Since the differential equations with delays are usually seen as dynamical systems of infinite dimension possessing more fruitful dynamics, the NDDEs, compared to the NODEs, own a stronger capacity of nonlinear representations.  Indeed, we analytically validate that the NDDEs are of universal approximators, and further articulate an extension of the NDDEs, where the initial function of the NDDEs is supposed to satisfy ODEs.   More importantly, we use several illustrative examples to demonstrate the outstanding capacities of the NDDEs and the NDDEs with ODEs' initial value.  More precisely, (1) we successfully model the delayed dynamics where the trajectories in the lower-dimensional phase space could be mutually intersected, while the traditional NODEs without any argumentation are not directly applicable for such modeling, and (2) we achieve lower loss and higher accuracy not only for the  data produced synthetically by complex models but also for the real-world image datasets, i.e., CIFAR10, MNIST and SVHN.   Our results on the NDDEs reveal that appropriately articulating the elements of dynamical systems into the network design is truly beneficial to promoting the network performance.

----

## [647] Capturing Label Characteristics in VAEs

**Authors**: *Tom Joy, Sebastian M. Schmon, Philip H. S. Torr, Siddharth Narayanaswamy, Tom Rainforth*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wQRlSUZ5V7B](https://openreview.net/forum?id=wQRlSUZ5V7B)

**Abstract**:

We present a principled approach to incorporating labels in variational autoencoders (VAEs) that captures the rich characteristic information associated with those labels. While prior work has typically conflated these by learning latent variables that directly correspond to label values, we argue this is contrary to the intended effect of supervision in VAEs—capturing rich label characteristics with the latents. For example, we may want to capture the characteristics of a face that make it look young, rather than just the age of the person. To this end, we develop a novel VAE model, the characteristic capturing VAE (CCVAE), which “reparameterizes” supervision through auxiliary variables and a concomitant variational objective. Through judicious structuring of mappings between latent and auxiliary variables, we show that the CCVAE can effectively learn meaningful representations of the characteristics of interest across a variety of supervision schemes. In particular, we show that the CCVAE allows for more effective and more general interventions to be performed, such as smooth traversals within the characteristics for a given label, diverse conditional generation, and transferring characteristics across datapoints.

----

## [648] Graph Edit Networks

**Authors**: *Benjamin Paassen, Daniele Grattarola, Daniele Zambon, Cesare Alippi, Barbara Hammer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dlEJsyHGeaL](https://openreview.net/forum?id=dlEJsyHGeaL)

**Abstract**:

While graph neural networks have made impressive progress in classification and regression, few approaches to date perform time series prediction on graphs, and those that do are mostly limited to edge changes. We suggest that graph edits are a more natural interface for graph-to-graph learning. In particular,  graph edits are general enough to describe any graph-to-graph change, not only edge changes; they are sparse, making them easier to understand for humans and more efficient computationally; and they are local, avoiding the need for pooling layers in graph neural networks. In this paper, we propose a novel output layer - the graph edit network - which takes node embeddings as input and generates a sequence of graph edits that transform the input graph to the output graph. We prove that a mapping between the node sets of two graphs is sufficient to construct training data for a graph edit network and that an optimal mapping yields edit scripts that are almost as short as the graph edit distance between the graphs. We further provide a proof-of-concept empirical evaluation on several graph dynamical systems, which are difficult to learn for baselines from the literature.

----

## [649] InfoBERT: Improving Robustness of Language Models from An Information Theoretic Perspective

**Authors**: *Boxin Wang, Shuohang Wang, Yu Cheng, Zhe Gan, Ruoxi Jia, Bo Li, Jingjing Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hpH98mK5Puk](https://openreview.net/forum?id=hpH98mK5Puk)

**Abstract**:

Large-scale language models such as BERT have achieved state-of-the-art performance across a wide range of NLP tasks. Recent studies, however, show that such BERT-based models are vulnerable facing the threats of textual adversarial attacks. We aim to address this problem from an information-theoretic perspective, and propose InfoBERT, a novel learning framework for robust ﬁne-tuning of pre-trained language models. InfoBERT contains two mutual-information-based regularizers for model training: (i) an Information Bottleneck regularizer, which suppresses noisy mutual information between the input and the feature representation; and (ii) a Robust Feature regularizer, which increases the mutual information between local robust features and global features. We provide a principled way to theoretically analyze and improve the robustness of representation learning for language models in both standard and adversarial training. Extensive experiments demonstrate that InfoBERT achieves state-of-the-art robust accuracy over several adversarial datasets on Natural Language Inference (NLI) and Question Answering (QA) tasks.
Our code is available at https://github.com/AI-secure/InfoBERT.

----

## [650] DrNAS: Dirichlet Neural Architecture Search

**Authors**: *Xiangning Chen, Ruochen Wang, Minhao Cheng, Xiaocheng Tang, Cho-Jui Hsieh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9FWas6YbmB3](https://openreview.net/forum?id=9FWas6YbmB3)

**Abstract**:

This paper proposes a novel differentiable architecture search method by formulating it into a distribution learning problem. We treat the continuously relaxed architecture mixing weight as random variables, modeled by Dirichlet distribution. With recently developed pathwise derivatives, the Dirichlet parameters can be easily optimized with gradient-based optimizer in an end-to-end manner. This formulation improves the generalization ability and induces stochasticity that naturally encourages exploration in the search space. Furthermore, to alleviate the large memory consumption of differentiable NAS, we propose a simple yet effective progressive learning scheme that enables searching directly on large-scale tasks, eliminating the gap between search and evaluation phases. Extensive experiments demonstrate the effectiveness of our method. Specifically, we obtain a test error of 2.46\% for CIFAR-10, 23.7\% for ImageNet under the mobile setting. On NAS-Bench-201, we also achieve state-of-the-art results on all three datasets and provide insights for the effective design of neural architecture search algorithms.

----

## [651] Drop-Bottleneck: Learning Discrete Compressed Representation for Noise-Robust Exploration

**Authors**: *Jaekyeom Kim, Minjung Kim, Dongyeon Woo, Gunhee Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1rxHOBjeDUW](https://openreview.net/forum?id=1rxHOBjeDUW)

**Abstract**:

We propose a novel information bottleneck (IB) method named Drop-Bottleneck, which discretely drops features that are irrelevant to the target variable. Drop-Bottleneck not only enjoys a simple and tractable compression objective but also additionally provides a deterministic compressed representation of the input variable, which is useful for inference tasks that require consistent representation. Moreover, it can jointly learn a feature extractor and select features considering each feature dimension's relevance to the target task, which is unattainable by most neural network-based IB methods. We propose an exploration method based on Drop-Bottleneck for reinforcement learning tasks. In a multitude of noisy and reward sparse maze navigation tasks in VizDoom (Kempka et al., 2016) and DMLab (Beattie et al., 2016), our exploration method achieves state-of-the-art performance. As a new IB framework, we demonstrate that Drop-Bottleneck outperforms Variational Information Bottleneck (VIB) (Alemi et al., 2017) in multiple aspects including adversarial robustness and dimensionality reduction.

----

## [652] Monte-Carlo Planning and Learning with Language Action Value Estimates

**Authors**: *Youngsoo Jang, Seokin Seo, Jongmin Lee, Kee-Eung Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7_G8JySGecm](https://openreview.net/forum?id=7_G8JySGecm)

**Abstract**:

Interactive Fiction (IF) games provide a useful testbed for language-based reinforcement learning agents, posing significant challenges of natural language understanding, commonsense reasoning, and non-myopic planning in the combinatorial search space. Agents based on standard planning algorithms struggle to play IF games due to the massive search space of language actions. Thus, language-grounded planning is a key ability of such agents, since inferring the consequence of language action based on semantic understanding can drastically improve search. In this paper, we introduce Monte-Carlo planning with Language Action Value Estimates (MC-LAVE) that combines a Monte-Carlo tree search with language-driven exploration. MC-LAVE invests more search effort into semantically promising language actions using locally optimistic language value estimates, yielding a significant reduction in the effective search space of language actions. We then present a reinforcement learning approach via MC-LAVE, which alternates between MC-LAVE planning and supervised learning of the self-generated language actions. In the experiments, we demonstrate that our method achieves new high scores in various IF games.

----

## [653] Robust early-learning: Hindering the memorization of noisy labels

**Authors**: *Xiaobo Xia, Tongliang Liu, Bo Han, Chen Gong, Nannan Wang, Zongyuan Ge, Yi Chang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Eql5b1_hTE4](https://openreview.net/forum?id=Eql5b1_hTE4)

**Abstract**:

The \textit{memorization effects} of deep networks show that they will first memorize training data with clean labels and then those with noisy labels. The \textit{early stopping} method therefore can be exploited for learning with noisy labels. However, the side effect brought by noisy labels will influence the memorization of clean labels before early stopping. In this paper, motivated by the \textit{lottery ticket hypothesis} which shows that only partial parameters are important for generalization, we find that only partial parameters are important for fitting clean labels and generalize well, which we term as \textit{critical parameters}; while the other parameters tend to fit noisy labels and cannot generalize well, which we term as \textit{non-critical parameters}. Based on this, we propose \textit{robust early-learning} to reduce the side effect of noisy labels before early stopping and thus enhance the memorization of clean labels. Specifically, in each iteration, we divide all parameters into the critical and non-critical ones, and then perform different update rules for different types of parameters. Extensive experiments on benchmark-simulated and real-world label-noise datasets demonstrate the superiority of the proposed method over the state-of-the-art label-noise learning methods.

----

## [654] Identifying Physical Law of Hamiltonian Systems via Meta-Learning

**Authors**: *Seungjun Lee, Haesang Yang, Woojae Seong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=45NZvF1UHam](https://openreview.net/forum?id=45NZvF1UHam)

**Abstract**:

Hamiltonian mechanics is an effective tool to represent many physical processes with concise yet well-generalized mathematical expressions. A well-modeled Hamiltonian makes it easy for researchers to analyze and forecast many related phenomena that are governed by the same physical law. However, in general, identifying a functional or shared expression of the Hamiltonian is very difficult. It requires carefully designed experiments and the researcher's insight that comes from years of experience. We propose that meta-learning algorithms can be potentially powerful data-driven tools for identifying the physical law governing Hamiltonian systems without any mathematical assumptions on the representation, but with observations from a set of systems governed by the same physical law. We show that a well meta-trained learner can identify the shared representation of the Hamiltonian by evaluating our method on several types of physical systems with various experimental settings.

----

## [655] Reweighting Augmented Samples by Minimizing the Maximal Expected Loss

**Authors**: *Mingyang Yi, Lu Hou, Lifeng Shang, Xin Jiang, Qun Liu, Zhi-Ming Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9G5MIc-goqB](https://openreview.net/forum?id=9G5MIc-goqB)

**Abstract**:

Data augmentation is an effective technique to improve the generalization of deep neural networks. However, previous data augmentation methods usually treat the augmented samples equally without considering their individual impacts on the model. To address this, for the augmented samples from the same training example, we propose to assign different weights to them. We construct the maximal expected loss which is the supremum over any reweighted loss on augmented samples. Inspired by adversarial training, we minimize this maximal expected loss (MMEL) and obtain a simple and interpretable closed-form solution: more attention should be paid to augmented samples with large loss values (i.e., harder examples). Minimizing this maximal expected loss enables the model to perform well under any reweighting strategy. The proposed method can generally be applied on top of any data augmentation methods. Experiments are conducted on both natural language understanding tasks with token-level data augmentation, and image classification tasks with commonly-used image augmentation techniques like random crop and horizontal flip. Empirical results show that the proposed method improves the generalization performance of the model.

----

## [656] Understanding and Improving Encoder Layer Fusion in Sequence-to-Sequence Learning

**Authors**: *Xuebo Liu, Longyue Wang, Derek F. Wong, Liang Ding, Lidia S. Chao, Zhaopeng Tu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=n1HD8M6WGn](https://openreview.net/forum?id=n1HD8M6WGn)

**Abstract**:

Encoder layer fusion (EncoderFusion) is a technique to fuse all the encoder layers (instead of the uppermost layer) for sequence-to-sequence (Seq2Seq) models, which has proven effective on various NLP tasks. However, it is still not entirely clear why and when EncoderFusion should work. In this paper, our main contribution is to take a step further in understanding EncoderFusion. Many of previous studies believe that the success of EncoderFusion comes from exploiting surface and syntactic information embedded in lower encoder layers. Unlike them, we find that the encoder embedding layer is more important than other intermediate encoder layers. In addition, the uppermost decoder layer consistently pays more attention to the encoder embedding layer across NLP tasks. Based on this observation, we propose a simple fusion method, SurfaceFusion, by fusing only the encoder embedding layer for the softmax layer. Experimental results show that SurfaceFusion outperforms EncoderFusion on several NLP benchmarks, including machine translation, text summarization, and grammatical error correction. It obtains the state-of-the-art performance on WMT16 Romanian-English and WMT14 English-French translation tasks. Extensive analyses reveal that SurfaceFusion learns more expressive bilingual word embeddings by building a closer relationship between relevant source and target embeddings. Source code is freely available at https://github.com/SunbowLiu/SurfaceFusion

----

## [657] Spatial Dependency Networks: Neural Layers for Improved Generative Image Modeling

**Authors**: *Ðorðe Miladinovic, Aleksandar Stanic, Stefan Bauer, Jürgen Schmidhuber, Joachim M. Buhmann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=I4c4K9vBNny](https://openreview.net/forum?id=I4c4K9vBNny)

**Abstract**:

How to improve generative modeling by better exploiting spatial regularities and coherence in images? We introduce a novel neural network for building image generators (decoders) and apply it to variational autoencoders (VAEs). In our spatial dependency networks (SDNs), feature maps at each level of a deep neural net are computed in a spatially coherent way, using a sequential gating-based mechanism that distributes contextual information across 2-D space. We show that augmenting the decoder of a hierarchical VAE by spatial dependency layers considerably improves density estimation over baseline convolutional architectures and the state-of-the-art among the models within the same class. Furthermore, we demonstrate that SDN can be applied to large images by synthesizing samples of high quality and coherence. In a vanilla VAE setting, we find that a powerful SDN decoder also improves learning disentangled representations, indicating that neural architectures play an important role in this task. Our results suggest favoring spatial dependency over convolutional layers in various VAE settings. The accompanying source code is given at https://github.com/djordjemila/sdn.

----

## [658] Deep Repulsive Clustering of Ordered Data Based on Order-Identity Decomposition

**Authors**: *Seon-Ho Lee, Chang-Su Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Yz-XtK5RBxB](https://openreview.net/forum?id=Yz-XtK5RBxB)

**Abstract**:

We propose the deep repulsive clustering (DRC) algorithm of ordered data for effective order learning. First, we develop the order-identity decomposition (ORID) network to divide the information of an object instance into an order-related feature and an identity feature. Then, we group object instances into clusters according to their identity features using a repulsive term. Moreover, we estimate the rank of a test instance, by comparing it with references within the same cluster. Experimental results on facial age estimation, aesthetic score regression, and historical color image classification show that the proposed algorithm can cluster ordered data effectively and also yield excellent rank estimation performance.

----

## [659] Revisiting Locally Supervised Learning: an Alternative to End-to-end Training

**Authors**: *Yulin Wang, Zanlin Ni, Shiji Song, Le Yang, Gao Huang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fAbkE6ant2](https://openreview.net/forum?id=fAbkE6ant2)

**Abstract**:

Due to the need to store the intermediate activations for back-propagation, end-to-end (E2E) training of deep networks usually suffers from high GPUs memory footprint. This paper aims to address this problem by revisiting the locally supervised learning, where a network is split into gradient-isolated modules and trained with local supervision. We experimentally show that simply training local modules with E2E loss tends to collapse task-relevant information at early layers, and hence hurts the performance of the full model. To avoid this issue, we propose an information propagation (InfoPro) loss, which encourages local modules to preserve as much useful information as possible, while progressively discard task-irrelevant information. As InfoPro loss is difficult to compute in its original form, we derive a feasible upper bound as a surrogate optimization objective, yielding a simple but effective algorithm. In fact, we show that the proposed method boils down to minimizing the combination of a reconstruction loss and a normal cross-entropy/contrastive term. Extensive empirical results on five datasets (i.e., CIFAR, SVHN, STL-10, ImageNet and Cityscapes) validate that InfoPro is capable of achieving competitive performance with less than 40% memory footprint compared to E2E training, while allowing using training data with higher-resolution or larger batch sizes under the same GPU memory constraint. Our method also enables training local modules asynchronously for potential training acceleration.

----

## [660] Nonseparable Symplectic Neural Networks

**Authors**: *Shiying Xiong, Yunjin Tong, Xingzhe He, Shuqi Yang, Cheng Yang, Bo Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=B5VvQrI49Pa](https://openreview.net/forum?id=B5VvQrI49Pa)

**Abstract**:

Predicting the behaviors of Hamiltonian systems has been drawing increasing attention in scientific machine learning. However, the vast majority of the literature was focused on predicting separable Hamiltonian systems with their kinematic and potential energy terms being explicitly decoupled, while building data-driven paradigms to predict nonseparable Hamiltonian systems that are ubiquitous in fluid dynamics and quantum mechanics were rarely explored. The main computational challenge lies in the effective embedding of symplectic priors to describe the inherently coupled evolution of position and momentum, which typically exhibits intricate dynamics. To solve the problem, we propose a novel neural network architecture, Nonseparable Symplectic Neural Networks (NSSNNs), to uncover and embed the symplectic structure of a nonseparable Hamiltonian system from limited observation data. The enabling mechanics of our approach is an augmented symplectic time integrator to decouple the position and momentum energy terms and facilitate their evolution. We demonstrated the efficacy and versatility of our method by predicting a wide range of Hamiltonian systems, both separable and nonseparable, including chaotic vortical flows. We showed the unique computational merits of our approach to yield long-term, accurate, and robust predictions for large-scale Hamiltonian systems by rigorously enforcing symplectomorphism.

----

## [661] Gradient Origin Networks

**Authors**: *Sam Bond-Taylor, Chris G. Willcocks*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0O_cQfw6uEh](https://openreview.net/forum?id=0O_cQfw6uEh)

**Abstract**:

This paper proposes a new type of generative model that is able to quickly learn a latent representation without an encoder. This is achieved using empirical Bayes to calculate the expectation of the posterior, which is implemented by initialising a latent vector with zeros, then using the gradient of the log-likelihood of the data with respect to this zero vector as new latent points. The approach has similar characteristics to autoencoders, but with a simpler architecture, and is demonstrated in a variational autoencoder equivalent that permits sampling. This also allows implicit representation networks to learn a space of implicit functions without requiring a hypernetwork, retaining their representation advantages across datasets. The experiments show that the proposed method converges faster, with significantly lower reconstruction error than autoencoders, while requiring half the parameters.

----

## [662] Learning to Sample with Local and Global Contexts in Experience Replay Buffer

**Authors**: *Youngmin Oh, Kimin Lee, Jinwoo Shin, Eunho Yang, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gJYlaqL8i8](https://openreview.net/forum?id=gJYlaqL8i8)

**Abstract**:

Experience replay, which enables the agents to remember and reuse experience from the past, has played a significant role in the success of off-policy reinforcement learning (RL). To utilize the experience replay efficiently, the existing sampling methods allow selecting out more meaningful experiences by imposing priorities on them based on certain metrics (e.g. TD-error). However, they may result in sampling highly biased, redundant transitions since they compute the sampling rate for each transition independently, without consideration of its importance in relation to other transitions. In this paper, we aim to address the issue by proposing a new learning-based sampling method that can compute the relative importance of transition. To this end, we design a novel permutation-equivariant neural architecture that takes contexts from not only features of each transition (local) but also those of others (global) as inputs. We validate our framework, which we refer to as Neural Experience Replay Sampler (NERS), on multiple benchmark tasks for both continuous and discrete control tasks and show that it can significantly improve the performance of various off-policy RL methods. Further analysis confirms that the improvements of the sample efficiency indeed are due to sampling diverse and meaningful transitions by NERS that considers both local and global contexts.

----

## [663] Provable Rich Observation Reinforcement Learning with Combinatorial Latent States

**Authors**: *Dipendra Misra, Qinghua Liu, Chi Jin, John Langford*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hx1IXFHAw7R](https://openreview.net/forum?id=hx1IXFHAw7R)

**Abstract**:

We propose a novel setting for reinforcement learning that combines two common real-world difficulties: presence of observations (such as camera images) and factored states (such as location of objects). In our setting, the agent receives observations generated stochastically from a "latent" factored state. These observations are "rich enough" to enable decoding of the latent state and remove partial observability concerns. Since the latent state is combinatorial, the size of state space is exponential in the number of latent factors. We create a learning algorithm FactoRL (Fact-o-Rel) for this setting,  which uses noise-contrastive learning to identify latent structures in emission processes and discover a factorized state space. We derive polynomial sample complexity guarantees for FactoRL which polynomially depend upon the number factors, and very weakly depend on the size of the observation space.  We also provide a guarantee of polynomial time complexity when given access to an efficient planning algorithm.

----

## [664] Sharper Generalization Bounds for Learning with Gradient-dominated Objective Functions

**Authors**: *Yunwen Lei, Yiming Ying*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=r28GdiQF7vM](https://openreview.net/forum?id=r28GdiQF7vM)

**Abstract**:

Stochastic optimization has become the workhorse behind many successful machine learning applications, which motivates a lot of theoretical analysis to understand its empirical behavior. As a comparison, there is far less work to study the generalization behavior especially in a non-convex learning setting. In this paper, we study the generalization behavior of stochastic optimization by leveraging the algorithmic stability for learning with $eta$-gradient-dominated objective functions. We develop generalization bounds of the order $O(1/(neta))$ plus the convergence rate of the optimization algorithm, where n is the sample size. Our stability analysis significantly improves the existing non-convex analysis by removing the bounded gradient assumption and implying better generalization bounds. We achieve this improvement by exploiting the smoothness of loss functions instead of the Lipschitz condition in Charles & Papailiopoulos (2018). We apply our general results to various stochastic optimization algorithms, which show clearly how the variance-reduction techniques improve not only training but also generalization. Furthermore, our discussion explains how interpolation helps generalization for highly expressive models.

----

## [665] Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets

**Authors**: *Hayeon Lee, Eunyoung Hyung, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rkQuFUmUOg3](https://openreview.net/forum?id=rkQuFUmUOg3)

**Abstract**:

Despite the success of recent Neural Architecture Search (NAS) methods on various tasks which have shown to output networks that largely outperform human-designed networks, conventional NAS methods have mostly tackled the optimization of searching for the network architecture for a single task (dataset), which does not generalize well across multiple tasks (datasets). Moreover, since such task-specific methods search for a neural architecture from scratch for every given task, they incur a large computational cost, which is problematic when the time and monetary budget are limited. In this paper, we propose an efficient NAS framework that is trained once on a database consisting of datasets and pretrained networks and can rapidly search for a neural architecture for a novel dataset. The proposed MetaD2A (Meta Dataset-to-Architecture) model can stochastically generate graphs (architectures) from a given set (dataset) via a cross-modal latent space learned with amortized meta-learning. Moreover, we also propose a meta-performance predictor to estimate and select the best architecture without direct training on target datasets. The experimental results demonstrate that our model meta-learned on subsets of ImageNet-1K and architectures from NAS-Bench 201 search space successfully generalizes to multiple unseen datasets including CIFAR-10 and CIFAR-100, with an average search time of 33 GPU seconds. Even under MobileNetV3 search space, MetaD2A is 5.5K times faster than NSGANetV2, a transferable NAS method, with comparable performance. We believe that the MetaD2A proposes a new research direction for rapid NAS as well as ways to utilize the knowledge from rich databases of datasets and architectures accumulated over the past years. Code is available at https://github.com/HayeonLee/MetaD2A

----

## [666] Relating by Contrasting: A Data-efficient Framework for Multimodal Generative Models

**Authors**: *Yuge Shi, Brooks Paige, Philip H. S. Torr, N. Siddharth*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vhKe9UFbrJo](https://openreview.net/forum?id=vhKe9UFbrJo)

**Abstract**:

Multimodal learning for generative models often refers to the learning of abstract concepts from the commonality of information in multiple modalities, such as vision and language. While it has proven effective for learning generalisable representations, the training of such models often requires a large amount of related multimodal data that shares commonality, which can be expensive to come by. To mitigate this, we develop a novel contrastive framework for generative model learning, allowing us to train the model not just by the commonality between modalities, but by the distinction between "related" and "unrelated" multimodal data. We show in experiments that our method enables data-efficient multimodal learning on challenging datasets for various multimodal VAE models. We also show that under our proposed framework, the generative model can accurately identify related samples from unrelated ones, making it possible to make use of the plentiful unlabeled, unpaired multimodal data.

----

## [667] FedMix: Approximation of Mixup under Mean Augmented Federated Learning

**Authors**: *Tehrim Yoon, Sumin Shin, Sung Ju Hwang, Eunho Yang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ogga20D2HO-](https://openreview.net/forum?id=Ogga20D2HO-)

**Abstract**:

Federated learning (FL) allows edge devices to collectively learn a model without directly sharing data within each device, thus preserving privacy and eliminating the need to store data globally. While there are promising results under the assumption of independent and identically distributed (iid) local data, current state-of-the-art algorithms suffer a performance degradation as the heterogeneity of local data across clients increases. To resolve this issue, we propose a simple framework, \emph{Mean Augmented Federated Learning (MAFL)}, where clients send and receive \emph{averaged} local data, subject to the privacy requirements of target applications. Under our framework, we propose a new augmentation algorithm, named \emph{FedMix}, which is inspired by a phenomenal yet simple data augmentation method, Mixup, but does not require local raw data to be directly shared among devices. Our method shows greatly improved performance in the standard benchmark datasets of FL, under highly non-iid federated settings, compared to conventional algorithms.

----

## [668] Generalized Variational Continual Learning

**Authors**: *Noel Loo, Siddharth Swaroop, Richard E. Turner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_IM-AfFhna9](https://openreview.net/forum?id=_IM-AfFhna9)

**Abstract**:

Continual learning deals with training models on new tasks and datasets in an online fashion. One strand of research has used probabilistic regularization for continual learning, with two of the main approaches in this vein being Online Elastic Weight Consolidation (Online EWC) and Variational Continual Learning (VCL). VCL employs variational inference, which in other settings has been improved empirically by applying likelihood-tempering. We show that applying this modification to VCL recovers Online EWC as a limiting case, allowing for interpolation between the two approaches. We term the general algorithm Generalized VCL (GVCL). In order to mitigate the observed overpruning effect of VI, we take inspiration from a common multi-task architecture, neural networks with task-specific FiLM layers, and find that this addition leads to significant performance gains, specifically for variational methods. In the small-data regime, GVCL strongly outperforms existing baselines. In larger datasets, GVCL with FiLM layers outperforms or is competitive with existing baselines in terms of accuracy, whilst also providing significantly better calibration.

----

## [669] Understanding and Improving Lexical Choice in Non-Autoregressive Translation

**Authors**: *Liang Ding, Longyue Wang, Xuebo Liu, Derek F. Wong, Dacheng Tao, Zhaopeng Tu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZTFeSBIX9C](https://openreview.net/forum?id=ZTFeSBIX9C)

**Abstract**:

Knowledge distillation (KD) is essential for training non-autoregressive translation (NAT) models by reducing the complexity of the raw data with an autoregressive teacher model. In this study, we empirically show that as a side effect of this training, the lexical choice errors on low-frequency words are propagated to the NAT model from the teacher model. To alleviate this problem, we propose to expose the raw data to NAT models to restore the useful information of low-frequency words, which are missed in the distilled data. To this end, we introduce an extra Kullback-Leibler divergence term derived by comparing the lexical choice of NAT model and that embedded in the raw data. Experimental results across language pairs and model architectures demonstrate the effectiveness and universality of the proposed approach.  Extensive analyses confirm our claim that our approach improves performance by reducing the lexical choice errors on low-frequency words.  Encouragingly, our approach pushes the SOTA NAT performance on the WMT14 English-German and WMT16 Romanian-English datasets up to 27.8 and 33.8 BLEU points, respectively.

----

## [670] Bayesian Context Aggregation for Neural Processes

**Authors**: *Michael Volpp, Fabian Flürenbrock, Lukas Großberger, Christian Daniel, Gerhard Neumann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ufZN2-aehFa](https://openreview.net/forum?id=ufZN2-aehFa)

**Abstract**:

Formulating scalable probabilistic regression models with reliable uncertainty estimates has been a long-standing challenge in machine learning research.  Recently, casting probabilistic regression as a multi-task learning problem in terms of conditional latent variable (CLV) models such as the Neural Process (NP) has shown promising results. In this paper, we focus on context aggregation, a central component of such architectures, which fuses information from multiple context data points. So far, this aggregation operation has been treated separately from the inference of a latent representation of the target function in CLV models. Our key contribution is to combine these steps into one holistic mechanism by phrasing context aggregation as a Bayesian inference problem. The resulting Bayesian Aggregation (BA) mechanism enables principled handling of task ambiguity, which is key for efficiently processing context information. We demonstrate on a range of challenging experiments that BA consistently improves upon the performance of traditional mean aggregation while remaining computationally efficient and fully compatible with existing NP-based models.

----

## [671] Variational Intrinsic Control Revisited

**Authors**: *Taehwan Kwon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=P0p33rgyoE](https://openreview.net/forum?id=P0p33rgyoE)

**Abstract**:

In this paper, we revisit variational intrinsic control (VIC), an unsupervised reinforcement learning method for finding the largest set of intrinsic options available to an agent. In the original work by Gregor et al. (2016), two VIC algorithms were proposed: one that represents the options explicitly, and the other that does it implicitly. We show that the intrinsic reward used in the latter is subject to bias in stochastic environments, causing convergence to suboptimal solutions. To correct this behavior, we propose two methods respectively based on the transitional probability model and Gaussian Mixture Model. We substantiate our claims through rigorous mathematical derivations and experimental analyses.

----

## [672] Implicit Gradient Regularization

**Authors**: *David G. T. Barrett, Benoit Dherin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3q5IqUrkcF](https://openreview.net/forum?id=3q5IqUrkcF)

**Abstract**:

Gradient descent can be surprisingly good at optimizing deep neural networks without overfitting and without explicit regularization. We find that the discrete steps of gradient descent implicitly regularize models by penalizing gradient descent trajectories that have large loss gradients. We call this Implicit Gradient Regularization (IGR) and we use backward error analysis to calculate the size of this regularization. We confirm empirically that implicit gradient regularization biases gradient descent toward flat minima, where test errors are small and solutions are robust to noisy parameter perturbations. Furthermore, we demonstrate that the implicit gradient regularization term can be used as an explicit regularizer, allowing us to control this gradient regularization directly. More broadly, our work indicates that backward error analysis is a useful theoretical approach to the perennial question of how learning rate, model size, and parameter regularization interact to determine the properties of overparameterized models optimized with gradient descent.

----

## [673] Return-Based Contrastive Representation Learning for Reinforcement Learning

**Authors**: *Guoqing Liu, Chuheng Zhang, Li Zhao, Tao Qin, Jinhua Zhu, Jian Li, Nenghai Yu, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_TM6rT7tXke](https://openreview.net/forum?id=_TM6rT7tXke)

**Abstract**:

Recently, various auxiliary tasks have been proposed to accelerate representation learning and improve sample efficiency in deep reinforcement learning (RL). However, existing auxiliary tasks do not take the characteristics of RL problems into consideration and are unsupervised. By leveraging returns, the most important feedback signals in RL, we propose a novel auxiliary task that forces the learnt representations to discriminate state-action pairs with different returns. Our auxiliary loss is theoretically justified to learn representations that capture the structure of a new form of state-action abstraction, under which state-action pairs with similar return distributions are aggregated together. Empirically, our algorithm outperforms strong baselines on complex tasks in Atari games and DeepMind Control suite, and achieves even better performance when combined with existing auxiliary tasks.

----

## [674] Scalable Bayesian Inverse Reinforcement Learning

**Authors**: *Alex James Chan, Mihaela van der Schaar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4qR3coiNaIv](https://openreview.net/forum?id=4qR3coiNaIv)

**Abstract**:

Bayesian inference over the reward presents an ideal solution to the ill-posed nature of the inverse reinforcement learning problem. Unfortunately current methods generally do not scale well beyond the small tabular setting due to the need for an inner-loop MDP solver, and even non-Bayesian methods that do themselves scale often require extensive interaction with the environment to perform well, being inappropriate for high stakes or costly applications such as healthcare. In this paper we introduce our method, Approximate Variational Reward Imitation Learning (AVRIL), that addresses both of these issues by jointly learning an approximate posterior distribution over the reward that scales to arbitrarily complicated state spaces alongside an appropriate policy in a completely offline manner through a variational approach to said latent reward. Applying our method to real medical data alongside classic control simulations, we demonstrate Bayesian reward inference in environments beyond the scope of current methods, as well as task performance competitive with focused offline imitation learning algorithms.

----

## [675] Exemplary Natural Images Explain CNN Activations Better than State-of-the-Art Feature Visualization

**Authors**: *Judy Borowski, Roland Simon Zimmermann, Judith Schepers, Robert Geirhos, Thomas S. A. Wallis, Matthias Bethge, Wieland Brendel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QO9-y8also-](https://openreview.net/forum?id=QO9-y8also-)

**Abstract**:

Feature visualizations such as synthetic maximally activating images are a widely used explanation method to better understand the information processing of convolutional neural networks (CNNs). At the same time, there are concerns that these visualizations might not accurately represent CNNs' inner workings. Here, we measure how much extremely activating images help humans to predict CNN activations.
Using a well-controlled psychophysical paradigm, we compare the informativeness of synthetic images by Olah et al. (2017) with a simple baseline visualization, namely exemplary natural images that also strongly activate a specific feature map. Given either synthetic or natural reference images, human participants choose which of two query images leads to strong positive activation. The experiment is designed to maximize participants' performance, and is the first to probe intermediate instead of final layer representations. We find that synthetic images indeed provide helpful information about feature map activations ($82\pm4\%$ accuracy; chance would be $50\%$). However, natural images --- originally intended to be a baseline --- outperform these synthetic images by a wide margin ($92\pm2\%$). Additionally, participants are faster and more confident for natural images, whereas subjective impressions about the interpretability of the feature visualizations by Olah et al. (2017) are mixed. The higher informativeness of natural images holds across most layers, for both expert and lay participants as well as for hand- and randomly-picked feature visualizations. Even if only a single reference image is given, synthetic images provide less information than natural images ($65\pm5\%$ vs. $73\pm4\%$). In summary, synthetic images from a popular feature visualization method are significantly less informative for assessing CNN activations than natural images. We argue that visualization methods should improve over this simple baseline.

----

## [676] LiftPool: Bidirectional ConvNet Pooling

**Authors**: *Jiaojiao Zhao, Cees G. M. Snoek*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kE3vd639uRW](https://openreview.net/forum?id=kE3vd639uRW)

**Abstract**:

Pooling is a critical operation in convolutional neural networks for increasing receptive fields and improving robustness to input variations. Most existing pooling operations downsample the feature maps,  which is a lossy process.   Moreover, they are not invertible: upsampling a downscaled feature map can not recover the lost information in the downsampling.  By adopting the philosophy of the classical Lifting Scheme from signal processing, we propose LiftPool for bidirectional pooling layers, including LiftDownPool and LiftUpPool.  LiftDownPool decomposes a feature map into various downsized sub-bands,  each of which contains information with different frequencies. As the pooling function in LiftDownPool is perfectly invertible, by performing LiftDownPool backward, a corresponding up-pooling layer LiftUpPool is able to generate a refined upsampled feature map using the detail subbands, which is useful for image-to-image translation challenges.  Experiments show the proposed methods achieve better results on image classification and semantic segmentation,  using various backbones. Moreover, LiftDownPool offers better robustness to input corruptions and perturbations.

----

## [677] Adversarial score matching and improved sampling for image generation

**Authors**: *Alexia Jolicoeur-Martineau, Rémi Piché-Taillefer, Ioannis Mitliagkas, Remi Tachet des Combes*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eLfqMl3z3lq](https://openreview.net/forum?id=eLfqMl3z3lq)

**Abstract**:

Denoising Score Matching with Annealed Langevin Sampling (DSM-ALS) has recently found success in generative modeling. The approach works by first training a neural network to estimate the score of a distribution, and then using Langevin dynamics to sample from the data distribution assumed by the score network. Despite the convincing visual quality of samples, this method appears to perform worse than Generative Adversarial Networks (GANs) under the Fréchet Inception Distance, a standard metric for generative models. We show that this apparent gap vanishes when denoising the final Langevin samples using the score network.
In addition, we propose two improvements to DSM-ALS:  1) Consistent Annealed Sampling as a more stable alternative to Annealed Langevin Sampling, and 2) a hybrid training formulation, composed of both Denoising Score Matching and adversarial objectives. By combining these two techniques and exploring different network architectures, we elevate score matching methods and obtain results competitive with state-of-the-art image generation on CIFAR-10.

----

## [678] Transient Non-stationarity and Generalisation in Deep Reinforcement Learning

**Authors**: *Maximilian Igl, Gregory Farquhar, Jelena Luketina, Wendelin Boehmer, Shimon Whiteson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Qun8fv4qSby](https://openreview.net/forum?id=Qun8fv4qSby)

**Abstract**:

Non-stationarity can arise in Reinforcement Learning (RL) even in stationary environments. For example, most RL algorithms collect new data throughout training, using a non-stationary behaviour policy. Due to the transience of this non-stationarity, it is often not explicitly addressed in deep RL and a single neural network is continually updated. However, we find evidence that neural networks exhibit a memory effect, where these transient non-stationarities can permanently impact the latent representation and adversely affect generalisation performance. Consequently, to improve generalisation of deep RL agents, we propose Iterated Relearning (ITER). ITER augments standard RL training by repeated knowledge transfer of the current policy into a freshly initialised network, which thereby experiences less non-stationarity during training. Experimentally, we show that ITER improves performance on the challenging generalisation benchmarks ProcGen and Multiroom.

----

## [679] On the Origin of Implicit Regularization in Stochastic Gradient Descent

**Authors**: *Samuel L. Smith, Benoit Dherin, David G. T. Barrett, Soham De*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rq_Qr0c1Hyo](https://openreview.net/forum?id=rq_Qr0c1Hyo)

**Abstract**:

For infinitesimal learning rates, stochastic gradient descent (SGD) follows the path of gradient flow on the full batch loss function. However moderately large learning rates can achieve higher test accuracies, and this generalization benefit is not explained by convergence bounds, since the learning rate which maximizes test accuracy is often larger than the learning rate which minimizes training loss. To interpret this phenomenon we prove that for SGD with random shuffling, the mean SGD iterate also stays close to the path of gradient flow if the learning rate is small and finite, but on a modified loss. This modified loss is composed of the original loss function and an implicit regularizer, which penalizes the norms of the minibatch gradients. Under mild assumptions, when the batch size is small the scale of the implicit regularization term is proportional to the ratio of the learning rate to the batch size. We verify empirically that explicitly including the implicit regularizer in the loss can enhance the test accuracy when the learning rate is small.

----

## [680] Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective

**Authors**: *Muhammet Balcilar, Guillaume Renton, Pierre Héroux, Benoit Gaüzère, Sébastien Adam, Paul Honeine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-qh0M9XWxnv](https://openreview.net/forum?id=-qh0M9XWxnv)

**Abstract**:

In the recent literature of Graph Neural Networks (GNN), the expressive power of models has been studied through their capability to distinguish if two given graphs are isomorphic or not. Since the graph isomorphism problem is NP-intermediate, and Weisfeiler-Lehman (WL) test can give sufficient but not enough evidence in polynomial time, the theoretical power of GNNs is usually evaluated by the equivalence of WL-test order, followed by an empirical analysis of the models on some reference inductive and transductive datasets. However, such analysis does not account the signal processing pipeline, whose capability is generally evaluated in the spectral domain. In this paper, we argue that a spectral analysis of GNNs behavior can provide a complementary point of view to go one step further in the understanding of GNNs. By bridging the gap between the spectral and spatial design of graph convolutions, we theoretically demonstrate some equivalence of the graph convolution process regardless it is designed in the spatial or the spectral domain. Using this connection, we managed to re-formulate most of the state-of-the-art graph neural networks into one common framework. This general framework allows to lead a spectral analysis of the most popular GNNs, explaining their performance and showing their limits according to spectral point of view. Our theoretical spectral analysis is confirmed by experiments on various graph databases. Furthermore, we demonstrate the necessity of high and/or band-pass filters on a graph dataset, while the majority of GNN is limited to only low-pass and inevitably it fails.

----

## [681] Pruning Neural Networks at Initialization: Why Are We Missing the Mark?

**Authors**: *Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ig-VyQc-MLK](https://openreview.net/forum?id=Ig-VyQc-MLK)

**Abstract**:

Recent work has explored the possibility of pruning neural networks at initialization. We assess proposals for doing so: SNIP (Lee et al., 2019), GraSP (Wang et al., 2020), SynFlow (Tanaka et al., 2020), and magnitude pruning. Although these methods surpass the trivial baseline of random pruning, they remain below the accuracy of magnitude pruning after training, and we endeavor to understand why. We show that, unlike pruning after training, randomly shuffling the weights these methods prune within each layer or sampling new initial values preserves or improves accuracy. As such, the per-weight pruning decisions made by these methods can be replaced by a per-layer choice of the fraction of weights to prune. This property suggests broader challenges with the underlying pruning heuristics, the desire to prune at initialization, or both.

----

## [682] SkipW: Resource Adaptable RNN with Strict Upper Computational Limit

**Authors**: *Tsiry Mayet, Anne Lambert, Pascal Leguyadec, Françoise Le Bolzer, François Schnitzler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=2CjEVW-RGOJ](https://openreview.net/forum?id=2CjEVW-RGOJ)

**Abstract**:

We introduce Skip-Window, a method to allow recurrent neural networks (RNNs) to trade off accuracy for computational cost during the analysis of a sequence. Similarly to existing approaches, Skip-Window extends existing RNN cells by adding a mechanism to encourage the model to process fewer inputs. Unlike existing approaches, Skip-Window is able to respect a strict computational budget, making this model more suitable for limited hardware. We evaluate this approach on two datasets: a human activity recognition task and adding task. Our results show that Skip-Window is able to exceed the accuracy of existing approaches for a lower computational cost while strictly limiting said cost.

----

## [683] Into the Wild with AudioScope: Unsupervised Audio-Visual Separation of On-Screen Sounds

**Authors**: *Efthymios Tzinis, Scott Wisdom, Aren Jansen, Shawn Hershey, Tal Remez, Dan Ellis, John R. Hershey*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MDsQkFP1Aw](https://openreview.net/forum?id=MDsQkFP1Aw)

**Abstract**:

Recent progress in deep learning has enabled many advances in sound separation and visual scene understanding. However, extracting sound sources which are apparent in natural videos remains an open problem. In this work, we present AudioScope, a novel audio-visual sound separation framework that can be trained without supervision to isolate on-screen sound sources from real in-the-wild videos. Prior audio-visual separation work assumed artificial limitations on the domain of sound classes (e.g., to speech or music), constrained the number of sources, and required strong sound separation or visual segmentation labels. AudioScope overcomes these limitations, operating on an open domain of sounds, with variable numbers of sources, and without labels or prior visual segmentation.  The training procedure for AudioScope uses mixture invariant training (MixIT) to separate synthetic mixtures of mixtures (MoMs) into individual sources, where noisy labels for mixtures are provided by an unsupervised audio-visual coincidence model. Using the noisy labels, along with attention between video and audio features, AudioScope learns to identify audio-visual similarity and to suppress off-screen sounds. We demonstrate the effectiveness of our approach using a dataset of video clips extracted from open-domain YFCC100m video data. This dataset contains a wide diversity of sound classes recorded in unconstrained conditions, making the application of previous methods unsuitable. For evaluation and semi-supervised experiments, we collected human labels for presence of on-screen and off-screen sounds on a small subset of clips.

----

## [684] Simple Augmentation Goes a Long Way: ADRL for DNN Quantization

**Authors**: *Lin Ning, Guoyang Chen, Weifeng Zhang, Xipeng Shen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Qr0aRliE_Hb](https://openreview.net/forum?id=Qr0aRliE_Hb)

**Abstract**:

Mixed precision quantization improves DNN performance by assigning different layers with different bit-width values. Searching for the optimal bit-width for each layer, however, remains a challenge. Deep Reinforcement Learning (DRL) shows some recent promise. It however suffers instability due to function approximation errors, causing large variances in the early training stages, slow convergence, and suboptimal policies in the mixed-precision quantization problem. This paper proposes augmented DRL (ADRL) as a way to alleviate these issues. This new strategy augments the neural networks in DRL with a complementary scheme to boost the performance of learning. The paper examines the effectiveness of ADRL both analytically and empirically, showing that it can produce more accurate quantized models than the state of the art DRL-based quantization while improving the learning speed by 4.5-64 times.

----

## [685] Few-Shot Bayesian Optimization with Deep Kernel Surrogates

**Authors**: *Martin Wistuba, Josif Grabocka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bJxgv5C3sYc](https://openreview.net/forum?id=bJxgv5C3sYc)

**Abstract**:

Hyperparameter optimization (HPO) is a central pillar in the automation of machine learning solutions and is mainly performed via Bayesian optimization, where a parametric surrogate is learned to approximate the black box response function (e.g. validation error). Unfortunately, evaluating the response function is computationally intensive. As a remedy, earlier work emphasizes the need for transfer learning surrogates which learn to optimize hyperparameters for an algorithm from other tasks. In contrast to previous work, we propose to rethink HPO as a few-shot learning problem in which we train a shared deep surrogate model to quickly adapt (with few response evaluations) to the response function of a new task. We propose the use of a deep kernel network for a Gaussian process surrogate that is meta-learned in an end-to-end fashion in order to jointly approximate the response functions of a collection of training data sets. As a result, the novel few-shot optimization of our deep kernel surrogate leads to new state-of-the-art results at HPO compared to several recent methods on diverse metadata sets.

----

## [686] AdaSpeech: Adaptive Text to Speech for Custom Voice

**Authors**: *Mingjian Chen, Xu Tan, Bohan Li, Yanqing Liu, Tao Qin, Sheng Zhao, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Drynvt7gg4L](https://openreview.net/forum?id=Drynvt7gg4L)

**Abstract**:

Custom voice, a specific text to speech (TTS) service in commercial speech platforms, aims to adapt a source TTS model to synthesize personal voice for a target speaker using few speech from her/him. Custom voice presents two unique challenges for TTS adaptation: 1) to support diverse customers, the adaptation model needs to handle diverse acoustic conditions which could be very different from source speech data, and 2) to support a large number of customers, the adaptation parameters need to be small enough for each target speaker to reduce memory usage while maintaining high voice quality. In this work, we propose AdaSpeech, an adaptive TTS system for high-quality and efficient customization of new voices. We design several techniques in AdaSpeech to address the two challenges in custom voice: 1) To handle different acoustic conditions, we model the acoustic information in both utterance and phoneme level. Specifically, we use one acoustic encoder to extract an utterance-level vector and another one to extract a sequence of phoneme-level vectors from the target speech during pre-training and fine-tuning; in inference, we extract the utterance-level vector from a reference speech and use an acoustic predictor to predict the phoneme-level vectors. 2) To better trade off the adaptation parameters and voice quality, we introduce conditional layer normalization in the mel-spectrogram decoder of AdaSpeech, and fine-tune this part in addition to speaker embedding for adaptation. We pre-train the source TTS model on LibriTTS datasets and fine-tune it on VCTK and LJSpeech datasets (with different acoustic conditions from LibriTTS) with few adaptation data, e.g., 20 sentences, about 1 minute speech. Experiment results show that AdaSpeech achieves much better adaptation quality than baseline methods, with only about 5K specific parameters for each speaker, which demonstrates its effectiveness for custom voice. The audio samples are available at https://speechresearch.github.io/adaspeech/.

----

## [687] HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients

**Authors**: *Enmao Diao, Jie Ding, Vahid Tarokh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TNkPBBYFkXg](https://openreview.net/forum?id=TNkPBBYFkXg)

**Abstract**:

Federated Learning (FL) is a method of training machine learning models on private data distributed over a large number of possibly heterogeneous clients such as mobile phones and IoT devices. In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation complexities and still produce a single global inference model. For the first time, our method challenges the underlying assumption of existing work that local models have to share the same architecture as the global model. We demonstrate several strategies to enhance FL training and conduct extensive empirical evaluations, including five computation complexity levels of three model architecture on three datasets. We show that adaptively distributing subnetworks according to clients' capabilities is both computation and communication efficient.

----

## [688] DINO: A Conditional Energy-Based GAN for Domain Translation

**Authors**: *Konstantinos Vougioukas, Stavros Petridis, Maja Pantic*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=WAISmwsqDsb](https://openreview.net/forum?id=WAISmwsqDsb)

**Abstract**:

Domain translation is the process of transforming data from one domain to another while preserving the common semantics. Some of the most popular domain translation systems are based on conditional generative adversarial networks, which use source domain data to drive the generator and as an input to the discriminator. However, this approach does not enforce the preservation of shared semantics since the conditional input can often be ignored by the discriminator. We propose an alternative method for conditioning and present a new framework, where two networks are simultaneously trained, in a supervised manner, to perform domain translation in opposite directions. Our method is not only better at capturing the shared information between two domains but is more generic and can be applied to a broader range of problems. The proposed framework performs well even in challenging cross-modal translations, such as video-driven speech reconstruction, for which other systems struggle to maintain correspondence.

----

## [689] Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning

**Authors**: *Tsung-Wei Ke, Jyh-Jing Hwang, Stella X. Yu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=N33d7wjgzde](https://openreview.net/forum?id=N33d7wjgzde)

**Abstract**:

Weakly supervised segmentation requires assigning a label to every pixel based on training instances with partial annotations such as image-level tags, object bounding boxes, labeled points and scribbles. This task is challenging, as coarse annotations (tags, boxes) lack precise pixel localization whereas sparse annotations (points, scribbles) lack broad region coverage. Existing methods tackle these two types of weak supervision differently: Class activation maps are used to localize coarse labels and iteratively refine the segmentation model, whereas conditional random fields are used to propagate sparse labels to the entire image.

We formulate weakly supervised segmentation as a semi-supervised metric learning problem, where pixels of the same (different) semantics need to be mapped to the same (distinctive) features. We propose 4 types of contrastive relationships between pixels and segments in the feature space, capturing low-level image similarity, semantic annotation, co-occurrence, and feature affinity They act as priors; the pixel-wise feature can be learned from training images with any partial annotations in a data-driven fashion. In particular, unlabeled pixels in training images participate not only in data-driven grouping within each image, but also in discriminative feature learning within and across images. We deliver a universal weakly supervised segmenter with significant gains on Pascal VOC and DensePose. Our code is publicly available at https://github.com/twke18/SPML.

----

## [690] PC2WF: 3D Wireframe Reconstruction from Raw Point Clouds

**Authors**: *Yujia Liu, Stefano D'Aronco, Konrad Schindler, Jan Dirk Wegner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8X2eaSZxTP](https://openreview.net/forum?id=8X2eaSZxTP)

**Abstract**:

We introduce PC2WF, the first end-to-end trainable deep network architecture to convert a 3D point cloud into a wireframe model. The network takes as input an unordered set of 3D points sampled from the surface of some object, and outputs a wireframe of that object, i.e., a sparse set of corner points linked by line segments. Recovering the wireframe is a challenging task, where the numbers of both vertices and edges are different for every instance, and a-priori unknown. Our architecture gradually builds up the model: It starts by encoding the points into feature vectors. Based on those features, it identifies a pool of candidate vertices, then prunes those candidates to a final set of corner vertices and refines their locations. Next, the corners are linked with an exhaustive set of candidate edges, which is again pruned to obtain the final wireframe. All steps are trainable, and errors can be backpropagated through the entire sequence. We validate the proposed model on a publicly available synthetic dataset, for which the ground truth wireframes are accessible, as well as on a new real-world dataset. Our model produces wireframe abstractions of good quality and outperforms several baselines.

----

## [691] Multi-resolution modeling of a discrete stochastic process identifies causes of cancer

**Authors**: *Adam Uri Yaari, Maxwell Sherman, Oliver Clarke Priebe, Po-Ru Loh, Boris Katz, Andrei Barbu, Bonnie Berger*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KtH8W3S_RE](https://openreview.net/forum?id=KtH8W3S_RE)

**Abstract**:

Detection of cancer-causing mutations within the vast and mostly unexplored human genome is a major challenge. Doing so requires modeling the background mutation rate, a highly non-stationary stochastic process, across regions of interest varying in size from one to millions of positions. Here, we present the split-Poisson-Gamma (SPG) distribution, an extension of the classical Poisson-Gamma formulation, to model a discrete stochastic process at multiple resolutions. We demonstrate that the probability model has a closed-form posterior, enabling efficient and accurate linear-time prediction over any length scale after the parameters of the model have been inferred a single time. We apply our framework to model mutation rates in tumors and show that model parameters can be accurately inferred from high-dimensional epigenetic data using a convolutional neural network, Gaussian process, and maximum-likelihood estimation. Our method is both more accurate and more efficient than existing models over a large range of length scales. We demonstrate the usefulness of multi-resolution modeling by detecting genomic elements that drive tumor emergence and are of vastly differing sizes.

----

## [692] C-Learning: Horizon-Aware Cumulative Accessibility Estimation

**Authors**: *Panteha Naderian, Gabriel Loaiza-Ganem, Harry J. Braviner, Anthony L. Caterini, Jesse C. Cresswell, Tong Li, Animesh Garg*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=W3Wf_wKmqm9](https://openreview.net/forum?id=W3Wf_wKmqm9)

**Abstract**:

Multi-goal reaching is an important problem in reinforcement learning needed to achieve algorithmic generalization. Despite recent advances in this field, current algorithms suffer from three major challenges: high sample complexity, learning only a single way of reaching the goals,  and difficulties in solving complex motion planning tasks. In order to address these limitations, we introduce the concept of cumulative accessibility functions, which measure the reachability of a goal from a given state within a specified horizon. We show that these functions obey a recurrence relation, which enables learning from offline interactions. We also prove that optimal cumulative accessibility functions are monotonic in the planning horizon. Additionally, our method can trade off speed and reliability in goal-reaching by suggesting multiple paths to a single goal depending on the provided horizon. We evaluate our approach on a set of multi-goal discrete and continuous control tasks. We show that our method outperforms state-of-the-art goal-reaching algorithms in success rate, sample complexity, and path optimality. Our code is available at https://github.com/layer6ai-labs/CAE, and additional visualizations can be found at https://sites.google.com/view/learning-cae/.

----

## [693] Shapley Explanation Networks

**Authors**: *Rui Wang, Xiaoqian Wang, David I. Inouye*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vsU0efpivw](https://openreview.net/forum?id=vsU0efpivw)

**Abstract**:

Shapley values have become one of the most popular feature attribution explanation methods. However, most prior work has focused on post-hoc Shapley explanations, which can be computationally demanding due to its exponential time complexity and preclude model regularization based on Shapley explanations during training. Thus, we propose to incorporate Shapley values themselves as latent representations in deep models  thereby making Shapley explanations first-class citizens in the modeling paradigm. This intrinsic explanation approach enables layer-wise explanations, explanation regularization of the model during training, and fast explanation computation at test time. We define the Shapley transform that transforms the input into a Shapley representation given a specific function. We operationalize the Shapley transform as a neural network module and construct both shallow and deep networks, called ShapNets, by composing Shapley modules. We prove that our Shallow ShapNets compute the exact Shapley values and our Deep ShapNets maintain the missingness and accuracy properties of Shapley values. We demonstrate on synthetic and real-world datasets that our ShapNets enable layer-wise Shapley explanations, novel Shapley regularizations during training, and fast computation while maintaining reasonable performance. Code is available at https://github.com/inouye-lab/ShapleyExplanationNetworks.

----

## [694] The role of Disentanglement in Generalisation

**Authors**: *Milton Llera Montero, Casimir J. H. Ludwig, Rui Ponte Costa, Gaurav Malhotra, Jeffrey S. Bowers*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qbH974jKUVy](https://openreview.net/forum?id=qbH974jKUVy)

**Abstract**:

Combinatorial generalisation — the ability to understand and produce novel combinations of familiar elements — is a core capacity of human intelligence that current AI systems struggle with. Recently, it has been suggested that learning disentangled representations may help address this problem. It is claimed that such representations should be able to capture the compositional structure of the world which can then be combined to support combinatorial generalisation. In this study, we systematically tested how the degree of disentanglement affects various forms of generalisation, including two forms of combinatorial generalisation that varied in difficulty. We trained three classes of variational autoencoders (VAEs) on two datasets on an unsupervised task by excluding combinations of generative factors during training. At test time we ask the models to reconstruct the missing combinations in order to measure generalisation performance. Irrespective of the degree of disentanglement, we found that the models supported only weak combinatorial generalisation. We obtained the same outcome when we directly input perfectly disentangled representations as the latents, and when we tested a model on a more complex task that explicitly required independent generative factors to be controlled. While learning disentangled representations does improve interpretability and sample efficiency in some downstream tasks, our results suggest that they are not sufficient for supporting more difficult forms of generalisation.

----

## [695] Learning N: M Fine-grained Structured Sparse Neural Networks From Scratch

**Authors**: *Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, Hongsheng Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=K9bw7vqp_s](https://openreview.net/forum?id=K9bw7vqp_s)

**Abstract**:

Sparsity in Deep Neural Networks (DNNs) has been widely studied to compress and accelerate the models on resource-constrained environments. It can be generally categorized into unstructured fine-grained sparsity that zeroes out multiple individual weights distributed across the neural network, and structured coarse-grained sparsity which prunes blocks of sub-networks of a neural network. Fine-grained sparsity can achieve a high compression ratio but is not hardware friendly and hence receives limited speed gains. On the other hand, coarse-grained sparsity cannot simultaneously achieve both apparent acceleration on modern GPUs and
decent performance. In this paper, we are the first to study training from scratch an N:M fine-grained structured sparse network, which can maintain the advantages of both unstructured fine-grained sparsity and structured coarse-grained sparsity simultaneously on specifically designed GPUs. Specifically, a 2 : 4 sparse network could achieve 2× speed-up without performance drop on Nvidia A100 GPUs. Furthermore, we propose a novel and effective ingredient, sparse-refined straight-through estimator (SR-STE), to alleviate the negative influence of the approximated gradients computed by vanilla STE during optimization. We also define a metric, Sparse Architecture Divergence (SAD), to measure the sparse network’s topology change during the training process. Finally, We justify SR-STE’s advantages with SAD and demonstrate the effectiveness of SR-STE by performing
comprehensive experiments on various tasks. Anonymous code and model will be at available at https://github.com/anonymous-NM-sparsity/NM-sparsity.

----

## [696] Unsupervised Meta-Learning through Latent-Space Interpolation in Generative Models

**Authors**: *Siavash Khodadadeh, Sharare Zehtabian, Saeed Vahidian, Weijia Wang, Bill Lin, Ladislau Bölöni*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XOjv2HxIF6i](https://openreview.net/forum?id=XOjv2HxIF6i)

**Abstract**:

Several recently proposed unsupervised meta-learning approaches rely on synthetic meta-tasks created using techniques such as random selection, clustering and/or augmentation. In this work, we describe a novel approach that generates meta-tasks using generative models. The proposed family of algorithms generate pairs of in-class and out-of-class samples from the latent space in a principled way, allowing us to create synthetic classes forming the training and validation data of a meta-task. We find that the proposed approach, LAtent Space Interpolation Unsupervised Meta-learning (LASIUM), outperforms or is competitive with current unsupervised learning baselines on few-shot classification tasks on the most widely used benchmark datasets.

----

## [697] On Data-Augmentation and Consistency-Based Semi-Supervised Learning

**Authors**: *Atin Ghosh, Alexandre H. Thiéry*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7FNqrcPtieT](https://openreview.net/forum?id=7FNqrcPtieT)

**Abstract**:

Recently proposed consistency-based Semi-Supervised Learning (SSL) methods such as the Pi-model, temporal ensembling, the mean teacher, or the virtual adversarial training, achieve the state of the art results in several SSL tasks. These methods can typically reach performances that are comparable to their fully supervised counterparts while using only a fraction of labelled examples. Despite these methodological advances, the understanding of these methods is still relatively limited. To make progress, we analyse (variations of) the Pi-model in settings where analytically tractable results can be obtained. We establish links with Manifold Tangent Classifiers and demonstrate that the quality of the perturbations is key to obtaining reasonable SSL performances. Furthermore, we propose a simple extension of the Hidden Manifold Model that naturally incorporates data-augmentation schemes and offers a tractable framework for understanding SSL methods.

----

## [698] Learning from Demonstration with Weakly Supervised Disentanglement

**Authors**: *Yordan Hristov, Subramanian Ramamoorthy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ldau9eHU-qO](https://openreview.net/forum?id=Ldau9eHU-qO)

**Abstract**:

Robotic manipulation tasks, such as wiping with a soft sponge, require control from multiple rich sensory modalities. Human-robot interaction, aimed at teach- ing robots, is difficult in this setting as there is potential for mismatch between human and machine comprehension of the rich data streams. We treat the task of interpretable learning from demonstration as an optimisation problem over a probabilistic generative model. To account for the high-dimensionality of the data, a high-capacity neural network is chosen to represent the model. The latent variables in this model are explicitly aligned with high-level notions and concepts that are manifested in a set of demonstrations. We show that such alignment is best achieved through the use of labels from the end user, in an appropriately restricted vocabulary, in contrast to the conventional approach of the designer picking a prior over the latent variables. Our approach is evaluated in the context of two table-top robot manipulation tasks performed by a PR2 robot – that of dabbing liquids with a sponge (forcefully pressing a sponge and moving it along a surface) and pouring between different containers. The robot provides visual information, arm joint positions and arm joint efforts. We have made videos of the tasks and data available - see supplementary materials at: https://sites.google.com/view/weak-label-lfd.

----

## [699] Neurally Augmented ALISTA

**Authors**: *Freya Behrens, Jonathan Sauder, Peter Jung*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=q_S44KLQ_Aa](https://openreview.net/forum?id=q_S44KLQ_Aa)

**Abstract**:

It is well-established that many iterative sparse reconstruction algorithms can be unrolled to yield a learnable neural network for improved empirical performance. A prime example is learned ISTA (LISTA) where weights, step sizes and thresholds are learned from training data. Recently, Analytic LISTA (ALISTA) has been introduced, combining the strong empirical performance of a fully learned approach like LISTA, while retaining theoretical guarantees of classical compressed sensing algorithms and significantly reducing the number of parameters to learn. However, these parameters are trained to work in expectation, often leading to suboptimal reconstruction of individual targets.  In this work we therefore introduce Neurally Augmented ALISTA, in which an LSTM network is used to compute step sizes and thresholds individually for each target vector during reconstruction. This adaptive approach is theoretically motivated by revisiting the recovery guarantees of ALISTA. We show that our approach further improves empirical performance in sparse reconstruction, in particular outperforming existing algorithms by an increasing margin as the compression ratio becomes more challenging.

----

## [700] Shape or Texture: Understanding Discriminative Features in CNNs

**Authors**: *Md. Amirul Islam, Matthew Kowal, Patrick Esser, Sen Jia, Björn Ommer, Konstantinos G. Derpanis, Neil D. B. Bruce*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NcFEZOi-rLa](https://openreview.net/forum?id=NcFEZOi-rLa)

**Abstract**:

Contrasting the previous evidence that neurons in the later layers of a Convolutional Neural Network (CNN) respond to complex object shapes, recent studies have shown that CNNs actually exhibit a 'texture bias': given an image with both texture and shape cues (e.g., a stylized image), a CNN is biased towards predicting the category corresponding to the texture. However, these previous studies conduct experiments on the final classification output of the network, and fail to robustly evaluate the bias contained (i) in the latent representations, and (ii) on a per-pixel level. In this paper, we design a series of experiments that overcome these issues. We do this with the goal of better understanding what type of shape information contained in the network is discriminative, where shape information is encoded, as well as when the network learns about object shape during training. We show that a network learns the majority of overall shape information at the first few epochs of training and that this information is largely encoded in the last few layers of a CNN. Finally, we show that the encoding of shape does not imply the encoding of localized per-pixel semantic information. The experimental results and findings provide a more accurate understanding of the behaviour of current CNNs, thus helping to inform future design choices.

----

## [701] Convex Potential Flows: Universal Probability Distributions with Optimal Transport and Convex Optimization

**Authors**: *Chin-Wei Huang, Ricky T. Q. Chen, Christos Tsirigotis, Aaron C. Courville*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=te7PVH1sPxJ](https://openreview.net/forum?id=te7PVH1sPxJ)

**Abstract**:

Flow-based models are powerful tools for designing probabilistic models with tractable density. This paper introduces Convex Potential Flows (CP-Flow), a natural and efficient parameterization of invertible models inspired by the optimal transport (OT) theory. CP-Flows are the gradient map of a strongly convex neural potential function. The convexity implies invertibility and allows us to resort to convex optimization to solve the convex conjugate for efficient inversion. To enable maximum likelihood training, we derive a new gradient estimator of the log-determinant of the Jacobian, which involves solving an inverse-Hessian vector product using the conjugate gradient method. The gradient estimator has constant-memory cost, and can be made effectively unbiased by reducing the error tolerance level of the convex optimization routine. Theoretically, we prove that CP-Flows are universal density approximators and are optimal in the OT sense. Our empirical results show that CP-Flow performs competitively on standard benchmarks of density estimation and variational inference.

----

## [702] Wasserstein Embedding for Graph Learning

**Authors**: *Soheil Kolouri, Navid Naderializadeh, Gustavo K. Rohde, Heiko Hoffmann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AAes_3W-2z](https://openreview.net/forum?id=AAes_3W-2z)

**Abstract**:

We present Wasserstein Embedding for Graph Learning (WEGL), a novel and fast framework for embedding entire graphs in a vector space, in which various machine learning models are applicable for graph-level prediction tasks. We leverage new insights on defining similarity between graphs as a function of the similarity between their node embedding distributions. Specifically, we use the Wasserstein distance to measure the dissimilarity between node embeddings of different graphs. Unlike prior work, we avoid pairwise calculation of distances between graphs and reduce the computational complexity from quadratic to linear in the number of graphs. WEGL calculates Monge maps from a reference distribution to each node embedding and, based on these maps, creates a fixed-sized vector representation of the graph. We evaluate our new graph embedding approach on various benchmark graph-property prediction tasks, showing state-of-the-art classification performance while having superior computational efficiency. The code is available at https://github.com/navid-naderi/WEGL.

----

## [703] Meta-learning with negative learning rates

**Authors**: *Alberto Bernacchia*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=60j5LygnmD](https://openreview.net/forum?id=60j5LygnmD)

**Abstract**:

Deep learning models require a large amount of data to perform well. When data is scarce for a target task, we can transfer the knowledge gained by training on similar tasks to quickly learn the target. A successful approach is meta-learning, or "learning to learn" a distribution of tasks, where "learning" is represented by an outer loop, and "to learn" by an inner loop of gradient descent. However, a number of recent empirical studies argue that the inner loop is unnecessary and more simple models work equally well or even better. We study the performance of MAML as a function of the learning rate of the inner loop, where zero learning rate implies that there is no inner loop. Using random matrix theory and exact solutions of linear models, we calculate an algebraic expression for the test loss of MAML applied to mixed linear regression and nonlinear regression with overparameterized models. Surprisingly, while the optimal learning rate for adaptation is positive, we find that the optimal learning rate for training is always negative, a setting that has never been considered before. Therefore, not only does the performance increase by decreasing the learning rate to zero, as suggested by recent work, but it can be increased even further by decreasing the learning rate to negative
values. These results help clarify under what circumstances meta-learning performs best.

----

## [704] Representing Partial Programs with Blended Abstract Semantics

**Authors**: *Maxwell I. Nye, Yewen Pu, Matthew Bowers, Jacob Andreas, Joshua B. Tenenbaum, Armando Solar-Lezama*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mCtadqIxOJ](https://openreview.net/forum?id=mCtadqIxOJ)

**Abstract**:

Synthesizing programs from examples requires searching over a vast, combinatorial space of possible programs. In this search process, a key challenge is representing the behavior of a partially written program before it can be executed, to judge if it is on the right track and predict where to search next. We introduce a general technique for representing partially written programs in a program synthesis engine. We take inspiration from the technique of abstract interpretation, in which an approximate execution model is used to determine if an unfinished program will eventually satisfy a goal specification. Here we learn an approximate execution model implemented as a modular neural network. By constructing compositional program representations that implicitly encode the interpretation semantics of the underlying programming language, we can represent partial programs using a flexible combination of concrete execution state and learned neural representations, using the learned approximate semantics when concrete semantics are not known (in unfinished parts of the program). We show that these hybrid neuro-symbolic representations enable execution-guided synthesizers to use more powerful language constructs, such as loops and higher-order functions, and can be used to synthesize programs more accurately for a given search budget than pure neural approaches in several domains.

----

## [705] Fast convergence of stochastic subgradient method under interpolation

**Authors**: *Huang Fang, Zhenan Fan, Michael P. Friedlander*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=w2mYg3d0eot](https://openreview.net/forum?id=w2mYg3d0eot)

**Abstract**:

This paper studies the behaviour of the stochastic subgradient descent (SSGD) method applied to over-parameterized nonsmooth optimization problems that satisfy an interpolation condition. By leveraging the composite structure of the empirical risk minimization problems, we prove that SSGD converges, respectively, with rates $O(1/\epsilon)$ and $O(\log(1/\epsilon))$ for convex and strongly-convex objectives when interpolation holds. These rates coincide with established rates for the stochastic gradient descent (SGD) method applied to smooth problems that also satisfy an interpolation condition. Our analysis provides a partial explanation for the empirical observation that sometimes SGD and SSGD behave similarly for training smooth and nonsmooth machine learning models. We also prove that the rate $O(1/\epsilon)$ is optimal for the subgradient method in the convex and interpolation setting.

----

## [706] A Hypergradient Approach to Robust Regression without Correspondence

**Authors**: *Yujia Xie, Yixiu Mao, Simiao Zuo, Hongteng Xu, Xiaojing Ye, Tuo Zhao, Hongyuan Zha*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=l35SB-_raSQ](https://openreview.net/forum?id=l35SB-_raSQ)

**Abstract**:

We consider a regression problem, where the correspondence between the input and output data is not available. Such shuffled data are commonly observed in many real world problems. Take flow cytometry as an example: the measuring instruments are unable to preserve the correspondence between the samples and the measurements. Due to the combinatorial nature of the problem, most of the existing methods are only applicable when the sample size is small, and are limited to linear regression models. To overcome such bottlenecks, we propose a new computational framework --- ROBOT --- for the shuffled regression problem, which is applicable to large data and complex models. Specifically, we propose to formulate regression without correspondence as a continuous optimization problem. Then by exploiting the interaction between the regression model and the data correspondence, we propose to develop a hypergradient approach based on differentiable programming techniques. Such a hypergradient approach essentially views the data correspondence as an operator of the regression model, and therefore it allows us to find a better descent direction for the model parameters by differentiating through the data correspondence. ROBOT is quite general, and can be further extended to an inexact correspondence setting, where the input and output data are not necessarily exactly aligned. Thorough numerical experiments show that ROBOT achieves better performance than existing methods in both linear and nonlinear regression tasks, including real-world applications such as flow cytometry and multi-object tracking.

----

## [707] On the role of planning in model-based deep reinforcement learning

**Authors**: *Jessica B. Hamrick, Abram L. Friesen, Feryal M. P. Behbahani, Arthur Guez, Fabio Viola, Sims Witherspoon, Thomas Anthony, Lars Holger Buesing, Petar Velickovic, Theophane Weber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IrM64DGB21](https://openreview.net/forum?id=IrM64DGB21)

**Abstract**:

Model-based planning is often thought to be necessary for deep, careful reasoning and generalization in artificial agents. While recent successes of model-based reinforcement learning (MBRL) with deep function approximation have strengthened this hypothesis, the resulting diversity of model-based methods has also made it difficult to track which components drive success and why. In this paper, we seek to disentangle the contributions of recent methods by focusing on three questions: (1) How does planning benefit MBRL agents? (2) Within planning, what choices drive performance? (3) To what extent does planning improve generalization? To answer these questions, we study the performance of MuZero (Schrittwieser et al., 2019), a state-of-the-art MBRL algorithm with strong connections and overlapping components with many other MBRL algorithms. We perform a number of interventions and ablations of MuZero across a wide range of environments, including control tasks, Atari, and 9x9 Go. Our results suggest the following: (1) Planning is most useful in the learning process, both for policy updates and for providing a more useful data distribution. (2) Using shallow trees with simple Monte-Carlo rollouts is as performant as more complex methods, except in the most difficult reasoning tasks. (3) Planning alone is insufficient to drive strong generalization. These results indicate where and how to utilize planning in reinforcement learning settings, and highlight a number of open questions for future MBRL research.

----

## [708] Trajectory Prediction using Equivariant Continuous Convolution

**Authors**: *Robin Walters, Jinxi Li, Rose Yu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=J8_GttYLFgr](https://openreview.net/forum?id=J8_GttYLFgr)

**Abstract**:

Trajectory prediction is a critical part of many AI applications, for example, the safe operation of autonomous vehicles. However, current methods are prone to making inconsistent and physically unrealistic predictions. We leverage insights from  fluid dynamics to overcome this limitation by considering internal symmetry in real-world trajectories. We propose a novel model, Equivariant Continous COnvolution (ECCO) for improved trajectory prediction.  ECCO uses rotationally-equivariant continuous convolutions to embed the symmetries of the system. On both vehicle and pedestrian trajectory datasets, ECCO attains competitive accuracy  with significantly fewer parameters. It is also more sample efficient, generalizing automatically from few data points in any orientation.  Lastly, ECCO improves generalization with equivariance, resulting in more physically consistent predictions.   Our method provides a fresh perspective towards increasing trust and transparency in deep learning models. Our code and data can be found at https://github.com/Rose-STL-Lab/ECCO.

----

## [709] Grounding Language to Autonomously-Acquired Skills via Goal Generation

**Authors**: *Ahmed Akakzia, Cédric Colas, Pierre-Yves Oudeyer, Mohamed Chetouani, Olivier Sigaud*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=chPj_I5KMHG](https://openreview.net/forum?id=chPj_I5KMHG)

**Abstract**:

We are interested in the autonomous acquisition of repertoires of skills. Language-conditioned reinforcement learning (LC-RL) approaches are great tools in this quest, as they allow to express abstract goals as sets of constraints on the states. However, most LC-RL agents are not autonomous and cannot learn without external instructions and feedback. Besides, their direct language condition cannot account for the goal-directed behavior of pre-verbal infants and strongly limits the expression of behavioral diversity for a given language input. To resolve these issues, we propose a new conceptual approach to language-conditioned RL: the Language-Goal-Behavior architecture (LGB). LGB decouples skill learning and language grounding via an intermediate semantic representation of the world. To showcase the properties of LGB, we present a specific implementation called DECSTR. DECSTR is an intrinsically motivated learning agent endowed with an innate semantic representation describing spatial relations between physical objects. In a first stage G -> B, it freely explores its environment and targets self-generated semantic configurations. In a second stage (L -> G), it trains a language-conditioned  goal generator to generate semantic goals that match the constraints expressed in language-based inputs. We showcase the additional properties of LGB w.r.t. both an end-to-end LC-RL approach and a similar approach leveraging non-semantic, continuous intermediate representations. Intermediate semantic representations help satisfy language commands in a diversity of ways, enable strategy switching after a failure and facilitate language grounding.

----

## [710] Chaos of Learning Beyond Zero-sum and Coordination via Game Decompositions

**Authors**: *Yun Kuen Cheung, Yixin Tao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=a3wKPZpGtCF](https://openreview.net/forum?id=a3wKPZpGtCF)

**Abstract**:

It is of primary interest for ML to understand how agents learn and interact dynamically in competitive environments and games (e.g. GANs). But this has been a difficult task, as irregular behaviors are commonly observed in such systems. This can be explained theoretically, for instance, by the works of Cheung and Piliouras (COLT 2019; NeurIPS 2020), which showed that in two-person zero-sum games, if agents employ one of the most well-known learning algorithms, Multiplicative Weights Update (MWU), then Lyapunov chaos occurs everywhere in the payoff space. In this paper, we study how persistent chaos can occur in the more general normal game settings, where the agents might have the motivation to coordinate (which is not true for zero-sum games) and the number of agents can be arbitrary.

We characterize bimatrix games where MWU, its optimistic variant (OMWU) or Follow-the-Regularized-Leader (FTRL) algorithms are Lyapunov chaotic almost everywhere in the payoff space. Technically, our characterization is derived by extending the volume-expansion argument of Cheung and Piliouras via the canonical game decomposition into zero-sum and coordination components. Interestingly, the two components induce opposite volume-changing behaviors, so the overall behavior can be analyzed by comparing the strengths of the components against each other. The comparison is done via our new notion of "matrix domination" or via a linear program. For multi-player games, we present a local equivalence of volume change between general games and graphical games, which is used to perform volume and chaos analyses of MWU and OMWU in potential games.

----

## [711] Isometric Transformation Invariant and Equivariant Graph Convolutional Networks

**Authors**: *Masanobu Horie, Naoki Morita, Toshiaki Hishinuma, Yu Ihara, Naoto Mitsume*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=FX0vR39SJ5q](https://openreview.net/forum?id=FX0vR39SJ5q)

**Abstract**:

Graphs are one of the most important data structures for representing pairwise relations between objects. Specifically, a graph embedded in a Euclidean space is essential to solving real problems, such as physical simulations. A crucial requirement for applying graphs in Euclidean spaces to physical simulations is learning and inferring the isometric transformation invariant and equivariant features in a computationally efficient manner. In this paper, we propose a set of transformation invariant and equivariant models based on graph convolutional networks, called IsoGCNs. We demonstrate that the proposed model has a competitive performance compared to state-of-the-art methods on tasks related to geometrical and physical simulation data. Moreover, the proposed model can scale up to graphs with 1M vertices and conduct an inference faster than a conventional finite element analysis, which the existing equivariant models cannot achieve.

----

## [712] R-GAP: Recursive Gradient Attack on Privacy

**Authors**: *Junyi Zhu, Matthew B. Blaschko*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RSU17UoKfJF](https://openreview.net/forum?id=RSU17UoKfJF)

**Abstract**:

Federated learning frameworks have been regarded as a promising approach to break the dilemma between demands on privacy and the promise of learning from large collections of distributed data. Many such frameworks only ask collaborators to share their local update of a common model, i.e. gradients with respect to locally stored data, instead of exposing their raw data to other collaborators. However, recent optimization-based gradient attacks show that raw data can often be accurately recovered from gradients. It has been shown that minimizing the Euclidean distance between true gradients and those calculated from estimated data is often effective in fully recovering private data. However, there is a fundamental lack of theoretical understanding of how and when gradients can lead to unique recovery of original data. Our research fills this gap by providing a closed-form recursive procedure to recover data from gradients in deep neural networks. We name it Recursive Gradient Attack on Privacy (R-GAP). Experimental results demonstrate that R-GAP  works as well as or even better than optimization-based approaches at a fraction of the computation under certain conditions. Additionally, we propose a Rank Analysis method, which can be used to estimate the risk of gradient attacks inherent in certain network architectures, regardless of whether an optimization-based or closed-form-recursive attack is used. Experimental results demonstrate the utility of the rank analysis towards improving the network's security. Source code is available for download from https://github.com/JunyiZhu-AI/R-GAP.

----

## [713] Multi-Level Local SGD: Distributed SGD for Heterogeneous Hierarchical Networks

**Authors**: *Timothy Castiglia, Anirban Das, Stacy Patterson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=C70cp4Cn32](https://openreview.net/forum?id=C70cp4Cn32)

**Abstract**:

We propose Multi-Level Local SGD, a distributed stochastic gradient method for learning a smooth, non-convex objective in a multi-level communication network with heterogeneous workers. Our network model consists of a set of disjoint sub-networks, with a single hub and multiple workers; further, workers may have different operating rates. The hubs exchange information with one another via a connected, but not necessarily complete communication network. In our algorithm, sub-networks execute a distributed SGD algorithm, using a hub-and-spoke paradigm, and the hubs periodically average their models with neighboring hubs. We first provide a unified mathematical framework that describes the Multi-Level Local SGD algorithm. We then present a theoretical analysis of the algorithm; our analysis shows the dependence of the convergence error on the worker node heterogeneity, hub network topology, and the number of local, sub-network, and global iterations. We illustrate the effectiveness of our algorithm in a multi-level network with slow workers via simulation-based experiments.

----

## [714] GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

**Authors**: *Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qrwe7XHTmYb](https://openreview.net/forum?id=qrwe7XHTmYb)

**Abstract**:

Neural network scaling has been critical for improving the model quality in many real-world machine learning applications with vast amounts of training data and compute. Although this trend of scaling is affirmed to be a sure-fire approach for better model quality, there are challenges on the path such as the computation cost,ease of programming, and efficient implementation on parallel devices.  In this paper we demonstrate conditional computation as a remedy to the above mentioned impediments, and demonstrate its efficacy and utility.  We make extensive use of GShard, a module composed of a set of lightweight annotation APIs and an extension to the XLA compiler to enable large scale models with up to trillions of parameters. GShard and conditional computation enable us to scale up multilingual neural machine translation Transformer model with Sparsely-Gated Mixture-of-Experts. We demonstrate that such a giant model with 600 billion parameters can efficiently be trained on 2048 TPU v3 cores in 4 days to achieve far superior quality for translation from 100 languages to English compared to the prior art.

----

## [715] Representation learning for improved interpretability and classification accuracy of clinical factors from EEG

**Authors**: *Garrett Honke, Irina Higgins, Nina Thigpen, Vladimir Miskovic, Katie Link, Sunny Duan, Pramod Gupta, Julia Klawohn, Greg Hajcak*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TVjLza1t4hI](https://openreview.net/forum?id=TVjLza1t4hI)

**Abstract**:

Despite extensive standardization, diagnostic interviews for mental health disorders encompass substantial subjective judgment. Previous studies have demonstrated that EEG-based neural measures can function as reliable objective correlates of depression, or even predictors of depression and its course. However, their clinical utility has not been fully realized because of 1) the lack of automated ways to deal with the inherent noise associated with EEG data at scale, and 2) the lack of knowledge of which aspects of the EEG signal may be markers of a clinical disorder. Here we adapt an unsupervised pipeline from the recent deep representation learning literature to address these problems by 1) learning a disentangled representation using $\beta$-VAE to denoise the signal, and 2) extracting interpretable features associated with a sparse set of clinical labels using a Symbol-Concept Association Network (SCAN). We demonstrate that our method is able to outperform the canonical hand-engineered baseline classification method on a number of factors, including participant age and depression diagnosis. Furthermore, our method recovers a representation that can be used to automatically extract denoised Event Related Potentials (ERPs) from novel, single EEG trajectories, and supports fast supervised re-mapping to various clinical labels, allowing clinicians to re-use a single EEG representation regardless of updates to the standardized diagnostic system. Finally, single factors of the learned disentangled representations often correspond to meaningful markers of clinical factors, as automatically detected by SCAN, allowing for human interpretability and post-hoc expert analysis of the recommendations made by the model.

----

## [716] Multiplicative Filter Networks

**Authors**: *Rizal Fathony, Anit Kumar Sahu, Devin Willmott, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OmtmcPkkhT](https://openreview.net/forum?id=OmtmcPkkhT)

**Abstract**:

Although deep networks are typically used to approximate functions over high dimensional inputs, recent work has increased interest in neural networks as function approximators for low-dimensional-but-complex functions, such as representing images as a function of pixel coordinates, solving differential equations, or representing signed distance fields or neural radiance fields.  Key to these recent successes has been the use of new elements such as sinusoidal nonlinearities, or Fourier features in positional encodings, which vastly outperform simple ReLU networks.  In this paper, we propose and empirically demonstrate that an arguably simpler class of function approximators can work just as well for such problems: multiplicative filter networks.  In these networks, we avoid traditional compositional depth altogether, and simply multiply together (linear functions of) sinusoidal or Gabor wavelet functions applied to the input.  This representation has the notable advantage that the entire function can simply be viewed as a linear function approximator over an exponential number of Fourier or Gabor basis functions, respectively.  Despite this simplicity, when compared to recent approaches that use Fourier features with ReLU networks or sinusoidal activation networks, we show that these multiplicative filter networks largely outperform or match the performance of these recent approaches on the domains highlighted in these past works.

----

## [717] Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks

**Authors**: *Róbert Csordás, Sjoerd van Steenkiste, Jürgen Schmidhuber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7uVcpu-gMD](https://openreview.net/forum?id=7uVcpu-gMD)

**Abstract**:

Neural networks (NNs) whose subnetworks implement reusable functions are expected to offer numerous advantages, including compositionality through efficient recombination of functional building blocks, interpretability, preventing catastrophic interference, etc. Understanding if and how NNs are modular could provide insights into how to improve them. Current inspection methods, however, fail to link modules to their functionality. In this paper, we present a novel method based on learning binary weight masks to identify individual weights and subnets responsible for specific functions. Using this powerful tool, we contribute an extensive study of emerging modularity in NNs that covers several standard architectures and datasets. We demonstrate how common NNs fail to reuse submodules and offer new insights into the related issue of systematic generalization on language tasks.

----

## [718] Modeling the Second Player in Distributionally Robust Optimization

**Authors**: *Paul Michel, Tatsunori Hashimoto, Graham Neubig*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZDnzZrTqU9N](https://openreview.net/forum?id=ZDnzZrTqU9N)

**Abstract**:

Distributionally robust optimization (DRO) provides a framework for training machine learning models that are able to perform well on a collection of related data distributions (the "uncertainty set"). This is done by solving a min-max game: the model is trained to minimize its maximum expected loss among all distributions in the uncertainty set. While careful design of the uncertainty set is critical to the success of the DRO procedure, previous work has been limited to relatively simple alternatives that keep the min-max optimization problem exactly tractable, such as $f$-divergence balls. In this paper, we argue instead for the use of neural generative models to characterize the worst-case distribution, allowing for more flexible and problem-specific selection of the uncertainty set. However, while simple conceptually, this approach poses a number of implementation and optimization challenges. To circumvent these issues, we propose a relaxation of the KL-constrained inner maximization objective that makes the DRO problem more amenable to gradient-based optimization of large scale generative models, and develop model selection heuristics to guide hyper-parameter search. On both toy settings and realistic NLP tasks, we find that the proposed approach yields models that are more robust than comparable baselines.

----

## [719] Private Post-GAN Boosting

**Authors**: *Marcel Neunhoeffer, Steven Wu, Cynthia Dwork*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6isfR3JCbi](https://openreview.net/forum?id=6isfR3JCbi)

**Abstract**:

Differentially private GANs have proven to be a promising approach for generating realistic synthetic data without compromising the privacy of individuals. Due to the privacy-protective noise introduced in the  training, the convergence of GANs becomes even more elusive, which often leads to poor utility in the output generator at the end of training. We propose Private post-GAN boosting (Private PGB), a differentially private method that combines samples produced by the sequence of generators obtained during GAN training to create a high-quality synthetic dataset. To that end, our method leverages the Private Multiplicative Weights method (Hardt and Rothblum, 2010) to reweight generated samples. We evaluate Private PGB on two dimensional toy data, MNIST images, US Census data and a standard machine learning prediction task. Our experiments show that Private PGB improves upon a standard private GAN approach across a collection of quality measures. We also provide a non-private variant of PGB that improves the data quality of standard GAN training.

----

## [720] Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis

**Authors**: *Rafael Valle, Kevin J. Shih, Ryan Prenger, Bryan Catanzaro*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ig53hpHxS4](https://openreview.net/forum?id=Ig53hpHxS4)

**Abstract**:

In this paper we propose Flowtron: an autoregressive flow-based generative network for text-to-speech synthesis with style transfer and speech variation. Flowtron borrows insights from Autoregressive Flows and revamps Tacotron 2 in order to provide high-quality and expressive mel-spectrogram synthesis. Flowtron is optimized by maximizing the likelihood of the training data, which makes training simple and stable. Flowtron learns an invertible mapping of data to a latent space that can be used to modulate many aspects of speech synthesis (timbre, expressivity, accent). Our mean opinion scores (MOS) show that Flowtron matches state-of-the-art TTS models in terms of speech quality. We provide results on speech variation, interpolation over time between samples and style transfer between seen and unseen speakers. Code and pre-trained models are publicly available at \href{https://github.com/NVIDIA/flowtron}{https://github.com/NVIDIA/flowtron}.

----

## [721] Learning Structural Edits via Incremental Tree Transformations

**Authors**: *Ziyu Yao, Frank F. Xu, Pengcheng Yin, Huan Sun, Graham Neubig*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=v9hAX77--cZ](https://openreview.net/forum?id=v9hAX77--cZ)

**Abstract**:

While most neural generative models generate outputs in a single pass, the human creative process is usually one of iterative building and refinement. Recent work has proposed models of editing processes, but these mostly focus on editing sequential data and/or only model a single editing pass. In this paper, we present a generic model for incremental editing of structured data (i.e. ''structural edits''). Particularly, we focus on tree-structured data, taking abstract syntax trees of computer programs as our canonical example. Our editor learns to iteratively generate tree edits (e.g. deleting or adding a subtree) and applies them to the partially edited data, thereby the entire editing process can be formulated as consecutive, incremental tree transformations. To show the unique benefits of modeling tree edits directly, we further propose a novel edit encoder for learning to represent edits, as well as an imitation learning method that allows the editor to be more robust. We evaluate our proposed editor on two source code edit datasets, where results show that, with the proposed edit encoder, our editor significantly improves accuracy over previous approaches that generate the edited program directly in one pass. Finally, we demonstrate that training our editor to imitate experts and correct its mistakes dynamically can further improve its performance.

----

## [722] Sample-Efficient Automated Deep Reinforcement Learning

**Authors**: *Jörg K. H. Franke, Gregor Köhler, André Biedenkapp, Frank Hutter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hSjxQ3B7GWq](https://openreview.net/forum?id=hSjxQ3B7GWq)

**Abstract**:

Despite significant progress in challenging problems across various domains, applying state-of-the-art deep reinforcement learning (RL) algorithms remains challenging due to their sensitivity to the choice of hyperparameters. This sensitivity can partly be attributed to the non-stationarity of the RL problem, potentially requiring different hyperparameter settings at various stages of the learning process. Additionally, in the RL setting, hyperparameter optimization (HPO) requires a large number of environment interactions, hindering the transfer of the successes in RL to real-world applications. In this work, we tackle the issues of sample-efficient and dynamic HPO in RL. We propose a population-based automated RL (AutoRL) framework to meta-optimize arbitrary off-policy RL algorithms. In this framework, we optimize the hyperparameters and also the neural architecture while simultaneously training the agent. By sharing the collected experience across the population, we substantially increase the sample efficiency of the meta-optimization. We demonstrate the capabilities of our sample-efficient AutoRL approach in a case study with the popular TD3 algorithm in the MuJoCo benchmark suite, where we reduce the number of environment interactions needed for meta-optimization by up to an order of magnitude compared to population-based training.

----

## [723] Unsupervised Discovery of 3D Physical Objects from Video

**Authors**: *Yilun Du, Kevin A. Smith, Tomer D. Ullman, Joshua B. Tenenbaum, Jiajun Wu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lf7st0bJIA5](https://openreview.net/forum?id=lf7st0bJIA5)

**Abstract**:

We study the problem of unsupervised physical object discovery. While existing frameworks aim to decompose scenes into 2D segments based off each object's appearance, we explore how physics, especially object interactions, facilitates disentangling of 3D geometry and position of objects from video, in an unsupervised manner. Drawing inspiration from developmental psychology, our Physical Object Discovery Network (POD-Net) uses both multi-scale pixel cues and physical motion cues to accurately segment observable and partially occluded objects of varying sizes, and infer properties of those objects. Our model reliably segments objects on both synthetic and real scenes.  The discovered object properties can also be used to reason about physical events.

----

## [724] Global optimality of softmax policy gradient with single hidden layer neural networks in the mean-field regime

**Authors**: *Andrea Agazzi, Jianfeng Lu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bB2drc7DPuB](https://openreview.net/forum?id=bB2drc7DPuB)

**Abstract**:

We study the problem of policy optimization for infinite-horizon discounted Markov Decision Processes with softmax policy and nonlinear function approximation trained with policy gradient algorithms. We concentrate on the training dynamics in the mean-field regime, modeling e.g. the behavior of wide single hidden layer neural networks, when exploration is encouraged through entropy regularization. The dynamics of these models is established as a Wasserstein gradient flow of distributions in parameter space.  We further prove global optimality of the fixed points of this dynamics  under mild conditions on their initialization.

----

## [725] Extracting Strong Policies for Robotics Tasks from Zero-Order Trajectory Optimizers

**Authors**: *Cristina Pinneri, Shambhuraj Sawant, Sebastian Blaes, Georg Martius*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Nc3TJqbcl3](https://openreview.net/forum?id=Nc3TJqbcl3)

**Abstract**:

Solving high-dimensional, continuous robotic tasks is a challenging optimization problem. Model-based methods that rely on zero-order optimizers like the cross-entropy method (CEM) have so far shown strong performance and are considered state-of-the-art in the model-based reinforcement learning community. However, this success comes at the cost of high computational complexity, being therefore not suitable for real-time control. In this paper, we propose a technique to jointly optimize the trajectory and distill a policy, which is essential for fast execution in real robotic systems. Our method builds upon standard approaches, like guidance cost and dataset aggregation, and introduces a novel adaptive factor which prevents the optimizer from collapsing to the learner's behavior at the beginning of the training. The extracted policies reach unprecedented performance on challenging tasks as making a humanoid stand up and opening a door without reward shaping

----

## [726] Temporally-Extended ε-Greedy Exploration

**Authors**: *Will Dabney, Georg Ostrovski, André Barreto*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ONBPHFZ7zG4](https://openreview.net/forum?id=ONBPHFZ7zG4)

**Abstract**:

Recent work on exploration in reinforcement learning (RL) has led to a series of increasingly complex solutions to the problem. This increase in complexity often comes at the expense of generality. Recent empirical studies suggest that, when applied to a broader set of domains, some sophisticated exploration methods are outperformed by simpler counterparts, such as ε-greedy. In this paper we propose an exploration algorithm that retains the simplicity of ε-greedy while reducing dithering. We build on a simple hypothesis: the main limitation of ε-greedy exploration is its lack of temporal persistence, which limits its ability to escape local optima. We propose a temporally extended form of ε-greedy that simply repeats the sampled action for a random duration. It turns out that, for many duration distributions, this suffices to improve exploration on a large set of domains. Interestingly, a class of distributions inspired by ecological models of animal foraging behaviour yields particularly strong performance.

----

## [727] Rapid Task-Solving in Novel Environments

**Authors**: *Samuel Ritter, Ryan Faulkner, Laurent Sartran, Adam Santoro, Matthew M. Botvinick, David Raposo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=F-mvpFpn_0q](https://openreview.net/forum?id=F-mvpFpn_0q)

**Abstract**:

We propose the challenge of rapid task-solving in novel environments (RTS), wherein an agent must solve a series of tasks as rapidly as possible in an unfamiliar environment. An effective RTS agent must balance between exploring the unfamiliar environment and solving its current task, all while building a model of the new environment over which it can plan when faced with later tasks. While modern deep RL agents exhibit some of these abilities in isolation, none are suitable for the full RTS challenge. To enable progress toward RTS, we introduce two challenge domains: (1) a minimal RTS challenge called the Memory&Planning Game and (2) One-Shot StreetLearn Navigation, which introduces scale and complexity from real-world data. We demonstrate that state-of-the-art deep RL agents fail at RTS in both domains, and that this failure is due to an inability to plan over gathered knowledge. We develop Episodic Planning Networks (EPNs) and show that deep-RL agents with EPNs excel at RTS, outperforming the nearest baseline by factors of 2-3 and learning to navigate held-out StreetLearn maps within a single episode. We show that EPNs learn to execute a value iteration-like planning algorithm and that they generalize to situations beyond their training experience.

----

## [728] Tradeoffs in Data Augmentation: An Empirical Study

**Authors**: *Raphael Gontijo Lopes, Sylvia J. Smullin, Ekin Dogus Cubuk, Ethan Dyer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZcKPWuhG6wy](https://openreview.net/forum?id=ZcKPWuhG6wy)

**Abstract**:

Though data augmentation has become a standard component of deep neural network training, the underlying mechanism behind the effectiveness of these techniques remains poorly understood. In practice, augmentation policies are often chosen using heuristics of distribution shift or augmentation diversity. Inspired by these, we conduct an empirical study to quantify how data augmentation improves model generalization. We introduce two interpretable and easy-to-compute measures: Affinity and Diversity. We find that augmentation performance is predicted not by either of these alone but by jointly optimizing the two.

----

## [729] Multiscale Score Matching for Out-of-Distribution Detection

**Authors**: *Ahsan Mahmood, Junier Oliva, Martin Andreas Styner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xoHdgbQJohv](https://openreview.net/forum?id=xoHdgbQJohv)

**Abstract**:

We present a new methodology for detecting out-of-distribution (OOD) images by utilizing norms of the score estimates at multiple noise scales. A score is defined to be the gradient of the log density with respect to the input data. Our methodology is completely unsupervised and follows a straight forward training scheme. First, we train a deep network to estimate scores for $L$ levels of noise. Once trained, we calculate the noisy score estimates for $N$ in-distribution samples and take the L2-norms across the input dimensions (resulting in an $N$x$L$ matrix). Then we train an auxiliary model (such as a Gaussian Mixture Model) to learn the in-distribution spatial regions in this $L$-dimensional space. This auxiliary model can now be used to identify points that reside outside the learned space. Despite its simplicity, our experiments show that this methodology significantly outperforms the state-of-the-art in detecting out-of-distribution images. For example, our method can effectively separate CIFAR-10 (inlier) and SVHN (OOD) images, a setting which has been previously shown to be difficult for deep likelihood models.

----

## [730] Understanding Over-parameterization in Generative Adversarial Networks

**Authors**: *Yogesh Balaji, Mohammadmahdi Sajedi, Neha Mukund Kalibhat, Mucong Ding, Dominik Stöger, Mahdi Soltanolkotabi, Soheil Feizi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=C3qvk5IQIJY](https://openreview.net/forum?id=C3qvk5IQIJY)

**Abstract**:

A broad class of unsupervised deep learning methods such as Generative Adversarial Networks (GANs) involve training of overparameterized models where the number of parameters of the model exceeds a certain threshold. Indeed, most successful GANs used in practice are trained using overparameterized generator and discriminator networks, both in terms of depth and width. A large body of work in supervised learning have shown the importance of model overparameterization in the convergence of the gradient descent (GD) to globally optimal solutions. In contrast, the unsupervised setting and GANs in particular involve non-convex concave mini-max optimization problems that are often trained using Gradient Descent/Ascent (GDA).
The role and benefits of model overparameterization in the convergence of GDA to a global saddle point in non-convex concave problems is far less understood. In this work, we present a comprehensive analysis of the importance of model overparameterization in GANs both theoretically and empirically. We theoretically show that in an overparameterized GAN model with a $1$-layer neural network generator and a linear discriminator, GDA converges to a global saddle point of the underlying non-convex concave min-max problem. To the best of our knowledge, this is the first result for global convergence of GDA in such settings. Our theory is based on a more general result that holds for a broader class of nonlinear generators and discriminators that obey certain assumptions (including deeper generators and random feature discriminators). Our theory utilizes and builds upon a novel connection with the convergence analysis of linear time-varying dynamical systems which may have broader implications for understanding the convergence behavior of GDA for non-convex concave problems involving overparameterized models. We also empirically study the role of model overparameterization in GANs using several large-scale experiments on CIFAR-10 and Celeb-A datasets. Our experiments show that overparameterization improves the quality of generated samples across various model architectures and datasets. Remarkably, we observe that overparameterization leads to faster and more stable convergence behavior of GDA across the board.

----

## [731] Go with the flow: Adaptive control for Neural ODEs

**Authors**: *Mathieu Chalvidal, Matthew Ricci, Rufin VanRullen, Thomas Serre*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=giit4HdDNa](https://openreview.net/forum?id=giit4HdDNa)

**Abstract**:

Despite their elegant formulation and lightweight memory cost, neural ordinary differential equations (NODEs) suffer from known representational limitations. In particular, the single flow learned by NODEs cannot express all homeomorphisms from a given data space to itself, and their static weight parameterization restricts the type of functions they can learn compared to discrete architectures with layer-dependent weights. Here, we describe a new module called neurally-controlled ODE (N-CODE) designed to improve the expressivity of NODEs. The parameters of N-CODE modules are dynamic variables governed by a trainable map from initial or current activation state, resulting in forms of open-loop and closed-loop control, respectively. A single module is sufficient for learning a distribution on non-autonomous flows that adaptively drive neural representations. We provide theoretical and empirical evidence that N-CODE circumvents limitations of previous NODEs models and show how increased model expressivity manifests in several supervised and unsupervised learning problems. These favorable empirical results indicate the potential of using data- and activity-dependent plasticity in neural networks across numerous domains.

----

## [732] Linear Last-iterate Convergence in Constrained Saddle-point Optimization

**Authors**: *Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, Haipeng Luo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dx11_7vm5_r](https://openreview.net/forum?id=dx11_7vm5_r)

**Abstract**:

Optimistic Gradient Descent Ascent (OGDA) and Optimistic Multiplicative Weights Update (OMWU) for saddle-point optimization have received growing attention due to their favorable last-iterate convergence. However, their behaviors for simple bilinear games over the probability simplex are still not fully understood --- previous analysis lacks explicit convergence rates, only applies to an exponentially small learning rate, or requires additional assumptions such as the uniqueness of the optimal solution.

In this work, we significantly expand the understanding of last-iterate convergence for OGDA and OMWU in the constrained setting. Specifically, for OMWU in bilinear games over the simplex, we show that when the equilibrium is unique, linear last-iterate convergence is achievable with a constant learning rate, which improves the result of (Daskalakis & Panageas, 2019) under the same assumption. We then significantly extend the results to more general objectives and feasible sets for the projected OGDA algorithm, by introducing a sufficient condition under which OGDA exhibits concrete last-iterate convergence rates with a constant learning rate. We show that bilinear games over any polytope satisfy this condition and OGDA converges exponentially fast even without the unique equilibrium assumption. Our condition also holds for strongly-convex-strongly-concave functions, recovering the result of (Hsieh et al., 2019). Finally, we provide experimental results to further support our theory.

----

## [733] Learning advanced mathematical computations from examples

**Authors**: *François Charton, Amaury Hayat, Guillaume Lample*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-gfhS00XfKj](https://openreview.net/forum?id=-gfhS00XfKj)

**Abstract**:

Using transformers over large generated datasets, we train models to learn mathematical properties of differential systems, such as local stability, behavior at infinity and controllability. We achieve near perfect prediction of qualitative characteristics, and good approximations of numerical features of the system. This demonstrates that neural networks can learn to perform complex computations, grounded in advanced theory, from examples, without built-in mathematical knowledge.

----

## [734] WaveGrad: Estimating Gradients for Waveform Generation

**Authors**: *Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NsMLjcFaO8O](https://openreview.net/forum?id=NsMLjcFaO8O)

**Abstract**:

This paper introduces WaveGrad, a conditional model for waveform generation which estimates gradients of the data density. The model is built on prior work on score matching and diffusion probabilistic models. It starts from a Gaussian white noise signal and iteratively refines the signal via a gradient-based sampler conditioned on the mel-spectrogram.
WaveGrad offers a natural way to trade inference speed for sample quality by adjusting the number of refinement steps, and bridges the gap between non-autoregressive and autoregressive models in terms of audio quality.
We find that it can generate high fidelity audio samples using as few as six iterations.
Experiments reveal WaveGrad to generate high fidelity audio, outperforming adversarial non-autoregressive baselines and matching a strong likelihood-based autoregressive baseline using fewer sequential operations.  Audio samples are available at https://wavegrad.github.io/.

----

## [735] SALD: Sign Agnostic Learning with Derivatives

**Authors**: *Matan Atzmon, Yaron Lipman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7EDgLu9reQD](https://openreview.net/forum?id=7EDgLu9reQD)

**Abstract**:

Learning 3D geometry directly from raw data, such as point clouds, triangle soups, or unoriented meshes is still a challenging task that feeds many downstream computer vision and graphics applications. 

In this paper, we introduce SALD: a method for learning implicit neural representations of shapes directly from raw data. We generalize sign agnostic learning (SAL) to include derivatives: given an unsigned distance function to the input raw data, we advocate a novel sign agnostic regression loss, incorporating both pointwise values and gradients of the unsigned distance function. Optimizing this loss leads to a signed implicit function solution, the zero level set of which is a high quality and valid manifold approximation to the input 3D data. The motivation behind SALD is that incorporating derivatives in a regression loss leads to a lower sample complexity, and consequently better fitting. In addition, we provide empirical evidence, as well as theoretical motivation in 2D that SAL enjoys a minimal surface property, favoring minimal area solutions. More importantly, we are able to show that this property still holds for SALD, i.e.,  with derivatives included.

We demonstrate the efficacy of SALD for shape space learning on two challenging datasets: ShapeNet that contains inconsistent orientation and non-manifold meshes, and D-Faust that contains raw 3D scans (triangle soups). On both these datasets, we present state-of-the-art results.

----

## [736] Generalized Energy Based Models

**Authors**: *Michael Arbel, Liang Zhou, Arthur Gretton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0PtUPB9z6qK](https://openreview.net/forum?id=0PtUPB9z6qK)

**Abstract**:

We introduce the Generalized Energy Based Model (GEBM) for generative modelling. These models combine two  trained components: a base distribution (generally an implicit model), which can learn the support of data with low intrinsic dimension in a high dimensional space; and an energy function, to refine the probability mass on the learned support. 
Both the energy function and base jointly constitute the final model, unlike GANs, which retain only the base distribution (the "generator").  
GEBMs are trained by alternating between learning the energy and the base. 
We show that both training stages are well-defined: the energy is learned by maximising a generalized likelihood, and the resulting energy-based loss provides informative gradients for learning the base.
Samples from the posterior on the latent space of the trained model can be obtained via MCMC, thus finding regions in this space that produce better quality samples.
Empirically, the GEBM samples on image-generation tasks are of much better quality than those from the learned generator alone, indicating that all else being equal, the GEBM will outperform a GAN of the same complexity. When using normalizing flows as base measures, GEBMs succeed on density modelling tasks returning comparable performance to direct maximum likelihood of the same networks.

----

## [737] Long Range Arena : A Benchmark for Efficient Transformers

**Authors**: *Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qVyeW-grC2k](https://openreview.net/forum?id=qVyeW-grC2k)

**Abstract**:

Transformers do not scale very well to long sequence lengths largely because of quadratic self-attention complexity. In the recent months, a wide spectrum of efficient, fast Transformers have been proposed to tackle this problem, more often than not claiming superior or comparable model quality to vanilla Transformer models. To this date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide spectrum of tasks and datasets makes it difficult to assess relative model quality amongst many models. This paper proposes a systematic and unified benchmark, Long Range Arena, specifically focused on evaluating model quality under long-context scenarios. Our benchmark is a suite of tasks consisting of sequences ranging from $1K$ to $16K$ tokens, encompassing a wide range of data types and modalities such as text, natural, synthetic images, and mathematical expressions requiring similarity, structural, and visual-spatial reasoning. We systematically evaluate ten well-established long-range Transformer models (Reformers, Linformers, Linear Transformers, Sinkhorn Transformers, Performers, Synthesizers, Sparse Transformers, and Longformers) on our newly proposed benchmark suite. Long Range Arena paves the way towards better understanding this class of efficient Transformer models, facilitates more research in this direction, and presents new challenging tasks to tackle.

----

## [738] Beyond Categorical Label Representations for Image Classification

**Authors**: *Boyuan Chen, Yu Li, Sunand Raghupathi, Hod Lipson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MyHwDabUHZm](https://openreview.net/forum?id=MyHwDabUHZm)

**Abstract**:

We find that the way we choose to represent data labels can have a profound effect on the quality of trained models. For example, training an image classifier to regress audio labels rather than traditional categorical probabilities produces a more reliable classification. This result is surprising, considering that audio labels are more complex than simpler numerical probabilities or text. We hypothesize that high dimensional, high entropy label representations are generally more useful because they provide a stronger error signal. We support this hypothesis with evidence from various label representations including constant matrices, spectrograms, shuffled spectrograms, Gaussian mixtures, and uniform random matrices of various dimensionalities. Our experiments reveal that high dimensional, high entropy labels achieve comparable accuracy to text (categorical) labels on standard image classification tasks, but features learned through our label representations exhibit more robustness under various adversarial attacks and better effectiveness with a limited amount of training data. These results suggest that label representation may play a more important role than previously thought.

----

## [739] CoCo: Controllable Counterfactuals for Evaluating Dialogue State Trackers

**Authors**: *Shiyang Li, Semih Yavuz, Kazuma Hashimoto, Jia Li, Tong Niu, Nazneen Fatema Rajani, Xifeng Yan, Yingbo Zhou, Caiming Xiong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eom0IUrF__F](https://openreview.net/forum?id=eom0IUrF__F)

**Abstract**:

Dialogue state trackers have made significant progress on benchmark datasets, but their generalization capability to novel and realistic scenarios beyond the held- out conversations is less understood. We propose controllable counterfactuals (COCO) to bridge this gap and evaluate dialogue state tracking (DST) models on novel scenarios, i.e., would the system successfully tackle the request if the user responded differently but still consistently with the dialogue flow? COCO leverages turn-level belief states as counterfactual conditionals to produce novel conversation scenarios in two steps: (i) counterfactual goal generation at turn- level by dropping and adding slots followed by replacing slot values, (ii) counterfactual conversation generation that is conditioned on (i) and consistent with the dialogue flow. Evaluating state-of-the-art DST models on MultiWOZ dataset with COCO-generated counterfactuals results in a significant performance drop of up to 30.8% (from 49.4% to 18.6%) in absolute joint goal accuracy. In comparison, widely used techniques like paraphrasing only affect the accuracy by at most 2%. Human evaluations show that COCO-generated conversations perfectly reflect the underlying user goal with more than 95% accuracy and are as human-like as the original conversations, further strengthening its reliability and promise to be adopted as part of the robustness evaluation of DST models.

----

## [740] Stochastic Security: Adversarial Defense Using Long-Run Dynamics of Energy-Based Models

**Authors**: *Mitch Hill, Jonathan Craig Mitchell, Song-Chun Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gwFTuzxJW0](https://openreview.net/forum?id=gwFTuzxJW0)

**Abstract**:

The vulnerability of deep networks to adversarial attacks is a central problem for deep learning from the perspective of both cognition and security. The current most successful defense method is to train a classifier using adversarial images created during learning. Another defense approach involves transformation or purification of the original input to remove adversarial signals before the image is classified. We focus on defending naturally-trained classifiers using Markov Chain Monte Carlo (MCMC) sampling with an Energy-Based Model (EBM) for adversarial purification. In contrast to adversarial training, our approach is intended to secure highly vulnerable pre-existing classifiers. To our knowledge, no prior defensive transformation is capable of securing naturally-trained classifiers, and our method is the first to validate a post-training defense approach that is distinct from current successful defenses which modify classifier training.

The memoryless behavior of long-run MCMC sampling will eventually remove adversarial signals, while metastable behavior preserves consistent appearance of MCMC samples after many steps to allow accurate long-run prediction. Balancing these factors can lead to effective purification and robust classification. We evaluate adversarial defense with an EBM using the strongest known attacks against purification. Our contributions are 1) an improved method for training EBM's with realistic long-run MCMC samples for effective purification, 2) an Expectation-Over-Transformation (EOT) defense that resolves ambiguities for evaluating stochastic defenses and from which the EOT attack naturally follows, and 3) state-of-the-art adversarial defense for naturally-trained classifiers and competitive defense compared to adversarial training on CIFAR-10, SVHN, and CIFAR-100. Our code and pre-trained models are available at https://github.com/point0bar1/ebm-defense.

----

## [741] X2T: Training an X-to-Text Typing Interface with Online Learning from User Feedback

**Authors**: *Jensen Gao, Siddharth Reddy, Glen Berseth, Nicholas Hardy, Nikhilesh Natraj, Karunesh Ganguly, Anca D. Dragan, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LiX3ECzDPHZ](https://openreview.net/forum?id=LiX3ECzDPHZ)

**Abstract**:

We aim to help users communicate their intent to machines using flexible, adaptive interfaces that translate arbitrary user input into desired actions. In this work, we focus on assistive typing applications in which a user cannot operate a keyboard, but can instead supply other inputs, such as webcam images that capture eye gaze or neural activity measured by a brain implant. Standard methods train a model on a fixed dataset of user inputs, then deploy a static interface that does not learn from its mistakes; in part, because extracting an error signal from user behavior can be challenging. We investigate a simple idea that would enable such interfaces to improve over time, with minimal additional effort from the user: online learning from user feedback on the accuracy of the interface's actions. In the typing domain, we leverage backspaces as feedback that the interface did not perform the desired action. We propose an algorithm called x-to-text (X2T) that trains a predictive model of this feedback signal, and uses this model to fine-tune any existing, default interface for translating user input into actions that select words or characters. We evaluate X2T through a small-scale online user study with 12 participants who type sentences by gazing at their desired words, a large-scale observational study on handwriting samples from 60 users, and a pilot study with one participant using an electrocorticography-based brain-computer interface. The results show that X2T learns to outperform a non-adaptive default interface, stimulates user co-adaptation to the interface, personalizes the interface to individual users, and can leverage offline data collected from the default interface to improve its initial performance and accelerate online learning.

----

## [742] Mapping the Timescale Organization of Neural Language Models

**Authors**: *Hsiang-Yun Sherry Chien, Jinhan Zhang, Christopher J. Honey*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=J3OUycKwz-](https://openreview.net/forum?id=J3OUycKwz-)

**Abstract**:

In the human brain, sequences of language input are processed within a distributed and hierarchical architecture, in which higher stages of processing encode contextual information over longer timescales. In contrast, in recurrent neural networks which perform natural language processing, we know little about how the multiple timescales of contextual information are functionally organized. Therefore, we applied tools developed in neuroscience to map the “processing timescales” of individual units within a word-level LSTM language model. This timescale-mapping method assigned long timescales to units previously found to track long-range syntactic dependencies. Additionally, the mapping revealed a small subset of the network (less than 15% of units) with long timescales and whose function had not previously been explored. We next probed the functional organization of the network by examining the relationship between the processing timescale of units and their network connectivity. We identified two classes of long-timescale units: “controller” units composed a densely interconnected subnetwork and strongly projected to the rest of the network, while “integrator” units showed the longest timescales in the network, and expressed projection profiles closer to the mean projection profile. Ablating integrator and controller units affected model performance at different positions within a sentence, suggesting distinctive functions of these two sets of units. Finally, we tested the generalization of these results to a character-level LSTM model and models with different architectures. In summary, we demonstrated a model-free technique for mapping the timescale organization in recurrent neural networks, and we applied this method to reveal the timescale and functional organization of neural language models

----

## [743] PDE-Driven Spatiotemporal Disentanglement

**Authors**: *Jérémie Donà, Jean-Yves Franceschi, Sylvain Lamprier, Patrick Gallinari*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vLaHRtHvfFp](https://openreview.net/forum?id=vLaHRtHvfFp)

**Abstract**:

A recent line of work in the machine learning community addresses the problem of predicting high-dimensional spatiotemporal phenomena by leveraging specific tools from the differential equations theory. Following this direction, we propose in this article a novel and general paradigm for this task based on a resolution method for partial differential equations: the separation of variables. This inspiration allows us to introduce a dynamical interpretation of spatiotemporal disentanglement. It induces a principled model based on learning disentangled spatial and temporal representations of a phenomenon to accurately predict future observations. We experimentally demonstrate the performance and broad applicability of our method against prior state-of-the-art models on physical and synthetic video datasets.

----

## [744] OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning

**Authors**: *Anurag Ajay, Aviral Kumar, Pulkit Agrawal, Sergey Levine, Ofir Nachum*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=V69LGwJ0lIN](https://openreview.net/forum?id=V69LGwJ0lIN)

**Abstract**:

Reinforcement learning (RL) has achieved impressive performance in a variety of online settings in which an agent’s ability to query the environment for transitions and rewards is effectively unlimited. However, in many practical applications, the situation is reversed:  an agent may have access to large amounts of undirected offline experience data, while access to the online environment is severely limited. In this work, we focus on this offline setting. Our main insight is that, when presented with offline data composed of a variety of behaviors, an effective way to leverage this data is to extract a continuous space of recurring and temporally extended primitive behaviors before using these primitives for downstream task learning.  Primitives extracted in this way serve two purposes:  they delineate the behaviors that are supported by the data from those that are not, making them useful for avoiding distributional shift in offline RL; and they provide a degree of temporal abstraction, which reduces the effective horizon yielding better learning in theory, and improved offline RL in practice. In addition to benefiting offline policy optimization, we show that performing offline primitive learning in this way can also be leveraged for improving few-shot imitation learning as well as exploration and transfer in online RL on a variety of benchmark domains. Visualizations and code are available at https://sites.google.com/view/opal-iclr

----

## [745] Does enhanced shape bias improve neural network robustness to common corruptions?

**Authors**: *Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, Jan Hendrik Metzen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yUxUNaj2Sl](https://openreview.net/forum?id=yUxUNaj2Sl)

**Abstract**:

Convolutional neural networks (CNNs) learn to extract representations of complex features, such as object shapes and textures to solve image recognition tasks. Recent work indicates that CNNs trained on ImageNet are biased towards features that encode textures and that these alone are sufficient to generalize to unseen test data from the same distribution as the training data but often fail to generalize to out-of-distribution data. It has been shown that augmenting the training data with different image styles decreases this texture bias in favor of increased shape bias while at the same time improving robustness to common corruptions, such as noise and blur. Commonly, this is interpreted as shape bias increasing corruption robustness. However, this relationship is only hypothesized. We perform a systematic study of different ways of composing inputs based on natural images, explicit edge information, and stylization. While stylization is essential for achieving high corruption robustness, we do not find a clear correlation between shape bias and robustness. We conclude that the data augmentation caused by style-variation  accounts for the improved corruption robustness and increased shape bias is only a byproduct.

----

## [746] Directed Acyclic Graph Neural Networks

**Authors**: *Veronika Thost, Jie Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JbuYF437WB6](https://openreview.net/forum?id=JbuYF437WB6)

**Abstract**:

Graph-structured data ubiquitously appears in science and engineering. Graph neural networks (GNNs) are designed to exploit the relational inductive bias exhibited in graphs; they have been shown to outperform other forms of neural networks in scenarios where structure information supplements node features. The most common GNN architecture aggregates information from neighborhoods based on message passing. Its generality has made it broadly applicable. In this paper, we focus on a special, yet widely used, type of graphs---DAGs---and inject a stronger inductive bias---partial ordering---into the neural network design. We propose the directed acyclic graph neural network, DAGNN, an architecture that processes information according to the flow defined by the partial order. DAGNN can be considered a framework that entails earlier works as special cases (e.g., models for trees and models updating node representations recurrently), but we identify several crucial components that prior architectures lack. We perform comprehensive experiments, including ablation studies, on representative DAG datasets (i.e., source code, neural architectures, and probabilistic graphical models) and demonstrate the superiority of DAGNN over simpler DAG architectures as well as general graph architectures.

----

## [747] QPLEX: Duplex Dueling Multi-Agent Q-Learning

**Authors**: *Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, Chongjie Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Rcmk0xxIQV](https://openreview.net/forum?id=Rcmk0xxIQV)

**Abstract**:

We explore value-based multi-agent reinforcement learning (MARL) in the popular paradigm of centralized training with decentralized execution (CTDE). CTDE has an important concept, Individual-Global-Max (IGM) principle, which requires the consistency between joint and local action selections to support efficient local decision-making. However, in order to achieve scalability, existing MARL methods either limit representation expressiveness of their value function classes or relax the IGM consistency, which may suffer from instability risk or may not perform well in complex domains. This paper presents a novel MARL approach, called duPLEX dueling multi-agent Q-learning (QPLEX), which takes a duplex dueling network architecture to factorize the joint value function. This duplex dueling structure encodes the IGM principle into the neural network architecture and thus enables efficient value function learning. Theoretical analysis shows that QPLEX achieves a complete IGM function class. Empirical experiments on StarCraft II micromanagement tasks demonstrate that QPLEX significantly outperforms state-of-the-art baselines in both online and offline data collection settings, and also reveal that QPLEX achieves high sample efficiency and can benefit from offline datasets without additional online exploration.

----

## [748] Learning Energy-Based Models by Diffusion Recovery Likelihood

**Authors**: *Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, Diederik P. Kingma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=v_1Soh8QUNc](https://openreview.net/forum?id=v_1Soh8QUNc)

**Abstract**:

While energy-based models (EBMs) exhibit a number of desirable properties, training and sampling on high-dimensional datasets remains challenging. Inspired by recent progress on diffusion probabilistic models, we present a diffusion recovery likelihood method to tractably learn and sample from a sequence of EBMs trained on increasingly noisy versions of a dataset. Each EBM is trained with recovery likelihood,  which maximizes the conditional probability of the data at a certain noise level given their noisy versions at a higher noise level. Optimizing recovery likelihood is more tractable than marginal likelihood, as sampling from the conditional distributions is much easier than sampling from the marginal distributions. After training, synthesized images can be generated by the sampling process that initializes from Gaussian white noise distribution and progressively samples the conditional distributions at decreasingly lower noise levels.  Our method generates high fidelity samples on various image datasets. On unconditional CIFAR-10 our method achieves FID 9.58 and inception score 8.30, superior to the majority of GANs. Moreover, we demonstrate that unlike previous work on EBMs, our long-run MCMC samples from the conditional distributions do not diverge and still represent realistic images, allowing us to accurately estimate the normalized density of data even for high-dimensional datasets. Our implementation is available at \url{https://github.com/ruiqigao/recovery_likelihood}.

----

## [749] Neural Networks for Learning Counterfactual G-Invariances from Single Environments

**Authors**: *S. Chandra Mouli, Bruno Ribeiro*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7t1FcJUWhi3](https://openreview.net/forum?id=7t1FcJUWhi3)

**Abstract**:

Despite —or maybe because of— their astonishing capacity to fit data, neural networks are believed to have difficulties extrapolating beyond training data distribution. This work shows that, for extrapolations based on finite transformation groups, a model’s inability to extrapolate is unrelated to its capacity. Rather, the shortcoming is inherited from a learning hypothesis: Examples not explicitly observed with infinitely many training examples have underspecified outcomes in the learner’s model. In order to endow neural networks with the ability to extrapolate over group transformations, we introduce a learning framework counterfactually-guided by the learning hypothesis that any group invariance to (known) transformation groups is mandatory even without evidence, unless the learner deems it inconsistent with the training data. Unlike existing invariance-driven methods for (counterfactual) extrapolations, this framework allows extrapolations from a single environment. Finally, we introduce sequence and image extrapolation tasks that validate our framework and showcase the shortcomings of traditional approaches.

----

## [750] Model-Based Offline Planning

**Authors**: *Arthur Argenson, Gabriel Dulac-Arnold*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OMNB1G5xzd4](https://openreview.net/forum?id=OMNB1G5xzd4)

**Abstract**:

Offline learning is a key part of making reinforcement learning (RL) useable in real systems. Offline RL looks at scenarios where there is data from a system's operation, but no direct access to the system when learning a policy. Recent work on training RL policies from offline data has shown results both with model-free policies learned directly from the data, or with planning on top of learnt models of the data. Model-free policies tend to be more performant, but are more opaque, harder to command externally, and less easy to integrate into larger systems. We propose an offline learner that generates a model that can be used to control the system directly through planning. This allows us to have easily controllable policies directly from data, without ever interacting with the system. We show the performance of our algorithm, Model-Based Offline Planning (MBOP) on a series of robotics-inspired tasks, and demonstrate its ability leverage planning to respect environmental constraints. We are able to find near-optimal polices for certain simulated systems from as little as 50 seconds of real-time system interaction, and create zero-shot goal-conditioned policies on a series of environments.

----

## [751] On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections

**Authors**: *Peizhao Li, Yifei Wang, Han Zhao, Pengyu Hong, Hongfu Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xgGS6PmzNq6](https://openreview.net/forum?id=xgGS6PmzNq6)

**Abstract**:

Disparate impact has raised serious concerns in machine learning applications and its societal impacts. In response to the need of mitigating discrimination, fairness has been regarded as a crucial property in algorithmic design. In this work, we study the problem of disparate impact on graph-structured data. Specifically, we focus on dyadic fairness, which articulates a fairness concept that a predictive relationship between two instances should be independent of the sensitive attributes. Based on this, we theoretically relate the graph connections to dyadic fairness on link predictive scores in learning graph neural networks, and reveal that regulating weights on existing edges in a graph contributes to dyadic fairness conditionally. Subsequently, we propose our algorithm, \textbf{FairAdj}, to empirically learn a fair adjacency matrix with proper graph structural constraints for fair link prediction, and in the meanwhile preserve predictive accuracy as much as possible. Empirical validation demonstrates that our method delivers effective dyadic fairness in terms of various statistics, and at the same time enjoys a favorable fairness-utility tradeoff.

----

## [752] Coping with Label Shift via Distributionally Robust Optimisation

**Authors**: *Jingzhao Zhang, Aditya Krishna Menon, Andreas Veit, Srinadh Bhojanapalli, Sanjiv Kumar, Suvrit Sra*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BtZhsSGNRNi](https://openreview.net/forum?id=BtZhsSGNRNi)

**Abstract**:

The label shift problem refers to the supervised learning setting where the train and test label distributions do not match. Existing work addressing label shift usually assumes access to an unlabelled test sample. This sample may be used to estimate the test label distribution, and to then train a suitably re-weighted classifier. While approaches using this idea have proven effective, their scope is limited as it is not always feasible to access the target domain; further, they require repeated retraining if the model is to be deployed in multiple test environments. Can one instead learn a single classifier that is robust to arbitrary label shifts from a broad family? In this paper, we answer this question by proposing a model that minimises an objective based on distributionally robust optimisation (DRO). We then design and analyse a gradient descent-proximal mirror ascent algorithm tailored for large-scale problems to optimise the proposed objective.  Finally, through experiments on CIFAR-100 and ImageNet, we show that our technique can significantly improve performance over a number of baselines in settings where label shift is present.

----

## [753] Faster Binary Embeddings for Preserving Euclidean Distances

**Authors**: *Jinjie Zhang, Rayan Saab*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YCXrx6rRCXO](https://openreview.net/forum?id=YCXrx6rRCXO)

**Abstract**:

We propose a fast, distance-preserving, binary embedding algorithm to transform a high-dimensional dataset $\mathcal{T}\subseteq\mathbb{R}^n$ into binary sequences in the cube $\{\pm 1\}^m$. When $\mathcal{T}$ consists of well-spread (i.e., non-sparse) vectors, our embedding method applies a stable noise-shaping quantization scheme to $A x$ where $A\in\mathbb{R}^{m\times n}$ is a sparse Gaussian random matrix. This contrasts with most binary embedding methods, which usually use $x\mapsto \mathrm{sign}(Ax)$ for the embedding. Moreover, we show that Euclidean distances among the elements of $\mathcal{T}$ are approximated by the $\ell_1$ norm on the images of $\{\pm 1\}^m$ under a fast linear transformation. This again contrasts with standard methods, where the Hamming distance is used instead.  Our method is both fast and memory efficient, with time complexity  $O(m)$ and space complexity $O(m)$ on well-spread data. When the data is not well-spread, we show that the approach still works provided that data is transformed via a Walsh-Hadamard matrix, but now the cost is $O(n\log n)$ per data point.  Further, we prove that the method is accurate and its associated error is comparable to that of a continuous valued Johnson-Lindenstrauss embedding plus a quantization error that admits a polynomial decay as the embedding dimension $m$ increases.
	Thus the length of the binary codes required to achieve a desired accuracy is quite small, and we show it can even be compressed further without compromising the accuracy. To illustrate our results, we test the proposed method on natural images and show that it achieves strong performance.

----

## [754] Learning and Evaluating Representations for Deep One-Class Classification

**Authors**: *Kihyuk Sohn, Chun-Liang Li, Jinsung Yoon, Minho Jin, Tomas Pfister*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HCSgyPUfeDj](https://openreview.net/forum?id=HCSgyPUfeDj)

**Abstract**:

We present a two-stage framework for deep one-class classification. We first learn self-supervised  representations from one-class data, and then build one-class classifiers on learned representations. The framework not only allows to learn better representations, but also permits building one-class classifiers that are faithful to the target task. We argue that classifiers inspired by the statistical perspective in generative or discriminative models are more effective than existing approaches, such as a normality score from a surrogate classifier. We thoroughly evaluate different self-supervised representation learning algorithms under the proposed framework for one-class classification. Moreover, we present a novel distribution-augmented contrastive learning that extends training distributions via data augmentation to obstruct the uniformity of contrastive representations. In experiments, we demonstrate state-of-the-art performance on visual domain one-class classification benchmarks, including novelty and anomaly detection. Finally, we present visual explanations, confirming that the decision-making process of deep one-class classifiers is intuitive to humans. The code is available at https://github.com/google-research/deep_representation_one_class.

----

## [755] Conditional Negative Sampling for Contrastive Learning of Visual Representations

**Authors**: *Mike Wu, Milan Mosse, Chengxu Zhuang, Daniel Yamins, Noah D. Goodman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=v8b3e5jN66j](https://openreview.net/forum?id=v8b3e5jN66j)

**Abstract**:

Recent methods for learning unsupervised visual representations, dubbed contrastive learning, optimize the noise-contrastive estimation (NCE) bound on mutual information between two transformations of an image. NCE typically uses randomly sampled negative examples to normalize the objective, but this may often include many uninformative examples either because they are too easy or too hard to discriminate. Taking inspiration from  metric learning, we show that choosing semi-hard negatives can yield stronger contrastive representations. To do this, we introduce a family of mutual information estimators that sample negatives conditionally -- in a "ring" around each positive. We prove that these estimators remain lower-bounds of mutual information, with higher bias but lower variance than NCE. Experimentally, we find our approach, applied on top of existing models (IR, CMC, and MoCo) improves accuracy by 2-5% absolute points in each case, measured by linear evaluation on four standard image benchmarks. Moreover, we find continued benefits when transferring features to a variety of new image distributions from the Meta-Dataset collection and to a variety of downstream tasks such as object detection, instance segmentation, and key-point detection.

----

## [756] On Position Embeddings in BERT

**Authors**: *Benyou Wang, Lifeng Shang, Christina Lioma, Xin Jiang, Hao Yang, Qun Liu, Jakob Grue Simonsen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=onxoVA9FxMw](https://openreview.net/forum?id=onxoVA9FxMw)

**Abstract**:

Various Position Embeddings (PEs) have been proposed in Transformer based architectures~(e.g. BERT) to model word order. These are empirically-driven and perform well, but no formal framework exists to systematically study them. To address this, we present three properties of PEs that capture word distance in vector space:  translation invariance, monotonicity, and  symmetry. These properties formally capture the behaviour of PEs and allow us to reinterpret sinusoidal PEs in a principled way.
Moreover, we propose a new probing test (called `identical word probing') and  mathematical  indicators to quantitatively detect the general  attention patterns with respect to the above properties. An empirical evaluation of seven PEs (and their combinations) for classification (GLUE) and span prediction (SQuAD) shows that: (1) both  classification and span prediction benefit from  translation invariance and local monotonicity, while symmetry slightly decreases performance;
(2) The fully-learnable absolute PE performs better in classification, while relative PEs perform better in span prediction.  We contribute the first formal and quantitative analysis of desiderata for PEs, and  a principled discussion about their correlation to the performance of typical downstream tasks.

----

## [757] Repurposing Pretrained Models for Robust Out-of-domain Few-Shot Learning

**Authors**: *Namyeong Kwon, Hwidong Na, Gabriel Huang, Simon Lacoste-Julien*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qkLMTphG5-h](https://openreview.net/forum?id=qkLMTphG5-h)

**Abstract**:

Model-agnostic meta-learning (MAML) is a popular method for few-shot learning but assumes that we have access to the meta-training set. In practice, training on the meta-training set may not always be an option due to data privacy concerns, intellectual property issues, or merely lack of computing resources. In this paper, we consider the novel problem of repurposing pretrained MAML checkpoints to solve new few-shot classification tasks. Because of the potential distribution mismatch, the original MAML steps may no longer be optimal. Therefore we propose an alternative meta-testing procedure and combine MAML gradient steps with adversarial training and uncertainty-based stepsize adaptation. Our method outperforms "vanilla" MAML on same-domain and cross-domains benchmarks using both SGD and Adam optimizers and shows improved robustness to the choice of base stepsize.

----

## [758] Dataset Meta-Learning from Kernel Ridge-Regression

**Authors**: *Timothy Nguyen, Zhourong Chen, Jaehoon Lee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=l-PrrQrK0QR](https://openreview.net/forum?id=l-PrrQrK0QR)

**Abstract**:

One of the most fundamental aspects of any machine learning algorithm is the training data used by the algorithm. 
We introduce the novel concept of $\epsilon$-approximation of datasets, obtaining datasets which are much smaller than or are significant corruptions of the original training data while maintaining similar performance. We introduce a meta-learning algorithm Kernel Inducing Points (KIP) for obtaining such remarkable datasets, drawing inspiration from recent developments in the correspondence between infinitely-wide neural networks and kernel ridge-regression (KRR). For KRR tasks, we demonstrate that KIP can compress datasets by one or two orders of magnitude, significantly improving previous dataset distillation and subset selection methods while obtaining state of the art results for MNIST and CIFAR10 classification. Furthermore, our KIP-learned datasets are transferable to the training of finite-width neural networks even beyond the lazy-training regime. Consequently, we obtain state of the art results for neural network dataset distillation with potential applications to privacy-preservation.

----

## [759] AdaFuse: Adaptive Temporal Fusion Network for Efficient Action Recognition

**Authors**: *Yue Meng, Rameswar Panda, Chung-Ching Lin, Prasanna Sattigeri, Leonid Karlinsky, Kate Saenko, Aude Oliva, Rogério Feris*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bM3L3I_853](https://openreview.net/forum?id=bM3L3I_853)

**Abstract**:

Temporal modelling is the key for efficient video action recognition. While understanding temporal information can improve recognition accuracy for dynamic actions, removing temporal redundancy and reusing past features can significantly save computation leading to efficient action recognition. In this paper, we introduce an adaptive temporal fusion network, called AdaFuse, that dynamically fuses channels from current and past feature maps for strong temporal modelling. Specifically, the necessary information from the historical convolution feature maps is fused with current pruned feature maps with the goal of improving both recognition accuracy and efficiency. In addition, we use a skipping operation to further reduce the computation cost of action recognition. Extensive experiments on SomethingV1 & V2, Jester and Mini-Kinetics show that our approach can achieve about 40% computation savings with comparable accuracy to state-of-the-art methods. The project page can be found at https://mengyuest.github.io/AdaFuse/

----

## [760] One Network Fits All? Modular versus Monolithic Task Formulations in Neural Networks

**Authors**: *Atish Agarwala, Abhimanyu Das, Brendan Juba, Rina Panigrahy, Vatsal Sharan, Xin Wang, Qiuyi Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uz5uw6gM0m](https://openreview.net/forum?id=uz5uw6gM0m)

**Abstract**:

Can deep learning solve multiple, very different tasks simultaneously? We investigate how the representations of the underlying tasks affect the ability of a single neural network to learn them jointly. We present theoretical and empirical findings that a single neural network is capable of simultaneously learning multiple tasks from a combined data set, for a variety of methods for representing tasks---for example, when the distinct tasks are encoded by well-separated clusters or decision trees over some task-code attributes. Indeed, more strongly, we present a novel analysis that shows that families of simple programming-like constructs for the codes encoding the tasks are learnable by two-layer neural networks with standard training. We study more generally how the complexity of learning such combined tasks grows with the complexity of the task codes; we find that learning many tasks can be provably hard, even though the individual tasks are easy to learn. We provide empirical support for the usefulness of the learning bounds by training networks on clusters, decision trees, and SQL-style aggregation.

----

## [761] Vector-output ReLU Neural Network Problems are Copositive Programs: Convex Analysis of Two Layer Networks and Polynomial-time Algorithms

**Authors**: *Arda Sahiner, Tolga Ergen, John M. Pauly, Mert Pilanci*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fGF8qAqpXXG](https://openreview.net/forum?id=fGF8qAqpXXG)

**Abstract**:

We describe the convex semi-infinite dual of the two-layer vector-output ReLU neural network training problem. This semi-infinite dual admits a finite dimensional representation, but its support is over a convex set which is difficult to characterize. In particular, we demonstrate that the non-convex neural network training problem is equivalent to a finite-dimensional convex copositive program. Our work is the first to identify this strong connection between the global optima of neural networks and those of copositive programs. We thus demonstrate how neural networks implicitly attempt to solve copositive programs via semi-nonnegative matrix factorization, and draw key insights from this formulation. We describe the first algorithms for provably finding the global minimum of the vector output neural network training problem, which are polynomial in the number of samples for a fixed data rank, yet exponential in the dimension. However, in the case of convolutional architectures, the computational complexity is exponential in only the filter size and polynomial in all other parameters. We describe the circumstances in which we can find the global optimum of this neural network training problem exactly with soft-thresholded SVD, and provide a copositive relaxation which is guaranteed to be exact for certain classes of problems, and which corresponds with the solution of Stochastic Gradient Descent in practice.

----

## [762] In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning

**Authors**: *Mamshad Nayeem Rizve, Kevin Duarte, Yogesh S. Rawat, Mubarak Shah*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-ODN6SbiUU](https://openreview.net/forum?id=-ODN6SbiUU)

**Abstract**:

The recent research in semi-supervised learning (SSL) is mostly dominated by consistency regularization based methods which achieve strong performance. However, they heavily rely on domain-specific data augmentations, which are not easy to generate for all data modalities. Pseudo-labeling (PL) is a general SSL approach that does not have this constraint but performs relatively poorly in its original formulation. We argue that PL underperforms due to the erroneous high confidence predictions from poorly calibrated models; these predictions generate many incorrect pseudo-labels, leading to noisy training. We propose an uncertainty-aware pseudo-label selection (UPS) framework which improves pseudo labeling accuracy by drastically reducing the amount of noise encountered in the training process. Furthermore, UPS generalizes the pseudo-labeling process, allowing for the creation of negative pseudo-labels; these negative pseudo-labels can be used for multi-label classification as well as negative learning to improve the single-label classification. We achieve strong performance when compared to recent SSL methods on the CIFAR-10 and CIFAR-100 datasets. Also, we demonstrate the versatility of our method on the video dataset UCF-101 and the multi-label dataset Pascal VOC.

----

## [763] MELR: Meta-Learning via Modeling Episode-Level Relationships for Few-Shot Learning

**Authors**: *Nanyi Fei, Zhiwu Lu, Tao Xiang, Songfang Huang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=D3PcGLdMx0](https://openreview.net/forum?id=D3PcGLdMx0)

**Abstract**:

Most recent few-shot learning (FSL) approaches are based on episodic training whereby each episode samples few training instances (shots) per class to imitate the test condition. However, this strict adhering to test condition has a negative side effect, that is, the trained model is susceptible to the poor sampling of few shots. In this work, for the first time, this problem is addressed by exploiting inter-episode relationships. Specifically, a novel meta-learning via modeling episode-level relationships (MELR) framework is proposed. By sampling two episodes containing the same set of classes for meta-training, MELR is designed to ensure that the meta-learned model is robust against the presence of poorly-sampled shots in the meta-test stage. This is achieved through two key components: (1) a Cross-Episode Attention Module (CEAM) to improve the ability of alleviating the effects of poorly-sampled shots, and (2) a Cross-Episode Consistency Regularization (CECR) to enforce that the two classifiers learned from the two episodes are consistent even when there are unrepresentative instances. Extensive experiments for non-transductive standard FSL on two benchmarks show that our MELR achieves 1.0%-5.0% improvements over the baseline (i.e., ProtoNet) used for FSL in our model and outperforms the latest competitors under the same settings.

----

## [764] Why resampling outperforms reweighting for correcting sampling bias with stochastic gradients

**Authors**: *Jing An, Lexing Ying, Yuhua Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iQQK02mxVIT](https://openreview.net/forum?id=iQQK02mxVIT)

**Abstract**:

A data set sampled from a certain population is biased if the subgroups of the population are sampled at proportions that are significantly different from their underlying proportions. Training machine learning models on biased data sets requires correction techniques to compensate for the bias. We consider two commonly-used techniques, resampling and reweighting, that rebalance the proportions of the subgroups to maintain the desired objective function. Though statistically equivalent, it has been observed that resampling outperforms reweighting when combined with stochastic gradient algorithms. By analyzing illustrative examples, we explain the reason behind this phenomenon using tools from dynamical stability and stochastic asymptotics. We also present experiments from regression, classification, and off-policy prediction to demonstrate that this is a general phenomenon. We argue that it is imperative to consider the objective function design and the optimization algorithm together while addressing the sampling bias.

----

## [765] Prediction and generalisation over directed actions by grid cells

**Authors**: *Changmin Yu, Timothy Behrens, Neil Burgess*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ptaz_zIFbX](https://openreview.net/forum?id=Ptaz_zIFbX)

**Abstract**:

Knowing how the effects of directed actions generalise to new situations (e.g. moving North, South, East and West, or turning left, right, etc.) is key to rapid generalisation across new situations. Markovian tasks can be characterised by a state space and a transition matrix and recent work has proposed that neural grid codes provide an efficient representation of the state space, as eigenvectors of a transition matrix reflecting diffusion across states, that allows efficient prediction of future state distributions. Here we extend the eigenbasis prediction model, utilising tools from Fourier analysis, to prediction over arbitrary translation-invariant directed transition structures (i.e. displacement and diffusion), showing that a single set of eigenvectors can support predictions over arbitrary directed actions via action-specific eigenvalues. We show how to define a "sense of direction" to combine actions to reach a target state (ignoring task-specific deviations from translation-invariance), and demonstrate that adding the Fourier representations to a deep Q network aids policy learning in continuous control tasks. We show the equivalence between the generalised prediction framework and traditional models of grid cell firing driven by self-motion to perform path integration, either using oscillatory interference (via Fourier components as velocity-controlled oscillators) or continuous attractor networks (via analysis of the update dynamics). We thus provide a unifying framework for the role of the grid system in predictive planning, sense of direction and path integration: supporting generalisable inference over directed actions across different tasks.

----

## [766] Hopper: Multi-hop Transformer for Spatiotemporal Reasoning

**Authors**: *Honglu Zhou, Asim Kadav, Farley Lai, Alexandru Niculescu-Mizil, Martin Renqiang Min, Mubbasir Kapadia, Hans Peter Graf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MaZFq7bJif7](https://openreview.net/forum?id=MaZFq7bJif7)

**Abstract**:

This paper considers the problem of spatiotemporal object-centric reasoning in videos. Central to our approach is the notion of object permanence, i.e., the ability to reason about the location of objects as they move through the video while being occluded, contained or carried by other objects. Existing deep learning based approaches often suffer from spatiotemporal biases when applied to video reasoning problems. We propose Hopper, which uses a Multi-hop Transformer for reasoning object permanence in videos. Given a video and a localization query, Hopper reasons over image and object tracks to automatically hop over critical frames in an iterative fashion to predict the final position of the object of interest. We demonstrate the effectiveness of using a contrastive loss to reduce spatiotemporal biases. We evaluate over CATER dataset and find that Hopper achieves 73.2% Top-1 accuracy using just 1 FPS by hopping through just a few critical frames. We also demonstrate Hopper can perform long-term reasoning by building a CATER-h dataset that requires multi-step reasoning to localize objects of interest correctly.

----

## [767] Sparse encoding for more-interpretable feature-selecting representations in probabilistic matrix factorization

**Authors**: *Joshua C. Chang, Patrick Fletcher, Jungmin Han, Ted L. Chang, Shashaank Vattikuti, Bart Desmet, Ayah Zirikly, Carson C. Chow*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=D_KeYoqCYC](https://openreview.net/forum?id=D_KeYoqCYC)

**Abstract**:

Dimensionality reduction methods for count data are critical to a wide range of applications in medical informatics and other fields where model interpretability is paramount. For such data, hierarchical Poisson matrix factorization (HPF) and other sparse probabilistic non-negative matrix factorization (NMF) methods are considered to be interpretable generative models. They consist of sparse transformations for decoding their learned representations into predictions. However, sparsity in representation decoding does not necessarily imply sparsity in the encoding of representations from the original data features.  HPF is often incorrectly interpreted in the literature as if it possesses encoder sparsity. The distinction between decoder sparsity and encoder sparsity is subtle but important. Due to the lack of encoder sparsity, HPF does not possess the column-clustering property of classical NMF -- the factor loading matrix does not sufficiently define how each factor is formed from the original features. We address this deficiency by self-consistently enforcing encoder sparsity, using a generalized additive model  (GAM), thereby allowing one to relate each representation coordinate to a subset of the original data features. In doing so, the method also gains the ability to perform feature selection. We demonstrate our method on simulated data and give an example of how encoder sparsity is of practical use in a concrete application of representing inpatient comorbidities in Medicare patients.

----

## [768] Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning

**Authors**: *Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, Daxin Jiang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xGZG2kS5bFk](https://openreview.net/forum?id=xGZG2kS5bFk)

**Abstract**:

Dancing to music is one of human's innate abilities since ancient times. In machine learning research, however, synthesizing dance movements from music is a challenging problem. Recently, researchers synthesize human motion sequences through autoregressive models like recurrent neural network (RNN). Such an approach often generates short sequences due to an accumulation of prediction errors that are fed back into the neural network. This problem becomes even more severe in the long motion sequence generation. Besides, the consistency between dance and music in terms of style, rhythm and beat is yet to be taken into account during modeling. In this paper, we formalize the music-driven dance generation as a sequence-to-sequence learning problem and devise a novel seq2seq architecture to efficiently process long sequences of music features and capture the fine-grained correspondence between music and dance. Furthermore, we propose a novel curriculum learning strategy to alleviate error accumulation of autoregressive models in long motion sequence generation, which gently changes the training process from a fully guided teacher-forcing scheme using the previous ground-truth movements, towards a less guided autoregressive scheme mostly using the generated movements instead. Extensive experiments show that our approach significantly outperforms the existing state-of-the-arts on automatic metrics and human evaluation. We also make a demo video to demonstrate the superior performance of our proposed approach at https://www.youtube.com/watch?v=lmE20MEheZ8.

----

## [769] PAC Confidence Predictions for Deep Neural Network Classifiers

**Authors**: *Sangdon Park, Shuo Li, Insup Lee, Osbert Bastani*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Qk-Wq5AIjpq](https://openreview.net/forum?id=Qk-Wq5AIjpq)

**Abstract**:

A key challenge for deploying deep neural networks (DNNs) in safety critical settings is the need to provide rigorous ways to quantify their uncertainty. In this paper, we propose a novel algorithm for constructing predicted classification confidences for DNNs that comes with provable correctness guarantees. Our approach uses Clopper-Pearson confidence intervals for the Binomial distribution in conjunction with the histogram binning approach to calibrated prediction. In addition, we demonstrate how our predicted confidences can be used to enable downstream guarantees in two settings: (i) fast DNN inference, where we demonstrate how to compose a fast but inaccurate DNN with an accurate but slow DNN in a rigorous way to improve performance without sacrificing accuracy, and (ii) safe planning, where we guarantee safety when using a DNN to predict whether a given action is safe based on visual observations. In our experiments, we demonstrate that our approach can be used to provide guarantees for state-of-the-art DNNs.

----

## [770] BREEDS: Benchmarks for Subpopulation Shift

**Authors**: *Shibani Santurkar, Dimitris Tsipras, Aleksander Madry*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mQPBmvyAuk](https://openreview.net/forum?id=mQPBmvyAuk)

**Abstract**:

We develop a methodology for assessing the robustness of models to subpopulation shift---specifically, their ability to generalize to novel data subpopulations that were not observed during training. Our approach leverages the class structure underlying existing datasets to control the data subpopulations that comprise the training and test distributions. This enables us to synthesize realistic distribution shifts whose sources can be precisely controlled and characterized, within existing
large-scale datasets. Applying this methodology to the ImageNet dataset, we create a suite of subpopulation shift benchmarks of varying granularity. We then validate that the corresponding shifts are tractable by obtaining human baselines. Finally, we utilize these benchmarks to measure the sensitivity of standard model architectures as well as the effectiveness of existing train-time robustness interventions.

----

## [771] Bypassing the Ambient Dimension: Private SGD with Gradient Subspace Identification

**Authors**: *Yingxue Zhou, Steven Wu, Arindam Banerjee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7dpmlkBuJFC](https://openreview.net/forum?id=7dpmlkBuJFC)

**Abstract**:

Differentially private SGD (DP-SGD) is one of the most popular methods for solving differentially private empirical risk minimization (ERM). Due to its noisy perturbation on each gradient update, the error rate of DP-SGD scales with the ambient dimension $p$, the number of parameters in the model. Such dependence can be problematic for over-parameterized models where $p \gg n$, the number of training samples. Existing lower bounds on private ERM show that such dependence on $p$ is inevitable in the worst case. In this paper, we circumvent the dependence on the ambient dimension by leveraging a low-dimensional structure of gradient space in deep networks---that is, the stochastic gradients for deep nets usually stay in a low dimensional subspace in the training process. We propose Projected DP-SGD that performs noise reduction by projecting the noisy gradients to a low-dimensional subspace, which is given by the top gradient eigenspace on a small public dataset. We provide a general sample complexity analysis on the public dataset for the gradient subspace identification problem and demonstrate that under certain low-dimensional assumptions the public sample complexity only grows logarithmically in $p$. Finally, we provide a theoretical analysis and empirical evaluations to show that our method can substantially improve the accuracy of DP-SGD in the high privacy regime (corresponding to low privacy loss $\epsilon$).

----

## [772] End-to-End Egospheric Spatial Memory

**Authors**: *Daniel James Lenton, Stephen James, Ronald Clark, Andrew J. Davison*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rRFIni1CYmy](https://openreview.net/forum?id=rRFIni1CYmy)

**Abstract**:

Spatial memory, or the ability to remember and recall specific locations and objects, is central to autonomous agents' ability to carry out tasks in real environments. However, most existing artificial memory modules are not very adept at storing spatial information. We propose a parameter-free module, Egospheric Spatial Memory (ESM), which encodes the memory in an ego-sphere around the agent, enabling expressive 3D representations. ESM can be trained end-to-end via either imitation or reinforcement learning, and improves both training efficiency and final performance against other memory baselines on both drone and manipulator visuomotor control tasks. The explicit egocentric geometry also enables us to seamlessly combine the learned controller with other non-learned modalities, such as local obstacle avoidance. We further show applications to semantic segmentation on the ScanNet dataset, where ESM naturally combines image-level and map-level inference modalities. Through our broad set of experiments, we show that ESM provides a general computation graph for embodied spatial reasoning, and the module forms a bridge between real-time mapping systems and differentiable memory architectures. Implementation at: https://github.com/ivy-dl/memory.

----

## [773] Evaluating the Disentanglement of Deep Generative Models through Manifold Topology

**Authors**: *Sharon Zhou, Eric Zelikman, Fred Lu, Andrew Y. Ng, Gunnar E. Carlsson, Stefano Ermon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=djwS0m4Ft_A](https://openreview.net/forum?id=djwS0m4Ft_A)

**Abstract**:

Learning disentangled representations is regarded as a fundamental task for improving the generalization, robustness, and interpretability of generative models. However, measuring disentanglement has been challenging and inconsistent, often dependent on an ad-hoc external model or specific to a certain dataset. To address this, we present a method for quantifying disentanglement that only uses the generative model, by measuring the topological similarity of conditional submanifolds in the learned representation. This method showcases both unsupervised and supervised variants. To illustrate the effectiveness and applicability of our method, we empirically evaluate several state-of-the-art models across multiple datasets. We find that our method ranks models similarly to existing methods. We make our code publicly available at https://github.com/stanfordmlgroup/disentanglement.

----

## [774] SCoRe: Pre-Training for Context Representation in Conversational Semantic Parsing

**Authors**: *Tao Yu, Rui Zhang, Alex Polozov, Christopher Meek, Ahmed Hassan Awadallah*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=oyZxhRI2RiE](https://openreview.net/forum?id=oyZxhRI2RiE)

**Abstract**:

Conversational Semantic Parsing (CSP) is the task of converting a sequence of natural language queries to formal language (e.g., SQL, SPARQL) that can be executed against a structured ontology (e.g.  databases, knowledge bases).  To accomplish  this  task,  a  CSP  system  needs  to  model  the  relation  between  the unstructured language utterance and the structured ontology while representing the multi-turn dynamics of the dialog. Pre-trained language models (LMs) are the state-of-the-art for various natural language processing tasks. However, existing pre-trained LMs that use language modeling training objectives over free-form text have limited ability to represent natural language references to contextual structural data. In this work, we present SCORE, a new pre-training approach for CSP tasks designed to induce representations that capture the alignment between the dialogue flow and the structural context. We demonstrate the broad applicability of SCORE to CSP tasks by combining SCORE with strong base systems on four different tasks (SPARC, COSQL, MWOZ, and SQA). We show that SCORE can improve the performance over all these base systems by a significant margin and achieves state-of-the-art results on three of them.

----

## [775] Decoupling Global and Local Representations via Invertible Generative Flows

**Authors**: *Xuezhe Ma, Xiang Kong, Shanghang Zhang, Eduard H. Hovy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iWLByfvUhN](https://openreview.net/forum?id=iWLByfvUhN)

**Abstract**:

In this work, we propose a new generative model that is capable of automatically decoupling global and local representations of images in an entirely unsupervised setting, by embedding a generative flow in the VAE framework to model the decoder.
Specifically, the proposed model utilizes the variational auto-encoding framework to learn a (low-dimensional) vector of latent variables to capture the global information of an image, which is fed as a conditional input to a flow-based invertible decoder with architecture borrowed from style transfer literature.
Experimental results on standard image benchmarks demonstrate the effectiveness of our model in terms of density estimation, image generation and unsupervised representation learning.
Importantly, this work demonstrates that with only architectural inductive biases, a generative model with a likelihood-based objective is capable of learning decoupled representations, requiring no explicit supervision.
The code for our model is available at \url{https://github.com/XuezheMax/wolf}.

----

## [776] Pre-training Text-to-Text Transformers for Concept-centric Common Sense

**Authors**: *Wangchunshu Zhou, Dong-Ho Lee, Ravi Kiran Selvam, Seyeon Lee, Xiang Ren*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3k20LAiHYL2](https://openreview.net/forum?id=3k20LAiHYL2)

**Abstract**:

Pretrained language models (PTLM) have achieved impressive results in a range of natural language understanding (NLU) and generation (NLG) tasks that require a syntactic and semantic understanding of the text. However, current pre-training objectives such as masked token prediction (for BERT-style PTLMs) and masked span infilling (for T5-style PTLMs) do not explicitly model the relational and compositional commonsense knowledge about everyday concepts, which is crucial to many downstream tasks requiring commonsense reasoning. To augment PTLMs with common sense, we propose generative and contrastive objectives as intermediate self-supervised pre-training tasks between general pre-training and downstream task-specific fine-tuning. We also propose a joint training framework to unify generative and contrastive objectives so that these objectives can be more effective.
Our proposed objectives can pack more commonsense knowledge into the parameters of a pre-trained text-to-text transformer without relying on external knowledge bases, yielding better performance on both NLU and NLG tasks. We apply our method on a pre-trained T5 model in an intermediate task transfer learning fashion to train a concept-aware language model (CALM) and experiment with five commonsense benchmarks (four NLU tasks and one NLG task). Experimental results show that CALM outperforms baseline methods by a consistent margin.

----

## [777] Local Search Algorithms for Rank-Constrained Convex Optimization

**Authors**: *Kyriakos Axiotis, Maxim Sviridenko*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tH6_VWZjoq](https://openreview.net/forum?id=tH6_VWZjoq)

**Abstract**:

We propose greedy and local search algorithms for rank-constrained convex optimization, namely solving $\underset{\mathrm{rank}(A)\leq r^*}{\min}\, R(A)$ given a convex function $R:\mathbb{R}^{m\times n}\rightarrow \mathbb{R}$ and a parameter $r^*$. These algorithms consist of repeating two steps: (a) adding a new rank-1 matrix to $A$ and (b) enforcing the rank constraint on $A$. We refine and improve the theoretical analysis of Shalev-Shwartz et al. (2011), and show that if the rank-restricted condition number of $R$ is $\kappa$, a solution $A$ with rank $O(r^*\cdot \min\{\kappa \log \frac{R(\mathbf{0})-R(A^*)}{\epsilon}, \kappa^2\})$ and $R(A) \leq R(A^*) + \epsilon$ can be recovered, where $A^*$ is the optimal solution. This significantly generalizes associated results on sparse convex optimization, as well as rank-constrained convex optimization for smooth functions. We then introduce new practical variants of these algorithms that have superior runtime and recover better solutions in practice. We demonstrate the versatility of these methods on a wide range of applications involving matrix completion and robust principal component analysis.

----

## [778] Combining Label Propagation and Simple Models out-performs Graph Neural Networks

**Authors**: *Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, Austin R. Benson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8E1-f3VhX1o](https://openreview.net/forum?id=8E1-f3VhX1o)

**Abstract**:

Graph Neural Networks (GNNs) are a predominant technique for learning over graphs. However, there is relatively little understanding of why GNNs are successful in practice and whether they are necessary for good performance. Here, we show that for many standard transductive node classification benchmarks, we can exceed or match the performance of state-of-the-art GNNs by combining shallow models that ignore the graph structure with two simple post-processing steps that exploit correlation in the label structure: (i) an “error correlation” that spreads residual errors in training data to correct errors in test data and (ii) a “prediction correlation” that smooths the predictions on the test data. We call this overall procedure Correct and Smooth (C&S), and the post-processing steps are implemented via simple modifications to standard label propagation techniques that have long been used in graph-based semi-supervised learning. Our approach exceeds or nearly matches the performance of state-of-the-art GNNs on a wide variety of benchmarks, with just a small fraction of the parameters and orders of magnitude faster runtime. For instance, we exceed the best-known GNN performance on the OGB-Products dataset with 137 times fewer parameters and greater than 100 times less training time. The performance of our methods highlights how directly incorporating label information into the learning algorithm (as is common in traditional methods) yields easy and substantial performance gains. We can also incorporate our techniques into big GNN models, providing modest gains in some cases.

----

## [779] Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning

**Authors**: *Beliz Gunel, Jingfei Du, Alexis Conneau, Veselin Stoyanov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cu7IUiOhujH](https://openreview.net/forum?id=cu7IUiOhujH)

**Abstract**:

State-of-the-art natural language understanding classification models follow two-stages: pre-training a large language model on an auxiliary task, and then fine-tuning the model on a task-specific labeled dataset using cross-entropy loss. However, the cross-entropy loss has several shortcomings that can lead to sub-optimal generalization and instability. Driven by the intuition that good generalization requires capturing the similarity between examples in one class and contrasting them with examples in other classes, we propose a supervised contrastive learning (SCL) objective for the fine-tuning stage. Combined with cross-entropy, our proposed SCL loss obtains significant improvements over a strong RoBERTa-Large baseline on multiple datasets of the GLUE benchmark in few-shot learning settings, without requiring specialized architecture, data augmentations, memory banks, or additional unsupervised data. Our proposed fine-tuning objective leads to models that are more robust to different levels of noise in the fine-tuning training data, and can generalize better to related tasks with limited labeled data.

----

## [780] SAFENet: A Secure, Accurate and Fast Neural Network Inference

**Authors**: *Qian Lou, Yilin Shen, Hongxia Jin, Lei Jiang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Cz3dbFm5u-](https://openreview.net/forum?id=Cz3dbFm5u-)

**Abstract**:

The advances in neural networks have driven many companies to provide prediction services to users in a wide range of applications. However, current prediction systems raise privacy concerns regarding the user's private data. A cryptographic neural network inference service is an efficient way to allow two parties to execute neural network inference without revealing either party’s data or model. Nevertheless, existing cryptographic neural network inference services suffer from huge running latency; in particular, the latency of communication-expensive cryptographic activation function is 3 orders of magnitude higher than plaintext-domain activation function. And activations are the necessary components of the modern neural networks. Therefore, slow cryptographic activation has become the primary obstacle of efficient cryptographic inference. 

In this paper, we propose a new technique, called SAFENet, to enable a Secure, Accurate and Fast nEural Network inference service. To speedup secure inference and guarantee inference accuracy, SAFENet includes channel-wise activation approximation with multiple-degree options. This is implemented by keeping the most useful activation channels and replacing the remaining, less useful, channels with various-degree polynomials. SAFENet also supports mixed-precision activation approximation by automatically assigning different replacement ratios to various layer; further increasing the approximation ratio and reducing inference latency. Our experimental results show SAFENet obtains the state-of-the-art inference latency and performance, reducing latency by $38\% \sim 61\%$ or improving accuracy by $1.8\% \sim 4\%$ over prior techniques on various encrypted datasets.

----

## [781] Provably robust classification of adversarial examples with detection

**Authors**: *Fatemeh Sheikholeslami, Ali Lotfi, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sRA5rLNpmQc](https://openreview.net/forum?id=sRA5rLNpmQc)

**Abstract**:

Adversarial attacks against deep networks can be defended against either by building robust classifiers or, by creating classifiers that can \emph{detect} the presence of adversarial perturbations.  Although it may intuitively seem easier to simply detect attacks rather than build a robust classifier, this has not bourne out in practice even empirically, as most detection methods have subsequently been broken by adaptive attacks, thus necessitating \emph{verifiable} performance for detection mechanisms.  In this paper, we propose a new method for jointly training a provably robust classifier and detector.  Specifically, we show that by introducing an additional "abstain/detection" into a classifier, we can modify existing certified defense mechanisms to allow the classifier to either robustly classify \emph{or} detect adversarial attacks.  We extend the common interval bound propagation (IBP) method for certified robustness under $\ell_\infty$ perturbations to account for our new robust objective, and show that the method outperforms traditional IBP used in isolation, especially for large perturbation sizes.  Specifically, tests on MNIST and CIFAR-10 datasets exhibit promising results, for example with provable robust error less than $63.63\%$ and $67.92\%$, for $55.6\%$ and $66.37\%$ natural error, for $\epsilon=8/255$ and $16/255$ on the CIFAR-10 dataset, respectively.

----

## [782] Saliency is a Possible Red Herring When Diagnosing Poor Generalization

**Authors**: *Joseph D. Viviano, Becks Simpson, Francis Dutil, Yoshua Bengio, Joseph Paul Cohen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=c9-WeM-ceB](https://openreview.net/forum?id=c9-WeM-ceB)

**Abstract**:

Poor generalization is one symptom of models that learn to predict target variables using spuriously-correlated image features present only in the training distribution instead of the true image features that denote a class. It is often thought that this can be diagnosed visually using attribution (aka saliency) maps. We study if this assumption is correct. In some prediction tasks, such as for medical images, one may have some images with masks drawn by a human expert, indicating a region of the image containing relevant information to make the prediction. We study multiple methods that take advantage of such auxiliary labels, by training networks to ignore distracting features which may be found outside of the region of interest. This mask information is only used during training and has an impact on generalization accuracy depending on the severity of the shift between the training and test distributions. Surprisingly, while these methods improve generalization performance in the presence of a covariate shift, there is no strong correspondence between the correction of attribution towards the features a human expert have labelled as important and generalization performance. These results suggest that the root cause of poor generalization may not always be spatially defined, and raise questions about the utility of masks as 'attribution priors' as well as saliency maps for explainable predictions.

----

## [783] Fourier Neural Operator for Parametric Partial Differential Equations

**Authors**: *Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, Anima Anandkumar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=c8P9NQVtmnO](https://openreview.net/forum?id=c8P9NQVtmnO)

**Abstract**:

The classical development of neural networks has primarily focused on learning mappings between finite-dimensional Euclidean spaces.  Recently, this has been generalized to neural operators that learn mappings between function spaces. For partial differential equations (PDEs), neural operators directly learn the mapping from any functional parametric dependence to the solution. Thus, they learn an entire family of PDEs, in contrast to classical methods which solve one instance of the equation. In this work, we formulate a new neural operator by parameterizing the integral kernel directly in Fourier space, allowing for an expressive and efficient architecture. We perform experiments on Burgers' equation, Darcy flow, and Navier-Stokes equation. The Fourier neural operator is the first ML-based method to successfully model turbulent flows with zero-shot super-resolution. It is up to three orders of magnitude faster compared to traditional PDE solvers. Additionally, it achieves superior accuracy compared to previous learning-based solvers under fixed resolution.

----

## [784] Combining Ensembles and Data Augmentation Can Harm Your Calibration

**Authors**: *Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W. Dusenberry, Jasper Snoek, Balaji Lakshminarayanan, Dustin Tran*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=g11CZSghXyY](https://openreview.net/forum?id=g11CZSghXyY)

**Abstract**:

Ensemble methods which average over multiple neural network predictions are a simple approach to improve a model’s calibration and robustness. Similarly, data augmentation techniques, which encode prior information in the form of invariant feature transformations, are effective for improving calibration and robustness. In this paper, we show a surprising pathology: combining ensembles and data augmentation can harm model calibration. This leads to a trade-off in practice, whereby improved accuracy by combining the two techniques comes at the expense of calibration. On the other hand, selecting only one of the techniques ensures good uncertainty estimates at the expense of accuracy. We investigate this pathology and identify a compounding under-confidence among methods which marginalize over sets of weights and data augmentation techniques which soften labels. Finally, we propose a simple correction, achieving the best of both worlds with significant accuracy and calibration gains over using only ensembles or data augmentation individually. Applying the correction produces new state-of-the art in uncertainty calibration and robustness across CIFAR-10, CIFAR-100, and ImageNet.

----

## [785] SOLAR: Sparse Orthogonal Learned and Random Embeddings

**Authors**: *Tharun Medini, Beidi Chen, Anshumali Shrivastava*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fw-BHZ1KjxJ](https://openreview.net/forum?id=fw-BHZ1KjxJ)

**Abstract**:

Dense embedding models are commonly deployed in commercial search engines, wherein all the document vectors are pre-computed, and near-neighbor search (NNS) is performed with the query vector to find relevant documents. However, the bottleneck of indexing a large number of dense vectors and performing an NNS hurts the query time and accuracy of these models. In this paper, we argue that high-dimensional and ultra-sparse embedding is a significantly superior alternative to dense low-dimensional embedding for both query efficiency and accuracy. Extreme sparsity eliminates the need for NNS by replacing them with simple lookups, while its high dimensionality ensures that the embeddings are informative even when sparse. However, learning extremely high dimensional embeddings leads to blow up in the model size. To make the training feasible, we propose a partitioning algorithm that learns such high dimensional embeddings across multiple GPUs without any communication. This is facilitated by our novel asymmetric mixture of Sparse, Orthogonal, Learned and Random (SOLAR) Embeddings. The label vectors are random, sparse, and near-orthogonal by design, while the query vectors are learned and sparse. We theoretically prove that our way of one-sided learning is equivalent to learning both query and label embeddings. With these unique properties, we can successfully train 500K dimensional SOLAR embeddings for the tasks of searching through 1.6M books and multi-label classification on the three largest public datasets. We achieve superior precision and recall compared to the respective state-of-the-art baselines for each task with up to 10 times faster speed.

----

## [786] Efficient Empowerment Estimation for Unsupervised Stabilization

**Authors**: *Ruihan Zhao, Kevin Lu, Pieter Abbeel, Stas Tiomkin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=u2YNJPcQlwq](https://openreview.net/forum?id=u2YNJPcQlwq)

**Abstract**:

Intrinsically motivated artificial agents learn advantageous behavior without externally-provided rewards. Previously, it was shown that maximizing mutual information between agent actuators and future states, known as the empowerment principle, enables unsupervised stabilization of dynamical systems at upright positions, which is a prototypical intrinsically motivated behavior for upright standing and walking. This follows from the coincidence between the objective of stabilization and the objective of empowerment. Unfortunately, sample-based estimation of this kind of mutual information is challenging. Recently, various variational lower bounds (VLBs) on empowerment have been proposed as solutions; however, they are often biased, unstable in training, and have high sample complexity. In this work, we propose an alternative solution based on a trainable representation of a dynamical system as a Gaussian channel, which allows us to efficiently calculate an unbiased estimator of empowerment by convex optimization. We demonstrate our solution for sample-based unsupervised stabilization on different dynamical control systems and show the advantages of our method by comparing it to the existing VLB approaches. Specifically, we show that our method has a lower sample complexity, is more stable in training, possesses the essential properties of the empowerment function, and allows estimation of empowerment from images. Consequently, our method opens a path to wider and easier adoption of empowerment for various applications.

----

## [787] More or Less: When and How to Build Convolutional Neural Network Ensembles

**Authors**: *Abdul Wasay, Stratos Idreos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=z5Z023VBmDZ](https://openreview.net/forum?id=z5Z023VBmDZ)

**Abstract**:

Convolutional neural networks are utilized to solve increasingly more complex problems and with more data. As a  result, researchers and practitioners seek to scale the representational power of such models by adding more parameters. However, increasing parameters requires additional critical resources in terms of memory and compute,  leading to increased training and inference cost. Thus a consistent challenge is to obtain as high as possible accuracy within a parameter budget. As neural network designers navigate this complex landscape, they are guided by conventional wisdom that is informed from past empirical studies. We identify a critical part of this design space that is not well-understood: How to decide between the alternatives of expanding a single convolutional network model or increasing the number of networks in the form of an ensemble. We study this question in detail across various network architectures and data sets. We build an extensive experimental framework that captures numerous angles of the possible design space in terms of how a new set of parameters can be used in a model.  We consider a holistic set of metrics such as training time, inference time, and memory usage. The framework provides a robust assessment by making sure it controls for the number of parameters. Contrary to conventional wisdom, we show that when we perform a holistic and robust assessment, we uncover a wide design space, where ensembles provide better accuracy, train faster, and deploy at speed comparable to single convolutional networks with the same total number of parameters.

----

## [788] VTNet: Visual Transformer Network for Object Goal Navigation

**Authors**: *Heming Du, Xin Yu, Liang Zheng*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=DILxQP08O3B](https://openreview.net/forum?id=DILxQP08O3B)

**Abstract**:

Object goal navigation aims to steer an agent towards a target object based on observations of the agent. It is of pivotal importance to design effective visual representations of the observed scene in determining navigation actions.  In this paper, we introduce a Visual Transformer Network (VTNet) for learning informative visual representation in navigation.  VTNet is a highly effective structure that embodies two key properties for visual representations: First, the relationships among all the object instances in a scene are exploited; Second, the spatial locations of objects and image regions are emphasized so that directional navigation signals can be learned. Furthermore, we also develop a pre-training scheme to associate the visual representations with navigation signals, and thus facilitate navigation policy learning. In a nutshell, VTNet embeds object and region features with their location cues as spatial-aware descriptors and then incorporates all the encoded descriptors through attention operations to achieve informative representation for navigation. Given such visual representations, agents are able to explore the correlations between visual observations and navigation actions. For example, an agent would prioritize ``turning right'' over ``turning left'' when the visual representation emphasizes on the right side of activation map. Experiments in the artificial environment AI2-Thor demonstrate that VTNet significantly outperforms state-of-the-art methods in unseen testing environments.

----

## [789] Class Normalization for (Continual)? Generalized Zero-Shot Learning

**Authors**: *Ivan Skorokhodov, Mohamed Elhoseiny*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7pgFL2Dkyyy](https://openreview.net/forum?id=7pgFL2Dkyyy)

**Abstract**:

Normalization techniques have proved to be a crucial ingredient of successful training in a traditional supervised learning regime. However, in the zero-shot learning (ZSL) world, these ideas have received only marginal attention. This work studies normalization in ZSL scenario from both theoretical and practical perspectives. First, we give a theoretical explanation to two popular tricks used in zero-shot learning: normalize+scale and attributes normalization and show that they help training by preserving variance during a forward pass. Next, we demonstrate that they are insufficient to normalize a deep ZSL model and propose Class Normalization (CN): a normalization scheme, which alleviates this issue both provably and in practice. Third, we show that ZSL models typically have more irregular loss surface compared to traditional classifiers and that the proposed method partially remedies this problem. Then, we test our approach on 4 standard ZSL datasets and outperform sophisticated modern SotA with a simple MLP optimized without any bells and whistles and having ~50 times faster training speed. Finally, we generalize ZSL to a broader problem — continual ZSL, and introduce some principled metrics and rigorous baselines for this new setup. The source code is available at https://github.com/universome/class-norm.

----

## [790] Batch Reinforcement Learning Through Continuation Method

**Authors**: *Yijie Guo, Shengyu Feng, Nicolas Le Roux, Ed H. Chi, Honglak Lee, Minmin Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=po-DLlBuAuz](https://openreview.net/forum?id=po-DLlBuAuz)

**Abstract**:

Many real-world applications of reinforcement learning (RL) require the agent to learn from a fixed set of trajectories, without collecting new interactions.  Policy optimization under this setting is extremely challenging as: 1) the geometry of the objective function is hard to optimize efficiently; 2) the shift of data distributions causes high noise in the value estimation. In this work, we propose a simple yet effective policy iteration approach to batch RL using global optimization techniques known as continuation.  By constraining the difference between the learned policy and the behavior policy that generates the fixed trajectories, and continuously relaxing the constraint, our method 1) helps the agent escape local optima; 2) reduces the error in policy evaluation in the optimization procedure.   We present results on a variety of control tasks, game environments, and a recommendation task to empirically demonstrate the efficacy of our proposed method.

----

## [791] Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning

**Authors**: *Zhiyuan Li, Yuping Luo, Kaifeng Lyu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AHOs7Sm5H7R](https://openreview.net/forum?id=AHOs7Sm5H7R)

**Abstract**:

Matrix factorization is a simple and natural test-bed to investigate the implicit regularization of gradient descent. Gunasekar et al. (2017) conjectured that gradient flow with infinitesimal initialization converges to the solution that minimizes the nuclear norm, but a series of recent papers argued that the language of norm minimization is not sufficient to give a full characterization for the implicit regularization. In this work, we provide theoretical and empirical evidence that for depth-2 matrix factorization, gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions. This generalizes the rank minimization view from previous works to a much broader setting and enables us to construct counter-examples to refute the conjecture from Gunasekar et al. (2017). We also extend the results to the case where depth >= 3, and we show that the benefit of being deeper is that the above convergence has a much weaker dependence over initialization magnitude so that this rank minimization is more likely to take effect for initialization with practical scale.

----

## [792] Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs

**Authors**: *Jonathan Frankle, David J. Schwab, Ari S. Morcos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vYeQQ29Tbvx](https://openreview.net/forum?id=vYeQQ29Tbvx)

**Abstract**:

A wide variety of deep learning techniques from style transfer to multitask learning rely on training affine transformations of features. Most prominent among these is the popular feature normalization technique BatchNorm, which normalizes activations and then subsequently applies a learned affine transform. In this paper, we aim to understand the role and expressive power of affine parameters used to transform features in this way. To isolate the contribution of these parameters from that of the learned features they transform, we investigate the performance achieved when training only these parameters in BatchNorm and freezing all weights at their random initializations. Doing so leads to surprisingly high performance considering the significant limitations that this style of training imposes. For example, sufficiently deep ResNets reach 82% (CIFAR-10) and 32% (ImageNet, top-5) accuracy in this configuration, far higher than when training an equivalent number of randomly chosen parameters elsewhere in the network. BatchNorm achieves this performance in part by naturally learning to disable around a third of the random features. Not only do these results highlight the expressive power of affine parameters in deep learning, but - in a broader sense - they characterize the expressive power of neural networks constructed simply by shifting and rescaling random features.

----

## [793] CompOFA - Compound Once-For-All Networks for Faster Multi-Platform Deployment

**Authors**: *Manas Sahni, Shreya Varshini, Alind Khare, Alexey Tumanov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IgIk8RRT-Z](https://openreview.net/forum?id=IgIk8RRT-Z)

**Abstract**:

The emergence of CNNs in mainstream deployment has necessitated methods to design and train efficient architectures tailored to maximize the accuracy under diverse hardware and latency constraints. To scale these resource-intensive tasks with an increasing number of deployment targets, Once-For-All (OFA) proposed an approach to jointly train several models at once with a constant training cost. However, this cost remains as high as 40-50 GPU days and also suffers from a combinatorial explosion of sub-optimal model configurations. We seek to reduce this search space -- and hence the training budget -- by constraining search to models close to the accuracy-latency Pareto frontier. We incorporate insights of compound relationships between model dimensions to build CompOFA, a design space smaller by several orders of magnitude.  Through experiments on ImageNet, we demonstrate that even with simple heuristics we can achieve a 2x reduction in training time and 216x speedup in model search/extraction time compared to the state of the art, without loss of Pareto optimality! We also show that this smaller design space is dense enough to support equally accurate models for a similar diversity of hardware and latency targets, while also reducing the complexity of the training and subsequent extraction algorithms. Our source code is available at https://github.com/gatech-sysml/CompOFA

----

## [794] Adaptive and Generative Zero-Shot Learning

**Authors**: *Yu-Ying Chou, Hsuan-Tien Lin, Tyng-Luh Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ahAUv8TI2Mz](https://openreview.net/forum?id=ahAUv8TI2Mz)

**Abstract**:

We address the problem of generalized zero-shot learning (GZSL) where the task is to predict the class label of a target image whether its label belongs to the seen or unseen category. Similar to ZSL, the learning setting assumes that all class-level semantic features are given, while only the images of seen classes are available for training. By exploring the correlation between image features and the corresponding semantic features, the main idea of the proposed approach is to enrich the semantic-to-visual (S2V) embeddings via a seamless fusion of adaptive and generative learning. To this end, we extend the semantic features of each class by supplementing image-adaptive attention so that the learned S2V embedding can account for not only inter-class but also intra-class variations. In addition, to break the limit of training with images only from seen classes, we design a generative scheme to simultaneously generate virtual class labels and their visual features by sampling and interpolating over seen counterparts. In inference, a testing image will give rise to two different S2V embeddings, seen and virtual. The former is used to decide whether the underlying label is of the unseen category or otherwise a specific seen class; the latter is to predict an unseen class label. To demonstrate the effectiveness of our method, we report state-of-the-art results on four standard GZSL datasets, including an ablation study of the proposed modules.

----

## [795] Learning to Make Decisions via Submodular Regularization

**Authors**: *Ayya Alieva, Aiden Aceves, Jialin Song, Stephen Mayo, Yisong Yue, Yuxin Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ac288vnG_7U](https://openreview.net/forum?id=ac288vnG_7U)

**Abstract**:

Many sequential decision making tasks can be viewed as combinatorial optimization problems over a large number of actions. When the cost of evaluating an action is high, even a greedy algorithm, which iteratively picks the best action given the history, is prohibitive to run. In this paper, we aim to learn a greedy heuristic for sequentially selecting actions as a surrogate for invoking the expensive oracle when evaluating an action. In particular, we focus on a class of combinatorial problems that can be solved via submodular maximization (either directly on the objective function or via submodular surrogates). We introduce a data-driven optimization framework based on the submodular-norm loss, a novel loss function that encourages the resulting objective to exhibit diminishing returns. Our framework outputs a surrogate objective that is efficient to train, approximately submodular, and can be made permutation-invariant. The latter two properties allow us to prove strong approximation guarantees for the learned greedy heuristic. Furthermore, we show that our model can be easily integrated with modern deep imitation learning pipelines for sequential prediction tasks. We demonstrate the performance of our algorithm on a variety of batched and sequential optimization tasks, including set cover, active learning, and Bayesian optimization for protein engineering.

----

## [796] Generating Furry Cars: Disentangling Object Shape and Appearance across Multiple Domains

**Authors**: *Utkarsh Ojha, Krishna Kumar Singh, Yong Jae Lee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=M88oFvqp_9](https://openreview.net/forum?id=M88oFvqp_9)

**Abstract**:

We consider the novel task of learning disentangled representations of object shape and appearance across multiple domains (e.g., dogs and cars).  The goal is to learn a generative model that learns an intermediate distribution, which borrows a subset of properties from each domain, enabling the generation of images that did not exist in any domain exclusively.  This challenging problem requires an accurate disentanglement of object shape, appearance, and background from each domain, so that the appearance and shape factors from the two domains can be interchanged. We augment an existing approach that can disentangle factors within a single domain but struggles to do so across domains.  Our key technical contribution is to represent object appearance with a differentiable histogram of visual features, and to optimize the generator so that two images with the same latent appearance factor but different latent shape factors produce similar histograms. On multiple multi-domain datasets, we demonstrate our method leads to accurate and consistent appearance and shape transfer across domains.

----

## [797] Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning

**Authors**: *Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O9bnihsFfXU](https://openreview.net/forum?id=O9bnihsFfXU)

**Abstract**:

We identify an implicit under-parameterization phenomenon in value-based deep RL methods that use bootstrapping: when value functions, approximated using deep neural networks, are trained with gradient descent using iterated regression onto target values generated by previous instances of the value network, more gradient updates decrease the expressivity of the current value network. We char- acterize this loss of expressivity via a drop in the rank of the learned value net- work features, and show that this typically corresponds to a performance drop. We demonstrate this phenomenon on Atari and Gym benchmarks, in both offline and online RL settings. We formally analyze this phenomenon and show that it results from a pathological interaction between bootstrapping and gradient-based optimization. We further show that mitigating implicit under-parameterization by controlling rank collapse can improve performance.

----

## [798] Disentangling 3D Prototypical Networks for Few-Shot Concept Learning

**Authors**: *Mihir Prabhudesai, Shamit Lal, Darshan Patil, Hsiao-Yu Tung, Adam W. Harley, Katerina Fragkiadaki*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-Lr-u0b42he](https://openreview.net/forum?id=-Lr-u0b42he)

**Abstract**:

We present neural architectures that disentangle RGB-D images into objects’ shapes and styles and a map of the background scene, and explore their applications for few-shot 3D object detection and few-shot concept classification. Our networks incorporate architectural biases that reflect the image formation process, 3D  geometry of the world scene, and shape-style interplay. They are trained end-to-end self-supervised by predicting views in static scenes, alongside a small number of 3D object boxes. Objects and scenes are represented in terms of 3D feature grids in the bottleneck of the network. We show the proposed 3D neural representations are compositional: they can generate novel 3D scene feature maps by mixing object shapes and styles, resizing and adding the resulting object 3D feature maps over background scene feature maps. We show object detectors trained on hallucinated 3D neural scenes generalize better to novel environments. We show classifiers for object categories, color, materials, and spatial relationships trained over the  disentangled 3D feature sub-spaces generalize better with dramatically fewer exemplars over the current state-of-the-art, and enable a visual question answering system that uses them as its modules to generalize one-shot to novel objects in the scene.

----

## [799] Anytime Sampling for Autoregressive Models via Ordered Autoencoding

**Authors**: *Yilun Xu, Yang Song, Sahaj Garg, Linyuan Gong, Rui Shu, Aditya Grover, Stefano Ermon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TSRTzJnuEBS](https://openreview.net/forum?id=TSRTzJnuEBS)

**Abstract**:

Autoregressive models are widely used for tasks such as image and audio generation. The sampling process of these models, however, does not allow interruptions and cannot adapt to real-time computational resources. This challenge impedes the deployment of powerful autoregressive models, which involve a slow sampling process that is sequential in nature and typically scales linearly with respect to the data dimension.  To address this difficulty, we propose a new family of autoregressive models that enables anytime sampling. Inspired by Principal Component Analysis, we learn a structured representation space where dimensions are ordered based on their importance with respect to reconstruction. Using an autoregressive model in this latent space, we trade off sample quality for computational efficiency by truncating the generation process before decoding into the original data space. Experimentally, we demonstrate in several image and audio generation tasks that sample quality degrades gracefully as we reduce the computational budget for sampling. The approach suffers almost no loss in sample quality (measured by FID) using only 60\% to 80\% of all latent dimensions for image data. Code is available at https://github.com/Newbeeer/Anytime-Auto-Regressive-Model.

----



[Go to the previous page](ICLR-2021-list03.md)

[Go to the next page](ICLR-2021-list05.md)

[Go to the catalog section](README.md)