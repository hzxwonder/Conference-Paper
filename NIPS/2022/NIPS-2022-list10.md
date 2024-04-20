## [1800] House of Cans: Covert Transmission of Internal Datasets via Capacity-Aware Neuron Steganography

**Authors**: *Xudong Pan, Shengyao Zhang, Mi Zhang, Yifan Yan, Min Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d65080f3be61f4dcc5ca4c293308104-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d65080f3be61f4dcc5ca4c293308104-Abstract-Conference.html)

**Abstract**:

In this paper, we present a capacity-aware neuron steganography scheme (i.e., Cans) to covertly transmit multiple private machine learning (ML) datasets via a scheduled-to-publish deep neural network (DNN) as the carrier model. Unlike existing steganography schemes which treat the DNN parameters as bit strings, \textit{Cans} for the first time exploits the learning capacity of the carrier model via a novel parameter sharing mechanism. Extensive evaluation shows, Cans is the first working scheme which can covertly transmit over $10000$ real-world data samples within a carrier model which has $220\times$ less parameters than the total size of the stolen data, and simultaneously transmit multiple heterogeneous datasets within a single carrier model, under a trivial distortion rate ($<10^{-5}$) and with almost no utility loss on the carrier model ($<1\%$). Besides, Cans implements by-design redundancy to be resilient against common post-processing techniques on the carrier model before the publishing.

----

## [1801] Polynomial-Time Optimal Equilibria with a Mediator in Extensive-Form Games

**Authors**: *Brian Hu Zhang, Tuomas Sandholm*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d823334fdccb62a544fa7643cf0615d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d823334fdccb62a544fa7643cf0615d-Abstract-Conference.html)

**Abstract**:

For common notions of correlated equilibrium in extensive-form games, computing an optimal (e.g., welfare-maximizing) equilibrium is NP-hard. Other equilibrium notions---communication and certification equilibria---augment the game with a mediator that has the power to both send and receive messages to and from the players---and, in particular, to remember the messages. In this paper, we investigate both notions in extensive-form games from a computational lens. We show that optimal equilibria in both notions can be computed in polynomial time, the latter under a natural additional assumption known in the literature. Our proof works by constructing a {\em mediator-augmented game} of polynomial size that explicitly represents the mediator's decisions and actions. Our framework allows us to define an entire family of equilibria by varying the mediator's information partition, the players' ability to lie, and the players' ability to deviate. From this perspective, we show that other notions of equilibrium, such as extensive-form correlated equilibrium, correspond to the mediator having imperfect recall. This shows that, at least among all these equilibrium notions, the hardness of computation is driven by the mediator's imperfect recall. As special cases of our general construction, we recover the polynomial-time algorithm of Conitzer & Sandholm [2004] for automated mechanism design in Bayes-Nash equilibria, and the correlation DAG algorithm of Zhang et al [2022] for optimal correlation. Our algorithm is especially scalable when the equilibrium notion is what we define as the full-certification equilibrium, where players cannot lie about their information but they can be silent. We back up our theoretical claims with experiments on a suite of standard benchmark games.

----

## [1802] Dungeons and Data: A Large-Scale NetHack Dataset

**Authors**: *Eric Hambro, Roberta Raileanu, Danielle Rothermel, Vegard Mella, Tim Rocktäschel, Heinrich Küttler, Naila Murray*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d9258fd703057246cb341e615426e2d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d9258fd703057246cb341e615426e2d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent breakthroughs in the development of agents to solve challenging sequential decision making problems such as Go, StarCraft, or DOTA, have relied on both simulated environments and large-scale datasets.  However, progress on this research has been hindered by the scarcity of open-sourced datasets and the prohibitive computational cost to work with them.  Here we present the NetHack Learning Dataset (NLD), a large and highly-scalable dataset of trajectories from the popular game of NetHack, which is both extremely challenging for current methods and very fast to run. NLD consists of three parts: 10 billion state transitions from 1.5 million human trajectories collected on the NAO public NetHack server from 2009 to 2020; 3 billion state-action-score transitions from 100,000 trajectories collected from the symbolic bot winner of the NetHack Challenge 2021; and, accompanying code for users to record, load and stream any collection of such trajectories in a highly compressed form.  We evaluate a wide range of existing algorithms for learning from demonstrations, showing that significant research advances are needed to fully leverage large-scale datasets for challenging sequential decision making tasks.

----

## [1803] Optimal Dynamic Regret in LQR Control

**Authors**: *Dheeraj Baby, Yu-Xiang Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d94bf4711fa459812437e5df5978551-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d94bf4711fa459812437e5df5978551-Abstract-Conference.html)

**Abstract**:

We consider the problem of nonstochastic control with a sequence of quadratic losses, i.e., LQR control. We provide an efficient online algorithm that achieves an optimal dynamic (policy) regret of $\tilde{O}(n^{1/3} \mathcal{TV}(M_{1:n}^{2/3}  \vee 1)$, where $\mathcal{TV}(M_{1:n})$ is the total variation of any oracle sequence of \emph{Disturbance Action} policies parameterized by $M_1,...,M_n$ --- chosen in hindsight to cater to unknown nonstationarity. The rate improves the best known rate of $\tilde{O}(\sqrt{n (\mathcal{TV}(M_{1:n})+1)} )$ for general convex losses and is information-theoretically optimal for LQR. Main technical components include the reduction of LQR to online linear regression with delayed feedback due to Foster & Simchowitz 2020, as well as a new \emph{proper} learning algorithm with an optimal $\tilde{O}(n^{1/3})$ dynamic regret on a family of "minibatched'' quadratic losses, which could be of independent interest.

----

## [1804] Gradient Descent Is Optimal Under Lower Restricted Secant Inequality And Upper Error Bound

**Authors**: *Charles Guille-Escuret, Adam Ibrahim, Baptiste Goujaud, Ioannis Mitliagkas*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9daab3b451038ed1bc5d8e9b77996b99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9daab3b451038ed1bc5d8e9b77996b99-Abstract-Conference.html)

**Abstract**:

The study of first-order optimization is sensitive to the assumptions made on the objective functions.These assumptions induce complexity classes which play a key role in worst-case analysis, includingthe fundamental concept of algorithm optimality. Recent work argues that strong convexity andsmoothness—popular assumptions in literature—lead to a pathological definition of the conditionnumber. Motivated by this result, we focus on the class of functionssatisfying a lower restricted secant inequality and an upper error bound. On top of being robust tothe aforementioned pathological behavior and including some non-convex functions, this pair ofconditions displays interesting geometrical properties. In particular, the necessary and sufficientconditions to interpolate a set of points and their gradients within the class can be separated intosimple conditions on each sampled gradient. This allows the performance estimation problem (PEP) to be solved analytically, leading to a lower boundon the convergence rate that proves gradient descent to be exactly optimal on this class of functionsamong all first-order algorithms.

----

## [1805] GenSDF: Two-Stage Learning of Generalizable Signed Distance Functions

**Authors**: *Gene Chou, Ilya Chugunov, Felix Heide*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9dfb5bc27e2d046199b38739e4ce64bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9dfb5bc27e2d046199b38739e4ce64bd-Abstract-Conference.html)

**Abstract**:

We investigate the generalization capabilities of neural signed distance functions (SDFs) for learning 3D object representations for unseen and unlabeled point clouds. Existing methods can fit SDFs to a handful of object classes and boast fine detail or fast inference speeds, but do not generalize well to unseen shapes. We introduce a two-stage semi-supervised meta-learning approach that transfers shape priors from labeled to unlabeled data to reconstruct unseen object categories. The first stage uses an episodic training scheme to simulate training on unlabeled data and meta-learns initial shape priors. The second stage then introduces unlabeled data with disjoint classes in a semi-supervised scheme to diversify these priors and achieve generalization. We assess our method on both synthetic data and real collected point clouds. Experimental results and analysis validate that our approach outperforms existing neural SDF methods and is capable of robust zero-shot inference on 100+ unseen classes. Code can be found at https://github.com/princeton-computational-imaging/gensdf

----

## [1806] Forecasting Human Trajectory from Scene History

**Authors**: *Mancheng Meng, Ziyan Wu, Terrence Chen, Xiran Cai, Xiang Sean Zhou, Fan Yang, Dinggang Shen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e3b203e72c4e058de26d02a92a81844-Abstract-Conference.html)

**Abstract**:

Predicting the future trajectory of a person remains a challenging problem, due to randomness and subjectivity. However, the moving patterns of human in constrained scenario typically conform to a limited number of regularities to a certain extent, because of the scenario restrictions (\eg, floor plan, roads and obstacles) and person-person or person-object interactivity. Thus, an individual person in this scenario should follow one of the regularities as well. In other words, a person's subsequent trajectory has likely been traveled by others. Based on this hypothesis, we propose to forecast a person's future trajectory by learning from the implicit scene regularities. We call the regularities, inherently derived from the past dynamics of the people and the environment in the scene,  \emph{scene history}. We categorize scene history information into two types: historical group trajectories and individual-surroundings interaction. To exploit these information for trajectory prediction, we propose a novel framework Scene History Excavating Network (SHENet), where the scene history is leveraged in a simple yet effective approach. In particular, we design two components, the group trajectory bank module to extract representative group trajectories as the candidate for future path, and the cross-modal interaction module to model the interaction between individual past trajectory and its surroundings for trajectory refinement, respectively.  In addition, to mitigate the uncertainty in the evaluation, caused by the aforementioned randomness and subjectivity, we propose to include smoothness into evaluation metrics. We conduct extensive evaluations to validate the efficacy of proposed framework on ETH, UCY, as well as a new, challenging benchmark dataset PAV, demonstrating superior performance compared to state-of-the-art methods.

----

## [1807] Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure

**Authors**: *Shaohua Fan, Xiao Wang, Yanhu Mo, Chuan Shi, Jian Tang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e47a0bc530cc88b09b7670d2c130a29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e47a0bc530cc88b09b7670d2c130a29-Abstract-Conference.html)

**Abstract**:

Most Graph Neural Networks (GNNs) predict the labels of unseen graphs by learning the correlation between the input graphs and labels. However, by presenting a graph classification investigation on the training graphs with severe bias, surprisingly, we discover that GNNs always tend to explore the spurious correlations to make decision, even if the causal correlation always exists. This implies that existing GNNs trained on such biased datasets will suffer from poor generalization capability.  By analyzing this problem in a causal view, we find that disentangling and decorrelating the causal and bias latent variables from the biased graphs are both crucial for debiasing. Inspired by this, we propose a general disentangled GNN framework to learn the causal substructure and bias substructure, respectively. Particularly,  we design a parameterized edge mask generator to explicitly split the input graph into causal and bias subgraphs. Then two GNN modules supervised by causal/bias-aware loss functions respectively are trained to encode causal and bias subgraphs into their corresponding representations. With the disentangled representations, we synthesize the counterfactual unbiased training samples to further decorrelate causal and bias variables. Moreover, to better benchmark the severe bias problem, we construct three new graph datasets, which have controllable bias degrees and are easier to visualize and explain. Experimental results well demonstrate that our approach achieves superior generalization performance over existing baselines. Furthermore, owing to the learned edge mask, the proposed model has appealing interpretability and transferability.

----

## [1808] Stochastic Online Learning with Feedback Graphs: Finite-Time and Asymptotic Optimality

**Authors**: *Teodor Vanislavov Marinov, Mehryar Mohri, Julian Zimmert*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e79aefb538d02a7c0610fa43bdb0d0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e79aefb538d02a7c0610fa43bdb0d0f-Abstract-Conference.html)

**Abstract**:

We revisit the problem of stochastic online learning with feedbackgraphs, with the goal of devising algorithms that are optimal, up toconstants, both asymptotically and in finite time. We show that,surprisingly, the notion of optimal finite-time regret is not auniquely defined property in this context and that, in general, itis decoupled from the asymptotic rate. We discuss alternativechoices and propose a notion of finite-time optimality that we argueis \emph{meaningful}. For that notion, we give an algorithm thatadmits quasi-optimal regret both in finite-time and asymptotically.

----

## [1809] Asymptotics of ℓ2 Regularized Network Embeddings

**Authors**: *Andrew Davison*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e800ca1a4898f49a77fc0fcf7ec77e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e800ca1a4898f49a77fc0fcf7ec77e5-Abstract-Conference.html)

**Abstract**:

A common approach to solving prediction tasks on large networks, such as node classification or link prediction, begin by learning a Euclidean embedding of the nodes of the network, from which traditional machine learning methods can then be applied. This includes methods such as DeepWalk and node2vec, which learn embeddings by optimizing stochastic losses formed over subsamples of the graph at each iteration of stochastic gradient descent. In this paper, we study the effects of adding an $\ell_2$ penalty of the embedding vectors to the training loss of these types of methods. We prove that, under some exchangeability assumptions on the graph, this asymptotically leads to learning a graphon with a nuclear-norm-type penalty, and give guarantees for the asymptotic distribution of the learned embedding vectors. In particular, the exact form of the penalty depends on the choice of subsampling method used as part of stochastic gradient descent. We also illustrate empirically that concatenating node covariates to $\ell_2$ regularized node2vec embeddings leads to comparable, when not superior, performance to methods which incorporate node covariates and the network structure in a non-linear manner..

----

## [1810] Differentiable hierarchical and surrogate gradient search for spiking neural networks

**Authors**: *Kaiwei Che, Luziwei Leng, Kaixuan Zhang, Jianguo Zhang, Qinghu Meng, Jie Cheng, Qinghai Guo, Jianxing Liao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e8c2895db691eaab85af37bddee75aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e8c2895db691eaab85af37bddee75aa-Abstract-Conference.html)

**Abstract**:

Spiking neural network (SNN) has been viewed as a potential candidate for the next generation of artificial intelligence with appealing characteristics such as sparse computation and inherent temporal dynamics. By adopting architectures of deep artificial neural networks (ANNs), SNNs are achieving competitive performances in benchmark tasks such as image classification. However, successful architectures of ANNs are not necessary ideal for SNN and when tasks become more diverse effective architectural variations could be critical. To this end, we develop a spike-based differentiable hierarchical search (SpikeDHS) framework, where spike-based computation is realized on both the cell and the layer level search space. Based on this framework, we find effective SNN architectures under limited computation cost. During the training of SNN, a suboptimal surrogate gradient function could lead to poor approximations of true gradients, making the network enter certain local minima. To address this problem, we extend the differential approach to surrogate gradient search where the SG function is efficiently optimized locally. Our models achieve state-of-the-art performances on classification of CIFAR10/100 and ImageNet with accuracy of 95.50%, 76.25% and 68.64%. On event-based deep stereo, our method finds optimal layer variation and surpasses the accuracy of specially designed ANNs meanwhile with 26$\times$ lower energy cost ($6.7\mathrm{mJ}$), demonstrating the advantage of SNN in processing highly sparse and dynamic signals. Codes are available at \url{https://github.com/Huawei-BIC/SpikeDHS}.

----

## [1811] On Embeddings for Numerical Features in Tabular Deep Learning

**Authors**: *Yury Gorishniy, Ivan Rubachev, Artem Babenko*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)

**Abstract**:

Recently, Transformer-like deep architectures have shown strong performance on tabular data problems. Unlike traditional models, e.g., MLP, these architectures map scalar values of numerical features to high-dimensional embeddings before mixing them in the main backbone. In this work, we argue that embeddings for numerical features are an underexplored degree of freedom in tabular DL, which allows constructing more powerful DL models and competing with gradient boosted decision trees (GBDT) on some GBDT-friendly benchmarks (that is, where GBDT outperforms conventional DL models). We start by describing two conceptually different approaches to building embedding modules: the first one is based on a piecewise linear encoding of scalar values, and the second one utilizes periodic activations. Then, we empirically demonstrate that these two approaches can lead to significant performance boosts compared to the embeddings based on conventional blocks such as linear layers and ReLU activations. Importantly, we also show that embedding numerical features is beneficial for many backbones, not only for Transformers. Specifically, after proper embeddings, simple MLP-like models can perform on par with the attention-based architectures. Overall, we highlight embeddings for numerical features as an important design aspect with good potential for further improvements in tabular DL. The source code is available at https://github.com/Yura52/tabular-dl-num-embeddings

----

## [1812] Visual Prompting via Image Inpainting

**Authors**: *Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, Alexei A. Efros*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

**Abstract**:

How does one adapt a pre-trained visual model to novel downstream tasks without task-specific finetuning or any model modification? Inspired by prompting in NLP, this paper investigates visual prompting: given input-output image example(s) of a new task at test time and a new input image, the goal is to automatically produce the output image, consistent with the given examples. We show that posing this problem as simple image inpainting -- literally just filling in a hole in a concatenated visual prompt image -- turns out to be surprisingly effective, provided that the inpainting algorithm has been trained on the right data. We train masked auto-encoders on a new dataset that we curated -- 88k unlabeled figures from academic papers sources on Arxiv. We apply visual prompting to these pretrained models and demonstrate results on various downstream image-to-image tasks, including foreground segmentation, single object detection, colorization, edge detection, etc. Project page: https://yossigandelsman.github.io/visual_prompt

----

## [1813] MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction

**Authors**: *Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, Andreas Geiger*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html)

**Abstract**:

In recent years, neural implicit surface reconstruction methods have become popular for multi-view 3D reconstruction. In contrast to traditional multi-view stereo methods, these approaches tend to produce smoother and more complete reconstructions due to the inductive smoothness bias of neural networks. State-of-the-art neural implicit methods allow for high-quality reconstructions of simple scenes from many input views. Yet, their performance drops significantly for larger and more complex scenes and scenes captured from sparse viewpoints. This is caused primarily by the inherent ambiguity in the RGB reconstruction loss that does not provide enough constraints, in particular in less-observed and textureless areas. Motivated by recent advances in the area of monocular geometry prediction, we systematically explore the utility these cues provide for improving neural implicit surface reconstruction. We demonstrate that depth and normal cues, predicted by general-purpose monocular estimators, significantly improve reconstruction quality and optimization time. Further, we analyse and investigate multiple design choices for representing neural implicit surfaces, ranging from monolithic MLP models over single-grid to multi-resolution grid representations. We observe that geometric monocular priors improve performance both for small-scale single-object as well as large-scale multi-object scenes, independent of the choice of representation.

----

## [1814] OpenAUC: Towards AUC-Oriented Open-Set Recognition

**Authors**: *Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f73d65a4186198152357be871345771-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f73d65a4186198152357be871345771-Abstract-Conference.html)

**Abstract**:

Traditional machine learning follows a close-set assumption that the training and test set share the same label space. While in many practical scenarios, it is inevitable that some test samples belong to unknown classes (open-set). To fix this issue, Open-Set Recognition (OSR), whose goal is to make correct predictions on both close-set samples and open-set samples, has attracted rising attention. In this direction, the vast majority of literature focuses on the pattern of open-set samples. However, how to evaluate model performance in this challenging task is still unsolved. In this paper, a systematic analysis reveals that most existing metrics are essentially inconsistent with the aforementioned goal of OSR: (1) For metrics extended from close-set classification, such as Open-set F-score, Youden's index, and Normalized Accuracy, a poor open-set prediction can escape from a low performance score with a superior close-set prediction. (2) Novelty detection AUC, which measures the ranking performance between close-set and open-set samples, ignores the close-set performance. To fix these issues, we propose a novel metric named OpenAUC. Compared with existing metrics, OpenAUC enjoys a concise pairwise formulation that evaluates open-set performance and close-set performance in a coupling manner. Further analysis shows that OpenAUC is free from the aforementioned inconsistency properties. Finally, an end-to-end learning method is proposed to minimize the OpenAUC risk, and the experimental results on popular benchmark datasets speak to its effectiveness.

----

## [1815] Reduction Algorithms for Persistence Diagrams of Networks: CoralTDA and PrunIT

**Authors**: *Cuneyt Gurcan Akcora, Murat Kantarcioglu, Yulia R. Gel, Baris Coskunuzer*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f81a6e7081497b2d458689a4ce39fc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f81a6e7081497b2d458689a4ce39fc7-Abstract-Conference.html)

**Abstract**:

Topological data analysis (TDA) delivers invaluable and complementary information on the intrinsic properties of data inaccessible to conventional methods. However, high computational costs remain the primary roadblock hindering the successful application of TDA in real-world studies, particularly with machine learning on large complex networks.Indeed, most modern networks such as citation, blockchain, and online social networks often have hundreds of thousands of vertices, making the application of existing TDA methods infeasible. We develop two new, remarkably simple but effective algorithms to compute the exact persistence diagrams of large graphs to address this major TDA limitation. First, we prove that $(k+1)$-core of a graph $G$ suffices to compute its $k^{th}$ persistence diagram, $PD_k(G)$. Second, we introduce a pruning algorithm for graphs to compute their persistence diagrams by removing the dominated vertices. Our experiments on large networks show that our novel approach can achieve computational gains up to 95%. The developed framework provides the first bridge between the graph theory and TDA, with applications in machine learning of large complex networks. Our implementation is available at https://github.com/cakcora/PersistentHomologyWithCoralPrunit.

----

## [1816] TreeMoCo: Contrastive Neuron Morphology Representation Learning

**Authors**: *Hanbo Chen, Jiawei Yang, Daniel Maxim Iascone, Lijuan Liu, Lei He, Hanchuan Peng, Jianhua Yao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f989633ffbd47a83caddacad0f0261f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f989633ffbd47a83caddacad0f0261f-Abstract-Conference.html)

**Abstract**:

Morphology of neuron trees is a key indicator to delineate neuronal cell-types, analyze brain development process, and evaluate pathological changes in neurological diseases. Traditional analysis mostly relies on heuristic features and visual inspections. A quantitative, informative, and comprehensive representation of neuron morphology is largely absent but desired. To fill this gap, in this work, we adopt a Tree-LSTM network to encode neuron morphology and introduce a self-supervised learning framework named TreeMoCo to learn features without the need for labels. We test TreeMoCo on 2403 high-quality 3D neuron reconstructions of mouse brains from three different public resources. Our results show that TreeMoCo is effective in both classifying major brain cell-types and identifying sub-types. To our best knowledge, TreeMoCo is the very first to explore learning the representation of neuron tree morphology with contrastive learning. It has a great potential to shed new light on quantitative neuron morphology analysis. Code is available at https://github.com/TencentAILabHealthcare/NeuronRepresentation.

----

## [1817] Compositional Generalization in Unsupervised Compositional Representation Learning: A Study on Disentanglement and Emergent Language

**Authors**: *Zhenlin Xu, Marc Niethammer, Colin Raffel*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9f9ecbf4062842df17ec3f4ea3ad7f54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9f9ecbf4062842df17ec3f4ea3ad7f54-Abstract-Conference.html)

**Abstract**:

Deep learning models struggle with compositional generalization, i.e. the ability to recognize or generate novel combinations of observed elementary concepts. In hopes of enabling compositional generalization, various unsupervised learning algorithms have been proposed with inductive biases that aim to induce compositional structure in learned representations (e.g. disentangled representation and emergent language learning). In this work, we evaluate these unsupervised learning algorithms in terms of how well they enable \textit{compositional generalization}. Specifically, our evaluation protocol focuses on whether or not it is easy to train a simple model on top of the learned representation that generalizes to new combinations of compositional factors. We systematically study three unsupervised representation learning algorithms - $\beta$-VAE, $\beta$-TCVAE, and emergent language (EL) autoencoders - on two datasets that allow directly testing compositional generalization. We find that directly using the bottleneck representation with simple models and few labels may lead to worse generalization than using representations from layers before or after the learned representation itself. In addition, we find that the previously proposed metrics for evaluating the levels of compositionality are not correlated with actual compositional generalization in our framework. Surprisingly, we find that increasing pressure to produce a disentangled representation (e.g. increasing $\beta$ in the $\beta$-VAE) produces representations with worse generalization, while representations from EL models show strong compositional generalization. Motivated by this observation, we further investigate the advantages of using EL to induce compositional structure in unsupervised representation learning, finding that it shows consistently stronger generalization than disentanglement models, especially when using less unlabeled data for unsupervised learning and fewer labels for downstream tasks. Taken together, our results shed new light onto the compositional generalization behavior of different unsupervised learning algorithms with a new setting to rigorously test this behavior, and suggest the potential benefits of developing EL learning algorithms for more generalizable representations. Our code is publicly available at https://github.com/wildphoton/Compositional-Generalization .

----

## [1818] Improving Zero-Shot Generalization in Offline Reinforcement Learning using Generalized Similarity Functions

**Authors**: *Bogdan Mazoure, Ilya Kostrikov, Ofir Nachum, Jonathan Tompson*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9fbdfded5c4d2969d889efc72f85c644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9fbdfded5c4d2969d889efc72f85c644-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) agents are widely used for solving complex sequential decision-making tasks, but still exhibit difficulty generalizing to scenarios not seen during training. While prior online approaches demonstrated that using additional signals beyond the reward function can lead to better generalization capabilities in RL agents, i.e. using self-supervised learning (SSL), they struggle in the offline RL setting, i.e. learning from a static dataset. We show that the performance of online algorithms for generalization in RL can be hindered in the offline setting due to poor estimation of similarity between observations. We propose a new theoretically-motivated framework called Generalized Similarity Functions (GSF), which uses contrastive learning to train an offline RL agent to aggregate observations based on the similarity of their expected future behavior, where we quantify this similarity using generalized value functions. We show that GSF is general enough to recover existing SSL objectives while improving zero-shot generalization performance on two complex pixel-based offline RL benchmarks.

----

## [1819] GAUDI: A Neural Architect for Immersive 3D Scene Generation

**Authors**: *Miguel Ángel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, Afshin Dehghan, Joshua M. Susskind*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html)

**Abstract**:

We introduce GAUDI, a generative model capable of capturing the distribution of complex and realistic 3D scenes that can be rendered immersively from a moving camera. We tackle this challenging problem with a scalable yet powerful approach, where we first optimize a latent representation that disentangles radiance fields and camera poses. This latent representation is then used to learn a generative model that enables both unconditional and conditional generation of 3D scenes. Our model generalizes previous works that focus on single objects by removing the assumption that the camera pose distribution can be shared across samples. We show that GAUDI obtains state-of-the-art performance in the unconditional generative setting across multiple datasets and allows for conditional generation of 3D scenes given conditioning variables like sparse image observations or text that describes the scene.

----

## [1820] Mask-based Latent Reconstruction for Reinforcement Learning

**Authors**: *Tao Yu, Zhizheng Zhang, Cuiling Lan, Yan Lu, Zhibo Chen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a0709efe5139939ab69902884ecad9c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a0709efe5139939ab69902884ecad9c1-Abstract-Conference.html)

**Abstract**:

For deep reinforcement learning (RL) from pixels, learning effective state representations is crucial for achieving high performance. However, in practice, limited experience and high-dimensional inputs prevent effective representation learning. To address this, motivated by the success of mask-based modeling in other research fields, we introduce mask-based reconstruction to promote state representation learning in RL. Specifically, we propose a simple yet effective self-supervised method, Mask-based Latent Reconstruction (MLR), to predict complete state representations in the latent space from the observations with spatially and temporally masked pixels. MLR enables better use of context information when learning state representations to make them more informative, which facilitates the training of RL agents. Extensive experiments show that our MLR significantly improves the sample efficiency in RL and outperforms the state-of-the-art sample-efficient RL methods on multiple continuous and discrete control benchmarks. Our code is available at https://github.com/microsoft/Mask-based-Latent-Reconstruction.

----

## [1821] Product Ranking for Revenue Maximization with Multiple Purchases

**Authors**: *Renzhe Xu, Xingxuan Zhang, Bo Li, Yafeng Zhang, Xiaolong Chen, Peng Cui*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a09e0dd6f92e402256725e15d3331811-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a09e0dd6f92e402256725e15d3331811-Abstract-Conference.html)

**Abstract**:

Product ranking is the core problem for revenue-maximizing online retailers. To design proper product ranking algorithms, various consumer choice models are proposed to characterize the consumers' behaviors when they are provided with a list of products. However, existing works assume that each consumer purchases at most one product or will keep viewing the product list after purchasing a product, which does not agree with the common practice in real scenarios. In this paper, we assume that each consumer can purchase multiple products at will. To model consumers' willingness to view and purchase, we set a random attention span and purchase budget, which determines the maximal amount of products that he/she views and purchases, respectively. Under this setting, we first design an optimal ranking policy when the online retailer can precisely model consumers' behaviors. Based on the policy, we further develop the Multiple-Purchase-with-Budget UCB (MPB-UCB) algorithms with $\tilde{O}(\sqrt{T})$ regret that estimate consumers' behaviors and maximize revenue simultaneously in online settings. Experiments on both synthetic and semi-synthetic datasets prove the effectiveness of the proposed algorithms.

----

## [1822] One Model to Edit Them All: Free-Form Text-Driven Image Manipulation with Semantic Modulations

**Authors**: *Yiming Zhu, Hongyu Liu, Yibing Song, Ziyang Yuan, Xintong Han, Chun Yuan, Qifeng Chen, Jue Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)

**Abstract**:

Free-form text prompts allow users to describe their intentions during image manipulation conveniently. Based on the visual latent space of StyleGAN[21] and text embedding space of CLIP[34], studies focus on how to map these two latent spaces for text-driven attribute manipulations. Currently, the latent mapping between these two spaces is empirically designed and confines that each manipulation model can only handle one fixed text prompt. In this paper, we propose a method named Free-Form CLIP (FFCLIP), aiming to  establish an automatic latent mapping so that one manipulation model handles free-form text prompts. Our FFCLIP has a cross-modality semantic modulation module containing semantic alignment and injection. The semantic alignment performs the automatic latent mapping via linear transformations with a cross attention mechanism. After alignment, we inject semantics from text prompt embeddings to the StyleGAN latent space. For one type of image (e.g., human portrait'), one FFCLIP model can be learned to handle free-form text prompts. Meanwhile, we observe that although each training text prompt only contains a single semantic meaning, FFCLIP can leverage text prompts with multiple semantic meanings for image manipulation. In the experiments, we evaluate FFCLIP on three types of images (i.e.,human portraits', cars', andchurches'). Both visual and numerical results show that FFCLIP effectively produces semantically accurate and visually realistic images. Project page:  https://github.com/KumapowerLIU/FFCLIP.

----

## [1823] Luckiness in Multiscale Online Learning

**Authors**: *Wouter M. Koolen, Muriel Felipe Pérez-Ortiz*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a0d2345b43e66fa946155c98899dc03b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a0d2345b43e66fa946155c98899dc03b-Abstract-Conference.html)

**Abstract**:

Algorithms for full-information online learning are classically tuned to minimize their worst-case regret. Modern algorithms additionally provide tighter guarantees outside the adversarial regime, most notably in the form of constant pseudoregret bounds under statistical margin assumptions. We investigate the multiscale extension of the problem where the loss ranges of the experts are vastly different. Here, the regret with respect to each expert needs to scale with its range, instead of the maximum overall range. We develop new multiscale algorithms, tuning schemes and analysis techniques to show that worst-case robustness and adaptation to easy data can be combined at a negligible cost. We further develop an extension with optimism and apply it to solve multiscale two-player zero-sum games. We demonstrate experimentally the superior performance of our scale-adaptive algorithm and discuss the subtle relationship of our results to Freund's 2016 open problem.

----

## [1824] Deep Active Learning by Leveraging Training Dynamics

**Authors**: *Haonan Wang, Wei Huang, Ziwei Wu, Hanghang Tong, Andrew Margenot, Jingrui He*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a102dd5931da01e1b40205490513304c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a102dd5931da01e1b40205490513304c-Abstract-Conference.html)

**Abstract**:

Active learning theories and methods have been extensively studied in classical statistical learning settings. However, deep active learning, i.e., active learning with deep learning models, is usually based on empirical criteria without solid theoretical justification, thus suffering from heavy doubts when some of those fail to provide benefits in applications. In this paper, by exploring the connection between the generalization performance and the training dynamics, we propose a theory-driven deep active learning method (dynamicAL) which selects samples to maximize training dynamics. In particular, we prove that the convergence speed of training and the generalization performance is positively correlated under the ultra-wide condition and show that maximizing the training dynamics leads to a better generalization performance. Furthermore, to scale up to large deep neural networks and data sets, we introduce two relaxations for the subset selection problem and reduce the time complexity from polynomial to constant. Empirical results show that dynamicAL not only outperforms the other baselines consistently but also scales well on large deep learning models. We hope our work inspires more attempts in bridging the theoretical findings of deep networks and practical impacts in deep active learning applications.

----

## [1825] Learning and Covering Sums of Independent Random Variables with Unbounded Support

**Authors**: *Alkis Kalavasis, Konstantinos Stavropoulos, Emmanouil Zampetakis*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html)

**Abstract**:

We study the problem of covering and learning sums $X = X_1 + \cdots + X_n$ of independent integer-valued random variables $X_i$ (SIIRVs) with infinite support. De et al. at FOCS 2018, showed that even when the collective support of $X_i$'s is of size $4$, the maximum value of the support necessarily appears in the sample complexity of learning $X$. In this work, we address two questions: (i) Are there general families of SIIRVs with infinite support that can be learned with sample complexity independent of both $n$ and the maximal element of the support? (ii) Are there general families of SIIRVs with infinite support that admit proper sparse covers in total variation distance? As for question (i), we provide a set of simple conditions that allow the infinitely supported SIIRV to be learned with complexity $ \text{poly}(1/\epsilon)$ bypassing the aforementioned lower bound. We further address question (ii) in the general setting where each variable $X_i$ has unimodal probability mass function and is a different member of some, possibly multi-parameter, exponential family $\mathcal{E}$ that satisfies some structural properties. These properties allow $\mathcal{E}$ to contain heavy tailed and non log-concave distributions. Moreover, we show that for every $\epsilon > 0$, and every $k$-parameter family $\mathcal{E}$ that satisfies some structural assumptions, there exists an algorithm with $\widetilde{O}(k) \cdot  \text{poly}(1/\epsilon)$ samples that learns a sum of $n$ arbitrary members of $\mathcal{E}$ within $\epsilon$ in TV distance. The output of the learning algorithm is also a sum of random variables within the family $\mathcal{E}$. En route, we prove that any discrete unimodal exponential family with bounded constant-degree central moments can be approximated by the family corresponding to a bounded subset of the initial (unbounded) parameter space.

----

## [1826] LGDN: Language-Guided Denoising Network for Video-Language Modeling

**Authors**: *Haoyu Lu, Mingyu Ding, Nanyi Fei, Yuqi Huo, Zhiwu Lu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a117a3cd54b7affad04618c77c2fb18b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a117a3cd54b7affad04618c77c2fb18b-Abstract-Conference.html)

**Abstract**:

Video-language modeling has attracted much attention with the rapid growth of web videos. Most existing methods assume that the video frames and text description are semantically correlated, and focus on video-language modeling at video level. However, this hypothesis often fails for two reasons: (1) With the rich semantics of video contents, it is difficult to cover all frames with a single video-level description; (2) A raw video typically has noisy/meaningless information (e.g., scenery shot, transition or teaser). Although a number of recent works deploy attention mechanism to alleviate this problem, the irrelevant/noisy information still makes it very difficult to address. To overcome such challenge, we thus propose an efficient and effective model, termed Language-Guided Denoising Network (LGDN), for video-language modeling. Different from most existing methods that utilize all extracted video frames, LGDN dynamically filters out the misaligned or redundant frames under the language supervision and obtains only 2--4 salient frames per video for cross-modal token-level alignment. Extensive experiments on five public datasets show that our LGDN outperforms the state-of-the-arts by large margins. We also provide detailed ablation study to reveal the critical importance of solving the noise issue, in hope of inspiring future video-language work.

----

## [1827] LieGG: Studying Learned Lie Group Generators

**Authors**: *Artem Moskalev, Anna Sepliarskaia, Ivan Sosnovik, Arnold W. M. Smeulders*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a120382cf4e2e06d94d7ae7ac96fbe25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a120382cf4e2e06d94d7ae7ac96fbe25-Abstract-Conference.html)

**Abstract**:

Symmetries built into a neural network have appeared to be very beneficial for a wide range of tasks as it saves the data to learn them. We depart from the position that when symmetries are not built into a model a priori, it is advantageous for robust networks to learn symmetries directly from the data to fit a task function. In this paper, we present a method to extract symmetries learned by a neural network and to evaluate the degree to which a network is invariant to them. With our method, we are able to explicitly retrieve learned invariances in a form of the generators of corresponding Lie-groups without prior knowledge of symmetries in the data. We use the proposed method to study how symmetrical properties depend on a neural network's parameterization and configuration. We found that the ability of a network to learn symmetries generalizes over a range of architectures. However, the quality of learned symmetries depends on the depth and the number of parameters.

----

## [1828] FourierNets enable the design of highly non-local optical encoders for computational imaging

**Authors**: *Diptodip Deb, Zhenfei Jiao, Ruth R. Sims, Alex B. Chen, Michael Broxton, Misha B. Ahrens, Kaspar Podgorski, Srinivas C. Turaga*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1263ffa557506ea29c54481788d518f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1263ffa557506ea29c54481788d518f-Abstract-Conference.html)

**Abstract**:

Differentiable simulations of optical systems can be combined with deep learning-based reconstruction networks to enable high performance computational imaging via end-to-end (E2E) optimization of both the optical encoder and the deep decoder. This has enabled imaging applications such as 3D localization microscopy, depth estimation, and lensless photography via the optimization of local optical encoders. More challenging computational imaging applications, such as 3D snapshot microscopy which compresses 3D volumes into single 2D images, require a highly non-local optical encoder. We show that existing deep network decoders have a locality bias which prevents the optimization of such highly non-local optical encoders. We address this with a decoder based on a shallow neural network architecture using global kernel Fourier convolutional neural networks (FourierNets). We show that FourierNets surpass existing deep network based decoders at reconstructing photographs captured by the highly non-local DiffuserCam optical encoder. Further, we show that FourierNets enable E2E optimization of highly non-local optical encoders for 3D snapshot microscopy. By combining FourierNets with a large-scale multi-GPU differentiable optical simulation, we are able to optimize non-local optical encoders 170$\times$ to 7372$\times$ larger than prior state of the art, and demonstrate the potential for ROI-type specific optical encoding with a programmable microscope.

----

## [1829] Benign Overfitting in Two-layer Convolutional Neural Networks

**Authors**: *Yuan Cao, Zixiang Chen, Misha Belkin, Quanquan Gu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a12c999be280372b157294e72a4bbc8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a12c999be280372b157294e72a4bbc8b-Abstract-Conference.html)

**Abstract**:

Modern neural networks often have great expressive power and can be trained to overfit the training data, while still achieving a good test performance. This phenomenon is referred to as “benign overfitting”. Recently, there emerges a line of works studying “benign overfitting” from the theoretical perspective. However, they are limited to linear models or kernel/random feature models, and there is still a lack of theoretical understanding about when and how benign overfitting occurs in neural networks. In this paper, we study the benign overfitting phenomenon in training a two-layer convolutional neural network (CNN). We show that when the signal-to-noise ratio satisfies a certain condition, a two-layer CNN trained by gradient descent can achieve arbitrarily small training and test loss. On the other hand, when this condition does not hold, overfitting becomes harmful and the obtained CNN can only achieve a constant level test loss. These together demonstrate a sharp phase transition between benign overfitting and harmful overfitting, driven by the signal-to-noise ratio. To the best of our knowledge, this is the first work that precisely characterizes the conditions under which benign overfitting can occur in training convolutional neural networks.

----

## [1830] Discovery of Single Independent Latent Variable

**Authors**: *Uri Shaham, Jonathan Svirsky, Ori Katz, Ronen Talmon*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a12e362d89d4e0b40760f839f91550ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a12e362d89d4e0b40760f839f91550ee-Abstract-Conference.html)

**Abstract**:

Latent variable discovery is a central problem in data analysis with a broad range of applications in applied science.In this work, we consider data given as an invertible mixture of two statistically independent components, and assume that one of the components is observed while the other is hidden. Our goal is to recover the hidden component.For this purpose, we propose an autoencoder equipped with a discriminator.Unlike the standard nonlinear ICA problem, which was shown to be non-identifiable, in the  special case of ICA we consider here, we show that our approach can recover the component of interest up to entropy-preserving transformation.We demonstrate the performance of the proposed approach in several tasks, including image synthesis, voice cloning, and fetal ECG extraction.

----

## [1831] Meta-ticket: Finding optimal subnetworks for few-shot learning within randomly initialized neural networks

**Authors**: *Daiki Chijiwa, Shin'ya Yamaguchi, Atsutoshi Kumagai, Yasutoshi Ida*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1563f83b580b83d2836abc6ea03280a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1563f83b580b83d2836abc6ea03280a-Abstract-Conference.html)

**Abstract**:

Few-shot learning for neural networks (NNs) is an important problem that aims to train NNs with a few data. The main challenge is how to avoid overfitting since over-parameterized NNs can easily overfit to such small dataset. Previous work (e.g. MAML by Finn et al. 2017) tackles this challenge by meta-learning, which learns how to learn from a few data by using various tasks. On the other hand, one conventional approach to avoid overfitting is restricting hypothesis spaces by endowing sparse NN structures like convolution layers in computer vision. However, although such manually-designed sparse structures are sample-efficient for sufficiently large datasets, they are still insufficient for few-shot learning. Then the following questions naturally arise: (1) Can we find sparse structures effective for few-shot learning by meta-learning? (2) What benefits will it bring in terms of meta-generalization? In this work, we propose a novel meta-learning approach, called Meta-ticket, to find optimal sparse subnetworks for few-shot learning within randomly initialized NNs. We empirically validated that Meta-ticket successfully discover sparse subnetworks that can learn specialized features for each given task. Due to this task-wise adaptation ability, Meta-ticket achieves superior meta-generalization compared to MAML-based methods especially with large NNs.

----

## [1832] LAION-5B: An open large-scale dataset for training next generation image-text models

**Authors**: *Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, Jenia Jitsev*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Groundbreaking language-vision architectures like CLIP and DALL-E proved the utility of training on large amounts of noisy image-text data, without relying on expensive accurate labels used in standard vision unimodal supervised learning. The resulting models showed capabilities of strong text-guided image generation and transfer to downstream tasks, while performing remarkably at zero-shot classification with noteworthy out-of-distribution robustness. Since then, large-scale language-vision models like ALIGN, BASIC, GLIDE, Flamingo and Imagen made further improvements. Studying the training and capabilities of such models requires datasets containing billions of image-text pairs. Until now, no datasets of this size have been made openly available for the broader research community. To address this problem and democratize research on large-scale multi-modal models, we present LAION-5B - a dataset consisting of 5.85 billion CLIP-filtered image-text pairs, of which 2.32B contain English language. We show successful replication and fine-tuning of foundational models like CLIP, GLIDE and Stable Diffusion using the dataset, and discuss further experiments enabled with an openly available dataset of this scale. Additionally we provide several nearest neighbor indices, an improved web-interface for dataset exploration and subset generation, and detection scores for watermark, NSFW, and toxic content detection.

----

## [1833] Constants of motion network

**Authors**: *Muhammad Firmansyah Kasim, Yi Heng Lim*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1a90fbba98b417c7cf53e75eb4ac933-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1a90fbba98b417c7cf53e75eb4ac933-Abstract-Conference.html)

**Abstract**:

The beauty of physics is that there is usually a conserved quantity in an always-changing system, known as the constant of motion. Finding the constant of motion is important in understanding the dynamics of the system, but typically requires mathematical proficiency and manual analytical work. In this paper, we present a neural network that can simultaneously learn the dynamics of the system and the constants of motion from data. By exploiting the discovered constants of motion, it can produce better predictions on dynamics and can work on a wider range of systems than Hamiltonian-based neural networks. In addition, the training progresses of our method can be used as an indication of the number of constants of motion in a system which could be useful in studying a novel physical system.

----

## [1834] HUMUS-Net: Hybrid Unrolled Multi-scale Network Architecture for Accelerated MRI Reconstruction

**Authors**: *Zalan Fabian, Berk Tinaz, Mahdi Soltanolkotabi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1bb3f96e255ae1e04325ae166bcef0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1bb3f96e255ae1e04325ae166bcef0f-Abstract-Conference.html)

**Abstract**:

In accelerated MRI reconstruction, the anatomy of a patient is recovered from a set of undersampled and noisy measurements. Deep learning approaches have been proven to be successful in solving this ill-posed inverse problem and are capable of producing very high quality reconstructions. However, current architectures heavily rely on convolutions, that are content-independent and have difficulties modeling long-range dependencies in images. Recently, Transformers, the workhorse of contemporary natural language processing, have emerged as powerful building blocks for a multitude of vision tasks. These models split input images into non-overlapping patches, embed the patches into lower-dimensional tokens and utilize a self-attention mechanism that does not suffer from the aforementioned weaknesses of convolutional architectures. However, Transformers incur extremely high compute and memory cost when 1) the input image resolution is high and 2) when the image needs to be split into a large number of patches to preserve fine detail information, both of which are typical in low-level vision problems such as MRI reconstruction, having a compounding effect. To tackle these challenges, we propose HUMUS-Net, a hybrid architecture that combines the beneficial implicit bias and efficiency of convolutions with the power of Transformer blocks in an unrolled and multi-scale network. HUMUS-Net extracts high-resolution features via convolutional blocks and refines low-resolution features via a novel Transformer-based multi-scale feature extractor. Features from both levels are then synthesized into a high-resolution output reconstruction. Our network establishes new state of the art on the largest publicly available MRI dataset, the fastMRI dataset. We further demonstrate the performance of HUMUS-Net on two other popular MRI datasets and perform fine-grained ablation studies to validate our design.

----

## [1835] A Damped Newton Method Achieves Global $\mathcal O \left(\frac{1}{k^2}\right)$ and Local Quadratic Convergence Rate

**Authors**: *Slavomír Hanzely, Dmitry Kamzolov, Dmitry Pasechnyuk, Alexander V. Gasnikov, Peter Richtárik, Martin Takác*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html)

**Abstract**:

In this paper, we present the first stepsize schedule for Newton method resulting in fast global and local convergence guarantees. In particular, we a) prove an $\mathcal O \left( 1/{k^2} \right)$ global rate, which matches the state-of-the-art global rate of cubically regularized Newton method of Polyak and Nesterov (2006) and of regularized Newton method of Mishchenko (2021), and the later variant of Doikov and Nesterov (2021), b) prove a local quadratic rate, which matches the best-known local rate of second-order methods, and c) our stepsize formula is simple, explicit, and does not require solving any subproblem. Our convergence proofs hold under affine-invariant assumptions closely related to the notion of self-concordance. Finally, our method has competitive performance when compared to existing baselines which share the same fast global convergence guarantees.

----

## [1836] Disentangling the Predictive Variance of Deep Ensembles through the Neural Tangent Kernel

**Authors**: *Seijin Kobayashi, Pau Vilimelis Aceituno, Johannes von Oswald*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference.html)

**Abstract**:

Identifying unfamiliar inputs, also known as out-of-distribution (OOD) detection, is a crucial property of any decision making process. A simple and empirically validated technique is based on deep ensembles where the variance of predictions over different neural networks acts as a substitute for input uncertainty. Nevertheless, a theoretical understanding of the inductive biases leading to the performance of deep ensemble's uncertainty estimation is missing. To improve our description of their behavior, we study deep ensembles with large layer widths operating in simplified linear training regimes, in which the functions trained with gradient descent can be described by the neural tangent kernel. We identify two sources of noise, each inducing a distinct inductive bias in the predictive variance at initialization. We further show theoretically and empirically that both noise sources affect the predictive variance of non-linear deep ensembles in toy models and realistic settings after training. Finally, we propose practical ways to eliminate part of these noise sources leading to significant changes and improved OOD detection in trained deep ensembles.

----

## [1837] High-dimensional limit theorems for SGD: Effective dynamics and critical scaling

**Authors**: *Gérard Ben Arous, Reza Gheissari, Aukosh Jagannath*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a224ff18cc99a71751aa2b79118604da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a224ff18cc99a71751aa2b79118604da-Abstract-Conference.html)

**Abstract**:

We study the scaling limits of stochastic gradient descent (SGD) with constant step-size in the high-dimensional regime. We prove limit theorems for the trajectories of summary statistics (i.e., finite-dimensional functions) of SGD as the dimension goes to infinity. Our approach allows one to choose the summary statistics that are tracked, the initialization, and the step-size. It yields both ballistic (ODE) and diffusive (SDE) limits, with the limit depending dramatically on the former choices. We find a critical scaling regime for the step-size below which this ``effective dynamics" matches gradient flow for the population loss, but at which, a new correction term appears which changes the phase diagram. About the fixed points of this effective dynamics, the corresponding diffusive limits can be quite complex and even degenerate. We demonstrate our approach on popular examples including estimation for spiked matrix and tensor models and classification via two-layer networks for binary and XOR-type Gaussian mixture models. These examples exhibit surprising phenomena including multimodal timescales to convergence as well as convergence to sub-optimal solutions with probability bounded away from zero from random (e.g., Gaussian) initializations.

----

## [1838] Online Deep Equilibrium Learning for Regularization by Denoising

**Authors**: *Jiaming Liu, Xiaojian Xu, Weijie Gan, Shirin Shoushtari, Ulugbek Kamilov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a2440e23f6a8c037eff1dc4f1156aa35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a2440e23f6a8c037eff1dc4f1156aa35-Abstract-Conference.html)

**Abstract**:

Plug-and-Play Priors (PnP) and Regularization by Denoising (RED) are widely-used frameworks for solving imaging inverse problems by computing fixed-points of operators combining physical measurement models and learned image priors. While traditional PnP/RED formulations have focused on priors specified using image denoisers, there is a growing interest in learning PnP/RED priors that are end-to-end optimal. The recent Deep Equilibrium Models (DEQ) framework has enabled memory-efficient end-to-end learning of PnP/RED priors by implicitly differentiating through the fixed-point equations without storing intermediate activation values.  However, the dependence of the computational/memory complexity of the measurement models in PnP/RED on the total number of measurements leaves DEQ impractical for many imaging applications. We propose ODER as a new strategy for improving the efficiency of DEQ through stochastic approximations of the measurement models. We theoretically analyze ODER giving insights into its convergence and ability to approximate the traditional DEQ approach. Our numerical results suggest the potential improvements in training/testing complexity due to ODER on three distinct imaging applications.

----

## [1839] Semantic Exploration from Language Abstractions and Pretrained Representations

**Authors**: *Allison C. Tam, Neil C. Rabinowitz, Andrew K. Lampinen, Nicholas A. Roy, Stephanie C. Y. Chan, DJ Strouse, Jane Wang, Andrea Banino, Felix Hill*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a28e024ccd623ed113fb19683fa0910d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a28e024ccd623ed113fb19683fa0910d-Abstract-Conference.html)

**Abstract**:

Effective exploration is a challenge in reinforcement learning (RL). Novelty-based exploration methods can suffer in high-dimensional state spaces, such as continuous partially-observable 3D environments. We address this challenge by defining novelty using semantically meaningful state abstractions, which can be found in learned representations shaped by natural language. In particular, we evaluate vision-language representations, pretrained on natural image captioning datasets. We show that these pretrained representations drive meaningful, task-relevant exploration and improve performance on 3D simulated environments. We also characterize why and how language provides useful abstractions for exploration by considering the impacts of using representations from a pretrained model, a language oracle, and several ablations. We demonstrate the benefits of our approach with on- and off-policy RL algorithms and in two very different task domains---one that stresses the identification and manipulation of everyday objects, and one that requires navigational exploration in an expansive world. Our results suggest that using language-shaped representations could improve exploration for various algorithms and agents in challenging environments.

----

## [1840] Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

**Authors**: *Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Wang, Mu Li, Dit-Yan Yeung*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html)

**Abstract**:

Conventionally, Earth system (e.g., weather and climate) forecasting relies on numerical simulation with complex physical models and hence is both expensive in computation and demanding on domain expertise. With the explosive growth of spatiotemporal Earth observation data in the past decade, data-driven models that apply Deep Learning (DL) are demonstrating impressive potential for various Earth system forecasting tasks. The Transformer as an emerging DL architecture, despite its broad success in other domains, has limited adoption in this area. In this paper, we propose Earthformer, a space-time Transformer for Earth system forecasting. Earthformer is based on a generic, flexible and efficient space-time attention block, named Cuboid Attention. The idea is to decompose the data into cuboids and apply cuboid-level self-attention in parallel. These cuboids are further connected with a collection of global vectors. We conduct experiments on the MovingMNIST dataset and a newly proposed chaotic $N$-body MNIST dataset to verify the effectiveness of cuboid attention and figure out the best design of Earthformer. Experiments on two real-world benchmarks about precipitation nowcasting and El Ni√±o/Southern Oscillation (ENSO) forecasting show that Earthformer achieves state-of-the-art performance.

----

## [1841] Benchopt: Reproducible, efficient and collaborative optimization benchmarks

**Authors**: *Thomas Moreau, Mathurin Massias, Alexandre Gramfort, Pierre Ablin, Pierre-Antoine Bannier, Benjamin Charlier, Mathieu Dagréou, Tom Dupré la Tour, Ghislain Durif, Cássio F. Dantas, Quentin Klopfenstein, Johan Larsson, En Lai, Tanguy Lefort, Benoît Malézieux, Badr Moufad, Binh T. Nguyen, Alain Rakotomamonjy, Zaccharie Ramzi, Joseph Salmon, Samuel Vaiter*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a30769d9b62c9b94b72e21e0ca73f338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a30769d9b62c9b94b72e21e0ca73f338-Abstract-Conference.html)

**Abstract**:

Numerical validation is at the core of machine learning research as it allows us to assess the actual impact of new methods, and to confirm the agreement between theory and practice. Yet, the rapid development of the field poses several challenges: researchers are confronted with a profusion of methods to compare, limited transparency and consensus on best practices, as well as tedious re-implementation work. As a result, validation is often very partial, which can lead to wrong conclusions that slow down the progress of research. We propose Benchopt, a collaborative framework to automatize, publish and reproduce optimization benchmarks in machine learning across programming languages and hardware architectures. Benchopt simplifies benchmarking for the community by providing an off-the-shelf tool for running, sharing and extending experiments. To demonstrate its broad usability, we showcase benchmarks on three standard ML tasks: $\ell_2$-regularized logistic regression, Lasso and ResNet18 training for image classification. These benchmarks highlight key practical findings that give a more nuanced view of state-of-the-art for these problems, showing that for practical evaluation, the devil is in the details.

----

## [1842] SketchBoost: Fast Gradient Boosted Decision Tree for Multioutput Problems

**Authors**: *Leonid Iosipoi, Anton Vakhrushev*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html)

**Abstract**:

Gradient Boosted Decision Tree (GBDT) is a widely-used machine learning algorithm that has been shown to achieve state-of-the-art results on many standard data science problems. We are interested in its application to multioutput problems when the output is highly multidimensional. Although there are highly effective GBDT implementations, their scalability to such problems is still unsatisfactory. In this paper, we propose novel methods aiming to accelerate the training process of GBDT in the multioutput scenario. The idea behind these methods lies in the approximate computation of a scoring function used to find the best split of decision trees. These methods are implemented in SketchBoost, which itself is integrated into our easily customizable Python-based GPU implementation of GBDT called Py-Boost. Our numerical study demonstrates that SketchBoost speeds up the training process of GBDT by up to over 40 times while achieving comparable or even better performance.

----

## [1843] Deep Attentive Belief Propagation: Integrating Reasoning and Learning for Solving Constraint Optimization Problems

**Authors**: *Yanchen Deng, Shufeng Kong, Caihua Liu, Bo An*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a375e3cb803e0d78fda4bb3933bd3a3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a375e3cb803e0d78fda4bb3933bd3a3a-Abstract-Conference.html)

**Abstract**:

Belief Propagation (BP) is an important message-passing algorithm for various reasoning tasks over graphical models, including solving the Constraint Optimization Problems (COPs). It has been shown that BP can achieve state-of-the-art performance on various benchmarks by mixing old and new messages before sending the new one, i.e., damping. However, existing methods on tuning a static damping factor for BP not only is laborious but also harms their performance. Moreover, existing BP  algorithms treat each variable node's neighbors equally when composing a new message, which also limits their exploration ability. To address these issues, we seamlessly integrate BP, Gated Recurrent Units (GRUs), and Graph Attention Networks (GATs) within the massage-passing framework to reason about dynamic weights and damping factors for composing new BP messages. Our model, Deep Attentive Belief Propagation (DABP), takes the factor graph and the BP messages in each iteration as the input and infers the optimal weights and damping factors through GRUs and GATs, followed by a multi-head attention layer. Furthermore, unlike existing neural-based BP variants, we propose a novel self-supervised learning algorithm for DABP with a smoothed solution cost, which does not require expensive training labels and also avoids the common out-of-distribution issue through efficient online learning. Extensive experiments show that our model significantly outperforms state-of-the-art baselines.

----

## [1844] Conservative Dual Policy Optimization for Efficient Model-Based Reinforcement Learning

**Authors**: *Shenao Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a3769fddee1b20552d2490c4ff18b136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a3769fddee1b20552d2490c4ff18b136-Abstract-Conference.html)

**Abstract**:

Provably efficient Model-Based Reinforcement Learning (MBRL) based on optimism or posterior sampling (PSRL) is ensured to attain the global optimality asymptotically by introducing the complexity measure of the model. However, the complexity might grow exponentially for the simplest nonlinear models, where global convergence is impossible within finite iterations. When the model suffers a large generalization error, which is quantitatively measured by the model complexity, the uncertainty can be large. The sampled model that current policy is greedily optimized upon will thus be unsettled, resulting in aggressive policy updates and over-exploration. In this work, we propose Conservative Dual Policy Optimization (CDPO) that involves a Referential Update and a Conservative Update. The policy is first optimized under a reference model, which imitates the mechanism of PSRL while offering more stability. A conservative range of randomness is guaranteed by maximizing the expectation of model value. Without harmful sampling procedures, CDPO can still achieve the same regret as PSRL. More importantly, CDPO enjoys monotonic policy improvement and global optimality simultaneously. Empirical results also validate the exploration efficiency of CDPO.

----

## [1845] Decentralized Training of Foundation Models in Heterogeneous Environments

**Authors**: *Binhang Yuan, Yongjun He, Jared Davis, Tianyi Zhang, Tri Dao, Beidi Chen, Percy Liang, Christopher Ré, Ce Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a37d615b61f999a5fa276adb14643476-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a37d615b61f999a5fa276adb14643476-Abstract-Conference.html)

**Abstract**:

Training foundation models, such as GPT-3 and PaLM, can be extremely expensive, often involving tens of thousands of GPUs running continuously for months. These models are typically trained in specialized clusters featuring fast, homogeneous interconnects and using carefully designed software systems that support both data parallelism and model/pipeline parallelism. Such dedicated clusters can be costly and difficult to obtain. Can we instead leverage the much greater amount of decentralized, heterogeneous, and lower-bandwidth interconnected compute? Previous works examining the heterogeneous, decentralized setting focus on relatively small models that can be trained in a purely data parallel manner. State-of-the-art schemes for model parallel foundation model training, such as Megatron and Deepspeed, only consider the homogeneous data center setting. In this paper, we present the first study of training large foundation models with model parallelism in a decentralized regime over a heterogeneous network. Our key technical contribution is a scheduling algorithm that allocates different computational “tasklets” in the training of foundation models to a group of decentralized GPU devices connected by a slow heterogeneous network. We provide a formal cost model and further propose an efficient evolutionary algorithm to find the optimal allocation strategy. We conduct extensive experiments that represent different scenarios for learning over geo-distributed devices simulated using real-world network measurements. In the most extreme case, across 8 different cities spanning 3 continents, our approach is 4.8× faster than prior state-of-the-art training systems.

----

## [1846] Cross Aggregation Transformer for Image Restoration

**Authors**: *Zheng Chen, Yulun Zhang, Jinjin Gu, Yongbing Zhang, Linghe Kong, Xin Yuan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a37fea8e67f907311826bc1ba2654d97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a37fea8e67f907311826bc1ba2654d97-Abstract-Conference.html)

**Abstract**:

Recently, Transformer architecture has been introduced into image restoration to replace convolution neural network (CNN) with surprising results. Considering the high computational complexity of Transformer with global attention, some methods use the local square window to limit the scope of self-attention. However, these methods lack direct interaction among different windows, which limits the establishment of long-range dependencies. To address the above issue, we propose a new image restoration model, Cross Aggregation Transformer (CAT). The core of our CAT is the Rectangle-Window Self-Attention (Rwin-SA), which utilizes horizontal and vertical rectangle window attention in different heads parallelly to expand the attention area and aggregate the features cross different windows. We also introduce the Axial-Shift operation for different window interactions. Furthermore, we propose the Locality Complementary Module to complement the self-attention mechanism, which incorporates the inductive bias of CNN (e.g., translation invariance and locality) into Transformer, enabling global-local coupling. Extensive experiments demonstrate that our CAT outperforms recent state-of-the-art methods on several image restoration applications. The code and models are available at https://github.com/zhengchen1999/CAT.

----

## [1847] Neural Payoff Machines: Predicting Fair and Stable Payoff Allocations Among Team Members

**Authors**: *Daphne Cornelisse, Thomas Rood, Yoram Bachrach, Mateusz Malinowski, Tal Kachman*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a38df2dd882bf7059a1914dd5547af87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a38df2dd882bf7059a1914dd5547af87-Abstract-Conference.html)

**Abstract**:

In many multi-agent settings, participants can form teams to achieve collective outcomes that may far surpass their individual capabilities. Measuring the relative contributions of agents and allocating them shares of the reward that promote long-lasting cooperation are difficult tasks. Cooperative game theory offers solution concepts identifying distribution schemes, such as the Shapley value, that fairly reflect the contribution of individuals to the performance of the team or the Core, which reduces the incentive of agents to abandon their team. Applications of such methods include identifying influential features and sharing the costs of joint ventures or team formation. Unfortunately, using these solutions requires tackling a computational barrier as they are hard to compute, even in restricted settings. In this work, we show how cooperative game-theoretic solutions can be distilled into a learned model by training neural networks to propose fair and stable payoff allocations. We show that our approach creates models that can generalize to games far from the training distribution and can predict solutions for more players than observed during training. An important application of our framework is Explainable AI: our approach can be used to speed-up Shapley value computations on many instances.

----

## [1848] Learning Recourse on Instance Environment to Enhance Prediction Accuracy

**Authors**: *Lokesh Nagalapatti, Guntakanti Sai Koushik, Abir De, Sunita Sarawagi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html)

**Abstract**:

Machine Learning models are often susceptible to poor performance on instances sampled from bad environments. For example, an image classifier could provide low accuracy on images captured under low lighting conditions. In high stake ML applications, such as AI-driven medical diagnostics, a better option could be to provide recourse in the form of  alternative environment settings in which to recapture the instance for more reliable diagnostics. In this paper, we propose a model called {\em RecourseNet} that learns to apply recourse on the space of environments so that the recoursed instances are amenable to better predictions by the classifier.   Learning to output optimal recourse is challenging because we do not assume access to the underlying physical process that generates the recoursed instances. Also, the optimal setting could be instance-dependent --- for example the best camera angle for object recognition could be a function of the object's shape. We propose a novel three-level training method that (a) Learns a classifier that is optimized for high performance under recourse, (b) Learns a recourse predictor when the training data may contain only limited instances under good environment settings, and (c) Triggers recourse selectively only when recourse is likely to improve classifier confidence.

----

## [1849] Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification

**Authors**: *Dandan Guo, Zhuo Li, Meixi Zheng, He Zhao, Mingyuan Zhou, Hongyuan Zha*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a39a9aceda771cded859ae7560530e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a39a9aceda771cded859ae7560530e09-Abstract-Conference.html)

**Abstract**:

Imbalanced data pose challenges for deep learning based classification models. One of the most widely-used approaches for tackling imbalanced data is re-weighting, where training samples are associated with different weights in the loss function. Most of existing re-weighting approaches treat the example weights as the learnable parameter and optimize the weights on the meta set, entailing expensive bilevel optimization. In this paper, we propose a novel re-weighting method based on optimal transport (OT) from a distributional point of view. Specifically, we view the training set as an imbalanced distribution over its samples, which is transported by OT to a balanced distribution obtained from the meta set. The weights of the training samples are the probability mass of the imbalanced distribution andlearned by minimizing the OT distance between the two distributions. Compared with existing methods, our proposed one disengages the dependence of the weight learning on the concerned classifier at each iteration. Experiments on image, text and point cloud datasets demonstrate that our proposed re-weighting method has excellent performance, achieving state-of-the-art results in many cases andproviding a promising tool for addressing the imbalanced classification issue. The code has been made available athttps://github.com/DandanGuo1993/reweight-imbalance-classification-with-OT.

----

## [1850] DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems

**Authors**: *Ruizhong Qiu, Zhiqing Sun, Yiming Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html)

**Abstract**:

Recently, deep reinforcement learning (DRL) models have shown promising results in solving NP-hard Combinatorial Optimization (CO) problems. However, most DRL solvers can only scale to a few hundreds of nodes for combinatorial optimization problems on graphs, such as the Traveling Salesman Problem (TSP).   This paper addresses the scalability challenge in large-scale combinatorial optimization by proposing a novel approach, namely, DIMES. Unlike previous DRL methods which suffer from costly autoregressive decoding or iterative refinements of discrete solutions, DIMES introduces a compact continuous space for parameterizing the underlying distribution of candidate solutions. Such a continuous space allows stable REINFORCE-based training and fine-tuning via massively parallel sampling. We further propose a meta-learning framework to enable the effective initialization of model parameters in the fine-tuning stage. Extensive experiments show that DIMES outperforms recent DRL-based methods on large benchmark datasets for Traveling Salesman Problems and Maximal Independent Set problems.

----

## [1851] Confident Approximate Policy Iteration for Efficient Local Planning in $q^\pi$-realizable MDPs

**Authors**: *Gellért Weisz, András György, Tadashi Kozuno, Csaba Szepesvári*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a3bfb116214815682a0d0d88ea95cd12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a3bfb116214815682a0d0d88ea95cd12-Abstract-Conference.html)

**Abstract**:

We consider approximate dynamic programming in $\gamma$-discounted Markov decision processes and apply it to approximate planning with linear value-function approximation. Our first contribution is a new variant of Approximate Policy Iteration (API), called Confident Approximate Policy Iteration (CAPI), which computes a deterministic stationary policy with an optimal error bound scaling linearly with the product of the effective horizon $H$ and the worst-case approximation error  $\epsilon$ of the action-value functions of stationary policies. This improvement over API (whose error scales with $H^2$) comes at the price of an $H$-fold increase in memory cost. Unlike Scherrer and Lesner [2012], who recommended computing a non-stationary policy to achieve a similar improvement (with the same memory overhead), we are able to stick to stationary policies. This allows for our second contribution, the application of CAPI to planning with local access to a simulator and $d$-dimensional linear function approximation. As such, we design a planning algorithm that applies CAPI to obtain a sequence of policies with successively refined accuracies on a dynamically evolving set of states. The algorithm outputs an $\tilde O(\sqrt{d}H\epsilon)$-optimal policy after issuing $\tilde O(dH^4/\epsilon^2)$ queries to the simulator, simultaneously achieving the optimal accuracy bound and the best known query complexity bound, while earlier algorithms in the literature achieve only one of them. This query complexity is shown to be tight in all parameters except $H$. These improvements come at the expense of a mild (polynomial) increase in memory and computational costs of both the algorithm and its output policy.

----

## [1852] Finding Correlated Equilibrium of Constrained Markov Game: A Primal-Dual Approach

**Authors**: *Ziyi Chen, Shaocong Ma, Yi Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a3f8f584febcc88ed8cdeb30b096db34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a3f8f584febcc88ed8cdeb30b096db34-Abstract-Conference.html)

**Abstract**:

Constrained Markov game is a fundamental problem that covers many applications, where multiple players compete with each other under behavioral constraints. The existing literature has proved the existence of Nash equilibrium for constrained Markov games, which turns out to be PPAD-complete and cannot be computed in polynomial time. In this work, we propose a surrogate notion of correlated equilibrium (CE) for constrained Markov games that can be computed in polynomial time, and study its fundamental properties. We show that the modification structure of CE of constrained Markov games is fundamentally different from that of unconstrained Markov games. Moreover, we prove that the corresponding Lagrangian function has zero duality gap. Based on this result, we develop the first primal-dual algorithm that provably converges to CE of constrained Markov games. In particular, we prove that both the duality gap and the constraint violation of the output policy converge at the rate $\mathcal{O}(\frac{1}{\sqrt{T}})$. Moreover, when adopting the V-learning algorithm as the subroutine in the primal update, our algorithm achieves an approximate CE with $\epsilon$ duality gap with the sample complexity $\mathcal{O}(H^9|\mathcal{S}||\mathcal{A}|^{2} \epsilon^{-4})$.

----

## [1853] NSNet: A General Neural Probabilistic Framework for Satisfiability Problems

**Authors**: *Zhaoyu Li, Xujie Si*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a40462acc6959034c6aa6dfb8e696415-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a40462acc6959034c6aa6dfb8e696415-Abstract-Conference.html)

**Abstract**:

We present the Neural Satisfiability Network (NSNet), a general neural framework that models satisfiability problems as probabilistic inference and meanwhile exhibits proper explainability. Inspired by the Belief Propagation (BP), NSNet uses a novel graph neural network (GNN) to parameterize BP in the latent space, where its hidden representations maintain the same probabilistic interpretation as BP.  NSNet can be flexibly configured to solve both SAT and #SAT problems by applying different learning objectives. For SAT, instead of directly predicting a satisfying assignment, NSNet performs marginal inference among all satisfying solutions, which we empirically find is more feasible for neural networks to learn. With the estimated marginals, a satisfying assignment can be efficiently generated by rounding and executing a stochastic local search. For #SAT, NSNet performs approximate model counting by learning the Bethe approximation of the partition function. Our evaluations show that NSNet achieves competitive results in terms of inference accuracy and time efficiency on multiple SAT and #SAT datasets.

----

## [1854] Brain Network Transformer

**Authors**: *Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, Carl Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a408234a9b80604a9cf6ca518e474550-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a408234a9b80604a9cf6ca518e474550-Abstract-Conference.html)

**Abstract**:

Human brains are commonly modeled as networks of Regions of Interest (ROIs) and their connections for the understanding of brain functions and mental disorders. Recently, Transformer-based models have been studied over different types of data, including graphs, shown to bring performance gains widely. In this work, we study Transformer-based models for brain network analysis. Driven by the unique properties of data, we model brain networks as graphs with nodes of fixed size and order, which allows us to (1) use connection profiles as node features to provide natural and low-cost positional information and (2) learn pair-wise connection strengths among ROIs with efficient attention weights across individuals that are predictive towards downstream analysis tasks. Moreover, we propose an Orthonormal Clustering Readout operation based on self-supervised soft clustering and orthonormal projection. This design accounts for the underlying functional modules that determine similar behaviors among groups of ROIs, leading to distinguishable cluster-aware node embeddings and informative graph embeddings. Finally, we re-standardize the evaluation pipeline on the only one publicly available large-scale brain network dataset of ABIDE, to enable meaningful comparison of different models. Experiment results show clear improvements of our proposed Brain Network Transformer on both the public ABIDE and our restricted ABCD datasets. The implementation is available at https://github.com/Wayfear/BrainNetworkTransformer.

----

## [1855] A Neural Corpus Indexer for Document Retrieval

**Authors**: *Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, Mao Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html)

**Abstract**:

Current state-of-the-art document retrieval solutions mainly follow an index-retrieve paradigm, where the index is hard to be directly optimized for the final retrieval target. In this paper, we aim to show that an end-to-end deep neural network unifying training and indexing stages can significantly improve the recall performance of traditional methods. To this end, we propose Neural Corpus Indexer (NCI), a sequence-to-sequence network that generates relevant document identifiers directly for a designated query. To optimize the recall performance of NCI, we invent a prefix-aware weight-adaptive decoder architecture, and leverage tailored techniques including query generation, semantic document identifiers, and consistency-based regularization. Empirical studies demonstrated the superiority of NCI on two commonly used academic benchmarks, achieving +21.4% and +16.8% relative enhancement for Recall@1 on NQ320k dataset and R-Precision on TriviaQA dataset, respectively, compared to the best baseline method.

----

## [1856] Perfect Sampling from Pairwise Comparisons

**Authors**: *Dimitris Fotakis, Alkis Kalavasis, Christos Tzamos*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a4628e9fbd3002a554923642f74d5d6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a4628e9fbd3002a554923642f74d5d6b-Abstract-Conference.html)

**Abstract**:

In this work, we study how to efficiently obtain perfect samples from a discrete distribution $\mathcal{D}$ given access only to pairwise comparisons of elements of its support. Specifically, we assume access to samples $(x, S)$, where $S$ is drawn from a distribution over sets $\mathcal{Q}$ (indicating the elements being compared), and $x$ is drawn from the conditional distribution $\mathcal{D}_S$ (indicating the winner of the comparison) and aim to output a clean sample $y$ distributed according to $\mathcal{D}$. We mainly focus on the case of pairwise comparisons where all sets $S$ have size 2. We design a Markov chain whose stationary distribution coincides with $\mathcal{D}$ and give an algorithm to obtain exact samples using the technique of Coupling from the Past. However, the sample complexity of this algorithm depends on the structure of the distribution $\mathcal{D}$ and can be even exponential in the support of $\mathcal{D}$ in many natural scenarios. Our main contribution is to provide an efficient exact sampling algorithm whose complexity does not depend on the structure of $\mathcal{D}$. To this end, we give a parametric Markov chain that mixes significantly faster given a good approximation to the stationary distribution. We can obtain such an approximation using an efficient learning from pairwise comparisons algorithm (Shah et al., JMLR 17, 2016). Our technique for speeding up sampling from a Markov chain whose stationary distribution is approximately known is simple, general and possibly of independent interest.

----

## [1857] Improved Utility Analysis of Private CountSketch

**Authors**: *Rasmus Pagh, Mikkel Thorup*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a47f5cdff1469751597d78e803fc590f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a47f5cdff1469751597d78e803fc590f-Abstract-Conference.html)

**Abstract**:

Sketching is an important tool for dealing with high-dimensional vectors that are sparse (or well-approximated by a sparse vector), especially useful in distributed, parallel, and streaming settings.It is known that sketches can be made differentially private by adding noise according to the sensitivity of the sketch, and this has been used in private analytics and federated learning settings.The post-processing property of differential privacy implies that \emph{all} estimates computed from the sketch can be released within the given privacy budget.In this paper we consider the classical CountSketch, made differentially private with the Gaussian mechanism, and give an improved analysis of its estimation error.Perhaps surprisingly, the privacy-utility trade-off is essentially the best one could hope for, independent of the number of repetitions in CountSketch:The error is almost identical to the error from non-private CountSketch plus the noise needed to make the vector private in the original, high-dimensional domain.

----

## [1858] Optimal Efficiency-Envy Trade-Off via Optimal Transport

**Authors**: *Steven Yin, Christian Kroer*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a488aa1a0c00d76db8a922ef7815a786-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a488aa1a0c00d76db8a922ef7815a786-Abstract-Conference.html)

**Abstract**:

We consider the problem of allocating a distribution of items to $n$ recipients where each recipient has to be allocated a fixed, pre-specified fraction of all items, while ensuring that each recipient does not experience too much envy.  We show that this problem can be formulated as a variant of the semi-discrete optimal transport (OT) problem, whose solution structure in this case has a concise representation and a simple geometric interpretation.  Unlike existing literature that treats envy-freeness as a hard constraint, our formulation allows us to \emph{optimally} trade off efficiency and envy continuously.  Additionally, we study the statistical properties of the space of our OT based allocation policies by showing a polynomial bound on the number of samples needed to approximate the optimal solution from samples.  Our approach is suitable for large-scale fair allocation problems such as the blood donation matching problem, and we show numerically that it performs well on a prior realistic data simulator.

----

## [1859] Non-Linear Coordination Graphs

**Authors**: *Yipeng Kang, Tonghan Wang, Qianlan Yang, Xiaoran Wu, Chongjie Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a48a6c9a03e87d6426b3f9bd18bbb86b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a48a6c9a03e87d6426b3f9bd18bbb86b-Abstract-Conference.html)

**Abstract**:

Value decomposition multi-agent reinforcement learning methods learn the global value function as a mixing of each agent's individual utility functions. Coordination graphs (CGs) represent a higher-order decomposition by incorporating pairwise payoff functions and thus is supposed to have a more powerful representational capacity. However, CGs decompose the global value function linearly over local value functions, severely limiting the complexity of the value function class that can be represented. In this paper, we propose the first non-linear coordination graph by extending CG value decomposition beyond the linear case. One major challenge is to conduct greedy action selections in this new function class to which commonly adopted DCOP algorithms are no longer applicable. We study how to solve this problem when mixing networks with LeakyReLU activation are used. An enumeration method with a global optimality guarantee is proposed and motivates an efficient iterative optimization method with a local optimality guarantee. We find that our method can achieve superior performance on challenging multi-agent coordination tasks like MACO.

----

## [1860] SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles

**Authors**: *Chejian Xu, Wenhao Ding, Weijie Lyu, Zuxin Liu, Shuai Wang, Yihan He, Hanjiang Hu, Ding Zhao, Bo Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As shown by recent studies, machine intelligence-enabled systems are vulnerable to test cases resulting from either adversarial manipulation or natural distribution shifts. This has raised great concerns about deploying machine learning algorithms for real-world applications, especially in safety-critical domains such as autonomous driving (AD). On the other hand, traditional AD testing on naturalistic scenarios requires hundreds of millions of driving miles due to the high dimensionality and rareness of the safety-critical scenarios in the real world. As a result, several approaches for autonomous driving evaluation have been explored, which are usually, however, based on different simulation platforms, types of safety-critical scenarios, scenario generation algorithms, and driving route variations. Thus, despite a large amount of effort in autonomous driving testing, it is still challenging to compare and understand the effectiveness and efficiency of different testing scenario generation algorithms and testing mechanisms under similar conditions. In this paper, we aim to provide the first unified platform SafeBench to integrate different types of safety-critical testing scenarios, scenario generation algorithms, and other variations such as driving routes and environments. In particular, we consider 8 safety-critical testing scenarios following National Highway Traffic Safety Administration (NHTSA) and develop 4 scenario generation algorithms considering 10 variations for each scenario. Meanwhile, we implement 4 deep reinforcement learning-based AD algorithms with 4 types of input (e.g., birdâ€™s-eye view, camera) to perform fair comparisons on SafeBench. We find our generated testing scenarios are indeed more challenging and observe the trade-off between the performance of AD agents under benign and safety-critical testing scenarios. We believe our unified platform SafeBench for large-scale and effective autonomous driving testing will motivate the development of new testing scenario generation and safe AD algorithms. SafeBench is available at https://safebench.github.io.

----

## [1861] Improving Diffusion Models for Inverse Problems using Manifold Constraints

**Authors**: *Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, Jong Chul Ye*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html)

**Abstract**:

Recently, diffusion models have been used to solve various inverse problems in an unsupervised manner with appropriate modifications to the sampling process. However, the current solvers, which recursively apply a reverse diffusion step followed by a projection-based measurement consistency step, often produce sub-optimal results. By studying the generative sampling path, here we show that current solvers throw the sample path off the data manifold, and hence the error accumulates. To address this, we propose an additional correction term inspired by the manifold constraint, which can be used synergistically with the previous solvers to make the iterations close to the manifold. The proposed manifold constraint is straightforward to implement within a few lines of code, yet boosts the performance by a surprisingly large margin. With extensive experiments, we show that our method is superior to the previous methods both theoretically and empirically, producing promising results in many applications such as image inpainting, colorization, and sparse-view computed tomography. Code available https://github.com/HJ-harry/MCG_diffusion

----

## [1862] Semi-supervised Vision Transformers at Scale

**Authors**: *Zhaowei Cai, Avinash Ravichandran, Paolo Favaro, Manchen Wang, Davide Modolo, Rahul Bhotika, Zhuowen Tu, Stefano Soatto*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html)

**Abstract**:

We study semi-supervised learning (SSL) for vision transformers (ViT), an under-explored topic despite the wide adoption of the ViT architectures to different tasks. To tackle this problem, we use a SSL pipeline, consisting of first un/self-supervised pre-training, followed by supervised fine-tuning, and finally semi-supervised fine-tuning. At the semi-supervised fine-tuning stage, we adopt an exponential moving average (EMA)-Teacher framework instead of the popular FixMatch, since the former is more stable and delivers higher accuracy for semi-supervised vision transformers. In addition, we propose a probabilistic pseudo mixup mechanism to interpolate unlabeled samples and their pseudo labels for improved regularization, which is important for training ViTs with weak inductive bias. Our proposed method, dubbed Semi-ViT, achieves comparable or better performance than the CNN counterparts in the semi-supervised classification setting. Semi-ViT also enjoys the scalability benefits of ViTs that can be readily scaled up to large-size models with increasing accuracy. For example, Semi-ViT-Huge achieves an impressive 80\% top-1 accuracy on ImageNet using only 1\% labels, which is comparable with Inception-v4 using 100\% ImageNet labels. The code is available at https://github.com/amazon-science/semi-vit.

----

## [1863] Fast Bayesian Estimation of Point Process Intensity as Function of Covariates

**Authors**: *Hideaki Kim, Taichi Asami, Hiroyuki Toda*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html)

**Abstract**:

In this paper, we tackle the Bayesian estimation of point process intensity as a function of covariates. We propose a novel augmentation of permanental process called augmented permanental process, a doubly-stochastic point process that uses a Gaussian process on covariate space to describe the Bayesian a priori uncertainty present in the square root of intensity, and derive a fast Bayesian estimation algorithm that scales linearly with data size without relying on either domain discretization or Markov Chain Monte Carlo computation. The proposed algorithm is based on a non-trivial finding that the representer theorem, one of the most desirable mathematical property for machine learning problems, holds for the augmented permanental process, which provides us with many significant computational advantages. We evaluate our algorithm on synthetic and real-world data, and show that it outperforms state-of-the-art methods in terms of predictive accuracy while being substantially faster than a conventional Bayesian method.

----

## [1864] Online PAC-Bayes Learning

**Authors**: *Maxime Haddouche, Benjamin Guedj*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a4d991d581accd2955a1e1928f4e6965-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a4d991d581accd2955a1e1928f4e6965-Abstract-Conference.html)

**Abstract**:

Most PAC-Bayesian bounds hold in the batch learning setting where data is collected at once, prior to inference or prediction. This somewhat departs from many contemporary learning problems where data streams are collected and the algorithms must dynamically adjust. We prove new PAC-Bayesian bounds in this online learning framework, leveraging an updated definition of regret, and we revisit classical PAC-Bayesian results with a batch-to-online conversion, extending their remit to the case of dependent data. Our results hold for bounded losses, potentially \emph{non-convex}, paving the way to promising developments in online learning.

----

## [1865] Deep Model Reassembly

**Authors**: *Xingyi Yang, Daquan Zhou, Songhua Liu, Jingwen Ye, Xinchao Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a4e683f0ce6b91e7fbdae9d32642d88f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a4e683f0ce6b91e7fbdae9d32642d88f-Abstract-Conference.html)

**Abstract**:

In this paper, we explore a novel knowledge-transfer task, termed as Deep  Model Reassembly (DeRy), for general-purpose model reuse.Given a collection of heterogeneous models pre-trained from distinct sources and with diverse architectures, the goal of DeRy, as its name implies, is to first dissect each model into distinctive building blocks, and then selectively reassemble the derived blocks to produce customized networks under both the hardware resource and performance constraints. Such ambitious nature of DeRy inevitably imposes significant challenges, including, in the first place, the feasibility of its solution. We strive to showcase that, through a dedicated paradigm proposed in this paper, DeRy can be made not only possibly but practically efficiently. Specifically, we conduct the partitions of all pre-trained networks jointly via a cover set optimization, and derive  a number of equivalence set, within each of which the network blocks are treated as functionally equivalent and hence interchangeable. The equivalence sets learned in this way, in turn, enable  picking and assembling blocks to customize networks subject to certain constraints, which is achieved via solving an integer program backed up with a training-free proxy to estimate the task performance. The reassembled models give rise to gratifying performances with the user-specified constraints satisfied. We demonstrate that on ImageNet, the best reassemble model achieves 78.6% top-1 accuracy without fine-tuning, which could be further elevated to 83.2% with end-to-end fine-tuning. Our code is available at https://github.com/Adamdad/DeRy.

----

## [1866] Fast Algorithms for Packing Proportional Fairness and its Dual

**Authors**: *Francisco Criado, David Martínez-Rubio, Sebastian Pokutta*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a548ef984f30bca3abdc09f43743827f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a548ef984f30bca3abdc09f43743827f-Abstract-Conference.html)

**Abstract**:

The proportional fair resource allocation problem is a major problem studied in flow control of networks, operations research, and economic theory, where it has found numerous applications. This problem, defined as the constrained maximization of $\sum_i \log x_i$, is known as the packing proportional fairness problem when the feasible set is defined by positive linear constraints and $x \in \mathbb{R}_{\geq 0}^n$. In this work, we present a distributed accelerated first-order method for this problem which improves upon previous approaches. We also design an algorithm for the optimization of its dual problem. Both algorithms are width-independent.

----

## [1867] A Closer Look at Prototype Classifier for Few-shot Image Classification

**Authors**: *Mingcheng Hou, Issei Sato*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html)

**Abstract**:

The prototypical network is a prototype classifier based on meta-learning and is widely used for few-shot learning because it classifies unseen examples by constructing class-specific prototypes without adjusting hyper-parameters during meta-testing.Interestingly, recent research has attracted a lot of attention, showing that training a new linear classifier, which does not use a meta-learning algorithm, performs comparably with the prototypical network.However, the training of a new linear classifier requires the retraining of the classifier every time a new class appears.In this paper, we analyze how a prototype classifier works equally well without training a new linear classifier or meta-learning.We experimentally find that directly using the feature vectors, which is extracted by using standard pre-trained models to construct a prototype classifier in meta-testing, does not perform as well as the prototypical network and training new linear classifiers on the feature vectors of pre-trained models.Thus, we derive a novel generalization bound for a prototypical classifier and show that the transformation of a feature vector can improve the performance of prototype classifiers.We experimentally investigate several normalization methods for minimizing the derived bound and find that the same performance can be obtained by using the L2 normalization and minimizing the ratio of the within-class variance to the between-class variance without training a new classifier or meta-learning.

----

## [1868] When are Offline Two-Player Zero-Sum Markov Games Solvable?

**Authors**: *Qiwen Cui, Simon S. Du*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a57483b394a3654f4317051e4ce3b2b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a57483b394a3654f4317051e4ce3b2b8-Abstract-Conference.html)

**Abstract**:

We study what dataset assumption permits solving offline two-player zero-sum Markov games. In stark contrast to the offline single-agent Markov decision process, we show that the single strategy concentration assumption is insufficient for learning the Nash equilibrium (NE) strategy in offline two-player zero-sum Markov games. On the other hand, we propose a new assumption named unilateral concentration and design a pessimism-type algorithm that is provably efficient under this assumption. In addition, we show that the unilateral concentration assumption is necessary for learning an NE strategy. Furthermore, our algorithm can achieve minimax sample complexity without any modification for two widely studied settings: dataset with uniform concentration assumption and turn-based Markov games. Our work serves as an important initial step towards understanding offline multi-agent reinforcement learning.

----

## [1869] BigBio: A Framework for Data-Centric Biomedical Natural Language Processing

**Authors**: *Jason A. Fries, Leon Weber, Natasha Seelam, Gabriel Altay, Debajyoti Datta, Samuele Garda, Sunny Kang, Rosaline Su, Wojciech Kusa, Samuel Cahyawijaya, Fabio Barth, Simon Ott, Matthias Samwald, Stephen H. Bach, Stella Biderman, Mario Sänger, Bo Wang, Alison Callahan, Daniel León Periñán, Théo Gigant, Patrick Haller, Jenny Chim, José D. Posada, John M. Giorgi, Karthik Rangasai Sivaraman, Marc Pàmies, Marianna Nezhurina, Robert Martin, Michael Cullan, Moritz Freidank, Nathan Dahlberg, Shubhanshu Mishra, Shamik Bose, Nicholas Broad, Yanis Labrak, Shlok Deshmukh, Sid Kiblawi, Ayush Singh, Minh Chien Vu, Trishala Neeraj, Jonas Golde, Albert Villanova del Moral, Benjamin Beilharz*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Training and evaluating language models increasingly requires the construction of meta-datasets -- diverse collections of curated data with clear provenance. Natural language prompting has recently lead to improved zero-shot generalization by transforming existing, supervised datasets into a variety of novel instruction tuning tasks, highlighting the benefits of meta-dataset curation. While successful in general-domain text, translating these data-centric approaches to biomedical language modeling remains challenging, as labeled biomedical datasets are significantly underrepresented in popular data hubs. To address this challenge, we introduce BigBio a community library of 126+ biomedical NLP datasets, currently covering 13 task categories and 10+ languages. BigBio facilitates reproducible meta-dataset curation via programmatic access to datasets and their metadata, and is compatible with current platforms for prompt engineering and end-to-end few/zero shot language model evaluation. We discuss our process for task schema harmonization, data auditing, contribution guidelines, and outline two illustrative use cases: zero-shot evaluation of biomedical prompts and large-scale, multi-task learning. BigBio is an ongoing community effort and is available at https://github.com/bigscience-workshop/biomedical

----

## [1870] Branch & Learn for Recursively and Iteratively Solvable Problems in Predict+Optimize

**Authors**: *Xinyi Hu, Jasper C. H. Lee, Jimmy H. M. Lee, Allen Z. Zhong*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a59a11e8580a7ac850cb792f6179c7a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a59a11e8580a7ac850cb792f6179c7a0-Abstract-Conference.html)

**Abstract**:

This paper proposes Branch & Learn, a framework for Predict+Optimize to tackle optimization problems containing parameters that are unknown at the time of solving. Given an optimization problem solvable by a recursive algorithm satisfying simple conditions, we show how a corresponding learning algorithm can be constructed directly and methodically from the recursive algorithm. Our framework applies also to iterative algorithms by viewing them as a degenerate form of recursion. Extensive experimentation shows better performance for our proposal over classical and state of the art approaches.

----

## [1871] Linear tree shap

**Authors**: *Peng Yu, Albert Bifet, Jesse Read, Chao Xu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a5a3b1ef79520b7cd122d888673a3ebc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a5a3b1ef79520b7cd122d888673a3ebc-Abstract-Conference.html)

**Abstract**:

Decision trees are well-known due to their ease of interpretability.To improve accuracy, we need to grow deep trees or ensembles of trees.These are hard to interpret, offsetting their original benefits. Shapley values have recently become a popular way to explain the predictions of tree-based machine learning models. It provides a linear weighting to features independent of the tree structure. The rise in popularity is mainly due to TreeShap, which solves a general exponential complexity problem in polynomial time. Following extensive adoption in the industry, more efficient algorithms are required. This paper presents a more efficient and straightforward algorithm: Linear TreeShap.Like TreeShap, Linear TreeShap is exact and requires the same amount of memory.

----

## [1872] Gradient Estimation with Discrete Stein Operators

**Authors**: *Jiaxin Shi, Yuhao Zhou, Jessica Hwang, Michalis K. Titsias, Lester Mackey*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a5a5b0ff87c59172a13342d428b1e033-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a5a5b0ff87c59172a13342d428b1e033-Abstract-Conference.html)

**Abstract**:

Gradient estimation---approximating the gradient of an  expectation  with respect to the parameters of a distribution---is central to the solution of  many machine learning problems.  However, when the distribution is discrete, most common gradient estimators suffer from excessive variance. To improve the quality of gradient estimation, we introduce a variance reduction technique based on Stein operators for discrete distributions. We then use this technique to build flexible control variates for the REINFORCE leave-one-out estimator.  Our control variates can be adapted online to minimize variance and do not require extra evaluations of the target function. In benchmark generative modeling tasks such as training binary variational autoencoders, our gradient estimator achieves substantially lower variance than state-of-the-art estimators with the same number of function evaluations.

----

## [1873] Rapidly Mixing Multiple-try Metropolis Algorithms for Model Selection Problems

**Authors**: *Hyunwoong Chang, Changwoo J. Lee, Zhao Tang Luo, Huiyan Sang, Quan Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a600cdf3a53f93bcb85cb37343a8d831-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a600cdf3a53f93bcb85cb37343a8d831-Abstract-Conference.html)

**Abstract**:

The multiple-try Metropolis (MTM) algorithm is an extension of the Metropolis-Hastings (MH) algorithm by selecting the proposed state among multiple trials according to some weight function. Although MTM has gained great popularity owing to its faster empirical convergence and mixing than the standard MH algorithm, its theoretical mixing property is rarely studied in the literature due to its complex proposal scheme. We prove that MTM can achieve a mixing time bound smaller than that of MH by a factor of the number of trials under a general setting applicable to high-dimensional model selection problems with discrete state spaces. Our theoretical results motivate a new class of weight functions called locally balanced weight functions and guide the choice of the number of trials, which leads to improved performance over standard MTM algorithms. We support our theoretical results by extensive simulation studies and real data applications with several Bayesian model selection problems.

----

## [1874] Distributionally Adaptive Meta Reinforcement Learning

**Authors**: *Anurag Ajay, Abhishek Gupta, Dibya Ghosh, Sergey Levine, Pulkit Agrawal*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html)

**Abstract**:

Meta-reinforcement learning algorithms provide a data-driven way to acquire policies that quickly adapt to many tasks with varying rewards or dynamics functions. However, learned meta-policies are often effective only on the exact task distribution on which they were trained and struggle in the presence of distribution shift of test-time rewards or transition dynamics. In this work, we develop a framework for meta-RL algorithms that are able to behave appropriately under test-time distribution shifts in the space of tasks. Our framework centers on an adaptive approach to distributional robustness that trains a population of meta-policies to be robust to varying levels of distribution shift. When evaluated on a potentially shifted test-time distribution of tasks, this allows us to choose the meta-policy with the most appropriate level of robustness, and use it to perform fast adaptation. We formally show how our framework allows for improved regret under distribution shift, and empirically show its efficacy on simulated robotics problems under a wide range of distribution shifts.

----

## [1875] Toward Efficient Robust Training against Union of $\ell_p$ Threat Models

**Authors**: *Gaurang Sriramanan, Maharshi Gor, Soheil Feizi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a627b9468c319c13a70b7c2fb8df65a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a627b9468c319c13a70b7c2fb8df65a3-Abstract-Conference.html)

**Abstract**:

The overwhelming vulnerability of deep neural networks to carefully crafted perturbations known as adversarial attacks has led to the development of various training techniques to produce robust models. While the primary focus of existing approaches has been directed toward addressing the worst-case performance achieved under a single-threat model, it is imperative that safety-critical systems are robust with respect to multiple threat models simultaneously. Existing approaches that address worst-case performance under the union of such threat models ($\ell_{\infty}, \ell_2, \ell_1$) either utilize adversarial training methods that require multi-step attacks which are computationally expensive in practice, or rely upon fine-tuning of pre-trained models that are robust with respect to a single-threat model. In this work, we show that by carefully choosing the objective function used for robust training, it is possible to achieve similar, or improved worst-case performance over a union of threat models while utilizing only single-step attacks, thereby achieving a significant reduction in computational resources necessary for training. Furthermore, prior work showed that adversarial training specific to the $\ell_1$ threat model is relatively difficult, to the extent that even multi-step adversarially trained models were shown to be prone to gradient-masking. However, the proposed method—when applied on the $\ell_1$ threat model specifically—enables us to obtain the first $\ell_1$ robust model trained solely with single-step adversaries. Finally, to demonstrate the merits of our approach, we utilize a modern set of attack evaluations to better estimate the worst-case performance under the considered union of threat models.

----

## [1876] Multi-view Subspace Clustering on Topological Manifold

**Authors**: *Shudong Huang, Hongjie Wu, Yazhou Ren, Ivor W. Tsang, Zenglin Xu, Wentao Feng, Jiancheng Lv*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6610efd6c767f63343a4ab28505212e-Abstract-Conference.html)

**Abstract**:

Multi-view subspace clustering aims to exploit a common affinity representation by means of self-expression. Plenty of works have been presented to boost the clustering performance, yet seldom considering the topological structure in data, which is crucial for clustering data on manifold. Orthogonal to existing works, in this paper, we argue that it is beneficial to explore the implied data manifold by learning the topological relationship between data points. Our model seamlessly integrates multiple affinity graphs into a consensus one with the topological relevance considered. Meanwhile, we manipulate the consensus graph by a connectivity constraint such that the connected components precisely indicate different clusters. Hence our model is able to directly obtain the final clustering result without reliance on any label discretization strategy as previous methods do. Experimental results on several benchmark datasets illustrate the effectiveness of the proposed model, compared to the state-of-the-art competitors over the clustering performance.

----

## [1877] CLEAR: Generative Counterfactual Explanations on Graphs

**Authors**: *Jing Ma, Ruocheng Guo, Saumitra Mishra, Aidong Zhang, Jundong Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a69d7f3a1340d55c720e572742439eaf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a69d7f3a1340d55c720e572742439eaf-Abstract-Conference.html)

**Abstract**:

Counterfactual explanations promote explainability in machine learning models by answering the question “how should the input instance be altered to obtain a desired predicted label?". The comparison of this instance before and after perturbation can enhance human interpretation. Most existing studies on counterfactual explanations are limited in tabular data or image data. In this paper, we study the problem of counterfactual explanation generation on graphs. A few studies have explored to generate counterfactual explanations on graphs, but many challenges of this problem are still not well-addressed: 1) optimizing in the discrete and disorganized space of graphs; 2) generalizing on unseen graphs; 3) maintaining the causality in the generated counterfactuals without prior knowledge of the causal model. To tackle these challenges, we propose a novel framework CLEAR which aims to generate counterfactual explanations on graphs for graph-level prediction models. Specifically, CLEAR leverages a graph variational autoencoder based mechanism to facilitate its optimization and generalization, and promotes causality by leveraging an auxiliary variable to better identify the causal model. Extensive experiments on both synthetic and real-world graphs validate the superiority of CLEAR over state-of-the-art counterfactual explanation methods on graphs in different aspects.

----

## [1878] Online Agnostic Multiclass Boosting

**Authors**: *Vinod Raman, Ambuj Tewari*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6acb2d482de9c708f5b03d5a70465d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6acb2d482de9c708f5b03d5a70465d2-Abstract-Conference.html)

**Abstract**:

Boosting is a fundamental approach in machine learning that enjoys both strong theoretical and practical guarantees. At a high-level, boosting algorithms cleverly aggregate weak learners to generate predictions with arbitrarily high accuracy. In this way, boosting algorithms convert weak learners into strong ones. Recently, Brukhim et al. [2020] extended boosting to the online agnostic binary classification setting. A key ingredient in their approach is a clean and simple reduction to online convex optimization, one that efficiently converts an arbitrary online convex optimizer to an agnostic online booster. In this work, we extend this reduction to multiclass problems and give the first boosting algorithm for online agnostic mutliclass classification.  Our reduction also enables the construction of algorithms for statistical agnostic, online realizable, and statistical realizable multiclass boosting.

----

## [1879] A contrastive rule for meta-learning

**Authors**: *Nicolas Zucchet, Simon Schug, Johannes von Oswald, Dominic Zhao, João Sacramento*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)

**Abstract**:

Humans and other animals are capable of improving their learning performance as they solve related tasks from a given problem domain, to the point of being able to learn from extremely limited data. While synaptic plasticity is generically thought to underlie learning in the brain, the precise neural and synaptic mechanisms by which learning processes improve through experience are not well understood. Here, we present a general-purpose, biologically-plausible meta-learning rule which estimates gradients with respect to the parameters of an underlying learning algorithm by simply running it twice. Our rule may be understood as a generalization of contrastive Hebbian learning to meta-learning and notably, it neither requires computing second derivatives nor going backwards in time, two characteristic features of previous gradient-based methods that are hard to conceive in physical neural circuits. We demonstrate the generality of our rule by applying it to two distinct models: a complex synapse with internal states which consolidate task-shared information, and a dual-system architecture in which a primary network is rapidly modulated by another one to learn the specifics of each task. For both models, our meta-learning rule matches or outperforms reference algorithms on a wide range of benchmark problems, while only using information presumed to be locally available at neurons and synapses. We corroborate these findings with a theoretical analysis of the gradient estimation error incurred by our rule.

----

## [1880] Distinguishing Learning Rules with Brain Machine Interfaces

**Authors**: *Jacob P. Portes, Christian Schmid, James M. Murray*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6d94c38506f16fb50894a5b555f2c9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6d94c38506f16fb50894a5b555f2c9a-Abstract-Conference.html)

**Abstract**:

Despite extensive theoretical work on biologically plausible learning rules, clear evidence about whether and how such rules are implemented in the brain has been difficult to obtain. We consider biologically plausible supervised- and reinforcement-learning rules and ask whether changes in network activity during learning can be used to determine which learning rule is being used. Supervised learning requires a credit-assignment model estimating the mapping from neural activity to behavior, and, in a biological organism, this model will inevitably be an imperfect approximation of the ideal mapping, leading to a bias in the direction of the weight updates relative to the true gradient. Reinforcement learning, on the other hand, requires no credit-assignment model and tends to make weight updates following the true gradient direction. We derive a metric to distinguish between learning rules by observing changes in the network activity during learning, given that the mapping from brain to behavior is known by the experimenter. Because brain-machine interface (BMI) experiments allow for precise knowledge of this mapping, we model a cursor-control BMI task using recurrent neural networks, showing that  learning rules can be distinguished in simulated experiments using only observations that a  neuroscience experimenter would plausibly have access to.

----

## [1881] Efficient Training of Low-Curvature Neural Networks

**Authors**: *Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, François Fleuret*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6ec568ede6584b20dccfb6c2e4f2b58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6ec568ede6584b20dccfb6c2e4f2b58-Abstract-Conference.html)

**Abstract**:

Standard deep neural networks often have excess non-linearity, making them susceptible to issues such as low adversarial robustness and gradient instability. Common methods to address these downstream issues, such as adversarial training, are expensive and often sacrifice predictive accuracy. In this work, we address the core issue of excess non-linearity via curvature, and demonstrate low-curvature neural networks (LCNNs) that obtain drastically lower curvature than standard models while exhibiting similar predictive performance. This leads to improved robustness and stable gradients, at a fraction of the cost of standard adversarial training. To achieve this, we decompose overall model curvature in terms of curvatures and slopes of its constituent layers. To enable efficient curvature minimization of constituent layers, we introduce two novel architectural components: first, a non-linearity called centered-softplus that is a stable variant of the softplus non-linearity, and second, a Lipschitz-constrained batch normalization layer.Our experiments show that LCNNs have lower curvature, more stable gradients and increased off-the-shelf adversarial robustness when compared to standard neural networks, all without affecting predictive performance. Our approach is easy to use and can be readily incorporated into existing neural network architectures.

----

## [1882] Infinite-Fidelity Coregionalization for Physical Simulation

**Authors**: *Shibo Li, Zheng Wang, Robert M. Kirby, Shandian Zhe*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6fcfd15cd01e4a550808c3e01f5583d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6fcfd15cd01e4a550808c3e01f5583d-Abstract-Conference.html)

**Abstract**:

Multi-fidelity modeling and learning is important in physical simulation related applications. It can leverage both low-fidelity and high-fidelity examples for training so as to reduce the cost of data generation yet still achieving good performance. While existing approaches only model finite, discrete fidelities, in practice, the feasible fidelity choice is often infinite, which can correspond to a continuous mesh spacing or finite element length. In this paper, we propose Infinite Fidelity Coregionalization (IFC). Given the data, our method can extract and exploit rich information within infinite, continuous fidelities to bolster the prediction accuracy. Our model can interpolate and/or extrapolate the predictions to novel fidelities that are not covered by the training data. Specifically, we introduce a low-dimensional latent output as a continuous function of the fidelity and input, and multiple it with a basis matrix to predict high-dimensional solution outputs. We model the latent output as a neural Ordinary Differential Equation (ODE) to capture the complex relationships within and integrate information throughout the continuous fidelities.  We then use Gaussian processes or another ODE to estimate the fidelity-varying bases. For efficient inference, we reorganize the bases as a tensor, and use a tensor-Gaussian variational posterior approximation to develop a scalable inference algorithm for massive outputs. We show the advantage of our method in several benchmark tasks in computational physics.

----

## [1883] Open High-Resolution Satellite Imagery: The WorldStrat Dataset - With Application to Super-Resolution

**Authors**: *Julien Cornebise, Ivan Orsolic, Freddie Kalaitzis*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a6fe99561d9eb9c90b322afe664587fd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a6fe99561d9eb9c90b322afe664587fd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Analyzing the planet at scale with satellite imagery and machine learning is a dream that has been constantly hindered by the cost of difficult-to-access highly-representative high-resolution imagery. To remediate this, we introduce here the  WorldStratified dataset. The largest and most varied such publicly available dataset, at Airbus SPOT 6/7 satellites' high resolution of up to 1.5 m/pixel, empowered by European Space Agency's Phi-Lab as part of the ESA-funded QueryPlanet project, we curate 10,000 sq km of unique locations to ensure stratified representation of all types of land-use across the world: from agriculture to ice caps, from forests to multiple urbanization densities. We also enrich those with locations typically under-represented in ML datasets: sites of humanitarian interest, illegal mining sites, and settlements of persons at risk. We temporally-match each high-resolution image with multiple low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites at 10 m/pixel. We accompany this dataset with an open-source Python package to: rebuild or extend the WorldStrat dataset, train and infer baseline algorithms, and learn with abundant tutorials, all compatible with the popular EO-learn toolbox. We hereby hope to foster broad-spectrum applications of ML to satellite imagery, and possibly develop from free public low-resolution Sentinel2 imagery the same power of analysis allowed by costly private high-resolution imagery. We illustrate this specific point by training and releasing several highly compute-efficient baselines on the task of Multi-Frame Super-Resolution. License-wise, the high-resolution Airbus imagery is CC-BY-NC, while the labels, Sentinel2 imagery, and trained weights are under CC-BY, and the source code under BSD, to allow for the widest use and dissemination. The dataset is available at \url{https://zenodo.org/record/6810792} and the software package at \url{https://github.com/worldstrat/worldstrat}.

----

## [1884] Evaluation beyond Task Performance: Analyzing Concepts in AlphaZero in Hex

**Authors**: *Charles Lovering, Jessica Forde, George Konidaris, Ellie Pavlick, Michael L. Littman*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a705747417d32ebf1916169e1a442274-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a705747417d32ebf1916169e1a442274-Abstract-Conference.html)

**Abstract**:

AlphaZero, an approach to reinforcement learning that couples neural networks and Monte Carlo tree search (MCTS), has produced state-of-the-art strategies for traditional board games like chess, Go, shogi, and Hex. While researchers and game commentators have suggested that AlphaZero uses concepts that humans consider important, it is unclear how these concepts are captured in the network. We investigate AlphaZero's internal representations in the game of Hex using two evaluation techniques from natural language processing (NLP): model probing and behavioral tests. In doing so, we introduce several new evaluation tools to the RL community, and illustrate how evaluations other than task performance can be used to provide a more complete picture of a model's strengths and weaknesses. Our analyses in the game of Hex reveal interesting patterns and generate some testable hypotheses about how such models learn in general. For example, we find that the MCTS discovers concepts before the neural network learns to encode them. We also find that concepts related to short-term end-game planning are best encoded in the final layers of the model, whereas concepts related to long-term planning are encoded in the middle layers of the model.

----

## [1885] Semi-supervised Semantic Segmentation with Prototype-based Consistency Regularization

**Authors**: *Haiming Xu, Lingqiao Liu, Qiuchen Bian, Zhen Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a70ee7ea485e4fd36abbfc4adf591c28-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a70ee7ea485e4fd36abbfc4adf591c28-Abstract-Conference.html)

**Abstract**:

Semi-supervised semantic segmentation requires the model to effectively propagate the label information from limited annotated images to unlabeled ones. A challenge for such a per-pixel prediction task is the large intra-class variation, i.e., regions belonging to the same class may exhibit a very different appearance even in the same picture. This diversity will make the label propagation hard from pixels to pixels. To address this problem, we propose a novel approach to regularize the distribution of within-class features to ease label propagation difficulty. Specifically, our approach encourages the consistency between the prediction from a linear predictor and the output from a prototype-based predictor, which implicitly encourages features from the same pseudo-class to be close to at least one within-class prototype while staying far from the other between-class prototypes. By further incorporating CutMix operations and a carefully-designed prototype maintenance strategy, we create a semi-supervised semantic segmentation algorithm that demonstrates superior performance over the state-of-the-art methods from extensive experimental evaluation on both Pascal VOC and Cityscapes benchmarks.

----

## [1886] Empirical Phase Diagram for Three-layer Neural Networks with Infinite Width

**Authors**: *Hanxu Zhou, Qixuan Zhou, Zhenyuan Jin, Tao Luo, Yaoyu Zhang, Zhi-Qin John Xu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html)

**Abstract**:

Substantial work indicates that the dynamics of neural networks (NNs) is closely related to their initialization of parameters. Inspired by the phase diagram for two-layer ReLU NNs with infinite width (Luo et al., 2021), we make a step towards drawing a phase diagram for three-layer ReLU NNs with infinite width. First, we derive a normalized gradient flow for three-layer ReLU NNs and obtain two key independent quantities to distinguish different dynamical regimes for common initialization methods. With carefully designed experiments and a large computation cost, for both synthetic datasets and real datasets, we find that the dynamics of each layer also could be divided into a linear regime and a condensed regime, separated by a critical regime. The criteria is the relative change of input weights (the input weight of a hidden neuron consists of the weight from its input layer to the hidden neuron and its bias term) as the width approaches infinity during the training, which tends to $0$, $+\infty$ and $O(1)$, respectively. In addition, we also demonstrate that different layers can lie in different dynamical regimes in a training process within a deep NN. In the condensed regime, we also observe the condensation of weights in isolated orientations with low complexity. Through experiments under three-layer condition, our phase diagram suggests a complicated dynamical regimes consisting of three possible regimes, together with their mixture, for deep NNs and provides a guidance for studying deep NNs in different initialization regimes, which reveals the possibility of completely different dynamics emerging within a deep NN for its different layers.

----

## [1887] Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms

**Authors**: *Hui En Pang, Zhongang Cai, Lei Yang, Tianwei Zhang, Ziwei Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a727a08774b61b0c754c2183d3ecd4fc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a727a08774b61b0c754c2183d3ecd4fc-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

3D human pose and shape estimation (a.k.a. ``human mesh recovery'') has achieved substantial progress. Researchers mainly focus on the development of novel algorithms, while less attention has been paid to other critical factors involved. This could lead to less optimal baselines, hindering the fair and faithful evaluations of newly designed methodologies. To address this problem, this work presents the \textit{first} comprehensive benchmarking study from three under-explored perspectives beyond algorithms. \emph{1) Datasets.} An analysis on 31 datasets reveals the distinct impacts of data samples: datasets featuring critical attributes (\emph{i.e.} diverse poses, shapes, camera characteristics, backbone features) are more effective. Strategical selection and combination of high-quality datasets can yield a significant boost to the model performance. \emph{2) Backbones.} Experiments with 10 backbones, ranging from CNNs to transformers, show the knowledge learnt from a proximity task is readily transferable to human mesh recovery. \emph{3) Training strategies.} Proper augmentation techniques and loss designs are crucial. With the above findings, we achieve a PA-MPJPE of 47.3 (mm) on the 3DPW test set with a relatively simple model. More importantly, we provide strong baselines for fair comparisons of algorithms, and recommendations for building effective training configurations in the future. Codebase is available at \url{https://github.com/smplbody/hmr-benchmarks}.

----

## [1888] TTOpt: A Maximum Volume Quantized Tensor Train-based Optimization and its Application to Reinforcement Learning

**Authors**: *Konstantin Sozykin, Andrei Chertkov, Roman Schutski, Anh-Huy Phan, Andrzej S. Cichocki, Ivan V. Oseledets*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a730abbcd6cf4a371ca9545db5922442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a730abbcd6cf4a371ca9545db5922442-Abstract-Conference.html)

**Abstract**:

We present a novel procedure for optimization based on the combination of efficient quantized tensor train representation and a generalized maximum matrix volume principle.We demonstrate the applicability of the new Tensor Train Optimizer (TTOpt) method for various tasks, ranging from minimization of multidimensional functions to reinforcement learning.Our algorithm compares favorably to popular gradient-free methods and outperforms them by the number of function evaluations or execution time, often by a significant margin.

----

## [1889] Diversified Recommendations for Agents with Adaptive Preferences

**Authors**: *William Brown, Arpit Agarwal*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html)

**Abstract**:

When an Agent visits a platform recommending a menu of content to select from, their choice of item depends not only on immutable preferences, but also on their prior engagements with the platform. The Recommender's primary objective is typically to encourage content consumption which optimizes some reward, such as ad revenue, but they often additionally aim to ensure that a sufficiently wide variety of content is consumed by the Agent over time. We formalize this problem as an adversarial bandit task. At each step, the Recommender presents a menu of $k$ (out of $n$) items to the Agent, who selects one item in the menu according to their unknown {\it preference model}, which maps their history of past items to relative selection probabilities. The Recommender then observes the Agent's selected item and receives bandit feedback of the item's (adversarial) reward. In addition to optimizing  reward from the selected items at each step, the Recommender must also ensure that the total distribution of chosen items has sufficiently high entropy. We define a class of preference models which are {\it locally learnable}, i.e.\ behavior over the entire domain can be estimated by only observing behavior in a small region; this includes models representable by bounded-degree polynomials as well as functions with a sparse Fourier basis. For this class, we give an algorithm for the Recommender which obtains $\tilde{O}(T^{3/4})$ regret against all  item distributions satisfying two conditions: they are sufficiently diversified, and they are {\it instantaneously  realizable} at any history by some distribution over menus. We show that these conditions are closely connected:  all sufficiently high-entropy distributions are instantaneously realizable at any history of selected items. We also give a set of negative results justifying our assumptions, in the form of a runtime lower bound for non-local learning and linear regret lower bounds for alternate benchmarks.

----

## [1890] A Mixture Of Surprises for Unsupervised Reinforcement Learning

**Authors**: *Andrew Zhao, Matthieu Gaetan Lin, Yangguang Li, Yong-Jin Liu, Gao Huang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a7667ee5d545a43d2f0fda98863c260e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a7667ee5d545a43d2f0fda98863c260e-Abstract-Conference.html)

**Abstract**:

Unsupervised reinforcement learning aims at learning a generalist policy in a reward-free manner for fast adaptation to downstream tasks. Most of the existing methods propose to provide an intrinsic reward based on surprise. Maximizing or minimizing surprise drives the agent to either explore or gain control over its environment. However, both strategies rely on a strong assumption: the entropy of the environment's dynamics is either high or low. This assumption may not always hold in real-world scenarios, where the entropy of the environment's dynamics may be unknown. Hence, choosing between the two objectives is a dilemma. We propose a novel yet simple mixture of policies to address this concern, allowing us to optimize an objective that simultaneously maximizes and minimizes the surprise. Concretely, we train one mixture component whose objective is to maximize the surprise and another whose objective is to minimize the surprise. Hence, our method does not make assumptions about the entropy of the environment's dynamics. We call our method a $\textbf{M}\text{ixture }\textbf{O}\text{f }\textbf{S}\text{urprise}\textbf{S}$ (MOSS) for unsupervised reinforcement learning. Experimental results show that our simple method achieves state-of-the-art performance on the URLB benchmark, outperforming previous pure surprise maximization-based objectives. Our code is available at: https://github.com/LeapLabTHU/MOSS.

----

## [1891] Deep Hierarchical Planning from Pixels

**Authors**: *Danijar Hafner, Kuang-Huei Lee, Ian Fischer, Pieter Abbeel*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html)

**Abstract**:

Intelligent agents need to select long sequences of actions to solve complex tasks. While humans easily break down tasks into subgoals and reach them through millions of muscle commands, current artificial intelligence is limited to tasks with horizons of a few hundred decisions, despite large compute budgets. Research on hierarchical reinforcement learning aims to overcome this limitation but has proven to be challenging, current methods rely on manually specified goal spaces or subtasks, and no general solution exists. We introduce Director, a practical method for learning hierarchical behaviors directly from pixels by planning inside the latent space of a learned world model. The high-level policy maximizes task and exploration rewards by selecting latent goals and the low-level policy learns to achieve the goals. Despite operating in latent space, the decisions are interpretable because the world model can decode goals into images for visualization. Director learns successful behaviors across a wide range of environments, including visual control, Atari games, and DMLab levels and outperforms exploration methods on tasks with very sparse rewards, including 3D maze traversal with a quadruped robot from an egocentric camera and proprioception, without access to the global position or top-down view used by prior work.

----

## [1892] PeRFception: Perception using Radiance Fields

**Authors**: *Yoonwoo Jeong, Seungjoo Shin, Junha Lee, Christopher B. Choy, Anima Anandkumar, Minsu Cho, Jaesik Park*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The recent progress in implicit 3D representation, i.e., Neural Radiance Fields (NeRFs), has made accurate and photorealistic 3D reconstruction possible in a differentiable manner. This new representation can effectively convey the information of hundreds of high-resolution images in one compact format and allows photorealistic synthesis of novel views. In this work, using the variant of NeRF called Plenoxels, we create the first large-scale radiance fields datasets  for perception tasks, called the PeRFception, which consists of two parts that incorporate both object-centric and scene-centric scans for classification and segmentation. It shows a significant memory compression rate (96.4\%) from the original dataset, while containing both 2D and 3D information in a unified form. We construct the  classification and segmentation models that directly take this radiance fields format as input and also propose a novel augmentation technique to avoid overfitting on backgrounds of images. The code and data are publicly available in "https://postech-cvlab.github.io/PeRFception/".

----

## [1893] Efficient Submodular Optimization under Noise: Local Search is Robust

**Authors**: *Lingxiao Huang, Yuyi Wang, Chunxue Yang, Huanjian Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html)

**Abstract**:

The problem of monotone submodular maximization has been studied extensively due to its wide range of applications. However, there are cases where one can only access the objective function in a distorted or noisy form because of the uncertain nature or the errors involved in the evaluation. This paper considers the problem of constrained monotone submodular maximization with noisy oracles introduced by Hassidim and Singer (2017). For a cardinality constraint, we propose an algorithm achieving a near-optimal (1-1/e-O(epsilon))-approximation guarantee (for arbitrary epsilon > 0) with only a polynomial number of queries to the noisy value oracle, which improves the exponential query complexity of Singer and Hassidim (2018). For general matroid constraints, we show the first constant approximation algorithm in the presence of noise. Our main approaches are to design a novel local search framework that can handle the effect of noise and to construct certain smoothing surrogate functions for noise reduction.

----

## [1894] On the Generalization Power of the Overfitted Three-Layer Neural Tangent Kernel Model

**Authors**: *Peizhong Ju, Xiaojun Lin, Ness B. Shroff*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a77eadda332b6d4a9ae1e0e4024555f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a77eadda332b6d4a9ae1e0e4024555f2-Abstract-Conference.html)

**Abstract**:

In this paper, we study the generalization performance of overparameterized 3-layer NTK models. We show that, for a specific set of ground-truth functions (which we refer to as the "learnable set"), the test error of the overfitted 3-layer NTK is upper bounded by an expression that decreases with the number of neurons of the two hidden layers. Different from 2-layer NTK where there exists only one hidden-layer, the 3-layer NTK involves interactions between two hidden-layers. Our upper bound reveals that, between the two hidden-layers, the test error descends faster with respect to the number of neurons in the second hidden-layer (the one closer to the output) than with respect to that in the first hidden-layer (the one closer to the input). We also show that the learnable set of 3-layer NTK without bias is no smaller than that of 2-layer NTK models with various choices of bias in the neurons. However, in terms of the actual generalization performance, our results suggest that 3-layer NTK is much less sensitive to the choices of bias than 2-layer NTK, especially when the input dimension is large.

----

## [1895] Chromatic Correlation Clustering, Revisited

**Authors**: *Qing Xiu, Kai Han, Jing Tang, Shuang Cui, He Huang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a781ff9cfb267277937db1818284739f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a781ff9cfb267277937db1818284739f-Abstract-Conference.html)

**Abstract**:

Chromatic Correlation Clustering (CCC) (introduced by Bonchi et al. [6]) is a natural generalization of the celebrated Correlation Clustering (CC) problem, introduced by Bonchi et al. [6]. It models objects with categorical pairwise relationships by an edge-colored graph, and has many applications in data mining, social networks and bioinformatics. We show that there exists a $2.5$-approximation to the CCC problem based on a Linear Programming (LP) approach, thus improving the best-known approximation ratio of 3 achieved by Klodt et al. [21] . We also present an efficient heuristic algorithm for CCC leveraging a greedy clustering strategy, and conduct extensive experiments to demonstrate the effectiveness and efficiency of our proposed algorithm.

----

## [1896] Gradient-Free Methods for Deterministic and Stochastic Nonsmooth Nonconvex Optimization

**Authors**: *Tianyi Lin, Zeyu Zheng, Michael I. Jordan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a78f142aec481e68c75276756e0a0d91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a78f142aec481e68c75276756e0a0d91-Abstract-Conference.html)

**Abstract**:

Nonsmooth nonconvex optimization problems broadly emerge in machine learning and business decision making, whereas two core challenges impede the development of efficient solution methods with finite-time convergence guarantee: the lack of computationally tractable optimality criterion and the lack of computationally powerful oracles. The contributions of this paper are two-fold. First, we establish the relationship between the celebrated Goldstein subdifferential~\citep{Goldstein-1977-Optimization} and uniform smoothing, thereby providing the basis and intuition for the design of gradient-free methods that guarantee the finite-time convergence to a set of Goldstein stationary points. Second, we propose the gradient-free method (GFM) and stochastic GFM for solving a class of nonsmooth nonconvex optimization problems and prove that both of them can return a $(\delta,\epsilon)$-Goldstein stationary point of a Lipschitz function $f$ at an expected convergence rate at $O(d^{3/2}\delta^{-1}\epsilon^{-4})$ where $d$ is the problem dimension. Two-phase versions of GFM and SGFM are also proposed and proven to achieve improved large-deviation results. Finally, we demonstrate the effectiveness of 2-SGFM on training ReLU neural networks with the \textsc{Minst} dataset.

----

## [1897] Multilingual Abusive Comment Detection at Scale for Indic Languages

**Authors**: *Vikram Gupta, Sumegh Roychowdhury, Mithun Das, Somnath Banerjee, Punyajoy Saha, Binny Mathew, Hastagiri Prakash Vanchinathan, Animesh Mukherjee*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a7c4163b33286261b24c72fd3d1707c9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Social media platforms were conceived to act as online town squares' where people could get together, share information and communicate with each other peacefully. However, harmful content borne out of bad actors are constantly plaguing these platforms slowly converting them intomosh pits' where the bad actors take the liberty to extensively abuse various marginalised groups. Accurate and timely detection of abusive content on social media platforms is therefore very important for facilitating safe interactions between users.  However, due to the small scale and sparse linguistic coverage of Indic abusive speech datasets, development of such algorithms for Indic social media users (one-sixth of global population) is severely impeded.To facilitate and encourage research in this important direction, we contribute for the first time MACD - a large-scale (150K), human-annotated, multilingual (5 languages), balanced (49\% abusive content) and diverse (70K users) abuse detection dataset of user comments, sourced from a popular social media platform - ShareChat. We also release AbuseXLMR, an abusive content detection model pretrained on large number of social media comments in 15+ Indic languages which outperforms XLM-R and MuRIL on multiple Indic datasets. Along with the annotations, we also release the mapping between comment, post and user id's to facilitate modelling the relationship between them. We share competitive monolingual, cross-lingual and few-shot baselines so that MACD can be used as a dataset benchmark for future research.

----

## [1898] Generalized Delayed Feedback Model with Post-Click Information in Recommender Systems

**Authors**: *Jia-Qi Yang, De-Chuan Zhan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a7f90da65dd41d699d00e95700e6fa1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a7f90da65dd41d699d00e95700e6fa1e-Abstract-Conference.html)

**Abstract**:

Predicting conversion rate (e.g., the probability that a user will purchase an item) is a fundamental problem in machine learning based recommender systems. However, accurate conversion labels are revealed after a long delay, which harms the timeliness of recommender systems. Previous literature concentrates on utilizing early conversions to mitigate such a delayed feedback problem. In this paper, we show that post-click user behaviors are also informative to conversion rate prediction and can be used to improve timeliness. We propose a generalized delayed feedback model (GDFM) that unifies both post-click behaviors and early conversions as stochastic post-click information, which could be utilized to train GDFM in a streaming manner efficiently. Based on GDFM, we further establish a novel perspective that the performance gap introduced by delayed feedback can be attributed to a temporal gap and a sampling gap. Inspired by our analysis, we propose to measure the quality of post-click information with a combination of temporal distance and sample complexity. The training objective is re-weighted accordingly to highlight informative and timely signals. We validate our analysis on public datasets, and experimental performance confirms the effectiveness of our method.

----

## [1899] A Communication-Efficient Distributed Gradient Clipping Algorithm for Training Deep Neural Networks

**Authors**: *Mingrui Liu, Zhenxun Zhuang, Yunwen Lei, Chunyang Liao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a7fa0a0d6b4bb14c659b9921e8e4a772-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a7fa0a0d6b4bb14c659b9921e8e4a772-Abstract-Conference.html)

**Abstract**:

In distributed training of deep neural networks, people usually run Stochastic Gradient Descent (SGD) or its variants on each machine and communicate with other machines periodically. However, SGD might converge slowly in training some deep neural networks (e.g., RNN, LSTM) because of the exploding gradient issue. Gradient clipping is usually employed to address this issue in the single machine setting, but exploring this technique in the distributed setting is still in its infancy: it remains mysterious whether the gradient clipping scheme can take advantage of multiple machines to enjoy parallel speedup. The main technical difficulty lies in dealing with nonconvex loss function, non-Lipschitz continuous gradient, and skipping communication rounds simultaneously. In this paper, we explore a relaxed-smoothness assumption of the loss landscape which LSTM was shown to satisfy in previous works, and design a communication-efficient gradient clipping algorithm. This algorithm can be run on multiple machines, where each machine employs a gradient clipping scheme and communicate with other machines after multiple steps of gradient-based updates. Our algorithm is proved to have $O\left(\frac{1}{N\epsilon^4}\right)$ iteration complexity and $O(\frac{1}{\epsilon^3})$ communication complexity for finding an $\epsilon$-stationary point in the homogeneous data setting, where $N$ is the number of machines. This indicates that our algorithm enjoys linear speedup and reduced communication rounds. Our proof relies on novel analysis techniques of estimating truncated random variables, which we believe are of independent interest. Our experiments on several benchmark datasets and various scenarios demonstrate that our algorithm indeed exhibits fast convergence speed in practice and thus validates our theory.

----

## [1900] On Analyzing Generative and Denoising Capabilities of Diffusion-based Deep Generative Models

**Authors**: *Kamil Deja, Anna Kuzina, Tomasz Trzcinski, Jakub M. Tomczak*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a7fe86385ab2aa74024c6ddb5ea38585-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a7fe86385ab2aa74024c6ddb5ea38585-Abstract-Conference.html)

**Abstract**:

Diffusion-based Deep Generative Models (DDGMs) offer state-of-the-art performance in generative modeling. Their main strength comes from their unique setup in which a model (the backward diffusion process) is trained to reverse the forward diffusion process, which gradually adds noise to the input signal. Although DDGMs are well studied, it is still unclear how the small amount of noise is transformed during the backward diffusion process. Here, we focus on analyzing this problem to gain more insight into the behavior of DDGMs and their denoising and generative capabilities. We observe a fluid transition point that changes the functionality of the backward diffusion process from generating a (corrupted) image from noise to denoising the corrupted image to the final sample. Based on this observation, we postulate to divide a DDGM into two parts: a denoiser and a generator. The denoiser could be parameterized by a denoising auto-encoder, while the generator is a diffusion-based model with its own set of parameters. We experimentally validate our proposition, showing its pros and cons.

----

## [1901] On the Tradeoff Between Robustness and Fairness

**Authors**: *Xinsong Ma, Zekai Wang, Weiwei Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a80ebbb4ec9e9b39789318a0a61e2e43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a80ebbb4ec9e9b39789318a0a61e2e43-Abstract-Conference.html)

**Abstract**:

Interestingly, recent experimental results [2, 26, 22] have identified a robust fairness phenomenon in adversarial training (AT), namely that a robust model well-trained by AT exhibits a remarkable disparity of standard accuracy and robust accuracy among different classes compared with natural training. However, the effect of different perturbation radii in AT on robust fairness has not been studied, and one natural question is raised: does a tradeoff exist between average robustness and robust fairness? Our extensive experimental results provide an affirmative answer to this question: with an increasing perturbation radius, stronger AT will lead to a larger class-wise disparity of robust accuracy. Theoretically, we analyze the class-wise performance of adversarially trained linear models with mixture Gaussian distribution. Our theoretical results support our observations. Moreover, our theory shows that  adversarial training easily leads to more serious robust fairness issue than natural training. Motivated by theoretical results, we propose a fairly adversarial training (FAT) method to mitigate the tradeoff between average robustness and robust fairness. Experimental results validate the effectiveness of our proposed method.

----

## [1902] Learning Expressive Meta-Representations with Mixture of Expert Neural Processes

**Authors**: *Qi Wang, Herke van Hoof*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html)

**Abstract**:

Neural processes (NPs) formulate exchangeable stochastic processes and are promising models for meta learning that do not require gradient updates during the testing phase. However, most NP variants place a strong emphasis on a global latent variable. This weakens the approximation power and restricts the scope of applications using NP variants, especially when data generative processes are complicated.To resolve these issues, we propose to combine the Mixture of Expert models with Neural Processes to develop more expressive exchangeable stochastic processes, referred to as Mixture of Expert Neural Processes (MoE-NPs). Then we apply MoE-NPs to both few-shot supervised learning and meta reinforcement learning tasks. Empirical results demonstrate MoE-NPs' strong generalization capability to unseen tasks in these benchmarks.

----

## [1903] Learning Physical Dynamics with Subequivariant Graph Neural Networks

**Authors**: *Jiaqi Han, Wenbing Huang, Hengbo Ma, Jiachen Li, Josh Tenenbaum, Chuang Gan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have become a prevailing tool for learning physical dynamics. However, they still encounter several challenges: 1) Physical laws abide by symmetry,  which is a vital inductive bias accounting for model generalization and should be incorporated into the model design. Existing simulators either consider insufficient symmetry, or enforce excessive equivariance in practice when symmetry is partially broken by gravity. 2) Objects in the physical world possess diverse shapes, sizes, and properties, which should be appropriately processed by the model. To tackle these difficulties, we propose a novel backbone, called Subequivariant Graph Neural Network, which 1) relaxes equivariance to subequivariance by considering external fields like gravity, where the universal approximation ability holds theoretically; 2) introduces a new subequivariant object-aware message passing for learning physical interactions between multiple objects of various shapes in particle-based representation; 3) operates in a hierarchical fashion, allowing for modeling long-range and complex interactions. Our model achieves on average over 3% enhancement in contact prediction accuracy across 8 scenarios on Physion and 2$\times$ lower rollout MSE on RigidFall compared with state-of-the-art GNN simulators, while exhibiting strong generalization and data efficiency.

----

## [1904] DiSC: Differential Spectral Clustering of Features

**Authors**: *Ram Dyuthi Sristi, Gal Mishne, Ariel Jaffe*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a84953147312ea2e8b020e53a267321b-Abstract-Conference.html)

**Abstract**:

Selecting subsets of features that differentiate between two conditions is a key task in a broad range of scientific domains. In many applications, the features of interest form clusters with similar effects on the data at hand. To recover such clusters we develop DiSC, a data-driven approach for detecting groups of features that differentiate between conditions. For each condition, we construct a graph whose nodes correspond to the features and whose weights are functions of the similarity between them for that condition. We then apply a spectral approach to compute subsets of nodes whose connectivity pattern differs significantly between the condition-specific feature graphs. On the theoretical front, we analyze our approach with a toy example based on the stochastic block model. We evaluate DiSC on a variety of datasets, including MNIST, hyperspectral imaging, simulated scRNA-seq and task fMRI, and demonstrate that DiSC uncovers features that better differentiate between conditions compared to competing methods.

----

## [1905] Learn what matters: cross-domain imitation learning with task-relevant embeddings

**Authors**: *Tim Franzmeyer, Philip H. S. Torr, João F. Henriques*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a862f5788fd09bb6843c694d8120d50c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a862f5788fd09bb6843c694d8120d50c-Abstract-Conference.html)

**Abstract**:

We study how an autonomous agent learns to perform a task from demonstrations in a different domain, such as a different environment or different agent. Such cross-domain imitation learning is required to, for example, train an artificial agent from demonstrations of a human expert. We propose a scalable framework that enables cross-domain imitation learning without access to additional demonstrations or further domain knowledge. We jointly train the learner agent's policy and learn a mapping between the learner and expert domains with adversarial training. We effect this by using a mutual information criterion to find an embedding of the expert's state space that contains task-relevant information and is invariant to domain specifics. This step significantly simplifies estimating the mapping between the learner and expert domains and hence facilitates end-to-end learning. We demonstrate successful transfer of policies between considerably different domains, without extra supervision such as additional demonstrations, and in situations where other methods fail.

----

## [1906] UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes

**Authors**: *Alexander Kolesnikov, André Susano Pinto, Lucas Beyer, Xiaohua Zhai, Jeremiah Harmsen, Neil Houlsby*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html)

**Abstract**:

We introduce UViM, a unified approach capable of modeling a wide range of computer vision tasks. In contrast to previous models, UViM has the same functional form for all tasks; it requires no task-specific modifications which require extensive human expertise. The approach involves two components: (I) a base model (feed-forward) which is trained to directly predict raw vision outputs, guided by a learned discrete code and (II) a language model (autoregressive) that is trained to generate the guiding code. These components complement each other: the language model is well-suited to modeling structured interdependent data, while the base model is efficient at dealing with high-dimensional outputs. We demonstrate the effectiveness of UViM on three diverse and challenging vision tasks: panoptic segmentation, depth prediction and image colorization, where we achieve competitive and near state-of-the-art results. Our experimental results suggest that UViM is a promising candidate for a unified modeling approach in computer vision.

----

## [1907] Rotation-Equivariant Conditional Spherical Neural Fields for Learning a Natural Illumination Prior

**Authors**: *James A. D. Gardner, Bernhard Egger, William Smith*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a875c5600e933e56aad7d63439b11b35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a875c5600e933e56aad7d63439b11b35-Abstract-Conference.html)

**Abstract**:

Inverse rendering is an ill-posed problem. Previous work has sought to resolve this by focussing on priors for object or scene shape or appearance. In this work, we instead focus on a prior for natural illuminations. Current methods rely on spherical harmonic lighting or other generic representations and, at best, a simplistic prior on the parameters. We propose a conditional neural field representation based on a variational auto-decoder with a SIREN network and, extending Vector Neurons, build equivariance directly into the network. Using this, we develop a rotation-equivariant, high dynamic range (HDR) neural illumination model that is compact and able to express complex, high-frequency features of natural environment maps. Training our model on a curated dataset of 1.6K HDR environment maps of natural scenes, we compare it against traditional representations, demonstrate its applicability for an inverse rendering task and show environment map completion from partial observations.

----

## [1908] Proximal Learning With Opponent-Learning Awareness

**Authors**: *Stephen Zhao, Chris Lu, Roger B. Grosse, Jakob N. Foerster*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a882dab38011264d2ca8dba3cca9faf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a882dab38011264d2ca8dba3cca9faf1-Abstract-Conference.html)

**Abstract**:

Learning With Opponent-Learning Awareness (LOLA) (Foerster et al. [2018a]) is a multi-agent reinforcement learning algorithm that typically learns reciprocity-based cooperation in partially competitive environments. However, LOLA often fails to learn such behaviour on more complex policy spaces parameterized by neural networks, partly because the update rule is sensitive to the policy parameterization. This problem is especially pronounced in the opponent modeling setting, where the opponent's policy is unknown and must be inferred from observations; in such settings, LOLA is ill-specified because behaviorally equivalent opponent policies can result in non-equivalent updates. To address this shortcoming, we reinterpret LOLA as approximating a proximal operator, and then derive a new algorithm, proximal LOLA (POLA), which uses the proximal formulation directly. Unlike LOLA, the POLA updates are parameterization invariant, in the sense that when the proximal objective has a unique optimum, behaviorally equivalent policies result in behaviorally equivalent updates. We then present practical approximations to the ideal POLA update, which we evaluate in several partially competitive environments with function approximation and opponent modeling. This empirically demonstrates that POLA achieves reciprocity-based cooperation more reliably than LOLA.

----

## [1909] HyperTree Proof Search for Neural Theorem Proving

**Authors**: *Guillaume Lample, Timothée Lacroix, Marie-Anne Lachaux, Aurélien Rodriguez, Amaury Hayat, Thibaut Lavril, Gabriel Ebner, Xavier Martinet*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html)

**Abstract**:

We propose an online training procedure for a transformer-based automated theorem prover. Our approach leverages a new search algorithm, HyperTree Proof Search (HTPS), that learns from previous proof searches through online training, allowing it to generalize to domains far from the training distribution. We report detailed ablations of our pipelineâ€™s main components by studying performance on three environments of increasing complexity. In particular, we show that with HTPS alone, a model trained on annotated proofs manages to prove 65.4% of a held-out set of Metamath theorems, significantly outperforming the previous state of the art of 56.5% by GPT-f. Online training on these unproved theorems increases accuracy to 82.6%. With a similar computational budget, we improve the state of the art on the Lean-based miniF2F-curriculum dataset from 31% to 42% proving accuracy.

----

## [1910] The Policy-gradient Placement and Generative Routing Neural Networks for Chip Design

**Authors**: *Ruoyu Cheng, Xianglong Lyu, Yang Li, Junjie Ye, Jianye Hao, Junchi Yan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a8b8c1ad51df1b93d9e3d1fca75debbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a8b8c1ad51df1b93d9e3d1fca75debbf-Abstract-Conference.html)

**Abstract**:

Placement and routing are two critical yet time-consuming steps of chip design in modern VLSI systems. Distinct from traditional heuristic solvers, this paper on one hand proposes an RL-based model for mixed-size macro placement, which differs from existing learning-based placers that often consider the macro by coarse grid-based mask. While the standard cells are placed via gradient-based GPU acceleration. On the other hand, a one-shot conditional generative routing model, which is composed of a special-designed input-size-adapting generator and a bi-discriminator, is devised to perform one-shot routing to the pins within each net, and the order of nets to route is adaptively learned. Combining these techniques, we develop a flexible and efficient neural pipeline, which to our best knowledge, is the first joint placement and routing network without involving any traditional heuristic solver. Experimental results on chip design benchmarks showcase the effectiveness of our approach, with code that will be made publicly available.

----

## [1911] IMED-RL: Regret optimal learning of ergodic Markov decision processes

**Authors**: *Fabien Pesquerel, Odalric-Ambrym Maillard*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a8c9f9ccc45771d2fd06bcd04ff3442e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a8c9f9ccc45771d2fd06bcd04ff3442e-Abstract-Conference.html)

**Abstract**:

We consider reinforcement learning in a discrete, undiscounted, infinite-horizon Markov decision problem (MDP) under the average reward criterion, and focus on the  minimization of the regret with respect to an optimal policy, when the learner does not know the rewards nor transitions of the MDP. In light of their success at regret minimization in multi-armed bandits, popular bandit strategies, such as the optimistic \texttt{UCB}, \texttt{KL-UCB} or the Bayesian Thompson sampling strategy, have been extended to the MDP setup. Despite some key successes, existing strategies for solving this problem either fail to be provably asymptotically optimal, or suffer from prohibitive burn-in phase and computational complexity when implemented in practice. In this work, we shed a novel light on regret minimization strategies, by extending to reinforcement learning the computationally appealing Indexed Minimum Empirical Divergence (\texttt{IMED}) bandit algorithm. Traditional asymptotic problem-dependent lower bounds on the regret are known under the assumption that the MDP is \emph{ergodic}. Under this assumption, we introduce \texttt{IMED-RL} and prove that its regret upper bound asymptotically matches the regret lower bound. We discuss both the case when the supports of transitions are unknown, and the more informative but a priori harder-to-exploit-optimally case when they are known. Rewards are assumed light-tailed, semi-bounded from above. Last, we provide numerical illustrations on classical tabular MDPs, \textit{ergodic} and \textit{communicative} only, showing the competitiveness of \texttt{IMED-RL} in finite-time against state-of-the-art algorithms. \texttt{IMED-RL} also benefits from a lighter complexity.

----

## [1912] Coresets for Wasserstein Distributionally Robust Optimization Problems

**Authors**: *Ruomin Huang, Jiawei Huang, Wenjie Liu, Hu Ding*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a8e0abdd6a58058d84369dadfcd0905a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a8e0abdd6a58058d84369dadfcd0905a-Abstract-Conference.html)

**Abstract**:

Wasserstein distributionally robust optimization (\textsf{WDRO}) is a popular model to enhance the robustness of machine learning with ambiguous data. However, the complexity of \textsf{WDRO} can be prohibitive in practice since solving its ``minimax'' formulation requires a great amount of computation. Recently, several fast \textsf{WDRO} training algorithms for some specific machine learning tasks (e.g., logistic regression) have been developed. However, the research on designing efficient algorithms for general large-scale \textsf{WDRO}s is still quite limited, to the best of our knowledge. \textit{Coreset} is an important  tool for compressing large dataset, and thus it has been widely applied to  reduce the computational complexities for many optimization problems. In this paper, we introduce a unified framework to construct the $\epsilon$-coreset for the general \textsf{WDRO} problems. Though it is challenging to obtain a conventional coreset for \textsf{WDRO}  due to the uncertainty issue of ambiguous data, we show that we can compute a ``dual coreset'' by using the strong duality property of \textsf{WDRO}. Also, the error introduced by the dual coreset can be theoretically guaranteed for the original \textsf{WDRO} objective. To construct the dual coreset, we propose a novel  grid sampling approach that is particularly suitable for the dual formulation of \textsf{WDRO}. Finally, we implement our coreset approach and illustrate its effectiveness for several \textsf{WDRO} problems in the experiments. See \href{https://arxiv.org/abs/2210.04260}{arXiv:2210.04260} for the full version of this paper. The code is available at \url{https://github.com/h305142/WDRO_coreset}.

----

## [1913] SAMURAI: Shape And Material from Unconstrained Real-world Arbitrary Image collections

**Authors**: *Mark Boss, Andreas Engelhardt, Abhishek Kar, Yuanzhen Li, Deqing Sun, Jonathan T. Barron, Hendrik P. A. Lensch, Varun Jampani*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html)

**Abstract**:

Inverse rendering of an object under entirely unknown capture conditions is a fundamental challenge in computer vision and graphics. Neural approaches such as NeRF have achieved photorealistic results on novel view synthesis, but they require known camera poses. Solving this problem with unknown camera poses is highly challenging as it requires joint optimization over shape, radiance, and pose. This problem is exacerbated when the input images are captured in the wild with varying backgrounds and illuminations. Standard pose estimation techniques fail in such image collections in the wild due to very few estimated correspondences across images. Furthermore, NeRF cannot relight a scene under any illumination, as it operates on radiance (the product of reflectance and illumination). We propose a joint optimization framework to estimate the shape,  BRDF, and per-image camera pose and illumination. Our method works on in-the-wild online image collections of an object and produces relightable 3D assets for several use-cases such as AR/VR. To our knowledge, our method is the first to tackle this severely unconstrained task with minimal user interaction.

----

## [1914] Automatic differentiation of nonsmooth iterative algorithms

**Authors**: *Jérôme Bolte, Edouard Pauwels, Samuel Vaiter*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9077da44185792cb63599cc9e0357bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9077da44185792cb63599cc9e0357bc-Abstract-Conference.html)

**Abstract**:

Differentiation along algorithms, i.e., piggyback propagation of derivatives, is now routinely used to differentiate iterative solvers in differentiable programming. Asymptotics is well understood for many smooth problems but the nondifferentiable case is hardly considered. Is there a limiting object for nonsmooth piggyback automatic differentiation (AD)? Does it have any variational meaning and can it be used effectively in machine learning? Is there a connection with classical derivative? All these questions are addressed under appropriate contractivity conditions in the framework of conservative derivatives which has proved useful in understanding nonsmooth AD. For nonsmooth piggyback iterations, we characterize the attractor set of nonsmooth piggyback iterations as a set-valued fixed point which remains in the conservative framework. This has various consequences and in particular almost everywhere convergence of classical derivatives. Our results are illustrated on parametric convex optimization problems with forward-backward, Douglas-Rachford and Alternating Direction of Multiplier algorithms as well as the Heavy-Ball method.

----

## [1915] Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark

**Authors**: *Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, Chunjing Xu, Hang Xu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a90b9a09a6ee43d6631cf42e225d73b4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a90b9a09a6ee43d6631cf42e225d73b4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Vision-Language Pre-training (VLP) models have shown remarkable performance on various downstream tasks. Their success heavily relies on the scale of pre-trained cross-modal datasets. However, the lack of large-scale datasets and benchmarks in Chinese hinders the development of Chinese VLP models and broader multilingual applications. In this work, we release a large-scale Chinese cross-modal dataset named Wukong, which contains 100 million Chinese image-text pairs collected from the web. Wukong aims to benchmark different multi-modal pre-training methods to facilitate the VLP research and community development. Furthermore, we release a group of models pre-trained with various image encoders (ViT-B/ViT-L/SwinT) and also apply advanced pre-training techniques into VLP such as locked-image text tuning, token-wise similarity in contrastive learning, and reduced-token interaction. Extensive experiments and a benchmarking of different downstream tasks including a new largest human-verified image-text test dataset are also provided. Experiments show that Wukong can serve as a promising Chinese pre-training dataset and benchmark for different cross-modal learning methods. For the zero-shot image classification task on 10 datasets, $Wukong_\text{ViT-L}$ achieves an average accuracy of 73.03%. For the image-text retrieval task, it achieves a mean recall of 71.6% on AIC-ICC which is 12.9% higher than WenLan 2.0. Also, our Wukong models are benchmarked on downstream tasks with other variants on multiple datasets, e.g., Flickr8K-CN, Flickr-30K-CN, COCO-CN, et al. More information can be referred to https://wukong-dataset.github.io/wukong-dataset/.

----

## [1916] Neural Abstractions

**Authors**: *Alessandro Abate, Alec Edwards, Mirco Giacobbe*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a922b7121007768f78f770c404415375-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a922b7121007768f78f770c404415375-Abstract-Conference.html)

**Abstract**:

We present a novel method for the safety verification of nonlinear dynamical models that uses neural networks to represent abstractions of their dynamics. Neural networks have extensively been used before as approximators; in this work, we make a step further and use them for the first time as abstractions. For a given dynamical model, our method synthesises a neural network that overapproximates its dynamics by ensuring an arbitrarily tight, formally certified bound on the approximation error. For this purpose, we employ a counterexample-guided inductive synthesis procedure. We show that this produces a neural ODE with non-deterministic disturbances that constitutes a formal abstraction of the concrete model under analysis. This guarantees a fundamental property: if the abstract model is safe, i.e., free from any initialised trajectory that reaches an undesirable state, then the concrete model is also safe. By using neural ODEs with ReLU activation functions as abstractions, we cast the safety verification problem for nonlinear dynamical models into that of hybrid automata with affine dynamics, which we verify using SpaceEx. We demonstrate that our approach performs comparably to the mature tool Flow* on existing benchmark nonlinear models. We additionally demonstrate and that it is effective on models that do not exhibit local Lipschitz continuity, which are out of reach to the existing technologies.

----

## [1917] Few-Shot Non-Parametric Learning with Deep Latent Variable Model

**Authors**: *Zhiying Jiang, Yiqin Dai, Ji Xin, Ming Li, Jimmy Lin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html)

**Abstract**:

Most real-world problems that machine learning algorithms are expected to solve face the situation with (1) unknown data distribution; (2) little domain-specific knowledge; and (3) datasets with limited annotation. We propose Non-Parametric learning by Compression with Latent Variables (NPC-LV), a learning framework for any dataset with abundant unlabeled data but very few labeled ones. By only training a generative model in an unsupervised way, the framework utilizes the data distribution to build a compressor. Using a compressor-based distance metric derived from Kolmogorov complexity, together with few labeled data, NPC-LV classifies without further training. We show that NPC-LV outperforms supervised methods on all three datasets on image classification in the low data regime and even outperforms semi-supervised learning methods on CIFAR-10. We demonstrate how and when negative evidence lowerbound (nELBO) can be used as an approximate compressed length for classification. By revealing the correlation between compression rate and classification accuracy, we illustrate that under NPC-LV how the improvement of generative models can enhance downstream classification accuracy.

----

## [1918] ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning

**Authors**: *Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, Hongsheng Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract-Conference.html)

**Abstract**:

Capitalizing on large pre-trained models for various downstream tasks of interest have recently emerged with promising performance. Due to the ever-growing model size, the standard full fine-tuning based task adaptation strategy becomes prohibitively costly in terms of model training and storage. This has led to a new research direction in parameter-efficient transfer learning. However, existing attempts typically focus on downstream tasks from the same modality (e.g., image understanding) of the pre-trained model. This creates a limit because in some specific modalities, (e.g., video understanding) such a strong pre-trained model with sufficient knowledge is less or not available. In this work, we investigate such a novel cross-modality transfer learning setting, namely parameter-efficient image-to-video transfer learning. To solve this problem, we propose a new Spatio-Temporal Adapter (ST-Adapter) for parameter-efficient fine-tuning per video task. With a built-in spatio-temporal reasoning capability in a compact design, ST-Adapter enables a pre-trained image model without temporal knowledge to reason about dynamic video content at a small ~8% per-task parameter cost, requiring approximately 20 times fewer updated parameters compared to previous work. Extensive experiments on video action recognition tasks show that our ST-Adapter can match or even outperform the strong full fine-tuning strategy and state-of-the-art video models, whilst enjoying the advantage of parameter efficiency.

----

## [1919] Distilled Gradient Aggregation: Purify Features for Input Attribution in the Deep Neural Network

**Authors**: *Giyoung Jeon, Haedong Jeong, Jaesik Choi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html)

**Abstract**:

Measuring the attribution of input features toward the model output is one of the popular post-hoc explanations on the Deep Neural Networks (DNNs). Among various approaches to compute the attribution, the gradient-based methods are widely used to generate attributions, because of its ease of implementation and the model-agnostic characteristic. However, existing gradient integration methods such as Integrated Gradients (IG) suffer from (1) the noisy attributions which cause the unreliability of the explanation, and (2) the selection for the integration path which determines the quality of explanations. FullGrad (FG) is an another approach to construct the reliable attributions by focusing the locality of piece-wise linear network with the bias gradient. Although FG has shown reasonable performance for the given input, as the shortage of the global property, FG is vulnerable to the small perturbation, while IG which includes the exploration over the input space is robust. In this work, we design a new input attribution method which adopt the strengths of both local and global attributions.In particular, we propose a novel approach to distill input features using weak and extremely positive contributor masks. We aggregate the intermediate local attributions obtained from the distillation sequence to provide reliable attribution. We perform the quantitative evaluation compared to various attribution methods and show that our method outperforms others. We also provide the qualitative result that our method obtains object-aligned and sharp attribution heatmap.

----

## [1920] Temporally Disentangled Representation Learning

**Authors**: *Weiran Yao, Guangyi Chen, Kun Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a938292feb86b94ebe3e6200ff7786ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a938292feb86b94ebe3e6200ff7786ef-Abstract-Conference.html)

**Abstract**:

Recently in the field of unsupervised representation learning, strong identifiability results for disentanglement of causally-related latent variables have been established by exploiting certain side information, such as class labels, in addition to independence. However, most existing work is constrained by functional form assumptions such as independent sources or further with linear transitions, and distribution assumptions such as stationary, exponential family distribution. It is unknown whether the underlying latent variables and their causal relations are identifiable if they have arbitrary, nonparametric causal influences in between.  In this work, we establish the identifiability theories of nonparametric latent causal processes from their nonlinear mixtures under fixed temporal causal influences and analyze how distribution changes can further benefit the disentanglement. We propose TDRL, a principled framework to recover time-delayed latent causal variables and identify their relations from measured sequential data under stationary environments and under different distribution shifts. Specifically, the framework can factorize unknown distribution shifts into transition distribution changes under fixed and time-varying latent causal relations, and under global changes in observation. Through experiments, we show that time-delayed latent causal influences are reliably identified and that our approach considerably outperforms existing baselines that do not correctly exploit this modular representation of changes.

----

## [1921] Can Adversarial Training Be Manipulated By Non-Robust Features?

**Authors**: *Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html)

**Abstract**:

Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attack, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness.

----

## [1922] On the Effectiveness of Fine-tuning Versus Meta-reinforcement Learning

**Authors**: *Mandi Zhao, Pieter Abbeel, Stephen James*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a951f595184aec1bb885ce165b47209a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a951f595184aec1bb885ce165b47209a-Abstract-Conference.html)

**Abstract**:

Intelligent agents should have the ability to leverage knowledge from previously learned tasks in order to learn new ones quickly and efficiently. Meta-learning approaches have emerged as a popular solution to achieve this. However, meta-reinforcement learning (meta-RL) algorithms have thus far been restricted to simple environments with narrow task distributions and have seen limited success. Moreover, the paradigm of pretraining followed by fine-tuning to adapt to new tasks has emerged as a simple yet effective solution in supervised learning. This calls into question the benefits of meta learning approaches also in reinforcement learning, which typically come at the cost of high complexity. We therefore investigate meta-RL approaches in a variety of vision-based benchmarks, including Procgen, RLBench, and Atari, where evaluations are made on completely novel tasks. Our findings show that when meta-learning approaches are evaluated on different tasks (rather than different variations of the same task), multi-task pretraining with fine-tuning on new tasks performs equally as well, or better, than meta-pretraining with meta test-time adaptation. This is encouraging for future research, as multi-task pretraining tends to be simpler and computationally cheaper than meta-RL. From these findings, we advocate for evaluating future meta-RL methods on more challenging tasks and including multi-task pretraining with fine-tuning as a simple, yet strong baseline.

----

## [1923] Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning

**Authors**: *Wenhao Ding, Haohong Lin, Bo Li, Ding Zhao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a96368eb38bce0956a1132154d70d72d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a96368eb38bce0956a1132154d70d72d-Abstract-Conference.html)

**Abstract**:

As a pivotal component to attaining generalizable solutions in human intelligence, reasoning provides great potential for reinforcement learning (RL) agents' generalization towards varied goals by summarizing part-to-whole arguments and discovering cause-and-effect relations. However, how to discover and represent causalities remains a huge gap that hinders the development of causal RL. In this paper, we augment Goal-Conditioned RL (GCRL) with Causal Graph (CG), a structure built upon the relation between objects and events. We novelly formulate the GCRL problem into variational likelihood maximization with CG as latent variables. To optimize the derived objective, we propose a framework with theoretical performance guarantees that alternates between two steps: using interventional data to estimate the posterior of CG; using CG to learn generalizable models and interpretable policies. Due to the lack of public benchmarks that verify generalization capability under reasoning, we design nine tasks and then empirically show the effectiveness of the proposed method against five baselines on these tasks. Further theoretical analysis shows that our performance improvement is attributed to the virtuous cycle of causal discovery, transition modeling, and policy training, which aligns with the experimental evidence in extensive ablation studies.

----

## [1924] WinoGAViL: Gamified Association Benchmark to Challenge Vision-and-Language Models

**Authors**: *Yonatan Bitton, Nitzan Bitton Guetta, Ron Yosef, Yuval Elovici, Mohit Bansal, Gabriel Stanovsky, Roy Schwartz*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a96fe863f85c59789bba63588a9557b4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a96fe863f85c59789bba63588a9557b4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While vision-and-language models perform well on tasks such as visual question answering, they struggle when it comes to basic human commonsense reasoning skills. In this work, we introduce WinoGAViL: an online game of vision-and-language associations (e.g., between werewolves and a full moon), used as a dynamic evaluation benchmark. Inspired by the popular card game Codenames, a spymaster gives a textual cue related to several visual candidates, and another player tries to identify them. Human players are rewarded for creating associations that are challenging for a rival AI model but still solvable by other human players. We use the game to collect 3.5K instances, finding that they are intuitive for humans (>90% Jaccard index) but challenging for state-of-the-art AI models, where the best model (ViLT) achieves a score of 52%, succeeding mostly where the cue is visually salient. Our analysis as well as the feedback we collect from players indicate that the collected associations require diverse reasoning skills, including general knowledge, common sense, abstraction, and more. We release the dataset, the code and the interactive game, allowing future data collection that can be used to develop models with better association abilities.

----

## [1925] Elucidating the Design Space of Diffusion-Based Generative Models

**Authors**: *Tero Karras, Miika Aittala, Timo Aila, Samuli Laine*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)

**Abstract**:

We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.

----

## [1926] TREC: Transient Redundancy Elimination-based Convolution

**Authors**: *Jiawei Guan, Feng Zhang, Jiesong Liu, Hsin-Hsuan Sung, Ruofan Wu, Xiaoyong Du, Xipeng Shen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html)

**Abstract**:

The intensive computations in convolutional neural networks (CNNs) pose challenges for resource-constrained devices; eliminating redundant computations from convolution is essential. This paper gives a principled method to detect and avoid transient redundancy, a type of redundancy existing in input data or activation maps and hence changing across inferences. By introducing a new form of convolution (TREC), this new method makes transient redundancy detection and avoidance an inherent part of the CNN architecture, and the determination of the best configurations for redundancy elimination part of CNN backward propagation. We provide a rigorous proof of the robustness and convergence of TREC-equipped CNNs. TREC removes over 96% computations and achieves 3.51x average speedups on microcontrollers with minimal (about 0.7%) accuracy loss.

----

## [1927] Chaotic Regularization and Heavy-Tailed Limits for Deterministic Gradient Descent

**Authors**: *Soon Hoe Lim, Yijun Wan, Umut Simsekli*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9a238a00f7e885389fc02d65fb00994-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9a238a00f7e885389fc02d65fb00994-Abstract-Conference.html)

**Abstract**:

Recent studies have shown that gradient descent (GD) can achieve improved generalization when its dynamics exhibits a chaotic behavior. However, to obtain the desired effect, the step-size should be chosen sufficiently large, a task which is problem dependent and can be difficult in practice. In this study, we incorporate a chaotic component to GD in a controlled manner, and introduce \emph{multiscale perturbed GD} (MPGD), a novel optimization framework where the GD recursion is augmented with chaotic perturbations that evolve via an independent dynamical system. We analyze MPGD from three different angles: (i) By building up on recent advances in rough paths theory, we show that, under appropriate assumptions, as the step-size decreases, the MPGD recursion converges weakly to a stochastic differential equation (SDE) driven by a heavy-tailed L\'{e}vy-stable process. (ii) By making connections to recently developed generalization bounds for heavy-tailed processes, we derive a generalization bound for the limiting SDE and relate the worst-case generalization error over the trajectories of the process to the parameters of MPGD. (iii) We analyze the implicit regularization effect brought by the dynamical regularization and show that, in the weak perturbation regime, MPGD introduces terms that penalize the Hessian of the loss function. Empirical results are provided to demonstrate the advantages of MPGD.

----

## [1928] DARE: Disentanglement-Augmented Rationale Extraction

**Authors**: *Linan Yue, Qi Liu, Yichao Du, Yanqing An, Li Wang, Enhong Chen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9a67d9309a28372dde3de2a1c837390-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9a67d9309a28372dde3de2a1c837390-Abstract-Conference.html)

**Abstract**:

Rationale extraction can be considered as a straightforward method of improving the model explainability, where rationales are a subsequence of the original inputs, and can be extracted to support the prediction results. Existing methods are mainly cascaded with the selector which extracts the rationale tokens, and the predictor which makes the prediction based on selected tokens. Since previous works fail to fully exploit the original input, where the information of non-selected tokens is ignored, in this paper, we propose a Disentanglement-Augmented Rationale Extraction (DARE) method, which encapsulates more information from the input to extract rationales. Specifically, it first disentangles the input into the rationale representations and the non-rationale ones, and then learns more comprehensive rationale representations for extracting by minimizing the mutual information (MI) between the two disentangled representations. Besides, to improve the performance of MI minimization, we develop a new MI estimator by exploring existing MI estimation methods. Extensive experimental results on three real-world datasets and simulation studies clearly validate the effectiveness of our proposed method. Code is released at https://github.com/yuelinan/DARE.

----

## [1929] The Gyro-Structure of Some Matrix Manifolds

**Authors**: *Xuan Son Nguyen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html)

**Abstract**:

In this paper, we study the gyrovector space structure (gyro-structure) of matrix manifolds. Our work is motivated by the success of hyperbolic neural networks (HNNs) that have demonstrated impressive performance in a variety of applications. At the heart of HNNs is the theory of gyrovector spaces that provides a powerful tool for studying hyperbolic geometry. Here we focus on two matrix manifolds, i.e., Symmetric Positive Definite (SPD) and Grassmann manifolds, and consider connecting the Riemannian geometry of these manifolds with the basic operations, i.e., the binary operation and scalar multiplication on gyrovector spaces. Our work reveals some interesting facts about SPD and Grassmann manifolds. First, SPD matrices with an Affine-Invariant (AI) or a Log-Euclidean (LE) geometry have rich structure with strong connection to hyperbolic geometry. Second, linear subspaces, when equipped with our proposed basic operations, form what we call gyrocommutative and gyrononreductive gyrogroups. Furthermore, they share remarkable analogies with gyrovector spaces. We demonstrate the applicability of our approach for human activity understanding and question answering.

----

## [1930] SMPL: Simulated Industrial Manufacturing and Process Control Learning Environments

**Authors**: *Mohan Zhang, Xiaozhou Wang, Benjamin Decardi-Nelson, Bo Song, An Zhang, Jinfeng Liu, Sile Tao, Jiayi Cheng, Xiaohong Liu, Dengdeng Yu, Matthew Poon, Animesh Garg*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9b3d7f65eebb083e5c7f8cf10e52528-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9b3d7f65eebb083e5c7f8cf10e52528-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Traditional biological and pharmaceutical manufacturing plants are controlled by human workers or pre-defined thresholds. Modernized factories have advanced process control algorithms such as model predictive control (MPC). However, there is little exploration of applying deep reinforcement learning to control manufacturing plants. One of the reasons is the lack of high fidelity simulations and standard APIs for benchmarking. To bridge this gap, we develop an easy-to-use library that includes five high-fidelity simulation environments: BeerFMTEnv, ReactorEnv, AtropineEnv, PenSimEnv and mAbEnv, which cover a wide range of manufacturing processes. We build these environments on published dynamics models. Furthermore, we benchmark online and offline, model-based and model-free reinforcement learning algorithms for comparisons of follow-up research.

----

## [1931] CoPur: Certifiably Robust Collaborative Inference via Feature Purification

**Authors**: *Jing Liu, Chulin Xie, Sanmi Koyejo, Bo Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/a9c7200b0f37dc58e6bb97d45ff8faf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/a9c7200b0f37dc58e6bb97d45ff8faf1-Abstract-Conference.html)

**Abstract**:

Collaborative inference leverages diverse features provided by different agents (e.g., sensors) for more accurate inference. A common setup is where each agent sends its embedded features instead of the raw data to the Fusion Center (FC) for joint prediction. In this setting, we consider the inference-time attacks when a small fraction of agents are compromised. The compromised agent either does not send embedded features to the FC, or sends arbitrarily embedded features. To address this, we propose a certifiably robust COllaborative inference framework via feature PURification (CoPur), by leveraging the block-sparse nature of adversarial perturbations on the feature vector, as well as exploring the underlying redundancy across the embedded features (by assuming the overall features lie on an underlying lower dimensional manifold). We theoretically show that the proposed feature purification method can robustly recover the true feature vector, despite adversarial corruptions and/or incomplete observations. We also propose and test an untargeted distributed feature-flipping attack, which is agnostic to the model, training data, label, as well as the features held by other agents, and is shown to be effective in attacking state-of-the-art defenses. Experiments on ExtraSensory and NUS-WIDE datasets show that CoPur significantly outperforms existing defenses in terms of robustness against targeted and untargeted adversarial attacks.

----

## [1932] Toward Understanding Privileged Features Distillation in Learning-to-Rank

**Authors**: *Shuo Yang, Sujay Sanghavi, Holakou Rahmanian, Jan Bakus, S. V. N. Vishwanathan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html)

**Abstract**:

In learning-to-rank problems, a \textit{privileged feature} is one that is available during model training, but not available at test time. Such features naturally arise in merchandised recommendation systems; for instance, "user clicked this item" as a feature is predictive of "user purchased this item" in the offline data, but is clearly not available during online serving. Another source of privileged features is those that are too expensive to compute online but feasible to be added offline. \textit{Privileged features distillation} (PFD) refers to a natural idea: train a "teacher" model using all features (including privileged ones) and then use it to train a "student" model that does not use the privileged features.  In this paper, we first study PFD empirically on three public ranking datasets and an industrial-scale ranking problem derived from Amazon's logs. We show that PFD outperforms several baselines (no-distillation, pretraining-finetuning, self-distillation, and generalized distillation) on all these datasets. Next, we analyze why and when PFD performs well via both empirical ablation studies and theoretical analysis for linear models. Both investigations uncover an interesting non-monotone behavior: as the predictive power of a privileged feature increases, the performance of the resulting student model initially increases but then decreases. We show the reason for the later decreasing performance is that a very predictive privileged teacher produces predictions with high variance, which lead to high variance student estimates and inferior testing performance.

----

## [1933] Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods

**Authors**: *Randall Balestriero, Yann LeCun*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aa56c74513a5e35768a11f4e82dd7ffb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aa56c74513a5e35768a11f4e82dd7ffb-Abstract-Conference.html)

**Abstract**:

Self-Supervised Learning (SSL) surmises that inputs and pairwise positive relationships are enough to learn meaningful representations. Although SSL has recently reached a milestone: outperforming supervised methods in many modalities\dots the theoretical foundations are limited, method-specific, and fail to provide principled design guidelines to practitioners. In this paper, we propose a unifying framework under the helm of spectral manifold learning. Through the course of this study, we will demonstrate that VICReg, SimCLR, BarlowTwins et al. correspond to eponymous spectral methods such as Laplacian Eigenmaps, ISOMAP et al.From this unified viewpoint, we obtain (i) the close-form optimal representation, (ii) the close-form optimal network parameters in the linear regime, (iii) the impact of the pairwise relations used during training on each of those quantities and on downstream task performances, and most importantly, (iv) the first theoretical bridge between contrastive and non-contrastive methods to global and local spectral methods respectively hinting at the benefits and limitations of each. For example, if the pairwise relation is aligned with the downstream task, all SSL methods produce optimal representations for that downstream task.

----

## [1934] Subgame Solving in Adversarial Team Games

**Authors**: *Brian Hu Zhang, Luca Carminati, Federico Cacciamani, Gabriele Farina, Pierriccardo Olivieri, Nicola Gatti, Tuomas Sandholm*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aa5f5e6eb6f613ec412f1d948dfa21a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aa5f5e6eb6f613ec412f1d948dfa21a5-Abstract-Conference.html)

**Abstract**:

In adversarial team games, a team of players sequentially faces a team of adversaries. These games are the simplest setting with multiple players where cooperation and competition coexist, and it is known that the information asymmetry among the team members makes equilibrium approximation computationally hard. Although much effort has been spent designing scalable algorithms, the problem of solving large game instances is open. In this paper, we extend the successful approach of solving huge two-player zero-sum games, where a blueprint strategy is computed offline by using an abstract version of the game and then it is refined online, that is, during a playthrough. In particular, to the best of our knowledge, our paper provides the first method for online strategy refinement via subgame solving in adversarial team games. Our method, based on the team belief DAG, generates a gadget game and then refine the blueprint strategy by using column-generation approaches in anytime fashion. If the blueprint is sparse, then our whole algorithm runs end-to-end in polynomial time given a best-response oracle; in particular, it avoids expanding the whole team belief DAG, which has exponential worst-case size. We apply our method to a standard test suite, and we empirically show the performance improvement of the strategies thanks to subgame solving.

----

## [1935] A framework for bilevel optimization that enables stochastic and global variance reduction algorithms

**Authors**: *Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, Thomas Moreau*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aa84ec1ac3f5fdcf77bce2c22705ab77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aa84ec1ac3f5fdcf77bce2c22705ab77-Abstract-Conference.html)

**Abstract**:

Bilevel optimization, the problem of minimizing a value function which involves the arg-minimum of another function, appears in many areas of machine learning. In a large scale empirical risk minimization setting where the number of samples is huge, it is crucial to develop stochastic methods, which only use a few samples at a time to progress. However, computing the gradient of the value function involves solving a linear system, which makes it difficult to derive unbiased stochastic estimates.To overcome this problem we introduce a novel framework, in which the solution of the inner problem, the solution of the linear system, and the main variable evolve at the same time. These directions are written as a sum, making it straightforward to derive unbiased estimates.The simplicity of our approach allows us to develop global variance reduction algorithms, where the dynamics of all variables is subject to variance reduction.We demonstrate that SABA, an adaptation of the celebrated SAGA algorithm in our framework, has $O(\frac1T)$ convergence rate, and that it achieves linear convergence under Polyak-Lojasciewicz assumption.This is the first stochastic algorithm for bilevel optimization that verifies either of these properties.Numerical experiments validate the usefulness of our method.

----

## [1936] Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution

**Authors**: *Leon Hetzel, Simon Böhm, Niki Kilbertus, Stephan Günnemann, Mohammad Lotfollahi, Fabian J. Theis*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html)

**Abstract**:

Single-cell transcriptomics enabled the study of cellular heterogeneity in response to perturbations at the resolution of individual cells. However, scaling high-throughput screens (HTSs) to measure cellular responses for many drugs remains a challenge due to technical limitations and, more importantly, the cost of such multiplexed experiments. Thus, transferring information from routinely performed bulk RNA HTS is required to enrich single-cell data meaningfully.We introduce chemCPA, a new encoder-decoder architecture to study the perturbational effects of unseen drugs. We combine the model with an architecture surgery for transfer learning and demonstrate how training on existing bulk RNA HTS datasets can improve generalisation performance. Better generalisation reduces the need for extensive and costly screens at single-cell resolution. We envision that our proposed method will facilitate more efficient experiment designs through its ability to generate in-silico hypotheses, ultimately accelerating drug discovery.

----

## [1937] Accelerated Projected Gradient Algorithms for Sparsity Constrained Optimization Problems

**Authors**: *Jan Harold Alcantara, Ching-pei Lee*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aab3003c922e0fcd2fd2c951fa3c03ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aab3003c922e0fcd2fd2c951fa3c03ad-Abstract-Conference.html)

**Abstract**:

We consider the projected gradient algorithm for the nonconvex best subset selection problem that minimizes a given empirical loss function under an $\ell_0$-norm constraint. Through decomposing the feasible set of the given sparsity constraint as a finite union of linear subspaces, we present two acceleration schemes with global convergence guarantees, one by same-space extrapolation and the other by subspace identification. The former fully utilizes the problem structure to greatly accelerate the optimization speed with only negligible additional cost. The latter leads to a two-stage meta-algorithm that first uses classical projected gradient iterations to identify the correct subspace containing an optimal solution, and then switches to a highly-efficient smooth optimization method in the identified subspace to attain superlinear convergence. Experiments demonstrate that the proposed accelerated algorithms are magnitudes faster than their non-accelerated counterparts as well as the state of the art.

----

## [1938] The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models

**Authors**: *Conglong Li, Minjia Zhang, Yuxiong He*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aac02401755a65904cf977a33136af4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aac02401755a65904cf977a33136af4a-Abstract-Conference.html)

**Abstract**:

Recent works have demonstrated great success in pre-training large-scale autoregressive language models (e.g., GPT-3) on massive GPUs. To reduce the wall-clock training time, a common practice is to increase the batch size and learning rate. However, such practice is often brittle and leads to a so-called stability-efficiency dilemma: increasing the batch sizes and learning rates leads to better training efficiency but can also result in training instability, leading to poor generalization accuracy or failed runs. To better understand this phenomenon, we conduct an in-depth analysis on large-scale pre-training experiments replicating the GPT-2 model with public dataset. We find that there is a strong correlation between training instability and extreme values of gradient variance. We further identify that samples with long sequence lengths contribute to these extreme gradient variance values, especially at the beginning of the training, indicating that long sequence length can be a main source of training instability.Based on the analysis, we present a simple yet effective Sequence Length Warmup method that aims to solve the training stability-efficiency dilemma by avoiding extreme gradient variance values. Moreover, we present a lightweight tuning strategy that allows us to tune our method with just a small portion of the expensive full training. Experiments replicating GPT-2 models (117M and 1.5B) show that our approach enables stable training with 8x larger batch size and 4x larger learning rate, whereas the baseline approach struggles with training instability. To achieve the same or better zero-shot evaluation results, our method reduces the required number of training tokens and wall clock time by up to 2.2x and 3.7x, respectively. Experiments replicating GPT-3 model (125M) show that our approach enables stable training with 8x larger batch size and 40x larger learning rate, and retains 99\% of the zero-shot accuracy on 11 tasks using 10x less data and 17x less time compared to the original GPT-3 training recipe, while the baseline diverges under the same settings and only retain 95\% of accuracy under lower learning rate.

----

## [1939] Trading off Image Quality for Robustness is not Necessary with Regularized Deterministic Autoencoders

**Authors**: *Amrutha Saseendran, Kathrin Skubch, Stefan Falkner, Margret Keuper*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aae3ff05a5638ce4e2ef2fbc04229797-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aae3ff05a5638ce4e2ef2fbc04229797-Abstract-Conference.html)

**Abstract**:

The susceptibility of Variational Autoencoders (VAEs) to adversarial attacks indicates the necessity to evaluate the robustness of the learned representations along with the generation performance. The vulnerability of VAEs has been attributed to the limitations associated with their variational formulation. Deterministic autoencoders could overcome the practical limitations associated with VAEs and offer a promising alternative for image generation applications. In this work, we propose an adversarially robust deterministic autoencoder with superior performance in terms of both generation and robustness of the learned representations. We introduce a regularization scheme to incorporate adversarially perturbed data points to the training pipeline without increasing the computational complexity or compromising the generation fidelity by leveraging a loss based on the two-point Kolmogorovâ€“Smirnov test between representations. We conduct extensive experimental studies on popular image benchmark datasets to quantify the robustness of the proposed approach based on the adversarial attacks targeted at VAEs. Our empirical findings show that the proposed method achieves significant performance in both robustness and fidelity when compared to the robust VAE models.

----

## [1940] Does Momentum Change the Implicit Regularization on Separable Data?

**Authors**: *Bohan Wang, Qi Meng, Huishuai Zhang, Ruoyu Sun, Wei Chen, Zhi-Ming Ma, Tie-Yan Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ab3f6bbe121a8f7a0263a9b393000741-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ab3f6bbe121a8f7a0263a9b393000741-Abstract-Conference.html)

**Abstract**:

The momentum acceleration technique is widely adopted in many optimization algorithms. However, there is no theoretical answer on how the momentum affects the generalization performance of the optimization algorithms. This paper studies this problem by analyzing the implicit regularization of momentum-based optimization. We prove that on the linear classification problem with separable data and exponential-tailed loss, gradient descent with momentum (GDM) converges to the $L^2$ max-margin solution, which is the same as vanilla gradient descent. That means gradient descent with momentum acceleration still converges to a low-complexity model, which guarantees their generalization. We then analyze the stochastic and adaptive variants of GDM (i.e., SGDM and deterministic Adam) and show they also converge to the $L^2$ max-margin solution.  Technically, the implicit regularization of SGDM is established based on a novel convergence analysis of SGDM under a general noise condition called affine noise variance condition. To the best of our knowledge, we are the first to derive SGDMâ€™s convergence under such an assumption. Numerical experiments are conducted to support our theoretical results.

----

## [1941] Generalization Gap in Amortized Inference

**Authors**: *Mingtian Zhang, Peter Hayes, David Barber*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ab41313eaa3cbedbe491c24cbfe6547d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ab41313eaa3cbedbe491c24cbfe6547d-Abstract-Conference.html)

**Abstract**:

The ability of likelihood-based probabilistic models to generalize to unseen data is central to many machine learning applications such as lossless compression. In this work,  we study the generalization of a popular class of probabilistic model - the Variational Auto-Encoder (VAE). We discuss the two generalization gaps that affect VAEs and show that overfitting is usually dominated by amortized inference. Based on this observation, we propose a new training objective that improves the generalization of amortized inference. We demonstrate how our method can improve performance in the context of image modeling and lossless compression.

----

## [1942] Top Two Algorithms Revisited

**Authors**: *Marc Jourdan, Rémy Degenne, Dorian Baudry, Rianne de Heide, Emilie Kaufmann*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ab5f5f22e3e09f4424592ffb06840ab0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ab5f5f22e3e09f4424592ffb06840ab0-Abstract-Conference.html)

**Abstract**:

Top two algorithms arose as an adaptation of Thompson sampling to best arm identification in multi-armed bandit models for parametric families of arms. They select the next arm to sample from by randomizing among two candidate arms, a leader and a challenger. Despite their good empirical performance, theoretical guarantees for fixed-confidence best arm identification have only been obtained when the arms are Gaussian with known variances. In this paper, we provide a general analysis of top-two methods, which identifies desirable properties of the leader, the challenger, and the (possibly non-parametric) distributions of the arms. As a result, we obtain theoretically supported top-two algorithms for best arm identification with bounded distributions. Our proof method demonstrates in particular that the sampling step used to select the leader inherited from Thompson sampling can be replaced by other choices, like selecting the empirical best arm.

----

## [1943] Relation-Constrained Decoding for Text Generation

**Authors**: *Xiang Chen, Zhixian Yang, Xiaojun Wan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ab63a1a325670278ba9b87fbc3e95e33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ab63a1a325670278ba9b87fbc3e95e33-Abstract-Conference.html)

**Abstract**:

The dominant paradigm for neural text generation nowadays is seq2seq learning with large-scale pretrained language models. However, it is usually difficult to manually constrain the generation process of these models. Prior studies have introduced Lexically Constrained Decoding (LCD) to ensure the presence of pre-specified words or phrases in the output. However, simply applying lexical constraints has no guarantee of the grammatical or semantic relations between words. Thus, more elaborate constraints are needed. To this end, we first propose a new constrained decoding scenario named Relation-Constrained Decoding (RCD), which requires the model's output to contain several given word pairs with respect to the given relations between them. For this scenario, we present a novel plug-and-play decoding algorithm named RElation-guided probability Surgery and bEam ALlocation (RESEAL), which can handle different categories of relations, e.g., syntactical relations or factual relations. Moreover, RESEAL can adaptively "reseal" the relations to form a high-quality sentence, which can be applied to the inference stage of any autoregressive text generation model. To evaluate our method, we first construct an RCD benchmark based on dependency relations from treebanks with annotated dependencies. Experimental results demonstrate that our approach can achieve better preservation of the input dependency relations compared to previous methods. To further illustrate the effectiveness of RESEAL, we apply our method to three downstream tasks: sentence summarization, fact-based text editing, and data-to-text generation. We observe an improvement in generation quality. The source code is available at https://github.com/CasparSwift/RESEAL.

----

## [1944] Learning General World Models in a Handful of Reward-Free Deployments

**Authors**: *Yingchen Xu, Jack Parker-Holder, Aldo Pacchiano, Philip J. Ball, Oleh Rybkin, Stephen Roberts, Tim Rocktäschel, Edward Grefenstette*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ab6a2c6ee757afe43882121281f6065c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ab6a2c6ee757afe43882121281f6065c-Abstract-Conference.html)

**Abstract**:

Building generally capable agents is a grand challenge for deep reinforcement learning (RL). To approach this challenge practically, we outline two key desiderata: 1) to facilitate generalization, exploration should be task agnostic; 2) to facilitate scalability, exploration policies should collect large quantities of data without costly centralized retraining. Combining these two properties, we introduce the reward-free deployment efficiency setting, a new paradigm for RL research. We then present CASCADE, a novel approach for self-supervised exploration in this new setting. CASCADE seeks to learn a world model by collecting data with a population of agents, using an information theoretic objective inspired by Bayesian Active Learning. CASCADE achieves this by specifically maximizing the diversity of trajectories sampled by the population through a novel cascading objective. We provide theoretical intuition for CASCADE which we show in a tabular setting improves upon naïve approaches that do not account for population diversity. We then demonstrate that CASCADE collects diverse task-agnostic datasets and learns agents that generalize zero-shot to novel, unseen downstream tasks on Atari, MiniGrid, Crafter and the DM Control Suite. Code and videos are available at https://ycxuyingchen.github.io/cascade/

----

## [1945] Bringing Image Scene Structure to Video via Frame-Clip Consistency of Object Tokens

**Authors**: *Elad Ben-Avraham, Roei Herzig, Karttikeya Mangalam, Amir Bar, Anna Rohrbach, Leonid Karlinsky, Trevor Darrell, Amir Globerson*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/abc1943857a42935ceacff03c524bb44-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/abc1943857a42935ceacff03c524bb44-Abstract-Conference.html)

**Abstract**:

Recent action recognition models have achieved impressive results by integrating objects, their locations and interactions. However, obtaining dense structured annotations for each frame is tedious and time-consuming, making these methods expensive to train and less scalable. At the same time, if a small set of annotated images is available, either within or outside the domain of interest, how could we leverage these for a video downstream task? We propose a learning framework StructureViT (SViT for short), which demonstrates how utilizing the structure of a small number of images only available during training can improve a video model. SViT relies on two key insights. First, as both images and videos contain structured information, we enrich a transformer model with a set of object tokens that can be used across images and videos. Second, the scene representations of individual frames in video should ``align'' with those of still images. This is achieved via a Frame-Clip Consistency loss, which ensures the flow of structured information between images and videos. We explore a particular instantiation of scene structure, namely a Hand-Object Graph, consisting of hands and objects with their locations as nodes, and physical relations of contact/no-contact as edges. SViT shows strong performance improvements on multiple video understanding tasks and datasets, including the first place in the Ego4D CVPR'22 Point of No Return Temporal Localization Challenge. For code and pretrained models, visit the project page at https://eladb3.github.io/SViT/.

----

## [1946] Probabilistic Transformer: Modelling Ambiguities and Distributions for RNA Folding and Molecule Design

**Authors**: *Jörg K. H. Franke, Frederic Runge, Frank Hutter*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/abf0ea3ae33d1a931483e327ff8d94f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/abf0ea3ae33d1a931483e327ff8d94f8-Abstract-Conference.html)

**Abstract**:

Our world is ambiguous and this is reflected in the data we use to train our algorithms. This is particularly true when we try to model natural processes where collected data is affected by noisy measurements and differences in measurement techniques. Sometimes, the process itself is ambiguous, such as in the case of RNA folding, where the same nucleotide sequence can fold into different structures. This suggests that a predictive model should have similar probabilistic characteristics to match the data it models. Therefore, we propose a hierarchical latent distribution to enhance one of the most successful deep learning models, the Transformer, to accommodate ambiguities and data distributions. We show the benefits of our approach (1) on a synthetic task that captures the ability to learn a hidden data distribution, (2) with state-of-the-art results in RNA folding that reveal advantages on highly ambiguous data, and (3) demonstrating its generative capabilities on property-based molecule design by implicitly learning the underlying distributions and outperforming existing work.

----

## [1947] PulseImpute: A Novel Benchmark Task for Pulsative Physiological Signal Imputation

**Authors**: *Maxwell A. Xu, Alexander Moreno, Supriya Nagesh, Varol Burak Aydemir, David W. Wetter, Santosh Kumar, James M. Rehg*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The promise of Mobile Health (mHealth) is the ability to use wearable sensors to monitor participant physiology at high frequencies during daily life to enable temporally-precise health interventions. However, a major challenge is frequent missing data. Despite a rich imputation literature, existing techniques are ineffective for the pulsative signals which comprise many mHealth applications, and a lack of available datasets has stymied progress. We address this gap with PulseImpute, the first large-scale pulsative signal imputation challenge which includes realistic mHealth missingness models, an extensive set of baselines, and clinically-relevant downstream tasks. Our baseline models include a novel transformer-based architecture designed to exploit the structure of pulsative signals. We hope that PulseImpute will enable the ML community to tackle this important and challenging task.

----

## [1948] Beyond Separability: Analyzing the Linear Transferability of Contrastive Representations to Related Subpopulations

**Authors**: *Jeff Z. HaoChen, Colin Wei, Ananya Kumar, Tengyu Ma*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html)

**Abstract**:

Contrastive learning is a highly effective method for learning representations from unlabeled data. Recent works show that contrastive representations can transfer across domains, leading to simple state-of-the-art algorithms for unsupervised domain adaptation. In particular, a linear classifier trained to separate the representations on the source domain can also predict classes on the target domain accurately, even though the representations of the two domains are far from each other. We refer to this phenomenon as linear transferability. This paper analyzes when and why contrastive representations exhibit linear transferability in a general unsupervised domain adaptation setting. We prove that linear transferability can occur when data from the same class in different domains (e.g., photo dogs and cartoon dogs) are more related with each other than data from different classes in different domains (e.g., photo dogs and cartoon cats) are. Our analyses are in a realistic regime where the source and target domains can have unbounded density ratios and be weakly related, and they have distant representations across domains.

----

## [1949] Precise Regret Bounds for Log-loss via a Truncated Bayesian Algorithm

**Authors**: *Changlong Wu, Mohsen Heidari, Ananth Grama, Wojciech Szpankowski*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac1887299ee703ba4e54f8c102161213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac1887299ee703ba4e54f8c102161213-Abstract-Conference.html)

**Abstract**:

We study sequential general online regression, known also as sequential probability assignments, under logarithmic loss when compared against a broad class of experts. We obtain tight, often matching, lower and upper bounds for sequential minimax regret, which is defined as the excess loss incurred by the predictor over the best expert in the class. After proving a general upper bound we consider some specific classes of experts from Lipschitz class to bounded Hessian class and derive matching lower and upper bounds with provably optimal constants. Our bounds work for a wide range of values of the data dimension and the number of rounds. To derive lower bounds, we use tools from information theory (e.g., Shtarkov sum) and for upper bounds, we resort to new "smooth truncated covering" of the class of experts. This allows us to find constructive proofs by applying a simple and novel truncated Bayesian algorithm. Our proofs are substantially simpler than the existing ones and yet provide tighter (and often optimal) bounds.

----

## [1950] What are the best Systems? New Perspectives on NLP Benchmarking

**Authors**: *Pierre Colombo, Nathan Noiry, Ekhine Irurozki, Stéphan Clémençon*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac4920f4085b5662133dd751493946a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac4920f4085b5662133dd751493946a6-Abstract-Conference.html)

**Abstract**:

In Machine Learning, a benchmark refers to an ensemble of datasets associated with one or multiple metrics together with a way to aggregate different systems performances. They are instrumental in {\it (i)}  assessing the progress of new methods along different axes and {\it (ii)} selecting the best systems for practical use. This is particularly the case for NLP with the development of large pre-trained models (\textit{e.g.} GPT, BERT) that are expected to generalize well on a variety of tasks. While the community mainly focused on developing new datasets and metrics, there has been little interest in the aggregation procedure, which is often reduced to a simple average over various performance measures. However, this procedure can be problematic when the metrics are on a different scale, which may lead to spurious conclusions. This paper proposes a new procedure to rank systems based on their performance across different tasks. Motivated by the social choice theory, the final system ordering is obtained through aggregating the rankings induced by each task and is theoretically grounded. We conduct extensive numerical experiments (on over 270k scores) to assess the soundness of our approach both on synthetic and real scores (\textit{e.g.} GLUE, EXTREM, SEVAL, TAC, FLICKR). In particular, we show that our method yields different conclusions on state-of-the-art systems than the mean-aggregation procedure while being both more reliable and robust.

----

## [1951] Learning from Label Proportions by Learning with Label Noise

**Authors**: *Jianxin Zhang, Yutong Wang, Clayton Scott*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac56fb3fab015124b541f6299016a21c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac56fb3fab015124b541f6299016a21c-Abstract-Conference.html)

**Abstract**:

Learning from label proportions (LLP) is a weakly supervised classification problem where data points are grouped into bags, and the label proportions within each bag are observed instead of the instance-level labels. The task is to learn a classifier to predict the labels of future individual instances. Prior work on LLP for multi-class data has yet to develop a theoretically grounded algorithm. In this work, we propose an approach to LLP based on a reduction to learning with label noise, using the forward correction (FC) loss of \textcite{Patrini2017MakingDN}. We establish an excess risk bound and generalization error analysis for our approach, while also extending the theory of the FC loss which may be of independent interest. Our approach demonstrates improved empirical performance in deep learning scenarios across multiple datasets and architectures, compared to the leading methods.

----

## [1952] Dynamics of SGD with Stochastic Polyak Stepsizes: Truly Adaptive Variants and Convergence to Exact Solution

**Authors**: *Antonio Orvieto, Simon Lacoste-Julien, Nicolas Loizou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)

**Abstract**:

Recently Loizou et al. (2021), proposed and analyzed stochastic gradient descent (SGD) with stochastic Polyak stepsize (SPS). The proposed SPS comes with strong convergence guarantees and competitive performance; however, it has two main drawbacks when it is used in non-over-parameterized regimes: (i) It requires a priori knowledge of the optimal mini-batch losses, which are not available when the interpolation condition is not satisfied (e.g., regularized objectives), and (ii) it guarantees convergence only to a neighborhood of the solution. In this work, we study the dynamics and the convergence properties of SGD equipped with new variants of the stochastic Polyak stepsize and provide solutions to both drawbacks of the original SPS. We first show that a simple modification of the original SPS that uses lower bounds instead of the optimal function values can directly solve issue (i). On the other hand, solving issue (ii) turns out to be more challenging and leads us to valuable insights into the method's behavior. We show that if interpolation is not satisfied, the correlation between SPS and stochastic gradients introduces a bias, which effectively distorts the expectation of the gradient signal near minimizers, leading to non-convergence - even if the stepsize is scaled down during training. To fix this issue, we propose DecSPS, a novel modification of SPS, which guarantees convergence to the exact minimizer - without a priori knowledge of the problem parameters. For strongly-convex optimization problems, DecSPS is the first stochastic adaptive optimization method that converges to the exact solution without restrictive assumptions like bounded iterates/gradients.

----

## [1953] MORA: Improving Ensemble Robustness Evaluation with Model Reweighing Attack

**Authors**: *Yunrui Yu, Xitong Gao, Cheng-Zhong Xu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac895e51849bfc99ae25e054fd4c2eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac895e51849bfc99ae25e054fd4c2eda-Abstract-Conference.html)

**Abstract**:

Adversarial attacks can deceive neural networks by adding tiny perturbations to their input data.  Ensemble defenses, which are trained to minimize attack transferability among sub-models, offer a promising research direction to improve robustness against such attacks while maintaining a high accuracy on natural inputs.  We discover, however, that recent state-of-the-art (SOTA) adversarial attack strategies cannot reliably evaluate ensemble defenses, sizeably overestimating their robustness.  This paper identifies the two factors that contribute to this behavior.  First, these defenses form ensembles that are notably difficult for existing gradient-based method to attack, due to gradient obfuscation.  Second, ensemble defenses diversify sub-model gradients, presenting a challenge to defeat all sub-models simultaneously, simply summing their contributions may counteract the overall attack objective; yet, we observe that ensemble may still be fooled despite most sub-models being correct.  We therefore introduce MORA, a model-reweighing attack to steer adversarial example synthesis by reweighing the importance of sub-model gradients.  MORA finds that recent ensemble defenses all exhibit varying degrees of overestimated robustness.  Comparing it against recent SOTA white-box attacks, it can converge orders of magnitude faster while achieving higher attack success rates across all ensemble models examined with three different ensemble modes (i.e, ensembling by either softmax, voting or logits).  In particular, most ensemble defenses exhibit near or exactly $0\%$ robustness against MORA with $\ell^\infty$ perturbation within $0.02$ on CIFAR-10, and $0.01$ on CIFAR-100.  We make MORA open source with reproducible results and pre-trained models; and provide a leaderboard of ensemble defenses under various attack strategies.

----

## [1954] Faster and Scalable Algorithms for Densest Subgraph and Decomposition

**Authors**: *Elfarouk Harb, Kent Quanrud, Chandra Chekuri*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html)

**Abstract**:

We study the densest subgraph problem (DSG) and the densest subgraph local decomposition problem (DSG-LD) in undirected graphs. We also consider supermodular generalizations of these problems. For large scale graphs simple iterative algorithms perform much better in practice than theoretically fast algorithms based on network-flow or LP solvers. Boob et al [1] recently gave a fast iterative algorithm called Greedy++ for DSG. It was shown in [2] that it converges to a $(1-\epsilon)$ relative approximation to the optimum density in $O(\frac{1}{\epsilon^2} \frac{\Delta(G)}{\lambda^*})$ iterations where $\Delta(G)$ is the maximum degree and $\lambda^*$ is the optimum density. Danisch et al. [3] gave an iterative algorithm based on the Frank-Wolfe algorithm for DSG-LD that takes $O(\frac{m\Delta(G) }{\epsilon^2})$ iterations to converge to an $\epsilon$-additive approximate local decomposition vector $\hat{b}$, where $m$ is number of edges in the graph.In this paper we give a new iterative algorithm for both problems that takes at most $O(\frac{\sqrt{m\Delta(G)}}{\epsilon})$ iterations to converge to an $\epsilon$-additive approximate local decomposition vector; each iteration can be implemented in $O(m)$ time. We describe a fractional peeling technique which has strong empirical performance as well as theoretical guarantees. The algorithm is scalable and simple, and can be applied to graphs with hundreds of millions of edges. We test our algorithm on real and synthetic data sets and show that it provides a significant benefit over previous algorithms. The algorithm and analysis extends to hypergraphs.

----

## [1955] Extrapolation and Spectral Bias of Neural Nets with Hadamard Product: a Polynomial Net Study

**Authors**: *Yongtao Wu, Zhenyu Zhu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/acb3565a58dea4c39c84af35d4225d97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/acb3565a58dea4c39c84af35d4225d97-Abstract-Conference.html)

**Abstract**:

Neural tangent kernel (NTK) is a powerful tool to analyze training dynamics of neural networks and their generalization bounds. The study on NTK has been devoted to typical neural network architectures, but it is incomplete for neural networks with Hadamard products (NNs-Hp), e.g., StyleGAN and polynomial neural networks (PNNs). In this work, we derive the finite-width NTK formulation for a special class of NNs-Hp, i.e., polynomial neural networks. We prove their equivalence to the kernel regression predictor with the associated NTK, which expands the application scope of NTK. Based on our results, we elucidate the separation of PNNs over standard neural networks with respect to extrapolation and spectral bias. Our two key insights are that when compared to standard neural networks, PNNs can fit more complicated functions in the extrapolation regime and admit a slower eigenvalue decay of the respective NTK, leading to a faster learning towards high-frequency functions. Besides, our theoretical results can be extended to other types of NNs-Hp, which expand the scope of our work. Our empirical results validate the separations in broader classes of NNs-Hp, which provide a good justification for a deeper understanding of neural architectures.

----

## [1956] Cost-Sensitive Self-Training for Optimizing Non-Decomposable Metrics

**Authors**: *Harsh Rangwani, Shrinivas Ramasubramanian, Sho Takemori, Kato Takashi, Yuhei Umeda, Venkatesh Babu R.*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/acb94e709f02895fd98b5867f0b184f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/acb94e709f02895fd98b5867f0b184f3-Abstract-Conference.html)

**Abstract**:

Self-training based semi-supervised learning algorithms have enabled the learning of highly accurate deep neural networks, using only a fraction of labeled data. However, the majority of work on self-training has focused on the objective of improving accuracy whereas practical machine learning systems can have complex goals (e.g. maximizing the minimum of recall across classes, etc.) that are non-decomposable in nature. In this work, we introduce the Cost-Sensitive Self-Training (CSST) framework which generalizes the self-training-based methods for optimizing non-decomposable metrics. We prove that our framework can better optimize the desired non-decomposable metric utilizing unlabeled data, under similar data distribution assumptions made for the analysis of self-training.  Using the proposed CSST framework, we obtain practical self-training methods (for both vision and NLP tasks) for optimizing different non-decomposable metrics using deep neural networks.  Our results demonstrate that CSST achieves an improvement over the state-of-the-art in majority of the cases across datasets and objectives.

----

## [1957] Tensor Wheel Decomposition and Its Tensor Completion Application

**Authors**: *Zhong-Cheng Wu, Ting-Zhu Huang, Liang-Jian Deng, Hong-Xia Dou, Deyu Meng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/acbfe708197ff78ad04cc1beb1710979-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/acbfe708197ff78ad04cc1beb1710979-Abstract-Conference.html)

**Abstract**:

Recently, tensor network (TN) decompositions have gained prominence in computer vision and contributed promising results to high-order data recovery tasks. However, current TN models are rather being developed towards more intricate structures to pursue incremental improvements, which instead leads to a dramatic increase in rank numbers, thus encountering laborious hyper-parameter selection, especially for higher-order cases. In this paper, we propose a novel TN decomposition, dubbed tensor wheel (TW) decomposition, in which a high-order tensor is represented by a set of latent factors mapped into a specific wheel topology. Such decomposition is constructed starting from analyzing the graph structure, aiming to more accurately characterize the complex interactions inside objectives while maintaining a lower hyper-parameter scale, theoretically alleviating the above deficiencies. Furthermore, to investigate the potentiality of TW decomposition, we provide its one numerical application, i.e., tensor completion (TC), yet develop an efficient proximal alternating minimization-based solving algorithm with guaranteed convergence. Experimental results elaborate that the proposed method is significantly superior to other tensor decomposition-based state-of-the-art methods on synthetic and real-world data, implying the merits of TW decomposition. The code is available at: https://github.com/zhongchengwu/code_TWDec.

----

## [1958] BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs

**Authors**: *Kay Liu, Yingtong Dou, Yue Zhao, Xueying Ding, Xiyang Hu, Ruitong Zhang, Kaize Ding, Canyu Chen, Hao Peng, Kai Shu, Lichao Sun, Jundong Li, George H. Chen, Zhihao Jia, Philip S. Yu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Detecting which nodes in graphs are outliers is a relatively new machine learning task with numerous applications. Despite the proliferation of algorithms developed in recent years for this task, there has been no standard comprehensive setting for performance evaluation. Consequently, it has been difficult to understand which methods work well and when under a broad range of settings. To bridge this gap, we present—to the best of our knowledge—the first comprehensive benchmark for unsupervised outlier node detection on static attributed graphs called BOND, with the following highlights. (1) We benchmark the outlier detection performance of 14 methods ranging from classical matrix factorization to the latest graph neural networks. (2) Using nine real datasets, our benchmark assesses how the different detection methods respond to two major types of synthetic outliers and separately to “organic” (real non-synthetic) outliers. (3) Using an existing random graph generation technique, we produce a family of synthetically generated datasets of different graph sizes that enable us to compare the running time and memory usage of the different outlier detection algorithms. Based on our experimental results, we discuss the pros and cons of existing graph outlier detection algorithms, and we highlight opportunities for future research. Importantly, our code is freely available and meant to be easily extendable: https://github.com/pygod-team/pygod/tree/main/benchmark

----

## [1959] Movement Penalized Bayesian Optimization with Application to Wind Energy Systems

**Authors**: *Shyam Sundhar Ramesh, Pier Giuseppe Sessa, Andreas Krause, Ilija Bogunovic*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/acde98fb254b8021d194ccdb80a1241e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/acde98fb254b8021d194ccdb80a1241e-Abstract-Conference.html)

**Abstract**:

Contextual Bayesian optimization (CBO) is a powerful framework for sequential decision-making given side information, with important applications, e.g., in wind energy systems. In this setting, the learner receives context (e.g., weather conditions) at each round, and has to choose an action (e.g., turbine parameters). Standard algorithms assume no cost for switching their decisions at every round. However, in many practical applications, there is a cost associated with such changes, which should be minimized. We introduce the episodic CBO with movement costs problem and, based on the online learning approach for metrical task systems of Coester and Lee (2019), propose a novel randomized mirror descent algorithm that makes use of Gaussian Process confidence bounds. We compare its performance with the offline optimal sequence for each episode and provide rigorous regret guarantees. We further demonstrate our approach on the important real-world application of altitude optimization for Airborne Wind Energy Systems. In the presence of substantial movement costs, our algorithm consistently outperforms standard CBO algorithms.

----

## [1960] Associating Objects and Their Effects in Video through Coordination Games

**Authors**: *Erika Lu, Forrester Cole, Weidi Xie, Tali Dekel, Bill Freeman, Andrew Zisserman, Michael Rubinstein*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html)

**Abstract**:

We explore a feed-forward approach for decomposing a video into layers, where each layer contains an object of interest along with its associated shadows, reflections, and other visual effects. This problem is challenging since associated effects vary widely with the 3D geometry and lighting conditions in the scene, and ground-truth labels for visual effects are difficult (and in some cases impractical) to collect. We take a self-supervised approach and train a neural network to produce a foreground image and alpha matte from a rough object segmentation mask under a reconstruction and sparsity loss. Under reconstruction loss, the layer decomposition problem is underdetermined: many combinations of layers may reconstruct the input video.Inspired by the game theory concept of focal points---or \emph{Schelling points}---we pose the problem as a coordination game, where each player (network) predicts the effects for a single object without knowledge of the other players' choices. The players learn to converge on the ``natural'' layer decomposition in order to maximize the likelihood of their choices aligning with the other players'. We train the network to play this game with itself, and show how to design the rules of this game so that the focal point lies at the correct layer decomposition. We demonstrate feed-forward results on a challenging synthetic dataset, then show that pretraining on this dataset significantly reduces optimization time for real videos.

----

## [1961] Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training

**Authors**: *Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ad1d7a4df30a9c0c46b387815a774a84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ad1d7a4df30a9c0c46b387815a774a84-Abstract-Conference.html)

**Abstract**:

Masked Autoencoders (MAE) have shown great potentials in self-supervised pre-training for language and 2D image transformers. However, it still remains an open question on how to exploit masked autoencoding for learning 3D representations of irregular point clouds. In this paper, we propose Point-M2AE, a strong Multi-scale MAE pre-training framework for hierarchical self-supervised learning of 3D point clouds. Unlike the standard transformer in MAE, we modify the encoder and decoder into pyramid architectures to progressively model spatial geometries and capture both fine-grained and high-level semantics of 3D shapes. For the encoder that downsamples point tokens by stages, we design a multi-scale masking strategy to generate consistent visible regions across scales, and adopt a local spatial self-attention mechanism during fine-tuning to focus on neighboring patterns. By multi-scale token propagation, the lightweight decoder gradually upsamples point tokens with complementary skip connections from the encoder, which further promotes the reconstruction from a global-to-local perspective. Extensive experiments demonstrate the state-of-the-art performance of Point-M2AE for 3D representation learning. With a frozen encoder after pre-training, Point-M2AE achieves 92.9% accuracy for linear SVM on ModelNet40, even surpassing some fully trained methods. By fine-tuning on downstream tasks, Point-M2AE achieves 86.43% accuracy on ScanObjectNN, +3.36% to the second-best, and largely benefits the few-shot classification, part segmentation and 3D object detection with the hierarchical pre-training scheme. Code is available at https://github.com/ZrrSkywalker/Point-M2AE.

----

## [1962] Exploring Example Influence in Continual Learning

**Authors**: *Qing Sun, Fan Lyu, Fanhua Shang, Wei Feng, Liang Wan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html)

**Abstract**:

Continual Learning (CL) sequentially learns new tasks like human beings, with the goal to achieve better Stability (S, remembering past tasks) and Plasticity (P, adapting to new tasks). Due to the fact that past training data is not available, it is valuable to explore the influence difference on S and P among training examples, which may improve the learning pattern towards better SP. Inspired by Influence Function (IF), we first study example influence via adding perturbation to example weight and computing the influence derivation. To avoid the storage and calculation burden of Hessian inverse in neural networks, we propose a simple yet effective MetaSP algorithm to simulate the two key steps in the computation of IF and obtain the S- and P-aware example influence. Moreover, we propose to fuse two kinds of example influence by solving a dual-objective optimization problem, and obtain a fused influence towards SP Pareto optimality. The fused influence can be used to control the update of model and optimize the storage of rehearsal. Empirical results show that our algorithm significantly outperforms state-of-the-art methods on both task- and class-incremental benchmark CL datasets.

----

## [1963] Subspace clustering in high-dimensions: Phase transitions & Statistical-to-Computational gap

**Authors**: *Luca Pesce, Bruno Loureiro, Florent Krzakala, Lenka Zdeborová*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ad3d0ac42b4b5cc3b5f0ca10107d5c84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ad3d0ac42b4b5cc3b5f0ca10107d5c84-Abstract-Conference.html)

**Abstract**:

A simple model to study subspace clustering is the high-dimensional $k$-Gaussian mixture model where the cluster means are sparse vectors. Here we provide an exact asymptotic characterization of the statistically optimal reconstruction error in this model in the high-dimensional regime with extensive sparsity, i.e. when the fraction of non-zero components of the cluster means $\rho$, as well as the ratio $\alpha$ between the number of samples and the dimension are fixed, while the dimension diverges. We identify the information-theoretic threshold below which obtaining a positive correlation with the true cluster means is statistically impossible. Additionally, we investigate the performance of the approximate message passing (AMP) algorithm analyzed via its state evolution, which is conjectured to be optimal among polynomial algorithm for this task. We identify in particular the existence of a statistical-to-computational gap between the algorithm that requires a signal-to-noise ratio $\lambda_{\text{alg}} \ge k  / \sqrt{\alpha}$ to perform better than random, and the information theoretic threshold at $\lambda_{\text{it}} \approx \sqrt{-k \rho \log{\rho}}  / \sqrt{\alpha}$. Finally, we discuss the case of sub-extensive sparsity $\rho$ by comparing the performance of the AMP with other sparsity-enhancing algorithms, such as sparse-PCA and diagonal thresholding.

----

## [1964] Self-Supervised Fair Representation Learning without Demographics

**Authors**: *Junyi Chai, Xiaoqian Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ad991bbc381626a8e44dc5414aa136a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ad991bbc381626a8e44dc5414aa136a8-Abstract-Conference.html)

**Abstract**:

Fairness has become an important topic in machine learning. Generally, most literature on fairness assumes that the sensitive information, such as gender or race, is present in the training set, and uses this information to mitigate bias. However, due to practical concerns like privacy and regulation, applications of these methods are restricted. Also, although much of the literature studies supervised learning, in many real-world scenarios, we want to utilize the large unlabelled dataset to improve the model's accuracy. Can we improve fair classification without sensitive information and without labels? To tackle the problem, in this paper, we propose a novel reweighing-based contrastive learning method. The goal of our method is to learn a generally fair representation without observing sensitive attributes.Our method assigns weights to training samples per iteration based on their gradient directions relative to the validation samples such that the average top-k validation loss is minimized. Compared with past fairness methods without demographics, our method is built on fully unsupervised training data and requires only a small labelled validation set. We provide rigorous theoretical proof of the convergence of our model. Experimental results show that our proposed method achieves better or comparable performance than state-of-the-art methods on three datasets in terms of accuracy and several fairness metrics.

----

## [1965] CEDe: A collection of expert-curated datasets with atom-level entity annotations for Optical Chemical Structure Recognition

**Authors**: *Rodrigo Hormazabal, Changyoung Park, Soonyoung Lee, Sehui Han, Yeonsik Jo, Jaewan Lee, Ahra Jo, Seung Hwan Kim, Jaegul Choo, Moontae Lee, Honglak Lee*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ada36dfeb684a5c11f783fc170c294fe-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/ada36dfeb684a5c11f783fc170c294fe-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Optical Chemical Structure Recognition (OCSR) deals with the translation from chemical images to molecular structures, this being the main way chemical compounds are depicted in scientific documents. Traditionally, rule-based methods have followed a framework based on the detection of chemical entities, such as atoms and bonds, followed by a compound structure reconstruction step. Recently, neural architectures analog to image captioning have been explored to solve this task, yet they still show to be data inefficient, using millions of examples just to show performances comparable with traditional methods. Looking to motivate and benchmark new approaches based on atomic-level entities detection and graph reconstruction, we present CEDe, a unique collection of chemical entity bounding boxes manually curated by experts for scientific literature datasets. These annotations combine to more than 700,000 chemical entity bounding boxes with the necessary information for structure reconstruction. Also, a large synthetic dataset containing one million molecular images and annotations is released in order to explore transfer-learning techniques that could help these architectures perform better under low-data regimes. Benchmarks show that detection-reconstruction based models can achieve performances on par with or better than image captioning-like models, even with 100x fewer training examples.

----

## [1966] How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders

**Authors**: *Qi Zhang, Yifei Wang, Yisen Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/adb2075b6dd31cb18dfa727240d2887e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/adb2075b6dd31cb18dfa727240d2887e-Abstract-Conference.html)

**Abstract**:

Masked Autoencoders (MAE) based on a reconstruction task have risen to be a promising paradigm for self-supervised learning (SSL) and achieve state-of-the-art performance across different benchmark datasets. However, despite its impressive empirical success, there is still limited theoretical understanding of it. In this paper, we propose a theoretical understanding of how masking matters for MAE to learn meaningful features. We establish a close connection between MAE and contrastive learning, which shows that MAE implicit aligns the mask-induced positive pairs. Built upon this connection, we develop the first downstream guarantees for MAE methods, and analyze the effect of mask ratio. Besides, as a result of the implicit alignment, we also point out the dimensional collapse issue of MAE, and propose a Uniformity-enhanced MAE (U-MAE) loss that can effectively address this issue and bring significant improvements on real-world datasets, including CIFAR-10, ImageNet-100, and ImageNet-1K. Code is available at https://github.com/zhangq327/U-MAE.

----

## [1967] Improving Out-of-Distribution Generalization by Adversarial Training with Structured Priors

**Authors**: *Qixun Wang, Yifei Wang, Hong Zhu, Yisen Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html)

**Abstract**:

Deep models often fail to generalize well in test domains when the data distribution differs from that in the training domain. Among numerous approaches to address this Out-of-Distribution (OOD) generalization problem, there has been a growing surge of interest in exploiting Adversarial Training (AT) to improve OOD performance. Recent works have revealed that the robust model obtained by conducting sample-wise AT also retains transferability to biased test domains. In this paper, we empirically show that sample-wise AT has limited improvement on OOD performance. Specifically, we find that AT can only maintain performance at smaller scales of perturbation while Universal AT (UAT) is more robust to larger-scale perturbations. This provides us with clues that adversarial perturbations with universal (low dimensional) structures can enhance the robustness against large data distribution shifts that are common in OOD scenarios. Inspired by this, we propose two AT variants with low-rank structures to train OOD-robust models. Extensive experiments on DomainBed benchmark show that our proposed approaches outperform Empirical Risk Minimization (ERM) and sample-wise AT. Our code is available at https://github.com/NOVAglow646/NIPS22-MAT-and-LDAT-for-OOD.

----

## [1968] Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Biomolecular Structures and Interaction Networks

**Authors**: *Arian R. Jamasb, Ramón Viñas Torné, Eric Ma, Yuanqi Du, Charles Harris, Kexin Huang, Dominic Hall, Pietro Lió, Tom L. Blundell*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ade039c1db0391106a3375bd2feb310a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ade039c1db0391106a3375bd2feb310a-Abstract-Conference.html)

**Abstract**:

Geometric deep learning has broad applications in biology, a domain where relational structure in data is often intrinsic to modelling  the underlying phenomena. Currently, efforts in both geometric deep learning and, more broadly, deep learning applied to biomolecular tasks have been hampered by a scarcity of appropriate datasets accessible to domain specialists and machine learning researchers alike. To address this, we introduce Graphein as a turn-key tool for transforming raw data from widely-used bioinformatics databases into machine learning-ready datasets in a high-throughput and flexible manner. Graphein is a Python library for constructing graph and surface-mesh representations of biomolecular structures, such as proteins, nucleic acids and small molecules, and biological interaction networks for computational analysis and machine learning. Graphein provides utilities for data retrieval from widely-used bioinformatics databases for structural data, including the Protein Data Bank, the AlphaFold Structure Database, chemical data from ZINC and ChEMBL, and for biomolecular interaction networks from STRINGdb, BioGrid, TRRUST and RegNetwork. The library interfaces with popular geometric deep learning libraries: DGL, Jraph, PyTorch Geometric and PyTorch3D though remains framework agnostic as it is built on top of the PyData ecosystem to enable inter-operability with scientific computing tools and libraries.  Graphein is designed to be highly flexible, allowing the user to specify each step of the data preparation, scalable to facilitate working with large protein complexes and interaction graphs, and contains useful pre-processing tools for preparing experimental files. Graphein facilitates network-based, graph-theoretic and topological analyses of structural and interaction datasets in a high-throughput manner. We envision that Graphein will facilitate developments in computational biology, graph representation learning and drug discovery. Availability and implementation: Graphein is written in Python. Source code, example usage and tutorials, datasets, and documentation are made freely available under the MIT License at the following URL: https://anonymous.4open.science/r/graphein-3472/README.md

----

## [1969] ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers

**Authors**: *Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html)

**Abstract**:

How to efficiently serve ever-larger trained natural language models in practice has become exceptionally challenging even for powerful cloud servers due to their prohibitive memory/computation requirements.In this work, we present an efficient and affordable post-training quantization approach to compress large Transformer-based models, termed as \OURS. \OURS is an end-to-end quantization and inference pipeline with three main components: (1) a fine-grained hardware-friendly quantization scheme for both weight and activations; (2) a novel affordable layer-by-layer knowledge distillation algorithm (\lwd) even without the original training data access;(3) a highly-optimized quantization system backend support to remove the quantization/dequantization overhead.As such, we are able to show that:(1) \OURS can reduce the precision for weight and activations to INT8 in a cost-free way for both \bert and \gpt-style models with minimal accuracy impact, which leads to up to 5.19x/4.16x speedup on \bert/\gpt-style models compared to FP16 inference, separately;(2) \OURS plus \lwd can affordably quantize the weights in the fully-connected module to INT4 along with INT8 weights in the attention module and INT8 activations, resulting in 3x memory footprint reduction compared to the FP16 model;(3) \OURS can be directly applied to two of the largest open-sourced language models, including \gptneox, for which our INT8 model achieves similar accuracy as the FP16 model but achieves 5.2x better efficiency.Our code is open-sourced at~\cite{code_compression}.

----

## [1970] Simplified Graph Convolution with Heterophily

**Authors**: *Sudhanshu Chanpuriya, Cameron Musco*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae07d152c51ea2ddae65aa7192eb5ff7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae07d152c51ea2ddae65aa7192eb5ff7-Abstract-Conference.html)

**Abstract**:

Recent work has shown that a simple, fast method called Simple Graph Convolution (SGC) (Wu et al., 2019), which eschews deep learning, is competitive with deep methods like graph convolutional networks (GCNs) (Kipf & Welling, 2017) in common graph machine learning benchmarks. The use of graph data in SGC implicitly assumes the common but not universal graph characteristic of homophily, wherein nodes link to nodes which are similar. Here we confirm that SGC is indeed ineffective for heterophilous (i.e., non-homophilous) graphs via experiments on synthetic and real-world datasets. We propose Adaptive Simple Graph Convolution (ASGC), which we show can adapt to both homophilous and heterophilous graph structure. Like SGC, ASGC is not a deep model, and hence is fast, scalable, and interpretable; further, we can prove performance guarantees on natural synthetic data models. Empirically, ASGC is often competitive with recent deep models at node classification on a benchmark of real-world datasets. The SGC paper questioned whether the complexity of graph neural networks is warranted for common graph problems involving homophilous networks; our results similarly suggest that, while deep learning often achieves the highest performance, heterophilous structure alone does not necessitate these more involved methods.

----

## [1971] Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse

**Authors**: *Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, Aurélien Lucchi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html)

**Abstract**:

Transformers have achieved remarkable success in several domains, ranging from natural language processing to computer vision. Nevertheless, it has been recently shown that stacking self-attention layers — the distinctive architectural component of Transformers — can result in rank collapse of the tokens’ representations at initialization. The question of if and how rank collapse affects training is still largely unanswered, and its investigation is necessary for a more comprehensive understanding of this architecture. In this work, we shed new light on the causes and the effects of this phenomenon. First, we show that rank collapse of the tokens’ representations hinders training by causing the gradients of the queries and keys to vanish at initialization. Furthermore, we provide a thorough description of the origin of rank collapse and discuss how to prevent it via an appropriate depth-dependent scaling of the residual branches. Finally, our analysis unveils that specific architectural hyperparameters affect the gradients of queries, keys and values differently, leading to disproportionate gradient norms. This suggests an explanation for the widespread use of adaptive methods for Transformers' optimization.

----

## [1972] Maximum a posteriori natural scene reconstruction from retinal ganglion cells with deep denoiser priors

**Authors**: *Eric Wu, Nora Brackbill, Alexander Sher, Alan M. Litke, Eero P. Simoncelli, E. J. Chichilnisky*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae447e9dbfdd1189966e894b85bea062-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae447e9dbfdd1189966e894b85bea062-Abstract-Conference.html)

**Abstract**:

Visual information arriving at the retina is transmitted to the brain by signals in the optic nerve, and the brain must rely solely on these signals to make inferences about the visual world. Previous work has probed the content of these signals by directly reconstructing images from retinal activity using linear regression or nonlinear regression with neural networks. Maximum a posteriori (MAP) reconstruction using retinal encoding models and separately-trained natural image priors offers a more general and principled approach. We develop a novel method for approximate MAP reconstruction that combines a generalized linear model for retinal responses to light, including their dependence on spike history and spikes of neighboring cells, with the image prior implicitly embedded in a deep convolutional neural network trained for image denoising. We use this method to reconstruct natural images from ex vivo simultaneously-recorded spikes of hundreds of retinal ganglion cells uniformly sampling a region of the retina. The method produces reconstructions that match or exceed the state-of-the-art in perceptual similarity and exhibit additional fine detail, while using substantially fewer model parameters than previous approaches. The use of more rudimentary encoding models (a linear-nonlinear-Poisson cascade) or image priors (a 1/f spectral model) significantly reduces reconstruction performance, indicating the essential role of both components in achieving high-quality reconstructed images from the retinal signal.

----

## [1973] Imbalance Trouble: Revisiting Neural-Collapse Geometry

**Authors**: *Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, Tina Behnia*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html)

**Abstract**:

Neural Collapse refers to the remarkable structural properties characterizing the geometry of class embeddings and classifier weights, found by deep nets when trained beyond zero training error. However, this characterization only holds for balanced data. Here we thus ask whether it can be made invariant to class imbalances. Towards this end, we adopt the unconstrained feature model (UFM), a recent theoretical model for studying neural collapse, and introduce $\text{\emph{Simplex-Encoded-Labels Interpolation}}$ (SELI) as an invariant characterization of the neural collapse phenomenon. Specifically, we prove for the UFM with cross-entropy loss and vanishing regularization that, irrespective of class imbalances, the embeddings and classifiers always interpolate a simplex-encoded label matrix and that their individual geometries are determined by the SVD factors of this same label matrix. We then present extensive experiments on synthetic and real datasets that confirm convergence to the SELI geometry. However, we caution that convergence worsens with increasing imbalances. We theoretically support this finding by showing that unlike the balanced case, when minorities are present, ridge-regularization plays a critical role in tweaking the geometry. This defines new questions and motivates further investigations into the impact of class imbalances on the rates at which first-order methods converge to their asymptotically preferred solutions.

----

## [1974] ProtoX: Explaining a Reinforcement Learning Agent via Prototyping

**Authors**: *Ronilo J. Ragodos, Tong Wang, Qihang Lin, Xun Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae5bf4f35236240c9460e761c60fa53d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae5bf4f35236240c9460e761c60fa53d-Abstract-Conference.html)

**Abstract**:

While deep reinforcement learning has proven to be successful in solving control tasks, the ``black-box'' nature of an agent has received increasing concerns. We propose a prototype-based post-hoc \emph{policy explainer}, ProtoX, that explains a black-box agent by prototyping the agent's behaviors into scenarios, each represented by a prototypical state. When learning prototypes, ProtoX considers both visual similarity and scenario similarity. The latter is unique to the reinforcement learning context since it explains why the same action is taken in visually different states. To teach ProtoX about visual similarity, we pre-train an encoder using contrastive learning via self-supervised learning to recognize states as similar if they occur close together in time and receive the same action from the black-box agent. We then add an isometry layer to allow ProtoX to adapt scenario similarity to the downstream task. ProtoX is trained via imitation learning using behavior cloning, and thus requires no access to the environment or agent. In addition to explanation fidelity, we  design different prototype shaping terms in the objective function to encourage better interpretability. We conduct various experiments to test ProtoX. Results show that ProtoX achieved high fidelity to the original black-box agent while providing meaningful and understandable explanations.

----

## [1975] NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation

**Authors**: *Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, Sung-Ju Lee*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae6c7dbd9429b3a75c41b5fb47e57c9e-Abstract-Conference.html)

**Abstract**:

Test-time adaptation (TTA) is an emerging paradigm that addresses distributional shifts between training and testing phases without additional data acquisition or labeling cost; only unlabeled test data streams are used for continual model adaptation. Previous TTA schemes assume that the test samples are independent and identically distributed (i.i.d.), even though they are often temporally correlated (non-i.i.d.) in application scenarios, e.g., autonomous driving. We discover that most existing TTA methods fail dramatically under such scenarios. Motivated by this, we present a new test-time adaptation scheme that is robust against non-i.i.d. test data streams. Our novelty is mainly two-fold: (a) Instance-Aware Batch Normalization (IABN) that corrects normalization for out-of-distribution samples, and (b) Prediction-balanced Reservoir Sampling (PBRS) that simulates i.i.d. data stream from non-i.i.d. stream in a class-balanced manner. Our evaluation with various datasets, including real-world non-i.i.d. streams, demonstrates that the proposed robust TTA not only outperforms state-of-the-art TTA algorithms in the non-i.i.d. setting, but also achieves comparable performance to those algorithms under the i.i.d. assumption. Code is available at https://github.com/TaesikGong/NOTE.

----

## [1976] Margin-Based Few-Shot Class-Incremental Learning with Class-Level Overfitting Mitigation

**Authors**: *Yixiong Zou, Shanghang Zhang, Yuhua Li, Ruixuan Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae817e85f71ef86d5c9566598e185b89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae817e85f71ef86d5c9566598e185b89-Abstract-Conference.html)

**Abstract**:

Few-shot class-incremental learning (FSCIL) is designed to incrementally recognize novel classes with only few training samples after the (pre-)training on base classes with sufficient samples, which focuses on both base-class performance and novel-class generalization. A well known modification to the base-class training is to apply a margin to the base-class classification. However, a dilemma exists that we can hardly achieve both good base-class performance and novel-class generalization simultaneously by applying the margin during the base-class training, which is still under explored. In this paper, we study the cause of such dilemma for FSCIL. We first interpret this dilemma as a class-level overfitting (CO) problem from the aspect of pattern learning, and then find its cause lies in the easily-satisfied constraint of learning margin-based patterns. Based on the analysis, we propose a novel margin-based FSCIL method to mitigate the CO problem by providing the pattern learning process with extra constraint from the margin-based patterns themselves. Extensive experiments on CIFAR100, Caltech-USCD Birds-200-2011 (CUB200), and miniImageNet demonstrate that the proposed method effectively mitigates the CO problem and achieves state-of-the-art performance.

----

## [1977] First Hitting Diffusion Models for Generating Manifold, Graph and Categorical Data

**Authors**: *Mao Ye, Lemeng Wu, Qiang Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ae87d80f5a0f3ee5c5643448f9599d1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ae87d80f5a0f3ee5c5643448f9599d1b-Abstract-Conference.html)

**Abstract**:

We propose a family of First Hitting Diffusion Models (FHDM), deep generative models that generate data with a diffusion process that terminates at a random first hitting time. This yields an extension of the standard fixed-time diffusion models that terminate at a pre-specified deterministic time. Although standard diffusion models are designed for continuous unconstrained data, FHDM is naturally designed to learn distributions on continuous as well as a range of discrete and structure domains. Moreover, FHDM  enables instance-dependent terminate time and accelerates the diffusion process to sample higher quality data with fewer diffusion steps. Technically, we train FHDM by maximum likelihood estimation on diffusion trajectories augmented from observed data with conditional first hitting processes (i.e., bridge) derived based on Doob's $h$-transform, deviating from the commonly used time-reversal mechanism. We apply FHDM to generate data in various domains such as point cloud (general continuous distribution),  climate and geographical events on earth (continuous distribution on the sphere),  unweighted graphs (distribution of binary matrices), and segmentation maps of 2D images (high-dimensional categorical distribution). We observe considerable improvement compared with the state-of-the-art approaches in both quality and speed.

----

## [1978] Forecasting Future World Events With Neural Networks

**Authors**: *Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, Dan Hendrycks*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aec870a6772336c15dac992c16f2e7c9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/aec870a6772336c15dac992c16f2e7c9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Forecasting future world events is a challenging but valuable task. Forecasts of climate, geopolitical conflict, pandemics and economic indicators help shape policy and decision making. In these domains, the judgment of expert humans contributes to the best forecasts. Given advances in language modeling, can these forecasts be automated? To this end, we introduce Autocast, a dataset containing thousands of forecasting questions and an accompanying news corpus. Questions are taken from forecasting tournaments, ensuring high quality, real-world importance, and diversity. The news corpus is organized by date, allowing us to precisely simulate the conditions under which humans made past forecasts (avoiding leakage from the future). Motivated by the difficulty of forecasting numbers across orders of magnitude (e.g. global cases of COVID-19 in 2022), we also curate IntervalQA, a dataset of numerical questions and metrics for calibration. We test language models on our forecasting task and find that performance is far below a human expert baseline. However, performance improves with increased model size and incorporation of relevant information from the news corpus. In sum, Autocast poses a novel challenge for large language models and improved performance could bring large practical benefits.

----

## [1979] Human-Robotic Prosthesis as Collaborating Agents for Symmetrical Walking

**Authors**: *Ruofan Wu, Junmin Zhong, Brent Wallace, Xiang Gao, He Huang, Jennie Si*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/aed42bb2e45857928418e4fe23d8cbcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/aed42bb2e45857928418e4fe23d8cbcb-Abstract-Conference.html)

**Abstract**:

This is the first attempt at considering human influence in the reinforcement learning control of a robotic lower limb prosthesis toward symmetrical walking in real world situations. We propose a collaborative multi-agent reinforcement learning (cMARL) solution framework for this highly complex and challenging human-prosthesis collaboration (HPC) problem. The design of an automatic controller of the robot within the HPC context is based on accessible physical features or measurements that are known to affect walking performance. Comparisons are made with the current state-of-the-art robot control designs, which are single-agent based, as well as existing MARL solution approaches tailored to the problem, including multi-agent deep deterministic policy gradient (MADDPG) and  counterfactual multi-agent policy gradient (COMA).  Results show that, when compared to these approaches, treating the human and robot as coupled agents and using estimated human adaption in robot control design can achieve lower stage cost, peak error, and symmetry value to ensure better human walking performance. Additionally, our approach accelerates learning of walking tasks and increases learning success rate. The proposed framework can potentially be further developed to examine how human and robotic lower limb prosthesis interact, an area that little is known about. Advancing cMARL toward real world applications such as HPC for normative walking sets a good example of how AI can positively impact on peopleâ€™s lives.

----

## [1980] Support Recovery in Sparse PCA with Incomplete Data

**Authors**: *Hanbyul Lee, Qifan Song, Jean Honorio*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af050c48a0d8162e46b3d1952e7e374f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af050c48a0d8162e46b3d1952e7e374f-Abstract-Conference.html)

**Abstract**:

We study a practical algorithm for sparse principal component analysis (PCA) of incomplete and noisy data.Our algorithm is based on the semidefinite program (SDP) relaxation of the non-convex $l_1$-regularized PCA problem.We provide theoretical and experimental evidence that SDP enables us to exactly recover the true support of the sparse leading eigenvector of the unknown true matrix, despite only observing an incomplete (missing uniformly at random) and noisy version of it.We derive sufficient conditions for exact recovery, which involve matrix incoherence, the spectral gap between the largest and second-largest eigenvalues, the observation probability and the noise variance.We validate our theoretical results with incomplete synthetic data, and show encouraging and meaningful results on a gene expression dataset.

----

## [1981] Exponentially Improving the Complexity of Simulating the Weisfeiler-Lehman Test with Graph Neural Networks

**Authors**: *Anders Aamand, Justin Y. Chen, Piotr Indyk, Shyam Narayanan, Ronitt Rubinfeld, Nicholas Schiefer, Sandeep Silwal, Tal Wagner*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af0ad514b9cda46bd49e14ee11e2672f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af0ad514b9cda46bd49e14ee11e2672f-Abstract-Conference.html)

**Abstract**:

Recent work shows that the expressive power of Graph Neural Networks (GNNs) in distinguishing non-isomorphic graphs is exactly the same as that of the Weisfeiler-Lehman (WL) graph test. In particular, they show that the WL test can be simulated by GNNs. However, those simulations involve neural networks for the “combine” function of size polynomial or even exponential in the number of graph nodes $n$, as well as feature vectors of length linear in $n$. We present an improved simulation of the WL test on GNNs with {\em exponentially} lower complexity. In particular,  the neural network implementing the  combine function  in each node has only $\mathrm{polylog}(n)$ parameters, and the feature vectors exchanged by the nodes of GNN consists of only $O(\log n)$ bits. We also give logarithmic lower bounds for the feature vector length and the size of the neural networks, showing the (near)-optimality of our construction.

----

## [1982] Zero-shot Transfer Learning within a Heterogeneous Graph via Knowledge Transfer Networks

**Authors**: *Minji Yoon, John Palowitch, Dustin Zelle, Ziniu Hu, Ruslan Salakhutdinov, Bryan Perozzi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html)

**Abstract**:

Data continuously emitted from industrial ecosystems such as social or e-commerce platforms are commonly represented as heterogeneous graphs (HG) composed of multiple node/edge types. State-of-the-art graph learning methods for HGs known as heterogeneous graph neural networks (HGNNs) are applied to learn deep context-informed node representations. However, many HG datasets from industrial applications suffer from label imbalance between node types. As there is no direct way to learn using labels rooted at different node types, HGNNs have been applied to only a few node types with abundant labels. We propose a zero-shot transfer learning module for HGNNs called a Knowledge Transfer Network (KTN) that transfers knowledge from label-abundant node types to zero-labeled node types through rich relational information given in the HG. KTN is derived from the theoretical relationship, which we introduce in this work, between distinct feature extractors for each node type given in an HGNN model. KTN improves the performance of 6 different types of HGNN models by up to 960% for inference on zero-labeled node types and outperforms state-of-the-art transfer learning baselines by up to 73% across 18 different transfer learning tasks on HGs.

----

## [1983] Nonlinear Sufficient Dimension Reduction with a Stochastic Neural Network

**Authors**: *Siqi Liang, Yan Sun, Faming Liang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af5509c72a244497c999ac39ba068ff4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af5509c72a244497c999ac39ba068ff4-Abstract-Conference.html)

**Abstract**:

Sufficient dimension reduction is a powerful tool to extract core information hidden in the high-dimensional data and has potentially many important applications in machine learning tasks. However, the existing nonlinear sufficient dimension reduction  methods often lack the scalability necessary for dealing with large-scale data.  We propose a new type of stochastic neural network under a rigorous probabilistic framework and show that it can be used for sufficient dimension reduction for large-scale data. The proposed stochastic neural network is trained using an adaptive stochastic gradient Markov chain Monte Carlo algorithm, whose convergence is rigorously studied in the paper as well. Through extensive experiments on real-world classification and regression problems, we show that the proposed method compares favorably with the existing  state-of-the-art sufficient dimension reduction methods and is computationally more efficient for large-scale data.

----

## [1984] Autoregressive Perturbations for Data Poisoning

**Authors**: *Pedro Sandoval Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David Jacobs*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af66ac99716a64476c07ae8b089d59f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af66ac99716a64476c07ae8b089d59f8-Abstract-Conference.html)

**Abstract**:

The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data ``unlearnable'' by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.

----

## [1985] NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification

**Authors**: *Qitian Wu, Wentao Zhao, Zenan Li, David P. Wipf, Junchi Yan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af790b7ae573771689438bbcfc5933fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af790b7ae573771689438bbcfc5933fe-Abstract-Conference.html)

**Abstract**:

Graph neural networks have been extensively studied for learning with inter-connected data. Despite this, recent evidence has revealed GNNs' deficiencies related to over-squashing, heterophily, handling long-range dependencies, edge incompleteness and particularly, the absence of graphs altogether. While a plausible solution is to learn new adaptive topology for message passing, issues concerning quadratic complexity hinder simultaneous guarantees for scalability and precision in large networks. In this paper, we introduce a novel all-pair message passing scheme for efficiently propagating node signals between arbitrary nodes, as an important building block for a new class of Transformer networks for node classification on large graphs, dubbed as NodeFormer. Specifically, the efficient computation is enabled by a kernerlized Gumbel-Softmax operator that reduces the algorithmic complexity to linearity w.r.t. node numbers for learning latent graph structures from large, potentially fully-connected graphs in a differentiable manner. We also provide accompanying theory as justification for our design. Extensive experiments demonstrate the promising efficacy of the method in various tasks including node classification on graphs (with up to 2M nodes) and graph-enhanced applications (e.g., image classification) where input graphs are missing. The codes are available at https://github.com/qitianwu/NodeFormer.

----

## [1986] Is Integer Arithmetic Enough for Deep Learning Training?

**Authors**: *Alireza Ghaffari, Marzieh S. Tahaei, Mohammadreza Tayaranian, Masoud Asgharian, Vahid Partovi Nia*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af835bd1b5b689c3f9d075ae5a15bf3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/af835bd1b5b689c3f9d075ae5a15bf3e-Abstract-Conference.html)

**Abstract**:

The ever-increasing computational complexity of deep learning models makes their training and deployment difficult on various cloud and edge platforms. Replacing floating-point arithmetic with low-bit integer arithmetic is a promising approach to save energy, memory footprint, and latency of deep learning models. As such, quantization has attracted the attention of researchers in recent years. However, using integer numbers to form a fully functional integer training pipeline including forward pass, back-propagation, and stochastic gradient descent is not studied in detail. Our empirical and mathematical results reveal that integer arithmetic seems to be enough to train deep learning models. Unlike recent proposals, instead of quantization, we directly switch the number representation of computations. Our novel training method forms a fully integer training pipeline that does not change the trajectory of the loss and accuracy compared to floating-point, nor does it need any special hyper-parameter tuning, distribution adjustment, or gradient clipping. Our experimental results show that our proposed method is effective in a wide variety of tasks such as classification (including vision transformers), object detection, and semantic segmentation.

----

## [1987] mRI: Multi-modal 3D Human Pose Estimation Dataset using mmWave, RGB-D, and Inertial Sensors

**Authors**: *Sizhe An, Yin Li, Ümit Y. Ogras*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/af9c9c6d2da701da5a0acf91ec217815-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/af9c9c6d2da701da5a0acf91ec217815-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The ability to estimate 3D human body pose and movement, also known as human pose estimation (HPE), enables many applications for home-based health monitoring, such as remote rehabilitation training. Several possible solutions have emerged using sensors ranging from RGB cameras, depth sensors, millimeter-Wave (mmWave) radars, and wearable inertial sensors. Despite previous efforts on datasets and benchmarks for HPE, few dataset exploits multiple modalities and focuses on home-based health monitoring. To bridge the gap, we present mRI, a multi-modal 3D human pose estimation dataset with mmWave, RGB-D, and Inertial Sensors. Our dataset consists of over 160k synchronized frames from 20 subjects performing rehabilitation exercises and supports the benchmarks of HPE and action detection. We perform extensive experiments using our dataset and delineate the strength of each modality. We hope that the release of mRI can catalyze the research in pose estimation, multi-modal learning, and action understanding, and more importantly facilitate the applications of home-based health monitoring.

----

## [1988] Effectiveness of Vision Transformer for Fast and Accurate Single-Stage Pedestrian Detection

**Authors**: *Jing Yuan, Panagiotis Barmpoutis, Tania Stathaki*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/afb8caec018d3c8f6ef8b81fa52386fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/afb8caec018d3c8f6ef8b81fa52386fe-Abstract-Conference.html)

**Abstract**:

Vision transformers have demonstrated remarkable performance on a variety of computer vision tasks. In this paper, we illustrate the effectiveness of the deformable vision transformer for single-stage pedestrian detection and propose a spatial and multi-scale feature enhancement module, which aims to achieve the optimal balance between speed and accuracy. Performance improvement with vision transformers on various commonly used single-stage structures is demonstrated. The design of the proposed architecture is investigated in depth. Comprehensive comparisons with state-of-the-art single- and two-stage detectors on different pedestrian datasets are performed. The proposed detector achieves leading performance on Caltech and Citypersons datasets among single- and two-stage methods using fewer parameters than the baseline. The log-average miss rates for Reasonable and Heavy are decreased to 2.6% and 28.0% on the Caltech test set, and 10.9% and 38.6% on the Citypersons validation set, respectively. The proposed method outperforms SOTA two-stage detectors in the Heavy subset on the Citypersons validation set with considerably faster inference speed.

----

## [1989] ESCADA: Efficient Safety and Context Aware Dose Allocation for Precision Medicine

**Authors**: *Ilker Demirel, Ahmet Alparslan Celik, Cem Tekin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/afddff15817993412489a7df483da7d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/afddff15817993412489a7df483da7d9-Abstract-Conference.html)

**Abstract**:

Finding an optimal individualized treatment regimen is considered one of the most challenging precision medicine problems. Various patient characteristics influence the response to the treatment, and hence, there is no one-size-fits-all regimen. Moreover, the administration of an unsafe dose during the treatment can have adverse effects on health. Therefore, a treatment model must ensure patient \emph{safety} while \emph{efficiently} optimizing the course of therapy. We study a prevalent medical problem where the treatment aims to keep a physiological variable in a safe range and preferably close to a target level, which we refer to as \emph{leveling}. Such a task may be relevant in numerous other domains as well. We propose ESCADA, a novel and generic multi-armed bandit (MAB) algorithm tailored for the leveling task, to make safe, personalized, and context-aware dose recommendations. We derive high probability upper bounds on its cumulative regret and safety guarantees. Following ESCADA's design, we also describe its Thompson sampling-based counterpart. We discuss why the straightforward adaptations of the classical MAB algorithms such as GP-UCB may not be a good fit for the leveling task. Finally, we make \emph{in silico} experiments on the bolus-insulin dose allocation problem in type-1 diabetes mellitus disease and compare our algorithms against the famous GP-UCB algorithm, the rule-based dose calculators, and a clinician.

----

## [1990] Grow and Merge: A Unified Framework for Continuous Categories Discovery

**Authors**: *Xinwei Zhang, Jianwen Jiang, Yutong Feng, Zhi-Fan Wu, Xibin Zhao, Hai Wan, Mingqian Tang, Rong Jin, Yue Gao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/afe37ac3ce109cd33a23a6b3ed0cfc21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/afe37ac3ce109cd33a23a6b3ed0cfc21-Abstract-Conference.html)

**Abstract**:

Although a number of studies are devoted to novel category discovery, most of them assume a static setting where both labeled and unlabeled data are given at once for finding new categories. In this work, we focus on the application scenarios where unlabeled data are continuously fed into the category discovery system. We refer to it as the {\bf Continuous Category Discovery} ({\bf CCD}) problem, which is significantly more challenging than the static setting. A common challenge faced by novel category discovery is that different sets of features are needed for classification and category discovery: class discriminative features are preferred for classification, while rich and diverse features are more suitable for new category mining. This challenge becomes more severe for dynamic setting as the system is asked to deliver good performance for known classes over time, and at the same time continuously discover new classes from unlabeled data. To address this challenge, we develop a framework of {\bf Grow and Merge} ({\bf GM}) that works by alternating between a growing phase and a merge phase: in the growing phase, it increases the diversity of features through a continuous self-supervised learning for effective category mining, and in the merging phase, it merges the grown model with a static one to ensure satisfying performance for known classes. Our extensive studies verify that the proposed GM framework is significantly more effective than the state-of-the-art approaches for continuous category discovery.

----

## [1991] Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and A New Dataset

**Authors**: *Yanjie Ze, Xiaolong Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html)

**Abstract**:

6D object pose estimation is one of the fundamental problems in computer vision and robotics research. While a lot of recent efforts have been made on generalizing pose estimation to novel object instances within the same category, namely category-level 6D pose estimation, it is still restricted in constrained environments given the limited number of annotated data. In this paper, we collect Wild6D, a new unlabeled RGBD object video dataset with diverse instances and backgrounds. We utilize this data to generalize category-level 6D object pose estimation in the wild with semi-supervised learning. We propose a new model, called Rendering for Pose estimation network RePoNet), that is jointly trained using the free ground-truths with the synthetic data, and a silhouette matching objective function on the real-world data. Without using any 3D annotations on real data, our method outperforms state-of-the-art methods on the previous dataset and our Wild6D test set (with manual annotations for evaluation) by a large margin.  Project page with Wild6D data:  \url{https://oasisyang.github.io/semi-pose/}.

----

## [1992] Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery

**Authors**: *Utkarsh Mall, Bharath Hariharan, Kavita Bala*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Satellite imagery is increasingly available, high resolution, and temporally detailed.  Changes in spatio-temporal datasets such as satellite images are particularly interesting as they reveal the many events and forces that shape our world.  However, finding such interesting and meaningful change events from the vast data is challenging.  In this paper, we present new datasets for such change events that include semantically meaningful events like road construction.  Instead of manually annotating the very large corpus of satellite images, we introduce a novel unsupervised approach that takes a large spatio-temporal dataset from satellite images and finds interesting change events.  To evaluate the meaningfulness on these datasets we create 2 benchmarks namely CaiRoad and CalFire which capture the events of road construction and forest fires.  These new benchmarks can be used to evaluate semantic retrieval/classification performance.  We explore these benchmarks qualitatively and quantitatively by using several methods and show that these new datasets are indeed challenging for many existing methods.

----

## [1993] Improved Algorithms for Neural Active Learning

**Authors**: *Yikun Ban, Yuheng Zhang, Hanghang Tong, Arindam Banerjee, Jingrui He*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html)

**Abstract**:

We improve the theoretical and empirical performance of neural-network(NN)-based active learning algorithms for the non-parametric streaming setting. In particular, we introduce two regret metrics by minimizing the population loss that are more suitable in active learning than the one used in state-of-the-art (SOTA) related work.  Then, the proposed algorithm leverages the powerful representation of NNs for both exploitation and exploration, has the query decision-maker tailored for $k$-class classification problems with the performance guarantee, utilizes the full feedback, and updates parameters in a more practical and efficient manner. These careful designs lead to an instance-dependent regret upper bound, roughly improving by a multiplicative factor $O(\log T)$ and removing the curse of input dimensionality. Furthermore, we show that the algorithm can achieve the same performance as the Bayes-optimal classifier in the long run under the hard-margin setting in classification problems. In the end, we use extensive experiments to evaluate the proposed algorithm and SOTA baselines, to show the improved empirical performance.

----

## [1994] Robust Calibration with Multi-domain Temperature Scaling

**Authors**: *Yaodong Yu, Stephen Bates, Yi Ma, Michael I. Jordan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b054fadf1ccd80b37d465f6082629934-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b054fadf1ccd80b37d465f6082629934-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification is essential for the reliable deployment of machine learning models to high-stakes application domains. Uncertainty quantification is all the more challenging when training distribution and test distribution are different, even if the distribution shifts are mild. Despite the ubiquity of distribution shifts in real-world applications, existing uncertainty quantification approaches mainly study the in-distribution setting where the train and test distributions are the same. In this paper, we develop a systematic calibration model to handle distribution shifts by leveraging data from multiple domains. Our proposed method---multi-domain temperature scaling---uses the heterogeneity in the domains to improve calibration robustness under distribution shift. Through experiments on three benchmark data sets, we find our proposed method outperforms existing methods as measured on both in-distribution and out-of-distribution test sets.

----

## [1995] Independence Testing-Based Approach to Causal Discovery under Measurement Error and Linear Non-Gaussian Models

**Authors**: *Haoyue Dai, Peter Spirtes, Kun Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b05bffeb1ef937677ef0e32f027b4c80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b05bffeb1ef937677ef0e32f027b4c80-Abstract-Conference.html)

**Abstract**:

Causal discovery aims to recover causal structures generating the observational data. Despite its success in certain problems, in many real-world scenarios the observed variables are not the target variables of interest, but the imperfect measures of the target variables. Causal discovery under measurement error aims to recover the causal graph among unobserved target variables from observations made with measurement error. We consider a specific formulation of the problem, where the unobserved target variables follow a linear non-Gaussian acyclic model, and the measurement process follows the random measurement error model. Existing methods on this formulation rely on non-scalable over-complete independent component analysis (OICA). In this work, we propose the Transformed Independent Noise (TIN) condition, which checks for independence between a specific linear transformation of some measured variables and certain other measured variables. By leveraging the non-Gaussianity and higher-order statistics of data, TIN is informative about the graph structure among the unobserved target variables. By utilizing TIN, the ordered group decomposition of the causal model is identifiable. In other words, we could achieve what once required OICA to achieve by only conducting independence tests. Experimental results on both synthetic and real-world data demonstrate the effectiveness and reliability of our method.

----

## [1996] CUP: Critic-Guided Policy Reuse

**Authors**: *Jin Zhang, Siyuan Li, Chongjie Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b09df3a10e26204136540ca59bc5a646-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b09df3a10e26204136540ca59bc5a646-Abstract-Conference.html)

**Abstract**:

The ability to reuse previous policies is an important aspect of human intelligence. To achieve efficient policy reuse, a Deep Reinforcement Learning (DRL) agent needs to decide when to reuse and which source policies to reuse. Previous methods solve this problem by introducing extra components to the underlying algorithm, such as hierarchical high-level policies over source policies, or estimations of source policies' value functions on the target task. However, training these components induces either optimization non-stationarity or heavy sampling cost, significantly impairing the effectiveness of transfer. To tackle this problem, we propose a novel policy reuse algorithm called Critic-gUided Policy reuse (CUP), which avoids training any extra components and efficiently reuses source policies. CUP utilizes the critic, a common component in actor-critic methods, to evaluate and choose source policies. At each state, CUP chooses the source policy that has the largest one-step improvement over the current target policy, and forms a guidance policy. The guidance policy is theoretically guaranteed to be a monotonic improvement over the current target policy. Then the target policy is regularized to imitate the guidance policy to perform efficient policy search. Empirical results demonstrate that CUP achieves efficient transfer and significantly outperforms baseline algorithms.

----

## [1997] Local Identifiability of Deep ReLU Neural Networks: the Theory

**Authors**: *Joachim Bona-Pellissier, François Malgouyres, François Bachoc*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b0ae046e198a5e43141519868a959c74-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b0ae046e198a5e43141519868a959c74-Abstract-Conference.html)

**Abstract**:

Is a sample rich enough to determine, at least locally, the parameters of a neural network? To answer this question, we introduce a new local parameterization of a given deep ReLU neural network by fixing the values of some of its weights. This allows us to define local lifting operators whose inverses are charts of a smooth manifold of a high dimensional space. The function implemented by the deep ReLU neural network composes the local lifting with a linear operator which depends on the sample. We derive from this convenient representation a geometrical necessary and sufficient condition of local identifiability. Looking at tangent spaces, the geometrical condition provides: 1/ a sharp and testable necessary condition of identifiability and 2/ a sharp and testable sufficient condition of local identifiability. The validity of the conditions can be tested numerically using backpropagation and matrix rank computations.

----

## [1998] DOMINO: Decomposed Mutual Information Optimization for Generalized Context in Meta-Reinforcement Learning

**Authors**: *Yao Mu, Yuzheng Zhuang, Fei Ni, Bin Wang, Jianyu Chen, Jianye Hao, Ping Luo*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b0b1cfc8ede53f452cabf8b9cf4eef76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b0b1cfc8ede53f452cabf8b9cf4eef76-Abstract-Conference.html)

**Abstract**:

Adapting to the changes in transition dynamics is essential in robotic applications. By learning a conditional policy with a compact context, context-aware meta-reinforcement learning provides a flexible way to adjust behavior according to dynamics changes. However, in real-world applications, the agent may encounter complex dynamics changes. Multiple confounders can influence the transition dynamics, making it challenging to infer accurate context for decision-making. This paper addresses such a challenge by decomposed mutual information optimization (DOMINO) for context learning, which explicitly learns a disentangled context to maximize the mutual information between the context and historical trajectories while minimizing the state transition prediction error. Our theoretical analysis shows that DOMINO can overcome the underestimation of the mutual information caused by multi-confounded challenges via learning disentangled context and reduce the demand for the number of samples collected in various environments. Extensive experiments show that the context learned by DOMINO benefits both model-based and model-free reinforcement learning algorithms for dynamics generalization in terms of sample efficiency and performance in unseen environments.

----

## [1999] Generalization Bounds with Minimal Dependency on Hypothesis Class via Distributionally Robust Optimization

**Authors**: *Yibo Zeng, Henry Lam*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b0dc3753faa0f55cb6e548bbe414bd08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b0dc3753faa0f55cb6e548bbe414bd08-Abstract-Conference.html)

**Abstract**:

Established approaches to obtain generalization bounds in data-driven optimization and machine learning mostly build on solutions from empirical risk minimization (ERM), which depend crucially on the functional complexity of the hypothesis class. In this paper, we present an alternate route to obtain these bounds on the solution from distributionally robust optimization (DRO), a recent data-driven optimization framework based on worst-case analysis and the notion of ambiguity set to capture statistical uncertainty. In contrast to the hypothesis class complexity in ERM, our DRO bounds depend on the ambiguity set geometry and its compatibility with the true loss function. Notably, when using statistical distances such as maximum mean discrepancy, Wasserstein distance, or $\phi$-divergence in the DRO, our analysis implies generalization bounds whose dependence on the hypothesis class appears the minimal possible: The bound depends solely on the true loss function, independent of any other candidates in the hypothesis class.  To our best knowledge, it is the first generalization bound of this type in the literature, and we hope our findings can open the door for a better understanding of DRO, especially its benefits on loss minimization and other machine learning applications.

----



[Go to the previous page](NIPS-2022-list09.md)

[Go to the next page](NIPS-2022-list11.md)

[Go to the catalog section](README.md)