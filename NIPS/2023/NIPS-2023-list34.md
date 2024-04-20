## [0] Entropic Neural Optimal Transport via Diffusion Processes

**Authors**: *Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry P. Vetrov, Evgeny Burnaev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeac51414a11484d048432f614d5bb1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeac51414a11484d048432f614d5bb1b-Abstract-Conference.html)

**Abstract**:

We propose a novel neural algorithm for the fundamental problem of computing the entropic optimal transport (EOT) plan between probability distributions which are accessible by samples. Our algorithm is based on the saddle point reformulation of the dynamic version of EOT which is known as the Schr√∂dinger Bridge problem. In contrast to the prior methods for large-scale EOT, our algorithm is end-to-end and consists of a single learning step, has fast inference procedure, and allows handling small values of the entropy regularization coefficient which is of particular importance in some applied problems. Empirically, we show the performance of the method on several large-scale EOT tasks. The code for the ENOT solver can be found at https://github.com/ngushchin/EntropicNeuralOptimalTransport

----

## [0] On the Convergence to a Global Solution of Shuffling-Type Gradient Algorithms

**Authors**: *Lam M. Nguyen, Trang H. Tran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html)

**Abstract**:

Stochastic gradient descent (SGD) algorithm is the method of choice in many machine learning tasks thanks to its scalability and efficiency in dealing with large-scale problems. In this paper, we focus on the shuffling version of SGD which matches the mainstream practical heuristics. We show the convergence to a global solution of shuffling SGD for a class of non-convex functions under over-parameterized settings. Our analysis employs more relaxed non-convex assumptions than previous literature. Nevertheless, we maintain the desired computational complexity as shuffling SGD has achieved in the general convex setting.

----

## [0] Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts

**Authors**: *Gleb Bazhenov, Denis Kuznedelev, Andrey Malinin, Artem Babenko, Liudmila Prokhorenkova*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html)

**Abstract**:

In reliable decision-making systems based on machine learning, models have to be robust to distributional shifts or provide the uncertainty of their predictions. In node-level problems of graph learning, distributional shifts can be especially complex since the samples are interdependent. To evaluate the performance of graph models, it is important to test them on diverse and meaningful distributional shifts. However, most graph benchmarks considering distributional shifts for node-level problems focus mainly on node features, while structural properties are also essential for graph problems. In this work, we propose a general approach for inducing diverse distributional shifts based on graph structure. We use this approach to create data splits according to several structural node properties: popularity, locality, and density. In our experiments, we thoroughly evaluate the proposed distributional shifts and show that they can be quite challenging for existing graph models. We also reveal that simple models often outperform more sophisticated methods on the considered structural shifts. Finally, our experiments provide evidence that there is a trade-off between the quality of learned representations for the base classification task under structural distributional shift and the ability to separate the nodes from different distributions using these representations.

----

## [0] Learning Causal Models under Independent Changes

**Authors**: *Sarah Mameche, David Kaltenpoth, Jilles Vreeken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eee6efe709623f36483e3fbb0bb513dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eee6efe709623f36483e3fbb0bb513dd-Abstract-Conference.html)

**Abstract**:

In many scientific applications, we observe a system in different conditions in which its components may change, rather than in isolation. In our work, we are interested in explaining the generating process of such a multi-context system using a finite mixture of causal mechanisms. Recent work shows that this causal model is identifiable from data, but is limited to settings where the sparse mechanism shift hypothesis holds and only a subset of the causal conditionals change. As this assumption is not easily verifiable in practice, we study the more general principle that mechanism shifts are independent, which we formalize using the algorithmic notion of independence. We introduce an approach for causal discovery beyond partially directed graphs using Gaussian Process models, and give conditions under which we provably identify the correct causal model. In our experiments, we show that our method performs well in a range of synthetic settings, on realistic gene expression simulations, as well as on real-world cell signaling data.

----

## [0] Universality and Limitations of Prompt Tuning

**Authors**: *Yihan Wang, Jatin Chauhan, Wei Wang, Cho-Jui Hsieh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eef6aecfe050b556c6a48d9c16b15558-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eef6aecfe050b556c6a48d9c16b15558-Abstract-Conference.html)

**Abstract**:

Despite the demonstrated empirical efficacy of prompt tuning to adapt a pretrained language model for a new task, the theoretical underpinnings of the difference between "tuning parameters before the input" against "the tuning of model weights" are limited. We thus take one of the first steps to understand the role of soft-prompt tuning for transformer-based architectures. By considering a general purpose architecture, we analyze prompt tuning from the lens of both: universal approximation and limitations with finite-depth fixed-weight pretrained transformers for continuous-valued functions. Our universality result guarantees the existence of a strong transformer with a prompt to approximate any sequence-to-sequence function in the set of Lipschitz functions. The limitations of prompt tuning for limited-depth transformers are first proved by constructing a set of datasets, that cannot be memorized by a prompt of any length for a given single encoder layer. We also provide a lower bound on the required number of tunable prompt parameters and compare the result with the number of parameters required for a low-rank update (based on LoRA) for a single-layer setting. We finally extend our analysis to multi-layer settings by providing sufficient conditions under which the transformer can at best learn datasets from invertible functions only. Our theoretical claims are also corroborated by empirical results.

----

## [0] Evaluating Neuron Interpretation Methods of NLP Models

**Authors**: *Yimin Fan, Fahim Dalvi, Nadir Durrani, Hassan Sajjad*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eef6cb60fd59b32d35718e176b4b08d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eef6cb60fd59b32d35718e176b4b08d6-Abstract-Conference.html)

**Abstract**:

Neuron interpretation offers valuable insights into how knowledge is structured within a deep neural network model. While a number of neuron interpretation methods have been proposed in the literature, the field lacks a comprehensive comparison among these methods. This gap hampers progress due to the absence of standardized metrics and benchmarks. The commonly used evaluation metric has limitations, and creating ground truth annotations for neurons is impractical. Addressing these challenges, we propose an evaluation framework based on voting theory. Our hypothesis posits that neurons consistently identified by different methods carry more significant information. We rigorously assess our framework across a diverse array of neuron interpretation methods. Notable findings include: i) despite the theoretical differences among the methods, neuron ranking methods share over 60% of their rankings when identifying salient neurons, ii) the neuron interpretation methods are most sensitive to the last layer representations, iii) Probeless neuron ranking emerges as the most consistent method.

----

## [0] How Re-sampling Helps for Long-Tail Learning?

**Authors**: *Jiang-Xin Shi, Tong Wei, Yuke Xiang, Yu-Feng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeffa70bcbbd43f6bd067edebc6595e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeffa70bcbbd43f6bd067edebc6595e8-Abstract-Conference.html)

**Abstract**:

Long-tail learning has received significant attention in recent years due to the challenge it poses with extremely imbalanced datasets. In these datasets, only a few classes (known as the head classes) have an adequate number of training samples, while the rest of the classes (known as the tail classes) are infrequent in the training data. Re-sampling is a classical and widely used approach for addressing class imbalance issues. Unfortunately, recent studies claim that re-sampling brings negligible performance improvements in modern long-tail learning tasks. This paper aims to investigate this phenomenon systematically. Our research shows that re-sampling can considerably improve generalization when the training images do not contain semantically irrelevant contexts. In other scenarios, however, it can learn unexpected spurious correlations between irrelevant contexts and target labels. We design experiments on two homogeneous datasets, one containing irrelevant context and the other not, to confirm our findings. To prevent the learning of spurious correlations, we propose a new context shift augmentation module that generates diverse training images for the tail class by maintaining a context bank extracted from the head-class images. Experiments demonstrate that our proposed module can boost the generalization and outperform other approaches, including class-balanced re-sampling, decoupled classifier re-training, and data augmentation methods. The source code is available at https://www.lamda.nju.edu.cn/code_CSA.ashx.

----

## [0] FIND: A Function Description Benchmark for Evaluating Interpretability Methods

**Authors**: *Sarah Schwettmann, Tamar Rott Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0164c1112f56246224af540857348f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0164c1112f56246224af540857348f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of trained neural networks, and accompanying descriptions of the kind we seek to generate. The functions are procedurally constructed across textual and numeric domains, and involve a range of real-world complexities, including noise, composition, approximation, and bias. We evaluate methods that use pretrained language models (LMs) to produce code-based and natural language descriptions of function behavior. Additionally, we introduce a new interactive method in which an Automated Interpretability Agent (AIA) generates function descriptions. We find that an AIA, built with an off-the-shelf LM augmented with black-box access to functions, can sometimes infer function structureâ€”acting as a scientist by forming hypotheses, proposing experiments, and updating descriptions in light of new data. However, FIND also reveals that LM-based descriptions capture global function behavior while missing local details. These results suggest that FIND will be useful for characterizing the performance of more sophisticated interpretability methods before they are applied to real-world models.

----

## [0] EgoTracks: A Long-term Egocentric Visual Object Tracking Dataset

**Authors**: *Hao Tang, Kevin J. Liang, Kristen Grauman, Matt Feiszli, Weiyao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef01d91aa87e7701aa9c8dc66a2d5bdb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef01d91aa87e7701aa9c8dc66a2d5bdb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Visual object tracking is a key component to many egocentric vision problems. However, the full spectrum of challenges of egocentric tracking faced by an embodied AI is underrepresented in many existing datasets; these tend to focus on relatively short, third-person videos. Egocentric video has several distinguishing characteristics from those commonly found in past datasets: frequent large camera motions and hand interactions with objects commonly lead to occlusions or objects exiting the frame, and object appearance can change rapidly due to widely different points of view, scale, or object states. Embodied tracking is also naturally long-term, and being able to consistently (re-)associate objects to their appearances and disappearances over as long as a lifetime is critical. Previous datasets under-emphasize this re-detection problem, and their "framed" nature has led to adoption of various spatiotemporal priors that we find do not necessarily generalize to egocentric video. We thus introduce EgoTracks, a new dataset for long-term egocentric visual object tracking. Sourced from the Ego4D dataset, this new dataset presents a significant challenge to recent state-of-the-art single-object tracking models, which we find score poorly on traditional tracking metrics for our new dataset, compared to popular benchmarks. We further show improvements that can be made to a STARK tracker to significantly increase its performance on egocentric data, resulting in a baseline model we call EgoSTARK. We publicly release our annotations and benchmark, hoping our dataset leads to further advancements in tracking.

----

## [0] Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models

**Authors**: *Andrew Luo, Maggie Henderson, Leila Wehbe, Michael J. Tarr*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html)

**Abstract**:

A long standing goal in neuroscience has been to elucidate the functional organization of the brain. Within higher visual cortex, functional accounts have remained relatively coarse, focusing on regions of interest (ROIs) and taking the form of selectivity for broad categories such as faces, places, bodies, food, or words. Because the identification of such ROIs has typically relied on manually assembled stimulus sets consisting of isolated objects in non-ecological contexts, exploring functional organization without robust a priori hypotheses has been challenging. To overcome these limitations, we introduce a data-driven approach in which we synthesize images predicted to activate a given brain region using paired natural images and fMRI recordings, bypassing the need for category-specific stimuli. Our approach -- Brain Diffusion for Visual Exploration ("BrainDiVE") -- builds on recent generative methods by combining large-scale diffusion models with brain-guided image synthesis. Validating our method, we demonstrate the ability to synthesize preferred images with appropriate semantic specificity for well-characterized category-selective ROIs. We then show that BrainDiVE can characterize differences between ROIs selective for the same high-level category. Finally we identify novel functional subdivisions within these ROIs, validated with behavioral data. These  results advance our understanding of the fine-grained functional organization of human visual cortex, and provide well-specified constraints for further examination of cortical organization using hypothesis-driven methods.

----

## [0] Query-based Temporal Fusion with Explicit Motion for 3D Object Detection

**Authors**: *Jinghua Hou, Zhe Liu, Dingkang Liang, Zhikang Zou, Xiaoqing Ye, Xiang Bai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0dcb44a47185f5bacac62571f6e920-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0dcb44a47185f5bacac62571f6e920-Abstract-Conference.html)

**Abstract**:

Effectively utilizing temporal information to improve 3D detection performance is vital for autonomous driving vehicles. Existing methods either conduct temporal fusion based on the dense BEV features or sparse 3D proposal features. However, the former does not pay more attention to foreground objects, leading to more computation costs and sub-optimal performance. The latter implements time-consuming operations to generate sparse 3D proposal features, and the performance is limited by the quality of 3D proposals. In this paper, we propose a simple and effective Query-based Temporal Fusion Network (QTNet). The main idea is to exploit the object queries in previous frames to enhance the representation of current object queries by the proposed Motion-guided Temporal Modeling (MTM) module, which utilizes the spatial position information of object queries along the temporal dimension to construct their relevance between adjacent frames reliably. Experimental results show our proposed QTNet outperforms BEV-based or proposal-based manners on the nuScenes dataset. Besides, the MTM is a plug-and-play module, which can be integrated into some advanced LiDAR-only or multi-modality 3D detectors and even brings new SOTA performance with negligible computation cost and latency on the nuScenes dataset. These experiments powerfully illustrate the superiority and generalization of our method. The code is available at https://github.com/AlmoonYsl/QTNet.

----

## [0] Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection

**Authors**: *Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan S. Kankanhalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html)

**Abstract**:

Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/EfficientACLvia_RCS.

----

## [0] A Finite-Sample Analysis of Payoff-Based Independent Learning in Zero-Sum Stochastic Games

**Authors**: *Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman E. Ozdaglar, Adam Wierman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef62614753535977071395fb1f1435be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef62614753535977071395fb1f1435be-Abstract-Conference.html)

**Abstract**:

In this work, we study two-player zero-sum stochastic games and develop a variant of the smoothed best-response learning dynamics that combines independent learning dynamics for matrix games with the minimax value iteration for stochastic games. The resulting learning dynamics are payoff-based, convergent, rational, and symmetric between the two players. Our theoretical results present to the best of our knowledge the first last-iterate finite-sample analysis of such independent learning dynamics. To establish the results, we develop a coupled Lyapunov drift approach to capture the evolution of multiple sets of coupled and stochastic iterates, which might be of independent interest.

----

## [0] Compact Neural Volumetric Video Representations with Dynamic Codebooks

**Authors**: *Haoyu Guo, Sida Peng, Yunzhi Yan, Linzhan Mou, Yujun Shen, Hujun Bao, Xiaowei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef63b00ad8475605b2eaf520747f61d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef63b00ad8475605b2eaf520747f61d4-Abstract-Conference.html)

**Abstract**:

This paper addresses the challenge of representing high-fidelity volumetric videos with low storage cost. Some recent feature grid-based methods have shown superior performance of fast learning implicit neural representations from input 2D images. However, such explicit representations easily lead to large model sizes when modeling dynamic scenes. To solve this problem, our key idea is reducing the spatial and temporal redundancy of feature grids, which intrinsically exist due to the self-similarity of scenes. To this end, we propose a novel neural representation, named dynamic codebook, which first merges similar features for the model compression and then compensates for the potential decline in rendering quality by a set of dynamic codes. Experiments on the NHR and DyNeRF datasets demonstrate that the proposed approach achieves state-of-the-art rendering quality, while being able to achieve more storage efficiency. The source code is available at https://github.com/zju3dv/compact_vv.

----

## [0] Towards Label-free Scene Understanding by Vision Foundation Models

**Authors**: *Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, Wenping Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef6c94e9cf4d169298479ee2e230ee13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef6c94e9cf4d169298479ee2e230ee13-Abstract-Conference.html)

**Abstract**:

Vision foundation models such as Contrastive Vision-Language Pre-training (CLIP) and Segment Anything (SAM) have demonstrated impressive zero-shot performance on image classification and segmentation tasks. However, the incorporation of CLIP and SAM for label-free scene understanding has yet to be explored. In this paper, we investigate the potential of vision foundation models in enabling networks to comprehend 2D and 3D worlds without labelled data. The primary challenge lies in effectively supervising networks under extremely noisy pseudo labels, which are generated by CLIP and further exacerbated during the propagation from the 2D to the 3D domain. To tackle these challenges, we propose a novel Cross-modality Noisy Supervision (CNS) method that leverages the strengths of CLIP and SAM to supervise 2D and 3D networks simultaneously. In particular, we introduce a prediction consistency regularization to co-train 2D and 3D networks, then further impose the networks' latent space consistency using the SAM's robust feature representation. Experiments conducted on diverse indoor and outdoor datasets demonstrate the superior performance of our method in understanding 2D and 3D open environments. Our 2D and 3D network achieves label-free semantic segmentation with 28.4\% and 33.5\% mIoU on ScanNet, improving 4.7\% and 7.9\%, respectively. For nuImages and nuScenes datasets, the performance is 22.1\% and 26.8\% with improvements of 3.5\% and 6.0\%, respectively. Code is available. (https://github.com/runnanchen/Label-Free-Scene-Understanding)

----

## [0] Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions

**Authors**: *Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, Elad Hazan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html)

**Abstract**:

Adaptive regularization methods that exploit more than the diagonal entries exhibit state of the art performance for many tasks, but can be prohibitive in terms of memory and running time. We find the spectra of the Kronecker-factored gradient covariance matrix in deep learning (DL) training tasks are concentrated on a small leading eigenspace that changes throughout training, motivating a low-rank sketching approach. We describe a generic method for reducing memory and compute requirements of maintaining a matrix preconditioner using the Frequent Directions (FD) sketch. While previous approaches have explored applying FD for second-order optimization, we present a novel analysis which allows efficient interpolation between resource requirements and the degradation in regret guarantees with rank $k$: in the online convex optimization (OCO) setting over dimension $d$, we match full-matrix $d^2$ memory regret using only $dk$ memory up to additive error in the bottom $d-k$ eigenvalues of the gradient covariance. Further, we show extensions of our work to Shampoo, resulting in a method competitive in quality with Shampoo and Adam, yet requiring only sub-linear memory for tracking second moments.

----

## [0] SatBird: a Dataset for Bird Species Distribution Modeling using Remote Sensing and Citizen Science Data

**Authors**: *Mélisande Teng, Amna Elmustafa, Benjamin Akera, Yoshua Bengio, Hager Radi Abdelwahed, Hugo Larochelle, David Rolnick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef7653bbc4655305efb89a32362e332a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef7653bbc4655305efb89a32362e332a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Biodiversity is declining at an unprecedented rate, impacting ecosystem services necessary to ensure food, water, and human health and well-being. Understanding the distribution of species and their habitats is crucial for conservation policy planning. However, traditional methods in ecology for species distribution models (SDMs) generally focus either on narrow sets of species or narrow geographical areas and there remain significant knowledge gaps about the distribution of species. A major reason for this is the limited availability of data traditionally used, due to the prohibitive amount of effort and expertise required for traditional field monitoring. The wide availability of remote sensing data and the growing adoption of citizen science tools to collect species observations data at low cost offer an opportunity for improving biodiversity monitoring and enabling the modelling of complex ecosystems. We introduce a novel task for mapping bird species to their habitats by predicting species encounter rates from satellite images, and present SatBird, a satellite dataset of locations in the USA with labels derived from presence-absence observation data from the citizen science database eBird, considering summer (breeding) and winter seasons. We also provide a dataset in Kenya representing low-data regimes. We additionally provide environmental data and species range maps for each location.  We benchmark a set of baselines on our dataset, including SOTA models for remote sensing tasks. SatBird opens up possibilities for scalably modelling properties of ecosystems worldwide.

----

## [0] DiffComplete: Diffusion-based Generative 3D Shape Completion

**Authors**: *Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Nießner, Chi-Wing Fu, Jiaya Jia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef7bd1f9cbf8a5ab7ddcaccd50699c90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef7bd1f9cbf8a5ab7ddcaccd50699c90-Abstract-Conference.html)

**Abstract**:

We introduce a new diffusion-based approach for shape completion on 3D range scans. Compared with prior deterministic and probabilistic methods, we strike a balance between realism, multi-modality, and high fidelity. We propose DiffComplete by casting shape completion as a generative task conditioned on the incomplete shape. Our key designs are two-fold. First, we devise a hierarchical feature aggregation mechanism to inject conditional features in a spatially-consistent manner. So, we can capture both local details and broader contexts of the conditional inputs to control the shape completion. Second, we propose an occupancy-aware fusion strategy in our model to enable the completion of multiple partial shapes and introduce higher flexibility on the input conditions. DiffComplete sets a new SOTA performance (e.g., 40% decrease on $l_1$ error) on two large-scale 3D shape completion benchmarks. Our completed shapes not only have a realistic outlook compared with the deterministic methods but also exhibit high similarity to the ground truths compared with the probabilistic alternatives. Further, DiffComplete has strong generalizability on objects of entirely unseen classes for both synthetic and real data, eliminating the need for model re-training in various applications.

----

## [0] Bayesian Risk-Averse Q-Learning with Streaming Observations

**Authors**: *Yuhao Wang, Enlu Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)

**Abstract**:

We consider a robust reinforcement learning problem, where a learning agent learns from a simulated training environment. To account for the model mis-specification between this training environment and the true environment due to lack of data, we adopt a formulation of Bayesian risk MDP (BRMDP) with infinite horizon, which uses Bayesian posterior to estimate the transition model and impose a risk functional to account for the model uncertainty. Observations from the real environment that is out of the agent's control arrive periodically and are utilized by the agent to update the Bayesian posterior to reduce model uncertainty. We theoretically demonstrate that BRMDP balances the trade-off between robustness and conservativeness, and we further develop a multi-stage Bayesian risk-averse Q-learning algorithm to solve BRMDP with streaming observations from real environment. The proposed algorithm learns a risk-averse yet optimal policy that depends on the availability of real-world observations. We provide a theoretical guarantee of strong convergence for the proposed algorithm.

----

## [0] On the Planning Abilities of Large Language Models - A Critical Investigation

**Authors**: *Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, Subbarao Kambhampati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html)

**Abstract**:

Intrigued by the claims of emergent reasoning capabilities in LLMs trained on general web corpora, in this paper, we set out to investigate their planning capabilities. We aim to evaluate (1) the effectiveness of LLMs in generating plans autonomously in commonsense planning tasks and (2) the potential of LLMs as a source of heuristic guidance for other agents (AI planners) in their planning tasks. We conduct a systematic study by generating a suite of instances on domains similar to the ones employed in the International Planning Competition and evaluate LLMs in two distinct modes: autonomous and heuristic. Our findings reveal that LLMsâ€™ ability to generate executable plans autonomously is rather limited, with the best model (GPT-4) having an average success rate of ~12% across the domains. However, the results in the heuristic mode show more promise. In the heuristic mode, we demonstrate that LLM-generated plans can improve the search process for underlying sound planners and additionally show that external verifiers can help provide feedback on the generated plans and back-prompt the LLM for better plan generation.

----

## [0] Is RLHF More Difficult than Standard RL? A Theoretical Perspective

**Authors**: *Yuanhao Wang, Qinghua Liu, Chi Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efb9629755e598c4f261c44aeb6fde5e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efb9629755e598c4f261c44aeb6fde5e-Abstract-Conference.html)

**Abstract**:

Reinforcement learning from Human Feedback (RLHF) learns from preference signals, while standard Reinforcement Learning (RL) directly learns from reward signals. Preferences arguably contain less information than rewards, which makes preference-based RL seemingly more difficult. This paper theoretically proves that, for a wide range of preference models, we can solve preference-based RL directly using existing algorithms and techniques for reward-based RL, with small or no extra costs. Specifically, (1) for preferences that are drawn from reward-based probabilistic models, we reduce the problem to robust reward-based RL that can tolerate small errors in rewards; (2) for general arbitrary preferences where the objective is to find the von Neumann winner, we reduce the problem to multiagent reward-based RL which finds Nash equilibria for factored Markov games under a restricted set of policies. The latter case can be further reduce to adversarial MDP when preferences only depend on the final state. We instantiate all reward-based RL subroutines by concrete provable algorithms, and apply our theory to a large class of models including tabular MDPs and MDPs with generic function approximation. We further provide guarantees when K-wise comparisons are available.

----

## [0] How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model

**Authors**: *Michael Hanna, Ollie Liu, Alexandre Variengien*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html)

**Abstract**:

Pre-trained language models can be surprisingly adept at tasks they were not explicitly trained on, but how they implement these capabilities is poorly understood. In this paper, we investigate the basic mathematical abilities often acquired by pre-trained language models. Concretely, we use mechanistic interpretability techniques to explain the (limited) mathematical abilities of GPT-2 small. As a case study, we examine its ability to take in sentences such as "The war lasted from the year 1732 to the year 17", and predict valid two-digit end years (years > 32). We first identify a circuit, a small subset of GPT-2 small's computational graph that computes this task's output. Then, we explain the role of each circuit component, showing that GPT-2 small's final multi-layer perceptrons boost the probability of end years greater than the start year. Finally, we find related tasks that activate our circuit. Our results suggest that GPT-2 small computes greater-than using a complex but general mechanism that activates across diverse contexts.

----

## [0] NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations

**Authors**: *Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, André Araújo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, Howard Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efc90033e6e1b05485312dd09fe302b8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/efc90033e6e1b05485312dd09fe302b8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent advances in neural reconstruction enable high-quality 3D object reconstruction from casually captured image collections. Current techniques mostly analyze their progress on relatively simple image collections where SfM techniques can provide ground-truth (GT) camera poses. We note that SfM techniques tend to fail on in-the-wild image collections such as image search results with varying backgrounds and illuminations. To enable systematic research progress on 3D reconstruction from casual image captures, we propose `NAVI': a new dataset of category-agnostic image collections of objects with high-quality 3D scans along with per-image 2D-3D alignments providing near-perfect GT camera parameters. These 2D-3D alignments allow us to extract accurate derivative annotations such as dense pixel correspondences, depth and segmentation maps. We demonstrate the use of NAVI image collections on different problem settings and show that NAVI enables more thorough evaluations that were not possible with existing datasets. We believe NAVI is beneficial for systematic research progress on 3D reconstruction and correspondence estimation.

----

## [0] Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation

**Authors**: *Jia-Xing Zhong, Ta Ying Cheng, Yuhang He, Kai Lu, Kaichen Zhou, Andrew Markham, Niki Trigoni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efca456a4e861f3b47455c44bb134424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efca456a4e861f3b47455c44bb134424-Abstract-Conference.html)

**Abstract**:

A truly generalizable approach to rigid segmentation and motion estimation is fundamental to 3D understanding of articulated objects and moving scenes. In view of the closely intertwined relationship between segmentation and motion estimates, we present an SE(3) equivariant architecture and a training strategy to tackle this task in an unsupervised manner. Our architecture is composed of two interconnected, lightweight heads. These heads predict segmentation masks using point-level invariant features and estimate motion from SE(3) equivariant features, all without the need for category information. Our training strategy is unified and can be implemented online, which jointly optimizes the predicted segmentation and motion by leveraging the interrelationships among scene flow, segmentation mask, and rigid transformations. We conduct experiments on four datasets to demonstrate the superiority of our method. The results show that our method excels in both model performance and computational efficiency, with only 0.25M parameters and 0.92G FLOPs. To the best of our knowledge, this is the first work designed for category-agnostic part-level SE(3) equivariance in dynamic point clouds.

----

## [0] Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning

**Authors**: *Fuqi Jia, Yuhang Dong, Minghao Liu, Pei Huang, Feifei Ma, Jian Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efcb5b06ce8bb672ffa26b9dc5cdd0f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efcb5b06ce8bb672ffa26b9dc5cdd0f9-Abstract-Conference.html)

**Abstract**:

Cylindrical Algebraic Decomposition (CAD) is one of the pillar algorithms of symbolic computation, and its worst-case complexity is double exponential to the number of variables. Researchers found that variable order dramatically affects efficiency and proposed various heuristics. The existing learning-based methods are all supervised learning methods that cannot cope with diverse polynomial sets.This paper proposes two Reinforcement Learning (RL) approaches combined with Graph Neural Networks (GNN) for Suggesting Variable Order (SVO). One is GRL-SVO(UP), a branching heuristic integrated with CAD. The other is GRL-SVO(NUP), a fast heuristic providing a total order directly. We generate a random dataset and collect a real-world dataset from SMT-LIB. The experiments show that our approaches outperform state-of-the-art learning-based heuristics and are competitive with the best expert-based heuristics. Interestingly, our models show a strong generalization ability, working well on various datasets even if they are only trained on a 3-var random dataset. The source code and data are available at https://github.com/dongyuhang22/GRL-SVO.

----

## [0] GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER

**Authors**: *Mingzhen Sun, Weining Wang, Zihan Qin, Jiahui Sun, Sihan Chen, Jing Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html)

**Abstract**:

Video generation necessitates both global coherence and local realism. This work presents a novel non-autoregressive method GLOBER, which first generates global features to obtain comprehensive global guidance and then synthesizes video frames based on the global features to generate coherent videos. Specifically, we propose a video auto-encoder, where a video encoder encodes videos into global features, and a video decoder, built on a diffusion model, decodes the global features and synthesizes video frames in a non-autoregressive manner. To achieve maximum flexibility, our video decoder perceives temporal information through normalized frame indexes, which enables it to synthesize arbitrary sub video clips with predetermined starting and ending frame indexes. Moreover, a novel adversarial loss is introduced to improve the global coherence and local realism between the synthesized video frames. Finally, we employ a diffusion-based video generator to fit the global features outputted by the video encoder for video generation. Extensive experimental results demonstrate the effectiveness and efficiency of our proposed method, and new state-of-the-art results have been achieved on multiple benchmarks.

----

## [0] Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models

**Authors**: *Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogério Feris, Shimon Ullman, Leonid Karlinsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efe406d6d2674d176cdcd958ce605d17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efe406d6d2674d176cdcd958ce605d17-Abstract-Conference.html)

**Abstract**:

Vision and Language (VL) models offer an effective method for aligning representation spaces of images and text allowing for numerous applications such as cross-modal retrieval, visual and multi-hop question answering, captioning, and many more. However, the aligned image-text spaces learned by all the popular VL models are still suffering from the so-called 'object bias' - their representations behave as 'bags of nouns' mostly ignoring or downsizing the attributes, relations, and states of objects described/appearing in texts/images. Although some great attempts at fixing these `compositional reasoning' issues were proposed in the recent literature, the problem is still far from being solved. In this paper, we uncover two factors limiting the VL models' compositional reasoning performance. These two factors are properties of the paired VL dataset used for finetuning (or pre-training) the VL model: (i) the caption quality, or in other words 'image-alignment', of the texts; and (ii) the 'density' of the captions in the sense of mentioning all the details appearing on the image. We propose a fine-tuning approach for automatically treating these factors on a standard collection of paired VL data (CC3M). Applied to CLIP, we demonstrate its significant compositional reasoning performance increase of up to $\sim27$\% over the base model, up to $\sim20$\% over the strongest baseline, and by $6.7$\% on average. Our code is provided in the Supplementary and would be released upon acceptance.

----

## [0] Latent SDEs on Homogeneous Spaces

**Authors**: *Sebastian Zeng, Florian Graf, Roland Kwitt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0172a5da5a2611e3dc0fe9c6e9a7480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0172a5da5a2611e3dc0fe9c6e9a7480-Abstract-Conference.html)

**Abstract**:

We consider the problem of variational Bayesian inference in a latent variable model where a (possibly complex) observed stochastic process is governed by the unobserved solution of a latent stochastic differential equation (SDE). Motivated by the challenges that arise when trying to learn a latent SDE in $\mathbb{R}^n$ from large-scale data, such as efficient gradient computation, we take a step back and study a specific subclass instead. In our case, the SDE evolves inside a homogeneous latent space and is induced by stochastic dynamics of the corresponding (matrix) Lie group. In the context of learning problems, SDEs on the $n$-dimensional unit sphere are arguably the most relevant incarnation of this setup. For variational inference, the sphere not only facilitates using a uniform prior on the initial state of the SDE, but we also obtain a particularly simple and intuitive expression for the KL divergence between the approximate posterior and prior process in the evidence lower bound. We provide empirical evidence that a latent SDE of the proposed type can be learned efficiently by means of an existing one-step geometric Euler-Maruyama scheme. Despite restricting ourselves to a less diverse class of SDEs, we achieve competitive or even state-of-the-art performance on a collection of time series interpolation and classification benchmarks.

----

## [0] Balancing Risk and Reward: A Batched-Bandit Strategy for Automated Phased Release

**Authors**: *Yufan Li, Jialiang Mao, Iavor Bojinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f02a7dd6bd3d038b51d092d99e74c638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f02a7dd6bd3d038b51d092d99e74c638-Abstract-Conference.html)

**Abstract**:

Phased releases are a common strategy in the technology industry for gradually releasing new products or updates through a sequence of A/B tests in which the number of treated units gradually grows until full deployment or deprecation. Performing phased releases in a principled way requires selecting the proportion of units assigned to the new release in a way that balances the risk of an adverse effect with the need to iterate and learn from the experiment rapidly. In this paper, we formalize this problem and propose an algorithm that automatically determines the release percentage at each stage in the schedule, balancing the need to control risk while maximizing ramp-up speed. Our framework models the challenge as a constrained batched bandit problem that ensures that our pre-specified experimental budget is not depleted with high probability. Our proposed algorithm leverages an adaptive Bayesian approach in which the maximal number of units assigned to the treatment is determined by the posterior distribution, ensuring that the probability of depleting the remaining budget is low. Notably, our approach analytically solves the ramp sizes by inverting probability bounds, eliminating the need for challenging rare-event Monte Carlo simulation. It only requires computing means and variances of outcome subsets, making it highly efficient and parallelizable.

----

## [0] Preconditioning Matters: Fast Global Convergence of Non-convex Matrix Factorization via Scaled Gradient Descent

**Authors**: *Xixi Jia, Hailin Wang, Jiangjun Peng, Xiangchu Feng, Deyu Meng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html)

**Abstract**:

Low-rank matrix factorization (LRMF) is a canonical problem in non-convex optimization, the objective function to be minimized is non-convex and even non-smooth, which makes the global convergence guarantee of gradient-based algorithm quite challenging. Recent work made a breakthrough on proving that standard gradient descent converges to the $\varepsilon$-global minima after $O( \frac{d \kappa^2}{\tau^2} {\rm ln} \frac{d \sigma_d}{\tau} + \frac{d \kappa^2}{\tau^2} {\rm ln} \frac{\sigma_d}{\varepsilon})$ iterations from small initialization with a very small learning rate (both are related to the small constant $\tau$). While the dependence of the convergence on the \textit{condition number} $\kappa$ and \textit{small learning rate} makes it not practical especially for ill-conditioned LRMF problem.In this paper, we show that precondition helps in accelerating the convergence and prove that the scaled gradient descent (ScaledGD) and its variant, alternating scaled gradient descent (AltScaledGD) converge to an $\varepsilon$-global minima after $O( {\rm ln} \frac{d}{\delta} + {\rm ln} \frac{d}{\varepsilon})$ iterations from general random initialization. Meanwhile, for small initialization as in gradient descent, both ScaledGD and AltScaledGD converge to $\varepsilon$-global minima after only $O({\rm ln} \frac{d}{\varepsilon})$ iterations. Furthermore, we prove that as a proximity to the alternating minimization, AltScaledGD converges faster than ScaledGD, its global convergence does not rely on small learning rate and small initialization, which certificates the advantages of AltScaledGD in LRMF.

----

## [0] Deep Insights into Noisy Pseudo Labeling on Graph Data

**Authors**: *Botao Wang, Jia Li, Yang Liu, Jiashun Cheng, Yu Rong, Wenjia Wang, Fugee Tsung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0318ba897cee71ce200e408dea6062e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0318ba897cee71ce200e408dea6062e-Abstract-Conference.html)

**Abstract**:

Pseudo labeling (PL) is a wide-applied strategy to enlarge the labeled dataset by self-annotating the potential samples during the training process. Several works have shown that it can improve the graph learning model performance in general. However, we notice that the incorrect labels can be fatal to the graph training process. Inappropriate PL may result in the performance degrading, especially on graph data where the noise can propagate. Surprisingly, the corresponding error is seldom theoretically analyzed in the literature. In this paper, we aim to give deep insights of PL on graph learning models. We first present the error analysis of PL strategy by showing that the error is bounded by the confidence of PL threshold and consistency of multi-view prediction. Then, we theoretically illustrate the effect of PL on convergence property. Based on the analysis, we propose a cautious pseudo labeling methodology in which we pseudo label the samples with highest confidence and multi-view consistency. Finally, extensive experiments demonstrate that the proposed strategy improves graph learning process and outperforms other PL strategies on link prediction and node classification tasks.

----

## [0] Predicting a Protein's Stability under a Million Mutations

**Authors**: *Jeffrey Ouyang-Zhang, Daniel Jesus Diaz, Adam R. Klivans, Philipp Krähenbühl*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html)

**Abstract**:

Stabilizing proteins is a foundational step in protein engineering. However, the evolutionary pressure of all extant proteins makes identifying the scarce number of mutations that will improve thermodynamic stability challenging. Deep learning has recently emerged as a powerful tool for identifying promising mutations.Existing approaches, however, are computationally expensive, as the number of model inferences scales with the number of mutations queried. Our main contribution is a simple, parallel decoding algorithm.Mutate Everything is capable of predicting the effect of all single and double mutations in one forward pass. It is even versatile enough to predict higher-order mutations with minimal computational overhead.We build Mutate Everything on top of ESM2 and AlphaFold, neither of which were trained to predict thermodynamic stability.We trained on the Mega-Scale cDNA proteolysis dataset and achieved state-of-the-art performance on single and higher-order mutations on S669, ProTherm, and ProteinGym datasets.Our code is available at https://github.com/jozhang97/MutateEverything.

----

## [0] Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems

**Authors**: *Fiona Lippert, Bart Kranstauber, Emiel van Loon, Patrick Forré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)

**Abstract**:

Probabilistic inference in high-dimensional state-space models is computationally challenging. For many spatiotemporal systems, however, prior knowledge about the dependency structure of state variables is available. We leverage this structure to develop a computationally efficient approach to state estimation and learning in graph-structured state-space models with (partially) unknown dynamics and limited historical data. Building on recent methods that combine ideas from deep learning with principled inference in Gaussian Markov random fields (GMRF), we reformulate graph-structured state-space models as Deep GMRFs defined by simple spatial and temporal graph layers. This results in a flexible spatiotemporal prior that can be learned efficiently from a single time sequence via variational inference. Under linear Gaussian assumptions, we retain a closed-form posterior, which can be sampled efficiently using the conjugate gradient method, scaling favourably compared to classical Kalman filter based approaches.

----

## [0] Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy

**Authors**: *Amit Daniely, Nati Srebro, Gal Vardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html)

**Abstract**:

Understanding when neural networks can be learned efficientlyis a fundamental question in learning theory.Existing hardness results suggest that assumptions on both the input distribution and the network's weights are necessary for obtaining efficient algorithms. Moreover, it was previously shown that depth-$2$ networks can be efficiently learned under the assumptions that the input distribution is Gaussian, and the weight matrix is non-degenerate. In this work, we study whether such assumptions may suffice for learning deeper networks and prove negative results. We show that learning depth-$3$ ReLU networks under the Gaussian input distribution is hard even in the smoothed-analysis framework, where a random noise is added to the network's parameters. It implies that learning depth-$3$ ReLU networks under the Gaussian distribution is hard even if the weight matrices are non-degenerate. Moreover, we consider depth-$2$ networks, and show hardness of learning in the smoothed-analysis framework, where both the network parameters and the input distribution are smoothed. Our hardness results are under a well-studied assumption on the existence of local pseudorandom generators.

----

## [0] LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

**Authors**: *Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html)

**Abstract**:

We present a novel vision-language prompt learning approach for few-shot out-of-distribution (OOD) detection. Few-shot OOD detection aims to detect OOD images from classes that are unseen during training using only a few labeled in-distribution (ID) images. While prompt learning methods such as CoOp have shown effectiveness and efficiency in few-shot ID classification, they still face limitations in OOD detection due to the potential presence of ID-irrelevant information in text embeddings. To address this issue, we introduce a new approach called $\textbf{Lo}$cal regularized $\textbf{Co}$ntext $\textbf{Op}$timization (LoCoOp), which performs OOD regularization that utilizes the portions of CLIP local features as OOD features during training. CLIP's local features have a lot of ID-irrelevant nuisances ($\textit{e.g.}$, backgrounds), and by learning to push them away from the ID class text embeddings, we can remove the nuisances in the ID class text embeddings and enhance the separation between ID and OOD. Experiments on the large-scale ImageNet OOD detection benchmarks demonstrate the superiority of our LoCoOp over zero-shot, fully supervised detection methods and prompt learning methods. Notably, even in a one-shot setting -- just one label per class, LoCoOp outperforms existing zero-shot and fully supervised detection methods. The code is available via https://github.com/AtsuMiyai/LoCoOp.

----

## [0] AdANNS: A Framework for Adaptive Semantic Search

**Authors**: *Aniket Rege, Aditya Kusupati, Sharan Ranjit S, Alan Fan, Qingqing Cao, Sham M. Kakade, Prateek Jain, Ali Farhadi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html)

**Abstract**:

Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture tail queries and data points, learned representations typically are _rigid, high-dimensional_ vectors that are generally used as-is in the entire ANNS pipeline and can lead to computationally expensive retrieval. In this paper, we argue that instead of rigid representations, different stages of ANNS can leverage _adaptive representations_ of varying capacities to achieve significantly better accuracy-compute trade-offs, i.e., stages of ANNS that can get away with more approximate computation should use a lower-capacity representation of the same data point. To this end, we introduce AdANNS, a novel ANNS design framework that explicitly leverages the flexibility of Matryoshka Representations. We demonstrate state-of-the-art accuracy-compute trade-offs using novel AdANNS-based key ANNS building blocks like search data structures (AdANNS-IVF) and quantization (AdANNS-OPQ). For example on ImageNet retrieval, AdANNS-IVF is up to $\mathbf{1.5}$% more accurate than the rigid representations-based IVF at the same compute budget; and matches accuracy while being up to $\mathbf{90}\times$ faster in _wall-clock time_. For Natural Questions, $32$-byte AdANNS-OPQ matches the accuracy of the $64$-byte OPQ baseline constructed using rigid representations -- _same accuracy at half the cost!_ We further show that the gains from AdANNS translate to modern-day composite ANNS indices that combine search structures and quantization. Finally, we demonstrate that AdANNS can enable inference-time adaptivity for compute-aware search on ANNS indices built non-adaptively on matryoshka representations. Code is open-sourced at https://github.com/RAIVNLab/AdANNS.

----

## [0] When Do Neural Nets Outperform Boosted Trees on Tabular Data?

**Authors**: *Duncan C. McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C., Ganesh Ramakrishnan, Micah Goldblum, Colin White*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Tabular data is one of the most commonly used types of data in machine learning. Despite recent advances in neural nets (NNs) for tabular data, there is still an active discussion on whether or not NNs generally outperform gradient-boosted decision trees (GBDTs) on tabular data, with several recent works arguing either that GBDTs consistently outperform NNs on tabular data, or vice versa. In this work, we take a step back and question the importance of this debate. To this end, we conduct the largest tabular data analysis to date, comparing 19 algorithms across 176 datasets, and we find that the 'NN vs. GBDT' debate is overemphasized: for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs. Next, we analyze dozens of metafeatures to determine what \emph{properties} of a dataset make NNs or GBDTs better-suited to perform well. For example, we find that GBDTs are much better than NNs at handling skewed or heavy-tailed feature distributions and other forms of dataset irregularities. Our insights act as a guide for practitioners to determine which techniques may work best on their dataset. Finally, with the goal of accelerating tabular data research, we release the TabZilla Benchmark Suite: a collection of the 36 'hardest' of the datasets we study. Our benchmark suite, codebase, and all raw results are available at https://github.com/naszilla/tabzilla.

----

## [0] Adversarially Robust Distributed Count Tracking via Partial Differential Privacy

**Authors**: *Zhongzheng Xiong, Xiaoyi Zhu, Zengfeng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html)

**Abstract**:

We study the distributed tracking model, also known as distributed functional monitoring. This model involves $k$ sites each receiving a stream of items and communicating with the central server. The server's task is to track a function of all items received thus far continuously, with minimum communication cost. For count tracking, it is known that there is a $\sqrt{k}$ gap in communication between deterministic and randomized algorithms. However, existing randomized algorithms assume an "oblivious adversary" who constructs the entire input streams before the algorithm starts. Here we consider adaptive adversaries who can choose new items based on previous answers from the algorithm. Deterministic algorithms are trivially robust to adaptive adversaries, while randomized ones may not. Therefore, we investigate whether the $\sqrt{k}$ advantage of randomized algorithms is from randomness itself or the oblivious adversary assumption. We provide an affirmative answer to this question by giving a robust algorithm with optimal communication. Existing robustification techniques do not yield optimal bounds due to the inherent challenges of the distributed nature of the problem. To address this, we extend the differential privacy framework by introducing "partial differential privacy" and proving a new generalization theorem. This theorem may have broader applications beyond robust count tracking, making it of independent interest.

----

## [0] Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models

**Authors**: *Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, Jong Chul Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0878b7efa656b3bbd407c9248d13751-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0878b7efa656b3bbd407c9248d13751-Abstract-Conference.html)

**Abstract**:

Despite the remarkable performance of text-to-image diffusion models in image generation tasks, recent studies have raised the issue that generated images sometimes cannot capture the intended semantic contents of the text prompts, which phenomenon is often called semantic misalignment. To address this, here we present a novel energy-based model (EBM) framework for adaptive context control by modeling the posterior of context vectors. Specifically, we first formulate EBMs of latent image representations and text embeddings in each cross-attention layer of the denoising autoencoder. Then, we obtain the gradient of the log posterior of context vectors, which can be updated and transferred to the subsequent cross-attention layer, thereby implicitly minimizing a nested hierarchy of energy functions. Our latent EBMs further allow zero-shot compositional generation as a linear combination of cross-attention outputs from different contexts. Using extensive experiments, we demonstrate that the proposed method is highly effective in handling various image generation tasks, including multi-concept generation, text-guided image inpainting, and real and synthetic image editing. Code: https://github.com/EnergyAttention/Energy-Based-CrossAttention.

----

## [0] Optimal Algorithms for the Inhomogeneous Spiked Wigner Model

**Authors**: *Aleksandr Pak, Justin Ko, Florent Krzakala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0a6b46b0183a62a2db973014e3429f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0a6b46b0183a62a2db973014e3429f4-Abstract-Conference.html)

**Abstract**:

We study a spiked Wigner problem with an inhomogeneous noise profile. Our aim in this problem is to recover the signal passed through an inhomogeneous low-rank matrix channel. While the information-theoretic performances are well-known, we focus on the algorithmic problem. First, we derive an approximate message-passing algorithm (AMP) for the inhomogeneous problem and show that its rigorous state evolution coincides with the information-theoretic optimal Bayes fixed-point equations. Second, we deduce a simple and efficient spectral method that outperforms PCA and is shown to match the information-theoretic transition.

----

## [0] Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer

**Authors**: *Jianwei Zhang, Suren Jayasuriya, Visar Berisha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0aa7e9e67515fa0c607c2959ccda6a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0aa7e9e67515fa0c607c2959ccda6a0-Abstract-Conference.html)

**Abstract**:

A good supervised embedding for a specific machine learning task is only sensitive to changes in the label of interest and is invariant to other confounding factors. We leverage the concept of repeatability from measurement theory to describe this property and propose to use the intra-class correlation coefficient (ICC) to evaluate the repeatability of embeddings. We then propose a novel regularizer, the ICC regularizer, as a complementary component for contrastive losses to guide deep neural networks to produce embeddings with higher repeatability. We use simulated data to explain why the ICC regularizer works better on minimizing the intra-class variance than the contrastive loss alone. We implement the ICC regularizer and apply it to three speech tasks: speaker verification, voice style conversion, and a clinical application for detecting dysphonic voice. The experimental results demonstrate that adding an ICC regularizer can improve the repeatability of learned embeddings compared to only using the contrastive loss; further, these embeddings lead to improved performance in these downstream tasks.

----

## [0] Momentum Provably Improves Error Feedback!

**Authors**: *Ilyas Fatkhullin, Alexander Tyurin, Peter Richtárik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)

**Abstract**:

Due to the high communication overhead when training machine learning models in a distributed environment, modern algorithms invariably rely on lossy communication compression. However, when untreated, the errors caused by compression propagate, and can lead to severely unstable behavior, including exponential divergence. Almost a decade ago, Seide et al. [2014] proposed an error feedback (EF) mechanism, which we refer to as EF14, as an immensely effective heuristic for mitigating this issue. However, despite steady algorithmic and theoretical  advances in the EF field in the last decade, our understanding is far from complete. In this work we address one of the most pressing issues. In particular, in the canonical nonconvex setting, all known variants of EF rely on very large batch sizes to converge, which can be prohibitive in practice. We propose a surprisingly simple fix which removes this issue both theoretically, and in practice: the application of Polyak's momentum to the latest incarnation of EF due to Richt√°rik et al. [2021] known as EF21. Our algorithm, for which we coin the name EF21-SGDM, improves the communication and sample complexities of previous error feedback algorithms under standard smoothness and bounded variance assumptions, and does not require any further strong assumptions such as bounded gradient dissimilarity. Moreover, we propose a double momentum version of our method that improves the complexities even further. Our proof seems to be novel even when compression is removed form the method, and as such, our proof technique is of independent interest in the study of nonconvex stochastic optimization enriched with Polyak's momentum.

----

## [0] Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes

**Authors**: *Emmeran Johnson, Ciara Pike-Burke, Patrick Rebeschini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html)

**Abstract**:

Policy Mirror Descent (PMD) is a general family of algorithms that covers a wide range of novel and fundamental methods in reinforcement learning. Motivated by the instability of policy iteration (PI) with inexact policy evaluation, unregularised PMD algorithmically regularises the policy improvement step of PI without regularising the objective function. With exact policy evaluation, PI is known to converge linearly with a rate given by the discount factor $\gamma$ of a Markov Decision Process. In this work, we bridge the gap between PI and PMD with exact policy evaluation and show that the dimension-free $\gamma$-rate of PI can be achieved by the general family of unregularised PMD algorithms under an adaptive step-size. We show that both the rate and step-size are unimprovable for PMD: we provide matching lower bounds that demonstrate that the $\gamma$-rate is optimal for PMD methods as well as PI and that the adaptive step-size is necessary to achieve it. Our work is the first to relate PMD to rate-optimality and step-size necessity. Our study of the convergence of PMD avoids the use of the performance difference lemma, which leads to a direct analysis of independent interest. We also extend the analysis to the inexact setting and establish the first dimension-optimal sample complexity for unregularised PMD under a generative model, improving upon the best-known result.

----

## [0] Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models

**Authors**: *Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhihua Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html)

**Abstract**:

Due to the ease of training, ability to scale, and high sample quality, diffusion models (DMs) have become the preferred option for generative modeling, with numerous pre-trained models available for a wide variety of datasets. Containing intricate information about data distributions, pre-trained DMs are valuable assets for downstream applications. In this work, we consider learning from pre-trained DMs and transferring their knowledge to other generative models in a data-free fashion. Specifically, we propose a general framework called Diff-Instruct to instruct the training of arbitrary generative models as long as the generated samples are differentiable with respect to the model parameters. Our proposed Diff-Instruct is built on a rigorous mathematical foundation where the instruction process directly corresponds to minimizing a novel divergence we call Integral Kullback-Leibler (IKL) divergence. IKL is tailored for DMs by calculating the integral of the KL divergence along a diffusion process, which we show to be more robust in comparing distributions with misaligned supports. We also reveal non-trivial connections of our method to existing works such as DreamFusion \citep{poole2022dreamfusion}, and generative adversarial training. To demonstrate the effectiveness and universality of Diff-Instruct, we consider two scenarios: distilling pre-trained diffusion models and refining existing GAN models. The experiments on distilling pre-trained diffusion models show that Diff-Instruct results in state-of-the-art single-step diffusion-based models. The experiments on refining GAN models show that the Diff-Instruct can consistently improve the pre-trained generators of GAN models across various settings. Our official code is released through \url{https://github.com/pkulwj1994/diff_instruct}.

----

## [0] Characterization of Overfitting in Robust Multiclass Classification

**Authors**: *Jingyuan Xu, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f144ab9985c739a5091ec188a2688644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f144ab9985c739a5091ec188a2688644-Abstract-Conference.html)

**Abstract**:

This paper considers the following question: Given the number of classes m, the number of robust accuracy queries k, and the number of test examples in the dataset n, how much can adaptive algorithms robustly overfit the test dataset? We solve this problem by equivalently giving near-matching upper and lower bounds of the robust overfitting bias in multiclass classification problems.

----

## [0] Expanding Small-Scale Datasets with Guided Imagination

**Authors**: *Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, Jiashi Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html)

**Abstract**:

The power of DNNs relies heavily on the quantity and quality of training data. However, collecting and annotating data on a large scale is often expensive and time-consuming. To address this issue, we explore a new task, termed dataset expansion, aimed at expanding a ready-to-use small dataset by automatically creating new labeled samples. To this end, we present a Guided Imagination Framework (GIF) that leverages cutting-edge generative models like DALL-E2 and Stable Diffusion (SD) to "imagine" and create informative new data from the input seed data. Specifically, GIF conducts data imagination by optimizing the latent features of the seed data in the semantically meaningful space of the prior model, resulting in the creation of photo-realistic images with new content. To guide the imagination towards creating informative samples for model training, we introduce two key criteria, i.e., class-maintained information boosting and sample diversity promotion. These criteria are verified to be essential for effective dataset expansion: GIF-SD obtains 13.5% higher model accuracy on natural image datasets than unguided expansion with SD. With these essential criteria, GIF successfully expands small datasets in various scenarios, boosting model accuracy by 36.9% on average over six natural image datasets and by 13.5% on average over three medical datasets. The source code is available at https://github.com/Vanint/DatasetExpansion.

----

## [0] Parallel-mentoring for Offline Model-based Optimization

**Authors**: *Can Chen, Christopher Beckham, Zixuan Liu, Xue (Steve) Liu, Chris Pal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f189e7580acad0fc7fd45405817ddee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f189e7580acad0fc7fd45405817ddee3-Abstract-Conference.html)

**Abstract**:

We study offline model-based optimization to maximize a black-box objective function with a static dataset of designs and scores. These designs encompass a variety of domains, including materials, robots, DNA sequences, and proteins. A common approach trains a proxy on the static dataset and performs gradient ascent to obtain new designs. However, this often results in poor designs due to the proxy inaccuracies for out-of-distribution designs. Recent studies indicate that (a) gradient ascent with a mean ensemble of proxies generally outperforms simple gradient ascent, and (b) a trained proxy provides weak ranking supervision signals for design selection. Motivated by (a) and (b), we propose $\textit{parallel-mentoring}$ as an effective and novel method that facilitates mentoring among proxies, creating a more robust ensemble to mitigate the out-of-distribution issue. We focus on the three-proxy case in the main paper and our method consists of two modules. The first module, $\textit{voting-based pairwise supervision}$, operates on three parallel proxies and captures their ranking supervision signals as pairwise comparison labels. These labels are combined through majority voting to generate consensus labels, which incorporates ranking supervision signals from all proxies and enables mutual mentoring. Yet, label noise arises due to possible incorrect consensus. To alleviate this, we introduce an $\textit{adaptive soft-labeling}$ module with soft-labels initialized as consensus labels. Based on bi-level optimization, this module fine-tunes proxies in the inner level and learns more accurate labels in the outer level to adaptively mentor proxies, resulting in a more robust ensemble. Experiments validate the effectiveness of our method. Our code is available here.

----

## [0] Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

**Authors**: *Chih-Yu Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H. Lang, Duane S. Boning*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html)

**Abstract**:

Time series anomaly detection is challenging due to the complexity and variety of patterns that can occur. One major difficulty arises from modeling time-dependent relationships to find contextual anomalies while maintaining detection accuracy for point anomalies. In this paper, we propose a framework for unsupervised time series anomaly detection that utilizes point-based and sequence-based reconstruction models. The point-based model attempts to quantify point anomalies, and the sequence-based model attempts to quantify both point and contextual anomalies. Under the formulation that the observed time point is a two-stage deviated value from a nominal time point, we introduce a nominality score calculated from the ratio of a combined value of the reconstruction errors. We derive an induced anomaly score by further integrating the nominality score and anomaly score, then theoretically prove the superiority of the induced anomaly score over the original anomaly score under certain conditions. Extensive studies conducted on several public datasets show that the proposed framework outperforms most state-of-the-art baselines for time series anomaly detection.

----

## [0] Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

**Authors**: *Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, Zhendong Niu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html)

**Abstract**:

Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and superior performance. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FreTS.

----

## [0] Q-DM: An Efficient Low-bit Quantized Diffusion Model

**Authors**: *Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, Baochang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1ee1cca0721de55bb35cf28ab95e1b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1ee1cca0721de55bb35cf28ab95e1b4-Abstract-Conference.html)

**Abstract**:

Denoising diffusion generative models are capable of generating high-quality data, but suffers from the computation-costly generation process, due to a iterative noise estimation using full-precision  networks. As an intuitive solution, quantization can significantly reduce the computational and memory consumption by low-bit parameters and operations. However, low-bit noise estimation networks in diffusion models (DMs) remain unexplored yet and perform much worse than the full-precision counterparts as observed in our experimental studies. In this paper, we first identify that the bottlenecks of low-bit quantized DMs come from a large distribution oscillation  on activations  and accumulated quantization error caused by the multi-step denoising process. To address these issues, we first develop a Timestep-aware Quantization (TaQ) method and a Noise-estimating Mimicking (NeM) scheme for low-bit quantized DMs (Q-DM) to effectively eliminate such oscillation and accumulated error respectively, leading to well-performed low-bit DMs. In this way, we propose an efficient Q-DM to calculate low-bit DMs by considering both training and inference process in the same framework. We evaluate our methods on popular DDPM and DDIM models. Extensive experimental results show that our method achieves a much better performance than the prior arts. For example, the 4-bit Q-DM theoretically accelerates the 1000-step DDPM by 7.8x and achieves a FID score of 5.17, on the unconditional CIFAR-10 dataset.

----

## [0] Beyond Exponential Graph: Communication-Efficient Topologies for Decentralized Learning via Finite-time Convergence

**Authors**: *Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, Makoto Yamada*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f201b3f3d0f08c6ab46c36b9052c1b64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f201b3f3d0f08c6ab46c36b9052c1b64-Abstract-Conference.html)

**Abstract**:

Decentralized learning has recently been attracting increasing attention for its applications in parallel computation and privacy preservation. Many recent studies stated that the underlying network topology with a faster consensus rate (a.k.a. spectral gap) leads to a better convergence rate and accuracy for decentralized learning. However, a topology with a fast consensus rate, e.g., the exponential graph, generally has a large maximum degree, which incurs significant communication costs. Thus, seeking topologies with both a fast consensus rate and small maximum degree is important. In this study, we propose a novel topology combining both a fast consensus rate and small maximum degree called the Base-$\left(k+1\right)$ Graph. Unlike the existing topologies, the Base-$\left(k+1\right)$ Graph enables all nodes to reach the exact consensus after a finite number of iterations for any number of nodes and maximum degree $k$. Thanks to this favorable property, the Base-$\left(k+1\right)$ Graph endows Decentralized SGD (DSGD) with both a faster convergence rate and more communication efficiency than the exponential graph. We conducted experiments with various topologies, demonstrating that the Base-$\left(k+1\right)$ Graph enables various decentralized learning methods to achieve higher accuracy with better communication efficiency than the existing topologies. Our code is available at https://github.com/yukiTakezawa/BaseGraph.

----

## [0] Alternating Updates for Efficient Transformers

**Authors**: *Cenk Baykal, Dylan J. Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, Xin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html)

**Abstract**:

It has been well established that increasing scale in deep transformer networks leads to improved quality and performance. However, this increase in scale often comes with prohibitive increases in compute cost and inference latency. We introduce Alternating Updates (AltUp), a simple-to-implement method to increase a model's capacity without the computational burden. AltUp enables the widening of the learned representation, i.e., the token embedding, while only incurring a negligible increase in latency. AltUp achieves this by working on a subblock of the widened representation at each layer and using a predict-and-correct mechanism to update the inactivated blocks. We present extensions of AltUp, such as its applicability to the sequence dimension, and demonstrate how AltUp can be synergistically combined with existing approaches, such as Sparse Mixture-of-Experts models, to obtain efficient models with even higher capacity. Our experiments on benchmark transformer models and language tasks demonstrate the consistent effectiveness of AltUp on a diverse set of scenarios. Notably, on SuperGLUE and SQuAD benchmarks, AltUp enables up to $87\%$ speedup relative to the dense baselines at the same accuracy.

----

## [0] Interpretable Prototype-based Graph Information Bottleneck

**Authors**: *Sangwoo Seo, Sungwon Kim, Chanyoung Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html)

**Abstract**:

The success of Graph Neural Networks (GNNs) has led to a need for understanding their decision-making process and providing explanations for their predictions, which has given rise to explainable AI (XAI) that offers transparent explanations for black-box models. Recently, the use of prototypes has successfully improved the explainability of models by learning prototypes to imply training graphs that affect the prediction. However, these approaches tend to provide prototypes with excessive information from the entire graph, leading to the exclusion of key substructures or the inclusion of irrelevant substructures, which can limit both the interpretability and the performance of the model in downstream tasks. In this work, we propose a novel framework of explainable GNNs, called interpretable Prototype-based Graph Information Bottleneck (PGIB) that incorporates prototype learning within the information bottleneck framework to provide prototypes with the key subgraph from the input graph that is important for the model prediction. This is the first work that incorporates prototype learning into the process of identifying the key subgraphs that have a critical impact on the prediction performance. Extensive experiments, including qualitative analysis, demonstrate that PGIB outperforms state-of-the-art methods in terms of both prediction performance and explainability.

----

## [0] Self-Chained Image-Language Model for Video Localization and Question Answering

**Authors**: *Shoubin Yu, Jaemin Cho, Prateek Yadav, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html)

**Abstract**:

Recent studies have shown promising results on utilizing large pre-trained image-language models for video question answering. While these image-language models can efficiently bootstrap the representation learning of video-language models, they typically concatenate uniformly sampled video frames as visual inputs without explicit language-aware, temporal modeling. When only a portion of a video input is relevant to the language query, such uniform frame sampling can often lead to missing important visual cues. Although humans often find a video moment to focus on and rewind the moment to answer questions, training a query-aware video moment localizer often requires expensive annotations and high computational costs. To address this issue, we propose Self-Chained Video Localization-Answering (SeViLA), a novel framework that leverages a single image-language model (BLIP- 2) to tackle both temporal keyframe localization and question answering on videos. SeViLA framework consists of two modules: Localizer and Answerer, where both are parameter-efficiently fine-tuned from BLIP-2. We propose two ways of chaining these modules for cascaded inference and self-refinement. First, in the forward chain, the Localizer finds multiple language-aware keyframes in a video, which the Answerer uses to predict the answer. Second, in the reverse chain, the Answerer generates keyframe pseudo-labels to refine the Localizer, alleviating the need for expensive video moment localization annotations. Our SeViLA framework outperforms several strong baselines/previous works on five challenging video question answering and event prediction benchmarks, and achieves the state-of-the-art in both fine-tuning (NExT-QA and STAR) and zero-shot (NExT-QA, STAR, How2QA, and VLEP) settings. We show a comprehensive analysis of our framework, including the impact of Localizer, comparisons of Localizer with other temporal localization models, pre-training/self-refinement of Localizer, and varying the number of keyframes.

----

## [0] The Tunnel Effect: Building Data Representations in Deep Neural Networks

**Authors**: *Wojciech Masarczyk, Mateusz Ostaszewski, Ehsan Imani, Razvan Pascanu, Piotr Milos, Tomasz Trzcinski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f249db9ab5975586f36df46f8958c008-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f249db9ab5975586f36df46f8958c008-Abstract-Conference.html)

**Abstract**:

Deep neural networks are widely known for their remarkable effectiveness across various tasks, with the consensus that deeper networks implicitly learn more complex data representations. This paper shows that sufficiently deep networks trained for supervised image classification split into two distinct parts that contribute to the resulting data representations differently. The initial layers create linearly-separable representations, while the subsequent layers, which we refer to as \textit{the tunnel}, compress these representations and have a minimal impact on the overall performance. We explore the tunnel's behavior through comprehensive empirical studies, highlighting that it emerges early in the training process. Its depth depends on the relation between the network's capacity and task complexity. Furthermore, we show that the tunnel degrades out-of-distribution generalization and discuss its implications for continual learning.

----

## [0] Restart Sampling for Improving Generative Processes

**Authors**: *Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, Tommi S. Jaakkola*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html)

**Abstract**:

Generative processes that involve solving differential equations, such as diffusion models, frequently necessitate balancing speed and quality. ODE-based samplers are fast but plateau in performance while SDE-based samplers deliver higher sample quality at the cost of increased sampling time.  We attribute this difference to sampling errors: ODE-samplers involve smaller discretization errors while stochasticity in SDE contracts accumulated errors. Based on these findings, we propose a novel sampling algorithm called \textit{Restart} in order to better balance discretization errors and contraction. The sampling method alternates between adding substantial noise in additional forward steps and strictly following a backward ODE. Empirically, Restart sampler surpasses previous SDE and ODE samplers in both speed and accuracy. Restart not only outperforms the previous best SDE results, but also accelerates the sampling speed by 10-fold / 2-fold on CIFAR-10 / ImageNet $64{\times} 64$. In addition, it attains significantly better sample quality than ODE samplers within comparable sampling times. Moreover, Restart better balances text-image alignment/visual quality versus diversity than previous samplers in the large-scale text-to-image Stable Diffusion model pre-trained on LAION $512{\times} 512$.  Code is available at https://github.com/Newbeeer/diffusion_restart_sampling

----

## [0] Constructing Non-isotropic Gaussian Diffusion Model Using Isotropic Gaussian Diffusion Model for Image Editing

**Authors**: *Xi Yu, Xiang Gu, Haozhi Liu, Jian Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)

**Abstract**:

Score-based diffusion models (SBDMs) have achieved state-of-the-art results in image generation. In this paper, we propose a Non-isotropic Gaussian Diffusion Model (NGDM) for image editing, which requires editing the source image while preserving the image regions irrelevant to the editing task. We construct NGDM by adding independent Gaussian noises with different variances to different image pixels. Instead of specifically training the NGDM, we rectify the NGDM into an isotropic Gaussian diffusion model with different pixels having different total forward diffusion time. We propose to reverse the diffusion by designing a sampling method that starts at different time for different pixels for denoising to generate images using the pre-trained isotropic Gaussian diffusion model. Experimental results show that NGDM achieves state-of-the-art performance for image editing tasks, considering the trade-off between the fidelity to the source image and alignment with the desired editing target.

----

## [0] Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models

**Authors**: *Haonan Duan, Adam Dziedzic, Nicolas Papernot, Franziska Boenisch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) are excellent in-context learners. However, the sensitivity of data contained in prompts raises privacy concerns. Our work first shows that these concerns are valid: we instantiate a simple but highly effective membership inference attack against the data used to prompt LLMs. To address this vulnerability, one could forego prompting and resort to fine-tuning LLMs with known algorithms for private gradient descent. However, this comes at the expense of the practicality and efficiency offered by prompting. Therefore, we propose to privately learn to prompt. We first show that soft prompts can be obtained privately through gradient descent on downstream data. However, this is not the case for discrete prompts. Thus, we orchestrate a noisy vote among an ensemble of LLMs presented with different prompts, i.e., a flock of stochastic parrots. The vote privately transfers the flock's knowledge into a single public prompt. We show that LLMs prompted with our private algorithms closely match the non-private baselines. For example, using GPT3 as the base model, we achieve a downstream accuracy of 92.7% on the sst2 dataset with $(\varepsilon=0.147, \delta=10^{-6})$-differential privacy vs. 95.2% for the non-private baseline. Through our experiments, we also show that our prompt-based approach is easily deployed with existing commercial~APIs.

----

## [0] Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation

**Authors**: *Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2957e48240c1d90e62b303574871b47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2957e48240c1d90e62b303574871b47-Abstract-Conference.html)

**Abstract**:

Preparing training data for deep vision models is a labor-intensive task. To address this, generative models have emerged as an effective solution for generating synthetic data. While current generative models produce image-level category labels, we propose a novel method for generating pixel-level semantic segmentation labels using the text-to-image generative model Stable Diffusion (SD). By utilizing the text prompts, cross-attention, and self-attention of SD, we introduce three new techniques: class-prompt appending, class-prompt cross-attention, and self-attention exponentiation. These techniques enable us to generate segmentation maps corresponding to synthetic images. These maps serve as pseudo-labels for training semantic segmenters, eliminating the need for labor-intensive pixel-wise annotation. To account for the imperfections in our pseudo-labels, we incorporate uncertainty regions into the segmentation, allowing us to disregard loss from those regions. We conduct evaluations on two datasets, PASCAL VOC and MSCOCO, and our approach significantly outperforms concurrent work. Our benchmarks and code will be released at https://github.com/VinAIResearch/Dataset-Diffusion.

----

## [0] ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition

**Authors**: *Aashaka Desai, Lauren Berger, Fyodor Minakov, Nessa Milano, Chinmay Singh, Kriston Pumphrey, Richard E. Ladner, Hal Daumé III, Alex X. Lu, Naomi Caselli, Danielle Bragg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Sign languages are used as a primary language by approximately 70 million D/deaf people world-wide. However, most communication technologies operate in spoken and written languages, creating inequities in access. To help tackle this problem, we release ASL Citizen, the first crowdsourced Isolated Sign Language Recognition (ISLR) dataset, collected with consent and containing 83,399 videos for 2,731 distinct signs filmed by 52 signers in a variety of environments. We propose that this dataset be used for sign language dictionary retrieval for American Sign Language (ASL), where a user demonstrates a sign to their webcam to retrieve matching signs from a dictionary. We show that training supervised machine learning classifiers with our dataset advances the state-of-the-art on metrics relevant for dictionary retrieval, achieving 63\% accuracy and a recall-at-10 of 91\%, evaluated entirely on videos of users who are not present in the training or validation sets.

----

## [0] Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification

**Authors**: *Rui Wang, Peipei Li, Huaibo Huang, Chunshui Cao, Ran He, Zhaofeng He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html)

**Abstract**:

We present a novel language-driven ordering alignment method for ordinal classification. The labels in ordinal classification contain additional ordering relations, making them prone to overfitting when relying solely on training data. Recent developments in pre-trained vision-language models inspire us to leverage the rich ordinal priors in human language by converting the original task into a vision-language alignment task. Consequently, we propose L2RCLIP, which fully utilizes the language priors from two perspectives. First, we introduce a complementary prompt tuning technique called RankFormer, designed to enhance the ordering relation of original rank prompts. It employs token-level attention with residual-style prompt blending in the word embedding space. Second, to further incorporate language priors, we revisit the approximate bound optimization of vanilla cross-entropy loss and restructure it within the cross-modal embedding space. Consequently, we propose a cross-modal ordinal pairwise loss to refine the CLIP feature space, where texts and images maintain both semantic alignment and ordering alignment.  Extensive experiments on three ordinal classification tasks, including facial age estimation, historical color image (HCI) classification, and aesthetic assessment demonstrate its promising performance.

----

## [0] GAUCHE: A Library for Gaussian Processes in Chemistry

**Authors**: *Ryan-Rhys Griffiths, Leo Klarner, Henry B. Moss, Aditya Ravuri, Sang Truong, Yuanqi Du, Samuel Stanton, Gary Tom, Bojana Rankovic, Arian Rokkum Jamasb, Aryan Deshwal, Julius Schwartz, Austin Tripp, Gregory Kell, Simon Frieder, Anthony Bourached, Alex Chan, Jacob Moss, Chengzhi Guo, Johannes Peter Dürholt, Saudamini Chaurasia, Ji Won Park, Felix Strieth-Kalthoff, Alpha A. Lee, Bingqing Cheng, Alán Aspuru-Guzik, Philippe Schwaller, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html)

**Abstract**:

We introduce GAUCHE, an open-source library for GAUssian processes in CHEmistry. Gaussian processes have long been a cornerstone of probabilistic machine learning, affording particular advantages for uncertainty quantification and Bayesian optimisation. Extending Gaussian processes to molecular representations, however, necessitates kernels defined over structured inputs such as graphs, strings and bit vectors. By providing such kernels in a modular, robust and easy-to-use framework, we seek to enable expert chemists and materials scientists to make use of state-of-the-art black-box optimization techniques. Motivated by scenarios frequently encountered in practice, we showcase applications for GAUCHE in molecular discovery, chemical reaction optimisation and protein design. The codebase is made available at https://github.com/leojklarner/gauche.

----

## [0] Exponentially Convergent Algorithms for Supervised Matrix Factorization

**Authors**: *Joowon Lee, Hanbaek Lyu, Weixin Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2c80b3c9cf8102d38c4b21af25d9740-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2c80b3c9cf8102d38c4b21af25d9740-Abstract-Conference.html)

**Abstract**:

Supervised matrix factorization (SMF) is a classical machine learning method that simultaneously seeks feature extraction and classification tasks, which are not necessarily a priori aligned objectives. Our goal is to use SMF to learn low-rank latent factors that offer interpretable, data-reconstructive, and class-discriminative features, addressing challenges posed by high-dimensional data. Training SMF model involves solving a nonconvex and possibly constrained optimization with at least three blocks of parameters. Known algorithms are either heuristic or provide weak convergence guarantees for special cases. In this paper, we provide a novel framework that `lifts' SMF as a low-rank matrix estimation problem in a combined factor space and propose an efficient algorithm that provably converges exponentially fast to a global minimizer of the objective with arbitrary initialization under mild assumptions. Our framework applies to a wide range of SMF-type problems for multi-class classification with auxiliary features. To showcase an application, we demonstrate that our algorithm successfully identified well-known cancer-associated gene groups for various cancers.

----

## [0] InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion

**Authors**: *Fangzhou Lin, Yun Yue, Ziming Zhang, Songlin Hou, Kazunori D. Yamada, Vijaya Kolachalama, Venkatesh Saligrama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract-Conference.html)

**Abstract**:

A point cloud is a discrete set of data points sampled from a 3D geometric surface. Chamfer distance (CD) is a popular metric and training loss to measure the distances between point clouds, but also well known to be sensitive to outliers. To address this issue, in this paper we propose InfoCD, a novel contrastive Chamfer distance loss to learn to spread the matched points for better distribution alignments between point clouds as well as accounting for a surface similarity estimator. We show that minimizing InfoCD is equivalent to maximizing a lower bound of the mutual information between the underlying geometric surfaces represented by the point clouds, leading to a regularized CD metric which is robust and computationally efficient for deep learning. We conduct comprehensive experiments for point cloud completion using InfoCD and observe significant improvements consistently over all the popular baseline networks trained with CD-based losses, leading to new state-of-the-art results on several benchmark datasets. Demo code is available at https://github.com/Zhang-VISLab/NeurIPS2023-InfoCD.

----

## [0] Differentially Private Statistical Inference through β-Divergence One Posterior Sampling

**Authors**: *Jack Jewson, Sahra Ghalebikesabi, Chris C. Holmes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3024ea88cec9f45a411cf4d51ab649c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3024ea88cec9f45a411cf4d51ab649c-Abstract-Conference.html)

**Abstract**:

Differential privacy guarantees allow the results of a statistical analysis involving sensitive data to be released without compromising the privacy of any individual taking part. Achieving such guarantees generally requires the injection of noise, either directly into parameter estimates or into the estimation process. Instead of artificially introducing perturbations, sampling from Bayesian posterior distributions has been shown to be a special case of the exponential mechanism, producing consistent,and efficient private estimates without altering the data generative process. The application of current approaches has, however, been limited by their strong bounding assumptions which do not hold for basic models, such as simple linear regressors.To ameliorate this, we propose $\beta$D-Bayes, a posterior sampling  scheme from a generalised posterior targeting the minimisation of the $\beta$-divergence between the model and the data generating process. This provides private estimation that is generally applicable without requiring changes to the underlying model and consistently learns the data generating parameter. We show that $\beta$D-Bayes produces more precise inference estimation for the same privacy guarantees, and further facilitates differentially private estimation of complex classifiers, and continuous regression models such as neural networks, which goes beyond what has been currently possible with private posterior sampling.

----

## [0] Doubly Robust Augmented Transfer for Meta-Reinforcement Learning

**Authors**: *Yuankun Jiang, Nuowen Kan, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f31bf160569618084ba9bdc2a8de29d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f31bf160569618084ba9bdc2a8de29d0-Abstract-Conference.html)

**Abstract**:

Meta-reinforcement learning (Meta-RL), though enabling a fast adaptation to learn new skills by exploiting the common structure shared among different tasks, suffers performance degradation in the sparse-reward setting. Current hindsight-based sample transfer approaches can alleviate this issue by transferring relabeled trajectories from other tasks to a new task so as to provide informative experience for the target reward function, but are unfortunately constrained with the unrealistic assumption that tasks differ only in reward functions. In this paper, we propose a doubly robust augmented transfer (DRaT) approach, aiming at addressing the more general sparse reward meta-RL scenario with both dynamics mismatches and varying reward functions across tasks. Specifically, we design a doubly robust augmented estimator for efficient value-function evaluation, which tackles dynamics mismatches with the optimal importance weight of transition distributions achieved by minimizing the theoretically derived upper bound of mean squared error (MSE) between the estimated values of transferred samples and their true values in the target task. Due to its intractability, we then propose an interval-based approximation to this optimal importance weight, which is guaranteed to cover the optimum with a constrained and sample-independent upper bound on the MSE approximation error. Based on our theoretical findings, we finally develop a DRaT algorithm for transferring informative samples across tasks during the training of meta-RL. We implement DRaT on an off-policy meta-RL baseline, and empirically show that it significantly outperforms other hindsight-based approaches on various sparse-reward MuJoCo locomotion tasks with varying dynamics and reward functions.

----

## [0] Evaluating Open-QA Evaluation

**Authors**: *Cunxiang Wang, Sirui Cheng, Qipeng Guo, Yuanhao Yue, Bowen Ding, Zhikun Xu, Yidong Wang, Xiangkun Hu, Zheng Zhang, Yue Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f323d594aa5d2c68154433a131c07959-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f323d594aa5d2c68154433a131c07959-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This study focuses on the evaluation of the Open Question Answering (Open-QA) task, which can directly estimate the factuality of large language models (LLMs). Current automatic evaluation methods have shown limitations, indicating that human evaluation still remains the most reliable approach. We introduce a new task, QA Evaluation (QA-Eval) and the corresponding dataset EVOUNA, designed to assess the accuracy of AI-generated answers in relation to standard answers within Open-QA. Our evaluation of these methods utilizes human-annotated results to measure their performance. Specifically, the work investigates methods that show high correlation with human evaluations, deeming them more reliable. We also discuss the pitfalls of current methods and methods to improve LLM-based evaluators. We believe this new QA-Eval task and corresponding dataset EVOUNA will facilitate the development of more effective automatic evaluation tools and prove valuable for future research in this area. All resources are available at https://github.com/wangcunxiang/QA-Eval and it is under the Apache-2.0 License.

----

## [0] Efficiently incorporating quintuple interactions into geometric deep learning force fields

**Authors**: *Zun Wang, Guoqing Liu, Yichi Zhou, Tong Wang, Bin Shao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f32b13bfc384b3b1d52d675b05f2bece-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f32b13bfc384b3b1d52d675b05f2bece-Abstract-Conference.html)

**Abstract**:

Machine learning force fields (MLFFs) have instigated a groundbreaking shift in molecular dynamics (MD) simulations across a wide range of fields, such as physics, chemistry, biology, and materials science. Incorporating higher order many-body interactions can enhance the expressiveness and accuracy of models. Recent models have achieved this by explicitly including up to four-body interactions. However, five-body interactions, which have relevance in various fields, are still challenging to incorporate efficiently into MLFFs. In this work, we propose the quintuple network (QuinNet), an end-to-end graph neural network that efficiently expresses many-body interactions up to five-body interactions with \emph{ab initio} accuracy. By analyzing the topology of diverse many-body interactions, we design the model architecture to efficiently and explicitly represent these interactions. We evaluate QuinNet on public datasets of small molecules, such as MD17 and its revised version, and show that it is compatible with other state-of-the-art models on these benchmarks. Moreover, QuinNet surpasses many leading models on larger and more complex molecular systems, such as MD22 and Chignolin, without increasing the computational complexity. We also use QuinNet as a force field for molecular dynamics (MD) simulations to demonstrate its accuracy and stability, and conduct an ablation study to elucidate the significance of five-body interactions. We open source our implementation at https://github.com/Zun-Wang/QuinNet.

----

## [0] Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning

**Authors**: *Stefan Stojanovic, Yassir Jedra, Alexandre Proutière*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f334c3375bd3744e98a0ca8eaa2403b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f334c3375bd3744e98a0ca8eaa2403b0-Abstract-Conference.html)

**Abstract**:

We study matrix estimation problems arising in reinforcement learning with low-rank structure. In low-rank bandits, the matrix to be recovered specifies the expected arm rewards, and for low-rank Markov Decision Processes (MDPs), it characterizes the transition kernel of the MDP. In both cases, each entry of the matrix carries important information, and we seek estimation methods with low entry-wise prediction error. Importantly, these methods further need to accommodate for inherent correlations in the available data (e.g. for MDPs, the data consists of system trajectories). We investigate the performance of  simple spectral-based matrix estimation approaches: we show that they efficiently recover the singular subspaces of the matrix and exhibit nearly-minimal entry-wise prediction error. These new results on low-rank matrix estimation make it possible to devise reinforcement learning algorithms that fully exploit the underlying low-rank structure. We provide two examples of such algorithms: a regret minimization algorithm for low-rank bandit problems, and a best policy identification algorithm for low-rank MDPs. Both algorithms yield state-of-the-art performance guarantees.

----

## [0] Function Space Bayesian Pseudocoreset for Bayesian Neural Networks

**Authors**: *Balhae Kim, Hyungi Lee, Juho Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f36a180277bd3d5781dc02245f9d5f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f36a180277bd3d5781dc02245f9d5f52-Abstract-Conference.html)

**Abstract**:

A Bayesian pseudocoreset is a compact synthetic dataset summarizing essential information of a large-scale dataset and thus can be used as a proxy dataset for scalable Bayesian inference. Typically, a Bayesian pseudocoreset is constructed by minimizing a divergence measure between the posterior conditioning on the pseudocoreset and the posterior conditioning on the full dataset. However, evaluating the divergence can be challenging, particularly for the models like deep neural networks having high-dimensional parameters. In this paper, we propose a novel Bayesian pseudocoreset construction method that operates on a function space. Unlike previous methods, which construct and match the coreset and full data posteriors in the space of model parameters (weights), our method constructs variational approximations to the coreset posterior on a function space and matches it to the full data posterior in the function space. By working directly on the function space, our method could bypass several challenges that may arise when working on a weight space, including limited scalability and multi-modality issue. Through various experiments, we demonstrate that the Bayesian pseudocoresets constructed from our method enjoys enhanced uncertainty quantification and better robustness across various model architectures.

----

## [0] One-step differentiation of iterative algorithms

**Authors**: *Jérôme Bolte, Edouard Pauwels, Samuel Vaiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3716db40060004d0629d4051b2c57ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3716db40060004d0629d4051b2c57ab-Abstract-Conference.html)

**Abstract**:

In appropriate frameworks, automatic differentiation is transparent to the user, at the cost of being a significant computational burden when the number of operations is large. For iterative algorithms, implicit differentiation alleviates this issue but requires custom implementation of Jacobian evaluation. In this paper, we study one-step differentiation, also known as Jacobian-free backpropagation, a method as easy as automatic differentiation and as performant as implicit differentiation for fast algorithms (e.g. superlinear optimization methods). We provide a complete theoretical approximation analysis with specific examples (Newton's method, gradient descent) along with its consequences in bilevel optimization. Several numerical examples illustrate the well-foundness of the one-step estimator.

----

## [0] Adaptive Principal Component Regression with Applications to Panel Data

**Authors**: *Anish Agarwal, Keegan Harris, Justin Whitehouse, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f37265d7493377170a3b4ba91823119a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f37265d7493377170a3b4ba91823119a-Abstract-Conference.html)

**Abstract**:

Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for counterfactual estimation of unit-specific treatment effects in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic interventions framework where data is collected via an adaptive intervention assignment policy.

----

## [0] VisAlign: Dataset for Measuring the Alignment between AI and Humans in Visual Perception

**Authors**: *Jiyoung Lee, Seungho Kim, Seunghyun Won, Joonseok Lee, Marzyeh Ghassemi, James Thorne, Jaeseok Choi, O.-Kil Kwon, Edward Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f37aba0f53fdb59f53254fe9098b2177-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f37aba0f53fdb59f53254fe9098b2177-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

AI alignment refers to models acting towards human-intended goals, preferences, or ethical principles. Analyzing the similarity between models and humans can be a proxy measure for ensuring AI safety. In this paper, we focus on the models' visual perception alignment with humans, further referred to as AI-human visual alignment. Specifically, we propose a new dataset for measuring AI-human visual alignment in terms of image classification. In order to evaluate AI-human visual alignment, a dataset should encompass samples with various scenarios and have gold human perception labels. Our dataset consists of three groups of samples, namely Must-Act (i.e., Must-Classify), Must-Abstain, and Uncertain, based on the quantity and clarity of visual information in an image and further divided into eight categories. All samples have a gold human perception label; even Uncertain (e.g., severely blurry) sample labels were obtained via crowd-sourcing. The validity of our dataset is verified by sampling theory, statistical theories related to survey design, and experts in the related fields. Using our dataset, we analyze the visual alignment and reliability of five popular visual perception models and seven abstention methods. Our code and data is available at https://github.com/jiyounglee-0523/VisAlign.

----

## [0] The Best of Both Worlds in Network Population Games: Reaching Consensus and Convergence to Equilibrium

**Authors**: *Shuyue Hu, Harold Soh, Georgios Piliouras*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f39931608cdc52d7d9f8ba7003af9136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f39931608cdc52d7d9f8ba7003af9136-Abstract-Conference.html)

**Abstract**:

Reaching consensus and convergence to equilibrium are two major challenges of multi-agent systems. Although each has attracted significant attention, relatively few studies address both challenges at the same time. This paper examines the connection between the notions of consensus and equilibrium in a multi-agent system where multiple interacting sub-populations coexist. We argue that consensus can be seen as an intricate component of intra-population stability, whereas equilibrium can be seen as encoding inter-population stability. We show that smooth fictitious play, a well-known learning model in game theory, can achieve both consensus and convergence to equilibrium in diverse multi-agent settings. Moreover, we show that the consensus formation process plays a crucial role in the seminal thorny problem of equilibrium selection in multi-agent learning.

----

## [0] L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors

**Authors**: *Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html)

**Abstract**:

Language-based colorization produces plausible and visually pleasing colors under the guidance of user-friendly natural language descriptions. Previous methods implicitly assume that users provide comprehensive color descriptions for most of the objects in the image, which leads to suboptimal performance. In this paper, we propose a unified model to perform language-based colorization with any-level descriptions. We leverage the pretrained cross-modality generative model for its robust language understanding and rich color priors to handle the inherent ambiguity of any-level descriptions. We further design modules to align with input conditions to preserve local spatial structures and prevent the ghosting effect. With the proposed novel sampling strategy, our model achieves instance-aware colorization in diverse and complex scenarios. Extensive experimental results demonstrate our advantages of effectively handling any-level descriptions and outperforming both language-based and automatic colorization methods. The code and pretrained modelsare available at: https://github.com/changzheng123/L-CAD.

----

## [0] Convolutional Neural Operators for robust and accurate learning of PDEs

**Authors**: *Bogdan Raonic, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html)

**Abstract**:

Although very successfully used in conventional machine learning, convolution based neural network architectures -- believed to be inconsistent in function space -- have been largely ignored in the context of learning solution operators of PDEs. Here, we present novel adaptations for convolutional neural networks to demonstrate that they are indeed able to process functions as inputs and outputs. The resulting architecture, termed as convolutional neural operators (CNOs), is designed specifically to preserve its underlying continuous nature, even when implemented in a discretized form on a computer. We prove a universality theorem to show that CNOs can approximate operators arising in PDEs to desired accuracy. CNOs are tested on a novel suite of benchmarks, encompassing a diverse set of PDEs with multi-scale solutions and are observed to significantly outperform baselines, paving the way for an alternative framework for robust and accurate operator learning.

----

## [0] Neural Image Compression: Generalization, Robustness, and Spectral Biases

**Authors**: *Kelsey Lieberman, James Diffenderfer, Charles Godfrey, Bhavya Kailkhura*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html)

**Abstract**:

Recent advances in neural image compression (NIC) have produced models that are starting to outperform classic codecs. While this has led to growing excitement about using NIC in real-world applications, the successful adoption of any machine learning system in the wild requires it to generalize (and be robust) to unseen distribution shifts at deployment. Unfortunately, current research lacks comprehensive datasets and informative tools to evaluate and understand NIC performance in real-world settings. To bridge this crucial gap, first, this paper presents a comprehensive benchmark suite to evaluate the out-of-distribution (OOD) performance of image compression methods. Specifically, we provide CLIC-C and Kodak-C by introducing 15 corruptions to the popular CLIC and Kodak benchmarks. Next, we propose spectrally-inspired inspection tools to gain deeper insight into errors introduced by image compression methods as well as their OOD performance. We then carry out a detailed performance comparison of several classic codecs and NIC variants, revealing intriguing findings that challenge our current understanding of the strengths and limitations of NIC. Finally, we corroborate our empirical findings with theoretical analysis, providing an in-depth view of the OOD performance of NIC and its dependence on the spectral properties of the data. Our benchmarks, spectral inspection tools, and findings provide a crucial bridge to the real-world adoption of NIC. We hope that our work will propel future efforts in designing robust and generalizable NIC methods. Code and data will be made available at https://github.com/klieberman/ood_nic.

----

## [0] Estimating Koopman operators with sketching to provably learn large scale dynamical systems

**Authors**: *Giacomo Meanti, Antoine Chatalic, Vladimir Kostic, Pietro Novelli, Massimiliano Pontil, Lorenzo Rosasco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3d1e34a15c0af0954ae36a7f811c754-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3d1e34a15c0af0954ae36a7f811c754-Abstract-Conference.html)

**Abstract**:

The theory of Koopman operators allows to deploy non-parametric machine learning algorithms to predict and analyze complex dynamical systems.Estimators such as principal component regression (PCR) or reduced rank regression (RRR) in kernel spaces can be shown to provably learn Koopman operators from finite empirical observations of the system's time evolution. Scaling these approaches to very long trajectories is a challenge and requires introducing suitable approximations to make computations feasible. In this paper, we boost the efficiency of different kernel-based Koopman operator estimators using random projections (sketching).We derive, implement and test the new ``sketched'' estimators with extensive experiments on synthetic and large-scale molecular dynamics datasets. Further, we establish non asymptotic error bounds giving a sharp characterization of the trade-offs between statistical learning rates and computational efficiency.Our empirical and theoretical analysis shows that the proposed estimators provide a sound and efficient way to learn large scale dynamical systems.In particular our experiments indicate that the proposed estimators retain the same accuracy of PCR or RRR, while being much faster.

----

## [0] Self-Adaptive Motion Tracking against On-body Displacement of Flexible Sensors

**Authors**: *Chengxu Zuo, Jiawei Fang, Shihui Guo, Yipeng Qin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html)

**Abstract**:

Flexible sensors are promising for ubiquitous sensing of human status due to their flexibility and easy integration as wearable systems. However, on-body displacement of sensors is inevitable since the device cannot be firmly worn at a fixed position across different sessions. This displacement issue causes complicated patterns and significant challenges to subsequent machine learning algorithms. Our work proposes a novel self-adaptive motion tracking network to address this challenge. Our network consists of three novel components: i) a light-weight learnable Affine Transformation layer whose parameters can be tuned to efficiently adapt to unknown displacements; ii) a Fourier-encoded LSTM network for better pattern identification; iii) a novel sequence discrepancy loss equipped with auxiliary regressors for unsupervised tuning of Affine Transformation parameters.

----

## [0] Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning

**Authors**: *Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html)

**Abstract**:

Offline multi-agent reinforcement learning is challenging due to the coupling effect of both distribution shift issue common in offline setting and the high dimension issue common in multi-agent setting, making the action out-of-distribution (OOD) and value overestimation phenomenon excessively severe. To mitigate this problem, we propose a novel multi-agent offline RL algorithm, named CounterFactual Conservative Q-Learning (CFCQL) to conduct conservative value estimation. Rather than regarding all the agents as a high dimensional single one and directly applying single agent conservative methods to it, CFCQL calculates conservative regularization for each agent separately in a counterfactual way and then linearly combines them to realize an overall conservative value estimation. We prove that it still enjoys the underestimation property and the performance guarantee as those single agent conservative methods do, but the induced regularization and safe policy improvement bound are independent of the agent number, which is therefore theoretically superior to the direct treatment referred to above, especially when the agent number is large. We further conduct experiments on four environments including both discrete and continuous action settings on both existing and our man-made datasets, demonstrating that CFCQL outperforms existing methods on most datasets and even with a remarkable margin on some of them.

----

## [0] Black-Box Differential Privacy for Interactive ML

**Authors**: *Haim Kaplan, Yishay Mansour, Shay Moran, Kobbi Nissim, Uri Stemmer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f418594e90047a10f4c158f70d6701cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f418594e90047a10f4c158f70d6701cc-Abstract-Conference.html)

**Abstract**:

In this work we revisit an interactive variant of joint differential privacy, recently introduced by Naor et al. [2023], and generalize it towards handling online processes in which existing privacy definitions seem too restrictive. We study basic properties of this definition and demonstrate that it satisfies (suitable variants) of group privacy, composition, and post processing.In order to demonstrate the advantages of this privacy definition compared to traditional forms of differential privacy,we consider the basic setting of online classification. We show that any (possibly non-private) learning rule can be effectively transformed to a private learning rule with only a polynomial overhead in the mistake bound. This demonstrates a stark difference with traditional forms of differential privacy, such as the one studied  by Golowich and Livni [2021], where only a double exponential overhead in the mistake bound is known (via an information theoretic upper bound).

----

## [0] Mnemosyne: Learning to Train Transformers with Transformers

**Authors**: *Deepali Jain, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Sumeet Singh, Vikas Sindhwani, Tingnan Zhang, Jie Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f41b6e5af73421e46ceed9cb036e72e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f41b6e5af73421e46ceed9cb036e72e7-Abstract-Conference.html)

**Abstract**:

In this work, we propose a new class of learnable optimizers, called Mnemosyne. It is based on the novel spatio-temporal low-rank implicit attention Transformers that can learn to train entire neural network architectures, including other Transformers, without any task-specific optimizer tuning. We show that Mnemosyne: (a) outperforms popular LSTM optimizers (also with new feature engineering to mitigate catastrophic forgetting of LSTMs), (b) can successfully train Transformers while using simple meta-training strategies that require minimal computational resources, (c) matches accuracy-wise SOTA hand-designed optimizers with carefully tuned hyper-parameters (often producing top performing models). Furthermore, Mnemosyne provides space complexity comparable to that of its hand-designed first-order counterparts, which allows it to scale to training larger sets of parameters. We conduct an extensive empirical evaluation of Mnemosyne on: (a) fine-tuning a wide range of Vision Transformers (ViTs) from medium-size architectures to massive ViT-Hs (36 layers, 16 heads), (b) pre-training BERT models and (c) soft prompt-tuning large 11B+ T5XXL models. We complement our results with a comprehensive theoretical analysis of the compact associative memory used by Mnemosyne which we believe was never done before.

----

## [0] M2Hub: Unlocking the Potential of Machine Learning for Materials Discovery

**Authors**: *Yuanqi Du, Yingheng Wang, Yining Huang, Jianan Canal Li, Yanqiao Zhu, Tian Xie, Chenru Duan, John M. Gregoire, Carla Pedro Gomes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f43380ca3f86cd989f3269583c3c8b55-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f43380ca3f86cd989f3269583c3c8b55-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce M$^2$Hub, a toolkit for advancing machine learning in materials discovery. Machine learning has achieved remarkable progress in modeling molecular structures, especially biomolecules for drug discovery. However, the development of machine learning approaches for modeling materials structures lag behind, which is partly due to the lack of an integrated platform that enables access to diverse tasks for materials discovery. To bridge this gap, M$^2$Hub will enable easy access to materials discovery tasks, datasets, machine learning methods, evaluations, and benchmark results that cover the entire workflow. Specifically, the first release of M$^2$Hub focuses on three key stages in materials discovery: virtual screening, inverse design, and molecular simulation, including 9 datasets that covers 6 types of materials with 56 tasks across 8 types of material properties. We further provide 2 synthetic datasets for the purpose of generative tasks on materials. In addition to random data splits, we also provide 3 additional data partitions to reflect the real-world materials discovery scenarios. State-of-the-art machine learning methods (including those are suitable for materials structures but never compared in the literature) are benchmarked on representative tasks. Our codes and library are publicly available at \url{https://github.com/yuanqidu/M2Hub}.

----

## [0] PoET: A generative model of protein families as sequences-of-sequences

**Authors**: *Timothy F. Truong Jr., Tristan Bepler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html)

**Abstract**:

Generative protein language models are a natural way to design new proteins with desired functions. However, current models are either difficult to direct to produce a protein from a specific family of interest, or must be trained on a large multiple sequence alignment (MSA) from the specific family of interest, making them unable to benefit from transfer learning across families. To address this, we propose Protein Evolutionary Transformer (PoET), an autoregressive generative model of whole protein families that learns to generate sets of related proteins as sequences-of-sequences across tens of millions of natural protein sequence clusters. PoET can be used as a retrieval-augmented language model to generate and score arbitrary modifications conditioned on any protein family of interest, and can extrapolate from short context lengths to generalize well even for small families. This is enabled by a unique Transformer layer; we model tokens sequentially within sequences while attending between sequences order invariantly, allowing PoET to scale to context lengths beyond those used during training. In extensive experiments on deep mutational scanning datasets, we show that PoET outperforms existing protein language models and evolutionary sequence models for variant function prediction across proteins of all MSA depths. We also demonstrate PoET's ability to controllably generate new protein sequences.

----

## [0] BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization

**Authors**: *Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract-Conference.html)

**Abstract**:

Despite the success of neural-based combinatorial optimization methods for end-to-end heuristic learning, out-of-distribution generalization remains a challenge. In this paper, we present a novel formulation of Combinatorial Optimization Problems (COPs) as Markov Decision Processes (MDPs) that effectively leverages common symmetries of COPs to improve out-of-distribution robustness. Starting from a direct MDP formulation of a constructive method, we introduce a generic way to reduce the state space, based on Bisimulation Quotienting (BQ) in MDPs. Then, for COPs with a recursive nature, we specialize the bisimulation and show how the reduced state exploits the symmetries of these problems and facilitates MDP solving. Our approach is principled and we prove that an optimal policy for the proposed BQ-MDP actually solves the associated COPs. We illustrate our approach on five classical problems: the Euclidean and Asymmetric Traveling Salesman, Capacitated Vehicle Routing, Orienteering and Knapsack Problems. Furthermore, for each problem, we introduce a simple attention-based policy network for the BQ-MDPs, which we train by imitation of (near) optimal solutions of small instances from a single distribution. We obtain new state-of-the-art results for the five COPs on both synthetic and realistic benchmarks. Notably, in contrast to most existing neural approaches, our learned policies show excellent generalization performance to much larger instances than seen during training, without any additional search procedure. Our code is available at: link.

----

## [0] Turbulence in Focus: Benchmarking Scaling Behavior of 3D Volumetric Super-Resolution with BLASTNet 20 Data

**Authors**: *Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, Alexei Y. Poludnenko, Matthias Ihme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f458af2455b1e12608c2a16c308d663d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f458af2455b1e12608c2a16c308d663d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Analysis of compressible turbulent flows is essential for applications related to propulsion, energy generation, and the environment. Here, we present BLASTNet 2.0, a 2.2 TB network-of-datasets containing 744 full-domain samples from 34 high-fidelity direct numerical simulations, which addresses the current limited availability of 3D high-fidelity reacting and non-reacting compressible turbulent flow simulation data.  With this data, we benchmark a total of 49 variations of five deep learning approaches for 3D super-resolution - which can be applied for improving scientific imaging, simulations, turbulence models, as well as in computer vision  applications. We perform neural scaling analysis on these models to examine the performance of different machine learning (ML) approaches, including two scientific ML techniques. We demonstrate that (i) predictive performance can scale with model size and cost, (ii) architecture matters significantly, especially for smaller models, and (iii) the benefits of physics-based losses can persist with increasing model size. The outcomes of this benchmark study are anticipated to offer insights that can aid the design of 3D super-resolution models, especially for turbulence models, while this data is expected to foster ML methods for a broad range of flow physics applications. This data is publicly available with download links and browsing tools consolidated at https://blastnet.github.io.

----

## [0] Neural Functional Transformers

**Authors**: *Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J. Zico Kolter, Chelsea Finn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4757db82a02eea015670ecca605d5cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4757db82a02eea015670ecca605d5cc-Abstract-Conference.html)

**Abstract**:

The recent success of neural networks as implicit representation of data has driven growing interest in neural functionals: models that can process other neural networks as input by operating directly over their weight spaces. Nevertheless, constructing expressive and efficient neural functional architectures that can handle high-dimensional weight-space objects remains challenging. This paper uses the attention mechanism to define a novel set of permutation equivariant weight-space layers and composes them into deep equivariant models called neural functional Transformers (NFTs). NFTs respect weight-space permutation symmetries while incorporating the advantages of attention, which have exhibited remarkable success across multiple domains. In experiments processing the weights of feedforward MLPs and CNNs, we find that NFTs match or exceed the performance of prior weight-space methods. We also leverage NFTs to develop Inr2Array, a novel method for computing permutation invariant latent representations from the weights of implicit neural representations (INRs). Our proposed method improves INR classification accuracy by up to $+17\\%$ over existing methods. We provide an implementation of our layers at https://github.com/AllanYangZhou/nfn.

----

## [0] LinkerNet: Fragment Poses and Linker Co-Design with 3D Equivariant Diffusion

**Authors**: *Jiaqi Guan, Xingang Peng, Peiqi Jiang, Yunan Luo, Jian Peng, Jianzhu Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html)

**Abstract**:

Targeted protein degradation techniques, such as PROteolysis TArgeting Chimeras (PROTACs), have emerged as powerful tools for selectively removing disease-causing proteins. One challenging problem in this field is designing a linker to connect different molecular fragments to form a stable drug-candidate molecule. Existing models for linker design assume that the relative positions of the fragments are known, which may not be the case in real scenarios. In this work, we address a more general problem where the poses of the fragments are unknown in 3D space. We develop a 3D equivariant diffusion model that jointly learns the generative process of both fragment poses and the 3D structure of the linker. By viewing fragments as rigid bodies, we design a fragment pose prediction module inspired by the Newton-Euler equations in rigid body mechanics. Empirical studies on ZINC and PROTAC-DB datasets demonstrate that our model can generate chemically valid, synthetically-accessible,  and low-energy molecules under both unconstrained and constrained generation settings.

----

## [0] One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning

**Authors**: *Marc Rigter, Bruno Lacerda, Nick Hawes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f49287371916715b9209fa41a275851e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f49287371916715b9209fa41a275851e-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) is suitable for safety-critical domains where online exploration is not feasible. In such domains, decision-making should take into consideration the risk of catastrophic outcomes. In other words, decision-making should be risk-averse. An additional challenge of offline RL is avoiding distributional shift, i.e. ensuring that  state-action pairs visited by the policy remain near those in the dataset. Previous offline RL algorithms that consider risk combine offline RL techniques (to avoid distributional shift), with risk-sensitive RL algorithms (to achieve risk-aversion). In this work, we propose risk-aversion as a mechanism to jointly address both of these issues. We propose a model-based approach, and use an ensemble of models to estimate epistemic uncertainty, in addition to aleatoric uncertainty. We train a policy that is risk-averse, and avoids high uncertainty actions. Risk-aversion to epistemic uncertainty prevents distributional shift, as areas not covered by the dataset have high epistemic uncertainty. Risk-aversion to aleatoric uncertainty discourages actions that are risky due to environment stochasticity. Thus, by considering epistemic uncertainty via a model ensemble and introducing risk-aversion, our algorithm (1R2R) avoids distributional shift in addition to achieving risk-aversion to aleatoric risk. Our experiments show that 1R2R achieves strong performance on deterministic benchmarks, and outperforms existing approaches for risk-sensitive objectives in stochastic domains.

----

## [0] Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture

**Authors**: *Daniel Y. Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin W. Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f498c1ce6bff52eb04febf87438dd84b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f498c1ce6bff52eb04febf87438dd84b-Abstract-Conference.html)

**Abstract**:

Machine learning models are increasingly being scaled in both sequence length and model dimension to reach longer contexts and better performance. However, existing architectures such as Transformers scale quadratically along both these axes. We ask: are there performant architectures that can scale sub-quadratically along sequence length and model dimension? We introduce Monarch Mixer (M2), a new architecture that uses the same sub-quadratic primitive along both sequence length and model dimension: Monarch matrices, a simple class of expressive structured matrices that captures many linear transforms, achieves high hardware efficiency on GPUs, and scales sub-quadratically. As a proof of concept, we explore the performance of M2 in three domains: non-causal BERT-style language modeling, ViT-style image classification, and causal GPT-style language modeling. For non-causal BERT-style modeling, M2 matches BERT-base and BERT-large in downstream GLUE quality with up to 27% fewer parameters, and achieves up to 9.1$\times$ higher throughput at sequence length 4K. On ImageNet, M2 outperforms ViT-b by 1% in accuracy, with only half the parameters. Causal GPT-style models introduce a technical challenge: enforcing causality via masking introduces a quadratic bottleneck. To alleviate this bottleneck, we develop a novel theoretical view of Monarch matrices based on multivariate polynomial evaluation and interpolation, which lets us parameterize M2 to be causal while remaining sub-quadratic. Using this parameterization, M2 matches GPT-style Transformers at 360M parameters in pretraining perplexity on The PILE—showing for the first time that it may be possible to match Transformer quality without attention or MLPs.

----

## [0] Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes

**Authors**: *Yizi Zhang, Tianxiao He, Julien Boussard, Charles Windolf, Olivier Winter, Eric Trautmann, Noam Roth, Hailey Barrell, Mark Churchland, Nicholas A Steinmetz, Erdem Varol, Cole L. Hurwitz, Liam Paninski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract-Conference.html)

**Abstract**:

Neural decoding and its applications to brain computer interfaces (BCI) are essential for understanding the association between neural activity and behavior. A prerequisite for many decoding approaches is spike sorting, the assignment of action potentials (spikes) to individual neurons. Current spike sorting algorithms, however, can be inaccurate and do not properly model uncertainty of spike assignments, therefore discarding information that could potentially improve decoding performance. Recent advances in high-density probes (e.g., Neuropixels) and computational methods now allow for extracting a rich set of spike features from unsorted data; these features can in turn be used to directly decode behavioral correlates. To this end, we propose a spike sorting-free decoding method that directly models the distribution of extracted spike features using a mixture of Gaussians (MoG) encoding the uncertainty of spike assignments, without aiming to solve the spike clustering problem explicitly. We allow the mixing proportion of the MoG to change over time in response to the behavior and develop variational inference methods to fit the resulting model and to perform decoding. We benchmark our method with an extensive suite of recordings from different animals and probe geometries, demonstrating that our proposed decoder can consistently outperform current methods based on thresholding (i.e. multi-unit activity) and spike sorting. Open source code is available at https://github.com/yzhang511/density_decoding.

----

## [0] SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models

**Authors**: *Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhi-Ming Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4a6806490d31216a3ba667eb240c897-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4a6806490d31216a3ba667eb240c897-Abstract-Conference.html)

**Abstract**:

Diffusion Probabilistic Models (DPMs) have achieved considerable success in generation tasks. As sampling from DPMs is equivalent to solving diffusion SDE or ODE which is time-consuming, numerous fast sampling methods built upon improved differential equation solvers are proposed. The majority of such techniques consider solving the diffusion ODE due to its superior efficiency. However, stochastic sampling could offer additional advantages in generating diverse and high-quality data. In this work, we engage in a comprehensive analysis of stochastic sampling from two aspects: variance-controlled diffusion SDE and linear multi-step SDE solver. Based on our analysis, we propose SA-Solver, which is an improved efficient stochastic Adams method for solving diffusion SDE to generate data with high quality. Our experiments show that SA-Solver achieves: 1) improved or comparable performance compared with the existing state-of-the-art (SOTA) sampling methods for few-step sampling; 2) SOTA FID on substantial benchmark datasets under a suitable number of function evaluations (NFEs).

----

## [0] Social Motion Prediction with Cognitive Hierarchies

**Authors**: *Wentao Zhu, Jason Qin, Yuke Lou, Hang Ye, Xiaoxuan Ma, Hai Ci, Yizhou Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b52b45a677d855dee0ca9ba1ddf638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b52b45a677d855dee0ca9ba1ddf638-Abstract-Conference.html)

**Abstract**:

Humans exhibit a remarkable capacity for anticipating the actions of others and planning their own actions accordingly. In this study, we strive to replicate this ability by addressing the social motion prediction problem. We introduce a new benchmark, a novel formulation, and a cognition-inspired framework. We present Wusi, a 3D multi-person motion dataset under the context of team sports, which features intense and strategic human interactions and diverse pose distributions. By reformulating the problem from a multi-agent reinforcement learning perspective, we incorporate behavioral cloning and generative adversarial imitation learning to boost learning efficiency and generalization. Furthermore, we take into account the cognitive aspects of the human social action planning process and develop a cognitive hierarchy framework to predict strategic human social interactions. We conduct comprehensive experiments to validate the effectiveness of our proposed dataset and approach.

----

## [0] Unbounded Differentially Private Quantile and Maximum Estimation

**Authors**: *David Durfee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b6ef2a78684dca2fb3f1c09372e041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b6ef2a78684dca2fb3f1c09372e041-Abstract-Conference.html)

**Abstract**:

In this work we consider the problem of differentially private computation ofquantiles for the data, especially the highest quantiles such as maximum, butwith an unbounded range for the dataset. We show that this can be doneefficiently through a simple invocation of $\texttt{AboveThreshold}$, asubroutine that is iteratively called in the fundamental Sparse VectorTechnique, even when there is no upper bound on the data. In particular, weshow that this procedure can give more accurate and robust estimates on thehighest quantiles with applications towards clipping that is essential fordifferentially private sum and mean estimation. In addition, we show how twoinvocations can handle the fully unbounded data setting. Within our study, weshow that an improved analysis of $\texttt{AboveThreshold}$ can improve theprivacy guarantees for the widely used Sparse Vector Technique that is ofindependent interest. We give a more general characterization of privacy lossfor $\texttt{AboveThreshold}$ which we immediately apply to our method forimproved privacy guarantees. Our algorithm only requires one $O(n)$ passthrough the data, which can be unsorted, and each subsequent query takes $O(1)$time. We empirically compare our unbounded algorithm with the state-of-the-artalgorithms in the bounded setting. For inner quantiles, we find that our methodoften performs better on non-synthetic datasets. For the maximal quantiles,which we apply to differentially private sum computation, we find that ourmethod performs significantly better.

----

## [0] How to Turn Your Knowledge Graph Embeddings into Generative Models

**Authors**: *Lorenzo Loconte, Nicola Di Mauro, Robert Peharz, Antonio Vergari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b768188be63b8d2680a46934fd295a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b768188be63b8d2680a46934fd295a-Abstract-Conference.html)

**Abstract**:

Some of the most successful knowledge graph embedding (KGE) models for link prediction – CP, RESCAL, TuckER, ComplEx – can be interpreted as energy-based models. Under this perspective they are not amenable for exact maximum-likelihood estimation (MLE), sampling and struggle to integrate logical constraints. This work re-interprets the score functions of these KGEs as circuits – constrained computational graphs allowing efficient marginalisation. Then, we design two recipes to obtain efficient generative circuit models by either restricting their activations to be non-negative or squaring their outputs. Our interpretation comes with little or no loss of performance for link prediction, while the circuits framework unlocks exact learning by MLE, efficient sampling of new triples, and guarantee that logical constraints are satisfied by design. Furthermore, our models scale more gracefully than the original KGEs on graphs with millions of entities.

----

## [0] Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer

**Authors**: *Zikai Xiao, Zihan Chen, Songshang Liu, Hualiang Wang, Yang Feng, Jin Hao, Joey Tianyi Zhou, Jian Wu, Howard H. Yang, Zuozhu Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html)

**Abstract**:

Data privacy and long-tailed distribution are the norms rather than the exception in many real-world tasks. This paper investigates a federated long-tailed learning (Fed-LT) task in which each client holds a locally heterogeneous dataset; if the datasets can be globally aggregated, they jointly exhibit a long-tailed distribution. Under such a setting, existing federated optimization and/or centralized long-tailed learning methods hardly apply due to challenges in (a) characterizing the global long-tailed distribution under privacy constraints and (b) adjusting the local learning strategy to cope with the head-tail imbalance. In response, we propose a method termed $\texttt{Fed-GraB}$, comprised of a Self-adjusting Gradient Balancer (SGB) module that re-weights clients' gradients in a closed-loop manner, based on the feedback of global long-tailed distribution evaluated by a Direct Prior Analyzer (DPA) module. Using $\texttt{Fed-GraB}$, clients can effectively alleviate the distribution drift caused by data heterogeneity during the model training process and obtain a global model with better performance on the minority classes while maintaining the performance of the majority classes. Extensive experiments demonstrate that $\texttt{Fed-GraB}$ achieves state-of-the-art performance on representative datasets such as CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist.

----

## [0] CityRefer: Geography-aware 3D Visual Grounding Dataset on City-scale Point Cloud Data

**Authors**: *Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.

----

## [0] GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

**Authors**: *Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The extraordinary ability of generative models to generate photographic images has intensified concerns about the spread of disinformation, thereby leading to the demand for detectors capable of distinguishing between AI-generated fake images and real images. However, the lack of large datasets containing images from the most advanced image generators poses an obstacle to the development of such detectors. In this paper, we introduce the GenImage dataset, which has the following advantages: 1) Plenty of Images, including over one million pairs of AI-generated fake images and collected real images. 2) Rich Image Content, encompassing a broad range of image classes. 3) State-of-the-art Generators, synthesizing images with advanced diffusion models and GANs. The aforementioned advantages allow the detectors trained on GenImage to undergo a thorough evaluation and demonstrate strong applicability to diverse images. We conduct a comprehensive analysis of the dataset and propose two tasks for evaluating the detection method in resembling real-world scenarios. The cross-generator image classification task measures the performance of a detector trained on one generator when tested on the others. The degraded image classification task assesses the capability of the detectors in handling degraded images such as low-resolution, blurred, and compressed images. With the GenImage dataset, researchers can effectively expedite the development and evaluation of superior AI-generated image detectors in comparison to prevailing methodologies.

----

## [0] On Differentially Private Sampling from Gaussian and Product Distributions

**Authors**: *Badih Ghazi, Xiao Hu, Ravi Kumar, Pasin Manurangsi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html)

**Abstract**:

We study the problem, where given a dataset of $n$ i.i.d. samples from an unknown distribution $P$, we seek to generate a sample from a distribution that is close to $P$ in total variation distance, under the constraint of differential privacy. We study the settings where $P$ is a multi-dimensional Gaussian distribution with different assumptions: known covariance, unknown bounded covariance, and unknown unbounded covariance. We present new differentially private sampling algorithms, and show that they achieve near-optimal sample complexity in the first two settings. Moreover, when $P$ is a product distribution on the binary hypercube, we obtain a pure-DP algorithm whereas only an approximate-DP algorithm (with slightly worse sample complexity) was previously known.

----

## [0] MedSat: A Public Health Dataset for England Featuring Medical Prescriptions and Satellite Imagery

**Authors**: *Sanja Scepanovic, Ivica Obadic, Sagar Joglekar, Laura Giustarini, Cristiano Nattero, Daniele Quercia, Xiaoxiang Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4fdf676c3b21f20f8c391d929188386-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4fdf676c3b21f20f8c391d929188386-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As extreme weather events become more frequent, understanding their impact on human health becomes increasingly crucial. However, the utilization of Earth Observation to effectively analyze the environmental context in relation to health remains limited. This limitation is primarily due to the lack of fine-grained spatial and temporal data in public and population health studies, hindering a comprehensive understanding of health outcomes. Additionally, obtaining appropriate environmental indices across different geographical levels and timeframes poses a challenge. For the years 2019 (pre-COVID) and 2020 (COVID), we collected spatio-temporal indicators for all Lower Layer Super Output Areas in England. These indicators included: i) 111 sociodemographic features linked to health in existing literature, ii) 43 environmental point features (e.g., greenery and air pollution levels), iii) 4 seasonal composite satellite images each with 11 bands, and iv) prescription prevalence associated with five medical conditions (depression, anxiety, diabetes, hypertension, and asthma), opioids and total prescriptions. We combined these indicators into a single MedSat dataset, the availability of which presents an opportunity for the machine learning community to develop new techniques specific to public health. These techniques would address challenges such as handling large and complex data volumes, performing effective feature engineering on environmental and sociodemographic factors, capturing spatial and temporal dependencies in the models, addressing imbalanced data distributions, developing novel computer vision methods for health modeling based on satellite imagery, ensuring model explainability, and achieving generalization beyond the specific geographical region.

----



[Go to the previous page](NIPS-2023-list3300.md)

[Go to the next page](NIPS-2023-list3302.md)

[Go to the catalog section](README.md)