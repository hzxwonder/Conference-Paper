## [0] HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception

**Authors**: *Junkun Yuan, Xinyu Zhang, Hao Zhou, Jian Wang, Zhongwei Qiu, Zhiyin Shao, Shaofeng Zhang, Sifan Long, Kun Kuang, Kun Yao, Junyu Han, Errui Ding, Lanfen Lin, Fei Wu, Jingdong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ed1c94a6c87276f25ebb65231c86c3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ed1c94a6c87276f25ebb65231c86c3e-Abstract-Conference.html)

**Abstract**:

Model pre-training is essential in human-centric perception. In this paper, we first introduce masked image modeling (MIM) as a pre-training approach for this task. Upon revisiting the MIM training strategy, we reveal that human structure priors offer significant potential. Motivated by this insight, we further incorporate an intuitive human structure prior - human parts - into pre-training. Specifically, we employ this prior to guide the mask sampling process. Image patches, corresponding to human part regions, have high priority to be masked out. This encourages the model to concentrate more on body structure information during pre-training, yielding substantial benefits across a range of human-centric perception tasks. To further capture human characteristics, we propose a structure-invariant alignment loss that enforces different masked views, guided by the human part prior, to be closely aligned for the same image. We term the entire method as HAP. HAP simply uses a plain ViT as the encoder yet establishes new state-of-the-art performance on 11 human-centric benchmarks, and on-par result on one dataset. For example, HAP achieves 78.1% mAP on MSMT17 for person re-identification, 86.54% mA on PA-100K for pedestrian attribute recognition, 78.2% AP on MS COCO for 2D pose estimation, and 56.0 PA-MPJPE on 3DPW for 3D pose and shape estimation.

----

## [0] Trust Your 𝛁: Gradient-based Intervention Targeting for Causal Discovery

**Authors**: *Mateusz Olko, Michal Zajac, Aleksandra Nowak, Nino Scherrer, Yashas Annadani, Stefan Bauer, Lukasz Kucinski, Piotr Milos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9eda77f505efbb89462970d739143f73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9eda77f505efbb89462970d739143f73-Abstract-Conference.html)

**Abstract**:

Inferring causal structure from data is a challenging task of fundamental importance in science. Often, observational data alone is not enough to uniquely identify a system’s causal structure. The use of interventional data can address this issue, however, acquiring these samples typically demands a considerable investment of time and physical or financial resources. In this work, we are concerned with the acquisition of interventional data in a targeted manner to minimize the number of required experiments. We propose a novel Gradient-based Intervention Targeting method, abbreviated GIT, that ’trusts’ the gradient estimator of a gradient-based causal discovery framework to provide signals for the intervention targeting function. We provide extensive experiments in simulated and real-world datasets and demonstrate that GIT performs on par with competitive baselines, surpassing them in the low-data regime.

----

## [0] SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions

**Authors**: *Yuseung Lee, Kunho Kim, Hyunjin Kim, Minhyuk Sung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html)

**Abstract**:

The remarkable capabilities of pretrained image diffusion models have been utilized not only for generating fixed-size images but also for creating panoramas. However, naive stitching of multiple images often results in visible seams. Recent techniques have attempted to address this issue by performing joint diffusions in multiple windows and averaging latent features in overlapping regions. However, these approaches, which focus on seamless montage generation, often yield incoherent outputs by blending different scenes within a single image. To overcome this limitation, we propose SyncDiffusion, a plug-and-play module that synchronizes multiple diffusions through gradient descent from a perceptual similarity loss. Specifically, we compute the gradient of the perceptual loss using the predicted denoised images at each denoising step, providing meaningful guidance for achieving coherent montages. Our experimental results demonstrate that our method produces significantly more coherent outputs compared to previous methods (66.35% vs. 33.65% in our user study) while still maintaining fidelity (as assessed by GIQA) and compatibility with the input prompt (as measured by CLIP score). We further demonstrate the versatility of our method across three plug-and-play applications: layout-guided image generation, conditional image generation and 360-degree panorama generation. Our project page is at https://syncdiffusion.github.io.

----

## [0] Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean

**Authors**: *Spyridon Kondylatos, Ioannis Prapas, Gustau Camps-Valls, Ioannis Papoutsis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ee3ed2dd656402f954ef9dc37e39f48-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ee3ed2dd656402f954ef9dc37e39f48-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Mesogeos, a large-scale multi-purpose dataset for wildfire modeling in the Mediterranean. Mesogeos integrates variables representing wildfire drivers (meteorology, vegetation, human activity) and historical records of wildfire ignitions and burned areas for 17 years (2006-2022). It is designed as a cloud-friendly spatio-temporal dataset, namely a datacube, harmonizing all variables in a grid of 1km x 1km x 1-day resolution. The datacube structure offers opportunities to assess machine learning (ML) usage in various wildfire modeling tasks. We extract two ML-ready datasets that establish distinct tracks to demonstrate this potential: (1) short-term wildfire danger forecasting and (2) final burned area estimation given the point of ignition. We define appropriate metrics and baselines to evaluate the performance of models in each track. By publishing the datacube, along with the code to create the ML datasets and models, we encourage the community to foster the implementation of additional tracks for mitigating the increasing threat of wildfires in the Mediterranean.

----

## [0] Deep learning with kernels through RKHM and the Perron-Frobenius operator

**Authors**: *Yuka Hashimoto, Masahiro Ikeda, Hachem Kadri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ef237e007e26180ce4d16738efdf83f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ef237e007e26180ce4d16738efdf83f-Abstract-Conference.html)

**Abstract**:

Reproducing kernel Hilbert $C^*$-module (RKHM) is a generalization of reproducing kernel Hilbert space (RKHS) by means of $C^*$-algebra, and the Perron-Frobenius operator is a linear operator related to the composition of functions. Combining these two concepts, we present deep RKHM, a deep learning framework for kernel methods. We derive a new Rademacher generalization bound in this setting and provide a theoretical interpretation of benign overfitting by means of Perron-Frobenius operators. By virtue of $C^*$-algebra, the dependency of the bound on output dimension is milder than existing bounds. We show that $C^*$-algebra is a suitable tool for deep learning with kernels, enabling us to take advantage of the product structure of operators and to provide a clear connection with convolutional neural networks. Our theoretical analysis provides a new lens through which one can design and analyze deep kernel methods.

----

## [0] SmoothHess: ReLU Network Feature Interactions via Stein's Lemma

**Authors**: *Max Torop, Aria Masoomi, Davin Hill, Kivanç Köse, Stratis Ioannidis, Jennifer G. Dy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html)

**Abstract**:

Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.

----

## [0] MLFMF: Data Sets for Machine Learning for Mathematical Formalization

**Authors**: *Andrej Bauer, Matej Petkovic, Ljupco Todorovski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9efe8db7fab57e19eed25718abedbbd2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9efe8db7fab57e19eed25718abedbbd2-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.

----

## [0] DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

**Authors**: *Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, Phillip Isola*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

**Abstract**:

Current perceptual similarity metrics operate at the level of pixels and patches. These metrics compare images in terms of their low-level colors and textures, but fail to capture mid-level similarities and differences in image layout, object pose, and semantic content. In this paper, we develop a perceptual metric that assesses images holistically. Our first step is to collect a new dataset of human similarity judgments over image pairs that are alike in diverse ways. Critical to this dataset is that judgments are nearly automatic and shared by all observers. To achieve this we use recent text-to-image models to create synthetic pairs that are perturbed along various dimensions. We observe that popular perceptual metrics fall short of explaining our new data, and we introduce a new metric, DreamSim, tuned to better align with human perception. We analyze how our metric is affected by different visual attributes, and find that it focuses heavily on foreground objects and semantic content while also being sensitive to color and layout. Notably, despite being trained on synthetic data, our metric generalizes to real images, giving strong results on retrieval and reconstruction tasks. Furthermore, our metric outperforms both prior learned metrics and recent large vision models on these tasks. Our project page: https://dreamsim-nights.github.io/

----

## [0] Explaining the Uncertain: Stochastic Shapley Values for Gaussian Process Models

**Authors**: *Siu Lun Chau, Krikamol Muandet, Dino Sejdinovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html)

**Abstract**:

We present a novel approach for explaining Gaussian processes (GPs) that can utilize the full analytical covariance structure present in GPs. Our method is based on the popular solution concept of Shapley values extended to stochastic cooperative games, resulting in explanations that are random variables. The GP explanations generated using our approach satisfy similar favorable axioms to standard Shapley values and possess a tractable covariance function across features and data observations. This covariance allows for quantifying explanation uncertainties and studying the statistical dependencies between explanations. We further extend our framework to the problem of predictive explanation, and propose a Shapley prior over the explanation function to predict Shapley values for new data based on previously computed ones. Our extensive illustrations demonstrate the effectiveness of the proposed approach.

----

## [0] WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes

**Authors**: *Satoshi Tsutsui, Winnie Pang, Bihan Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f34484e5b8d87f09cc58c292a1c9f5d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f34484e5b8d87f09cc58c292a1c9f5d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The examination of blood samples at a microscopic level plays a fundamental role in clinical diagnostics. For instance, an in-depth study of White Blood Cells (WBCs), a crucial component of our blood, is essential for diagnosing blood-related diseases such as leukemia and anemia. While multiple datasets containing WBC images have been proposed, they mostly focus on cell categorization, often lacking the necessary morphological details to explain such categorizations, despite the importance of explainable artificial intelligence (XAI) in medical domains. This paper seeks to address this limitation by introducing comprehensive annotations for WBC images. Through collaboration with pathologists, a thorough literature review, and manual inspection of microscopic images, we have identified 11 morphological attributes associated with the cell and its components (nucleus, cytoplasm, and granules). We then annotated ten thousand WBC images with these attributes, resulting in 113k labels (11 attributes x 10.3k images). Annotating at this level of detail and scale is unprecedented, offering unique value to AI in pathology. Moreover, we conduct experiments to predict these attributes from cell images, and also demonstrate specific applications that can benefit from our detailed annotations. Overall, our dataset paves the way for interpreting WBC recognition models, further advancing XAI in the fields of pathology and hematology.

----

## [0] Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling

**Authors**: *Haotao Wang, Ziyu Jiang, Yuning You, Yan Han, Gaowen Liu, Jayanth Srinivasa, Ramana Kompella, Zhangyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) have found extensive applications in learning from graph data. However, real-world graphs often possess diverse structures and comprise nodes and edges of varying types. To bolster the generalization capacity of GNNs, it has become customary to augment training graph structures through techniques like graph augmentations and large-scale pre-training on a wider array of graphs. Balancing this diversity while avoiding increased computational costs and the notorious trainability issues of GNNs is crucial. This study introduces the concept of Mixture-of-Experts (MoE) to GNNs, with the aim of augmenting their capacity to adapt to a diverse range of training graph structures, without incurring explosive computational overhead. The proposed Graph Mixture of Experts (GMoE) model empowers individual nodes in the graph to dynamically and adaptively select more general information aggregation experts. These experts are trained to capture distinct subgroups of graph structures and to incorporate information with varying hop sizes, where those with larger hop sizes specialize in gathering information over longer distances. The effectiveness of GMoE is validated through a series of experiments on a diverse set of tasks, including graph, node, and link prediction, using the OGB benchmark. Notably, it enhances ROC-AUC by $1.81\%$ in ogbg-molhiv and by $1.40\%$ in ogbg-molbbbp, when compared to the non-MoE baselines. Our code is publicly available at https://github.com/VITA-Group/Graph-Mixture-of-Experts.

----

## [0] Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction

**Authors**: *Quentin Delfosse, Hikaru Shindo, Devendra Singh Dhami, Kristian Kersting*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f42f06a54ce3b709ad78d34c73e4363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f42f06a54ce3b709ad78d34c73e4363-Abstract-Conference.html)

**Abstract**:

The limited priors required by neural networks make them the dominating choice to encode and learn policies using reinforcement learning (RL). However, they are also black-boxes, making it hard to understand the agent's behavior, especially when working on the image level. Therefore, neuro-symbolic RL aims at creating policies that are interpretable in the first place.Unfortunately, interpretability is not explainability. To achieve both, we introduce Neurally gUided Differentiable loGic policiEs (NUDGE). NUDGE exploits trained neural network-based agents to guide the search of candidate-weighted logic rules, then uses differentiable logic to train the logic agents. Our experimental evaluation demonstrates that NUDGE agents can induce interpretable and explainable policies while outperforming purely neural ones and showing good flexibility to environments of different initial states and problem sizes.

----

## [0] Personalized Dictionary Learning for Heterogeneous Datasets

**Authors**: *Geyu Liang, Naichen Shi, Raed Al Kontar, Salar Fattahi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f6f790f28a31fba89644f09faf4e0cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f6f790f28a31fba89644f09faf4e0cb-Abstract-Conference.html)

**Abstract**:

We introduce a relevant yet challenging problem named Personalized Dictionary Learning (PerDL), where the goal is to learn sparse linear representations from heterogeneous datasets that share some commonality. In PerDL, we model each dataset's shared and unique features as global and local dictionaries. Challenges for PerDL not only are inherited from classical dictionary learning(DL), but also arise due to the unknown nature of the shared and unique features. In this paper, we rigorously formulate this problem and provide conditions under which the global and local dictionaries can be provably disentangled. Under these conditions, we provide a meta-algorithm called Personalized Matching and Averaging (PerMA) that can recover both global and local dictionaries from heterogeneous datasets. PerMA is highly efficient; it converges to the ground truth at a linear rate under suitable conditions. Moreover, it automatically borrows strength from strong learners to improve the prediction of weak learners. As a general framework for extracting global and local dictionaries, we show the application of PerDL in different learning tasks, such as training with imbalanced datasets and video surveillance.

----

## [0] Graph-Structured Gaussian Processes for Transferable Graph Learning

**Authors**: *Jun Wu, Lisa Ainsworth, Andrew Leakey, Haixun Wang, Jingrui He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f7f2f57d8eaf44b2f09020f64ff6d96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f7f2f57d8eaf44b2f09020f64ff6d96-Abstract-Conference.html)

**Abstract**:

Transferable graph learning involves knowledge transferability from a source graph to a relevant target graph. The major challenge of transferable graph learning is the distribution shift between source and target graphs induced by individual node attributes and complex graph structures. To solve this problem, in this paper, we propose a generic graph-structured Gaussian process framework (GraphGP) for adaptively transferring knowledge across graphs with either homophily or heterophily assumptions. Specifically, GraphGP is derived from a novel graph structure-aware neural network in the limit on the layer width. The generalization analysis of GraphGP explicitly investigates the connection between knowledge transferability and graph domain similarity. Extensive experiments on several transferable graph learning benchmarks demonstrate the efficacy of GraphGP over state-of-the-art Gaussian process baselines.

----

## [0] Language Models are Weak Learners

**Authors**: *Hariharan Manikandan, Yiding Jiang, J. Zico Kolter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f94298bac4668db4dc77ddb0a244301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f94298bac4668db4dc77ddb0a244301-Abstract-Conference.html)

**Abstract**:

A central notion in practical and theoretical machine learning is that of a weak learner, classifiers that achieve better-than-random performance (on any given distribution over data), even by a small margin.  Such weak learners form the practical basis for canonical machine learning methods such as boosting.  In this work, we illustrate that prompt-based large language models can operate effectively as said weak learners.  Specifically, we illustrate the use of a large language model (LLM) as a weak learner in a boosting algorithm applied to tabular data.  We show that by providing (properly sampled according to the distribution of interest) text descriptions of tabular data samples, LLMs can produce a summary of the samples that serves as a template for classification, and achieves the aim of acting as a weak learner on this task.  We incorporate these models into a boosting approach, which in many settings can leverage the knowledge within the LLM to outperform traditional tree-based boosting.  The model outperforms both few-shot learning and occasionally even more involved fine-tuning procedures, particularly for some tasks involving small numbers of data points.  The results illustrate the potential for prompt-based LLMs to function not just as few-shot learners themselves, but as components of larger machine learning models.

----

## [0] SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models

**Authors**: *Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, Animesh Garg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9fa03b16dbd6cabc7601fe98c6ec291e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9fa03b16dbd6cabc7601fe98c6ec291e-Abstract-Conference.html)

**Abstract**:

Object-centric learning aims to represent visual data with a set of object entities (a.k.a. slots), providing structured representations that enable systematic generalization.Leveraging advanced architectures like Transformers, recent approaches have made significant progress in unsupervised object discovery.In addition, slot-based representations hold great potential for generative modeling, such as controllable image generation and object manipulation in image editing.However, current slot-based methods often produce blurry images and distorted objects, exhibiting poor generative modeling capabilities.In this paper, we focus on improving slot-to-image decoding, a crucial aspect for high-quality visual generation.We introduce SlotDiffusion -- an object-centric Latent Diffusion Model (LDM) designed for both image and video data.Thanks to the powerful modeling capacity of LDMs, SlotDiffusion surpasses previous slot models in unsupervised object segmentation and visual generation across six datasets.Furthermore, our learned object features can be utilized by existing object-centric dynamics models, improving video prediction quality and downstream temporal reasoning tasks.Finally, we demonstrate the scalability of SlotDiffusion to unconstrained real-world datasets such as PASCAL VOC and COCO, when integrated with self-supervised pre-trained image encoders.

----

## [0] ZoomTrack: Target-aware Non-uniform Resizing for Efficient Visual Tracking

**Authors**: *Yutong Kou, Jin Gao, Bing Li, Gang Wang, Weiming Hu, Yizheng Wang, Liang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9fc291fef2f9607a46777d367f900a15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9fc291fef2f9607a46777d367f900a15-Abstract-Conference.html)

**Abstract**:

Recently, the transformer has enabled the speed-oriented trackers to approach state-of-the-art (SOTA) performance with high-speed thanks to the smaller input size or the lighter feature extraction backbone, though they still substantially lag behind their corresponding performance-oriented versions. In this paper, we demonstrate that it is possible to narrow or even close this gap while achieving high tracking speed based on the smaller input size. To this end, we non-uniformly resize the cropped image to have a smaller input size while the resolution of the area where the target is more likely to appear is higher and vice versa. This enables us to solve the dilemma of attending to a larger visual field while retaining more raw information for the target despite a smaller input size. Our formulation for the non-uniform resizing can be efficiently solved through quadratic programming (QP) and naturally integrated into most of the crop-based local trackers. Comprehensive experiments on five challenging datasets based on two kinds  of transformer trackers, \ie, OSTrack and TransT, demonstrate consistent improvements over them. In particular, applying our method to the speed-oriented version of OSTrack even outperforms its performance-oriented counterpart by 0.6\% AUC on TNL2K, while running 50\% faster and saving over 55\% MACs. Codes and models are available at https://github.com/Kou-99/ZoomTrack.

----

## [0] Improving neural network representations using human similarity judgments

**Authors**: *Lukas Muttenthaler, Lorenz Linhardt, Jonas Dippel, Robert A. Vandermeulen, Katherine L. Hermann, Andrew K. Lampinen, Simon Kornblith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)

**Abstract**:

Deep neural networks have reached human-level performance on many computer vision tasks. However, the objectives used to train these networks enforce only that similar images are embedded at similar locations in the representation space, and do not directly constrain the global structure of the resulting space. Here, we explore the impact of supervising this global structure by linearly aligning it with human similarity judgments. We find that a naive approach leads to large changes in local representational structure that harm downstream performance. Thus, we propose a novel method that aligns the global structure of representations while preserving their local structure. This global-local transform considerably improves accuracy across a variety of few-shot learning and anomaly detection tasks. Our results indicate that human visual representations are globally organized in a way that facilitates learning from few examples, and incorporating this global structure into neural network representations improves performance on downstream tasks.

----

## [0] Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery

**Authors**: *Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html)

**Abstract**:

The strength of modern generative models lies in their ability to be controlled through prompts. Hard prompts comprise interpretable words and tokens, and are typically hand-crafted by humans.  Soft prompts, on the other hand, consist of continuous feature vectors.  These can be discovered using powerful optimization methods, but they cannot be easily edited, re-used across models, or plugged into a text-based interface. We describe an easy-to-use approach to automatically optimize hard text prompts through efficient gradient-based optimization. Our approach can be readily applied to text-to-image and text-only applications alike. This method allows API users to easily generate, discover, and mix and match image concepts without prior knowledge of how to prompt the model. Furthermore, using our method, we can bypass token-level content filters imposed by Midjourney by optimizing through the open-sourced text encoder.

----

## [0] Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm

**Authors**: *Jie Hao, Kaiyi Ji, Mingrui Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0251e494a7e75d59e06d37e646f46b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0251e494a7e75d59e06d37e646f46b7-Abstract-Conference.html)

**Abstract**:

Coreset is a small set that provides a data summary for a large dataset, such that training solely on the small set achieves competitive performance compared with a large dataset. In rehearsal-based continual learning, the coreset is typically used in the memory replay buffer to stand for representative samples in previous tasks, and the coreset selection procedure is typically formulated as a bilevel problem. However, the typical bilevel formulation for coreset selection explicitly performs optimization over discrete decision variables with greedy search, which is computationally expensive. Several works consider other formulations to address this issue, but they ignore the nested nature of bilevel optimization problems and may not solve the bilevel coreset selection problem accurately. To address these issues, we propose a new bilevel formulation, where the inner problem tries to find a model which minimizes the expected training error sampled from a given probability distribution, and the outer problem aims to learn the probability distribution with approximately $K$ (coreset size) nonzero entries such that learned model in the inner problem minimizes the training error over the whole data. To ensure the learned probability has approximately $K$ nonzero entries, we introduce a novel regularizer based on the smoothed top-$K$ loss in the upper problem. We design a new optimization algorithm that provably converges to the $\epsilon$-stationary point with $O(1/\epsilon^4)$ computational complexity. We conduct extensive experiments in various settings in continual learning, including balanced data, imbalanced data, and label noise, to show that our proposed formulation and new algorithm significantly outperform competitive baselines. From bilevel optimization point of view, our algorithm significantly improves the vanilla greedy coreset selection method in terms of running time on continual learning benchmark datasets. The code is available at https://github.com/MingruiLiu-ML-Lab/Bilevel-Coreset-Selection-via-Regularization.

----

## [0] HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork

**Authors**: *Bipasha Sen, Gaurav Singh, Aditya Agarwal, Rohith Agaram, K. Madhava Krishna, Srinath Sridhar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRF) have become an increasingly popular representation to capture high-quality appearance and shape of scenes and objects. However, learning generalizable NeRF priors over categories of scenes or objects has been challenging due to the high dimensionality of network weight space. To address the limitations of existing work on generalization, multi-view consistency and to improve quality, we propose HyP-NeRF, a latent conditioning method for learning generalizable category-level NeRF priors using hypernetworks. Rather than using hypernetworks to estimate only the weights of a NeRF, we estimate both the weights and the multi-resolution hash encodings resulting in significant quality gains. To improve quality even further, we incorporate a denoise and finetune strategy that denoises images rendered from NeRFs estimated by the hypernetwork and finetunes it while retaining multiview consistency. These improvements enable us to use HyP-NeRF as a generalizable prior for multiple downstream tasks including NeRF reconstruction from single-view or cluttered scenes and text-to-NeRF. We provide qualitative comparisons and evaluate HyP-NeRF on three tasks: generalization, compression, and retrieval, demonstrating our state-of-the-art results.

----

## [0] MultiVENT: Multilingual Videos of Events and Aligned Natural Text

**Authors**: *Kate Sanders, David Etter, Reno Kriz, Benjamin Van Durme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a054ff49751dbc991ec30ae479397c3d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a054ff49751dbc991ec30ae479397c3d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Everyday news coverage has shifted from traditional broadcasts towards a wide range of presentation formats such as first-hand, unedited video footage. Datasets that reflect the diverse array of multimodal, multilingual news sources available online could be used to teach models to benefit from this shift, but existing news video datasets focus on traditional news broadcasts produced for English-speaking audiences. We address this limitation by constructing MultiVENT, a dataset of multilingual, event-centric videos grounded in text documents across five target languages. MultiVENT includes both news broadcast videos and non-professional event footage, which we use to analyze the state of online news videos and how they can be leveraged to build robust, factually accurate models. Finally, we provide a  model for complex, multilingual video retrieval to serve as a baseline for information retrieval using MultiVENT.

----

## [0] GEO-Bench: Toward Foundation Models for Earth Monitoring

**Authors**: *Alexandre Lacoste, Nils Lehmann, Pau Rodríguez, Evan D. Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vázquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

----

## [0] Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

**Authors**: *Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, Kai Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html)

**Abstract**:

In the past years, YOLO-series models have emerged as the leading approaches in the area of real-time object detection. Many studies pushed up the baseline to a higher level by modifying the architecture, augmenting data and designing new losses. However, we find previous models still suffer from information fusion problem, although Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) have alleviated this. Therefore, this study provides an advanced Gatherand-Distribute mechanism (GD) mechanism, which is realized with convolution and self-attention operations. This new designed model named as Gold-YOLO, which boosts the multi-scale feature fusion capabilities and achieves an ideal balance between latency and accuracy across all model scales. Additionally, we implement MAE-style pretraining in the YOLO-series for the first time, allowing YOLOseries models could be to benefit from unsupervised pretraining. Gold-YOLO-N attains an outstanding 39.9% AP on the COCO val2017 datasets and 1030 FPS on a T4 GPU, which outperforms the previous SOTA model YOLOv6-3.0-N with similar FPS by +2.4%. The PyTorch code is available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO, and the MindSpore code is available at https://gitee.com/mindspore/models/tree/master/research/cv/Gold_YOLO.

----

## [0] Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First

**Authors**: *Zheng Zhang, Junxiang Wang, Liang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a07e5160196058120105ad7cb3505d3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a07e5160196058120105ad7cb3505d3c-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have achieved great success in representing data with dependencies by recursively propagating and aggregating messages along the edges. However, edges in real-world graphs often have varying degrees of difficulty, and some edges may even be noisy to the downstream tasks. Therefore, existing GNNs may lead to suboptimal learned representations because they usually treat every edge in the graph equally. On the other hand, Curriculum Learning (CL), which mimics the human learning principle of learning data samples in a meaningful order, has been shown to be effective in improving the generalization ability and robustness of representation learners by gradually proceeding from easy to more difficult samples during training. Unfortunately, existing CL strategies are designed for independent data samples and cannot trivially generalize to handle data dependencies. To address these issues, we propose a novel CL strategy to gradually incorporate more edges into training according to their difficulty from easy to hard, where the degree of difficulty is measured by how well the edges are expected given the model training status. We demonstrate the strength of our proposed method in improving the generalization ability and robustness of learned representations through extensive experiments on nine synthetic datasets and nine real-world datasets. The code for our proposed method is available at https://github.com/rollingstonezz/Curriculumlearningfor_GNNs

----

## [0] Unified Lower Bounds for Interactive High-dimensional Estimation under Information Constraints

**Authors**: *Jayadev Acharya, Clément L. Canonne, Ziteng Sun, Himanshu Tyagi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a07e87ecfa8a651d62257571669b0150-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a07e87ecfa8a651d62257571669b0150-Abstract-Conference.html)

**Abstract**:

We consider distributed parameter estimation using interactive protocols subject to local information constraints such as bandwidth limitations, local differential privacy, and restricted measurements. We provide a unified framework enabling us to derive a variety of (tight) minimax lower bounds for different parametric families of distributions, both continuous and discrete, under any $\ell_p$ loss. Our lower bound framework is versatile and yields “plug-and-play” bounds that are widely applicable to a large range of estimation problems, and, for the prototypical case of the Gaussian family, circumvents limitations of previous techniques. In particular, our approach recovers bounds obtained using data processing inequalities and Cramér–Rao bounds, two other alternative approaches for proving lower bounds in our setting of interest. Further, for the families considered, we complement our lower bounds with matching upper bounds.

----

## [0] Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching

**Authors**: *Junsheng Zhou, Baorui Ma, Wenyuan Zhang, Yi Fang, Yu-Shen Liu, Zhizhong Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)

**Abstract**:

Cross-modality registration between 2D images captured by cameras and 3D point clouds from LiDARs is a crucial task in computer vision and robotic. Previous methods estimate 2D-3D correspondences by matching point and pixel patterns learned by neural networks, and use Perspective-n-Points (PnP) to estimate rigid transformation during post-processing. However, these methods struggle to map points and pixels to a shared latent space robustly since points and pixels have very different characteristics with patterns learned in different manners (MLP and CNN), and they also fail to construct supervision directly on the transformation since the PnP is non-differentiable, which leads to unstable registration results. To address these problems, we propose to learn a structured cross-modality latent space to represent pixel features and 3D features via a differentiable probabilistic PnP solver. Specifically, we design a triplet network to learn VoxelPoint-to-Pixel matching, where we represent 3D elements using both voxels and points to learn the cross-modality latent space with pixels. We design both the voxel and pixel branch based on CNNs to operate convolutions on voxels/pixels represented in grids, and integrate an additional point branch to regain the information lost during voxelization. We train our framework end-to-end by imposing supervisions directly on the predicted pose distribution with a probabilistic PnP solver. To explore distinctive patterns of cross-modality features, we design a novel loss with adaptive-weighted optimization for cross-modality feature description. The experimental results on KITTI and nuScenes datasets show significant improvements over the state-of-the-art methods.

----

## [0] Ecosystem-level Analysis of Deployed Machine Learning Reveals Homogeneous Outcomes

**Authors**: *Connor Toups, Rishi Bommasani, Kathleen Creel, Sarah H. Bana, Dan Jurafsky, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0b1082fc7823c4c68abcab4fa850e9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0b1082fc7823c4c68abcab4fa850e9c-Abstract-Conference.html)

**Abstract**:

Machine learning is traditionally studied at the model level: researchers measure and improve the accuracy, robustness, bias, efficiency, and other dimensions of specific models. In practice, however, the societal impact of any machine learning model is partially determined by the context into which it is deployed. To capture this, we introduce ecosystem-level analysis: rather than analyzing a single model, we consider the collection of models that are deployed in a given context. For example, ecosystem-level analysis in hiring recognizes that a job candidateâ€™s outcomes are determined not only by a single hiring algorithm or firm but instead by the collective decisions of all the firms to which the candidate applied. Across three modalities (text, images, speech) and 11 datasets, we establish a clear trend: deployed machine learning is prone to systemic failure, meaning some users are exclusively misclassified by all models available. Even when individual models improve at the population level over time, we find these improvements rarely reduce the prevalence of systemic failure. Instead, the benefits of these improvements predominantly accrue to individuals who are already correctly classified by other models. In light of these trends, we analyze medical imaging for dermatology, a setting where the costs of systemic failure are especially high. While traditional analyses reveal that both models and humans exhibit racial performance disparities, ecosystem-level analysis reveals new forms of racial disparity in model predictions that do not present in human predictions. These examples demonstrate that ecosystem-level analysis has unique strengths in characterizing the societal impact of machine learning.

----

## [0] MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion

**Authors**: *Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, Yasutaka Furukawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0da690a47b2f52faa63f6fe054057b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0da690a47b2f52faa63f6fe054057b5-Abstract-Conference.html)

**Abstract**:

This paper introduces MVDiffusion, a simple yet effective method for generating consistent multi-view images from text prompts given pixel-to-pixel correspondences (e.g., perspective crops from a panorama or multi-view images given depth maps and poses). Unlike prior methods that rely on iterative image warping and inpainting, MVDiffusion simultaneously generates all images with a global awareness, effectively addressing the prevalent error accumulation issue. At its core, MVDiffusion processes perspective images in parallel with a pre-trained text-to-image diffusion model, while integrating novel correspondence-aware attention layers to facilitate cross-view interactions. For panorama generation, while only trained with 10k panoramas, MVDiffusion is able to generate high-resolution photorealistic images for arbitrary texts or extrapolate one perspective image to a 360-degree view. For multi-view depth-to-image generation, MVDiffusion demonstrates state-of-the-art performance for texturing a scene mesh. The project page is at https://mvdiffusion.github.io/.

----

## [0] The geometry of hidden representations of large transformer models

**Authors**: *Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0e66093d7168b40246af1cddc025daa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0e66093d7168b40246af1cddc025daa-Abstract-Conference.html)

**Abstract**:

Large transformers are powerful architectures used for self-supervised data analysis across various data types, including protein sequences, images, and text. In these models, the semantic structure of the dataset emerges from a sequence of transformations between one representation and the next. We characterize the geometric and statistical properties of these representations and how they change as we move through the layers.By analyzing the intrinsic dimension (ID) and neighbor composition, we find that the representations evolve similarly in transformers trained on protein language taskand image reconstruction tasks. In the first layers, the data manifold expands, becoming high-dimensional, and then contracts significantly in the intermediate layers. In the last part of the model, the ID remains approximately constant or forms a second shallow peak. We show that the semantic information of the dataset is better expressed at the end of the first peak, and this phenomenon can be observed across many models trained on diverse datasets.Based on our findings, we point out an explicit strategy to identify, without supervision, the layers that maximize semantic content: representations at intermediate layers corresponding to a relative minimum of the ID profile are more suitable for downstream learning tasks.

----

## [0] Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration

**Authors**: *Guangyu Shen, Siyuan Cheng, Guanhong Tao, Kaiyuan Zhang, Yingqi Liu, Shengwei An, Shiqing Ma, Xiangyu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html)

**Abstract**:

Object detection models are vulnerable to backdoor or trojan attacks, where an attacker can inject malicious triggers into the model, leading to altered behavior during inference. As a defense mechanism, trigger inversion leverages optimization to reverse-engineer triggers and identify compromised models. While existing trigger inversion methods assume that each instance from the support set is equally affected by the injected trigger, we observe that the poison effect can vary significantly across bounding boxes in object detection models due to its dense prediction nature, leading to an undesired optimization objective misalignment issue for existing trigger reverse-engineering methods. To address this challenge, we propose the first object detection backdoor detection framework Django (Detecting Trojans in Object Detection Models via Gaussian Focus Calibration). It leverages a dynamic Gaussian weighting scheme that prioritizes more vulnerable victim boxes and assigns appropriate coefficients to calibrate the optimization objective during trigger inversion. In addition, we combine Django with a novel label proposal pre-processing technique to enhance its efficiency. We evaluate Django on 3 object detection image datasets, 3 model architectures, and 2 types of attacks, with a total of 168 models. Our experimental results show that Django outperforms 6 state-of-the-art baselines, with up to 38% accuracy improvement and 10x reduced overhead. The code is available at https://github.com/PurduePAML/DJGO.

----

## [0] CORNN: Convex optimization of recurrent neural networks for rapid inference of neural dynamics

**Authors**: *Fatih Dinc, Adam Shai, Mark Schnitzer, Hidenori Tanaka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a103529738706979331778377f2d5864-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a103529738706979331778377f2d5864-Abstract-Conference.html)

**Abstract**:

Advances in optical and electrophysiological recording technologies have made it possible to record the dynamics of thousands of neurons, opening up new possibilities for interpreting and controlling large neural populations in behaving animals. A promising way to extract computational principles from these large datasets is to train data-constrained recurrent neural networks (dRNNs). Performing this training in real-time could open doors for research techniques and medical applications to model and control interventions at single-cell resolution and drive desired forms of animal behavior. However, existing training algorithms for dRNNs are inefficient and have limited scalability, making it a challenge to analyze large neural recordings even in offline scenarios. To address these issues, we introduce a training method termed Convex Optimization of Recurrent Neural Networks (CORNN). In studies of simulated recordings, CORNN attained training speeds $\sim$100-fold faster than traditional optimization approaches while maintaining or enhancing modeling accuracy. We further validated CORNN on simulations with thousands of cells that performed simple computations such as those of a 3-bit flip-flop or the execution of a timed response. Finally, we showed that CORNN can robustly reproduce network dynamics and underlying attractor structures despite mismatches between generator and inference models, severe subsampling of observed neurons, or mismatches in neural time-scales. Overall, by training dRNNs with millions of parameters in subminute processing times on a standard computer, CORNN constitutes a first step towards real-time network reproduction constrained on large-scale neural recordings and a powerful computational tool for advancing the understanding of neural computation.

----

## [0] A Unified Framework for Rank-based Loss Minimization

**Authors**: *Rufeng Xiao, Yuze Ge, Rujun Jiang, Yifan Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html)

**Abstract**:

The empirical loss, commonly referred to as the average loss, is extensively utilized for training machine learning models. However, in order to address the diverse performance requirements of machine learning models, the use of the rank-based loss is prevalent, replacing the empirical loss in many cases. The rank-based loss comprises a weighted sum of sorted individual losses, encompassing both convex losses like the spectral risk, which includes the empirical risk and conditional value-at-risk, and nonconvex losses such as the human-aligned risk and the sum of the ranked range loss. In this paper, we introduce a unified framework for the optimization of the rank-based loss through the utilization of a proximal alternating direction method of multipliers. We demonstrate the convergence and convergence rate of the proposed algorithm under mild conditions. Experiments conducted on synthetic and real datasets illustrate the effectiveness and efficiency of the proposed algorithm.

----

## [0] LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas

**Authors**: *Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, Charles Sutton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a10da26f47120217c1b7c2aeb2979048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a10da26f47120217c1b7c2aeb2979048-Abstract-Conference.html)

**Abstract**:

Search is an important technique in program synthesis that allows for adaptive strategies such as focusing on particular search directions based on execution results. Several prior works have demonstrated that neural models are effective at guiding program synthesis searches. However, a common drawback of those approaches is the inability to handle iterative loops, higher-order functions, or lambda functions, thus limiting prior neural searches from synthesizing longer and more general programs. We address this gap by designing a search algorithm called LambdaBeam that can construct arbitrary lambda functions that compose operations within a given DSL. We create semantic vector representations of the execution behavior of the lambda functions and train a neural policy network to choose which lambdas to construct during search, and pass them as arguments to higher-order functions to perform looping computations. Our experiments show that LambdaBeam outperforms neural, symbolic, and LLM-based techniques in an integer list manipulation domain.

----

## [0] HQA-Attack: Toward High Quality Black-Box Hard-Label Adversarial Attack on Text

**Authors**: *Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a124b5e7385d35e5c8ad05d192106e19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a124b5e7385d35e5c8ad05d192106e19-Abstract-Conference.html)

**Abstract**:

Black-box hard-label adversarial attack on text is a practical and challenging task, as the text data space is inherently discrete and non-differentiable, and only the predicted label is accessible. Research on this problem is still in the embryonic stage and only a few methods are available. Nevertheless, existing methods rely on the complex heuristic algorithm or unreliable gradient estimation strategy, which probably fall into the local optimum and inevitably consume numerous queries, thus are difficult to craft satisfactory adversarial examples with high semantic similarity and low perturbation rate in a limited query budget. To alleviate above issues, we propose a simple yet effective framework to generate high quality textual adversarial examples under the black-box hard-label attack scenarios, named HQA-Attack. Specifically, after initializing an adversarial example randomly, HQA-attack first constantly substitutes original words back as many as possible, thus shrinking the perturbation rate. Then it leverages the synonym set of the remaining changed words to further optimize the adversarial example with the direction which can improve the semantic similarity and satisfy the adversarial condition simultaneously. In addition, during the optimizing procedure, it searches a transition synonym word for each changed word, thus avoiding traversing the whole synonym set and reducing the query number to some extent. Extensive experimental results on five text classification datasets, three natural language inference datasets and two real-world APIs have shown that the proposed HQA-Attack method outperforms other strong baselines significantly.

----

## [0] Augmentation-free Dense Contrastive Distillation for Efficient Semantic Segmentation

**Authors**: *Jiawei Fan, Chao Li, Xiaolong Liu, Meina Song, Anbang Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html)

**Abstract**:

In recent years, knowledge distillation methods based on contrastive learning have achieved promising results on image classification and object detection tasks. However, in this line of research, we note that less attention is paid to semantic segmentation. Existing methods heavily rely on data augmentation and memory buffer, which entail high computational resource demands when applying them to handle semantic segmentation that requires to preserve high-resolution feature maps for making dense pixel-wise predictions. In order to address this problem, we present Augmentation-free Dense Contrastive Knowledge Distillation (Af-DCD), a new contrastive distillation learning paradigm to train compact and accurate deep neural networks for semantic segmentation applications. Af-DCD leverages a masked feature mimicking strategy, and formulates a novel contrastive learning loss via taking advantage of tactful feature partitions across both channel and spatial dimensions, allowing to effectively transfer dense and structured local knowledge learnt by the teacher model to a target student model while maintaining training efficiency. Extensive experiments on five mainstream benchmarks with various teacher-student network pairs demonstrate the effectiveness of our approach. For instance, DeepLabV3-Res18|DeepLabV3-MBV2 model trained by Af-DCD reaches 77.03\%|76.38\% mIOU on Cityscapes dataset when choosing DeepLabV3-Res101 as the teacher, setting new performance records. Besides that, Af-DCD achieves an absolute mIOU improvement of 3.26\%|3.04\%|2.75\%|2.30\%|1.42\% compared with individually trained counterpart on Cityscapes|Pascal VOC|Camvid|ADE20K|COCO-Stuff-164K. Code is available at https://github.com/OSVAI/Af-DCD.

----

## [0] On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets

**Authors**: *Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Different distribution shifts require different algorithmic and operational  interventions. Methodological research must be grounded by the specific  shifts they address.  Although nascent benchmarks provide a promising  empirical foundation, they \emph{implicitly} focus on covariate  shifts, and the validity of empirical findings depends on the type of shift,   e.g., previous observations on algorithmic performance can fail to be valid when  the $Y|X$ distribution changes.  We conduct a thorough investigation of  natural shifts in 5 tabular datasets over 86,000 model configurations, and  find that $Y|X$-shifts are most prevalent.  To encourage researchers to  develop a refined language for distribution shifts, we build ``WhyShift``, an empirical testbed of curated real-world shifts where  we characterize the type of shift we benchmark performance over.  Since  $Y|X$-shifts are prevalent in tabular settings, we \emph{identify covariate  regions} that suffer the biggest $Y|X$-shifts and discuss implications for  algorithmic and data-based interventions.  Our testbed highlights the  importance of future research that builds an understanding of why  distributions differ.

----

## [0] Diverse Community Data for Benchmarking Data Privacy Algorithms

**Authors**: *Aniruddha Sen, Christine Task, Dhruv Kapur, Gary Howarth, Karan Bhagat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a15032f8199511ced4d7a8e2bbb487a5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a15032f8199511ced4d7a8e2bbb487a5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The Collaborative Research Cycle (CRC) is a National Institute of Standards and Technology (NIST) benchmarking program intended to strengthen understanding of tabular data deidentification technologies. Deidentification algorithms are vulnerable to the same bias and privacy issues that impact other data analytics and machine learning applications, and it can even amplify those issues by contaminating downstream applications. This paper summarizes four CRC contributions: theoretical work on the relationship between diverse populations and challenges for equitable deidentification; public benchmark data focused on diverse populations and challenging features; a comprehensive open source suite of evaluation metrology for deidentified datasets; and an archive of more than 450 deidentified data samples from a broad range of techniques. The initial set of evaluation results demonstrate the value of the CRC tools for investigations in this field.

----

## [0] Coneheads: Hierarchy Aware Attention

**Authors**: *Albert Tseng, Tao Yu, Toni J. B. Liu, Christopher De Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a17251f8d595179eef5e466b1f5f7a85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a17251f8d595179eef5e466b1f5f7a85-Abstract-Conference.html)

**Abstract**:

Attention networks such as transformers have achieved state-of-the-art performance in many domains. These networks rely heavily on the dot product attention operator, which computes the similarity between two points by taking their inner product.However, the inner product does not explicitly model the complex structural properties of real world datasets, such as hierarchies between data points.To remedy this, we introduce cone attention, a drop-in replacement for dot product attention based on hyperbolic entailment cones.Cone attention associates two points by the depth of their lowest common ancestor in a hierarchy defined by hyperbolic cones, which intuitively measures the divergence of two points and gives a $\textit{hierarchy aware}$ similarity score.We test cone attention on a wide variety of models and tasks and show that it improves task-level performance over dot product attention and other baselines, and is able to match dot-product attention with significantly fewer parameters.Our results suggest that cone attention is an effective way to capture hierarchical relationships when calculating attention.

----

## [0] Benchmark of Machine Learning Force Fields for Semiconductor Simulations: Datasets, Metrics, and Comparative Analysis

**Authors**: *Geonu Kim, Byunggook Na, Gunhee Kim, Hyuntae Cho, Seungjin Kang, Hee Sun Lee, Saerom Choi, Heejae Kim, Seungwon Lee, Yongdeok Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As semiconductor devices become miniaturized and their structures become more complex, there is a growing need for large-scale atomic-level simulations as a less costly alternative to the trial-and-error approach during development.Although machine learning force fields (MLFFs) can meet the accuracy and scale requirements for such simulations, there are no open-access benchmarks for semiconductor materials.Hence, this study presents a comprehensive benchmark suite that consists of two semiconductor material datasets and ten MLFF models with six evaluation metrics. We select two important semiconductor thin-film materials silicon nitride and hafnium oxide, and generate their datasets using computationally expensive density functional theory simulations under various scenarios at a cost of 2.6k GPU days.Additionally, we provide a variety of architectures as baselines: descriptor-based fully connected neural networks and graph neural networks with rotational invariant or equivariant features.We assess not only the accuracy of energy and force predictions but also five additional simulation indicators to determine the practical applicability of MLFF models in molecular dynamics simulations.To facilitate further research, our benchmark suite is available at https://github.com/SAITPublic/MLFF-Framework.

----

## [0] Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks

**Authors**: *Aoxiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1c716638d9b618a1a40a96f473c8250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1c716638d9b618a1a40a96f473c8250-Abstract-Conference.html)

**Abstract**:

No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models againstadversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial videoâ€™s estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.

----

## [0] Unsupervised Behavior Extraction via Random Intent Priors

**Authors**: *Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, Chongjie Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1c8a68e52499c9396854e3f967e37c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1c8a68e52499c9396854e3f967e37c0-Abstract-Conference.html)

**Abstract**:

Reward-free data is abundant and contains rich prior knowledge of human behaviors, but it is not well exploited by offline reinforcement learning (RL) algorithms. In this paper, we propose UBER, an unsupervised approach to extract useful behaviors from offline reward-free datasets via diversified rewards. UBER assigns different pseudo-rewards sampled from a given prior distribution to different agents to extract a diverse set of behaviors, and reuse them as candidate policies to facilitate the learning of new tasks. Perhaps surprisingly, we show that rewards generated from random neural networks are sufficient to extract diverse and useful behaviors, some even close to expert ones. We provide both empirical and theoretical evidences to justify the use of random priors for the reward function. Experiments on multiple benchmarks showcase UBER's ability to learn effective and diverse behavior sets that enhance sample efficiency for online RL, outperforming existing baselines. By reducing reliance on human supervision, UBER broadens the applicability of RL to real-world scenarios with abundant reward-free data.

----

## [0] Deconstructing Data Reconstruction: Multiclass, Weight Decay and General Losses

**Authors**: *Gon Buzaglo, Niv Haim, Gilad Yehudai, Gal Vardi, Yakir Oz, Yaniv Nikankin, Michal Irani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1d20cc72a21ef971d7e49a90d8fa56f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1d20cc72a21ef971d7e49a90d8fa56f-Abstract-Conference.html)

**Abstract**:

Memorization of training data is an active research area, yet our understanding of the inner workings of neural networks is still in its infancy.Recently, Haim et al. 2022 proposed a scheme to reconstruct training samples from multilayer perceptron binary classifiers, effectively demonstrating that a large portion of training samples are encoded in the parameters of such networks.In this work, we extend their findings in several directions, including reconstruction from multiclass and convolutional neural networks. We derive a more general reconstruction scheme which is applicable to a wider range of loss functions such as regression losses. Moreover, we study the various factors that contribute to networks' susceptibility to such reconstruction schemes. Intriguingly, we observe that using weight decay during training increases reconstructability both in terms of quantity and quality. Additionally, we examine the influence of the number of neurons relative to the number of training samples on the reconstructability.Code: https://github.com/gonbuzaglo/decoreco

----

## [0] Information Maximizing Curriculum: A Curriculum-Based Approach for Learning Versatile Skills

**Authors**: *Denis Blessing, Onur Celik, Xiaogang Jia, Moritz Reuss, Maximilian Li, Rudolf Lioutikov, Gerhard Neumann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1e6783e4d739196cad3336f12d402bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1e6783e4d739196cad3336f12d402bf-Abstract-Conference.html)

**Abstract**:

Imitation learning uses data for training policies to solve complex tasks. However,when the training data is collected from human demonstrators, it often leadsto multimodal distributions because of the variability in human actions. Mostimitation learning methods rely on a maximum likelihood (ML) objective to learna parameterized policy, but this can result in suboptimal or unsafe behavior dueto the mode-averaging property of the ML objective. In this work, we proposeInformation Maximizing Curriculum, a curriculum-based approach that assignsa weight to each data point and encourages the model to specialize in the data itcan represent, effectively mitigating the mode-averaging problem by allowing themodel to ignore data from modes it cannot represent. To cover all modes and thus,enable versatile behavior, we extend our approach to a mixture of experts (MoE)policy, where each mixture component selects its own subset of the training datafor learning. A novel, maximum entropy-based objective is proposed to achievefull coverage of the dataset, thereby enabling the policy to encompass all modeswithin the data distribution. We demonstrate the effectiveness of our approach oncomplex simulated control tasks using versatile human demonstrations, achievingsuperior performance compared to state-of-the-art methods.

----

## [0] Unleash the Potential of Image Branch for Cross-modal 3D Object Detection

**Authors**: *Yifan Zhang, Qijian Zhang, Junhui Hou, Yixuan Yuan, Guoliang Xing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html)

**Abstract**:

To achieve reliable and precise scene understanding, autonomous vehicles typically incorporate multiple sensing modalities to capitalize on their complementary attributes. However, existing cross-modal 3D detectors do not fully utilize the image domain information to address the bottleneck issues of the LiDAR-based detectors. This paper presents a new cross-modal 3D object detector, namely UPIDet, which aims to unleash the potential of the image branch from two aspects. First, UPIDet introduces a new 2D auxiliary task called normalized local coordinate map estimation. This approach enables the learning of local spatial-aware features from the image modality to supplement sparse point clouds. Second, we discover that the representational capability of the point cloud backbone can be enhanced through the gradients backpropagated from the training objectives of the image branch, utilizing a succinct and effective point-to-pixel module. Extensive experiments and ablation studies validate the effectiveness of our method. Notably, we achieved the top rank in the highly competitive cyclist class of the KITTI benchmark at the time of submission. The source code is available at https://github.com/Eaphan/UPIDet.

----

## [0] Model Sparsity Can Simplify Machine Unlearning

**Authors**: *Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, Sijia Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html)

**Abstract**:

In response to recent data regulation requirements, machine unlearning (MU) has emerged as a critical process to remove the influence of specific examples from a given model. Although exact unlearning can be achieved through complete model retraining using the remaining dataset, the associated computational costs have driven the development of efficient, approximate unlearning techniques. Moving beyond data-centric MU approaches, our study introduces a novel model-based perspective: model sparsification via weight pruning, which is capable of reducing the gap between exact unlearning and approximate unlearning. We show in both theory and practice that model sparsity can boost the multi-criteria unlearning performance of an approximate unlearner, closing the approximation gap, while continuing to be efficient. This leads to a new MU paradigm,    termed prune first, then unlearn, which infuses a sparse prior to the unlearning process. Building on this insight, we also develop a sparsity-aware unlearning method that utilizes sparsity regularization to enhance the training process of approximate unlearning. Extensive experiments show that our proposals consistently benefit MU in various unlearning scenarios. A notable highlight is the 77% unlearning efficacy gain of fine-tuning (one of the simplest approximate unlearning methods) when using our proposed sparsity-aware unlearning method. Furthermore, we showcase the practical impact of our proposed MU methods through two specific use cases: defending against backdoor attacks, and enhancing transfer learning through source class removal. These applications demonstrate the versatility and effectiveness of our approaches in addressing a variety of machine learning challenges beyond unlearning for data privacy. Codes are available at https://github.com/OPTML-Group/Unlearn-Sparse.

----

## [0] IDRNet: Intervention-Driven Relation Network for Semantic Segmentation

**Authors**: *Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, Lequan Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a216c27f2f3160b1785c057fa510fdf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a216c27f2f3160b1785c057fa510fdf1-Abstract-Conference.html)

**Abstract**:

Co-occurrent visual patterns suggest that pixel relation modeling facilitates dense prediction tasks, which inspires the development of numerous context modeling paradigms, \emph{e.g.}, multi-scale-driven and similarity-driven context schemes. Despite the impressive results, these existing paradigms often suffer from inadequate or ineffective contextual information aggregation due to reliance on large amounts of predetermined priors. To alleviate the issues, we propose a novel \textbf{I}ntervention-\textbf{D}riven \textbf{R}elation \textbf{Net}work (\textbf{IDRNet}), which leverages a deletion diagnostics procedure to guide the modeling of contextual relations among different pixels. Specifically, we first group pixel-level representations into semantic-level representations with the guidance of pseudo labels and further improve the distinguishability of the grouped representations with a feature enhancement module. Next, a deletion diagnostics procedure is conducted to model relations of these semantic-level representations via perceiving the network outputs and the extracted relations are utilized to guide the semantic-level representations to interact with each other.  Finally, the interacted representations are utilized to augment original pixel-level representations for final predictions. Extensive experiments are conducted to validate the effectiveness of IDRNet quantitatively and qualitatively. Notably, our intervention-driven context scheme brings consistent performance improvements to state-of-the-art segmentation frameworks and achieves competitive results on popular benchmark datasets, including ADE20K, COCO-Stuff, PASCAL-Context, LIP, and Cityscapes.

----

## [0] Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width

**Authors**: *Dayal Singh Kalra, Maissam Barkeshli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a23598416361c7a9860164155e6ddd0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a23598416361c7a9860164155e6ddd0b-Abstract-Conference.html)

**Abstract**:

We systematically analyze optimization dynamics in deep neural networks (DNNs) trained with stochastic gradient descent (SGD) and study the effect of learning rate $\eta$, depth $d$, and width $w$ of the neural network. By analyzing the maximum eigenvalue $\lambda^H_t$ of the Hessian of the loss, which is a measure of sharpness of the loss landscape, we find that the dynamics can show four distinct regimes: (i) an early time transient regime, (ii) an intermediate saturation regime, (iii) a progressive sharpening regime, and (iv) a late time "edge of stability" regime. The early and intermediate regimes (i) and (ii) exhibit a rich phase diagram depending on $\eta \equiv c / \lambda_0^H $, $d$, and $w$. We identify several critical values of $c$, which separate qualitatively distinct phenomena in the early time dynamics of training loss and sharpness. Notably, we discover the opening up of a "sharpness reduction" phase, where sharpness decreases at early times, as $d$ and $ 1/w$ are increased.

----

## [0] Neural Algorithmic Reasoning Without Intermediate Supervision

**Authors**: *Gleb Rodionov, Liudmila Prokhorenkova*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2370db7c99791ad5d9f3ef48ad6d464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2370db7c99791ad5d9f3ef48ad6d464-Abstract-Conference.html)

**Abstract**:

Neural algorithmic reasoning is an emerging area of machine learning focusing on building models that can imitate the execution of classic algorithms, such as sorting, shortest paths, etc. One of the main challenges is to learn algorithms that are able to generalize to out-of-distribution data, in particular with significantly larger input sizes. Recent work on this problem has demonstrated the advantages of learning algorithms step-by-step, giving models access to all intermediate steps of the original algorithm. In this work, we instead focus on learning neural algorithmic reasoning only from the input-output pairs without appealing to the intermediate supervision. We propose simple but effective architectural improvements and also build a self-supervised objective that can regularise intermediate computations of the model without access to the algorithm trajectory. We demonstrate that our approach is competitive to its trajectory-supervised counterpart on tasks from the CLRS Algorithmic Reasoning Benchmark and achieves new state-of-the-art results for several problems, including sorting, where we obtain significant improvements. Thus, learning without intermediate supervision is a promising direction for further research on neural reasoners.

----

## [0] On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

**Authors**: *Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, Sungroh Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2374637af47ac9471b43c99b68acf27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2374637af47ac9471b43c99b68acf27-Abstract-Conference.html)

**Abstract**:

Successful detection of Out-of-Distribution (OoD) data is becoming increasingly important to ensure safe deployment of neural networks. One of the main challenges in OoD detection is that neural networks output overconfident predictions on OoD data, make it difficult to determine OoD-ness of data solely based on their predictions. Outlier exposure addresses this issue by introducing an additional loss that encourages low-confidence predictions on OoD data during training. While outlier exposure has shown promising potential in improving OoD detection performance, all previous studies on outlier exposure have been limited to utilizing visual outliers. Drawing inspiration from the recent advancements in vision-language pre-training, this paper venture out to the uncharted territory of textual outlier exposure. First, we uncover the benefits of using textual outliers by replacing real or virtual outliers in the image-domain with textual equivalents. Then, we propose various ways of generating preferable textual outliers. Our extensive experiments demonstrate that generated textual outliers achieve competitive performance on large-scale OoD and hard OoD benchmarks. Furthermore, we conduct empirical analyses of textual outliers to provide primary criteria for designing advantageous textual outliers: near-distribution, descriptiveness, and inclusion of visual semantics.

----

## [0] Estimating Propensity for Causality-based Recommendation without Exposure Data

**Authors**: *Zhongzhou Liu, Yuan Fang, Min Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a237f11d6aad94f59a182d70405d3fdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a237f11d6aad94f59a182d70405d3fdb-Abstract-Conference.html)

**Abstract**:

Causality-based recommendation systems focus on the causal effects of user-item interactions resulting from item exposure (i.e., which items are recommended or exposed to the user), as opposed to conventional correlation-based recommendation. They are gaining popularity due to their multi-sided benefits to users, sellers and platforms alike. However, existing causality-based recommendation methods require additional input in the form of exposure data and/or propensity scores (i.e., the probability of exposure) for training. Such data, crucial for modeling causality in recommendation, are often not available in real-world situations due to technical or privacy constraints. In this paper, we bridge the gap by proposing a new framework, called Propensity Estimation for Causality-based Recommendation (PropCare). It can estimate the propensity and exposure from a more practical setup, where only interaction data are available without any ground truth on exposure or propensity in training and inference. We demonstrate that, by relating the pairwise characteristics between propensity and item popularity, PropCare enables competitive causality-based recommendation given only the conventional interaction data. We further present a theoretical analysis on the bias of the causal effect under our model estimation.  Finally, we empirically evaluate PropCare through both quantitative and qualitative experiments.

----

## [0] A Robust Exact Algorithm for the Euclidean Bipartite Matching Problem

**Authors**: *Akshaykumar Gattani, Sharath Raghvendra, Pouyan Shirzadian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a24a75ef009ee73b160653c16b18f00e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a24a75ef009ee73b160653c16b18f00e-Abstract-Conference.html)

**Abstract**:

Algorithms for the minimum-cost bipartite matching can be used to estimate Wasserstein distance between two distributions.Given two sets $A$ and $B$ of $n$ points in a $2$-dimensional Euclidean space, one can use a fast implementation of the Hungarian method to compute a minimum-cost bipartite matching of $A$ and $B$ in $\tilde{O}(n^2)$ time. Let $\Delta$ be the spread, i.e., the ratio of the distance of the farthest to the closest pair of points in $A\cup B$. In this paper, we present a new algorithm to compute a minimum-cost bipartite matching of $A$ and $B$ with a similar worst-case execution time of $\tilde{O}(n^2 \log \Delta)$. However, when $A$ and $B$ are drawn independently and identically from a fixed distribution that is not known to the algorithm, the execution time of our algorithm is, in expectation, $\tilde{O}(n^{7/4}\log \Delta)$.To the best of our knowledge, our algorithm is the first one to achieve a sub-quadratic execution time even for stochastic point sets with real-valued coordinates.Our algorithm extends to any dimension $d$, where it runs in $\tilde{O}(n^{2-\frac{1}{2d}}\Phi(n))$ time for stochastic point sets $A$ and $B$; here $\Phi(n)$ is the query/update time of a dynamic weighted nearest neighbor data structure. Our algorithm can be seen as a careful adaptation of the Hungarian method in the geometric divide-and-conquer framework.

----

## [0] Content-based Unrestricted Adversarial Attack

**Authors**: *Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html)

**Abstract**:

Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack (ACA) based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4\% and 16.8-48.0\% in normally trained models and defense methods, respectively.

----

## [0] On Dynamic Programming Decompositions of Static Risk Measures in Markov Decision Processes

**Authors**: *Jia Lin Hau, Erick Delage, Mohammad Ghavamzadeh, Marek Petrik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a264726ebd222124514a32bf0143b83d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a264726ebd222124514a32bf0143b83d-Abstract-Conference.html)

**Abstract**:

Optimizing static risk-averse objectives in Markov decision processes is difficult because they do not admit standard dynamic programming equations common in Reinforcement Learning (RL) algorithms. Dynamic programming decompositions that augment the state space with discrete risk levels have recently gained popularity in the RL community. Prior work has shown that these decompositions are optimal when the risk level is discretized sufficiently. However, we show that these popular decompositions for Conditional-Value-at-Risk (CVaR) and Entropic-Value-at-Risk (EVaR) are inherently suboptimal regardless of the discretization level. In particular, we show that a saddle point property assumed to hold in prior literature may be violated. However, a decomposition does hold for Value-at-Risk and our proof demonstrates how this risk measure differs from CVaR and EVaR. Our findings are significant because risk-averse algorithms are used in high-stake environments, making their correctness much more critical.

----

## [0] Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models

**Authors**: *Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip H. S. Torr, Volker Tresp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2a544e43acb8b954dc5846ff0d77ad5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2a544e43acb8b954dc5846ff0d77ad5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. As test samples in real-world applications usually differ from adaptation data, the robustness of these adaptation methods against distribution shifts are essential. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that increasing the number of adaptation data and parameters does not guarantee enhanced robustness; instead, it results in even lower robustness. We hope this study could benefit future research in the development of robust multimodal adaptation methods. The benchmark, code, and dataset used in this study can be accessed at https://adarobustness.github.io.

----

## [0] Evaluating the Moral Beliefs Encoded in LLMs

**Authors**: *Nino Scherrer, Claudia Shi, Amir Feder, David M. Blei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html)

**Abstract**:

This paper presents a case study on the design, administration, post-processing,  and evaluation of surveys on large language models (LLMs). It comprises two components:(1) A statistical method for eliciting beliefs encoded in LLMs. We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice.(2) We apply this method to study what moral beliefs are encoded in different LLMs, especially in ambiguous cases where the right choice is not obvious.We design a large-scale survey comprising 680 high-ambiguity moral scenarios (e.g., "Should I tell a white lie?") and 687 low-ambiguity moral scenarios (e.g., "Should I stop for a pedestrian on the road?"). Each scenario includes a description, two possible actions, and auxiliary labels indicating violated rules (e.g., "do not kill"). We administer the survey to 28 open- and closed-source LLMs.We find that (a) in unambiguous scenarios, most models ``choose" actions that align with commonsense. In ambiguous cases, most models express uncertainty.(b) Some models are uncertain about choosing the commonsense action because their responses are sensitive to the question-wording.(c) Some models reflect clear preferences in ambiguous scenarios. Specifically, closed-source models tend to agree with each other.

----

## [0] Enhancing Adversarial Robustness via Score-Based Optimization

**Authors**: *Boya Zhang, Weijian Luo, Zhihua Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2e707354da36956945dbb288efe82b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2e707354da36956945dbb288efe82b3-Abstract-Conference.html)

**Abstract**:

Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data  in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

----

## [0] Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation

**Authors**: *Giorgio Giannone, Akash Srivastava, Ole Winther, Faez Ahmed*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html)

**Abstract**:

Generative models have significantly influenced both vision and language domains, ushering in innovative multimodal applications. Although these achievements have motivated exploration in scientific and engineering fields, challenges emerge, particularly in constrained settings with limited data where precision is crucial. Traditional engineering optimization methods rooted in physics often surpass generative models in these contexts. To address these challenges, we introduce Diffusion Optimization Models (DOM) and Trajectory Alignment (TA), a learning framework that demonstrates the efficacy of aligning the sampling trajectory of diffusion models with the trajectory derived from physics-based iterative optimization methods. This alignment ensures that the sampling process remains grounded in the underlying physical principles. This alignment eliminates the need for costly preprocessing, external surrogate models, or extra labeled data, generating feasible and high-performance designs efficiently. We apply our framework to structural topology optimization, a fundamental problem in mechanical design, evaluating its performance on in- and out-of-distribution configurations. Our results demonstrate that TA outperforms state-of-the-art deep generative models on in-distribution configurations and halves the inference computational cost. When coupled with a few steps of optimization, it also improves manufacturability for out-of-distribution conditions. DOM's efficiency and performance improvements significantly expedite design processes and steer them toward optimal and manufacturable outcomes, highlighting the potential of generative models in data-driven design.

----

## [0] Optimal cross-learning for contextual bandits with unknown context distributions

**Authors**: *Jon Schneider, Julian Zimmert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3017a8d202a433be56a3dfdcac6c8eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3017a8d202a433be56a3dfdcac6c8eb-Abstract-Conference.html)

**Abstract**:

We consider the problem of designing contextual bandit algorithms in the ``cross-learning'' setting of Balseiro et al., where the learner observes the loss for the action they play in all possible contexts, not just the context of the current round. We specifically consider the setting where losses are chosen adversarially and contexts are sampled i.i.d. from an unknown distribution. In this setting, we resolve an open problem of Balseiro et al. by providing an efficient algorithm with a nearly tight (up to logarithmic factors) regret bound of $\widetilde{O}(\sqrt{TK})$, independent of the number of contexts. As a consequence, we obtain the first nearly tight regret bounds for the problems of learning to bid in first-price auctions (under unknown value distributions) and sleeping bandits with a stochastic action set.At the core of our algorithm is a novel technique for coordinating the execution of a learning algorithm over multiple epochs in such a way to remove correlations between estimation of the unknown distribution and the actions played by the algorithm. This technique may be of independent interest for other learning problems involving estimation of an unknown context distribution.

----

## [0] Conservative Offline Policy Adaptation in Multi-Agent Games

**Authors**: *Chengjie Wu, Pingzhong Tang, Jun Yang, Yujing Hu, Tangjie Lv, Changjie Fan, Chongjie Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a31253f4871694f09541122d6b6f5ad1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a31253f4871694f09541122d6b6f5ad1-Abstract-Conference.html)

**Abstract**:

Prior research on policy adaptation in multi-agent games has often relied on online interaction with the target agent in training, which can be expensive and impractical in real-world scenarios. Inspired by recent progress in offline reinforcement learn- ing, this paper studies offline policy adaptation, which aims to utilize the target agentâ€™s behavior data to exploit its weakness or enable effective cooperation. We investigate its distinct challenges of distributional shift and risk-free deviation, and propose a novel learning objective, conservative offline adaptation, that optimizes the worst-case performance against any dataset consistent proxy models. We pro- pose an efficient algorithm called Constrained Self-Play (CSP) that incorporates dataset information into regularized policy learning. We prove that CSP learns a near-optimal risk-free offline adaptation policy upon convergence. Empirical results demonstrate that CSP outperforms non-conservative baselines in various environments, including Maze, predator-prey, MuJoCo, and Google Football.

----

## [0] Bounding the Invertibility of Privacy-preserving Instance Encoding using Fisher Information

**Authors**: *Kiwan Maeng, Chuan Guo, Sanjay Kariyappa, G. Edward Suh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a344f7f474958cc0775be7e46bc94309-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a344f7f474958cc0775be7e46bc94309-Abstract-Conference.html)

**Abstract**:

Privacy-preserving instance encoding aims to encode raw data into feature vectors without revealing their privacy-sensitive information. When designed properly, these encodings can be used for downstream ML applications such as training and inference with limited privacy risk. However, the vast majority of existing schemes do not theoretically justify that their encoding is non-invertible, and their privacy-enhancing properties are only validated empirically against a limited set of attacks. In this paper, we propose a theoretically-principled measure for the invertibility of instance encoding based on Fisher information that is broadly applicable to a wide range of popular encoders. We show that dFIL can be used to bound the invertibility of encodings both theoretically and empirically, providing an intuitive interpretation of the privacy of instance encoding.

----

## [0] Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

**Authors**: *Yuxin Pan, Yize Chen, Fangzhen Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a345ed605675c7c484e740a8ceaa6b45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a345ed605675c7c484e740a8ceaa6b45-Abstract-Conference.html)

**Abstract**:

Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints.  While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound  by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.

----

## [0] Promises and Pitfalls of Threshold-based Auto-labeling

**Authors**: *Harit Vishwakarma, Heguang Lin, Frederic Sala, Ramya Korlakai Vinayak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a355051cc32d36e2a971de190701745a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a355051cc32d36e2a971de190701745a-Abstract-Conference.html)

**Abstract**:

Creating large-scale high-quality labeled datasets is a major bottleneck in supervised machine learning workflows. Threshold-based auto-labeling (TBAL), where validation data obtained from humans is used to find a confidence threshold above which the data is machine-labeled, reduces reliance on manual annotation. TBAL is emerging as a widely-used solution in practice. Given the long shelf-life and diverse usage of the resulting datasets, understanding when the data obtained by such auto-labeling systems can be relied on is crucial. This is the first work to analyze TBAL systems and derive sample complexity bounds on the amount of human-labeled validation data required for guaranteeing the quality of machine-labeled data. Our results provide two crucial insights. First, reasonable chunks of unlabeled data can be automatically and accurately labeled by seemingly bad models. Second, a hidden downside of TBAL systems is potentially prohibitive validation data usage. Together, these insights describe the promise and pitfalls of using such systems. We validate our theoretical guarantees with extensive experiments on synthetic and real datasets.

----

## [0] CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society

**Authors**: *Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html)

**Abstract**:

The rapid advancement of chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents, and provides insight into their “cognitive” processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing . Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of a society of agents, providing a valuable resource for investigating conversational language models. In particular, we conduct comprehensive studies on instruction-following cooperation in multi-agent settings. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond: https://github.com/camel-ai/camel.

----

## [0] Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis

**Authors**: *Abhinav Nippani, Dongyue Li, Haotian Ju, Haris K. Koutsopoulos, Hongyang R. Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We consider the problem of traffic accident analysis on a road network based on road network connections and traffic volume. Previous works have designed various deep-learning methods using historical records to predict traffic accident occurrences. However, there is a lack of consensus on how accurate existing methods are, and a fundamental issue is the lack of public accident datasets for comprehensive evaluations. This paper constructs a large-scale, unified dataset of traffic accident records from official reports of various states in the US, totaling 9 million records, accompanied by road networks and traffic volume reports. Using this new dataset, we evaluate existing deep-learning methods for predicting the occurrence of accidents on road networks. Our main finding is that graph neural networks such as GraphSAGE can accurately predict the number of accidents on roads with less than 22% mean absolute error (relative to the actual count) and whether an accident will occur or not with over 87% AUROC, averaged over states. We achieve these results by using multitask learning to account for cross-state variabilities (e.g., availability of accident labels) and transfer learning to combine traffic volume with accident prediction. Ablation studies highlight the importance of road graph-structural features, amongst other features. Lastly, we discuss the implications of the analysis and develop a package for easily using our new dataset.

----

## [0] SUBP: Soft Uniform Block Pruning for 1×N Sparse CNNs Multithreading Acceleration

**Authors**: *Jingyang Xiang, Siqi Li, Jun Chen, Guang Dai, Shipeng Bai, Yukai Ma, Yong Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html)

**Abstract**:

The study of sparsity in Convolutional Neural Networks (CNNs) has become widespread to compress and accelerate models in environments with limited resources. By constraining N consecutive weights along the output channel to be group-wise non-zero, the recent network with 1$\times$N sparsity has received tremendous popularity for its three outstanding advantages: 1) A large amount of storage space saving by a \emph{Block Sparse Row} matrix. 2) Excellent performance at a high sparsity. 3) Significant speedups on CPUs with Advanced Vector Extensions. Recent work requires selecting and fine-tuning 1$\times$N sparse weights based on dense pre-trained weights, leading to the problems such as expensive training cost and memory access, sub-optimal model quality, as well as unbalanced workload across threads (different sparsity across output channels). To overcome them, this paper proposes a novel \emph{\textbf{S}oft \textbf{U}niform \textbf{B}lock \textbf{P}runing} (SUBP) approach to train a uniform 1$\times$N sparse structured network from scratch. Specifically, our approach tends to repeatedly allow pruned blocks to regrow to the network based on block angular redundancy and importance sampling in a uniform manner throughout the training process. It not only makes the model less dependent on pre-training, reduces the model redundancy and the risk of pruning the important blocks permanently but also achieves balanced workload. Empirically, on ImageNet, comprehensive experiments across various CNN architectures show that our SUBP consistently outperforms existing 1$\times$N and structured sparsity methods based on pre-trained models or training from scratch. Source codes and models are available at \url{https://github.com/JingyangXiang/SUBP}.

----

## [0] Adaptive Linear Estimating Equations

**Authors**: *Mufang Ying, Koulik Khamaru, Cun-Hui Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html)

**Abstract**:

Sequential data collection has emerged as a widely adopted technique for enhancing the efficiency of data gathering processes. Despite its advantages, such data collection mechanism often introduces complexities to the statistical inference procedure. For instance, the ordinary least squares (OLS) estimator in an adaptive linear regression model can exhibit non-normal asymptotic behavior, posing challenges for accurate inference and interpretation. In this paper, we propose a general method for constructing debiased estimator which remedies this issue. It makes use of the idea of adaptive linear estimating equations, and we establish theoretical guarantees of asymptotic normality, supplemented by discussions on achieving near-optimal asymptotic variance. A salient feature of our estimator is that in the context of multi-armed bandits,  our estimator retains the non-asymptotic performance of the least squares estimator while obtaining asymptotic normality property. Consequently, this work helps connect two fruitful paradigms of adaptive inference: a) non-asymptotic inference using concentration inequalities  and b) asymptotic inference via asymptotic normality.

----

## [0] Robust Knowledge Transfer in Tiered Reinforcement Learning

**Authors**: *Jiawei Huang, Niao He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a39ab46bf619ada0e90ceed846648a81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a39ab46bf619ada0e90ceed846648a81-Abstract-Conference.html)

**Abstract**:

In this paper, we study the Tiered Reinforcement Learning setting, a parallel transfer learning framework, where the goal is to transfer knowledge from the low-tier (source) task to the high-tier (target) task to reduce the exploration risk of the latter while solving the two tasks in parallel. Unlike previous work, we do not assume the low-tier and high-tier tasks share the same dynamics or reward functions, and focus on robust knowledge transfer without prior knowledge on the task similarity. We identify a natural and necessary condition called the ``Optimal Value Dominance'' for our objective. Under this condition, we propose novel online learning algorithms such that, for the high-tier task, it can achieve constant regret on partial states depending on the task similarity and retain near-optimal regret when the two tasks are dissimilar, while for the low-tier task, it can keep near-optimal without making sacrifice. Moreover, we further study the setting with multiple low-tier tasks, and propose a novel transfer source selection mechanism, which can ensemble the information from all low-tier tasks and allow provable benefits on a much larger state-action space.

----

## [0] Bypassing the Simulator: Near-Optimal Adversarial Linear Contextual Bandits

**Authors**: *Haolin Liu, Chen-Yu Wei, Julian Zimmert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3a661eb3308d0bb686f6a4bac521032-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3a661eb3308d0bb686f6a4bac521032-Abstract-Conference.html)

**Abstract**:

We consider the adversarial linear contextual bandit problem, where the loss vectors are selected fully adversarially and the per-round action set (i.e. the context) is drawn from a fixed distribution. Existing methods for this problem either require access to a simulator to generate free i.i.d. contexts, achieve a sub-optimal regret no better than $\tilde{\mathcal{O}}(T^{\frac{5}{6}})$, or are computationally inefficient. We greatly improve these results by achieving a regret of $\tilde{\mathcal{O}}(\sqrt{T})$ without a simulator, while maintaining computational efficiency when the action set in each round is small. In the special case of sleeping bandits with adversarial loss and stochastic arm availability, our result answers affirmatively the open question by [SGV20]  on whether there exists a polynomial-time algorithm with  $poly(d)\sqrt{T}$ regret. Our approach naturally handles the case where the loss is linear up to an additive misspecification error, and our regret shows near-optimal dependence on the magnitude of the error.

----

## [0] GenEval: An object-focused framework for evaluating text-to-image alignment

**Authors**: *Dhruba Ghosh, Hannaneh Hajishirzi, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent breakthroughs in diffusion models, multimodal pretraining, and efficient finetuning have led to an explosion of text-to-image generative models. Given human evaluation is expensive and difficult to scale, automated methods are critical for evaluating the increasingly large number of new models. However, most current automated evaluation metrics like FID or CLIPScore only offer a distribution-level measure of image quality or image-text alignment, and are unsuited for fine-grained or instance-level analysis. In this paper, we introduce GenEval, an object-focused framework to evaluate compositional image properties such as object co-occurrence, position, count, and color. We show that current object detection models can be leveraged to evaluate text-to-image models on a variety of generation tasks with strong human agreement, and that other discriminative vision models can be linked to this pipeline to further verify properties like object color. We then evaluate several open-source text-to-image models and analyze their relative reasoning capabilities on our benchmark. We find that recent models demonstrate significant improvement on these tasks, though they are still lacking in complex capabilities such as spatial relations and attribute binding. Finally, we demonstrate how GenEval might be used to help discover existing failure modes, in order to inform development of the next generation of text-to-image models. Our code to run the GenEval framework will be made publicly available at https://github.com/djghosh13/geneval.

----

## [0] Generalization in the Face of Adaptivity: A Bayesian Perspective

**Authors**: *Moshe Shenfeld, Katrina Ligett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3c01875a052f81d27a5211df096cd91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3c01875a052f81d27a5211df096cd91-Abstract-Conference.html)

**Abstract**:

Repeated use of a data sample via adaptively chosen queries can rapidly lead to overfitting, wherein the empirical evaluation of queries on the sample significantly deviates from their mean with respect to the underlying data distribution. It turns out that simple noise addition algorithms suffice to prevent this issue, and differential privacy-based analysis of these algorithms shows that they can handle an asymptotically optimal number of queries.  However, differential privacy's worst-case nature entails scaling such noise to the range of the queries even for highly-concentrated queries, or introducing more complex algorithms.In this paper, we prove that straightforward noise-addition algorithms already provide variance-dependent guarantees that also extend to unbounded queries. This improvement stems from a novel characterization that illuminates the core problem of adaptive data analysis. We show that the harm of adaptivity results from the covariance between the new query and a Bayes factor-based measure of how much information about the data sample was encoded in the responses given to past queries. We then leverage this characterization to introduce a new data-dependent stability notion that can bound this covariance.

----

## [0] Convergence of Adam Under Relaxed Assumptions

**Authors**: *Haochuan Li, Alexander Rakhlin, Ali Jadbabaie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html)

**Abstract**:

In this paper, we provide a rigorous proof of convergence of the Adaptive Moment Estimate (Adam) algorithm for a wide class of optimization objectives. Despite the popularity and efficiency of the Adam algorithm in training deep neural networks, its theoretical properties are not yet fully understood, and existing convergence proofs require unrealistically strong assumptions, such as globally bounded gradients, to show the convergence to stationary points. In this paper, we show that Adam provably converges to $\epsilon$-stationary points with $\mathcal{O}(\epsilon^{-4})$ gradient complexity under far more realistic conditions. The key to our analysis is a new proof of boundedness of gradients along the optimization trajectory of Adam, under a generalized smoothness assumption according to which the local smoothness (i.e., Hessian norm when it exists) is bounded by a sub-quadratic function of the gradient norm. Moreover, we propose a variance-reduced version of Adam with an accelerated gradient complexity of $\mathcal{O}(\epsilon^{-3})$.

----

## [0] On the Convergence of Encoder-only Shallow Transformers

**Authors**: *Yongtao Wu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3cf318fbeec1126da21e9185ae9908c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3cf318fbeec1126da21e9185ae9908c-Abstract-Conference.html)

**Abstract**:

In this paper, we aim to build the global convergence theory of encoder-only shallow Transformers under a realistic setting from the perspective of architectures, initialization, and scaling under a finite width regime. The difficulty lies in how to tackle the softmax in self-attention mechanism, the core ingredient of Transformer. In particular, we diagnose the scaling scheme, carefully tackle the input/output of softmax, and prove that quadratic overparameterization is sufficient for global convergence of our shallow Transformers under commonly-used He/LeCun initialization in practice. Besides, neural tangent kernel (NTK) based analysis is also given, which facilitates a comprehensive comparison. Our theory demonstrates the separation on the importance of different scaling schemes and initialization. We believe our results can pave the way for a better understanding of modern Transformers, particularly on training dynamics.

----

## [0] SoundCam: A Dataset for Finding Humans Using Room Acoustics

**Authors**: *Mason Wang, Samuel Clarke, Jui-Hsien Wang, Ruohan Gao, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4289154c9209b679ac761a50d5fec3a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4289154c9209b679ac761a50d5fec3a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A room’s acoustic properties are a product of the room’s geometry, the objects within the room, and their specific positions. A room’s acoustic properties can be characterized by its impulse response (RIR) between a source and listener location, or roughly inferred from recordings of natural signals present in the room. Variations in the positions of objects in a room can effect measurable changes in the room’s acoustic properties, as characterized by the RIR. Existing datasets of RIRs either do not systematically vary positions of objects in an environment, or they consist of only simulated RIRs. We present SoundCam, the largest dataset of unique RIRs from in-the-wild rooms publicly released to date. It includes 5,000 10-channel real-world measurements of room impulse responses and 2,000 10-channel recordings of music in three different rooms, including a controlled acoustic lab, an in-the-wild living room, and a conference room, with different humans in positions throughout each room. We show that these measurements can be used for interesting tasks, such as detecting and identifying humans, and tracking their positions.

----

## [0] Accelerated On-Device Forward Neural Network Training with Module-Wise Descending Asynchronism

**Authors**: *Xiaohan Zhao, Hualin Zhang, Zhouyuan Huo, Bin Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a42d8f43fae4d267e3084b10056153f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a42d8f43fae4d267e3084b10056153f7-Abstract-Conference.html)

**Abstract**:

On-device learning faces memory constraints when optimizing or fine-tuning on edge devices with limited resources. Current techniques for training deep models on edge devices rely heavily on backpropagation. However, its high memory usage calls for a reassessment of its dominance.In this paper, we propose forward gradient descent (FGD) as a potential solution to overcome the memory capacity limitation in on-device learning. However, FGD's dependencies across layers hinder parallel computation and can lead to inefficient resource utilization.To mitigate this limitation, we propose AsyncFGD, an asynchronous framework that decouples dependencies, utilizes module-wise stale parameters, and maximizes parallel computation. We demonstrate its convergence to critical points through rigorous theoretical analysis.Empirical evaluations conducted on NVIDIA's AGX Orin, a popular embedded device,  show that AsyncFGD reduces memory consumption and enhances hardware efficiency, offering a novel approach to on-device learning.

----

## [0] Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

**Authors**: *Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, Jieping Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4316bb210a59fb7aafeca5dd21c2703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4316bb210a59fb7aafeca5dd21c2703-Abstract-Conference.html)

**Abstract**:

For a machine learning model deployed in real world scenarios, the ability of detecting out-of-distribution (OOD) samples is indispensable and challenging. Most existing OOD detection methods focused on exploring advanced training skills or training-free tricks to prevent the model from yielding overconfident confidence score for unknown samples. The training-based methods require expensive training cost and rely on OOD samples which are not always available, while most training-free methods can not efficiently utilize the prior information from the training data. In this work, we propose an \textbf{O}ptimal \textbf{P}arameter and \textbf{N}euron \textbf{P}runing (\textbf{OPNP}) approach, which aims to identify and remove those parameters and neurons that lead to over-fitting. The main method is divided into two steps. In the first step, we evaluate the sensitivity of the model parameters and neurons by averaging gradients over all training samples. In the second step, the parameters and neurons with exceptionally large or close to zero sensitivities are removed for prediction. Our proposal is training-free, compatible with other post-hoc methods, and exploring the information from all training data. Extensive experiments are performed on multiple OOD detection tasks and model architectures, showing that our proposed OPNP consistently outperforms the existing methods by a large margin.

----

## [0] Unbalanced Low-rank Optimal Transport Solvers

**Authors**: *Meyer Scetbon, Michal Klein, Giovanni Palla, Marco Cuturi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a439259e78294c38d157a51a2c40486b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a439259e78294c38d157a51a2c40486b-Abstract-Conference.html)

**Abstract**:

The relevance of optimal transport methods to machine learning has long been hindered by two salient limitations.First, the $O(n^3)$ computational cost of standard sample-based solvers (when used on batches of $n$ samples) is prohibitive.Second, the mass conservation constraint makes OT solvers too rigid in practice: because they must match \textit{all} points from both measures, their output can be heavily influenced by outliers.A flurry of recent works in OT has addressed these computational and modelling limitations, but has resulted in two separate strains of methods:While the computational outlook was much improved by entropic regularization, more recent $O(n)$ linear-time \textit{low-rank} solvers hold the promise to scale up OT further.On the other hand, modelling rigidities have been eased owing to unbalanced variants of OT, that rely on penalization terms to promote, rather than impose, mass conservation.The goal of this paper is to merge these two strains, to achieve the promise of \textit{both} versatile/scalable unbalanced/low-rank OT solvers. We propose custom algorithms to implement these extensions for the linear OT problem and its Fused-Gromov-Wasserstein generalization, and demonstrate their practical relevance to challenging spatial transcriptomics matching problems.

----

## [0] Geodesic Multi-Modal Mixup for Robust Fine-Tuning

**Authors**: *Changdae Oh, Junhyuk So, Hoyoon Byun, YongTaek Lim, Minchul Shin, Jong-June Jeon, Kyungwoo Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45296e83b19f656392e0130d9e53cb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45296e83b19f656392e0130d9e53cb1-Abstract-Conference.html)

**Abstract**:

Pre-trained multi-modal models, such as CLIP, provide transferable embeddings and show promising results in diverse applications. However, the analysis of learned multi-modal embeddings is relatively unexplored, and the embedding transferability can be improved. In this work, we observe that CLIP holds separated embedding subspaces for two different modalities, and then we investigate it through the lens of \textit{uniformity-alignment} to measure the quality of learned representation. Both theoretically and empirically, we show that CLIP retains poor uniformity and alignment even after fine-tuning. Such a lack of alignment and uniformity might restrict the transferability and robustness of embeddings. To this end, we devise a new fine-tuning method for robust representation equipping better alignment and uniformity. First, we propose a \textit{Geodesic Multi-Modal Mixup} that mixes the embeddings of image and text to generate hard negative samples on the hypersphere. Then, we fine-tune the model on hard negatives as well as original negatives and positives with contrastive loss. Based on the theoretical analysis about hardness guarantee and limiting behavior, we justify the use of our method. Extensive experiments on retrieval, calibration, few- or zero-shot classification (under distribution shift), embedding arithmetic, and image captioning further show that our method provides transferable representations, enabling robust model adaptation on diverse tasks.

----

## [0] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

**Authors**: *Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html)

**Abstract**:

Large language models(LLMs) have sparked a new wave of exciting AI applications. Hosting these models at scale requires significant memory resources. One crucial memory bottleneck for the deployment stems from the context window. It is commonly recognized that model weights are memory hungry; however, the size of key-value embedding stored during the generation process (KV cache) can easily surpass the model size. The enormous size of the KV cache puts constraints on the inference batch size, which is crucial for high throughput inference workload. Inspired by an interesting observation of the attention scores, we hypothesize the persistence of importance: only pivotal tokens, which had a substantial influence at one step, will significantly influence future generations. Based on our empirical verification and theoretical analysis around this hypothesis, we propose scissorhands, a system that maintains the memory usage of the KV cache at a fixed budget without finetuning the model. In essence, Scissorhands manages the KV cache by storing the pivotal tokens with a higher probability. We validate that scissorhands reduces the inference memory usage of the KV cache by up to 5$\times$ without compromising model quality. We further demonstrate that scissorhands can be combined with 4-bit quantization, traditionally used to compress model weights, to achieve up to 20$\times$ compression.

----

## [0] Asymmetric Certified Robustness via Feature-Convex Neural Networks

**Authors**: *Samuel Pfrommer, Brendon G. Anderson, Julien Piet, Somayeh Sojoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45b205c10ef082515cacae80555bbef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45b205c10ef082515cacae80555bbef-Abstract-Conference.html)

**Abstract**:

Real-world adversarial attacks on machine learning models often feature an asymmetric structure wherein adversaries only attempt to induce false negatives (e.g., classify a spam email as not spam). We formalize the asymmetric robustness certification problem and correspondingly present the feature-convex neural network architecture, which composes an input-convex neural network (ICNN) with a Lipschitz continuous feature map in order to achieve asymmetric adversarial robustness. We consider the aforementioned binary setting with one "sensitive" class, and for this class we prove deterministic, closed-form, and easily-computable certified robust radii for arbitrary $\ell_p$-norms. We theoretically justify the use of these models by characterizing their decision region geometry, extending the universal approximation theorem for ICNN regression to the classification setting, and proving a lower bound on the probability that such models perfectly fit even unstructured uniformly distributed data in sufficiently high dimensions. Experiments on Malimg malware classification and subsets of the MNIST, Fashion-MNIST, and CIFAR-10 datasets show that feature-convex classifiers attain substantial certified $\ell_1$, $\ell_2$, and $\ell_{\infty}$-radii while being far more computationally efficient than competitive baselines.

----

## [0] A Unified Fast Gradient Clipping Framework for DP-SGD

**Authors**: *Weiwei Kong, Andrés Muñoz Medina*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45d344b28179c8da7646bc38ff50ad8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45d344b28179c8da7646bc38ff50ad8-Abstract-Conference.html)

**Abstract**:

A well-known numerical bottleneck in the differentially-private stochastic gradient descent (DP-SGD) algorithm is the computation of the gradient norm for each example in a large input batch. When the loss function in DP-SGD is consists of an intermediate linear operation, existing methods in the literature have proposed decompositions of gradients that are amenable to fast norm computations. In this paper, we present a framework that generalizes the above approach to arbitrary (possibly nonlinear) intermediate operations. Moreover, we show that for certain operations, such as fully-connected and embedding layer computations, further improvements to the runtime and storage costs of existing decompositions can be deduced using certain components of our framework. Finally, preliminary numerical experiments are given to demonstrate the substantial effects of the aforementioned improvements.

----

## [0] Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization

**Authors**: *Xiangsen Wang, Haoran Xu, Yinan Zheng, Xianyuan Zhan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) has received considerable attention in recent years due to its attractive capability of learning policies from offline datasets without environmental interactions. Despite some success in the single-agent setting, offline multi-agent RL (MARL) remains to be a challenge. The large joint state-action space and the coupled multi-agent behaviors pose extra complexities for offline policy optimization. Most existing offline MARL studies simply apply offline data-related regularizations on individual agents, without fully considering the multi-agent system at the global level. In this work, we present OMIGA, a new offline multi-agent RL algorithm with implicit global-to-local value regularization. OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. Based on comprehensive experiments on the offline multi-agent MuJoCo and StarCraft II micro-management tasks, we show that OMIGA achieves superior performance over the state-of-the-art offline MARL methods in almost all tasks.

----

## [0] Benchmarking Large Language Models on CMExam - A comprehensive Chinese Medical Exam Dataset

**Authors**: *Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, Lei Zhu, Michael Lingzhi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent advancements in large language models (LLMs) have transformed the field of question answering (QA). However, evaluating LLMs in the medical field is challenging due to the lack of standardized and comprehensive datasets. To address this gap, we introduce CMExam, sourced from the Chinese National Medical Licensing Examination. CMExam consists of 60K+ multiple-choice questions for standardized and objective evaluations, as well as solution explanations for model reasoning evaluation in an open-ended manner. For in-depth analyses of LLMs, we invited medical professionals to label five additional question-wise annotations, including disease groups, clinical departments, medical disciplines, areas of competency, and question difficulty levels. Alongside the dataset, we further conducted thorough experiments with representative LLMs and QA algorithms on CMExam. The results show that GPT-4 had the best accuracy of 61.6% and a weighted F1 score of 0.617. These results highlight a great disparity when compared to human accuracy, which stood at 71.6%. For explanation tasks, while LLMs could generate relevant reasoning and demonstrate improved performance after finetuning, they fall short of a desired standard, indicating ample room for improvement. To the best of our knowledge, CMExam is the first Chinese medical exam dataset to provide comprehensive medical annotations. The experiments and findings of LLM evaluation also provide valuable insights into the challenges and potential solutions in developing Chinese medical QA systems and LLM evaluation pipelines.

----

## [0] A Logic for Expressing Log-Precision Transformers

**Authors**: *William Merrill, Ashish Sabharwal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html)

**Abstract**:

One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed that finite-precision transformer classifiers can be equivalently expressed in a generalization of first-order logic. However, finite-precision transformers are a weak transformer variant because, as we show, a single head can only attend to a constant number of tokens and, in particular, cannot represent uniform attention. Since attending broadly is a core capability for transformers, we ask whether a minimally more expressive model that can attend universally can also be characterized in logic. To this end, we analyze transformers whose forward pass is computed in $\log n$ precision on contexts of length $n$. We prove any log-precision transformer classifier can be equivalently expressed as a first-order logic sentence that, in addition to standard universal and existential quantifiers, may also contain majority-vote quantifiers. This is the tightest known upper bound and first logical characterization of log-precision transformers.

----

## [0] Universal Prompt Tuning for Graph Neural Networks

**Authors**: *Taoran Fang, Yunchao Zhang, Yang Yang, Chunping Wang, Lei Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html)

**Abstract**:

In recent years, prompt tuning has sparked a research surge in adapting pre-trained models. Unlike the unified pre-training strategy employed in the language field, the graph field exhibits diverse pre-training strategies, posing challenges in designing appropriate prompt-based tuning methods for graph neural networks. While some pioneering work has devised specialized prompting functions for models that employ edge prediction as their pre-training tasks, these methods are limited to specific pre-trained GNN models and lack broader applicability. In this paper, we introduce a universal prompt-based tuning method called Graph Prompt Feature (GPF) for pre-trained GNN models under any pre-training strategy. GPF operates on the input graph's feature space and can theoretically achieve an equivalent effect to any form of prompting function. Consequently, we no longer need to illustrate the prompting function corresponding to each pre-training strategy explicitly. Instead, we employ GPF to obtain the prompted graph for the downstream task in an adaptive manner. We provide rigorous derivations to demonstrate the universality of GPF and make guarantee of its effectiveness. The experimental results under various pre-training strategies indicate that our method performs better than fine-tuning, with an average improvement of about 1.4% in full-shot scenarios and about 3.2% in few-shot scenarios. Moreover, our method significantly outperforms existing specialized prompt-based tuning methods when applied to models utilizing the pre-training strategy they specialize in. These numerous advantages position our method as a compelling alternative to fine-tuning for downstream adaptations.

----

## [0] Stochastic Approximation Approaches to Group Distributionally Robust Optimization

**Authors**: *Lijun Zhang, Peng Zhao, Zhen-Hua Zhuang, Tianbao Yang, Zhi-Hua Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html)

**Abstract**:

This paper investigates group distributionally robust optimization (GDRO), with the purpose to learn a model that performs well over $m$ different distributions. First, we formulate GDRO as a stochastic convex-concave saddle-point problem, and demonstrate that stochastic mirror descent (SMD), using $m$ samples in each iteration, achieves an $O(m (\log m)/\epsilon^2)$ sample complexity for finding an $\epsilon$-optimal solution, which matches the $\Omega(m/\epsilon^2)$ lower bound up to a logarithmic factor. Then, we make use of techniques from online learning to reduce the number of samples required in each round from $m$ to $1$, keeping the same sample complexity. Specifically, we cast GDRO as a two-players game where one player simply performs SMD and the other executes an online algorithm for non-oblivious multi-armed bandits. Next, we consider a more practical scenario where the number of samples that can be drawn from each distribution is different, and propose a novel formulation of weighted GDRO, which allows us to derive distribution-dependent convergence rates.  Denote by $n_i$ the sample budget for the $i$-th distribution, and assume $n_1 \geq n_2 \geq \cdots \geq n_m$. In the first approach, we incorporate non-uniform sampling into SMD such that the sample budget is satisfied in expectation, and prove that the excess risk of the $i$-th distribution decreases at an $O(\sqrt{n_1 \log m}/n_i)$ rate. In the second approach, we use mini-batches to meet the budget exactly and also reduce the variance in stochastic gradients, and then leverage stochastic mirror-prox algorithm, which can exploit small variances, to optimize a carefully designed weighted GDRO problem. Under appropriate conditions, it attains an $O((\log m)/\sqrt{n_i})$ convergence rate, which almost matches the optimal $O(\sqrt{1/n_i})$ rate of only learning from the $i$-th distribution with $n_i$ samples.

----

## [0] Learning Efficient Surrogate Dynamic Models with Graph Spline Networks

**Authors**: *Chuanbo Hua, Federico Berto, Michael Poli, Stefano Massaroli, Jinkyoo Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4c3a66ed818455b8bbe591b6a5d0f56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4c3a66ed818455b8bbe591b6a5d0f56-Abstract-Conference.html)

**Abstract**:

While complex simulations of physical systems have been widely used in engineering and scientific computing, lowering their often prohibitive computational requirements has only recently been tackled by deep learning approaches. In this paper, we present GraphSplineNets, a novel deep-learning method to speed up the forecasting of physical systems by reducing the grid size and number of iteration steps of deep surrogate models. Our method uses two differentiable orthogonal spline collocation methods to efficiently predict response at any location in time and space. Additionally, we introduce an adaptive collocation strategy in space to prioritize sampling from the most important regions. GraphSplineNets improve the accuracy-speedup tradeoff in forecasting various dynamical systems with increasing complexity, including the heat equation, damped wave propagation, Navier-Stokes equations, and real-world ocean currents in both regular and irregular domains.

----

## [0] Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing

**Authors**: *Wei Dong, Dawei Yan, Zhijun Lin, Peng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)

**Abstract**:

The advent of high-capacity pre-trained models has revolutionized problem-solving in computer vision, shifting the focus from training task-specific models to adapting pre-trained models. Consequently, effectively adapting large pre-trained models to downstream tasks in an efficient manner has become a prominent research area. Existing solutions primarily concentrate on designing lightweight adapters and their interaction with pre-trained models, with the goal of minimizing the number of parameters requiring updates. In this study, we propose a novel Adapter Re-Composing (ARC) strategy that addresses efficient pre-trained model adaptation from a fresh perspective. Our approach considers the reusability of adaptation parameters and introduces a parameter-sharing scheme. Specifically, we leverage symmetric down-/up-projections to construct bottleneck operations, which are shared across layers. By learning low-dimensional re-scaling coefficients, we can effectively re-compose layer-adaptive adapters. This parameter-sharing strategy in adapter design allows us to further reduce the number of new parameters while maintaining satisfactory performance, thereby offering a promising approach to compress the adaptation cost. We conduct experiments on 24 downstream image classification tasks using various Vision Transformer variants to evaluate our method. The results demonstrate that our approach achieves compelling transfer learning performance with a reduced parameter count. Our code is available at https://github.com/DavidYanAnDe/ARC.

----

## [0] Hardness of Low Rank Approximation of Entrywise Transformed Matrix Products

**Authors**: *Tamás Sarlós, Xingyou Song, David P. Woodruff, Richard Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4d92f656cc99f60fe1bfc98386aee34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4d92f656cc99f60fe1bfc98386aee34-Abstract-Conference.html)

**Abstract**:

Inspired by fast algorithms in natural language processing, we study low rank approximation in the entrywise transformed setting where we want to find a good rank $k$ approximation to $f(U \cdot V)$, where $U, V^\top \in \mathbb{R}^{n \times r}$ are given, $r = O(\log(n))$, and $f(x)$ is a general scalar function. Previous work in sublinear low rank approximation has shown that if both (1) $U = V^\top$ and (2) $f(x)$ is a PSD kernel function, then there is an $O(nk^{\omega-1})$ time constant relative error approximation algorithm, where $\omega \approx 2.376$ is the exponent of matrix multiplication. We give the first conditional time hardness results for this problem, demonstrating that both conditions (1) and (2) are in fact necessary for getting better than $n^{2-o(1)}$ time for a relative error low rank approximation for a wide class of functions. We give novel reductions from the Strong Exponential Time Hypothesis (SETH) that rely on lower bounding the leverage scores of flat sparse vectors and hold even when the rank of the transformed matrix $f(UV)$ and the target rank are $n^{o(1)}$, and when $U = V^\top$. Furthermore, even when $f(x) = x^p$ is a simple polynomial, we give runtime lower bounds in the case when $U \neq V^\top$ of the form $\Omega(\min(n^{2-o(1)}, \Omega(2^p)))$. Lastly, we demonstrate that our lower bounds are tight by giving an $O(n \cdot \text{poly}(k, 2^p, 1/\epsilon))$ time relative error approximation algorithm and a fast $O(n \cdot \text{poly}(k, p, 1/\epsilon))$ additive error approximation using fast tensor-based sketching. Additionally, since our low rank algorithms rely on matrix-vector product subroutines, our lower bounds extend to show that computing $f(UV)W$, for even a small matrix $W$, requires $\Omega(n^{2-o(1)})$ time.

----

## [0] Efficient Training of Energy-Based Models Using Jarzynski Equality

**Authors**: *Davide Carbone, Mengjian Hua, Simon Coste, Eric Vanden-Eijnden*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4ddb865e0a8ca3cca43fd7387b4b0da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4ddb865e0a8ca3cca43fd7387b4b0da-Abstract-Conference.html)

**Abstract**:

Energy-based models (EBMs) are generative models inspired by statistical physics with a wide range of applications in unsupervised learning.  Their performance is well measured by the cross-entropy (CE) of the model distribution relative to the data distribution. Using the CE as the objective for training is however challenging because the computation of its gradient with respect to the model parameters requires sampling the model distribution. Here we show how results for nonequilibrium thermodynamics based on Jarzynski equality together with tools from sequential Monte-Carlo sampling can be used to perform this computation efficiently and avoid the uncontrolled approximations made using the standard contrastive divergence algorithm.  Specifically, we introduce a modification of the unadjusted Langevin algorithm (ULA) in which each walker acquires a  weight that enables the estimation of the gradient of the cross-entropy at any step during GD, thereby bypassing sampling biases induced by slow mixing of ULA. We illustrate these results with numerical experiments on Gaussian mixture distributions as well as the MNIST and CIFAR-10 datasets. We show that the proposed approach outperforms  methods based on the contrastive divergence algorithm in all the considered situations.

----

## [0] High Precision Causal Model Evaluation with Conditional Randomization

**Authors**: *Chao Ma, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4dfcbcfa1f0425cd18aafa35a68019a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4dfcbcfa1f0425cd18aafa35a68019a-Abstract-Conference.html)

**Abstract**:

The gold standard for causal model evaluation involves comparing model predictions with true effects estimated from randomized controlled trials (RCT). However, RCTs are not always feasible or ethical to perform. In contrast, conditionally randomized experiments based on inverse probability weighting (IPW) offer a more realistic approach but may suffer from high estimation variance. To tackle this challenge and enhance causal model evaluation in real-world conditional randomization settings, we introduce a novel low-variance estimator for causal error, dubbed as the pairs estimator. By applying the same IPW estimator to both the model and true experimental effects, our estimator effectively cancels out the variance due to IPW and achieves a smaller asymptotic variance. Empirical studies demonstrate the improved of our estimator, highlighting its potential on achieving near-RCT performance. Our method offers a simple yet powerful solution to evaluate causal inference models in conditional randomization settings without complicated modification of the IPW estimator itself, paving the way for more robust and reliable model assessments.

----

## [0] Reducing Blackwell and Average Optimality to Discounted MDPs via the Blackwell Discount Factor

**Authors**: *Julien Grand-Clément, Marek Petrik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4e720ce31ccd8ba747d8863e1580fa8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4e720ce31ccd8ba747d8863e1580fa8-Abstract-Conference.html)

**Abstract**:

We introduce the Blackwell discount factor for Markov Decision Processes (MDPs). Classical objectives for MDPs include discounted, average, and Blackwell optimality. Many existing approaches to computing average-optimal policies solve for discount-optimal policies with a discount factor close to $1$, but they only work under strong or hard-to-verify assumptions on the MDP structure  such as unichain or ergodicity. We are the first to highlight the shortcomings of the classical definition of Blackwell optimality, which does not lead to simple algorithms for computing Blackwell-optimal policies and overlooks the pathological behaviors of optimal policies as regards the discount factors. To resolve this issue, in this paper, we show that when the discount factor is larger than  the Blackwell discount factor $\gamma_{\sf bw}$, all discount-optimal policies become Blackwell- and average-optimal, and we derive a general upper bound on $\gamma_{\sf bw}$. Our upper bound on $\gamma_{\sf bw}$, parametrized by the bit-size of the rewards and transition probabilities of the MDP instance, provides the first reduction from average and Blackwell optimality to discounted optimality, without any assumptions, along with new polynomial-time algorithms. Our work brings new ideas from polynomials and algebraic numbers to the analysis of MDPs. Our results also apply to robust MDPs, enabling the first algorithms to compute robust Blackwell-optimal policies.

----

## [0] Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits

**Authors**: *Muhammad Faaiz Taufiq, Arnaud Doucet, Rob Cornish, Jean-Francois Ton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a51f974947c42b40a40a882a7d9b2479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a51f974947c42b40a40a882a7d9b2479-Abstract-Conference.html)

**Abstract**:

Off-Policy Evaluation (OPE) in contextual bandits is crucial for assessing new policies using existing data without costly experimentation. However, current OPE methods, such as Inverse Probability Weighting (IPW) and Doubly Robust (DR) estimators, suffer from high variance, particularly in cases of low overlap between target and behaviour policies or large action and context spaces. In this paper, we introduce a new OPE estimator for contextual bandits, the Marginal Ratio (MR) estimator, which focuses on the shift in the marginal distribution of outcomes $Y$ instead of the policies themselves. Through rigorous theoretical analysis, we demonstrate the benefits of the MR estimator compared to conventional methods like IPW and DR in terms of variance reduction. Additionally, we establish a connection between the MR estimator and the state-of-the-art Marginalized Inverse Propensity Score (MIPS) estimator, proving that MR achieves lower variance among a generalized family of MIPS estimators. We further illustrate the utility of the MR estimator in causal inference settings, where it exhibits enhanced performance in estimating Average Treatment Effects (ATE). Our experiments on synthetic and real-world datasets corroborate our theoretical findings and highlight the practical advantages of the MR estimator in OPE for contextual bandits.

----

## [0] SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs

**Authors**: *Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin P. Murphy, Alexander G. Hauptmann, Lu Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a526cc8f6ffb74bedb6ff313e3fdb450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a526cc8f6ffb74bedb6ff313e3fdb450-Abstract-Conference.html)

**Abstract**:

In this work, we introduce Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform both understanding and generation tasks involving non-linguistic modalities such as images or videos. SPAE converts between raw pixels and interpretable lexical tokens (or words) extracted from the LLM's vocabulary. The resulting tokens capture both the rich semantic meaning and the fine-grained details needed for visual reconstruction, effectively translating the visual content into a language comprehensible to the LLM, and empowering it to perform a wide array of multimodal tasks. Our approach is validated through in-context learning experiments with frozen PaLM 2 and GPT 3.5 on a diverse set of image understanding and generation tasks.Our method marks the first successful attempt to enable a frozen LLM to generate image content while surpassing state-of-the-art performance in image understanding tasks, under the same setting, by over 25%.

----

## [0] Energy-based learning algorithms for analog computing: a comparative study

**Authors**: *Benjamin Scellier, Maxence Ernoult, Jack D. Kendall, Suhas Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html)

**Abstract**:

Energy-based learning algorithms have recently gained a surge of interest due to their compatibility with analog (post-digital) hardware. Existing algorithms include contrastive learning (CL), equilibrium propagation (EP) and coupled learning (CpL), all consisting in contrasting two states, and differing in the type of perturbation used to obtain the second state from the first one. However, these algorithms have never been explicitly compared on equal footing with same models and datasets, making it difficult to assess their scalability and decide which one to select in practice. In this work, we carry out a comparison of seven learning algorithms, namely CL and different variants of EP and CpL depending on the signs of the perturbations. Specifically, using these learning algorithms, we train deep convolutional Hopfield networks (DCHNs) on five vision tasks (MNIST, F-MNIST, SVHN, CIFAR-10 and CIFAR-100). We find that, while all algorithms yield comparable performance on MNIST, important differences in performance arise as the difficulty of the task increases. Our key findings reveal that negative perturbations are better than positive ones, and highlight the centered variant of EP (which uses two perturbations of opposite sign) as the best-performing algorithm. We also endorse these findings with theoretical arguments. Additionally, we establish new SOTA results with DCHNs on all five datasets, both in performance and speed. In particular, our DCHN simulations are 13.5 times faster with respect to Laborieux et al. (2021), which we achieve thanks to the use of a novel energy minimisation algorithm based on asynchronous updates, combined with reduced precision (16 bits).

----

## [0] Distribution Learnability and Robustness

**Authors**: *Shai Ben-David, Alex Bie, Gautam Kamath, Tosca Lechner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5321f64005b0d4a94d0b18e84e19f48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5321f64005b0d4a94d0b18e84e19f48-Abstract-Conference.html)

**Abstract**:

We examine the relationship between learnability and robust learnability for the problem of distribution learning.We show that learnability implies robust learnability if the adversary can only perform additive contamination (and consequently, under Huber contamination), but not if the adversary is allowed to perform subtractive contamination. Thus, contrary to other learning settings (e.g., PAC learning of function classes), realizable learnability does not imply agnostic learnability. We also explore related implications in the context of compression schemes and  differentially private learnability.

----

## [0] Behavior Alignment via Reward Function Optimization

**Authors**: *Dhawal Gupta, Yash Chandak, Scott M. Jordan, Philip S. Thomas, Bruno C. da Silva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html)

**Abstract**:

Designing reward functions for efficiently guiding reinforcement learning (RL) agents toward specific behaviors is a complex task.This is challenging since it requires the identification of reward structures that are not sparse and that avoid inadvertently inducing undesirable behaviors. Naively modifying the reward structure to offer denser and more frequent feedback can lead to unintended outcomes and promote behaviors that are not aligned with the designer's intended goal. Although potential-based reward shaping is often suggested as a remedy, we systematically investigate settings where deploying it often significantly impairs performance. To address these issues, we introduce a new framework that uses a bi-level objective to learn \emph{behavior alignment reward functions}. These functions integrate auxiliary rewards reflecting a designer's heuristics and domain knowledge with the environment's primary rewards. Our approach automatically determines the most effective way to blend these types of feedback, thereby enhancing robustness against heuristic reward misspecification. Remarkably, it can also adapt an agent's policy optimization process to mitigate suboptimalities resulting from limitations and biases inherent in the underlying RL algorithms. We evaluate our method's efficacy on a diverse set of tasks, from small-scale experiments to high-dimensional control challenges. We investigate heuristic auxiliary rewards of varying quality---some of which are beneficial and others detrimental to the learning process. Our results show that our framework offers a robust and principled way to integrate designer-specified heuristics. It not only addresses key shortcomings of existing approaches but also consistently leads to high-performing solutions, even when given misaligned or poorly-specified auxiliary reward functions.

----

## [0] Tuning Multi-mode Token-level Prompt Alignment across Modalities

**Authors**: *Dongsheng Wang, Miaoge Li, Xinyang Liu, Mingsheng Xu, Bo Chen, Hanwang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a547d86953a4e36aa8a1390e6f4708e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a547d86953a4e36aa8a1390e6f4708e2-Abstract-Conference.html)

**Abstract**:

Advancements in prompt tuning of vision-language models have underscored their potential in enhancing open-world visual concept comprehension. However, prior works only primarily focus on single-mode (only one prompt for each modality) and holistic level (image or sentence) semantic alignment, which fails to capture the sample diversity, leading to sub-optimal prompt discovery. To address the limitation, we propose a multi-mode token-level tuning framework that leverages the optimal transportation to learn and align a set of prompt tokens across modalities. Specifically, we rely on two essential factors: 1) multi-mode prompts discovery, which guarantees diverse semantic representations, and 2) token-level alignment, which helps explore fine-grained similarity. Consequently, the similarity can be calculated as a hierarchical transportation problem between the modality-specific sets. Extensive experiments on popular image recognition benchmarks show the superior generalization and few-shot abilities of our approach. The qualitative analysis demonstrates that the learned prompt tokens have the ability to capture diverse visual concepts.

----

## [0] Censored Sampling of Diffusion Models Using 3 Minutes of Human Feedback

**Authors**: *Taeho Yoon, Kibeom Myoung, Keon Lee, Jaewoong Cho, Albert No, Ernest K. Ryu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5755ccd0efeca8852ae0a1193f319f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5755ccd0efeca8852ae0a1193f319f6-Abstract-Conference.html)

**Abstract**:

Diffusion models have recently shown remarkable success in high-quality image generation. Sometimes, however, a pre-trained diffusion model exhibits partial misalignment in the sense that the model can generate good images, but it sometimes outputs undesirable images. If so, we simply need to prevent the generation of the bad images, and we call this task censoring. In this work, we present censored generation with a pre-trained diffusion model using a reward model trained on minimal human feedback. We show that censoring can be accomplished with extreme human feedback efficiency and that labels generated with a mere few minutes of human feedback are sufficient.

----

## [0] Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics

**Authors**: *Leon Klein, Andrew Y. K. Foong, Tor Erlend Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, Ryota Tomioka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a598c367280f9054434fdcc227ce4d38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a598c367280f9054434fdcc227ce4d38-Abstract-Conference.html)

**Abstract**:

*Molecular dynamics* (MD) simulation is a widely used technique to simulate molecular systems, most commonly at the all-atom resolution where equations of motion are integrated with timesteps on the order of femtoseconds ($1\textrm{fs}=10^{-15}\textrm{s}$). MD is often used to compute equilibrium properties, which requires sampling from an equilibrium distribution such as the Boltzmann distribution. However, many important processes, such as binding and folding, occur over timescales of milliseconds or beyond, and cannot be efficiently sampled with conventional MD.Furthermore, new MD simulations need to be performed for each molecular system studied.We present *Timewarp*, an enhanced sampling method which uses a normalising flow as a proposal distribution in a Markov chain Monte Carlo method targeting the Boltzmann distribution. The flow is trained offline on MD trajectories and learns to make large steps in time, simulating the molecular dynamics of $10^{5} - 10^{6} \textrm{fs}$.Crucially, Timewarp is *transferable* between molecular systems: once trained, we show that it generalises to unseen small peptides (2-4 amino acids) at all-atom resolution, exploring their metastable states and providing wall-clock acceleration of sampling compared to standard MD.Our method constitutes an important step towards general, transferable algorithms for accelerating MD.

----



[Go to the previous page](NIPS-2023-list2200.md)

[Go to the next page](NIPS-2023-list2202.md)

[Go to the catalog section](README.md)