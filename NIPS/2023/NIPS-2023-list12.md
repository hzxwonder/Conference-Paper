## [2200] HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception

        **Authors**: *Junkun Yuan, Xinyu Zhang, Hao Zhou, Jian Wang, Zhongwei Qiu, Zhiyin Shao, Shaofeng Zhang, Sifan Long, Kun Kuang, Kun Yao, Junyu Han, Errui Ding, Lanfen Lin, Fei Wu, Jingdong Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ed1c94a6c87276f25ebb65231c86c3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ed1c94a6c87276f25ebb65231c86c3e-Abstract-Conference.html)

        **Abstract**:

        Model pre-training is essential in human-centric perception. In this paper, we first introduce masked image modeling (MIM) as a pre-training approach for this task. Upon revisiting the MIM training strategy, we reveal that human structure priors offer significant potential. Motivated by this insight, we further incorporate an intuitive human structure prior - human parts - into pre-training. Specifically, we employ this prior to guide the mask sampling process. Image patches, corresponding to human part regions, have high priority to be masked out. This encourages the model to concentrate more on body structure information during pre-training, yielding substantial benefits across a range of human-centric perception tasks. To further capture human characteristics, we propose a structure-invariant alignment loss that enforces different masked views, guided by the human part prior, to be closely aligned for the same image. We term the entire method as HAP. HAP simply uses a plain ViT as the encoder yet establishes new state-of-the-art performance on 11 human-centric benchmarks, and on-par result on one dataset. For example, HAP achieves 78.1% mAP on MSMT17 for person re-identification, 86.54% mA on PA-100K for pedestrian attribute recognition, 78.2% AP on MS COCO for 2D pose estimation, and 56.0 PA-MPJPE on 3DPW for 3D pose and shape estimation.

        ----

        ## [2201] Trust Your 𝛁: Gradient-based Intervention Targeting for Causal Discovery

        **Authors**: *Mateusz Olko, Michal Zajac, Aleksandra Nowak, Nino Scherrer, Yashas Annadani, Stefan Bauer, Lukasz Kucinski, Piotr Milos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9eda77f505efbb89462970d739143f73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9eda77f505efbb89462970d739143f73-Abstract-Conference.html)

        **Abstract**:

        Inferring causal structure from data is a challenging task of fundamental importance in science. Often, observational data alone is not enough to uniquely identify a system’s causal structure. The use of interventional data can address this issue, however, acquiring these samples typically demands a considerable investment of time and physical or financial resources. In this work, we are concerned with the acquisition of interventional data in a targeted manner to minimize the number of required experiments. We propose a novel Gradient-based Intervention Targeting method, abbreviated GIT, that ’trusts’ the gradient estimator of a gradient-based causal discovery framework to provide signals for the intervention targeting function. We provide extensive experiments in simulated and real-world datasets and demonstrate that GIT performs on par with competitive baselines, surpassing them in the low-data regime.

        ----

        ## [2202] SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions

        **Authors**: *Yuseung Lee, Kunho Kim, Hyunjin Kim, Minhyuk Sung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract-Conference.html)

        **Abstract**:

        The remarkable capabilities of pretrained image diffusion models have been utilized not only for generating fixed-size images but also for creating panoramas. However, naive stitching of multiple images often results in visible seams. Recent techniques have attempted to address this issue by performing joint diffusions in multiple windows and averaging latent features in overlapping regions. However, these approaches, which focus on seamless montage generation, often yield incoherent outputs by blending different scenes within a single image. To overcome this limitation, we propose SyncDiffusion, a plug-and-play module that synchronizes multiple diffusions through gradient descent from a perceptual similarity loss. Specifically, we compute the gradient of the perceptual loss using the predicted denoised images at each denoising step, providing meaningful guidance for achieving coherent montages. Our experimental results demonstrate that our method produces significantly more coherent outputs compared to previous methods (66.35% vs. 33.65% in our user study) while still maintaining fidelity (as assessed by GIQA) and compatibility with the input prompt (as measured by CLIP score). We further demonstrate the versatility of our method across three plug-and-play applications: layout-guided image generation, conditional image generation and 360-degree panorama generation. Our project page is at https://syncdiffusion.github.io.

        ----

        ## [2203] Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean

        **Authors**: *Spyridon Kondylatos, Ioannis Prapas, Gustau Camps-Valls, Ioannis Papoutsis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ee3ed2dd656402f954ef9dc37e39f48-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ee3ed2dd656402f954ef9dc37e39f48-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce Mesogeos, a large-scale multi-purpose dataset for wildfire modeling in the Mediterranean. Mesogeos integrates variables representing wildfire drivers (meteorology, vegetation, human activity) and historical records of wildfire ignitions and burned areas for 17 years (2006-2022). It is designed as a cloud-friendly spatio-temporal dataset, namely a datacube, harmonizing all variables in a grid of 1km x 1km x 1-day resolution. The datacube structure offers opportunities to assess machine learning (ML) usage in various wildfire modeling tasks. We extract two ML-ready datasets that establish distinct tracks to demonstrate this potential: (1) short-term wildfire danger forecasting and (2) final burned area estimation given the point of ignition. We define appropriate metrics and baselines to evaluate the performance of models in each track. By publishing the datacube, along with the code to create the ML datasets and models, we encourage the community to foster the implementation of additional tracks for mitigating the increasing threat of wildfires in the Mediterranean.

        ----

        ## [2204] Deep learning with kernels through RKHM and the Perron-Frobenius operator

        **Authors**: *Yuka Hashimoto, Masahiro Ikeda, Hachem Kadri*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ef237e007e26180ce4d16738efdf83f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ef237e007e26180ce4d16738efdf83f-Abstract-Conference.html)

        **Abstract**:

        Reproducing kernel Hilbert $C^*$-module (RKHM) is a generalization of reproducing kernel Hilbert space (RKHS) by means of $C^*$-algebra, and the Perron-Frobenius operator is a linear operator related to the composition of functions. Combining these two concepts, we present deep RKHM, a deep learning framework for kernel methods. We derive a new Rademacher generalization bound in this setting and provide a theoretical interpretation of benign overfitting by means of Perron-Frobenius operators. By virtue of $C^*$-algebra, the dependency of the bound on output dimension is milder than existing bounds. We show that $C^*$-algebra is a suitable tool for deep learning with kernels, enabling us to take advantage of the product structure of operators and to provide a clear connection with convolutional neural networks. Our theoretical analysis provides a new lens through which one can design and analyze deep kernel methods.

        ----

        ## [2205] SmoothHess: ReLU Network Feature Interactions via Stein's Lemma

        **Authors**: *Max Torop, Aria Masoomi, Davin Hill, Kivanç Köse, Stratis Ioannidis, Jennifer G. Dy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ef5e965720193681fc8d16372ac4717-Abstract-Conference.html)

        **Abstract**:

        Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.

        ----

        ## [2206] MLFMF: Data Sets for Machine Learning for Mathematical Formalization

        **Authors**: *Andrej Bauer, Matej Petkovic, Ljupco Todorovski*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9efe8db7fab57e19eed25718abedbbd2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9efe8db7fab57e19eed25718abedbbd2-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.

        ----

        ## [2207] DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

        **Authors**: *Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, Phillip Isola*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html)

        **Abstract**:

        Current perceptual similarity metrics operate at the level of pixels and patches. These metrics compare images in terms of their low-level colors and textures, but fail to capture mid-level similarities and differences in image layout, object pose, and semantic content. In this paper, we develop a perceptual metric that assesses images holistically. Our first step is to collect a new dataset of human similarity judgments over image pairs that are alike in diverse ways. Critical to this dataset is that judgments are nearly automatic and shared by all observers. To achieve this we use recent text-to-image models to create synthetic pairs that are perturbed along various dimensions. We observe that popular perceptual metrics fall short of explaining our new data, and we introduce a new metric, DreamSim, tuned to better align with human perception. We analyze how our metric is affected by different visual attributes, and find that it focuses heavily on foreground objects and semantic content while also being sensitive to color and layout. Notably, despite being trained on synthetic data, our metric generalizes to real images, giving strong results on retrieval and reconstruction tasks. Furthermore, our metric outperforms both prior learned metrics and recent large vision models on these tasks. Our project page: https://dreamsim-nights.github.io/

        ----

        ## [2208] Explaining the Uncertain: Stochastic Shapley Values for Gaussian Process Models

        **Authors**: *Siu Lun Chau, Krikamol Muandet, Dino Sejdinovic*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f0b1220028dfa2ee82ca0a0e0fc52d1-Abstract-Conference.html)

        **Abstract**:

        We present a novel approach for explaining Gaussian processes (GPs) that can utilize the full analytical covariance structure present in GPs. Our method is based on the popular solution concept of Shapley values extended to stochastic cooperative games, resulting in explanations that are random variables. The GP explanations generated using our approach satisfy similar favorable axioms to standard Shapley values and possess a tractable covariance function across features and data observations. This covariance allows for quantifying explanation uncertainties and studying the statistical dependencies between explanations. We further extend our framework to the problem of predictive explanation, and propose a Shapley prior over the explanation function to predict Shapley values for new data based on previously computed ones. Our extensive illustrations demonstrate the effectiveness of the proposed approach.

        ----

        ## [2209] WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes

        **Authors**: *Satoshi Tsutsui, Winnie Pang, Bihan Wen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f34484e5b8d87f09cc58c292a1c9f5d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f34484e5b8d87f09cc58c292a1c9f5d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The examination of blood samples at a microscopic level plays a fundamental role in clinical diagnostics. For instance, an in-depth study of White Blood Cells (WBCs), a crucial component of our blood, is essential for diagnosing blood-related diseases such as leukemia and anemia. While multiple datasets containing WBC images have been proposed, they mostly focus on cell categorization, often lacking the necessary morphological details to explain such categorizations, despite the importance of explainable artificial intelligence (XAI) in medical domains. This paper seeks to address this limitation by introducing comprehensive annotations for WBC images. Through collaboration with pathologists, a thorough literature review, and manual inspection of microscopic images, we have identified 11 morphological attributes associated with the cell and its components (nucleus, cytoplasm, and granules). We then annotated ten thousand WBC images with these attributes, resulting in 113k labels (11 attributes x 10.3k images). Annotating at this level of detail and scale is unprecedented, offering unique value to AI in pathology. Moreover, we conduct experiments to predict these attributes from cell images, and also demonstrate specific applications that can benefit from our detailed annotations. Overall, our dataset paves the way for interpreting WBC recognition models, further advancing XAI in the fields of pathology and hematology.

        ----

        ## [2210] Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling

        **Authors**: *Haotao Wang, Ziyu Jiang, Yuning You, Yan Han, Gaowen Liu, Jayanth Srinivasa, Ramana Kompella, Zhangyang Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f4064d145bad5e361206c3303bda7b8-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs) have found extensive applications in learning from graph data. However, real-world graphs often possess diverse structures and comprise nodes and edges of varying types. To bolster the generalization capacity of GNNs, it has become customary to augment training graph structures through techniques like graph augmentations and large-scale pre-training on a wider array of graphs. Balancing this diversity while avoiding increased computational costs and the notorious trainability issues of GNNs is crucial. This study introduces the concept of Mixture-of-Experts (MoE) to GNNs, with the aim of augmenting their capacity to adapt to a diverse range of training graph structures, without incurring explosive computational overhead. The proposed Graph Mixture of Experts (GMoE) model empowers individual nodes in the graph to dynamically and adaptively select more general information aggregation experts. These experts are trained to capture distinct subgroups of graph structures and to incorporate information with varying hop sizes, where those with larger hop sizes specialize in gathering information over longer distances. The effectiveness of GMoE is validated through a series of experiments on a diverse set of tasks, including graph, node, and link prediction, using the OGB benchmark. Notably, it enhances ROC-AUC by $1.81\%$ in ogbg-molhiv and by $1.40\%$ in ogbg-molbbbp, when compared to the non-MoE baselines. Our code is publicly available at https://github.com/VITA-Group/Graph-Mixture-of-Experts.

        ----

        ## [2211] Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction

        **Authors**: *Quentin Delfosse, Hikaru Shindo, Devendra Singh Dhami, Kristian Kersting*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f42f06a54ce3b709ad78d34c73e4363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f42f06a54ce3b709ad78d34c73e4363-Abstract-Conference.html)

        **Abstract**:

        The limited priors required by neural networks make them the dominating choice to encode and learn policies using reinforcement learning (RL). However, they are also black-boxes, making it hard to understand the agent's behavior, especially when working on the image level. Therefore, neuro-symbolic RL aims at creating policies that are interpretable in the first place.Unfortunately, interpretability is not explainability. To achieve both, we introduce Neurally gUided Differentiable loGic policiEs (NUDGE). NUDGE exploits trained neural network-based agents to guide the search of candidate-weighted logic rules, then uses differentiable logic to train the logic agents. Our experimental evaluation demonstrates that NUDGE agents can induce interpretable and explainable policies while outperforming purely neural ones and showing good flexibility to environments of different initial states and problem sizes.

        ----

        ## [2212] Personalized Dictionary Learning for Heterogeneous Datasets

        **Authors**: *Geyu Liang, Naichen Shi, Raed Al Kontar, Salar Fattahi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f6f790f28a31fba89644f09faf4e0cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f6f790f28a31fba89644f09faf4e0cb-Abstract-Conference.html)

        **Abstract**:

        We introduce a relevant yet challenging problem named Personalized Dictionary Learning (PerDL), where the goal is to learn sparse linear representations from heterogeneous datasets that share some commonality. In PerDL, we model each dataset's shared and unique features as global and local dictionaries. Challenges for PerDL not only are inherited from classical dictionary learning(DL), but also arise due to the unknown nature of the shared and unique features. In this paper, we rigorously formulate this problem and provide conditions under which the global and local dictionaries can be provably disentangled. Under these conditions, we provide a meta-algorithm called Personalized Matching and Averaging (PerMA) that can recover both global and local dictionaries from heterogeneous datasets. PerMA is highly efficient; it converges to the ground truth at a linear rate under suitable conditions. Moreover, it automatically borrows strength from strong learners to improve the prediction of weak learners. As a general framework for extracting global and local dictionaries, we show the application of PerDL in different learning tasks, such as training with imbalanced datasets and video surveillance.

        ----

        ## [2213] Graph-Structured Gaussian Processes for Transferable Graph Learning

        **Authors**: *Jun Wu, Lisa Ainsworth, Andrew Leakey, Haixun Wang, Jingrui He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f7f2f57d8eaf44b2f09020f64ff6d96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f7f2f57d8eaf44b2f09020f64ff6d96-Abstract-Conference.html)

        **Abstract**:

        Transferable graph learning involves knowledge transferability from a source graph to a relevant target graph. The major challenge of transferable graph learning is the distribution shift between source and target graphs induced by individual node attributes and complex graph structures. To solve this problem, in this paper, we propose a generic graph-structured Gaussian process framework (GraphGP) for adaptively transferring knowledge across graphs with either homophily or heterophily assumptions. Specifically, GraphGP is derived from a novel graph structure-aware neural network in the limit on the layer width. The generalization analysis of GraphGP explicitly investigates the connection between knowledge transferability and graph domain similarity. Extensive experiments on several transferable graph learning benchmarks demonstrate the efficacy of GraphGP over state-of-the-art Gaussian process baselines.

        ----

        ## [2214] Language Models are Weak Learners

        **Authors**: *Hariharan Manikandan, Yiding Jiang, J. Zico Kolter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9f94298bac4668db4dc77ddb0a244301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9f94298bac4668db4dc77ddb0a244301-Abstract-Conference.html)

        **Abstract**:

        A central notion in practical and theoretical machine learning is that of a weak learner, classifiers that achieve better-than-random performance (on any given distribution over data), even by a small margin.  Such weak learners form the practical basis for canonical machine learning methods such as boosting.  In this work, we illustrate that prompt-based large language models can operate effectively as said weak learners.  Specifically, we illustrate the use of a large language model (LLM) as a weak learner in a boosting algorithm applied to tabular data.  We show that by providing (properly sampled according to the distribution of interest) text descriptions of tabular data samples, LLMs can produce a summary of the samples that serves as a template for classification, and achieves the aim of acting as a weak learner on this task.  We incorporate these models into a boosting approach, which in many settings can leverage the knowledge within the LLM to outperform traditional tree-based boosting.  The model outperforms both few-shot learning and occasionally even more involved fine-tuning procedures, particularly for some tasks involving small numbers of data points.  The results illustrate the potential for prompt-based LLMs to function not just as few-shot learners themselves, but as components of larger machine learning models.

        ----

        ## [2215] SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models

        **Authors**: *Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, Animesh Garg*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9fa03b16dbd6cabc7601fe98c6ec291e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9fa03b16dbd6cabc7601fe98c6ec291e-Abstract-Conference.html)

        **Abstract**:

        Object-centric learning aims to represent visual data with a set of object entities (a.k.a. slots), providing structured representations that enable systematic generalization.Leveraging advanced architectures like Transformers, recent approaches have made significant progress in unsupervised object discovery.In addition, slot-based representations hold great potential for generative modeling, such as controllable image generation and object manipulation in image editing.However, current slot-based methods often produce blurry images and distorted objects, exhibiting poor generative modeling capabilities.In this paper, we focus on improving slot-to-image decoding, a crucial aspect for high-quality visual generation.We introduce SlotDiffusion -- an object-centric Latent Diffusion Model (LDM) designed for both image and video data.Thanks to the powerful modeling capacity of LDMs, SlotDiffusion surpasses previous slot models in unsupervised object segmentation and visual generation across six datasets.Furthermore, our learned object features can be utilized by existing object-centric dynamics models, improving video prediction quality and downstream temporal reasoning tasks.Finally, we demonstrate the scalability of SlotDiffusion to unconstrained real-world datasets such as PASCAL VOC and COCO, when integrated with self-supervised pre-trained image encoders.

        ----

        ## [2216] ZoomTrack: Target-aware Non-uniform Resizing for Efficient Visual Tracking

        **Authors**: *Yutong Kou, Jin Gao, Bing Li, Gang Wang, Weiming Hu, Yizheng Wang, Liang Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9fc291fef2f9607a46777d367f900a15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9fc291fef2f9607a46777d367f900a15-Abstract-Conference.html)

        **Abstract**:

        Recently, the transformer has enabled the speed-oriented trackers to approach state-of-the-art (SOTA) performance with high-speed thanks to the smaller input size or the lighter feature extraction backbone, though they still substantially lag behind their corresponding performance-oriented versions. In this paper, we demonstrate that it is possible to narrow or even close this gap while achieving high tracking speed based on the smaller input size. To this end, we non-uniformly resize the cropped image to have a smaller input size while the resolution of the area where the target is more likely to appear is higher and vice versa. This enables us to solve the dilemma of attending to a larger visual field while retaining more raw information for the target despite a smaller input size. Our formulation for the non-uniform resizing can be efficiently solved through quadratic programming (QP) and naturally integrated into most of the crop-based local trackers. Comprehensive experiments on five challenging datasets based on two kinds  of transformer trackers, \ie, OSTrack and TransT, demonstrate consistent improvements over them. In particular, applying our method to the speed-oriented version of OSTrack even outperforms its performance-oriented counterpart by 0.6\% AUC on TNL2K, while running 50\% faster and saving over 55\% MACs. Codes and models are available at https://github.com/Kou-99/ZoomTrack.

        ----

        ## [2217] Improving neural network representations using human similarity judgments

        **Authors**: *Lukas Muttenthaler, Lorenz Linhardt, Jonas Dippel, Robert A. Vandermeulen, Katherine L. Hermann, Andrew K. Lampinen, Simon Kornblith*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks have reached human-level performance on many computer vision tasks. However, the objectives used to train these networks enforce only that similar images are embedded at similar locations in the representation space, and do not directly constrain the global structure of the resulting space. Here, we explore the impact of supervising this global structure by linearly aligning it with human similarity judgments. We find that a naive approach leads to large changes in local representational structure that harm downstream performance. Thus, we propose a novel method that aligns the global structure of representations while preserving their local structure. This global-local transform considerably improves accuracy across a variety of few-shot learning and anomaly detection tasks. Our results indicate that human visual representations are globally organized in a way that facilitates learning from few examples, and incorporating this global structure into neural network representations improves performance on downstream tasks.

        ----

        ## [2218] Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery

        **Authors**: *Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html)

        **Abstract**:

        The strength of modern generative models lies in their ability to be controlled through prompts. Hard prompts comprise interpretable words and tokens, and are typically hand-crafted by humans.  Soft prompts, on the other hand, consist of continuous feature vectors.  These can be discovered using powerful optimization methods, but they cannot be easily edited, re-used across models, or plugged into a text-based interface. We describe an easy-to-use approach to automatically optimize hard text prompts through efficient gradient-based optimization. Our approach can be readily applied to text-to-image and text-only applications alike. This method allows API users to easily generate, discover, and mix and match image concepts without prior knowledge of how to prompt the model. Furthermore, using our method, we can bypass token-level content filters imposed by Midjourney by optimizing through the open-sourced text encoder.

        ----

        ## [2219] Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm

        **Authors**: *Jie Hao, Kaiyi Ji, Mingrui Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0251e494a7e75d59e06d37e646f46b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0251e494a7e75d59e06d37e646f46b7-Abstract-Conference.html)

        **Abstract**:

        Coreset is a small set that provides a data summary for a large dataset, such that training solely on the small set achieves competitive performance compared with a large dataset. In rehearsal-based continual learning, the coreset is typically used in the memory replay buffer to stand for representative samples in previous tasks, and the coreset selection procedure is typically formulated as a bilevel problem. However, the typical bilevel formulation for coreset selection explicitly performs optimization over discrete decision variables with greedy search, which is computationally expensive. Several works consider other formulations to address this issue, but they ignore the nested nature of bilevel optimization problems and may not solve the bilevel coreset selection problem accurately. To address these issues, we propose a new bilevel formulation, where the inner problem tries to find a model which minimizes the expected training error sampled from a given probability distribution, and the outer problem aims to learn the probability distribution with approximately $K$ (coreset size) nonzero entries such that learned model in the inner problem minimizes the training error over the whole data. To ensure the learned probability has approximately $K$ nonzero entries, we introduce a novel regularizer based on the smoothed top-$K$ loss in the upper problem. We design a new optimization algorithm that provably converges to the $\epsilon$-stationary point with $O(1/\epsilon^4)$ computational complexity. We conduct extensive experiments in various settings in continual learning, including balanced data, imbalanced data, and label noise, to show that our proposed formulation and new algorithm significantly outperform competitive baselines. From bilevel optimization point of view, our algorithm significantly improves the vanilla greedy coreset selection method in terms of running time on continual learning benchmark datasets. The code is available at https://github.com/MingruiLiu-ML-Lab/Bilevel-Coreset-Selection-via-Regularization.

        ----

        ## [2220] HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork

        **Authors**: *Bipasha Sen, Gaurav Singh, Aditya Agarwal, Rohith Agaram, K. Madhava Krishna, Srinath Sridhar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a03037317560b8c5f2fb4b6466d4c439-Abstract-Conference.html)

        **Abstract**:

        Neural Radiance Fields (NeRF) have become an increasingly popular representation to capture high-quality appearance and shape of scenes and objects. However, learning generalizable NeRF priors over categories of scenes or objects has been challenging due to the high dimensionality of network weight space. To address the limitations of existing work on generalization, multi-view consistency and to improve quality, we propose HyP-NeRF, a latent conditioning method for learning generalizable category-level NeRF priors using hypernetworks. Rather than using hypernetworks to estimate only the weights of a NeRF, we estimate both the weights and the multi-resolution hash encodings resulting in significant quality gains. To improve quality even further, we incorporate a denoise and finetune strategy that denoises images rendered from NeRFs estimated by the hypernetwork and finetunes it while retaining multiview consistency. These improvements enable us to use HyP-NeRF as a generalizable prior for multiple downstream tasks including NeRF reconstruction from single-view or cluttered scenes and text-to-NeRF. We provide qualitative comparisons and evaluate HyP-NeRF on three tasks: generalization, compression, and retrieval, demonstrating our state-of-the-art results.

        ----

        ## [2221] MultiVENT: Multilingual Videos of Events and Aligned Natural Text

        **Authors**: *Kate Sanders, David Etter, Reno Kriz, Benjamin Van Durme*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a054ff49751dbc991ec30ae479397c3d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a054ff49751dbc991ec30ae479397c3d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Everyday news coverage has shifted from traditional broadcasts towards a wide range of presentation formats such as first-hand, unedited video footage. Datasets that reflect the diverse array of multimodal, multilingual news sources available online could be used to teach models to benefit from this shift, but existing news video datasets focus on traditional news broadcasts produced for English-speaking audiences. We address this limitation by constructing MultiVENT, a dataset of multilingual, event-centric videos grounded in text documents across five target languages. MultiVENT includes both news broadcast videos and non-professional event footage, which we use to analyze the state of online news videos and how they can be leveraged to build robust, factually accurate models. Finally, we provide a  model for complex, multilingual video retrieval to serve as a baseline for information retrieval using MultiVENT.

        ----

        ## [2222] GEO-Bench: Toward Foundation Models for Earth Monitoring

        **Authors**: *Alexandre Lacoste, Nils Lehmann, Pau Rodríguez, Evan D. Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vázquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.

        ----

        ## [2223] Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

        **Authors**: *Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, Kai Han*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html)

        **Abstract**:

        In the past years, YOLO-series models have emerged as the leading approaches in the area of real-time object detection. Many studies pushed up the baseline to a higher level by modifying the architecture, augmenting data and designing new losses. However, we find previous models still suffer from information fusion problem, although Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) have alleviated this. Therefore, this study provides an advanced Gatherand-Distribute mechanism (GD) mechanism, which is realized with convolution and self-attention operations. This new designed model named as Gold-YOLO, which boosts the multi-scale feature fusion capabilities and achieves an ideal balance between latency and accuracy across all model scales. Additionally, we implement MAE-style pretraining in the YOLO-series for the first time, allowing YOLOseries models could be to benefit from unsupervised pretraining. Gold-YOLO-N attains an outstanding 39.9% AP on the COCO val2017 datasets and 1030 FPS on a T4 GPU, which outperforms the previous SOTA model YOLOv6-3.0-N with similar FPS by +2.4%. The PyTorch code is available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO, and the MindSpore code is available at https://gitee.com/mindspore/models/tree/master/research/cv/Gold_YOLO.

        ----

        ## [2224] Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First

        **Authors**: *Zheng Zhang, Junxiang Wang, Liang Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a07e5160196058120105ad7cb3505d3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a07e5160196058120105ad7cb3505d3c-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have achieved great success in representing data with dependencies by recursively propagating and aggregating messages along the edges. However, edges in real-world graphs often have varying degrees of difficulty, and some edges may even be noisy to the downstream tasks. Therefore, existing GNNs may lead to suboptimal learned representations because they usually treat every edge in the graph equally. On the other hand, Curriculum Learning (CL), which mimics the human learning principle of learning data samples in a meaningful order, has been shown to be effective in improving the generalization ability and robustness of representation learners by gradually proceeding from easy to more difficult samples during training. Unfortunately, existing CL strategies are designed for independent data samples and cannot trivially generalize to handle data dependencies. To address these issues, we propose a novel CL strategy to gradually incorporate more edges into training according to their difficulty from easy to hard, where the degree of difficulty is measured by how well the edges are expected given the model training status. We demonstrate the strength of our proposed method in improving the generalization ability and robustness of learned representations through extensive experiments on nine synthetic datasets and nine real-world datasets. The code for our proposed method is available at https://github.com/rollingstonezz/Curriculumlearningfor_GNNs

        ----

        ## [2225] Unified Lower Bounds for Interactive High-dimensional Estimation under Information Constraints

        **Authors**: *Jayadev Acharya, Clément L. Canonne, Ziteng Sun, Himanshu Tyagi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a07e87ecfa8a651d62257571669b0150-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a07e87ecfa8a651d62257571669b0150-Abstract-Conference.html)

        **Abstract**:

        We consider distributed parameter estimation using interactive protocols subject to local information constraints such as bandwidth limitations, local differential privacy, and restricted measurements. We provide a unified framework enabling us to derive a variety of (tight) minimax lower bounds for different parametric families of distributions, both continuous and discrete, under any $\ell_p$ loss. Our lower bound framework is versatile and yields “plug-and-play” bounds that are widely applicable to a large range of estimation problems, and, for the prototypical case of the Gaussian family, circumvents limitations of previous techniques. In particular, our approach recovers bounds obtained using data processing inequalities and Cramér–Rao bounds, two other alternative approaches for proving lower bounds in our setting of interest. Further, for the families considered, we complement our lower bounds with matching upper bounds.

        ----

        ## [2226] Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching

        **Authors**: *Junsheng Zhou, Baorui Ma, Wenyuan Zhang, Yi Fang, Yu-Shen Liu, Zhizhong Han*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)

        **Abstract**:

        Cross-modality registration between 2D images captured by cameras and 3D point clouds from LiDARs is a crucial task in computer vision and robotic. Previous methods estimate 2D-3D correspondences by matching point and pixel patterns learned by neural networks, and use Perspective-n-Points (PnP) to estimate rigid transformation during post-processing. However, these methods struggle to map points and pixels to a shared latent space robustly since points and pixels have very different characteristics with patterns learned in different manners (MLP and CNN), and they also fail to construct supervision directly on the transformation since the PnP is non-differentiable, which leads to unstable registration results. To address these problems, we propose to learn a structured cross-modality latent space to represent pixel features and 3D features via a differentiable probabilistic PnP solver. Specifically, we design a triplet network to learn VoxelPoint-to-Pixel matching, where we represent 3D elements using both voxels and points to learn the cross-modality latent space with pixels. We design both the voxel and pixel branch based on CNNs to operate convolutions on voxels/pixels represented in grids, and integrate an additional point branch to regain the information lost during voxelization. We train our framework end-to-end by imposing supervisions directly on the predicted pose distribution with a probabilistic PnP solver. To explore distinctive patterns of cross-modality features, we design a novel loss with adaptive-weighted optimization for cross-modality feature description. The experimental results on KITTI and nuScenes datasets show significant improvements over the state-of-the-art methods.

        ----

        ## [2227] Ecosystem-level Analysis of Deployed Machine Learning Reveals Homogeneous Outcomes

        **Authors**: *Connor Toups, Rishi Bommasani, Kathleen Creel, Sarah H. Bana, Dan Jurafsky, Percy Liang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0b1082fc7823c4c68abcab4fa850e9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0b1082fc7823c4c68abcab4fa850e9c-Abstract-Conference.html)

        **Abstract**:

        Machine learning is traditionally studied at the model level: researchers measure and improve the accuracy, robustness, bias, efficiency, and other dimensions of specific models. In practice, however, the societal impact of any machine learning model is partially determined by the context into which it is deployed. To capture this, we introduce ecosystem-level analysis: rather than analyzing a single model, we consider the collection of models that are deployed in a given context. For example, ecosystem-level analysis in hiring recognizes that a job candidateâ€™s outcomes are determined not only by a single hiring algorithm or firm but instead by the collective decisions of all the firms to which the candidate applied. Across three modalities (text, images, speech) and 11 datasets, we establish a clear trend: deployed machine learning is prone to systemic failure, meaning some users are exclusively misclassified by all models available. Even when individual models improve at the population level over time, we find these improvements rarely reduce the prevalence of systemic failure. Instead, the benefits of these improvements predominantly accrue to individuals who are already correctly classified by other models. In light of these trends, we analyze medical imaging for dermatology, a setting where the costs of systemic failure are especially high. While traditional analyses reveal that both models and humans exhibit racial performance disparities, ecosystem-level analysis reveals new forms of racial disparity in model predictions that do not present in human predictions. These examples demonstrate that ecosystem-level analysis has unique strengths in characterizing the societal impact of machine learning.

        ----

        ## [2228] MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion

        **Authors**: *Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, Yasutaka Furukawa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0da690a47b2f52faa63f6fe054057b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0da690a47b2f52faa63f6fe054057b5-Abstract-Conference.html)

        **Abstract**:

        This paper introduces MVDiffusion, a simple yet effective method for generating consistent multi-view images from text prompts given pixel-to-pixel correspondences (e.g., perspective crops from a panorama or multi-view images given depth maps and poses). Unlike prior methods that rely on iterative image warping and inpainting, MVDiffusion simultaneously generates all images with a global awareness, effectively addressing the prevalent error accumulation issue. At its core, MVDiffusion processes perspective images in parallel with a pre-trained text-to-image diffusion model, while integrating novel correspondence-aware attention layers to facilitate cross-view interactions. For panorama generation, while only trained with 10k panoramas, MVDiffusion is able to generate high-resolution photorealistic images for arbitrary texts or extrapolate one perspective image to a 360-degree view. For multi-view depth-to-image generation, MVDiffusion demonstrates state-of-the-art performance for texturing a scene mesh. The project page is at https://mvdiffusion.github.io/.

        ----

        ## [2229] The geometry of hidden representations of large transformer models

        **Authors**: *Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a0e66093d7168b40246af1cddc025daa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a0e66093d7168b40246af1cddc025daa-Abstract-Conference.html)

        **Abstract**:

        Large transformers are powerful architectures used for self-supervised data analysis across various data types, including protein sequences, images, and text. In these models, the semantic structure of the dataset emerges from a sequence of transformations between one representation and the next. We characterize the geometric and statistical properties of these representations and how they change as we move through the layers.By analyzing the intrinsic dimension (ID) and neighbor composition, we find that the representations evolve similarly in transformers trained on protein language taskand image reconstruction tasks. In the first layers, the data manifold expands, becoming high-dimensional, and then contracts significantly in the intermediate layers. In the last part of the model, the ID remains approximately constant or forms a second shallow peak. We show that the semantic information of the dataset is better expressed at the end of the first peak, and this phenomenon can be observed across many models trained on diverse datasets.Based on our findings, we point out an explicit strategy to identify, without supervision, the layers that maximize semantic content: representations at intermediate layers corresponding to a relative minimum of the ID profile are more suitable for downstream learning tasks.

        ----

        ## [2230] Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration

        **Authors**: *Guangyu Shen, Siyuan Cheng, Guanhong Tao, Kaiyuan Zhang, Yingqi Liu, Shengwei An, Shiqing Ma, Xiangyu Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a102d6cb996be3482c059c1e18bbe523-Abstract-Conference.html)

        **Abstract**:

        Object detection models are vulnerable to backdoor or trojan attacks, where an attacker can inject malicious triggers into the model, leading to altered behavior during inference. As a defense mechanism, trigger inversion leverages optimization to reverse-engineer triggers and identify compromised models. While existing trigger inversion methods assume that each instance from the support set is equally affected by the injected trigger, we observe that the poison effect can vary significantly across bounding boxes in object detection models due to its dense prediction nature, leading to an undesired optimization objective misalignment issue for existing trigger reverse-engineering methods. To address this challenge, we propose the first object detection backdoor detection framework Django (Detecting Trojans in Object Detection Models via Gaussian Focus Calibration). It leverages a dynamic Gaussian weighting scheme that prioritizes more vulnerable victim boxes and assigns appropriate coefficients to calibrate the optimization objective during trigger inversion. In addition, we combine Django with a novel label proposal pre-processing technique to enhance its efficiency. We evaluate Django on 3 object detection image datasets, 3 model architectures, and 2 types of attacks, with a total of 168 models. Our experimental results show that Django outperforms 6 state-of-the-art baselines, with up to 38% accuracy improvement and 10x reduced overhead. The code is available at https://github.com/PurduePAML/DJGO.

        ----

        ## [2231] CORNN: Convex optimization of recurrent neural networks for rapid inference of neural dynamics

        **Authors**: *Fatih Dinc, Adam Shai, Mark Schnitzer, Hidenori Tanaka*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a103529738706979331778377f2d5864-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a103529738706979331778377f2d5864-Abstract-Conference.html)

        **Abstract**:

        Advances in optical and electrophysiological recording technologies have made it possible to record the dynamics of thousands of neurons, opening up new possibilities for interpreting and controlling large neural populations in behaving animals. A promising way to extract computational principles from these large datasets is to train data-constrained recurrent neural networks (dRNNs). Performing this training in real-time could open doors for research techniques and medical applications to model and control interventions at single-cell resolution and drive desired forms of animal behavior. However, existing training algorithms for dRNNs are inefficient and have limited scalability, making it a challenge to analyze large neural recordings even in offline scenarios. To address these issues, we introduce a training method termed Convex Optimization of Recurrent Neural Networks (CORNN). In studies of simulated recordings, CORNN attained training speeds $\sim$100-fold faster than traditional optimization approaches while maintaining or enhancing modeling accuracy. We further validated CORNN on simulations with thousands of cells that performed simple computations such as those of a 3-bit flip-flop or the execution of a timed response. Finally, we showed that CORNN can robustly reproduce network dynamics and underlying attractor structures despite mismatches between generator and inference models, severe subsampling of observed neurons, or mismatches in neural time-scales. Overall, by training dRNNs with millions of parameters in subminute processing times on a standard computer, CORNN constitutes a first step towards real-time network reproduction constrained on large-scale neural recordings and a powerful computational tool for advancing the understanding of neural computation.

        ----

        ## [2232] A Unified Framework for Rank-based Loss Minimization

        **Authors**: *Rufeng Xiao, Yuze Ge, Rujun Jiang, Yifan Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a10946e1f46e1ffc0daf37cb2abfdcad-Abstract-Conference.html)

        **Abstract**:

        The empirical loss, commonly referred to as the average loss, is extensively utilized for training machine learning models. However, in order to address the diverse performance requirements of machine learning models, the use of the rank-based loss is prevalent, replacing the empirical loss in many cases. The rank-based loss comprises a weighted sum of sorted individual losses, encompassing both convex losses like the spectral risk, which includes the empirical risk and conditional value-at-risk, and nonconvex losses such as the human-aligned risk and the sum of the ranked range loss. In this paper, we introduce a unified framework for the optimization of the rank-based loss through the utilization of a proximal alternating direction method of multipliers. We demonstrate the convergence and convergence rate of the proposed algorithm under mild conditions. Experiments conducted on synthetic and real datasets illustrate the effectiveness and efficiency of the proposed algorithm.

        ----

        ## [2233] LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas

        **Authors**: *Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, Charles Sutton*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a10da26f47120217c1b7c2aeb2979048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a10da26f47120217c1b7c2aeb2979048-Abstract-Conference.html)

        **Abstract**:

        Search is an important technique in program synthesis that allows for adaptive strategies such as focusing on particular search directions based on execution results. Several prior works have demonstrated that neural models are effective at guiding program synthesis searches. However, a common drawback of those approaches is the inability to handle iterative loops, higher-order functions, or lambda functions, thus limiting prior neural searches from synthesizing longer and more general programs. We address this gap by designing a search algorithm called LambdaBeam that can construct arbitrary lambda functions that compose operations within a given DSL. We create semantic vector representations of the execution behavior of the lambda functions and train a neural policy network to choose which lambdas to construct during search, and pass them as arguments to higher-order functions to perform looping computations. Our experiments show that LambdaBeam outperforms neural, symbolic, and LLM-based techniques in an integer list manipulation domain.

        ----

        ## [2234] HQA-Attack: Toward High Quality Black-Box Hard-Label Adversarial Attack on Text

        **Authors**: *Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a124b5e7385d35e5c8ad05d192106e19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a124b5e7385d35e5c8ad05d192106e19-Abstract-Conference.html)

        **Abstract**:

        Black-box hard-label adversarial attack on text is a practical and challenging task, as the text data space is inherently discrete and non-differentiable, and only the predicted label is accessible. Research on this problem is still in the embryonic stage and only a few methods are available. Nevertheless, existing methods rely on the complex heuristic algorithm or unreliable gradient estimation strategy, which probably fall into the local optimum and inevitably consume numerous queries, thus are difficult to craft satisfactory adversarial examples with high semantic similarity and low perturbation rate in a limited query budget. To alleviate above issues, we propose a simple yet effective framework to generate high quality textual adversarial examples under the black-box hard-label attack scenarios, named HQA-Attack. Specifically, after initializing an adversarial example randomly, HQA-attack first constantly substitutes original words back as many as possible, thus shrinking the perturbation rate. Then it leverages the synonym set of the remaining changed words to further optimize the adversarial example with the direction which can improve the semantic similarity and satisfy the adversarial condition simultaneously. In addition, during the optimizing procedure, it searches a transition synonym word for each changed word, thus avoiding traversing the whole synonym set and reducing the query number to some extent. Extensive experimental results on five text classification datasets, three natural language inference datasets and two real-world APIs have shown that the proposed HQA-Attack method outperforms other strong baselines significantly.

        ----

        ## [2235] Augmentation-free Dense Contrastive Distillation for Efficient Semantic Segmentation

        **Authors**: *Jiawei Fan, Chao Li, Xiaolong Liu, Meina Song, Anbang Yao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a12779b5e802668df1cbc73fa00da62f-Abstract-Conference.html)

        **Abstract**:

        In recent years, knowledge distillation methods based on contrastive learning have achieved promising results on image classification and object detection tasks. However, in this line of research, we note that less attention is paid to semantic segmentation. Existing methods heavily rely on data augmentation and memory buffer, which entail high computational resource demands when applying them to handle semantic segmentation that requires to preserve high-resolution feature maps for making dense pixel-wise predictions. In order to address this problem, we present Augmentation-free Dense Contrastive Knowledge Distillation (Af-DCD), a new contrastive distillation learning paradigm to train compact and accurate deep neural networks for semantic segmentation applications. Af-DCD leverages a masked feature mimicking strategy, and formulates a novel contrastive learning loss via taking advantage of tactful feature partitions across both channel and spatial dimensions, allowing to effectively transfer dense and structured local knowledge learnt by the teacher model to a target student model while maintaining training efficiency. Extensive experiments on five mainstream benchmarks with various teacher-student network pairs demonstrate the effectiveness of our approach. For instance, DeepLabV3-Res18|DeepLabV3-MBV2 model trained by Af-DCD reaches 77.03\%|76.38\% mIOU on Cityscapes dataset when choosing DeepLabV3-Res101 as the teacher, setting new performance records. Besides that, Af-DCD achieves an absolute mIOU improvement of 3.26\%|3.04\%|2.75\%|2.30\%|1.42\% compared with individually trained counterpart on Cityscapes|Pascal VOC|Camvid|ADE20K|COCO-Stuff-164K. Code is available at https://github.com/OSVAI/Af-DCD.

        ----

        ## [2236] On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets

        **Authors**: *Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Different distribution shifts require different algorithmic and operational  interventions. Methodological research must be grounded by the specific  shifts they address.  Although nascent benchmarks provide a promising  empirical foundation, they \emph{implicitly} focus on covariate  shifts, and the validity of empirical findings depends on the type of shift,   e.g., previous observations on algorithmic performance can fail to be valid when  the $Y|X$ distribution changes.  We conduct a thorough investigation of  natural shifts in 5 tabular datasets over 86,000 model configurations, and  find that $Y|X$-shifts are most prevalent.  To encourage researchers to  develop a refined language for distribution shifts, we build ``WhyShift``, an empirical testbed of curated real-world shifts where  we characterize the type of shift we benchmark performance over.  Since  $Y|X$-shifts are prevalent in tabular settings, we \emph{identify covariate  regions} that suffer the biggest $Y|X$-shifts and discuss implications for  algorithmic and data-based interventions.  Our testbed highlights the  importance of future research that builds an understanding of why  distributions differ.

        ----

        ## [2237] Diverse Community Data for Benchmarking Data Privacy Algorithms

        **Authors**: *Aniruddha Sen, Christine Task, Dhruv Kapur, Gary Howarth, Karan Bhagat*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a15032f8199511ced4d7a8e2bbb487a5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a15032f8199511ced4d7a8e2bbb487a5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The Collaborative Research Cycle (CRC) is a National Institute of Standards and Technology (NIST) benchmarking program intended to strengthen understanding of tabular data deidentification technologies. Deidentification algorithms are vulnerable to the same bias and privacy issues that impact other data analytics and machine learning applications, and it can even amplify those issues by contaminating downstream applications. This paper summarizes four CRC contributions: theoretical work on the relationship between diverse populations and challenges for equitable deidentification; public benchmark data focused on diverse populations and challenging features; a comprehensive open source suite of evaluation metrology for deidentified datasets; and an archive of more than 450 deidentified data samples from a broad range of techniques. The initial set of evaluation results demonstrate the value of the CRC tools for investigations in this field.

        ----

        ## [2238] Coneheads: Hierarchy Aware Attention

        **Authors**: *Albert Tseng, Tao Yu, Toni J. B. Liu, Christopher De Sa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a17251f8d595179eef5e466b1f5f7a85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a17251f8d595179eef5e466b1f5f7a85-Abstract-Conference.html)

        **Abstract**:

        Attention networks such as transformers have achieved state-of-the-art performance in many domains. These networks rely heavily on the dot product attention operator, which computes the similarity between two points by taking their inner product.However, the inner product does not explicitly model the complex structural properties of real world datasets, such as hierarchies between data points.To remedy this, we introduce cone attention, a drop-in replacement for dot product attention based on hyperbolic entailment cones.Cone attention associates two points by the depth of their lowest common ancestor in a hierarchy defined by hyperbolic cones, which intuitively measures the divergence of two points and gives a $\textit{hierarchy aware}$ similarity score.We test cone attention on a wide variety of models and tasks and show that it improves task-level performance over dot product attention and other baselines, and is able to match dot-product attention with significantly fewer parameters.Our results suggest that cone attention is an effective way to capture hierarchical relationships when calculating attention.

        ----

        ## [2239] Benchmark of Machine Learning Force Fields for Semiconductor Simulations: Datasets, Metrics, and Comparative Analysis

        **Authors**: *Geonu Kim, Byunggook Na, Gunhee Kim, Hyuntae Cho, Seungjin Kang, Hee Sun Lee, Saerom Choi, Heejae Kim, Seungwon Lee, Yongdeok Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        As semiconductor devices become miniaturized and their structures become more complex, there is a growing need for large-scale atomic-level simulations as a less costly alternative to the trial-and-error approach during development.Although machine learning force fields (MLFFs) can meet the accuracy and scale requirements for such simulations, there are no open-access benchmarks for semiconductor materials.Hence, this study presents a comprehensive benchmark suite that consists of two semiconductor material datasets and ten MLFF models with six evaluation metrics. We select two important semiconductor thin-film materials silicon nitride and hafnium oxide, and generate their datasets using computationally expensive density functional theory simulations under various scenarios at a cost of 2.6k GPU days.Additionally, we provide a variety of architectures as baselines: descriptor-based fully connected neural networks and graph neural networks with rotational invariant or equivariant features.We assess not only the accuracy of energy and force predictions but also five additional simulation indicators to determine the practical applicability of MLFF models in molecular dynamics simulations.To facilitate further research, our benchmark suite is available at https://github.com/SAITPublic/MLFF-Framework.

        ----

        ## [2240] Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks

        **Authors**: *Aoxiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1c716638d9b618a1a40a96f473c8250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1c716638d9b618a1a40a96f473c8250-Abstract-Conference.html)

        **Abstract**:

        No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models againstadversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial videoâ€™s estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.

        ----

        ## [2241] Unsupervised Behavior Extraction via Random Intent Priors

        **Authors**: *Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, Chongjie Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1c8a68e52499c9396854e3f967e37c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1c8a68e52499c9396854e3f967e37c0-Abstract-Conference.html)

        **Abstract**:

        Reward-free data is abundant and contains rich prior knowledge of human behaviors, but it is not well exploited by offline reinforcement learning (RL) algorithms. In this paper, we propose UBER, an unsupervised approach to extract useful behaviors from offline reward-free datasets via diversified rewards. UBER assigns different pseudo-rewards sampled from a given prior distribution to different agents to extract a diverse set of behaviors, and reuse them as candidate policies to facilitate the learning of new tasks. Perhaps surprisingly, we show that rewards generated from random neural networks are sufficient to extract diverse and useful behaviors, some even close to expert ones. We provide both empirical and theoretical evidences to justify the use of random priors for the reward function. Experiments on multiple benchmarks showcase UBER's ability to learn effective and diverse behavior sets that enhance sample efficiency for online RL, outperforming existing baselines. By reducing reliance on human supervision, UBER broadens the applicability of RL to real-world scenarios with abundant reward-free data.

        ----

        ## [2242] Deconstructing Data Reconstruction: Multiclass, Weight Decay and General Losses

        **Authors**: *Gon Buzaglo, Niv Haim, Gilad Yehudai, Gal Vardi, Yakir Oz, Yaniv Nikankin, Michal Irani*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1d20cc72a21ef971d7e49a90d8fa56f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1d20cc72a21ef971d7e49a90d8fa56f-Abstract-Conference.html)

        **Abstract**:

        Memorization of training data is an active research area, yet our understanding of the inner workings of neural networks is still in its infancy.Recently, Haim et al. 2022 proposed a scheme to reconstruct training samples from multilayer perceptron binary classifiers, effectively demonstrating that a large portion of training samples are encoded in the parameters of such networks.In this work, we extend their findings in several directions, including reconstruction from multiclass and convolutional neural networks. We derive a more general reconstruction scheme which is applicable to a wider range of loss functions such as regression losses. Moreover, we study the various factors that contribute to networks' susceptibility to such reconstruction schemes. Intriguingly, we observe that using weight decay during training increases reconstructability both in terms of quantity and quality. Additionally, we examine the influence of the number of neurons relative to the number of training samples on the reconstructability.Code: https://github.com/gonbuzaglo/decoreco

        ----

        ## [2243] Information Maximizing Curriculum: A Curriculum-Based Approach for Learning Versatile Skills

        **Authors**: *Denis Blessing, Onur Celik, Xiaogang Jia, Moritz Reuss, Maximilian Li, Rudolf Lioutikov, Gerhard Neumann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1e6783e4d739196cad3336f12d402bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1e6783e4d739196cad3336f12d402bf-Abstract-Conference.html)

        **Abstract**:

        Imitation learning uses data for training policies to solve complex tasks. However,when the training data is collected from human demonstrators, it often leadsto multimodal distributions because of the variability in human actions. Mostimitation learning methods rely on a maximum likelihood (ML) objective to learna parameterized policy, but this can result in suboptimal or unsafe behavior dueto the mode-averaging property of the ML objective. In this work, we proposeInformation Maximizing Curriculum, a curriculum-based approach that assignsa weight to each data point and encourages the model to specialize in the data itcan represent, effectively mitigating the mode-averaging problem by allowing themodel to ignore data from modes it cannot represent. To cover all modes and thus,enable versatile behavior, we extend our approach to a mixture of experts (MoE)policy, where each mixture component selects its own subset of the training datafor learning. A novel, maximum entropy-based objective is proposed to achievefull coverage of the dataset, thereby enabling the policy to encompass all modeswithin the data distribution. We demonstrate the effectiveness of our approach oncomplex simulated control tasks using versatile human demonstrations, achievingsuperior performance compared to state-of-the-art methods.

        ----

        ## [2244] Unleash the Potential of Image Branch for Cross-modal 3D Object Detection

        **Authors**: *Yifan Zhang, Qijian Zhang, Junhui Hou, Yixuan Yuan, Guoliang Xing*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a1f0c0cd6caaa4863af5f12608edf63e-Abstract-Conference.html)

        **Abstract**:

        To achieve reliable and precise scene understanding, autonomous vehicles typically incorporate multiple sensing modalities to capitalize on their complementary attributes. However, existing cross-modal 3D detectors do not fully utilize the image domain information to address the bottleneck issues of the LiDAR-based detectors. This paper presents a new cross-modal 3D object detector, namely UPIDet, which aims to unleash the potential of the image branch from two aspects. First, UPIDet introduces a new 2D auxiliary task called normalized local coordinate map estimation. This approach enables the learning of local spatial-aware features from the image modality to supplement sparse point clouds. Second, we discover that the representational capability of the point cloud backbone can be enhanced through the gradients backpropagated from the training objectives of the image branch, utilizing a succinct and effective point-to-pixel module. Extensive experiments and ablation studies validate the effectiveness of our method. Notably, we achieved the top rank in the highly competitive cyclist class of the KITTI benchmark at the time of submission. The source code is available at https://github.com/Eaphan/UPIDet.

        ----

        ## [2245] Model Sparsity Can Simplify Machine Unlearning

        **Authors**: *Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, Sijia Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html)

        **Abstract**:

        In response to recent data regulation requirements, machine unlearning (MU) has emerged as a critical process to remove the influence of specific examples from a given model. Although exact unlearning can be achieved through complete model retraining using the remaining dataset, the associated computational costs have driven the development of efficient, approximate unlearning techniques. Moving beyond data-centric MU approaches, our study introduces a novel model-based perspective: model sparsification via weight pruning, which is capable of reducing the gap between exact unlearning and approximate unlearning. We show in both theory and practice that model sparsity can boost the multi-criteria unlearning performance of an approximate unlearner, closing the approximation gap, while continuing to be efficient. This leads to a new MU paradigm,    termed prune first, then unlearn, which infuses a sparse prior to the unlearning process. Building on this insight, we also develop a sparsity-aware unlearning method that utilizes sparsity regularization to enhance the training process of approximate unlearning. Extensive experiments show that our proposals consistently benefit MU in various unlearning scenarios. A notable highlight is the 77% unlearning efficacy gain of fine-tuning (one of the simplest approximate unlearning methods) when using our proposed sparsity-aware unlearning method. Furthermore, we showcase the practical impact of our proposed MU methods through two specific use cases: defending against backdoor attacks, and enhancing transfer learning through source class removal. These applications demonstrate the versatility and effectiveness of our approaches in addressing a variety of machine learning challenges beyond unlearning for data privacy. Codes are available at https://github.com/OPTML-Group/Unlearn-Sparse.

        ----

        ## [2246] IDRNet: Intervention-Driven Relation Network for Semantic Segmentation

        **Authors**: *Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, Lequan Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a216c27f2f3160b1785c057fa510fdf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a216c27f2f3160b1785c057fa510fdf1-Abstract-Conference.html)

        **Abstract**:

        Co-occurrent visual patterns suggest that pixel relation modeling facilitates dense prediction tasks, which inspires the development of numerous context modeling paradigms, \emph{e.g.}, multi-scale-driven and similarity-driven context schemes. Despite the impressive results, these existing paradigms often suffer from inadequate or ineffective contextual information aggregation due to reliance on large amounts of predetermined priors. To alleviate the issues, we propose a novel \textbf{I}ntervention-\textbf{D}riven \textbf{R}elation \textbf{Net}work (\textbf{IDRNet}), which leverages a deletion diagnostics procedure to guide the modeling of contextual relations among different pixels. Specifically, we first group pixel-level representations into semantic-level representations with the guidance of pseudo labels and further improve the distinguishability of the grouped representations with a feature enhancement module. Next, a deletion diagnostics procedure is conducted to model relations of these semantic-level representations via perceiving the network outputs and the extracted relations are utilized to guide the semantic-level representations to interact with each other.  Finally, the interacted representations are utilized to augment original pixel-level representations for final predictions. Extensive experiments are conducted to validate the effectiveness of IDRNet quantitatively and qualitatively. Notably, our intervention-driven context scheme brings consistent performance improvements to state-of-the-art segmentation frameworks and achieves competitive results on popular benchmark datasets, including ADE20K, COCO-Stuff, PASCAL-Context, LIP, and Cityscapes.

        ----

        ## [2247] Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width

        **Authors**: *Dayal Singh Kalra, Maissam Barkeshli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a23598416361c7a9860164155e6ddd0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a23598416361c7a9860164155e6ddd0b-Abstract-Conference.html)

        **Abstract**:

        We systematically analyze optimization dynamics in deep neural networks (DNNs) trained with stochastic gradient descent (SGD) and study the effect of learning rate $\eta$, depth $d$, and width $w$ of the neural network. By analyzing the maximum eigenvalue $\lambda^H_t$ of the Hessian of the loss, which is a measure of sharpness of the loss landscape, we find that the dynamics can show four distinct regimes: (i) an early time transient regime, (ii) an intermediate saturation regime, (iii) a progressive sharpening regime, and (iv) a late time "edge of stability" regime. The early and intermediate regimes (i) and (ii) exhibit a rich phase diagram depending on $\eta \equiv c / \lambda_0^H $, $d$, and $w$. We identify several critical values of $c$, which separate qualitatively distinct phenomena in the early time dynamics of training loss and sharpness. Notably, we discover the opening up of a "sharpness reduction" phase, where sharpness decreases at early times, as $d$ and $ 1/w$ are increased.

        ----

        ## [2248] Neural Algorithmic Reasoning Without Intermediate Supervision

        **Authors**: *Gleb Rodionov, Liudmila Prokhorenkova*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2370db7c99791ad5d9f3ef48ad6d464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2370db7c99791ad5d9f3ef48ad6d464-Abstract-Conference.html)

        **Abstract**:

        Neural algorithmic reasoning is an emerging area of machine learning focusing on building models that can imitate the execution of classic algorithms, such as sorting, shortest paths, etc. One of the main challenges is to learn algorithms that are able to generalize to out-of-distribution data, in particular with significantly larger input sizes. Recent work on this problem has demonstrated the advantages of learning algorithms step-by-step, giving models access to all intermediate steps of the original algorithm. In this work, we instead focus on learning neural algorithmic reasoning only from the input-output pairs without appealing to the intermediate supervision. We propose simple but effective architectural improvements and also build a self-supervised objective that can regularise intermediate computations of the model without access to the algorithm trajectory. We demonstrate that our approach is competitive to its trajectory-supervised counterpart on tasks from the CLRS Algorithmic Reasoning Benchmark and achieves new state-of-the-art results for several problems, including sorting, where we obtain significant improvements. Thus, learning without intermediate supervision is a promising direction for further research on neural reasoners.

        ----

        ## [2249] On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

        **Authors**: *Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, Sungroh Yoon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2374637af47ac9471b43c99b68acf27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2374637af47ac9471b43c99b68acf27-Abstract-Conference.html)

        **Abstract**:

        Successful detection of Out-of-Distribution (OoD) data is becoming increasingly important to ensure safe deployment of neural networks. One of the main challenges in OoD detection is that neural networks output overconfident predictions on OoD data, make it difficult to determine OoD-ness of data solely based on their predictions. Outlier exposure addresses this issue by introducing an additional loss that encourages low-confidence predictions on OoD data during training. While outlier exposure has shown promising potential in improving OoD detection performance, all previous studies on outlier exposure have been limited to utilizing visual outliers. Drawing inspiration from the recent advancements in vision-language pre-training, this paper venture out to the uncharted territory of textual outlier exposure. First, we uncover the benefits of using textual outliers by replacing real or virtual outliers in the image-domain with textual equivalents. Then, we propose various ways of generating preferable textual outliers. Our extensive experiments demonstrate that generated textual outliers achieve competitive performance on large-scale OoD and hard OoD benchmarks. Furthermore, we conduct empirical analyses of textual outliers to provide primary criteria for designing advantageous textual outliers: near-distribution, descriptiveness, and inclusion of visual semantics.

        ----

        ## [2250] Estimating Propensity for Causality-based Recommendation without Exposure Data

        **Authors**: *Zhongzhou Liu, Yuan Fang, Min Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a237f11d6aad94f59a182d70405d3fdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a237f11d6aad94f59a182d70405d3fdb-Abstract-Conference.html)

        **Abstract**:

        Causality-based recommendation systems focus on the causal effects of user-item interactions resulting from item exposure (i.e., which items are recommended or exposed to the user), as opposed to conventional correlation-based recommendation. They are gaining popularity due to their multi-sided benefits to users, sellers and platforms alike. However, existing causality-based recommendation methods require additional input in the form of exposure data and/or propensity scores (i.e., the probability of exposure) for training. Such data, crucial for modeling causality in recommendation, are often not available in real-world situations due to technical or privacy constraints. In this paper, we bridge the gap by proposing a new framework, called Propensity Estimation for Causality-based Recommendation (PropCare). It can estimate the propensity and exposure from a more practical setup, where only interaction data are available without any ground truth on exposure or propensity in training and inference. We demonstrate that, by relating the pairwise characteristics between propensity and item popularity, PropCare enables competitive causality-based recommendation given only the conventional interaction data. We further present a theoretical analysis on the bias of the causal effect under our model estimation.  Finally, we empirically evaluate PropCare through both quantitative and qualitative experiments.

        ----

        ## [2251] A Robust Exact Algorithm for the Euclidean Bipartite Matching Problem

        **Authors**: *Akshaykumar Gattani, Sharath Raghvendra, Pouyan Shirzadian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a24a75ef009ee73b160653c16b18f00e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a24a75ef009ee73b160653c16b18f00e-Abstract-Conference.html)

        **Abstract**:

        Algorithms for the minimum-cost bipartite matching can be used to estimate Wasserstein distance between two distributions.Given two sets $A$ and $B$ of $n$ points in a $2$-dimensional Euclidean space, one can use a fast implementation of the Hungarian method to compute a minimum-cost bipartite matching of $A$ and $B$ in $\tilde{O}(n^2)$ time. Let $\Delta$ be the spread, i.e., the ratio of the distance of the farthest to the closest pair of points in $A\cup B$. In this paper, we present a new algorithm to compute a minimum-cost bipartite matching of $A$ and $B$ with a similar worst-case execution time of $\tilde{O}(n^2 \log \Delta)$. However, when $A$ and $B$ are drawn independently and identically from a fixed distribution that is not known to the algorithm, the execution time of our algorithm is, in expectation, $\tilde{O}(n^{7/4}\log \Delta)$.To the best of our knowledge, our algorithm is the first one to achieve a sub-quadratic execution time even for stochastic point sets with real-valued coordinates.Our algorithm extends to any dimension $d$, where it runs in $\tilde{O}(n^{2-\frac{1}{2d}}\Phi(n))$ time for stochastic point sets $A$ and $B$; here $\Phi(n)$ is the query/update time of a dynamic weighted nearest neighbor data structure. Our algorithm can be seen as a careful adaptation of the Hungarian method in the geometric divide-and-conquer framework.

        ----

        ## [2252] Content-based Unrestricted Adversarial Attack

        **Authors**: *Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html)

        **Abstract**:

        Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack (ACA) based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4\% and 16.8-48.0\% in normally trained models and defense methods, respectively.

        ----

        ## [2253] On Dynamic Programming Decompositions of Static Risk Measures in Markov Decision Processes

        **Authors**: *Jia Lin Hau, Erick Delage, Mohammad Ghavamzadeh, Marek Petrik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a264726ebd222124514a32bf0143b83d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a264726ebd222124514a32bf0143b83d-Abstract-Conference.html)

        **Abstract**:

        Optimizing static risk-averse objectives in Markov decision processes is difficult because they do not admit standard dynamic programming equations common in Reinforcement Learning (RL) algorithms. Dynamic programming decompositions that augment the state space with discrete risk levels have recently gained popularity in the RL community. Prior work has shown that these decompositions are optimal when the risk level is discretized sufficiently. However, we show that these popular decompositions for Conditional-Value-at-Risk (CVaR) and Entropic-Value-at-Risk (EVaR) are inherently suboptimal regardless of the discretization level. In particular, we show that a saddle point property assumed to hold in prior literature may be violated. However, a decomposition does hold for Value-at-Risk and our proof demonstrates how this risk measure differs from CVaR and EVaR. Our findings are significant because risk-averse algorithms are used in high-stake environments, making their correctness much more critical.

        ----

        ## [2254] Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models

        **Authors**: *Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip H. S. Torr, Volker Tresp*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2a544e43acb8b954dc5846ff0d77ad5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2a544e43acb8b954dc5846ff0d77ad5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. As test samples in real-world applications usually differ from adaptation data, the robustness of these adaptation methods against distribution shifts are essential. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that increasing the number of adaptation data and parameters does not guarantee enhanced robustness; instead, it results in even lower robustness. We hope this study could benefit future research in the development of robust multimodal adaptation methods. The benchmark, code, and dataset used in this study can be accessed at https://adarobustness.github.io.

        ----

        ## [2255] Evaluating the Moral Beliefs Encoded in LLMs

        **Authors**: *Nino Scherrer, Claudia Shi, Amir Feder, David M. Blei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2cf225ba392627529efef14dc857e22-Abstract-Conference.html)

        **Abstract**:

        This paper presents a case study on the design, administration, post-processing,  and evaluation of surveys on large language models (LLMs). It comprises two components:(1) A statistical method for eliciting beliefs encoded in LLMs. We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice.(2) We apply this method to study what moral beliefs are encoded in different LLMs, especially in ambiguous cases where the right choice is not obvious.We design a large-scale survey comprising 680 high-ambiguity moral scenarios (e.g., "Should I tell a white lie?") and 687 low-ambiguity moral scenarios (e.g., "Should I stop for a pedestrian on the road?"). Each scenario includes a description, two possible actions, and auxiliary labels indicating violated rules (e.g., "do not kill"). We administer the survey to 28 open- and closed-source LLMs.We find that (a) in unambiguous scenarios, most models ``choose" actions that align with commonsense. In ambiguous cases, most models express uncertainty.(b) Some models are uncertain about choosing the commonsense action because their responses are sensitive to the question-wording.(c) Some models reflect clear preferences in ambiguous scenarios. Specifically, closed-source models tend to agree with each other.

        ----

        ## [2256] Enhancing Adversarial Robustness via Score-Based Optimization

        **Authors**: *Boya Zhang, Weijian Luo, Zhihua Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2e707354da36956945dbb288efe82b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2e707354da36956945dbb288efe82b3-Abstract-Conference.html)

        **Abstract**:

        Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data  in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

        ----

        ## [2257] Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation

        **Authors**: *Giorgio Giannone, Akash Srivastava, Ole Winther, Faez Ahmed*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a2fe4bb50fc6f3564cee1551d6309fea-Abstract-Conference.html)

        **Abstract**:

        Generative models have significantly influenced both vision and language domains, ushering in innovative multimodal applications. Although these achievements have motivated exploration in scientific and engineering fields, challenges emerge, particularly in constrained settings with limited data where precision is crucial. Traditional engineering optimization methods rooted in physics often surpass generative models in these contexts. To address these challenges, we introduce Diffusion Optimization Models (DOM) and Trajectory Alignment (TA), a learning framework that demonstrates the efficacy of aligning the sampling trajectory of diffusion models with the trajectory derived from physics-based iterative optimization methods. This alignment ensures that the sampling process remains grounded in the underlying physical principles. This alignment eliminates the need for costly preprocessing, external surrogate models, or extra labeled data, generating feasible and high-performance designs efficiently. We apply our framework to structural topology optimization, a fundamental problem in mechanical design, evaluating its performance on in- and out-of-distribution configurations. Our results demonstrate that TA outperforms state-of-the-art deep generative models on in-distribution configurations and halves the inference computational cost. When coupled with a few steps of optimization, it also improves manufacturability for out-of-distribution conditions. DOM's efficiency and performance improvements significantly expedite design processes and steer them toward optimal and manufacturable outcomes, highlighting the potential of generative models in data-driven design.

        ----

        ## [2258] Optimal cross-learning for contextual bandits with unknown context distributions

        **Authors**: *Jon Schneider, Julian Zimmert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3017a8d202a433be56a3dfdcac6c8eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3017a8d202a433be56a3dfdcac6c8eb-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of designing contextual bandit algorithms in the ``cross-learning'' setting of Balseiro et al., where the learner observes the loss for the action they play in all possible contexts, not just the context of the current round. We specifically consider the setting where losses are chosen adversarially and contexts are sampled i.i.d. from an unknown distribution. In this setting, we resolve an open problem of Balseiro et al. by providing an efficient algorithm with a nearly tight (up to logarithmic factors) regret bound of $\widetilde{O}(\sqrt{TK})$, independent of the number of contexts. As a consequence, we obtain the first nearly tight regret bounds for the problems of learning to bid in first-price auctions (under unknown value distributions) and sleeping bandits with a stochastic action set.At the core of our algorithm is a novel technique for coordinating the execution of a learning algorithm over multiple epochs in such a way to remove correlations between estimation of the unknown distribution and the actions played by the algorithm. This technique may be of independent interest for other learning problems involving estimation of an unknown context distribution.

        ----

        ## [2259] Conservative Offline Policy Adaptation in Multi-Agent Games

        **Authors**: *Chengjie Wu, Pingzhong Tang, Jun Yang, Yujing Hu, Tangjie Lv, Changjie Fan, Chongjie Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a31253f4871694f09541122d6b6f5ad1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a31253f4871694f09541122d6b6f5ad1-Abstract-Conference.html)

        **Abstract**:

        Prior research on policy adaptation in multi-agent games has often relied on online interaction with the target agent in training, which can be expensive and impractical in real-world scenarios. Inspired by recent progress in offline reinforcement learn- ing, this paper studies offline policy adaptation, which aims to utilize the target agentâ€™s behavior data to exploit its weakness or enable effective cooperation. We investigate its distinct challenges of distributional shift and risk-free deviation, and propose a novel learning objective, conservative offline adaptation, that optimizes the worst-case performance against any dataset consistent proxy models. We pro- pose an efficient algorithm called Constrained Self-Play (CSP) that incorporates dataset information into regularized policy learning. We prove that CSP learns a near-optimal risk-free offline adaptation policy upon convergence. Empirical results demonstrate that CSP outperforms non-conservative baselines in various environments, including Maze, predator-prey, MuJoCo, and Google Football.

        ----

        ## [2260] Bounding the Invertibility of Privacy-preserving Instance Encoding using Fisher Information

        **Authors**: *Kiwan Maeng, Chuan Guo, Sanjay Kariyappa, G. Edward Suh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a344f7f474958cc0775be7e46bc94309-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a344f7f474958cc0775be7e46bc94309-Abstract-Conference.html)

        **Abstract**:

        Privacy-preserving instance encoding aims to encode raw data into feature vectors without revealing their privacy-sensitive information. When designed properly, these encodings can be used for downstream ML applications such as training and inference with limited privacy risk. However, the vast majority of existing schemes do not theoretically justify that their encoding is non-invertible, and their privacy-enhancing properties are only validated empirically against a limited set of attacks. In this paper, we propose a theoretically-principled measure for the invertibility of instance encoding based on Fisher information that is broadly applicable to a wide range of popular encoders. We show that dFIL can be used to bound the invertibility of encodings both theoretically and empirically, providing an intuitive interpretation of the privacy of instance encoding.

        ----

        ## [2261] Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

        **Authors**: *Yuxin Pan, Yize Chen, Fangzhen Lin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a345ed605675c7c484e740a8ceaa6b45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a345ed605675c7c484e740a8ceaa6b45-Abstract-Conference.html)

        **Abstract**:

        Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints.  While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound  by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.

        ----

        ## [2262] Promises and Pitfalls of Threshold-based Auto-labeling

        **Authors**: *Harit Vishwakarma, Heguang Lin, Frederic Sala, Ramya Korlakai Vinayak*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a355051cc32d36e2a971de190701745a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a355051cc32d36e2a971de190701745a-Abstract-Conference.html)

        **Abstract**:

        Creating large-scale high-quality labeled datasets is a major bottleneck in supervised machine learning workflows. Threshold-based auto-labeling (TBAL), where validation data obtained from humans is used to find a confidence threshold above which the data is machine-labeled, reduces reliance on manual annotation. TBAL is emerging as a widely-used solution in practice. Given the long shelf-life and diverse usage of the resulting datasets, understanding when the data obtained by such auto-labeling systems can be relied on is crucial. This is the first work to analyze TBAL systems and derive sample complexity bounds on the amount of human-labeled validation data required for guaranteeing the quality of machine-labeled data. Our results provide two crucial insights. First, reasonable chunks of unlabeled data can be automatically and accurately labeled by seemingly bad models. Second, a hidden downside of TBAL systems is potentially prohibitive validation data usage. Together, these insights describe the promise and pitfalls of using such systems. We validate our theoretical guarantees with extensive experiments on synthetic and real datasets.

        ----

        ## [2263] CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society

        **Authors**: *Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html)

        **Abstract**:

        The rapid advancement of chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents, and provides insight into their “cognitive” processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing . Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of a society of agents, providing a valuable resource for investigating conversational language models. In particular, we conduct comprehensive studies on instruction-following cooperation in multi-agent settings. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond: https://github.com/camel-ai/camel.

        ----

        ## [2264] Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis

        **Authors**: *Abhinav Nippani, Dongyue Li, Haotian Ju, Haris K. Koutsopoulos, Hongyang R. Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We consider the problem of traffic accident analysis on a road network based on road network connections and traffic volume. Previous works have designed various deep-learning methods using historical records to predict traffic accident occurrences. However, there is a lack of consensus on how accurate existing methods are, and a fundamental issue is the lack of public accident datasets for comprehensive evaluations. This paper constructs a large-scale, unified dataset of traffic accident records from official reports of various states in the US, totaling 9 million records, accompanied by road networks and traffic volume reports. Using this new dataset, we evaluate existing deep-learning methods for predicting the occurrence of accidents on road networks. Our main finding is that graph neural networks such as GraphSAGE can accurately predict the number of accidents on roads with less than 22% mean absolute error (relative to the actual count) and whether an accident will occur or not with over 87% AUROC, averaged over states. We achieve these results by using multitask learning to account for cross-state variabilities (e.g., availability of accident labels) and transfer learning to combine traffic volume with accident prediction. Ablation studies highlight the importance of road graph-structural features, amongst other features. Lastly, we discuss the implications of the analysis and develop a package for easily using our new dataset.

        ----

        ## [2265] SUBP: Soft Uniform Block Pruning for 1×N Sparse CNNs Multithreading Acceleration

        **Authors**: *Jingyang Xiang, Siqi Li, Jun Chen, Guang Dai, Shipeng Bai, Yukai Ma, Yong Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a36c3dbe676fa8445715a31a90c66ab3-Abstract-Conference.html)

        **Abstract**:

        The study of sparsity in Convolutional Neural Networks (CNNs) has become widespread to compress and accelerate models in environments with limited resources. By constraining N consecutive weights along the output channel to be group-wise non-zero, the recent network with 1$\times$N sparsity has received tremendous popularity for its three outstanding advantages: 1) A large amount of storage space saving by a \emph{Block Sparse Row} matrix. 2) Excellent performance at a high sparsity. 3) Significant speedups on CPUs with Advanced Vector Extensions. Recent work requires selecting and fine-tuning 1$\times$N sparse weights based on dense pre-trained weights, leading to the problems such as expensive training cost and memory access, sub-optimal model quality, as well as unbalanced workload across threads (different sparsity across output channels). To overcome them, this paper proposes a novel \emph{\textbf{S}oft \textbf{U}niform \textbf{B}lock \textbf{P}runing} (SUBP) approach to train a uniform 1$\times$N sparse structured network from scratch. Specifically, our approach tends to repeatedly allow pruned blocks to regrow to the network based on block angular redundancy and importance sampling in a uniform manner throughout the training process. It not only makes the model less dependent on pre-training, reduces the model redundancy and the risk of pruning the important blocks permanently but also achieves balanced workload. Empirically, on ImageNet, comprehensive experiments across various CNN architectures show that our SUBP consistently outperforms existing 1$\times$N and structured sparsity methods based on pre-trained models or training from scratch. Source codes and models are available at \url{https://github.com/JingyangXiang/SUBP}.

        ----

        ## [2266] Adaptive Linear Estimating Equations

        **Authors**: *Mufang Ying, Koulik Khamaru, Cun-Hui Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a399456a191ca36c7c78dff367887f0a-Abstract-Conference.html)

        **Abstract**:

        Sequential data collection has emerged as a widely adopted technique for enhancing the efficiency of data gathering processes. Despite its advantages, such data collection mechanism often introduces complexities to the statistical inference procedure. For instance, the ordinary least squares (OLS) estimator in an adaptive linear regression model can exhibit non-normal asymptotic behavior, posing challenges for accurate inference and interpretation. In this paper, we propose a general method for constructing debiased estimator which remedies this issue. It makes use of the idea of adaptive linear estimating equations, and we establish theoretical guarantees of asymptotic normality, supplemented by discussions on achieving near-optimal asymptotic variance. A salient feature of our estimator is that in the context of multi-armed bandits,  our estimator retains the non-asymptotic performance of the least squares estimator while obtaining asymptotic normality property. Consequently, this work helps connect two fruitful paradigms of adaptive inference: a) non-asymptotic inference using concentration inequalities  and b) asymptotic inference via asymptotic normality.

        ----

        ## [2267] Robust Knowledge Transfer in Tiered Reinforcement Learning

        **Authors**: *Jiawei Huang, Niao He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a39ab46bf619ada0e90ceed846648a81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a39ab46bf619ada0e90ceed846648a81-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the Tiered Reinforcement Learning setting, a parallel transfer learning framework, where the goal is to transfer knowledge from the low-tier (source) task to the high-tier (target) task to reduce the exploration risk of the latter while solving the two tasks in parallel. Unlike previous work, we do not assume the low-tier and high-tier tasks share the same dynamics or reward functions, and focus on robust knowledge transfer without prior knowledge on the task similarity. We identify a natural and necessary condition called the ``Optimal Value Dominance'' for our objective. Under this condition, we propose novel online learning algorithms such that, for the high-tier task, it can achieve constant regret on partial states depending on the task similarity and retain near-optimal regret when the two tasks are dissimilar, while for the low-tier task, it can keep near-optimal without making sacrifice. Moreover, we further study the setting with multiple low-tier tasks, and propose a novel transfer source selection mechanism, which can ensemble the information from all low-tier tasks and allow provable benefits on a much larger state-action space.

        ----

        ## [2268] Bypassing the Simulator: Near-Optimal Adversarial Linear Contextual Bandits

        **Authors**: *Haolin Liu, Chen-Yu Wei, Julian Zimmert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3a661eb3308d0bb686f6a4bac521032-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3a661eb3308d0bb686f6a4bac521032-Abstract-Conference.html)

        **Abstract**:

        We consider the adversarial linear contextual bandit problem, where the loss vectors are selected fully adversarially and the per-round action set (i.e. the context) is drawn from a fixed distribution. Existing methods for this problem either require access to a simulator to generate free i.i.d. contexts, achieve a sub-optimal regret no better than $\tilde{\mathcal{O}}(T^{\frac{5}{6}})$, or are computationally inefficient. We greatly improve these results by achieving a regret of $\tilde{\mathcal{O}}(\sqrt{T})$ without a simulator, while maintaining computational efficiency when the action set in each round is small. In the special case of sleeping bandits with adversarial loss and stochastic arm availability, our result answers affirmatively the open question by [SGV20]  on whether there exists a polynomial-time algorithm with  $poly(d)\sqrt{T}$ regret. Our approach naturally handles the case where the loss is linear up to an additive misspecification error, and our regret shows near-optimal dependence on the magnitude of the error.

        ----

        ## [2269] GenEval: An object-focused framework for evaluating text-to-image alignment

        **Authors**: *Dhruba Ghosh, Hannaneh Hajishirzi, Ludwig Schmidt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recent breakthroughs in diffusion models, multimodal pretraining, and efficient finetuning have led to an explosion of text-to-image generative models. Given human evaluation is expensive and difficult to scale, automated methods are critical for evaluating the increasingly large number of new models. However, most current automated evaluation metrics like FID or CLIPScore only offer a distribution-level measure of image quality or image-text alignment, and are unsuited for fine-grained or instance-level analysis. In this paper, we introduce GenEval, an object-focused framework to evaluate compositional image properties such as object co-occurrence, position, count, and color. We show that current object detection models can be leveraged to evaluate text-to-image models on a variety of generation tasks with strong human agreement, and that other discriminative vision models can be linked to this pipeline to further verify properties like object color. We then evaluate several open-source text-to-image models and analyze their relative reasoning capabilities on our benchmark. We find that recent models demonstrate significant improvement on these tasks, though they are still lacking in complex capabilities such as spatial relations and attribute binding. Finally, we demonstrate how GenEval might be used to help discover existing failure modes, in order to inform development of the next generation of text-to-image models. Our code to run the GenEval framework will be made publicly available at https://github.com/djghosh13/geneval.

        ----

        ## [2270] Generalization in the Face of Adaptivity: A Bayesian Perspective

        **Authors**: *Moshe Shenfeld, Katrina Ligett*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3c01875a052f81d27a5211df096cd91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3c01875a052f81d27a5211df096cd91-Abstract-Conference.html)

        **Abstract**:

        Repeated use of a data sample via adaptively chosen queries can rapidly lead to overfitting, wherein the empirical evaluation of queries on the sample significantly deviates from their mean with respect to the underlying data distribution. It turns out that simple noise addition algorithms suffice to prevent this issue, and differential privacy-based analysis of these algorithms shows that they can handle an asymptotically optimal number of queries.  However, differential privacy's worst-case nature entails scaling such noise to the range of the queries even for highly-concentrated queries, or introducing more complex algorithms.In this paper, we prove that straightforward noise-addition algorithms already provide variance-dependent guarantees that also extend to unbounded queries. This improvement stems from a novel characterization that illuminates the core problem of adaptive data analysis. We show that the harm of adaptivity results from the covariance between the new query and a Bayes factor-based measure of how much information about the data sample was encoded in the responses given to past queries. We then leverage this characterization to introduce a new data-dependent stability notion that can bound this covariance.

        ----

        ## [2271] Convergence of Adam Under Relaxed Assumptions

        **Authors**: *Haochuan Li, Alexander Rakhlin, Ali Jadbabaie*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3cc50126338b175e56bb3cad134db0b-Abstract-Conference.html)

        **Abstract**:

        In this paper, we provide a rigorous proof of convergence of the Adaptive Moment Estimate (Adam) algorithm for a wide class of optimization objectives. Despite the popularity and efficiency of the Adam algorithm in training deep neural networks, its theoretical properties are not yet fully understood, and existing convergence proofs require unrealistically strong assumptions, such as globally bounded gradients, to show the convergence to stationary points. In this paper, we show that Adam provably converges to $\epsilon$-stationary points with $\mathcal{O}(\epsilon^{-4})$ gradient complexity under far more realistic conditions. The key to our analysis is a new proof of boundedness of gradients along the optimization trajectory of Adam, under a generalized smoothness assumption according to which the local smoothness (i.e., Hessian norm when it exists) is bounded by a sub-quadratic function of the gradient norm. Moreover, we propose a variance-reduced version of Adam with an accelerated gradient complexity of $\mathcal{O}(\epsilon^{-3})$.

        ----

        ## [2272] On the Convergence of Encoder-only Shallow Transformers

        **Authors**: *Yongtao Wu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a3cf318fbeec1126da21e9185ae9908c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a3cf318fbeec1126da21e9185ae9908c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we aim to build the global convergence theory of encoder-only shallow Transformers under a realistic setting from the perspective of architectures, initialization, and scaling under a finite width regime. The difficulty lies in how to tackle the softmax in self-attention mechanism, the core ingredient of Transformer. In particular, we diagnose the scaling scheme, carefully tackle the input/output of softmax, and prove that quadratic overparameterization is sufficient for global convergence of our shallow Transformers under commonly-used He/LeCun initialization in practice. Besides, neural tangent kernel (NTK) based analysis is also given, which facilitates a comprehensive comparison. Our theory demonstrates the separation on the importance of different scaling schemes and initialization. We believe our results can pave the way for a better understanding of modern Transformers, particularly on training dynamics.

        ----

        ## [2273] SoundCam: A Dataset for Finding Humans Using Room Acoustics

        **Authors**: *Mason Wang, Samuel Clarke, Jui-Hsien Wang, Ruohan Gao, Jiajun Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4289154c9209b679ac761a50d5fec3a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4289154c9209b679ac761a50d5fec3a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        A room’s acoustic properties are a product of the room’s geometry, the objects within the room, and their specific positions. A room’s acoustic properties can be characterized by its impulse response (RIR) between a source and listener location, or roughly inferred from recordings of natural signals present in the room. Variations in the positions of objects in a room can effect measurable changes in the room’s acoustic properties, as characterized by the RIR. Existing datasets of RIRs either do not systematically vary positions of objects in an environment, or they consist of only simulated RIRs. We present SoundCam, the largest dataset of unique RIRs from in-the-wild rooms publicly released to date. It includes 5,000 10-channel real-world measurements of room impulse responses and 2,000 10-channel recordings of music in three different rooms, including a controlled acoustic lab, an in-the-wild living room, and a conference room, with different humans in positions throughout each room. We show that these measurements can be used for interesting tasks, such as detecting and identifying humans, and tracking their positions.

        ----

        ## [2274] Accelerated On-Device Forward Neural Network Training with Module-Wise Descending Asynchronism

        **Authors**: *Xiaohan Zhao, Hualin Zhang, Zhouyuan Huo, Bin Gu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a42d8f43fae4d267e3084b10056153f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a42d8f43fae4d267e3084b10056153f7-Abstract-Conference.html)

        **Abstract**:

        On-device learning faces memory constraints when optimizing or fine-tuning on edge devices with limited resources. Current techniques for training deep models on edge devices rely heavily on backpropagation. However, its high memory usage calls for a reassessment of its dominance.In this paper, we propose forward gradient descent (FGD) as a potential solution to overcome the memory capacity limitation in on-device learning. However, FGD's dependencies across layers hinder parallel computation and can lead to inefficient resource utilization.To mitigate this limitation, we propose AsyncFGD, an asynchronous framework that decouples dependencies, utilizes module-wise stale parameters, and maximizes parallel computation. We demonstrate its convergence to critical points through rigorous theoretical analysis.Empirical evaluations conducted on NVIDIA's AGX Orin, a popular embedded device,  show that AsyncFGD reduces memory consumption and enhances hardware efficiency, offering a novel approach to on-device learning.

        ----

        ## [2275] Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

        **Authors**: *Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, Jieping Ye*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4316bb210a59fb7aafeca5dd21c2703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4316bb210a59fb7aafeca5dd21c2703-Abstract-Conference.html)

        **Abstract**:

        For a machine learning model deployed in real world scenarios, the ability of detecting out-of-distribution (OOD) samples is indispensable and challenging. Most existing OOD detection methods focused on exploring advanced training skills or training-free tricks to prevent the model from yielding overconfident confidence score for unknown samples. The training-based methods require expensive training cost and rely on OOD samples which are not always available, while most training-free methods can not efficiently utilize the prior information from the training data. In this work, we propose an \textbf{O}ptimal \textbf{P}arameter and \textbf{N}euron \textbf{P}runing (\textbf{OPNP}) approach, which aims to identify and remove those parameters and neurons that lead to over-fitting. The main method is divided into two steps. In the first step, we evaluate the sensitivity of the model parameters and neurons by averaging gradients over all training samples. In the second step, the parameters and neurons with exceptionally large or close to zero sensitivities are removed for prediction. Our proposal is training-free, compatible with other post-hoc methods, and exploring the information from all training data. Extensive experiments are performed on multiple OOD detection tasks and model architectures, showing that our proposed OPNP consistently outperforms the existing methods by a large margin.

        ----

        ## [2276] Unbalanced Low-rank Optimal Transport Solvers

        **Authors**: *Meyer Scetbon, Michal Klein, Giovanni Palla, Marco Cuturi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a439259e78294c38d157a51a2c40486b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a439259e78294c38d157a51a2c40486b-Abstract-Conference.html)

        **Abstract**:

        The relevance of optimal transport methods to machine learning has long been hindered by two salient limitations.First, the $O(n^3)$ computational cost of standard sample-based solvers (when used on batches of $n$ samples) is prohibitive.Second, the mass conservation constraint makes OT solvers too rigid in practice: because they must match \textit{all} points from both measures, their output can be heavily influenced by outliers.A flurry of recent works in OT has addressed these computational and modelling limitations, but has resulted in two separate strains of methods:While the computational outlook was much improved by entropic regularization, more recent $O(n)$ linear-time \textit{low-rank} solvers hold the promise to scale up OT further.On the other hand, modelling rigidities have been eased owing to unbalanced variants of OT, that rely on penalization terms to promote, rather than impose, mass conservation.The goal of this paper is to merge these two strains, to achieve the promise of \textit{both} versatile/scalable unbalanced/low-rank OT solvers. We propose custom algorithms to implement these extensions for the linear OT problem and its Fused-Gromov-Wasserstein generalization, and demonstrate their practical relevance to challenging spatial transcriptomics matching problems.

        ----

        ## [2277] Geodesic Multi-Modal Mixup for Robust Fine-Tuning

        **Authors**: *Changdae Oh, Junhyuk So, Hoyoon Byun, YongTaek Lim, Minchul Shin, Jong-June Jeon, Kyungwoo Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45296e83b19f656392e0130d9e53cb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45296e83b19f656392e0130d9e53cb1-Abstract-Conference.html)

        **Abstract**:

        Pre-trained multi-modal models, such as CLIP, provide transferable embeddings and show promising results in diverse applications. However, the analysis of learned multi-modal embeddings is relatively unexplored, and the embedding transferability can be improved. In this work, we observe that CLIP holds separated embedding subspaces for two different modalities, and then we investigate it through the lens of \textit{uniformity-alignment} to measure the quality of learned representation. Both theoretically and empirically, we show that CLIP retains poor uniformity and alignment even after fine-tuning. Such a lack of alignment and uniformity might restrict the transferability and robustness of embeddings. To this end, we devise a new fine-tuning method for robust representation equipping better alignment and uniformity. First, we propose a \textit{Geodesic Multi-Modal Mixup} that mixes the embeddings of image and text to generate hard negative samples on the hypersphere. Then, we fine-tune the model on hard negatives as well as original negatives and positives with contrastive loss. Based on the theoretical analysis about hardness guarantee and limiting behavior, we justify the use of our method. Extensive experiments on retrieval, calibration, few- or zero-shot classification (under distribution shift), embedding arithmetic, and image captioning further show that our method provides transferable representations, enabling robust model adaptation on diverse tasks.

        ----

        ## [2278] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

        **Authors**: *Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html)

        **Abstract**:

        Large language models(LLMs) have sparked a new wave of exciting AI applications. Hosting these models at scale requires significant memory resources. One crucial memory bottleneck for the deployment stems from the context window. It is commonly recognized that model weights are memory hungry; however, the size of key-value embedding stored during the generation process (KV cache) can easily surpass the model size. The enormous size of the KV cache puts constraints on the inference batch size, which is crucial for high throughput inference workload. Inspired by an interesting observation of the attention scores, we hypothesize the persistence of importance: only pivotal tokens, which had a substantial influence at one step, will significantly influence future generations. Based on our empirical verification and theoretical analysis around this hypothesis, we propose scissorhands, a system that maintains the memory usage of the KV cache at a fixed budget without finetuning the model. In essence, Scissorhands manages the KV cache by storing the pivotal tokens with a higher probability. We validate that scissorhands reduces the inference memory usage of the KV cache by up to 5$\times$ without compromising model quality. We further demonstrate that scissorhands can be combined with 4-bit quantization, traditionally used to compress model weights, to achieve up to 20$\times$ compression.

        ----

        ## [2279] Asymmetric Certified Robustness via Feature-Convex Neural Networks

        **Authors**: *Samuel Pfrommer, Brendon G. Anderson, Julien Piet, Somayeh Sojoudi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45b205c10ef082515cacae80555bbef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45b205c10ef082515cacae80555bbef-Abstract-Conference.html)

        **Abstract**:

        Real-world adversarial attacks on machine learning models often feature an asymmetric structure wherein adversaries only attempt to induce false negatives (e.g., classify a spam email as not spam). We formalize the asymmetric robustness certification problem and correspondingly present the feature-convex neural network architecture, which composes an input-convex neural network (ICNN) with a Lipschitz continuous feature map in order to achieve asymmetric adversarial robustness. We consider the aforementioned binary setting with one "sensitive" class, and for this class we prove deterministic, closed-form, and easily-computable certified robust radii for arbitrary $\ell_p$-norms. We theoretically justify the use of these models by characterizing their decision region geometry, extending the universal approximation theorem for ICNN regression to the classification setting, and proving a lower bound on the probability that such models perfectly fit even unstructured uniformly distributed data in sufficiently high dimensions. Experiments on Malimg malware classification and subsets of the MNIST, Fashion-MNIST, and CIFAR-10 datasets show that feature-convex classifiers attain substantial certified $\ell_1$, $\ell_2$, and $\ell_{\infty}$-radii while being far more computationally efficient than competitive baselines.

        ----

        ## [2280] A Unified Fast Gradient Clipping Framework for DP-SGD

        **Authors**: *Weiwei Kong, Andrés Muñoz Medina*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a45d344b28179c8da7646bc38ff50ad8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a45d344b28179c8da7646bc38ff50ad8-Abstract-Conference.html)

        **Abstract**:

        A well-known numerical bottleneck in the differentially-private stochastic gradient descent (DP-SGD) algorithm is the computation of the gradient norm for each example in a large input batch. When the loss function in DP-SGD is consists of an intermediate linear operation, existing methods in the literature have proposed decompositions of gradients that are amenable to fast norm computations. In this paper, we present a framework that generalizes the above approach to arbitrary (possibly nonlinear) intermediate operations. Moreover, we show that for certain operations, such as fully-connected and embedding layer computations, further improvements to the runtime and storage costs of existing decompositions can be deduced using certain components of our framework. Finally, preliminary numerical experiments are given to demonstrate the substantial effects of the aforementioned improvements.

        ----

        ## [2281] Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization

        **Authors**: *Xiangsen Wang, Haoran Xu, Yinan Zheng, Xianyuan Zhan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a46c84276e3a4249ab7dbf3e069baf7f-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (RL) has received considerable attention in recent years due to its attractive capability of learning policies from offline datasets without environmental interactions. Despite some success in the single-agent setting, offline multi-agent RL (MARL) remains to be a challenge. The large joint state-action space and the coupled multi-agent behaviors pose extra complexities for offline policy optimization. Most existing offline MARL studies simply apply offline data-related regularizations on individual agents, without fully considering the multi-agent system at the global level. In this work, we present OMIGA, a new offline multi-agent RL algorithm with implicit global-to-local value regularization. OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. Based on comprehensive experiments on the offline multi-agent MuJoCo and StarCraft II micro-management tasks, we show that OMIGA achieves superior performance over the state-of-the-art offline MARL methods in almost all tasks.

        ----

        ## [2282] Benchmarking Large Language Models on CMExam - A comprehensive Chinese Medical Exam Dataset

        **Authors**: *Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, Lei Zhu, Michael Lingzhi Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a48ad12d588c597f4725a8b84af647b5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recent advancements in large language models (LLMs) have transformed the field of question answering (QA). However, evaluating LLMs in the medical field is challenging due to the lack of standardized and comprehensive datasets. To address this gap, we introduce CMExam, sourced from the Chinese National Medical Licensing Examination. CMExam consists of 60K+ multiple-choice questions for standardized and objective evaluations, as well as solution explanations for model reasoning evaluation in an open-ended manner. For in-depth analyses of LLMs, we invited medical professionals to label five additional question-wise annotations, including disease groups, clinical departments, medical disciplines, areas of competency, and question difficulty levels. Alongside the dataset, we further conducted thorough experiments with representative LLMs and QA algorithms on CMExam. The results show that GPT-4 had the best accuracy of 61.6% and a weighted F1 score of 0.617. These results highlight a great disparity when compared to human accuracy, which stood at 71.6%. For explanation tasks, while LLMs could generate relevant reasoning and demonstrate improved performance after finetuning, they fall short of a desired standard, indicating ample room for improvement. To the best of our knowledge, CMExam is the first Chinese medical exam dataset to provide comprehensive medical annotations. The experiments and findings of LLM evaluation also provide valuable insights into the challenges and potential solutions in developing Chinese medical QA systems and LLM evaluation pipelines.

        ----

        ## [2283] A Logic for Expressing Log-Precision Transformers

        **Authors**: *William Merrill, Ashish Sabharwal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html)

        **Abstract**:

        One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed that finite-precision transformer classifiers can be equivalently expressed in a generalization of first-order logic. However, finite-precision transformers are a weak transformer variant because, as we show, a single head can only attend to a constant number of tokens and, in particular, cannot represent uniform attention. Since attending broadly is a core capability for transformers, we ask whether a minimally more expressive model that can attend universally can also be characterized in logic. To this end, we analyze transformers whose forward pass is computed in $\log n$ precision on contexts of length $n$. We prove any log-precision transformer classifier can be equivalently expressed as a first-order logic sentence that, in addition to standard universal and existential quantifiers, may also contain majority-vote quantifiers. This is the tightest known upper bound and first logical characterization of log-precision transformers.

        ----

        ## [2284] Universal Prompt Tuning for Graph Neural Networks

        **Authors**: *Taoran Fang, Yunchao Zhang, Yang Yang, Chunping Wang, Lei Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html)

        **Abstract**:

        In recent years, prompt tuning has sparked a research surge in adapting pre-trained models. Unlike the unified pre-training strategy employed in the language field, the graph field exhibits diverse pre-training strategies, posing challenges in designing appropriate prompt-based tuning methods for graph neural networks. While some pioneering work has devised specialized prompting functions for models that employ edge prediction as their pre-training tasks, these methods are limited to specific pre-trained GNN models and lack broader applicability. In this paper, we introduce a universal prompt-based tuning method called Graph Prompt Feature (GPF) for pre-trained GNN models under any pre-training strategy. GPF operates on the input graph's feature space and can theoretically achieve an equivalent effect to any form of prompting function. Consequently, we no longer need to illustrate the prompting function corresponding to each pre-training strategy explicitly. Instead, we employ GPF to obtain the prompted graph for the downstream task in an adaptive manner. We provide rigorous derivations to demonstrate the universality of GPF and make guarantee of its effectiveness. The experimental results under various pre-training strategies indicate that our method performs better than fine-tuning, with an average improvement of about 1.4% in full-shot scenarios and about 3.2% in few-shot scenarios. Moreover, our method significantly outperforms existing specialized prompt-based tuning methods when applied to models utilizing the pre-training strategy they specialize in. These numerous advantages position our method as a compelling alternative to fine-tuning for downstream adaptations.

        ----

        ## [2285] Stochastic Approximation Approaches to Group Distributionally Robust Optimization

        **Authors**: *Lijun Zhang, Peng Zhao, Zhen-Hua Zhuang, Tianbao Yang, Zhi-Hua Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4b6ad6b48850c0c331d1259fc66a69c-Abstract-Conference.html)

        **Abstract**:

        This paper investigates group distributionally robust optimization (GDRO), with the purpose to learn a model that performs well over $m$ different distributions. First, we formulate GDRO as a stochastic convex-concave saddle-point problem, and demonstrate that stochastic mirror descent (SMD), using $m$ samples in each iteration, achieves an $O(m (\log m)/\epsilon^2)$ sample complexity for finding an $\epsilon$-optimal solution, which matches the $\Omega(m/\epsilon^2)$ lower bound up to a logarithmic factor. Then, we make use of techniques from online learning to reduce the number of samples required in each round from $m$ to $1$, keeping the same sample complexity. Specifically, we cast GDRO as a two-players game where one player simply performs SMD and the other executes an online algorithm for non-oblivious multi-armed bandits. Next, we consider a more practical scenario where the number of samples that can be drawn from each distribution is different, and propose a novel formulation of weighted GDRO, which allows us to derive distribution-dependent convergence rates.  Denote by $n_i$ the sample budget for the $i$-th distribution, and assume $n_1 \geq n_2 \geq \cdots \geq n_m$. In the first approach, we incorporate non-uniform sampling into SMD such that the sample budget is satisfied in expectation, and prove that the excess risk of the $i$-th distribution decreases at an $O(\sqrt{n_1 \log m}/n_i)$ rate. In the second approach, we use mini-batches to meet the budget exactly and also reduce the variance in stochastic gradients, and then leverage stochastic mirror-prox algorithm, which can exploit small variances, to optimize a carefully designed weighted GDRO problem. Under appropriate conditions, it attains an $O((\log m)/\sqrt{n_i})$ convergence rate, which almost matches the optimal $O(\sqrt{1/n_i})$ rate of only learning from the $i$-th distribution with $n_i$ samples.

        ----

        ## [2286] Learning Efficient Surrogate Dynamic Models with Graph Spline Networks

        **Authors**: *Chuanbo Hua, Federico Berto, Michael Poli, Stefano Massaroli, Jinkyoo Park*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4c3a66ed818455b8bbe591b6a5d0f56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4c3a66ed818455b8bbe591b6a5d0f56-Abstract-Conference.html)

        **Abstract**:

        While complex simulations of physical systems have been widely used in engineering and scientific computing, lowering their often prohibitive computational requirements has only recently been tackled by deep learning approaches. In this paper, we present GraphSplineNets, a novel deep-learning method to speed up the forecasting of physical systems by reducing the grid size and number of iteration steps of deep surrogate models. Our method uses two differentiable orthogonal spline collocation methods to efficiently predict response at any location in time and space. Additionally, we introduce an adaptive collocation strategy in space to prioritize sampling from the most important regions. GraphSplineNets improve the accuracy-speedup tradeoff in forecasting various dynamical systems with increasing complexity, including the heat equation, damped wave propagation, Navier-Stokes equations, and real-world ocean currents in both regular and irregular domains.

        ----

        ## [2287] Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing

        **Authors**: *Wei Dong, Dawei Yan, Zhijun Lin, Peng Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)

        **Abstract**:

        The advent of high-capacity pre-trained models has revolutionized problem-solving in computer vision, shifting the focus from training task-specific models to adapting pre-trained models. Consequently, effectively adapting large pre-trained models to downstream tasks in an efficient manner has become a prominent research area. Existing solutions primarily concentrate on designing lightweight adapters and their interaction with pre-trained models, with the goal of minimizing the number of parameters requiring updates. In this study, we propose a novel Adapter Re-Composing (ARC) strategy that addresses efficient pre-trained model adaptation from a fresh perspective. Our approach considers the reusability of adaptation parameters and introduces a parameter-sharing scheme. Specifically, we leverage symmetric down-/up-projections to construct bottleneck operations, which are shared across layers. By learning low-dimensional re-scaling coefficients, we can effectively re-compose layer-adaptive adapters. This parameter-sharing strategy in adapter design allows us to further reduce the number of new parameters while maintaining satisfactory performance, thereby offering a promising approach to compress the adaptation cost. We conduct experiments on 24 downstream image classification tasks using various Vision Transformer variants to evaluate our method. The results demonstrate that our approach achieves compelling transfer learning performance with a reduced parameter count. Our code is available at https://github.com/DavidYanAnDe/ARC.

        ----

        ## [2288] Hardness of Low Rank Approximation of Entrywise Transformed Matrix Products

        **Authors**: *Tamás Sarlós, Xingyou Song, David P. Woodruff, Richard Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4d92f656cc99f60fe1bfc98386aee34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4d92f656cc99f60fe1bfc98386aee34-Abstract-Conference.html)

        **Abstract**:

        Inspired by fast algorithms in natural language processing, we study low rank approximation in the entrywise transformed setting where we want to find a good rank $k$ approximation to $f(U \cdot V)$, where $U, V^\top \in \mathbb{R}^{n \times r}$ are given, $r = O(\log(n))$, and $f(x)$ is a general scalar function. Previous work in sublinear low rank approximation has shown that if both (1) $U = V^\top$ and (2) $f(x)$ is a PSD kernel function, then there is an $O(nk^{\omega-1})$ time constant relative error approximation algorithm, where $\omega \approx 2.376$ is the exponent of matrix multiplication. We give the first conditional time hardness results for this problem, demonstrating that both conditions (1) and (2) are in fact necessary for getting better than $n^{2-o(1)}$ time for a relative error low rank approximation for a wide class of functions. We give novel reductions from the Strong Exponential Time Hypothesis (SETH) that rely on lower bounding the leverage scores of flat sparse vectors and hold even when the rank of the transformed matrix $f(UV)$ and the target rank are $n^{o(1)}$, and when $U = V^\top$. Furthermore, even when $f(x) = x^p$ is a simple polynomial, we give runtime lower bounds in the case when $U \neq V^\top$ of the form $\Omega(\min(n^{2-o(1)}, \Omega(2^p)))$. Lastly, we demonstrate that our lower bounds are tight by giving an $O(n \cdot \text{poly}(k, 2^p, 1/\epsilon))$ time relative error approximation algorithm and a fast $O(n \cdot \text{poly}(k, p, 1/\epsilon))$ additive error approximation using fast tensor-based sketching. Additionally, since our low rank algorithms rely on matrix-vector product subroutines, our lower bounds extend to show that computing $f(UV)W$, for even a small matrix $W$, requires $\Omega(n^{2-o(1)})$ time.

        ----

        ## [2289] Efficient Training of Energy-Based Models Using Jarzynski Equality

        **Authors**: *Davide Carbone, Mengjian Hua, Simon Coste, Eric Vanden-Eijnden*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4ddb865e0a8ca3cca43fd7387b4b0da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4ddb865e0a8ca3cca43fd7387b4b0da-Abstract-Conference.html)

        **Abstract**:

        Energy-based models (EBMs) are generative models inspired by statistical physics with a wide range of applications in unsupervised learning.  Their performance is well measured by the cross-entropy (CE) of the model distribution relative to the data distribution. Using the CE as the objective for training is however challenging because the computation of its gradient with respect to the model parameters requires sampling the model distribution. Here we show how results for nonequilibrium thermodynamics based on Jarzynski equality together with tools from sequential Monte-Carlo sampling can be used to perform this computation efficiently and avoid the uncontrolled approximations made using the standard contrastive divergence algorithm.  Specifically, we introduce a modification of the unadjusted Langevin algorithm (ULA) in which each walker acquires a  weight that enables the estimation of the gradient of the cross-entropy at any step during GD, thereby bypassing sampling biases induced by slow mixing of ULA. We illustrate these results with numerical experiments on Gaussian mixture distributions as well as the MNIST and CIFAR-10 datasets. We show that the proposed approach outperforms  methods based on the contrastive divergence algorithm in all the considered situations.

        ----

        ## [2290] High Precision Causal Model Evaluation with Conditional Randomization

        **Authors**: *Chao Ma, Cheng Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4dfcbcfa1f0425cd18aafa35a68019a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4dfcbcfa1f0425cd18aafa35a68019a-Abstract-Conference.html)

        **Abstract**:

        The gold standard for causal model evaluation involves comparing model predictions with true effects estimated from randomized controlled trials (RCT). However, RCTs are not always feasible or ethical to perform. In contrast, conditionally randomized experiments based on inverse probability weighting (IPW) offer a more realistic approach but may suffer from high estimation variance. To tackle this challenge and enhance causal model evaluation in real-world conditional randomization settings, we introduce a novel low-variance estimator for causal error, dubbed as the pairs estimator. By applying the same IPW estimator to both the model and true experimental effects, our estimator effectively cancels out the variance due to IPW and achieves a smaller asymptotic variance. Empirical studies demonstrate the improved of our estimator, highlighting its potential on achieving near-RCT performance. Our method offers a simple yet powerful solution to evaluate causal inference models in conditional randomization settings without complicated modification of the IPW estimator itself, paving the way for more robust and reliable model assessments.

        ----

        ## [2291] Reducing Blackwell and Average Optimality to Discounted MDPs via the Blackwell Discount Factor

        **Authors**: *Julien Grand-Clément, Marek Petrik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a4e720ce31ccd8ba747d8863e1580fa8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a4e720ce31ccd8ba747d8863e1580fa8-Abstract-Conference.html)

        **Abstract**:

        We introduce the Blackwell discount factor for Markov Decision Processes (MDPs). Classical objectives for MDPs include discounted, average, and Blackwell optimality. Many existing approaches to computing average-optimal policies solve for discount-optimal policies with a discount factor close to $1$, but they only work under strong or hard-to-verify assumptions on the MDP structure  such as unichain or ergodicity. We are the first to highlight the shortcomings of the classical definition of Blackwell optimality, which does not lead to simple algorithms for computing Blackwell-optimal policies and overlooks the pathological behaviors of optimal policies as regards the discount factors. To resolve this issue, in this paper, we show that when the discount factor is larger than  the Blackwell discount factor $\gamma_{\sf bw}$, all discount-optimal policies become Blackwell- and average-optimal, and we derive a general upper bound on $\gamma_{\sf bw}$. Our upper bound on $\gamma_{\sf bw}$, parametrized by the bit-size of the rewards and transition probabilities of the MDP instance, provides the first reduction from average and Blackwell optimality to discounted optimality, without any assumptions, along with new polynomial-time algorithms. Our work brings new ideas from polynomials and algebraic numbers to the analysis of MDPs. Our results also apply to robust MDPs, enabling the first algorithms to compute robust Blackwell-optimal policies.

        ----

        ## [2292] Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits

        **Authors**: *Muhammad Faaiz Taufiq, Arnaud Doucet, Rob Cornish, Jean-Francois Ton*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a51f974947c42b40a40a882a7d9b2479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a51f974947c42b40a40a882a7d9b2479-Abstract-Conference.html)

        **Abstract**:

        Off-Policy Evaluation (OPE) in contextual bandits is crucial for assessing new policies using existing data without costly experimentation. However, current OPE methods, such as Inverse Probability Weighting (IPW) and Doubly Robust (DR) estimators, suffer from high variance, particularly in cases of low overlap between target and behaviour policies or large action and context spaces. In this paper, we introduce a new OPE estimator for contextual bandits, the Marginal Ratio (MR) estimator, which focuses on the shift in the marginal distribution of outcomes $Y$ instead of the policies themselves. Through rigorous theoretical analysis, we demonstrate the benefits of the MR estimator compared to conventional methods like IPW and DR in terms of variance reduction. Additionally, we establish a connection between the MR estimator and the state-of-the-art Marginalized Inverse Propensity Score (MIPS) estimator, proving that MR achieves lower variance among a generalized family of MIPS estimators. We further illustrate the utility of the MR estimator in causal inference settings, where it exhibits enhanced performance in estimating Average Treatment Effects (ATE). Our experiments on synthetic and real-world datasets corroborate our theoretical findings and highlight the practical advantages of the MR estimator in OPE for contextual bandits.

        ----

        ## [2293] SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs

        **Authors**: *Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin P. Murphy, Alexander G. Hauptmann, Lu Jiang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a526cc8f6ffb74bedb6ff313e3fdb450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a526cc8f6ffb74bedb6ff313e3fdb450-Abstract-Conference.html)

        **Abstract**:

        In this work, we introduce Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform both understanding and generation tasks involving non-linguistic modalities such as images or videos. SPAE converts between raw pixels and interpretable lexical tokens (or words) extracted from the LLM's vocabulary. The resulting tokens capture both the rich semantic meaning and the fine-grained details needed for visual reconstruction, effectively translating the visual content into a language comprehensible to the LLM, and empowering it to perform a wide array of multimodal tasks. Our approach is validated through in-context learning experiments with frozen PaLM 2 and GPT 3.5 on a diverse set of image understanding and generation tasks.Our method marks the first successful attempt to enable a frozen LLM to generate image content while surpassing state-of-the-art performance in image understanding tasks, under the same setting, by over 25%.

        ----

        ## [2294] Energy-based learning algorithms for analog computing: a comparative study

        **Authors**: *Benjamin Scellier, Maxence Ernoult, Jack D. Kendall, Suhas Kumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a52b0d191b619477cc798d544f4f0e4b-Abstract-Conference.html)

        **Abstract**:

        Energy-based learning algorithms have recently gained a surge of interest due to their compatibility with analog (post-digital) hardware. Existing algorithms include contrastive learning (CL), equilibrium propagation (EP) and coupled learning (CpL), all consisting in contrasting two states, and differing in the type of perturbation used to obtain the second state from the first one. However, these algorithms have never been explicitly compared on equal footing with same models and datasets, making it difficult to assess their scalability and decide which one to select in practice. In this work, we carry out a comparison of seven learning algorithms, namely CL and different variants of EP and CpL depending on the signs of the perturbations. Specifically, using these learning algorithms, we train deep convolutional Hopfield networks (DCHNs) on five vision tasks (MNIST, F-MNIST, SVHN, CIFAR-10 and CIFAR-100). We find that, while all algorithms yield comparable performance on MNIST, important differences in performance arise as the difficulty of the task increases. Our key findings reveal that negative perturbations are better than positive ones, and highlight the centered variant of EP (which uses two perturbations of opposite sign) as the best-performing algorithm. We also endorse these findings with theoretical arguments. Additionally, we establish new SOTA results with DCHNs on all five datasets, both in performance and speed. In particular, our DCHN simulations are 13.5 times faster with respect to Laborieux et al. (2021), which we achieve thanks to the use of a novel energy minimisation algorithm based on asynchronous updates, combined with reduced precision (16 bits).

        ----

        ## [2295] Distribution Learnability and Robustness

        **Authors**: *Shai Ben-David, Alex Bie, Gautam Kamath, Tosca Lechner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5321f64005b0d4a94d0b18e84e19f48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5321f64005b0d4a94d0b18e84e19f48-Abstract-Conference.html)

        **Abstract**:

        We examine the relationship between learnability and robust learnability for the problem of distribution learning.We show that learnability implies robust learnability if the adversary can only perform additive contamination (and consequently, under Huber contamination), but not if the adversary is allowed to perform subtractive contamination. Thus, contrary to other learning settings (e.g., PAC learning of function classes), realizable learnability does not imply agnostic learnability. We also explore related implications in the context of compression schemes and  differentially private learnability.

        ----

        ## [2296] Behavior Alignment via Reward Function Optimization

        **Authors**: *Dhawal Gupta, Yash Chandak, Scott M. Jordan, Philip S. Thomas, Bruno C. da Silva*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5357781c204d4412e44ed9cbcdb08d5-Abstract-Conference.html)

        **Abstract**:

        Designing reward functions for efficiently guiding reinforcement learning (RL) agents toward specific behaviors is a complex task.This is challenging since it requires the identification of reward structures that are not sparse and that avoid inadvertently inducing undesirable behaviors. Naively modifying the reward structure to offer denser and more frequent feedback can lead to unintended outcomes and promote behaviors that are not aligned with the designer's intended goal. Although potential-based reward shaping is often suggested as a remedy, we systematically investigate settings where deploying it often significantly impairs performance. To address these issues, we introduce a new framework that uses a bi-level objective to learn \emph{behavior alignment reward functions}. These functions integrate auxiliary rewards reflecting a designer's heuristics and domain knowledge with the environment's primary rewards. Our approach automatically determines the most effective way to blend these types of feedback, thereby enhancing robustness against heuristic reward misspecification. Remarkably, it can also adapt an agent's policy optimization process to mitigate suboptimalities resulting from limitations and biases inherent in the underlying RL algorithms. We evaluate our method's efficacy on a diverse set of tasks, from small-scale experiments to high-dimensional control challenges. We investigate heuristic auxiliary rewards of varying quality---some of which are beneficial and others detrimental to the learning process. Our results show that our framework offers a robust and principled way to integrate designer-specified heuristics. It not only addresses key shortcomings of existing approaches but also consistently leads to high-performing solutions, even when given misaligned or poorly-specified auxiliary reward functions.

        ----

        ## [2297] Tuning Multi-mode Token-level Prompt Alignment across Modalities

        **Authors**: *Dongsheng Wang, Miaoge Li, Xinyang Liu, Mingsheng Xu, Bo Chen, Hanwang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a547d86953a4e36aa8a1390e6f4708e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a547d86953a4e36aa8a1390e6f4708e2-Abstract-Conference.html)

        **Abstract**:

        Advancements in prompt tuning of vision-language models have underscored their potential in enhancing open-world visual concept comprehension. However, prior works only primarily focus on single-mode (only one prompt for each modality) and holistic level (image or sentence) semantic alignment, which fails to capture the sample diversity, leading to sub-optimal prompt discovery. To address the limitation, we propose a multi-mode token-level tuning framework that leverages the optimal transportation to learn and align a set of prompt tokens across modalities. Specifically, we rely on two essential factors: 1) multi-mode prompts discovery, which guarantees diverse semantic representations, and 2) token-level alignment, which helps explore fine-grained similarity. Consequently, the similarity can be calculated as a hierarchical transportation problem between the modality-specific sets. Extensive experiments on popular image recognition benchmarks show the superior generalization and few-shot abilities of our approach. The qualitative analysis demonstrates that the learned prompt tokens have the ability to capture diverse visual concepts.

        ----

        ## [2298] Censored Sampling of Diffusion Models Using 3 Minutes of Human Feedback

        **Authors**: *Taeho Yoon, Kibeom Myoung, Keon Lee, Jaewoong Cho, Albert No, Ernest K. Ryu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5755ccd0efeca8852ae0a1193f319f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5755ccd0efeca8852ae0a1193f319f6-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have recently shown remarkable success in high-quality image generation. Sometimes, however, a pre-trained diffusion model exhibits partial misalignment in the sense that the model can generate good images, but it sometimes outputs undesirable images. If so, we simply need to prevent the generation of the bad images, and we call this task censoring. In this work, we present censored generation with a pre-trained diffusion model using a reward model trained on minimal human feedback. We show that censoring can be accomplished with extreme human feedback efficiency and that labels generated with a mere few minutes of human feedback are sufficient.

        ----

        ## [2299] Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics

        **Authors**: *Leon Klein, Andrew Y. K. Foong, Tor Erlend Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, Ryota Tomioka*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a598c367280f9054434fdcc227ce4d38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a598c367280f9054434fdcc227ce4d38-Abstract-Conference.html)

        **Abstract**:

        *Molecular dynamics* (MD) simulation is a widely used technique to simulate molecular systems, most commonly at the all-atom resolution where equations of motion are integrated with timesteps on the order of femtoseconds ($1\textrm{fs}=10^{-15}\textrm{s}$). MD is often used to compute equilibrium properties, which requires sampling from an equilibrium distribution such as the Boltzmann distribution. However, many important processes, such as binding and folding, occur over timescales of milliseconds or beyond, and cannot be efficiently sampled with conventional MD.Furthermore, new MD simulations need to be performed for each molecular system studied.We present *Timewarp*, an enhanced sampling method which uses a normalising flow as a proposal distribution in a Markov chain Monte Carlo method targeting the Boltzmann distribution. The flow is trained offline on MD trajectories and learns to make large steps in time, simulating the molecular dynamics of $10^{5} - 10^{6} \textrm{fs}$.Crucially, Timewarp is *transferable* between molecular systems: once trained, we show that it generalises to unseen small peptides (2-4 amino acids) at all-atom resolution, exploring their metastable states and providing wall-clock acceleration of sampling compared to standard MD.Our method constitutes an important step towards general, transferable algorithms for accelerating MD.

        ----

        ## [2300] Harnessing Hard Mixed Samples with Decoupled Regularizer

        **Authors**: *Zicheng Liu, Siyuan Li, Ge Wang, Lirong Wu, Cheng Tan, Stan Z. Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5c47c1b7adf19e8dc633812a4acf6d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5c47c1b7adf19e8dc633812a4acf6d2-Abstract-Conference.html)

        **Abstract**:

        Mixup is an efficient data augmentation approach that improves the generalization of neural networks by smoothing the decision boundary with mixed data. Recently, dynamic mixup methods have improved previous \textit{static} policies effectively (e.g., linear interpolation) by maximizing target-related salient regions in mixed samples, but excessive additional time costs are not acceptable. These additional computational overheads mainly come from optimizing the mixed samples according to the mixed labels. However, we found that the extra optimizing step may be redundant because label-mismatched mixed samples are informative hard mixed samples for deep models to localize discriminative features. In this paper, we thus are not trying to propose a more complicated dynamic mixup policy but rather an efficient mixup objective function with decoupled regularizer, named decoupled mixup (DM). The primary effect is that DM can adaptively utilize those hard mixed samples to mine discriminative features without losing the original smoothness of mixup. As a result, DM enables static mixup methods to achieve comparable or even exceed the performance of dynamic methods without any extra computation. This also leads to an interesting objective design problem for mixup training that we need to focus on both smoothing the decision boundaries and identifying discriminative features. Extensive experiments on supervised and semi-supervised learning benchmarks across seven datasets validate the effectiveness of DM.

        ----

        ## [2301] The Utility of "Even if" Semifactual Explanation to Optimise Positive Outcomes

        **Authors**: *Eoin M. Kenny, Weipeng Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e146ca55a2b18be41942cfa677123d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e146ca55a2b18be41942cfa677123d-Abstract-Conference.html)

        **Abstract**:

        When users receive either a positive or negative outcome from an automated system, Explainable AI (XAI) has almost exclusively focused on how to mutate negative outcomes into positive ones by crossing a decision boundary using counterfactuals (e.g., "If you earn 2k more, we will accept your loan application"). Here, we instead focus on positive outcomes, and take the novel step of using XAI to optimise them (e.g., "Even if you wish to half your down-payment, we will still accept your loan application"). Explanations such as these that employ "even if..." reasoning, and do not cross a decision boundary, are known as semifactuals. To instantiate semifactuals in this context, we introduce the concept of Gain (i.e., how much a user stands to benefit from the explanation), and consider the first causal formalisation of semifactuals. Tests on benchmark datasets show our algorithms are better at maximising gain compared to prior work, and that causality is important in the process. Most importantly however, a user study supports our main hypothesis by showing people find semifactual explanations more useful than counterfactuals when they receive the positive outcome of a loan acceptance.

        ----

        ## [2302] VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models

        **Authors**: *Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html)

        **Abstract**:

        Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLATTACK to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multi-modal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multi-modal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level.  We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLATTACK framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models.

        ----

        ## [2303] Mode Connectivity in Auction Design

        **Authors**: *Christoph Hertrich, Yixin Tao, László A. Végh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5e4907a40c0dcb8433a35c714ba9d79-Abstract-Conference.html)

        **Abstract**:

        Optimal auction design is a fundamental problem in algorithmic game theory. This problem is notoriously difficult already in very simple settings. Recent work in differentiable economics showed that neural networks can efficiently learn known optimal auction mechanisms and discover interesting new ones. In an attempt to theoretically justify their empirical success, we focus on one of the first such networks, RochetNet, and a generalized version for affine maximizer auctions. We prove that they satisfy mode connectivity, i.e., locally optimal solutions are connected by a simple, piecewise linear path such that every solution on the path is almost as good as one of the two local optima. Mode connectivity has been recently investigated as an intriguing empirical and theoretically justifiable property of neural networks used for prediction problems. Our results give the first such analysis in the context of differentiable economics, where neural networks are used directly for solving non-convex optimization problems.

        ----

        ## [2304] Katakomba: Tools and Benchmarks for Data-Driven NetHack

        **Authors**: *Vladislav Kurenkov, Alexander Nikulin, Denis Tarasov, Sergey Kolesnikov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a5f596699d8d4637532f955c7f2860f4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a5f596699d8d4637532f955c7f2860f4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        NetHack is known as the frontier of reinforcement learning research where learning-based methods still need to catch up to rule-based solutions. One of the promising directions for a breakthrough is using pre-collected datasets similar to recent developments in robotics, recommender systems, and more under the umbrella of offline reinforcement learning (ORL). Recently, a large-scale NetHack dataset was released; while it was a necessary step forward, it has yet to gain wide adoption in the ORL community. In this work, we argue that there are three major obstacles for adoption: tool-wise, implementation-wise, and benchmark-wise. To address them, we develop an open-source library that provides workflow fundamentals familiar to the ORL community: pre-defined D4RL-style tasks, uncluttered baseline implementations, and reliable evaluation tools with accompanying configs and logs synced to the cloud.

        ----

        ## [2305] Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model

        **Authors**: *Peter Súkeník, Marco Mondelli, Christoph H. Lampert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a60c43ba078b723d3d517d28c50ded4c-Abstract-Conference.html)

        **Abstract**:

        Neural collapse (NC) refers to the surprising structure of the last layer of deep neural networks in the terminal phase of gradient descent training. Recently, an increasing amount of experimental evidence has pointed to the propagation of NC to earlier layers of neural networks. However, while the NC in the last layer is well studied theoretically, much less is known about its multi-layered counterpart - deep neural collapse (DNC). In particular, existing work focuses either on linear layers or only on the last two layers at the price of an extra assumption. Our work fills this gap by generalizing the established analytical framework for NC - the unconstrained features model - to multiple non-linear layers. Our key technical contribution is to show that, in a deep unconstrained features model, the unique global optimum for binary classification exhibits all the properties typical of DNC. This explains the existing experimental evidence of DNC. We also empirically show that (i) by optimizing deep unconstrained features models via gradient descent, the resulting solution agrees well with our theory, and (ii) trained networks recover the unconstrained features suitable for the occurrence of DNC, thus supporting the validity of this modeling principle.

        ----

        ## [2306] IEBins: Iterative Elastic Bins for Monocular Depth Estimation

        **Authors**: *Shuwei Shao, Zhongcai Pei, Xingming Wu, Zhong Liu, Weihai Chen, Zhengguo Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html)

        **Abstract**:

        Monocular depth estimation (MDE) is a fundamental topic of geometric computer vision and a core technique for many downstream applications. Recently, several methods reframe the MDE as a classification-regression problem where a linear combination of probabilistic distribution and bin centers is used to predict depth. In this paper, we propose a novel concept of iterative elastic bins (IEBins) for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. Furthermore, we develop a dedicated framework composed of a feature extractor and an iterative optimizer that has powerful temporal context modeling capabilities benefiting from the GRU-based architecture. Extensive experiments on the KITTI, NYU-Depth-v2 and SUN RGB-D datasets demonstrate that the proposed method surpasses prior state-of-the-art competitors. The source code is publicly available at https://github.com/ShuweiShao/IEBins.

        ----

        ## [2307] Fine-Tuning Language Models with Just Forward Passes

        **Authors**: *Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html)

        **Abstract**:

        Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zerothorder optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same budget. We conduct comprehensive experiments across model types (masked and autoregressive LMs), model scales (up to 66B), and downstream tasks (classification, multiple-choice, and generation). Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear probing; (2) MeZO achieves comparable performance to fine-tuning with backpropagation across multiple tasks, with up to 12× memory reduction and up to 2× GPU-hour reduction in our implementation; (3) MeZO is compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix tuning; (4) MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1). We support our empirical findings with theoretical insights, highlighting how adequate pre-training and task prompts enable MeZO to fine-tune huge models, despite classical ZO analyses suggesting otherwise.

        ----

        ## [2308] HEDNet: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds

        **Authors**: *Gang Zhang, Junnan Chen, Guohuan Gao, Jianmin Li, Xiaolin Hu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a64e641fa00a7eb9500cb7e1835d0495-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a64e641fa00a7eb9500cb7e1835d0495-Abstract-Conference.html)

        **Abstract**:

        3D object detection in point clouds is important for autonomous driving systems. A primary challenge in 3D object detection stems from the sparse distribution of points within the 3D scene. Existing high-performance methods typically employ 3D sparse convolutional neural networks with small kernels to extract features. To reduce computational costs, these methods resort to submanifold sparse convolutions, which prevent the information exchange among spatially disconnected features. Some recent approaches have attempted to address this problem by introducing large-kernel convolutions or self-attention mechanisms, but they either achieve limited accuracy improvements or incur excessive computational costs. We propose HEDNet, a hierarchical encoder-decoder network for 3D object detection, which leverages encoder-decoder blocks to capture long-range dependencies among features in the spatial space, particularly for large and distant objects. We conducted extensive experiments on the Waymo Open and nuScenes datasets. HEDNet achieved superior detection accuracy on both datasets than previous state-of-the-art methods with competitive efficiency. The code is available at https://github.com/zhanggang001/HEDNet.

        ----

        ## [2309] FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning

        **Authors**: *Jinyuan Jia, Zhuowen Yuan, Dinuka Sahabandu, Luyao Niu, Arezoo Rajabi, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract-Conference.html)

        **Abstract**:

        Federated learning (FL) provides a distributed training paradigm where multiple clients can jointly train a global model without sharing their local data. However, recent studies have shown that FL offers an additional surface for backdoor attacks. For instance, an attacker can compromise a subset of clients and thus corrupt the global model to misclassify an input with a backdoor trigger as the adversarial target. Existing defenses for FL against backdoor attacks usually detect and exclude the corrupted information from the compromised clients based on a static attacker model. However, such defenses are inadequate against dynamic attackers who strategically adapt their attack strategies. To bridge this gap, we model the strategic interactions between the defender and dynamic attackers as a minimax game. Based on the analysis of the game, we design an interactive defense mechanism FedGame. We prove that under mild assumptions, the global model trained with FedGame under backdoor attacks is close to that trained without attacks. Empirically, we compare FedGame with multiple state-of-the-art baselines on several benchmark datasets under various attacks. We show that FedGame can effectively defend against strategic attackers and achieves significantly higher robustness than baselines. Our code is available at: https://github.com/AI-secure/FedGame.

        ----

        ## [2310] Ensemble-based Deep Reinforcement Learning for Vehicle Routing Problems under Distribution Shift

        **Authors**: *Yuan Jiang, Zhiguang Cao, Yaoxin Wu, Wen Song, Jie Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a68120d2eb2f53f7d9e71547591aef11-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a68120d2eb2f53f7d9e71547591aef11-Abstract-Conference.html)

        **Abstract**:

        While performing favourably on the independent and identically distributed (i.i.d.) instances, most of the existing neural methods for vehicle routing problems (VRPs) struggle to generalize in the presence of a distribution shift. To tackle this issue, we propose an ensemble-based deep reinforcement learning method for VRPs, which learns a group of diverse sub-policies to cope with various instance distributions. In particular, to prevent convergence of the parameters to the same one, we enforce diversity across sub-policies by leveraging Bootstrap with random initialization. Moreover, we also explicitly pursue inequality between sub-policies by exploiting regularization terms during training to further enhance diversity. Experimental results show that our method is able to outperform the state-of-the-art neural baselines on randomly generated instances of various distributions, and also generalizes favourably on the benchmark instances from TSPLib and CVRPLib, which confirmed the effectiveness of the whole method and the respective designs.

        ----

        ## [2311] Module-wise Training of Neural Networks via the Minimizing Movement Scheme

        **Authors**: *Skander Karkar, Ibrahim Ayed, Emmanuel de Bézenac, Patrick Gallinari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6a1e4c756d700d9aedcc1896a7e6fb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6a1e4c756d700d9aedcc1896a7e6fb0-Abstract-Conference.html)

        **Abstract**:

        Greedy layer-wise or module-wise training of neural networks is compelling in constrained and on-device settings where memory is limited, as it circumvents a number of problems of end-to-end back-propagation. However, it suffers from a stagnation problem, whereby early layers overfit and deeper layers stop increasing the test accuracy after a certain depth. We propose to solve this issue by introducing a simple module-wise regularization inspired by the minimizing movement scheme for gradient flows in distribution space. We call the method TRGL for Transport Regularized Greedy Learning and study it theoretically, proving that it leads to greedy modules that are regular and that progressively solve the task. Experimentally, we show improved accuracy of module-wise training of various architectures such as ResNets, Transformers and VGG, when our regularization is added, superior to that of other module-wise training methods and often to end-to-end training, with as much as 60% less memory usage.

        ----

        ## [2312] POMDP Planning for Object Search in Partially Unknown Environment

        **Authors**: *Yongbo Chen, Hanna Kurniawati*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6d7226db2ff3643d8624624e3859c19-Abstract-Conference.html)

        **Abstract**:

        Efficiently searching for target objects in complex environments that contain various types of furniture, such as shelves, tables, and beds, is crucial for mobile robots, but it poses significant challenges due to various factors such as localization errors, limited field of view, and visual occlusion. To address this problem, we propose a Partially Observable Markov Decision Process (POMDP) formulation with a growing state space for object search in a 3D region. We solve this POMDP by carefully designing a perception module and developing a planning algorithm, called Growing Partially Observable Monte-Carlo Planning (GPOMCP), based on online Monte-Carlo tree search and belief tree reuse with a novel upper confidence bound. We have demonstrated that belief tree reuse is reasonable and achieves good performance when the belief differences are limited. Additionally, we introduce a guessed target object with an updating grid world to guide the search in the information-less and reward-less cases, like the absence of any detected objects. We tested our approach using Gazebo simulations on four scenarios of target finding in a realistic indoor living environment with the Fetch robot simulator. Compared to the baseline approaches, which are based on POMCP, our results indicate that our approach enables the robot to find the target object with a higher success rate faster while using the same computational requirements.

        ----

        ## [2313] On the Statistical Consistency of Risk-Sensitive Bayesian Decision-Making

        **Authors**: *Prateek Jaiswal, Harsha Honnappa, Vinayak A. Rao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html)

        **Abstract**:

        We study data-driven decision-making problems in the Bayesian framework, where the expectation in the Bayes risk is replaced by a risk-sensitive entropic risk measure with respect to the posterior distribution. We focus on problems where calculating the posterior distribution is intractable, a typical situation in modern applications with large datasets and complex data generating models. We leverage a dual representation of the entropic risk measure to introduce a novel risk-sensitive variational Bayesian (RSVB) framework for jointly computing a risk-sensitive posterior approximation and the corresponding decision rule. Our general framework includes \textit{loss-calibrated} VB (Lacoste-Julien et al. [2011] ) as a special case. We also study the impact of these computational approximations on the predictive performance of the inferred decision rules. We compute the convergence rates of the RSVB approximate posterior and the corresponding optimal value. We illustrate our theoretical findings in parametric and nonparametric settings with the help of three examples.

        ----

        ## [2314] Does Graph Distillation See Like Vision Dataset Counterpart?

        **Authors**: *Beining Yang, Kai Wang, Qingyun Sun, Cheng Ji, Xingcheng Fu, Hao Tang, Yang You, Jianxin Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6efa49c54bedf4411f1bcd32f15937a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6efa49c54bedf4411f1bcd32f15937a-Abstract-Conference.html)

        **Abstract**:

        Training on large-scale graphs has achieved remarkable results in graph representation learning, but its cost and storage have attracted increasing concerns. Existing graph condensation methods primarily focus on optimizing the feature matrices of condensed graphs while overlooking the impact of the structure information from the original graphs. To investigate the impact of the structure information, we conduct analysis from the spectral domain and empirically identify substantial Laplacian Energy Distribution (LED) shifts in previous works. Such shifts lead to poor performance in cross-architecture generalization and specific tasks, including anomaly detection and link prediction. In this paper, we propose a novel Structure-broadcasting Graph Dataset Distillation (\textbf{SGDD}) scheme for broadcasting the original structure information to the generation of the synthetic one, which explicitly prevents overlooking the original structure information. Theoretically, the synthetic graphs by SGDD are expected to have smaller LED shifts than previous works, leading to superior performance in both cross-architecture settings and specific tasks.We validate the proposed SGDD~across 9 datasets and achieve state-of-the-art results on all of them: for example, on YelpChi dataset, our approach maintains 98.6\% test accuracy of training on the original graph dataset with 1,000 times saving on the scale of the graph. Moreover, we empirically evaluate there exist 17.6\% $\sim$ 31.4\% reductions in LED shift crossing 9 datasets. Extensive experiments and analysis verify the effectiveness and necessity of the proposed designs. The code will be made public.

        ----

        ## [2315] Online Learning under Adversarial Nonlinear Constraints

        **Authors**: *Pavel Kolev, Georg Martius, Michael Muehlebach*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6f2763089c0bd8f56006c42f09ee24c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6f2763089c0bd8f56006c42f09ee24c-Abstract-Conference.html)

        **Abstract**:

        In many applications, learning systems are required to process continuous non-stationary data streams.We study this problem in an online learning framework and propose an algorithm that can deal with adversarial time-varying and nonlinear constraints.As we show in our work, the algorithm called Constraint Violation Velocity Projection (CVV-Pro) achieves $\sqrt{T}$ regret and converges to the feasible set at a rate of $1/\sqrt{T}$, despite the fact that the feasible set is slowly time-varying and a priori unknown to the learner. CVV-Pro only relies on local sparse linear approximations of the feasible set and therefore avoids optimizing over the entire set at each iteration, which is in sharp contrast to projected gradients or Frank-Wolfe methods. We also empirically evaluate our algorithm on two-player games, where the players are subjected to a shared constraint.

        ----

        ## [2316] Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents

        **Authors**: *Woojun Kim, Yongjae Shin, Jongeui Park, Youngchul Sung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a6f6a5c517b2b92f3d309786af64086c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a6f6a5c517b2b92f3d309786af64086c-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning (RL) has achieved remarkable success in solving complex tasks through its integration with deep neural networks (DNNs) as function approximators. However, the reliance on DNNs has introduced a new challenge called primacy bias, whereby these function approximators tend to prioritize early experiences, leading to overfitting. To alleviate this bias, a reset method has been proposed, which involves periodic resets of a portion or the entirety of a deep RL agent while preserving the replay buffer. However, the use of this method can result in performance collapses after executing the reset, raising concerns from the perspective of safe RL and regret minimization. In this paper, we propose a novel reset-based method that leverages deep ensemble learning to address the limitations of the vanilla reset method and enhance sample efficiency. The effectiveness of the proposed method is validated through various experiments including those in the domain of safe RL. Numerical results demonstrate its potential for real-world applications requiring high sample efficiency and safety considerations.

        ----

        ## [2317] Im-Promptu: In-Context Composition from Image Prompts

        **Authors**: *Bhishma Dedhia, Michael Chang, Jake Snell, Tom Griffiths, Niraj K. Jha*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a71c1931d3fb8ba564f7458d0657d0b1-Abstract-Conference.html)

        **Abstract**:

        Large language models are few-shot learners that can solve diverse tasks from a handful of demonstrations. This implicit understanding of tasks suggests that the attention mechanisms over word tokens may play a role in analogical reasoning. In this work, we investigate whether analogical reasoning can enable in-context composition over composable elements of visual stimuli. First, we introduce a suite of three benchmarks to test the generalization properties of a visual in-context learner. We formalize the notion of an analogy-based in-context learner and use it to design a meta-learning framework called Im-Promptu. Whereas the requisite token granularity for language is well established, the appropriate compositional granularity for enabling in-context generalization in visual stimuli is usually unspecified. To this end, we use Im-Promptu to train multiple agents with different levels of compositionality, including vector representations, patch representations, and object slots. Our experiments reveal tradeoffs between extrapolation abilities and the degree of compositionality, with non-compositional representations extending learned composition rules to unseen domains but performing poorly on combinatorial tasks. Patch-based representations require patches to contain entire objects for robust extrapolation. At the same time, object-centric tokenizers coupled with a cross-attention module generate consistent and high-fidelity solutions, with these inductive biases being particularly crucial for compositional generalization. Lastly, we demonstrate a use case of Im-Promptu as an intuitive programming interface for image generation.

        ----

        ## [2318] Sequential Predictive Two-Sample and Independence Testing

        **Authors**: *Aleksandr Podkopaev, Aaditya Ramdas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a72b207734d6112f6b47447e46be40e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a72b207734d6112f6b47447e46be40e9-Abstract-Conference.html)

        **Abstract**:

        We study the problems of sequential nonparametric two-sample and independence testing. Sequential tests process data online and allow using observed data to decide whether to stop and reject the null hypothesis or to collect more data, while maintaining type I error control. We build upon the principle of (nonparametric) testing by betting, where a gambler places bets on future observations and their wealth measures evidence against the null hypothesis. While recently developed kernel-based betting strategies often work well on simple distributions, selecting a suitable kernel for high-dimensional or structured data, such as images, is often nontrivial. To address this drawback, we design prediction-based betting strategies that rely on the following fact: if a sequentially updated predictor starts to consistently determine (a) which distribution an instance is drawn from, or (b) whether an instance is drawn from the joint distribution or the product of the marginal distributions (the latter produced by external randomization), it provides evidence against the two-sample or independence nulls respectively. We empirically demonstrate the superiority of our tests over kernel-based approaches under structured settings. Our tests can be applied beyond the case of independent and identically distributed data, remaining valid and powerful even when the data distribution drifts over time.

        ----

        ## [2319] Towards Symmetry-Aware Generation of Periodic Materials

        **Authors**: *Youzhi Luo, Chengkai Liu, Shuiwang Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of generating periodic materials with deep models. While symmetry-aware molecule generation has been studied extensively, periodic materials possess different symmetries, which have not been completely captured by existing methods.In this work, we propose SyMat, a novel material generation approach that can capture physical symmetries of periodic material structures. SyMat generates atom types and lattices of materials through generating atom type sets, lattice lengths and lattice angles with a variational auto-encoder model. In addition, SyMat employs a score-based diffusion model to generate atom coordinates of materials, in which a novel symmetry-aware probabilistic model is used in the coordinate diffusion process. We show that SyMat is theoretically invariant to all symmetry transformations on materials and demonstrate that SyMat achieves promising performance on random generation and property optimization tasks. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

        ----

        ## [2320] DICES Dataset: Diversity in Conversational AI Evaluation for Safety

        **Authors**: *Lora Aroyo, Alex S. Taylor, Mark Díaz, Christopher Homan, Alicia Parrish, Gregory Serapio-García, Vinodkumar Prabhakaran, Ding Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a74b697bce4cac6c91896372abaa8863-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Machine learning approaches often require training and evaluation datasets with a clear separation between positive and negative examples. This requirement overly simplifies the natural subjectivity present in many tasks, and obscures the inherent diversity in human perceptions and opinions about many content items. Preserving the variance in content and diversity in human perceptions in datasets is often quite expensive and laborious. This is especially troubling when building safety datasets for conversational AI systems, as safety is socio-culturally situated in this context. To demonstrate this crucial aspect of conversational AI safety, and to facilitate in-depth model performance analyses, we introduce the DICES (Diversity In Conversational AI Evaluation for Safety) dataset that contains fine-grained demographics information about raters, high replication of ratings per item to ensure statistical power for analyses, and encodes rater votes as distributions across different demographics to allow for in-depth explorations of different aggregation strategies. The DICES dataset enables the observation and measurement of variance, ambiguity, and diversity in the context of safety for conversational AI. We further describe a set of metrics that show how rater diversity influences safety perception across different geographic regions, ethnicity groups, age groups, and genders. The goal of the DICES dataset is to be used as a shared resource and benchmark that respects diverse perspectives during safety evaluation of conversational AI systems.

        ----

        ## [2321] SALSA VERDE: a machine learning attack on LWE with sparse small secrets

        **Authors**: *Cathy Yuanchen Li, Emily Wenger, Zeyuan Allen-Zhu, François Charton, Kristin E. Lauter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a75db7d2ee1e4bee8fb819979b0a6cad-Abstract-Conference.html)

        **Abstract**:

        Learning with Errors (LWE) is a hard math problem used in post-quantum cryptography. Homomorphic Encryption (HE) schemes rely on the hardness of the LWE problem for their security, and two LWE-based cryptosystems were recently standardized by NIST for digital signatures and key exchange (KEM).  Thus, it is critical to continue assessing the security of LWE and specific parameter choices. For example, HE uses secrets with small entries, and the HE community has considered standardizing small sparse secrets to improve efficiency and functionality.  However, prior work, SALSA and PICANTE, showed that ML attacks can recover sparse binary secrets. Building on these, we propose VERDE, an improved ML attack that can recover sparse binary, ternary, and narrow Gaussian secrets. Using improved preprocessing and secret recovery techniques, VERDE can attack LWE with larger dimensions ($n=512$) and smaller moduli ($\log_2 q=12$ for $n=256$), using less time and power. We propose novel architectures for scaling. Finally, we develop a theory that explains the success of ML LWE attacks.

        ----

        ## [2322] Designing Robust Transformers using Robust Kernel Density Estimation

        **Authors**: *Xing Han, Tongzheng Ren, Tan Nguyen, Khai Nguyen, Joydeep Ghosh, Nhat Ho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html)

        **Abstract**:

        Transformer-based architectures have recently exhibited remarkable successes across different domains beyond just powering large language models. However, existing approaches typically focus on predictive accuracy and computational cost, largely ignoring certain other practical issues such as robustness to contaminated samples. In this paper, by re-interpreting the self-attention mechanism as a non-parametric kernel density estimator, we adapt classical robust kernel density estimation methods to develop novel classes of transformers that are resistant to adversarial attacks and data contamination. We first propose methods that down-weight outliers in RKHS when computing the self-attention operations. We empirically show that these methods produce improved performance over existing state-of-the-art methods, particularly on image data under adversarial attacks. Then we leverage the median-of-means principle to obtain another efficient approach that results in noticeably enhanced performance and robustness on language modeling and time series classification tasks. Our methods can be combined with existing transformers to augment their robust properties, thus promising to impact a wide variety of applications.

        ----

        ## [2323] Benchmarking Distribution Shift in Tabular Data with TableShift

        **Authors**: *Josh Gardner, Zoran Popovic, Ludwig Schmidt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Robustness to distribution shift has become a growing concern for text and image models as they transition from research subjects to deployment in the real world. However, high-quality benchmarks for distribution shift in tabular machine learning tasks are still lacking despite the widespread real-world use of tabular data and differences in the models used for tabular data in comparison to text and images. As a consequence, the robustness of tabular models to distribution shift is poorly understood. To address this issue, we introduce TableShift, a distribution shift benchmark for tabular data. TableShift contains 15 binary classification tasks in total, each with an associated shift, and includes a diverse set of data sources, prediction targets, and distribution shifts. The benchmark covers domains including finance, education, public policy, healthcare, and civic participation, and is accessible using only a few lines of Python code via the TableShift API. We conduct a large-scale study comparing several state-of-the-art tabular data models alongside robust learning and domain generalization methods on the benchmark tasks. Our study demonstrates (1) a linear trend between in-distribution (ID) and out-of-distribution (OOD) accuracy; (2) domain robustness methods can reduce shift gaps but at the cost of reduced ID accuracy; (3) a strong relationship between shift gap (difference between ID and OOD performance) and shifts in the label distribution. The benchmark data, Python package, model implementations, and more information about TableShift are available at https://github.com/mlfoundations/tableshift and https://tableshift.org .

        ----

        ## [2324] Weakly Supervised 3D Open-vocabulary Segmentation

        **Authors**: *Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu, Yingchen Yu, Abdulmotaleb El-Saddik, Christian Theobalt, Eric P. Xing, Shijian Lu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a76b693f36916a5ed84d6e5b39a0dc03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a76b693f36916a5ed84d6e5b39a0dc03-Abstract-Conference.html)

        **Abstract**:

        Open-vocabulary segmentation of 3D scenes is a fundamental function of human perception and thus a crucial objective in computer vision research. However, this task is heavily impeded by the lack of large-scale and diverse 3D open-vocabulary segmentation datasets for training robust and generalizable models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation models helps but it compromises the open-vocabulary feature as the 2D models are mostly finetuned with close-vocabulary datasets. We tackle the challenges in 3D open-vocabulary segmentation by exploiting pre-trained foundation models CLIP and DINO in a weakly supervised manner. Specifically, given only the open-vocabulary text descriptions of the objects in a scene, we distill the open-vocabulary multimodal knowledge and object reasoning capability of CLIP and DINO into a neural radiance field (NeRF), which effectively lifts 2D features into view-consistent 3D segmentation. A notable aspect of our approach is that it does not require any manual segmentation annotations for either the foundation models or the distillation process. Extensive experiments show that our method even outperforms fully supervised models trained with segmentation annotations in certain scenes, suggesting that 3D open-vocabulary segmentation can be effectively learned from 2D images and text-image pairs. Code is available at https://github.com/Kunhao-Liu/3D-OVS.

        ----

        ## [2325] Better Private Linear Regression Through Better Private Feature Selection

        **Authors**: *Travis Dick, Jennifer Gillenwater, Matthew Joseph*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a79699db176ed0efc04a9da171e52112-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a79699db176ed0efc04a9da171e52112-Abstract-Conference.html)

        **Abstract**:

        Existing work on differentially private linear regression typically assumes that end users can precisely set data bounds or algorithmic hyperparameters. End users often struggle to meet these requirements without directly examining the data (and violating privacy). Recent work has attempted to develop solutions that shift these burdens from users to algorithms, but they struggle to provide utility as the feature dimension grows. This work extends these algorithms to higher-dimensional problems by introducing a differentially private feature selection method based on Kendall rank correlation. We prove a utility guarantee for the setting where features are normally distributed and conduct experiments across 25 datasets. We find that adding this private feature selection step before regression significantly broadens the applicability of ``plug-and-play'' private linear regression algorithms at little additional cost to privacy, computation, or decision-making by the end user.

        ----

        ## [2326] Geometric Neural Diffusion Processes

        **Authors**: *Emile Mathieu, Vincent Dutordoir, Michael Hutchinson, Valentin De Bortoli, Yee Whye Teh, Richard E. Turner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)

        **Abstract**:

        Denoising diffusion models have proven to be a flexible and effective paradigm for generative modelling.Their recent extension to infinite dimensional Euclidean spaces has allowed for the modelling of stochastic processes.However, many problems in the natural sciences incorporate symmetries and involve data living in non-Euclidean spaces.In this work, we extend the framework of diffusion models to incorporate a series of geometric priors in infinite-dimension modelling.We do so by a) constructing a noising process which admits, as limiting distribution, a geometric Gaussian process that transforms under the symmetry group of interest, and b) approximating the score with a neural network that is equivariant w.r.t. this group.We show that with these conditions, the generative functional model admits the same symmetry.We demonstrate scalability and capacity of the model, using a novel Langevin-based conditional sampler, to fit complex scalar and vector fields, with Euclidean and spherical codomain, on synthetic and real-world weather data.

        ----

        ## [2327] Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations

        **Authors**: *Yiheng Lin, James A. Preiss, Emile Anand, Yingying Li, Yisong Yue, Adam Wierman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7a7180fe7f82ff98eee0827c5e9c141-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7a7180fe7f82ff98eee0827c5e9c141-Abstract-Conference.html)

        **Abstract**:

        We study online adaptive policy selection in systems with time-varying costs and dynamics. We develop the Gradient-based Adaptive Policy Selection (GAPS) algorithm together with a general analytical framework for online policy selection via online optimization. Under our proposed notion of contractive policy classes, we show that GAPS approximates the behavior of an ideal online gradient descent algorithm on the policy parameters while requiring less information and computation. When convexity holds, our algorithm is the first to achieve optimal policy regret. When convexity does not hold, we provide the first local regret bound for online policy selection. Our numerical experiments show that GAPS can adapt to changing environments more quickly than existing benchmarks.

        ----

        ## [2328] IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

        **Authors**: *Pascal Leroy, Pablo G. Morato, Jonathan Pisane, Athanasios Kolios, Damien Ernst*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7a7c0c92f195cce85f99768621ac6c0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7a7c0c92f195cce85f99768621ac6c0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce IMP-MARL, an open-source suite of multi-agent reinforcement learning (MARL) environments for large-scale Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.In IMP, a multi-component engineering system is subject to a risk of failure due to its components' damage condition.Specifically, each agent plans inspections and repairs for a specific system component, aiming to minimise maintenance costs while cooperating to minimise system failure risk.With IMP-MARL, we release several environments including one related to offshore wind structural systems, in an effort to meet today's needs to improve management strategies to support sustainable and reliable energy systems.Supported by IMP practical engineering environments featuring up to 100 agents, we conduct a benchmark campaign, where the scalability and performance of state-of-the-art cooperative MARL methods are compared against expert-based heuristic policies. The results reveal that centralised training with decentralised execution methods scale better with the number of agents than fully centralised or decentralised RL approaches, while also outperforming expert-based heuristic policies in most IMP environments.Based on our findings, we additionally outline remaining cooperation and scalability challenges that future MARL methods should still address.Through IMP-MARL, we encourage the implementation of new environments and the further development of MARL methods.

        ----

        ## [2329] Learning Multi-agent Behaviors from Distributed and Streaming Demonstrations

        **Authors**: *Shicheng Liu, Minghui Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7affe50ab177b9a7f0a05f07a9ca205-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7affe50ab177b9a7f0a05f07a9ca205-Abstract-Conference.html)

        **Abstract**:

        This paper considers the problem of inferring the behaviors of multiple interacting experts by estimating their reward functions and constraints where the distributed demonstrated trajectories are sequentially revealed to a group of learners. We formulate the problem as a distributed online bi-level optimization problem where the outer-level problem is to estimate the reward functions and the inner-level problem is to learn the constraints and corresponding policies. We propose a novel ``multi-agent behavior inference from distributed and streaming demonstrations" (MA-BIRDS) algorithm that allows the learners to solve the outer-level and inner-level problems in a single loop through intermittent communications. We formally guarantee that the distributed learners achieve consensus on reward functions, constraints, and policies, the average local regret (over $N$ online iterations) decreases at the rate of $O(1/N^{1-\eta_1}+1/N^{1-\eta_2}+1/N)$, and the cumulative constraint violation increases sub-linearly at the rate of $O(N^{\eta_2}+1)$ where $\eta_1,\eta_2\in (1/2,1)$.

        ----

        ## [2330] CAPro: Webly Supervised Learning with Cross-modality Aligned Prototypes

        **Authors**: *Yulei Qin, Xingyu Chen, Yunhang Shen, Chaoyou Fu, Yun Gu, Ke Li, Xing Sun, Rongrong Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a7e0d77325db843fd5baf1298163e89a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a7e0d77325db843fd5baf1298163e89a-Abstract-Conference.html)

        **Abstract**:

        Webly supervised learning has attracted increasing attention for its effectiveness in exploring publicly accessible data at scale without manual annotation. However, most existing methods of learning with web datasets are faced with challenges from label noise, and they have limited assumptions on clean samples under various noise. For instance, web images retrieved with queries of ”tiger cat“ (a cat species) and ”drumstick“ (a musical instrument) are almost dominated by images of tigers and chickens, which exacerbates the challenge of fine-grained visual concept learning. In this case, exploiting both web images and their associated texts is a requisite solution to combat real-world noise. In this paper, we propose Cross-modality Aligned Prototypes (CAPro), a unified prototypical contrastive learning framework to learn visual representations with correct semantics. For one thing, we leverage textual prototypes, which stem from the distinct concept definition of classes, to select clean images by text matching and thus disambiguate the formation of visual prototypes. For another, to handle missing and mismatched noisy texts, we resort to the visual feature space to complete and enhance individual texts and thereafter improve text matching. Such semantically aligned visual prototypes are further polished up with high-quality samples, and engaged in both cluster regularization and noise removal. Besides, we propose collective bootstrapping to encourage smoother and wiser label reference from appearance-similar instances in a manner of dictionary look-up. Extensive experiments on WebVision1k and NUS-WIDE (Web) demonstrate that CAPro well handles realistic noise under both single-label and multi-label scenarios. CAPro achieves new state-of-the-art performance and exhibits robustness to open-set recognition. Codes are available at https://github.com/yuleiqin/capro.

        ----

        ## [2331] Diversify & Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement

        **Authors**: *Daesol Cho, Seungjae Lee, H. Jin Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a815fe7cad6af20a6c118f2072a881d2-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) often faces the challenges of uninformed search problems where the agent should explore without access to the domain knowledge such as characteristics of the environment or external rewards. To tackle these challenges, this work proposes a new approach for curriculum RL called $\textbf{D}$iversify for $\textbf{D}$isagreement \& $\textbf{C}$onquer ($\textbf{D2C}$). Unlike previous curriculum learning methods, D2C requires only a few examples of desired outcomes and works in any environment, regardless of its geometry or the distribution of the desired outcome examples. The proposed method performs diversification of the goal-conditional classifiers to identify similarities between visited and desired outcome states and ensures that the classifiers disagree on states from out-of-distribution, which enables quantifying the unexplored region and designing an arbitrary goal-conditioned intrinsic reward signal in a simple and intuitive way. The proposed method then employs bipartite matching to define a curriculum learning objective that produces a sequence of well-adjusted intermediate goals, which enable the agent to automatically explore and conquer the unexplored region. We present experimental results demonstrating that D2C outperforms prior curriculum RL methods in both quantitative and qualitative aspects, even with the arbitrarily distributed desired outcome examples.

        ----

        ## [2332] Distribution-Free Model-Agnostic Regression Calibration via Nonparametric Methods

        **Authors**: *Shang Liu, Zhongze Cai, Xiaocheng Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a81dc87f7b3b7ab8489d5bb48c4a8d92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a81dc87f7b3b7ab8489d5bb48c4a8d92-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider the uncertainty quantification problem for regression models. Specifically, we consider an individual calibration objective for characterizing the quantiles of the prediction model. While such an objective is well-motivated from downstream tasks such as newsvendor cost, the existing methods have been largely heuristic and lack of statistical guarantee in terms of individual calibration. We show via simple examples that the existing methods focusing on population-level calibration guarantees such as average calibration or sharpness can lead to harmful and unexpected results. We propose simple nonparametric calibration methods that are agnostic of the underlying prediction model and enjoy both computational efficiency and statistical consistency. Our approach enables a better understanding of the possibility of individual calibration, and we establish matching upper and lower bounds for the calibration error of our proposed methods. Technically, our analysis combines the nonparametric analysis with a covering number argument for parametric analysis, which advances the existing theoretical analyses in the literature of nonparametric density estimation and quantile bandit problems. Importantly, the nonparametric perspective sheds new theoretical insights into regression calibration in terms of the curse of dimensionality and reconciles the existing results on the impossibility of individual calibration. To our knowledge, we make the first effort to reach both individual calibration and finite-sample guarantee with minimal assumptions in terms of conformal prediction. Numerical experiments show the advantage of such a simple approach under various metrics, and also under covariates shift. We hope our work provides a simple benchmark and a starting point of theoretical ground for future research on regression calibration.

        ----

        ## [2333] Synthetic-to-Real Pose Estimation with Geometric Reconstruction

        **Authors**: *Qiuxia Lin, Kerui Gu, Linlin Yang, Angela Yao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8223b0ad64007423ffb308b0dd92298-Abstract-Conference.html)

        **Abstract**:

        Pose estimation is remarkably successful under supervised learning, but obtaining annotations, especially for new deployments, is costly and time-consuming. This work tackles adapting models trained on synthetic data to real-world target domains with only unlabelled data. A common approach is model fine-tuning with pseudo-labels from the target domain; yet many pseudo-labelling strategies cannot provide sufficient high-quality pose labels. This work proposes a reconstruction-based strategy as a complement to pseudo-labelling for synthetic-to-real domain adaptation. We generate the driving image by geometrically transforming a base image according to the predicted keypoints and enforce a reconstruction loss to refine the predictions. It provides a novel solution to effectively correct confident yet inaccurate keypoint locations through image reconstruction in domain adaptation. Our approach outperforms the previous state-of-the-arts by 8% for PCK on four large-scale hand and human real-world datasets. In particular, we excel on endpoints such as fingertips and head, with 7.2% and 29.9% improvements in PCK.

        ----

        ## [2334] Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies

        **Authors**: *Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, Yonghong Tian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html)

        **Abstract**:

        Vanilla spiking neurons in Spiking Neural Networks (SNNs) use charge-fire-reset neuronal dynamics, which can only be simulated serially and can hardly learn long-time dependencies. We find that when removing reset, the neuronal dynamics can be reformulated in a non-iterative form and parallelized. By rewriting neuronal dynamics without reset to a general formulation, we propose the Parallel Spiking Neuron (PSN), which generates hidden states that are independent of their predecessors, resulting in parallelizable neuronal dynamics and extremely high simulation speed. The weights of inputs in the PSN are fully connected, which maximizes the utilization of temporal information. To avoid the use of future inputs for step-by-step inference, the weights of the PSN can be masked, resulting in the masked PSN. By sharing weights across time-steps based on the masked PSN, the sliding PSN is proposed to handle sequences of varying lengths. We evaluate the PSN family on simulation speed and temporal/static data classification, and the results show the overwhelming advantage of the PSN family in efficiency and accuracy. To the best of our knowledge, this is the first study about parallelizing spiking neurons and can be a cornerstone for the spiking deep learning research. Our codes are available at https://github.com/fangwei123456/Parallel-Spiking-Neuron.

        ----

        ## [2335] Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment

        **Authors**: *Zihui Xue, Kristen Grauman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a845fdc3f87751710218718adb634fe7-Abstract-Conference.html)

        **Abstract**:

        The egocentric and exocentric viewpoints of a human activity look dramatically different, yet invariant representations to link them are essential for many potential applications in robotics and augmented reality.  Prior work is limited to learning view-invariant features from paired synchronized viewpoints.  We relax that strong data assumption and propose to learn fine-grained action features that are invariant to the viewpoints by aligning egocentric and exocentric videos in time, even when not captured simultaneously or in the same environment. To this end, we propose AE2, a self-supervised embedding approach with two key designs: (1) an object-centric encoder that explicitly focuses on regions corresponding to hands and active objects; (2) a contrastive-based alignment objective that leverages temporally reversed frames as negative samples. For evaluation, we establish a benchmark for fine-grained video understanding in the ego-exo context, comprising four datasets---including an ego tennis forehand dataset we collected, along with dense per-frame labels we annotated for each dataset. On the four datasets, our AE2 method strongly outperforms prior work in a variety of fine-grained downstream tasks, both in regular and cross-view settings.

        ----

        ## [2336] Training Private Models That Know What They Don't Know

        **Authors**: *Stephan Rabanser, Anvith Thudi, Abhradeep Guha Thakurta, Krishnamurthy Dvijotham, Nicolas Papernot*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8526465a91166fbb90aaa8452b21eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8526465a91166fbb90aaa8452b21eda-Abstract-Conference.html)

        **Abstract**:

        Training reliable deep learning models which avoid making overconfident but incorrect predictions is a longstanding challenge. This challenge is further exacerbated when learning has to be differentially private: protection provided to sensitive data comes at the price of injecting additional randomness into the learning process. In this work, we conduct a thorough empirical investigation of selective classifiers---that can abstain under uncertainty---under a differential privacy constraint. We find that some popular selective prediction approaches are ineffective in a differentially private setting because they increase the risk of privacy leakage. At the same time, we identify that a recent approach that only uses checkpoints produced by an off-the-shelf private learning algorithm stands out as particularly suitable under DP. Further, we show that differential privacy does not just harm utility but also degrades selective classification performance. To analyze this effect across privacy levels, we propose a novel evaluation mechanism which isolates selective prediction performance across model utility levels at full coverage. Our experimental results show that recovering the performance level attainable by non-private models is possible but comes at a considerable coverage cost as the privacy budget decreases.

        ----

        ## [2337] Direct Preference Optimization: Your Language Model is Secretly a Reward Model

        **Authors**: *Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)

        **Abstract**:

        While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper, we leverage a mapping between reward functions and optimal policies to show that this constrained reward maximization problem can be optimized exactly with a single stage of policy training, essentially solving a classification problem on the human preference data. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for fitting a reward model, sampling from the LM during fine-tuning, or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds RLHF's ability to control sentiment of generations and improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

        ----

        ## [2338] Diffusion Representation for Asymmetric Kernels via Magnetic Transform

        **Authors**: *Mingzhen He, Fan He, Ruikai Yang, Xiaolin Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a86b7a9bf7647d6f9f9168d8167d9283-Abstract-Conference.html)

        **Abstract**:

        As a nonlinear dimension reduction technique, the diffusion map (DM) has been widely used. In DM, kernels play an important role for capturing the nonlinear relationship of data. However, only symmetric kernels can be used now, which prevents the use of DM in directed graphs, trophic networks, and other real-world scenarios where the intrinsic and extrinsic geometries in data are asymmetric. A promising technique is the magnetic transform which  converts an asymmetric matrix to a Hermitian one. However, we are facing essential problems, including how diffusion distance could be preserved and how divergence could be avoided during diffusion process. Via theoretical proof, we successfully establish a diffusion representation framework with the magnetic transform, named MagDM. The effectiveness and robustness for dealing data endowed with asymmetric proximity are demonstrated on three synthetic datasets and two trophic networks.

        ----

        ## [2339] Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

        **Authors**: *Pritam Sarkar, Ahmad Beirami, Ali Etemad*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a86d17b6cd70366d56ab48d2a05a4df1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a86d17b6cd70366d56ab48d2a05a4df1-Abstract-Conference.html)

        **Abstract**:

        Video self-supervised learning (VSSL) has made significant progress in recent years. However, the exact behavior and dynamics of these models under different forms of distribution shift are not yet known. In this paper, we comprehensively study the behavior of six popular self-supervised methods (v-SimCLR, v-MoCo, v-BYOL, v-SimSiam, v-DINO, v-MAE) in response to various forms of natural distribution shift, i.e., (i) context shift, (ii) viewpoint shift, (iii) actor shift, (iv) source shift, (v) generalizability to unknown classes (zero-shot), and (vi) open-set recognition. To perform this extensive study, we carefully craft a test bed consisting of 17 in-distribution and out-of-distribution benchmark pairs using available public datasets and a series of evaluation protocols to stress-test the different methods under the intended shifts. Our study uncovers a series of intriguing findings and interesting behaviors of VSSL methods. For instance, we observe that while video models generally struggle with context shifts, v-MAE and supervised learning exhibit more robustness. Moreover, our study shows that v-MAE is a strong temporal learner, whereas contrastive methods, v-SimCLR and v-MoCo, exhibit strong performances against viewpoint shifts. When studying the notion of open-set recognition, we notice a trade-off between closed-set and open-set recognition performance if the pretrained VSSL encoders are used without finetuning. We hope that our work will contribute to the development of robust video representation learning frameworks for various real-world scenarios. The project page and code are available at: https://pritamqu.github.io/OOD-VSSL.

        ----

        ## [2340] M2SODAI: Multi-Modal Maritime Object Detection Dataset With RGB and Hyperspectral Image Sensors

        **Authors**: *Jonggyu Jang, Sangwoo Oh, Youjin Kim, Dongmin Seo, Youngchol Choi, Hyun Jong Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8757b889350a3782b384a3ec0dfbae9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8757b889350a3782b384a3ec0dfbae9-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Object detection in aerial images is a growing area of research, with maritime object detection being a particularly important task for reliable surveillance, monitoring, and active rescuing. Notwithstanding astonishing advances of computer visiontechnologies, detecting ships and floating matters in these images are challenging due to factors such as object distance. What makes it worse is pervasive sea surface effects such as sunlight reflection, wind, and waves. Hyperspectral image (HSI) sensors, providing more than 100 channels in wavelengths of visible and near-infrared, can extract intrinsic information of materials from a few pixels of HSIs.The advent of HSI sensors motivates us to leverage HSIs to circumvent false positives due to the sea surface effects.Unfortunately, there are few public HSI datasets due to the high cost and labor involved in collecting them, hindering object detection research based on HSIs. We have collected and annotated a new dataset called ``Multi-Modal Ship and flOating matter Detection in Aerial Images (M$^{2}$SODAI),'', which includes synchronized image pairs of RGB and HSI data, along with bounding box labels for nearly 6,000 instances per category. We also propose a new multi-modal extension of the feature pyramid network called DoubleFPN.Extensive experiments on our benchmark demonstrate that fusion of RGB and HSI data can enhance mAP, especially in the presence of the sea surface effects.

        ----

        ## [2341] Projection-Free Methods for Solving Nonconvex-Concave Saddle Point Problems

        **Authors**: *Morteza Boroun, Erfan Yazdandoost Hamedani, Afrooz Jalilzadeh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a899a801fab59f14777fcc08842b6fc5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a899a801fab59f14777fcc08842b6fc5-Abstract-Conference.html)

        **Abstract**:

        In this paper, we investigate a class of constrained saddle point (SP) problems where the objective function is nonconvex-concave and smooth. This class of problems has wide applicability in machine learning, including robust multi-class classification and dictionary learning. Several projection-based primal-dual methods have been developed to tackle this problem; however, the availability of methods with projection-free oracles remains limited. To address this gap, we propose efficient single-loop projection-free methods reliant on first-order information. In particular, using regularization and nested approximation techniques, we propose a primal-dual conditional gradient method that solely employs linear minimization oracles to handle constraints. Assuming that the constraint set in the maximization is strongly convex, our method achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-6})$ iterations. When the projection onto the constraint set of maximization is easy to compute, we propose a one-sided projection-free method that achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-4})$ iterations. Moreover, we present improved iteration complexities of our methods under a strong concavity assumption. To the best of our knowledge, our proposed algorithms are among the first projection-free methods with convergence guarantees for solving nonconvex-concave SP problems.

        ----

        ## [2342] Better Correlation and Robustness: A Distribution-Balanced Self-Supervised Learning Framework for Automatic Dialogue Evaluation

        **Authors**: *Peiwen Yuan, Xinglin Wang, Jiayi Shi, Bin Sun, Yiwei Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8b148559549ce33261e79b4400e0d77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8b148559549ce33261e79b4400e0d77-Abstract-Conference.html)

        **Abstract**:

        Turn-level dialogue evaluation models (TDEMs), using self-supervised learning (SSL) framework, have achieved state-of-the-art performance in open-domain dialogue evaluation. However, these models inevitably face two potential problems. First, they have low correlations with humans on medium coherence samples as the SSL framework often brings training data with unbalanced coherence distribution. Second, the SSL framework leads TDEM to nonuniform score distribution. There is a danger that the nonuniform score distribution will weaken the robustness of TDEM through our theoretical analysis. To tackle these problems, we propose Better Correlation and Robustness (BCR), a distribution-balanced self-supervised learning framework for TDEM. Given a dialogue dataset, BCR offers an effective training set reconstructing method to provide coherence-balanced training signals and further facilitate balanced evaluating abilities of TDEM. To get a uniform score distribution, a novel loss function is proposed, which can adjust adaptively according to the uniformity of score distribution estimated by kernel density estimation. Comprehensive experiments on 17 benchmark datasets show that vanilla BERT-base using BCR outperforms SOTA methods significantly by 11.3% on average. BCR also demonstrates strong generalization ability as it can lead multiple SOTA methods to attain better correlation and robustness.

        ----

        ## [2343] Learning Topology-Agnostic EEG Representations with Geometry-Aware Modeling

        **Authors**: *Ke Yi, Yansen Wang, Kan Ren, Dongsheng Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8c893712cb7858e49631fb03c941f8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8c893712cb7858e49631fb03c941f8d-Abstract-Conference.html)

        **Abstract**:

        Large-scale pre-training has shown great potential to enhance models on downstream tasks in vision and language. Developing similar techniques for scalp electroencephalogram (EEG) is suitable since unlabelled data is plentiful. Meanwhile, various sampling channel selections and inherent structural and spatial information bring challenges and avenues to improve existing pre-training strategies further. In order to break boundaries between different EEG resources and facilitate cross-dataset EEG pre-training, we propose to map all kinds of channel selections to a unified topology. We further introduce MMM, a pre-training framework with Multi-dimensional position encoding, Multi-level channel hierarchy, and Multi-stage pre-training strategy built on the unified topology to obtain topology-agnostic representations. Experiments demonstrate that our approach yields impressive improvements over previous state-of-the-art techniques on emotional recognition benchmark datasets.

        ----

        ## [2344] Correlation Aware Sparsified Mean Estimation Using Random Projection

        **Authors**: *Shuli Jiang, Pranay Sharma, Gauri Joshi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8e21789027e92739f89df92cc172bcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8e21789027e92739f89df92cc172bcf-Abstract-Conference.html)

        **Abstract**:

        We study the problem of communication-efficient distributed vector mean estimation, which is a commonly used subroutine in distributed optimization and Federated Learning (FL). Rand-$k$ sparsification is a commonly used technique to reduce communication cost, where each client sends $k < d$ of its coordinates to the server. However, Rand-$k$ is agnostic to any correlations, that might exist between clients in practical scenarios. The recently proposed Rand-$k$-Spatial estimator leverages the cross-client correlation information at the server to improve Rand-$k$'s performance. Yet, the performance of Rand-$k$-Spatial is suboptimal, and improving mean estimation is key to a faster convergence in distributed optimization. We propose the Rand-Proj-Spatial estimator with a more flexible encoding-decoding procedure, which generalizes the encoding of Rand-$k$ by projecting the client vectors to a random $k$-dimensional subspace. We utilize Subsampled Randomized Hadamard Transform (SRHT) as the projection matrix, and show that Rand-Proj-Spatial with SRHT outperforms Rand-$k$-Spatial, using the correlation information more efficiently. Furthermore, we propose an approach to incorporate varying degrees of correlation, and suggest a practical variant of Rand-Proj-Spatial when the correlation information is not available to the server. Finally, experiments on real-world distributed optimization tasks showcase the superior performance of Rand-Proj-Spatial compared to Rand-$k$-Spatial and other more sophisticated sparsification techniques.

        ----

        ## [2345] Accelerating Value Iteration with Anchoring

        **Authors**: *Jongmin Lee, Ernest Ryu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f2713b5c6bdcd3d264f1aa9b9c6f03-Abstract-Conference.html)

        **Abstract**:

        Value Iteration (VI) is foundational to the theory and practice of modern reinforcement learning, and it is known to converge at a $\mathcal{O}(\gamma^k)$-rate. Surprisingly, however, the optimal rate for the VI setup was not known, and finding a general acceleration mechanism has been an open problem. In this paper, we present the first accelerated VI for both the Bellman consistency and optimality operators. Our method, called Anc-VI, is based on an \emph{anchoring} mechanism (distinct from Nesterov's acceleration), and it reduces the Bellman error faster than standard VI. In particular, Anc-VI exhibits a $\mathcal{O}(1/k)$-rate for $\gamma\approx 1$ or even $\gamma=1$, while standard VI has rate $\mathcal{O}(1)$ for $\gamma\ge 1-1/k$, where $k$ is the iteration count. We also provide a complexity lower bound matching the upper bound up to a constant factor of $4$, thereby establishing optimality of the accelerated rate of Anc-VI. Finally, we show that the anchoring mechanism provides the same benefit in the approximate VI and Gauss--Seidel VI setups as well.

        ----

        ## [2346] Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion

        **Authors**: *Yang Liu, Feng Wang, Naiyan Wang, Zhaoxiang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f7f12b29d9b8c227785f6b529f63b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f7f12b29d9b8c227785f6b529f63b7-Abstract-Conference.html)

        **Abstract**:

        Radar is ubiquitous in autonomous driving systems due to its low cost and good adaptability to bad weather. Nevertheless, the radar detection performance is usually inferior because its point cloud is sparse and not accurate due to the poor azimuth and elevation resolution. Moreover, point cloud generation algorithms already drop weak signals to reduce the false targets which may be suboptimal for the use of deep fusion. In this paper, we propose a novel method named EchoFusion to skip the existing radar signal processing pipeline and then incorporate the radar raw data with other sensors. Specifically, we first generate the Bird's Eye View (BEV) queries and then take corresponding spectrum features from radar to fuse with other sensors. By this approach, our method could utilize both rich and lossless distance and speed clues from radar echoes and rich semantic clues from images, making our method surpass all existing methods on the RADIal dataset, and approach the performance of LiDAR. The code will be released on https://github.com/tusen-ai/EchoFusion.

        ----

        ## [2347] D4: Improving LLM Pretraining via Document De-Duplication and Diversification

        **Authors**: *Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/a8f8cbd7f7a5fb2c837e578c75e5b615-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Over recent years, an increasing amount of compute and data has been poured into training large language models (LLMs), usually by doing one-pass learning on as many tokens as possible randomly selected from large-scale web corpora. While training on ever-larger portions of the internet leads to consistent performance improvements, the size of these improvements diminishes with scale, and there has been little work exploring the effect of data selection on pre-training and downstream performance beyond simple de-duplication methods such as MinHash. Here, we show that careful data selection (on top of de-duplicated data) via pre-trained model embeddings can speed up training (20% efficiency gains) and improves average downstream accuracy on 16 NLP tasks (up to 2%) at the 6.7B model scale. Furthermore, we show that repeating data intelligently consistently outperforms baseline training (while repeating random data performs worse than baseline training). Our results indicate that clever data selection can significantly improve LLM pre-training, calls into question the common practice of training for a single epoch on as much data as possible, and demonstrates a path to keep improving our models past the limits of randomly sampling web data.

        ----

        ## [2348] Effective Bayesian Heteroscedastic Regression with Deep Neural Networks

        **Authors**: *Alexander Immer, Emanuele Palumbo, Alexander Marx, Julia E. Vogt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html)

        **Abstract**:

        Flexibly quantifying both irreducible aleatoric and model-dependent epistemic uncertainties plays an important role for complex regression problems. While deep neural networks in principle can provide this flexibility and learn heteroscedastic aleatoric uncertainties through non-linear functions, recent works highlight that maximizing the log likelihood objective parameterized by mean and variance can lead to compromised mean fits since the gradient are scaled by the predictive variance, and propose adjustments in line with this premise. We instead propose to use the natural parametrization of the Gaussian, which has been shown to be more stable for heteroscedastic regression based on non-linear feature maps and Gaussian processes. Further, we emphasize the significance of principled regularization of the network parameters and prediction. We therefore propose an efficient Laplace approximation for heteroscedastic neural networks that allows automatic regularization through empirical Bayes and provides epistemic uncertainties, both of which improve generalization.We showcase on a range of regression problems—including a new heteroscedastic image regression benchmark—that our methods are scalable, improve over previous approaches for heteroscedastic regression, and provide epistemic uncertainty without requiring hyperparameter tuning.

        ----

        ## [2349] Multi-task learning with summary statistics

        **Authors**: *Parker Knight, Rui Duan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a924b7178e5975dfed1de235f0b72973-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a924b7178e5975dfed1de235f0b72973-Abstract-Conference.html)

        **Abstract**:

        Multi-task learning has emerged as a powerful machine learning paradigm for integrating data from multiple sources, leveraging similarities between tasks to improve overall model performance. However, the application of multi-task learning to real-world settings is hindered by data-sharing constraints, especially in healthcare settings. To address this challenge, we propose a flexible multi-task learning framework utilizing summary statistics from various sources. Additionally, we present an adaptive parameter selection approach based on a variant of Lepski's method, allowing for data-driven tuning parameter selection when only summary statistics are accessible. Our systematic non-asymptotic analysis characterizes the performance of the proposed methods under various regimes of the source datasets' sample complexity and overlap.  We demonstrate our theoretical findings and the performance of the method  through extensive simulations. This work offers a more flexible tool for training related models across various domains, with  practical implications in genetic risk prediction and many other fields.

        ----

        ## [2350] Estimating Noise Correlations Across Continuous Conditions With Wishart Processes

        **Authors**: *Amin Nejatbakhsh, Isabel Garon, Alex Williams*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a935ba2236c6ba0fb620f23354e789ff-Abstract-Conference.html)

        **Abstract**:

        The signaling capacity of a neural population depends on the scale and orientation of its covariance across trials. Estimating this "noise" covariance is challenging and is thought to require a large number of stereotyped trials. New approaches are therefore needed to interrogate the structure of neural noise across rich, naturalistic behaviors and sensory experiences, with few trials per condition. Here, we exploit the fact that conditions are smoothly parameterized in many experiments and leverage Wishart process models to pool statistical power from trials in neighboring conditions. We demonstrate that these models perform favorably on experimental data from the mouse visual cortex and monkey motor cortex relative to standard covariance estimators. Moreover, they produce smooth estimates of covariance as a function of stimulus parameters, enabling estimates of noise correlations in entirely unseen conditions as well as continuous estimates of Fisher information—a commonly used measure of signal fidelity. Together, our results suggest that Wishart processes are broadly applicable tools for quantification and uncertainty estimation of noise correlations in trial-limited regimes, paving the way toward understanding the role of noise in complex neural computations and behavior.

        ----

        ## [2351] Toward Understanding Generative Data Augmentation

        **Authors**: *Chenyu Zheng, Guoqiang Wu, Chongxuan Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html)

        **Abstract**:

        Generative data augmentation, which scales datasets by obtaining fake labeled examples from a trained conditional generative model, boosts classification performance in various learning tasks including (semi-)supervised learning, few-shot learning, and adversarially robust learning. However, little work has theoretically investigated the effect of generative data augmentation. To fill this gap, we establish a general stability bound in this not independently and identicallydistributed (non-i.i.d.) setting, where the learned distribution is dependent on the original train set and generally not the same as the true distribution. Our theoretical result includes the divergence between the learned distribution and the true distribution. It shows that generative data augmentation can enjoy a faster learning rate when the order of divergence term is $o(\max\left( \log(m)\beta_m, 1 / \sqrt{m})\right)$, where $m$ is the train set size and $\beta_m$ is the corresponding stability constant. We further specify the learning setup to the Gaussian mixture model and generative adversarial nets. We prove that in both cases, though generative data augmentation does not enjoy a faster learning rate, it can improve the learning guarantees at a constant level when the train set is small, which is significant when the awful overfitting occurs. Simulation results on the Gaussian mixture model and empirical results on generative adversarial nets support our theoretical conclusions.

        ----

        ## [2352] TOA: Task-oriented Active VQA

        **Authors**: *Xiaoying Xing, Mingfu Liang, Ying Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a95cc4f370bcc418e7b57d6512e28f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a95cc4f370bcc418e7b57d6512e28f52-Abstract-Conference.html)

        **Abstract**:

        Knowledge-based visual question answering (VQA) requires external knowledge to answer the question about an image. Early methods explicitly retrieve knowledge from external knowledge bases, which often introduce noisy information. Recently large language models like GPT-3 have shown encouraging performance as implicit knowledge source and revealed planning abilities. However, current large language models can not effectively understand image inputs, thus it remains an open problem to extract the image information and input to large language models. Prior works have used image captioning and object descriptions to represent the image. However, they may either drop the essential visual information to answer the question correctly or involve irrelevant objects to the task-of-interest. To address this problem, we propose to let large language models make an initial hypothesis according to their knowledge, then actively collect the visual evidence required to verify the hypothesis. In this way, the model can attend to the essential visual information in a task-oriented manner. We leverage several vision modules from the perspectives of spatial attention (i.e., Where to look) and attribute attention (i.e., What to look), which is similar to human cognition. The experiments show that our proposed method outperforms the baselines on open-ended knowledge-based VQA datasets and presents clear reasoning procedure with better interpretability.

        ----

        ## [2353] Universal Gradient Descent Ascent Method for Nonconvex-Nonconcave Minimax Optimization

        **Authors**: *Taoli Zheng, Linglingzhi Zhu, Anthony Man-Cho So, Jose H. Blanchet, Jiajin Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a961dea42c23c3c0d01b79918701fb6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a961dea42c23c3c0d01b79918701fb6e-Abstract-Conference.html)

        **Abstract**:

        Nonconvex-nonconcave minimax optimization has received intense attention over the last decade due to its broad applications in machine learning. Most existing algorithms rely on one-sided information, such as the convexity (resp. concavity) of the primal (resp. dual) functions, or other specific structures, such as the Polyak-Łojasiewicz (PŁ) and Kurdyka-Łojasiewicz (KŁ) conditions. However, verifying these regularity conditions is challenging in practice. To meet this challenge, we propose a novel universally applicable single-loop algorithm, the doubly smoothed gradient descent ascent method (DS-GDA), which naturally balances the primal and dual updates. That is, DS-GDA with the same hyperparameters is able to uniformly solve nonconvex-concave, convex-nonconcave, and nonconvex-nonconcave problems with one-sided KŁ properties, achieving convergence with $\mathcal{O}(\epsilon^{-4})$ complexity. Sharper (even optimal) iteration complexity can be obtained when the KŁ exponent is known. Specifically, under the one-sided KŁ condition with exponent $\theta\in(0,1)$, DS-GDA converges with an iteration complexity of $\mathcal{O}(\epsilon^{-2\max\\{2\theta,1\\}})$. They all match the corresponding best results in the literature. Moreover, we show that DS-GDA is practically applicable to general nonconvex-nonconcave problems even without any regularity conditions, such as the PŁ condition, KŁ condition, or weak Minty variational inequalities condition.  For various challenging nonconvex-nonconcave examples in the literature, including *Forsaken*, *Bilinearly-coupled minimax*, *Sixth-order polynomial*, and *PolarGame*, the proposed DS-GDA can all get rid of limit cycles. To the best of our knowledge, this is the first first-order algorithm to achieve convergence on all of these formidable problems.

        ----

        ## [2354] On Evaluating Adversarial Robustness of Large Vision-Language Models

        **Authors**: *Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Cheung, Min Lin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html)

        **Abstract**:

        Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, multimodal generation exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulnerable modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where adversaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a surprisingly high success rate for generating targeted responses. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: https://yunqing-me.github.io/AttackVLM/.

        ----

        ## [2355] Generator Born from Classifier

        **Authors**: *Runpeng Yu, Xinchao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97f0218b49bc17ea3f121a0e724f028-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97f0218b49bc17ea3f121a0e724f028-Abstract-Conference.html)

        **Abstract**:

        In this paper, we make a bold attempt toward an ambitious task: given a pre-trained classifier, we aim to reconstruct an image generator, without relying on any data samples. From a black-box perspective, this challenge seems intractable, since it inevitably involves identifying the inverse function for a classifier, which is, by nature, an information extraction process. As such, we resort to leveraging the knowledge encapsulated within the parameters of the neural network. Grounded on the theory of Maximum-Margin Bias of gradient descent, we propose a novel learning paradigm, in which the generator is trained to ensure that the convergence conditions of the network parameters are satisfied over the generated distribution of the samples. Empirical validation from various image generation tasks substantiates the efficacy of our strategy.

        ----

        ## [2356] Scattering Vision Transformer: Spectral Mixing Matters

        **Authors**: *Badri N. Patro, Vijay Srinivas Agneeswaran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a97f8072e51a785434b2da3e9cbf5aae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a97f8072e51a785434b2da3e9cbf5aae-Abstract-Conference.html)

        **Abstract**:

        Vision transformers have gained significant attention and achieved state-of-the-art performance in various computer vision tasks, including image classification, instance segmentation, and object detection. However, challenges remain in addressing attention complexity and effectively capturing fine-grained information within images. Existing solutions often resort to down-sampling operations, such as pooling, to reduce computational cost. Unfortunately, such operations are non-invertible and can result in information loss. In this paper, we present a novel approach called Scattering Vision Transformer (SVT) to tackle these challenges. SVT incorporates a spectrally scattering network that enables the capture of intricate image details. SVT overcomes the invertibility issue associated with down-sampling operations by separating low-frequency and high-frequency components. Furthermore, SVT introduces a unique spectral gating network utilizing Einstein multiplication for token and channel mixing, effectively reducing complexity. We show that SVT achieves state-of-the-art performance on the ImageNet dataset with a significant reduction in a number of parameters and FLOPS. SVT shows 2\% improvement over LiTv2 and iFormer. SVT-H-S reaches 84.2\% top-1 accuracy, while SVT-H-B reaches 85.2\% (state-of-art for base versions) and SVT-H-L reaches 85.7\% (again state-of-art for large versions). SVT also shows comparable results in other vision tasks such as instance segmentation. SVT also outperforms other transformers in transfer learning on standard datasets such as CIFAR10, CIFAR100, Oxford Flower, and Stanford Car datasets.  The project page is available on this webpage.\url{https://badripatro.github.io/svt/}.

        ----

        ## [2357] Task-aware world model learning with meta weighting via bi-level optimization

        **Authors**: *Huining Yuan, Hongkun Dou, Xingyu Jiang, Yue Deng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a995960dd0193654d6b18eca4ac5b936-Abstract-Conference.html)

        **Abstract**:

        Aligning the world model with the environment for the agentâ€™s specific task is crucial in model-based reinforcement learning. While value-equivalent models may achieve better task awareness than maximum-likelihood models, they sacrifice a large amount of semantic information and face implementation issues. To combine the benefits of both types of models, we propose Task-aware Environment Modeling Pipeline with bi-level Optimization (TEMPO), a bi-level model learning framework that introduces an additional level of optimization on top of a maximum-likelihood model by incorporating a meta weighter network that weights each training sample. The meta weighter in the upper level learns to generate novel sample weights by minimizing a proposed task-aware model loss. The model in the lower level focuses on important samples while maintaining rich semantic information in state representations. We evaluate TEMPO on a variety of continuous and discrete control tasks from the DeepMind Control Suite and Atari video games. Our results demonstrate that TEMPO achieves state-of-the-art performance regarding asymptotic performance, training stability, and convergence speed.

        ----

        ## [2358] LEPARD: Learning Explicit Part Discovery for 3D Articulated Shape Reconstruction

        **Authors**: *Di Liu, Anastasis Stathopoulos, Qilong Zhangli, Yunhe Gao, Dimitris N. Metaxas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a99f50fb024a56d15f057a1830ed0a00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a99f50fb024a56d15f057a1830ed0a00-Abstract-Conference.html)

        **Abstract**:

        Reconstructing the 3D articulated shape of an animal from a single in-the-wild image is a challenging task. We propose LEPARD, a learning-based framework that discovers semantically meaningful 3D parts and reconstructs 3D shapes in a part-based manner. This is advantageous as 3D parts are robust to pose variations due to articulations and their shape is typically simpler than the overall shape of the object. In our framework, the parts are explicitly represented as parameterized primitive surfaces with global and local deformations in 3D that deform to match the image evidence. We propose a kinematics-inspired optimization to guide each transformation of the primitive deformation given 2D evidence. Similar to recent approaches, LEPARD is only trained using off-the-shelf deep features from DINO and does not require any form of 2D or 3D annotations. Experiments on 3D animal shape reconstruction, demonstrate significant improvement over existing alternatives in terms of both the overall reconstruction performance as well as the ability to discover semantically meaningful and consistent parts.

        ----

        ## [2359] Curriculum Learning With Infant Egocentric Videos

        **Authors**: *Saber Sheybani, Himanshu Hansaria, Justin Wood, Linda B. Smith, Zoran Tiganj*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html)

        **Abstract**:

        Infants possess a remarkable ability to rapidly learn and process visual inputs. As an infant's mobility increases, so does the variety and dynamics of their visual inputs. Is this change in the properties of the visual inputs beneficial or even critical for the proper development of the visual system? To address this question, we used video recordings from infants wearing head-mounted cameras to train a variety of self-supervised learning models. Critically, we separated the infant data by age group and evaluated the importance of training with a curriculum aligned with developmental order. We found that initiating learning with the data from the youngest age group provided the strongest learning signal and led to the best learning outcomes in terms of downstream task performance. We then showed that the benefits of the data from the youngest age group are due to the slowness and simplicity of the visual experience. The results provide strong empirical evidence for the importance of the properties of the early infant experience and developmental progression in training. More broadly, our approach and findings take a noteworthy step towards reverse engineering the learning mechanisms in newborn brains using image-computable models from artificial intelligence.

        ----

        ## [2360] MarioGPT: Open-Ended Text2Level Generation through Large Language Models

        **Authors**: *Shyam Sudhakaran, Miguel González Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, Sebastian Risi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9bbeb2858dfbdbd4c19814e5d80ec60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9bbeb2858dfbdbd4c19814e5d80ec60-Abstract-Conference.html)

        **Abstract**:

        Procedural Content Generation (PCG) is a technique to generate complex and diverse environments in an automated way. However, while generating content with PCG methods is often straightforward, generating meaningful content that reflects specific  intentions and constraints remains challenging. Furthermore, many PCG algorithms lack the ability to generate content in an open-ended manner. Recently, Large Language Models (LLMs) have  shown to be incredibly effective in many diverse domains. These trained LLMs can be fine-tuned, re-using information and accelerating training for new tasks. Here, we introduce MarioGPT, a fine-tuned GPT2 model trained to generate tile-based game levels, in our case Super Mario Bros levels. MarioGPT can not only generate diverse levels, but can be  text-prompted for controllable level generation, addressing one of the key challenges of current PCG techniques.  As far as we know, MarioGPT is the first text-to-level model and combined with novelty search it enables the  generation of diverse levels with varying play-style dynamics (i.e. player paths) and the open-ended discovery of an increasingly diverse range of content. Code available at https://github.com/shyamsn97/mario-gpt.

        ----

        ## [2361] Is Learning in Games Good for the Learners?

        **Authors**: *William Brown, Jon Schneider, Kiran Vodrahalli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/a9ea92ef18aae17627d133534209e640-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/a9ea92ef18aae17627d133534209e640-Abstract-Conference.html)

        **Abstract**:

        We consider a number of questions related to tradeoffs between reward and regret in repeated gameplay between two agents. To facilitate this,  we introduce a notion of generalized equilibrium which allows for asymmetric regret constraints, and yields polytopes of feasible values for each agent and pair of regret constraints, where we show that any such equilibrium is reachable by a pair of algorithms which maintain their regret guarantees against arbitrary opponents. As a central example, we highlight the case one agent is no-swap and the other's regret is unconstrained. We show that this captures an extension of Stackelberg equilibria with a matching optimal value, and that there exists a wide class of games where a player can significantly increase their utility by deviating from a no-swap-regret algorithm against a no-swap learner (in fact, almost any game without pure Nash equilibria is of this form). Additionally, we make use of generalized equilibria to consider tradeoffs in terms of the opponent's algorithm choice. We give a tight characterization for the maximal reward obtainable against some no-regret learner, yet we also show a class of games in which this is bounded away from the value obtainable against the class of common "mean-based" no-regret algorithms. Finally, we consider the question of learning reward-optimal strategies via repeated play with a no-regret agent when the game is initially unknown. Again we show tradeoffs depending on the opponent's learning algorithm: the Stackelberg strategy is learnable in exponential time with any no-regret agent (and in polynomial time with any no-adaptive-regret agent) for any game where it is learnable via queries, and there are games where it is learnable in polynomial time against any no-swap-regret agent but requires exponential time against a mean-based no-regret agent.

        ----

        ## [2362] The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit

        **Authors**: *Lorenzo Noci, Chuning Li, Mufan Bill Li, Bobby He, Thomas Hofmann, Chris J. Maddison, Dan Roy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html)

        **Abstract**:

        In deep learning theory, the covariance matrix of the representations serves as aproxy to examine the network’s trainability. Motivated by the success of Transform-ers, we study the covariance matrix of a modified Softmax-based attention modelwith skip connections in the proportional limit of infinite-depth-and-width. Weshow that at initialization the limiting distribution can be described by a stochasticdifferential equation (SDE) indexed by the depth-to-width ratio. To achieve awell-defined stochastic limit, the Transformer’s attention mechanism is modifiedby centering the Softmax output at identity, and scaling the Softmax logits by awidth-dependent temperature parameter. We examine the stability of the networkthrough the corresponding SDE, showing how the scale of both the drift and diffu-sion can be elegantly controlled with the aid of residual connections. The existenceof a stable SDE implies that the covariance structure is well-behaved, even for verylarge depth and width, thus preventing the notorious issues of rank degeneracyin deep attention models. Finally, we show, through simulations, that the SDEprovides a surprisingly good description of the corresponding finite-size model.We coin the name shaped Transformer for these architectural modifications.

        ----

        ## [2363] Decompose Novel into Known: Part Concept Learning For 3D Novel Class Discovery

        **Authors**: *Tingyu Weng, Jun Xiao, Haiyong Jiang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa31eee8f2351176ddd4d14646d4a950-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa31eee8f2351176ddd4d14646d4a950-Abstract-Conference.html)

        **Abstract**:

        In this work, we address 3D novel class discovery (NCD) that discovers novel classes from an unlabeled dataset by leveraging the knowledge of disjoint known classes. The key challenge of 3D NCD is that learned features by known class recognition are heavily biased and hinder generalization to novel classes. Since geometric parts are more generalizable across different classes, we propose to decompose novel into known parts, coined DNIK, to mitigate the above problems. DNIK learns a part concept bank encoding rich part geometric patterns from known classes so that novel 3D shapes can be represented as part concept compositions to facilitate cross-category generalization. Moreover, we formulate three constraints on part concepts to ensure diverse part concepts without collapsing. A part relation encoding module (PRE) is also developed to leverage part-wise spatial relations for better recognition. We construct three 3D NCD tasks for evaluation and extensive experiments show that our method achieves significantly superior results than SOTA baselines (+11.7%, +14.1%, and +16.3% improvements on average for three tasks, respectively). Code and data will be released.

        ----

        ## [2364] Long Sequence Hopfield Memory

        **Authors**: *Hamza Tahir Chaudhry, Jacob A. Zavatone-Veth, Dmitry Krotov, Cengiz Pehlevan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa32ebcdd2ce1bed4ef7f456fc8fa5c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa32ebcdd2ce1bed4ef7f456fc8fa5c1-Abstract-Conference.html)

        **Abstract**:

        Sequence memory is an essential attribute of natural and artificial intelligence that enables agents to encode, store, and retrieve complex sequences of stimuli and actions. Computational models of sequence memory have been proposed where recurrent Hopfield-like neural networks are trained with temporally asymmetric Hebbian rules. However, these networks suffer from limited sequence capacity (maximal length of the stored sequence) due to interference between the memories. Inspired by recent work on Dense Associative Memories, we expand the sequence capacity of these models by introducing a nonlinear interaction term, enhancing separation between the patterns. We derive novel scaling laws for sequence capacity with respect to network size, significantly outperforming existing scaling laws for models based on traditional Hopfield networks, and verify these theoretical results with numerical simulation. Moreover, we introduce a generalized pseudoinverse rule to recall sequences of highly correlated patterns. Finally, we extend this model to store sequences with variable timing between states' transitions and describe a biologically-plausible implementation, with connections to motor neuroscience.

        ----

        ## [2365] Provably Safe Reinforcement Learning with Step-wise Violation Constraints

        **Authors**: *Nuoya Xiong, Yihan Du, Longbo Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa3e67220ca4cd50010165c950fc8056-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa3e67220ca4cd50010165c950fc8056-Abstract-Conference.html)

        **Abstract**:

        We investigate a novel safe reinforcement learning problem with step-wise violation constraints. Our problem differs from existing works in that we focus on stricter step-wise violation constraints and do not assume the existence of safe actions, making our formulation more suitable for safety-critical applications that need to ensure safety in all decision steps but may not always possess safe actions, e.g., robot control and autonomous driving.We propose an efficient algorithm SUCBVI, which guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ or gap-dependent $\widetilde{\mathcal{O}}(S/\mathcal{C}_{\mathrm{gap}} + S^2AH^2)$ step-wise violation and $\widetilde{\mathcal{O}}(\sqrt{H^3SAT})$ regret. Lower bounds are provided to validate the optimality in both violation and  regret performance with respect to the number of states $S$ and the total number of steps $T$. Moreover, we further study an innovative safe reward-free exploration problem with step-wise violation constraints. For this problem, we design algorithm SRF-UCRL to find a near-optimal safe policy, which achieves nearly state-of-the-art  sample complexity $\widetilde{\mathcal{O}}((\frac{S^2AH^2}{\varepsilon}+\frac{H^4SA}{\varepsilon^2})(\log(\frac{1}{\delta})+S))$, and guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ violation during exploration.  Experimental results demonstrate the  superiority of our algorithms in safety performance and corroborate our theoretical results.

        ----

        ## [2366] Human spatiotemporal pattern learning as probabilistic program synthesis

        **Authors**: *Tracey Mills, Josh Tenenbaum, Samuel Cheyette*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5c083f9d387c49514eb5c4dc2dc16b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5c083f9d387c49514eb5c4dc2dc16b-Abstract-Conference.html)

        **Abstract**:

        People are adept at learning a wide variety of structured patterns from small amounts of data, presenting a conundrum from the standpoint of the bias-variance tradeoff: what kinds of representations and algorithms support the joint flexibility and data-paucity of human learning? One possibility is that people "learn by programming": inducing probabilistic models to fit observed data. Here, we experimentally test human learning in the domain of structured 2-dimensional patterns, using a task in which participants repeatedly predicted where a dot would move based on its previous trajectory. We evaluate human performance against standard parametric and non-parametric time-series models, as well as two Bayesian program synthesis models whose hypotheses vary in their degree of structure: a compositional Gaussian Process model and a structured "Language of Thought" (LoT) model. We find that signatures of human pattern learning are best explained by the LoT model, supporting the idea that the flexibility and data-efficiency of human structure learning can be understood as probabilistic inference over an expressive space of programs.

        ----

        ## [2367] Fair Allocation of Indivisible Chores: Beyond Additive Costs

        **Authors**: *Bo Li, Fangxiao Wang, Yu Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5d22c77b380e2261332bb641b3c2e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5d22c77b380e2261332bb641b3c2e3-Abstract-Conference.html)

        **Abstract**:

        We study the maximin share (MMS) fair allocation of $m$ indivisible tasks to $n$ agents who have costs for completing the assigned tasks.It is known that exact MMS fairness cannot be guaranteed, and so far the best-known approximation for additive cost functions is $\frac{13}{11}$ by Huang and Segal-Halevi [EC, 2023]; however, beyond additivity, very little is known. In this work, we first prove that no algorithm can ensure better than $\min\{n,\frac{\log m}{\log \log m}\}$-approximation if the cost functions are submodular. This result also shows a sharp contrast with the allocation of goods where constant approximations exist as shown by Barman and Krishnamurthy [TEAC, 2020] and Ghodsi et al. [AIJ, 2022]. We then prove that for subadditive costs, there always exists an allocation that is $\min\{n,\lceil\log m\rceil\}$-approximation, and thus the approximation ratio is asymptotically tight.Besides multiplicative approximation, we also consider the ordinal relaxation, 1-out-of-$d$ MMS, which was recently proposed by Hosseini et al. [JAIR and AAMAS, 2022]. Our impossibility result implies that for any $d\ge 2$, a 1-out-of-$d$ MMS allocation may not exist.Due to these hardness results for general subadditive costs, we turn to studying two specific subadditive costs, namely, bin packing and job scheduling. For both settings, we show that constant approximate allocations exist for both multiplicative and ordinal relaxations of MMS.

        ----

        ## [2368] Robust Second-Order Nonconvex Optimization and Its Application to Low Rank Matrix Sensing

        **Authors**: *Shuyao Li, Yu Cheng, Ilias Diakonikolas, Jelena Diakonikolas, Rong Ge, Stephen J. Wright*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa5f224975a67914067519faddeacba3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa5f224975a67914067519faddeacba3-Abstract-Conference.html)

        **Abstract**:

        Finding an approximate second-order stationary point (SOSP) is a well-studied and fundamental problem in stochastic nonconvex optimization with many applications in machine learning.However, this problem is poorly understood in the presence of outliers, limiting the use of existing nonconvex algorithms in adversarial settings.In this paper, we study the problem of finding SOSPs in the strong contamination model, where a constant fraction of datapoints are arbitrarily corrupted.We introduce a general framework for efficiently finding an approximate SOSP with \emph{dimension-independent} accuracy guarantees, using $\widetilde{O}({D^2}/{\epsilon})$ samples where $D$ is the ambient dimension and $\epsilon$ is the fraction of corrupted datapoints.As a concrete application of our framework, we apply it to the problem of low rank matrix sensing, developing efficient and provably robust algorithms that can tolerate corruptions in both the sensing matrices and the measurements.In addition, we establish a Statistical Query lower bound providing evidence that the quadratic dependence on $D$ in the sample complexity is necessary for computationally efficient algorithms.

        ----

        ## [2369] Incentivized Communication for Federated Bandits

        **Authors**: *Zhepei Wei, Chuanhao Li, Haifeng Xu, Hongning Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa61d142c0081a8259a6372a3bb0af2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa61d142c0081a8259a6372a3bb0af2b-Abstract-Conference.html)

        **Abstract**:

        Most existing works on federated bandits take it for granted that all clients are altruistic about sharing their data with the server for the collective good whenever needed. Despite their compelling theoretical guarantee on performance and communication efficiency, this assumption is overly idealistic and oftentimes violated in practice, especially when the algorithm is operated over self-interested clients, who are reluctant to share data without explicit benefits. Negligence of such self-interested behaviors can significantly affect the learning efficiency and even the practical operability of federated bandit learning. In light of this, we aim to spark new insights into this under-explored research area by formally introducing an incentivized communication problem for federated bandits, where the server shall motivate clients to share data by providing incentives. Without loss of generality, we instantiate this bandit problem with the contextual linear setting and propose the first incentivized communication protocol, namely, Inc-FedUCB, that achieves near-optimal regret with provable communication and incentive cost guarantees. Extensive empirical experiments on both synthetic and real-world datasets further validate the effectiveness of the proposed method across various environments.

        ----

        ## [2370] Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand

        **Authors**: *Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, Bo Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa6287ca31ae1474ea802342d0c8ba63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa6287ca31ae1474ea802342d0c8ba63-Abstract-Conference.html)

        **Abstract**:

        The prosperity of deep neural networks (DNNs) is largely benefited from open-source datasets, based on which users can evaluate and improve their methods. In this paper, we revisit backdoor-based dataset ownership verification (DOV), which is currently the only feasible approach to protect the copyright of open-source datasets. We reveal that these methods are fundamentally harmful given that they could introduce malicious misclassification behaviors to watermarked DNNs by the adversaries. In this paper, we design DOV from another perspective by making watermarked models (trained on the protected dataset) correctly classify some `hard' samples that will be misclassified by the benign model. Our method is inspired by the generalization property of DNNs, where we find a \emph{hardly-generalized domain} for the original dataset (as its \emph{domain watermark}). It can be easily learned with the protected dataset containing modified samples. Specifically, we formulate the domain generation as a bi-level optimization and propose to optimize a set of visually-indistinguishable clean-label modified data with similar effects to domain-watermarked samples from the hardly-generalized domain to ensure watermark stealthiness. We also design a hypothesis-test-guided ownership verification via our domain watermark and provide the theoretical analyses of our method. Extensive experiments on three benchmark datasets are conducted, which verify the effectiveness of our method and its resistance to potential adaptive methods.

        ----

        ## [2371] DISCO-10M: A Large-Scale Music Dataset

        **Authors**: *Luca A. Lanzendörfer, Florian Grötschla, Emil Funke, Roger Wattenhofer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa7ef4c0f4aaabf376088a1a74e09d4c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa7ef4c0f4aaabf376088a1a74e09d4c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Music datasets play a crucial role in advancing research in machine learning for music. However, existing music datasets suffer from limited size, accessibility, and lack of audio resources. To address these shortcomings, we present DISCO-10M, a novel and extensive music dataset that surpasses the largest previously available music dataset by an order of magnitude. To ensure high-quality data, we implement a multi-stage filtering process. This process incorporates similarities based on textual descriptions and audio embeddings. Moreover, we provide precomputed CLAP embeddings alongside DISCO-10M, facilitating direct application on various downstream tasks. These embeddings enable efficient exploration of machine learning applications on the provided data. With DISCO-10M, we aim to democratize and facilitate new research to help advance the development of novel machine learning models for music: https://huggingface.co/DISCOX

        ----

        ## [2372] Guide Your Agent with Adaptive Multimodal Rewards

        **Authors**: *Changyeon Kim, Younggyo Seo, Hao Liu, Lisa Lee, Jinwoo Shin, Honglak Lee, Kimin Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html)

        **Abstract**:

        Developing an agent capable of adapting to unseen environments remains a difficult challenge in imitation learning. This work presents Adaptive Return-conditioned Policy (ARP), an efficient framework designed to enhance the agent's generalization ability using natural language task descriptions and pre-trained multimodal encoders. Our key idea is to calculate a similarity between visual observations and natural language instructions in the pre-trained multimodal embedding space (such as CLIP) and use it as a reward signal. We then train a return-conditioned policy using expert demonstrations labeled with multimodal rewards. Because the multimodal rewards provide adaptive signals at each timestep, our ARP effectively mitigates the goal misgeneralization. This results in superior generalization performances even when faced with unseen text instructions, compared to existing text-conditioned policies. To improve the quality of rewards, we also introduce a fine-tuning method for pre-trained multimodal encoders, further enhancing the performance. Video demonstrations and source code are available on the project website: \url{https://sites.google.com/view/2023arp}.

        ----

        ## [2373] Fine-Grained Theoretical Analysis of Federated Zeroth-Order Optimization

        **Authors**: *Jun Chen, Hong Chen, Bin Gu, Hao Deng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aaa973f65b98c96e5f850d706464a3c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aaa973f65b98c96e5f850d706464a3c4-Abstract-Conference.html)

        **Abstract**:

        Federated zeroth-order optimization (FedZO) algorithm enjoys the advantages of both zeroth-order optimization and federated learning, and has shown exceptional performance on black-box attack and softmax regression tasks. However, there is no generalization analysis for FedZO, and its analysis on computing convergence rate is slower than the corresponding first-order optimization setting. This paper aims to establish systematic theoretical assessments of FedZO by developing the analysis technique of on-average model stability. We establish the first generalization error bound of FedZO under the Lipschitz continuity and smoothness conditions. Then, refined generalization and optimization bounds are provided by replacing bounded gradient with heavy-tailed gradient noise and utilizing the second-order Taylor expansion for gradient approximation. With the help of a new error decomposition strategy, our theoretical analysis is also extended to the asynchronous case. For FedZO, our fine-grained analysis fills the theoretical gap on the generalization guarantees and polishes the convergence characterization of the computing algorithm.

        ----

        ## [2374] Sparse Deep Learning for Time Series Data: Theory and Applications

        **Authors**: *Mingxuan Zhang, Yan Sun, Faming Liang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aaa9c20f0a217a1aef6fa5d97f310292-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aaa9c20f0a217a1aef6fa5d97f310292-Abstract-Conference.html)

        **Abstract**:

        Sparse deep learning has become a popular technique for improving the performance of deep neural networks in areas such as uncertainty quantification, variable selection, and large-scale network compression. However, most existing research has focused on problems where the observations are independent and identically distributed (i.i.d.), and there has been little work on the problems where the observations are dependent, such as time series data and sequential data in natural language processing. This paper aims to address this gap by studying the theory for sparse deep learning with dependent data. We show that sparse recurrent neural networks (RNNs) can be consistently estimated, and their predictions are asymptotically normally distributed under appropriate assumptions, enabling the prediction uncertainty to be correctly quantified. Our numerical results show that sparse deep learning outperforms state-of-the-art methods, such as conformal predictions, in prediction uncertainty quantification for time series data. Furthermore, our results indicate that the proposed method can consistently identify the autoregressive order for time series data and outperform existing methods in large-scale model compression. Our proposed method has important practical implications in fields such as finance, healthcare, and energy, where both accurate point estimates and prediction uncertainty quantification are of concern.

        ----

        ## [2375] Imbalanced Mixed Linear Regression

        **Authors**: *Pini Zilber, Boaz Nadler*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aad615d33ba5071045656ba24d800c7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aad615d33ba5071045656ba24d800c7b-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of mixed linear regression (MLR), where each observed sample belongs to one of $K$ unknown linear models. In practical applications, the mixture of the $K$ models may be imbalanced with a significantly different number of samples from each model. Unfortunately, most MLR methods do not perform well in such settings. Motivated by this practical challenge, in this work we propose Mix-IRLS, a novel, simple and fast algorithm for MLR with excellent performance on both balanced and imbalanced mixtures.In contrast to popular approaches that recover the $K$ models simultaneously, Mix-IRLS does it sequentially using tools from robust regression. Empirically, beyond imbalanced mixtures, Mix-IRLS succeeds in a broad range of additional settings where other methods fail, including small sample sizes, presence of outliers, and an unknown number of models $K$. Furthermore, Mix-IRLS outperforms competing methods on several real-world datasets, in some cases by a large margin. We complement our empirical results by deriving a recovery guarantee for Mix-IRLS, which highlights its advantage on imbalanced mixtures.

        ----

        ## [2376] GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference

        **Authors**: *Ziang Li, Mengda Yang, Yaxin Liu, Juan Wang, Hongxin Hu, Wenzhe Yi, Xiaoyang Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)

        **Abstract**:

        Split Inference (SI) is an emerging deep learning paradigm that addresses computational constraints on edge devices and preserves data privacy through collaborative edge-cloud approaches. However, SI is vulnerable to Data Reconstruction Attacks (DRA), which aim to reconstruct users' private prediction instances. Existing attack methods suffer from various limitations. Optimization-based DRAs do not leverage public data effectively, while Learning-based DRAs depend heavily on auxiliary data quantity and distribution similarity. Consequently, these approaches yield unsatisfactory attack results and are sensitive to defense mechanisms. To overcome these challenges, we propose a GAN-based LAtent Space Search attack (GLASS) that harnesses abundant prior knowledge from public data using advanced StyleGAN technologies. Additionally, we introduce GLASS++ to enhance reconstruction stability. Our approach represents the first GAN-based DRA against SI, and extensive evaluation across different split points and adversary setups demonstrates its state-of-the-art performance. Moreover, we thoroughly examine seven defense mechanisms, highlighting our method's capability to reveal private information even in the presence of these defenses.

        ----

        ## [2377] Random-Access Infinite Context Length for Transformers

        **Authors**: *Amirkeivan Mohtashami, Martin Jaggi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab05dc8bf36a9f66edbff6992ec86f56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab05dc8bf36a9f66edbff6992ec86f56-Abstract-Conference.html)

        **Abstract**:

        While Transformers have shown remarkable success in natural language processing, their attention mechanism's large memory requirements have limited their ability to handle longer contexts. Prior approaches, such as recurrent memory or retrieval-based augmentation, have either compromised the random-access flexibility of attention (i.e., the capability to select any token in the entire context) or relied on separate mechanisms for relevant context retrieval, which may not be compatible with the model's attention. In this paper, we present a novel approach that allows access to the complete context while retaining random-access flexibility, closely resembling running attention on the entire context. Our method uses a landmark token to represent each block of the input and trains the attention to use it for selecting relevant blocks, enabling retrieval of blocks directly through the attention mechanism instead of by relying on a separate mechanism. Our approach seamlessly integrates with specialized data structures and the system's memory hierarchy, enabling processing of arbitrarily long context lengths. We demonstrate that our method can obtain comparable performance with Transformer-XL while significantly reducing the number of retrieved tokens in each step. Finally, we show that fine-tuning LLaMA 7B with our method successfully extends its context length capacity to over 32k tokens, allowing for inference at the context lengths of GPT-4. We release the implementation of landmark attention and the code to reproduce our experiments at https://github.com/epfml/landmark-attention/.

        ----

        ## [2378] Egocentric Planning for Scalable Embodied Task Achievement

        **Authors**: *Xiaotian Liu, Héctor Palacios, Christian Muise*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab0b1be09c317cb068aecfa7fa86a7e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab0b1be09c317cb068aecfa7fa86a7e3-Abstract-Conference.html)

        **Abstract**:

        Embodied agents face significant challenges when tasked with performing actions in diverse environments, particularly in generalizing across object types and executing suitable actions to accomplish tasks. Furthermore, agents should exhibit robustness, minimizing the execution of illegal actions. In this work, we present Egocentric Planning, an innovative approach that combines symbolic planning and Object-oriented POMDPs to solve tasks in complex environments, harnessing existing models for visual perception and natural language processing. We evaluated our approach in ALFRED, a simulated environment designed for domestic tasks, and demonstrated its high scalability, achieving an impressive 36.07\% unseen success rate in the ALFRED benchmark and winning the ALFRED challenge at CVPR Embodied AI workshop. Our method requires reliable perception and the specification or learning of a symbolic description of the preconditions and effects of the agent's actions, as well as what object types reveal information about others. It can naturally scale to solve new tasks beyond ALFRED, as long as they can be solved using the available skills. This work offers a solid baseline for studying end-to-end and hybrid methods that aim to generalize to new tasks, including recent approaches relying on LLMs, but often struggle to scale to long sequences of actions or produce robust plans for novel tasks.

        ----

        ## [2379] Removing Hidden Confounding in Recommendation: A Unified Multi-Task Learning Approach

        **Authors**: *Haoxuan Li, Kunhan Wu, Chunyuan Zheng, Yanghao Xiao, Hao Wang, Zhi Geng, Fuli Feng, Xiangnan He, Peng Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab3f114401f0523ca1cc09de0621f400-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab3f114401f0523ca1cc09de0621f400-Abstract-Conference.html)

        **Abstract**:

        In recommender systems, the collected data used for training is always subject to selection bias, which poses a great challenge for unbiased learning. Previous studies proposed various debiasing methods based on observed user and item features, but ignored the effect of hidden confounding. To address this problem, recent works suggest the use of sensitivity analysis for worst-case control of the unknown true propensity, but only valid when the true propensity is near to the nominal propensity within a finite bound. In this paper, we first perform theoretical analysis to reveal the possible failure of previous approaches, including propensity-based, multi-task learning, and bi-level optimization methods, in achieving unbiased learning when hidden confounding is present. Then, we propose a unified multi-task learning approach to remove hidden confounding, which uses a few unbiased ratings to calibrate the learned nominal propensities and nominal error imputations from biased data. We conduct extensive experiments on three publicly available benchmark datasets containing a fully exposed large-scale industrial dataset, validating the effectiveness of the proposed methods in removing hidden confounding.

        ----

        ## [2380] Generative Category-level Object Pose Estimation via Diffusion Models

        **Authors**: *Jiyao Zhang, Mingdong Wu, Hao Dong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab59d149fc0c2c9039d3e3049f7914b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab59d149fc0c2c9039d3e3049f7914b1-Abstract-Conference.html)

        **Abstract**:

        Object pose estimation plays a vital role in embodied AI and computer vision, enabling intelligent agents to comprehend and interact with their surroundings. Despite the practicality of category-level pose estimation, current approaches encounter challenges with partially observed point clouds, known as the multihypothesis issue. In this study, we propose a novel solution by reframing categorylevel object pose estimation as conditional generative modeling, departing from traditional point-to-point regression. Leveraging score-based diffusion models, we estimate object poses by sampling candidates from the diffusion model and aggregating them through a two-step process: filtering out outliers via likelihood estimation and subsequently mean-pooling the remaining candidates. To avoid the costly integration process when estimating the likelihood, we introduce an alternative method that distils an energy-based model from the original score-based model, enabling end-to-end likelihood estimation. Our approach achieves state-of-the-art performance on the REAL275 dataset, surpassing 50% and 60% on strict 5 ◦ 2cm and 5 ◦ 5cm metrics, respectively. Furthermore, our method demonstrates strong generalization to novel categories without the need for fine-tuning and can readily adapt to object pose tracking tasks, yielding comparable results to the current state-of-the-art baselines. Our checkpoints and demonstrations can be found at https://sites.google.com/view/genpose.

        ----

        ## [2381] On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective

        **Authors**: *Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab5a2bf4385bee44f3919060b184605b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab5a2bf4385bee44f3919060b184605b-Abstract-Conference.html)

        **Abstract**:

        Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating saliency maps, and counterfactual explanations. However, saliency maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the saliency maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties:They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the saliency map for such models, we prove this gradient encodes  both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another.  Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the saliency map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.

        ----

        ## [2382] DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models

        **Authors**: *Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract-Conference.html)

        **Abstract**:

        Current deep networks are very data-hungry and benefit from training on large-scale datasets, which are often time-consuming to collect and annotate. By contrast, synthetic data can be generated infinitely using generative models such as DALL-E and diffusion models, with minimal effort and cost. In this paper, we present DatasetDM, a generic dataset generation model that can produce diverse syntheticimages and the corresponding high-quality perception annotations (e.g., segmentation masks, and depth). Our method builds upon the pre-trained diffusion model and extends text-guided image synthesis to perception data generation. We show that the rich latent code of the diffusion model can be effectively decoded as accurate perception annotations using a decoder module. Training the decoder only needs less than 1% (around 100 images) of manually labeled images, enabling the generation of an infinitely large annotated dataset. Then these synthetic data can be used for training various perception models on downstream tasks. To showcase the power of the proposed approach, we generate datasets with rich dense pixel-wise labels for a wide range of downstream tasks, including semantic15segmentation, instance segmentation, and depth estimation. Notably, it achieves 1) state-of-the-art results on semantic segmentation and instance segmentation; 2) significantly more efficient and robust in domain generalization than the real data; 3) state-of-the-art results in zero-shot segmentation setting; and 4) flexibility for efficient application and novel task composition (e.g., image editing)

        ----

        ## [2383] No-Regret Online Prediction with Strategic Experts

        **Authors**: *Omid Sadeghi, Maryam Fazel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ab9f9cfe97da3665e08f50ade9f8c4d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ab9f9cfe97da3665e08f50ade9f8c4d6-Abstract-Conference.html)

        **Abstract**:

        We study a generalization of the online binary prediction with expert advice framework where at each round, the learner is allowed to pick $m\geq 1$ experts from a pool of $K$ experts and the overall utility is a modular or submodular function of the chosen experts. We focus on the setting in which experts act strategically and aim to maximize their influence on the algorithm's predictions by potentially misreporting their beliefs about the events. Among others, this setting finds applications in forecasting competitions where the learner seeks not only to make predictions by aggregating different forecasters but also to rank them according to their relative performance. Our goal is to design algorithms that satisfy the following two requirements: 1) \emph{Incentive-compatible}: Incentivize the experts to report their beliefs truthfully, and 2) \emph{No-regret}: Achieve sublinear regret with respect to the true beliefs of the best fixed set of $m$ experts in hindsight. Prior works have studied this framework when $m=1$ and provided incentive-compatible no-regret algorithms for the problem. We first show that a simple reduction of our problem to the $m=1$ setting is neither efficient nor effective. Then, we provide algorithms that utilize the specific structure of the utility functions to achieve the two desired goals.

        ----

        ## [2384] Learning Unseen Modality Interaction

        **Authors**: *Yunhua Zhang, Hazel Doughty, Cees Snoek*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abb4847bbd60f38b1b7649d26c7a0067-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abb4847bbd60f38b1b7649d26c7a0067-Abstract-Conference.html)

        **Abstract**:

        Multimodal learning assumes all modality combinations of interest are available during training to learn cross-modal correspondences. In this paper, we challenge this modality-complete assumption for multimodal learning and instead strive for generalization to unseen modality combinations during inference. We pose the problem of unseen modality interaction and introduce a first solution. It exploits a module that projects the multidimensional features of different modalities into a common space with rich information preserved. This allows the information to be accumulated with a simple summation operation across available modalities. To reduce overfitting to less discriminative modality combinations during training, we further improve the model learning with pseudo-supervision indicating the reliability of a modalityâ€™s prediction. We demonstrate that our approach is effective for diverse tasks and modalities by evaluating it for multimodal video classification, robot state regression, and multimedia retrieval. Project website: https://xiaobai1217.github.io/Unseen-Modality-Interaction/.

        ----

        ## [2385] Autonomous Capability Assessment of Sequential Decision-Making Systems in Stochastic Settings

        **Authors**: *Pulkit Verma, Rushang Karia, Siddharth Srivastava*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abbb7f20cdffdd3bb7d98447f60b0b0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abbb7f20cdffdd3bb7d98447f60b0b0c-Abstract-Conference.html)

        **Abstract**:

        It is essential for users to understand what their AI systems can and can't do in order to use them safely. However, the problem of enabling users to assess AI systems with sequential decision-making (SDM) capabilities is relatively understudied. This paper presents a new approach for modeling the capabilities of black-box AI systems that can plan and act, along with the possible effects and requirements for executing those capabilities in stochastic settings. We present an active-learning approach that can effectively interact with a black-box SDM system and learn an interpretable probabilistic model describing its capabilities. Theoretical analysis of the approach identifies the conditions under which the learning process is guaranteed to converge to the correct model of the agent; empirical evaluations on different agents and simulated scenarios show that this approach is few-shot generalizable and can effectively describe the capabilities of arbitrary black-box SDM agents in a sample-efficient manner.

        ----

        ## [2386] Model-Free Active Exploration in Reinforcement Learning

        **Authors**: *Alessio Russo, Alexandre Proutière*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abbbb25cddb2c2cd08714e6bfa2f0634-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abbbb25cddb2c2cd08714e6bfa2f0634-Abstract-Conference.html)

        **Abstract**:

        We study the problem of exploration in Reinforcement Learning and present a novel model-free solution. We adopt an information-theoretical viewpoint and start from the  instance-specific lower bound of the number of samples that have to be collected to identify a nearly-optimal policy. Deriving this lower bound along with the optimal exploration strategy entails solving an intricate optimization problem and requires a model of the system. In turn, most existing sample optimal exploration algorithms rely on estimating the model. We derive an approximation of the instance-specific lower bound that only involves quantities that can be inferred using model-free approaches. Leveraging this approximation, we devise an ensemble-based model-free exploration strategy  applicable to both tabular and continuous Markov decision processes. Numerical results demonstrate that our strategy is able to identify efficient policies faster than state-of-the-art exploration approaches.

        ----

        ## [2387] Universality laws for Gaussian mixtures in generalized linear models

        **Authors**: *Yatin Dandi, Ludovic Stephan, Florent Krzakala, Bruno Loureiro, Lenka Zdeborová*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abccb8a90b30d45b948360ba41f5a20f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abccb8a90b30d45b948360ba41f5a20f-Abstract-Conference.html)

        **Abstract**:

        A recent line of work in high-dimensional statistics working under the Gaussian mixture hypothesis has led to a number of results in the context of empirical risk minimization, Bayesian uncertainty quantification, separation of kernel methods and neural networks, ensembling and fluctuation of random features. We provide rigorous proofs for the applicability of these results to a general class of datasets $(\mathbf{x_i},y_i, {i=1,\dots,n})$ containing independent samples from a mixture distribution $\sum_{c\in\mathcal{C}} \rho_{c}P_{c}^{\mathbf{x}}$. Specifically, we consider the hypothesis class of generalized linear models $\hat{y} = F(\mathbf{\Theta}^{\top}\mathbf{x})$ and investigate the asymptotic joint statistics of a family of generalized linear estimators $(\mathbf{\Theta}^{(1)}, \dots, \mathbf{\Theta}^{(M)})$, obtained either from (a) minimizing an empirical risk $\hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y})$ or (b) sampling from the associated Gibbs measure $\exp(-\beta n \hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y}))$. Our main contribution is to characterize under which conditions the asymptotic joint statistics of this family depends (on a weak sense) only on the means and covariances of the class conditional features distribution $P_{c}^{\mathbf{x}}$. This allows us to prove the universality of different quantities of interest, including training, generalization errors, as well as the geometrical properties and correlations of the estimators.

        ----

        ## [2388] ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers

        **Authors**: *Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, Lei Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abe1eb21ceb046209c96a0f5e7544ccc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abe1eb21ceb046209c96a0f5e7544ccc-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) excel at implementing code from functionality descriptions but struggle with algorithmic problems that require not only implementation but also identification of the suitable algorithm. Moreover, LLM-generated programs lack guaranteed correctness and require human verification. To address these challenges, we propose ALGO, a framework that synthesizes Algorithmic programs with LLM-Generated Oracles to guide the generation and verify their correctness. ALGO first generates a reference oracle by prompting an LLM to exhaustively enumerate all the combinations of relevant variables. This oracle is then utilized to guide an arbitrary search strategy in exploring the algorithm space and to verify the synthesized algorithms. Our study shows that the LLM-generatedoracles are correct for 88% of the cases. With the oracles as verifiers, ALGO can be integrated with any existing code generation model in a model-agnostic manner to enhance its performance. Experiments show that when equipped with ALGO, we achieve an 8× better one-submission pass rate over the Codex model and a 2.6× better one-submission pass rate over CodeT, the current state-of-the-art model on CodeContests. We can also get 1.3× better pass rate over the ChatGPT Code Interpreter on unseen problems. The problem set we used for testing, the prompts we used, the verifier and solution programs, and the test cases generated by ALGOare available at https://github.com/zkx06111/ALGO.

        ----

        ## [2389] Private Everlasting Prediction

        **Authors**: *Moni Naor, Kobbi Nissim, Uri Stemmer, Chao Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html)

        **Abstract**:

        A private learner is trained on a sample of labeled points and generates a hypothesis that can be used for predicting the labels of newly sampled points while protecting the privacy of the training set [Kasiviswannathan et al., FOCS 2008]. Past research uncovered that private learners may need to exhibit significantly higher sample complexity than non-private learners as is the case of learning of one-dimensional threshold functions [Bun et al., FOCS 2015, Alon et al., STOC 2019].We explore prediction as an alternative to learning. A predictor answers a stream of classification queries instead of outputting a hypothesis. Earlier work has considered a private prediction model with a single classification query [Dwork and Feldman, COLT 2018]. We observe that when answering a stream of queries, a predictor must modify the hypothesis it uses over time, and in a manner that  cannot rely solely on the training set.We introduce {\em private everlasting prediction} taking into account the privacy of both the training set {\em and} the (adaptively chosen) queries made to the predictor. We then present a generic construction of private everlasting predictors in the PAC model.The sample complexity of the initial training sample in our construction is quadratic (up to polylog factors) in the VC dimension of the concept class. Our construction allows prediction for all concept classes with finite VC dimension, and in particular threshold functions over infinite domains, for which (traditional) private learning is known to be impossible.

        ----

        ## [2390] A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation

        **Authors**: *Thomas Fel, Victor Boutin, Louis Béthune, Rémi Cadène, Mazda Moayeri, Léo Andéol, Mathieu Chalvidal, Thomas Serre*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html)

        **Abstract**:

        In recent years, concept-based approaches have emerged as some of the most promising explainability methods to help us interpret the decisions of Artificial Neural Networks (ANNs). These methods seek to discover intelligible visual ``concepts'' buried within the complex patterns of ANN activations in two key steps: (1) concept extraction followed by (2) importance estimation. While these two steps are shared across methods, they all differ in their specific implementations. Here, we introduce a unifying theoretical framework that recast the first step -- concept extraction problem -- as a special case of dictionary learning, and we formalize the second step -- concept importance estimation -- as a more general form of attribution method.This framework offers several advantages as it allows us: (i) to propose new evaluation metrics for comparing different concept extraction approaches; (ii) to leverage modern attribution methods and evaluation metrics to extend and systematically evaluate state-of-the-art concept-based approaches and importance estimation techniques; (iii)  to derive theoretical guarantees regarding the optimality of such methods. We further leverage our framework to try to tackle a crucial question in explainability: how to efficiently identify clusters of data points that are classified based on a similar shared strategy.To illustrate these findings and to highlight the main strategies of a model, we introduce a visual representation called the strategic cluster graph. Finally, we present Lens, a dedicated website that offers a complete compilation of these visualizations for all classes of the ImageNet dataset.

        ----

        ## [2391] Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

        **Authors**: *Fenja Falta, Christoph Großbröhmer, Alessa Hering, Alexander Bigalke, Mattias P. Heinrich*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        A popular benchmark for intra-patient lung registration is provided by the DIR-LAB COPDgene dataset consisting of large-motion in- and expiratory breath-hold CT pairs. This dataset alone, however, does not provide enough samples to properly train state-of-the-art deep learning methods. Other public datasets often also provide only small sample sizes or include primarily small motions between scans that do not translate well to larger deformations. For point-based geometric registration, the PVT1010 dataset provides a large number of vessel point clouds without any correspondences and a labeled test set corresponding to the COPDgene cases. However, the absence of correspondences for supervision complicates training, and a fair comparison with image-based algorithms is infeasible, since CT scans for the training data are not publicly available.We here provide a combined benchmark for image- and point-based registration approaches. We curated a total of 248 public multi-centric in- and expiratory lung CT scans from 124 patients, which show large motion between scans, processed them to ensure sufficient homogeneity between the data and generated vessel point clouds that are well distributed even deeper inside the lungs. For supervised training, we provide vein and artery segmentations of the vessels and multiple thousand image-derived keypoint correspondences for each pair. For validation, we provide multiple scan pairs with manual landmark annotations. Finally, as first baselines on our new benchmark, we evaluate several image and point cloud registration methods on the dataset.

        ----

        ## [2392] An Iterative Self-Learning Framework for Medical Domain Generalization

        **Authors**: *Zhenbang Wu, Huaxiu Yao, David M. Liebovitz, Jimeng Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac0035c349f3fe8af6a93fe44697b5bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac0035c349f3fe8af6a93fe44697b5bd-Abstract-Conference.html)

        **Abstract**:

        Deep learning models have been widely used to assist doctors with clinical decision-making. However, these models often encounter a significant performance drop when applied to data that differs from the distribution they were trained on. This challenge is known as the domain shift problem. Existing domain generalization algorithms attempt to address this problem by assuming the availability of domain IDs and training a single model to handle all domains. However, in healthcare settings, patients can be classified into numerous latent domains, where the actual domain categorizations are unknown. Furthermore, each patient domain exhibits distinct clinical characteristics, making it sub-optimal to train a single model for all domains. To overcome these limitations, we propose SLGD, a self-learning framework that iteratively discovers decoupled domains and trains personalized classifiers for each decoupled domain. We evaluate the generalizability of SLGD across spatial and temporal data distribution shifts on two real-world public EHR datasets: eICU and MIMIC-IV. Our results show that SLGD achieves up to 11% improvement in the AUPRC score over the best baseline.

        ----

        ## [2393] A benchmark of categorical encoders for binary classification

        **Authors**: *Federico Matteucci, Vadim Arzamasov, Klemens Böhm*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac01e21bb14609416760f790dd8966ae-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Categorical encoders transform categorical features into numerical representations that are indispensable for a wide range of machine learning models.Existing encoder benchmark studies lack generalizability because of their limited choice of (1) encoders, (2) experimental factors, and (3) datasets. Additionally, inconsistencies arise from the adoption of varying aggregation strategies.This paper is the most comprehensive benchmark of categorical encoders to date, including an extensive evaluation of 32 configurations of encoders from diverse families, with 36 combinations of experimental factors, and on 50 datasets.The study shows the profound influence of dataset selection, experimental factors, and aggregation strategies on the benchmark's conclusions~---~aspects disregarded in previous encoder benchmarks.Our code is available at \url{https://github.com/DrCohomology/EncoderBenchmarking}.

        ----

        ## [2394] Structure of universal formulas

        **Authors**: *Dmitry Yarotsky*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac04e54e0a2d1927d60709019e4e7870-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac04e54e0a2d1927d60709019e4e7870-Abstract-Conference.html)

        **Abstract**:

        By universal formulas we understand parameterized analytic expressions that have a fixed complexity, but nevertheless can approximate any continuous function on a compact set. There exist various examples of such formulas, including some in the form of neural networks. In this paper we analyze the essential structural elements of these highly expressive models. We introduce a hierarchy of expressiveness classes connecting the global approximability property to the weaker property of infinite VC dimension, and prove a series of classification results for several increasingly complex functional families. In particular, we introduce a general family of polynomially-exponentially-algebraic functions that, as we prove, is subject to polynomial constraints. As a consequence, we show that fixed-size neural networks with not more than one layer of neurons having transcendental activations (e.g., sine or standard sigmoid) cannot in general approximate functions on arbitrary finite sets. On the other hand, we give examples of functional families, including two-hidden-layer neural networks, that approximate functions on arbitrary finite sets, but fail to do that  on the whole domain of definition.

        ----

        ## [2395] Model-enhanced Vector Index

        **Authors**: *Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang, Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding, Xupeng Miao, Haonan Wang, Bochen Pang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Qi Zhang, Fan Yang, Xing Xie, Mao Yang, Bin Cui*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac112e8ffc4e5b9ece32070440a8ca43-Abstract-Conference.html)

        **Abstract**:

        Embedding-based retrieval methods construct vector indices to search for document representations that are most similar to the query representations. They are widely used in document retrieval due to low latency and decent recall performance. Recent research indicates that deep retrieval solutions offer better model quality, but are hindered by unacceptable serving latency and the inability to support document updates. In this paper, we aim to enhance the vector index with end-to-end deep generative models, leveraging the differentiable advantages of deep retrieval models while maintaining desirable serving efficiency. We propose Model-enhanced Vector Index (MEVI), a differentiable model-enhanced index empowered by a twin-tower representation model. MEVI leverages a Residual Quantization (RQ) codebook to bridge the sequence-to-sequence deep retrieval and embedding-based models. To substantially reduce the inference time, instead of decoding the unique document ids in long sequential steps, we first generate some semantic virtual cluster ids of candidate documents in a small number of steps, and then leverage the well-adapted embedding vectors to further perform a fine-grained search for the relevant documents in the candidate virtual clusters. We empirically show that our model achieves better performance on the commonly used academic benchmarks MSMARCO Passage and Natural Questions, with comparable serving latency to dense retrieval solutions.

        ----

        ## [2396] Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models

        **Authors**: *Tianxiang Gao, Xiaokai Huo, Hailiang Liu, Hongyang Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac24656b0b5f543b202f748d62041637-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac24656b0b5f543b202f748d62041637-Abstract-Conference.html)

        **Abstract**:

        Neural networks with wide layers have attracted significant attention due to their equivalence to Gaussian processes, enabling perfect fitting of training data while maintaining generalization performance, known as benign overfitting. However, existing results mainly focus on shallow or finite-depth networks, necessitating a comprehensive analysis of wide neural networks with infinite-depth layers, such as neural ordinary differential equations (ODEs) and deep equilibrium models (DEQs). In this paper, we specifically investigate the deep equilibrium model (DEQ), an infinite-depth neural network with shared weight matrices across layers. Our analysis reveals that as the width of DEQ layers approaches infinity, it converges to a Gaussian process, establishing what is known as the Neural Network and Gaussian Process (NNGP) correspondence. Remarkably, this convergence holds even when the limits of depth and width are interchanged, which is not observed in typical infinite-depth Multilayer Perceptron (MLP) networks. Furthermore, we demonstrate that the associated Gaussian vector remains non-degenerate for any pairwise distinct input data, ensuring a strictly positive smallest eigenvalue of the corresponding kernel matrix using the NNGP kernel. These findings serve as fundamental elements for studying the training and generalization of DEQs, laying the groundwork for future research in this area.

        ----

        ## [2397] Tree Variational Autoencoders

        **Authors**: *Laura Manduchi, Moritz Vandenhirtz, Alain Ryser, Julia E. Vogt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac58b418745b3e5f10c80110c963969f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac58b418745b3e5f10c80110c963969f-Abstract-Conference.html)

        **Abstract**:

        We propose Tree Variational Autoencoder (TreeVAE), a new generative hierarchical clustering model  that learns a flexible tree-based posterior distribution over latent variables. TreeVAE hierarchically divides samples according to their intrinsic characteristics, shedding light on hidden structures in the data. It adapts its architecture to discover the optimal tree for encoding dependencies between latent variables. The proposed tree-based generative architecture enables lightweight conditional inference and improves generative performance by utilizing specialized leaf decoders.   We show that TreeVAE uncovers underlying clusters in the data and finds meaningful hierarchical relations between the different groups on a variety of datasets, including real-world imaging data.   We present empirically that TreeVAE provides a more competitive log-likelihood lower bound than the sequential counterparts.   Finally, due to its generative nature, TreeVAE is able to generate new samples from the discovered clusters via conditional sampling.

        ----

        ## [2398] Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes

        **Authors**: *Yihong Sun, Bharath Hariharan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac5c594dedf66affb098c39a3bcfdb3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac5c594dedf66affb098c39a3bcfdb3d-Abstract-Conference.html)

        **Abstract**:

        Unsupervised monocular depth estimation techniques have demonstrated encouraging results but typically assume that the scene is static. These techniques suffer when trained on dynamical scenes, where apparent object motion can equally be explained by hypothesizing the object's independent motion, or by altering its depth. This ambiguity causes depth estimators to predict erroneous depth for moving objects. To resolve this issue, we introduce Dynamo-Depth, an unifying approach that disambiguates dynamical motion by jointly learning monocular depth, 3D independent flow field, and motion segmentation from unlabeled monocular videos. Specifically, we offer our key insight that a good initial estimation of motion segmentation is sufficient for jointly learning depth and independent motion despite the fundamental underlying ambiguity. Our proposed method achieves state-of-the-art performance on monocular depth estimation on Waymo Open and nuScenes Dataset with significant improvement in the depth of moving objects. Code and additional results are available at https://dynamo-depth.github.io.

        ----

        ## [2399] LIMA: Less Is More for Alignment

        **Authors**: *Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)

        **Abstract**:

        Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history.Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data.In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43\% of cases; this statistic is as high as 58\% when compared to Bard and 65\% versus DaVinci003, which was trained with human feedback.Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.

        ----

        

[Go to the previous page](NIPS-2023-list11.md)

[Go to the next page](NIPS-2023-list13.md)

[Go to the catalog section](README.md)