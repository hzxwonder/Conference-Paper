## [0] Emergent Communication in Interactive Sketch Question Answering

**Authors**: *Zixing Lei, Yiming Zhang, Yuxin Xiong, Siheng Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/746cf1bc2337700f7f0c35c7b02638cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/746cf1bc2337700f7f0c35c7b02638cc-Abstract-Conference.html)

**Abstract**:

Vision-based emergent communication (EC) aims to learn to communicate through sketches and demystify the evolution of human communication. Ironically, previous works neglect multi-round interaction, which is indispensable in human communication. To fill this gap, we first introduce a novel Interactive Sketch Question Answering (ISQA) task, where two collaborative players are interacting through sketches to answer a question about an image. To accomplish this task, we design a new and efficient interactive EC system, which can achieve an effective balance among three evaluation factors, including the question answering accuracy, drawing complexity and human interpretability. Our experimental results demonstrate that multi-round interactive mechanism facilitates tar- geted and efficient communication between intelligent agents. The code will be released.

----

## [0] Computing Approximate 𝓁p Sensitivities

**Authors**: *Swati Padmanabhan, David P. Woodruff, Richard Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/746d2254f6892f3badadb07cc9c0f0da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/746d2254f6892f3badadb07cc9c0f0da-Abstract-Conference.html)

**Abstract**:

Recent works in dimensionality reduction for regression tasks have introduced the notion of sensitivity, an estimate of the importance of a specific datapoint in a dataset, offering provable guarantees on the quality of the approximation after removing low-sensitivity datapoints via subsampling. However, fast algorithms for approximating sensitivities, which we show is equivalent to approximate regression, are known for only the $\ell_2$ setting, in which they are popularly termed leverage scores. In this work, we provide the first efficient algorithms for approximating $\ell_p$ sensitivities and other summary statistics of a given matrix. In particular, for a given $n \times d$ matrix, we compute $\alpha$-approximation to its $\ell_1$ sensitivities at the cost of $n/\alpha$ sensitivity computations. For estimating the total $\ell_p$ sensitivity (i.e. the sum of $\ell_p$ sensitivities), we provide an algorithm based on importance sampling of $\ell_p$ Lewis weights, which computes a constant factor approximation at the cost of roughly $\sqrt{d}$ sensitivity computations, with no polynomial dependence on $n$. Furthermore, we estimate the maximum $\ell_1$ sensitivity up to a $\sqrt{d}$ factor in $O(d)$ sensitivity computations. We also generalize these results to $\ell_p$ norms.  Lastly, we experimentally show that for a wide class of structured matrices in real-world datasets, the total sensitivity can be quickly approximated and is significantly smaller than the theoretical prediction, demonstrating that real-world datasets have on average low intrinsic effective dimensionality.

----

## [0] Automated Classification of Model Errors on ImageNet

**Authors**: *Momchil Peychev, Mark Niklas Müller, Marc Fischer, Martin T. Vechev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7480ed13740773505262791131c12b89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7480ed13740773505262791131c12b89-Abstract-Conference.html)

**Abstract**:

While the ImageNet dataset has been driving computer vision research over the past decade, significant label noise and ambiguity have made top-1 accuracy an insufficient measure of further progress. To address this, new label-sets and evaluation protocols have been proposed for ImageNet showing that state-of-the-art models already achieve over 95% accuracy and shifting the focus on investigating why the remaining errors persist.Recent work in this direction employed a panel of experts to manually categorize all remaining classification errors for two selected models. However, this process is time-consuming, prone to inconsistencies, and requires trained experts, making it unsuitable for regular model evaluation thus limiting its utility. To overcome these limitations, we propose the first automated error classification framework, a valuable tool to study how modeling choices affect error distributions. We use our framework to comprehensively evaluate the error distribution of over 900 models. Perhaps surprisingly, we find that across model architectures, scales, and pre-training corpora, top-1 accuracy is a strong predictor for the portion of all error types. In particular, we observe that the portion of severe errors drops significantly with top-1 accuracy indicating that, while it underreports a model's true performance, it remains a valuable performance metric.We release all our code at https://github.com/eth-sri/automated-error-analysis.

----

## [0] Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent

**Authors**: *Jihao Andreas Lin, Javier Antorán, Shreyas Padhy, David Janz, José Miguel Hernández-Lobato, Alexander Terenin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7482e8ce4139df1a2d8195a0746fa713-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7482e8ce4139df1a2d8195a0746fa713-Abstract-Conference.html)

**Abstract**:

Gaussian processes are a powerful framework for quantifying uncertainty and for sequential decision-making but are limited by the requirement of solving linear systems. In general, this has a cubic cost in dataset size and is sensitive to conditioning. We explore stochastic gradient algorithms as a computationally efficient method of approximately solving these linear systems: we develop low-variance optimization objectives for sampling from the posterior and extend these to inducing points. Counterintuitively, stochastic gradient descent often produces accurate predictions, even in cases where it does not converge quickly to the optimum. We explain this through a spectral characterization of the implicit bias from non-convergence. We show that stochastic gradient descent produces predictive distributions close to the true posterior both in regions with sufficient data coverage, and in regions sufficiently far away from the data. Experimentally, stochastic gradient descent achieves state-of-the-art performance on sufficiently large-scale or ill-conditioned regression tasks. Its uncertainty estimates match the performance of significantly more expensive baselines on a large-scale Bayesian~optimization~task.

----

## [0] Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning

**Authors**: *Yihua Zhang, Yimeng Zhang, Aochuan Chen, Jinghan Jia, Jiancheng Liu, Gaowen Liu, Mingyi Hong, Shiyu Chang, Sijia Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html)

**Abstract**:

Massive data is often considered essential for deep learning applications, but it also incurs significant computational and infrastructural costs. Therefore, dataset pruning (DP) has emerged as an effective way to improve data efficiency by identifying and removing redundant training samples without sacrificing performance. In this work, we aim to address the problem of DP for transfer learning, i.e., how to prune a source dataset for improved pretraining efficiency and lossless finetuning accuracy on downstream target tasks. To our best knowledge, the problem of DP for transfer learning remains open, as previous studies have primarily addressed DP and transfer learning as separate problems. By contrast, we establish a unified viewpoint to integrate DP with transfer learning and find that existing DP methods are not suitable for the transfer learning paradigm. We then propose two new DP methods, label mapping and feature mapping, for supervised and self-supervised pretraining settings respectively, by revisiting the DP problem through the lens of source-target domain mapping. Furthermore, we demonstrate the effectiveness of our approach on numerous transfer learning tasks. We show that source data classes can be pruned by up to $40\%\sim 80\%$ without sacrificing the downstream performance, resulting in a significant $2\sim 5\times$ speed-up during the pretraining stage. Besides, our proposal exhibits broad applicability and can improve other computationally intensive transfer learning techniques, such as adversarial pretraining.

----

## [0] On Slicing Optimality for Mutual Information

**Authors**: *Ammar Fayad, Majd Ibrahim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/749b64078a64fa5734a49fb40bc9fd65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/749b64078a64fa5734a49fb40bc9fd65-Abstract-Conference.html)

**Abstract**:

Measuring dependence between two random variables is of great importance in various domains but is difficult to compute in today's complex environments with high-dimensional data. Recently, slicing methods have shown to be a scalable approach to measuring mutual information (MI) between high-dimensional variables by projecting these variables into one-dimensional spaces. Unfortunately, these methods use uniform distributions of slicing directions, which generally discard informative features between variables and thus lead to inaccurate quantification of dependence. In this paper, we propose a principled framework that searches for an \textit{optimal} distribution of slices for MI. Importantly, we answer theoretical questions about finding the optimal slicing distribution in the context of MI and develop corresponding theoretical analyses. We also develop a practical algorithm, connecting our theoretical results with modern machine learning frameworks. Through comprehensive experiments in benchmark domains, we demonstrate significant gains in our information measure than state-of-the-art baselines.

----

## [0] OpenIllumination: A Multi-Illumination Dataset for Inverse Rendering Evaluation on Real Objects

**Authors**: *Isabella Liu, Linghao Chen, Ziyang Fu, Liwen Wu, Haian Jin, Zhong Li, Chin Ming Ryan Wong, Yi Xu, Ravi Ramamoorthi, Zexiang Xu, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce OpenIllumination, a real-world dataset containing over 108K images of 64 objects with diverse materials, captured under 72 camera views and a large number of different illuminations. For each image in the dataset, we provide accurate camera parameters, illumination ground truth, and foreground segmentation masks. Our dataset enables the quantitative evaluation of most inverse rendering and material decomposition methods for real objects. We examine several state-of-the-art inverse rendering methods on our dataset and compare their performances. The dataset and code can be found on the project page: https://oppo-us-research.github.io/OpenIllumination.

----

## [0] Language Model Tokenizers Introduce Unfairness Between Languages

**Authors**: *Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html)

**Abstract**:

Recent language models have shown impressive multilingual performance, even when not explicitly trained for it.Despite this, there are concerns about the quality of their outputs across different languages.In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked.The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases.These disparities persist even for tokenizers that are intentionally trained for multilingual support.Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs.This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models.Therefore, we make the case that we should train future language models using multilingually fair subword tokenizers.

----

## [0] Interaction Measures, Partition Lattices and Kernel Tests for High-Order Interactions

**Authors**: *Zhaolu Liu, Robert L. Peach, Pedro A. M. Mediano, Mauricio Barahona*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74f11936d6144eae43730e1a49365479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74f11936d6144eae43730e1a49365479-Abstract-Conference.html)

**Abstract**:

Models that rely solely on pairwise relationships often fail to capture the complete statistical structure of the complex multivariate data found in diverse domains, such as socio-economic, ecological, or biomedical systems. Non-trivial dependencies between groups of more than two variables can play a significant role in the analysis and modelling of such systems, yet extracting such high-order interactions from data remains challenging. Here, we introduce a hierarchy of $d$-order ($d \geq 2$) interaction measures, increasingly inclusive of possible factorisations of the joint probability distribution, and define non-parametric, kernel-based tests to establish systematically the statistical significance of $d$-order interactions. We also establish mathematical links with lattice theory, which elucidate the derivation of the interaction measures and their composite permutation tests; clarify the connection of simplicial complexes with kernel matrix centring; and provide a means to enhance computational efficiency. We illustrate our results numerically with validations on synthetic data, and through an application to neuroimaging data.

----

## [0] Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?

**Authors**: *Haitao Mao, Zhikai Chen, Wei Jin, Haoyu Han, Yao Ma, Tong Zhao, Neil Shah, Jiliang Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74f1edadbdf495e7258ee8db7b1d3acd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74f1edadbdf495e7258ee8db7b1d3acd-Abstract-Conference.html)

**Abstract**:

Recent studies on Graph Neural Networks(GNNs) provide both empirical and theoretical evidence supporting their effectiveness in capturing structural patterns on both homophilic and certain heterophilic graphs. Notably, most real-world homophilic and heterophilic graphs are comprised of a mixture of nodes in both homophilic and heterophilic structural patterns, exhibiting a structural disparity. However, the analysis of GNN performance with respect to nodes exhibiting different structural patterns, e.g., homophilic nodes in heterophilic graphs, remains rather limited. In the present study, we provide evidence that Graph Neural Networks(GNNs) on node classification typically perform admirably on homophilic nodes within homophilic graphs and heterophilic nodes within heterophilic graphs while struggling on the opposite node set, exhibiting a performance disparity. We theoretically and empirically identify effects of GNNs on testing nodes exhibiting distinct structural patterns. We then propose a rigorous, non-i.i.d PAC-Bayesian generalization bound for GNNs, revealing reasons for the performance disparity, namely the aggregated feature distance and homophily ratio difference between training and testing nodes. Furthermore, we demonstrate the practical implications of our new findings via (1) elucidating the effectiveness of deeper GNNs; and (2) revealing an over-looked distribution shift factor on graph out-of-distribution problem and proposing a new scenario accordingly.

----

## [0] Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment

**Authors**: *Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html)

**Abstract**:

Spatio-Temporal Graph (STG) forecasting is a fundamental task in many real-world applications. Spatio-Temporal Graph Neural Networks have emerged as the most popular method for STG forecasting, but they often struggle with temporal out-of-distribution (OoD) issues and dynamic spatial causation. In this paper, we propose a novel framework called CaST to tackle these two challenges via causal treatments. Concretely, leveraging a causal lens, we first build a structural causal model to decipher the data generation process of STGs. To handle the temporal OoD issue, we employ the back-door adjustment by a novel disentanglement block to separate the temporal environments from input data. Moreover, we utilize the front-door adjustment and adopt edge-level convolution to model the ripple effect of causation. Experiments results on three real-world datasets demonstrate the effectiveness of CaST, which consistently outperforms existing methods with good interpretability. Our source code is available at https://github.com/yutong-xia/CaST.

----

## [0] Greedy Poisson Rejection Sampling

**Authors**: *Gergely Flamich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fb3d526c7d8bd0c3e4b71704bb5abf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fb3d526c7d8bd0c3e4b71704bb5abf-Abstract-Conference.html)

**Abstract**:

One-shot channel simulation is a fundamental data compression problem concerned with encoding a single sample from a target distribution $Q$ using a coding distribution $P$ using as few bits as possible on average. Algorithms that solve this problem find applications in neural data compression and differential privacy and can serve as a more efficient and natural alternative to quantization-based methods. Unfortunately, existing solutions are too slow or have limited applicability, preventing their widespread adaptation. In this paper, we conclusively solve one-shot channel simulation for one-dimensional problems where the target-proposal density ratio is unimodal by describing an algorithm with optimal runtime. We achieve this by constructing a rejection sampling procedure equivalent to greedily searching over the points of a Poisson process. Hence, we call our algorithm greedy Poisson rejection sampling (GPRS) and analyze the correctness and time complexity of several of its variants. Finally, we empirically verify our theorems, demonstrating that GPRS significantly outperforms the current state-of-the-art method, A* coding.

----

## [0] Uncertainty Quantification via Neural Posterior Principal Components

**Authors**: *Elias Nehme, Omer Yair, Tomer Michaeli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification is crucial for the deployment of image restoration models in safety-critical domains, like autonomous driving and biological imaging. To date, methods for uncertainty visualization have mainly focused on per-pixel estimates. Yet, a heatmap of per-pixel variances is typically of little practical use, as it does not capture the strong correlations between pixels. A more natural measure of uncertainty corresponds to the variances along the principal components (PCs) of the posterior distribution. Theoretically, the PCs can be computed by applying PCA on samples generated from a conditional generative model for the input image. However, this requires generating a very large number of samples at test time, which is painfully slow with the current state-of-the-art (diffusion) models. In this work, we present a method for predicting the PCs of the posterior distribution for any input image, in a single forward pass of a neural network. Our method can either wrap around a pre-trained model that was trained to minimize the mean square error (MSE), or can be trained from scratch to output both a predicted image and the posterior PCs. We showcase our method on multiple inverse problems in imaging, including denoising, inpainting, super-resolution, and biological image-to-image translation. Our method reliably conveys instance-adaptive uncertainty directions, achieving uncertainty quantification comparable with posterior samplers while being orders of magnitude faster. Code and examples are available on our webpage.

----

## [0] Deep Reinforcement Learning with Plasticity Injection

**Authors**: *Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75101364dc3aa7772d27528ea504472b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75101364dc3aa7772d27528ea504472b-Abstract-Conference.html)

**Abstract**:

A growing body of evidence suggests that neural networks employed in deep reinforcement learning (RL) gradually lose their plasticity, the ability to learn from new data; however, the analysis and mitigation of this phenomenon is hampered by the complex relationship between plasticity, exploration, and performance in RL. This paper introduces plasticity injection, a minimalistic intervention that increases the network plasticity without changing the number of trainable parameters or biasing the predictions. The applications of this intervention are two-fold: first, as a diagnostic tool â€” if injection increases the performance, we may conclude that an agent's network was losing its plasticity. This tool allows us to identify a subset of Atari environments where the lack of plasticity causes performance plateaus, motivating future studies on understanding and combating plasticity loss. Second, plasticity injection can be used to improve the computational efficiency of RL training if the agent has to re-learn from scratch due to exhausted plasticity or by growing the agent's network dynamically without compromising performance. The results on Atari show that plasticity injection attains stronger performance compared to alternative methods while being computationally efficient.

----

## [0] StreamNet: Memory-Efficient Streaming Tiny Deep Learning Inference on the Microcontroller

**Authors**: *Hong-Sheng Zheng, Yu-Yuan Liu, Chen-Fong Hsu, Tsung Tai Yeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7526508f11bbe0a123af62b9dab1fbe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7526508f11bbe0a123af62b9dab1fbe1-Abstract-Conference.html)

**Abstract**:

With the emerging Tiny Machine Learning (TinyML) inference applications, there is a growing interest when deploying TinyML models on the low-power Microcontroller Unit (MCU). However, deploying TinyML models on MCUs reveals several challenges due to the MCU’s resource constraints, such as small flash memory, tight SRAM memory budget, and slow CPU performance. Unlike typical layer-wise inference, patch-based inference reduces the peak usage of SRAM memory on MCUs by saving small patches rather than the entire tensor in the SRAM memory. However, the processing of patch-based inference tremendously increases the amount of MACs against the layer-wise method. Thus, this notoriously computational overhead makes patch-based inference undesirable on MCUs. This work designs StreamNet that employs the stream buffer to eliminate the redundant computation of patch-based inference. StreamNet uses 1D and 2D streaming processing and provides an parameter selection algorithm that automatically improve the performance of patch-based inference with minimal requirements on the MCU’s SRAM memory space. In 10 TinyML models, StreamNet-2D achieves a geometric mean of 7.3X speedup and saves 81\% of MACs over the state-of-the-art patch-based inference.

----

## [0] Estimating and Controlling for Equalized Odds via Sensitive Attribute Predictors

**Authors**: *Beepul Bharti, Paul H. Yi, Jeremias Sulam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/752820c79b4ebb72809014bdfdedd603-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/752820c79b4ebb72809014bdfdedd603-Abstract-Conference.html)

**Abstract**:

As the use of machine learning models in real world high-stakes decision settings continues to grow, it is highly important that we are able to audit and control for any potential fairness violations these models may exhibit towards certain groups. To do so, one naturally requires access to sensitive attributes, such as demographics, biological sex, or other potentially sensitive features that determine group membership. Unfortunately, in many settings, this information is often unavailable. In this work we study the well known equalized odds (EOD) definition of fairness. In a setting without sensitive attributes, we first provide tight and computable upper bounds for the EOD violation of a predictor. These bounds precisely reflect the worst possible EOD violation. Second, we demonstrate how one can provably control the worst-case EOD by a new post-processing correction method. Our results characterize when directly controlling for EOD with respect to the predicted sensitive attributes is -- and when is not -- optimal when it comes to controlling worst-case EOD. Our results hold under assumptions that are milder than previous works, and we illustrate these results with experiments on synthetic and real datasets.

----

## [0] Segment Any Point Cloud Sequences by Distilling Vision Foundation Models

**Authors**: *Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html)

**Abstract**:

Recent advancements in vision foundation models (VFMs) have opened up new possibilities for versatile and efficient visual perception. In this work, we introduce Seal, a novel framework that harnesses VFMs for segmenting diverse automotive point cloud sequences. Seal exhibits three appealing properties: i) Scalability: VFMs are directly distilled into point clouds, obviating the need for annotations in either 2D or 3D during pretraining. ii) Consistency: Spatial and temporal relationships are enforced at both the camera-to-LiDAR and point-to-segment regularization stages, facilitating cross-modal representation learning. iii) Generalizability: Seal enables knowledge transfer in an off-the-shelf manner to downstream tasks involving diverse point clouds, including those from real/synthetic, low/high-resolution, large/small-scale, and clean/corrupted datasets. Extensive experiments conducted on eleven different point cloud datasets showcase the effectiveness and superiority of Seal. Notably, Seal achieves a remarkable 45.0% mIoU on nuScenes after linear probing, surpassing random initialization by 36.9% mIoU and outperforming prior arts by 6.1% mIoU. Moreover, Seal demonstrates significant performance gains over existing methods across 20 different few-shot fine-tuning tasks on all eleven tested point cloud datasets. The code is available at this link.

----

## [0] Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings

**Authors**: *Giovanni de Felice, John Yannis Goulermas, Vladimir V. Gusev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html)

**Abstract**:

Kernel design is a pivotal but challenging aspect of time series analysis, especially in the context of small datasets. In recent years, Reservoir Computing (RC) has emerged as a powerful tool to compare time series based on the underlying dynamics of the generating process rather than the observed data. However, the performance of RC highly depends on the hyperparameter setting, which is hard to interpret and costly to optimize because of the recurrent nature of RC. Here, we present a new kernel for time series based on the recently established equivalence between reservoir dynamics and Nonlinear Vector AutoRegressive (NVAR) processes. The kernel is non-recurrent and depends on a small set of meaningful hyperparameters, for which we suggest an effective heuristic. We demonstrate excellent performance on a wide range of real-world classification tasks, both in terms of accuracy and speed. This further advances the understanding of RC representation learning models and extends the typical use of the NVAR framework to kernel design and representation of real-world time series data.

----

## [0] SPA: A Graph Spectral Alignment Perspective for Domain Adaptation

**Authors**: *Zhiqing Xiao, Haobo Wang, Ying Jin, Lei Feng, Gang Chen, Fei Huang, Junbo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/754e80f98b2a141942f45a0eeb258a3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/754e80f98b2a141942f45a0eeb258a3c-Abstract-Conference.html)

**Abstract**:

Unsupervised domain adaptation (UDA) is a pivotal form in machine learning to extend the in-domain model to the distinctive target domains where the data distributions differ. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. In this work, we introduce a novel graph SPectral Alignment (SPA) framework to tackle the tradeoff. The core of our method is briefly condensed as follows: (i)-by casting the DA problem to graph primitives, SPA composes a coarse graph alignment mechanism with a novel spectral regularizer towards aligning the domain graphs in eigenspaces; (ii)-we further develop a fine-grained message propagation module --- upon a novel neighbor-aware self-training mechanism --- in order for enhanced discriminability in the target domain. On standardized benchmarks, the extensive experiments of SPA demonstrate that its performance has surpassed the existing cutting-edge DA methods. Coupled with dense model analysis, we conclude that our approach indeed possesses superior efficacy, robustness, discriminability, and transferability. Code and data are available at: https://github.com/CrownX/SPA.

----

## [0] CosNet: A Generalized Spectral Kernel Network

**Authors**: *Yanfang Xue, Pengfei Fang, Jinyue Tian, Shipeng Zhu, Hui Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html)

**Abstract**:

Complex-valued representation exists inherently in the time-sequential data that can be derived from the integration of harmonic waves. The non-stationary spectral kernel, realizing a complex-valued feature mapping, has shown its potential to analyze the time-varying statistical characteristics of the time-sequential data, as a result of the modeling frequency parameters. However, most existing spectral kernel-based methods eliminate the imaginary part, thereby limiting the representation power of the spectral kernel. To tackle this issue, we propose a generalized spectral kernel network, namely, \underline{Co}mplex-valued \underline{s}pectral kernel \underline{Net}work (CosNet), which includes spectral kernel mapping generalization (SKMG) module and complex-valued spectral kernel embedding (CSKE) module. Concretely, the SKMG module is devised to generalize the spectral kernel mapping in the real number domain to the complex number domain, recovering the inherent complex-valued representation for the real-valued data. Then a following CSKE module is further developed to combine the complex-valued spectral kernels and neural networks to effectively capture long-range or periodic relations of the data. Along with the CosNet, we study the effect of the complex-valued spectral kernel mapping via theoretically analyzing the bound of covering number and generalization error. Extensive experiments demonstrate that CosNet performs better than the mainstream kernel methods and complex-valued neural networks.

----

## [0] A Theory of Unsupervised Translation Motivated by Understanding Animal Communication

**Authors**: *Shafi Goldwasser, David F. Gruber, Adam Tauman Kalai, Orr Paradise*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7571c9d44179c7988178593c5b62a9b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7571c9d44179c7988178593c5b62a9b6-Abstract-Conference.html)

**Abstract**:

Neural networks are capable of translating between languagesâ€”in some cases even between two languages where there is little or no access to parallel translations, in what is known as Unsupervised Machine Translation (UMT). Given this progress, it is intriguing to ask whether machine learning tools can ultimately enable understanding animal communication, particularly that of highly intelligentanimals. We propose a theoretical framework for analyzing UMT when no parallel translations are available and when it cannot be assumed that the source and target corpora address related subject domains or posses similar linguistic structure. Weexemplify this theory with two stylized models of language, for which our framework provides bounds on necessary sample complexity; the bounds are formally proven and experimentally verified on synthetic data. These bounds show that the error rates are inversely related to the language complexity and amount of common ground. This suggests that unsupervised translation of animal communication may be feasible if the communication system is sufficiently complex.

----

## [0] Adversarial Self-Training Improves Robustness and Generalization for Gradual Domain Adaptation

**Authors**: *Lianghe Shi, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75b0edb869e2cd509d64d0e8ff446bc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75b0edb869e2cd509d64d0e8ff446bc1-Abstract-Conference.html)

**Abstract**:

Gradual Domain Adaptation (GDA), in which the learner is provided with additional intermediate domains, has been theoretically and empirically studied in many contexts. Despite its vital role in security-critical scenarios, the adversarial robustness of the GDA model remains unexplored. In this paper, we adopt the effective gradual self-training method and replace vanilla self-training with adversarial self-training (AST). AST first predicts labels on the unlabeled data and then adversarially trains the model on the pseudo-labeled distribution. Intriguingly, we find that gradual AST improves not only adversarial accuracy but also clean accuracy on the target domain. We reveal that this is because adversarial training (AT) performs better than standard training when the pseudo-labels contain a portion of incorrect labels. Accordingly, we first present the generalization error bounds for gradual AST in a multiclass classification setting. We then use the optimal value of the Subset Sum Problem to bridge the standard error on a real distribution and the adversarial error on a pseudo-labeled distribution. The result indicates that AT may obtain a tighter bound than standard training on data with incorrect pseudo-labels. We further present an example of a conditional Gaussian distribution to provide more insights into why gradual AST can improve the clean accuracy for GDA.

----

## [0] TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials

**Authors**: *Guillem Simeon, Gianni De Fabritiis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75c2ec5f98d7b2f50ad68033d2c07086-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75c2ec5f98d7b2f50ad68033d2c07086-Abstract-Conference.html)

**Abstract**:

The development of efficient machine learning models for molecular systems representation is becoming crucial in scientific research. We introduce TensorNet, an innovative O(3)-equivariant message-passing neural network architecture that leverages Cartesian tensor representations. By using Cartesian tensor atomic embeddings, feature mixing is simplified through matrix product operations. Furthermore, the cost-effective decomposition of these tensors into rotation group irreducible representations allows for the separate processing of scalars, vectors, and tensors when necessary. Compared to higher-rank spherical tensor models, TensorNet demonstrates state-of-the-art performance with significantly fewer parameters. For small molecule potential energies, this can be achieved even with a single interaction layer. As a result of all these properties, the model's computational cost is substantially decreased. Moreover, the accurate prediction of vector and tensor molecular quantities on top of potential energies and forces is possible. In summary, TensorNet's framework opens up a new space for the design of state-of-the-art equivariant models.

----

## [0] Multi-Player Zero-Sum Markov Games with Networked Separable Interactions

**Authors**: *Chanwoo Park, Kaiqing Zhang, Asuman E. Ozdaglar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75c411b0a06fa9e78f2a516b57b2ce62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75c411b0a06fa9e78f2a516b57b2ce62-Abstract-Conference.html)

**Abstract**:

We study a new class of  Markov games, \textit{(multi-player) zero-sum Markov Games} with {\it Networked separable interactions} (zero-sum NMGs), to model the local interaction structure in non-cooperative multi-agent sequential decision-making. We define a zero-sum NMG as a model where {the payoffs of the auxiliary games associated with each state are zero-sum and} have some separable (i.e., polymatrix) structure across the neighbors over some interaction network. We first identify the necessary and sufficient conditions under which an MG can be presented as a zero-sum NMG, and show that the set of Markov coarse correlated equilibrium (CCE) collapses to the set of Markov Nash equilibrium (NE) in these games, in that the {product of} per-state marginalization of the former for all players yields the latter. Furthermore,  we show that finding approximate Markov \emph{stationary}  CCE  in infinite-horizon discounted zero-sum NMGs is \texttt{PPAD}-hard, unless the underlying network has a ``star topology''. Then, we propose fictitious-play-type dynamics, the classical learning dynamics in normal-form games, for zero-sum NMGs, and establish convergence guarantees to Markov stationary NE under a star-shaped network structure. Finally, in light of the hardness result, we focus on computing a Markov \emph{non-stationary} NE and provide finite-iteration guarantees for a series of value-iteration-based algorithms. We also provide numerical experiments to corroborate our theoretical results.

----

## [0] Continuous-Time Functional Diffusion Processes

**Authors**: *Giulio Franzese, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html)

**Abstract**:

We introduce Functional Diffusion Processes (FDPs), which generalize score-based diffusion models to infinite-dimensional function spaces. FDPs require a new mathematical framework to describe the forward and backward dynamics, and several extensions to derive practical training objectives. These include infinite-dimensional versions of Girsanov theorem, in order to be able to compute an ELBO, and of the sampling theorem, in order to guarantee that functional evaluations in a countable set of points are equivalent to infinite-dimensional functions. We use FDPs to build a new breed of generative models in function spaces, which do not require specialized network architectures, and that can work with any kind of continuous data.Our results on real data show that FDPs achieve high-quality image generation, using a simple MLP architecture with orders of magnitude fewer parameters than existing diffusion models.

----

## [0] Knowledge-based in silico models and dataset for the comparative evaluation of mammography AI for a range of breast characteristics, lesion conspicuities and doses

**Authors**: *Elena Sizikova, Niloufar Saharkhiz, Diksha Sharma, Miguel A. Lago, Berkman Sahiner, Jana G. Delfino, Aldo Badano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75d0956c9594f47bfb86a07bef58d4b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/75d0956c9594f47bfb86a07bef58d4b0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

To generate evidence regarding the safety and efficacy of artificial intelligence (AI) enabled medical devices, AI models need to be evaluated on a diverse population of patient cases, some of which may not be readily available. We propose an evaluation approach for testing medical imaging AI models that relies on in silico imaging pipelines in which stochastic digital models of human anatomy (in object space) with and without pathology are imaged using a digital replica imaging acquisition system to generate realistic synthetic image datasets. Here, we release M-SYNTH, a dataset of cohorts with four breast fibroglandular density distributions imaged at different exposure levels using Monte Carlo x-ray simulations with the publicly available Virtual Imaging Clinical Trial for Regulatory Evaluation (VICTRE) toolkit. We utilize the synthetic dataset to analyze AI model performance and find that model performance decreases with increasing breast density and increases with higher mass density, as expected. As exposure levels decrease, AI model performance drops with the highest performance achieved at exposure levels lower than the nominal recommended dose for the breast type.

----

## [0] Is Distance Matrix Enough for Geometric Deep Learning?

**Authors**: *Zian Li, Xiyuan Wang, Yinan Huang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are often used for tasks involving the 3D geometry of a given graph, such as molecular dynamics simulation. While incorporating Euclidean distance into Message Passing Neural Networks (referred to as Vanilla DisGNN) is a straightforward way to learn the geometry, it has been demonstrated that Vanilla DisGNN is geometrically incomplete. In this work, we first construct families of novel and symmetric geometric graphs that Vanilla DisGNN cannot distinguish even when considering all-pair distances, which greatly expands the existing counterexample families. Our counterexamples show the inherent limitation of Vanilla DisGNN to capture symmetric geometric structures. We then propose $k$-DisGNNs, which can effectively exploit the rich geometry contained in the distance matrix. We demonstrate the high expressive power of $k$-DisGNNs from three perspectives: 1. They can learn high-order geometric information that cannot be captured by Vanilla DisGNN. 2. They can unify some existing well-designed geometric models. 3. They are universal function approximators from geometric graphs to scalars (when $k\geq 2$) and vectors (when $k\geq 3$). Most importantly, we establish a connection between geometric deep learning (GDL) and traditional graph representation learning (GRL), showing that those highly expressive GNN models originally designed for GRL can also be applied to GDL with impressive performance, and that existing complicated, equivariant models are not the only solution. Experiments verify our theory. Our $k$-DisGNNs achieve many new state-of-the-art results on MD17.

----

## [0] Optimized Covariance Design for AB Test on Social Network under Interference

**Authors**: *Qianyi Chen, Bo Li, Lu Deng, Yong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html)

**Abstract**:

Online A/B tests have become increasingly popular and important for social platforms. However, accurately estimating the global average treatment effect (GATE) has proven to be challenging due to network interference, which violates the Stable Unit Treatment Value Assumption (SUTVA) and poses great challenge to experimental design. Existing network experimental design research was mostly based on the unbiased Horvitz-Thompson (HT) estimator with substantial data trimming to ensure unbiasedness at the price of high resultant estimation variance. In this paper, we strive to balance the bias and variance in designing randomized network experiments.  Under a potential outcome model with 1-hop interference, we derive the bias and variance of the standard HT estimator and reveal their relation to the network topological structure and the covariance of the treatment assignment vector. We then propose to formulate the experimental design problem as to optimize the covariance matrix of the treatment assignment vector to achieve the bias and variance balance by minimizing the mean squared error (MSE) of the estimator. An efficient projected gradient descent algorithm is presented to the implement of the desired randomization scheme. Finally, we carry out extensive  simulation studies to demonstrate the advantages of our proposed method over other existing methods in many settings, with different levels of model misspecification.

----

## [0] AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis

**Authors**: *Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html)

**Abstract**:

Can machines recording an audio-visual scene produce realistic, matching audio-visual experiences at novel positions and novel view directions? We answer it by studying a new task---real-world audio-visual scene synthesis---and a first-of-its-kind NeRF-based approach for multimodal learning. Concretely, given a video recording of an audio-visual scene, the task is to synthesize new videos with spatial audios along arbitrary novel camera trajectories in that scene. We propose an acoustic-aware audio generation module that integrates prior knowledge of audio propagation into NeRF, in which we implicitly associate audio generation with the 3D geometry and material properties of a visual environment. Furthermore, we present a coordinate transformation module that expresses a view direction relative to the sound source, enabling the model to learn sound source-centric acoustic fields. To facilitate the study of this new task, we collect a high-quality Real-World Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method on this real-world dataset and the simulation-based SoundSpaces dataset. Notably, we refer readers to view our demo videos for convincing comparisons.

----

## [0] Is This Loss Informative? Faster Text-to-Image Customization by Tracking Objective Dynamics

**Authors**: *Anton Voronov, Mikhail Khoroshikh, Artem Babenko, Max Ryabinin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760e8857c7660fe50bac933161b14f41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760e8857c7660fe50bac933161b14f41-Abstract-Conference.html)

**Abstract**:

Text-to-image generation models represent the next step of evolution in image synthesis, offering a natural way to achieve flexible yet fine-grained control over the result.One emerging area of research is the fast adaptation of large text-to-image models to smaller datasets or new visual concepts.However, many efficient methods of adaptation have a long training time, which limits their practical applications, slows down experiments, and spends excessive GPU resources.In this work, we study the training dynamics of popular text-to-image personalization methods (such as Textual Inversion or DreamBooth), aiming to speed them up.We observe that most concepts are learned at early stages and do not improve in quality later, but standard training convergence metrics fail to indicate that.Instead, we propose a simple drop-in early stopping criterion that only requires computing the regular training objective on a fixed set of inputs for all training iterations.Our experiments on Stable Diffusion for 48 different concepts and three personalization methods demonstrate the competitive performance of our approach, which makes adaptation up to 8 times faster with no significant drops in quality.

----

## [0] DesCo: Learning Object Recognition with Rich Language Descriptions

**Authors**: *Liunian Harold Li, Zi-Yi Dou, Nanyun Peng, Kai-Wei Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/761c3284ee4859bff3c7e5d9299a45ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/761c3284ee4859bff3c7e5d9299a45ee-Abstract-Conference.html)

**Abstract**:

Recent development in vision-language approaches has instigated a paradigm shift in learning visual recognition models from language supervision. These approaches align objects with language queries (e.g. "a photo of a cat") and thus improve the models' adaptability to novel objects and domains. Recent studies have attempted to query these models with complex language expressions that include specifications of fine-grained details, such as colors, shapes, and relations. However, simply incorporating language descriptions into queries does not guarantee accurate interpretation by the models. In fact, our experiments show that GLIP, a state-of-the-art vision-language model for object detection, often disregards contextual information in the language descriptions and instead relies heavily on detecting objects solely by their names. To tackle the challenge, we propose a new description-conditioned (DesCo) paradigm of learning object recognition models with rich language descriptions consisting of two innovations: 1) we employ a large language model as a commonsense knowledge engine to generate rich language descriptions of objects; 2) we design context-sensitive queries to improve the model's ability in deciphering intricate nuances embedded within descriptions and enforce the model to focus on context rather than object names alone.  On two novel object detection benchmarks, LVIS and OminiLabel, under the zero-shot detection setting, our approach achieves 34.8 APr minival (+9.1) and 29.3 AP (+3.6), respectively, surpassing the prior state-of-the-art models, GLIP and FIBER, by a large margin.

----

## [0] On the Variance, Admissibility, and Stability of Empirical Risk Minimization

**Authors**: *Gil Kur, Eli Putterman, Alexander Rakhlin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7644353d580a9e027e0069d6480d971b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7644353d580a9e027e0069d6480d971b-Abstract-Conference.html)

**Abstract**:

It is well known that Empirical Risk Minimization (ERM) may attain minimax suboptimal rates in terms of the mean squared error (Birgé and Massart, 1993). In this paper, we prove that, under relatively mild assumptions, the suboptimality of ERM must be due to its bias. Namely, the variance error term of ERM (in terms of the bias and variance decomposition) enjoys the minimax rate. In the fixed design setting, we provide an elementary proof of this result using the probabilistic method. Then, we extend our proof to the random design setting for various models. In addition, we provide a simple proof of Chatterjee’s admissibility theorem (Chatterjee, 2014, Theorem 1.4), which states that in the fixed design setting, ERM cannot be ruled out as an optimal method, and then we extend this result to the random design setting. We also show that our estimates imply stability of ERM, complementing the main result of Caponnetto and Rakhlin (2006) for non-Donsker classes. Finally, we highlight the somewhat irregular nature of the loss landscape of ERM in the non-Donsker regime, by showing that functions can be close to ERM, in terms of $L_2$ distance, while still being far from almost-minimizers of the empirical loss.

----

## [0] NetHack is Hard to Hack

**Authors**: *Ulyana Piterbarg, Lerrel Pinto, Rob Fergus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html)

**Abstract**:

Neural policy learning methods have achieved remarkable results in various control problems, ranging from Atari games to simulated locomotion. However, these methods struggle in long-horizon tasks, especially in open-ended environments with multi-modal observations, such as the popular dungeon-crawler game, NetHack. Intriguingly, the NeurIPS 2021 NetHack Challenge revealed that symbolic agents outperformed neural approaches by over four times in median game score. In this paper, we delve into the reasons behind this performance gap and present an extensive study on neural policy learning for NetHack. To conduct this study, we analyze the winning symbolic agent, extending its codebase to track internal strategy selection in order to generate one of the largest available demonstration datasets. Utilizing this dataset, we examine (i) the advantages of an action hierarchy; (ii) enhancements in neural architecture; and (iii) the integration of reinforcement learning with imitation learning. Our investigations produce a state-of-the-art neural agent that surpasses previous fully neural policies by 127% in offline settings and 25% in online settings on median game score. However, we also demonstrate that mere scaling is insufficient to bridge the performance gap with the best symbolic models or even the top human players.

----

## [0] SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Benjamin Ellis, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob N. Foerster, Shimon Whiteson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The availability of challenging benchmarks has played a key role in the recent progress of machine learning. In cooperative multi-agent reinforcement learning, the StarCraft Multi-Agent Challenge (SMAC) has become a popular testbed for centralised training with decentralised execution. However, after years of sustained improvement on SMAC, algorithms now achieve near-perfect performance. In this work, we conduct new analysis demonstrating that SMAC lacks the stochasticity and partial observability to require complex closed-loop policies. In particular, we show that an open-loop policy conditioned only on the timestep can achieve non-trivial win rates for many SMAC scenarios. To address this limitation, we introduce SMACv2, a new version of the benchmark where scenarios are procedurally generated and require agents to generalise to previously unseen settings (from the same distribution) during evaluation. We also introduce the extended partial observability challenge (EPO), which augments SMACv2 to ensure meaningful partial observability. We show that these changes ensure the benchmarkrequires the use of closed-loop policies. We evaluate state-of-the-art algorithms on SMACv2 and show that it presents significant challenges not present in the original benchmark.  Our analysis illustrates that SMACv2 addresses the discovered deficiencies of SMAC and can help benchmark the next generation of MARL methods. Videos of training are available on our website.

----

## [0] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios

**Authors**: *Yazhe Niu, Yuan Pu, Zhenjie Yang, Xueyan Li, Tong Zhou, Jiyuan Ren, Shuai Hu, Hongsheng Li, Yu Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Building agents based on tree-search planning capabilities with learned models has achieved remarkable success in classic decision-making problems, such as Go and Atari.However, it has been deemed challenging or even infeasible to extend Monte Carlo Tree Search (MCTS) based algorithms to diverse real-world applications, especially when these environments involve complex action spaces and significant simulation costs, or inherent stochasticity.In this work, we introduce LightZero, the first unified benchmark for deploying MCTS/MuZero in general sequential decision scenarios. Specificially, we summarize the most critical challenges in designing a general MCTS-style decision-making solver, then decompose the tightly-coupled algorithm and system design of tree-search RL methods into distinct sub-modules.By incorporating more appropriate exploration and optimization strategies, we can significantly enhance these sub-modules and construct powerful LightZero agents to tackle tasks across a wide range of domains, such as board games, Atari, MuJoCo, MiniGrid and GoBigger.Detailed benchmark results reveal the significant potential of such methods in building scalable and efficient decision intelligence.The code is available as part of OpenDILab at https://github.com/opendilab/LightZero.

----

## [0] Improving Diffusion-Based Image Synthesis with Context Prediction

**Authors**: *Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, Bin Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7664a7e946a84ac5e97649a967717cf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7664a7e946a84ac5e97649a967717cf2-Abstract-Conference.html)

**Abstract**:

Diffusion models are a new class of generative models, and have dramatically promoted image generation with unprecedented quality and diversity. Existing diffusion models mainly try to reconstruct input image from a corrupted one with a pixel-wise or feature-wise constraint along spatial axes. However, such point-based reconstruction may fail to make each predicted pixel/feature fully preserve its neighborhood context, impairing diffusion-based image synthesis. As a powerful source of automatic supervisory signal, context has been well studied for learning representations. Inspired by this, we for the first time propose ConPreDiff to improve diffusion-based image synthesis with context prediction. We explicitly reinforce each point to predict its neighborhood context (i.e., multi-stride pixels/features) with a context decoder at the end of diffusion denoising blocks in training stage, and remove the decoder for inference. In this way, each point can better reconstruct itself by preserving its semantic connections with neighborhood context. This new paradigm of ConPreDiff can generalize to arbitrary discrete and continuous diffusion backbones without introducing extra parameters in sampling procedure. Extensive experiments are conducted on unconditional image generation, text-to-image generation and image inpainting tasks. Our ConPreDiff consistently outperforms previous methods and achieves new SOTA text-to-image generation results on MS-COCO, with a zero-shot FID score of 6.21.

----

## [0] Adversarial Robustness through Random Weight Sampling

**Authors**: *Yanxiang Ma, Minjing Dong, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/766f407b7b4a82135da23b32f0cbaff3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/766f407b7b4a82135da23b32f0cbaff3-Abstract-Conference.html)

**Abstract**:

Deep neural networks have been found to be vulnerable in a variety of tasks. Adversarial attacks can manipulate network outputs, resulting in incorrect predictions. Adversarial defense methods aim to improve the adversarial robustness of networks by countering potential attacks. In addition to traditional defense approaches, randomized defense mechanisms have recently received increasing attention from researchers. These methods introduce different types of perturbations during the inference phase to destabilize adversarial attacks.Although promising empirical results have been demonstrated by these approaches, the defense performance is quite sensitive to the randomness parameters, which are always manually tuned without further analysis. On the contrary, we propose incorporating random weights into the optimization to fully exploit the potential of randomized defense. To perform better optimization of randomness parameters, we conduct a theoretical analysis of the connections between randomness parameters and gradient similarity as well as natural performance. From these two aspects, we suggest imposing theoretically-guided constraints on random weights during optimizations, as these weights play a critical role in balancing natural performance and adversarial robustness. We derive both the upper and lower bounds of random weight parameters by considering prediction bias and gradient similarity. In this study, we introduce the Constrained Trainable Random Weight (CTRW), which adds random weight parameters to the optimization and includes a constraint guided by the upper and lower bounds to achieve better trade-offs between natural and robust accuracy. We evaluate the effectiveness of CTRW on several datasets and benchmark convolutional neural networks. Our results indicate that our model achieves a robust accuracy approximately 16% to 17% higher than the baseline model under PGD-20 and 22% to 25% higher on Auto Attack.

----

## [0] PyNeRF: Pyramidal Neural Radiance Fields

**Authors**: *Haithem Turki, Michael Zollhöfer, Christian Richardt, Deva Ramanan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/767c1b5f7c03d9299e493bc9e1feeba6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/767c1b5f7c03d9299e493bc9e1feeba6-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRFs) can be dramatically accelerated by spatial grid representations. However, they do not explicitly reason about scale and so introduce aliasing artifacts when reconstructing scenes captured at different camera distances. Mip-NeRF and its extensions propose scale-aware renderers that project volumetric frustums rather than point samples. But such approaches rely on positional encodings that are not readily compatible with grid methods. We propose a simple modification to grid-based models by training model heads at different spatial grid resolutions. At render time, we simply use coarser grids to render samples that cover larger volumes. Our method can be easily applied to existing accelerated NeRF methods and significantly improves rendering quality (reducing error rates by 20–90% across synthetic and unbounded real-world scenes) while incurring minimal performance overhead (as each model head is quick to evaluate). Compared to Mip-NeRF, we reduce error rates by 20% while training over 60x faster.

----

## [0] Universal Online Learning with Gradient Variations: A Multi-layer Online Ensemble Approach

**Authors**: *Yu-Hu Yan, Peng Zhao, Zhi-Hua Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an online convex optimization approach with two different levels of adaptivity. On a higher level, our approach is agnostic to the unknown types and curvatures of the online functions, while at a lower level, it can exploit the unknown niceness of the environments and attain problem-dependent guarantees. Specifically, we obtain $\mathcal{O}(\log V_T)$, $\mathcal{O}(d \log V_T)$ and $\hat{\mathcal{O}}(\sqrt{V_T})$ regret bounds for strongly convex, exp-concave and convex loss functions, respectively, where $d$ is the dimension, $V_T$ denotes problem-dependent gradient variations and the $\hat{\mathcal{O}}(\cdot)$-notation omits $\log V_T$ factors. Our result not only safeguards the worst-case guarantees but also directly implies the small-loss bounds in analysis. Moreover, when applied to adversarial/stochastic convex optimization and game theory problems, our result enhances the existing universal guarantees. Our approach is based on a multi-layer online ensemble framework incorporating novel ingredients, including a carefully designed optimism for unifying diverse function types and cascaded corrections for algorithmic stability. Notably, despite its multi-layer structure, our algorithm necessitates only one gradient query per round, making it favorable when the gradient evaluation is time-consuming. This is facilitated by a novel regret decomposition equipped with carefully designed surrogate losses.

----

## [0] Information Theoretic Lower Bounds for Information Theoretic Upper Bounds

**Authors**: *Roi Livni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/768396006e9214568dba5aae9dd312c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/768396006e9214568dba5aae9dd312c5-Abstract-Conference.html)

**Abstract**:

We examine the relationship between the mutual information between the output model and the empirical sample and the algorithm's generalization in the context of stochastic convex optimization. Despite increasing interest in information-theoretic generalization bounds, it is uncertain if these bounds can provide insight into the exceptional performance of various learning algorithms. Our study of stochastic convex optimization reveals that, for true risk minimization, dimension-dependent mutual information is necessary. This indicates that existing information-theoretic generalization bounds fall short in capturing the generalization capabilities of algorithms like SGD and regularized ERM, which have dimension-independent sample complexity.

----

## [0] CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift

**Authors**: *Siddhartha Laghuvarapu, Zhen Lin, Jimeng Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html)

**Abstract**:

In drug discovery, it is vital to confirm the predictions of pharmaceutical properties from computational models using costly wet-lab experiments. Hence, obtaining reliable uncertainty estimates is crucial for prioritizing drug molecules for subsequent experimental validation. Conformal Prediction (CP) is a promising tool for creating such prediction sets for molecular properties with a coverage guarantee. However, the exchangeability assumption of CP is often challenged with covariate shift in drug discovery tasks: Most datasets contain limited labeled data, which may not be representative of the vast chemical space from which molecules are drawn. To address this limitation, we propose a method called CoDrug that employs an energy-based model leveraging both training data and unlabelled data, and  Kernel Density Estimation (KDE) to assess the densities of a molecule set. The estimated densities are then used to weigh the molecule samples while building prediction sets and rectifying for distribution shift. In extensive experiments involving realistic distribution drifts in various small-molecule drug discovery tasks,  we demonstrate the ability of CoDrug to provide valid prediction sets and its utility in addressing the distribution shift arising from de novo drug design models. On average, using CoDrug can reduce the coverage gap by over 35% when compared to conformal prediction sets not adjusted for covariate shift.

----

## [0] TWIGMA: A dataset of AI-Generated Images with Metadata From Twitter

**Authors**: *Yiqun Chen, James Y. Zou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/769b70d1a9a6b21af53c00d0b322c763-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/769b70d1a9a6b21af53c00d0b322c763-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent progress in generative artificial intelligence (gen-AI) has enabled the generation of photo-realistic and artistically-inspiring photos at a single click, catering to millions of users online. To explore how people use gen-AI models such as DALLE and StableDiffusion, it is critical to understand the themes, contents, and variations present in the AI-generated photos. In this work, we introduce TWIGMA (TWItter Generative-ai images with MetadatA), a comprehensive dataset encompassing over 800,000 gen-AI images collected from Jan 2021 to March 2023 on Twitter, with associated metadata (e.g., tweet text, creation date, number of likes). Through a comparative analysis of TWIGMA with natural images and human artwork, we find that gen-AI images possess distinctive characteristics and exhibit, on average, lower variability when compared to their non-gen-AI counterparts. Additionally, we find that the similarity between a gen-AI image and natural images is inversely correlated with the number of likes. Finally, we observe a longitudinal shift in the themes of AI-generated images on Twitter, with users increasingly sharing artistically sophisticated content such as intricate human portraits, whereas their interest in simple subjects such as natural scenes and animals has decreased. Our analyses and findings underscore the significance of TWIGMA as a unique data resource for studying AI-generated images.

----

## [0] Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation

**Authors**: *Berivan Isik, Wei-Ning Chen, Ayfer Özgür, Tsachy Weissman, Albert No*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html)

**Abstract**:

We study the mean estimation problem under communication and local differential privacy constraints. While previous work has proposed order-optimal algorithms for the same problem (i.e., asymptotically optimal as we spend more bits), exact optimality (in the non-asymptotic setting) still has not been achieved. In this work, we take a step towards characterizing the exact-optimal approach in the presence of shared randomness (a random variable shared between the server and the user) and identify several conditions for exact optimality. We prove that one of the conditions is to utilize a rotationally symmetric shared random codebook. Based on this, we propose a randomization mechanism where the codebook is a randomly rotated simplex -- satisfying the properties of the exact-optimal codebook. The proposed mechanism is based on a $k$-closest encoding which we prove to be exact-optimal for the randomly rotated simplex codebook.

----

## [0] Estimating Generic 3D Room Structures from 2D Annotations

**Authors**: *Denys Rozumnyi, Stefan Popov, Kevis-Kokitsi Maninis, Matthias Nießner, Vittorio Ferrari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76bf913ad349686b2aa552a1c6ee0a2e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/76bf913ad349686b2aa552a1c6ee0a2e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Indoor rooms are among the most common use cases in 3D scene understanding. Current state-of-the-art methods for this task are driven by large annotated datasets. Room layouts are especially important, consisting of structural elements in 3D, such as wall, floor, and ceiling. However, they are difficult to annotate, especially on pure RGB video. We propose a novel method to produce generic 3D room layouts just from 2D segmentation masks, which are easy to annotate for humans. Based on these 2D annotations, we automatically reconstruct 3D plane equations for the structural elements and their spatial extent in the scene, and connect adjacent elements at the appropriate contact edges. We annotate and publicly release 2246 3D room layouts on the RealEstate10k dataset, containing YouTube videos. We demonstrate the high quality of these 3D layouts annotations with extensive experiments.

----

## [0] Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces

**Authors**: *Sungbin Lim, Eun-Bi Yoon, Taehyun Byun, Taewon Kang, Seungwoo Kim, Kyungjae Lee, Sungjoon Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76c6f9f2475b275b92d03a83ea270af4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76c6f9f2475b275b92d03a83ea270af4-Abstract-Conference.html)

**Abstract**:

Continuous-time score-based generative models consist of a pair of stochastic differential equations (SDEs)—a forward SDE that smoothly transitions data into a noise space and a reverse SDE that incrementally eliminates noise from a Gaussian prior distribution to generate data distribution samples—are intrinsically connected by the time-reversal theory on diffusion processes. In this paper, we investigate the use of stochastic evolution equations in Hilbert spaces, which expand the applicability of SDEs in two aspects: sample space and evolution operator, so they enable encompassing recent variations of diffusion models, such as generating functional data or replacing drift coefficients with image transformation. To this end, we derive a generalized time-reversal formula to build a bridge between probabilistic diffusion models and stochastic evolution equations and propose a score-based generative model called Hilbert Diffusion Model (HDM). Combining with Fourier neural operator, we verify the superiority of HDM for sampling functions from functional datasets with a power of kernel two-sample test of 4.2 on Quadratic, 0.2 on Melbourne, and 3.6 on Gridwatch, which outperforms existing diffusion models formulated in function spaces. Furthermore, the proposed method shows its strength in motion synthesis tasks by utilizing the Wiener process with values in Hilbert space. Finally, our empirical results on image datasets also validate a connection between HDM and diffusion models using heat dissipation, revealing the potential for exploring evolution operators and sample spaces.

----

## [0] Unlocking Feature Visualization for Deep Network with MAgnitude Constrained Optimization

**Authors**: *Thomas Fel, Thibaut Boissin, Victor Boutin, Agustin Picard, Paul Novello, Julien Colin, Drew Linsley, Tom Rousseau, Rémi Cadène, Lore Goetschalckx, Laurent Gardes, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76d2f8e328e1081c22a77ca0fa330ca5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76d2f8e328e1081c22a77ca0fa330ca5-Abstract-Conference.html)

**Abstract**:

Feature visualization has gained significant popularity as an explainability method, particularly after the influential work by Olah et al. in 2017. Despite its success, its widespread adoption has been limited due to issues in scaling to deeper neural networks and the reliance on tricks to generate interpretable images. Here, we describe MACO, a simple approach to address these shortcomings. It consists in optimizing solely an image's phase spectrum while keeping its magnitude constant to ensure that the generated explanations lie in the space of natural images. Our approach yields significantly better results -- both qualitatively and quantitatively -- unlocking efficient and interpretable feature visualizations for state-of-the-art neural networks. We also show that our approach exhibits an attribution mechanism allowing to augment feature visualizations with spatial importance. Furthermore, we enable quantitative evaluation of feature visualizations by introducing 3 metrics: transferability, plausibility, and alignment with natural images. We validate our method on various applications and we introduce a website featuring MACO visualizations for all classes of the ImageNet dataset, which will be made available upon acceptance. Overall, our study unlocks feature visualizations for the largest, state-of-the-art classification networks without resorting to any parametric prior image model, effectively advancing a field that has been stagnating since 2017 (Olah et al, 2017).

----

## [0] Exact recovery and Bregman hard clustering of node-attributed Stochastic Block Model

**Authors**: *Maximilien Dreveton, Felipe S. Fernandes, Daniel R. Figueiredo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/770b3ecb70147a2d2f18d2964fafcdd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/770b3ecb70147a2d2f18d2964fafcdd5-Abstract-Conference.html)

**Abstract**:

Classic network clustering tackles the problem of identifying sets of nodes (communities) that have similar connection patterns. However, in many scenarios nodes also have attributes that are correlated and can also be used to identify node clusters. Thus, network information (edges) and node information (attributes) can be jointly leveraged to design high-performance clustering algorithms. Under a general model for the network and node attributes, this work establishes an information-theoretic criteria for the exact recovery of community labels and characterizes a phase transition determined by the Chernoff-Hellinger divergence of the model. The criteria shows how network and attribute information can be exchanged in order to have exact recovery (e.g., more reliable network information requires less reliable attribute information). This work also presents an iterative clustering algorithm that maximizes the joint likelihood, assuming that the probability distribution of network interactions and node attributes belong to exponential families. This covers a broad range of possible interactions (e.g., edges with weights) and attributes (e.g., non-Gaussian models) while also exploring the connection between exponential families and Bregman divergences. Extensive numerical experiments using synthetic and real data indicate that the proposed algorithm outperforms algorithms that leverage only network or only attribute information as well as recently proposed algorithms that perform clustering using both sources of information. The contributions of this work provide insights into the fundamental limits and practical techniques for inferring community labels on node-attributed networks.

----

## [0] Learning to Receive Help: Intervention-Aware Concept Embedding Models

**Authors**: *Mateo Espinosa Zarlenga, Katie Collins, Krishnamurthy Dvijotham, Adrian Weller, Zohreh Shams, Mateja Jamnik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)

**Abstract**:

Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories at train-time. This conditions IntCEMs to effectively select and receive concept interventions when deployed at test-time. Our experiments show that IntCEMs significantly outperform state-of-the-art concept-interpretable models when provided with test-time concept interventions, demonstrating the effectiveness of our approach.

----

## [0] Tracr: Compiled Transformers as a Laboratory for Interpretability

**Authors**: *David Lindner, János Kramár, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, Vladimir Mikulik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html)

**Abstract**:

We show how to "compile" human-readable programs into standard decoder-only transformer models. Our compiler, Tracr, generates models with known structure. This structure can be used to design experiments. For example, we use it to study "superposition" in transformers that execute multi-step algorithms. Additionally, the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods. Commonly, because the "programs" learned by transformers are unknown it is unclear whether an interpretation succeeded. We demonstrate our approach by implementing and examining programs including computing token frequencies, sorting, and parenthesis checking. We provide an open-source implementation of Tracr at https://github.com/google-deepmind/tracr.

----

## [0] KAKURENBO: Adaptively Hiding Samples in Deep Neural Network Training

**Authors**: *Truong Thao Nguyen, Balazs Gerofi, Edgar Josafat Martinez-Noriega, François Trahay, Mohamed Wahib*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7712b1075f5e0eae297702845714098f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7712b1075f5e0eae297702845714098f-Abstract-Conference.html)

**Abstract**:

This paper proposes a method for hiding the least-important samples during the training of deep neural networks to increase efficiency, i.e., to reduce the cost of training. Using information about the loss and prediction confidence during training, we adaptively find samples to exclude in a given epoch based on their contribution to the overall learning process, without significantly degrading accuracy. We explore the converge properties when accounting for the reduction in the number of SGD updates.  Empirical results on various large-scale datasets and models used directly in image classification and segmentation show that while the with-replacement importance sampling algorithm performs poorly on large datasets,  our method can reduce total training time by up to 22\% impacting accuracy only by 0.4\% compared to the baseline.

----

## [0] Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation

**Authors**: *Dapeng Hu, Jian Liang, Jun Hao Liew, Chuhui Xue, Song Bai, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html)

**Abstract**:

Unsupervised domain adaptation (UDA) has been widely applied in improving model generalization on unlabeled target data. However, accurately selecting the best UDA model for the target domain is challenging due to the absence of labeled target data and domain distribution shifts. Traditional model selection approaches involve training extra models with source data to estimate the target validation risk. Recent studies propose practical methods that are based on measuring various properties of model predictions on target data. Although effective for some UDA models, these methods often lack stability and may lead to poor selections for other UDA models.In this paper, we present MixVal, an innovative model selection method that operates solely with unlabeled target data during inference. MixVal leverages mixed target samples with pseudo labels to directly probe the learned target structure by each UDA model. Specifically, MixVal employs two distinct types of probes: the intra-cluster mixed samples for evaluating neighborhood density and the inter-cluster mixed samples for investigating the classification boundary. With this comprehensive probing strategy, MixVal elegantly combines the strengths of two state-of-the-art model selection methods, Entropy and SND. We extensively evaluate MixVal on 11 UDA methods across 4 adaptation settings, including classification and segmentation tasks. Experimental results consistently demonstrate that MixVal achieves state-of-the-art performance and maintains exceptional stability in model selection. Code is available at \url{https://github.com/LHXXHB/MixVal}.

----

## [0] Payoff-based Learning with Matrix Multiplicative Weights in Quantum Games

**Authors**: *Kyriakos Lotidis, Panayotis Mertikopoulos, Nicholas Bambos, Jose H. Blanchet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77307e2e3f326335dfeb94ab47f7a6c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77307e2e3f326335dfeb94ab47f7a6c0-Abstract-Conference.html)

**Abstract**:

In this paper, we study the problem of learning in quantum games - and other classes of semidefinite games - with scalar, payoff-based feedback.For concreteness, we focus on the widely used matrix multiplicative weights (MMW) algorithm and, instead of requiring players to have full knowledge of the game (and/or each other's chosen states), we introduce a suite of minimal-information matrix multiplicative weights (3MW) methods tailored to different information frameworks.The main difficulty to attaining convergence in this setting is that, in contrast to classical finite games, quantum games have an infinite continuum of pure states (the quantum equivalent of pure strategies), so standard importance-weighting techniques for estimating payoff vectors cannot be employed.Instead, we borrow ideas from bandit convex optimization and we design a zeroth-order gradient sampler adapted to the semidefinite geometry of the problem at hand.As a first result, we show that the 3MW method with deterministic payoff feedback retains the $\mathcal{O}(1/\sqrt{T})$ convergence rate of the vanilla, full information MMW algorithm in quantum min-max games, even though the players only observe a single scalar.Subsequently, we relax the algorithm's information requirements even further and we provide a 3MW method that only requires players to observe a random realization of their payoff observable, and converges to equilibrium at an $\mathcal{O}(T^{-1/4})$ rate.Finally, going beyond zero-sum games, we show that a regularized variant of the proposed 3MW method guarantees local convergence with high probability to all equilibria that satisfy a certain first-order stability condition.

----

## [0] Deep Stochastic Processes via Functional Markov Transition Operators

**Authors**: *Jin Xu, Emilien Dupont, Kaspar Märtens, Thomas Rainforth, Yee Whye Teh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html)

**Abstract**:

We introduce Markov Neural Processes (MNPs), a new class of Stochastic Processes (SPs) which are constructed by stacking sequences of neural parameterised Markov transition operators in function space. We prove that these Markov transition operators can preserve the exchangeability and consistency of SPs. Therefore, the proposed iterative construction adds substantial flexibility and expressivity to the original framework of Neural Processes (NPs) without compromising consistency or adding restrictions. Our experiments demonstrate clear advantages of MNPs over baseline models on a variety of tasks.

----

## [0] Quilt-1M: One Million Image-Text Pairs for Histopathology

**Authors**: *Wisdom Oluchi Ikezogwo, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Stefan Chan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, Linda G. Shapiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has slowed comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians.From YouTube, we curate QUILT: a large-scale vision-language dataset consisting of $802, 144$ image and text pairs.QUILT was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition.In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples.We combine QUILT with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even larger dataset: QUILT-1M, with $1$M paired image-text samples, marking it as the largest vision-language histopathology dataset to date. We demonstrate the value of QUILT-1M by fine-tuning a pre-trained CLIP model. Our model outperforms state-of-the-art models on both zero-shot and linear probing tasks for classifying new histopathology images across $13$ diverse patch-level datasets of $8$ different sub-pathologies and cross-modal retrieval tasks.

----

## [0] A Computation and Communication Efficient Method for Distributed Nonconvex Problems in the Partial Participation Setting

**Authors**: *Alexander Tyurin, Peter Richtárik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/778ff1fcfb6d6707fc015908a1845b62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/778ff1fcfb6d6707fc015908a1845b62-Abstract-Conference.html)

**Abstract**:

We present a new method that includes three key components of distributed optimization and federated learning: variance reduction of stochastic gradients, partial participation, and compressed communication. We prove that the new method has optimal oracle complexity and state-of-the-art communication complexity in the partial participation setting. Regardless of the communication compression feature, our method successfully combines variance reduction and partial participation: we get the optimal oracle complexity, never need the participation of all nodes, and do not require the bounded gradients (dissimilarity) assumption.

----

## [0] Optimistic Active Exploration of Dynamical Systems

**Authors**: *Bhavya Sukhija, Lenart Treven, Cansu Sancaktar, Sebastian Blaes, Stelian Coros, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html)

**Abstract**:

Reinforcement learning algorithms commonly seek to optimize policies for solving one particular task. How should we explore an unknown dynamical system such that the estimated model allows us to solve multiple downstream tasks in a zero-shot manner? In this paper, we address this challenge, by developing an algorithm -- OPAX -- for active exploration. OPAX uses well-calibrated probabilistic models to quantify the epistemic uncertainty about the unknown dynamics. It optimistically---w.r.t. to plausible dynamics---maximizes the information gain between the unknown dynamics and state observations. We show how the resulting optimization problem can be reduced to an optimal control problem that can be solved at each episode using standard approaches.  We analyze our algorithm for general models, and, in the case of Gaussian process dynamics, we give a sample complexity bound andshow that the epistemic uncertainty converges to zero. In our experiments, we compare OPAX with other heuristic active exploration approaches on several environments. Our experiments show that OPAX is not only theoretically sound but also performs well for zero-shot planning on novel downstream tasks.

----

## [0] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

**Authors**: *Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77c33e6a367922d003ff102ffb92b658-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77c33e6a367922d003ff102ffb92b658-Abstract-Conference.html)

**Abstract**:

Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are numerous AI models available for various domains and modalities, they cannot handle complicated AI tasks autonomously. Considering large language models (LLMs) have exhibited exceptional abilities in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks, with language serving as a generic interface to empower this. Based on this philosophy, we present HuggingGPT, an LLM-powered agent that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Hugging Face, HuggingGPT can tackle a wide range of sophisticated AI tasks spanning different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards the realization of artificial general intelligence.

----

## [0] Multi-Step Generalized Policy Improvement by Leveraging Approximate Models

**Authors**: *Lucas Nunes Alegre, Ana L. C. Bazzan, Ann Nowé, Bruno C. da Silva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html)

**Abstract**:

We introduce a principled method for performing zero-shot transfer in reinforcement learning (RL) by exploiting approximate models of the environment. Zero-shot transfer in RL has been investigated by leveraging methods rooted in generalized policy improvement (GPI) and successor features (SFs). Although computationally efficient, these methods are model-free: they analyze a library of policies---each solving a particular task---and identify which action the agent should take. We investigate the more general setting where, in addition to a library of policies, the agent has access to an approximate environment model. Even though model-based RL algorithms can identify near-optimal policies, they are typically computationally intensive. We introduce $h$-GPI, a multi-step extension of GPI that interpolates between these extremes---standard model-free GPI and fully model-based planning---as a function of a parameter, $h$, regulating the amount of time the agent has to reason. We prove that $h$-GPI's performance lower bound is strictly better than GPI's, and show that $h$-GPI generally outperforms GPI as $h$ increases. Furthermore, we prove that as $h$ increases, $h$-GPI's performance becomes arbitrarily less susceptible to sub-optimality in the agent's policy library. Finally, we introduce novel bounds characterizing the gains achievable by $h$-GPI as a function of approximation errors in both the agent's policy library and its (possibly learned) model. These bounds strictly generalize those known in the literature. We evaluate $h$-GPI on challenging tabular and continuous-state problems under value function approximation and show that it consistently outperforms GPI and state-of-the-art competing methods under various levels of approximation errors.

----

## [0] GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients

**Authors**: *Sima Behpour, Thang Long Doan, Xin Li, Wenbin He, Liang Gou, Liu Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77cf940349218069bbc230fc2c9c8a21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77cf940349218069bbc230fc2c9c8a21-Abstract-Conference.html)

**Abstract**:

Detecting out-of-distribution (OOD) data is crucial for ensuring the safe deployment of machine learning models in real-world applications. However, existing OOD detection approaches primarily rely on the feature maps or the full gradient space information to derive OOD scores neglecting the role of \textbf{most important parameters} of the pre-trained network over In-Distribution data. In this study, we propose a novel approach called GradOrth to facilitate OOD detection based on one intriguing observation that the important features to identify OOD data lie in the lower-rank subspace of in-distribution (ID) data.In particular, we identify OOD data by computing the norm of gradient projection on \textit{the subspaces considered \textbf{important} for the in-distribution data}. A large orthogonal projection value (i.e. a small projection value) indicates the sample as OOD as it captures a weak correlation of the in-distribution (ID) data. This simple yet effective method exhibits outstanding performance, showcasing a notable reduction in the average false positive rate at a 95\% true positive rate (FPR95) of up to 8\% when compared to the current state-of-the-art methods.

----

## [0] Learning to Modulate pre-trained Models in RL

**Authors**: *Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html)

**Abstract**:

Reinforcement Learning (RL) has been successful in various domains like robotics, game playing, and simulation. While RL agents have shown impressive capabilities in their specific tasks, they insufficiently adapt to new tasks. In supervised learning, this adaptation problem is addressed by large-scale pre-training followed by fine-tuning to new down-stream tasks. Recently, pre-training on multiple tasks has been gaining traction in RL. However, fine-tuning a pre-trained model often suffers from catastrophic forgetting. That is, the performance on the pre-training tasks deteriorates when fine-tuning on new tasks. To investigate the catastrophic forgetting phenomenon, we first jointly pre-train a model on datasets from two benchmark suites, namely Meta-World and DMControl. Then, we evaluate and compare a variety of fine-tuning methods prevalent in natural language processing, both in terms of performance on new tasks, and how well performance on pre-training tasks is retained. Our study shows that with most fine-tuning approaches, the performance on pre-training tasks deteriorates significantly. Therefore, we propose a novel method, Learning-to-Modulate (L2M), that avoids the degradation of learned skills by modulating the information flow of the frozen pre-trained model via a learnable modulation pool. Our method achieves state-of-the-art performance on the Continual-World benchmark, while retaining performance on the pre-training tasks. Finally, to aid future research in this area, we release a dataset encompassing 50 Meta-World and 16 DMControl tasks.

----

## [0] Injecting Multimodal Information into Rigid Protein Docking via Bi-level Optimization

**Authors**: *Ruijia Wang, YiWu Sun, Yujie Luo, Shaochuan Li, Cheng Yang, Xingyi Cheng, Hui Li, Chuan Shi, Le Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77fa0e7d45c6687f1958de0b31e9fc05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77fa0e7d45c6687f1958de0b31e9fc05-Abstract-Conference.html)

**Abstract**:

The structure of protein-protein complexes is critical for understanding binding dynamics, biological mechanisms, and intervention strategies. Rigid protein docking, a fundamental problem in this field, aims to predict the 3D structure of complexes from their unbound states without conformational changes. In this scenario, we have access to two types of valuable information: sequence-modal information, such as coevolutionary data obtained from multiple sequence alignments, and structure-modal information, including the 3D conformations of rigid structures. However, existing docking methods typically utilize single-modal information, resulting in suboptimal predictions. In this paper, we propose xTrimoBiDock (or BiDock for short), a novel rigid docking model that effectively integrates sequence- and structure-modal information through bi-level optimization. Specifically, a cross-modal transformer combines multimodal information to predict an inter-protein distance map. To achieve rigid docking, the roto-translation transformation is optimized to align the docked pose with the predicted distance map. In order to tackle this bi-level optimization problem, we unroll the gradient descent of the inner loop and further derive a better initialization for roto-translation transformation based on spectral estimation. Compared to baselines, BiDock achieves a promising result of a maximum 234% relative improvement in challenging antibody-antigen docking problem.

----

## [0] Uncertainty-Aware Alignment Network for Cross-Domain Video-Text Retrieval

**Authors**: *Xiaoshuai Hao, Wanqian Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78526d7ad4a2532bd91416e948b9644c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78526d7ad4a2532bd91416e948b9644c-Abstract-Conference.html)

**Abstract**:

Video-text retrieval is an important but challenging research task in the multimedia community.  In this paper, we address the challenge task of Unsupervised Domain Adaptation Video-text Retrieval (UDAVR), assuming that training (source) data and testing (target) data are from different domains. Previous approaches are mostly derived from classification based domain adaptation methods, which are neither multi-modal nor suitable for retrieval task.  In addition, as to the pairwise misalignment issue in target domain, i.e., no pairwise annotations between target videos and texts, the existing method assumes that a video corresponds to a text. Yet we empirically find that in the real scene, one text usually corresponds to multiple videos and vice versa. To tackle this one-to-many issue, we propose a novel method named Uncertainty-aware Alignment Network (UAN). Specifically, we first introduce the multimodal mutual information module to balance the minimization of domain shift in a smooth manner. To tackle the multimodal uncertainties pairwise misalignment in target domain, we propose the Uncertainty-aware Alignment Mechanism (UAM) to fully exploit the semantic information of both modalities in target domain. Extensive experiments in the context of domain-adaptive video-text retrieval demonstrate that our proposed method consistently outperforms multiple baselines, showing a superior generalization ability for target data.

----

## [0] Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?

**Authors**: *Jialu Gao, Kaizhe Hu, Guowei Xu, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7866ff509c822c2e58d20d00154a15a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7866ff509c822c2e58d20d00154a15a2-Abstract-Conference.html)

**Abstract**:

Pre-trained text-to-image generative models can produce diverse, semantically rich, and realistic images from natural language descriptions. Compared with language, images usually convey information with more details and less ambiguity. In this study, we propose Learning from the Void (LfVoid), a method that leverages the power of pre-trained text-to-image models and advanced image editing techniques to guide robot learning. Given natural language instructions, LfVoid can edit the original observations to obtain goal images, such as "wiping" a stain off a table. Subsequently, LfVoid trains an ensembled goal discriminator on the generated image to provide reward signals for a reinforcement learning agent, guiding it to achieve the goal. The ability of LfVoid to learn with zero in-domain training on expert demonstrations or true goal observations (the void) is attributed to the utilization of knowledge from web-scale generative models. We evaluate LfVoid across three simulated tasks and validate its feasibility in the corresponding real-world scenarios. In addition, we offer insights into the key considerations for the effective integration of visual generative models into robot learning workflows. We posit that our work represents an initial step towards the broader application of pre-trained visual generative models in the robotics field. Our project page: https://lfvoid-rl.github.io/.

----

## [0] H3T: Efficient Integration of Memory Optimization and Parallelism for Large-scale Transformer Training

**Authors**: *Yuzhong Wang, Xu Han, Weilin Zhao, Guoyang Zeng, Zhiyuan Liu, Maosong Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7886b89aced4d37dd25a6f32854bf3f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7886b89aced4d37dd25a6f32854bf3f9-Abstract-Conference.html)

**Abstract**:

In recent years, big models based on Transformers have achieved state-of-the-art performance on many artificial intelligence (AI) tasks.Despite the success of these Transformer-based models, their huge parameter size poses a serious challenge to their training, both from the storage and computation perspectives.To this end, memory optimization (e.g., rematerialization and offloading) and parallelism (e.g., data parallelism and model parallelism) are widely explored to make training Transformers more efficient.In this paper, we propose a framework to automatically find an efficient integration of memory optimization and parallelism for High-Throughput Transformer Training (named H3T), which is rarely considered by existing efforts for training big Transformer-based models.Specifically, we design search algorithms to combine appropriate memory optimization strategies and parallelism schemes to achieve a balance between memory overhead and training efficiency.We implement H3T based on an open-source toolkit BMTrain and then use H3T to train the Transformers of different sizes to evaluate the efficiency of H3T.The experimental results show that H3T outperforms the most popular deep learning (DL) toolkit Megatron-DeepSpeed by $1.2\times \sim 4.3\times$ training speed while reducing $34.6\% \sim 80.5\%$ of memory overhead.Moreover, H3T can use only 64 NVIDIA A100 GPUs to train GPT-3-175B, which is very difficult for existing DL toolkits. The source code is available at https://github.com/OpenBMB/BMTrain/tree/h3t.

----

## [0] Binarized Spectral Compressive Imaging

**Authors**: *Yuanhao Cai, Yuxin Zheng, Jing Lin, Xin Yuan, Yulun Zhang, Haoqian Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/788e086c07b8d6fa6b279df56e512312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/788e086c07b8d6fa6b279df56e512312-Abstract-Conference.html)

**Abstract**:

Existing deep learning models for hyperspectral image (HSI) reconstruction achieve good performance but require powerful hardwares with enormous memory and computational resources. Consequently, these methods can hardly be deployed on resource-limited mobile devices. In this paper, we propose a novel method, Binarized Spectral-Redistribution Network (BiSRNet), for efficient and practical HSI restoration from compressed measurement in snapshot compressive imaging (SCI) systems. Firstly, we redesign a compact and easy-to-deploy base model to be binarized. Then we present the basic unit, Binarized Spectral-Redistribution Convolution (BiSR-Conv). BiSR-Conv can adaptively redistribute the HSI representations before binarizing activation and uses a scalable hyperbolic tangent function to closer approximate the Sign function in backpropagation. Based on our BiSR-Conv, we customize four binarized convolutional modules to address the dimension mismatch and propagate full-precision information throughout the whole network. Finally, our BiSRNet is derived by using the proposed techniques to binarize the base model. Comprehensive quantitative and qualitative experiments manifest that our proposed BiSRNet outperforms state-of-the-art binarization algorithms. Code and models are publicly available at https://github.com/caiyuanhao1998/BiSCI

----

## [0] When Can We Track Significant Preference Shifts in Dueling Bandits?

**Authors**: *Joe Suk, Arpit Agarwal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78ccee9dfbcf84840165ab4093715969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78ccee9dfbcf84840165ab4093715969-Abstract-Conference.html)

**Abstract**:

The $K$-armed dueling bandits problem, where the feedback is in the form of noisy pairwise preferences, has been widely studied due its applications in information retrieval, recommendation systems, etc. Motivated by concerns that user preferences/tastes can evolve over time, we consider the problem of _dueling bandits with distribution shifts_. Specifically, we study the recent notion of _significant shifts_ (Suk and Kpotufe, 2022), and ask whether one can design an _adaptive_ algorithm for the dueling problem with $O(\sqrt{K\tilde{L}T})$ dynamic regret,where $\tilde{L}$ is the (unknown) number of significant shifts in preferences. We show that the answer to this question depends on the properties of underlying preference distributions. Firstly,  we give an impossibility result that rules out any algorithm with $O(\sqrt{K\tilde{L}T})$ dynamic regret under the well-studied Condorcet and SST classes of preference distributions. Secondly, we show that $\text{SST}\cap \text{STI}$ is the largest amongst popular classes of preference distributions where it is possible to design such an algorithm. Overall, our results provides an almost complete resolution of the above question for the hierarchy of  distribution classes.

----

## [0] Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization

**Authors**: *Haitz Sáez de Ocáriz Borde, Alvaro Arroyo, Ismael Morales, Ingmar Posner, Xiaowen Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html)

**Abstract**:

Recent research indicates that the performance of machine learning models can be improved by aligning the geometry of the latent space with the underlying data structure. Rather than relying solely on Euclidean space, researchers have proposed using hyperbolic and spherical spaces with constant curvature, or combinations thereof, to better model the latent space and enhance model performance. However, little attention has been given to the problem of automatically identifying the optimal latent geometry for the downstream task. We mathematically define this novel formulation and coin it as neural latent geometry search (NLGS). More specifically, we introduce an initial attempt to search for a latent geometry composed of a product of constant curvature model spaces with a small number of query evaluations, under some simplifying assumptions. To accomplish this, we propose a novel notion of distance between candidate latent geometries based on the Gromov-Hausdorff distance from metric geometry. In order to compute the Gromov-Hausdorff distance, we introduce a mapping function that enables the comparison of different manifolds by embedding them in a common high-dimensional ambient space. We then design a graph search space based on the notion of smoothness between latent geometries and employ the calculated distances as an additional inductive bias. Finally, we use Bayesian optimization to search for the optimal latent geometry in a query-efficient manner. This is a general method which can be applied to search for the optimal latent geometry for a variety of models and downstream tasks. We perform experiments on synthetic and real-world datasets to identify the optimal latent geometry for multiple machine learning problems.

----

## [0] Scientific Document Retrieval using Multi-level Aspect-based Queries

**Authors**: *Jianyou Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78f9c04bdcb06f1ada3902912d8b64ba-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/78f9c04bdcb06f1ada3902912d8b64ba-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high costs and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task,  $\textbf{S}$cientific $\textbf{Do}$cument $\textbf{R}$etrieval using $\textbf{M}$ulti-level $\textbf{A}$spect-based qu$\textbf{E}$ries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for evaluating the viability of Large Language Models (LLMs) such as ChatGPT-3.5 for expert-level dataset annotation tasks. The application of Anno-GPT to annotate the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these complex queries, our DORIS-MAE dataset can be extended to over 4,000 sub-query test cases without requiring additional annotation. We evaluated 17 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights DORIS-MAE's challenges and the need for better approaches to handle complex, multifaceted queries in scientific research. Our dataset and codebase are available at https://github.com/Real-Doris-Mae/Doris-Mae-Dataset .

----

## [0] Beyond Confidence: Reliable Models Should Also Consider Atypicality

**Authors**: *Mert Yüksekgönül, Linjun Zhang, James Y. Zou, Carlos Guestrin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7900318ffaf5e9bc60250f134c6cc3c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7900318ffaf5e9bc60250f134c6cc3c7-Abstract-Conference.html)

**Abstract**:

While most machine learning models can provide confidence in their predictions, confidence is insufficient to understand a prediction's reliability. For instance, the model may have a low confidence prediction if the input is not well-represented in the training dataset or if the input is inherently ambiguous. In this work, we investigate the relationship between how atypical~(rare) a sample or a class is and the reliability of a model's predictions. We first demonstrate that atypicality is strongly related to miscalibration and accuracy. In particular, we empirically show that predictions for atypical inputs or atypical classes are more overconfident and have lower accuracy. Using these insights, we show incorporating atypicality improves uncertainty quantification and model performance for discriminative neural networks and large language models. In a case study, we show that using atypicality improves the performance of a skin lesion classifier across different skin tone groups without having access to the group attributes. Overall, we propose that models should use not only confidence but also atypicality to improve uncertainty quantification and performance. Our results demonstrate that simple post-hoc atypicality estimators can provide significant value.

----

## [0] Reversible and irreversible bracket-based dynamics for deep graph neural networks

**Authors**: *Anthony Gruber, Kookjin Lee, Nathaniel Trask*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7903af0a1cffb43dbb2f8160d110a5f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7903af0a1cffb43dbb2f8160d110a5f3-Abstract-Conference.html)

**Abstract**:

Recent works have shown that physics-inspired architectures allow the training of deep graph neural networks (GNNs) without oversmoothing. The role of these physics is unclear, however, with successful examples of both reversible (e.g., Hamiltonian) and irreversible (e.g., diffusion) phenomena producing comparable results despite diametrically opposed mechanisms, and further complications arising due to empirical departures from mathematical theory. This work presents a series of novel GNN architectures based upon structure-preserving bracket-based dynamical systems, which are provably guaranteed to either conserve energy or generate positive dissipation with increasing depth.  It is shown that the theoretically principled framework employed here allows for inherently explainable constructions, which contextualize departures from theory in current architectures and better elucidate the roles of reversibility and irreversibility in network performance. Code is available at the Github repository \url{https://github.com/natrask/BracketGraphs}.

----

## [0] In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer

**Authors**: *Yuzhou Cao, Hussein Mozannar, Lei Feng, Hongxin Wei, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html)

**Abstract**:

Enabling machine learning classifiers to defer their decision to a downstream expert when the expert is more accurate will ensure improved safety and performance. This objective can be achieved with the learning-to-defer framework which aims to jointly learn how to classify and how to defer to the expert. In recent studies, it has been theoretically shown that popular estimators for learning to defer parameterized with softmax provide unbounded estimates for the likelihood of deferring which makes them uncalibrated. However, it remains unknown whether this is due to the widely used softmax parameterization and if we can find a softmax-based estimator that is both statistically consistent and possesses a valid probability estimator. In this work, we first show that the cause of the miscalibrated and unbounded estimator in prior literature is due to the symmetric nature of the surrogate losses used and not due to softmax. We then propose a novel statistically consistent asymmetric softmax-based surrogate loss that can produce valid estimates without the issue of unboundedness. We further analyze the non-asymptotic properties of our proposed method and empirically validate its performance and calibration on benchmark datasets.

----

## [0] Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection

**Authors**: *Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79206ac5b7e88eeeed74997f3b6f4c7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79206ac5b7e88eeeed74997f3b6f4c7f-Abstract-Conference.html)

**Abstract**:

Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS. The code will be released at https://github.com/OpenDriveLab/Birds-eye-view-Perception.

----

## [0] No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions

**Authors**: *Tiancheng Jin, Junyan Liu, Chloé Rouyer, William Chang, Chen-Yu Wei, Haipeng Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79358587d84628728199059f648824e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79358587d84628728199059f648824e6-Abstract-Conference.html)

**Abstract**:

Existing online learning algorithms for adversarial Markov Decision Processes achieve $\mathcal{O}(\sqrt{T})$ regret after $T$ rounds of interactions even if the loss functions are chosen arbitrarily by an adversary, with the caveat that the transition function has to be fixed.This is because it has been shown that adversarial transition functions make no-regret learning impossible.Despite such impossibility results, in this work, we develop algorithms that can handle both adversarial losses and adversarial transitions, with regret increasing smoothly in the degree of maliciousness of the adversary.More concretely, we first propose an algorithm that enjoys $\widetilde{\mathcal{O}}(\sqrt{T} + C^{P})$ regret where $C^{P}$ measures how adversarial the transition functions are and can be at most $\mathcal{O}(T)$.While this algorithm itself requires knowledge of $C^{P}$, we further develop a black-box reduction approach that removes this requirement.Moreover, we also show that further refinements of the algorithm not only maintains the same regret bound, but also simultaneously adapts to easier environments (where losses are generated in a certain stochastically constrained manner as in [Jin et al. 2021]) and achieves $\widetilde{\mathcal{O}}(U + \sqrt{UC^{L}}  + C^{P})$ regret, where $U$ is some standard gap-dependent coefficient and $C^{L}$ is the amount of corruption on losses.

----

## [0] Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark

**Authors**: *Lukasz Augustyniak, Szymon Wozniak, Marcin Gruza, Piotr Gramacki, Krzysztof Rajda, Mikolaj Morzy, Tomasz Kajdanowicz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite impressive advancements in multilingual corpora collection and model training, developing large-scale deployments of multilingual models still presents a significant challenge. This is particularly true for language tasks that are culture-dependent. One such example is the area of multilingual sentiment analysis, where affective markers can be subtle and deeply ensconced in culture.This work presents the most extensive open massively multilingual corpus of datasets for training sentiment models. The corpus consists of 79 manually selected datasets from over 350 datasets reported in the scientific literature based on strict quality criteria. The corpus covers 27 languages representing 6 language families. Datasets can be queried using several linguistic and functional features. In addition, we present a multi-faceted sentiment classification benchmark summarizing hundreds of experiments conducted on different base models, training objectives, dataset collections, and fine-tuning strategies.

----

## [0] Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations

**Authors**: *Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/796455f65fd2cbe049112a2d2d4488cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/796455f65fd2cbe049112a2d2d4488cb-Abstract-Conference.html)

**Abstract**:

Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.

----

## [0] Ignorance is Bliss: Robust Control via Information Gating

**Authors**: *Manan Tomar, Riashat Islam, Matthew E. Taylor, Sergey Levine, Philip Bachman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/797be96e4481c3fe5d675c1ba5352969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/797be96e4481c3fe5d675c1ba5352969-Abstract-Conference.html)

**Abstract**:

Informational parsimony provides a useful inductive bias for learning representations that achieve better generalization by being robust to noise and spurious correlations. We propose information gating as a way to learn parsimonious representations that identify the minimal information required for a task. When gating information, we can learn to reveal as little information as possible so that a task remains solvable, or hide as little information as possible so that a task becomes unsolvable. We gate information using a differentiable parameterization of the signal-to-noise ratio, which can be applied to arbitrary values in a network, e.g., erasing pixels at the input layer or activations in some intermediate layer. When gating at the input layer, our models learn which visual cues matter for a given task. When gating intermediate layers, our models learn which activations are needed for subsequent stages of computation. We call our approach InfoGating. We apply InfoGating to various objectives such as multi-step forward and inverse dynamics models, Q-learning, and behavior cloning, highlighting how InfoGating can naturally help in discarding information not relevant for control. Results show that learning to identify and use minimal information can improve generalization in downstream tasks. Policies based on InfoGating are considerably more robust to irrelevant visual features, leading to improved pretraining and finetuning of RL models.

----

## [0] Reduced Policy Optimization for Continuous Control with Hard Constraints

**Authors**: *Shutong Ding, Jingya Wang, Yali Du, Ye Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7984e22a06eb5f0e35d745cb38345983-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7984e22a06eb5f0e35d745cb38345983-Abstract-Conference.html)

**Abstract**:

Recent advances in constrained reinforcement learning (RL) have endowed reinforcement learning with certain safety guarantees. However, deploying existing constrained RL algorithms in continuous control tasks with general hard constraints remains challenging, particularly in those situations with non-convex hard constraints. Inspired by the generalized reduced gradient (GRG) algorithm, a classical constrained optimization technique, we propose a reduced policy optimization (RPO) algorithm that combines RL with GRG to address general hard constraints. RPO partitions actions into basic actions and nonbasic actions following the GRG method and outputs the basic actions via a policy network. Subsequently, RPO calculates the nonbasic actions by solving equations based on equality constraints using the obtained basic actions. The policy network is then updated by implicitly differentiating nonbasic actions with respect to basic actions. Additionally, we introduce an action projection procedure based on the reduced gradient and apply a modified Lagrangian relaxation technique to ensure inequality constraints are satisfied. To the best of our knowledge, RPO is the first attempt that introduces GRG to RL as a way of efficiently handling both equality and inequality hard constraints. It is worth noting that there is currently a lack of RL environments with complex hard constraints, which motivates us to develop three new benchmarks: two robotics manipulation tasks and a smart grid operation control task. With these benchmarks, RPO achieves better performance than previous constrained RL algorithms in terms of both cumulative reward and constraint violation. We believe RPO, along with the new benchmarks, will open up new opportunities for applying RL to real-world problems with complex constraints.

----

## [0] ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning

**Authors**: *Mingyu Xu, Zheng Lian, Lei Feng, Bin Liu, Jianhua Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html)

**Abstract**:

Noisy partial label learning (noisy PLL) is an important branch of weakly supervised learning. Unlike PLL where the ground-truth label must conceal in the candidate label set, noisy PLL relaxes this constraint and allows the ground-truth label may not be in the candidate label set. To address this challenging problem, most of the existing works attempt to detect noisy samples and estimate the ground-truth label for each noisy sample. However, detection errors are unavoidable. These errors can accumulate during training and continuously affect model optimization. To this end, we propose a novel framework for noisy PLL with theoretical interpretations, called ``Adjusting Label Importance Mechanism (ALIM)''. It aims to reduce the negative impact of detection errors by trading off the initial candidate set and model outputs. ALIM is a plug-in strategy that can be integrated with existing PLL approaches. Experimental results on multiple benchmark datasets demonstrate that our method can achieve state-of-the-art performance on noisy PLL. Our code is available at: https://github.com/zeroQiaoba/ALIM.

----

## [0] Conditional Score Guidance for Text-Driven Image-to-Image Translation

**Authors**: *Hyunsoo Lee, Minsoo Kang, Bohyung Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/799f81cfa0611f93586c007024041460-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/799f81cfa0611f93586c007024041460-Abstract-Conference.html)

**Abstract**:

We present a novel algorithm for text-driven image-to-image translation based on a pretrained text-to-image diffusion model. Our method aims to generate a target image by selectively editing regions of interest in a source image, defined by a modifying text, while preserving the remaining parts.In contrast to existing techniques that solely rely on a target prompt, we introduce a new score function that additionally considers both the source image and the source text prompt, tailored to address specific translation tasks. To this end, we derive the conditional score function in a principled way, decomposing it into the standard score and a guiding term for target image generation.For the gradient computation about the guiding term, we assume a Gaussian distribution for the posterior distribution and estimate its mean and variance to adjust the gradient without additional training.In addition, to improve the quality of the conditional score guidance, we incorporate a simple yet effective mixup technique, which combines two cross-attention maps derived from the source and target latents.This strategy is effective for promoting a desirable fusion of the invariant parts in the source image and the edited regions aligned with the target prompt, leading to high-fidelity target image generation.Through comprehensive experiments, we demonstrate that our approach achieves outstanding image-to-image translation performance on various tasks.Code is available at https://github.com/Hleephilip/CSG.

----

## [0] A Unified Approach to Count-Based Weakly Supervised Learning

**Authors**: *Vinay Shukla, Zhe Zeng, Kareem Ahmed, Guy Van den Broeck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html)

**Abstract**:

High-quality labels are often very scarce, whereas unlabeled data with inferred weak labels occurs more naturally. In many cases, these weak labels dictate the frequency of each respective class over a set of instances. In this paper, we develop a unified approach to learning from such weakly-labeled data, which we call *count-based weakly-supervised learning*. At the heart of our approach is the ability to compute the probability of exactly $k$ out of $n$ outputs being set to true. This computation is differentiable, exact, and efficient. Building upon the previous computation, we derive a *count loss* penalizing the model for deviations in its distribution from an arithmetic constraint defined over label counts.

----

## [0] Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars

**Authors**: *Kaiyue Wen, Yuchen Li, Bingbin Liu, Andrej Risteski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html)

**Abstract**:

Transformer interpretability aims to understand the algorithm implemented by a learned Transformer by examining various aspects of the model, such as the weight matrices or the attention patterns.In this work, through a combination of theoretical results and carefully controlled experiments on synthetic data, we take a critical viewof methods that exclusively focus on individual parts of the model, rather than consider the network as a whole.We consider a simple synthetic setup of learning a (bounded) Dyck language. Theoretically, we show that the set of models that (exactly or approximately) solve this task satisfy a structural characterization derived from ideas in formal languages (the pumping lemma).We use this characterization to show that the set of optima is qualitatively rich; in particular, the attention pattern of a single layer can be "nearly randomized", while preserving the functionality of the network.We also show via extensive experiments that these constructions are not merely a theoretical artifact: even with severe constraints to the architecture of the model, vastly different solutions can be reached via standard training. Thus, interpretability claims based on inspecting individual heads or weight matrices in the Transformer can be misleading.

----

## [0] GEQ: Gaussian Kernel Inspired Equilibrium Models

**Authors**: *Mingjie Li, Yisen Wang, Zhouchen Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79cab89b43ac21c6941ad9735df95d30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79cab89b43ac21c6941ad9735df95d30-Abstract-Conference.html)

**Abstract**:

Despite the connection established by optimization-induced deep equilibrium models (OptEqs) between their output and the underlying hidden optimization problems, the performance of it along with its related works is still not good enough especially when compared to deep networks. One key factor responsible for this performance limitation is the use of linear kernels to extract features in these models. To address this issue, we propose a novel approach by replacing its linear kernel with a new function that can readily capture nonlinear feature dependencies in the input data. Drawing inspiration from classical machine learning algorithms, we introduce Gaussian kernels as the alternative function and then propose our new equilibrium model, which we refer to as GEQ. By leveraging Gaussian kernels, GEQ can effectively extract the nonlinear information embedded within the input features, surpassing the performance of the original OptEqs. Moreover, GEQ can be perceived as a weight-tied neural network with infinite width and depth. GEQ also enjoys better theoretical properties and improved overall performance. Additionally, our GEQ exhibits enhanced stability when confronted with various samples. We further substantiate the effectiveness and stability of GEQ through a series of comprehensive experiments.

----

## [0] Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric

**Authors**: *Yiming Wang, Ming Yang, Renzhi Dong, Binbin Sun, Furui Liu, Leong Hou U*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79f7f00cbe3003cea4d0c2326b4c0b42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79f7f00cbe3003cea4d0c2326b4c0b42-Abstract-Conference.html)

**Abstract**:

Reward shaping is an effective technique for integrating domain knowledge into reinforcement learning (RL). However, traditional approaches like potential-based reward shaping totally rely on manually designing shaping reward functions, which significantly restricts exploration efficiency and introduces human cognitive biases.While a number of RL methods have been proposed to boost exploration by designing an intrinsic reward signal as exploration bonus. Nevertheless, these methods heavily rely on the count-based episodic term in their exploration bonus which falls short in scalability. To address these limitations, we propose a general end-to-end potential-based exploration bonus for deep RL via potentials of state discrepancy, which motivates the agent to discover novel states and provides them with denser rewards without manual intervention. Specifically, we measure the novelty of adjacent states by calculating their distance using the bisimulation metric-based potential function, which enhances agent's exploration and ensures policy invariance. In addition, we offer a theoretical guarantee on our inverse dynamic bisimulation metric, bounding the value difference and ensuring that the agent explores states with higher TD error, thus significantly improving training efficiency. The proposed approach is named \textbf{LIBERTY} (exp\textbf{L}oration v\textbf{I}a \textbf{B}isimulation m\textbf{E}t\textbf{R}ic-based s\textbf{T}ate discrepanc\textbf{Y}) which is comprehensively evaluated on the MuJoCo and the Arcade Learning Environments. Extensive experiments have verified the superiority and scalability of our algorithm compared with other competitive methods.

----

## [0] What's Left? Concept Grounding with Logic-Enhanced Foundation Models

**Authors**: *Joy Hsu, Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html)

**Abstract**:

Recent works such as VisProg and ViperGPT have smartly composed foundation models for visual reasoning—using large language models (LLMs) to produce programs that can be executed by pre-trained vision-language models. However, they operate in limited domains, such as 2D images, not fully exploiting the generalization of language: abstract concepts like “left” can also be grounded in 3D, temporal, and action data, as in moving to your left. This limited generalization stems from these inference-only methods’ inability to learn or adapt pre-trained models to a new domain. We propose the Logic-Enhanced FoundaTion Model (LEFT), a unified framework that learns to ground and reason with concepts across domains with a differentiable, domain-independent, first-order logic-based program executor. LEFT has an LLM interpreter that outputs a program represented in a general, logic-based reasoning language, which is shared across all domains and tasks. LEFT’s executor then executes the program with trainable domain-specific grounding modules. We show that LEFT flexibly learns concepts in four domains: 2D images, 3D scenes, human motions, and robotic manipulation. It exhibits strong reasoning ability in a wide variety of tasks, including those that are complex and not seen during training, and can be easily applied to new domains.

----

## [0] Recovering from Out-of-sample States via Inverse Dynamics in Offline Reinforcement Learning

**Authors**: *Ke Jiang, Jia-Yu Yao, Xiaoyang Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a0f7e9d9b42b26e5bfc9ba4c6e5287c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a0f7e9d9b42b26e5bfc9ba4c6e5287c-Abstract-Conference.html)

**Abstract**:

In this paper we deal with the state distributional shift problem commonly encountered in offline reinforcement learning during test, where the agent tends to take unreliable actions at out-of-sample (unseen) states. Our idea is to encourage the agent to follow the so called state recovery principle when taking actions, i.e., besides long-term return, the immediate consequences of the current action should also be taken into account and those capable of recovering the state distribution of the behavior policy are preferred. For this purpose, an inverse dynamics model is learned and employed to guide the state recovery behavior of the new policy. Theoretically, we show that the proposed method helps aligning the transited state distribution of the new policy with the offline dataset at out-of-sample states, without the need of  explicitly predicting the transited state distribution, which is usually difficult in high-dimensional and complicated environments. The effectiveness and feasibility of the proposed method is demonstrated with the state-of-the-art performance on the general offline RL benchmarks.

----

## [0] VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

**Authors**: *Li-Wei H. Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari D. Clifford, Roger G. Mark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

False arrhythmia alarms in intensive care units (ICUs) are a continuing problem despite considerable effort from industrial and academic algorithm developers. Of all life-threatening arrhythmias, ventricular tachycardia (VT) stands out as the most challenging arrhythmia to detect reliably. We introduce a new annotated VT alarm database, VTaC (Ventricular Tachycardia annotated alarms from ICUs)  consisting of  over 5,000 waveform recordings with VT alarms triggered by bedside monitors in the ICU. Each VT alarm waveform in the dataset has been labeled by at least two independent human expert annotators. The dataset encompasses data collected from ICUs in two major US hospitals and includes data from three leading bedside monitor manufacturers, providing a diverse and representative collection of alarm waveform data. Each waveform recording comprises at least two electrocardiogram (ECG) leads and one or more pulsatile waveforms, such as photoplethysmogram (PPG or PLETH) and arterial blood pressure (ABP) waveforms. We demonstrate the utility of this new benchmark dataset for the task of false arrhythmia alarm reduction, and present performance of multiple machine learning approaches, including conventional supervised machine learning, deep learning, semi-supervised learning, and generative approaches for the task of VT false alarm reduction.

----

## [0] TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation

**Authors**: *Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Yu Qiao, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html)

**Abstract**:

Training on large-scale datasets can boost the performance of video instance segmentation while the annotated datasets for VIS are hard to scale up due to the high labor cost. What we possess are numerous isolated filed-specific datasets, thus, it is appealing to jointly train models across the aggregation of datasets to enhance data volume and diversity. However, due to the heterogeneity in category space, as mask precision increase with the data volume, simply utilizing multiple datasets will dilute the attention of models on different taxonomy. Thus, increasing the data scale and enriching taxonomy space while improving classification precision is important. In this work, we analyze that providing extra taxonomy information can help models concentrate on specific taxonomy, and propose our model named Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation (TMT-VIS) to address this vital challenge. Specifically, we design a two-stage taxonomy aggregation module that first compiles taxonomy information from input videos and then aggregates these taxonomy priors into instance queries before the transformer decoder. We conduct extensive experimental evaluations on four popular and challenging benchmarks, including YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and UVO. Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all these benchmarks. These appealing and encouraging results demonstrate the effectiveness and generality of our proposed approach. The code and trained models will be publicly available.

----

## [0] Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities

**Authors**: *Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, Lorenzo Torresani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a65606fa1a6849450550325832036e5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a65606fa1a6849450550325832036e5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human activities are goal-oriented and hierarchical, comprising primary goals at the top level, sequences of steps and substeps in the middle, and atomic actions at the lowest level. Recognizing human activities thus requires relating atomic actions and steps to their functional objectives (what the actions contribute to) and modeling their sequential and hierarchical dependencies towards achieving the goals. Current activity recognition research has primarily focused on only the lowest levels of this hierarchy, i.e., atomic or low-level actions, often in trimmed videos with annotations spanning only a few seconds. In this work, we introduce Ego4D Goal-Step, a new set of annotations on the recently released Ego4D with a novel hierarchical taxonomy of goal-oriented activity labels. It provides dense annotations for 48K procedural step segments (430 hours) and high-level goal annotations for 2,807 hours of Ego4D videos. Compared to existing procedural video datasets, it is substantially larger in size, contains hierarchical action labels (goals - steps - substeps), and provides goal-oriented auxiliary information including natural language summary description, step completion status, and step-to-goal relevance information. We take a data-driven approach to build our taxonomy, resulting in dense step annotations that do not suffer from poor label-data alignment issues resulting from a taxonomy defined a priori. Through comprehensive evaluations and analyses, we demonstrate how Ego4D Goal-Step supports exploring various questions in procedural activity understanding, including goal inference, step prediction, hierarchical relation learning, and long-term temporal modeling.

----

## [0] The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter

**Authors**: *Ajay Jaiswal, Shiwei Liu, Tianlong Chen, Zhangyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a69ab48efcbb0153e72d458fb091969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a69ab48efcbb0153e72d458fb091969-Abstract-Conference.html)

**Abstract**:

Large pre-trained transformers are $\textit{show-stealer}$ in modern-day deep learning, and it becomes crucial to comprehend the parsimonious patterns that exist within them as they grow in scale. With exploding parameter counts, Lottery Ticket Hypothesis (LTH) and its variants, have lost their pragmatism in sparsifying them due to high computation and memory bottleneck of repetitive $\textit{train-prune-retrain}$ routine of iterative magnitude pruning (IMP) which worsens with increasing model size. In this paper, we comprehensively study $\textit{induced sparse patterns}$ across multiple large pre-trained vision and language transformers. We propose the existence of -- $\textbf{essential sparsity}$ defined with a $\textbf{sharp dropping point}$ beyond which the performance declines much faster w.r.t the rise of sparsity level, when we directly remove weights with the smallest magnitudes in $\textbf{one-shot}$. We also present an intriguing emerging phenomenon of $\textbf{abrupt sparsification}$ during the pre-training of BERT, i.e., BERT suddenly becomes heavily sparse in pre-training after certain iterations. Moreover, our observations also indicate a $\textbf{counter-intuitive}$ finding that BERT trained with a larger amount of pre-training data tends to have a better ability to condense knowledge in comparatively relatively fewer parameters. Lastly, we investigate the effect of the pre-training loss on essential sparsity and discover that self-supervised learning (SSL) objectives trigger stronger emergent sparsification properties than supervised learning (SL). All our codes will be publicly available.

----

## [0] Hypervolume Maximization: A Geometric View of Pareto Set Learning

**Authors**: *Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, Qingfu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html)

**Abstract**:

This paper presents a novel approach to multiobjective algorithms aimed at modeling the Pareto set using neural networks. Whereas previous methods mainly focused on identifying a finite number of solutions, our approach allows for the direct modeling of the entire Pareto set. Furthermore, we establish an equivalence between learning the complete Pareto set and maximizing the associated hypervolume, which enables the convergence analysis of hypervolume (as a new metric) for Pareto set learning. Specifically, our new analysis framework reveals the connection between the learned Pareto solution and its representation in a polar coordinate system. We evaluate our proposed approach on various benchmark problems and real-world problems, and the encouraging results make it a potentially viable alternative to existing multiobjective algorithms. Code is available at \url{https://github.com/xzhang2523/hvpsl/tree/master}.

----

## [0] Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics

**Authors**: *Koen Minartz, Yoeri Poels, Simon M. Koop, Vlado Menkovski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html)

**Abstract**:

Neural networks are emerging as a tool for scalable data-driven simulation of high-dimensional dynamical systems, especially in settings where numerical methods are infeasible or computationally expensive. Notably, it has been shown that incorporating domain symmetries in deterministic neural simulators can substantially improve their accuracy, sample efficiency, and parameter efficiency. However, to incorporate symmetries in probabilistic neural simulators that can simulate stochastic phenomena, we need a model that produces equivariant distributions over trajectories, rather than equivariant function approximations. In this paper, we propose Equivariant Probabilistic Neural Simulation (EPNS), a framework for autoregressive probabilistic modeling of equivariant distributions over system evolutions. We use EPNS to design models for a stochastic n-body system and stochastic cellular dynamics. Our results show that EPNS considerably outperforms existing neural network-based methods for probabilistic simulation. More specifically, we demonstrate that incorporating equivariance in EPNS improves simulation quality, data efficiency, rollout stability, and uncertainty quantification. We conclude that EPNS is a promising method for efficient and effective data-driven probabilistic simulation in a diverse range of domains.

----

## [0] Collaborative Alignment of NLP Models

**Authors**: *Fereshte Khani, Marco Túlio Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html)

**Abstract**:

Despite substantial advancements, Natural Language Processing (NLP) models often require post-training adjustments to enforce business rules, rectify undesired behavior, and align with user values.  These adjustments involve operationalizing "concepts"â€”dictating desired model responses to certain inputs. However, it's difficult for a single entity to enumerate and define all possible concepts, indicating a need for a multi-user, collaborative model alignment framework. Moreover, the exhaustive delineation of a concept is challenging, and an improper approach can create shortcuts or interfere with original data or other concepts.To address these challenges, we introduce CoAlign, a framework that enables multi-user interaction with the model, thereby mitigating individual limitations. CoAlign aids users in operationalizing their concepts using Large Language Models, and relying on the principle that NLP models exhibit simpler behaviors in local regions. Our main insight is learning a \emph{local} model for each concept, and a \emph{global} model to integrate the original data with all concepts.We then steer a large language model to generate instances within concept boundaries where local and global disagree.Our experiments show CoAlign is effective at helping multiple users operationalize concepts and avoid interference for a variety of scenarios, tasks, and models.

----

## [0] PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change

**Authors**: *Karthik Valmeekam, Matthew Marquez, Alberto Olmo Hernandez, Sarath Sreedharan, Subbarao Kambhampati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a92bcdede88c7afd108072faf5485c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a92bcdede88c7afd108072faf5485c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Generating plans of action, and reasoning about change have long been considered a core competence of intelligent agents. It is thus no surprise that evaluating the planning and reasoning capabilities of large language models (LLMs) has become a hot topic of research. Most claims about LLM planning capabilities are however based on common sense tasks–where it becomes hard to tell whether LLMs are planning or merely retrieving from their vast world knowledge.  There is a strong need for systematic and extensible planning benchmarks with sufficient diversity to evaluate whether LLMs have innate planning capabilities. Motivated by this, we propose PlanBench, an extensible benchmark suite based on the kinds of domains used in the automated planning community, especially in the International Planning Competition, to test the capabilities of LLMs in planning or reasoning about actions and change. PlanBench provides sufficient diversity in both the task domains and the specific planning capabilities. Our studies also show that on many critical capabilities–including plan generation–LLM performance falls quite short, even with the SOTA models. PlanBench can thus function as a useful marker of progress of LLMs in planning and reasoning.

----

## [0] Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model

**Authors**: *Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)

**Abstract**:

In the field of behavior-related brain computation, it is necessary to align raw neural signals against the drastic domain shift among them. A foundational framework within neuroscience research posits that trial-based neural population activities rely on low-dimensional latent dynamics, thus focusing on the latter greatly facilitates the alignment procedure. Despite this field's progress, existing methods ignore the intrinsic spatio-temporal structure during the alignment phase. Hence, their solutions usually lead to poor quality in latent dynamics structures and overall performance. To tackle this problem, we propose an alignment method ERDiff, which leverages the expressivity of the diffusion model to preserve the spatio-temporal structure of latent dynamics. Specifically, the latent dynamics structures of the source domain are first extracted by a diffusion model. Then, under the guidance of this diffusion model, such structures are well-recovered through a maximum likelihood alignment procedure in the target domain. We first demonstrate the effectiveness of our proposed method on a synthetic dataset. Then, when applied to neural recordings from the non-human primate motor cortex, under both cross-day and inter-subject settings, our method consistently manifests its capability of preserving the spatio-temporal structure of latent dynamics and outperforms existing approaches in alignment goodness-of-fit and neural decoding performance.

----

## [0] Closing the gap between the upper bound and lower bound of Adam's iteration complexity

**Authors**: *Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, Wei Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html)

**Abstract**:

Recently,  Arjevani et al. [1]  establish a lower bound of iteration complexity for the first-order optimization under an $L$-smooth condition and a bounded noise variance assumption. However, a thorough review of existing literature on Adam's convergence reveals a noticeable gap: none of them meet the above lower bound. In this paper, we close the gap by deriving a new convergence guarantee of Adam, with only an $L$-smooth condition and a bounded noise variance assumption. Our results remain valid across a broad spectrum of hyperparameters. Especially with properly chosen hyperparameters, we derive an upper bound of the iteration complexity of Adam and show that it meets the lower bound for first-order optimizers. To the best of our knowledge, this is the first to establish such a tight upper bound for Adam's convergence. Our proof utilizes novel techniques to handle the entanglement between momentum and adaptive learning rate and to convert the first-order term in the Descent Lemma to the gradient norm, which may be of independent interest.

----

## [0] Deep Patch Visual Odometry

**Authors**: *Zachary Teed, Lahav Lipson, Jia Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ac484b0f1a1719ad5be9aa8c8455fbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ac484b0f1a1719ad5be9aa8c8455fbb-Abstract-Conference.html)

**Abstract**:

We propose Deep Patch Visual Odometry (DPVO), a new deep learning system for monocular Visual Odometry (VO). DPVO uses a novel recurrent network architecture designed for tracking image patches across time. Recent approaches to VO have significantly improved the state-of-the-art accuracy by using deep networks to predict dense flow between video frames. However, using dense flow incurs a large computational cost, making these previous methods impractical for many use cases. Despite this, it has been assumed that dense flow is important as it provides additional redundancy against incorrect matches. DPVO disproves this assumption, showing that it is possible to get the best accuracy and efficiency by exploiting the advantages of sparse patch-based matching over dense flow. DPVO introduces a novel recurrent update operator for patch based correspondence coupled with differentiable bundle adjustment. On Standard benchmarks, DPVO outperforms all prior work, including the learning-based state-of-the-art VO-system (DROID) using a third of the memory while running 3x faster on average. Code is available at https://github.com/princeton-vl/DPVO

----

## [0] BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information

**Authors**: *Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, Deepak Ramachandran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Automated reasoning with unstructured natural text is a key requirement for many potential applications of NLP and for developing robust AI systems. Recently, Language Models (LMs) have demonstrated complex reasoning capacities even without any finetuning. However, existing evaluation for automated reasoning assumes access to a consistent and coherent set of information over which models reason. When reasoning in the real-world, the available information is frequently inconsistent or contradictory, and therefore models need to be equipped with a strategy to resolve such conflicts when they arise. One widely-applicable way of resolving conflicts is to impose preferences over information sources (e.g., based on source credibility or information recency) and adopt the source with higher preference. In this paper, we formulate the problem of reasoning with contradictory information guided by preferences over sources as the classical problem of defeasible reasoning, and develop a dataset called BoardgameQA for measuring the reasoning capacity of LMs in this setting. BoardgameQA also incorporates reasoning with implicit background knowledge, to better reflect reasoning problems in downstream applications. We benchmark various LMs on BoardgameQA and the results reveal a significant gap in the reasoning capacity of state-of-the-art LMs on this problem, showing that reasoning with conflicting information does not surface out-of-the-box in LMs. While performance can be improved with finetuning, it nevertheless remains poor.

----

## [0] Isometric Quotient Variational Auto-Encoders for Structure-Preserving Representation Learning

**Authors**: *In Huh, Changwook Jeong, Jae Myung Choe, Younggu Kim, Daesin Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html)

**Abstract**:

We study structure-preserving low-dimensional representation of a data manifold embedded in a high-dimensional observation space based on variational auto-encoders (VAEs). We approach this by decomposing the data manifold $\mathcal{M}$ as $\mathcal{M} = \mathcal{M} / G \times G$, where $G$ and $\mathcal{M} / G$ are a group of symmetry transformations and a quotient space of $\mathcal{M}$ up to $G$, respectively. From this perspective, we define the structure-preserving representation of such a manifold as a latent space $\mathcal{Z}$ which is isometrically isomorphic (i.e., distance-preserving) to the quotient space $\mathcal{M} / G$ rather $\mathcal{M}$ (i.e., symmetry-preserving). To this end, we propose a novel auto-encoding framework, named isometric quotient VAEs (IQVAEs), that can extract the quotient space from observations and learn the Riemannian isometry of the extracted quotient in an unsupervised manner. Empirical proof-of-concept experiments reveal that the proposed method can find a meaningful representation of the learned data and outperform other competitors for downstream tasks.

----

## [0] SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents

**Authors**: *Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang, Yongbin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b16688a2b053a1b01474ab5c78ce662-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b16688a2b053a1b01474ab5c78ce662-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Task-oriented dialogue (TOD) models have made significant progress in recent years. However, previous studies primarily focus on datasets written by annotators, which has resulted in a gap between academic research and real-world spoken con- versation scenarios. While several small-scale spoken TOD datasets are proposed to address robustness issues such as ASR errors, they ignore the unique challenges in spoken conversation. To tackle the limitations, we introduce SpokenWOZ, a large-scale speech-text dataset for spoken TOD, containing 8 domains, 203k turns, 5.7k dialogues and 249 hours of audios from human-to-human spoken conversations. SpokenWOZ further incorporates common spoken characteristics such as word-by-word processing and reasoning in spoken language. Based on these characteristics, we present cross-turn slot and reasoning slot detection as new challenges. We conduct experiments on various baselines, including text-modal models, newly proposed dual-modal models, and LLMs, e.g., ChatGPT. The results show that the current models still have substantial room for improvement in spoken conversation, where the most advanced dialogue state tracker only achieves 25.65% in joint goal accuracy and the SOTA end-to-end model only correctly completes the user request in 52.1% of dialogues. Our dataset, code, and leaderboard are available at https://spokenwoz.github.io/SpokenWOZ-github.io/.

----

## [0] Fast Partitioned Learned Bloom Filter

**Authors**: *Atsuki Sato, Yusuke Matsui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html)

**Abstract**:

A Bloom filter is a memory-efficient data structure for approximate membership queries used in numerous fields of computer science.Recently, learned Bloom filters that achieve better memory efficiency using machine learning models have attracted attention.One such filter, the partitioned learned Bloom filter (PLBF), achieves excellent memory efficiency.However, PLBF requires a $\mathcal{O}(N^3k)$ time complexity to construct the data structure, where $N$ and $k$ are the hyperparameters of PLBF.One can improve memory efficiency by increasing $N$, but the construction time becomes extremely long.Thus, we propose two methods that can reduce the construction time while maintaining the memory efficiency of PLBF.First, we propose fast PLBF, which can construct the same data structure as PLBF with a smaller time complexity $\mathcal{O}(N^2k)$.Second, we propose fast PLBF++, which can construct the data structure with even smaller time complexity $\mathcal{O}(Nk\log N + Nk^2)$.Fast PLBF++ does not necessarily construct the same data structure as PLBF.Still, it is almost as memory efficient as PLBF, and it is proved that fast PLBF++ has the same data structure as PLBF when the distribution satisfies a certain constraint.Our experimental results from real-world datasets show that (i) fast PLBF and fast PLBF++ can construct the data structure up to 233 and 761 times faster than PLBF, (ii) fast PLBF can achieve the same memory efficiency as PLBF, and (iii) fast PLBF++ can achieve almost the same memory efficiency as PLBF.The codes are available at [this https URL](https://github.com/atsukisato/FastPLBF).

----



[Go to the previous page](NIPS-2023-list1600.md)

[Go to the next page](NIPS-2023-list1602.md)

[Go to the catalog section](README.md)