## [1600] Emergent Communication in Interactive Sketch Question Answering

**Authors**: *Zixing Lei, Yiming Zhang, Yuxin Xiong, Siheng Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/746cf1bc2337700f7f0c35c7b02638cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/746cf1bc2337700f7f0c35c7b02638cc-Abstract-Conference.html)

**Abstract**:

Vision-based emergent communication (EC) aims to learn to communicate through sketches and demystify the evolution of human communication. Ironically, previous works neglect multi-round interaction, which is indispensable in human communication. To fill this gap, we first introduce a novel Interactive Sketch Question Answering (ISQA) task, where two collaborative players are interacting through sketches to answer a question about an image. To accomplish this task, we design a new and efficient interactive EC system, which can achieve an effective balance among three evaluation factors, including the question answering accuracy, drawing complexity and human interpretability. Our experimental results demonstrate that multi-round interactive mechanism facilitates tar- geted and efficient communication between intelligent agents. The code will be released.

----

## [1601] Computing Approximate 𝓁p Sensitivities

**Authors**: *Swati Padmanabhan, David P. Woodruff, Richard Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/746d2254f6892f3badadb07cc9c0f0da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/746d2254f6892f3badadb07cc9c0f0da-Abstract-Conference.html)

**Abstract**:

Recent works in dimensionality reduction for regression tasks have introduced the notion of sensitivity, an estimate of the importance of a specific datapoint in a dataset, offering provable guarantees on the quality of the approximation after removing low-sensitivity datapoints via subsampling. However, fast algorithms for approximating sensitivities, which we show is equivalent to approximate regression, are known for only the $\ell_2$ setting, in which they are popularly termed leverage scores. In this work, we provide the first efficient algorithms for approximating $\ell_p$ sensitivities and other summary statistics of a given matrix. In particular, for a given $n \times d$ matrix, we compute $\alpha$-approximation to its $\ell_1$ sensitivities at the cost of $n/\alpha$ sensitivity computations. For estimating the total $\ell_p$ sensitivity (i.e. the sum of $\ell_p$ sensitivities), we provide an algorithm based on importance sampling of $\ell_p$ Lewis weights, which computes a constant factor approximation at the cost of roughly $\sqrt{d}$ sensitivity computations, with no polynomial dependence on $n$. Furthermore, we estimate the maximum $\ell_1$ sensitivity up to a $\sqrt{d}$ factor in $O(d)$ sensitivity computations. We also generalize these results to $\ell_p$ norms.  Lastly, we experimentally show that for a wide class of structured matrices in real-world datasets, the total sensitivity can be quickly approximated and is significantly smaller than the theoretical prediction, demonstrating that real-world datasets have on average low intrinsic effective dimensionality.

----

## [1602] Automated Classification of Model Errors on ImageNet

**Authors**: *Momchil Peychev, Mark Niklas Müller, Marc Fischer, Martin T. Vechev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7480ed13740773505262791131c12b89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7480ed13740773505262791131c12b89-Abstract-Conference.html)

**Abstract**:

While the ImageNet dataset has been driving computer vision research over the past decade, significant label noise and ambiguity have made top-1 accuracy an insufficient measure of further progress. To address this, new label-sets and evaluation protocols have been proposed for ImageNet showing that state-of-the-art models already achieve over 95% accuracy and shifting the focus on investigating why the remaining errors persist.Recent work in this direction employed a panel of experts to manually categorize all remaining classification errors for two selected models. However, this process is time-consuming, prone to inconsistencies, and requires trained experts, making it unsuitable for regular model evaluation thus limiting its utility. To overcome these limitations, we propose the first automated error classification framework, a valuable tool to study how modeling choices affect error distributions. We use our framework to comprehensively evaluate the error distribution of over 900 models. Perhaps surprisingly, we find that across model architectures, scales, and pre-training corpora, top-1 accuracy is a strong predictor for the portion of all error types. In particular, we observe that the portion of severe errors drops significantly with top-1 accuracy indicating that, while it underreports a model's true performance, it remains a valuable performance metric.We release all our code at https://github.com/eth-sri/automated-error-analysis.

----

## [1603] Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent

**Authors**: *Jihao Andreas Lin, Javier Antorán, Shreyas Padhy, David Janz, José Miguel Hernández-Lobato, Alexander Terenin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7482e8ce4139df1a2d8195a0746fa713-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7482e8ce4139df1a2d8195a0746fa713-Abstract-Conference.html)

**Abstract**:

Gaussian processes are a powerful framework for quantifying uncertainty and for sequential decision-making but are limited by the requirement of solving linear systems. In general, this has a cubic cost in dataset size and is sensitive to conditioning. We explore stochastic gradient algorithms as a computationally efficient method of approximately solving these linear systems: we develop low-variance optimization objectives for sampling from the posterior and extend these to inducing points. Counterintuitively, stochastic gradient descent often produces accurate predictions, even in cases where it does not converge quickly to the optimum. We explain this through a spectral characterization of the implicit bias from non-convergence. We show that stochastic gradient descent produces predictive distributions close to the true posterior both in regions with sufficient data coverage, and in regions sufficiently far away from the data. Experimentally, stochastic gradient descent achieves state-of-the-art performance on sufficiently large-scale or ill-conditioned regression tasks. Its uncertainty estimates match the performance of significantly more expensive baselines on a large-scale Bayesian~optimization~task.

----

## [1604] Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning

**Authors**: *Yihua Zhang, Yimeng Zhang, Aochuan Chen, Jinghan Jia, Jiancheng Liu, Gaowen Liu, Mingyi Hong, Shiyu Chang, Sijia Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html)

**Abstract**:

Massive data is often considered essential for deep learning applications, but it also incurs significant computational and infrastructural costs. Therefore, dataset pruning (DP) has emerged as an effective way to improve data efficiency by identifying and removing redundant training samples without sacrificing performance. In this work, we aim to address the problem of DP for transfer learning, i.e., how to prune a source dataset for improved pretraining efficiency and lossless finetuning accuracy on downstream target tasks. To our best knowledge, the problem of DP for transfer learning remains open, as previous studies have primarily addressed DP and transfer learning as separate problems. By contrast, we establish a unified viewpoint to integrate DP with transfer learning and find that existing DP methods are not suitable for the transfer learning paradigm. We then propose two new DP methods, label mapping and feature mapping, for supervised and self-supervised pretraining settings respectively, by revisiting the DP problem through the lens of source-target domain mapping. Furthermore, we demonstrate the effectiveness of our approach on numerous transfer learning tasks. We show that source data classes can be pruned by up to $40\%\sim 80\%$ without sacrificing the downstream performance, resulting in a significant $2\sim 5\times$ speed-up during the pretraining stage. Besides, our proposal exhibits broad applicability and can improve other computationally intensive transfer learning techniques, such as adversarial pretraining.

----

## [1605] On Slicing Optimality for Mutual Information

**Authors**: *Ammar Fayad, Majd Ibrahim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/749b64078a64fa5734a49fb40bc9fd65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/749b64078a64fa5734a49fb40bc9fd65-Abstract-Conference.html)

**Abstract**:

Measuring dependence between two random variables is of great importance in various domains but is difficult to compute in today's complex environments with high-dimensional data. Recently, slicing methods have shown to be a scalable approach to measuring mutual information (MI) between high-dimensional variables by projecting these variables into one-dimensional spaces. Unfortunately, these methods use uniform distributions of slicing directions, which generally discard informative features between variables and thus lead to inaccurate quantification of dependence. In this paper, we propose a principled framework that searches for an \textit{optimal} distribution of slices for MI. Importantly, we answer theoretical questions about finding the optimal slicing distribution in the context of MI and develop corresponding theoretical analyses. We also develop a practical algorithm, connecting our theoretical results with modern machine learning frameworks. Through comprehensive experiments in benchmark domains, we demonstrate significant gains in our information measure than state-of-the-art baselines.

----

## [1606] OpenIllumination: A Multi-Illumination Dataset for Inverse Rendering Evaluation on Real Objects

**Authors**: *Isabella Liu, Linghao Chen, Ziyang Fu, Liwen Wu, Haian Jin, Zhong Li, Chin Ming Ryan Wong, Yi Xu, Ravi Ramamoorthi, Zexiang Xu, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce OpenIllumination, a real-world dataset containing over 108K images of 64 objects with diverse materials, captured under 72 camera views and a large number of different illuminations. For each image in the dataset, we provide accurate camera parameters, illumination ground truth, and foreground segmentation masks. Our dataset enables the quantitative evaluation of most inverse rendering and material decomposition methods for real objects. We examine several state-of-the-art inverse rendering methods on our dataset and compare their performances. The dataset and code can be found on the project page: https://oppo-us-research.github.io/OpenIllumination.

----

## [1607] Language Model Tokenizers Introduce Unfairness Between Languages

**Authors**: *Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html)

**Abstract**:

Recent language models have shown impressive multilingual performance, even when not explicitly trained for it.Despite this, there are concerns about the quality of their outputs across different languages.In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked.The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases.These disparities persist even for tokenizers that are intentionally trained for multilingual support.Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs.This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models.Therefore, we make the case that we should train future language models using multilingually fair subword tokenizers.

----

## [1608] Interaction Measures, Partition Lattices and Kernel Tests for High-Order Interactions

**Authors**: *Zhaolu Liu, Robert L. Peach, Pedro A. M. Mediano, Mauricio Barahona*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74f11936d6144eae43730e1a49365479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74f11936d6144eae43730e1a49365479-Abstract-Conference.html)

**Abstract**:

Models that rely solely on pairwise relationships often fail to capture the complete statistical structure of the complex multivariate data found in diverse domains, such as socio-economic, ecological, or biomedical systems. Non-trivial dependencies between groups of more than two variables can play a significant role in the analysis and modelling of such systems, yet extracting such high-order interactions from data remains challenging. Here, we introduce a hierarchy of $d$-order ($d \geq 2$) interaction measures, increasingly inclusive of possible factorisations of the joint probability distribution, and define non-parametric, kernel-based tests to establish systematically the statistical significance of $d$-order interactions. We also establish mathematical links with lattice theory, which elucidate the derivation of the interaction measures and their composite permutation tests; clarify the connection of simplicial complexes with kernel matrix centring; and provide a means to enhance computational efficiency. We illustrate our results numerically with validations on synthetic data, and through an application to neuroimaging data.

----

## [1609] Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?

**Authors**: *Haitao Mao, Zhikai Chen, Wei Jin, Haoyu Han, Yao Ma, Tong Zhao, Neil Shah, Jiliang Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74f1edadbdf495e7258ee8db7b1d3acd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74f1edadbdf495e7258ee8db7b1d3acd-Abstract-Conference.html)

**Abstract**:

Recent studies on Graph Neural Networks(GNNs) provide both empirical and theoretical evidence supporting their effectiveness in capturing structural patterns on both homophilic and certain heterophilic graphs. Notably, most real-world homophilic and heterophilic graphs are comprised of a mixture of nodes in both homophilic and heterophilic structural patterns, exhibiting a structural disparity. However, the analysis of GNN performance with respect to nodes exhibiting different structural patterns, e.g., homophilic nodes in heterophilic graphs, remains rather limited. In the present study, we provide evidence that Graph Neural Networks(GNNs) on node classification typically perform admirably on homophilic nodes within homophilic graphs and heterophilic nodes within heterophilic graphs while struggling on the opposite node set, exhibiting a performance disparity. We theoretically and empirically identify effects of GNNs on testing nodes exhibiting distinct structural patterns. We then propose a rigorous, non-i.i.d PAC-Bayesian generalization bound for GNNs, revealing reasons for the performance disparity, namely the aggregated feature distance and homophily ratio difference between training and testing nodes. Furthermore, we demonstrate the practical implications of our new findings via (1) elucidating the effectiveness of deeper GNNs; and (2) revealing an over-looked distribution shift factor on graph out-of-distribution problem and proposing a new scenario accordingly.

----

## [1610] Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment

**Authors**: *Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fa3651b41560e1c7555e0958c70333-Abstract-Conference.html)

**Abstract**:

Spatio-Temporal Graph (STG) forecasting is a fundamental task in many real-world applications. Spatio-Temporal Graph Neural Networks have emerged as the most popular method for STG forecasting, but they often struggle with temporal out-of-distribution (OoD) issues and dynamic spatial causation. In this paper, we propose a novel framework called CaST to tackle these two challenges via causal treatments. Concretely, leveraging a causal lens, we first build a structural causal model to decipher the data generation process of STGs. To handle the temporal OoD issue, we employ the back-door adjustment by a novel disentanglement block to separate the temporal environments from input data. Moreover, we utilize the front-door adjustment and adopt edge-level convolution to model the ripple effect of causation. Experiments results on three real-world datasets demonstrate the effectiveness of CaST, which consistently outperforms existing methods with good interpretability. Our source code is available at https://github.com/yutong-xia/CaST.

----

## [1611] Greedy Poisson Rejection Sampling

**Authors**: *Gergely Flamich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fb3d526c7d8bd0c3e4b71704bb5abf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fb3d526c7d8bd0c3e4b71704bb5abf-Abstract-Conference.html)

**Abstract**:

One-shot channel simulation is a fundamental data compression problem concerned with encoding a single sample from a target distribution $Q$ using a coding distribution $P$ using as few bits as possible on average. Algorithms that solve this problem find applications in neural data compression and differential privacy and can serve as a more efficient and natural alternative to quantization-based methods. Unfortunately, existing solutions are too slow or have limited applicability, preventing their widespread adaptation. In this paper, we conclusively solve one-shot channel simulation for one-dimensional problems where the target-proposal density ratio is unimodal by describing an algorithm with optimal runtime. We achieve this by constructing a rejection sampling procedure equivalent to greedily searching over the points of a Poisson process. Hence, we call our algorithm greedy Poisson rejection sampling (GPRS) and analyze the correctness and time complexity of several of its variants. Finally, we empirically verify our theorems, demonstrating that GPRS significantly outperforms the current state-of-the-art method, A* coding.

----

## [1612] Uncertainty Quantification via Neural Posterior Principal Components

**Authors**: *Elias Nehme, Omer Yair, Tomer Michaeli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification is crucial for the deployment of image restoration models in safety-critical domains, like autonomous driving and biological imaging. To date, methods for uncertainty visualization have mainly focused on per-pixel estimates. Yet, a heatmap of per-pixel variances is typically of little practical use, as it does not capture the strong correlations between pixels. A more natural measure of uncertainty corresponds to the variances along the principal components (PCs) of the posterior distribution. Theoretically, the PCs can be computed by applying PCA on samples generated from a conditional generative model for the input image. However, this requires generating a very large number of samples at test time, which is painfully slow with the current state-of-the-art (diffusion) models. In this work, we present a method for predicting the PCs of the posterior distribution for any input image, in a single forward pass of a neural network. Our method can either wrap around a pre-trained model that was trained to minimize the mean square error (MSE), or can be trained from scratch to output both a predicted image and the posterior PCs. We showcase our method on multiple inverse problems in imaging, including denoising, inpainting, super-resolution, and biological image-to-image translation. Our method reliably conveys instance-adaptive uncertainty directions, achieving uncertainty quantification comparable with posterior samplers while being orders of magnitude faster. Code and examples are available on our webpage.

----

## [1613] Deep Reinforcement Learning with Plasticity Injection

**Authors**: *Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75101364dc3aa7772d27528ea504472b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75101364dc3aa7772d27528ea504472b-Abstract-Conference.html)

**Abstract**:

A growing body of evidence suggests that neural networks employed in deep reinforcement learning (RL) gradually lose their plasticity, the ability to learn from new data; however, the analysis and mitigation of this phenomenon is hampered by the complex relationship between plasticity, exploration, and performance in RL. This paper introduces plasticity injection, a minimalistic intervention that increases the network plasticity without changing the number of trainable parameters or biasing the predictions. The applications of this intervention are two-fold: first, as a diagnostic tool â€” if injection increases the performance, we may conclude that an agent's network was losing its plasticity. This tool allows us to identify a subset of Atari environments where the lack of plasticity causes performance plateaus, motivating future studies on understanding and combating plasticity loss. Second, plasticity injection can be used to improve the computational efficiency of RL training if the agent has to re-learn from scratch due to exhausted plasticity or by growing the agent's network dynamically without compromising performance. The results on Atari show that plasticity injection attains stronger performance compared to alternative methods while being computationally efficient.

----

## [1614] StreamNet: Memory-Efficient Streaming Tiny Deep Learning Inference on the Microcontroller

**Authors**: *Hong-Sheng Zheng, Yu-Yuan Liu, Chen-Fong Hsu, Tsung Tai Yeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7526508f11bbe0a123af62b9dab1fbe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7526508f11bbe0a123af62b9dab1fbe1-Abstract-Conference.html)

**Abstract**:

With the emerging Tiny Machine Learning (TinyML) inference applications, there is a growing interest when deploying TinyML models on the low-power Microcontroller Unit (MCU). However, deploying TinyML models on MCUs reveals several challenges due to the MCU’s resource constraints, such as small flash memory, tight SRAM memory budget, and slow CPU performance. Unlike typical layer-wise inference, patch-based inference reduces the peak usage of SRAM memory on MCUs by saving small patches rather than the entire tensor in the SRAM memory. However, the processing of patch-based inference tremendously increases the amount of MACs against the layer-wise method. Thus, this notoriously computational overhead makes patch-based inference undesirable on MCUs. This work designs StreamNet that employs the stream buffer to eliminate the redundant computation of patch-based inference. StreamNet uses 1D and 2D streaming processing and provides an parameter selection algorithm that automatically improve the performance of patch-based inference with minimal requirements on the MCU’s SRAM memory space. In 10 TinyML models, StreamNet-2D achieves a geometric mean of 7.3X speedup and saves 81\% of MACs over the state-of-the-art patch-based inference.

----

## [1615] Estimating and Controlling for Equalized Odds via Sensitive Attribute Predictors

**Authors**: *Beepul Bharti, Paul H. Yi, Jeremias Sulam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/752820c79b4ebb72809014bdfdedd603-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/752820c79b4ebb72809014bdfdedd603-Abstract-Conference.html)

**Abstract**:

As the use of machine learning models in real world high-stakes decision settings continues to grow, it is highly important that we are able to audit and control for any potential fairness violations these models may exhibit towards certain groups. To do so, one naturally requires access to sensitive attributes, such as demographics, biological sex, or other potentially sensitive features that determine group membership. Unfortunately, in many settings, this information is often unavailable. In this work we study the well known equalized odds (EOD) definition of fairness. In a setting without sensitive attributes, we first provide tight and computable upper bounds for the EOD violation of a predictor. These bounds precisely reflect the worst possible EOD violation. Second, we demonstrate how one can provably control the worst-case EOD by a new post-processing correction method. Our results characterize when directly controlling for EOD with respect to the predicted sensitive attributes is -- and when is not -- optimal when it comes to controlling worst-case EOD. Our results hold under assumptions that are milder than previous works, and we illustrate these results with experiments on synthetic and real datasets.

----

## [1616] Segment Any Point Cloud Sequences by Distilling Vision Foundation Models

**Authors**: *Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html)

**Abstract**:

Recent advancements in vision foundation models (VFMs) have opened up new possibilities for versatile and efficient visual perception. In this work, we introduce Seal, a novel framework that harnesses VFMs for segmenting diverse automotive point cloud sequences. Seal exhibits three appealing properties: i) Scalability: VFMs are directly distilled into point clouds, obviating the need for annotations in either 2D or 3D during pretraining. ii) Consistency: Spatial and temporal relationships are enforced at both the camera-to-LiDAR and point-to-segment regularization stages, facilitating cross-modal representation learning. iii) Generalizability: Seal enables knowledge transfer in an off-the-shelf manner to downstream tasks involving diverse point clouds, including those from real/synthetic, low/high-resolution, large/small-scale, and clean/corrupted datasets. Extensive experiments conducted on eleven different point cloud datasets showcase the effectiveness and superiority of Seal. Notably, Seal achieves a remarkable 45.0% mIoU on nuScenes after linear probing, surpassing random initialization by 36.9% mIoU and outperforming prior arts by 6.1% mIoU. Moreover, Seal demonstrates significant performance gains over existing methods across 20 different few-shot fine-tuning tasks on all eleven tested point cloud datasets. The code is available at this link.

----

## [1617] Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings

**Authors**: *Giovanni de Felice, John Yannis Goulermas, Vladimir V. Gusev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html)

**Abstract**:

Kernel design is a pivotal but challenging aspect of time series analysis, especially in the context of small datasets. In recent years, Reservoir Computing (RC) has emerged as a powerful tool to compare time series based on the underlying dynamics of the generating process rather than the observed data. However, the performance of RC highly depends on the hyperparameter setting, which is hard to interpret and costly to optimize because of the recurrent nature of RC. Here, we present a new kernel for time series based on the recently established equivalence between reservoir dynamics and Nonlinear Vector AutoRegressive (NVAR) processes. The kernel is non-recurrent and depends on a small set of meaningful hyperparameters, for which we suggest an effective heuristic. We demonstrate excellent performance on a wide range of real-world classification tasks, both in terms of accuracy and speed. This further advances the understanding of RC representation learning models and extends the typical use of the NVAR framework to kernel design and representation of real-world time series data.

----

## [1618] SPA: A Graph Spectral Alignment Perspective for Domain Adaptation

**Authors**: *Zhiqing Xiao, Haobo Wang, Ying Jin, Lei Feng, Gang Chen, Fei Huang, Junbo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/754e80f98b2a141942f45a0eeb258a3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/754e80f98b2a141942f45a0eeb258a3c-Abstract-Conference.html)

**Abstract**:

Unsupervised domain adaptation (UDA) is a pivotal form in machine learning to extend the in-domain model to the distinctive target domains where the data distributions differ. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. In this work, we introduce a novel graph SPectral Alignment (SPA) framework to tackle the tradeoff. The core of our method is briefly condensed as follows: (i)-by casting the DA problem to graph primitives, SPA composes a coarse graph alignment mechanism with a novel spectral regularizer towards aligning the domain graphs in eigenspaces; (ii)-we further develop a fine-grained message propagation module --- upon a novel neighbor-aware self-training mechanism --- in order for enhanced discriminability in the target domain. On standardized benchmarks, the extensive experiments of SPA demonstrate that its performance has surpassed the existing cutting-edge DA methods. Coupled with dense model analysis, we conclude that our approach indeed possesses superior efficacy, robustness, discriminability, and transferability. Code and data are available at: https://github.com/CrownX/SPA.

----

## [1619] CosNet: A Generalized Spectral Kernel Network

**Authors**: *Yanfang Xue, Pengfei Fang, Jinyue Tian, Shipeng Zhu, Hui Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html)

**Abstract**:

Complex-valued representation exists inherently in the time-sequential data that can be derived from the integration of harmonic waves. The non-stationary spectral kernel, realizing a complex-valued feature mapping, has shown its potential to analyze the time-varying statistical characteristics of the time-sequential data, as a result of the modeling frequency parameters. However, most existing spectral kernel-based methods eliminate the imaginary part, thereby limiting the representation power of the spectral kernel. To tackle this issue, we propose a generalized spectral kernel network, namely, \underline{Co}mplex-valued \underline{s}pectral kernel \underline{Net}work (CosNet), which includes spectral kernel mapping generalization (SKMG) module and complex-valued spectral kernel embedding (CSKE) module. Concretely, the SKMG module is devised to generalize the spectral kernel mapping in the real number domain to the complex number domain, recovering the inherent complex-valued representation for the real-valued data. Then a following CSKE module is further developed to combine the complex-valued spectral kernels and neural networks to effectively capture long-range or periodic relations of the data. Along with the CosNet, we study the effect of the complex-valued spectral kernel mapping via theoretically analyzing the bound of covering number and generalization error. Extensive experiments demonstrate that CosNet performs better than the mainstream kernel methods and complex-valued neural networks.

----

## [1620] A Theory of Unsupervised Translation Motivated by Understanding Animal Communication

**Authors**: *Shafi Goldwasser, David F. Gruber, Adam Tauman Kalai, Orr Paradise*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7571c9d44179c7988178593c5b62a9b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7571c9d44179c7988178593c5b62a9b6-Abstract-Conference.html)

**Abstract**:

Neural networks are capable of translating between languagesâ€”in some cases even between two languages where there is little or no access to parallel translations, in what is known as Unsupervised Machine Translation (UMT). Given this progress, it is intriguing to ask whether machine learning tools can ultimately enable understanding animal communication, particularly that of highly intelligentanimals. We propose a theoretical framework for analyzing UMT when no parallel translations are available and when it cannot be assumed that the source and target corpora address related subject domains or posses similar linguistic structure. Weexemplify this theory with two stylized models of language, for which our framework provides bounds on necessary sample complexity; the bounds are formally proven and experimentally verified on synthetic data. These bounds show that the error rates are inversely related to the language complexity and amount of common ground. This suggests that unsupervised translation of animal communication may be feasible if the communication system is sufficiently complex.

----

## [1621] Adversarial Self-Training Improves Robustness and Generalization for Gradual Domain Adaptation

**Authors**: *Lianghe Shi, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75b0edb869e2cd509d64d0e8ff446bc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75b0edb869e2cd509d64d0e8ff446bc1-Abstract-Conference.html)

**Abstract**:

Gradual Domain Adaptation (GDA), in which the learner is provided with additional intermediate domains, has been theoretically and empirically studied in many contexts. Despite its vital role in security-critical scenarios, the adversarial robustness of the GDA model remains unexplored. In this paper, we adopt the effective gradual self-training method and replace vanilla self-training with adversarial self-training (AST). AST first predicts labels on the unlabeled data and then adversarially trains the model on the pseudo-labeled distribution. Intriguingly, we find that gradual AST improves not only adversarial accuracy but also clean accuracy on the target domain. We reveal that this is because adversarial training (AT) performs better than standard training when the pseudo-labels contain a portion of incorrect labels. Accordingly, we first present the generalization error bounds for gradual AST in a multiclass classification setting. We then use the optimal value of the Subset Sum Problem to bridge the standard error on a real distribution and the adversarial error on a pseudo-labeled distribution. The result indicates that AT may obtain a tighter bound than standard training on data with incorrect pseudo-labels. We further present an example of a conditional Gaussian distribution to provide more insights into why gradual AST can improve the clean accuracy for GDA.

----

## [1622] TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials

**Authors**: *Guillem Simeon, Gianni De Fabritiis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75c2ec5f98d7b2f50ad68033d2c07086-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75c2ec5f98d7b2f50ad68033d2c07086-Abstract-Conference.html)

**Abstract**:

The development of efficient machine learning models for molecular systems representation is becoming crucial in scientific research. We introduce TensorNet, an innovative O(3)-equivariant message-passing neural network architecture that leverages Cartesian tensor representations. By using Cartesian tensor atomic embeddings, feature mixing is simplified through matrix product operations. Furthermore, the cost-effective decomposition of these tensors into rotation group irreducible representations allows for the separate processing of scalars, vectors, and tensors when necessary. Compared to higher-rank spherical tensor models, TensorNet demonstrates state-of-the-art performance with significantly fewer parameters. For small molecule potential energies, this can be achieved even with a single interaction layer. As a result of all these properties, the model's computational cost is substantially decreased. Moreover, the accurate prediction of vector and tensor molecular quantities on top of potential energies and forces is possible. In summary, TensorNet's framework opens up a new space for the design of state-of-the-art equivariant models.

----

## [1623] Multi-Player Zero-Sum Markov Games with Networked Separable Interactions

**Authors**: *Chanwoo Park, Kaiqing Zhang, Asuman E. Ozdaglar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75c411b0a06fa9e78f2a516b57b2ce62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75c411b0a06fa9e78f2a516b57b2ce62-Abstract-Conference.html)

**Abstract**:

We study a new class of  Markov games, \textit{(multi-player) zero-sum Markov Games} with {\it Networked separable interactions} (zero-sum NMGs), to model the local interaction structure in non-cooperative multi-agent sequential decision-making. We define a zero-sum NMG as a model where {the payoffs of the auxiliary games associated with each state are zero-sum and} have some separable (i.e., polymatrix) structure across the neighbors over some interaction network. We first identify the necessary and sufficient conditions under which an MG can be presented as a zero-sum NMG, and show that the set of Markov coarse correlated equilibrium (CCE) collapses to the set of Markov Nash equilibrium (NE) in these games, in that the {product of} per-state marginalization of the former for all players yields the latter. Furthermore,  we show that finding approximate Markov \emph{stationary}  CCE  in infinite-horizon discounted zero-sum NMGs is \texttt{PPAD}-hard, unless the underlying network has a ``star topology''. Then, we propose fictitious-play-type dynamics, the classical learning dynamics in normal-form games, for zero-sum NMGs, and establish convergence guarantees to Markov stationary NE under a star-shaped network structure. Finally, in light of the hardness result, we focus on computing a Markov \emph{non-stationary} NE and provide finite-iteration guarantees for a series of value-iteration-based algorithms. We also provide numerical experiments to corroborate our theoretical results.

----

## [1624] Continuous-Time Functional Diffusion Processes

**Authors**: *Giulio Franzese, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75cd262a3fd8e76e37bb7941db141a1d-Abstract-Conference.html)

**Abstract**:

We introduce Functional Diffusion Processes (FDPs), which generalize score-based diffusion models to infinite-dimensional function spaces. FDPs require a new mathematical framework to describe the forward and backward dynamics, and several extensions to derive practical training objectives. These include infinite-dimensional versions of Girsanov theorem, in order to be able to compute an ELBO, and of the sampling theorem, in order to guarantee that functional evaluations in a countable set of points are equivalent to infinite-dimensional functions. We use FDPs to build a new breed of generative models in function spaces, which do not require specialized network architectures, and that can work with any kind of continuous data.Our results on real data show that FDPs achieve high-quality image generation, using a simple MLP architecture with orders of magnitude fewer parameters than existing diffusion models.

----

## [1625] Knowledge-based in silico models and dataset for the comparative evaluation of mammography AI for a range of breast characteristics, lesion conspicuities and doses

**Authors**: *Elena Sizikova, Niloufar Saharkhiz, Diksha Sharma, Miguel A. Lago, Berkman Sahiner, Jana G. Delfino, Aldo Badano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75d0956c9594f47bfb86a07bef58d4b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/75d0956c9594f47bfb86a07bef58d4b0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

To generate evidence regarding the safety and efficacy of artificial intelligence (AI) enabled medical devices, AI models need to be evaluated on a diverse population of patient cases, some of which may not be readily available. We propose an evaluation approach for testing medical imaging AI models that relies on in silico imaging pipelines in which stochastic digital models of human anatomy (in object space) with and without pathology are imaged using a digital replica imaging acquisition system to generate realistic synthetic image datasets. Here, we release M-SYNTH, a dataset of cohorts with four breast fibroglandular density distributions imaged at different exposure levels using Monte Carlo x-ray simulations with the publicly available Virtual Imaging Clinical Trial for Regulatory Evaluation (VICTRE) toolkit. We utilize the synthetic dataset to analyze AI model performance and find that model performance decreases with increasing breast density and increases with higher mass density, as expected. As exposure levels decrease, AI model performance drops with the highest performance achieved at exposure levels lower than the nominal recommended dose for the breast type.

----

## [1626] Is Distance Matrix Enough for Geometric Deep Learning?

**Authors**: *Zian Li, Xiyuan Wang, Yinan Huang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are often used for tasks involving the 3D geometry of a given graph, such as molecular dynamics simulation. While incorporating Euclidean distance into Message Passing Neural Networks (referred to as Vanilla DisGNN) is a straightforward way to learn the geometry, it has been demonstrated that Vanilla DisGNN is geometrically incomplete. In this work, we first construct families of novel and symmetric geometric graphs that Vanilla DisGNN cannot distinguish even when considering all-pair distances, which greatly expands the existing counterexample families. Our counterexamples show the inherent limitation of Vanilla DisGNN to capture symmetric geometric structures. We then propose $k$-DisGNNs, which can effectively exploit the rich geometry contained in the distance matrix. We demonstrate the high expressive power of $k$-DisGNNs from three perspectives: 1. They can learn high-order geometric information that cannot be captured by Vanilla DisGNN. 2. They can unify some existing well-designed geometric models. 3. They are universal function approximators from geometric graphs to scalars (when $k\geq 2$) and vectors (when $k\geq 3$). Most importantly, we establish a connection between geometric deep learning (GDL) and traditional graph representation learning (GRL), showing that those highly expressive GNN models originally designed for GRL can also be applied to GDL with impressive performance, and that existing complicated, equivariant models are not the only solution. Experiments verify our theory. Our $k$-DisGNNs achieve many new state-of-the-art results on MD17.

----

## [1627] Optimized Covariance Design for AB Test on Social Network under Interference

**Authors**: *Qianyi Chen, Bo Li, Lu Deng, Yong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html)

**Abstract**:

Online A/B tests have become increasingly popular and important for social platforms. However, accurately estimating the global average treatment effect (GATE) has proven to be challenging due to network interference, which violates the Stable Unit Treatment Value Assumption (SUTVA) and poses great challenge to experimental design. Existing network experimental design research was mostly based on the unbiased Horvitz-Thompson (HT) estimator with substantial data trimming to ensure unbiasedness at the price of high resultant estimation variance. In this paper, we strive to balance the bias and variance in designing randomized network experiments.  Under a potential outcome model with 1-hop interference, we derive the bias and variance of the standard HT estimator and reveal their relation to the network topological structure and the covariance of the treatment assignment vector. We then propose to formulate the experimental design problem as to optimize the covariance matrix of the treatment assignment vector to achieve the bias and variance balance by minimizing the mean squared error (MSE) of the estimator. An efficient projected gradient descent algorithm is presented to the implement of the desired randomization scheme. Finally, we carry out extensive  simulation studies to demonstrate the advantages of our proposed method over other existing methods in many settings, with different levels of model misspecification.

----

## [1628] AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis

**Authors**: *Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760dff0f9c0e9ed4d7e22918c73351d4-Abstract-Conference.html)

**Abstract**:

Can machines recording an audio-visual scene produce realistic, matching audio-visual experiences at novel positions and novel view directions? We answer it by studying a new task---real-world audio-visual scene synthesis---and a first-of-its-kind NeRF-based approach for multimodal learning. Concretely, given a video recording of an audio-visual scene, the task is to synthesize new videos with spatial audios along arbitrary novel camera trajectories in that scene. We propose an acoustic-aware audio generation module that integrates prior knowledge of audio propagation into NeRF, in which we implicitly associate audio generation with the 3D geometry and material properties of a visual environment. Furthermore, we present a coordinate transformation module that expresses a view direction relative to the sound source, enabling the model to learn sound source-centric acoustic fields. To facilitate the study of this new task, we collect a high-quality Real-World Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method on this real-world dataset and the simulation-based SoundSpaces dataset. Notably, we refer readers to view our demo videos for convincing comparisons.

----

## [1629] Is This Loss Informative? Faster Text-to-Image Customization by Tracking Objective Dynamics

**Authors**: *Anton Voronov, Mikhail Khoroshikh, Artem Babenko, Max Ryabinin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/760e8857c7660fe50bac933161b14f41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/760e8857c7660fe50bac933161b14f41-Abstract-Conference.html)

**Abstract**:

Text-to-image generation models represent the next step of evolution in image synthesis, offering a natural way to achieve flexible yet fine-grained control over the result.One emerging area of research is the fast adaptation of large text-to-image models to smaller datasets or new visual concepts.However, many efficient methods of adaptation have a long training time, which limits their practical applications, slows down experiments, and spends excessive GPU resources.In this work, we study the training dynamics of popular text-to-image personalization methods (such as Textual Inversion or DreamBooth), aiming to speed them up.We observe that most concepts are learned at early stages and do not improve in quality later, but standard training convergence metrics fail to indicate that.Instead, we propose a simple drop-in early stopping criterion that only requires computing the regular training objective on a fixed set of inputs for all training iterations.Our experiments on Stable Diffusion for 48 different concepts and three personalization methods demonstrate the competitive performance of our approach, which makes adaptation up to 8 times faster with no significant drops in quality.

----

## [1630] DesCo: Learning Object Recognition with Rich Language Descriptions

**Authors**: *Liunian Harold Li, Zi-Yi Dou, Nanyun Peng, Kai-Wei Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/761c3284ee4859bff3c7e5d9299a45ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/761c3284ee4859bff3c7e5d9299a45ee-Abstract-Conference.html)

**Abstract**:

Recent development in vision-language approaches has instigated a paradigm shift in learning visual recognition models from language supervision. These approaches align objects with language queries (e.g. "a photo of a cat") and thus improve the models' adaptability to novel objects and domains. Recent studies have attempted to query these models with complex language expressions that include specifications of fine-grained details, such as colors, shapes, and relations. However, simply incorporating language descriptions into queries does not guarantee accurate interpretation by the models. In fact, our experiments show that GLIP, a state-of-the-art vision-language model for object detection, often disregards contextual information in the language descriptions and instead relies heavily on detecting objects solely by their names. To tackle the challenge, we propose a new description-conditioned (DesCo) paradigm of learning object recognition models with rich language descriptions consisting of two innovations: 1) we employ a large language model as a commonsense knowledge engine to generate rich language descriptions of objects; 2) we design context-sensitive queries to improve the model's ability in deciphering intricate nuances embedded within descriptions and enforce the model to focus on context rather than object names alone.  On two novel object detection benchmarks, LVIS and OminiLabel, under the zero-shot detection setting, our approach achieves 34.8 APr minival (+9.1) and 29.3 AP (+3.6), respectively, surpassing the prior state-of-the-art models, GLIP and FIBER, by a large margin.

----

## [1631] On the Variance, Admissibility, and Stability of Empirical Risk Minimization

**Authors**: *Gil Kur, Eli Putterman, Alexander Rakhlin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7644353d580a9e027e0069d6480d971b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7644353d580a9e027e0069d6480d971b-Abstract-Conference.html)

**Abstract**:

It is well known that Empirical Risk Minimization (ERM) may attain minimax suboptimal rates in terms of the mean squared error (Birgé and Massart, 1993). In this paper, we prove that, under relatively mild assumptions, the suboptimality of ERM must be due to its bias. Namely, the variance error term of ERM (in terms of the bias and variance decomposition) enjoys the minimax rate. In the fixed design setting, we provide an elementary proof of this result using the probabilistic method. Then, we extend our proof to the random design setting for various models. In addition, we provide a simple proof of Chatterjee’s admissibility theorem (Chatterjee, 2014, Theorem 1.4), which states that in the fixed design setting, ERM cannot be ruled out as an optimal method, and then we extend this result to the random design setting. We also show that our estimates imply stability of ERM, complementing the main result of Caponnetto and Rakhlin (2006) for non-Donsker classes. Finally, we highlight the somewhat irregular nature of the loss landscape of ERM in the non-Donsker regime, by showing that functions can be close to ERM, in terms of $L_2$ distance, while still being far from almost-minimizers of the empirical loss.

----

## [1632] NetHack is Hard to Hack

**Authors**: *Ulyana Piterbarg, Lerrel Pinto, Rob Fergus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html)

**Abstract**:

Neural policy learning methods have achieved remarkable results in various control problems, ranging from Atari games to simulated locomotion. However, these methods struggle in long-horizon tasks, especially in open-ended environments with multi-modal observations, such as the popular dungeon-crawler game, NetHack. Intriguingly, the NeurIPS 2021 NetHack Challenge revealed that symbolic agents outperformed neural approaches by over four times in median game score. In this paper, we delve into the reasons behind this performance gap and present an extensive study on neural policy learning for NetHack. To conduct this study, we analyze the winning symbolic agent, extending its codebase to track internal strategy selection in order to generate one of the largest available demonstration datasets. Utilizing this dataset, we examine (i) the advantages of an action hierarchy; (ii) enhancements in neural architecture; and (iii) the integration of reinforcement learning with imitation learning. Our investigations produce a state-of-the-art neural agent that surpasses previous fully neural policies by 127% in offline settings and 25% in online settings on median game score. However, we also demonstrate that mere scaling is insufficient to bridge the performance gap with the best symbolic models or even the top human players.

----

## [1633] SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Benjamin Ellis, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob N. Foerster, Shimon Whiteson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/764c18ad230f9e7bf6a77ffc2312c55e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The availability of challenging benchmarks has played a key role in the recent progress of machine learning. In cooperative multi-agent reinforcement learning, the StarCraft Multi-Agent Challenge (SMAC) has become a popular testbed for centralised training with decentralised execution. However, after years of sustained improvement on SMAC, algorithms now achieve near-perfect performance. In this work, we conduct new analysis demonstrating that SMAC lacks the stochasticity and partial observability to require complex closed-loop policies. In particular, we show that an open-loop policy conditioned only on the timestep can achieve non-trivial win rates for many SMAC scenarios. To address this limitation, we introduce SMACv2, a new version of the benchmark where scenarios are procedurally generated and require agents to generalise to previously unseen settings (from the same distribution) during evaluation. We also introduce the extended partial observability challenge (EPO), which augments SMACv2 to ensure meaningful partial observability. We show that these changes ensure the benchmarkrequires the use of closed-loop policies. We evaluate state-of-the-art algorithms on SMACv2 and show that it presents significant challenges not present in the original benchmark.  Our analysis illustrates that SMACv2 addresses the discovered deficiencies of SMAC and can help benchmark the next generation of MARL methods. Videos of training are available on our website.

----

## [1634] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios

**Authors**: *Yazhe Niu, Yuan Pu, Zhenjie Yang, Xueyan Li, Tong Zhou, Jiyuan Ren, Shuai Hu, Hongsheng Li, Yu Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Building agents based on tree-search planning capabilities with learned models has achieved remarkable success in classic decision-making problems, such as Go and Atari.However, it has been deemed challenging or even infeasible to extend Monte Carlo Tree Search (MCTS) based algorithms to diverse real-world applications, especially when these environments involve complex action spaces and significant simulation costs, or inherent stochasticity.In this work, we introduce LightZero, the first unified benchmark for deploying MCTS/MuZero in general sequential decision scenarios. Specificially, we summarize the most critical challenges in designing a general MCTS-style decision-making solver, then decompose the tightly-coupled algorithm and system design of tree-search RL methods into distinct sub-modules.By incorporating more appropriate exploration and optimization strategies, we can significantly enhance these sub-modules and construct powerful LightZero agents to tackle tasks across a wide range of domains, such as board games, Atari, MuJoCo, MiniGrid and GoBigger.Detailed benchmark results reveal the significant potential of such methods in building scalable and efficient decision intelligence.The code is available as part of OpenDILab at https://github.com/opendilab/LightZero.

----

## [1635] Improving Diffusion-Based Image Synthesis with Context Prediction

**Authors**: *Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, Bin Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7664a7e946a84ac5e97649a967717cf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7664a7e946a84ac5e97649a967717cf2-Abstract-Conference.html)

**Abstract**:

Diffusion models are a new class of generative models, and have dramatically promoted image generation with unprecedented quality and diversity. Existing diffusion models mainly try to reconstruct input image from a corrupted one with a pixel-wise or feature-wise constraint along spatial axes. However, such point-based reconstruction may fail to make each predicted pixel/feature fully preserve its neighborhood context, impairing diffusion-based image synthesis. As a powerful source of automatic supervisory signal, context has been well studied for learning representations. Inspired by this, we for the first time propose ConPreDiff to improve diffusion-based image synthesis with context prediction. We explicitly reinforce each point to predict its neighborhood context (i.e., multi-stride pixels/features) with a context decoder at the end of diffusion denoising blocks in training stage, and remove the decoder for inference. In this way, each point can better reconstruct itself by preserving its semantic connections with neighborhood context. This new paradigm of ConPreDiff can generalize to arbitrary discrete and continuous diffusion backbones without introducing extra parameters in sampling procedure. Extensive experiments are conducted on unconditional image generation, text-to-image generation and image inpainting tasks. Our ConPreDiff consistently outperforms previous methods and achieves new SOTA text-to-image generation results on MS-COCO, with a zero-shot FID score of 6.21.

----

## [1636] Adversarial Robustness through Random Weight Sampling

**Authors**: *Yanxiang Ma, Minjing Dong, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/766f407b7b4a82135da23b32f0cbaff3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/766f407b7b4a82135da23b32f0cbaff3-Abstract-Conference.html)

**Abstract**:

Deep neural networks have been found to be vulnerable in a variety of tasks. Adversarial attacks can manipulate network outputs, resulting in incorrect predictions. Adversarial defense methods aim to improve the adversarial robustness of networks by countering potential attacks. In addition to traditional defense approaches, randomized defense mechanisms have recently received increasing attention from researchers. These methods introduce different types of perturbations during the inference phase to destabilize adversarial attacks.Although promising empirical results have been demonstrated by these approaches, the defense performance is quite sensitive to the randomness parameters, which are always manually tuned without further analysis. On the contrary, we propose incorporating random weights into the optimization to fully exploit the potential of randomized defense. To perform better optimization of randomness parameters, we conduct a theoretical analysis of the connections between randomness parameters and gradient similarity as well as natural performance. From these two aspects, we suggest imposing theoretically-guided constraints on random weights during optimizations, as these weights play a critical role in balancing natural performance and adversarial robustness. We derive both the upper and lower bounds of random weight parameters by considering prediction bias and gradient similarity. In this study, we introduce the Constrained Trainable Random Weight (CTRW), which adds random weight parameters to the optimization and includes a constraint guided by the upper and lower bounds to achieve better trade-offs between natural and robust accuracy. We evaluate the effectiveness of CTRW on several datasets and benchmark convolutional neural networks. Our results indicate that our model achieves a robust accuracy approximately 16% to 17% higher than the baseline model under PGD-20 and 22% to 25% higher on Auto Attack.

----

## [1637] PyNeRF: Pyramidal Neural Radiance Fields

**Authors**: *Haithem Turki, Michael Zollhöfer, Christian Richardt, Deva Ramanan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/767c1b5f7c03d9299e493bc9e1feeba6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/767c1b5f7c03d9299e493bc9e1feeba6-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRFs) can be dramatically accelerated by spatial grid representations. However, they do not explicitly reason about scale and so introduce aliasing artifacts when reconstructing scenes captured at different camera distances. Mip-NeRF and its extensions propose scale-aware renderers that project volumetric frustums rather than point samples. But such approaches rely on positional encodings that are not readily compatible with grid methods. We propose a simple modification to grid-based models by training model heads at different spatial grid resolutions. At render time, we simply use coarser grids to render samples that cover larger volumes. Our method can be easily applied to existing accelerated NeRF methods and significantly improves rendering quality (reducing error rates by 20–90% across synthetic and unbounded real-world scenes) while incurring minimal performance overhead (as each model head is quick to evaluate). Compared to Mip-NeRF, we reduce error rates by 20% while training over 60x faster.

----

## [1638] Universal Online Learning with Gradient Variations: A Multi-layer Online Ensemble Approach

**Authors**: *Yu-Hu Yan, Peng Zhao, Zhi-Hua Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an online convex optimization approach with two different levels of adaptivity. On a higher level, our approach is agnostic to the unknown types and curvatures of the online functions, while at a lower level, it can exploit the unknown niceness of the environments and attain problem-dependent guarantees. Specifically, we obtain $\mathcal{O}(\log V_T)$, $\mathcal{O}(d \log V_T)$ and $\hat{\mathcal{O}}(\sqrt{V_T})$ regret bounds for strongly convex, exp-concave and convex loss functions, respectively, where $d$ is the dimension, $V_T$ denotes problem-dependent gradient variations and the $\hat{\mathcal{O}}(\cdot)$-notation omits $\log V_T$ factors. Our result not only safeguards the worst-case guarantees but also directly implies the small-loss bounds in analysis. Moreover, when applied to adversarial/stochastic convex optimization and game theory problems, our result enhances the existing universal guarantees. Our approach is based on a multi-layer online ensemble framework incorporating novel ingredients, including a carefully designed optimism for unifying diverse function types and cascaded corrections for algorithmic stability. Notably, despite its multi-layer structure, our algorithm necessitates only one gradient query per round, making it favorable when the gradient evaluation is time-consuming. This is facilitated by a novel regret decomposition equipped with carefully designed surrogate losses.

----

## [1639] Information Theoretic Lower Bounds for Information Theoretic Upper Bounds

**Authors**: *Roi Livni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/768396006e9214568dba5aae9dd312c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/768396006e9214568dba5aae9dd312c5-Abstract-Conference.html)

**Abstract**:

We examine the relationship between the mutual information between the output model and the empirical sample and the algorithm's generalization in the context of stochastic convex optimization. Despite increasing interest in information-theoretic generalization bounds, it is uncertain if these bounds can provide insight into the exceptional performance of various learning algorithms. Our study of stochastic convex optimization reveals that, for true risk minimization, dimension-dependent mutual information is necessary. This indicates that existing information-theoretic generalization bounds fall short in capturing the generalization capabilities of algorithms like SGD and regularized ERM, which have dimension-independent sample complexity.

----

## [1640] CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift

**Authors**: *Siddhartha Laghuvarapu, Zhen Lin, Jimeng Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html)

**Abstract**:

In drug discovery, it is vital to confirm the predictions of pharmaceutical properties from computational models using costly wet-lab experiments. Hence, obtaining reliable uncertainty estimates is crucial for prioritizing drug molecules for subsequent experimental validation. Conformal Prediction (CP) is a promising tool for creating such prediction sets for molecular properties with a coverage guarantee. However, the exchangeability assumption of CP is often challenged with covariate shift in drug discovery tasks: Most datasets contain limited labeled data, which may not be representative of the vast chemical space from which molecules are drawn. To address this limitation, we propose a method called CoDrug that employs an energy-based model leveraging both training data and unlabelled data, and  Kernel Density Estimation (KDE) to assess the densities of a molecule set. The estimated densities are then used to weigh the molecule samples while building prediction sets and rectifying for distribution shift. In extensive experiments involving realistic distribution drifts in various small-molecule drug discovery tasks,  we demonstrate the ability of CoDrug to provide valid prediction sets and its utility in addressing the distribution shift arising from de novo drug design models. On average, using CoDrug can reduce the coverage gap by over 35% when compared to conformal prediction sets not adjusted for covariate shift.

----

## [1641] TWIGMA: A dataset of AI-Generated Images with Metadata From Twitter

**Authors**: *Yiqun Chen, James Y. Zou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/769b70d1a9a6b21af53c00d0b322c763-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/769b70d1a9a6b21af53c00d0b322c763-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent progress in generative artificial intelligence (gen-AI) has enabled the generation of photo-realistic and artistically-inspiring photos at a single click, catering to millions of users online. To explore how people use gen-AI models such as DALLE and StableDiffusion, it is critical to understand the themes, contents, and variations present in the AI-generated photos. In this work, we introduce TWIGMA (TWItter Generative-ai images with MetadatA), a comprehensive dataset encompassing over 800,000 gen-AI images collected from Jan 2021 to March 2023 on Twitter, with associated metadata (e.g., tweet text, creation date, number of likes). Through a comparative analysis of TWIGMA with natural images and human artwork, we find that gen-AI images possess distinctive characteristics and exhibit, on average, lower variability when compared to their non-gen-AI counterparts. Additionally, we find that the similarity between a gen-AI image and natural images is inversely correlated with the number of likes. Finally, we observe a longitudinal shift in the themes of AI-generated images on Twitter, with users increasingly sharing artistically sophisticated content such as intricate human portraits, whereas their interest in simple subjects such as natural scenes and animals has decreased. Our analyses and findings underscore the significance of TWIGMA as a unique data resource for studying AI-generated images.

----

## [1642] Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation

**Authors**: *Berivan Isik, Wei-Ning Chen, Ayfer Özgür, Tsachy Weissman, Albert No*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html)

**Abstract**:

We study the mean estimation problem under communication and local differential privacy constraints. While previous work has proposed order-optimal algorithms for the same problem (i.e., asymptotically optimal as we spend more bits), exact optimality (in the non-asymptotic setting) still has not been achieved. In this work, we take a step towards characterizing the exact-optimal approach in the presence of shared randomness (a random variable shared between the server and the user) and identify several conditions for exact optimality. We prove that one of the conditions is to utilize a rotationally symmetric shared random codebook. Based on this, we propose a randomization mechanism where the codebook is a randomly rotated simplex -- satisfying the properties of the exact-optimal codebook. The proposed mechanism is based on a $k$-closest encoding which we prove to be exact-optimal for the randomly rotated simplex codebook.

----

## [1643] Estimating Generic 3D Room Structures from 2D Annotations

**Authors**: *Denys Rozumnyi, Stefan Popov, Kevis-Kokitsi Maninis, Matthias Nießner, Vittorio Ferrari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76bf913ad349686b2aa552a1c6ee0a2e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/76bf913ad349686b2aa552a1c6ee0a2e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Indoor rooms are among the most common use cases in 3D scene understanding. Current state-of-the-art methods for this task are driven by large annotated datasets. Room layouts are especially important, consisting of structural elements in 3D, such as wall, floor, and ceiling. However, they are difficult to annotate, especially on pure RGB video. We propose a novel method to produce generic 3D room layouts just from 2D segmentation masks, which are easy to annotate for humans. Based on these 2D annotations, we automatically reconstruct 3D plane equations for the structural elements and their spatial extent in the scene, and connect adjacent elements at the appropriate contact edges. We annotate and publicly release 2246 3D room layouts on the RealEstate10k dataset, containing YouTube videos. We demonstrate the high quality of these 3D layouts annotations with extensive experiments.

----

## [1644] Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces

**Authors**: *Sungbin Lim, Eun-Bi Yoon, Taehyun Byun, Taewon Kang, Seungwoo Kim, Kyungjae Lee, Sungjoon Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76c6f9f2475b275b92d03a83ea270af4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76c6f9f2475b275b92d03a83ea270af4-Abstract-Conference.html)

**Abstract**:

Continuous-time score-based generative models consist of a pair of stochastic differential equations (SDEs)—a forward SDE that smoothly transitions data into a noise space and a reverse SDE that incrementally eliminates noise from a Gaussian prior distribution to generate data distribution samples—are intrinsically connected by the time-reversal theory on diffusion processes. In this paper, we investigate the use of stochastic evolution equations in Hilbert spaces, which expand the applicability of SDEs in two aspects: sample space and evolution operator, so they enable encompassing recent variations of diffusion models, such as generating functional data or replacing drift coefficients with image transformation. To this end, we derive a generalized time-reversal formula to build a bridge between probabilistic diffusion models and stochastic evolution equations and propose a score-based generative model called Hilbert Diffusion Model (HDM). Combining with Fourier neural operator, we verify the superiority of HDM for sampling functions from functional datasets with a power of kernel two-sample test of 4.2 on Quadratic, 0.2 on Melbourne, and 3.6 on Gridwatch, which outperforms existing diffusion models formulated in function spaces. Furthermore, the proposed method shows its strength in motion synthesis tasks by utilizing the Wiener process with values in Hilbert space. Finally, our empirical results on image datasets also validate a connection between HDM and diffusion models using heat dissipation, revealing the potential for exploring evolution operators and sample spaces.

----

## [1645] Unlocking Feature Visualization for Deep Network with MAgnitude Constrained Optimization

**Authors**: *Thomas Fel, Thibaut Boissin, Victor Boutin, Agustin Picard, Paul Novello, Julien Colin, Drew Linsley, Tom Rousseau, Rémi Cadène, Lore Goetschalckx, Laurent Gardes, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/76d2f8e328e1081c22a77ca0fa330ca5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/76d2f8e328e1081c22a77ca0fa330ca5-Abstract-Conference.html)

**Abstract**:

Feature visualization has gained significant popularity as an explainability method, particularly after the influential work by Olah et al. in 2017. Despite its success, its widespread adoption has been limited due to issues in scaling to deeper neural networks and the reliance on tricks to generate interpretable images. Here, we describe MACO, a simple approach to address these shortcomings. It consists in optimizing solely an image's phase spectrum while keeping its magnitude constant to ensure that the generated explanations lie in the space of natural images. Our approach yields significantly better results -- both qualitatively and quantitatively -- unlocking efficient and interpretable feature visualizations for state-of-the-art neural networks. We also show that our approach exhibits an attribution mechanism allowing to augment feature visualizations with spatial importance. Furthermore, we enable quantitative evaluation of feature visualizations by introducing 3 metrics: transferability, plausibility, and alignment with natural images. We validate our method on various applications and we introduce a website featuring MACO visualizations for all classes of the ImageNet dataset, which will be made available upon acceptance. Overall, our study unlocks feature visualizations for the largest, state-of-the-art classification networks without resorting to any parametric prior image model, effectively advancing a field that has been stagnating since 2017 (Olah et al, 2017).

----

## [1646] Exact recovery and Bregman hard clustering of node-attributed Stochastic Block Model

**Authors**: *Maximilien Dreveton, Felipe S. Fernandes, Daniel R. Figueiredo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/770b3ecb70147a2d2f18d2964fafcdd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/770b3ecb70147a2d2f18d2964fafcdd5-Abstract-Conference.html)

**Abstract**:

Classic network clustering tackles the problem of identifying sets of nodes (communities) that have similar connection patterns. However, in many scenarios nodes also have attributes that are correlated and can also be used to identify node clusters. Thus, network information (edges) and node information (attributes) can be jointly leveraged to design high-performance clustering algorithms. Under a general model for the network and node attributes, this work establishes an information-theoretic criteria for the exact recovery of community labels and characterizes a phase transition determined by the Chernoff-Hellinger divergence of the model. The criteria shows how network and attribute information can be exchanged in order to have exact recovery (e.g., more reliable network information requires less reliable attribute information). This work also presents an iterative clustering algorithm that maximizes the joint likelihood, assuming that the probability distribution of network interactions and node attributes belong to exponential families. This covers a broad range of possible interactions (e.g., edges with weights) and attributes (e.g., non-Gaussian models) while also exploring the connection between exponential families and Bregman divergences. Extensive numerical experiments using synthetic and real data indicate that the proposed algorithm outperforms algorithms that leverage only network or only attribute information as well as recently proposed algorithms that perform clustering using both sources of information. The contributions of this work provide insights into the fundamental limits and practical techniques for inferring community labels on node-attributed networks.

----

## [1647] Learning to Receive Help: Intervention-Aware Concept Embedding Models

**Authors**: *Mateo Espinosa Zarlenga, Katie Collins, Krishnamurthy Dvijotham, Adrian Weller, Zohreh Shams, Mateja Jamnik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)

**Abstract**:

Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories at train-time. This conditions IntCEMs to effectively select and receive concept interventions when deployed at test-time. Our experiments show that IntCEMs significantly outperform state-of-the-art concept-interpretable models when provided with test-time concept interventions, demonstrating the effectiveness of our approach.

----

## [1648] Tracr: Compiled Transformers as a Laboratory for Interpretability

**Authors**: *David Lindner, János Kramár, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, Vladimir Mikulik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html)

**Abstract**:

We show how to "compile" human-readable programs into standard decoder-only transformer models. Our compiler, Tracr, generates models with known structure. This structure can be used to design experiments. For example, we use it to study "superposition" in transformers that execute multi-step algorithms. Additionally, the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods. Commonly, because the "programs" learned by transformers are unknown it is unclear whether an interpretation succeeded. We demonstrate our approach by implementing and examining programs including computing token frequencies, sorting, and parenthesis checking. We provide an open-source implementation of Tracr at https://github.com/google-deepmind/tracr.

----

## [1649] KAKURENBO: Adaptively Hiding Samples in Deep Neural Network Training

**Authors**: *Truong Thao Nguyen, Balazs Gerofi, Edgar Josafat Martinez-Noriega, François Trahay, Mohamed Wahib*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7712b1075f5e0eae297702845714098f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7712b1075f5e0eae297702845714098f-Abstract-Conference.html)

**Abstract**:

This paper proposes a method for hiding the least-important samples during the training of deep neural networks to increase efficiency, i.e., to reduce the cost of training. Using information about the loss and prediction confidence during training, we adaptively find samples to exclude in a given epoch based on their contribution to the overall learning process, without significantly degrading accuracy. We explore the converge properties when accounting for the reduction in the number of SGD updates.  Empirical results on various large-scale datasets and models used directly in image classification and segmentation show that while the with-replacement importance sampling algorithm performs poorly on large datasets,  our method can reduce total training time by up to 22\% impacting accuracy only by 0.4\% compared to the baseline.

----

## [1650] Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation

**Authors**: *Dapeng Hu, Jian Liang, Jun Hao Liew, Chuhui Xue, Song Bai, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html)

**Abstract**:

Unsupervised domain adaptation (UDA) has been widely applied in improving model generalization on unlabeled target data. However, accurately selecting the best UDA model for the target domain is challenging due to the absence of labeled target data and domain distribution shifts. Traditional model selection approaches involve training extra models with source data to estimate the target validation risk. Recent studies propose practical methods that are based on measuring various properties of model predictions on target data. Although effective for some UDA models, these methods often lack stability and may lead to poor selections for other UDA models.In this paper, we present MixVal, an innovative model selection method that operates solely with unlabeled target data during inference. MixVal leverages mixed target samples with pseudo labels to directly probe the learned target structure by each UDA model. Specifically, MixVal employs two distinct types of probes: the intra-cluster mixed samples for evaluating neighborhood density and the inter-cluster mixed samples for investigating the classification boundary. With this comprehensive probing strategy, MixVal elegantly combines the strengths of two state-of-the-art model selection methods, Entropy and SND. We extensively evaluate MixVal on 11 UDA methods across 4 adaptation settings, including classification and segmentation tasks. Experimental results consistently demonstrate that MixVal achieves state-of-the-art performance and maintains exceptional stability in model selection. Code is available at \url{https://github.com/LHXXHB/MixVal}.

----

## [1651] Payoff-based Learning with Matrix Multiplicative Weights in Quantum Games

**Authors**: *Kyriakos Lotidis, Panayotis Mertikopoulos, Nicholas Bambos, Jose H. Blanchet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77307e2e3f326335dfeb94ab47f7a6c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77307e2e3f326335dfeb94ab47f7a6c0-Abstract-Conference.html)

**Abstract**:

In this paper, we study the problem of learning in quantum games - and other classes of semidefinite games - with scalar, payoff-based feedback.For concreteness, we focus on the widely used matrix multiplicative weights (MMW) algorithm and, instead of requiring players to have full knowledge of the game (and/or each other's chosen states), we introduce a suite of minimal-information matrix multiplicative weights (3MW) methods tailored to different information frameworks.The main difficulty to attaining convergence in this setting is that, in contrast to classical finite games, quantum games have an infinite continuum of pure states (the quantum equivalent of pure strategies), so standard importance-weighting techniques for estimating payoff vectors cannot be employed.Instead, we borrow ideas from bandit convex optimization and we design a zeroth-order gradient sampler adapted to the semidefinite geometry of the problem at hand.As a first result, we show that the 3MW method with deterministic payoff feedback retains the $\mathcal{O}(1/\sqrt{T})$ convergence rate of the vanilla, full information MMW algorithm in quantum min-max games, even though the players only observe a single scalar.Subsequently, we relax the algorithm's information requirements even further and we provide a 3MW method that only requires players to observe a random realization of their payoff observable, and converges to equilibrium at an $\mathcal{O}(T^{-1/4})$ rate.Finally, going beyond zero-sum games, we show that a regularized variant of the proposed 3MW method guarantees local convergence with high probability to all equilibria that satisfy a certain first-order stability condition.

----

## [1652] Deep Stochastic Processes via Functional Markov Transition Operators

**Authors**: *Jin Xu, Emilien Dupont, Kaspar Märtens, Thomas Rainforth, Yee Whye Teh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html)

**Abstract**:

We introduce Markov Neural Processes (MNPs), a new class of Stochastic Processes (SPs) which are constructed by stacking sequences of neural parameterised Markov transition operators in function space. We prove that these Markov transition operators can preserve the exchangeability and consistency of SPs. Therefore, the proposed iterative construction adds substantial flexibility and expressivity to the original framework of Neural Processes (NPs) without compromising consistency or adding restrictions. Our experiments demonstrate clear advantages of MNPs over baseline models on a variety of tasks.

----

## [1653] Quilt-1M: One Million Image-Text Pairs for Histopathology

**Authors**: *Wisdom Oluchi Ikezogwo, Mehmet Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Stefan Chan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, Linda G. Shapiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/775ec578876fa6812c062644964b9870-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has slowed comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians.From YouTube, we curate QUILT: a large-scale vision-language dataset consisting of $802, 144$ image and text pairs.QUILT was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition.In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples.We combine QUILT with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even larger dataset: QUILT-1M, with $1$M paired image-text samples, marking it as the largest vision-language histopathology dataset to date. We demonstrate the value of QUILT-1M by fine-tuning a pre-trained CLIP model. Our model outperforms state-of-the-art models on both zero-shot and linear probing tasks for classifying new histopathology images across $13$ diverse patch-level datasets of $8$ different sub-pathologies and cross-modal retrieval tasks.

----

## [1654] A Computation and Communication Efficient Method for Distributed Nonconvex Problems in the Partial Participation Setting

**Authors**: *Alexander Tyurin, Peter Richtárik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/778ff1fcfb6d6707fc015908a1845b62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/778ff1fcfb6d6707fc015908a1845b62-Abstract-Conference.html)

**Abstract**:

We present a new method that includes three key components of distributed optimization and federated learning: variance reduction of stochastic gradients, partial participation, and compressed communication. We prove that the new method has optimal oracle complexity and state-of-the-art communication complexity in the partial participation setting. Regardless of the communication compression feature, our method successfully combines variance reduction and partial participation: we get the optimal oracle complexity, never need the participation of all nodes, and do not require the bounded gradients (dissimilarity) assumption.

----

## [1655] Optimistic Active Exploration of Dynamical Systems

**Authors**: *Bhavya Sukhija, Lenart Treven, Cansu Sancaktar, Sebastian Blaes, Stelian Coros, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html)

**Abstract**:

Reinforcement learning algorithms commonly seek to optimize policies for solving one particular task. How should we explore an unknown dynamical system such that the estimated model allows us to solve multiple downstream tasks in a zero-shot manner? In this paper, we address this challenge, by developing an algorithm -- OPAX -- for active exploration. OPAX uses well-calibrated probabilistic models to quantify the epistemic uncertainty about the unknown dynamics. It optimistically---w.r.t. to plausible dynamics---maximizes the information gain between the unknown dynamics and state observations. We show how the resulting optimization problem can be reduced to an optimal control problem that can be solved at each episode using standard approaches.  We analyze our algorithm for general models, and, in the case of Gaussian process dynamics, we give a sample complexity bound andshow that the epistemic uncertainty converges to zero. In our experiments, we compare OPAX with other heuristic active exploration approaches on several environments. Our experiments show that OPAX is not only theoretically sound but also performs well for zero-shot planning on novel downstream tasks.

----

## [1656] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

**Authors**: *Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77c33e6a367922d003ff102ffb92b658-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77c33e6a367922d003ff102ffb92b658-Abstract-Conference.html)

**Abstract**:

Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are numerous AI models available for various domains and modalities, they cannot handle complicated AI tasks autonomously. Considering large language models (LLMs) have exhibited exceptional abilities in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks, with language serving as a generic interface to empower this. Based on this philosophy, we present HuggingGPT, an LLM-powered agent that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Hugging Face, HuggingGPT can tackle a wide range of sophisticated AI tasks spanning different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards the realization of artificial general intelligence.

----

## [1657] Multi-Step Generalized Policy Improvement by Leveraging Approximate Models

**Authors**: *Lucas Nunes Alegre, Ana L. C. Bazzan, Ann Nowé, Bruno C. da Silva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html)

**Abstract**:

We introduce a principled method for performing zero-shot transfer in reinforcement learning (RL) by exploiting approximate models of the environment. Zero-shot transfer in RL has been investigated by leveraging methods rooted in generalized policy improvement (GPI) and successor features (SFs). Although computationally efficient, these methods are model-free: they analyze a library of policies---each solving a particular task---and identify which action the agent should take. We investigate the more general setting where, in addition to a library of policies, the agent has access to an approximate environment model. Even though model-based RL algorithms can identify near-optimal policies, they are typically computationally intensive. We introduce $h$-GPI, a multi-step extension of GPI that interpolates between these extremes---standard model-free GPI and fully model-based planning---as a function of a parameter, $h$, regulating the amount of time the agent has to reason. We prove that $h$-GPI's performance lower bound is strictly better than GPI's, and show that $h$-GPI generally outperforms GPI as $h$ increases. Furthermore, we prove that as $h$ increases, $h$-GPI's performance becomes arbitrarily less susceptible to sub-optimality in the agent's policy library. Finally, we introduce novel bounds characterizing the gains achievable by $h$-GPI as a function of approximation errors in both the agent's policy library and its (possibly learned) model. These bounds strictly generalize those known in the literature. We evaluate $h$-GPI on challenging tabular and continuous-state problems under value function approximation and show that it consistently outperforms GPI and state-of-the-art competing methods under various levels of approximation errors.

----

## [1658] GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients

**Authors**: *Sima Behpour, Thang Long Doan, Xin Li, Wenbin He, Liang Gou, Liu Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77cf940349218069bbc230fc2c9c8a21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77cf940349218069bbc230fc2c9c8a21-Abstract-Conference.html)

**Abstract**:

Detecting out-of-distribution (OOD) data is crucial for ensuring the safe deployment of machine learning models in real-world applications. However, existing OOD detection approaches primarily rely on the feature maps or the full gradient space information to derive OOD scores neglecting the role of \textbf{most important parameters} of the pre-trained network over In-Distribution data. In this study, we propose a novel approach called GradOrth to facilitate OOD detection based on one intriguing observation that the important features to identify OOD data lie in the lower-rank subspace of in-distribution (ID) data.In particular, we identify OOD data by computing the norm of gradient projection on \textit{the subspaces considered \textbf{important} for the in-distribution data}. A large orthogonal projection value (i.e. a small projection value) indicates the sample as OOD as it captures a weak correlation of the in-distribution (ID) data. This simple yet effective method exhibits outstanding performance, showcasing a notable reduction in the average false positive rate at a 95\% true positive rate (FPR95) of up to 8\% when compared to the current state-of-the-art methods.

----

## [1659] Learning to Modulate pre-trained Models in RL

**Authors**: *Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html)

**Abstract**:

Reinforcement Learning (RL) has been successful in various domains like robotics, game playing, and simulation. While RL agents have shown impressive capabilities in their specific tasks, they insufficiently adapt to new tasks. In supervised learning, this adaptation problem is addressed by large-scale pre-training followed by fine-tuning to new down-stream tasks. Recently, pre-training on multiple tasks has been gaining traction in RL. However, fine-tuning a pre-trained model often suffers from catastrophic forgetting. That is, the performance on the pre-training tasks deteriorates when fine-tuning on new tasks. To investigate the catastrophic forgetting phenomenon, we first jointly pre-train a model on datasets from two benchmark suites, namely Meta-World and DMControl. Then, we evaluate and compare a variety of fine-tuning methods prevalent in natural language processing, both in terms of performance on new tasks, and how well performance on pre-training tasks is retained. Our study shows that with most fine-tuning approaches, the performance on pre-training tasks deteriorates significantly. Therefore, we propose a novel method, Learning-to-Modulate (L2M), that avoids the degradation of learned skills by modulating the information flow of the frozen pre-trained model via a learnable modulation pool. Our method achieves state-of-the-art performance on the Continual-World benchmark, while retaining performance on the pre-training tasks. Finally, to aid future research in this area, we release a dataset encompassing 50 Meta-World and 16 DMControl tasks.

----

## [1660] Injecting Multimodal Information into Rigid Protein Docking via Bi-level Optimization

**Authors**: *Ruijia Wang, YiWu Sun, Yujie Luo, Shaochuan Li, Cheng Yang, Xingyi Cheng, Hui Li, Chuan Shi, Le Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/77fa0e7d45c6687f1958de0b31e9fc05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/77fa0e7d45c6687f1958de0b31e9fc05-Abstract-Conference.html)

**Abstract**:

The structure of protein-protein complexes is critical for understanding binding dynamics, biological mechanisms, and intervention strategies. Rigid protein docking, a fundamental problem in this field, aims to predict the 3D structure of complexes from their unbound states without conformational changes. In this scenario, we have access to two types of valuable information: sequence-modal information, such as coevolutionary data obtained from multiple sequence alignments, and structure-modal information, including the 3D conformations of rigid structures. However, existing docking methods typically utilize single-modal information, resulting in suboptimal predictions. In this paper, we propose xTrimoBiDock (or BiDock for short), a novel rigid docking model that effectively integrates sequence- and structure-modal information through bi-level optimization. Specifically, a cross-modal transformer combines multimodal information to predict an inter-protein distance map. To achieve rigid docking, the roto-translation transformation is optimized to align the docked pose with the predicted distance map. In order to tackle this bi-level optimization problem, we unroll the gradient descent of the inner loop and further derive a better initialization for roto-translation transformation based on spectral estimation. Compared to baselines, BiDock achieves a promising result of a maximum 234% relative improvement in challenging antibody-antigen docking problem.

----

## [1661] Uncertainty-Aware Alignment Network for Cross-Domain Video-Text Retrieval

**Authors**: *Xiaoshuai Hao, Wanqian Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78526d7ad4a2532bd91416e948b9644c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78526d7ad4a2532bd91416e948b9644c-Abstract-Conference.html)

**Abstract**:

Video-text retrieval is an important but challenging research task in the multimedia community.  In this paper, we address the challenge task of Unsupervised Domain Adaptation Video-text Retrieval (UDAVR), assuming that training (source) data and testing (target) data are from different domains. Previous approaches are mostly derived from classification based domain adaptation methods, which are neither multi-modal nor suitable for retrieval task.  In addition, as to the pairwise misalignment issue in target domain, i.e., no pairwise annotations between target videos and texts, the existing method assumes that a video corresponds to a text. Yet we empirically find that in the real scene, one text usually corresponds to multiple videos and vice versa. To tackle this one-to-many issue, we propose a novel method named Uncertainty-aware Alignment Network (UAN). Specifically, we first introduce the multimodal mutual information module to balance the minimization of domain shift in a smooth manner. To tackle the multimodal uncertainties pairwise misalignment in target domain, we propose the Uncertainty-aware Alignment Mechanism (UAM) to fully exploit the semantic information of both modalities in target domain. Extensive experiments in the context of domain-adaptive video-text retrieval demonstrate that our proposed method consistently outperforms multiple baselines, showing a superior generalization ability for target data.

----

## [1662] Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?

**Authors**: *Jialu Gao, Kaizhe Hu, Guowei Xu, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7866ff509c822c2e58d20d00154a15a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7866ff509c822c2e58d20d00154a15a2-Abstract-Conference.html)

**Abstract**:

Pre-trained text-to-image generative models can produce diverse, semantically rich, and realistic images from natural language descriptions. Compared with language, images usually convey information with more details and less ambiguity. In this study, we propose Learning from the Void (LfVoid), a method that leverages the power of pre-trained text-to-image models and advanced image editing techniques to guide robot learning. Given natural language instructions, LfVoid can edit the original observations to obtain goal images, such as "wiping" a stain off a table. Subsequently, LfVoid trains an ensembled goal discriminator on the generated image to provide reward signals for a reinforcement learning agent, guiding it to achieve the goal. The ability of LfVoid to learn with zero in-domain training on expert demonstrations or true goal observations (the void) is attributed to the utilization of knowledge from web-scale generative models. We evaluate LfVoid across three simulated tasks and validate its feasibility in the corresponding real-world scenarios. In addition, we offer insights into the key considerations for the effective integration of visual generative models into robot learning workflows. We posit that our work represents an initial step towards the broader application of pre-trained visual generative models in the robotics field. Our project page: https://lfvoid-rl.github.io/.

----

## [1663] H3T: Efficient Integration of Memory Optimization and Parallelism for Large-scale Transformer Training

**Authors**: *Yuzhong Wang, Xu Han, Weilin Zhao, Guoyang Zeng, Zhiyuan Liu, Maosong Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7886b89aced4d37dd25a6f32854bf3f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7886b89aced4d37dd25a6f32854bf3f9-Abstract-Conference.html)

**Abstract**:

In recent years, big models based on Transformers have achieved state-of-the-art performance on many artificial intelligence (AI) tasks.Despite the success of these Transformer-based models, their huge parameter size poses a serious challenge to their training, both from the storage and computation perspectives.To this end, memory optimization (e.g., rematerialization and offloading) and parallelism (e.g., data parallelism and model parallelism) are widely explored to make training Transformers more efficient.In this paper, we propose a framework to automatically find an efficient integration of memory optimization and parallelism for High-Throughput Transformer Training (named H3T), which is rarely considered by existing efforts for training big Transformer-based models.Specifically, we design search algorithms to combine appropriate memory optimization strategies and parallelism schemes to achieve a balance between memory overhead and training efficiency.We implement H3T based on an open-source toolkit BMTrain and then use H3T to train the Transformers of different sizes to evaluate the efficiency of H3T.The experimental results show that H3T outperforms the most popular deep learning (DL) toolkit Megatron-DeepSpeed by $1.2\times \sim 4.3\times$ training speed while reducing $34.6\% \sim 80.5\%$ of memory overhead.Moreover, H3T can use only 64 NVIDIA A100 GPUs to train GPT-3-175B, which is very difficult for existing DL toolkits. The source code is available at https://github.com/OpenBMB/BMTrain/tree/h3t.

----

## [1664] Binarized Spectral Compressive Imaging

**Authors**: *Yuanhao Cai, Yuxin Zheng, Jing Lin, Xin Yuan, Yulun Zhang, Haoqian Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/788e086c07b8d6fa6b279df56e512312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/788e086c07b8d6fa6b279df56e512312-Abstract-Conference.html)

**Abstract**:

Existing deep learning models for hyperspectral image (HSI) reconstruction achieve good performance but require powerful hardwares with enormous memory and computational resources. Consequently, these methods can hardly be deployed on resource-limited mobile devices. In this paper, we propose a novel method, Binarized Spectral-Redistribution Network (BiSRNet), for efficient and practical HSI restoration from compressed measurement in snapshot compressive imaging (SCI) systems. Firstly, we redesign a compact and easy-to-deploy base model to be binarized. Then we present the basic unit, Binarized Spectral-Redistribution Convolution (BiSR-Conv). BiSR-Conv can adaptively redistribute the HSI representations before binarizing activation and uses a scalable hyperbolic tangent function to closer approximate the Sign function in backpropagation. Based on our BiSR-Conv, we customize four binarized convolutional modules to address the dimension mismatch and propagate full-precision information throughout the whole network. Finally, our BiSRNet is derived by using the proposed techniques to binarize the base model. Comprehensive quantitative and qualitative experiments manifest that our proposed BiSRNet outperforms state-of-the-art binarization algorithms. Code and models are publicly available at https://github.com/caiyuanhao1998/BiSCI

----

## [1665] When Can We Track Significant Preference Shifts in Dueling Bandits?

**Authors**: *Joe Suk, Arpit Agarwal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78ccee9dfbcf84840165ab4093715969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78ccee9dfbcf84840165ab4093715969-Abstract-Conference.html)

**Abstract**:

The $K$-armed dueling bandits problem, where the feedback is in the form of noisy pairwise preferences, has been widely studied due its applications in information retrieval, recommendation systems, etc. Motivated by concerns that user preferences/tastes can evolve over time, we consider the problem of _dueling bandits with distribution shifts_. Specifically, we study the recent notion of _significant shifts_ (Suk and Kpotufe, 2022), and ask whether one can design an _adaptive_ algorithm for the dueling problem with $O(\sqrt{K\tilde{L}T})$ dynamic regret,where $\tilde{L}$ is the (unknown) number of significant shifts in preferences. We show that the answer to this question depends on the properties of underlying preference distributions. Firstly,  we give an impossibility result that rules out any algorithm with $O(\sqrt{K\tilde{L}T})$ dynamic regret under the well-studied Condorcet and SST classes of preference distributions. Secondly, we show that $\text{SST}\cap \text{STI}$ is the largest amongst popular classes of preference distributions where it is possible to design such an algorithm. Overall, our results provides an almost complete resolution of the above question for the hierarchy of  distribution classes.

----

## [1666] Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization

**Authors**: *Haitz Sáez de Ocáriz Borde, Alvaro Arroyo, Ismael Morales, Ingmar Posner, Xiaowen Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/78efbc5386c5a7c241e7fcc482d3c3dc-Abstract-Conference.html)

**Abstract**:

Recent research indicates that the performance of machine learning models can be improved by aligning the geometry of the latent space with the underlying data structure. Rather than relying solely on Euclidean space, researchers have proposed using hyperbolic and spherical spaces with constant curvature, or combinations thereof, to better model the latent space and enhance model performance. However, little attention has been given to the problem of automatically identifying the optimal latent geometry for the downstream task. We mathematically define this novel formulation and coin it as neural latent geometry search (NLGS). More specifically, we introduce an initial attempt to search for a latent geometry composed of a product of constant curvature model spaces with a small number of query evaluations, under some simplifying assumptions. To accomplish this, we propose a novel notion of distance between candidate latent geometries based on the Gromov-Hausdorff distance from metric geometry. In order to compute the Gromov-Hausdorff distance, we introduce a mapping function that enables the comparison of different manifolds by embedding them in a common high-dimensional ambient space. We then design a graph search space based on the notion of smoothness between latent geometries and employ the calculated distances as an additional inductive bias. Finally, we use Bayesian optimization to search for the optimal latent geometry in a query-efficient manner. This is a general method which can be applied to search for the optimal latent geometry for a variety of models and downstream tasks. We perform experiments on synthetic and real-world datasets to identify the optimal latent geometry for multiple machine learning problems.

----

## [1667] Scientific Document Retrieval using Multi-level Aspect-based Queries

**Authors**: *Jianyou Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/78f9c04bdcb06f1ada3902912d8b64ba-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/78f9c04bdcb06f1ada3902912d8b64ba-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high costs and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task,  $\textbf{S}$cientific $\textbf{Do}$cument $\textbf{R}$etrieval using $\textbf{M}$ulti-level $\textbf{A}$spect-based qu$\textbf{E}$ries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for evaluating the viability of Large Language Models (LLMs) such as ChatGPT-3.5 for expert-level dataset annotation tasks. The application of Anno-GPT to annotate the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these complex queries, our DORIS-MAE dataset can be extended to over 4,000 sub-query test cases without requiring additional annotation. We evaluated 17 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights DORIS-MAE's challenges and the need for better approaches to handle complex, multifaceted queries in scientific research. Our dataset and codebase are available at https://github.com/Real-Doris-Mae/Doris-Mae-Dataset .

----

## [1668] Beyond Confidence: Reliable Models Should Also Consider Atypicality

**Authors**: *Mert Yüksekgönül, Linjun Zhang, James Y. Zou, Carlos Guestrin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7900318ffaf5e9bc60250f134c6cc3c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7900318ffaf5e9bc60250f134c6cc3c7-Abstract-Conference.html)

**Abstract**:

While most machine learning models can provide confidence in their predictions, confidence is insufficient to understand a prediction's reliability. For instance, the model may have a low confidence prediction if the input is not well-represented in the training dataset or if the input is inherently ambiguous. In this work, we investigate the relationship between how atypical~(rare) a sample or a class is and the reliability of a model's predictions. We first demonstrate that atypicality is strongly related to miscalibration and accuracy. In particular, we empirically show that predictions for atypical inputs or atypical classes are more overconfident and have lower accuracy. Using these insights, we show incorporating atypicality improves uncertainty quantification and model performance for discriminative neural networks and large language models. In a case study, we show that using atypicality improves the performance of a skin lesion classifier across different skin tone groups without having access to the group attributes. Overall, we propose that models should use not only confidence but also atypicality to improve uncertainty quantification and performance. Our results demonstrate that simple post-hoc atypicality estimators can provide significant value.

----

## [1669] Reversible and irreversible bracket-based dynamics for deep graph neural networks

**Authors**: *Anthony Gruber, Kookjin Lee, Nathaniel Trask*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7903af0a1cffb43dbb2f8160d110a5f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7903af0a1cffb43dbb2f8160d110a5f3-Abstract-Conference.html)

**Abstract**:

Recent works have shown that physics-inspired architectures allow the training of deep graph neural networks (GNNs) without oversmoothing. The role of these physics is unclear, however, with successful examples of both reversible (e.g., Hamiltonian) and irreversible (e.g., diffusion) phenomena producing comparable results despite diametrically opposed mechanisms, and further complications arising due to empirical departures from mathematical theory. This work presents a series of novel GNN architectures based upon structure-preserving bracket-based dynamical systems, which are provably guaranteed to either conserve energy or generate positive dissipation with increasing depth.  It is shown that the theoretically principled framework employed here allows for inherently explainable constructions, which contextualize departures from theory in current architectures and better elucidate the roles of reversibility and irreversibility in network performance. Code is available at the Github repository \url{https://github.com/natrask/BracketGraphs}.

----

## [1670] In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer

**Authors**: *Yuzhou Cao, Hussein Mozannar, Lei Feng, Hongxin Wei, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/791d3337291b2c574545aeecfa75484c-Abstract-Conference.html)

**Abstract**:

Enabling machine learning classifiers to defer their decision to a downstream expert when the expert is more accurate will ensure improved safety and performance. This objective can be achieved with the learning-to-defer framework which aims to jointly learn how to classify and how to defer to the expert. In recent studies, it has been theoretically shown that popular estimators for learning to defer parameterized with softmax provide unbounded estimates for the likelihood of deferring which makes them uncalibrated. However, it remains unknown whether this is due to the widely used softmax parameterization and if we can find a softmax-based estimator that is both statistically consistent and possesses a valid probability estimator. In this work, we first show that the cause of the miscalibrated and unbounded estimator in prior literature is due to the symmetric nature of the surrogate losses used and not due to softmax. We then propose a novel statistically consistent asymmetric softmax-based surrogate loss that can produce valid estimates without the issue of unboundedness. We further analyze the non-asymptotic properties of our proposed method and empirically validate its performance and calibration on benchmark datasets.

----

## [1671] Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection

**Authors**: *Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79206ac5b7e88eeeed74997f3b6f4c7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79206ac5b7e88eeeed74997f3b6f4c7f-Abstract-Conference.html)

**Abstract**:

Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS. The code will be released at https://github.com/OpenDriveLab/Birds-eye-view-Perception.

----

## [1672] No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions

**Authors**: *Tiancheng Jin, Junyan Liu, Chloé Rouyer, William Chang, Chen-Yu Wei, Haipeng Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79358587d84628728199059f648824e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79358587d84628728199059f648824e6-Abstract-Conference.html)

**Abstract**:

Existing online learning algorithms for adversarial Markov Decision Processes achieve $\mathcal{O}(\sqrt{T})$ regret after $T$ rounds of interactions even if the loss functions are chosen arbitrarily by an adversary, with the caveat that the transition function has to be fixed.This is because it has been shown that adversarial transition functions make no-regret learning impossible.Despite such impossibility results, in this work, we develop algorithms that can handle both adversarial losses and adversarial transitions, with regret increasing smoothly in the degree of maliciousness of the adversary.More concretely, we first propose an algorithm that enjoys $\widetilde{\mathcal{O}}(\sqrt{T} + C^{P})$ regret where $C^{P}$ measures how adversarial the transition functions are and can be at most $\mathcal{O}(T)$.While this algorithm itself requires knowledge of $C^{P}$, we further develop a black-box reduction approach that removes this requirement.Moreover, we also show that further refinements of the algorithm not only maintains the same regret bound, but also simultaneously adapts to easier environments (where losses are generated in a certain stochastically constrained manner as in [Jin et al. 2021]) and achieves $\widetilde{\mathcal{O}}(U + \sqrt{UC^{L}}  + C^{P})$ regret, where $U$ is some standard gap-dependent coefficient and $C^{L}$ is the amount of corruption on losses.

----

## [1673] Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark

**Authors**: *Lukasz Augustyniak, Szymon Wozniak, Marcin Gruza, Piotr Gramacki, Krzysztof Rajda, Mikolaj Morzy, Tomasz Kajdanowicz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7945ab41f2aada1247a7c95e75cdf6c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite impressive advancements in multilingual corpora collection and model training, developing large-scale deployments of multilingual models still presents a significant challenge. This is particularly true for language tasks that are culture-dependent. One such example is the area of multilingual sentiment analysis, where affective markers can be subtle and deeply ensconced in culture.This work presents the most extensive open massively multilingual corpus of datasets for training sentiment models. The corpus consists of 79 manually selected datasets from over 350 datasets reported in the scientific literature based on strict quality criteria. The corpus covers 27 languages representing 6 language families. Datasets can be queried using several linguistic and functional features. In addition, we present a multi-faceted sentiment classification benchmark summarizing hundreds of experiments conducted on different base models, training objectives, dataset collections, and fine-tuning strategies.

----

## [1674] Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations

**Authors**: *Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/796455f65fd2cbe049112a2d2d4488cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/796455f65fd2cbe049112a2d2d4488cb-Abstract-Conference.html)

**Abstract**:

Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.

----

## [1675] Ignorance is Bliss: Robust Control via Information Gating

**Authors**: *Manan Tomar, Riashat Islam, Matthew E. Taylor, Sergey Levine, Philip Bachman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/797be96e4481c3fe5d675c1ba5352969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/797be96e4481c3fe5d675c1ba5352969-Abstract-Conference.html)

**Abstract**:

Informational parsimony provides a useful inductive bias for learning representations that achieve better generalization by being robust to noise and spurious correlations. We propose information gating as a way to learn parsimonious representations that identify the minimal information required for a task. When gating information, we can learn to reveal as little information as possible so that a task remains solvable, or hide as little information as possible so that a task becomes unsolvable. We gate information using a differentiable parameterization of the signal-to-noise ratio, which can be applied to arbitrary values in a network, e.g., erasing pixels at the input layer or activations in some intermediate layer. When gating at the input layer, our models learn which visual cues matter for a given task. When gating intermediate layers, our models learn which activations are needed for subsequent stages of computation. We call our approach InfoGating. We apply InfoGating to various objectives such as multi-step forward and inverse dynamics models, Q-learning, and behavior cloning, highlighting how InfoGating can naturally help in discarding information not relevant for control. Results show that learning to identify and use minimal information can improve generalization in downstream tasks. Policies based on InfoGating are considerably more robust to irrelevant visual features, leading to improved pretraining and finetuning of RL models.

----

## [1676] Reduced Policy Optimization for Continuous Control with Hard Constraints

**Authors**: *Shutong Ding, Jingya Wang, Yali Du, Ye Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7984e22a06eb5f0e35d745cb38345983-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7984e22a06eb5f0e35d745cb38345983-Abstract-Conference.html)

**Abstract**:

Recent advances in constrained reinforcement learning (RL) have endowed reinforcement learning with certain safety guarantees. However, deploying existing constrained RL algorithms in continuous control tasks with general hard constraints remains challenging, particularly in those situations with non-convex hard constraints. Inspired by the generalized reduced gradient (GRG) algorithm, a classical constrained optimization technique, we propose a reduced policy optimization (RPO) algorithm that combines RL with GRG to address general hard constraints. RPO partitions actions into basic actions and nonbasic actions following the GRG method and outputs the basic actions via a policy network. Subsequently, RPO calculates the nonbasic actions by solving equations based on equality constraints using the obtained basic actions. The policy network is then updated by implicitly differentiating nonbasic actions with respect to basic actions. Additionally, we introduce an action projection procedure based on the reduced gradient and apply a modified Lagrangian relaxation technique to ensure inequality constraints are satisfied. To the best of our knowledge, RPO is the first attempt that introduces GRG to RL as a way of efficiently handling both equality and inequality hard constraints. It is worth noting that there is currently a lack of RL environments with complex hard constraints, which motivates us to develop three new benchmarks: two robotics manipulation tasks and a smart grid operation control task. With these benchmarks, RPO achieves better performance than previous constrained RL algorithms in terms of both cumulative reward and constraint violation. We believe RPO, along with the new benchmarks, will open up new opportunities for applying RL to real-world problems with complex constraints.

----

## [1677] ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning

**Authors**: *Mingyu Xu, Zheng Lian, Lei Feng, Bin Liu, Jianhua Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html)

**Abstract**:

Noisy partial label learning (noisy PLL) is an important branch of weakly supervised learning. Unlike PLL where the ground-truth label must conceal in the candidate label set, noisy PLL relaxes this constraint and allows the ground-truth label may not be in the candidate label set. To address this challenging problem, most of the existing works attempt to detect noisy samples and estimate the ground-truth label for each noisy sample. However, detection errors are unavoidable. These errors can accumulate during training and continuously affect model optimization. To this end, we propose a novel framework for noisy PLL with theoretical interpretations, called ``Adjusting Label Importance Mechanism (ALIM)''. It aims to reduce the negative impact of detection errors by trading off the initial candidate set and model outputs. ALIM is a plug-in strategy that can be integrated with existing PLL approaches. Experimental results on multiple benchmark datasets demonstrate that our method can achieve state-of-the-art performance on noisy PLL. Our code is available at: https://github.com/zeroQiaoba/ALIM.

----

## [1678] Conditional Score Guidance for Text-Driven Image-to-Image Translation

**Authors**: *Hyunsoo Lee, Minsoo Kang, Bohyung Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/799f81cfa0611f93586c007024041460-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/799f81cfa0611f93586c007024041460-Abstract-Conference.html)

**Abstract**:

We present a novel algorithm for text-driven image-to-image translation based on a pretrained text-to-image diffusion model. Our method aims to generate a target image by selectively editing regions of interest in a source image, defined by a modifying text, while preserving the remaining parts.In contrast to existing techniques that solely rely on a target prompt, we introduce a new score function that additionally considers both the source image and the source text prompt, tailored to address specific translation tasks. To this end, we derive the conditional score function in a principled way, decomposing it into the standard score and a guiding term for target image generation.For the gradient computation about the guiding term, we assume a Gaussian distribution for the posterior distribution and estimate its mean and variance to adjust the gradient without additional training.In addition, to improve the quality of the conditional score guidance, we incorporate a simple yet effective mixup technique, which combines two cross-attention maps derived from the source and target latents.This strategy is effective for promoting a desirable fusion of the invariant parts in the source image and the edited regions aligned with the target prompt, leading to high-fidelity target image generation.Through comprehensive experiments, we demonstrate that our approach achieves outstanding image-to-image translation performance on various tasks.Code is available at https://github.com/Hleephilip/CSG.

----

## [1679] A Unified Approach to Count-Based Weakly Supervised Learning

**Authors**: *Vinay Shukla, Zhe Zeng, Kareem Ahmed, Guy Van den Broeck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html)

**Abstract**:

High-quality labels are often very scarce, whereas unlabeled data with inferred weak labels occurs more naturally. In many cases, these weak labels dictate the frequency of each respective class over a set of instances. In this paper, we develop a unified approach to learning from such weakly-labeled data, which we call *count-based weakly-supervised learning*. At the heart of our approach is the ability to compute the probability of exactly $k$ out of $n$ outputs being set to true. This computation is differentiable, exact, and efficient. Building upon the previous computation, we derive a *count loss* penalizing the model for deviations in its distribution from an arithmetic constraint defined over label counts.

----

## [1680] Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars

**Authors**: *Kaiyue Wen, Yuchen Li, Bingbin Liu, Andrej Risteski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html)

**Abstract**:

Transformer interpretability aims to understand the algorithm implemented by a learned Transformer by examining various aspects of the model, such as the weight matrices or the attention patterns.In this work, through a combination of theoretical results and carefully controlled experiments on synthetic data, we take a critical viewof methods that exclusively focus on individual parts of the model, rather than consider the network as a whole.We consider a simple synthetic setup of learning a (bounded) Dyck language. Theoretically, we show that the set of models that (exactly or approximately) solve this task satisfy a structural characterization derived from ideas in formal languages (the pumping lemma).We use this characterization to show that the set of optima is qualitatively rich; in particular, the attention pattern of a single layer can be "nearly randomized", while preserving the functionality of the network.We also show via extensive experiments that these constructions are not merely a theoretical artifact: even with severe constraints to the architecture of the model, vastly different solutions can be reached via standard training. Thus, interpretability claims based on inspecting individual heads or weight matrices in the Transformer can be misleading.

----

## [1681] GEQ: Gaussian Kernel Inspired Equilibrium Models

**Authors**: *Mingjie Li, Yisen Wang, Zhouchen Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79cab89b43ac21c6941ad9735df95d30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79cab89b43ac21c6941ad9735df95d30-Abstract-Conference.html)

**Abstract**:

Despite the connection established by optimization-induced deep equilibrium models (OptEqs) between their output and the underlying hidden optimization problems, the performance of it along with its related works is still not good enough especially when compared to deep networks. One key factor responsible for this performance limitation is the use of linear kernels to extract features in these models. To address this issue, we propose a novel approach by replacing its linear kernel with a new function that can readily capture nonlinear feature dependencies in the input data. Drawing inspiration from classical machine learning algorithms, we introduce Gaussian kernels as the alternative function and then propose our new equilibrium model, which we refer to as GEQ. By leveraging Gaussian kernels, GEQ can effectively extract the nonlinear information embedded within the input features, surpassing the performance of the original OptEqs. Moreover, GEQ can be perceived as a weight-tied neural network with infinite width and depth. GEQ also enjoys better theoretical properties and improved overall performance. Additionally, our GEQ exhibits enhanced stability when confronted with various samples. We further substantiate the effectiveness and stability of GEQ through a series of comprehensive experiments.

----

## [1682] Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric

**Authors**: *Yiming Wang, Ming Yang, Renzhi Dong, Binbin Sun, Furui Liu, Leong Hou U*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79f7f00cbe3003cea4d0c2326b4c0b42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79f7f00cbe3003cea4d0c2326b4c0b42-Abstract-Conference.html)

**Abstract**:

Reward shaping is an effective technique for integrating domain knowledge into reinforcement learning (RL). However, traditional approaches like potential-based reward shaping totally rely on manually designing shaping reward functions, which significantly restricts exploration efficiency and introduces human cognitive biases.While a number of RL methods have been proposed to boost exploration by designing an intrinsic reward signal as exploration bonus. Nevertheless, these methods heavily rely on the count-based episodic term in their exploration bonus which falls short in scalability. To address these limitations, we propose a general end-to-end potential-based exploration bonus for deep RL via potentials of state discrepancy, which motivates the agent to discover novel states and provides them with denser rewards without manual intervention. Specifically, we measure the novelty of adjacent states by calculating their distance using the bisimulation metric-based potential function, which enhances agent's exploration and ensures policy invariance. In addition, we offer a theoretical guarantee on our inverse dynamic bisimulation metric, bounding the value difference and ensuring that the agent explores states with higher TD error, thus significantly improving training efficiency. The proposed approach is named \textbf{LIBERTY} (exp\textbf{L}oration v\textbf{I}a \textbf{B}isimulation m\textbf{E}t\textbf{R}ic-based s\textbf{T}ate discrepanc\textbf{Y}) which is comprehensively evaluated on the MuJoCo and the Arcade Learning Environments. Extensive experiments have verified the superiority and scalability of our algorithm compared with other competitive methods.

----

## [1683] What's Left? Concept Grounding with Logic-Enhanced Foundation Models

**Authors**: *Joy Hsu, Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html)

**Abstract**:

Recent works such as VisProg and ViperGPT have smartly composed foundation models for visual reasoning—using large language models (LLMs) to produce programs that can be executed by pre-trained vision-language models. However, they operate in limited domains, such as 2D images, not fully exploiting the generalization of language: abstract concepts like “left” can also be grounded in 3D, temporal, and action data, as in moving to your left. This limited generalization stems from these inference-only methods’ inability to learn or adapt pre-trained models to a new domain. We propose the Logic-Enhanced FoundaTion Model (LEFT), a unified framework that learns to ground and reason with concepts across domains with a differentiable, domain-independent, first-order logic-based program executor. LEFT has an LLM interpreter that outputs a program represented in a general, logic-based reasoning language, which is shared across all domains and tasks. LEFT’s executor then executes the program with trainable domain-specific grounding modules. We show that LEFT flexibly learns concepts in four domains: 2D images, 3D scenes, human motions, and robotic manipulation. It exhibits strong reasoning ability in a wide variety of tasks, including those that are complex and not seen during training, and can be easily applied to new domains.

----

## [1684] Recovering from Out-of-sample States via Inverse Dynamics in Offline Reinforcement Learning

**Authors**: *Ke Jiang, Jia-Yu Yao, Xiaoyang Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a0f7e9d9b42b26e5bfc9ba4c6e5287c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a0f7e9d9b42b26e5bfc9ba4c6e5287c-Abstract-Conference.html)

**Abstract**:

In this paper we deal with the state distributional shift problem commonly encountered in offline reinforcement learning during test, where the agent tends to take unreliable actions at out-of-sample (unseen) states. Our idea is to encourage the agent to follow the so called state recovery principle when taking actions, i.e., besides long-term return, the immediate consequences of the current action should also be taken into account and those capable of recovering the state distribution of the behavior policy are preferred. For this purpose, an inverse dynamics model is learned and employed to guide the state recovery behavior of the new policy. Theoretically, we show that the proposed method helps aligning the transited state distribution of the new policy with the offline dataset at out-of-sample states, without the need of  explicitly predicting the transited state distribution, which is usually difficult in high-dimensional and complicated environments. The effectiveness and feasibility of the proposed method is demonstrated with the state-of-the-art performance on the general offline RL benchmarks.

----

## [1685] VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

**Authors**: *Li-Wei H. Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari D. Clifford, Roger G. Mark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

False arrhythmia alarms in intensive care units (ICUs) are a continuing problem despite considerable effort from industrial and academic algorithm developers. Of all life-threatening arrhythmias, ventricular tachycardia (VT) stands out as the most challenging arrhythmia to detect reliably. We introduce a new annotated VT alarm database, VTaC (Ventricular Tachycardia annotated alarms from ICUs)  consisting of  over 5,000 waveform recordings with VT alarms triggered by bedside monitors in the ICU. Each VT alarm waveform in the dataset has been labeled by at least two independent human expert annotators. The dataset encompasses data collected from ICUs in two major US hospitals and includes data from three leading bedside monitor manufacturers, providing a diverse and representative collection of alarm waveform data. Each waveform recording comprises at least two electrocardiogram (ECG) leads and one or more pulsatile waveforms, such as photoplethysmogram (PPG or PLETH) and arterial blood pressure (ABP) waveforms. We demonstrate the utility of this new benchmark dataset for the task of false arrhythmia alarm reduction, and present performance of multiple machine learning approaches, including conventional supervised machine learning, deep learning, semi-supervised learning, and generative approaches for the task of VT false alarm reduction.

----

## [1686] TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation

**Authors**: *Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Yu Qiao, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html)

**Abstract**:

Training on large-scale datasets can boost the performance of video instance segmentation while the annotated datasets for VIS are hard to scale up due to the high labor cost. What we possess are numerous isolated filed-specific datasets, thus, it is appealing to jointly train models across the aggregation of datasets to enhance data volume and diversity. However, due to the heterogeneity in category space, as mask precision increase with the data volume, simply utilizing multiple datasets will dilute the attention of models on different taxonomy. Thus, increasing the data scale and enriching taxonomy space while improving classification precision is important. In this work, we analyze that providing extra taxonomy information can help models concentrate on specific taxonomy, and propose our model named Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation (TMT-VIS) to address this vital challenge. Specifically, we design a two-stage taxonomy aggregation module that first compiles taxonomy information from input videos and then aggregates these taxonomy priors into instance queries before the transformer decoder. We conduct extensive experimental evaluations on four popular and challenging benchmarks, including YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and UVO. Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all these benchmarks. These appealing and encouraging results demonstrate the effectiveness and generality of our proposed approach. The code and trained models will be publicly available.

----

## [1687] Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities

**Authors**: *Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, Lorenzo Torresani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a65606fa1a6849450550325832036e5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a65606fa1a6849450550325832036e5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human activities are goal-oriented and hierarchical, comprising primary goals at the top level, sequences of steps and substeps in the middle, and atomic actions at the lowest level. Recognizing human activities thus requires relating atomic actions and steps to their functional objectives (what the actions contribute to) and modeling their sequential and hierarchical dependencies towards achieving the goals. Current activity recognition research has primarily focused on only the lowest levels of this hierarchy, i.e., atomic or low-level actions, often in trimmed videos with annotations spanning only a few seconds. In this work, we introduce Ego4D Goal-Step, a new set of annotations on the recently released Ego4D with a novel hierarchical taxonomy of goal-oriented activity labels. It provides dense annotations for 48K procedural step segments (430 hours) and high-level goal annotations for 2,807 hours of Ego4D videos. Compared to existing procedural video datasets, it is substantially larger in size, contains hierarchical action labels (goals - steps - substeps), and provides goal-oriented auxiliary information including natural language summary description, step completion status, and step-to-goal relevance information. We take a data-driven approach to build our taxonomy, resulting in dense step annotations that do not suffer from poor label-data alignment issues resulting from a taxonomy defined a priori. Through comprehensive evaluations and analyses, we demonstrate how Ego4D Goal-Step supports exploring various questions in procedural activity understanding, including goal inference, step prediction, hierarchical relation learning, and long-term temporal modeling.

----

## [1688] The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter

**Authors**: *Ajay Jaiswal, Shiwei Liu, Tianlong Chen, Zhangyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a69ab48efcbb0153e72d458fb091969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a69ab48efcbb0153e72d458fb091969-Abstract-Conference.html)

**Abstract**:

Large pre-trained transformers are $\textit{show-stealer}$ in modern-day deep learning, and it becomes crucial to comprehend the parsimonious patterns that exist within them as they grow in scale. With exploding parameter counts, Lottery Ticket Hypothesis (LTH) and its variants, have lost their pragmatism in sparsifying them due to high computation and memory bottleneck of repetitive $\textit{train-prune-retrain}$ routine of iterative magnitude pruning (IMP) which worsens with increasing model size. In this paper, we comprehensively study $\textit{induced sparse patterns}$ across multiple large pre-trained vision and language transformers. We propose the existence of -- $\textbf{essential sparsity}$ defined with a $\textbf{sharp dropping point}$ beyond which the performance declines much faster w.r.t the rise of sparsity level, when we directly remove weights with the smallest magnitudes in $\textbf{one-shot}$. We also present an intriguing emerging phenomenon of $\textbf{abrupt sparsification}$ during the pre-training of BERT, i.e., BERT suddenly becomes heavily sparse in pre-training after certain iterations. Moreover, our observations also indicate a $\textbf{counter-intuitive}$ finding that BERT trained with a larger amount of pre-training data tends to have a better ability to condense knowledge in comparatively relatively fewer parameters. Lastly, we investigate the effect of the pre-training loss on essential sparsity and discover that self-supervised learning (SSL) objectives trigger stronger emergent sparsification properties than supervised learning (SL). All our codes will be publicly available.

----

## [1689] Hypervolume Maximization: A Geometric View of Pareto Set Learning

**Authors**: *Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, Qingfu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html)

**Abstract**:

This paper presents a novel approach to multiobjective algorithms aimed at modeling the Pareto set using neural networks. Whereas previous methods mainly focused on identifying a finite number of solutions, our approach allows for the direct modeling of the entire Pareto set. Furthermore, we establish an equivalence between learning the complete Pareto set and maximizing the associated hypervolume, which enables the convergence analysis of hypervolume (as a new metric) for Pareto set learning. Specifically, our new analysis framework reveals the connection between the learned Pareto solution and its representation in a polar coordinate system. We evaluate our proposed approach on various benchmark problems and real-world problems, and the encouraging results make it a potentially viable alternative to existing multiobjective algorithms. Code is available at \url{https://github.com/xzhang2523/hvpsl/tree/master}.

----

## [1690] Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics

**Authors**: *Koen Minartz, Yoeri Poels, Simon M. Koop, Vlado Menkovski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a8d388b7a17df480856dff1cc079b08-Abstract-Conference.html)

**Abstract**:

Neural networks are emerging as a tool for scalable data-driven simulation of high-dimensional dynamical systems, especially in settings where numerical methods are infeasible or computationally expensive. Notably, it has been shown that incorporating domain symmetries in deterministic neural simulators can substantially improve their accuracy, sample efficiency, and parameter efficiency. However, to incorporate symmetries in probabilistic neural simulators that can simulate stochastic phenomena, we need a model that produces equivariant distributions over trajectories, rather than equivariant function approximations. In this paper, we propose Equivariant Probabilistic Neural Simulation (EPNS), a framework for autoregressive probabilistic modeling of equivariant distributions over system evolutions. We use EPNS to design models for a stochastic n-body system and stochastic cellular dynamics. Our results show that EPNS considerably outperforms existing neural network-based methods for probabilistic simulation. More specifically, we demonstrate that incorporating equivariance in EPNS improves simulation quality, data efficiency, rollout stability, and uncertainty quantification. We conclude that EPNS is a promising method for efficient and effective data-driven probabilistic simulation in a diverse range of domains.

----

## [1691] Collaborative Alignment of NLP Models

**Authors**: *Fereshte Khani, Marco Túlio Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a8fa1382ea068f3f402b72081df16be-Abstract-Conference.html)

**Abstract**:

Despite substantial advancements, Natural Language Processing (NLP) models often require post-training adjustments to enforce business rules, rectify undesired behavior, and align with user values.  These adjustments involve operationalizing "concepts"â€”dictating desired model responses to certain inputs. However, it's difficult for a single entity to enumerate and define all possible concepts, indicating a need for a multi-user, collaborative model alignment framework. Moreover, the exhaustive delineation of a concept is challenging, and an improper approach can create shortcuts or interfere with original data or other concepts.To address these challenges, we introduce CoAlign, a framework that enables multi-user interaction with the model, thereby mitigating individual limitations. CoAlign aids users in operationalizing their concepts using Large Language Models, and relying on the principle that NLP models exhibit simpler behaviors in local regions. Our main insight is learning a \emph{local} model for each concept, and a \emph{global} model to integrate the original data with all concepts.We then steer a large language model to generate instances within concept boundaries where local and global disagree.Our experiments show CoAlign is effective at helping multiple users operationalize concepts and avoid interference for a variety of scenarios, tasks, and models.

----

## [1692] PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change

**Authors**: *Karthik Valmeekam, Matthew Marquez, Alberto Olmo Hernandez, Sarath Sreedharan, Subbarao Kambhampati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7a92bcdede88c7afd108072faf5485c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7a92bcdede88c7afd108072faf5485c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Generating plans of action, and reasoning about change have long been considered a core competence of intelligent agents. It is thus no surprise that evaluating the planning and reasoning capabilities of large language models (LLMs) has become a hot topic of research. Most claims about LLM planning capabilities are however based on common sense tasks–where it becomes hard to tell whether LLMs are planning or merely retrieving from their vast world knowledge.  There is a strong need for systematic and extensible planning benchmarks with sufficient diversity to evaluate whether LLMs have innate planning capabilities. Motivated by this, we propose PlanBench, an extensible benchmark suite based on the kinds of domains used in the automated planning community, especially in the International Planning Competition, to test the capabilities of LLMs in planning or reasoning about actions and change. PlanBench provides sufficient diversity in both the task domains and the specific planning capabilities. Our studies also show that on many critical capabilities–including plan generation–LLM performance falls quite short, even with the SOTA models. PlanBench can thus function as a useful marker of progress of LLMs in planning and reasoning.

----

## [1693] Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model

**Authors**: *Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7abbcb05a5d55157ede410bb718e32d7-Abstract-Conference.html)

**Abstract**:

In the field of behavior-related brain computation, it is necessary to align raw neural signals against the drastic domain shift among them. A foundational framework within neuroscience research posits that trial-based neural population activities rely on low-dimensional latent dynamics, thus focusing on the latter greatly facilitates the alignment procedure. Despite this field's progress, existing methods ignore the intrinsic spatio-temporal structure during the alignment phase. Hence, their solutions usually lead to poor quality in latent dynamics structures and overall performance. To tackle this problem, we propose an alignment method ERDiff, which leverages the expressivity of the diffusion model to preserve the spatio-temporal structure of latent dynamics. Specifically, the latent dynamics structures of the source domain are first extracted by a diffusion model. Then, under the guidance of this diffusion model, such structures are well-recovered through a maximum likelihood alignment procedure in the target domain. We first demonstrate the effectiveness of our proposed method on a synthetic dataset. Then, when applied to neural recordings from the non-human primate motor cortex, under both cross-day and inter-subject settings, our method consistently manifests its capability of preserving the spatio-temporal structure of latent dynamics and outperforms existing approaches in alignment goodness-of-fit and neural decoding performance.

----

## [1694] Closing the gap between the upper bound and lower bound of Adam's iteration complexity

**Authors**: *Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, Wei Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Abstract-Conference.html)

**Abstract**:

Recently,  Arjevani et al. [1]  establish a lower bound of iteration complexity for the first-order optimization under an $L$-smooth condition and a bounded noise variance assumption. However, a thorough review of existing literature on Adam's convergence reveals a noticeable gap: none of them meet the above lower bound. In this paper, we close the gap by deriving a new convergence guarantee of Adam, with only an $L$-smooth condition and a bounded noise variance assumption. Our results remain valid across a broad spectrum of hyperparameters. Especially with properly chosen hyperparameters, we derive an upper bound of the iteration complexity of Adam and show that it meets the lower bound for first-order optimizers. To the best of our knowledge, this is the first to establish such a tight upper bound for Adam's convergence. Our proof utilizes novel techniques to handle the entanglement between momentum and adaptive learning rate and to convert the first-order term in the Descent Lemma to the gradient norm, which may be of independent interest.

----

## [1695] Deep Patch Visual Odometry

**Authors**: *Zachary Teed, Lahav Lipson, Jia Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ac484b0f1a1719ad5be9aa8c8455fbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ac484b0f1a1719ad5be9aa8c8455fbb-Abstract-Conference.html)

**Abstract**:

We propose Deep Patch Visual Odometry (DPVO), a new deep learning system for monocular Visual Odometry (VO). DPVO uses a novel recurrent network architecture designed for tracking image patches across time. Recent approaches to VO have significantly improved the state-of-the-art accuracy by using deep networks to predict dense flow between video frames. However, using dense flow incurs a large computational cost, making these previous methods impractical for many use cases. Despite this, it has been assumed that dense flow is important as it provides additional redundancy against incorrect matches. DPVO disproves this assumption, showing that it is possible to get the best accuracy and efficiency by exploiting the advantages of sparse patch-based matching over dense flow. DPVO introduces a novel recurrent update operator for patch based correspondence coupled with differentiable bundle adjustment. On Standard benchmarks, DPVO outperforms all prior work, including the learning-based state-of-the-art VO-system (DROID) using a third of the memory while running 3x faster on average. Code is available at https://github.com/princeton-vl/DPVO

----

## [1696] BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information

**Authors**: *Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, Deepak Ramachandran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Automated reasoning with unstructured natural text is a key requirement for many potential applications of NLP and for developing robust AI systems. Recently, Language Models (LMs) have demonstrated complex reasoning capacities even without any finetuning. However, existing evaluation for automated reasoning assumes access to a consistent and coherent set of information over which models reason. When reasoning in the real-world, the available information is frequently inconsistent or contradictory, and therefore models need to be equipped with a strategy to resolve such conflicts when they arise. One widely-applicable way of resolving conflicts is to impose preferences over information sources (e.g., based on source credibility or information recency) and adopt the source with higher preference. In this paper, we formulate the problem of reasoning with contradictory information guided by preferences over sources as the classical problem of defeasible reasoning, and develop a dataset called BoardgameQA for measuring the reasoning capacity of LMs in this setting. BoardgameQA also incorporates reasoning with implicit background knowledge, to better reflect reasoning problems in downstream applications. We benchmark various LMs on BoardgameQA and the results reveal a significant gap in the reasoning capacity of state-of-the-art LMs on this problem, showing that reasoning with conflicting information does not surface out-of-the-box in LMs. While performance can be improved with finetuning, it nevertheless remains poor.

----

## [1697] Isometric Quotient Variational Auto-Encoders for Structure-Preserving Representation Learning

**Authors**: *In Huh, Changwook Jeong, Jae Myung Choe, Younggu Kim, Daesin Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html)

**Abstract**:

We study structure-preserving low-dimensional representation of a data manifold embedded in a high-dimensional observation space based on variational auto-encoders (VAEs). We approach this by decomposing the data manifold $\mathcal{M}$ as $\mathcal{M} = \mathcal{M} / G \times G$, where $G$ and $\mathcal{M} / G$ are a group of symmetry transformations and a quotient space of $\mathcal{M}$ up to $G$, respectively. From this perspective, we define the structure-preserving representation of such a manifold as a latent space $\mathcal{Z}$ which is isometrically isomorphic (i.e., distance-preserving) to the quotient space $\mathcal{M} / G$ rather $\mathcal{M}$ (i.e., symmetry-preserving). To this end, we propose a novel auto-encoding framework, named isometric quotient VAEs (IQVAEs), that can extract the quotient space from observations and learn the Riemannian isometry of the extracted quotient in an unsupervised manner. Empirical proof-of-concept experiments reveal that the proposed method can find a meaningful representation of the learned data and outperform other competitors for downstream tasks.

----

## [1698] SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents

**Authors**: *Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang, Yongbin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b16688a2b053a1b01474ab5c78ce662-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b16688a2b053a1b01474ab5c78ce662-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Task-oriented dialogue (TOD) models have made significant progress in recent years. However, previous studies primarily focus on datasets written by annotators, which has resulted in a gap between academic research and real-world spoken con- versation scenarios. While several small-scale spoken TOD datasets are proposed to address robustness issues such as ASR errors, they ignore the unique challenges in spoken conversation. To tackle the limitations, we introduce SpokenWOZ, a large-scale speech-text dataset for spoken TOD, containing 8 domains, 203k turns, 5.7k dialogues and 249 hours of audios from human-to-human spoken conversations. SpokenWOZ further incorporates common spoken characteristics such as word-by-word processing and reasoning in spoken language. Based on these characteristics, we present cross-turn slot and reasoning slot detection as new challenges. We conduct experiments on various baselines, including text-modal models, newly proposed dual-modal models, and LLMs, e.g., ChatGPT. The results show that the current models still have substantial room for improvement in spoken conversation, where the most advanced dialogue state tracker only achieves 25.65% in joint goal accuracy and the SOTA end-to-end model only correctly completes the user request in 52.1% of dialogues. Our dataset, code, and leaderboard are available at https://spokenwoz.github.io/SpokenWOZ-github.io/.

----

## [1699] Fast Partitioned Learned Bloom Filter

**Authors**: *Atsuki Sato, Yusuke Matsui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b2e844c52349134268e819a9b56b9e8-Abstract-Conference.html)

**Abstract**:

A Bloom filter is a memory-efficient data structure for approximate membership queries used in numerous fields of computer science.Recently, learned Bloom filters that achieve better memory efficiency using machine learning models have attracted attention.One such filter, the partitioned learned Bloom filter (PLBF), achieves excellent memory efficiency.However, PLBF requires a $\mathcal{O}(N^3k)$ time complexity to construct the data structure, where $N$ and $k$ are the hyperparameters of PLBF.One can improve memory efficiency by increasing $N$, but the construction time becomes extremely long.Thus, we propose two methods that can reduce the construction time while maintaining the memory efficiency of PLBF.First, we propose fast PLBF, which can construct the same data structure as PLBF with a smaller time complexity $\mathcal{O}(N^2k)$.Second, we propose fast PLBF++, which can construct the data structure with even smaller time complexity $\mathcal{O}(Nk\log N + Nk^2)$.Fast PLBF++ does not necessarily construct the same data structure as PLBF.Still, it is almost as memory efficient as PLBF, and it is proved that fast PLBF++ has the same data structure as PLBF when the distribution satisfies a certain constraint.Our experimental results from real-world datasets show that (i) fast PLBF and fast PLBF++ can construct the data structure up to 233 and 761 times faster than PLBF, (ii) fast PLBF can achieve the same memory efficiency as PLBF, and (iii) fast PLBF++ can achieve almost the same memory efficiency as PLBF.The codes are available at [this https URL](https://github.com/atsukisato/FastPLBF).

----

## [1700] Instructing Goal-Conditioned Reinforcement Learning Agents with Temporal Logic Objectives

**Authors**: *Wenjie Qiu, Wensen Mao, He Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b35a69f434b5eb07ed1b1ef16ace52c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b35a69f434b5eb07ed1b1ef16ace52c-Abstract-Conference.html)

**Abstract**:

Goal-conditioned reinforcement learning (RL) is a powerful approach for learning general-purpose skills by reaching diverse goals. However, it has limitations when it comes to task-conditioned policies, where goals are specified by temporally extended instructions written in the Linear Temporal Logic (LTL) formal language. Existing approaches for finding LTL-satisfying policies rely on sampling a large set of LTL instructions during training to adapt to unseen tasks at inference time. However, these approaches do not guarantee generalization to out-of-distribution LTL objectives, which may have increased complexity. In this paper, we propose a novel approach to address this challenge. We show that simple goal-conditioned RL agents can be instructed to follow arbitrary LTL specifications without additional training over the LTL task space. Unlike existing approaches that focus on LTL specifications expressible as regular expressions, our technique is unrestricted and generalizes to $\omega$-regular expressions. Experiment results demonstrate the effectiveness of our approach in adapting goal-conditioned RL agents to satisfy complex temporal logic task specifications zero-shot.

----

## [1701] Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement

**Authors**: *Jinbiao Chen, Zizhen Zhang, Zhiguang Cao, Yaoxin Wu, Yining Ma, Te Ye, Jiahai Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b5ae891000049b91b3b62de596b1560-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b5ae891000049b91b3b62de596b1560-Abstract-Conference.html)

**Abstract**:

Most of existing neural methods for multi-objective combinatorial optimization (MOCO) problems solely rely on decomposition, which often leads to repetitive solutions for the respective subproblems, thus a limited Pareto set. Beyond decomposition, we propose a novel neural heuristic with diversity enhancement (NHDE) to produce more Pareto solutions from two perspectives. On the one hand, to hinder duplicated solutions for different subproblems, we propose an indicator-enhanced deep reinforcement learning method to guide the model, and design a heterogeneous graph attention mechanism to capture the relations between the instance graph and the Pareto front graph. On the other hand, to excavate more solutions in the neighborhood of each subproblem, we present a multiple Pareto optima strategy to sample and preserve desirable solutions. Experimental results on classic MOCO problems show that our NHDE is able to generate a Pareto front with higher diversity, thereby achieving superior overall performance. Moreover, our NHDE is generic and can be applied to different neural methods for MOCO.

----

## [1702] Multi-Agent First Order Constrained Optimization in Policy Space

**Authors**: *Youpeng Zhao, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract-Conference.html)

**Abstract**:

In the realm of multi-agent reinforcement learning (MARL), achieving high performance is crucial for a successful multi-agent system.Meanwhile, the ability to avoid unsafe actions is becoming an urgent and imperative problem to solve for real-life applications. Whereas, it is still challenging to develop a safety-aware method for multi-agent systems in MARL. In this work, we introduce a novel approach called Multi-Agent First Order Constrained Optimization in Policy Space (MAFOCOPS), which effectively addresses the dual objectives of attaining satisfactory performance and enforcing safety constraints. Using data generated from the current policy, MAFOCOPS first finds the optimal update policy by solving a constrained optimization problem in the nonparameterized policy space. Then, the update policy is projected back into the parametric policy space to achieve a feasible policy. Notably, our method is first-order in nature, ensuring the ease of implementation, and exhibits an approximate upper bound on the worst-case constraint violation. Empirical results show that our approach achieves remarkable performance while satisfying safe constraints on several safe MARL benchmarks.

----

## [1703] This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations

**Authors**: *Chiyu Ma, Brandon Zhao, Chaofan Chen, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html)

**Abstract**:

We present ProtoConcepts, a method for interpretable image classification combining deep learning and case-based reasoning using prototypical parts. Existing work in prototype-based image classification uses a "this looks like that'' reasoning process, which dissects a test image by finding prototypical parts and combining evidence from these prototypes to make a final classification. However, all of the existing prototypical part-based image classifiers provide only one-to-one comparisons, where a single training image patch serves as a prototype to compare with a part of our test image. With these single-image comparisons, it can often be difficult to identify the underlying concept being compared (e.g., "is it comparing the color or the shape?''). Our proposed method modifies the architecture of prototype-based networks to instead learn prototypical concepts which are visualized using multiple image patches. Having multiple visualizations of the same prototype allows us to more easily identify the concept captured by that prototype (e.g., "the test image and the related training patches are all the same shade of blue''), and allows our model to create richer, more interpretable visual explanations. Our experiments show that our ``this looks like those'' reasoning process can be applied as a modification to a wide range of existing prototypical image classification networks while achieving comparable accuracy on benchmark datasets.

----

## [1704] Speculative Decoding with Big Little Decoder

**Authors**: *Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, Kurt Keutzer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html)

**Abstract**:

The recent emergence of Large Language Models based on the Transformer architecture has enabled dramatic advancements in the field of Natural Language Processing. However, these models have long inference latency, which limits their deployment and makes them prohibitively expensive for various real-time applications. The inference latency is further exacerbated by autoregressive generative tasks, as models need to run iteratively to generate tokens sequentially without leveraging token-level parallelization. To address this, we propose Big Little Decoder (BiLD), a framework that can improve inference efficiency and latency for a wide range of text generation applications. The BiLD framework contains two models with different sizes that collaboratively generate text. The small model runs autoregressively to generate text with a low inference cost, and the large model is only invoked occasionally to refine the small modelâ€™s inaccurate predictions in a non-autoregressive manner. To coordinate the small and large models, BiLD introduces two simple yet effective policies: (1) the fallback policy that determines when to hand control over to the large model; and (2) the rollback policy that determines when the large model needs to correct the small model's inaccurate predictions. To evaluate our framework across different tasks and models, we apply BiLD to various text generation scenarios encompassing machine translation on IWSLT 2017 De-En and WMT 2014 De-En, and summarization on XSUM and CNN/DailyMail. On an NVIDIA T4 GPU, our framework achieves a speedup of up to 2.12x speedup with minimal generation quality degradation. Furthermore, our framework is fully plug-and-play and can be applied without any modifications in the training process or model architecture. Our code is open-sourced.

----

## [1705] Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

**Authors**: *Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey I. Nikolenko, Evgeny Burnaev, Serguei Barannikov, Irina Piontkovskaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7baa48bc166aa2013d78cbdc15010530-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7baa48bc166aa2013d78cbdc15010530-Abstract-Conference.html)

**Abstract**:

Rapidly increasing quality of AI-generated content makes it difficult to distinguish between human and AI-generated texts, which may lead to undesirable consequences for society. Therefore, it becomes increasingly important to study the properties of human texts that are invariant over text domains and various proficiency of human writers, can be easily calculated for any language, and can robustly separate natural and AI-generated texts regardless of the generation model and sampling method. In this work, we propose such an invariant of human texts, namely the intrinsic dimensionality of the manifold underlying the set of embeddings of a given text sample. We show that the average intrinsic dimensionality of fluent texts in natural language is hovering around the value $9$ for several alphabet-based languages and around $7$ for Chinese, while the average intrinsic dimensionality of AI-generated texts for each language is $\approx 1.5$ lower, with a clear statistical separation between human-generated and AI-generated distributions. This property allows us to build a score-based artificial text detector. The proposed detector's accuracy is stable over text domains, generator models, and human writer proficiency levels, outperforming SOTA detectors in model-agnostic and cross-domain scenarios by a significant margin.

----

## [1706] Replicable Clustering

**Authors**: *Hossein Esfandiari, Amin Karbasi, Vahab Mirrokni, Grigoris Velegkas, Felix Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bc3fe234454107149fa9d44faacaa64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bc3fe234454107149fa9d44faacaa64-Abstract-Conference.html)

**Abstract**:

We design replicable algorithms in the context of statistical clustering under the recently introduced notion of replicability from Impagliazzo et al. [2022]. According to this definition, a clustering algorithm is replicable if, with high probability, its output induces the exact same partition of the sample space after two executions on different inputs drawn from the same distribution, when its internal randomness is shared across the executions. We propose such algorithms for the statistical $k$-medians, statistical $k$-means, and statistical $k$-centers problems by utilizing approximation routines for their combinatorial counterparts in a black-box manner. In particular, we demonstrate a replicable $O(1)$-approximation algorithm for statistical Euclidean $k$-medians ($k$-means) with $\operatorname{poly}(d)$ sample complexity. We also describe an $O(1)$-approximation algorithm with an additional $O(1)$-additive error for statistical Euclidean $k$-centers, albeit with $\exp(d)$ sample complexity. In addition, we provide experiments on synthetic distributions in 2D using the $k$-means++ implementation from sklearn as a black-box that validate our theoretical results.

----

## [1707] Counterfactual Memorization in Neural Language Models

**Authors**: *Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html)

**Abstract**:

Modern neural language models that are widely used in various NLP tasks risk memorizing sensitive information from their training data.Understanding this memorization is important in real world applications and also from a learning-theoretical perspective. An open question in previous studies of language model memorization is how to filter out ``common'' memorization. In fact, most memorization criteria strongly correlate with the number of occurrences in the training set, capturing memorized familiar phrases, public knowledge, templated texts, or other repeated data.We formulate a notion of counterfactual memorization which characterizes how a model's predictions change if a particular document is omitted during training.We identify and study counterfactually-memorized training examples in standard text datasets.We estimate the influence of each memorized training example on the validation set and on generated texts, showing how this can provide direct evidence of the source of memorization at test time.

----

## [1708] Learning Generalizable Agents via Saliency-guided Features Decorrelation

**Authors**: *Sili Huang, Yanchao Sun, Jifeng Hu, Siyuan Guo, Hechang Chen, Yi Chang, Lichao Sun, Bo Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bd4a7d0e6773072c2e3c77b11d93065-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bd4a7d0e6773072c2e3c77b11d93065-Abstract-Conference.html)

**Abstract**:

In visual-based Reinforcement Learning (RL), agents often struggle to generalize well to environmental variations in the state space that were not observed during training. The variations can arise in both task-irrelevant features, such as background noise, and task-relevant features, such as robot configurations, that are related to the optimal decisions. To achieve generalization in both situations, agents are required to accurately understand the impact of changed features on the decisions, i.e., establishing the true associations between changed features and decisions in the policy model. However, due to the inherent correlations among features in the state space, the associations between features and decisions become entangled, making it difficult for the policy to distinguish them. To this end, we propose Saliency-Guided Features Decorrelation (SGFD) to eliminate these correlations through sample reweighting. Concretely, SGFD consists of two core techniques: Random Fourier Functions (RFF) and the saliency map. RFF is utilized to estimate the complex non-linear correlations in high-dimensional images, while the saliency map is designed to identify the changed features. Under the guidance of the saliency map, SGFD employs sample reweighting to minimize the estimated correlations related to changed features, thereby achieving decorrelation in visual RL tasks. Our experimental results demonstrate that SGFD can generalize well on a wide range of test environments and significantly outperforms state-of-the-art methods in handling both task-irrelevant variations and task-relevant variations.

----

## [1709] You Only Condense Once: Two Rules for Pruning Condensed Datasets

**Authors**: *Yang He, Lingao Xiao, Joey Tianyi Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bdd36a198a8408f444834039b09f518-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bdd36a198a8408f444834039b09f518-Abstract-Conference.html)

**Abstract**:

Dataset condensation is a crucial tool for enhancing training efficiency by reducing the size of the training dataset, particularly in on-device scenarios. However, these scenarios have two significant challenges: 1) the varying computational resources available on the devices require a dataset size different from the pre-defined condensed dataset, and 2) the limited computational resources often preclude the possibility of conducting additional condensation processes. We introduce You Only Condense Once (YOCO) to overcome these limitations. On top of one condensed dataset, YOCO produces smaller condensed datasets with two embarrassingly simple dataset pruning rules: Low LBPE Score and Balanced Construction. YOCO offers two key advantages: 1) it can flexibly resize the dataset to fit varying computational constraints, and 2) it eliminates the need for extra condensation processes, which can be computationally prohibitive. Experiments validate our findings on networks including ConvNet, ResNet and DenseNet, and datasets including CIFAR-10, CIFAR-100 and ImageNet. For example, our YOCO surpassed various dataset condensation and dataset pruning methods on CIFAR-10 with ten Images Per Class (IPC), achieving 6.98-8.89% and 6.31-23.92% accuracy gains, respectively. The code is available at: https://github.com/he-y/you-only-condense-once.

----

## [1710] Provably Efficient Offline Reinforcement Learning in Regular Decision Processes

**Authors**: *Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bf3e93543a612b75b6373178ba1faa4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bf3e93543a612b75b6373178ba1faa4-Abstract-Conference.html)

**Abstract**:

This paper deals with offline (or batch) Reinforcement Learning (RL) in episodic Regular Decision Processes (RDPs). RDPs are the subclass of Non-Markov Decision Processes where the dependency on the history of past events can be captured by a finite-state automaton. We consider a setting where the automaton that underlies the RDP is unknown, and a learner strives to learn a near-optimal policy using pre-collected data, in the form of non-Markov sequences of observations, without further exploration. We present RegORL, an algorithm that suitably combines automata learning techniques and state-of-the-art algorithms for offline RL in MDPs. RegORL has a modular design allowing one to use any off-the-shelf offline RL algorithm in MDPs. We report a non-asymptotic high-probability sample complexity bound for RegORL to yield an $\varepsilon$-optimal policy, which makes appear a notion of concentrability relevant for RDPs. Furthermore, we present a sample complexity lower bound for offline RL in RDPs. To our best knowledge, this is the first work presenting a provably efficient algorithm for offline learning in RDPs.

----

## [1711] CP-SLAM: Collaborative Neural Point-based SLAM System

**Authors**: *Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, Zhaopeng Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c10e259c7e56fa218ee03d9ae7d728e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c10e259c7e56fa218ee03d9ae7d728e-Abstract-Conference.html)

**Abstract**:

This paper presents a collaborative implicit neural simultaneous localization and mapping (SLAM) system with RGB-D image sequences, which consists of complete front-end and back-end modules including odometry, loop detection, sub-map fusion, and global refinement. In order to enable all these modules in a unified framework, we propose a novel neural point based 3D scene representation in which each point maintains a learnable neural feature for scene encoding and is associated with a certain keyframe. Moreover, a distributed-to-centralized learning strategy is proposed for the collaborative implicit SLAM to improve consistency and cooperation. A novel global optimization framework is also proposed to improve the system accuracy like traditional bundle adjustment. Experiments on various datasets demonstrate the superiority of the proposed method in both camera tracking and mapping.

----

## [1712] The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation

**Authors**: *Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, David J. Fleet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c119415672ae2186e17d492e1d5da2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c119415672ae2186e17d492e1d5da2f-Abstract-Conference.html)

**Abstract**:

Denoising diffusion probabilistic models have transformed image generation with their impressive fidelity and diversity.We show that they also excel in estimating optical flow and monocular depth, surprisingly without task-specific architectures and loss functions that are predominant for these tasks. Compared to the point estimates of conventional regression-based methods, diffusion models also enable Monte Carlo inference, e.g., capturing uncertainty and ambiguity in flow and depth.With self-supervised pre-training, the combined use of synthetic and real data for supervised training, and technical innovations (infilling and step-unrolled denoising diffusion training) to handle noisy-incomplete training data, one can train state-of-the-art diffusion models for depth and optical flow estimation, with additional zero-shot coarse-to-fine refinement for high resolution estimates. Extensive experiments focus on quantitative performance against benchmarks, ablations, and the model's ability to capture uncertainty and multimodality, and impute missing values. Our model obtains a state-of-the-art relative depth error of 0.074 on the indoor NYU benchmark and an Fl-all score of 3.26\% on the KITTI  optical flow benchmark, about 25\% better than the best published method.

----

## [1713] Efficient Testable Learning of Halfspaces with Adversarial Label Noise

**Authors**: *Ilias Diakonikolas, Daniel Kane, Vasilis Kontonis, Sihan Liu, Nikos Zarifis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html)

**Abstract**:

We give the first polynomial-time algorithm for the testable learning of halfspaces in the presence of adversarial label noise under the Gaussian distribution. In the recently introduced testable learning model, one is required to produce a tester-learner such that if the data passes the tester, then one can trust the output of the robust learner on the data. Our tester-learner runs in time $\text{poly}(d/\epsilon)$ and outputs a halfspace with misclassification error $O(\text{opt})+\epsilon$, where $\text{opt}$ is the 0-1 error of the best fitting halfspace. At a technical level, our algorithm employs an iterative soft localization technique enhanced with appropriate testers to ensure that the data distribution is sufficiently similar to a Gaussian. Finally, our algorithm can be readily adapted to yield an efficient and testable active learner requiring only $d ~ \text{polylog}(1/\epsilon)$ labeled examples.

----

## [1714] Achieving O(ε-15) Complexity in Hessian/Jacobian-free Stochastic Bilevel Optimization

**Authors**: *Yifan Yang, Peiyao Xiao, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html)

**Abstract**:

In this paper, we revisit the bilevel optimization problem, in which the upper-level objective function is generally nonconvex and the lower-level objective function is strongly convex. Although this type of problem has been studied extensively, it still remains an open question how to achieve an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity in Hessian/Jacobian-free stochastic bilevel optimization without any second-order derivative computation. To fill this gap, we propose a novel Hessian/Jacobian-free bilevel optimizer named FdeHBO, which features a simple fully single-loop structure, a projection-aided finite-difference Hessian/Jacobian-vector approximation, and momentum-based updates. Theoretically, we show that FdeHBO requires $\mathcal{O}(\epsilon^{-1.5})$ iterations (each using $\mathcal{O}(1)$ samples and only first-order gradient information) to find an $\epsilon$-accurate stationary point. As far as we know, this is the first Hessian/Jacobian-free method with an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity for nonconvex-strongly-convex stochastic bilevel optimization.

----

## [1715] Robust and Actively Secure Serverless Collaborative Learning

**Authors**: *Nicholas Franzese, Adam Dziedzic, Christopher A. Choquette-Choo, Mark R. Thomas, Muhammad Ahmad Kaleem, Stephan Rabanser, Congyu Fang, Somesh Jha, Nicolas Papernot, Xiao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c5a4b7a31dffef8ce296deedb6214a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c5a4b7a31dffef8ce296deedb6214a9-Abstract-Conference.html)

**Abstract**:

Collaborative machine learning (ML) is widely used to enable institutions to learn better models from distributed data. While collaborative approaches to learning intuitively protect user data, they remain vulnerable to either the server, the clients, or both, deviating from the protocol. Indeed, because the protocol is asymmetric, a malicious server can abuse its power to reconstruct client data points. Conversely, malicious clients can corrupt learning with malicious updates. Thus, both clients and servers require a guarantee when the other cannot be trusted to fully cooperate. In this work, we propose a peer-to-peer (P2P) learning scheme that is secure against malicious servers and robust to malicious clients. Our core contribution is a generic framework that transforms any (compatible) algorithm for robust aggregation of model updates to the setting where servers and clients can act maliciously. Finally, we demonstrate the computational efficiency of our approach even with 1-million parameter models trained by 100s of peers on standard datasets.

----

## [1716] Birder: Communication-Efficient 1-bit Adaptive Optimizer for Practical Distributed DNN Training

**Authors**: *Hanyang Peng, Shuang Qin, Yue Yu, Jin Wang, Hui Wang, Ge Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c72fcd7b6bffc3864c7152ab5a2dd83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c72fcd7b6bffc3864c7152ab5a2dd83-Abstract-Conference.html)

**Abstract**:

Various gradient compression algorithms have been proposed to alleviate the communication bottleneck in distributed learning, and they have demonstrated effectiveness in terms of  high compression ratios and theoretical low communication complexity. However, when it comes to practically training modern deep neural networks (DNNs),  these algorithms have yet to match the inference performance of uncompressed SGD-momentum  (SGDM) and adaptive optimizers (e.g.,Adam). More importantly,  recent studies suggest that these algorithms actually offer no speed advantages over SGDM/Adam when used with common  distributed DNN training frameworks ( e.g., DistributedDataParallel (DDP)) in the typical settings, due to heavy compression/decompression computation or incompatibility with the efficient All-Reduce or the requirement of uncompressed warmup at the early stage. For these reasons, we propose a novel 1-bit adaptive optimizer, dubbed *Bi*nary *r*andomization a*d*aptive optimiz*er* (**Birder**). The quantization of Birder can be easily and lightly computed, and it does not require warmup with its uncompressed version in the beginning. Also, we devise Hierarchical-1-bit-All-Reduce to further lower the communication volume. We theoretically prove that it promises the same convergence rate as the Adam. Extensive experiments, conducted on 8 to 64 GPUs (1 to 8 nodes) using DDP, demonstrate that Birder achieves comparable inference performance to uncompressed SGDM/Adam, with up to ${2.5 \times}$ speedup for training ResNet-50 and ${6.3\times}$ speedup for training BERT-Base. Code is publicly available at https://openi.pcl.ac.cn/c2net_optim/Birder.

----

## [1717] MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition

**Authors**: *Nicolas Menet, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c7a12559be4501f70d221352514397c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c7a12559be4501f70d221352514397c-Abstract-Conference.html)

**Abstract**:

With the advent of deep learning, progressively larger neural networks have been designed to solve complex tasks. We take advantage of these capacity-rich models to lower the cost of inference by exploiting computation in superposition. To reduce the computational burden per input, we propose Multiple-Input-Multiple-Output Neural Networks (MIMONets) capable of handling many inputs at once. MIMONets augment various deep neural network architectures with variable binding mechanisms to represent an arbitrary number of inputs in a compositional data structure via fixed-width distributed representations. Accordingly, MIMONets adapt nonlinear neural transformations to process the data structure holistically, leading to a speedup nearly proportional to the number of superposed input items in the data structure. After processing in superposition, an unbinding mechanism recovers each transformed input of interest. MIMONets also provide a dynamic trade-off between accuracy and throughput by an instantaneous on-demand switching between a set of accuracy-throughput operating points, yet within a single set of fixed parameters. We apply the concept of MIMONets to both CNN and Transformer architectures resulting in MIMOConv and MIMOFormer, respectively. Empirical evaluations show that MIMOConv achieves $\approx 2$–$4\times$ speedup at an accuracy delta within [+0.68, -3.18]% compared to WideResNet CNNs on CIFAR10 and CIFAR100.  Similarly, MIMOFormer can handle $2$–$4$ inputs at once while maintaining a high average accuracy within a [-1.07, -3.43]% delta on the long range arena benchmark. Finally, we provide mathematical bounds on the interference between superposition channels in MIMOFormer. Our code is available at https://github.com/IBM/multiple-input-multiple-output-nets.

----

## [1718] C-Disentanglement: Discovering Causally-Independent Generative Factors under an Inductive Bias of Confounder

**Authors**: *Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ca55c8276acf1f0aa996cd3622d1df4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ca55c8276acf1f0aa996cd3622d1df4-Abstract-Conference.html)

**Abstract**:

Representation learning assumes that real-world data is generated by a few semantically meaningful generative factors (i.e., sources of variation) and aims to discover them in the latent space. These factors are expected to be causally disentangled, meaning that distinct factors are encoded into separate latent variables, and changes in one factor will not affect the values of the others. Compared to statistical independence, causal disentanglement allows more controllable data generation, improved robustness, and better generalization. However, most existing work assumes unconfoundedness in the discovery process, that there are no common causes to the generative factors and thus obtain only statistical independence. In this paper, we recognize the importance of modeling confounders in discovering causal generative factors. Unfortunately, such factors are not identifiable without proper inductive bias. We fill the gap by introducing a framework entitled Confounded-Disentanglement (C-Disentanglement), the first framework that explicitly introduces the inductive bias of confounder via labels from domain expertise. In addition, we accordingly propose an approach to sufficiently identify the causally-disentangled factors under any inductive bias of the confounder.  We conduct extensive experiments on both synthetic and real-world datasets. Our method demonstrates competitive results compared to various SOTA baselines in obtaining causally disentangled features and downstream tasks under domain shifts.

----

## [1719] Representation Learning via Consistent Assignment of Views over Random Partitions

**Authors**: *Thalles Santos Silva, Adín Ramírez Rivera*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html)

**Abstract**:

We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views' assignments.  Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP's representations are suitable for learning downstream tasks. We evaluate CARP's representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, $k$-NN, $k$-means, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks.In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.

----

## [1720] Federated Multi-Objective Learning

**Authors**: *Haibo Yang, Zhuqing Liu, Jia Liu, Chaosheng Dong, Michinari Momma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cb2c2a8d35576c00078b6591ec26a7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cb2c2a8d35576c00078b6591ec26a7d-Abstract-Conference.html)

**Abstract**:

In recent years, multi-objective optimization (MOO) emerges as a foundational problem underpinning many multi-agent multi-task learning applications. However, existing algorithms in MOO literature remain limited to centralized learning settings, which do not satisfy the distributed nature and data privacy needs of such multi-agent multi-task learning applications. This motivates us to propose a new federated multi-objective learning (FMOL) framework with multiple clients distributively and collaboratively solving an MOO problem while keeping their training data private. Notably, our FMOL framework allows a different set of objective functions across different clients to support a wide range of applications, which advances and generalizes the MOO formulation to the federated learning paradigm for the first time. For this FMOL framework, we propose two new federated multi-objective optimization (FMOO) algorithms called federated multi-gradient descent averaging (FMGDA) and federated stochastic multi-gradient descent averaging (FSMGDA). Both algorithms allow local updates to significantly reduce communication costs, while achieving the {\em same} convergence rates as those of their algorithmic counterparts in the single-objective federated learning. Our extensive experiments also corroborate the efficacy of our proposed FMOO algorithms.

----

## [1721] MARBLE: Music Audio Representation Benchmark for Universal Evaluation

**Authors**: *Ruibin Yuan, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, Le Zhuo, Yiqi Liu, Jiawen Huang, Zeyue Tian, Binyue Deng, Ningzhi Wang, Chenghua Lin, Emmanouil Benetos, Anton Ragni, Norbert Gyenge, Roger B. Dannenberg, Wenhu Chen, Gus Xia, Wei Xue, Si Liu, Shi Wang, Ruibo Liu, Yike Guo, Jie Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the era of extensive intersection between art and Artificial Intelligence (AI), such as image generation and fiction co-creation, AI for music remains relatively nascent, particularly in music understanding. This is evident in the limited work on deep music representations, the scarcity of large-scale datasets, and the absence of a universal and community-driven benchmark. To address this issue, we introduce the Music Audio Representation Benchmark for universaL Evaluation, termed MARBLE. It aims to provide a benchmark for various Music Information Retrieval (MIR) tasks by defining a comprehensive taxonomy with four hierarchy levels, including acoustic, performance, score, and high-level description. We then establish a unified protocol based on 18 tasks on 12 public-available datasets, providing a fair and standard assessment of representations of all open-sourced pre-trained models developed on music recordings as baselines. Besides, MARBLE offers an easy-to-use, extendable, and reproducible suite for the community, with a clear statement on copyright issues on datasets. Results suggest recently proposed large-scale pre-trained musical language models perform the best in most tasks, with room for further improvement. The leaderboard and toolkit repository are published to promote future music AI research.

----

## [1722] Language Models can Solve Computer Tasks

**Authors**: *Geunwoo Kim, Pierre Baldi, Stephen McAleer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html)

**Abstract**:

Agents capable of carrying out general tasks on a computer can improve efficiency and productivity by automating repetitive tasks and assisting in complex problem-solving. Ideally, such agents should be able to solve new computer tasks presented to them through natural language commands. However, previous approaches to this problem require large amounts of expert demonstrations and task-specific reward functions, both of which are impractical for new tasks. In this work, we show that a pre-trained large language model (LLM) agent can execute computer tasks guided by natural language using a simple prompting scheme where the agent \textbf{R}ecursively \textbf{C}riticizes and \textbf{I}mproves its output (RCI). The RCI approach significantly outperforms existing LLM methods for automating computer tasks and surpasses supervised learning (SL) and reinforcement learning (RL) approaches on the MiniWoB++ benchmark. We compare multiple LLMs and find that RCI with the InstructGPT-3+RLHF LLM is state-of-the-art on MiniWoB++, using only a handful of demonstrations per task rather than tens of thousands, and without a task-specific reward function. Furthermore, we demonstrate RCI prompting's effectiveness in enhancing LLMs' reasoning abilities on a suite of natural language reasoning tasks, outperforming chain of thought (CoT) prompting with external feedback. We find that RCI combined with CoT performs better than either separately. Our code can be found here: https://github.com/posgnu/rci-agent.

----

## [1723] An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement

**Authors**: *Gaku Morio, Christopher D. Manning*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ccaa4f9a89cce6619093226f26b84e6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ccaa4f9a89cce6619093226f26b84e6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As societal awareness of climate change grows, corporate climate policy engagements are attracting attention.We propose a dataset to estimate corporate climate policy engagement from various PDF-formatted documents.Our dataset comes from LobbyMap (a platform operated by global think tank InfluenceMap) that provides engagement categories and stances on the documents.To convert the LobbyMap data into the structured dataset, we developed a pipeline using text extraction and OCR.Our contributions are: (i) Building an NLP dataset including 10K documents on corporate climate policy engagement. (ii) Analyzing the properties and challenges of the dataset. (iii) Providing experiments for the dataset using pre-trained language models.The results show that while Longformer outperforms baselines and other pre-trained models, there is still room for significant improvement.We hope our work begins to bridge research on NLP and climate change.

----

## [1724] Robustness Guarantees for Adversarially Trained Neural Networks

**Authors**: *Poorya Mianjy, Raman Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cde9bd7774c9f5056cb6e5474fbadff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cde9bd7774c9f5056cb6e5474fbadff-Abstract-Conference.html)

**Abstract**:

We study robust adversarial training of two-layer neural networks as a bi-level optimization problem. In particular, for the inner loop that implements the adversarial attack during training using projected gradient descent (PGD), we propose maximizing a \emph{lower bound} on the $0/1$-loss by reflecting a surrogate loss about the origin. This allows us to give a convergence guarantee for the inner-loop PGD attack. Furthermore, assuming the data is linearly separable, we provide precise iteration complexity results for end-to-end adversarial training, which holds for any width and initialization. We provide empirical evidence to support our theoretical results.

----

## [1725] Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning

**Authors**: *Jialong Wu, Haoyu Ma, Chaoyi Deng, Mingsheng Long*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html)

**Abstract**:

Unsupervised pre-training methods utilizing large and diverse datasets have achieved tremendous success across a range of domains. Recent work has investigated such unsupervised pre-training methods for model-based reinforcement learning (MBRL) but is limited to domain-specific or simulated data. In this paper, we study the problem of pre-training world models with abundant in-the-wild videos for efficient learning of downstream visual control tasks. However, in-the-wild videos are complicated with various contextual factors, such as intricate backgrounds and textured appearance, which precludes a world model from extracting shared world knowledge to generalize better. To tackle this issue, we introduce Contextualized World Models (ContextWM) that explicitly separate context and dynamics modeling to overcome the complexity and diversity of in-the-wild videos and facilitate knowledge transfer between distinct scenes. Specifically, a contextualized extension of the latent dynamics model is elaborately realized by incorporating a context encoder to retain contextual information and empower the image decoder, which encourages the latent dynamics model to concentrate on essential temporal variations. Our experiments show that in-the-wild video pre-training equipped with ContextWM can significantly improve the sample efficiency of MBRL in various domains, including robotic manipulation, locomotion, and autonomous driving. Code is available at this repository: https://github.com/thuml/ContextWM.

----

## [1726] Strategyproof Voting under Correlated Beliefs

**Authors**: *Daniel Halpern, Rachel Li, Ariel D. Procaccia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cefded8659ccc899196860af674b596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cefded8659ccc899196860af674b596-Abstract-Conference.html)

**Abstract**:

In voting theory, when voters have ranked preferences over candidates, the celebrated Gibbard-Satterthwaite Theorem essentially rules out the existence of reasonable strategyproof methods for picking a winner. What if we weaken strategyproofness to only hold for Bayesian voters with beliefs over others' preferences? When voters believe other participants' rankings are drawn independently from a fixed distribution, the impossibility persists. However, it is quite reasonable for a voter to believe that other votes are correlated, either to each other or to their own ranking. We consider such beliefs induced by classic probabilistic models in social choice such as the Mallows, Placket-Luce, and Thurstone-Mosteller models. We single out the plurality rule (choosing the candidate ranked first most often) as a particularly promising choice as it is strategyproof for a large class of beliefs containing the specific ones we introduce. Further, we show that plurality is unique among positional scoring rules in having this property: no other scoring rule is strategyproof for beliefs induced by the Mallows model when there are a sufficient number of voters. Finally, we give examples of prominent non-scoring voting rules failing to be strategyproof on beliefs in this class, further bolstering the case for plurality.

----

## [1727] PCF-GAN: generating sequential data via the characteristic function of measures on the path space

**Authors**: *Hang Lou, Siran Li, Hao Ni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d0e867582cdc156fd280d5a6aa1be08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d0e867582cdc156fd280d5a6aa1be08-Abstract-Conference.html)

**Abstract**:

Generating high-fidelity time series data using generative adversarial networks (GANs) remains a challenging task, as it is difficult to capture the temporal dependence of joint probability distributions induced by time-series data. Towards this goal, a key step is the development of an effective discriminator to distinguish between time series distributions. We propose the so-called PCF-GAN, a novel GAN that incorporates the path characteristic function (PCF) as the principled representation of time series distribution into the discriminator to enhance its generative performance.  On the one hand, we establish theoretical foundations of the PCF distance by proving its characteristicity, boundedness, differentiability with respect to generator parameters, and weak continuity, which ensure the stability and feasibility of training the PCF-GAN. On the other hand, we design efficient initialisation and optimisation schemes for PCFs to strengthen the discriminative power and accelerate training efficiency. To further boost the capabilities of complex time series generation, we integrate the auto-encoder structure via sequential embedding into the PCF-GAN, which provides additional reconstruction functionality. Extensive numerical experiments on various datasets demonstrate the consistently superior performance of PCF-GAN over state-of-the-art baselines, in both generation and reconstruction quality.

----

## [1728] A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods

**Authors**: *Veit David Wild, Sahra Ghalebikesabi, Dino Sejdinovic, Jeremias Knoblauch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d25b1db211d99d5750ec45d65fd6e4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d25b1db211d99d5750ec45d65fd6e4e-Abstract-Conference.html)

**Abstract**:

We establish the first mathematically rigorous link between Bayesian, variational Bayesian, and ensemble methods. A key step towards this it to reformulate the non-convex optimisation problem typically encountered in deep learning as a convex optimisation in the space of probability measures. On a technical level, our contribution amounts to studying generalised variational inference through the lense of Wasserstein gradient flows. The result is a unified theory of various seemingly disconnected approaches that are commonly used for uncertainty quantification in deep learning---including deep ensembles and (variational) Bayesian methods. This offers a fresh perspective on the reasons behind the success of deep ensembles over procedures based on parameterised variational inference, and allows the derivation of new ensembling schemes with convergence guarantees. We showcase this by proposing a family of interacting deep ensembles with direct parallels to the interactions of particle systems in thermodynamics, and use our theory to prove the convergence of these algorithms to a well-defined global minimiser on the space of probability measures.

----

## [1729] Markovian Sliced Wasserstein Distances: Beyond Independent Projections

**Authors**: *Khai Nguyen, Tongzheng Ren, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html)

**Abstract**:

Sliced Wasserstein (SW) distance suffers from redundant projections due to independent uniform random projecting directions. To partially overcome the issue, max K sliced Wasserstein  (Max-K-SW) distance ($K\geq 1$), seeks the best discriminative orthogonal projecting directions. Despite being able to reduce the number of projections, the metricity of the Max-K-SW cannot be guaranteed in practice due to the non-optimality of the optimization. Moreover,  the orthogonality constraint is also computationally expensive and might not be effective. To address the problem, we introduce a new family of SW distances, named Markovian sliced Wasserstein (MSW) distance, which imposes a first-order Markov structure on projecting directions. We discuss various members of the MSW by specifying the Markov structure including the prior distribution, the transition distribution, and the burning and thinning technique. Moreover, we investigate the theoretical properties of  MSW including topological properties (metricity, weak convergence, and connection to other distances), statistical properties (sample complexity, and Monte Carlo estimation error), and computational properties (computational complexity and memory complexity). Finally, we compare MSW distances with previous SW variants in various applications such as gradient flows, color transfer, and deep generative modeling to demonstrate the favorable performance of the MSW.

----

## [1730] Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning

**Authors**: *Changyu Chen, Ramesha Karunasena, Thanh Hong Nguyen, Arunesh Sinha, Pradeep Varakantham*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d4c0094ae32530494c71468558ab5b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d4c0094ae32530494c71468558ab5b1-Abstract-Conference.html)

**Abstract**:

Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists. In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy â€” the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.

----

## [1731] On the impact of activation and normalization in obtaining isometric embeddings at initialization

**Authors**: *Amir Joudaki, Hadi Daneshmand, Francis R. Bach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html)

**Abstract**:

In this paper, we explore the structure of the penultimate Gram matrix in deep neural networks, which contains the pairwise inner products of outputs corresponding to a batch of inputs. In several architectures it has been observed that this Gram matrix becomes degenerate with depth at initialization, which dramatically slows training. Normalization layers, such as batch or layer normalization, play a pivotal role in preventing the rank collapse issue. Despite promising advances, the existing theoretical results do not extend to layer normalization, which is widely used in transformers, and can not quantitatively characterize the role of non-linear activations.  To bridge this gap, we prove that layer normalization, in conjunction with activation layers, biases the Gram matrix of a multilayer perceptron towards the identity matrix at an exponential rate with depth at initialization. We quantify this rate using the Hermite expansion of the activation function.

----

## [1732] Uni3DETR: Unified 3D Detection Transformer

**Authors**: *Zhenyu Wang, Ya-Li Li, Xi Chen, Hengshuang Zhao, Shengjin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d60bfd8458b67acbbaf18b892338d00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d60bfd8458b67acbbaf18b892338d00-Abstract-Conference.html)

**Abstract**:

Existing point cloud based 3D detectors are designed for the particular scene, either indoor or outdoor ones. Because of the substantial differences in object distribution and point density within point clouds collected from various environments, coupled with the intricate nature of 3D metrics, there is still a lack of a unified network architecture that can accommodate diverse scenes. In this paper, we propose Uni3DETR, a unified 3D detector that addresses indoor and outdoor 3D detection within the same framework. Specifically, we employ the detection transformer with point-voxel interaction for object prediction, which leverages voxel features and points for cross-attention and behaves resistant to the discrepancies from data. We then propose the mixture of query points, which sufficiently exploits global information for dense small-range indoor scenes and local information for large-range sparse outdoor ones. Furthermore, our proposed decoupled IoU provides an easy-to-optimize training target for localization by disentangling the $xy$ and $z$ space. Extensive experiments validate that Uni3DETR exhibits excellent performance consistently on both indoor and outdoor 3D detection. In contrast to previous specialized detectors, which may perform well on some particular datasets but suffer a substantial degradation  on different scenes, Uni3DETR demonstrates the strong generalization ability under heterogeneous conditions (Fig. 1).

----

## [1733] SceneScape: Text-Driven Consistent Scene Generation

**Authors**: *Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html)

**Abstract**:

We present a method for text-driven perpetual view generation -- synthesizing long-term videos of various scenes solely, given an input text prompt describing the scene and camera poses. We introduce a novel framework that generates such videos in an online fashion by combining the generative power of a pre-trained text-to-image model with the geometric priors learned by a pre-trained monocular depth prediction model. To tackle the pivotal challenge of achieving 3D consistency, i.e., synthesizing videos that depict geometrically-plausible scenes, we deploy an online test-time training to encourage the predicted depth map of the current frame to be geometrically consistent with the synthesized scene. The depth maps are used to construct a \emph{unified} mesh representation of the scene, which is progressively constructed along the video generation process. In contrast to previous works, which are applicable only to limited domains, our method generates diverse scenes, such as walkthroughs in spaceships, caves, or ice castles.

----

## [1734] RDumb: A simple approach that questions our progress in continual test-time adaptation

**Authors**: *Ori Press, Steffen Schneider, Matthias Kümmerer, Matthias Bethge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d640f377893fc5f22b5610e175ef7c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d640f377893fc5f22b5610e175ef7c3-Abstract-Conference.html)

**Abstract**:

Test-Time Adaptation (TTA) allows to update pre-trained models to changing data distributions at deployment time. While early work tested these algorithms for individual fixed distribution shifts, recent work proposed and applied methods for continual adaptation over long timescales. To examine the reported progress in the field, we propose the Continually Changing Corruptions (CCC) benchmark to measure asymptotic performance of TTA techniques. We find that eventually all but one state-of-the-art methods collapse and perform worse than a non-adapting model, including models specifically proposed to be robust to performance collapse. In addition, we introduce a simple baseline, "RDumb", that periodically resets the model to its pretrained state. RDumb performs better or on par with the previously proposed state-of-the-art in all considered benchmarks.Our results show that previous TTA approaches are neither effective at regularizing adaptation to avoid collapse nor able to outperform a simplistic resetting strategy.

----

## [1735] Swap Agnostic Learning, or Characterizing Omniprediction via Multicalibration

**Authors**: *Parikshit Gopalan, Michael P. Kim, Omer Reingold*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d693203215325902ff9dbdd067a50ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d693203215325902ff9dbdd067a50ac-Abstract-Conference.html)

**Abstract**:

We introduce and study the notion of Swap Agnostic Learning.The problem can be phrased as a game between a *predictor* and an *adversary*:  first, the predictor selects a hypothesis $h$; then, the adversary plays in response, and for each level set of the predictor, selects a loss-minimizing hypothesis $c_v \in \mathcal{C}$; the predictor wins if $h$ competes with the adaptive adversary's loss.Despite the strength of the adversary, our main result demonstrates the feasibility Swap Agnostic Learning for any convex loss.Somewhat surprisingly, the result follows by proving an *equivalence* between Swap Agnostic Learning and swap variants of the recent notions Omniprediction (ITCS'22) and Multicalibration (ICML'18).Beyond this equivalence, we establish further connections to the literature on Outcome Indistinguishability (STOC'20, ITCS'23), revealing a unified notion of OI that captures all existing notions of omniprediction and multicalibration.

----

## [1736] AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation

**Authors**: *Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Yelong Shen, Jian Jiao, Juntao Li, Zhongyu Wei, Jian Guo, Nan Duan, Weizhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html)

**Abstract**:

Diffusion models have gained significant attention in the realm of image generation due to their exceptional performance. Their success has been recently expanded to text generation via generating all tokens within a sequence concurrently. However, natural language exhibits a far more pronounced sequential dependency in comparison to images, and the majority of existing language models are trained with a left-to-right auto-regressive approach.To account for the inherent sequential characteristic of natural language, we introduce  Auto-Regressive Diffusion (AR-Diffusion). AR-Diffusion ensures that the generation of tokens on the right depends on the generated ones on the left, a mechanism achieved through employing a dynamic number of denoising steps that vary based on token position. This results in tokens on the left undergoing fewer denoising steps than those on the right, thereby enabling them to generate earlier and subsequently influence the generation of tokens on the right.In a series of experiments on various text generation tasks, including text summarization, machine translation, and common sense generation, AR-Diffusion clearly demonstrated its superiority over existing diffusion language models and that it can be $100\times\sim600\times$ faster when achieving comparable results. Our code is available at https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion.

----

## [1737] Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations

**Authors**: *Tsai Hor Chan, Kin Wai Lau, Jiajun Shen, Guosheng Yin, Lequan Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7da558c6bd476ba77f5ba712626bba1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7da558c6bd476ba77f5ba712626bba1a-Abstract-Conference.html)

**Abstract**:

Uncertainty estimation aims to evaluate the confidence of a trained deep neural network. However, existing uncertainty estimation approaches rely on low-dimensional distributional assumptions and thus suffer from the high dimensionality of latent features. Existing approaches tend to focus on uncertainty on discrete classification probabilities, which leads to poor generalizability to uncertainty estimation for other tasks. Moreover, most of the literature requires seeing the out-of-distribution (OOD) data in the training for better estimation of uncertainty, which limits the uncertainty estimation performance in practice because the OOD data are typically unseen. To overcome these  limitations, we propose a new framework using data-adaptive high-dimensional hypothesis testing for uncertainty estimation, which leverages the statistical properties of the feature representations. Our method directly operates on latent representations and thus does not require retraining the feature encoder under a modified objective. The test statistic relaxes the feature distribution assumptions to high dimensionality, and it is more discriminative to uncertainties in the latent representations. We demonstrate that encoding features with Bayesian neural networks can enhance testing performance and lead to more accurate uncertainty estimation. We further introduce a family-wise testing procedure to determine the optimal threshold of OOD detection, which minimizes the false discovery rate (FDR). Extensive experiments validate the satisfactory performance of our framework on uncertainty estimation and task-specific prediction over a variety of competitors. The experiments on the OOD detection task also show satisfactory performance of our method when the OOD data are unseen in the training. Codes are available at https://github.com/HKU-MedAI/bnn_uncertainty.

----

## [1738] Algorithmic Regularization in Tensor Optimization: Towards a Lifted Approach in Matrix Sensing

**Authors**: *Ziye Ma, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7db2348b5bfeca620aa7327df815adcc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7db2348b5bfeca620aa7327df815adcc-Abstract-Conference.html)

**Abstract**:

Gradient descent (GD) is crucial for generalization in machine learning models, as it induces implicit regularization, promoting compact representations. In this work, we examine the role of GD in inducing implicit regularization for tensor optimization, particularly within the context of the lifted matrix sensing framework. This framework has been recently proposed to address the non-convex matrix sensing problem by transforming spurious solutions into strict saddles when optimizing over symmetric, rank-1 tensors. We show that, with sufficiently small initialization scale, GD applied to this lifted problem results in approximate rank-1 tensors and critical points with escape directions. Our findings underscore the significance of the tensor parametrization of matrix sensing, in combination with first-order methods, in achieving global optimality in such problems.

----

## [1739] A General Theory of Correct, Incorrect, and Extrinsic Equivariance

**Authors**: *Dian Wang, Xupeng Zhu, Jung Yeon Park, Mingxi Jia, Guanang Su, Robert Platt, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7dc7793c89b93887e126a86f22ef63c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7dc7793c89b93887e126a86f22ef63c6-Abstract-Conference.html)

**Abstract**:

Although equivariant machine learning has proven effective at many tasks, success depends heavily on the assumption that the ground truth function is symmetric over the entire domain matching the symmetry in an equivariant neural network. A missing piece in the equivariant learning literature is the analysis of equivariant networks when symmetry exists only partially in the domain. In this work, we present a general theory for such a situation. We propose pointwise definitions of correct, incorrect, and extrinsic equivariance, which allow us to quantify continuously the degree of each type of equivariance a function displays. We then study the impact of various degrees of incorrect or extrinsic symmetry on model error. We prove error lower bounds for invariant or equivariant networks in classification or regression settings with partially incorrect symmetry. We also analyze the potentially harmful effects of extrinsic equivariance. Experiments validate these results in three different environments.

----

## [1740] Analyzing Vision Transformers for Image Classification in Class Embedding Space

**Authors**: *Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html)

**Abstract**:

Despite the growing use of transformer models in computer vision, a mechanistic understanding of these networks is still needed. This work introduces a method to reverse-engineer Vision Transformers trained to solve image classification tasks. Inspired by previous research in NLP, we demonstrate how the inner representations at any level of the hierarchy can be projected onto the learned class embedding space to uncover how these networks build categorical representations for their predictions. We use our framework to show how image tokens develop class-specific representations that depend on attention mechanisms and contextual information, and give insights on how self-attention and MLP layers differentially contribute to this categorical composition. We additionally demonstrate that this method (1) can be used to determine the parts of an image that would be important for detecting the class of interest, and (2) exhibits significant advantages over traditional linear probing approaches. Taken together, our results position our proposed framework as a powerful tool for mechanistic interpretability and explainability research.

----

## [1741] Toward Re-Identifying Any Animal

**Authors**: *Bingliang Jiao, Lingqiao Liu, Liying Gao, Ruiqi Wu, Guosheng Lin, Peng Wang, Yanning Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7df69dbf39705c7a39b40f2d70e806c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7df69dbf39705c7a39b40f2d70e806c1-Abstract-Conference.html)

**Abstract**:

The current state of re-identification (ReID) models poses limitations to their applicability in the open world, as they are primarily designed and trained for specific categories like person or vehicle. In light of the importance of ReID technology for tracking wildlife populations and migration patterns, we propose a new task called ``Re-identify Any Animal in the Wild'' (ReID-AW). This task aims to develop a ReID model capable of handling any unseen wildlife category it encounters. To address this challenge, we have created a comprehensive dataset called Wildlife-71, which includes ReID data from 71 different wildlife categories. This dataset is the first of its kind to encompass multiple object categories in the realm of ReID. Furthermore, we have developed a universal re-identification model named UniReID specifically for the ReID-AW task. To enhance the model's adaptability to the target category, we employ a dynamic prompting mechanism using category-specific visual prompts. These prompts are generated based on knowledge gained from a set of pre-selected images within the target category. Additionally, we leverage explicit semantic knowledge derived from the large-scale pre-trained language model, GPT-4. This allows UniReID to focus on regions that are particularly useful for distinguishing individuals within the target category. Extensive experiments have demonstrated the remarkable generalization capability of our UniReID model. It showcases promising performance in handling arbitrary wildlife categories, offering significant advancements in the field of ReID for wildlife conservation and research purposes.

----

## [1742] Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications

**Authors**: *Darshil Doshi, Tianyu He, Andrey Gromov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e02f2910ea7911a37c4691f4201c878-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e02f2910ea7911a37c4691f4201c878-Abstract-Conference.html)

**Abstract**:

Deep neural networks are notorious for defying theoretical treatment. However, when the number of parameters in each layer tends to infinity, the network function is a Gaussian process (GP) and quantitatively predictive description is possible. Gaussian approximation allows one to formulate criteria for selecting hyperparameters, such as variances of weights and biases, as well as the learning rate. These criteria rely on the notion of criticality defined for deep neural networks. In this work we describe a new practical way to diagnose criticality. We introduce *partial Jacobians* of a network, defined as derivatives of preactivations in layer $l$ with respect to preactivations in layer $l_0\leq l$. We derive recurrence relations for the norms of partial Jacobians and utilize these relations to analyze criticality of deep fully connected neural networks with LayerNorm and/or residual connections. We derive and implement a simple and cheap numerical test that allows one to select optimal initialization for a broad class of deep neural networks; containing fully connected, convolutional and normalization layers. Using these tools we show quantitatively that proper stacking of the LayerNorm (applied to preactivations) and residual connections leads to an architecture that is critical for any initialization. Finally, we apply our methods to analyze ResNet and MLP-Mixer architectures; demonstrating the everywhere-critical regime.

----

## [1743] Trading-off price for data quality to achieve fair online allocation

**Authors**: *Mathieu Molina, Nicolas Gast, Patrick Loiseau, Vianney Perchet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html)

**Abstract**:

We consider the problem of online allocation subject to a long-term fairness penalty. Contrary to existing works, however, we do not assume that the decision-maker observes the protected attributes---which is often unrealistic in practice. Instead they can purchase data that help estimate them from sources of different quality; and hence reduce the fairness penalty at some cost. We model this problem as a multi-armed bandit problem where each arm corresponds to the choice of a data source, coupled with the fair online allocation problem. We propose an algorithm that jointly solves both problems and show that it has a regret bounded by $\mathcal{O}(\sqrt{T})$. A key difficulty is that the rewards received by selecting a source are correlated by the fairness penalty, which leads to a need for randomization (despite a stochastic setting). Our algorithm takes into account contextual information available before the source selection, and can adapt to many different fairness notions.

----

## [1744] Asymptotics of Bayesian Uncertainty Estimation in Random Features Regression

**Authors**: *Youngsoo Baek, Samuel Berchuck, Sayan Mukherjee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html)

**Abstract**:

In this paper we compare and contrast the behavior of the posterior predictive distribution to the risk of the the maximum a posteriori estimator for the random features regression model in the overparameterized regime.  We will focus on the variance of the  posterior predictive distribution (Bayesian model average) and compare its asymptotics to that of the risk of the MAP estimator. In the regime where the model dimensions grow faster than any constant multiple of the number of samples, asymptotic agreement between these two quantities is governed by the phase transition in the signal-to-noise ratio. They also asymptotically agree with each other when the number of samples grow faster than any constant multiple of model dimensions. Numerical simulations illustrate finer distributional properties of the two quantities for finite dimensions. We conjecture they have Gaussian fluctuations and exhibit similar properties as found by previous authors in a Gaussian sequence model, this is of independent theoretical interest.

----

## [1745] Decentralized Matrix Sensing: Statistical Guarantees and Fast Convergence

**Authors**: *Marie Maros, Gesualdo Scutari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e6f445a74cdb71931aac64f1e3f49c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e6f445a74cdb71931aac64f1e3f49c9-Abstract-Conference.html)

**Abstract**:

We explore the matrix sensing problem from near-isotropic linear measurements, distributed across a network of agents modeled as an undirected graph, with no centralized node. We provide the first study of statistical, computational/communication guarantees for a decentralized gradient algorithm that solves the (nonconvex) Burer-Monteiro type decomposition associated to the low-rank matrix estimation. With small random initialization, the algorithm displays an approximate two-phase convergence: (i) a spectral phase that aligns the iterates' column space with the underlying low-rank matrix, mimicking centralized spectral initialization (not directly implementable over networks); and (ii) a local refinement phase that diverts the iterates from certain degenerate saddle points, while ensuring swift convergence to the underlying low-rank matrix. Central to our analysis is a novel "in-network" Restricted Isometry Property which accommodates for the decentralized nature of the optimization, revealing an intriguing interplay between sample complexity and network connectivity, topology, and communication complexity.

----

## [1746] Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies

**Authors**: *Michael Beukman, Devon Jarvis, Richard Klein, Steven James, Benjamin Rosman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e7b768198d24d883d69704eee57efb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e7b768198d24d883d69704eee57efb0-Abstract-Conference.html)

**Abstract**:

While reinforcement learning has achieved remarkable successes in several domains, its real-world application is limited due to many methods failing to generalise to unfamiliar conditions. In this work, we consider the problem of generalising to new transition dynamics, corresponding to cases in which the environment's response to the agent's actions differs. For example, the gravitational force exerted on a robot depends on its mass and changes the robot's mobility. Consequently, in such cases, it is necessary to condition an agent's actions on extrinsic state information and pertinent contextual information reflecting how the environment responds. While the need for context-sensitive policies has been established, the manner in which context is incorporated architecturally has received less attention. Thus, in this work, we present an investigation into how context information should be incorporated into behaviour learning to improve generalisation.  To this end, we introduce a neural network architecture, the Decision Adapter, which generates the weights of an adapter module and conditions the behaviour of an agent on the context information. We show that the Decision Adapter is a useful generalisation of a previously proposed architecture and empirically demonstrate that it results in superior generalisation performance compared to previous approaches in several environments. Beyond this, the Decision Adapter is more robust to irrelevant distractor variables than several alternative methods.

----

## [1747] Goal Driven Discovery of Distributional Differences via Language Descriptions

**Authors**: *Ruiqi Zhong, Peter Zhang, Steve Li, Jinwoo Ahn, Dan Klein, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html)

**Abstract**:

Exploring large corpora can generate useful discoveries but is time-consuming for humans.    We formulate a new task, D5, that automatically discovers differences between two large corpora in a goal-driven way.     The task input is a problem comprising a user-specified research goal (“comparing the side effects of drug A and drug”) and a corpus pair (two large collections of patients' self-reported reactions after taking each drug).     The output is a goal-related description (discovery) of how these corpora differ (patients taking drug A “mention feelings of paranoia” more often).    We build a D5 system, and to quantitatively evaluate its performance, we 1) build a diagnostic benchmark, SynD5, to test whether it can recover known differences between two synthetic corpora, and 2) contribute a meta-dataset, OpenD5, aggregating 675 open-ended problems ranging across business, social sciences, humanities, machine learning, and health.    With both synthetic and real datasets, we confirm that language models can leverage the user-specified goals to propose more relevant candidate discoveries, and they sometimes produce discoveries previously unknown to the authors, including demographic differences in discussion topics, political stances in speech, insights in commercial reviews, and error patterns in NLP models.    Finally, we discuss the limitations of the current D5 system, which discovers correlation rather than causation and has the potential to reinforce societal biases when misused; therefore, practitioners should treat the outputs of our system with caution.

----

## [1748] Convex and Non-convex Optimization Under Generalized Smoothness

**Authors**: *Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, Ali Jadbabaie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e8bb8d17bb1cb24dfe972a2f8ff2500-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e8bb8d17bb1cb24dfe972a2f8ff2500-Abstract-Conference.html)

**Abstract**:

Classical analysis of convex and non-convex optimization methods often requires the Lipschitz continuity of the gradient, which limits the analysis to functions bounded by quadratics. Recent work relaxed this requirement to a non-uniform smoothness condition with the Hessian norm  bounded by an affine function of the gradient norm, and proved convergence in the non-convex setting via gradient clipping, assuming bounded noise. In this paper, we further generalize this non-uniform smoothness condition and develop a simple, yet powerful analysis technique that bounds the gradients along the trajectory, thereby leading to  stronger results for both convex and non-convex optimization problems. In particular, we obtain the classical convergence rates for (stochastic) gradient descent and Nesterov's accelerated gradient method in the convex and/or non-convex setting under this general smoothness condition. The new analysis approach does not require gradient clipping and allows heavy-tailed noise with bounded variance in the stochastic setting.

----

## [1749] DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement

**Authors**: *Wenxin Tai, Yue Lei, Fan Zhou, Goce Trajcevski, Ting Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e966a12c2d6307adb8809aaa9acf057-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e966a12c2d6307adb8809aaa9acf057-Abstract-Conference.html)

**Abstract**:

Speech enhancement (SE) aims to improve the intelligibility and quality of speech in the presence of non-stationary additive noise. Deterministic deep learning models have traditionally been used for SE, but recent studies have shown that generative approaches, such as denoising diffusion probabilistic models (DDPMs), can also be effective. However, incorporating condition information into DDPMs for SE remains a challenge. We propose a model-agnostic method called DOSE that employs two efficient condition-augmentation techniques to address this challenge, based on two key insights: (1) We force the model to prioritize the condition factor when generating samples by training it with dropout operation; (2) We inject the condition information into the sampling process by providing an informative adaptive prior. Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and inference efficiency. Codes are publicly available at https://github.com/ICDM-UESTC/DOSE.

----

## [1750] ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns

**Authors**: *Ren Li, Benoît Guillard, Pascal Fua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e976afe805026f7d378a583af5ea9a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e976afe805026f7d378a583af5ea9a2-Abstract-Conference.html)

**Abstract**:

Many approaches to draping individual garments on human body models are realistic, fast, and yield outputs that are differentiable with respect to the body shape on which they are draped. However, they are either unable to handle multi-layered clothing, which is prevalent in everyday dress, or restricted to bodies in T-pose. In this paper, we introduce a parametric garment representation model that addresses these limitations. As in models used by clothing designers, each garment consists of individual 2D panels. Their 2D shape is defined by a Signed Distance Function and 3D shape by a 2D to 3D mapping. The 2D parameterization enables easy detection of potential collisions and the 3D parameterization handles complex shapes effectively. We show that this combination is faster and yields higher quality reconstructions than purely implicit surface representations, and makes the recovery of layered garments from images possible thanks to its differentiability. Furthermore, it supports rapid editing of garment shapes and texture by modifying individual 2D panels.

----

## [1751] Optimality of Message-Passing Architectures for Sparse Graphs

**Authors**: *Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html)

**Abstract**:

We study the node classification problem on feature-decorated graphs in the sparse setting, i.e., when the expected degree of a node is $O(1)$ in the number of nodes, in the fixed-dimensional asymptotic regime, i.e., the dimension of the feature data is fixed while the number of nodes is large. Such graphs are typically known to be locally tree-like. We introduce a notion of Bayes optimality for node classification tasks, called asymptotic local Bayes optimality, and compute the optimal classifier according to this criterion for a fairly general statistical data model with arbitrary distributions of the node features and edge connectivity. The optimal classifier is implementable using a message-passing graph neural network architecture. We then compute the generalization error of this classifier and compare its performance against existing learning methods theoretically on a well-studied statistical model with naturally identifiable signal-to-noise ratios (SNRs) in the data. We find that the optimal message-passing architecture interpolates between a standard MLP in the regime of low graph signal and a typical convolution in the regime of high graph signal. Furthermore, we prove a corresponding non-asymptotic result.

----

## [1752] Distribution-Free Statistical Dispersion Control for Societal Applications

**Authors**: *Zhun Deng, Thomas P. Zollo, Jake Snell, Toniann Pitassi, Richard S. Zemel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ea46207ec9bda974b140fe11d8dd727-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ea46207ec9bda974b140fe11d8dd727-Abstract-Conference.html)

**Abstract**:

Explicit finite-sample statistical guarantees on model performance are an important ingredient in responsible machine learning.  Previous work has focused mainly on bounding either the expected loss of a predictor or the probability that an individual prediction will incur a loss value in a specified range.  However, for many high-stakes applications it is crucial to understand and control the \textit{dispersion} of a loss distribution, or the extent to which different members of a population experience unequal effects of algorithmic decisions. We initiate the study of distribution-free control of statistical dispersion measures with societal implications and propose a simple yet flexible framework that allows us to handle a much richer class of statistical functionals beyond previous work. Our methods are verified through experiments in toxic comment detection, medical imaging, and film recommendation.

----

## [1753] Switching Temporary Teachers for Semi-Supervised Semantic Segmentation

**Authors**: *Jaemin Na, Jung-Woo Ha, Hyung Jin Chang, Dongyoon Han, Wonjun Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7eeb42802d3750ca59e8a0523068e9e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7eeb42802d3750ca59e8a0523068e9e6-Abstract-Conference.html)

**Abstract**:

The teacher-student framework, prevalent in semi-supervised semantic segmentation, mainly employs the exponential moving average (EMA) to update a single teacher's weights based on the student's. However, EMA updates raise a problem in that the weights of the teacher and student are getting coupled, causing a potential performance bottleneck. Furthermore, this problem may become more severe when training with more complicated labels such as segmentation masks but with few annotated data. This paper introduces Dual Teacher, a simple yet effective approach that employs dual temporary teachers aiming to alleviate the coupling problem for the student. The temporary teachers work in shifts and are progressively improved, so consistently prevent the teacher and student from becoming excessively close. Specifically, the temporary teachers periodically take turns generating pseudo-labels to train a student model and maintain the distinct characteristics of the student model for each epoch. Consequently, Dual Teacher achieves competitive performance on the PASCAL VOC, Cityscapes, and ADE20K benchmarks with remarkably shorter training times than state-of-the-art methods. Moreover, we demonstrate that our approach is model-agnostic and compatible with both CNN- and Transformer-based models. Code is available at https://github.com/naver-ai/dual-teacher.

----

## [1754] Extremal Domain Translation with Neural Optimal Transport

**Authors**: *Milena Gazdieva, Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html)

**Abstract**:

In many unpaired image domain translation problems, e.g., style transfer or super-resolution, it is important to keep the translated image similar to its respective input image. We propose the extremal transport (ET) which is a mathematical formalization of the theoretically best possible unpaired translation between a pair of domains w.r.t. the given similarity function. Inspired by the recent advances in neural optimal transport (OT), we propose a scalable algorithm to approximate ET maps as a limit of partial OT maps. We test our algorithm on toy examples and on the unpaired image-to-image translation task. The code is publicly available at https://github.com/milenagazdieva/ExtremalNeuralOptimalTransport

----

## [1755] Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

**Authors**: *Yijia Cheng, Xin Liu, Jingyu Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f05193e5487287a890df7fbc3554427-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f05193e5487287a890df7fbc3554427-Abstract-Conference.html)

**Abstract**:

Capturing screen contents by smartphone cameras has become a common way for information sharing. However, these images and videos are often degraded by moiré patterns, which are caused by frequency aliasing between the camera filter array and digital display grids. We observe that the moiré patterns in raw domain is simpler than those in sRGB domain, and the moiré patterns in raw color channels have different properties. Therefore, we propose an image and video demoiréing network tailored for raw inputs. We introduce a color-separated feature branch, and it is fused with the traditional feature-mixed branch via channel and spatial modulations. Specifically, the channel modulation utilizes modulated color-separated features to enhance the color-mixed features. The spatial modulation utilizes the feature with large receptive field to modulate the feature with small receptive field. In addition, we build the first well-aligned raw video demoiréing (RawVDemoiré) dataset and propose an efficient temporal alignment method by inserting alternating patterns. Experiments demonstrate that our method achieves state-of-the-art performance for both image and video demoiréing. Our dataset and code will be released after the acceptance of this work.

----

## [1756] On Imitation in Mean-field Games

**Authors**: *Giorgia Ramponi, Pavel Kolev, Olivier Pietquin, Niao He, Mathieu Laurière, Matthieu Geist*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html)

**Abstract**:

We explore the problem of imitation learning (IL) in the context of mean-field games (MFGs), where the goal is to imitate the behavior of a population of agents following a Nash equilibrium policy according to some unknown payoff function. IL in MFGs presents new challenges compared to single-agent IL, particularly when both the reward function and the transition kernel depend on the population distribution. In this paper, departing from the existing literature on IL for MFGs, we introduce a new solution concept called the Nash imitation gap. Then we show that when only the reward depends on the population distribution, IL in MFGs can be reduced to single-agent IL with similar guarantees. However, when the dynamics is population-dependent, we provide a novel upper-bound that suggests IL is harder in this setting. To address this issue, we propose a new adversarial formulation where the reinforcement learning problem is replaced by a mean-field control (MFC) problem, suggesting progress in IL within MFGs may have to build upon MFC.

----

## [1757] CluB: Cluster Meets BEV for LiDAR-Based 3D Object Detection

**Authors**: *Yingjie Wang, Jiajun Deng, Yuenan Hou, Yao Li, Yu Zhang, Jianmin Ji, Wanli Ouyang, Yanyong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f2fc4053a66edfa430bcdf9a6ff3b17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f2fc4053a66edfa430bcdf9a6ff3b17-Abstract-Conference.html)

**Abstract**:

Currently, LiDAR-based 3D detectors are broadly categorized into two groups, namely, BEV-based detectors and cluster-based detectors.BEV-based detectors capture the contextual information from the Bird's Eye View (BEV) and fill their center voxels via feature diffusion with a stack of convolution layers, which, however, weakens the capability of presenting an object with the center point.On the other hand, cluster-based detectors exploit the voting mechanism and aggregate the foreground points into object-centric clusters for further prediction.In this paper, we explore how to effectively combine these two complementary representations into a unified framework.Specifically, we propose a new 3D object detection framework, referred to as CluB, which incorporates an auxiliary cluster-based branch into the BEV-based detector by enriching the object representation at both feature and query levels.Technically, CluB is comprised of two steps.First, we construct a cluster feature diffusion module to establish the association between cluster features and BEV features in a subtle and adaptive fashion. Based on that, an imitation loss is introduced to distill object-centric knowledge from the cluster features to the BEV features.Second, we design a cluster query generation module to leverage the voting centers directly from the cluster branch, thus enriching the diversity of object queries.Meanwhile, a direction loss is employed to encourage a more accurate voting center for each cluster.Extensive experiments are conducted on Waymo and nuScenes datasets, and our CluB achieves state-of-the-art performance on both benchmarks.

----

## [1758] Probabilistic Exponential Integrators

**Authors**: *Nathanael Bosch, Philipp Hennig, Filip Tronarp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html)

**Abstract**:

Probabilistic solvers provide a flexible and efficient framework for simulation, uncertainty quantification, and inference in dynamical systems. However, like standard solvers, they suffer performance penalties for certain stiff systems, where small steps are required not for reasons of numerical accuracy but for the sake of stability. This issue is greatly alleviated in semi-linear problems by the probabilistic exponential integrators developed in this paper. By including the fast, linear dynamics in the prior, we arrive at a class of probabilistic integrators with favorable properties. Namely, they are proven to be L-stable, and in a certain case reduce to a classic exponential integrator---with the added benefit of providing a probabilistic account of the numerical error. The method is also generalized to arbitrary non-linear systems by imposing piece-wise semi-linearity on the prior via Jacobians of the vector field at the previous estimates, resulting in probabilistic exponential Rosenbrock methods. We evaluate the proposed methods on multiple stiff differential equations and demonstrate their improved stability and efficiency over established probabilistic solvers. The present contribution thus expands the range of problems that can be effectively tackled within probabilistic numerics.

----

## [1759] Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers

**Authors**: *Yiwei Lu, Yaoliang Yu, Xinlin Li, Vahid Partovi Nia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html)

**Abstract**:

In neural network binarization, BinaryConnect (BC) and its variants are considered the standard. These methods apply the sign function in their forward pass and their respective gradients are backpropagated to update the weights. However, the derivative of the sign function is zero whenever defined, which consequently freezes training. Therefore, implementations of BC (e.g., BNN) usually replace the derivative of sign in the backward computation with identity or other approximate gradient alternatives. Although such practice works well empirically, it is largely a heuristic or ``training trick.'' We aim at shedding some light on these training tricks from the optimization perspective. Building from existing theory on ProxConnect (PC, a generalization of BC), we (1) equip PC with different forward-backward quantizers and obtain ProxConnect++ (PC++) that includes existing binarization techniques as special cases; (2) derive a principled way to synthesize forward-backward quantizers with automatic theoretical guarantees; (3) illustrate our theory by proposing an enhanced binarization algorithm BNN++; (4) conduct image classification experiments on CNNs and vision transformers, and empirically verify that BNN++ generally achieves competitive results on binarizing these models.

----

## [1760] QH9: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules

**Authors**: *Haiyang Yu, Meng Liu, Youzhi Luo, Alex Strasser, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f755e271717450020fda40f020922dd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f755e271717450020fda40f020922dd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Supervised machine learning approaches have been increasingly used in accelerating electronic structure prediction as surrogates of first-principle computational methods, such as density functional theory (DFT). While numerous quantum chemistry datasets focus on chemical properties and atomic forces, the ability to achieve accurate and efficient prediction of the Hamiltonian matrix is highly desired, as it is the most important and fundamental physical quantity that determines the quantum states of physical systems and chemical properties. In this work, we generate a new Quantum Hamiltonian dataset, named as QH9, to provide precise Hamiltonian matrices for 2,399 molecular dynamics trajectories and 130,831  stable molecular geometries, based on the QM9 dataset. By designing benchmark tasks with various molecules, we show that current machine learning models have the capacity to predict Hamiltonian matrices for arbitrary molecules. Both the QH9 dataset and the baseline models are provided to the community through an open-source benchmark, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications. Our benchmark is publicly available at \url{https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench}.

----

## [1761] CorresNeRF: Image Correspondence Priors for Neural Radiance Fields

**Authors**: *Yixing Lao, Xiaogang Xu, Zhipeng Cai, Xihui Liu, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f77492bb8070a5c825a87c0c5181da2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f77492bb8070a5c825a87c0c5181da2-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRFs) have achieved impressive results in novel view synthesis and surface reconstruction tasks. However, their performance suffers under challenging scenarios with sparse input views. We present CorresNeRF, a novel method that leverages image correspondence priors computed by off-the-shelf methods to supervise NeRF training. We design adaptive processes for augmentation and filtering to generate dense and high-quality correspondences. The correspondences are then used to regularize NeRF training via the correspondence pixel reprojection and depth loss terms. We evaluate our methods on novel view synthesis and surface reconstruction tasks with density-based and SDF-based NeRF models on different datasets. Our method outperforms previous methods in both photometric and geometric metrics. We show that this simple yet effective technique of using correspondence priors can be applied as a plug-and-play module across different NeRF variants. The project page is at https://yxlao.github.io/corres-nerf/.

----

## [1762] Score-based Data Assimilation

**Authors**: *François Rozet, Gilles Louppe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html)

**Abstract**:

Data assimilation, in its most comprehensive form, addresses the Bayesian inverse problem of identifying plausible state trajectories that explain noisy or incomplete observations of stochastic dynamical systems. Various approaches have been proposed to solve this problem, including particle-based and variational methods. However, most algorithms depend on the transition dynamics for inference, which becomes intractable for long time horizons or for high-dimensional systems with complex dynamics, such as oceans or atmospheres. In this work, we introduce score-based data assimilation for trajectory inference. We learn a score-based generative model of state trajectories based on the key insight that the score of an arbitrarily long trajectory can be decomposed into a series of scores over short segments. After training, inference is carried out using the score model, in a non-autoregressive manner by generating all states simultaneously. Quite distinctively, we decouple the observation model from the training procedure and use it only at inference to guide the generative process, which enables a wide range of zero-shot observation scenarios. We present theoretical and empirical evidence supporting the effectiveness of our method.

----

## [1763] Mr HiSum: A Large-scale Dataset for Video Highlight Detection and Summarization

**Authors**: *Jinhwan Sul, Jihoon Han, Joonseok Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f880e3a325b06e3601af1384a653038-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f880e3a325b06e3601af1384a653038-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Video highlight detection is a task to automatically select the most engaging moments from a long video. This problem is highly challenging since it aims to learn a general way of finding highlights from a variety of videos in the real world.The task has an innate subjectivity because the definition of a highlight differs across individuals. Therefore, to detect consistent and meaningful highlights, prior benchmark datasets have been labeled by multiple (5-20) raters. Due to the high cost of manual labeling, most existing public benchmarks are in extremely small scale, containing only a few tens or hundreds of videos. This insufficient benchmark scale causes multiple issues such as unstable evaluation or high sensitivity in traintest splits. We present Mr. HiSum, a large-scale dataset for video highlight detection and summarization, containing 31,892 videos and reliable labels aggregated over 50,000+ users per video. We empirically prove reliability of the labels as frame importance by cross-dataset transfer and user study.

----

## [1764] Sharp Bounds for Generalized Causal Sensitivity Analysis

**Authors**: *Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f8b8bc8ebac661c442c4dafd5d98c08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f8b8bc8ebac661c442c4dafd5d98c08-Abstract-Conference.html)

**Abstract**:

Causal inference from observational data is crucial for many disciplines such as medicine and economics. However, sharp bounds for causal effects under relaxations of the unconfoundedness assumption (causal sensitivity analysis) are subject to ongoing research. So far, works with sharp bounds are restricted to fairly simple settings (e.g., a single binary treatment). In this paper, we propose a unified framework for causal sensitivity analysis under unobserved confounding in various settings. For this, we propose a flexible generalization of the marginal sensitivity model (MSM) and then derive sharp bounds for a large class of causal effects. This includes (conditional) average treatment effects, effects for mediation analysis and path analysis, and distributional effects. Furthermore, our sensitivity model is applicable to discrete, continuous, and time-varying treatments. It allows us to interpret the partial identification problem under unobserved confounding as a distribution shift in the latent confounders while evaluating the causal effect of interest. In the special case of a single binary treatment, our bounds for (conditional) average treatment effects coincide with recent optimality results for causal sensitivity analysis. Finally, we propose a scalable algorithm to estimate our sharp bounds from observational data.

----

## [1765] Supported Value Regularization for Offline Reinforcement Learning

**Authors**: *Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fa46657df480226112d5be3faf096c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fa46657df480226112d5be3faf096c4-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning suffers from the extrapolation error and value overestimation caused by out-of-distribution (OOD) actions. To mitigate this issue, value regularization approaches aim to penalize the learned value functions to assign lower values to OOD actions. However, existing value regularization methods lack a proper distinction between the regularization effects on in-distribution (ID) and OOD actions, and fail to guarantee optimal convergence results of the policy. To this end, we propose Supported Value Regularization (SVR), which penalizes the Q-values for all OOD actions while maintaining standard Bellman updates for ID ones. Specifically, we utilize the bias of importance sampling to compute the summation of Q-values over the entire OOD region, which serves as the penalty for policy evaluation. This design automatically separates the regularization for ID and OOD actions without manually distinguishing between them. In tabular MDP, we show that the policy evaluation operator of SVR is a contraction, whose fixed point outputs unbiased Q-values for ID actions and underestimated Q-values for OOD actions. Furthermore, the policy iteration with SVR guarantees strict policy improvement until convergence to the optimal support-constrained policy in the dataset. Empirically, we validate the theoretical properties of SVR in a tabular maze environment and demonstrate its state-of-the-art performance on a range of continuous control tasks in the D4RL benchmark.

----

## [1766] Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective

**Authors**: *Yingying Fan, Yu Wu, Bo Du, Yutian Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fbae0a0885d3d688840bd34e4a8a698-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fbae0a0885d3d688840bd34e4a8a698-Abstract-Conference.html)

**Abstract**:

We focus on the weakly-supervised audio-visual video parsing task (AVVP), which aims to identify and locate all the events in audio/visual modalities. Previous works only concentrate on video-level overall label denoising across modalities, but overlook the segment-level label noise, where adjacent video segments (i.e., 1-second video clips) may contain different events. However, recognizing events on the segment is challenging because its label could be any combination of events that occur in the video. To address this issue, we consider tackling AVVP from the language perspective, since language could freely describe how various events appear in each segment beyond fixed labels. Specifically, we design language prompts to describe all cases of event appearance for each video. Then, the similarity between language prompts and segments is calculated, where the event of the most similar prompt is regarded as the segment-level label. In addition, to deal with the mislabeled segments, we propose to perform dynamic re-weighting on the unreliable segments to adjust their labels. Experiments show that our simple yet effective approach outperforms state-of-the-art methods by a large margin.

----

## [1767] Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones

**Authors**: *Asanobu Kitamoto, Jared Hwang, Bastien Vuillod, Lucas Gautier, Yingtao Tian, Tarin Clanuwat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fc36bce5de315751001981baaf4751a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fc36bce5de315751001981baaf4751a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper presents the official release of the Digital Typhoon dataset, the longest typhoon satellite image dataset for 40+ years aimed at benchmarking machine learning models for long-term spatio-temporal data. To build the dataset, we developed a workflow to create an infrared typhoon-centered image for cropping using Lambert azimuthal equal-area projection referring to the best track data. We also address data quality issues such as inter-satellite calibration to create a homogeneous dataset. To take advantage of the dataset, we organized machine learning tasks by the types and targets of inference, with other tasks for meteorological analysis, societal impact, and climate change. The benchmarking results on the analysis, forecasting, and reanalysis for the intensity suggest that the dataset is challenging for recent deep learning models, due to many choices that affect the performance of various models. This dataset reduces the barrier for machine learning researchers to meet large-scale real-world events called tropical cyclones and develop machine learning models that may contribute to advancing scientific knowledge on tropical cyclones as well as solving societal and sustainability issues such as disaster reduction and climate change. The dataset is publicly available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ and https://github.com/kitamoto-lab/digital-typhoon/.

----

## [1768] Maximum Independent Set: Self-Training through Dynamic Programming

**Authors**: *Lorenzo Brusca, Lars C. P. M. Quaedvlieg, Stratis Skoulakis, Grigorios Chrysos, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fe3170d88a8310ca86df2843f54236c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fe3170d88a8310ca86df2843f54236c-Abstract-Conference.html)

**Abstract**:

This work presents a graph neural network (GNN) framework for solving the maximum independent set (MIS) problem, inspired by dynamic programming (DP). Specifically, given a graph, we propose a DP-like recursive algorithm based on GNNs that firstly constructs two smaller sub-graphs, predicts the one with the larger MIS, and then uses it in the next recursive call. To train our algorithm, we require annotated comparisons of different graphs concerning their MIS size. Annotating the comparisons with the output of our algorithm leads to a self-training process that results in more accurate self-annotation of the comparisons and vice versa. We provide numerical evidence showing the superiority of our method vs prior methods in multiple synthetic and real-world datasets.

----

## [1769] Reference-Based POMDPs

**Authors**: *Edward Kim, Yohan Karunanayake, Hanna Kurniawati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ffb2b550ff6a75c536b279348a93fb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ffb2b550ff6a75c536b279348a93fb0-Abstract-Conference.html)

**Abstract**:

Making good decisions in partially observable and non-deterministic scenarios is a crucial capability for robots. A Partially Observable Markov Decision Process (POMDP) is a general framework for the above problem. Despite advances in POMDP solving, problems with long planning horizons and evolving environments remain difficult to solve even by the best approximate solvers today. To alleviate this difficulty, we propose a slightly modified POMDP problem, called a Reference-Based POMDP, where the objective is to balance between maximizing the expected total reward and being close to a given reference (stochastic) policy. The optimal policy of a Reference-Based POMDP can be computed via iterative expectations using the given reference policy, thereby avoiding exhaustive enumeration of actions at each belief node of the search tree. We demonstrate theoretically that the standard POMDP under stochastic policies is related to the Reference-Based POMDP. To demonstrate the feasibility of exploiting the formulation, we present a basic algorithm RefSolver. Results from experiments on long-horizon navigation problems indicate that this basic algorithm substantially outperforms POMCP.

----

## [1770] Siamese Masked Autoencoders

**Authors**: *Agrim Gupta, Jiajun Wu, Jia Deng, Fei-Fei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html)

**Abstract**:

Establishing correspondence between images or scenes is a significant challenge in computer vision, especially given occlusions, viewpoint changes, and varying object appearances. In this paper, we present Siamese Masked Autoencoders (SiamMAE), a simple extension of Masked Autoencoders (MAE) for learning visual correspondence from videos. SiamMAE operates on pairs of randomly sampled video frames and asymmetrically masks them. These frames are processed independently by an encoder network, and a decoder composed of a sequence of cross-attention layers is tasked with predicting the missing patches in the future frame. By masking a large fraction (95%) of patches in the future frame while leaving the past frame unchanged, SiamMAE encourages the network to focus on object motion and learn object-centric representations. Despite its conceptual simplicity, features learned via SiamMAE outperform state-of-the-art self-supervised methods on video object segmentation, pose keypoint propagation, and semantic part propagation tasks. SiamMAE achieves competitive results without relying on data augmentation, handcrafted tracking-based pretext tasks, or other techniques to prevent representational collapse.

----

## [1771] Score-based Generative Models with Lévy Processes

**Authors**: *Eun-Bi Yoon, Keehun Park, Sungwoong Kim, Sungbin Lim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html)

**Abstract**:

Investigating the optimal stochastic process beyond Gaussian for noise injection in a score-based generative model remains an open question. Brownian motion is a light-tailed process with continuous paths, which leads to a slow convergence rate for the Number of Function Evaluation (NFE). Recent studies have shown that diffusion models suffer from mode-collapse issues on imbalanced data.In order to overcome the limitations of Brownian motion, we introduce a novel score-based generative model referred to as Lévy-Itō Model (LIM). This model utilizes isotropic $\alpha$-stable Lévy processes. We first derive an exact reverse-time stochastic differential equation driven by the Lévy process and develop the corresponding fractional denoising score matching. The proposed generative model takes advantage of the heavy-tailed properties of the Lévy process. Our experimental results show LIM allows for faster and more diverse sampling while maintaining high fidelity compared to existing diffusion models across various image datasets such as CIFAR10, CelebA, and imbalanced dataset CIFAR10LT. Comparing our results to those of DDPM with 3.21 Fréchet Inception Distance (FID) and 0.6437 Recall on the CelebA dataset, we achieve 1.58 FID and 0.7006 Recall using the same architecture. LIM shows the best performance in NFE 500 with $2\times$ faster total wall-clock time than the baseline.

----

## [1772] 3D Indoor Instance Segmentation in an Open-World

**Authors**: *Mohamed El Amine Boudjoghra, Salwa K. Al Khatib, Jean Lahoud, Hisham Cholakkal, Rao Muhammad Anwer, Salman H. Khan, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html)

**Abstract**:

Existing 3D instance segmentation methods typically assume that all semantic classes to be segmented would be available during training and only seen categories are segmented at inference. We argue that such a closed-world assumption is restrictive and explore for the first time 3D indoor instance segmentation in an open-world setting, where the model is allowed to distinguish a set of known classes as well as identify an unknown object as unknown and then later incrementally learning the semantic category of the unknown when the corresponding category labels are available. To this end, we introduce an open-world 3D indoor instance segmentation method, where an auto-labeling scheme is employed to produce pseudo-labels during training and induce separation to separate known and unknown category labels. We further improve the pseudo-labels quality at inference by adjusting the unknown class probability based on the objectness score distribution. We also introduce carefully curated open-world splits leveraging realistic scenarios based on inherent object distribution, region-based indoor scene exploration and randomness aspect of open-world classes. Extensive experiments reveal the efficacy of the proposed contributions leading to promising open-world 3D instance segmentation performance. Code and splits are available at: https://github.com/aminebdj/3D-OWIS.

----

## [1773] AbDiffuser: full-atom generation of in-vitro functioning antibodies

**Authors**: *Karolis Martinkus, Jan Ludwiczak, Wei-Ching Liang, Julien Lafrance-Vanasse, Isidro Hötzel, Arvind Rajpal, Yan Wu, Kyunghyun Cho, Richard Bonneau, Vladimir Gligorijevic, Andreas Loukas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html)

**Abstract**:

We introduce AbDiffuser, an equivariant and physics-informed diffusion model for the joint generation of antibody 3D structures and sequences. AbDiffuser is built on top of a new representation of protein structure, relies on a novel architecture for aligned proteins, and utilizes strong diffusion priors to improve the denoising process. Our approach improves protein diffusion by taking advantage of domain knowledge and physics-based constraints; handles sequence-length changes; and reduces memory complexity by an order of magnitude, enabling backbone and side chain generation. We validate AbDiffuser in silico and in vitro. Numerical experiments showcase the ability of AbDiffuser to generate antibodies that closely track the sequence and structural properties of a reference set. Laboratory experiments confirm that all 16 HER2 antibodies discovered were expressed at high levels and that 57.1% of the selected designs were tight binders.

----

## [1774] Structure Learning with Adaptive Random Neighborhood Informed MCMC

**Authors**: *Xitong Liang, Alberto Caron, Samuel Livingstone, Jim E. Griffin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8027ace571384361920665f1d1b69758-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8027ace571384361920665f1d1b69758-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce a novel MCMC sampler, PARNI-DAG, for a fully-Bayesian approach to the problem of structure learning under observational data. Under the assumption of causal sufficiency, the algorithm allows for approximate sampling directly from the posterior distribution on Directed Acyclic Graphs (DAGs). PARNI-DAG performs efficient sampling of DAGs via locally informed, adaptive random neighborhood proposal that results in better mixing properties. In addition, to ensure better scalability with the number of nodes, we couple PARNI-DAG with a pre-tuning procedure of the sampler's parameters that exploits a skeleton graph derived through some constraint-based or scoring-based algorithms. Thanks to these novel features, PARNI-DAG quickly converges to high-probability regions and is less likely to get stuck in local modes in the presence of high correlation between nodes in high-dimensional settings. After introducing the technical novelties in PARNI-DAG, we empirically demonstrate its mixing efficiency and accuracy in learning DAG structures on a variety of experiments.

----

## [1775] Reining Generalization in Offline Reinforcement Learning via Representation Distinction

**Authors**: *Yi Ma, Hongyao Tang, Dong Li, Zhaopeng Meng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html)

**Abstract**:

Offline Reinforcement Learning (RL) aims to address the challenge of distribution shift between the dataset and the learned policy, where the value of out-of-distribution (OOD) data may be erroneously estimated due to overgeneralization. It has been observed that a considerable portion of the benefits derived from the conservative terms designed by existing offline RL approaches originates from their impact on the learned representation. This observation prompts us to scrutinize the learning dynamics of offline RL, formalize the process of generalization, and delve into the prevalent overgeneralization issue in offline RL. We then investigate the potential to rein the generalization from the representation perspective to enhance offline RL. Finally, we present  Representation Distinction (RD), an innovative plug-in method for improving offline RL algorithm performance by explicitly differentiating between the representations of in-sample and OOD state-action pairs generated by the learning policy. Considering scenarios in which the learning policy mirrors the behavioral policy and similar samples may be erroneously distinguished, we suggest a dynamic adjustment mechanism for RD based on an OOD data generator to prevent data representation collapse and further enhance policy performance. We demonstrate the efficacy of our approach by applying RD to specially-designed backbone algorithms and widely-used offline RL algorithms. The proposed RD method significantly improves their performance across various continuous control tasks on D4RL datasets, surpassing several state-of-the-art offline RL algorithms.

----

## [1776] BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning

**Authors**: *Xuan Chen, Wenbo Guo, Guanhong Tao, Xiangyu Zhang, Dawn Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/802e90325f4c8546e13e5763b2ecab88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/802e90325f4c8546e13e5763b2ecab88-Abstract-Conference.html)

**Abstract**:

Backdoor attacks pose a severe threat to the supply chain management of deep reinforcement learning (DRL) policies. Despite initial defenses proposed in recent studies, these methods have very limited generalizability and scalability. To address this issue, we propose BIRD, a technique to detect and remove backdoors from a pretrained DRL policy in a clean environment without requiring any knowledge about the attack specifications and accessing its training process. By analyzing the unique properties and behaviors of backdoor attacks, we formulate trigger restoration as an optimization problem and design a novel metric to detect backdoored policies. We also design a finetuning method to remove the backdoor, while maintaining the agent's performance in the clean environment. We evaluate BIRD against three backdoor attacks in ten different single-agent or multi-agent environments. Our results verify the effectiveness, efficiency, and generalizability of BIRD, as well as its robustness to different attack variations and adaptions.

----

## [1777] Cluster-aware Semi-supervised Learning: Relational Knowledge Distillation Provably Learns Clustering

**Authors**: *Yijun Dong, Kevin Miller, Qi Lei, Rachel Ward*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8037f47a6254eb60899a644bd90b4f6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8037f47a6254eb60899a644bd90b4f6a-Abstract-Conference.html)

**Abstract**:

Despite the empirical success and practical significance of (relational) knowledge distillation that matches (the relations of) features between teacher and student models, the corresponding theoretical interpretations remain limited for various knowledge distillation paradigms. In this work, we take an initial step toward a theoretical understanding of relational knowledge distillation (RKD), with a focus on semi-supervised classification problems. We start by casting RKD as spectral clustering on a population-induced graph unveiled by a teacher model. Via a notion of clustering error that quantifies the discrepancy between the predicted and ground truth clusterings, we illustrate that RKD over the population provably leads to low clustering error. Moreover, we provide a sample complexity bound for RKD with limited unlabeled samples. For semi-supervised learning, we further demonstrate the label efficiency of RKD through a general framework of cluster-aware semi-supervised learning that assumes low clustering errors. Finally, by unifying data augmentation consistency regularization into this cluster-aware framework, we show that despite the common effect of learning accurate clusterings, RKD facilitates a "global" perspective through spectral clustering, whereas consistency regularization focuses on a "local" perspective via expansion.

----

## [1778] Bicriteria Multidimensional Mechanism Design with Side Information

**Authors**: *Siddharth Prasad, Maria-Florina Balcan, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8039ca1e9860daab3a79e45d010d5398-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8039ca1e9860daab3a79e45d010d5398-Abstract-Conference.html)

**Abstract**:

We develop a versatile new methodology for multidimensional mechanism design that incorporates side information about agent types to generate high social welfare and high revenue simultaneously. Prominent sources of side information in practice include predictions from a machine-learning model trained on historical agent data, advice from domain experts, and even the mechanism designer's own gut instinct. In this paper we adopt a prior-free perspective that makes no assumptions on the correctness, accuracy, or source of the side information. First, we design a meta-mechanism that integrates input side information with an improvement of the classical VCG mechanism. The welfare, revenue, and incentive properties of our meta-mechanism are characterized by novel constructions we introduce based on the notion of a weakest competitor, which is an agent that has the smallest impact on welfare. We show that our meta-mechanism, when carefully instantiated, simultaneously achieves strong welfare and revenue guarantees parameterized by errors in the side information. When the side information is highly informative and accurate, our mechanism achieves welfare and revenue competitive with the total social surplus, and its performance decays continuously and gradually as the quality of the side information decreases. Finally, we apply our meta-mechanism to a setting where each agent's type is determined by a constant number of parameters. Specifically, agent types lie on constant-dimensional subspaces (of the potentially high-dimensional ambient type space) that are known to the mechanism designer. We use our meta-mechanism to obtain the first known welfare and revenue guarantees in this setting.

----

## [1779] Are These the Same Apple? Comparing Images Based on Object Intrinsics

**Authors**: *Klemen Kotar, Stephen Tian, Hong-Xing Yu, Dan Yamins, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/803c6ab3d62346e004ef70211d2d15b8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/803c6ab3d62346e004ef70211d2d15b8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The human visual system can effortlessly recognize an object under different extrinsic factors such as lighting, object poses, and background, yet current computer vision systems often struggle with these variations. An important step to understanding and improving artificial vision systems is to measure image similarity purely based on intrinsic object properties that define object identity. This problem has been studied in the computer vision literature as re-identification, though mostly restricted to specific object categories such as people and cars. We propose to extend it to general object categories, exploring an image similarity metric based on object intrinsics. To benchmark such measurements, we collect the Common paired objects Under differenT Extrinsics (CUTE) dataset of 18, 000 images of 180 objects under different extrinsic factors such as lighting, poses, and imaging conditions. While existing methods such as LPIPS and CLIP scores do not measure object intrinsics well, we find that combining deep features learned from contrastive self-supervised learning with foreground filtering is a simple yet effective approach to approximating the similarity. We conduct an extensive survey of pre-trained features and foreground extraction methods to arrive at a strong baseline that best measures intrinsic object-centric image similarity among current methods. Finally, we demonstrate that our approach can aid in downstream applications such as acting as an analog for human subjects and improving generalizable re-identification. Please see our project website at https://s-tian.github.io/projects/cute/ for visualizations of the data and demos of our metric.

----

## [1780] The ToMCAT Dataset

**Authors**: *Adarsh Pyarelal, Eric Duong, Caleb Shibu, Paulo Soares, Savannah Boyd, Payal Khosla, Valeria A. Pfeifer, Diheng Zhang, Eric Andrews, Rick Champlin, Vincent Raymond, Meghavarshini Krishnaswamy, Clayton T. Morrison, Emily Butler, Kobus Barnard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/803d8d4b4a549d0d062fc704f8659ce3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/803d8d4b4a549d0d062fc704f8659ce3-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present a rich, multimodal dataset consisting of data from 40 teams of three humans conducting simulated urban search-and-rescue (SAR) missions in a Minecraft-based testbed, collected for the Theory of Mind-based Cognitive Architecture for Teams (ToMCAT) project. Modalities include two kinds of brain scan data---functional near-infrared spectroscopy (fNIRS) and electroencephalography (EEG), as well as skin conductance, heart rate, eye tracking, face images, spoken dialog audio data with automatic speech recognition (ASR) transcriptions, game screenshots, gameplay data, game performance data, demographic data, and self-report questionnaires. Each team undergoes up to six consecutive phases: three behavioral tasks, one mission training session, and two collaborative SAR missions. As time-synchronized multimodal data collected under a variety of circumstances, this dataset will support studying a large variety of research questions on topics including teamwork, coordination, plan recognition, affective computing, physiological linkage, entrainment, and dialog understanding.  We provide an initial public release of the de-identified data, along with analyses illustrating the utility of this dataset to both computer scientists and social scientists.

----

## [1781] Exploring Diverse In-Context Configurations for Image Captioning

**Authors**: *Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, Xin Geng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/804b5e300c9ed4e3ea3b073f186f4adc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/804b5e300c9ed4e3ea3b073f186f4adc-Abstract-Conference.html)

**Abstract**:

After discovering that Language Models (LMs) can be good in-context few-shot learners, numerous strategies have been proposed to optimize in-context sequence configurations. Recently, researchers in Vision-Language (VL) domains also develop their few-shot learners, while they only use the simplest way, \ie, randomly sampling, to configure in-context image-text pairs. In order to explore the effects of varying configurations on VL in-context learning, we devised four strategies for image selection and four for caption assignment to configure in-context image-text pairs for image captioning. Here Image Captioning is used as the case study since it can be seen as the visually-conditioned LM. Our comprehensive experiments yield two counter-intuitive but valuable insights, highlighting the distinct characteristics of VL in-context learning due to multi-modal synergy, as compared to the NLP case. Furthermore, in our exploration of optimal combination strategies, we observed an average performance enhancement of 20.9 in CIDEr scores compared to the baseline. The code is given in https://github.com/yongliang-wu/ExploreCfg.

----

## [1782] DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis

**Authors**: *Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann, Christian Theobalt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/805c06617d2b643278936daadfde4280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/805c06617d2b643278936daadfde4280-Abstract-Conference.html)

**Abstract**:

Generating controllable and photorealistic digital human avatars is a long-standing and important problem in Vision and Graphics. Recent methods have shown great progress in terms of either photorealism or inference speed while the combination of the two desired properties still remains unsolved. To this end, we propose a novel method, called DELIFFAS, which parameterizes the appearance of the human as a surface light field that is attached to a controllable and deforming human mesh model. At the core, we represent the light field around the human with a deformable two-surface parameterization, which enables fast and accurate inference of the human appearance. This allows perceptual supervision on the full image compared to previous approaches that could only supervise individual pixels or small patches due to their slow runtime. Our carefully designed human representation and supervision strategy leads to state-of-the-art synthesis results and inference time. The video results and code are available at https://vcai.mpi-inf.mpg.de/projects/DELIFFAS.

----

## [1783] Zero-Shot Anomaly Detection via Batch Normalization

**Authors**: *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html)

**Abstract**:

Anomaly detection (AD) plays a crucial role in many safety-critical application domains. The challenge of adapting an anomaly detector to drift in the normal data distribution, especially when no training data is available for the "new normal," has led to the development of zero-shot AD techniques. In this paper, we propose a simple yet effective method called Adaptive Centered Representations (ACR) for zero-shot batch-level AD. Our approach trains off-the-shelf deep anomaly detectors (such as deep SVDD) to adapt to a set of inter-related training data distributions in combination with batch normalization, enabling automatic zero-shot generalization for unseen AD tasks. This simple recipe, batch normalization plus meta-training, is a highly effective and versatile tool. Our results demonstrate the first zero-shot AD results for tabular data and outperform existing methods in zero-shot anomaly detection and segmentation on image data from specialized domains.

----

## [1784] What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement

**Authors**: *Yotam Alexander, Nimrod De La Vega, Noam Razin, Nadav Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/808a79d149c9dd8338d789881c9dab4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/808a79d149c9dd8338d789881c9dab4c-Abstract-Conference.html)

**Abstract**:

The question of what makes a data distribution suitable for deep learning is a fundamental open problem. Focusing on locally connected neural networks (a prevalent family of architectures that includes convolutional and recurrent neural networks as well as local self-attention models), we address this problem by adopting theoretical tools from quantum physics. Our main theoretical result states that a certain locally connected neural network is capable of accurate prediction over a data distribution if and only if the data distribution admits low quantum entanglement under certain canonical partitions of features. As a practical application of this result, we derive a preprocessing method for enhancing the suitability of a data distribution to locally connected neural networks. Experiments with widespread models over various datasets demonstrate our findings. We hope that our use of quantum entanglement will encourage further adoption of tools from physics for formally reasoning about the relation between deep learning and real-world data.

----

## [1785] Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

**Authors**: *Qiong Wu, Wei Yu, Yiyi Zhou, Shubin Huang, Xiaoshuai Sun, Rongrong Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80e354fdac2c7fbf439a51f4853edbac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80e354fdac2c7fbf439a51f4853edbac-Abstract-Conference.html)

**Abstract**:

With ever increasing parameters and computation, vision-language pre-trained (VLP) models exhibit prohibitive expenditure in downstream task adaption. Recent endeavors mainly focus on parameter efficient transfer learning (PETL) for VLP models by only updating a small number of parameters. However, excessive computational overhead still plagues the application of VLPs. In this paper, we aim at parameter and computation efficient transfer learning (PCETL) for VLP models. In particular, PCETL not only needs to limit the number of trainable parameters in VLP models, but also to reduce the computational redundancy during inference, thus enabling a more efficient transfer. To approach this target, we propose a novel dynamic architecture skipping (DAS) approach towards effective PCETL. Instead of directly optimizing the intrinsic architectures of VLP models, DAS first observes the significances of their modules to downstream tasks via a reinforcement learning (RL) based process, and then skips the redundant ones with lightweight networks, i.e. adapters, according to the obtained rewards. In this case, the VLP model can well maintain the scale of trainable parameters while speeding up its inference on downstream tasks. To validate DAS, we apply it to two representative VLP models, namely ViLT and METER, and conduct extensive experiments on a bunch of VL tasks. The experimental results not only show the great advantages of DAS in reducing computational complexity, e.g. -11.97%  FLOPs of METER on VQA2.0, but also confirm its competitiveness against existing PETL methods in terms of parameter scale and performance. Our source code is given in our appendix.

----

## [1786] A Dynamical System View of Langevin-Based Non-Convex Sampling

**Authors**: *Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80f253dcb51cd2af7ce54e9379fb3521-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80f253dcb51cd2af7ce54e9379fb3521-Abstract-Conference.html)

**Abstract**:

Non-convex sampling is a key challenge in machine learning, central to non-convex optimization in deep learning as well as to approximate probabilistic inference. Despite its significance, theoretically there remain some important challenges: Existing guarantees suffer from the drawback of lacking guarantees for the last-iterates, and little is known beyond the elementary schemes of stochastic gradient Langevin dynamics. To address these issues, we develop a novel framework that lifts the above issues by harnessing several tools from the theory of dynamical systems. Our key result is that, for a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood. Coupled with standard assumptions of MCMC sampling, our theory immediately yields the last-iterate Wasserstein convergence of many advanced sampling schemes such as mirror Langevin, proximal, randomized mid-point, and Runge-Kutta methods.

----

## [1787] OKRidge: Scalable Optimal k-Sparse Ridge Regression

**Authors**: *Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80f48ffa8022773973a4a5cec7cce19c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80f48ffa8022773973a4a5cec7cce19c-Abstract-Conference.html)

**Abstract**:

We consider an important problem in scientific discovery, namely identifying sparse governing equations for nonlinear dynamical systems. This involves solving sparse ridge regression problems to provable optimality in order to determine which terms drive the underlying dynamics. We propose a fast algorithm, OKRidge, for sparse ridge regression, using a novel lower bound calculation involving, first, a saddle point formulation, and from there, either solving (i) a linear system or (ii) using an ADMM-based approach, where the proximal operators can be efficiently evaluated by solving another linear system and an isotonic regression problem. We also propose a method to warm-start our solver, which leverages a beam search. Experimentally, our methods attain provable optimality with run times that are orders of magnitude faster than those of the existing MIP formulations solved by the commercial solver Gurobi.

----

## [1788] Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise

**Authors**: *Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80fe51a7d8d0c73ff7439c2a2554ed53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80fe51a7d8d0c73ff7439c2a2554ed53-Abstract-Conference.html)

**Abstract**:

Standard diffusion models involve an image transform  -- adding Gaussian noise -- and an image restoration operator that inverts this degradation.  We observe that the generative behavior of diffusion models is not strongly dependent on the choice of image degradation, and in fact, an entire family of generative models can be constructed by varying this choice. Even when using completely deterministic degradations (e.g., blur, masking, and more), the training and test-time update rules that underlie diffusion models can be easily generalized to create generative models. The success of these fully deterministic models calls into question the community's understanding of diffusion models, which relies on noise in either gradient Langevin dynamics or variational inference and paves the way for generalized diffusion models that invert arbitrary processes.

----

## [1789] Towards the Difficulty for a Deep Neural Network to Learn Concepts of Different Complexities

**Authors**: *Dongrui Liu, Huiqi Deng, Xu Cheng, Qihan Ren, Kangrui Wang, Quanshi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8143b8c73073a9a23b9c18e400066471-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8143b8c73073a9a23b9c18e400066471-Abstract-Conference.html)

**Abstract**:

This paper theoretically explains the intuition that simple concepts are more likely to be learned by deep neural networks (DNNs) than complex concepts. In fact, recent studies have observed [24, 15] and proved [26] the emergence of interactive concepts in a DNN, i.e., it is proven that a DNN usually only encodes a small number of interactive concepts, and can be considered to use their interaction effects to compute inference scores. Each interactive concept is encoded by the DNN to represent the collaboration between a set of input variables. Therefore, in this study, we aim to theoretically explain that interactive concepts involving more input variables (i.e., more complex concepts) are more difficult to learn. Our finding clarifies the exact conceptual complexity that boosts the learning difficulty.

----

## [1790] Limits, approximation and size transferability for GNNs on sparse graphs via graphops

**Authors**: *Thien Le, Stefanie Jegelka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8154c89c8d3612d39fd1ed6a20f4bab1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8154c89c8d3612d39fd1ed6a20f4bab1-Abstract-Conference.html)

**Abstract**:

Can graph neural networks generalize to graphs that are different from the graphs they were trained on, e.g., in size? In this work, we study this question from a theoretical perspective. While recent work established such transferability and approximation results via graph limits, e.g., via graphons, these only apply nontrivially to dense graphs. To include frequently encountered sparse graphs such as bounded-degree or power law graphs, we take a perspective of taking limits of operators derived from graphs, such as the aggregation operation that makes up GNNs. This leads to the recently introduced limit notion of graphops (Backhausz and Szegedy, 2022). We demonstrate how the operator perspective allows us to develop quantitative bounds on the distance between a finite GNN and its limit on an infinite graph, as well as the distance between the GNN on graphs of different sizes that share structural properties, under a regularity assumption verified for various graph sequences. Our results hold for dense and sparse graphs, and various notions of graph limits.

----

## [1791] The Adversarial Consistency of Surrogate Risks for Binary Classification

**Authors**: *Natalie Frank, Jonathan Niles-Weed*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81858558b55a8c63763cfe088090242a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81858558b55a8c63763cfe088090242a-Abstract-Conference.html)

**Abstract**:

We study the consistency of surrogate risks for robust binary classification.It is common to learn robust classifiers by adversarial training, which seeks to minimize the expected $0$-$1$ loss when each example can be maliciously corrupted within a small ball.We give a simple and complete characterization of the set of surrogate loss functions that are \emph{consistent}, i.e., that can replace the $0$-$1$ loss without affecting the minimizing sequences of the original adversarial risk, for any data distribution.We also prove a quantitative version of adversarial consistency for the $\rho$-margin loss.Our results reveal that the class of adversarially consistent surrogates is substantially smaller than in the standard setting, where many common surrogates are known to be consistent.

----

## [1792] The Cambridge Law Corpus: A Corpus for Legal AI Research

**Authors**: *Andreas Östling, Holli Sargeant, Huiyuan Xie, Ludwig Bull, Alexander Terenin, Leif Jonsson, Måns Magnusson, Felix Steffek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce the Cambridge Law Corpus (CLC), a corpus for legal AI research. It consists of over 250 000 court cases from the UK. Most cases are from the 21st century, but the corpus includes cases as old as the 16th century. This paper presents the first release of the corpus, containing the raw text and meta-data. Together with the corpus, we provide annotations on case outcomes for 638 cases, done by legal experts. Using our annotated data, we have trained and evaluated case outcome extraction with GPT-3, GPT-4 and RoBERTa models to provide benchmarks. We include an extensive legal and ethical discussion to address the potentially sensitive nature of this material. As a consequence, the corpus will only be released for research purposes under certain restrictions.

----

## [1793] Large Language Models of Code Fail at Completing Code with Potential Bugs

**Authors**: *Tuan Dinh, Jinman Zhao, Samson Tan, Renato Negrinho, Leonard Lausen, Sheng Zha, George Karypis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html)

**Abstract**:

Large language models of code (Code-LLMs) have recently brought tremendous advances to code completion, a fundamental feature of programming assistance and code intelligence. However, most existing works ignore the possible presence of bugs in the code context for generation, which are inevitable in software development. Therefore, we introduce and study the buggy-code completion problem, inspired by the realistic scenario of real-time code suggestion where the code context contains potential bugs â€“ anti-patterns that can become bugs in the completed program. To systematically study the task, we introduce two datasets: one with synthetic bugs derived from semantics-altering operator changes (buggy-HumanEval) and one with realistic bugs derived from user submissions to coding problems (buggy-FixEval). We find that the presence of potential bugs significantly degrades the generation performance of the high-performing Code-LLMs. For instance, the passing rates of CODEGEN-2B-MONO on test cases of buggy-HumanEval drop more than 50% given a single potential bug in the context. Finally, we investigate several post-hoc methods for mitigating the adverse effect of potential bugs and find that there remains a large gap in post-mitigation performance.

----

## [1794] Doubly-Robust Self-Training

**Authors**: *Banghua Zhu, Mingyu Ding, Philip L. Jacobson, Ming Wu, Wei Zhan, Michael I. Jordan, Jiantao Jiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819f426947c27eb5067bb6fdbdde93dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/819f426947c27eb5067bb6fdbdde93dd-Abstract-Conference.html)

**Abstract**:

Self-training is a well-established technique in semi-supervised learning, which leverages unlabeled data by generating pseudo-labels and incorporating them with a limited labeled dataset for training. The effectiveness of self-training heavily relies on the accuracy of these pseudo-labels. In this paper, we introduce doubly-robust self-training, an innovative semi-supervised algorithm that provably balances between two extremes. When pseudo-labels are entirely incorrect, our method reduces to a training process solely using labeled data. Conversely, when pseudo-labels are completely accurate, our method transforms into a training process utilizing all pseudo-labeled data and labeled data, thus increasing the effective sample size. Through empirical evaluations on both the ImageNet dataset for image classification and the nuScenes autonomous driving dataset for 3D object detection, we demonstrate the superiority of the doubly-robust loss over the self-training baseline.

----

## [1795] FairLISA: Fair User Modeling with Limited Sensitive Attributes Information

**Authors**: *Zheng Zhang, Qi Liu, Hao Jiang, Fei Wang, Yan Zhuang, Le Wu, Weibo Gao, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81a12aed87eb9c75dfdf91ed99d5519d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81a12aed87eb9c75dfdf91ed99d5519d-Abstract-Conference.html)

**Abstract**:

User modeling techniques profile users' latent characteristics (e.g., preference) from their observed behaviors, and play a crucial role in decision-making. Unfortunately, traditional user models may unconsciously capture biases related to sensitive attributes (e.g., gender) from behavior data, even when this sensitive information is not explicitly provided. This can lead to unfair issues and discrimination against certain groups based on these sensitive attributes.  Recent studies have been proposed to improve fairness by explicitly decorrelating user modeling results and sensitive attributes. However, most existing approaches assume that fully sensitive attribute labels are available in the training set, which is unrealistic due to collection limitations like privacy concerns, and hence bear the limitation of performance. In this paper, we focus on a practical situation with limited sensitive data and propose a novel FairLISA framework, which can efficiently utilize data with known and unknown sensitive attributes to facilitate fair model training. We first propose a novel theoretical perspective to build the relationship between data with both known and unknown sensitive attributes with the fairness objective.  Then, based on this, we provide a general adversarial framework to effectively leverage the whole user data for fair user modeling. We conduct experiments on representative user modeling tasks including recommender system and cognitive diagnosis. The results demonstrate that our FairLISA can effectively improve fairness while retaining high accuracy in scenarios with different ratios of missing sensitive attributes.

----

## [1796] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model

**Authors**: *Kenneth Li, Oam Patel, Fernanda B. Viégas, Hanspeter Pfister, Martin Wattenberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)

**Abstract**:

We introduce Inference-Time Intervention (ITI), a technique designed to enhance the "truthfulness" of large language models (LLMs). ITI operates by shifting model activations during inference, following a learned set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from $32.5\%$ to $65.1\%$. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.

----

## [1797] Composable Coresets for Determinant Maximization: Greedy is Almost Optimal

**Authors**: *Siddharth Gollapudi, Sepideh Mahabadi, Varun Sivashankar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81c565e605161fcf25d08aa230431eba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81c565e605161fcf25d08aa230431eba-Abstract-Conference.html)

**Abstract**:

Given a set of $n$ vectors in $\mathbb{R}^d$, the goal of the \emph{determinant maximization} problem is to pick $k$ vectors with the maximum volume. Determinant maximization is the MAP-inference task for determinantal point processes (DPP) and has recently received considerable attention for modeling diversity. As most applications for the problem use large amounts of data, this problem has been studied in the relevant \textit{composable coreset} setting.In particular, [Indyk-Mahabadi-OveisGharan-Rezaei--SODA'20, ICML'19] showed that one can get composable coresets with optimal approximation factor of $\tilde O(k)^k$ for the problem, and that a local search algorithm achieves an almost optimal approximation guarantee of $O(k)^{2k}$.In this work, we show that the widely-used Greedy algorithm also provides composable coresets with an almost optimal approximation factor of $O(k)^{3k}$, which improves over the previously known guarantee of $C^{k^2}$, and supports the prior experimental results showing the practicality of the greedy algorithm as a coreset.Our main result follows by showing a local optimality property for Greedy:swapping a single point from the greedy solution with a vector that was not picked by the greedy algorithm can increase the volume by a factor of at most $(1+\sqrt{k})$. This is tight up to the additive constant $1$. Finally, our experiments show that the local optimality of the greedy algorithm is even lower than the theoretical bound on real data sets.

----

## [1798] ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab

**Authors**: *Jieming Cui, Ziren Gong, Baoxiong Jia, Siyuan Huang, Zilong Zheng, Jianzhu Ma, Yixin Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81c7202dbd3cd3006b35a58a076195c0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/81c7202dbd3cd3006b35a58a076195c0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The challenge of replicating research results has posed a significant impediment to the field of molecular biology. The advent of modern intelligent systems has led to notable progress in various domains. Consequently, we embarked on an investigation of intelligent monitoring systems as a means of tackling the issue of the reproducibility crisis. Specifically, we first curate a comprehensive multimodal dataset, named ProBio, as an initial step towards this objective. This dataset comprises fine-grained hierarchical annotations intended for the purpose of studying activity understanding in BioLab. Next, we devise two challenging benchmarks, transparent solution tracking and multimodal action recognition, to emphasize the unique characteristics and difficulties associated with activity understanding in BioLab settings. Finally, we provide a thorough experimental evaluation of contemporary video understanding models and highlight their limitations in this specialized domain to identify potential avenues for future research. We hope \dataset with associated benchmarks may garner increased focus on modern AI techniques in the realm of molecular biology.

----

## [1799] Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases

**Authors**: *Mazda Moayeri, Wenxiao Wang, Sahil Singla, Soheil Feizi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html)

**Abstract**:

We present a simple but effective method to measure and mitigate model biases caused by reliance on spurious cues. Instead of requiring costly changes to one's data or model training, our method better utilizes the data one already has by sorting them. Specifically, we rank images within their classes based on spuriosity (the degree to which common spurious cues are present), proxied via deep neural features of an interpretable network. With spuriosity rankings, it is easy to identify minority subpopulations (i.e. low spuriosity images) and assess model bias as the gap in accuracy between high and low spuriosity images. One can even efficiently remove a model's bias at little cost to accuracy by finetuning its classification head on low spuriosity images, resulting in fairer treatment of samples regardless of spuriosity. We demonstrate our method on ImageNet, annotating $5000$ class-feature dependencies ($630$ of which we find to be spurious) and generating a dataset of $325k$ soft segmentations for these features along the way. Having computed spuriosity rankings via the identified spurious neural features, we assess biases for $89$ diverse models and find that class-wise biases are highly correlated across models. Our results suggest that model bias due to spurious feature reliance is influenced far more by what the model is trained on than how it is trained.

----



[Go to the previous page](NIPS-2023-list8.md)

[Go to the next page](NIPS-2023-list10.md)

[Go to the catalog section](README.md)