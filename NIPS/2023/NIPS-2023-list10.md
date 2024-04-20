## [0] An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions

**Authors**: *Yingtai Xiao, Guanlin He, Danfeng Zhang, Daniel Kifer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/414f4c9fe9653e5de98fad6964d50315-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/414f4c9fe9653e5de98fad6964d50315-Abstract-Conference.html)

**Abstract**:

Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly run out of memory, even for relatively small datasets).

----

## [0] Recovering Unbalanced Communities in the Stochastic Block Model with Application to Clustering with a Faulty Oracle

**Authors**: *Chandra Sekhar Mukherjee, Pan Peng, Jiapeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41623b137cd34807f56028aa9f6f84a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41623b137cd34807f56028aa9f6f84a7-Abstract-Conference.html)

**Abstract**:

The stochastic block model (SBM) is a fundamental model for studying graph clustering or community detection in networks. It has received great attention in the last decade and the balanced case, i.e., assuming all clusters have large size, has been well studied. However, our understanding of SBM with unbalanced communities (arguably, more relevant in practice) is still limited. In this paper, we provide a simple SVD-based algorithm for recovering the communities in the SBM with communities of varying sizes.We improve upon a result of Ailon, Chen and Xu [ICML 2013; JMLR 2015] by removing the assumption that there is a large interval such that the sizes of clusters do not fall in, and also remove the dependency of the size of the recoverable clusters on the number of underlying clusters. We further complement our theoretical improvements with experimental comparisons.Under the planted clique conjecture, the size of the clusters that can be recovered by our algorithm is nearly optimal (up to poly-logarithmic factors) when the probability parameters are constant. As a byproduct, we obtain an efficient clustering algorithm with sublinear query complexity in a faulty oracle model, which is capable of detecting all clusters larger than $\tilde{\Omega}({\sqrt{n}})$, even in the presence of $\Omega(n)$ small clusters in the graph. In contrast, previous efficient algorithms that use a sublinear number of queries are incapable of recovering any large clusters if there are more than $\tilde{\Omega}(n^{2/5})$ small clusters.

----

## [0] Transition-constant Normalization for Image Enhancement

**Authors**: *Jie Huang, Man Zhou, Jinghao Zhang, Gang Yang, Mingde Yao, Chongyi Li, Zhiwei Xiong, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4163873c9ad623a87989d0a6eefe9442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4163873c9ad623a87989d0a6eefe9442-Abstract-Conference.html)

**Abstract**:

Normalization techniques that capture image style by statistical representation have become a popular component in deep neural networks.Although image enhancement can be considered as a form of style transformation, there has been little exploration of how normalization affect the enhancement performance. To fully leverage the potential of normalization, we present a novel Transition-Constant Normalization (TCN) for various image enhancement tasks.Specifically, it consists of two streams of normalization operations arranged under an invertible constraint, along with a feature sub-sampling operation that satisfies the normalization constraint.TCN enjoys several merits, including being parameter-free, plug-and-play, and incurring no additional computational costs.We provide various formats to utilize TCN for image enhancement, including seamless  integration with enhancement networks, incorporation into encoder-decoder architectures for downsampling, and implementation of efficient architectures.Through extensive experiments on multiple image enhancement tasks, like low-light enhancement, exposure correction, SDR2HDR translation, and image dehazing, our TCN consistently demonstrates performance improvements.Besides, it showcases extensive ability in other tasks including pan-sharpening and medical segmentation.The code is available at  \textit{\textcolor{blue}{https://github.com/huangkevinj/TCNorm}}.

----

## [0] Unexpected Improvements to Expected Improvement for Bayesian Optimization

**Authors**: *Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, Eytan Bakshy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html)

**Abstract**:

Expected Improvement (EI) is arguably the most popular acquisition function in Bayesian optimization and has found countless successful applications, but its performance is often exceeded by that of more recent methods. Notably, EI and its variants, including for the parallel and multi-objective settings, are challenging to optimize because their acquisition values vanish numerically in many regions. This difficulty generally increases as the number of observations, dimensionality of the search space, or the number of constraints grow, resulting in performance that is inconsistent across the literature and most often sub-optimal. Herein, we propose LogEI, a new family of acquisition functions whose members either have identical or approximately equal optima as their canonical counterparts, but are substantially easier to optimize numerically. We demonstrate that numerical pathologies manifest themselves in “classic” analytic EI, Expected Hypervolume Improvement (EHVI), as well as their constrained, noisy, and parallel variants, and propose corresponding reformulations that remedy these pathologies. Our empirical results show that members of the LogEI family of acquisition functions substantially improve on the optimization performance of their canonical counterparts and surprisingly, are on par with or exceed the performance of recent state-of-the-art acquisition functions, highlighting the understated role of numerical optimization in the literature.

----

## [0] Pseudo-Likelihood Inference

**Authors**: *Theo Gruner, Boris Belousov, Fabio Muratore, Daniel Palenicek, Jan R. Peters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41aa1c9f57ea83d7c41f0d3e98ed3dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41aa1c9f57ea83d7c41f0d3e98ed3dd4-Abstract-Conference.html)

**Abstract**:

Simulation-Based Inference (SBI) is a common name for an emerging family of approaches that infer the model parameters when the likelihood is intractable. Existing SBI methods either approximate the likelihood, such as Approximate Bayesian Computation (ABC) or directly model the posterior, such as Sequential Neural Posterior Estimation (SNPE). While ABC is efficient on low-dimensional problems, on higher-dimensional tasks, it is generally outperformed by SNPE, which leverages function approximation. In this paper, we propose Pseudo-Likelihood Inference (PLI), a new method that brings neural approximation into ABC, making it competitive on challenging Bayesian system identification tasks. By utilizing integral probability metrics, we introduce a smooth likelihood kernel with an adaptive bandwidth that is updated based on information-theoretic trust regions. Thanks to this formulation, our method (i) allows for optimizing neural posteriors via gradient descent, (ii) does not rely on summary statistics, and (iii) enables multiple observations as input. In comparison to SNPE, it leads to improved performance when more data is available. The effectiveness of PLI is evaluated on four classical SBI benchmark tasks and on a highly dynamic physical system, showing particular advantages on stochastic simulations and multi-modal posterior landscapes.

----

## [0] Calibrating "Cheap Signals" in Peer Review without a Prior

**Authors**: *Yuxuan Lu, Yuqing Kong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41badd36e935f8a80175e95d8bc6192e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41badd36e935f8a80175e95d8bc6192e-Abstract-Conference.html)

**Abstract**:

Peer review lies at the core of the academic process, but even well-intentioned reviewers can still provide noisy ratings. While ranking papers by average ratings may reduce noise, varying noise levels and systematic biases stemming from ``cheap'' signals (e.g. author identity, proof length) can lead to unfairness. Detecting and correcting bias is challenging, as ratings are subjective and unverifiable. Unlike previous works relying on prior knowledge or historical data, we propose a one-shot noise calibration process without any prior information. We ask reviewers to predict others' scores and use these predictions for calibration. Assuming reviewers adjust their predictions according to the noise, we demonstrate that the calibrated score results in a more robust ranking compared to average ratings, even with varying noise levels and biases.In detail, we show that the error probability of the calibrated score approaches zero as the number of reviewers increases and is significantly lower compared to average ratings when the number of reviewers is small.

----

## [0] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds

**Authors**: *Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, Jian Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41bcc9d3bddd9c90e1f44b29e26d97ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41bcc9d3bddd9c90e1f44b29e26d97ff-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models can create stunning images from natural language descriptions that rival the work of professional artists and photographers. However, these models are large, with complex network architectures and tens of denoising iterations, making them computationally expensive and slow to run. As a result, high-end GPUs and cloud-based inference are required to run diffusion models at scale. This is costly and has privacy implications, especially when user data is sent to a third party. To overcome these challenges, we present a generic approach that, for the first time, unlocks running text-to-image diffusion models on mobile devices in **less than 2 seconds**.  We achieve so by introducing efficient network architecture and improving step distillation. Specifically, we propose an efficient UNet by identifying the redundancy of the original model and reducing the computation of the image decoder via data distillation. Further, we enhance the step distillation by exploring training strategies and introducing regularization from classifier-free guidance. Our extensive experiments on MS-COCO show that our model with $8$ denoising steps achieves better FID and CLIP scores than Stable Diffusion v$1.5$ with $50$ steps. Our work democratizes content creation by bringing powerful text-to-image diffusion models to the hands of users.

----

## [0] LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference

**Authors**: *Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html)

**Abstract**:

The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The  deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2Ã— latency speedup relative to CryptoGCN, while preserving an inference accuracy of ~75\% and notably reducing multiplication depth. Additionally, LinGCN proves scalable for larger models, delivering a substantial 85.78\% accuracy with 6371s latency, a 10.47\% accuracy improvement over CryptoGCN.

----

## [0] Spectral Evolution and Invariance in Linear-width Neural Networks

**Authors**: *Zhichao Wang, Andrew Engel, Anand D. Sarwate, Ioana Dumitriu, Tony Chiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41ed4bd197d0a5fa036d361c1fc606ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41ed4bd197d0a5fa036d361c1fc606ad-Abstract-Conference.html)

**Abstract**:

We investigate the spectral properties of linear-width feed-forward neural networks, where the sample size is asymptotically proportional to network width. Empirically, we show that the spectra of weight in this high dimensional regime are invariant when trained by gradient descent for small constant learning rates; we provide a theoretical justification for this observation and prove the invariance of the bulk spectra for both conjugate and neural tangent kernels. We demonstrate similar characteristics when training with stochastic gradient descent with small learning rates. When the learning rate is large, we exhibit the emergence of an outlier whose corresponding eigenvector is aligned with the training data structure. We also show that after adaptive gradient training, where a lower test error and feature learning emerge, both weight and kernel matrices exhibit heavy tail behavior. Simple examples are provided to explain when heavy tails can have better generalizations. We exhibit different spectral properties such as invariant bulk, spike, and heavy-tailed distribution from a two-layer neural network using different training strategies, and then correlate them to the feature learning. Analogous phenomena also appear when we train conventional neural networks with real-world data. We conclude that monitoring the evolution of the spectra during training is an essential step toward understanding the training dynamics and feature learning.

----

## [0] Paxion: Patching Action Knowledge in Video-Language Foundation Models

**Authors**: *Zhenhailong Wang, Ansel Blume, Sha Li, Genglin Liu, Jaemin Cho, Zineng Tang, Mohit Bansal, Heng Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html)

**Abstract**:

Action knowledge involves the understanding of textual, visual, and temporal aspects of actions. We introduce the Action Dynamics Benchmark (ActionBench) containing two carefully designed probing tasks: Action Antonym and Video Reversal, which targets multimodal alignment capabilities and temporal understanding skills of the model, respectively. Despite recent video-language models’ (VidLM) impressive performance on various benchmark tasks, our diagnostic tasks reveal their surprising deficiency (near-random performance) in action knowledge, suggesting that current models rely on object recognition abilities as a shortcut for action understanding. To remedy this, we propose a novel framework, Paxion, along with a new Discriminative Video Dynamics Modeling (DVDM) objective. The Paxion framework utilizes a Knowledge Patcher network to encode new action knowledge and a Knowledge Fuser component to integrate the Patcher into frozen VidLMs without compromising their existing capabilities. Due to limitations of the widely-used Video-Text Contrastive (VTC) loss for learning action knowledge, we introduce the DVDM objective to train the Knowledge Patcher. DVDM forces the model to encode the correlation between the action text and the correct ordering of video frames. Our extensive analyses show that Paxion and DVDM together effectively fill the gap in action knowledge understanding (~50% → 80%), while maintaining or improving performance on a wide spectrum of both object- and action-centric downstream tasks.

----

## [0] ProPILE: Probing Privacy Leakage in Large Language Models

**Authors**: *Siwon Kim, Sangdoo Yun, Hwaran Lee, Martin Gubri, Sungroh Yoon, Seong Joon Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/420678bb4c8251ab30e765bc27c3b047-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/420678bb4c8251ab30e765bc27c3b047-Abstract-Conference.html)

**Abstract**:

The rapid advancement and widespread use of large language models (LLMs) have raised significant concerns regarding the potential leakage of personally identifiable information (PII). These models are often trained on vast quantities of web-collected data, which may inadvertently include sensitive personal data. This paper presents ProPILE, a novel probing tool designed to empower data subjects, or the owners of the PII, with awareness of potential PII leakage in LLM-based services. ProPILE lets data subjects formulate prompts based on their own PII to evaluate the level of privacy intrusion in LLMs. We demonstrate its application on the OPT-1.3B model trained on the publicly available Pile dataset. We show how hypothetical data subjects may assess the likelihood of their PII being included in the Pile dataset being revealed. ProPILE can also be leveraged by LLM service providers to effectively evaluate their own levels of PII leakage with more powerful prompts specifically tuned for their in-house models. This tool represents a pioneering step towards empowering the data subjects for their awareness and control over their own data on the web.

----

## [0] Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension

**Authors**: *Moritz Haas, David Holzmüller, Ulrike von Luxburg, Ingo Steinwart*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/421f83663c02cdaec8c3c38337709989-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/421f83663c02cdaec8c3c38337709989-Abstract-Conference.html)

**Abstract**:

The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting, where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a sequence of spiky-smooth kernels with large derivatives. Using neural tangent kernels, we translate our results to wide neural networks. We prove that while infinite-width networks do not overfit benignly with the ReLU activation, this can be fixed by adding small high-frequency fluctuations to the activation function. Our experiments verify that such neural networks, while overfitting, can indeed generalize well even on low-dimensional data sets.

----

## [0] The Goldilocks of Pragmatic Understanding: Fine-Tuning Strategy Matters for Implicature Resolution by LLMs

**Authors**: *Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, Edward Grefenstette*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4241fec6e94221526b0a9b24828bb774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4241fec6e94221526b0a9b24828bb774-Abstract-Conference.html)

**Abstract**:

Despite widespread use of LLMs as conversational agents, evaluations of performance fail to capture a crucial aspect of communication: interpreting language in context---incorporating its pragmatics. Humans interpret language using beliefs and prior knowledge about the world. For example, we intuitively understand the response "I wore gloves" to the question "Did you leave fingerprints?" as meaning "No". To investigate whether LLMs have the ability to make this type of inference, known as an implicature, we design a simple task and evaluate four categories of widely used state-of-the-art models. We find that, despite only evaluating on utterances that require a binary inference (yes or no), models in three of these categories perform close to random. However, LLMs instruction-tuned at the example-level perform significantly better. These results suggest that certain fine-tuning strategies are far better at inducing pragmatic understanding in models. We present our findings as the starting point for further research into evaluating how LLMs interpret language in context and to drive the development of more pragmatic and useful models of human discourse.

----

## [0] Slow and Weak Attractor Computation Embedded in Fast and Strong E-I Balanced Neural Dynamics

**Authors**: *Xiaohan Lin, Liyuan Li, Boxin Shi, Tiejun Huang, Yuanyuan Mi, Si Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/425ee25d6c22ef98b67328273b8f95d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/425ee25d6c22ef98b67328273b8f95d5-Abstract-Conference.html)

**Abstract**:

Attractor networks require neuronal connections to be highly structured in order to maintain attractor states that represent information, while excitation and inhibition balanced networks (E-INNs) require neuronal connections to be random and sparse to generate irregular neuronal firings. Despite being regarded as canonical models of neural circuits, both types of networks are usually studied in isolation, and it remains unclear how they coexist in the brain, given their very different structural demands. In this study, we investigate the compatibility of continuous attractor neural networks (CANNs) and E-INNs. In line with recent experimental data, we find that a neural circuit can exhibit both the traits of CANNs and E-INNs if the neuronal synapses consist of two sets: one set is strong and fast for irregular firing, and the other set is weak and slow for attractor dynamics. Our results from simulations and theoretical analysis reveal that the network also exhibits enhanced performance compared to the case of using only one set of synapses, with accelerated convergence of attractor states and retained E-I balanced condition for localized input. We also apply the network model to solve a real-world tracking problem and demonstrate that it can track fast-moving objects well. We hope that this study provides insight into how structured neural computations are realized by irregular firings of neurons.

----

## [0] Test-time Training for Matching-based Video Object Segmentation

**Authors**: *Juliette Bertrand, Giorgos Kordopatis-Zilos, Yannis Kalantidis, Giorgos Tolias*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html)

**Abstract**:

The video object segmentation (VOS) task involves the segmentation of an object over time based on a single initial mask. Current state-of-the-art approaches use a memory of previously processed frames and rely on matching to estimate segmentation masks of subsequent frames. Lacking any adaptation mechanism, such methods are prone to test-time distribution shifts. This work focuses on matching-based VOS under distribution shifts such as video corruptions, stylization, and sim-to-real transfer. We explore test-time training strategies that are agnostic to the specific task as well as strategies that are designed specifically for VOS. This includes a variant based on mask cycle consistency tailored to matching-based VOS methods. The experimental results on common benchmarks demonstrate that the proposed test-time training yields significant improvements in performance. In particular for the sim-to-real scenario and despite using only a single test video, our approach manages to recover a substantial portion of the performance gain achieved through training on real videos. Additionally, we introduce DAVIS-C, an augmented version of the popular DAVIS test set, featuring extreme distribution shifts like image-/video-level corruptions and stylizations. Our results illustrate that test-time training enhances performance even in these challenging cases.

----

## [0] Causal Effect Regularization: Automated Detection and Removal of Spurious Correlations

**Authors**: *Abhinav Kumar, Amit Deshpande, Amit Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html)

**Abstract**:

In many classification datasets, the task labels are spuriously correlated with some input attributes. Classifiers trained on such datasets often rely on these attributes for prediction, especially when the spurious correlation is high, and thus fail togeneralize whenever there is a shift in the attributes’ correlation at deployment. If we assume that the spurious attributes are known a priori, several methods have been proposed to learn a classifier that is invariant to the specified attributes. However, in real-world data, information about spurious attributes is typically unavailable. Therefore, we propose a method that automatically identifies spurious attributes by estimating their causal effect on the label and then uses a regularization objective to mitigate the classifier’s reliance on them. Although causal effect of an attribute on the label is not always identified, we present two commonly occurring data-generating processes where the effect can be identified. Compared to recent work for identifying spurious attributes, we find that our method, AutoACER, ismore accurate in removing the attribute from the learned model, especially when spurious correlation is high. Specifically, across synthetic, semi-synthetic, and real-world datasets, AutoACER shows significant improvement in a metric used to quantify the dependence of a classifier on spurious attributes ($\Delta$Prob), while obtaining better or similar accuracy. Empirically we find that AutoACER mitigatesthe reliance on spurious attributes even under noisy estimation of causal effects or when the causal effect is not identified. To explain the empirical robustness of our method, we create a simple linear classification task with two sets of attributes: causal and spurious. Under this setting, we prove that AutoACER only requires the ranking of estimated causal effects to be correct across attributes to select thecorrect classifier.

----

## [0] Multi-resolution Spectral Coherence for Graph Generation with Score-based Diffusion

**Authors**: *Hyuna Cho, Minjae Jeong, Sooyeon Jeon, Sungsoo Ahn, Won Hwa Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/427f20d90386fd27804f1831d6a3d48f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/427f20d90386fd27804f1831d6a3d48f-Abstract-Conference.html)

**Abstract**:

Successful graph generation depends on the accurate estimation of the joint distribution of graph components such as nodes and edges from training data. While recent deep neural networks have demonstrated sampling of realistic graphs together with diffusion models, however, they still suffer from oversmoothing problems which are inherited from conventional graph convolution and thus high-frequency characteristics of nodes and edges become intractable. To overcome such issues and generate graphs with high fidelity, this paper introduces a novel approach that captures the dependency between nodes and edges at multiple resolutions in the spectral space. By modeling the joint distribution of node and edge signals in a shared graph wavelet space, together with a score-based diffusion model, we propose a Wavelet Graph Diffusion Model (Wave-GD) which lets us sample synthetic graphs with real-like frequency characteristics of nodes and edges. Experimental results on four representative benchmark datasets validate the superiority of the Wave-GD over existing approaches, highlighting its potential for a wide range of applications that involve graph data.

----

## [0] Real-World Image Super-Resolution as Multi-Task Learning

**Authors**: *Wenlong Zhang, Xiaohui Li, Guangyuan Shi, Xiangyu Chen, Yu Qiao, Xiaoyun Zhang, Xiao-Ming Wu, Chao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42806406dd99e30c3796bc98b2670fa2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42806406dd99e30c3796bc98b2670fa2-Abstract-Conference.html)

**Abstract**:

In this paper, we take a new look at real-world image super-resolution (real-SR) from a multi-task learning perspective. We demonstrate that the conventional formulation of real-SR can be viewed as solving multiple distinct degradation tasks using a single shared model. This poses a challenge known as task competition or task conflict in multi-task learning, where certain tasks dominate the learning process, resulting in poor performance on other tasks. This problem is exacerbated in the case of real-SR, due to the involvement of numerous degradation tasks. To address the issue of task competition in real-SR, we propose a task grouping approach. Our approach efficiently identifies the degradation tasks where a real-SR model falls short and groups these unsatisfactory tasks into multiple task groups. We then utilize the task groups to fine-tune the real-SR model in a simple way, which effectively mitigates task competition and facilitates knowledge transfer. Extensive experiments demonstrate our method achieves significantly enhanced performance across a wide range of degradation scenarios.

----

## [0] Exact Representation of Sparse Networks with Symmetric Nonnegative Embeddings

**Authors**: *Sudhanshu Chanpuriya, Ryan A. Rossi, Anup B. Rao, Tung Mai, Nedim Lipka, Zhao Song, Cameron Musco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/428ceef2cd8a53add7213e04d1746479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/428ceef2cd8a53add7213e04d1746479-Abstract-Conference.html)

**Abstract**:

Graph models based on factorization of the adjacency matrix often fail to capture network structures related to links between dissimilar nodes (heterophily). We introduce a novel graph factorization model that leverages two nonnegative vectors per node to interpretably account for links between both similar and dissimilar nodes. We prove that our model can exactly represent any graph with low arboricity, a property that many real-world networks satisfy; our proof also applies to related models but has much greater scope than the closest prior bound, which is based on low max degree. Our factorization also has compelling properties besides expressiveness: due to its symmetric structure and nonnegativity, fitting the model inherently finds node communities, and the model's link predictions can be interpreted in terms of these communities. In experiments on real-world networks, we demonstrate our factorization's effectiveness on a variety of tasks, including community detection and link prediction.

----

## [0] Data-Centric Learning from Unlabeled Graphs with Diffusion Model

**Authors**: *Gang Liu, Eric Inae, Tong Zhao, Jiaxin Xu, Tengfei Luo, Meng Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4290cccf23be59e42a575d026ccbeeb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4290cccf23be59e42a575d026ccbeeb8-Abstract-Conference.html)

**Abstract**:

Graph property prediction tasks are important and numerous. While each task offers a small size of labeled examples, unlabeled graphs have been collected from various sources and at a large scale. A conventional approach is training a model with the unlabeled graphs on self-supervised tasks and then fine-tuning the model on the prediction tasks. However, the self-supervised task knowledge could not be aligned or sometimes conflicted with what the predictions needed. In this paper, we propose to extract the knowledge underlying the large set of unlabeled graphs as a specific set of useful data points to augment each property prediction model. We use a diffusion model to fully utilize the unlabeled graphs and design two new objectives to guide the model's denoising process with each task's labeled data to generate task-specific graph examples and their labels. Experiments demonstrate that our data-centric approach performs significantly better than fifteen existing various methods on fifteen tasks. The performance improvement brought by unlabeled data is visible as the generated labeled examples unlike the self-supervised learning.

----

## [0] Wasserstein Gradient Flows for Optimizing Gaussian Mixture Policies

**Authors**: *Hanna Ziesche, Leonel Rozo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/429b5216a4d08850c586fbf809e17877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/429b5216a4d08850c586fbf809e17877-Abstract-Conference.html)

**Abstract**:

Robots often rely on a repertoire of previously-learned motion policies for performing tasks of diverse complexities.  When facing unseen task conditions or when new task requirements arise, robots must adapt their motion policies accordingly. In this context, policy optimization is the \emph{de facto} paradigm to adapt robot policies as a function of task-specific objectives.  Most commonly-used motion policies carry particular structures that are often overlooked in policy optimization algorithms.  We instead propose to leverage the structure of probabilistic policies by casting the policy optimization as an optimal transport problem. Specifically, we focus on robot motion policies that build on Gaussian mixture models (GMMs) and formulate the policy optimization as a Wassertein gradient flow over the GMMs space. This naturally allows us to constrain the policy updates via the $L^2$-Wasserstein distance between GMMs to enhance the stability of the policy optimization process. Furthermore, we leverage the geometry of the Bures-Wasserstein manifold to optimize the Gaussian distributions of the GMM policy via Riemannian optimization. We evaluate our approach on common robotic settings: Reaching motions, collision-avoidance behaviors, and multi-goal tasks. Our results show that our method outperforms common policy optimization baselines in terms of task success rate and low-variance solutions.

----

## [0] Change point detection and inference in multivariate non-parametric models under mixing conditions

**Authors**: *Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, Oscar Hernan Madrid Padilla, Yi Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have HÃ¶lder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant.  Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator.  Numerical studies are provided to complement our theoretical findings.

----

## [0] Near-optimal learning with average Hölder smoothness

**Authors**: *Guy Kornowski, Steve Hanneke, Aryeh Kontorovich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42afce512806ab874b9f99ed9a08055e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42afce512806ab874b9f99ed9a08055e-Abstract-Conference.html)

**Abstract**:

We generalize the notion of average Lipschitz smoothness proposed by Ashlagi et al. (COLT 2021) by extending it to Hölder smoothness. This measure of the "effective smoothness" of a function is sensitive to the underlying distribution and can be dramatically smaller than its classic "worst-case" Hölder constant.We consider both the realizable and the agnostic (noisy) regression settings, proving upper and lower risk bounds in terms of the average Hölder smoothness; these rates improve upon both previously known rates even in the special case of average Lipschitz smoothness.Moreover, our lower bound is tight in the realizable setting up to log factors, thus we establish the minimax rate.From an algorithmic perspective, since our notion of average smoothness is defined with respect to the unknown underlying distribution, the learner does not have an explicit representation of the function class, hence is unable to execute ERM. Nevertheless, we provide distinct learning algorithms that achieve both (nearly) optimal learning rates.Our results hold in any totally bounded metric space, and are stated in terms of its intrinsic geometry.Overall, our results show that the classic worst-case notion of Hölder smoothness can be essentially replaced by its average, yielding considerably sharper guarantees.

----

## [0] Neural-Logic Human-Object Interaction Detection

**Authors**: *Liulei Li, Jianan Wei, Wenguan Wang, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html)

**Abstract**:

The interaction decoder utilized in prevalent Transformer-based HOI detectors typically accepts pre-composed human-object pairs as inputs. Though achieving remarkable performance, such a paradigm lacks feasibility and cannot explore novel combinations over entities during decoding. We present LogicHOI, a new HOI detector that leverages neural-logic reasoning and Transformer to infer feasible interactions between. entities. Specifically, we modify. self-attention mechanism in the vanilla Transformer, enabling it to reason over the ⟨ human, action, object ⟩ triplet and constitute novel interactions. Meanwhile, such a reasoning process is guided by two crucial properties for understanding HOI: affordances (the potential actions an object can facilitate) and proxemics (the spatial relations between humans and objects). We formulate these two properties in first-order logic and ground them into continuous space to constrain the learning process of our approach, leading to improved performance and zero-shot generalization capabilities. We evaluate L OGIC HOI on V-COCO and HICO-DET under both normal and zero-shot setups, achieving significant improvements over existing methods.

----

## [0] Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness

**Authors**: *Suraj Srinivas, Sebastian Bordt, Himabindu Lakkaraju*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42bbe2bfdbbfcadda643e8f89025716c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42bbe2bfdbbfcadda643e8f89025716c-Abstract-Conference.html)

**Abstract**:

One of the remarkable properties of robust computer vision models is that their input-gradients are often aligned with human perception, referred to in the literature as perceptually-aligned gradients (PAGs). Despite only being trained for classification, PAGs cause robust models to have rudimentary generative capabilities, including image generation, denoising, and in-painting. However, the underlying mechanisms behind these phenomena remain unknown. In this work, we provide a first explanation of PAGs via \emph{off-manifold robustness}, which states that models must be more robust off- the data manifold than they are on-manifold. We first demonstrate theoretically that off-manifold robustness leads input gradients to lie approximately on the data manifold, explaining their perceptual alignment. We then show that Bayes optimal models satisfy off-manifold robustness, and confirm the same empirically for robust models trained via gradient norm regularization, randomized smoothing, and adversarial training with projected gradient descent. Quantifying the perceptual alignment of model gradients via their similarity with the gradients of generative models, we show that off-manifold robustness correlates well with perceptual alignment. Finally, based on the levels of on- and off-manifold robustness, we identify three different regimes of robustness that affect both perceptual alignment and model accuracy: weak robustness, bayes-aligned robustness, and excessive robustness. Code is available at https://github.com/tml-tuebingen/pags.

----

## [0] Inferring the Future by Imagining the Past

**Authors**: *Kartik Chandra, Tony Chen, Tzu-Mao Li, Jonathan Ragan-Kelley, Josh Tenenbaum*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42c3438f432bc62014ce65af880e0d94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42c3438f432bc62014ce65af880e0d94-Abstract-Conference.html)

**Abstract**:

A single panel of a comic book can say a lot: it can depict not only where the characters currently are, but also their motions, their motivations, their emotions, and what they might do next. More generally, humans routinely infer complex sequences of past and future events from a static snapshot of a dynamic scene, even in situations they have never seen before.In this paper, we model how humans make such rapid and flexible inferences. Building on a long line of work in cognitive science, we offer a Monte Carlo algorithm whose inferences correlate well with human intuitions in a wide variety of domains, while only using a small, cognitively-plausible number of samples. Our key technical insight is a surprising connection between our inference problem and Monte Carlo path tracing, which allows us to apply decades of ideas from the computer graphics community to this seemingly-unrelated theory of mind task.

----

## [0] The Grand Illusion: The Myth of Software Portability and Implications for ML Progress

**Authors**: *Fraser Mince, Dzung Dinh, Jonas Kgomo, Neil Thompson, Sara Hooker*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42c40aff7814e9796266e12053b1c610-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42c40aff7814e9796266e12053b1c610-Abstract-Conference.html)

**Abstract**:

Pushing the boundaries of machine learning often requires exploring different hardware and software combinations. However, this ability to experiment with different systems can be at odds with the drive for efficiency, which has produced increasingly specialized AI hardware and incentivized consolidation around a narrow set of ML frameworks. Exploratory research can be further restricted if software and hardware are co-evolving, making it even harder to stray away from a given tooling stack. While this friction increasingly impacts the rate of innovation in machine learning, to our knowledge the lack of portability in tooling has not been quantified. In this work we ask: How portable are popular ML software frameworks? We conduct a large scale study of the portability of mainstream ML frameworks across different hardware types. Our findings paint an uncomfortable picture -- frameworks can lose more than 40% of their key functions when ported to other hardware. Worse, even when functions are portable, the slowdown in their performance can be extreme. Collectively, our results reveal how costly straying from a narrow set of hardware-software combinations can be - and thus how specialization incurs an exploration cost that can impede innovation in machine learning research.

----

## [0] Computing Optimal Nash Equilibria in Multiplayer Games

**Authors**: *Youzhi Zhang, Bo An, Venkatramanan Siva Subrahmanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42cac45fb00f7038c892f1a1bfc216d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42cac45fb00f7038c892f1a1bfc216d3-Abstract-Conference.html)

**Abstract**:

Designing efficient algorithms to compute a Nash Equilibrium (NE) in multiplayer games is still an open challenge. In this paper, we focus on computing an NE that optimizes a given objective function. For example, when there is a team of players independently playing against an adversary in a game (e.g., several groups in a forest trying to interdict illegal loggers in green security games), these team members may need to find an NE minimizing the adversaryâ€™s utility. Finding an optimal NE in multiplayer games can be formulated as a mixed-integer bilinear program by introducing auxiliary variables to represent bilinear terms, leading to a huge number of bilinear terms, making it hard to solve. To overcome this challenge, we first propose a general framework for this formulation based on a set of correlation plans. We then develop a novel algorithm called CRM based on this framework, which uses correlation plans with their relations to strictly reduce the feasible solution space after the convex relaxation of bilinear terms while minimizing the number of correlation plans to significantly reduce the number of bilinear terms. We show that our techniques can significantly reduce the time complexity and CRM can be several orders of magnitude faster than the state-of-the-art baseline.

----

## [0] AND: Adversarial Neural Degradation for Learning Blind Image Super-Resolution

**Authors**: *Fangzhou Luo, Xiaolin Wu, Yanhui Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html)

**Abstract**:

Learnt deep neural networks for image super-resolution fail easily if the assumed degradation model in training mismatches that of the real degradation source at the inference stage. Instead of attempting to exhaust all degradation variants in simulation, which is unwieldy and impractical, we propose a novel adversarial neural degradation (AND) model that can, when trained in conjunction with a deep restoration neural network under a minmax criterion, generate a wide range of highly nonlinear complex degradation effects without any explicit supervision. The AND model has a unique advantage over the current state of the art in that it can generalize much better to unseen degradation variants and hence deliver significantly improved restoration performance on real-world images.

----

## [0] Into the LAION's Den: Investigating Hate in Multimodal Datasets

**Authors**: *Abeba Birhane, Vinay Uday Prabhu, Sanghyun Han, Vishnu Boddeti, Sasha Luccioni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42f225509e8263e2043c9d834ccd9a2b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/42f225509e8263e2043c9d834ccd9a2b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

`Scale the model, scale the data, scale the compute' is the reigning sentiment in the world of generative AI today. While the impact of model scaling has been extensively studied, we are only beginning to scratch the surface of data scaling and its consequences. This is especially of critical importance in the context of vision-language datasets such as LAION. These datasets are continually growing in size and are built based on large-scale internet dumps such as the Common Crawl, which is known to have numerous drawbacks ranging from quality, legality, and content. The datasets then serve as the backbone for large generative models, contributing to the operationalization and perpetuation of harmful societal and historical biases and stereotypes. In this paper, we investigate the effect of scaling datasets on hateful content through a comparative audit of two datasets: LAION-400M and LAION-2B. Our results show that hate content increased by nearly 12% with dataset scale, measured both qualitatively and quantitatively using a metric that we term as Hate Content Rate (HCR). We also found that filtering dataset contents based on Not Safe For Work (NSFW) values calculated based on images alone does not exclude all the harmful content in alt-text. Instead, we found that trace amounts of hateful, targeted, and aggressive text remain even when carrying out conservative filtering. We end with a reflection and a discussion of the significance of our results for dataset curation and usage in the AI community.Code and the meta-data assets curated in this paper are publicly available at https://github.com/vinayprabhu/hate_scaling. Content warning: This paper contains examples of hateful text that might be disturbing, distressing, and/or offensive.

----

## [0] SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation

**Authors**: *Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, Jian Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce an SE(3) diffusion model-based point cloud registration framework for 6D object pose estimation in real-world scenarios. Our approach formulates the 3D registration task as a denoising diffusion process, which progressively refines the pose of the source point cloud to obtain a precise alignment with the model point cloud. Training our framework involves two operations: An SE(3) diffusion process and an SE(3) reverse process. The SE(3) diffusion process gradually perturbs the optimal rigid transformation of a pair of point clouds by continuously injecting noise (perturbation transformation). By contrast, the SE(3) reverse process focuses on learning a denoising network that refines the noisy transformation step-by-step, bringing it closer to the optimal transformation for accurate pose estimation. Unlike standard diffusion models used in linear Euclidean spaces, our diffusion model operates on the SE(3) manifold. This requires exploiting the linear Lie algebra $\mathfrak{se}(3)$ associated with SE(3) to constrain the transformation transitions during the diffusion and reverse processes. Additionally, to effectively train our denoising network, we derive a registration-specific variational lower bound as the optimization objective for model learning. Furthermore, we show that our denoising network can be constructed with a surrogate registration model, making our approach applicable to different deep registration networks. Extensive experiments demonstrate that our diffusion registration framework presents outstanding pose estimation performance on the real-world TUD-L, LINEMOD, and Occluded-LINEMOD datasets.

----

## [0] RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions

**Authors**: *Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit Cottereau, Wei Tsang Ooi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Depth estimation from monocular images is pivotal for real-world visual perception systems. While current learning-based depth estimation models train and test on meticulously curated data, they often overlook out-of-distribution (OoD) situations. Yet, in practical settings -- especially safety-critical ones like autonomous driving -- common corruptions can arise. Addressing this oversight, we introduce a comprehensive robustness test suite, RoboDepth, encompassing 18 corruptions spanning three categories: i) weather and lighting conditions; ii) sensor failures and movement; and iii) data processing anomalies. We subsequently benchmark 42 depth estimation models across indoor and outdoor scenes to assess their resilience to these corruptions. Our findings underscore that, in the absence of a dedicated robustness evaluation framework, many leading depth estimation models may be susceptible to typical corruptions. We delve into design considerations for crafting more robust depth estimation models, touching upon pre-training, augmentation, modality, model capacity, and learning paradigms. We anticipate our benchmark will establish a foundational platform for advancing robust OoD depth estimation.

----

## [0] Fed-CO2: Cooperation of Online and Offline Models for Severe Data Heterogeneity in Federated Learning

**Authors**: *Zhongyi Cai, Ye Shi, Wei Huang, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) has emerged as a promising distributed learning paradigm that enables multiple clients to learn a global model collaboratively without sharing their private data. However, the effectiveness of FL is highly dependent on the quality of the data that is being used for training. In particular, data heterogeneity issues, such as label distribution skew and feature skew, can significantly impact the performance of FL. Previous studies in FL have primarily focused on addressing label distribution skew data heterogeneity, while only a few recent works have made initial progress in tackling feature skew issues. Notably, these two forms of data heterogeneity have been studied separately and have not been well explored within a unified FL framework. To address this gap, we propose Fed-CO$_2$, a universal FL framework that handles both label distribution skew and feature skew within a Cooperation mechanism between the Online and Offline models. Specifically, the online model learns general knowledge that is shared among all clients, while the offline model is trained locally to learn the specialized knowledge of each individual client. To further enhance model cooperation in the presence of feature shifts, we design an intra-client knowledge transfer mechanism that reinforces mutual learning between the online and offline models, and an inter-client knowledge transfer mechanism to increase the modelsâ€™ domain generalization ability. Extensive experiments show that our Fed-CO$_2$ outperforms a wide range of existing personalized federated learning algorithms in terms of handling label distribution skew and feature skew, both individually and collectively. The empirical results are supported by our convergence analyses in a simplified setting.

----

## [0] Combating Bilateral Edge Noise for Robust Link Prediction

**Authors**: *Zhanke Zhou, Jiangchao Yao, Jiaxu Liu, Xiawei Guo, Quanming Yao, LI He, Liang Wang, Bo Zheng, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/435986a8cc3e0667648df5d1c2d55c83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/435986a8cc3e0667648df5d1c2d55c83-Abstract-Conference.html)

**Abstract**:

Although link prediction on graphs has achieved great success with the development of graph neural networks (GNNs), the potential robustness under the edge noise is still less investigated. To close this gap, we first conduct an empirical study to disclose that the edge noise bilaterally perturbs both input topology and target label, yielding severe performance degradation and representation collapse. To address this dilemma, we propose an information-theory-guided principle, Robust Graph Information Bottleneck (RGIB), to extract reliable supervision signals and avoid representation collapse. Different from the basic information bottleneck, RGIB further decouples and balances the mutual dependence among graph topology, target labels, and representation, building new learning objectives for robust representation against the bilateral noise. Two instantiations, RGIB-SSL and RGIB-REP, are explored to leverage the merits of different methodologies, i.e., self-supervised learning and data reparameterization, for implicit and explicit data denoising, respectively. Extensive experiments on six datasets and three GNNs with diverse noisy scenarios verify the effectiveness of our RGIB instantiations. The code is publicly available at: https://github.com/tmlr-group/RGIB.

----

## [0] SyncTREE: Fast Timing Analysis for Integrated Circuit Design through a Physics-informed Tree-based Graph Neural Network

**Authors**: *Yuting Hu, Jiajie Li, Florian Klemme, Gi-Joon Nam, Tengfei Ma, Hussam Amrouch, Jinjun Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/435e8fbbfc2c6072d4f3a5cb6e56a39a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/435e8fbbfc2c6072d4f3a5cb6e56a39a-Abstract-Conference.html)

**Abstract**:

Nowadays integrated circuits (ICs) are underpinning all major information technology innovations including the current trends of artificial intelligence (AI). Modern IC designs often involve analyses of complex phenomena (such as timing, noise, and power etc.) for tens of billions of electronic components, like resistance (R), capacitance (C), transistors and gates, interconnected in various complex structures. Those analyses often need to strike a balance between accuracy and speed as those analyses need to be carried out many times throughout the entire IC design cycles. With the advancement of AI, researchers also start to explore news ways in leveraging AI to improve those analyses. This paper focuses on one of the most important analyses, timing analysis for interconnects. Since IC interconnects can be represented as an RC-tree, a specialized graph as tree, we design a novel tree-based graph neural network, SyncTREE, to speed up the timing analysis by incorporating both the structural and physical properties of electronic circuits. Our major innovations include (1) a two-pass message-passing (bottom-up and top-down) for graph embedding, (2) a tree contrastive loss to guide learning, and (3) a closed formular-based approach to conduct fast timing.  Our experiments show that, compared to conventional GNN models, SyncTREE achieves the best timing prediction in terms of both delays and slews, all in reference to the industry golden numerical analyses results on real IC design data.

----

## [0] Hierarchical Open-vocabulary Universal Image Segmentation

**Authors**: *Xudong Wang, Shufan Li, Konstantinos Kallidromitis, Yusuke Kato, Kazuki Kozuka, Trevor Darrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43663f64775ae439ec52b64305d219d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43663f64775ae439ec52b64305d219d3-Abstract-Conference.html)

**Abstract**:

Open-vocabulary image segmentation aims to partition an image into semantic regions according to arbitrary text descriptions. However, complex visual scenes can be naturally decomposed into simpler parts and abstracted at multiple lev4 els of granularity, introducing inherent segmentation ambiguity. Unlike existing methods that typically sidestep this ambiguity and treat it as an external factor, our approach actively incorporates a hierarchical representation encompassing different semantic-levels into the learning process. We propose a decoupled text-image fusion mechanism and representation learning modules for both “things” and “stuff”. Additionally, we systematically examine the differences that exist in the textual and visual features between these types of categories. Our resulting model, named HIPIE, tackles HIerarchical, oPen-vocabulary, and unIvErsal segmentation tasks within a unified framework. Benchmarked on diverse datasets, e.g., ADE20K,COCO, Pascal-VOC Part, and RefCOCO/RefCOCOg, HIPIE achieves the state-of14 the-art results at various levels of image comprehension, including semantic-level (e.g., semantic segmentation), instance-level (e.g., panoptic/referring segmentationand object detection), as well as part-level (e.g., part/subpart segmentation) tasks.

----

## [0] Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach

**Authors**: *Jinqiu Jin, Haoxuan Li, Fuli Feng, Sihao Ding, Peng Wu, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html)

**Abstract**:

Item-side group fairness (IGF) requires a recommendation model to treat different item groups similarly, and has a crucial impact on information diffusion, consumption activity, and market equilibrium. Previous IGF notions only focus on the direct utility of the item exposures, i.e., the exposure numbers across different item groups. Nevertheless, the item exposures also facilitate utility gained from the neighboring users via social influence, called social utility, such as information sharing on the social media. To fill this gap, this paper introduces two social attribute-aware IGF metrics, which require similar user social attributes on the exposed items across the different item groups. In light of the trade-off between the direct utility and social utility, we formulate a new multi-objective optimization problem for training recommender models with flexible trade-off while ensuring controllable accuracy. To solve this problem, we develop a gradient-based optimization algorithm and theoretically show that the proposed algorithm can find Pareto optimal solutions with varying trade-off and guaranteed accuracy. Extensive experiments on two real-world datasets validate the effectiveness of our approach.

----

## [0] Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos

**Authors**: *Matthew Chang, Aditya Prakash, Saurabh Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/437cd2749391ad40f67e4dd1d87c4596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/437cd2749391ad40f67e4dd1d87c4596-Abstract-Conference.html)

**Abstract**:

The analysis and use of egocentric videos for robotics tasks is made challenging by occlusion and the visual mismatch between the human hand and a robot end-effector. Past work views the human hand as a nuisance and removes it from the scene. However, the hand also provides a valuable signal for learning. In this work, we propose to extract a factored representation of the scene that separates the agent (human hand) and the environment. This alleviates both occlusion and mismatch while preserving the signal, thereby easing the design of models for downstream robotics tasks. At the heart of this factorization is our proposed Video Inpainting via Diffusion Model (VIDM) that leverages both a prior on real-world images (through a large-scale pre-trained diffusion model) and the appearance of the object in earlier frames of the video (through attention). Our experiments demonstrate the effectiveness of VIDM at improving the in-painting quality in egocentric videos and the power of our factored representation for numerous tasks: object detection, 3D reconstruction of manipulated objects, and learning of reward functions, policies, and affordances from videos.

----

## [0] Generating Images with Multimodal Language Models

**Authors**: *Jing Yu Koh, Daniel Fried, Russ Salakhutdinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43a69d143273bd8215578bde887bb552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43a69d143273bd8215578bde887bb552-Abstract-Conference.html)

**Abstract**:

We propose a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces. Our model demonstrates a wide suite of multimodal capabilities: image retrieval, novel image generation, and multimodal dialogue. Ours is the first approach capable of conditioning on arbitrarily interleaved image and text inputs to generate coherent image (and text) outputs. To achieve strong performance on image generation, we propose an efficient mapping network to ground the LLM to an off-the-shelf text-to-image generation model. This mapping network translates hidden representations of text into the embedding space of the visual models, enabling us to leverage the strong text representations of the LLM for visual outputs. Our approach outperforms baseline generation models on tasks with longer and more complex language. In addition to novel image generation, our model is also capable of image retrieval from a prespecified dataset, and decides whether to retrieve or generate at inference time. This is done with a learnt decision module which conditions on the hidden representations of the LLM. Our model exhibits a wider range of capabilities compared to prior multimodal language models. It can process image-and-text inputs, and produce retrieved images, generated images, and generated text â€” outperforming non-LLM based generation models across several text-to-image tasks that measure context dependence.

----

## [0] MoVie: Visual Model-Based Policy Adaptation for View Generalization

**Authors**: *Sizhe Yang, Yanjie Ze, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html)

**Abstract**:

Visual Reinforcement Learning (RL) agents trained on limited views face significant challenges in generalizing their learned abilities to unseen views. This inherent difficulty is known as the problem of $\textit{view generalization}$. In this work, we systematically categorize this fundamental problem into four distinct and highly challenging scenarios that closely resemble real-world situations. Subsequently, we propose a straightforward yet effective approach to enable successful adaptation of visual $\textbf{Mo}$del-based policies for $\textbf{Vie}$w generalization ($\textbf{MoVie}$) during test time, without any need for explicit reward signals and any modification during training time. Our method demonstrates substantial advancements across all four scenarios encompassing a total of $\textbf{18}$ tasks sourced from DMControl, xArm, and Adroit, with a relative improvement of $\mathbf{33}$%, $\mathbf{86}$%, and $\mathbf{152}$% respectively. The superior results highlight the immense potential of our approach for real-world robotics applications. Code and videos are available at https://yangsizhe.github.io/MoVie/.

----

## [0] Does Visual Pretraining Help End-to-End Reasoning?

**Authors**: *Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, Cordelia Schmid*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html)

**Abstract**:

We aim to investigate whether end-to-end learning of visual reasoning can be achieved with general-purpose neural networks, with the help of visual pretraining. A positive result would refute the common belief that explicit visual abstraction (e.g. object detection) is essential for compositional generalization on visual reasoning, and confirm the feasibility of a neural network ''generalist'' to solve visual recognition and reasoning tasks. We propose a simple and general self-supervised framework which ''compresses'' each video frame into a small set of tokens with a transformer network, and reconstructs the remaining frames based on the compressed temporal context. To minimize the reconstruction loss, the network must learn a compact representation for each image, as well as capture temporal dynamics and object permanence from temporal context. We perform evaluation on two visual reasoning benchmarks, CATER and ACRE. We observe that pretraining is essential to achieve compositional generalization for end-to-end visual reasoning. Our proposed framework outperforms traditional supervised pretraining, including image classification and explicit object detection, by large margins.

----

## [0] Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems

**Authors**: *Lingbing Guo, Weiqing Wang, Zhuo Chen, Ningyu Zhang, Zequn Sun, Yixuan Lai, Qiang Zhang, Huajun Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43e8fd8b9581faa71a6a61602bc28435-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43e8fd8b9581faa71a6a61602bc28435-Abstract-Conference.html)

**Abstract**:

Reasoning system dynamics is one of the most important analytical approaches for many scientific studies. With the initial state of a system as input, the recent graph neural networks (GNNs)-based methods are capable of predicting the future state distant in time with high accuracy. Although these methods have diverse designs in modeling the coordinates and interacting forces of the system, we show that they actually share a common paradigm that learns the integration of the velocity over the interval between the initial and terminal coordinates. However, their integrand is constant w.r.t. time. Inspired by this observation, we propose a new approach to predict the integration based on several velocity estimations with Newton–Cotes formulas and prove its effectiveness theoretically. Extensive experiments on several benchmarks empirically demonstrate consistent and significant improvement compared with the state-of-the-art methods.

----

## [0] Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation

**Authors**: *Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)

**Abstract**:

Program synthesis has been long studied with recent approaches focused on directly using the power of Large Language Models (LLMs) to generate code. Programming benchmarks, with curated synthesis problems and test-cases, are used to measure the performance of various LLMs on code synthesis. However, these test-cases can be limited in both quantity and quality for fully assessing the functional correctness of the generated code. Such limitation in the existing benchmarks begs the following question: In the era of LLMs, is the code generated really correct? To answer this, we propose EvalPlus â€“ a code synthesis evaluation framework to rigorously benchmark the functional correctness of LLM-synthesized code. EvalPlus augments a given evaluation dataset with large amounts of test-cases newly produced by an automatic test input generator, powered by both LLM- and mutation-based strategies. While EvalPlus is general, we extend the test-cases of the popular HumanEval benchmark by 80x to build HumanEval+. Our extensive evaluation across 26 popular LLMs (e.g., GPT-4 and ChatGPT) demonstrates that HumanEval+ is able to catch significant amounts of previously undetected wrong code synthesized by LLMs, reducing the pass@k by up-to 19.3-28.9%. We also surprisingly found that test insufficiency can lead to mis-ranking. For example, both WizardCoder-CodeLlama and Phind-CodeLlama now outperform ChatGPT on HumanEval+, while none of them could on HumanEval. Our work not only indicates that prior popular code synthesis evaluation results do not accurately reflect the true performance of LLMs for code synthesis, but also opens up a new direction to improve such programming benchmarks through automated testing. We have open-sourced our tools, enhanced datasets as well as all LLM-generated code at https://github.com/evalplus/evalplus to facilitate and accelerate future LLM-for-code research.

----

## [0] LeanDojo: Theorem Proving with Retrieval-Augmented Language Models

**Authors**: *Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J. Prenger, Animashree Anandkumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selectionâ€”a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.

----

## [0] Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections

**Authors**: *Talia Konkle, George A. Alvarez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/444b09beab8438d4a58e9bc694dca32a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/444b09beab8438d4a58e9bc694dca32a-Abstract-Conference.html)

**Abstract**:

Given the rich visual information available in each glance, humans can internally direct their visual attention to enhance goal-relevant information---a capacity often absent in standard vision models.  Here we introduce cognitively and biologically-inspired long-range modulatory pathways to enable `cognitive steeringâ€™ in vision models.  First, we show that models equipped with these feedback pathways naturally show improved image recognition, adversarial robustness, and increased brain alignment, relative to baseline models. Further,  these feedback projections from the final layer of the vision backbone provide a meaningful steering interface, where goals can be specified as vectors in the output space.  We show that there are effective ways to steer the model that dramatically improve recognition of categories in composite images of multiple categories, succeeding where baseline feed-forward models without flexible steering fail. And, our multiplicative modulatory motif prevents rampant hallucination of the top-down goal category, dissociating what the model is looking for, from what it is looking at. Thus, these long-range modulatory pathways enable new behavioral capacities for goal-directed visual encoding, offering a flexible communication interface between cognitive and visual systems.

----

## [0] Neuro-symbolic Learning Yielding Logical Constraints

**Authors**: *Zenan Li, Yunpeng Huang, Zhaoyu Li, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, Jian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4459c3c143db74ee52afebdf56836375-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4459c3c143db74ee52afebdf56836375-Abstract-Conference.html)

**Abstract**:

Neuro-symbolic systems combine the abilities of neural perception and logical reasoning. However, end-to-end learning of neuro-symbolic systems is still an unsolved challenge. This paper proposes a natural framework that fuses neural network training, symbol grounding, and logical constraint synthesis into a coherent and efficient end-to-end learning process. The capability of this framework comes from the improved interactions between the neural and the symbolic parts of the system in both the training and inference stages. Technically, to bridge the gap between the continuous neural network and the discrete logical constraint, we introduce a difference-of-convex programming technique to relax the logical constraints while maintaining their precision. We also employ cardinality constraints as the language for logical constraint learning and incorporate a trust region method to avoid the degeneracy of logical constraint in learning. Both theoretical analyses and empirical evaluations substantiate the effectiveness of the proposed framework.

----

## [0] Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models

**Authors**: *Aaron J. Havens, Alexandre Araujo, Siddharth Garg, Farshad Khorrami, Bin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html)

**Abstract**:

Recently, deep equilibrium models (DEQs) have drawn increasing attention from the machine learning community. However, DEQs are much less understood in terms of certified robustness than their explicit network counterparts. In this paper, we advance the understanding of certified robustness of DEQs via exploiting the connections between various Lipschitz network parameterizations for both explicit and implicit models. Importantly, we show that various popular Lipschitz network structures, including convex potential layers (CPL), SDP-based Lipschitz layers (SLL), almost orthogonal layers (AOL), Sandwich layers, and monotone DEQs (MonDEQ) can all be reparameterized as special cases of the Lipschitz-bounded equilibrium networks (LBEN) without changing the prescribed Lipschitz constant in the original network parameterization. A key feature of our reparameterization technique is that it preserves the Lipschitz prescription used in different structures. This opens the possibility of achieving improved certified robustness of DEQs via a combination of network reparameterization, structure-preserving regularization, and LBEN-based fine-tuning. We also support our theoretical understanding with new empirical results, which show that our proposed method improves the certified robust accuracy of DEQs on classification tasks. All codes and experiments are made available at \url{https://github.com/AaronHavens/ExploitingLipschitzDEQ}.

----

## [0] A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings

**Authors**: *Nathaniel Lahn, Sharath Raghvendra, Kaiyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/448444518637da106d978ae7409d9789-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/448444518637da106d978ae7409d9789-Abstract-Conference.html)

**Abstract**:

Optimal Transport is a popular distance metric for measuring similarity between distributions. Exact and approximate combinatorial algorithms for computing the optimal transport distance are hard to parallelize. This has motivated the development of numerical solvers (e.g. Sinkhorn method) that can exploit GPU parallelism and produce approximate solutions. We introduce the first parallel combinatorial algorithm to find an additive $\varepsilon$-approximation of the OT distance. The parallel complexity of our algorithm is $O(\log(n)/ \varepsilon^2)$ where $n$ is the total support size for the input distributions. In Massive Parallel Computation (MPC) frameworks such as Hadoop and MapReduce, our algorithm computes an $\varepsilon$-approximate transport plan in $O(\log (\log (n/\varepsilon))/\varepsilon^2)$ rounds with $O(n/\varepsilon)$ space per machine; all prior algorithms in the MPC framework take $\Omega(\log n)$ rounds. We also provide a GPU-friendly matrix-based interpretation of our algorithm where each step of the algorithm is row or column manipulation of the matrix. Experiments suggest that our combinatorial algorithm is faster than the state-of-the-art approximate solvers in the GPU, especially for higher values of $n$.

----

## [0] RegBN: Batch Normalization of Multimodal Data with Regularization

**Authors**: *Morteza Ghahremani, Christian Wachinger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4488bf8354049b1cd592b6418dc30466-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4488bf8354049b1cd592b6418dc30466-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a surge of interest in integrating high-dimensional data captured by multisource sensors, driven by the impressive success of neural networks in integrating multimodal data. However, the integration of heterogeneous multimodal data poses a significant challenge, as confounding effects and dependencies among such heterogeneous data sources introduce unwanted variability and bias, leading to suboptimal performance of multimodal models. Therefore, it becomes crucial to normalize the low- or high-level features extracted from data modalities before their fusion takes place. This paper introduces RegBN, a novel approach for multimodal Batch Normalization with REGularization. RegBN uses the Frobenius norm as a regularizer term to address the side effects of confounders and underlying dependencies among different data sources. The proposed method generalizes well across multiple modalities and eliminates the need for learnable parameters, simplifying training and inference. We validate the effectiveness of RegBN on eight databases from five research areas, encompassing diverse modalities such as language, audio, image, video, depth, tabular, and 3D MRI. The proposed method demonstrates broad applicability across different architectures such as multilayer perceptrons, convolutional neural networks, and vision transformers, enabling effective normalization of both low- and high-level features in multimodal neural networks. RegBN is available at https://mogvision.github.io/RegBN.

----

## [0] LLM-Pruner: On the Structural Pruning of Large Language Models

**Authors**: *Xinyin Ma, Gongfan Fang, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. The code will be made public.

----

## [0] Nearly Optimal VC-Dimension and Pseudo-Dimension Bounds for Deep Neural Network Derivatives

**Authors**: *Yahong Yang, Haizhao Yang, Yang Xiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/449a016a6ce6fba3fe50d05482abf836-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/449a016a6ce6fba3fe50d05482abf836-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of  nearly optimal Vapnik--Chervonenkis dimension (VC-dimension) and pseudo-dimension estimations of the derivative functions of deep neural networks (DNNs). Two important applications of these estimations include: 1) Establishing a nearly tight approximation result of DNNs in the Sobolev space; 2)  Characterizing the generalization error of machine learning methods with loss functions involving function derivatives. This theoretical investigation fills the gap of learning error estimations for a wide range of physics-informed machine learning models and applications including generative models, solving partial differential equations, operator learning, network compression, distillation, regularization, etc.

----

## [0] ClimateSet: A Large-Scale Climate Model Dataset for Machine Learning

**Authors**: *Julia Kaltenborn, Charlotte E. E. Lange, Venkatesh Ramesh, Philippe Brouillard, Yaniv Gurwicz, Chandni Nagda, Jakob Runge, Peer Nowack, David Rolnick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44a6769fe6c695f8dfb347c649f7c9f0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/44a6769fe6c695f8dfb347c649f7c9f0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Climate models have been key for assessing the impact of climate change and simulating future climate scenarios. The machine learning (ML) community has taken an increased interest in supporting climate scientists’ efforts on various tasks such as climate model emulation, downscaling, and prediction tasks. Many of those tasks have been addressed on datasets created with single climate models. However, both the climate science and ML communities have suggested that to address those tasks at scale, we need large, consistent, and ML-ready climate model datasets. Here, we introduce ClimateSet, a dataset containing the inputs and outputs of 36 climate models from the Input4MIPs and CMIP6 archives. In addition, we provide a modular dataset pipeline for retrieving and preprocessing additional climate models and scenarios. We showcase the potential of our dataset by using it as a benchmark for ML-based climate model emulation. We gain new insights about the performance and generalization capabilities of the different ML models by analyzing their performance across different climate models. Furthermore, the dataset can be used to train an ML emulator on several climate models instead of just one. Such a “super emulator” can quickly project new climate change scenarios, complementing existing scenarios already provided to policymakers. We believe ClimateSet will create the basis needed for the ML community to tackle climate-related tasks at scale.

----

## [0] Near-Optimal Bounds for Learning Gaussian Halfspaces with Random Classification Noise

**Authors**: *Ilias Diakonikolas, Jelena Diakonikolas, Daniel Kane, Puqian Wang, Nikos Zarifis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44c150733f9c5b6f98cb0caad0c664c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44c150733f9c5b6f98cb0caad0c664c7-Abstract-Conference.html)

**Abstract**:

We study the problem of learning general (i.e., not necessarily homogeneous) halfspaces with Random Classification Noise under the Gaussian distribution. We establish nearly-matching algorithmic and Statistical Query (SQ) lower bound results revealing a surprising information-computation gap for this basic problem. Specifically, the sample complexity of this learning problem is $\widetilde{\Theta}(d/\epsilon)$, where $d$ is the dimension and $\epsilon$ is the excess error. Our positive result is a computationally efficient learning algorithm with sample complexity$\tilde{O}(d/\epsilon + d/\max(p, \epsilon))^2)$, where $p$ quantifies the bias of the target halfspace. On the lower bound side, we show that any efficient SQ algorithm (or low-degree test)for the problem requires sample complexity at least $\Omega(d^{1/2}/(\max(p, \epsilon))^2)$. Our lower bound suggests that this quadratic dependence on $1/\epsilon$ is inherent for efficient algorithms.

----

## [0] Explain Any Concept: Segment Anything Meets Concept-Based Explanation

**Authors**: *Ao Sun, Pingchuan Ma, Yuanyuan Yuan, Shuai Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html)

**Abstract**:

EXplainable AI (XAI) is an essential topic to improve human understanding of deep neural networks (DNNs) given their black-box internals. For computer vision tasks, mainstream pixel-based XAI methods explain DNN decisions by identifying important pixels, and emerging concept-based XAI explore forming explanations with concepts (e.g., a head in an image). However, pixels are generally hard to interpret and sensitive to the imprecision of XAI methods, whereas “concepts” in prior works require human annotation or are limited to pre-defined concept sets. On the other hand, driven by large-scale pre-training, Segment Anything Model (SAM) has been demonstrated as a powerful and promotable framework for performing precise and comprehensive instance segmentation, enabling automatic preparation of concept sets from a given image. This paper for the first time explores using SAM to augment concept-based XAI. We offer an effective and flexible concept-based explanation method, namely Explain Any Concept (EAC), which explains DNN decisions with any concept. While SAM is highly effective and offers an “out-of-the-box” instance segmentation, it is costly when being integrated into defacto XAI pipelines. We thus propose a lightweight per-input equivalent (PIE) scheme, enabling efficient explanation with a surrogate model. Our evaluation  over two popular datasets (ImageNet and COCO) illustrate the highly encouraging performance of EAC over commonly-used XAI methods.

----

## [0] Data-Driven Network Neuroscience: On Data Collection and Benchmark

**Authors**: *Jiaxing Xu, Yunhan Yang, David Tse Jung Huang, Sophi Shilpa Gururajapathy, Yiping Ke, Miao Qiao, Alan Wang, Haribalan Kumar, Josh McGeown, Eryn Kwon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44e3a3115ca26e5127851acd0cedd0d9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/44e3a3115ca26e5127851acd0cedd0d9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper presents a comprehensive and quality collection of functional human brain network data for potential research in the intersection of neuroscience, machine learning, and graph analytics. Anatomical and functional MRI images have been used to understand the functional connectivity of the human brain and are particularly important in identifying underlying neurodegenerative conditions such as Alzheimer's, Parkinson's, and Autism. Recently, the study of the brain in the form of brain networks using machine learning and graph analytics has become increasingly popular, especially to predict the early onset of these conditions. A brain network, represented as a graph, retains rich structural and positional information that traditional examination methods are unable to capture. However, the lack of publicly accessible brain network data prevents researchers from data-driven explorations. One of the main difficulties lies in the complicated domain-specific preprocessing steps and the exhaustive computation required to convert the data from MRI images into brain networks. We bridge this gap by collecting a large amount of MRI images from public databases and a private source, working with domain experts to make sensible design choices, and preprocessing the MRI images to produce a collection of brain network datasets. The datasets originate from 6 different sources, cover 4 brain conditions, and consist of a total of 2,702 subjects. We test our graph datasets on 12 machine learning models to provide baselines and validate the data quality on a recent graph analysis model. To lower the barrier to entry and promote the research in this interdisciplinary field, we release our brain network data and complete preprocessing details including codes at https://doi.org/10.17608/k6.auckland.21397377 and https://github.com/brainnetuoa/datadrivennetwork_neuroscience.

----

## [0] No-Regret Learning with Unbounded Losses: The Case of Logarithmic Pooling

**Authors**: *Eric Neyman, Tim Roughgarden*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44ecfb60950e868a13172b935b7964a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44ecfb60950e868a13172b935b7964a9-Abstract-Conference.html)

**Abstract**:

For each of $T$ time steps, $m$ experts report probability distributions over $n$ outcomes; we wish to learn to aggregate these forecasts in a way that attains a no-regret guarantee. We focus on the fundamental and practical aggregation method known as *logarithmic pooling* -- a weighted average of log odds -- which is in a certain sense the optimal choice of pooling method if one is interested in minimizing log loss (as we take to be our loss function). We consider the problem of learning the best set of parameters (i.e. expert weights) in an online adversarial setting.  We assume (by necessity) that the adversarial choices of outcomes and forecasts are consistent, in the sense that experts report calibrated forecasts. Imposing this constraint creates a (to our knowledge) novel semi-adversarial setting in which the adversary retains a large amount of flexibility. In this setting, we present an algorithm based on online mirror descent that learns expert weights in a way that attains $O(\sqrt{T} \log T)$ expected regret as compared with the best weights in hindsight.

----

## [0] PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation

**Authors**: *Jialu Li, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html)

**Abstract**:

Vision-and-Language Navigation requires the agent to follow language instructions to navigate through 3D environments. One main challenge in Vision-and-Language Navigation is the limited availability of photorealistic training environments, which makes it hard to generalize to new and unseen environments. To address this problem, we propose PanoGen, a generation method that can potentially create an infinite number of diverse panoramic environments conditioned on text. Specifically, we collect room descriptions by captioning the room images in existing Matterport3D environments, and leverage a state-of-the-art text-to-image diffusion model to generate the new panoramic environments. We use recursive outpainting over the generated images to create consistent 360-degree panorama views. Our new panoramic environments share similar semantic information with the original environments by conditioning on text descriptions, which ensures the co-occurrence of objects in the panorama follows human intuition, and creates enough diversity in room appearance and layout with image outpainting. Lastly, we explore two ways of utilizing PanoGen in VLN pre-training and fine-tuning. We generate instructions for paths in our PanoGen environments with a speaker built on a pre-trained vision-and-language model for VLN pre-training, and augment the visual observation with our panoramic environments during agents' fine-tuning to avoid overfitting to seen environments. Empirically, learning with our PanoGen environments achieves the new state-of-the-art on the Room-to-Room, Room-for-Room, and CVDN datasets. Besides, we find that pre-training with our PanoGen speaker data is especially effective for CVDN, which has under-specified instructions and needs commonsense knowledge to reach the target. Lastly, we show that the agent can benefit from training with more generated panoramic environments, suggesting promising results for scaling up the PanoGen environments to enhance agents' generalization to unseen environments.

----

## [0] Scaling laws for language encoding models in fMRI

**Authors**: *Richard Antonello, Aditya R. Vaidya, Alexander Huth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4533e4a352440a32558c1c227602c323-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4533e4a352440a32558c1c227602c323-Abstract-Conference.html)

**Abstract**:

Representations from transformer-based unidirectional language models are known to be effective at predicting brain responses to natural language. However, most studies comparing language models to brains have used GPT-2 or similarly sized language models. Here we tested whether larger open-source models such as those from the OPT and LLaMA families are better at predicting brain responses recorded using fMRI. Mirroring scaling results from other contexts, we found that brain prediction performance scales logarithmically with model size from 125M to 30B parameter models, with ~15% increased encoding performance as measured by correlation with a held-out test set across 3 subjects. Similar log-linear behavior was observed when scaling the size of the fMRI training set. We also characterized scaling for acoustic encoding models that use HuBERT, WavLM, and Whisper, and we found comparable improvements with model size. A noise ceiling analysis of these large, high-performance encoding models showed that performance is nearing the theoretical maximum for brain areas such as the precuneus and higher auditory cortex. These results suggest that increasing scale in both models and data will yield incredibly effective models of language processing in the brain, enabling better scientific understanding as well as applications such as decoding.

----

## [0] Optimal Rates for Bandit Nonstochastic Control

**Authors**: *Y. Jennifer Sun, Stephen H. Newman, Elad Hazan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45591d6727f0e127295f8d16adba6b23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45591d6727f0e127295f8d16adba6b23-Abstract-Conference.html)

**Abstract**:

Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) control are foundational and extensively researched problems in optimal control. We investigate LQR and LQG problems with semi-adversarial perturbations and time-varying adversarial bandit loss functions. The best-known sublinear regret algorithm~\cite{gradu2020non} has a $T^{\frac{3}{4}}$ time horizon dependence, and its authors posed an open question about whether a tight rate of $\sqrt{T}$ could be achieved. We answer in the affirmative, giving an algorithm for bandit LQR and LQG which attains optimal regret, up to logarithmic factors. A central component of our method is a new scheme for bandit convex optimization with memory, which is of independent interest.

----

## [0] Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection

**Authors**: *Yuxin Cao, Yian Li, Yumeng Zhu, Derui Wang, Minhui Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/456f9445d0fa1a932d19584ab788c787-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/456f9445d0fa1a932d19584ab788c787-Abstract-Conference.html)

**Abstract**:

Anti-spoofing detection has become a necessity for face recognition systems due to the security threat posed by spoofing attacks. Despite great success in traditional attacks, most deep-learning-based methods perform poorly in 3D masks, which can highly simulate real faces in appearance and structure, suffering generalizability insufficiency while focusing only on the spatial domain with single frame input. This has been mitigated by the recent introduction of a biomedical technology called rPPG (remote photoplethysmography). However, rPPG-based methods are sensitive to noisy interference and require at least one second (> 25 frames) of observation time, which induces high computational overhead. To address these challenges, we propose a novel 3D mask detection framework, called FASTEN (Flow-Attention-based Spatio-Temporal aggrEgation Network). We tailor the network for focusing more on fine-grained details in large movements, which can eliminate redundant spatio-temporal feature interference and quickly capture splicing traces of 3D masks in fewer frames. Our proposed network contains three key modules: 1) a facial optical flow network to obtain non-RGB inter-frame flow information; 2) flow attention to assign different significance to each frame; 3) spatio-temporal aggregation to aggregate high-level spatial features and temporal transition features. Through extensive experiments, FASTEN only requires five frames of input and outperforms eight competitors for both intra-dataset and cross-dataset evaluations in terms of multiple detection metrics. Moreover, FASTEN has been deployed in real-world mobile devices for practical 3D mask detection.

----

## [0] On the Last-iterate Convergence in Time-varying Zero-sum Games: Extra Gradient Succeeds where Optimism Fails

**Authors**: *Yi Feng, Hu Fu, Qun Hu, Ping Li, Ioannis Panageas, Bo Peng, Xiao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/457ab261562014550e53351422f69834-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/457ab261562014550e53351422f69834-Abstract-Conference.html)

**Abstract**:

Last-iterate convergence has received extensive study in two player zero-sum games starting from bilinear, convex-concave up to settings that satisfy the MVI condition. Typical methods that exhibit last-iterate convergence for the aforementioned games include extra-gradient (EG) and optimistic gradient descent ascent (OGDA). However, all the established last-iterate convergence results hold for the restrictive setting where the underlying repeated game does not change over time.Recently, a line of research has focused on regret analysis of OGDA  in time-varying games, i.e., games where payoffs evolve with time; the last-iterate behavior of OGDA and EG in time-varying environments remains unclear though. In this paper, we study the last-iterate behavior of various algorithms in two types of unconstrained, time-varying, bilinear zero-sum games: periodic and convergent perturbed games. These models expand upon the usual repeated game formulation and incorporate external environmental factors, such as the seasonal effects on species competition and vanishing external noise. In periodic games, we prove that EG will converge while OGDA and momentum method will diverge. This is quite surprising, as to the best of our knowledge, it is the first result that indicates EG and OGDA have qualitatively different last-iterate behaviors and do not exhibit similar behavior. In convergent perturbed games, we prove all these algorithms converge as long as the game itself stabilizes with a faster rate than $1/t$.

----

## [0] Taking the neural sampling code very seriously: A data-driven approach for evaluating generative models of the visual system

**Authors**: *Suhas Shrinivasan, Konstantin-Klemens Lurz, Kelli Restivo, George H. Denfield, Andreas S. Tolias, Edgar Y. Walker, Fabian H. Sinz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/458d9f2dd5c7565af60143630dc62f10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/458d9f2dd5c7565af60143630dc62f10-Abstract-Conference.html)

**Abstract**:

Prevailing theories of perception hypothesize that the brain implements perception via Bayesian inference in a generative model of the world.One prominent theory, the Neural Sampling Code (NSC), posits that neuronal responses to a stimulus represent samples from the posterior distribution over latent world state variables that cause the stimulus.Although theoretically elegant, NSC does not specify the exact form of the generative model or prescribe how to link the theory to recorded neuronal activity.Previous works assume simple generative models and test their qualitative agreement with neurophysiological data.Currently, there is no precise alignment of the normative theory with neuronal recordings, especially in response to natural stimuli, and a quantitative, experimental evaluation of models under NSC has been lacking.Here, we propose a novel formalization of NSC, that (a) allows us to directly fit NSC generative models to recorded neuronal activity in response to natural images, (b) formulate richer and more flexible generative models, and (c) employ standard metrics to quantitatively evaluate different generative models under NSC.Furthermore, we derive a stimulus-conditioned predictive model of neuronal responses from the trained generative model using our formalization that we compare to neural system identification models.We demonstrate our approach by fitting and comparing classical- and flexible deep learning-based generative models on population recordings from the macaque primary visual cortex (V1) to natural images, and show that the flexible models outperform classical models in both their generative- and predictive-model performance.Overall, our work is an important step towards a quantitative evaluation of NSC. It provides a framework that lets us \textit{learn} the generative model directly from neuronal population recordings, paving the way for an experimentally-informed understanding of probabilistic computational principles underlying perception and behavior.

----

## [0] Can semi-supervised learning use all the data effectively? A lower bound perspective

**Authors**: *Alexandru Tifrea, Gizem Yüce, Amartya Sanyal, Fanny Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/458fa8ee331566383d8e74bdb647f829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/458fa8ee331566383d8e74bdb647f829-Abstract-Conference.html)

**Abstract**:

Prior theoretical and empirical works have established that semi-supervised learning algorithms can leverage the unlabeled data to improve over the labeled sample complexity of supervised learning (SL) algorithms. However, existing theoretical work focuses on regimes where the unlabeled data is sufficient to learn a good decision boundary using unsupervised learning (UL) alone. This begs the question: Can SSL algorithms simultaneously improve upon both UL and SL? To this end, we derive a tight lower bound for 2-Gaussian mixture models that explicitly depends on the labeled and the unlabeled dataset size as well as the signal-to-noise ratio of the mixture distribution. Surprisingly, our result implies that no SSL algorithm improves upon the minimax-optimal statistical error rates of SL or UL algorithms for these distributions. Nevertheless, in our real-world experiments, SSL algorithms can often outperform UL and SL algorithms. In summary, our work suggests that while it is possible to prove the performance gains of SSL algorithms, this would require careful tracking of constants in the theoretical analysis.

----

## [0] Evolving Standardization for Continual Domain Generalization over Temporal Drift

**Authors**: *Mixue Xie, Shuang Li, Longhui Yuan, Chi Harold Liu, Zehui Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/459a911eb49cd2e0192055ee156d04e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/459a911eb49cd2e0192055ee156d04e5-Abstract-Conference.html)

**Abstract**:

The capability of generalizing to out-of-distribution data is crucial for the deployment of machine learning models in the real world. Existing domain generalization (DG) mainly embarks on offline and discrete scenarios, where multiple source domains are simultaneously accessible and the distribution shift among domains is abrupt and violent. Nevertheless, such setting may not be universally applicable to all real-world applications, as there are cases where the data distribution gradually changes over time due to various factors, e.g., the process of aging. Additionally, as the domain constantly evolves, new domains will continually emerge. Re-training and updating models with both new and previous domains using existing DG methods can be resource-intensive and inefficient. Therefore, in this paper, we present a problem formulation for Continual Domain Generalization over Temporal Drift (CDGTD). CDGTD addresses the challenge of gradually shifting data distributions over time, where domains arrive sequentially and models can only access the data of the current domain. The goal is to generalize to unseen domains that are not too far into the future. To this end, we propose an Evolving Standardization (EvoS) method, which characterizes the evolving pattern of feature distribution and mitigates the distribution shift by standardizing features with generated statistics of corresponding domain. Specifically, inspired by the powerful ability of transformers to model sequence relations, we design a multi-scale attention module (MSAM) to learn the evolving pattern under sliding time windows of different lengths. MSAM can generate statistics of current domain based on the statistics of previous domains and the learned evolving pattern. Experiments on multiple real-world datasets including images and texts validate the efficacy of our EvoS.

----

## [0] Learning the Efficient Frontier

**Authors**: *Philippe Chatigny, Ivan Sergienko, Ryan Ferguson, Jordan Weir, Maxime Bergeron*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html)

**Abstract**:

The efficient frontier (EF) is a fundamental resource allocation problem where one has to find an optimal portfolio maximizing a reward at a given level of risk. This optimal solution is traditionally found by solving a convex optimization problem. In this paper, we introduce NeuralEF: a fast neural approximation framework that robustly forecasts the result of the EF convex optimizations problems with respect to heterogeneous linear constraints and variable number of optimization inputs. By reformulating an optimization problem as a sequence to sequence problem, we show that NeuralEF is a viable solution to accelerate large-scale simulation while handling discontinuous behavior.

----

## [0] Dissecting Chain-of-Thought: Compositionality through In-Context Filtering and Learning

**Authors**: *Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, Samet Oymak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html)

**Abstract**:

Chain-of-thought (CoT) is a method that enables language models to handle complex reasoning tasks by decomposing them into simpler steps. Despite its success, the underlying mechanics of CoT are not yet fully understood. In an attempt to shed light on this, our study investigates the impact of CoT on the ability of transformers to in-context learn a simple to study, yet general family of compositional functions: multi-layer perceptrons (MLPs). In this setting, we find that the success of CoT can be attributed to breaking down in-context learning of a compositional function into two distinct phases: focusing on and filtering data related to each step of the composition and in-context learning the single-step composition function. Through both experimental and theoretical evidence, we demonstrate how CoT significantly reduces the sample complexity of in-context learning (ICL) and facilitates the learning of complex functions that non-CoT methods struggle with. Furthermore, we illustrate how transformers can transition from vanilla in-context learning to mastering a compositional function with CoT by simply incorporating additional layers that perform the necessary data-filtering for CoT via the attention mechanism. In addition to these test-time benefits, we show CoT helps accelerate pretraining by learning shortcuts to represent complex functions and filtering plays an important role in this process. These findings collectively provide insights into the mechanics of CoT, inviting further investigation of its role in complex reasoning tasks.

----

## [0] Improving multimodal datasets with image captioning

**Authors**: *Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45e604a3e33d10fba508e755faa72345-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/45e604a3e33d10fba508e755faa72345-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Massive web datasets play a key role in the success of large vision-language models like CLIP and Flamingo. However, the raw web data is noisy, and existing filtering methods to reduce noise often come at the expense of data diversity. Our work focuses on caption quality as one major source of noise, and studies how generated captions can increase the utility of web-scraped datapoints with nondescript text. Through exploring different mixing strategies for raw and generated captions, we outperform the best filtering method proposed by the DataComp benchmark by 2% on ImageNet and 4% on average across 38 tasks, given a candidate pool of 128M image-text pairs. Our best approach is also 2x better at Flickr and MS-COCO retrieval. We then analyze what makes synthetic captions an effective source of text supervision. In experimenting with different image captioning models, we also demonstrate that the performance of a model on standard image captioning benchmarks (e.g., NoCaps CIDEr) is not a reliable indicator of the utility of the captions it generates for multimodal training. Finally, our experiments with using generated captions at DataComp's large scale (1.28B image-text pairs) offer insights into the limitations of synthetic text, as well as the importance of image curation with increasing training data quantity. The synthetic captions used in our experiments are now available on HuggingFace.

----

## [0] ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation

**Authors**: *Sungduk Yu, Walter M. Hannah, Liran Peng, Jerry Lin, Mohamed Aziz Bhouri, Ritwik Gupta, Björn Lütjens, Justus C. Will, Gunnar Behrens, Julius Busecke, Nora Loose, Charles Stern, Tom Beucler, Bryce E. Harrop, Benjamin R. Hillman, Andrea M. Jenney, Savannah L. Ferretti, Nana Liu, Animashree Anandkumar, Noah D. Brenowitz, Veronika Eyring, Nicholas Geneva, Pierre Gentine, Stephan Mandt, Jaideep Pathak, Akshay Subramaniam, Carl Vondrick, Rose Yu, Laure Zanna, Tian Zheng, Ryan Abernathey, Fiaz Ahmed, David C. Bader, Pierre Baldi, Elizabeth A. Barnes, Christopher S. Bretherton, Peter M. Caldwell, Wayne Chuang, Yilun Han, Yu Huang, Fernando Iglesias-Suarez, Sanket R. Jantre, Karthik Kashinath, Marat Khairoutdinov, Thorsten Kurth, Nicholas J. Lutsko, Po-Lun Ma, Griffin Mooers, J. David Neelin, David A. Randall, Sara Shamekh, Mark Taylor, Nathan M. Urban, Janni Yuval, Guang Zhang, Mike Pritchard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45fbcc01349292f5e059a0b8b02c8c3f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/45fbcc01349292f5e059a0b8b02c8c3f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modern climate projections lack adequate spatial and temporal resolution due to computational constraints. A consequence is inaccurate and imprecise predictions of critical processes such as storms. Hybrid methods that combine physics with machine learning (ML) have introduced a new generation of higher fidelity climate simulators that can sidestep Moore's Law by outsourcing compute-hungry, short, high-resolution simulations to ML emulators. However, this hybrid ML-physics simulation approach requires domain-specific treatment and has been inaccessible to ML experts because of lack of training data and relevant, easy-to-use workflows. We present ClimSim, the largest-ever dataset designed for hybrid ML-physics research. It comprises multi-scale climate simulations, developed by a consortium of climate scientists and ML researchers. It consists of 5.7 billion pairs of multivariate input and output vectors that isolate the influence of locally-nested, high-resolution, high-fidelity physics on a host climate simulator's macro-scale physical state.The dataset is global in coverage, spans multiple years at high sampling frequency, and is designed such that resulting emulators are compatible with downstream coupling into operational climate simulators. We implement a range of deterministic and stochastic regression baselines to highlight the ML challenges and their scoring. The data (https://huggingface.co/datasets/LEAP/ClimSim_high-res) and code (https://leap-stc.github.io/ClimSim) are released openly to support the development of hybrid ML-physics and high-fidelity climate simulations for the benefit of science and society.

----

## [0] Relative Entropic Optimal Transport: a (Prior-aware) Matching Perspective to (Unbalanced) Classification

**Authors**: *Liangliang Shi, Haoyu Zhen, Gu Zhang, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4621451c25a7aa175dc00e5dd4a243a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4621451c25a7aa175dc00e5dd4a243a3-Abstract-Conference.html)

**Abstract**:

Classification is a fundamental problem in machine learning, and considerable efforts have been recently devoted to the demanding long-tailed setting due to its prevalence in nature. Departure from the Bayesian framework, this paper rethinks classification from a matching perspective by studying the matching probability between samples and labels with optimal transport (OT) formulation. Specifically, we first propose a new variant of optimal transport, called Relative Entropic Optimal Transport (RE-OT), which guides the coupling solution to a known prior information matrix. We gives some theoretical results and their proof for RE-OT and surprisingly find RE-OT can help to deblur for barycenter images. Then we adopt inverse RE-OT for training long-tailed data and find that the loss derived from RE-OT has a similar form to Softmax-based cross-entropy loss, indicating a close connection between optimal transport and classification and the potential for transferring concepts between these two academic fields, such as barycentric projection in OT, which can map the labels back to the feature space. We further derive an epoch-varying RE-OT loss, and do the experiments on unbalanced image classification,  molecule classification, instance segmentation and representation learning. Experimental results show its effectiveness.

----

## [0] Connecting Multi-modal Contrastive Representations

**Authors**: *Zehan Wang, Yang Zhao, Xize Cheng, Haifeng Huang, Jiageng Liu, Aoxiong Yin, Li Tang, Linjun Li, Yongqi Wang, Ziang Zhang, Zhou Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46362971bfc3a97e6a271f2eb90fba17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46362971bfc3a97e6a271f2eb90fba17-Abstract-Conference.html)

**Abstract**:

Multi-modal Contrastive Representation (MCR) learning aims to encode different modalities into a semantically aligned shared space. This paradigm shows remarkable generalization ability on numerous downstream tasks across various modalities. However, the reliance on massive high-quality data pairs limits its further development on more modalities. This paper proposes a novel training-efficient method for learning MCR without paired data called Connecting Multi-modal Contrastive Representations (C-MCR). Specifically, given two existing MCRs pre-trained on $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ modality pairs, we project them to a new space and use the data from the overlapping modality $\mathcal{B}$ to aligning the two MCRs in the new space. Meanwhile, since the modality pairs $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ are already aligned within each MCR, the connection learned by overlapping modality can also be transferred to non-overlapping modality pair $(\mathcal{A}$, $\mathcal{C})$. To unleash the potential of C-MCR, we further introduce a semantic-enhanced inter- and intra-MCR connection method. We first enhance the semantic consistency and completion of embeddings across different modalities for more robust alignment. Then we utilize the inter-MCR alignment to establish the connection, and employ the intra-MCR alignment to better maintain the connection for inputs from non-overlapping modalities. To demonstrate the effectiveness of C-MCR, we take the field of audio-visual and 3D-language learning as examples. Specifically, we connect CLIP and CLAP via texts to derive audio-visual representations, and integrate CLIP and ULIP via images for 3D-language representations. Remarkably, without using any paired data, C-MCR for audio-visual achieves state-of-the-art performance on audio-image retrieval, audio-visual source localization, and counterfactual audio-image recognition tasks. Furthermore, C-MCR for 3D-language also attains advanced zero-shot 3D point cloud classification accuracy on ModelNet40. Our project page is available at \url{https://c-mcr.github.io/C-MCR/}

----

## [0] Boosting Learning for LDPC Codes to Improve the Error-Floor Performance

**Authors**: *Heeyoul Kwak, Daeyoung Yun, Yongjune Kim, Sang-Hyo Kim, Jong-Seon No*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html)

**Abstract**:

Low-density parity-check (LDPC) codes have been successfully commercialized in communication systems due to their strong error correction capabilities and simple decoding process. However, the error-floor phenomenon of LDPC codes, in which the error rate stops decreasing rapidly at a certain level, presents challenges for achieving extremely low error rates and deploying LDPC codes in scenarios demanding ultra-high reliability. In this work, we propose training methods for neural min-sum (NMS) decoders to eliminate the error-floor effect. First, by leveraging the boosting learning technique of ensemble networks, we divide the decoding network into two neural decoders and train the post decoder to be specialized for uncorrected words that the first decoder fails to correct. Secondly, to address the vanishing gradient issue in training, we introduce a block-wise training schedule that locally trains a block of weights while retraining the preceding block. Lastly, we show that assigning different weights to unsatisfied check nodes effectively lowers the error-floor with a minimal number of weights. By applying these training methods to standard LDPC codes, we achieve the best error-floor performance compared to other decoding methods. The proposed NMS decoder, optimized solely through novel training methods without additional modules, can be integrated into existing LDPC decoders without incurring extra hardware costs. The source code is available at https://github.com/ghy1228/LDPCErrorFloor.

----

## [0] Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping

**Authors**: *Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html)

**Abstract**:

The use of anthropomorphic robotic hands for assisting individuals in situations where human hands may be unavailable or unsuitable has gained significant importance. In this paper, we propose a novel task called human-assisting dexterous grasping that aims to train a policy for controlling a robotic hand's fingers to assist users in grasping objects. Unlike conventional dexterous grasping, this task presents a more complex challenge as the policy needs to adapt to diverse user intentions, in addition to the object's geometry.  We address this challenge by proposing an approach consisting of two sub-modules: a hand-object-conditional grasping primitive called Grasping Gradient Field (GraspGF), and a history-conditional residual policy.  GraspGF learns 'how' to grasp by estimating the gradient of a synthesised success grasping example set, while the residual policy determines 'when' and at what speed the grasping action should be executed based on the trajectory history. Experimental results demonstrate the superiority of our proposed method compared to baselines, highlighting the user-awareness and practicality in real-world applications. The codes and demonstrations can be viewed at https://sites.google.com/view/graspgf.

----

## [0] Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration

**Authors**: *Zhihan Liu, Miao Lu, Wei Xiong, Han Zhong, Hao Hu, Shenao Zhang, Sirui Zheng, Zhuoran Yang, Zhaoran Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4640d5da5888238b9de7e0dbacd2c605-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4640d5da5888238b9de7e0dbacd2c605-Abstract-Conference.html)

**Abstract**:

In reinforcement learning (RL), balancing exploration and exploitation is crucial for achieving an optimal policy in a sample-efficient way. To this end, existing sample- efficient algorithms typically consist of three components: estimation, planning, and exploration. However, to cope with general function approximators, most of them involve impractical algorithmic components to incentivize exploration, such as data-dependent level-set constraints or complicated sampling procedures. To address this challenge, we propose an easy-to-implement RL framework called Maximize to Explore (MEX), which only needs to optimize unconstrainedly a single objective that integrates the estimation and planning components while balancing exploration and exploitation automatically. Theoretically, we prove that the MEX achieves a sublinear regret with general function approximators and is extendable to the zero-sum Markov game setting. Meanwhile, we adapt deep RL baselines to design practical versions of MEX in both the model-based and model-free settings, which outperform baselines in various MuJoCo environments with sparse reward by a stable margin. Compared with existing sample-efficient algorithms with general function approximators, MEX achieves similar sample efficiency while also enjoying a lower computational cost and is more compatible with modern deep RL methods.

----

## [0] Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks

**Authors**: *Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Liu Linc, Yang Feng, Lin Lai, Hongyang Qin, Minwen Deng, Juchao Zhuo, Deheng Ye, Qiang Fu, Yang Guang, Wei Yang, Lanxiao Huang, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/464fefa022aaefc85d901317bbf13f85-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/464fefa022aaefc85d901317bbf13f85-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The advancement of Offline Reinforcement Learning (RL) and Offline Multi-Agent Reinforcement Learning (MARL) critically depends on the availability of high-quality, pre-collected offline datasets that represent real-world complexities and practical applications. However, existing datasets often fall short in their simplicity and lack of realism. To address this gap, we propose Hokoff, a comprehensive set of pre-collected datasets that covers both offline RL and offline MARL, accompanied by a robust framework, to facilitate further research. This data is derived from Honor of Kings, a recognized Multiplayer Online Battle Arena (MOBA) game known for its intricate nature, closely resembling real-life situations. Utilizing this framework, we benchmark a variety of offline RL and offline MARL algorithms. We also introduce a novel baseline algorithm tailored for the inherent hierarchical action space of the game. We reveal the incompetency of current offline RL approaches in handling task complexity, generalization and multi-task learning.

----

## [0] Learning and Collusion in Multi-unit Auctions

**Authors**: *Simina Brânzei, Mahsa Derakhshan, Negin Golrezaei, Yanjun Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4661b55200c03a8c4bb9c2974b4fb12d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4661b55200c03a8c4bb9c2974b4fb12d-Abstract-Conference.html)

**Abstract**:

In a carbon auction, licenses for CO2 emissions are allocated among multiple interested players. Inspired by this setting, we consider repeated multi-unit auctions with uniform pricing, which are widely used in practice. Our contribution is to analyze these auctions in both the offline and online settings, by designing efficient bidding algorithms with low regret and giving regret lower bounds. We also analyze the quality of the equilibria  in  two  main variants of the auction, finding that one  variant is susceptible to collusion among the bidders while the other is not.

----

## [0] One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization

**Authors**: *Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html)

**Abstract**:

Single image 3D reconstruction is an important but challenging task that requires extensive knowledge of our natural world. Many existing methods solve this problem by optimizing a neural radiance field under the guidance of 2D diffusion models but suffer from lengthy optimization time, 3D inconsistency results, and poor geometry. In this work, we propose a novel method that takes a single image of any object as input and generates a full 360-degree 3D textured mesh in a single feed-forward pass. Given a single image, we first use a view-conditioned 2D diffusion model, Zero123, to generate multi-view images for the input view, and then aim to lift them up to 3D space. Since traditional reconstruction methods struggle with inconsistent multi-view predictions, we build our 3D reconstruction module upon an SDF-based generalizable neural surface reconstruction method and propose several critical training strategies to enable the reconstruction of 360-degree meshes. Without costly optimizations, our method reconstructs 3D shapes in significantly less time than existing methods. Moreover, our method favors better geometry, generates more 3D consistent results, and adheres more closely to the input image. We evaluate our approach on both synthetic data and in-the-wild images and demonstrate its superiority in terms of both mesh quality and runtime. In addition, our approach can seamlessly support the text-to-3D task by integrating with off-the-shelf text-to-image diffusion models.

----

## [0] VeriX: Towards Verified Explainability of Deep Neural Networks

**Authors**: *Min Wu, Haoze Wu, Clark W. Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46907c2ff9fafd618095161d76461842-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46907c2ff9fafd618095161d76461842-Abstract-Conference.html)

**Abstract**:

We present VeriX (Verified eXplainability), a system for producing optimal robust explanations and generating counterfactuals along decision boundaries of machine learning models. We build such explanations and counterfactuals iteratively using constraint solving techniques and a heuristic based on feature-level sensitivity ranking. We evaluate our method on image recognition benchmarks and a real-world scenario of autonomous aircraft taxiing.

----

## [0] Generalized test utilities for long-tail performance in extreme multi-label classification

**Authors**: *Erik Schultheis, Marek Wydmuch, Wojciech Kotlowski, Rohit Babbar, Krzysztof Dembczynski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46994b3d6dd0fd5fca5f780af6259db5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46994b3d6dd0fd5fca5f780af6259db5-Abstract-Conference.html)

**Abstract**:

Extreme multi-label classification (XMLC) is the task of selecting a small subset of relevant labels from a very large set of possible labels. As such, it is characterized by long-tail labels, i.e., most labels have very few positive instances. With standard performance measures such as precision@k, a classifier can ignore tail labels and still report good performance. However, it is often argued that correct predictions in the tail are more "interesting" or "rewarding," but the community has not yet settled on a metric capturing this intuitive concept. The existing propensity-scored metrics fall short on this goal by confounding the problems of long-tail and missing labels. In this paper, we analyze generalized metrics budgeted "at k" as an alternative solution. To tackle the challenging problem of optimizing these metrics, we formulate it in the expected test utility (ETU) framework, which aims to optimize the expected performance on a given test set. We derive optimal prediction rules and construct their computationally efficient approximations with provable regret guarantees and being robust against model misspecification. Our algorithm, based on block coordinate descent, scales effortlessly to XMLC problems and obtains promising results in terms of long-tail performance.

----

## [0] Compositional Foundation Models for Hierarchical Planning

**Authors**: *Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi S. Jaakkola, Joshua B. Tenenbaum, Leslie Pack Kaelbling, Akash Srivastava, Pulkit Agrawal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html)

**Abstract**:

To make effective decisions in novel environments with long-horizon goals, it is crucial to engage in hierarchical reasoning across spatial and temporal scales. This entails planning abstract subgoal sequences, visually reasoning about the underlying plans, and executing actions in accordance with the devised plan through visual-motor control. We propose Compositional Foundation Models for Hierarchical Planning (HiP), a foundation model which leverages multiple expert foundation model trained on language, vision and action data individually jointly together to solve long-horizon tasks. We use a large language model to construct symbolic plans that are grounded in the environment through a large video diffusion model. Generated video plans are then grounded to visual-motor control, through an inverse dynamics model that infers actions from generated videos. To enable effective reasoning within this hierarchy, we enforce consistency between the models via iterative refinement. We illustrate the efficacy and adaptability of our approach in three different long-horizon table-top manipulation tasks.

----

## [0] Diffusion Model for Graph Inverse Problems: Towards Effective Source Localization on Complex Networks

**Authors**: *Xin Yan, Hui Fang, Qiang He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html)

**Abstract**:

Information diffusion problems, such as the spread of epidemics or rumors, are widespread in society. The inverse problems of graph diffusion, which involve locating the sources and identifying the paths of diffusion based on currently observed diffusion graphs, are crucial to controlling the spread of information. The problem of localizing the source of diffusion is highly ill-posed, presenting a major obstacle in accurately assessing the uncertainty involved. Besides, while comprehending how information diffuses through a graph is crucial, there is a scarcity of research on reconstructing the paths of information propagation. To tackle these challenges, we propose a probabilistic model called DDMSL (Discrete Diffusion Model for Source Localization). Our approach is based on the natural diffusion process of information propagation over complex networks, which can be formulated using a message-passing function. First, we model the forward diffusion of information using Markov chains. Then, we design a reversible residual network to construct a denoising-diffusion model in discrete space for both source localization and reconstruction of information diffusion paths. We provide rigorous theoretical guarantees for DDMSL and demonstrate its effectiveness through extensive experiments on five real-world datasets.

----

## [0] UniT: A Unified Look at Certified Robust Training against Text Adversarial Perturbation

**Authors**: *Muchao Ye, Ziyi Yin, Tianrong Zhang, Tianyu Du, Jinghui Chen, Ting Wang, Fenglong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46b065f7d301a15a23909f6cad409a97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46b065f7d301a15a23909f6cad409a97-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a surge of certified robust training pipelines against text adversarial perturbation constructed by synonym substitutions. Given a base model, existing pipelines provide prediction certificates either in the discrete word space or the continuous latent space. However, they are isolated from each other with a structural gap. We observe that existing training frameworks need unification to provide stronger certified robustness. Additionally,  they mainly focus on building the certification process but neglect to improve the robustness of the base model. To mitigate the aforementioned limitations, we propose a unified framework named UniT that enables us to train flexibly in either fashion by working in the word embedding space. It can provide a stronger robustness guarantee obtained directly from the word embedding space without extra modules. In addition, we introduce the decoupled regularization (DR) loss to improve the robustness of the base model, which includes two separate robustness regularization terms for the feature extraction and classifier modules. Experimental results on widely used text classification datasets further demonstrate the effectiveness of the designed unified framework and the proposed DR loss for improving the certified robust accuracy.

----

## [0] Convergence of Alternating Gradient Descent for Matrix Factorization

**Authors**: *Rachel A. Ward, Tamara G. Kolda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46c10f6c8ea5aa6f267bcdabcb123f97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46c10f6c8ea5aa6f267bcdabcb123f97-Abstract-Conference.html)

**Abstract**:

We consider alternating gradient descent (AGD) with fixed step size applied to the asymmetric matrix factorization objective.  We show that, for a rank-$r$ matrix $A \in \mathbb{R}^{m \times n}$,  $T = C ( \frac{\sigma_1(A)}{\sigma_r(A)} )^2 \log(1/\epsilon)$  iterations of alternating gradient descent suffice to reach an $\epsilon$-optimal factorization   $\| A - X_{T} Y_{T}' \|^2 \leq \epsilon \| A \|^2$   with high probability  starting from an atypical random initialization. The  factors have rank $d \geq r$ so that $X_{T}\in \mathbb{R}^{m \times d}$ and $Y_{T} \in\mathbb{R}^{n \times d}$, and mild overparameterization suffices for the constant  $C$ in the iteration complexity $T$ to be an absolute constant.   Experiments suggest that our proposed initialization is not merely of theoretical benefit, but rather significantly improves the convergence rate of gradient descent in practice. Our proof is conceptually simple: a uniform Polyak-Lojasiewicz (PL) inequality and uniform Lipschitz smoothness constant are guaranteed for a sufficient number of iterations, starting from our random initialization.  Our proof method should be useful for extending and simplifying convergence analyses for a broader class of nonconvex low-rank factorization problems.

----

## [0] SPRING: Studying Papers and Reasoning to play Games

**Authors**: *Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Russ Salakhutdinov, Amos Azaria, Tom M. Mitchell, Yuanzhi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html)

**Abstract**:

Open-world survival games pose significant challenges for AI algorithms due to their multi-tasking, deep exploration, and goal prioritization requirements. Despite reinforcement learning (RL) being popular for solving games, its high sample complexity limits its effectiveness in complex open-world games like Crafter or Minecraft. We propose a novel approach, SPRING, to read Crafter's original academic paper and use the knowledge learned to reason and play the game through a large language model (LLM).Prompted with the LaTeX source as game context and a description of the agent's current observation, our SPRING framework employs a directed acyclic graph (DAG) with game-related questions as nodes and dependencies as edges. We identify the optimal action to take in the environment by traversing the DAG and calculating LLM responses for each node in topological order, with the LLM's answer to final node directly translating to environment actions.In our experiments, we study the quality of in-context "reasoning" induced by different forms of prompts under the setting of the Crafter environment. Our experiments suggest that LLMs, when prompted with consistent chain-of-thought, have great potential in completing sophisticated high-level trajectories. Quantitatively, SPRING with GPT-4 outperforms all state-of-the-art RL baselines, trained for 1M steps, without any training. Finally, we show the potential of Crafter as a test bed for LLMs. Code at github.com/holmeswww/SPRING

----

## [0] Hybrid Search for Efficient Planning with Completeness Guarantees

**Authors**: *Kalle Kujanpää, Joni Pajarinen, Alexander Ilin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46d26daeb05fbbcfe5f3d8f7ca756e16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46d26daeb05fbbcfe5f3d8f7ca756e16-Abstract-Conference.html)

**Abstract**:

Solving complex planning problems has been a long-standing challenge in computer science. Learning-based subgoal search methods have shown promise in tackling these problems, but they often suffer from a lack of completeness guarantees, meaning that they may fail to find a solution even if one exists. In this paper, we propose an efficient approach to augment a subgoal search method to achieve completeness in discrete action spaces. Specifically, we augment the high-level search with low-level actions to execute a multi-level (hybrid) search, which we call complete subgoal search. This solution achieves the best of both worlds: the practical efficiency of high-level search and the completeness of low-level search. We apply the proposed search method to a recently proposed subgoal search algorithm and evaluate the algorithm trained on offline data on complex planning problems. We demonstrate that our complete subgoal search not only guarantees completeness but can even improve performance in terms of search expansions for instances that the high-level could solve without low-level augmentations. Our approach makes it possible to apply subgoal-level planning for systems where completeness is a critical requirement.

----

## [0] Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation

**Authors**: *Jianing Zhu, Yu Geng, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46d943bc6a15a57c923829efc0db7c7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46d943bc6a15a57c923829efc0db7c7a-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection is important for deploying reliable machine learning models on real-world applications. Recent advances in outlier exposure have shown promising results on OOD detection via fine-tuning model with informatively sampled auxiliary outliers. However, previous methods assume that the collected outliers can be sufficiently large and representative to cover the boundary between ID and OOD data, which might be impractical and challenging. In this work, we propose a novel framework, namely, Diversified Outlier Exposure (DivOE), for effective OOD detection via informative extrapolation based on the given auxiliary outliers. Specifically, DivOE introduces a new learning objective, which diversifies the auxiliary distribution by explicitly synthesizing more informative outliers for extrapolation during training. It leverages a multi-step optimization method to generate novel outliers beyond the original ones, which is compatible with many variants of outlier exposure. Extensive experiments and analyses have been conducted to characterize and demonstrate the effectiveness of the proposed DivOE. The code is publicly available at: https://github.com/tmlr-group/DivOE.

----

## [0] Attacks on Online Learners: a Teacher-Student Analysis

**Authors**: *Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46e37aeccafc3b4b697b17b8a36f3b30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46e37aeccafc3b4b697b17b8a36f3b30-Abstract-Conference.html)

**Abstract**:

Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.

----

## [0] Delayed Algorithms for Distributed Stochastic Weakly Convex Optimization

**Authors**: *Wenzhi Gao, Qi Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html)

**Abstract**:

This paper studies delayed stochastic algorithms for weakly convex optimization in a distributed network with workers connected to a master node.  Recently, Xu~et~al.~2022  showed that an inertial stochastic subgradient method   converges at a rate of $\mathcal{O}(\tau_{\text{max}}/\sqrt{K})$ which depends on the maximum information delay $\tau_{\text{max}}$. In this work, we show that the delayed stochastic subgradient method ($\texttt{DSGD}$) obtains a tighter convergence rate which depends on the expected delay $\bar{\tau}$. Furthermore, for an important class of composition weakly convex problems, we develop a new delayed stochastic prox-linear ($\texttt{DSPL}$) method in which the delays only affect the high-order term in the rate and hence,  are negligible after a certain number of $\texttt{DSPL}$  iterations.  In addition, we demonstrate the robustness of our proposed algorithms against arbitrary delays.  By incorporating a simple safeguarding step in both methods, we achieve convergence rates that depend solely on the number of workers, eliminating the effect of delays. Our numerical experiments further confirm the empirical superiority of our proposed methods.

----

## [0] Grounding Neural Inference with Satisfiability Modulo Theories

**Authors**: *Zifan Wang, Saranya Vijayakumar, Kaiji Lu, Vijay Ganesh, Somesh Jha, Matt Fredrikson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47167991e38c65a72914763c11cd8d23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47167991e38c65a72914763c11cd8d23-Abstract-Conference.html)

**Abstract**:

Recent techniques that integrate solver layers into Deep Neural Networks (DNNs) have shown promise in bridging a long-standing gap between inductive learning and symbolic reasoning techniques. In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer.Using this approach, one can encode rich domain knowledge into the network in the form of mathematical formulas.In the forward pass, the solver uses symbols produced by prior layers, along with these formulas, to construct inferences; in the backward pass, the solver informs updates to the network, driving it towards representations that are compatible with the solver's theory.Notably, the solver need not be differentiable. We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.

----

## [0] D2CSG: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts

**Authors**: *Fenggen Yu, Qimin Chen, Maham Tanveer, Ali Mahdavi-Amiri, Hao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4732d425125832887f6c5a9675d49ead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4732d425125832887f6c5a9675d49ead-Abstract-Conference.html)

**Abstract**:

We present D$^2$CSG, a neural model composed of two dual and complementary network branches, with dropouts, for unsupervised learning of compact constructive solid geometry (CSG) representations of 3D CAD shapes. Our network is trained to reconstruct a 3D shape by a fixed-order assembly of quadric primitives, with both branches producing a union of primitive intersections or inverses. A key difference between D$^2$CSG and all prior neural CSG models is its dedicated residual branch to assemble the potentially complex shape complement, which is subtracted from an overall shape modeled by the cover branch. With the shape complements, our network is provably general, while the weight dropout further improves compactness of the CSG tree by removing redundant primitives. We demonstrate both quantitatively and qualitatively that D$^2$CSG produces compact CSG reconstructions with superior quality and more natural primitives than all existing alternatives, especially over complex and high-genus CAD shapes.

----

## [0] Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering

**Authors**: *Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html)

**Abstract**:

Knowledge-based Visual Question Answering (KB-VQA) requires VQA systems to utilize knowledge from external knowledge bases to answer visually-grounded questions. Retrieval-Augmented Visual Question Answering (RA-VQA), a strong framework to tackle KB-VQA, first retrieves related documents with Dense Passage Retrieval (DPR) and then uses them to answer questions. This paper proposes Fine-grained Late-interaction Multi-modal Retrieval (FLMR) which significantly improves knowledge retrieval in RA-VQA. FLMR addresses two major limitations in RA-VQA's retriever: (1) the image representations obtained via image-to-text transforms can be incomplete and inaccurate and (2) similarity scores between queries and documents are computed with one-dimensional embeddings, which can be insensitive to finer-grained similarities.FLMR overcomes these limitations by obtaining image representations that complement those from the image-to-text transform using a vision model aligned with an existing text-based retriever through a simple alignment network. FLMR also encodes images and questions using multi-dimensional embeddings to capture finer-grained similarities between queries and documents. FLMR significantly improves the original RA-VQA retriever's PRRecall@5 by approximately 8\%. Finally, we equipped RA-VQA with two state-of-the-art large multi-modal/language models to achieve $\sim62$% VQA score in the OK-VQA dataset.

----

## [0] Iteratively Learn Diverse Strategies with State Distance Information

**Authors**: *Wei Fu, Weihua Du, Jingwei Li, Sunli Chen, Jingzhao Zhang, Yi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/473aadf077f8464dbae7e9600d9be6c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/473aadf077f8464dbae7e9600d9be6c4-Abstract-Conference.html)

**Abstract**:

In complex reinforcement learning (RL) problems, policies with similar rewards may have substantially different behaviors. It remains a fundamental challenge to optimize rewards while also discovering as many diverse strategies as possible, which can be crucial in many practical applications. Our study examines two design choices for tackling this challenge, i.e., diversity measure and computation framework. First, we find that with existing diversity measures, visually indistinguishable policies can still yield high diversity scores. To accurately capture the behavioral difference, we propose to incorporate the state-space distance information into the diversity measure. In addition, we examine two common computation frameworks for this problem, i.e., population-based training (PBT) and iterative learning (ITR). We show that although PBT is the precise problem formulation, ITR can achieve comparable diversity scores with higher computation efficiency, leading to improved solution quality in practice. Based on our analysis, we further combine ITR with two tractable realizations of the state-distance-based diversity measures and develop a novel diversity-driven RL algorithm, State-based Intrinsic-reward Policy Optimization (SIPO), with provable convergence properties. We empirically examine SIPO across three domains from robot locomotion to multi-agent games. In all of our testing environments, SIPO consistently produces strategically diverse and human-interpretable policies that cannot be discovered by existing baselines.

----

## [0] Neural Fields with Hard Constraints of Arbitrary Differential Order

**Authors**: *Fangcheng Zhong, Kyle Fogarty, Param Hanji, Tianhao Wu, Alejandro Sztrajman, Andrew Spielberg, Andrea Tagliasacchi, Petra Bosilj, Cengiz Öztireli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html)

**Abstract**:

While deep learning techniques have become extremely popular for solving a broad range of optimization problems, methods to enforce hard constraints during optimization, particularly on deep neural networks, remain underdeveloped. Inspired by the rich literature on meshless interpolation and its extension to spectral collocation methods in scientific computing, we develop a series of approaches for enforcing hard constraints on neural fields, which we refer to as Constrained Neural Fields (CNF). The constraints can be specified as a linear operator applied to the neural field and its derivatives. We also design specific model representations and training strategies for problems where standard models may encounter difficulties, such as conditioning of the system, memory consumption, and capacity of the network when being constrained. Our approaches are demonstrated in a wide range of real-world applications. Additionally, we develop a framework that enables highly efficient model and constraint specification, which can be readily applied to any downstream task where hard constraints need to be explicitly satisfied during optimization.

----

## [0] Thinker: Learning to Plan and Act

**Authors**: *Stephen Chung, Ivan Anokhin, David Krueger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html)

**Abstract**:

We propose the Thinker algorithm, a novel approach that enables reinforcement learning agents to autonomously interact with and utilize a learned world model. The Thinker algorithm wraps the environment with a world model and introduces new actions designed for interacting with the world model. These model-interaction actions enable agents to perform planning by proposing alternative plans to the world model before selecting a final action to execute in the environment. This approach eliminates the need for handcrafted planning algorithms by enabling the agent to learn how to plan autonomously and allows for easy interpretation of the agent's plan with visualization. We demonstrate the algorithm's effectiveness through experimental results in the game of Sokoban and the Atari 2600 benchmark, where the Thinker algorithm achieves state-of-the-art performance and competitive results, respectively. Visualizations of agents trained with the Thinker algorithm demonstrate that they have learned to plan effectively with the world model to select better actions. Thinker is the first work showing that an RL agent can learn to plan with a learned world model in complex environments.

----

## [0] Near-Optimal k-Clustering in the Sliding Window Model

**Authors**: *David P. Woodruff, Peilin Zhong, Samson Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html)

**Abstract**:

Clustering is an important technique for identifying structural information in large-scale data analysis, where the underlying dataset may be too large to store. In many applications, recent data can provide more accurate information and thus older data past a certain time is expired. The sliding window model captures these desired properties and thus there has been substantial interest in clustering in the sliding window model. In this paper, we give the first algorithm that achieves near-optimal $(1+\varepsilon)$-approximation to $(k,z)$-clustering in the sliding window model. Our algorithm uses $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ words of space when the points are from $[\Delta]^d$, thus significantly improving on works by Braverman et. al. (SODA 2016), Borassi et. al. (NeurIPS 2021), and Epasto et. al. (SODA 2022).Along the way, we develop a data structure for clustering called an online coreset, which outputs a coreset not only for the end of a stream, but also for all prefixes of the stream. Our online coreset samples $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ points from the stream. We then show that any online coreset requires $\Omega\left(\frac{k}{\varepsilon^2}\log n\right)$ samples, which shows a separation between the problem of constructing an offline coreset, i.e., constructing online coresets is strictly harder. Our results also extend to general metrics on $[\Delta]^d$ and are near-optimal in light of a $\Omega\left(\frac{k}{\varepsilon^{2+z}}\right)$ lower bound for the size of an offline coreset.

----

## [0] SynMob: Creating High-Fidelity Synthetic GPS Trajectory Dataset for Urban Mobility Analysis

**Authors**: *Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, James Jian Qiao Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4786c0d1b9687a841bc579b0b8b01b8e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/4786c0d1b9687a841bc579b0b8b01b8e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Urban mobility analysis has been extensively studied in the past decade using a vast amount of GPS trajectory data, which reveals hidden patterns in movement and human activity within urban landscapes. Despite its significant value, the availability of such datasets often faces limitations due to privacy concerns, proprietary barriers, and quality inconsistencies. To address these challenges, this paper presents a synthetic trajectory dataset with high fidelity, offering a general solution to these data accessibility issues. Specifically, the proposed dataset adopts a diffusion model as its synthesizer, with the primary aim of accurately emulating the spatial-temporal behavior of the original trajectory data. These synthesized data can retain the geo-distribution and statistical properties characteristic of real-world datasets. Through rigorous analysis and case studies, we validate the high similarity and utility between the proposed synthetic trajectory dataset and real-world counterparts. Such validation underscores the practicality of synthetic datasets for urban mobility analysis and advocates for its wider acceptance within the research community. Finally, we publicly release the trajectory synthesizer and datasets, aiming to enhance the quality and availability of synthetic trajectory datasets and encourage continued contributions to this rapidly evolving field. The dataset is released for public online availability https://github.com/Applied-Machine-Learning-Lab/SynMob.

----

## [0] Window-Based Distribution Shift Detection for Deep Neural Networks

**Authors**: *Guy Bar-Shalom, Yonatan Geifman, Ran El-Yaniv*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4791edcba96fbd82a8962b0f790b52c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4791edcba96fbd82a8962b0f790b52c9-Abstract-Conference.html)

**Abstract**:

To deploy and operate deep neural models in production, the quality of their predictions, which might be contaminated benignly or manipulated maliciously by input distributional deviations, must be monitored and assessed. Specifically, we study the case of monitoring the healthy operation of a deep neural network (DNN) receiving a stream of data, with the aim of detecting input distributional deviations over which the quality of the network's predictions is potentially damaged. Using selective prediction principles, we propose a distribution deviation detection method for DNNs. The proposed method is derived from a tight coverage generalization bound computed over a sample of instances drawn from the true underlying distribution. Based on this bound, our detector continuously monitors the operation of the network over a test window and fires off an alarm whenever a deviation is detected. Our novel detection method performs on-par or better than the state-of-the-art, while consuming substantially lower computation time (five orders of magnitude reduction) and space complexity. Unlike previous methods, which require at least linear dependence on the size of the source distribution for each detection, rendering them inapplicable to ``Google-Scale'' datasets, our approach eliminates this dependence, making it suitable for real-world applications. Code is available at https://github.com/BarSGuy/Window-Based-Distribution-Shift-Detection.

----

## [0] Towards Label Position Bias in Graph Neural Networks

**Authors**: *Haoyu Han, Xiaorui Liu, Feng Shi, MohamadAli Torkamani, Charu C. Aggarwal, Jiliang Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4798eef078de031518beaf54f4b5fb5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4798eef078de031518beaf54f4b5fb5f-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as a powerful tool for semi-supervised node classification tasks. However, recent studies have revealed various biases in GNNs stemming from both node features and graph topology. In this work, we uncover a new bias - label position bias, which indicates that the node closer to the labeled nodes tends to perform better. We introduce a new metric, the Label Proximity Score, to quantify this bias, and find that it is closely related to performance disparities. To address the label position bias, we propose a novel optimization framework for learning a label position unbiased graph structure, which can be applied to existing GNNs. Extensive experiments demonstrate that our proposed method not only outperforms backbone methods but also significantly mitigates the issue of label position bias in GNNs.

----

## [0] Label Robust and Differentially Private Linear Regression: Computational and Statistical Efficiency

**Authors**: *Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, Arun Sai Suggala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47e74fca60b4af4846b7abab188b85f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47e74fca60b4af4846b7abab188b85f2-Abstract-Conference.html)

**Abstract**:

We study the canonical problem of linear regression under $(\varepsilon,\delta)$-differential privacy when the datapoints are sampled i.i.d.~from a distribution and a fraction of response variables are adversarially corrupted. We provide the first provably efficient -- both computationally and statistically -- method for this problem, assuming standard assumptions on the data distribution. Our algorithm is a variant of the popular differentially private stochastic gradient descent (DP-SGD) algorithm with two key innovations: a full-batch gradient descent to improve sample complexity and a novel adaptive clipping to guarantee robustness. Our method requires only linear time in input size, and still matches the information theoretical optimal sample complexity up to a data distribution dependent condition number factor.  Interestingly, the same algorithm, when applied to a setting where there is no adversarial corruption, still improves upon the existing state-of-the-art and achieves a near optimal sample complexity.

----

## [0] Explainable and Efficient Randomized Voting Rules

**Authors**: *Soroush Ebadian, Aris Filos-Ratsikas, Mohamad Latifian, Nisarg Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47eb2874a790d5b1f554b9bb93b3de9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47eb2874a790d5b1f554b9bb93b3de9d-Abstract-Conference.html)

**Abstract**:

With a rapid growth in the deployment of AI tools for making critical decisions (or aiding humans in doing so), there is a growing demand to be able to explain to the stakeholders how these tools arrive at a decision. Consequently, voting is frequently used to make such decisions due to its inherent explainability. Recent work suggests that using randomized (as opposed to deterministic) voting rules can lead to significant efficiency gains measured via the distortion framework. However, rules that use intricate randomization can often become too complex to explain to the stakeholders; losing explainability can eliminate the key advantage of voting over black-box AI tools, which may outweigh the efficiency gains.We study the efficiency gains which can be unlocked by using voting rules that add a simple randomization step to a deterministic rule, thereby retaining explainability. We focus on two such families of rules, randomized positional scoring rules and random committee member rules, and show, theoretically and empirically, that they indeed achieve explainability and efficiency simultaneously to some extent.

----

## [0] Conformal PID Control for Time Series Prediction

**Authors**: *Anastasios Angelopoulos, Emmanuel J. Candès, Ryan J. Tibshirani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html)

**Abstract**:

We study the problem of uncertainty quantification for time series prediction, with the goal of providing easy-to-use  algorithms with formal guarantees. The algorithms we present build upon ideas from conformal prediction and control theory, are able to prospectively model conformal scores in an online setting, and adapt to the presence of systematic errors due to seasonality, trends, and general distribution shifts. Our theory both simplifies and strengthens existing analyses in online conformal prediction. Experiments on 4-week-ahead forecasting of statewide COVID-19 death counts in the U.S. show an improvement in coverage over the ensemble forecaster used inofficial CDC communications. We also run experiments on predicting electricity demand, market returns, and temperature using autoregressive, Theta, Prophet, and Transformer models. We provide an extendable codebase for testing our methods and for the integration of new algorithms, data sets, and forecasting rules at this link.

----



[Go to the previous page](NIPS-2023-list900.md)

[Go to the next page](NIPS-2023-list902.md)

[Go to the catalog section](README.md)