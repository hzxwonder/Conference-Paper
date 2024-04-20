## [0] A Theory of Multimodal Learning

**Authors**: *Zhou Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b316495425d076b4abffc065a64c2cca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b316495425d076b4abffc065a64c2cca-Abstract-Conference.html)

**Abstract**:

Human perception of the empirical world involves recognizing the diverse appearances, or 'modalities', of underlying objects. Despite the longstanding consideration of this perspective in philosophy and cognitive science, the study of multimodality remains relatively under-explored within the field of machine learning. Nevertheless, current studies of multimodal machine learning are limited to empirical practices, lacking theoretical foundations beyond heuristic arguments. An intriguing finding from the practice of multimodal learning is that a model trained on multiple modalities can outperform a finely-tuned unimodal model, even on unimodal tasks. This paper provides a theoretical framework that explains this phenomenon, by studying generalization properties of multimodal learning algorithms. We demonstrate that multimodal learning allows for a superior generalization bound compared to unimodal learning, up to a factor of $O(\sqrt{n})$, where $n$ represents the sample size. Such advantage occurs when both connection and heterogeneity exist between the modalities.

----

## [0] IDEA: An Invariant Perspective for Efficient Domain Adaptive Image Retrieval

**Authors**: *Haixin Wang, Hao Wu, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, Xiao Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b33ad9d46ab2a23b6783d954121d26e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b33ad9d46ab2a23b6783d954121d26e3-Abstract-Conference.html)

**Abstract**:

In this paper, we investigate the problem of unsupervised domain adaptive hashing, which leverage knowledge from a label-rich source domain to expedite learning to hash on a label-scarce target domain. Although numerous existing approaches attempt to incorporate transfer learning techniques into deep hashing frameworks, they often neglect the essential invariance for adequate alignment between these two domains. Worse yet, these methods fail to distinguish between causal and non-causal effects embedded in images, rendering cross-domain retrieval ineffective. To address these challenges, we propose an Invariance-acquired Domain AdaptivE HAshing (IDEA) model. Our IDEA first decomposes each image into a causal feature representing label information, and a non-causal feature indicating domain information. Subsequently, we generate discriminative hash codes using causal features with consistency learning on both source and target domains. More importantly, we employ a generative model for synthetic samples to simulate the intervention of various non-causal effects, ultimately minimizing their impact on hash codes for domain invariance. Comprehensive experiments conducted on benchmark datasets validate the superior performance of our IDEA compared to a variety of competitive baselines.

----

## [0] Learning Provably Robust Estimators for Inverse Problems via Jittering

**Authors**: *Anselm Krainovic, Mahdi Soltanolkotabi, Reinhard Heckel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3411e30afa6caeefa4d6d39a5ea84cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3411e30afa6caeefa4d6d39a5ea84cd-Abstract-Conference.html)

**Abstract**:

Deep neural networks provide excellent performance for inverse problems such as denoising. However, neural networks can be sensitive to adversarial or worst-case perturbations. This raises the question of whether such networks can be trained efficiently to be worst-case robust. In this paper, we investigate whether jittering, a simple regularization technique that adds isotropic Gaussian noise during training, is effective for learning worst-case robust estimators for inverse problems. While well studied for prediction in classification tasks, the effectiveness of jittering for inverse problems has not been systematically investigated. In this paper, we present a novel analytical characterization of the optimal $\ell_2$-worst-case robust estimator for linear denoising and show that jittering yields optimal robust denoisers. Furthermore, we examine jittering empirically via training deep neural networks (U-nets) for natural image denoising, deconvolution, and accelerated magnetic resonance imaging (MRI). The results show that jittering significantly enhances the worst-case robustness, but can be suboptimal for inverse problems beyond denoising. Moreover, our results imply that training on real data which often contains slight noise is somewhat robustness enhancing.

----

## [0] On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

**Authors**: *Rajat Modi, Vibhav Vineet, Yogesh S. Rawat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper explores the impact of occlusions in video action detection. We facilitatethis study by introducing five new benchmark datasets namely O-UCF and O-JHMDB consisting of synthetically controlled static/dynamic occlusions, OVIS-UCF and OVIS-JHMDB consisting of occlusions with realistic motions and Real-OUCF for occlusions in realistic-world scenarios. We formally confirm an intuitiveexpectation: existing models suffer a lot as occlusion severity is increased andexhibit different behaviours when occluders are static vs when they are moving.We discover several intriguing phenomenon emerging in neural nets: 1) transformerscan naturally outperform CNN models which might have even used occlusion as aform of data augmentation during training 2) incorporating symbolic-componentslike capsules to such backbones allows them to bind to occluders never even seenduring training and 3) Islands of agreement (similar to the ones hypothesized inHinton et Alâ€™s GLOM) can emerge in realistic images/videos without instance-levelsupervision, distillation or contrastive-based objectives(eg. video-textual training).Such emergent properties allow us to derive simple yet effective training recipeswhich lead to robust occlusion models inductively satisfying the first two stages ofthe binding mechanism (grouping/segregation). Models leveraging these recipesoutperform existing video action-detectors under occlusion by 32.3% on O-UCF,32.7% on O-JHMDB & 2.6% on Real-OUCF in terms of the vMAP metric. The code for this work has been released at https: //github.com/rajatmodi62/OccludedActionBenchmark.

----

## [0] Black-box Backdoor Defense via Zero-shot Image Purification

**Authors**: *Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, Ninghao Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html)

**Abstract**:

Backdoor attacks inject poisoned samples into the training data, resulting in the misclassification of the poisoned input during a model's deployment. Defending against such attacks is challenging, especially for real-world black-box models where only query access is permitted. In this paper, we propose a novel defense framework against backdoor attacks through Zero-shot Image Purification (ZIP). Our framework can be applied to poisoned models without requiring internal information about the model or any prior knowledge of the clean/poisoned samples. Our defense framework involves two steps. First, we apply a linear transformation (e.g., blurring) on the poisoned image to destroy the backdoor pattern. Then, we use a pre-trained diffusion model to recover the missing semantic information removed by the transformation. In particular, we design a new reverse process by using the transformed image to guide the generation of high-fidelity purified images, which works in zero-shot settings. We evaluate our ZIP framework on multiple datasets with different types of attacks. Experimental results demonstrate the superiority of our ZIP framework compared to state-of-the-art backdoor defense baselines. We believe that our results will provide valuable insights for future defense methods for black-box models. Our code is available at https://github.com/sycny/ZIP.

----

## [0] Beyond NTK with Vanilla Gradient Descent: A Mean-Field Analysis of Neural Networks with Polynomial Width, Samples, and Time

**Authors**: *Arvind Mahankali, Haochen Zhang, Kefan Dong, Margalit Glasgow, Tengyu Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3748cdac932d91f0a51a37db90dec50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3748cdac932d91f0a51a37db90dec50-Abstract-Conference.html)

**Abstract**:

Despite recent theoretical progress on the non-convex optimization of two-layer neural networks, it is still an open question whether gradient descent on neural networks without unnatural modifications can achieve better sample complexity than kernel methods. This paper provides a clean mean-field analysis of projected gradient flow on polynomial-width two-layer neural networks. Different from prior works, our analysis does not require unnatural modifications of the optimization algorithm. We prove that with sample size $n = O(d^{3.1})$ where $d$ is the dimension of the inputs, the network trained with projected gradient flow converges in polynomial time to a non-trivial error that is not achievable by kernel methods using $n \ll  d^4$ samples, hence demonstrating a clear separation between unmodified gradient descent and NTK. As a corollary, we show that projected gradient descent with a positive learning rate and a polynomial number of iterations converges to low error with the same sample complexity.

----

## [0] Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding

**Authors**: *Zhejun Zhang, Alexander Liniger, Christos Sakaridis, Fisher Yu, Luc Van Gool*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b37c2e26b75ee02fcabd65a2a0367136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b37c2e26b75ee02fcabd65a2a0367136-Abstract-Conference.html)

**Abstract**:

The real-world deployment of an autonomous driving system requires its components to run on-board and in real-time, including the motion prediction module that predicts the future trajectories of surrounding traffic participants. Existing agent-centric methods have demonstrated outstanding performance on public benchmarks. However, they suffer from high computational overhead and poor scalability as the number of agents to be predicted increases. To address this problem, we introduce the K-nearest neighbor attention with relative pose encoding (KNARPE), a novel attention mechanism allowing the pairwise-relative representation to be used by Transformers. Then, based on KNARPE we present the Heterogeneous Polyline Transformer with Relative pose encoding (HPTR), a hierarchical framework enabling asynchronous token update during the online inference. By sharing contexts among agents and reusing the unchanged contexts, our approach is as efficient as scene-centric methods, while performing on par with state-of-the-art agent-centric methods. Experiments on Waymo and Argoverse-2 datasets show that HPTR achieves superior performance among end-to-end methods that do not apply expensive post-processing or model ensembling. The code is available at https://github.com/zhejz/HPTR.

----

## [0] Customizable Image Synthesis with Multiple Subjects

**Authors**: *Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html)

**Abstract**:

Synthesizing images with user-specified subjects has received growing attention due to its practical applications. Despite the recent success in single subject customization, existing algorithms suffer from high training cost and low success rate along with increased number of subjects. Towards controllable image synthesis with multiple subjects as the constraints, this work studies how to efficiently represent a particular subject as well as how to appropriately compose different subjects. We find that the text embedding regarding the subject token already serves as a simple yet effective representation that supports arbitrary combinations without any model tuning. Through learning a residual on top of the base embedding, we manage to robustly shift the raw subject to the customized subject given various text conditions. We then propose to employ layout, a very abstract and easy-to-obtain prior, as the spatial guidance for subject arrangement. By rectifying the activations in the cross-attention map, the layout appoints and separates the location of different subjects in the image, significantly alleviating the interference across them. Using cross-attention map as the intermediary, we could strengthen the signal of target subjects and weaken the signal of irrelevant subjects within a certain region, significantly alleviating the interference across subjects. Both qualitative and quantitative experimental results demonstrate our superiority over state-of-the-art alternatives under a variety of settings for multi-subject customization.

----

## [0] How do Minimum-Norm Shallow Denoisers Look in Function Space?

**Authors**: *Chen Zeno, Greg Ongie, Yaniv Blumenfeld, Nir Weinberger, Daniel Soudry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html)

**Abstract**:

Neural network (NN) denoisers are an essential building block in many common tasks, ranging from image reconstruction to image generation. However, the success of these models is not well understood from a theoretical perspective. In this paper, we aim to characterize the functions realized by shallow ReLU NN denoisers --- in the common theoretical setting of interpolation (i.e., zero training loss) with a minimal representation cost (i.e., minimal $\ell^2$ norm weights). First, for univariate data, we derive a closed form for the NN denoiser function, find it is contractive toward the clean data points, and prove it generalizes better than the empirical MMSE estimator at a low noise level. Next, for multivariate data, we find the NN denoiser functions in a closed form under various geometric assumptions on the training data: data contained in a low-dimensional subspace, data contained in a union of one-sided rays, or several types of simplexes. These functions decompose into a sum of simple rank-one piecewise linear interpolations aligned with edges and/or faces connecting training samples. We empirically verify this alignment phenomenon on synthetic data and real images.

----

## [0] Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping

**Authors**: *Feng Zhang, Ming Tian, Zhiqiang Li, Bin Xu, Qingbo Lu, Changxin Gao, Nong Sang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3a08d179347e33414badadf100e4e8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3a08d179347e33414badadf100e4e8d-Abstract-Conference.html)

**Abstract**:

Tone mapping aims to convert high dynamic range (HDR) images to low dynamic range (LDR) representations, a critical task in the camera imaging pipeline. In recent years, 3-Dimensional LookUp Table (3D LUT) based methods have gained attention due to their ability to strike a favorable balance between enhancement performance and computational efficiency. However, these methods often fail to deliver satisfactory results in local areas since the look-up table is a global operator for tone mapping, which works based on pixel values and fails to incorporate crucial local information. To this end, this paper aims to address this issue by exploring a novel strategy that integrates global and local operators by utilizing closed-form Laplacian pyramid decomposition and reconstruction. Specifically, we employ image-adaptive 3D LUTs to manipulate the tone in the low-frequency image by leveraging the specific characteristics of the frequency information. Furthermore, we utilize local Laplacian filters to refine the edge details in the high-frequency components in an adaptive manner. Local Laplacian filters are widely used to preserve edge details in photographs, but their conventional usage involves manual tuning and fixed implementation within camera imaging pipelines or photo editing tools. We propose to learn parameter value maps progressively for local Laplacian filters from annotated data using a lightweight network. Our model achieves simultaneous global tone manipulation and local edge detail preservation in an end-to-end manner. Extensive experimental results on two benchmark datasets demonstrate that the proposed method performs favorably against state-of-the-art methods.

----

## [0] Masked Image Residual Learning for Scaling Deeper Vision Transformers

**Authors**: *Guoxi Huang, Hongtao Fu, Adrian G. Bors*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3bac97f3227c52c0179a6d967480867-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3bac97f3227c52c0179a6d967480867-Abstract-Conference.html)

**Abstract**:

Deeper Vision Transformers (ViTs) are more challenging to train. We expose a degradation problem in deeper layers of ViT when using masked image modeling (MIM) for pre-training.To ease the training of deeper ViTs, we introduce a self-supervised learning framework called  $\textbf{M}$asked $\textbf{I}$mage $\textbf{R}$esidual $\textbf{L}$earning ($\textbf{MIRL}$), which significantly alleviates the degradation problem, making scaling ViT along depth a promising direction for performance upgrade. We reformulate the pre-training objective for deeper layers of ViT as learning to recover the residual of the masked image.We provide extensive empirical evidence showing that deeper ViTs can be effectively optimized using MIRL and easily gain accuracy from increased depth. With the same level of computational complexity as ViT-Base and ViT-Large, we instantiate $4.5{\times}$ and $2{\times}$ deeper ViTs, dubbed ViT-S-54 and ViT-B-48.The deeper ViT-S-54, costing $3{\times}$ less than ViT-Large, achieves performance on par with ViT-Large.ViT-B-48 achieves 86.2\% top-1 accuracy on ImageNet. On one hand, deeper ViTs pre-trained with MIRL exhibit excellent generalization capabilities on downstream tasks, such as object detection and semantic segmentation. On the other hand, MIRL demonstrates high pre-training efficiency. With less pre-training time, MIRL yields competitive performance compared to other approaches.

----

## [0] Revisiting Area Convexity: Faster Box-Simplex Games and Spectrahedral Generalizations

**Authors**: *Arun Jambulapati, Kevin Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3bec3f5ad96055b7f60c93edc3606c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3bec3f5ad96055b7f60c93edc3606c8-Abstract-Conference.html)

**Abstract**:

We investigate area convexity [Sherman17], a mysterious tool introduced to tackle optimization problems under the challenging $\ell_\infty$ geometry. We develop a deeper understanding of its relationship with conventional analyses of extragradient methods [Nemirovski04, Nesterov07]. We also give improved solvers for the subproblems required by variants of the [Sherman17] algorithm, designed through the lens of relative smoothness [BBT17, LFN18}.Leveraging these new tools, we give a state-of-the-art first-order algorithm for solving box-simplex games (a primal-dual formulation of $\ell_\infty$ regression) in a $d \times n$ matrix with bounded rows, using $O(\log d \cdot \epsilon^{-1})$ matrix-vector queries. As a consequence, we obtain improved complexities for approximate maximum flow, optimal transport, min-mean-cycle, and other basic combinatorial optimization problems. We also develop a near-linear time algorithm for a matrix generalization of box-simplex games, capturing a family of problems closely related to semidefinite programs recently used as subroutines in robust statistics and numerical linear algebra.

----

## [0] Adversarial Learning for Feature Shift Detection and Correction

**Authors**: *Míriam Barrabés, Daniel Mas Montserrat, Margarita Geleta, Xavier Giró-i-Nieto, Alexander G. Ioannidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3cd64ddad0a28da0f28a0e03a73ea7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3cd64ddad0a28da0f28a0e03a73ea7d-Abstract-Conference.html)

**Abstract**:

Data shift is a phenomenon present in many real-world applications, and while there are multiple methods attempting to detect shifts, the task of localizing and correcting the features originating such shifts has not been studied in depth. Feature shifts can occur in many datasets, including in multi-sensor data, where some sensors are malfunctioning, or in tabular and structured data, including biomedical, financial, and survey data, where faulty standardization and data processing pipelines can lead to erroneous features. In this work, we explore using the principles of adversarial learning, where the information from several discriminators trained to distinguish between two distributions is used to both detect the corrupted features and fix them in order to remove the distribution shift between datasets. We show that mainstream supervised classifiers, such as random forest or gradient boosting trees, combined with simple iterative heuristics, can localize and correct feature shifts, outperforming current statistical and neural network-based techniques. The code is available at https://github.com/AI-sandbox/DataFix.

----

## [0] General Munchausen Reinforcement Learning with Tsallis Kullback-Leibler Divergence

**Authors**: *Lingwei Zhu, Zheng Chen, Matthew Schlegel, Martha White*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3e866c228f8f4ea18021ae63aea5453-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3e866c228f8f4ea18021ae63aea5453-Abstract-Conference.html)

**Abstract**:

Many policy optimization approaches in reinforcement learning incorporate a Kullback-Leilbler (KL) divergence to the previous policy, to prevent the policy from changing too quickly. This idea was initially proposed in a seminal paper on Conservative Policy Iteration, with approximations given by algorithms like TRPO and Munchausen Value Iteration (MVI). We continue this line of work by investigating a generalized KL divergence---called the Tsallis KL divergence. Tsallis KL defined by the $q$-logarithm is a strict generalization, as $q = 1$ corresponds to the standard KL divergence; $q > 1$ provides a range of new options. We characterize the types of policies learned under the Tsallis KL, and motivate when $q >1$ could be beneficial.  To obtain a practical algorithm that incorporates Tsallis KL regularization, we extend MVI, which is one of the simplest approaches to incorporate KL regularization. We show that this generalized MVI($q$) obtains significant improvements over the standard MVI($q = 1$) across 35 Atari games.

----

## [0] Residual Alignment: Uncovering the Mechanisms of Residual Networks

**Authors**: *Jianing Li, Vardan Papyan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3f48945f6fb402b4b5cdcf490e72847-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3f48945f6fb402b4b5cdcf490e72847-Abstract-Conference.html)

**Abstract**:

The ResNet architecture has been widely adopted in deep learning due to its significant boost to performance through the use of simple skip connections, yet the underlying mechanisms leading to its success remain largely unknown. In this paper, we conduct a thorough empirical study of the ResNet architecture in classification tasks by linearizing its constituent residual blocks using Residual Jacobians and measuring their singular value decompositions. Our measurements ([code](https://colab.research.google.com/drive/1yKjEg2yF616tnZFAfuN0aQ-E9v3JmyjN?usp=sharing)) reveal a process called Residual Alignment (RA) characterized by four properties:- **(RA1):** intermediate representations of a given input are *equispaced* on a *line*, embedded in high dimensional space, as observed by Gai and Zhang [2021];- **(RA2):** top left and right singular vectors of Residual Jacobians align with each other and across different depths;- **(RA3):** Residual Jacobians are at most rank $C$ for fully-connected ResNets, where $C$ is the number of classes; and- **(RA4):** top singular values of Residual Jacobians scale inversely with depth.RA consistently occurs in models that generalize well, in both fully-connected and convolutional architectures, across various depths and widths, for varying numbers of classes, on all tested benchmark datasets, but ceases to occur once the skip connections are removed. It also provably occurs in a novel mathematical model we propose. This phenomenon reveals a strong alignment between residual branches of a ResNet (RA2+4), imparting a highly rigid geometric structure to the intermediate representations as they progress *linearly* through the network (RA1) up to the final layer, where they undergo Neural Collapse.

----

## [0] Globally injective and bijective neural operators

**Authors**: *Takashi Furuya, Michael Puthawala, Matti Lassas, Maarten V. de Hoop*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html)

**Abstract**:

Recently there has been great interest in operator learning, where networks learn operators between function spaces from an essentially infinite-dimensional perspective. In this work we present results for when the operators learned by these networks are injective and surjective. As a warmup, we combine prior work in both the finite-dimensional ReLU and operator learning setting by giving sharp conditions under which ReLU layers with linear neural operators are injective. We then consider the case when the activation function is pointwise bijective and obtain sufficient conditions for the layer to be injective. We remark that this question, while trivial in the finite-rank setting, is subtler in the infinite-rank setting and is proven using tools from Fredholm theory. Next, we prove that our supplied injective neural operators are universal approximators and that their implementation, with finite-rank neural networks, are still injective. This ensures that injectivity is not 'lost' in the transcription from analytical operators to their finite-rank implementation with networks. Finally, we conclude with an increase in abstraction and consider general conditions when subnetworks, which may have many layers, are injective and surjective and provide an exact inversion from a 'linearization.â€™ This section uses general arguments from Fredholm theory and Leray-Schauder degree theory for non-linear integral equations to analyze the mapping properties of neural operators in function spaces. These results apply to subnetworks formed from the layers considered in this work, under natural conditions. We believe that our work has applications in Bayesian uncertainty quantification where injectivity enables likelihood estimation and in inverse problems where surjectivity and injectivity corresponds to existence and uniqueness of the solutions, respectively.

----

## [0] On the Convergence of CART under Sufficient Impurity Decrease Condition

**Authors**: *Rahul Mazumder, Haoyue Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b418964bafb4fdd9aef9017301323a8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b418964bafb4fdd9aef9017301323a8a-Abstract-Conference.html)

**Abstract**:

The decision tree is a flexible machine-learning model that finds its success in numerous applications. It is usually fitted in a recursively greedy manner using CART. In this paper, we study the convergence rate of CART under a regression setting. First, we prove an upper bound on the prediction error of CART under a sufficient impurity decrease (SID) condition \cite{chi2020asymptotic} -- our result is an improvement over the known result by \cite{chi2020asymptotic} under a similar assumption. We show via examples that this error bound cannot be further improved by more than a constant or a log factor. Second, we introduce a few easy-to-check sufficient conditions of the SID condition. In particular, we show that the SID condition can be satisfied by an additive model when the component functions satisfy a ``locally reverse Poincare inequality". We discuss a few familiar function classes in non-parametric estimation to demonstrate the usefulness of this conception.

----

## [0] PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis

**Authors**: *Ali TehraniJamsaz, Quazi Ishtiaque Mahmud, Le Chen, Nesreen K. Ahmed, Ali Jannesari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b41907dd4df5c60f86216b73fe0c7465-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b41907dd4df5c60f86216b73fe0c7465-Abstract-Conference.html)

**Abstract**:

The remarkable growth and significant success of machine learning have expanded its applications into programming languages and program analysis. However, a key challenge in adopting the latest machine learning methods is the representation of programming languages which has a direct impact on the ability of machine learning methods to reason about programs. The absence of numerical awareness, aggregate data structure information, and improper way of presenting variables in previous representation works have limited their performances.  To overcome the limitations and challenges of current program representations, we propose a novel graph-based program representation called PERFOGRAPH. PERFOGRAPH can capture numerical information and the aggregate data structure by introducing new nodes and edges. Furthermore, we propose an adapted embedding method to incorporate numerical awareness.These enhancements make PERFOGRAPH a highly flexible and scalable representation that can effectively capture programs' intricate dependencies and semantics. Consequently, it serves as a powerful tool for various applications such as program analysis, performance optimization, and parallelism discovery. Our experimental results demonstrate that PERFOGRAPH outperforms existing representations and sets new state-of-the-art results by reducing the error rate by 7.4% (AMD dataset) and 10% (NVIDIA dataset) in the well-known Device Mapping challenge. It also sets new state-of-the-art results in various performance optimization tasks like Parallelism Discovery and Numa and Prefetchers Configuration prediction.

----

## [0] Penalising the biases in norm regularisation enforces sparsity

**Authors**: *Etienne Boursier, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b444ad72520a5f5c467343be88e352ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b444ad72520a5f5c467343be88e352ed-Abstract-Conference.html)

**Abstract**:

Controlling the parameters' norm often yields good generalisation when training neural networks. Beyond simple intuitions, the relation between regularising parameters' norm and obtained estimators remains theoretically misunderstood. For one hidden ReLU layer networks with unidimensional data, this work shows the parameters' norm required to represent a function is given by the total variation of its second derivative, weighted by a $\sqrt{1+x^2}$ factor. Notably, this weighting factor disappears when the norm of bias terms is not regularised. The presence of this additional weighting factor is of utmost significance as it is shown to enforce the uniqueness and sparsity (in the number of kinks) of the minimal norm interpolator. Conversely, omitting the bias' norm  allows for non-sparse solutions.Penalising the bias terms in the regularisation, either explicitly or implicitly, thus leads to sparse estimators.

----

## [0] Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation

**Authors**: *SeungHwan An, Jong-June Jeon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b456a00e145ad56f6f251f79f8c8a7de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b456a00e145ad56f6f251f79f8c8a7de-Abstract-Conference.html)

**Abstract**:

The Gaussianity assumption has been consistently criticized as a main limitation of the Variational Autoencoder (VAE) despite its efficiency in computational modeling. In this paper, we propose a new approach that expands the model capacity (i.e., expressive power of distributional family) without sacrificing the computational advantages of the VAE framework. Our VAE model's decoder is composed of an infinite mixture of asymmetric Laplace distribution, which possesses general distribution fitting capabilities for continuous variables. Our model is represented by a special form of a nonparametric M-estimator for estimating general quantile functions, and we theoretically establish the relevance between the proposed model and quantile estimation. We apply the proposed model to synthetic data generation, and particularly, our model demonstrates superiority in easily adjusting the level of data privacy.

----

## [0] Optimistic Meta-Gradients

**Authors**: *Sebastian Flennerhag, Tom Zahavy, Brendan O'Donoghue, Hado Philip van Hasselt, András György, Satinder Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b46bc1449205888e1883f692aff1a252-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b46bc1449205888e1883f692aff1a252-Abstract-Conference.html)

**Abstract**:

We study the connection between gradient-based meta-learning and convex optimisation. We observe that gradient descent with momentum is a special case of meta-gradients, and building on recent results in optimisation, we prove convergence rates for meta learning in the single task setting. While a meta-learned update rule can yield faster convergence up to constant factor, it is not sufficient for acceleration. Instead, some form of optimism is required. We show that optimism in meta-learning can be captured through the recently proposed Bootstrapped Meta-Gradient (Flennerhag et. al., 2022) method, providing deeper insight into its underlying mechanics.

----

## [0] Norm-guided latent space exploration for text-to-image generation

**Authors**: *Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, Gal Chechik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b49213694c3e752252d62ca360b72a36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b49213694c3e752252d62ca360b72a36-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models show great potential in synthesizing a large variety of concepts in new compositions and scenarios. However, the latent space of initial seeds is still not well understood and its structure was shown to impact the generation of various concepts. Specifically, simple operations like interpolation and finding the centroid of a set of seeds perform poorly when using standard Euclidean or spherical metrics in the latent space. This paper makes the observation that, in current training procedures, diffusion models observed inputs with a narrow range of norm values. This has strong implications for methods that rely on seed manipulation for image generation, with applications to few-shot and long-tail learning tasks. To address this issue, we propose a novel method for interpolating between two seeds and demonstrate that it defines a new non-Euclidean metric that takes into account a norm-based prior on seeds. We describe a simple yet efficient algorithm for approximating this interpolation procedure and use it to further define centroids in the latent seed space. We show that our new interpolation and centroid techniques significantly enhance the generation of rare concept images. This further leads to state-of-the-art performance on few-shot and long-tail benchmarks, improving prior approaches in terms of generation speed, image quality, and semantic content.

----

## [0] Scale Alone Does not Improve Mechanistic Interpretability in Vision Models

**Authors**: *Roland S. Zimmermann, Thomas Klein, Wieland Brendel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4aadf04d6fde46346db455402860708-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4aadf04d6fde46346db455402860708-Abstract-Conference.html)

**Abstract**:

In light of the recent widespread adoption of AI systems, understanding the internal information processing of neural networks has become increasingly critical. Most recently, machine vision has seen remarkable progress by scaling neural networks to unprecedented levels in dataset and model size. We here ask whether this extraordinary increase in scale also positively impacts the field of mechanistic interpretability. In other words, has our understanding of the inner workings of scaled neural networks improved as well? We use a psychophysical paradigm to quantify one form of mechanistic interpretability for a diverse suite of nine models and find no scaling effect for interpretability - neither for model nor dataset size. Specifically, none of the investigated state-of-the-art models are easier to interpret than the GoogLeNet model from almost a decade ago. Latest-generation vision models appear even less interpretable than older architectures, hinting at a regression rather than improvement, with modern models sacrificing interpretability for accuracy. These results highlight the need for models explicitly designed to be mechanistically interpretable and the need for more helpful interpretability methods to increase our understanding of networks at an atomic level. We release a dataset containing more than 130'000 human responses from our psychophysical evaluation of 767 units across nine models. This dataset facilitates research on automated instead of human-based interpretability evaluations, which can ultimately be leveraged to directly optimize the mechanistic interpretability of models.

----

## [0] The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications

**Authors**: *Mirac Suzgun, Luke Melas-Kyriazi, Suproteem K. Sarkar, Scott Duke Kominers, Stuart M. Shieber*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4b02a09f2e6ad29fdbeb1386d68f4c4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4b02a09f2e6ad29fdbeb1386d68f4c4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Innovation is a major driver of economic and social development, and information about many kinds of innovation is embedded in semi-structured data from patents and patent applications. Though the impact and novelty of innovations expressed in patent data are difficult to measure through traditional means, machine learning offers a promising set of techniques for evaluating novelty, summarizing contributions, and embedding semantics. In this paper, we introduce the Harvard USPTO Patent Dataset (HUPD), a large-scale, well-structured, and multi-purpose corpus of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions of patent applications, not the final versions of granted patents, allowing us to study patentability at the time of filing using NLP methods for the first time. It is also novel in its inclusion of rich structured data alongside the text of patent filings: By providing each applicationâ€™s metadata along with all of its text fields, HUPD enables researchers to perform new sets of NLP tasks that leverage variation in structured covariates. As a case study on the types of research HUPD makes possible, we introduce a new task to the NLP community -- patent acceptance prediction. We additionally show the structured metadata provided in HUPD allows us to conduct explicit studies of concept shifts for this task. We find that performance on patent acceptance prediction decays when models trained in one context are evaluated on different innovation categories and over time. Finally, we demonstrate how HUPD can be used for three additional tasks: Multi-class classification of patent subject areas, language modeling, and abstractive summarization. Put together, our publicly-available dataset aims to advance research extending language and classification models to diverse and dynamic real-world data distributions.

----

## [0] MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

**Authors**: *Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html)

**Abstract**:

Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.

----

## [0] Minimax Risks and Optimal Procedures for Estimation under Functional Local Differential Privacy

**Authors**: *Bonwoo Lee, Jeongyoun Ahn, Cheolwoo Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4dde7f1bc45bf9c0fda8db8f272b758-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4dde7f1bc45bf9c0fda8db8f272b758-Abstract-Conference.html)

**Abstract**:

As concerns about data privacy continue to grow, differential privacy (DP) has emerged as a fundamental concept that aims to guarantee privacy by ensuring individuals' indistinguishability in data analysis. Local differential privacy (LDP) is a rigorous type of DP that requires individual data to be privatized before being sent to the collector, thus removing the need for a trusted third party to collect data. Among the numerous (L)DP-based approaches, functional DP has gained considerable attention in the DP community because it connects DP to statistical decision-making by formulating it as a hypothesis-testing problem and also exhibits Gaussian-related properties. However, the utility of privatized data is generally lower than that of non-private data, prompting research into optimal mechanisms that maximize the statistical utility for given privacy constraints. In this study, we investigate how functional LDP preserves the statistical utility by analyzing minimax risks of univariate mean estimation as well as nonparametric density estimation. We leverage the contraction property of functional LDP mechanisms and classical information-theoretical bounds to derive private minimax lower bounds. Our theoretical study reveals that it is possible to establish an interpretable, continuous balance between the statistical utility and privacy level, which has not been achieved under the $\epsilon$-LDP framework. Furthermore, we suggest minimax optimal mechanisms based on Gaussian LDP (a type of functional LDP) that achieve the minimax upper bounds and show via a numerical study that they are superior to the counterparts derived under $\epsilon$-LDP. The theoretical and empirical findings of this work suggest that Gaussian LDP should be considered a reliable standard for LDP.

----

## [0] Switching Autoregressive Low-rank Tensor Models

**Authors**: *Hyun Dong Lee, Andrew Warrington, Joshua I. Glaser, Scott W. Linderman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4e3fea367538ea6b1b5ba6ebf5c39a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4e3fea367538ea6b1b5ba6ebf5c39a8-Abstract-Conference.html)

**Abstract**:

An important problem in time-series analysis is modeling systems with time-varying dynamics.  Probabilistic models with joint continuous and discrete latent states offer interpretable, efficient, and experimentally useful descriptions of such data.  Commonly used models include autoregressive hidden Markov models (ARHMMs) and switching linear dynamical systems (SLDSs), each with its own advantages and disadvantages.  ARHMMs permit exact inference and easy parameter estimation, but are parameter intensive when modeling long dependencies, and hence are prone to overfitting.  In contrast, SLDSs can capture long-range dependencies in a parameter efficient way through Markovian latent dynamics, but present an intractable likelihood and a challenging parameter estimation task.  In this paper, we propose switching autoregressive low-rank tensor SALT models, which retain the advantages of both approaches while ameliorating the weaknesses.  SALT parameterizes the tensor of an ARHMM with a low-rank factorization to control the number of parameters and allow longer range dependencies without overfitting.  We prove theoretical and discuss practical connections between SALT, linear dynamical systems, and SLDSs.  We empirically demonstrate quantitative advantages of SALT models on a range of simulated and real prediction tasks, including behavioral and neural datasets.  Furthermore, the learned low-rank tensor provides novel insights into temporal dependencies within each discrete state.

----

## [0] From Tempered to Benign Overfitting in ReLU Neural Networks

**Authors**: *Guy Kornowski, Gilad Yehudai, Ohad Shamir*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b52e8c6c1a798fed53ac2e6a5e23ddc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b52e8c6c1a798fed53ac2e6a5e23ddc8-Abstract-Conference.html)

**Abstract**:

Overparameterized neural networks (NNs) are observed to generalize well even when trained to perfectly fit noisy data. This phenomenon motivated a large body of work on "benign overfitting", where interpolating predictors achieve near-optimal performance. Recently, it was conjectured and empirically observed that the behavior of NNs is often better described as "tempered overfitting", where the performance is non-optimal yet also non-trivial, and degrades as a function of the noise level. However, a theoretical justification of this claim for non-linear NNs has been lacking so far. In this work, we provide several results that aim at bridging these complementing views. We study a simple classification setting with 2-layer ReLU NNs, and prove that under various assumptions, the type of overfitting transitions from tempered in the extreme case of one-dimensional data, to benign in high dimensions. Thus, we show that the input dimension has a crucial role on the overfitting profile in this setting, which we also validate empirically for intermediate dimensions. Overall, our results shed light on the intricate connections between the dimension, sample size, architecture and training algorithm on the one hand, and the type of resulting overfitting on the other hand.

----

## [0] Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images

**Authors**: *Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b54d1757c190ba20dbc4f9e4a2f54149-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b54d1757c190ba20dbc4f9e4a2f54149-Abstract-Conference.html)

**Abstract**:

Watermarking the outputs of generative models is a crucial technique for tracing copyright and preventing potential harm from AI-generated content. In this paper, we introduce a novel technique called Tree-Ring Watermarking that robustly fingerprints diffusion model outputs.  Unlike existing methods that perform post-hoc modifications to images after sampling, Tree-Ring Watermarking subtly influences the entire sampling process, resulting in a model fingerprint that is invisible to humans. The watermark embeds a pattern into the initial noise vector used for sampling. These patterns are structured in Fourier space so that they are invariant to convolutions, crops, dilations, flips, and rotations.  After image generation, the watermark signal is detected by inverting the diffusion process to retrieve the noise vector, which is then checked for the embedded signal.  We demonstrate that this technique can be easily applied to arbitrary diffusion models, including text-conditioned Stable Diffusion, as a plug-in with negligible loss in FID. Our watermark is semantically hidden in the image space and is far more robust than watermarking alternatives that are currently deployed.

----

## [0] MVDoppler: Unleashing the Power of Multi-View Doppler for MicroMotion-based Gait Classification

**Authors**: *Soheil Hor, Shubo Yang, Jaeho Choi, Amin Arbabian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5727c1bab903e0ff21cec84a9a7f5a6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5727c1bab903e0ff21cec84a9a7f5a6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modern perception systems rely heavily on high-resolution cameras, LiDARs, and advanced deep neural networks, enabling exceptional performance across various applications. However, these optical systems predominantly depend on geometric features and shapes of objects, which can be challenging to capture in long-range perception applications. To overcome this limitation, alternative approaches such as Doppler-based perception using high-resolution radars have been proposed. Doppler-based systems are capable of measuring micro-motions of targets remotely and with very high precision. When compared to geometric features, the resolution of micro-motion features exhibits significantly greater resilience to the influence of distance. However, the true potential of Doppler-based perception has yet to be fully realized due to several factors. These include the unintuitive nature of Doppler signals, the limited availability of public Doppler datasets, and the current datasets' inability to capture the specific co-factors that are unique to Doppler-based perception, such as the effect of the radar's observation angle and the target's motion trajectory.This paper introduces a new large multi-view Doppler dataset together with baseline perception models for micro-motion-based gait analysis and classification. The dataset captures the impact of the subject's walking trajectory and radar's observation angle on the classification performance. Additionally, baseline multi-view data fusion techniques are provided to mitigate these effects. This work demonstrates that sub-second micro-motion snapshots can be sufficient for reliable detection of hand movement patterns and even changes in a pedestrian's walking behavior when distracted by their phone. Overall, this research not only showcases the potential of Doppler-based perception, but also offers valuable solutions to tackle its fundamental challenges.

----

## [0] Understanding and Improving Ensemble Adversarial Defense

**Authors**: *Yian Deng, Tingting Mu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html)

**Abstract**:

The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense,  demonstrating a provable 0-1 loss reduction on challenging sample sets in adversarial defense scenarios. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques,  iGAT is capable of boosting their performance by up to 17\%  evaluated using  CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.

----

## [0] Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions

**Authors**: *Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5a801e6bc4f4ffa3e6786518a324488-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5a801e6bc4f4ffa3e6786518a324488-Abstract-Conference.html)

**Abstract**:

Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training  (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.

----

## [0] A Massive Scale Semantic Similarity Dataset of Historical English

**Authors**: *Emily Silcock, Abhishek Arora, Melissa Dell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5ae304ecd18c5d4ac4a011ab086ba23-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5ae304ecd18c5d4ac4a011ab086ba23-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in the presence of substantial noise and abridgement. The headlines of reproduced articles form positive semantic similarity pairs. The resulting publicly available HEADLINES dataset is significantly larger than most existing semantic similarity datasets and covers a much longer span of time. It will facilitate the application of contrastively trained semantic similarity models to a variety of tasks, including the study of semantic change across space and time.

----

## [0] Joint Prompt Optimization of Stacked LLMs using Variational Inference

**Authors**: *Alessandro Sordoni, Eric Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini, Friederike Niedtner, Nicolas Le Roux*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) can be seen as atomic units of computation mapping sequences to a distribution over sequences. Thus, they can be seen as stochastic language layers in a language network, where the learnable parameters are the natural language prompts at each layer. By stacking two such layers and feeding the output of one layer to the next, we obtain a Deep Language Network (DLN). We first show how to effectively perform prompt optimization for a 1-Layer language network (DLN-1). Then, we present an extension that applies to 2-layer DLNs (DLN-2), where two prompts must be learned. The key idea is to consider the output of the first layer as a latent variable, which requires inference, and prompts to be learned as the parameters of the generative distribution. We first test the effectiveness of DLN-1 in multiple reasoning and natural language understanding tasks. Then, we show that DLN-2 can reach higher performance than a single layer, showing promise that we might reach comparable performance to GPT-4, even when each LLM in the network is smaller and less powerful.

----

## [0] On the Properties of Kullback-Leibler Divergence Between Multivariate Gaussian Distributions

**Authors**: *Yufeng Zhang, Jialu Pan, Li Ken Li, Wanwei Liu, Zhenbang Chen, Xinwang Liu, Ji Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5b4d92374323c53c24bbbc8ee0e715c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5b4d92374323c53c24bbbc8ee0e715c-Abstract-Conference.html)

**Abstract**:

Kullback-Leibler (KL) divergence is one of the most important measures to calculate the difference between probability distributions. In this paper, we theoretically study several properties of KL divergence between multivariate Gaussian distributions. Firstly, for any two $n$-dimensional Gaussian distributions $\mathcal{N}_1$ and $\mathcal{N}_2$, we prove that when $KL(\mathcal{N}_2||\mathcal{N}_1)\leq \varepsilon\ (\varepsilon>0)$ the supremum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ is $(1/2)\left((-W_{0}(-e^{-(1+2\varepsilon)}))^{-1}+\log(-W_{0}(-e^{-(1+2\varepsilon)})) -1 \right)$, where $W_0$ is the principal branch of Lambert $W$ function.For small $\varepsilon$, the supremum is $\varepsilon + 2\varepsilon^{1.5} + O(\varepsilon^2)$. This quantifies the approximate symmetry of small KL divergence between Gaussian distributions. We further derive the infimum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ when $KL(\mathcal{N}_2||\mathcal{N}_1)\geq M\ (M>0)$. We give the conditions when the supremum and infimum can be attained. Secondly, for any three $n$-dimensional Gaussian distributions $\mathcal{N}_1$, $\mathcal{N}_2$, and $\mathcal{N}_3$, we theoretically show that an upper bound of $KL(\mathcal{N}_1||\mathcal{N}_3)$ is $3\varepsilon_1+3\varepsilon_2+2\sqrt{\varepsilon_1\varepsilon_2}+o(\varepsilon_1)+o(\varepsilon_2)$ when $KL(\mathcal{N}_1||\mathcal{N}_2)\leq \varepsilon_1$ and $KL(\mathcal{N}_2||\mathcal{N}_3)\leq \varepsilon_2$ ($\varepsilon_1,\varepsilon_2\ge 0$). This reveals that KL divergence between Gaussian distributions follows a relaxed triangle inequality. Note that, all these bounds in the theorems presented in this work are independent of the dimension $n$. Finally, we discuss several applications of our theories in deep learning, reinforcement learning, and sample complexity research.

----

## [0] Implicit Bias of (Stochastic) Gradient Descent for Rank-1 Linear Neural Network

**Authors**: *Bochen Lyu, Zhanxing Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)

**Abstract**:

Studying the implicit bias of gradient descent (GD) and stochastic gradient descent (SGD) is critical to unveil the underlying mechanism of deep learning. Unfortunately, even for standard linear networks in regression setting, a comprehensive characterization of the implicit bias is still an open problem. This paper proposes to investigate a new proxy model of standard linear network,  rank-1 linear network, where each weight matrix is parameterized as a rank-1 form. For over-parameterized regression problem, we  precisely analyze the implicit bias of GD and SGD---by identifying a “potential” function such that GD converges to its minimizer constrained by zero training error (i.e., interpolation solution), and further characterizing the role of the noise introduced by  SGD in perturbing the form of this potential. Our results explicitly connect the depth of the network and the initialization with the implicit bias of GD and SGD. Furthermore, we emphasize a new implicit bias of SGD jointly induced by stochasticity and over-parameterization, which can reduce the dependence of the SGD's solution on the initialization. Our findings regarding the implicit bias are different from that of a recently popular model, the diagonal linear network. We highlight that the induced bias of our rank-1 model is more consistent with standard linear network while the diagonal one is not. This suggests that the proposed rank-1 linear network might be a plausible proxy for standard linear net.

----

## [0] AdaPlanner: Adaptive Planning from Feedback with Language Models

**Authors**: *Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have recently demonstrated the potential in acting as autonomous agents for sequential decision-making tasks. However, most existing methods either take actions greedily without planning or rely on static plans that are not adaptable to environmental feedback. Consequently, the sequential decision-making performance of LLM agents degenerates with problem complexity and plan horizons increase. We propose a closed-loop approach, AdaPlanner, which allows the LLM agent to refine its self-generated plan adaptively in response to environmental feedback. In AdaPlanner, the LLM agent adaptively refines its plan from feedback with both in-plan and out-of-plan refinement strategies. To mitigate hallucination, we develop a code-style LLM prompt structure that facilitates plan generation across a variety of tasks, environments, and agent capabilities. Furthermore, we propose a skill discovery mechanism that leverages successful plans as few-shot exemplars, enabling the agent to plan and refine with fewer task demonstrations. Our experiments in the ALFWorld and MiniWoB++ environments demonstrate that AdaPlanner outperforms state-of-the-art baselines by 3.73% and 4.11% while utilizing 2x and 600x fewer samples, respectively. The implementation of AdaPlanner is available at https://github.com/haotiansun14/AdaPlanner.

----

## [0] Fairness Aware Counterfactuals for Subgroups

**Authors**: *Loukas Kavouras, Konstantinos Tsopelas, Giorgos Giannopoulos, Dimitris Sacharidis, Eleni Psaroudaki, Nikolaos Theologitis, Dimitrios Rontogiannis, Dimitris Fotakis, Ioannis Z. Emiris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html)

**Abstract**:

In this work, we present Fairness Aware Counterfactuals for Subgroups (FACTS), a framework for auditing subgroup fairness through counterfactual explanations. We start with revisiting (and generalizing) existing notions and introducing new, more refined notions of subgroup fairness. We aim to (a) formulate different aspects of the difficulty of individuals in certain subgroups to achieve recourse, i.e. receive the desired outcome, either at the micro level, considering members of the subgroup individually, or at the macro level, considering the subgroup as a whole, and (b) introduce notions of subgroup fairness that are robust, if not totally oblivious, to the cost of achieving recourse. We accompany these notions with an efficient, model-agnostic, highly parameterizable, and explainable framework for evaluating subgroup fairness. We demonstrate the advantages, the wide applicability, and the efficiency of our approach through a thorough experimental evaluation on different benchmark datasets.

----

## [0] ProteinShake: Building datasets and benchmarks for deep learning on protein structures

**Authors**: *Tim Kucera, Carlos G. Oliver, Dexiong Chen, Karsten M. Borgwardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6167294ed3d6fc61e11e1592ce5cb77-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6167294ed3d6fc61e11e1592ce5cb77-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present ProteinShake, a Python software package that simplifies datasetcreation and model evaluation for deep learning on protein structures. Users cancreate custom datasets or load an extensive set of pre-processed datasets fromthe Protein Data Bank (PDB) and AlphaFoldDB. Each dataset is associated withprediction tasks and evaluation functions covering a broad array of biologicalchallenges. A benchmark on these tasks shows that pre-training almost alwaysimproves performance, the optimal data modality (graphs, voxel grids, or pointclouds) is task-dependent, and models struggle to generalize to new structures.ProteinShake makes protein structure data easily accessible and comparisonamong models straightforward, providing challenging benchmark settings withreal-world implications.ProteinShake is available at: https://proteinshake.ai

----

## [0] Lovász Principle for Unsupervised Graph Representation Learning

**Authors**: *Ziheng Sun, Chris Ding, Jicong Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b61da4f02b271cb7b5e3d538e2b78fb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b61da4f02b271cb7b5e3d538e2b78fb9-Abstract-Conference.html)

**Abstract**:

This paper focuses on graph-level representation learning that aims to represent graphs as vectors that can be directly utilized in downstream tasks such as graph classification. We propose a novel graph-level representation learning principle called Lovász principle, which is motivated by the Lovász number in graph theory. The Lovász number of a graph is a real number that is an upper bound for graph Shannon capacity and is strongly connected with various global characteristics of the graph. Specifically, we show that the handle vector for computing the Lovász number is potentially a suitable choice for graph representation, as it captures a graph's global properties, though a direct application of the handle vector is difficult and problematic. We propose to use neural networks to address the problems and hence provide the Lovász principle. Moreover, we propose an enhanced Lovász principle that is able to exploit the subgraph Lovász numbers directly and efficiently. The experiments demonstrate that our Lovász principles achieve competitive performance compared to the baselines in unsupervised and semi-supervised graph-level representation learning tasks. The code of our Lovász principles is publicly available on GitHub.

----

## [0] ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation

**Authors**: *Chenyang Le, Yao Qian, Long Zhou, Shujie Liu, Yanmin Qian, Michael Zeng, Xuedong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6262f7a34e5d641cdb3d33dc9ad1a5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6262f7a34e5d641cdb3d33dc9ad1a5a-Abstract-Conference.html)

**Abstract**:

Joint speech-language training is challenging due to the large demand for training data and GPU consumption, as well as the modality gap between speech and language. We present ComSL, a speech-language model built atop a composite architecture of public pre-trained speech-only and language-only models and optimized data-efficiently for spoken language tasks. Particularly, we propose to incorporate cross-modality learning into transfer learning and conduct them simultaneously for downstream tasks in a multi-task learning manner. Our approach has demonstrated effectiveness in end-to-end speech-to-text translation tasks, achieving a new state-of-the-art average BLEU score of 31.5 on the multilingual speech to English text translation task for 21 languages, as measured on the public CoVoST2 evaluation set.

----

## [0] Reverse Engineering Self-Supervised Learning

**Authors**: *Ido Ben-Shaul, Ravid Shwartz-Ziv, Tomer Galanti, Shai Dekel, Yann LeCun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b63ad8c24354b0e5bcb7aea16490beab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b63ad8c24354b0e5bcb7aea16490beab-Abstract-Conference.html)

**Abstract**:

Understanding the learned representation and underlying mechanisms of Self-Supervised Learning (SSL) often poses a challenge. In this paper, we ‘reverse engineer’ SSL, conducting an in-depth empirical analysis of its learned internal representations, encompassing diverse models, architectures, and hyperparameters. Our study reveals an intriguing process within the SSL training: an inherent facilitation of semantic label-based clustering, which is surprisingly driven by the regularization component of the SSL objective. This clustering not only enhances downstream classification, but also compresses the information. We further illustrate that the alignment of the SSL-trained representation is more pronounced with semantic classes rather than random functions. Remarkably, the learned representations align with semantic classes across various hierarchical levels, with this alignment intensifying when going deeper into the network. This ‘reverse engineering’ approach provides valuable insights into the inner mechanism of SSL and their influences on the performance across different class sets.

----

## [0] DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning

**Authors**: *Alexander H. Liu, Heng-Jui Chang, Michael Auli, Wei-Ning Hsu, Jim Glass*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6404bf461c3c3186bdf5f55756af908-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6404bf461c3c3186bdf5f55756af908-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce self-distillation and online clustering for self-supervised speech representation learning (DinoSR) which combines masked language modeling, self-distillation, and online clustering. We show that these concepts complement each other and result in a strong representation learning model for speech. DinoSR first extracts contextualized embeddings from the input audio with a teacher network, then runs an online clustering system on the embeddings to yield a machine-discovered phone inventory, and finally uses the discretized tokens to guide a student network. We show that DinoSR surpasses previous state-of-the-art performance in several downstream tasks, and provide a detailed analysis of the model and the learned discrete units.

----

## [0] 4M: Massively Multimodal Masked Modeling

**Authors**: *David Mizrahi, Roman Bachmann, Oguzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6446566965fa38e183650728ab70318-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6446566965fa38e183650728ab70318-Abstract-Conference.html)

**Abstract**:

Current machine learning models for vision are often highly specialized and limited to a single modality and task. In contrast, recent large language models exhibit a wide range of capabilities, hinting at a possibility for similarly versatile models in computer vision.In this paper, we take a step in this direction and propose a multimodal training scheme called 4M. It consists of training a single unified Transformer encoder-decoder using a masked modeling objective across a wide range of input/output modalities  â€“ including text, images, geometric, and semantic modalities, as well as neural network feature maps. 4M achieves scalability by unifying the representation space of all modalities through mapping them into discrete tokens and performing multimodal masked modeling on a small randomized subset of tokens.4M leads to models that exhibit several key capabilities: (1) they can perform a diverse set of vision tasks out of the box, (2) they excel when fine-tuned for unseen downstream tasks or new input modalities, and (3) they can function as a generative model that can be conditioned on arbitrary modalities, enabling a wide variety of expressive multimodal editing capabilities with remarkable flexibility.Through experimental analyses, we demonstrate the potential of 4M for training versatile and scalable foundation models for vision tasks, setting the stage for further exploration in multimodal learning for vision and other domains.

----

## [0] Non-Rigid Shape Registration via Deep Functional Maps Prior

**Authors**: *Puhua Jiang, Mingze Sun, Ruqi Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a learning-based framework for non-rigid shape registra- tion without correspondence supervision. Traditional shape registration techniques typically rely on correspondences induced by extrinsic proximity, therefore can fail in the presence of large intrinsic deformations. Spectral mapping methods overcome this challenge by embedding shapes into, geometric or learned, high- dimensional spaces, where shapes are easier to align. However, due to the dependency on abstract, non-linear embedding schemes, the latter can be vulnerable with respect to perturbed or alien input. In light of this, our framework takes the best of both worlds. Namely, we deform source mesh towards the target point cloud, guided by correspondences induced by high-dimensional embeddings learned from deep functional maps (DFM). In particular, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Moreover, in order to alleviate the requirement of extrinsically aligned input, we train an orientation regressor on a set of aligned synthetic shapes independent of the training shapes for DFM. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic defor- mations, in which case neither traditional registration methods nor intrinsic methods work. The code is available at https://github.com/rqhuang88/DFR.

----

## [0] Game Solving with Online Fine-Tuning

**Authors**: *Ti-Rong Wu, Hung Guei, Ting-Han Wei, Chung-Chin Shih, Jui-Te Chin, I-Chen Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b663eb1512ce6c268e3e56f34c6d2959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b663eb1512ce6c268e3e56f34c6d2959-Abstract-Conference.html)

**Abstract**:

Game solving is a similar, yet more difficult task than mastering a game. Solving a game typically means to find the game-theoretic value (outcome given optimal play), and optionally a full strategy to follow in order to achieve that outcome. The AlphaZero algorithm has demonstrated super-human level play, and its powerful policy and value predictions have also served as heuristics in game solving. However, to solve a game and obtain a full strategy, a winning response must be found for all possible moves by the losing player. This includes very poor lines of play from the losing side, for which the AlphaZero self-play process will not encounter. AlphaZero-based heuristics can be highly inaccurate when evaluating these out-of-distribution positions, which occur throughout the entire search. To address this issue, this paper investigates applying online fine-tuning while searching and proposes two methods to learn tailor-designed heuristics for game solving. Our experiments show that using online fine-tuning can solve a series of challenging 7x7 Killall-Go problems, using only 23.54\% of computation time compared to the baseline without online fine-tuning. Results suggest that the savings scale with problem size. Our method can further be extended to any tree search algorithm for problem solving. Our code is available at https://rlg.iis.sinica.edu.tw/papers/neurips2023-online-fine-tuning-solver.

----

## [0] Beyond probability partitions: Calibrating neural networks with semantic aware grouping

**Authors**: *Jia-Qi Yang, De-Chuan Zhan, Le Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html)

**Abstract**:

Research has shown that deep networks tend to be overly optimistic about their predictions, leading to an underestimation of prediction errors. Due to the limited nature of data, existing studies have proposed various methods based on model prediction probabilities to bin the data and evaluate calibration error. We propose a more generalized definition of calibration error called Partitioned Calibration Error (PCE), revealing that the key difference among these calibration error metrics lies in how the data space is partitioned. We put forth an intuitive proposition that an accurate model should be calibrated across any partition, suggesting that the input space partitioning can extend beyond just the partitioning of prediction probabilities, and include partitions directly related to the input. Through semantic-related partitioning functions, we demonstrate that the relationship between model accuracy and calibration lies in the granularity of the partitioning function. This highlights the importance of partitioning criteria for training a calibrated and accurate model. To validate the aforementioned analysis, we propose a method that involves jointly learning a semantic aware grouping function based on deep model features and logits to partition the data space into subsets. Subsequently, a separate calibration function is learned for each subset. Experimental results demonstrate that our approach achieves significant performance improvements across multiple datasets and network architectures, thus highlighting the importance of the partitioning function for calibration.

----

## [0] Identifiable Contrastive Learning with Automatic Feature Importance Discovery

**Authors**: *Qi Zhang, Yifei Wang, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html)

**Abstract**:

Existing contrastive learning methods rely on pairwise sample contrast $z_x^\top z_{x'}$ to learn data representations, but the learned features often lack clear interpretability from a human perspective. Theoretically, it lacks feature identifiability and different initialization may lead to totally different features. In this paper, we study a new method named tri-factor contrastive learning (triCL) that involves a 3-factor contrast in the form of $z_x^\top S z_{x'}$, where $S=\text{diag}(s_1,\dots,s_k)$ is a learnable diagonal matrix that automatically captures the importance of each feature. We show that by this simple extension, triCL can not only obtain identifiable features that eliminate randomness but also obtain more interpretable features that are ordered according to the importance matrix $S$. We show that features with high importance have nice interpretability by capturing common classwise features, and obtain superior performance when evaluated for image retrieval using a few features. The proposed triCL objective is general and can be applied to different contrastive learning methods like SimCLR and CLIP. We believe that it is a better alternative to existing 2-factor contrastive learning by improving its identifiability and interpretability with minimal overhead. Code is available at https://github.com/PKU-ML/Tri-factor-Contrastive-Learning.

----

## [0] Revisiting Out-of-distribution Robustness in NLP: Benchmarks, Analysis, and LLMs Evaluations

**Authors**: *Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper reexamines the research on out-of-distribution (OOD) robustness in the field of NLP. We find that the distribution shift settings in previous studies commonly lack adequate challenges, hindering the accurate evaluation of OOD robustness. To address these issues, we propose a benchmark construction protocol that ensures clear differentiation and challenging distribution shifts. Then we introduceBOSS, a Benchmark suite for Out-of-distribution robustneSS evaluation covering 5 tasks and 20 datasets. Based on BOSS, we conduct a series of experiments on pretrained language models for analysis and evaluation of OOD robustness. First, for vanilla fine-tuning, we examine the relationship between in-distribution (ID) and OOD performance. We identify three typical types that unveil the inner learningmechanism, which could potentially facilitate the forecasting of OOD robustness, correlating with the advancements on ID datasets. Then, we evaluate 5 classic methods on BOSS and find that, despite exhibiting some effectiveness in specific cases, they do not offer significant improvement compared to vanilla fine-tuning. Further, we evaluate 5 LLMs with various adaptation paradigms and find that when sufficient ID data is available, fine-tuning domain-specific models outperform LLMs on ID examples significantly. However, in the case of OOD instances, prioritizing LLMs with in-context learning yields better results. We identify that both fine-tuned small models and LLMs face challenges in effectively addressing downstream tasks. The code is public at https://github.com/lifan-yuan/OOD_NLP.

----

## [0] Towards Consistent Video Editing with Text-to-Image Diffusion Models

**Authors**: *Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6c05f8254a00709e16fb0fdaae56cd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6c05f8254a00709e16fb0fdaae56cd8-Abstract-Conference.html)

**Abstract**:

Existing works have advanced Text-to-Image (TTI) diffusion models for video editing in a one-shot learning manner. Despite their low requirements of data and computation, these methods might produce results of unsatisfied consistency with text prompt as well as temporal sequence, limiting their applications in the real world. In this paper, we propose to address the above issues with a novel EI$^2$ model towards Enhancing vIdeo Editing consIstency of TTI-based frameworks. Specifically, we analyze and find that the inconsistent problem is caused by newly added modules into TTI models for learning temporal information. These modules lead to covariate shift in the feature space, which harms the editing capability. Thus, we design EI$^2$ to tackle the above drawbacks with two classical modules: Shift-restricted Temporal Attention Module (STAM) and Fine-coarse Frame Attention Module (FFAM). First, through theoretical analysis, we demonstrate that covariate shift is highly related to Layer Normalization, thus STAM employs a Instance Centering layer replacing it to preserve the distribution of temporal features.  In addition, STAM employs an attention layer with normalized mapping to transform temporal features while constraining the variance shift.  As the second part, we incorporate STAM with a novel FFAM, which efficiently leverages fine-coarse spatial information of overall frames to further enhance temporal consistency. Extensive experiments demonstrate the superiority of the proposed EI$^2$ model.

----

## [0] Federated Spectral Clustering via Secure Similarity Reconstruction

**Authors**: *Dong Qiao, Chris Ding, Jicong Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6cd2650926d332c86a84c48529cc421-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6cd2650926d332c86a84c48529cc421-Abstract-Conference.html)

**Abstract**:

Federated learning has a significant advantage in protecting information privacy. Many scholars proposed various secure learning methods within the framework of federated learning but the study on secure federated unsupervised learning especially clustering is limited. We in this work propose a secure kernelized factorization method for federated spectral clustering on distributed dataset. The method is non-trivial because the kernel or similarity matrix for spectral clustering is computed by data pairs, which violates the principle of privacy protection. Our method implicitly constructs an approximation for the kernel matrix on distributed data such that we can perform spectral clustering under the constraint of privacy protection. We provide a convergence guarantee of the optimization algorithm, reconstruction error bounds of the Gaussian kernel matrix, and the sufficient condition of correct clustering of our method. We also present some results of differential privacy. Numerical results on synthetic and real datasets demonstrate that the proposed method is efficient and accurate in comparison to the baselines.

----

## [0] CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation

**Authors**: *Yexiong Lin, Yu Yao, Xiaolong Shi, Mingming Gong, Xu Shen, Dong Xu, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6d67c380f8bde2adc4247d0036c0c73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6d67c380f8bde2adc4247d0036c0c73-Abstract-Conference.html)

**Abstract**:

Label noise widely exists in large-scale image datasets. To mitigate the side effects of label noise, state-of-the-art methods focus on selecting confident examples by leveraging semi-supervised learning. Existing research shows that the ability to extract hard confident examples, which are close to the decision boundary, significantly influences the generalization ability of the learned classifier.In this paper, we find that a key reason for some hard examples being close to the decision boundary is due to the entanglement of style factors with content factors. The hard examples become more discriminative when we focus solely on content factors, such as semantic information, while ignoring style factors. Nonetheless, given only noisy data, content factors are not directly observed and have to be inferred.To tackle the problem of inferring content factors for classification when learning with noisy labels, our objective is to ensure that the content factors of all examples in the same underlying clean class remain unchanged as their style information changes.To achieve this, we utilize different data augmentation techniques to alter the styles while regularizing content factors based on some confident examples. By training existing methods with our inferred content factors, CS-Isolate proves their effectiveness in learning hard examples on benchmark datasets. The implementation is available at https://github.com/tmllab/2023NeurIPSCS-isolate.

----

## [0] Conditional independence testing under misspecified inductive biases

**Authors**: *Felipe Maia Polo, Yuekai Sun, Moulinath Banerjee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6f2b16abf590e80c9df30bb5f8e2b7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6f2b16abf590e80c9df30bb5f8e2b7d-Abstract-Conference.html)

**Abstract**:

Conditional independence (CI) testing is a fundamental and challenging task in modern statistics and machine learning. Many modern methods for CI testing rely on powerful supervised learning methods to learn regression functions or Bayes predictors as an intermediate step; we refer to this class of tests as regression-based tests. Although these methods are guaranteed to control Type-I error when the supervised learning methods accurately estimate the regression functions or Bayes predictors of interest, their behavior is less understood when they fail due to misspecified inductive biases; in other words, when the employed models are not flexible enough or when the training algorithm does not induce the desired predictors. Then, we study the performance of regression-based CI tests under misspecified inductive biases. Namely, we propose new approximations or upper bounds for the testing errors of three regression-based tests that depend on misspecification errors. Moreover, we introduce the Rao-Blackwellized Predictor Test (RBPT), a regression-based CI test robust against misspecified inductive biases. Finally, we conduct experiments with artificial and real data, showcasing the usefulness of our theory and methods.

----

## [0] Blurred-Dilated Method for Adversarial Attacks

**Authors**: *Yang Deng, Weibin Wu, Jianping Zhang, Zibin Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) are vulnerable to adversarial attacks, which lead to incorrect predictions. In black-box settings, transfer attacks can be conveniently used to generate adversarial examples. However, such examples tend to overfit the specific architecture and feature representations of the source model, resulting in poor attack performance against other target models. To overcome this drawback, we propose a novel model modification-based transfer attack: Blurred-Dilated method (BD) in this paper. In summary, BD works by reducing downsampling while introducing BlurPool and dilated convolutions in the source model. Then BD employs the modified source model to generate adversarial samples. We think that BD can more comprehensively preserve the feature information than the original source model. It thus enables more thorough destruction of the image features, which can improve the transferability of the generated adversarial samples. Extensive experiments on the ImageNet dataset show that adversarial examples generated by BD achieve significantly higher transferability than the state-of-the-art baselines. Besides, BD can be conveniently combined with existing black-box attack techniques to further improve their performance.

----

## [0] Towards Distribution-Agnostic Generalized Category Discovery

**Authors**: *Jianhong Bai, Zuozhu Liu, Hualiang Wang, Ruizhe Chen, Lianrui Mu, Xiaomeng Li, Joey Tianyi Zhou, Yang Feng, Jian Wu, Haoji Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html)

**Abstract**:

Data imbalance and open-ended distribution are two intrinsic characteristics of the real visual world. Though encouraging progress has been made in tackling each challenge separately, few works dedicated to combining them towards real-world scenarios. While several previous works have focused on classifying close-set samples and detecting open-set samples during testing, it's still essential to be able to classify unknown subjects as human beings. In this paper, we formally define a more realistic task as distribution-agnostic generalized category discovery (DA-GCD): generating fine-grained predictions for both close- and open-set classes in a long-tailed open-world setting. To tackle the challenging problem, we propose a Self-Balanced Co-Advice contrastive framework (BaCon), which consists of a contrastive-learning branch and a pseudo-labeling branch, working collaboratively to provide interactive supervision to resolve the DA-GCD task. In particular, the contrastive-learning branch provides reliable distribution estimation to regularize the predictions of the pseudo-labeling branch, which in turn guides contrastive learning through self-balanced knowledge transfer and a proposed novel contrastive loss. We compare BaCon with state-of-the-art methods from two closely related fields: imbalanced semi-supervised learning and generalized category discovery. The effectiveness of BaCon is demonstrated with superior performance over all baselines and comprehensive analysis across various datasets. Our code is publicly available.

----

## [0] Stable Diffusion is Unstable

**Authors**: *Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html)

**Abstract**:

Recently, text-to-image models have been thriving. Despite their powerful generative capacity, our research has uncovered a lack of robustness in this generation process. Specifically, the introduction of small perturbations to the text prompts can result in the blending of primary subjects with other categories or their complete disappearance in the generated images. In this paper, we propose Auto-attack on Text-to-image Models (ATM), a gradient-based approach, to effectively and efficiently generate such perturbations. By learning a Gumbel Softmax distribution, we can make the discrete process of word replacement or extension continuous, thus ensuring the differentiability of the perturbation generation. Once the distribution is learned, ATM can sample multiple attack samples simultaneously. These attack samples can prevent the generative model from generating the desired subjects without tampering with the category keywords in the prompt. ATM has achieved a 91.1\% success rate in short-text attacks and an 81.2\% success rate in long-text attacks. Further empirical analysis revealed three attack patterns based on: 1) variability in generation speed, 2) similarity of coarse-grained characteristics, and 3) polysemy of words. The code is available at https://github.com/duchengbin8/StableDiffusionis_Unstable

----

## [0] A Competitive Algorithm for Agnostic Active Learning

**Authors**: *Yihan Zhou, Eric Price*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7385cb3fa76a0aeedb23d4163640db0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7385cb3fa76a0aeedb23d4163640db0-Abstract-Conference.html)

**Abstract**:

For some hypothesis classes and input distributions, \emph{active}  agnostic learning needs exponentially fewer samples than passive  learning; for other classes and distributions, it offers little to  no improvement.  The most popular algorithms for agnostic active  learning express their performance in terms of a parameter called  the disagreement coefficient, but it is known that these algorithms  are inefficient on some inputs.  We take a different approach to agnostic active learning, getting an  algorithm that is \emph{competitive} with the optimal algorithm for  any binary hypothesis class $H$ and distribution $\mathcal{D}_X$ over $X$.  In particular, if any algorithm can use $m^*$ queries to get  $O(\eta)$ error, then our algorithm uses $O(m^* \log H)$ queries to  get $O(\eta)$ error.  Our algorithm lies in the vein of the  splitting-based approach of Dasgupta [2004], which gets a similar  result for the realizable ($\eta = 0$) setting.  We also show that it is NP-hard to do better than our algorithm's  $O(\log H)$ overhead in general.

----

## [0] Efficient Hyper-parameter Optimization with Cubic Regularization

**Authors**: *Zhenqian Shen, Hansi Yang, Yong Li, James T. Kwok, Quanming Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7500454af92cf3934eb1cc2d59abbdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7500454af92cf3934eb1cc2d59abbdf-Abstract-Conference.html)

**Abstract**:

As hyper-parameters are ubiquitous and can significantly affect the model performance, hyper-parameter optimization is extremely important in machine learning. In this paper, we consider a sub-class of hyper-parameter optimization problems, where the hyper-gradients are not available. Such problems frequently appear when the performance metric is non-differentiable or the hyper-parameter is not continuous. However, existing algorithms, like Bayesian optimization and reinforcement learning, often get trapped in local optimals with poor performance. To address the above limitations, we propose to use cubic regularization to accelerate convergence and avoid saddle points. First, we adopt stochastic relaxation, which allows obtaining gradient and Hessian information without hyper-gradients. Then, we exploit the rich curvature information by cubic regularization. Theoretically, we prove that the proposed method can converge to approximate second-order stationary points, and the convergence is also guaranteed when the lower-level problem is inexactly solved. Experiments on synthetic and real-world data demonstrate the effectiveness of our proposed method.

----

## [0] Joint Attribute and Model Generalization Learning for Privacy-Preserving Action Recognition

**Authors**: *Duo Peng, Li Xu, Qiuhong Ke, Ping Hu, Jun Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b762632135b16f1225672f9fe2a9740b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b762632135b16f1225672f9fe2a9740b-Abstract-Conference.html)

**Abstract**:

Privacy-Preserving Action Recognition (PPAR) aims to transform raw videos into anonymous ones to prevent privacy leakage while maintaining action clues, which is an increasingly important problem in intelligent vision applications. Despite recent efforts in this task, it is still challenging to deal with novel privacy attributes and novel privacy attack models that are unavailable during the training phase. In this paper, from the perspective of meta-learning (learning to learn), we propose a novel Meta Privacy-Preserving Action Recognition (MPPAR) framework to improve both generalization abilities above (i.e., generalize to novel privacy attributes and novel privacy attack models) in a unified manner. Concretely, we simulate train/test task shifts by constructing disjoint support/query sets w.r.t. privacy attributes or attack models. Then, a virtual training and testing scheme is applied based on support/query sets to provide feedback to optimize the model's learning toward better generalization. Extensive experiments demonstrate the effectiveness and generalization of the proposed framework compared to state-of-the-arts.

----

## [0] 3D-Aware Visual Question Answering about Parts, Poses and Occlusions

**Authors**: *Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, Alan L. Yuille*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b783c44ba9adbc30344473dc633b4869-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b783c44ba9adbc30344473dc633b4869-Abstract-Conference.html)

**Abstract**:

Despite rapid progress in Visual question answering (\textit{VQA}), existing datasets and models mainly focus on testing reasoning in 2D.  However, it is important that VQA models also understand the 3D structure of visual scenes, for example to support tasks like navigation or manipulation.  This includes an understanding of the 3D object pose, their parts and occlusions.   In this work, we introduce the task of 3D-aware VQA, which focuses on challenging questions that require a compositional reasoning over the 3D structure of visual scenes.   We address 3D-aware VQA from both the dataset and the model perspective.   First, we introduce Super-CLEVR-3D, a compositional reasoning dataset that contains questions about object parts, their 3D poses, and occlusions.   Second, we propose PO3D-VQA, a 3D-aware VQA model that marries two powerful ideas: probabilistic neural symbolic program execution for reasoning and deep neural networks with 3D generative representations of objects for robust visual recognition.  Our experimental results show our model PO3D-VQA outperforms existing methods significantly, but we still observe a significant performance gap compared to 2D VQA benchmarks, indicating that 3D-aware VQA remains an important open research area.

----

## [0] MixFormerV2: Efficient Fully Transformer Tracking

**Authors**: *Yutao Cui, Tianhui Song, Gangshan Wu, Limin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7870bd43b2d133a1ed95582ae5d82a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7870bd43b2d133a1ed95582ae5d82a4-Abstract-Conference.html)

**Abstract**:

Transformer-based trackers have achieved strong accuracy on the standard benchmarks. However, their efficiency remains an obstacle to practical deployment on both GPU and CPU platforms. In this paper, to overcome this issue, we propose a fully transformer tracking framework, coined as \emph{MixFormerV2}, without any dense convolutional operation and complex score prediction module. Our key design is to introduce four special prediction tokens and concatenate them with the tokens from target template and search areas. Then, we apply the unified transformer backbone on these mixed token sequence. These prediction tokens are able to capture the complex correlation between target template and search area via mixed attentions. Based on them, we can easily predict the tracking box and estimate its confidence score through simple MLP heads. To further improve the efficiency of MixFormerV2, we present a new distillation-based model reduction paradigm, including dense-to-sparse distillation and deep-to-shallow distillation. The former one aims to transfer knowledge from the dense-head based MixViT to our fully transformer tracker, while the latter one is used to prune some layers of the backbone. We instantiate two types of MixForemrV2, where the MixFormerV2-B achieves an AUC of 70.6\% on LaSOT and AUC of 56.7\% on TNL2k with a high GPU speed of 165 FPS, and the MixFormerV2-S surpasses FEAR-L by 2.7\% AUC on LaSOT with a real-time CPU speed.

----

## [0] Generalized Information-theoretic Multi-view Clustering

**Authors**: *Weitian Huang, Sirui Yang, Hongmin Cai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7aa34d2d24f9bab3056993b7bfa0f1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7aa34d2d24f9bab3056993b7bfa0f1b-Abstract-Conference.html)

**Abstract**:

In an era of more diverse data modalities, multi-view clustering has become a fundamental tool for comprehensive data analysis and exploration. However, existing multi-view unsupervised learning methods often rely on strict assumptions on semantic consistency among samples. In this paper, we reformulate the multi-view clustering problem from an information-theoretic perspective and propose a general theoretical model. In particular, we define three desiderata under multi-view unsupervised learning in terms of mutual information, namely, comprehensiveness, concentration, and cross-diversity. The multi-view variational lower bound is then obtained by approximating the samples' high-dimensional mutual information. The Kullbackâ€“Leibler divergence is utilized to deduce sample assignments. Ultimately the information-based multi-view clustering model leverages deep neural networks and Stochastic Gradient Variational Bayes to achieve representation learning and clustering simultaneously. Extensive experiments on both synthetic and real datasets with wide types demonstrate that the proposed method exhibits a more stable and superior clustering performance than state-of-the-art algorithms.

----

## [0] Counterfactual Evaluation of Peer-Review Assignment Policies

**Authors**: *Martin Saveski, Steven Jecmen, Nihar B. Shah, Johan Ugander*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html)

**Abstract**:

Peer review assignment algorithms aim to match research papers to suitable expert reviewers, working to maximize the quality of the resulting reviews. A key challenge in designing effective assignment policies is evaluating how changes to the assignment algorithm map to changes in review quality. In this work, we leverage recently proposed policies that introduce randomness in peer-review assignment—in order to mitigate fraud—as a valuable opportunity to evaluate counterfactual assignment policies. Specifically, we exploit how such randomized assignments provide a positive probability of observing the reviews of many assignment policies of interest. To address challenges in applying standard off-policy evaluation methods, such as violations of positivity, we introduce novel methods for partial identification based on monotonicity and Lipschitz smoothness assumptions for the mapping between reviewer-paper covariates and outcomes. We apply our methods to peer-review data from two computer science venues: the TPDP'21 workshop (95 papers and 35 reviewers) and the AAAI'22 conference (8,450 papers and 3,145 reviewers). We consider estimates of (i) the effect on review quality when changing weights in the assignment algorithm, e.g., weighting reviewers' bids vs. textual similarity (between the review's past papers and the submission), and (ii) the "cost of randomization", capturing the difference in expected quality between the perturbed and unperturbed optimal match. We find that placing higher weight on text similarity results in higher review quality and that introducing randomization in the reviewer-paper assignment only marginally reduces the review quality. Our methods for partial identification may be of independent interest, while our off-policy approach can likely find use in evaluating a broad class of algorithmic matching systems.

----

## [0] Temporal Causal Mediation through a Point Process: Direct and Indirect Effects of Healthcare Interventions

**Authors**: *Çaglar Hizli, St John, Anne Juuti, Tuure Saarinen, Kirsi Pietiläinen, Pekka Marttinen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7d9b1d4a9464d5d1ece82198e351349-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7d9b1d4a9464d5d1ece82198e351349-Abstract-Conference.html)

**Abstract**:

Deciding on an appropriate intervention requires a causal model of a treatment, the outcome, and potential mediators. Causal mediation analysis lets us distinguish between direct and indirect effects of the intervention, but has mostly been studied in a static setting. In healthcare, data come in the form of complex, irregularly sampled time-series, with dynamic interdependencies between a treatment, outcomes, and mediators across time. Existing approaches to dynamic causal mediation analysis are limited to regular measurement intervals, simple parametric models, and disregard long-range mediator--outcome interactions. To address these limitations, we propose a non-parametric mediator--outcome model where the mediator is assumed to be a temporal point process that interacts with the outcome process. With this model, we estimate the direct and indirect effects of an external intervention on the outcome, showing how each of these affects the whole future trajectory. We demonstrate on semi-synthetic data that our method can accurately estimate direct and indirect effects. On real-world healthcare data, our model infers clinically  meaningful direct and indirect effect trajectories for blood glucose after a surgery.

----

## [0] Randomized and Deterministic Maximin-share Approximations for Fractionally Subadditive Valuations

**Authors**: *Hannaneh Akrami, Kurt Mehlhorn, Masoud Seddighin, Golnoosh Shahkarami*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7ed46bd87cd51d4c031b96d9b1a8eb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7ed46bd87cd51d4c031b96d9b1a8eb6-Abstract-Conference.html)

**Abstract**:

We consider the problem of guaranteeing maximin-share ($\MMS$) when allocating a set of indivisible items to a set of agents with fractionally subadditive ($\XOS$) valuations.  For $\XOS$ valuations, it has been previously shown that for some instances no allocation can guarantee a fraction better than $1/2$ of maximin-share to all the agents. Also, a deterministic allocation exists that guarantees $0.219225$ of the maximin-share of each agent. Our results involve both deterministic and randomized allocations. On the deterministic side, we improve the best approximation guarantee for fractionally subadditive valuations to $3/13 = 0.230769$. We develop new ideas on allocating large items in our allocation algorithm which might be of independent interest. Furthermore, we investigate randomized algorithms and the Best-of-both-worlds fairness guarantees. We propose a randomized allocation that is $1/4$-$\MMS$ ex-ante and $1/8$-$\MMS$ ex-post for $\XOS$ valuations. Moreover, we prove an upper bound of $3/4$ on the ex-ante guarantee for this class of valuations.

----

## [0] Causal normalizing flows: from theory to practice

**Authors**: *Adrián Javaloy, Pablo Sánchez-Martín, Isabel Valera*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html)

**Abstract**:

In this work, we deepen on the use of normalizing flows for causal reasoning. Specifically, we first leverage recent results on non-linear ICA to show that causal models are identifiable from observational data given a causal ordering, and thus can be recovered using autoregressive normalizing flows (NFs). Second, we analyze different design and learning choices for causal normalizing flows to capture the underlying causal data-generating process. Third, we describe how to implement the do-operator in causal NFs, and thus, how to answer interventional and counterfactual questions. Finally, in our experiments, we validate our design and training choices through a comprehensive ablation study; compare causal NFs to other approaches for approximating causal models; and empirically demonstrate that causal NFs can be used to address real-world problems—where the presence of mixed discrete-continuous data and partial knowledge on the causal graph is the norm. The code for this work can be found at https://github.com/psanch21/causal-flows.

----

## [0] Maximum Average Randomly Sampled: A Scale Free and Non-parametric Algorithm for Stochastic Bandits

**Authors**: *Masoud Moravej Khorasani, Erik Weyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b84adff45775e92a45f0cd87c37f5ce9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b84adff45775e92a45f0cd87c37f5ce9-Abstract-Conference.html)

**Abstract**:

Upper Confidence Bound (UCB) methods are one of the most effective methods in dealing with the exploration-exploitation trade-off in online decision-making problems. The confidence bounds utilized in UCB methods tend to be constructed based on concentration equalities which are usually dependent on a parameter of scale (e.g. a bound on the payoffs, a variance, or a subgaussian parameter) that must be known in advance. The necessity of knowing a scale parameter a priori and the fact that the confidence bounds only use the tail information can deteriorate the performance of the UCB methods.Here we propose a data-dependent UCB algorithm called MARS (Maximum Average Randomly Sampled) in a non-parametric setup for multi-armed bandits with symmetric rewards. The algorithm does not depend on any scaling, and the data-dependent upper confidence bound is constructed based on the maximum average of randomly sampled rewards inspired by the work of Hartigan in the 1960s and 70s. A regret bound for the multi-armed bandit problem is derived under the same assumptions as for the $\psi$-UCB method without incorporating any correction factors. The method is illustrated and compared with baseline algorithms in numerical experiments.

----

## [0] Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer

**Authors**: *Bowen Tan, Yun Zhu, Lijuan Liu, Eric P. Xing, Zhiting Hu, Jindong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b860c0c546f4a3a786f9c9468228c99f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b860c0c546f4a3a786f9c9468228c99f-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) such as T0, FLAN, and OPT-IML excel in multi-tasking under a unified instruction-following paradigm, where they also exhibit remarkable generalization abilities to unseen tasks. Despite their impressive performance, these LLMs, with sizes ranging from several billion to hundreds of billions of parameters, demand substantial computational resources, making their training and inference expensive and inefficient. Furthermore, adapting these models to downstream applications, particularly complex tasks, is often unfeasible due to the extensive hardware requirements for finetuning, even when utilizing parameter-efficient approaches such as prompt tuning. Additionally, the most powerful multi-task LLMs, such as OPT-IML-175B and FLAN-PaLM-540B, are not publicly accessible, severely limiting their customization potential. To address these challenges, we introduce a pretrained small scorer, \textit{Cappy}, designed to enhance the performance and efficiency of multi-task LLMs. With merely 360 million parameters, Cappy functions either independently on classification tasks or serve as an auxiliary component for LLMs, boosting their performance. Moreover, Cappy enables efficiently integrating downstream supervision without requiring LLM finetuning nor the access to their parameters. Our experiments demonstrate that, when working independently on 11 language understanding tasks from PromptSource, Cappy outperforms LLMs that are several orders of magnitude larger. Besides, on 45 complex tasks from BIG-Bench, Cappy boosts the performance of the advanced multi-task LLM, FLAN-T5, by a large margin. Furthermore, Cappy is flexible to cooperate with other LLM adaptations, including finetuning and in-context learning, offering additional performance enhancement.

----

## [0] Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks

**Authors**: *Qi Xu, Yuyuan Gao, Jiangrong Shen, Yaxin Li, Xuming Ran, Huajin Tang, Gang Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8734840bf65c8facd619f5105c6acd0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8734840bf65c8facd619f5105c6acd0-Abstract-Conference.html)

**Abstract**:

Spiking neural networks (SNNs) serve as one type of efficient model to process spatio-temporal patterns in time series, such as the Address-Event Representation data collected from Dynamic Vision Sensor (DVS). Although convolutional SNNs have achieved remarkable performance on these AER datasets, benefiting from the predominant spatial feature extraction ability of convolutional structure, they ignore temporal features related to sequential time points. In this paper, we develop a recurrent spiking neural network (RSNN) model embedded with an advanced spiking convolutional block attention module (SCBAM) component to combine both spatial and temporal features of spatio-temporal patterns. It invokes the history information in spatial and temporal channels adaptively through SCBAM, which brings the advantages of efficient memory calling and history redundancy elimination. The performance of our model was evaluated in DVS128-Gesture dataset and other time-series datasets. The experimental results show that the proposed SRNN-SCBAM model makes better use of the history information in spatial and temporal dimensions with less memory space, and achieves higher accuracy compared to other models.

----

## [0] Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method

**Authors**: *Qihang Fang, Yafei Song, Keqiang Li, Liefeng Bo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87738474533cab76c7bee4e08443aca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87738474533cab76c7bee4e08443aca-Abstract-Conference.html)

**Abstract**:

A neural radiance field (NeRF) enables the synthesis of cutting-edge realistic novel view images of a 3D scene. It includes density and color fields to model the shape and radiance of a scene, respectively. Supervised by the photometric loss in an end-to-end training manner, NeRF inherently suffers from the shape-radiance ambiguity problem, i.e., it can perfectly fit training views but does not guarantee decoupling the two fields correctly. To deal with this issue, existing works have incorporated prior knowledge to provide an independent supervision signal for the density field, including total variation loss, sparsity loss, distortion loss, etc. These losses are based on general assumptions about the density field, e.g., it should be smooth, sparse, or compact, which are not adaptive to a specific scene. In this paper, we propose a more adaptive method to reduce the shape-radiance ambiguity. The key is a rendering method that is only based on the density field. Specifically, we first estimate the color field based on the density field and posed images in a closed form. Then NeRF's rendering process can proceed. We address the problems in estimating the color field, including occlusion and non-uniformly distributed views. Afterwards, it is applied to regularize NeRF's density field. As our regularization is guided by photometric loss, it is more adaptive compared to existing ones. Experimental results show that our method improves the density field of NeRF both qualitatively and quantitatively. Our code is available at https://github.com/qihangGH/Closed-form-color-field.

----

## [0] Text-to-Image Diffusion Models are Zero Shot Classifiers

**Authors**: *Kevin Clark, Priyank Jaini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87bdcf963cad3d0b265fcb78ae7d11e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87bdcf963cad3d0b265fcb78ae7d11e-Abstract-Conference.html)

**Abstract**:

The excellent generative capabilities of text-to-image diffusion models suggest they learn informative representations of image-text data.However, what knowledge their representations capture is not fully understood, and they have not been thoroughly explored on downstream tasks.We investigate diffusion models by proposing a method for evaluating them as zero-shot classifiers.The key idea is using a diffusion model's ability to denoise a noised image given a text description of a label as a proxy for that label's likelihood.We apply our method to Stable Diffusion and Imagen, using it to probe fine-grained aspects of the models' knowledge and comparing them with CLIP's zero-shot abilities. They perform competitively with CLIP on a wide range of zero-shot image classification datasets. Additionally, they achieve state-of-the-art results on shape/texture bias tests and can successfully perform attribute binding while CLIP cannot.Although generative pre-training is prevalent in NLP, visual foundation models often use other methods such as contrastive learning. Based on our findings, we argue that generative pre-training should be explored as a compelling alternative for vision and vision-language problems.

----

## [0] MADG: Margin-based Adversarial Learning for Domain Generalization

**Authors**: *Aveen Dayal, Vimal K. B., Linga Reddy Cenkeramaddi, C. Krishna Mohan, Abhinav Kumar, Vineeth N. Balasubramanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87d9d19ecb5927f7e18c537908610ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87d9d19ecb5927f7e18c537908610ef-Abstract-Conference.html)

**Abstract**:

Domain Generalization (DG) techniques have emerged as a popular approach to address the challenges of domain shift in Deep Learning (DL), with the goal of generalizing well to the target domain unseen during the training. In recent years, numerous methods have been proposed to address the DG setting, among which one popular approach is the adversarial learning-based methodology. The main idea behind adversarial DG methods is to learn domain-invariant features by minimizing a discrepancy metric. However, most adversarial DG methods use 0-1 loss based $\mathcal{H}\Delta\mathcal{H}$ divergence metric. In contrast, the margin loss-based discrepancy metric has the following advantages: more informative, tighter, practical, and efficiently optimizable. To mitigate this gap, this work proposes a novel adversarial learning DG algorithm, $\textbf{MADG}$, motivated by a margin loss-based discrepancy metric. The proposed $\textbf{MADG}$ model learns domain-invariant features across all source domains and uses adversarial training to generalize well to the unseen target domain. We also provide a theoretical analysis of the proposed $\textbf{MADG}$ model based on the unseen target error bound. Specifically, we construct the link between the source and unseen domains in the real-valued hypothesis space and derive the generalization bound using margin loss and Rademacher complexity. We extensively experiment with the $\textbf{MADG}$ model on popular real-world DG datasets, VLCS, PACS, OfficeHome, DomainNet, and TerraIncognita. We evaluate the proposed algorithm on DomainBed's benchmark and observe consistent performance across all the datasets.

----

## [0] Bridging RL Theory and Practice with the Effective Horizon

**Authors**: *Cassidy Laidlaw, Stuart J. Russell, Anca D. Dragan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (RL) works impressively in some environments and fails catastrophically in others. Ideally, RL theory should be able to provide an understanding of why this is, i.e. bounds predictive of practical performance. Unfortunately, current theory does not quite have this ability. We compare standard deep RL algorithms to prior sample complexity bounds by introducing a new dataset, BRIDGE. It consists of 155 MDPs from common deep RL benchmarks, along with their corresponding tabular representations, which enables us to exactly compute instance-dependent bounds. We find that prior bounds do not correlate well with when deep RL succeeds vs. fails, but discover a surprising property that does. When actions with the highest Q-values under the random policy also have the highest Q-values under the optimal policy—i.e., when it is optimal to act greedily with respect to the random's policy Q function—deep RL tends to succeed; when they don't, deep RL tends to fail. We generalize this property into a new complexity measure of an MDP that we call the effective horizon, which roughly corresponds to how many steps of lookahead search would be needed in that MDP in order to identify the next optimal action, when leaf nodes are evaluated with random rollouts. Using BRIDGE, we show that the effective horizon-based bounds are more closely reflective of the empirical performance of PPO and DQN than prior sample complexity bounds across four metrics. We also show that, unlike existing bounds, the effective horizon can predict the effects of using reward shaping or a pre-trained exploration policy. Our code and data are available at https://github.com/cassidylaidlaw/effective-horizon.

----

## [0] Fine-Grained Human Feedback Gives Better Rewards for Language Model Training

**Authors**: *Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, Hannaneh Hajishirzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html)

**Abstract**:

Language models (LMs) often exhibit undesirable text generation behaviors, including generating false, toxic, or irrelevant outputs. Reinforcement learning from human feedback (RLHF)---where human preference judgments on LM outputs are transformed into a learning signal---has recently shown promise in addressing these issues. However, such holistic feedback conveys limited information on long text outputs; it does not indicate which aspects of the outputs influenced user preference; e.g., which parts contain what type(s) of errors. In this paper, we use fine-grained human feedback (e.g., which sentence is false, which sub-sentence is irrelevant) as an explicit training signal. We introduce Fine-Grained RLHF, a framework that enables training and learning from reward functions that are fine-grained in two respects: (1) density, providing a reward after every segment (e.g., a sentence) is generated; and (2) incorporating multiple reward models associated with different feedback types (e.g., factual incorrectness, irrelevance, and information incompleteness). We conduct experiments on detoxification and long-form question answering to illustrate how learning with this reward function leads to improved performance, supported by both automatic and human evaluation. Additionally, we show that LM behaviors can be customized using different combinations of fine-grained reward models. We release all data, collected human feedback, and codes at https://FineGrainedRLHF.github.io.

----

## [0] Towards Optimal Effective Resistance Estimation

**Authors**: *Rajat Vadiraj Dwaraknath, Ishani Karmarkar, Aaron Sidford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8e2046160a568145af6d42eeef199f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8e2046160a568145af6d42eeef199f4-Abstract-Conference.html)

**Abstract**:

We provide new algorithms and conditional hardness for the problem of estimating effective resistances in $n$-node $m$-edge undirected, expander graphs. We provide an $\widetilde{O}(m\epsilon^{-1})$-time algorithm that produces with high probability, an $\widetilde{O}(n\epsilon^{-1})$-bit sketch from which the effective resistance between any pair of nodes can be estimated, to $(1 \pm \epsilon)$-multiplicative accuracy, in $\widetilde{O}(1)$-time. Consequently, we obtain an $\widetilde{O}(m\epsilon^{-1})$-time algorithm for estimating the effective resistance of all edges in such graphs, improving (for sparse graphs) on the previous fastest runtimes of $\widetilde{O}(m\epsilon^{-3/2})$ [Chu et. al. 2018] and $\widetilde{O}(n^2\epsilon^{-1})$ [Jambulapati, Sidford, 2018] for general graphs and $\widetilde{O}(m + n\epsilon^{-2})$ for expanders [Li, Sachdeva 2022]. We complement this result by showing a conditional lower bound that a broad set of algorithms for computing such estimates of the effective resistances between all pairs of nodes require $\widetilde{\Omega}(n^2 \epsilon^{-1/2})$-time, improving upon the previous best such lower bound of $\widetilde{\Omega}(n^2 \epsilon^{-1/13})$ [Musco et. al. 2017]. Further, we leverage the tools underlying these results to obtain improved algorithms and conditional hardness for more general problems of sketching the pseudoinverse of positive semidefinite matrices and estimating functions of their eigenvalues.

----

## [0] TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning

**Authors**: *Shuo Sun, Molei Qin, Wentao Zhang, Haochong Xia, Chuqiao Zong, Jie Ying, Yonggang Xie, Lingxuan Zhao, Xinrun Wang, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The financial markets, which involve over \$90 trillion market capitals, attract the attention of innumerable profit-seeking investors globally. Recent explosion of reinforcement learning in financial trading (RLFT) research has shown stellar performance on many quantitative trading tasks. However, it is still challenging to deploy reinforcement learning (RL) methods into real-world financial markets due to the highly composite nature of this domain, which entails design choices and interactions between components that collect financial data, conduct feature engineering, build market environments, make investment decisions, evaluate model behaviors and offers user interfaces. Despite the availability of abundant financial data and advanced RL techniques, a remarkable gap still exists between the potential and realized utilization of RL in financial trading. In particular, orchestrating an RLFT project lifecycle poses challenges in engineering (i.e. hard to build), benchmarking (i.e. hard to compare) and usability (i.e. hard to optimize, maintain and use). To overcome these challenges, we introduce TradeMaster, a holistic open-source RLFT platform that serves as a i) software toolkit, ii) empirical benchmark, and iii) user interface. Our ultimate goal is to provide infrastructures for transparent and reproducible RLFT research and facilitate their real-world deployment with industry impact. TradeMaster will be updated continuously and welcomes contributions from both RL and finance communities.

----

## [0] Towards Optimal Caching and Model Selection for Large Model Inference

**Authors**: *Banghua Zhu, Ying Sheng, Lianmin Zheng, Clark W. Barrett, Michael I. Jordan, Jiantao Jiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b914a8fcea5c176cf1ed75c762ce27fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b914a8fcea5c176cf1ed75c762ce27fd-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) and other large foundation models have achieved impressive results, but their size exacerbates existing resource consumption and latency challenges. In particular, the large-scale deployment of these models is hindered by the significant resource requirements during inference. In this paper, we study  two  approaches for mitigating these challenges: employing a cache to store previous queries and learning a model selector to choose from an ensemble of models for query processing.Theoretically, we provide an optimal algorithm for jointly optimizing both approaches  to reduce the inference cost in both offline and online tabular settings. By combining a caching algorithm, namely Greedy Dual Size with Frequency (GDSF) or Least Expected Cost (LEC), with a model selector, we achieve optimal rates in both offline and online settings. Empirically, simulations show that our caching and model selection algorithm greatly improves over the baselines, with up to $50\times$ improvement over the baseline when the ratio between the maximum cost and minimum cost is $100$.  Experiments on real datasets show a $4.3\times$ improvement in FLOPs over the baseline when the ratio for FLOPs is $10$, and a $1.8\times$ improvement in latency when the ratio for average latency is $1.85$.

----

## [0] Deep Non-line-of-sight Imaging from Under-scanning Measurements

**Authors**: *Yue Li, Yueyi Zhang, Juntian Ye, Feihu Xu, Zhiwei Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b91cc0a242e6518ee731f74e82b2eebd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b91cc0a242e6518ee731f74e82b2eebd-Abstract-Conference.html)

**Abstract**:

Active confocal non-line-of-sight (NLOS) imaging has successfully enabled seeing around corners relying on high-quality transient measurements. However, acquiring spatial-dense transient measurement is time-consuming, raising the question of how to reconstruct satisfactory results from under-scanning measurements (USM). The existing solutions, involving the traditional algorithms, however, are hindered by unsatisfactory results or long computing times. To this end, we propose the first deep-learning-based approach to NLOS imaging from USM. Our proposed end-to-end network is composed of two main components: the transient recovery network (TRN) and the volume reconstruction network (VRN). Specifically, TRN takes the under-scanning measurements as input, utilizes a multiple kernel feature extraction module and a multiple feature fusion module, and outputs sufficient-scanning measurements at the high-spatial resolution. Afterwards, VRN incorporates the linear physics prior of the light-path transport model and reconstructs the hidden volume representation. Besides, we introduce regularized constraints that enhance the perception of more local details while suppressing smoothing effects. The proposed method achieves superior performance on both synthetic data and public real-world data, as demonstrated by extensive experimental results with different under-scanning grids. Moreover, the proposed method delivers impressive robustness at an extremely low scanning grid (i.e., 8$\times$8) and offers high-speed inference (i.e., 50 times faster than the existing iterative solution).

----

## [0] Learning Adversarial Low-rank Markov Decision Processes with Unknown Transition and Full-information Feedback

**Authors**: *Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, Xuezhou Zhang, Shuai Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b93fda2862db7a7ac4a5c412adfb1ac2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b93fda2862db7a7ac4a5c412adfb1ac2-Abstract-Conference.html)

**Abstract**:

In this work, we study the low-rank MDPs with adversarially changed losses in the full-information feedback setting. In particular, the unknown transition probability kernel admits a low-rank matrix decomposition \citep{REPUCB22}, and the loss functions may change adversarially but are revealed to the learner at the end of each episode. We propose a policy optimization-based algorithm POLO, and we prove that it attains the $\widetilde{O}(K^{\frac{5}{6}}A^{\frac{1}{2}}d\ln(1+M)/(1-\gamma)^2)$ regret guarantee, where $d$ is rank of the transition kernel (and hence the dimension of the unknown representations), $A$ is the cardinality of the action space, $M$ is the cardinality of the model class that contains all the plausible representations, and $\gamma$ is the discounted factor. Notably, our algorithm is oracle-efficient and has a regret guarantee with no dependence on the size of potentially arbitrarily large state space. Furthermore, we also prove an $\Omega(\frac{\gamma^2}{1-\gamma} \sqrt{d A K})$ regret lower bound for this problem, showing that low-rank MDPs are statistically more difficult to learn than linear MDPs in the regret minimization setting. To the best of our knowledge, we present the first algorithm that interleaves representation learning, exploration, and exploitation to achieve the sublinear regret guarantee for RL with nonlinear function approximation and adversarial losses.

----

## [0] UE4-NeRF: Neural Radiance Field for Real-Time Rendering of Large-Scale Scene

**Authors**: *Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, Mohammed Bennamoun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRF) is a novel implicit 3D reconstruction method that shows immense potential and has been gaining increasing attention. It enables the reconstruction of 3D scenes solely from a set of photographs. However, its real-time rendering capability, especially for interactive real-time rendering of large-scale scenes, still has significant limitations. To address these challenges, in this paper, we propose a novel neural rendering system called UE4-NeRF, specifically designed for real-time rendering of large-scale scenes. We partitioned each large scene into different sub-NeRFs. In order to represent the partitioned independent scene, we initialize polygonal meshes by constructing multiple regular octahedra within the scene and  the vertices of the polygonal faces are continuously optimized during the training process. Drawing inspiration from Level of Detail (LOD) techniques, we trained meshes of varying levels of detail for different observation levels. Our approach combines with the rasterization pipeline in Unreal Engine 4 (UE4), achieving real-time rendering of large-scale scenes at 4K resolution with a frame rate of up to 43 FPS. Rendering within UE4 also facilitates  scene editing in subsequent stages. Furthermore, through experiments, we have demonstrated that our method achieves rendering quality comparable to state-of-the-art approaches. Project page: https://jamchaos.github.io/UE4-NeRF/.

----

## [0] H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection

**Authors**: *Yi Yu, Xue Yang, Qingyun Li, Yue Zhou, Feipeng Da, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html)

**Abstract**:

With the rapidly increasing demand for oriented object detection, e.g. in autonomous driving and remote sensing, the recently proposed paradigm involving weakly-supervised detector H2RBox for learning rotated box (RBox) from the more readily-available horizontal box (HBox) has shown promise. This paper presents H2RBox-v2, to further bridge the gap between HBox-supervised and RBox-supervised oriented object detection. Specifically, we propose to leverage the reflection symmetry via flip and rotate consistencies, using a weakly-supervised network branch similar to H2RBox, together with a novel self-supervised branch that learns orientations from the symmetry inherent in visual objects. The detector is further stabilized and enhanced by practical techniques to cope with peripheral issues e.g. angular periodicity. To our best knowledge, H2RBox-v2 is the first symmetry-aware self-supervised paradigm for oriented object detection. In particular, our method shows less susceptibility to low-quality annotation and insufficient training data compared to H2RBox. Specifically, H2RBox-v2 achieves very close performance to a rotation annotation trained counterpart -- Rotated FCOS: 1) DOTA-v1.0/1.5/2.0: 72.31%/64.76%/50.33% vs. 72.44%/64.53%/51.77%; 2) HRSC: 89.66% vs. 88.99%; 3) FAIR1M: 42.27% vs. 41.25%.

----

## [0] The Waymo Open Sim Agents Challenge

**Authors**: *Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nicholas Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, Dragomir Anguelov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Simulation with realistic, interactive agents represents a key task for autonomous vehicle software development. In this work, we introduce the Waymo Open Sim Agents Challenge (WOSAC). WOSAC is the first public challenge to tackle this task and propose corresponding metrics. The goal of the challenge is to stimulate the design of realistic simulators that can be used to evaluate and train a behavior model for autonomous driving. We outline our evaluation methodology, present results for a number of different baseline simulation agent methods, and analyze several submissions to the 2023 competition which ran from March 16, 2023 to May 23, 2023. The WOSAC evaluation server remains open for submissions and we discuss open problems for the task.

----

## [0] Online RL in Linearly qπ-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore

**Authors**: *Gellért Weisz, András György, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b973a107336177a274069cefb011244c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b973a107336177a274069cefb011244c-Abstract-Conference.html)

**Abstract**:

We consider online reinforcement learning (RL) in episodic Markov decision processes (MDPs) under the linear $q^\pi$-realizability assumption, where it is assumed that the action-values of all policies can be  expressed as linear functions of state-action features. This class is known to be more general than  linear MDPs, where the transition kernel and the reward function are assumed to be linear functions of the feature vectors. As our first contribution, we show that the difference between the two classes is the presence of states in linearly $q^\pi$-realizable MDPs where for any policy, all the actions have  approximately equal values, and skipping over these states by following an arbitrarily fixed policy in those states transforms the problem to a linear MDP. Based on this observation, we derive a novel (computationally inefficient) learning algorithm for linearly $q^\pi$-realizable MDPs that simultaneously learns what states should be skipped over and runs another learning algorithm on the linear MDP hidden in the problem. The method returns an $\epsilon$-optimal policy after $\text{polylog}(H, d)/\epsilon^2$ interactions with the MDP, where $H$ is the time horizon and $d$ is the dimension of the feature vectors, giving the first polynomial-sample-complexity online RL algorithm for this setting. The results are proved for the misspecified case, where the sample complexity is shown to degrade gracefully with the misspecification error.

----

## [0] On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks

**Authors**: *Laura Fee Nern, Harsh Raj, Maurice André Georgi, Yash Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html)

**Abstract**:

As large-scale training regimes have gained popularity, the use of pretrained models for downstream tasks has become common practice in machine learning. While pretraining has been shown to enhance the performance of models in practice, the transfer of robustness properties from pretraining to downstream tasks remains poorly understood. In this study, we demonstrate that the robustness of a linear predictor on downstream tasks can be constrained by the robustness of its underlying representation, regardless of the protocol used for pretraining. We prove (i) a bound on the loss that holds independent of any downstream task, as well as (ii) a criterion for robust classification in particular. We validate our theoretical results in practical applications, show how our results can be used for calibrating expectations of downstream robustness, and when our results are useful for optimal transfer learning. Taken together, our results offer an initial step towards characterizing the requirements of the representation function for reliable post-adaptation performance.

----

## [0] Entropy-dissipation Informed Neural Network for McKean-Vlasov Type PDEs

**Authors**: *Zebang Shen, Zhenfu Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9a17133e3943509243b5e197c1c23b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9a17133e3943509243b5e197c1c23b2-Abstract-Conference.html)

**Abstract**:

The McKean-Vlasov equation (MVE) describes the collective behavior of particles subject to drift, diffusion, and mean-field interaction. In physical systems, the interaction term can be singular, i.e. it diverges when two particles collide. Notable examples of such interactions include the Coulomb interaction, fundamental in plasma physics, and the Biot-Savart interaction, present in the vorticity formulation of the 2D Navier-Stokes equation (NSE) in fluid dynamics. Solving MVEs that involve singular interaction kernels presents a significant challenge, especially when aiming to provide rigorous theoretical guarantees. In this work, we propose a novel approach based on the concept of entropy dissipation in the underlying system. We derive a potential function that effectively controls the KL divergence between a hypothesis solution and the ground truth. Building upon this theoretical foundation, we introduce the Entropy-dissipation Informed Neural Network (EINN) framework for solving MVEs. In EINN, we utilize neural networks (NN) to approximate the underlying velocity field and minimize the proposed potential function. By leveraging the expressive power of NNs, our approach offers a promising avenue for tackling the complexities associated with singular interactions. To assess the empirical performance of our method, we compare EINN with SOTA NN-based MVE solvers. The results demonstrate the effectiveness of our approach in solving MVEs across various example problems.

----

## [0] Revisiting Visual Model Robustness: A Frequency Long-Tailed Distribution View

**Authors**: *Zhiyu Lin, Yifei Gao, Yunfan Yang, Jitao Sang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9a4d7b88a41652c63962ebcc21701b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9a4d7b88a41652c63962ebcc21701b7-Abstract-Conference.html)

**Abstract**:

A widely discussed hypothesis regarding the cause of visual models' lack of robustness is that they can exploit human-imperceptible high-frequency components (HFC) in images, which in turn leads to model vulnerabilities, such as the adversarial examples. However, (1) inconsistent findings regarding the validation of this hypothesis reflect in a limited understanding of HFC, and (2) solutions inspired by the hypothesis tend to involve a robustness-accuracy trade-off and leaning towards suppressing the model's learning on HFC. In this paper, inspired by the long-tailed characteristic observed in frequency spectrum, we first formally define the HFC from long-tailed perspective and then revisit the relationship between HFC and model robustness. In the frequency long-tailed scenario, experimental results on common datasets and various network structures consistently indicate that models in standard training exhibit high sensitivity to HFC. We investigate the reason of the sensitivity, which reflects in model's under-fitting behavior on HFC. Furthermore, the cause of the model's under-fitting behavior is attributed to the limited information content in HFC. Based on these findings, we propose a Balance Spectrum Sampling (BaSS) strategy, which effectively counteracts the long-tailed effect and enhances the model's learning on HFC. Extensive experimental results demonstrate that our method achieves a substantially better robustness-accuracy trade-off when combined with existing defense methods, while also indicating the potential of encouraging HFC learning in improving model performance.

----

## [0] How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization

**Authors**: *Hai Zhang, Hang Yu, Junqiao Zhao, Di Zhang, Xiao Zhang, Hongtu Zhou, Chang Huang, Chen Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9b4f084b2e6709a2bfad0f601271aec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9b4f084b2e6709a2bfad0f601271aec-Abstract-Conference.html)

**Abstract**:

Designing and deriving effective model-based reinforcement learning (MBRL) algorithms with a performance improvement guarantee is challenging, mainly attributed to the high coupling between model learning and policy optimization. Many prior methods that rely on return discrepancy to guide model learning ignore the impacts of model shift, which can lead to performance deterioration due to excessive model updates. Other methods use performance difference bound to explicitly consider model shift. However, these methods rely on a fixed threshold to constrain model shift, resulting in a heavy dependence on the threshold and a lack of adaptability during the training process. In this paper, we theoretically derive an optimization objective that can unify model shift and model bias and then formulate a fine-tuning process. This process adaptively adjusts the model updates to get a performance improvement guarantee while avoiding model overfitting. Based on these, we develop a straightforward algorithm USB-PO (Unified model Shift and model Bias Policy Optimization). Empirical results show that USB-PO achieves state-of-the-art performance on several challenging benchmark tasks.

----

## [0] Dynamic Pricing and Learning with Bayesian Persuasion

**Authors**: *Shipra Agrawal, Yiding Feng, Wei Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html)

**Abstract**:

We consider a novel dynamic pricing and learning setting where in addition to setting prices of products in sequential rounds, the seller also ex-ante commits to ‘advertising schemes’. That is, in the beginning of each round the seller can decide what kind of signal they will provide to the buyer about the product’s quality upon realization. Using the popular Bayesian persuasion framework to model the effect of these signals on the buyers’ valuation and purchase responses, we formulate the problem of finding an optimal design of the advertising scheme along with a pricing scheme that maximizes the seller’s expected revenue. Without any apriori knowledge of the buyers’ demand function, our goal is to design an online algorithm that can use past purchase responses to adaptively learn the optimal pricing and advertising strategy. We study the regret of the algorithm when compared to the optimal clairvoyant price and advertisingscheme. Our main result is a computationally efficient online algorithm that achieves an $O(T^{2/3}(m \log T )^{1/3})$ regret bound when the valuation function is linear in the product quality. Here $m$ is the cardinality of the discrete product quality domain and $T$ is the time horizon. This result requires some natural monotonicity and Lipschitz assumptions on the valuation function, but no Lipschitz or smoothness assumption on the buyers’ demand function. For constant $m$, our result matches the regret lower bound for dynamic pricing within logarithmic factors, which is a special case of our problem. We also obtain several improved results for the widely considered special case of additive valuations, including an $\tilde{O}(T^{2/3})$ regret bound independent of $m$ when $m\le T^{1/3}$.

----

## [0] RH-BrainFS: Regional Heterogeneous Multimodal Brain Networks Fusion Strategy

**Authors**: *Hongting Ye, Yalu Zheng, Yueying Li, Ke Zhang, Youyong Kong, Yonggui Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9c353d02e565f0f7cba94c4f3584eaa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9c353d02e565f0f7cba94c4f3584eaa-Abstract-Conference.html)

**Abstract**:

Multimodal fusion has become an important research technique in neuroscience that completes downstream tasks by extracting complementary information from multiple modalities. Existing multimodal research on brain networks mainly focuses on two modalities, structural connectivity (SC) and functional connectivity (FC). Recently, extensive literature has shown that the relationship between SC and FC is complex and not a simple one-to-one mapping. The coupling of structure and function at the regional level is heterogeneous. However, all previous studies have neglected the modal regional heterogeneity between SC and FC and fused their representations via "simple patterns", which are inefficient ways of multimodal fusion and affect the overall performance of the model. In this paper, to alleviate the issue of regional heterogeneity of multimodal brain networks, we propose a novel Regional Heterogeneous multimodal Brain networks Fusion Strategy (RH-BrainFS). Briefly, we introduce a brain subgraph networks module to extract regional characteristics of brain networks, and further use a new transformer-based fusion bottleneck module to alleviate the issue of regional heterogeneity between SC and FC. To the best of our knowledge, this is the first paper to explicitly state the issue of structural-functional modal regional heterogeneity and to propose asolution. Extensive experiments demonstrate that the proposed method outperforms several state-of-the-art methods in a variety of neuroscience tasks.

----

## [0] To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis

**Authors**: *Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, Yang You*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html)

**Abstract**:

Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text data on the web is likely to be approaching its scaling limit for LLMs. To further enhance LLMs, a straightforward approach is to repeat the pre-training data for additional epochs. In this study, we empirically investigate three key aspects under this approach. First, we explore the consequences of repeating pre-training data, revealing that the model is susceptible to overfitting, leading to multi-epoch degradation. Second, we examine the key factors contributing to multi-epoch degradation, finding that significant factors include dataset size, model parameters, and training objectives, while less influential factors consist of dataset quality and model FLOPs. Finally, we explore whether widely used regularization can alleviate multi-epoch degradation. Most regularization techniques do not yield significant improvements, except for dropout, which demonstrates remarkable effectiveness but requires careful tuning when scaling up the model size. Additionally, we discover that leveraging mixture-of-experts (MoE) enables cost-effective and efficient hyper-parameter tuning for computationally intensive dense LLMs with comparable trainable parameters, potentially impacting efficient LLM development on a broader scale.

----

## [0] A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs

**Authors**: *Zhaocheng Zhu, Xinyu Yuan, Michael Galkin, Louis-Pascal A. C. Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html)

**Abstract**:

Reasoning on large-scale knowledge graphs has been long dominated by embedding methods. While path-based methods possess the inductive capacity that embeddings lack, their scalability is limited by the exponential number of paths. Here we present A*Net, a scalable path-based method for knowledge graph reasoning. Inspired by the A* algorithm for shortest path problems, our A*Net learns a priority function to select important nodes and edges at each iteration, to reduce time and memory footprint for both training and inference. The ratio of selected nodes and edges can be specified to trade off between performance and efficiency. Experiments on both transductive and inductive knowledge graph reasoning benchmarks show that A*Net achieves competitive performance with existing state-of-the-art path-based methods, while merely visiting 10% nodes and 10% edges at each iteration. On a million-scale dataset ogbl-wikikg2, A*Net not only achieves a new state-of-the-art result, but also converges faster than embedding methods. A*Net is the first path-based method for knowledge graph reasoning at such scale.

----

## [0] HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

**Authors**: *Shengcao Cao, Dhiraj Joshi, Liangyan Gui, Yu-Xiong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9ecf4d84999a61783c360c3782e801e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9ecf4d84999a61783c360c3782e801e-Abstract-Conference.html)

**Abstract**:

The human visual perception system demonstrates exceptional capabilities in learning without explicit supervision and understanding the part-to-whole composition of objects. Drawing inspiration from these two abilities, we propose Hierarchical Adaptive Self-Supervised Object Detection (HASSOD), a novel approach that learns to detect objects and understand their compositions without human supervision. HASSOD employs a hierarchical adaptive clustering strategy to group regions into object masks based on self-supervised visual representations, adaptively determining the number of objects per image. Furthermore, HASSOD identifies the hierarchical levels of objects in terms of composition, by analyzing coverage relations between masks and constructing tree structures. This additional self-supervised learning task leads to improved detection performance and enhanced interpretability. Lastly, we abandon the inefficient multi-round self-training process utilized in prior methods and instead adapt the Mean Teacher framework from semi-supervised learning, which leads to a smoother and more efficient training process. Through extensive experiments on prevalent image datasets, we demonstrate the superiority of HASSOD over existing methods, thereby advancing the state of the art in self-supervised object detection. Notably, we improve Mask AR from 20.2 to 22.5 on LVIS, and from 17.0 to 26.0 on SA-1B. Project page: https://HASSOD-NeurIPS23.github.io.

----

## [0] Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons

**Authors**: *Luke Taylor, Andrew King, Nicol S. Harper*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9f253c2758a323f9d2095f91de9a974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9f253c2758a323f9d2095f91de9a974-Abstract-Conference.html)

**Abstract**:

The adaptive leaky integrate-and-fire (ALIF) model is fundamental within computational neuroscience and has been instrumental in studying our brains $\textit{in silico}$. Due to the sequential nature of simulating these neural models, a commonly faced issue is the speed-accuracy trade-off: either accurately simulate a neuron using a small discretisation time-step (DT), which is slow, or more quickly simulate a neuron using a larger DT and incur a loss in simulation accuracy. Here we provide a solution to this dilemma, by algorithmically reinterpreting the ALIF model, reducing the sequential simulation complexity and permitting a more efficient parallelisation on GPUs. We computationally validate our implementation to obtain over a $50\times$ training speedup using small DTs on synthetic benchmarks. We also obtained a comparable performance to the standard ALIF implementation on different supervised classification tasks - yet in a fraction of the training time. Lastly, we showcase how our model makes it possible to quickly and accurately fit real electrophysiological recordings of cortical neurons, where very fine sub-millisecond DTs are crucial for capturing exact spike timing.

----

## [0] Re-Think and Re-Design Graph Neural Networks in Spaces of Continuous Graph Diffusion Functionals

**Authors**: *Tingting Dan, Jiaqi Ding, Ziquan Wei, Shahar Z. Kovalsky, Minjeong Kim, Won Hwa Kim, Guorong Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9fd027eb16434174b8bb3d3b18110af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9fd027eb16434174b8bb3d3b18110af-Abstract-Conference.html)

**Abstract**:

Graphs are ubiquitous in various domains, such as social networks and biological systems. Despite the great successes of graph neural networks (GNNs) in modeling and analyzing complex graph data, the inductive bias of locality assumption, which involves exchanging information only within neighboring connected nodes, restricts GNNs in capturing long-range dependencies and global patterns in graphs. Inspired by the classic Brachistochrone problem, we seek how to devise a new inductive bias for cutting-edge graph application and present a general framework through the lens of variational analysis. The backbone of our framework is a two-way mapping between the discrete GNN model and continuous diffusion functional, which allows us to design application-specific objective function in the continuous domain and engineer discrete deep model with mathematical guarantees. First, we address over-smoothing in current GNNs. Specifically, our inference reveals that the existing layer-by-layer models of graph embedding learning are equivalent to a ${\ell _2}$-norm integral functional of graph gradients, which is the underlying cause of the over-smoothing problem. Similar to edge-preserving filters in image denoising, we introduce the total variation (TV) to promote alignment of the graph diffusion pattern with the global information present in community topologies. On top of this, we devise a new selective mechanism for inductive bias that can be easily integrated into existing GNNs and effectively address the trade-off between model depth and over-smoothing. Second, we devise a novel generative adversarial network (GAN) to predict the spreading flows in the graph through a neural transport equation. To avoid the potential issue of vanishing flows, we tailor the objective function to minimize the transportation within each community while maximizing the inter-community flows. Our new GNN models achieve state-of-the-art (SOTA) performance on graph learning benchmarks such as Cora, Citeseer, and Pubmed.

----

## [0] Adapting Fairness Interventions to Missing Values

**Authors**: *Raymond Feng, Flávio Calmon, Hao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba0ad9d1e0c737800b2340b9cd68c208-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba0ad9d1e0c737800b2340b9cd68c208-Abstract-Conference.html)

**Abstract**:

Missing values in real-world data pose a significant and unique challenge to algorithmic fairness. Different demographic groups may be unequally affected by missing data, and the standard procedure for handling missing values where first data is imputed, then the imputed data is used for classification—a procedure referred to as "impute-then-classify"—can exacerbate discrimination. In this paper, we analyze how missing values affect algorithmic fairness. We first prove that training a classifier from imputed data can significantly worsen the achievable values of group fairness and average accuracy. This is because imputing data results in the loss of the missing pattern of the data, which often conveys information about the predictive label. We present scalable and adaptive algorithms for fair classification with missing values. These algorithms can be combined with any preexisting fairness-intervention algorithm to handle all possible missing patterns while preserving information encoded within the missing patterns. Numerical experiments with state-of-the-art fairness interventions demonstrate that our adaptive algorithms consistently achieve higher fairness and accuracy than impute-then-classify across different datasets.

----

## [0] Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantisation

**Authors**: *Christopher Subia-Waud, Srinandan Dasmahapatra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba178fab60f9306a0b2d7ec8973715a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba178fab60f9306a0b2d7ec8973715a6-Abstract-Conference.html)

**Abstract**:

Weight-sharing quantization has emerged as a technique to reduce energy expenditure during inference in large neural networks by constraining their weights to a limited set of values. However, existing methods often assume weights are treated solely based on value, neglecting the unique role of weight position. This paper proposes a probabilistic framework based on Bayesian neural networks (BNNs) and a variational relaxation to identify which weights can be moved to which cluster center and to what degree based on their individual position-specific learned uncertainty distributions. We introduce a new initialization setting and a regularization term, enabling the training of BNNs with complex dataset-model combinations. Leveraging the flexibility of weight values from probability distributions, we enhance noise resilience and compressibility. Our iterative clustering procedure demonstrates superior compressibility and higher accuracy compared to state-of-the-art methods on both ResNet models and the more complex transformer-based architectures. In particular, our method outperforms the state-of-the-art quantization method top-1 accuracy by 1.6\% on ImageNet using DeiT-Tiny, with its 5 million+ weights now represented by only 296 unique values.  Code available at https://github.com/subiawaud/PWFN.

----

## [0] PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks

**Authors**: *Ian Char, Jeff Schneider*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)

**Abstract**:

Deep reinforcement learning (RL) has shown immense potential for learning to control systems through data alone. However, one challenge deep RL faces is that the full state of the system is often not observable. When this is the case, the policy needs to leverage the history of observations to infer the current state. At the same time, differences between the training and testing environments makes it critical for the policy not to overfit to the sequence of observations it sees at training time. As such, there is an important balancing act between having the history encoder be flexible enough to extract relevant information, yet be robust to changes in the environment. To strike this balance, we look to the PID controller for inspiration. We assert the PID controller's success shows that only summing and differencing are needed to accumulate information over time for many control tasks. Following this principle, we propose two architectures for encoding history: one that directly uses PID features and another that extends these core ideas and can be used in arbitrary control tasks. When compared with prior approaches, our encoders produce policies that are often more robust and achieve better performance on a variety of tracking tasks. Going beyond tracking tasks, our policies achieve 1.7x better performance on average over previous state-of-the-art methods on a suite of locomotion control tasks.

----

## [0] SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems

**Authors**: *Christopher Yeh, Victor Li, Rajeev Datta, Julio Arroyo, Nicolas Christianson, Chi Zhang, Yize Chen, Mohammad Mehdi Hosseini, Azarang Golmohammadi, Yuanyuan Shi, Yisong Yue, Adam Wierman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba74855789913e5ed36f87288af79e5b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba74855789913e5ed36f87288af79e5b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts. In this paper, we present SustainGym, a suite of five environments designed to test the performance of RL algorithms on realistic sustainable energy system tasks, ranging from electric vehicle charging to carbon-aware data center job scheduling. The environments test RL algorithms under realistic distribution shifts as well as in multi-agent settings. We show that standard off-the-shelf RL algorithms leave significant room for improving performance and highlight the challenges ahead for introducing RL to real-world sustainability tasks.

----

## [0] Policy Gradient for Rectangular Robust Markov Decision Processes

**Authors**: *Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y. Levy, Shie Mannor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html)

**Abstract**:

Policy gradient methods have become a standard for training reinforcement learning agents in a scalable and efficient manner. However, they do not account for transition uncertainty, whereas learning robust policies can be computationally expensive. In this paper, we introduce robust policy gradient (RPG), a policy-based method that efficiently solves rectangular robust Markov decision processes (MDPs). We provide a closed-form expression for the worst occupation measure. Incidentally, we find that the worst kernel is a rank-one perturbation of the nominal. Combining the worst occupation measure with a robust Q-value estimation yields an explicit form of the robust gradient. Our resulting RPG can be estimated from data with the same time complexity as its non-robust equivalent. Hence, it relieves the computational burden of convex optimization problems required for training robust policies by current policy gradient approaches.

----

## [0] MultiFusion: Fusing Pre-Trained Models for Multi-Lingual, Multi-Modal Image Generation

**Authors**: *Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andrés Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html)

**Abstract**:

The recent popularity of text-to-image diffusion models (DM) can largely be attributed to the intuitive interface they provide to users. The intended generation can be expressed in natural language, with the model producing faithful interpretations of text prompts. However, expressing complex or nuanced ideas in text alone can be difficult. To ease image generation, we propose MultiFusion that allows one to express complex and nuanced concepts with arbitrarily interleaved inputs of multiple modalities and languages. MultiFusion leverages pre-trained models and aligns them for integration into a cohesive system, thereby avoiding the need for extensive training from scratch. Our experimental results demonstrate the efficient transfer of capabilities from individual modules to the downstream model. Specifically, the fusion of all independent components allows the image generation module to utilize multilingual, interleaved multimodal inputs despite being trained solely on monomodal data in a single language.

----



[Go to the previous page](NIPS-2023-list2500.md)

[Go to the next page](NIPS-2023-list2502.md)

[Go to the catalog section](README.md)