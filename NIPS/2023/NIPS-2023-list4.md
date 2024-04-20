## [0] Zero-shot causal learning

**Authors**: *Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Surina, Michihiro Yasunaga, Sara Oblak, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15ddb1773510075ef44981cdb204330b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15ddb1773510075ef44981cdb204330b-Abstract-Conference.html)

**Abstract**:

Predicting how different interventions will causally affect a specific individual is important in a variety of domains such as personalized medicine, public policy, and online marketing. There are a large number of methods to predict the effect of an existing intervention based on historical data from individuals who received it. However, in many settings it is important to predict the effects of novel interventions (e.g., a newly invented drug), which these methods do not address.Here, we consider zero-shot causal learning: predicting the personalized effects of a novel intervention. We propose CaML, a causal meta-learning framework which formulates the personalized prediction of each intervention's effect as a task. CaML trains a single meta-model across thousands of tasks, each constructed by sampling an intervention, its recipients, and its nonrecipients. By leveraging both intervention information (e.g., a drug's attributes) and individual features (e.g., a patient's history), CaML is able to predict the personalized effects of novel interventions that do not exist at the time of training. Experimental results on real world datasets in large-scale medical claims and cell-line perturbations demonstrate the effectiveness of our approach. Most strikingly, CaML's zero-shot predictions outperform even strong baselines trained directly on data from the test interventions.

----

## [0] Learning Modulated Transformation in GANs

**Authors**: *Ceyuan Yang, Qihang Zhang, Yinghao Xu, Jiapeng Zhu, Yujun Shen, Bo Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f1dbc086bfd94d8c32557b573cbe18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f1dbc086bfd94d8c32557b573cbe18-Abstract-Conference.html)

**Abstract**:

The success of style-based generators largely benefits from style modulation,which helps take care of the cross-instance variation within data. However, theinstance-wise stochasticity is typically introduced via regular convolution, wherekernels interact with features at some fixed locations, limiting its capacity formodeling geometric variation. To alleviate this problem, we equip the generatorin generative adversarial networks (GANs) with a plug-and-play module, termedas modulated transformation module (MTM). This module predicts spatial offsetsunder the control of latent codes, based on which the convolution operation canbe applied at variable locations for different instances, and hence offers the modelan additional degree of freedom to handle geometry deformation. Extensiveexperiments suggest that our approach can be faithfully generalized to variousgenerative tasks, including image generation, 3D-aware image synthesis, andvideo generation, and get compatible with state-of-the-art frameworks withoutany hyper-parameter tuning. It is noteworthy that, towards human generation onthe challenging TaiChi dataset, we improve the FID of StyleGAN3 from 21.36 to13.60, demonstrating the efficacy of learning modulated geometry transformation.Code and models are available at https://github.com/limbo0000/mtm.

----

## [0] Active Negative Loss Functions for Learning with Noisy Labels

**Authors**: *Xichen Ye, Xiaoqiang Li, Songmin Dai, Tong Liu, Yan Sun, Weiqin Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html)

**Abstract**:

Robust loss functions are essential for training deep neural networks in the presence of noisy labels. Some robust loss functions use Mean Absolute Error (MAE) as its necessary component. For example, the recently proposed Active Passive Loss (APL) uses MAE as its passive loss function. However, MAE treats every sample equally, slows down the convergence and can make training difficult. In this work, we propose a new class of theoretically robust passive loss functions different from MAE, namely Normalized Negative Loss Functions (NNLFs), which focus more on memorized clean samples. By replacing the MAE in APL with our proposed NNLFs, we improve APL and propose a new framework called Active Negative Loss (ANL). Experimental results on benchmark and real-world datasets demonstrate that the new set of loss functions created by our ANL framework can outperform state-of-the-art methods. The code is available athttps://github.com/Virusdoll/Active-Negative-Loss.

----

## [0] Compositional Generalization from First Principles

**Authors**: *Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f6a10899f557ce53fe39939af6f930-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f6a10899f557ce53fe39939af6f930-Abstract-Conference.html)

**Abstract**:

Leveraging the compositional nature of our world to expedite learning and facilitate generalization is a hallmark of human perception. In machine learning, on the other hand, achieving compositional generalization has proven to be an elusive goal, even for models with explicit compositional priors. To get a better handle on compositional generalization, we here approach it from the bottom up: Inspired by identifiable representation learning, we investigate compositionality as a property of the data-generating process rather than the data itself. This reformulation enables us to derive mild conditions on only the support of the training distribution and the model architecture, which are sufficient for compositional generalization. We further demonstrate how our theoretical framework applies to real-world scenarios and validate our findings empirically. Our results set the stage for a principled theoretical study of compositional generalization.

----

## [0] PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas

**Authors**: *Zheng Chen, Yan-Pei Cao, Yuan-Chen Guo, Chen Wang, Ying Shan, Song-Hai Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html)

**Abstract**:

Achieving an immersive experience enabling users to explore virtual environments with six degrees of freedom (6DoF) is essential for various applications such as virtual reality (VR). Wide-baseline panoramas are commonly used in these applications to reduce network bandwidth and storage requirements. However, synthesizing novel views from these panoramas remains a key challenge. Although existing neural radiance field methods can produce photorealistic views under narrow-baseline and dense image captures, they tend to overfit the training views when dealing with wide-baseline panoramas due to the difficulty in learning accurate geometry from sparse $360^{\circ}$ views. To address this problem, we propose PanoGRF, Generalizable Spherical Radiance Fields for Wide-baseline Panoramas, which construct spherical radiance fields incorporating $360^{\circ}$ scene priors. Unlike generalizable radiance fields trained on perspective images, PanoGRF avoids the information loss from panorama-to-perspective conversion and directly aggregates geometry and appearance features of 3D sample points from each panoramic view based on spherical projection. Moreover, as some regions of the panorama are only visible from one view while invisible from others under wide baseline settings, PanoGRF incorporates $360^{\circ}$ monocular depth priors into spherical depth estimation to improve the geometry features. Experimental results on multiple panoramic datasets demonstrate that PanoGRF significantly outperforms state-of-the-art generalizable view synthesis methods for wide-baseline panoramas (e.g., OmniSyn) and perspective images (e.g., IBRNet, NeuRay).

----

## [0] A Heat Diffusion Perspective on Geodesic Preserving Dimensionality Reduction

**Authors**: *Guillaume Huguet, Alexander Tong, Edward De Brouwer, Yanlei Zhang, Guy Wolf, Ian Adelstein, Smita Krishnaswamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16063a1c0f0cddd4894585cf44cebb2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16063a1c0f0cddd4894585cf44cebb2c-Abstract-Conference.html)

**Abstract**:

Diffusion-based manifold learning methods have proven useful in representation learning and dimensionality reduction of modern high dimensional, high throughput, noisy datasets. Such datasets are especially present in fields like biology and physics. While it is thought that these methods preserve underlying manifold structure of data by learning a proxy for geodesic distances, no specific theoretical links have been established. Here, we establish such a link via results in Riemannian geometry explicitly connecting heat diffusion to manifold distances. In this process, we also formulate a more general heat kernel based manifold embedding method that we call heat geodesic embeddings. This novel perspective makes clearer the choices available in manifold learning and denoising. Results show that our method outperforms existing state of the art in preserving ground truth manifold distances,  and preserving cluster structure in toy datasets. We also showcase our method on single cell RNA-sequencing datasets with both continuum and cluster structure, where our method enables interpolation of withheld timepoints of data. Finally, we show that parameters of our more general method can be configured to give results similar to PHATE (a state-of-the-art diffusion based manifold learning method) as well as SNE (an attraction/repulsion neighborhood based method that forms the basis of t-SNE).

----

## [0] Finite-Time Analysis of Single-Timescale Actor-Critic

**Authors**: *Xuyang Chen, Lin Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/160adf2dc118a920e7858484b92a37d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/160adf2dc118a920e7858484b92a37d8-Abstract-Conference.html)

**Abstract**:

Actor-critic methods have achieved significant success in many challenging applications. However, its finite-time convergence is still poorly understood in the most practical single-timescale form. Existing works on analyzing single-timescale actor-critic have been limited to i.i.d. sampling or tabular setting for simplicity. We investigate the more practical online single-timescale actor-critic algorithm on continuous state space, where the critic assumes linear function approximation and updates with a single Markovian sample per actor step.  Previous analysis has been unable to establish the convergence for such a challenging scenario. We demonstrate that the online single-timescale actor-critic method provably finds an $\epsilon$-approximate stationary point with $\widetilde{\mathcal{O}}(\epsilon^{-2})$ sample complexity under standard assumptions, which can be further improved to $\mathcal{O}(\epsilon^{-2})$ under the i.i.d. sampling. Our novel framework systematically evaluates and controls the error propagation between the actor and critic. It offers a promising approach for analyzing other single-timescale reinforcement learning algorithms as well.

----

## [0] VanillaNet: the Power of Minimalism in Deep Learning

**Authors**: *Hanting Chen, Yunhe Wang, Jianyuan Guo, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html)

**Abstract**:

At the heart of foundation models is the philosophy of "more is different", exemplified by the astonishing success in computer vision and natural language processing. However, the challenges of optimization and inherent complexity of transformer models call for a paradigm shift towards simplicity. In this study, we introduce VanillaNet, a neural network architecture that embraces elegance in design. By avoiding high depth, shortcuts, and intricate operations like self-attention, VanillaNet is refreshingly concise yet remarkably powerful. Each layer is carefully crafted to be compact and straightforward, with nonlinear activation functions pruned after training to restore the original architecture. VanillaNet overcomes the challenges of inherent complexity, making it ideal for resource-constrained environments. Its easy-to-understand and highly simplified architecture opens new possibilities for efficient deployment. Extensive experimentation demonstrates that VanillaNet delivers performance on par with renowned deep neural networks and vision transformers, showcasing the power of minimalism in deep learning. This visionary journey of VanillaNet has significant potential to redefine the landscape and challenge the status quo of foundation model, setting a new path for elegant and effective model design. Pre-trained models and codes are available at https://github.com/huawei-noah/VanillaNet and https://gitee.com/mindspore/models/tree/master/research/cv/vanillanet

----

## [0] Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs

**Authors**: *Dominik Straub, Matthias Schultheis, Heinz Koeppl, Constantin A. Rothkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16347f6e665376fd9a9a290dbfe0db5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16347f6e665376fd9a9a290dbfe0db5b-Abstract-Conference.html)

**Abstract**:

Inverse optimal control can be used to characterize behavior in sequential decision-making tasks. Most existing work, however, is limited to fully observable or linear systems, or requires the action signals to be known. Here, we introduce a probabilistic approach to inverse optimal control for partially observable stochastic non-linear systems with unobserved action signals, which unifies previous approaches to inverse optimal control with maximum causal entropy formulations. Using an explicit model of the noise characteristics of the sensory and motor systems of the agent in conjunction with local linearization techniques, we derive an approximate likelihood function for the model parameters, which can be computed within a single forward pass. We present quantitative evaluations on stochastic and partially observable versions of two classic control tasks and two human behavioral tasks. Importantly, we show that our method can disentangle perceptual factors and behavioral costs despite the fact that epistemic and pragmatic actions are intertwined in sequential decision-making under uncertainty, such as in active sensing and active learning. The proposed method has broad applicability, ranging from imitation learning to sensorimotor neuroscience.

----

## [0] TIES-Merging: Resolving Interference When Merging Models

**Authors**: *Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)

**Abstract**:

Transfer learning – i.e., further fine-tuning a pre-trained model on a downstream task – can confer significant advantages, including improved downstream performance, faster convergence, and better sample efficiency. These advantages have led to a proliferation of task-specific fine-tuned models, which typically can only perform a single task and do not benefit from one another. Recently, model merging techniques have emerged as a solution to combine multiple task-specific models into a single multitask model without performing additional training. However, existing merging methods often ignore the interference between parameters of different models, resulting in large performance drops when merging multiple models. In this paper, we demonstrate that prior merging techniques inadvertently lose valuable information due to two major sources of interference: (a) interference due to redundant parameter values and (b) disagreement on the sign of a given parameter’s values across models. To address this, we propose our method, TrIm, Elect Sign & Merge (TIES-Merging), which introduces three novel steps when merging models: (1) resetting parameters that only changed a small amount during fine-tuning, (2) resolving sign conflicts, and (3) merging only the parameters that are in alignment with the final agreed-upon sign. We find that TIES-Merging outperforms existing methods in diverse settings covering a range of modalities, domains, number of tasks, model sizes, architectures, and fine-tuning settings. We further analyze the impact of different types of interference on model parameters, highlight the importance of signs, and show that estimating the signs using the validation data could further improve performance.

----

## [0] 3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive Physics under Challenging Scenes

**Authors**: *Haotian Xue, Antonio Torralba, Josh Tenenbaum, Dan Yamins, Yunzhu Li, Hsiao-Yu Tung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/164687cb815daae754d33364716e65e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/164687cb815daae754d33364716e65e6-Abstract-Conference.html)

**Abstract**:

Given a visual scene, humans have strong intuitions about how a scene can evolve over time under given actions. The intuition, often termed visual intuitive physics, is a critical ability that allows us to make effective plans to manipulate the scene to achieve desired outcomes without relying on extensive trial and error. In this paper, we present a framework capable of learning 3D-grounded visual intuitive physics models from videos of complex scenes with fluids. Our method is composed of a conditional Neural Radiance Field (NeRF)-style visual frontend and a 3D point-based dynamics prediction backend, using which we can impose strong relational and structural inductive bias to capture the structure of the underlying environment. Unlike existing intuitive point-based dynamics works that rely on the supervision of dense point trajectory from simulators, we relax the requirements and only assume access to multi-view RGB images and (imperfect) instance masks acquired using color prior. This enables the proposed model to handle scenarios where accurate point estimation and tracking are hard or impossible. We generate datasets including three challenging scenarios involving fluid, granular materials, and rigid objects in the simulation. The datasets do not include any dense particle information so most previous 3D-based intuitive physics pipelines can barely deal with that. We show our model can make long-horizon future predictions by learning from raw images and significantly outperforms models that do not employ an explicit 3D representation space. We also show that once trained, our model can achieve strong generalization in complex scenarios under extrapolate settings.

----

## [0] Entropy-based Training Methods for Scalable Neural Implicit Samplers

**Authors**: *Weijian Luo, Boya Zhang, Zhihua Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1646e34971facbcda3727d1dc28ab635-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1646e34971facbcda3727d1dc28ab635-Abstract-Conference.html)

**Abstract**:

Efficiently sampling from un-normalized target distributions is a fundamental problem in scientific computing and machine learning. Traditional approaches such as Markov Chain Monte Carlo (MCMC) guarantee asymptotically unbiased samples from such distributions but suffer from computational inefficiency, particularly when dealing with high-dimensional targets, as they require numerous iterations to generate a batch of samples. In this paper, we introduce an efficient and scalable neural implicit sampler that overcomes these limitations. The implicit sampler can generate large batches of samples with low computational costs by leveraging a neural transformation that directly maps easily sampled latent vectors to target samples without the need for iterative procedures. To train the neural implicit samplers, we introduce two novel methods: the KL training method and the Fisher training method. The former method minimizes the Kullback-Leibler divergence, while the latter minimizes the Fisher divergence between the sampler and the target distributions. By employing the two training methods, we effectively optimize the neural implicit samplers to learn and generate from the desired target distribution. To demonstrate the effectiveness, efficiency, and scalability of our proposed samplers, we evaluate them on three sampling benchmarks with different scales. These benchmarks include sampling from 2D targets, Bayesian inference, and sampling from high-dimensional energy-based models (EBMs). Notably, in the experiment involving high-dimensional EBMs, our sampler produces samples that are comparable to those generated by MCMC-based methods while being more than 100 times more efficient, showcasing the efficiency of our neural sampler. Besides the theoretical contributions and strong empirical performances, the proposed neural samplers and corresponding training methods will shed light on further research on developing efficient samplers for various applications beyond the ones explored in this study.

----

## [0] Direct Diffusion Bridge using Data Consistency for Inverse Problems

**Authors**: *Hyungjin Chung, Jeongsol Kim, Jong Chul Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/165b0e600b1721bd59526131eb061092-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/165b0e600b1721bd59526131eb061092-Abstract-Conference.html)

**Abstract**:

Diffusion model-based inverse problem solvers have shown impressive performance, but are limited in speed, mostly as they require reverse diffusion sampling starting from noise. Several recent works have tried to alleviate this problem by building a diffusion process, directly bridging the clean and the corrupted for specific inverse problems. In this paper, we first unify these existing works under the name Direct Diffusion Bridges (DDB), showing that while motivated by different theories, the resulting algorithms only differ in the choice of parameters. Then, we highlight a critical limitation of the current DDB framework, namely that it does not ensure data consistency. To address this problem, we propose a modified inference procedure that imposes data consistency without the need for fine-tuning. We term the resulting method data Consistent DDB (CDDB), which outperforms its inconsistent counterpart in terms of both perception and distortion metrics, thereby effectively pushing the Pareto-frontier toward the optimum. Our proposed method achieves state-of-the-art results on both evaluation criteria, showcasing its superiority over existing methods. Code is open-sourced here.

----

## [0] Mask Propagation for Efficient Video Semantic Segmentation

**Authors**: *Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, Bohan Zhuang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/167bcf2af2cd08fcf75b932022db0311-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/167bcf2af2cd08fcf75b932022db0311-Abstract-Conference.html)

**Abstract**:

Video Semantic Segmentation (VSS) involves assigning a semantic label to each pixel in a video sequence. Prior work in this field has demonstrated promising results by extending image semantic segmentation models to exploit temporal relationships across video frames; however, these approaches often incur significant computational costs. In this paper, we propose an efficient mask propagation framework for VSS, called MPVSS. Our approach first employs a strong query-based image segmentor on sparse key frames to generate accurate binary masks and class predictions. We then design a flow estimation module utilizing the learned queries to generate a set of segment-aware flow maps, each associated with a mask prediction from the key frame. Finally, the mask-flow pairs are warped to serve as the mask predictions for the non-key frames. By reusing predictions from key frames, we circumvent the need to process a large volume of video frames individually with resource-intensive segmentors, alleviating temporal redundancy and significantly reducing computational costs. Extensive experiments on VSPW and Cityscapes demonstrate that our mask propagation framework achieves SOTA accuracy and efficiency trade-offs. For instance, our best model with Swin-L backbone outperforms the SOTA MRCFA using MiT-B5 by 4.0% mIoU, requiring only 26% FLOPs on the VSPW dataset. Moreover, our framework reduces up to 4Ã— FLOPs compared to the per-frame Mask2Former baseline with only up to 2% mIoU degradation on the Cityscapes validation set. Code is available at https://github.com/ziplab/MPVSS.

----

## [0] Private Distribution Learning with Public Data: The View from Sample Compression

**Authors**: *Shai Ben-David, Alex Bie, Clément L. Canonne, Gautam Kamath, Vikrant Singhal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1687466683649e8bdcdec0e3f5c8de64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1687466683649e8bdcdec0e3f5c8de64-Abstract-Conference.html)

**Abstract**:

We study the problem of private distribution learning with access to public data. In this setup, which we refer to as *public-private learning*, the learner is given public and private samples drawn from an unknown distribution $p$ belonging to a class $\mathcal Q$, with the goal of outputting an estimate of $p$ while adhering to privacy constraints (here, pure differential privacy) only with respect to the private samples.     We show that the public-private learnability of a class $\mathcal Q$ is connected to the existence of a sample compression scheme for $\mathcal Q$, as well as to an intermediate notion we refer to as \emph{list learning}. Leveraging this connection: (1) approximately recovers previous results on Gaussians over $\mathbb R^d$; and (2) leads to new ones, including sample complexity upper bounds for arbitrary $k$-mixtures of Gaussians over $\mathbb R^d$, results for agnostic and distribution-shift resistant learners, as well as closure properties for public-private learnability under taking mixtures and products of distributions. Finally, via the connection to list learning, we show that for Gaussians in $\mathbb R^d$, at least $d$ public samples are necessary for private learnability, which is close to the known upper bound of $d+1$ public samples.

----

## [0] ChessGPT: Bridging Policy Learning and Language Modeling

**Authors**: *Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

When solving decision-making tasks, humans typically depend on information from two key sources: (1) Historical policy data, which provides interaction replay from the environment, and (2) Analytical insights in natural language form, exposing the invaluable thought process or strategic considerations. Despite this, the majority of preceding research focuses on only one source: they either use historical replay exclusively to directly learn policy or value functions, or engaged in language model training utilizing mere language corpus. In this paper, we argue that a powerful autonomous agent should cover both sources. Thus, we propose ChessGPT, a GPT model bridging policy learning and language modeling by integrating data from these two sources in Chess games. Specifically, we build a large-scale game and language dataset related to chess. Leveraging the dataset, we showcase two model examples ChessCLIP and ChessGPT, integrating policy learning and language modeling. Finally, we propose a full evaluation framework for evaluating language model's chess ability. Experimental results validate our model and dataset's effectiveness. We open source our code, model, and dataset at https://github.com/waterhorse1/ChessGPT.

----

## [0] Fitting trees to 𝓁1-hyperbolic distances

**Authors**: *Joon-Hyeok Yim, Anna A. Gilbert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16bce4070c4e23434451b180348e3814-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16bce4070c4e23434451b180348e3814-Abstract-Conference.html)

**Abstract**:

Building trees to represent or to fit distances is a critical component of phylogenetic analysis, metric embeddings, approximation algorithms, geometric graph neural nets, and the analysis of hierarchical data. Much of the previous algorithmic work, however, has focused on generic metric spaces (i.e., those with no \emph{a priori} constraints). Leveraging several ideas from the mathematical analysis of hyperbolic geometry and geometric group theory, we study the tree fitting problem as finding the relation between the hyperbolicity (ultrametricity) vector and the error of tree (ultrametric) embedding. That is, we define a vector of hyperbolicity (ultrametric) values over all triples of points and compare the $\ell_p$ norms of this vector with the $\ell_q$ norm of the distortion of the best tree fit to the distances. This formulation allows us to define the average hyperbolicity (ultrametricity) in terms of a normalized $\ell_1$ norm of the hyperbolicity vector. Furthermore, we can interpret the classical tree fitting result of Gromov as a $p = q = \infty$ result. We present an algorithm \textsc{HCCRootedTreeFit} such that the $\ell_1$ error of the output embedding is analytically bounded in terms of the $\ell_1$-norm of the hyperbolicity vector (i.e., $p = q = 1$) and that this result is tight. Furthermore, this algorithm has significantly different theoretical and empirical performance as compared to Gromov's result and related algorithms. Finally, we show using \textsc{HCCRootedTreeFit} and related tree fitting algorithms, that supposedly standard data sets for hierarchical data analysis and geometric graph neural networks have radically different tree fits than those of synthetic, truly tree-like data sets, suggesting that a much more refined analysis of these standard data sets is called for.

----

## [0] Learning Robust Statistics for Simulation-based Inference under Model Misspecification

**Authors**: *Daolang Huang, Ayush Bharti, Amauri H. Souza, Luigi Acerbi, Samuel Kaski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16c5b4102a6b6eb061e502ce6736ad8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16c5b4102a6b6eb061e502ce6736ad8a-Abstract-Conference.html)

**Abstract**:

Simulation-based inference (SBI) methods such as approximate Bayesian computation (ABC),  synthetic likelihood, and neural posterior estimation (NPE) rely on simulating statistics to infer parameters of intractable likelihood models. However, such methods are known to yield untrustworthy and misleading inference outcomes under model misspecification, thus hindering their widespread applicability. In this work, we propose the first general approach to handle model misspecification that works across different classes of SBI methods. Leveraging the fact that the choice of statistics determines the degree of misspecification in SBI, we introduce a regularized loss function that penalizes those statistics that increase the mismatch between the data and the model. Taking NPE and ABC as use cases, we demonstrate the superior performance of our method on high-dimensional time-series models that are artificially misspecified. We also apply our method to real data from the field of radio propagation where the model is known to be misspecified. We show empirically that the method yields robust inference in misspecified scenarios, whilst still being accurate when the model is well-specified.

----

## [0] Block-State Transformers

**Authors**: *Jonathan Pilault, Mahan Fathi, Orhan Firat, Chris Pal, Pierre-Luc Bacon, Ross Goroshin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16ccd203e9e3696a7ab0dcf568316379-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16ccd203e9e3696a7ab0dcf568316379-Abstract-Conference.html)

**Abstract**:

State space models (SSMs) have shown impressive results on tasks that require modeling long-range dependencies and efficiently scale to long sequences owing to their subquadratic runtime complexity.Originally designed for continuous signals, SSMs have shown superior performance on a plethora of tasks, in vision and audio; however, SSMs still lag Transformer performance in Language Modeling tasks.In this work, we propose a hybrid layer named Block-State Transformer (BST), that internally combines an SSM sublayer for long-range contextualization, and a Block Transformer sublayer for short-term representation of sequences.We study three different, and completely parallelizable, variants that integrate SSMs and block-wise attention.We show that our model outperforms similar Transformer-based architectures on language modeling perplexity and generalizes to longer sequences. In addition, the Block-State Transformer demonstrates a more than tenfold increase in speed at the layer level compared to the Block-Recurrent Transformer when model parallelization is employed.

----

## [0] Explaining Predictive Uncertainty with Information Theoretic Shapley Values

**Authors**: *David S. Watson, Joshua O'Hara, Niek Tax, Richard Mudd, Ido Guy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html)

**Abstract**:

Researchers in explainable artificial intelligence have developed numerous methods for helping users understand the predictions of complex supervised learning models. By contrast, explaining the $\textit{uncertainty}$ of model outputs has received relatively little attention. We adapt the popular Shapley value framework to explain various types of predictive uncertainty, quantifying each feature's contribution to the conditional entropy of individual model outputs. We consider games with modified characteristic functions and find deep connections between the resulting Shapley values and fundamental quantities from information theory and conditional independence testing. We outline inference procedures for finite sample error rate control with provable guarantees, and implement efficient algorithms that perform well in a range of experiments on real and simulated data. Our method has applications to covariate shift detection, active learning, feature selection, and active feature-value acquisition.

----

## [0] Learning to Taste: A Multimodal Wine Dataset

**Authors**: *Thoranna Bender, Simon Møe Sørensen, Alireza Kashani, Kristjan Eldjarn Hjorleifsson, Grethe Hyldig, Søren Hauberg, Serge J. Belongie, Frederik Warburg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/170035f97007fdfa665880107b56f384-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/170035f97007fdfa665880107b56f384-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present WineSensed, a large multimodal wine dataset for studying the relations between visual perception, language, and flavor. The dataset encompasses 897k images of wine labels and 824k reviews of wines curated from the Vivino platform. It has over 350k unique vintages, annotated with year, region, rating, alcohol percentage, price, and grape composition. We obtained fine-grained flavor annotations on a subset by conducting a wine-tasting experiment with 256 participants who were asked to rank wines based on their similarity in flavor, resulting in more than 5k pairwise flavor distances. We propose a low-dimensional concept embedding algorithm that combines human experience with automatic machine similarity kernels. We demonstrate that this shared concept embedding space improves upon separate embedding spaces for coarse flavor classification  (alcohol percentage, country, grape, price, rating) and representing human perception of flavor.

----

## [0] CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning

**Authors**: *Charles Guille-Escuret, Pau Rodríguez, David Vázquez, Ioannis Mitliagkas, João Monteiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1700ad4e6252e8f2955909f96367b34d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1700ad4e6252e8f2955909f96367b34d-Abstract-Conference.html)

**Abstract**:

Handling out-of-distribution (OOD) samples has become a major stake in the real-world deployment of machine learning systems. This work explores the use of self-supervised contrastive learning to the simultaneous detection of two types of OOD samples: unseen classes and adversarial perturbations. First, we pair self-supervised contrastive learning with the maximum mean discrepancy (MMD) two-sample test. This approach enables us to robustly test whether two independent sets of samples originate from the same distribution, and we demonstrate its effectiveness by discriminating between CIFAR-10 and CIFAR-10.1 with higher confidence than previous work. Motivated by this success, we introduce CADet (Contrastive Anomaly Detection), a novel method for OOD detection of single samples. CADet draws inspiration from MMD, but leverages the similarity between contrastive transformations of a same sample. CADet outperforms existing adversarial detection methods in identifying adversarially perturbed samples on ImageNet and achieves comparable performance to unseen label detection methods on two challenging benchmarks: ImageNet-O and iNaturalist. Significantly, CADet is fully self-supervised and requires neither labels for in-distribution samples nor access to OOD examples.

----

## [0] PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning

**Authors**: *Neeratyoy Mallik, Edward Bergman, Carl Hvarfner, Danny Stoll, Maciej Janowski, Marius Lindauer, Luigi Nardi, Frank Hutter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html)

**Abstract**:

Hyperparameters of Deep Learning (DL) pipelines are crucial for their downstream performance. While a large number of methods for Hyperparameter Optimization (HPO) have been developed, their incurred costs are often untenable for modern DL.Consequently, manual experimentation is still the most prevalent approach to optimize hyperparameters, relying on the researcher's intuition, domain knowledge, and cheap preliminary explorations.To resolve this misalignment between HPO algorithms and DL researchers, we propose PriorBand, an HPO algorithm tailored to DL, able to utilize both expert beliefs and cheap proxy tasks. Empirically, we demonstrate PriorBand's efficiency across a range of DL benchmarks and show its gains under informative expert input and robustness against poor expert beliefs.

----

## [0] Towards Efficient Image Compression Without Autoregressive Models

**Authors**: *Muhammad Salman Ali, Yeongwoong Kim, Maryam Qamar, Sung-Chang Lim, Donghyun Kim, Chaoning Zhang, Sung-Ho Bae, Hui Yong Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/170dc3e41f2d03e327e04dbab0fccbfb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/170dc3e41f2d03e327e04dbab0fccbfb-Abstract-Conference.html)

**Abstract**:

Recently, learned image compression (LIC) has garnered increasing interest with its rapidly improving performance surpassing conventional codecs. A key ingredient of LIC is a hyperprior-based entropy model, where the underlying joint probability of the latent image features is modeled as a product of Gaussian distributions from each latent element. Since latents from the actual images are not spatially independent, autoregressive (AR) context based entropy models were proposed to handle the discrepancy between the assumed distribution and the actual distribution. Though the AR-based models have proven effective, the computational complexity is significantly increased due to the inherent sequential nature of the algorithm. In this paper, we present a novel alternative to the AR-based approach that can provide a significantly better trade-off between performance and complexity. To minimize the discrepancy, we introduce a correlation loss that forces the latents to be spatially decorrelated and better fitted to the independent probability model. Our correlation loss is proved to act as a general plug-in for the hyperprior (HP) based learned image compression methods. The performance gain from our correlation loss is ‘free’ in terms of computation complexity for both inference time and decoding time. To our knowledge, our method  gives the best trade-off between the complexity and performance: combined with the Checkerboard-CM, it attains 90% and when combined with ChARM-CM, it attains 98% of the AR-based BD-Rate gains yet is around 50 times and 30 times faster than AR-based methods respectively

----

## [0] De novo Drug Design using Reinforcement Learning with Multiple GPT Agents

**Authors**: *Xiuyuan Hu, Guoqing Liu, Yang Zhao, Hao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1737656c4dc65027939e47e4587ce95e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1737656c4dc65027939e47e4587ce95e-Abstract-Conference.html)

**Abstract**:

De novo drug design is a pivotal issue in pharmacology and a new area of focus in AI for science research. A central challenge in this field is to generate molecules with specific properties while also producing a wide range of diverse candidates. Although advanced technologies such as transformer models and reinforcement learning have been applied in drug design, their potential has not been fully realized. Therefore, we propose MolRL-MGPT, a reinforcement learning algorithm with multiple GPT agents for drug molecular generation. To promote molecular diversity, we encourage the agents to collaborate in searching for desirable molecules in diverse directions. Our algorithm has shown promising results on the GuacaMol benchmark and exhibits efficacy in designing inhibitors against SARS-CoV-2 protein targets. The codes are available at: https://github.com/HXYfighter/MolRL-MGPT.

----

## [0] Pointwise uncertainty quantification for sparse variational Gaussian process regression with a Brownian motion prior

**Authors**: *Luke Travis, Kolyan Ray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/176a579942089c4cdc70136c567932ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/176a579942089c4cdc70136c567932ab-Abstract-Conference.html)

**Abstract**:

We study pointwise estimation and uncertainty quantification for a sparse variational Gaussian process method with eigenvector inducing variables. For a rescaled Brownian motion prior, we derive theoretical guarantees and limitations for the frequentist size and coverage of pointwise credible sets. For sufficiently many inducing variables, we precisely characterize the asymptotic frequentist coverage, deducing when credible sets from this variational method are conservative and when overconfident/misleading. We numerically illustrate the applicability of our results and discuss connections with other common Gaussian process priors.

----

## [0] Few-shot Generation via Recalling Brain-Inspired Episodic-Semantic Memory

**Authors**: *Zhibin Duan, Zhiyi Lv, Chaojie Wang, Bo Chen, Bo An, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html)

**Abstract**:

Aimed at adapting a generative model to a novel generation task with only a few given data samples, the capability of few-shot generation is crucial for many real-world applications with limited data, \emph{e.g.}, artistic domains.Instead of training from scratch, recent works tend to leverage the prior knowledge stored in previous datasets, which is quite similar to the memory mechanism of human intelligence, but few of these works directly imitate the memory-recall mechanism that humans make good use of in accomplishing creative tasks,  \emph{e.g.}, painting and writing.Inspired by the memory mechanism of human brain, in this work, we carefully design a variational structured memory module (VSM), which can simultaneously store both episodic and semantic memories to assist existing generative models efficiently recall these memories during sample generation.Meanwhile, we introduce a bionic memory updating strategy for the conversion between episodic and semantic memories, which can also model the uncertainty during conversion.Then, we combine the developed VSM with various generative models under the Bayesian framework, and evaluate these memory-augmented generative models with few-shot generation tasks, demonstrating the effectiveness of our methods.

----

## [0] Balancing memorization and generalization in RNNs for high performance brain-machine Interfaces

**Authors**: *Joseph T. Costello, Hisham Temmar, Luis Cubillos, Matthew Mender, Dylan Wallace, Matt S. Willsey, Parag G. Patil, Cynthia A. Chestek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html)

**Abstract**:

Brain-machine interfaces (BMIs) can restore motor function to people with paralysis but are currently limited by the accuracy of real-time decoding algorithms. Recurrent neural networks (RNNs) using modern training techniques have shown promise in accurately predicting movements from neural signals but have yet to be rigorously evaluated against other decoding algorithms in a closed-loop setting. Here we compared RNNs to other neural network architectures in real-time, continuous decoding of finger movements using intracortical signals from nonhuman primates. Across one and two finger online tasks, LSTMs (a type of RNN) outperformed convolutional and transformer-based neural networks, averaging 18% higher throughput than the convolution network. On simplified tasks with a reduced movement set, RNN decoders were allowed to memorize movement patterns and matched able-bodied control. Performance gradually dropped as the number of distinct movements increased but did not go below fully continuous decoder performance. Finally, in a two-finger task where one degree-of-freedom had poor input signals, we recovered functional control using RNNs trained to act both like a movement classifier and continuous decoder. Our results suggest that RNNs can enable functional real-time BMI control by learning and generating accurate movement patterns.

----

## [0] Saddle-to-Saddle Dynamics in Diagonal Linear Networks

**Authors**: *Scott Pesme, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html)

**Abstract**:

In this paper we fully describe the trajectory of gradient flow over $2$-layer diagonal linear networks for the regression setting in the limit of vanishing initialisation. We show that the limiting flow successively jumps from a saddle of the training loss to another until reaching the minimum $\ell_1$-norm solution. We explicitly characterise the visited saddles as well as the jump times through a recursive algorithm reminiscent of the LARS algorithm used for computing the Lasso path.  Starting from the zero vector, coordinates are successively activated until the minimum $\ell_1$-norm solution is recovered, revealing an incremental learning. Our proof leverages a convenient arc-length time-reparametrisation which enables to keep track of the transitions between the jumps. Our analysis requires negligible assumptions on the data, applies to both under and overparametrised settings and covers complex cases where there is no monotonicity of the number of active coordinates. We provide numerical experiments to support our findings.

----

## [0] Encoding Human Behavior in Information Design through Deep Learning

**Authors**: *Guanghui Yu, Wei Tang, Saumik Narayanan, Chien-Ju Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17d0a21da4ec2c12b4f07fa2e34e4d6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17d0a21da4ec2c12b4f07fa2e34e4d6c-Abstract-Conference.html)

**Abstract**:

We initiate the study of $\textit{behavioral information design}$ through deep learning. In information design, a $\textit{sender}$ aims to persuade a $\textit{receiver}$ to take certain actions by strategically revealing information. We address scenarios in which the receiver might exhibit different behavior patterns other than the standard Bayesian rational assumption. We propose HAIDNet, a neural-network-based optimization framework for information design that can adapt to multiple representations of human behavior. Through extensive simulation, we show that HAIDNet can not only recover information policies that are near-optimal compared with known analytical solutions, but also can extend to designing information policies for settings that are computationally challenging (e.g., when there are multiple receivers) or for settings where there are no known solutions in general (e.g., when the receiver behavior does not follow the Bayesian rational assumption). We also conduct real-world human-subject experiments and demonstrate that our framework can capture human behavior from data and lead to more effective information policy for real-world human receivers.

----

## [0] Collaboratively Learning Linear Models with Structured Missing Data

**Authors**: *Chen Cheng, Gary Cheng, John C. Duchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17f158c25b08758cf650130f7f173e51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17f158c25b08758cf650130f7f173e51-Abstract-Conference.html)

**Abstract**:

We study the problem of collaboratively learning least squares estimates for $m$ agents. Each agent observes a different subset of the features---e.g., containing data collected from sensors of varying resolution. Our goal is to determine how to coordinate the agents in order to produce the best estimator for each agent. We propose a distributed, semi-supervised algorithm Collab, consisting of three steps: local training, aggregation, and distribution. Our procedure does not require communicating the labeled data, making it communication efficient and useful in settings where the labeled data is inaccessible. Despite this handicap, our procedure is nearly asymptotically, local-minimax optimal---even among estimators allowed to communicate the labeled data such as imputation methods. We test our method on US Census data. We also discuss generalizations of our method to non-Gaussian feature settings, non-linear settings, and Federated Learning.

----

## [0] Generating Behaviorally Diverse Policies with Latent Diffusion Models

**Authors**: *Shashank Hegde, Sumeet Batra, K. R. Zentner, Gaurav S. Sukhatme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180d4373aca26bd86bf45fc50d1a709f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180d4373aca26bd86bf45fc50d1a709f-Abstract-Conference.html)

**Abstract**:

Recent progress in Quality Diversity Reinforcement Learning (QD-RL) has enabled learning a collection of behaviorally diverse, high performing policies. However, these methods typically involve storing thousands of policies, which results in high space-complexity and poor scaling to additional behaviors. Condensing the archive into a single model while retaining the performance and coverage of theoriginal collection of policies has proved challenging. In this work, we propose using diffusion models to distill the archive into a single generative model over policy parameters. We show that our method achieves a compression ratio of 13x while recovering 98% of the original rewards and 89% of the original humanoid archive coverage. Further, the conditioning mechanism of diffusion models allowsfor flexibly selecting and sequencing behaviors, including using language. Project website: https://sites.google.com/view/policydiffusion/home.

----

## [0] Incentives in Private Collaborative Machine Learning

**Authors**: *Rachael Hwee Ling Sim, Yehong Zhang, Nghia Hoang, Xinyi Xu, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180f1a1de4244c009ff0848c55ae54a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180f1a1de4244c009ff0848c55ae54a5-Abstract-Conference.html)

**Abstract**:

Collaborative machine learning involves training models on data from multiple parties but must incentivize their participation. Existing data valuation methods fairly value and reward each party based on  shared data or model parameters but neglect the privacy risks involved. To address this, we introduce differential privacy (DP) as an incentive. Each party can select its required DP guarantee and perturb its sufficient statistic (SS) accordingly. The mediator values the perturbed SS by the Bayesian surprise it elicits about the model parameters. As our valuation function enforces a privacy-valuation trade-off, parties are deterred from selecting excessive DP guarantees that reduce the utility of the grand coalition's model. Finally, the mediator rewards each party with different posterior samples of the model parameters. Such rewards still satisfy existing incentives like fairness but additionally preserve DP and a high similarity to the grand coalition's posterior. We empirically demonstrate the effectiveness and practicality of our approach on synthetic and real-world datasets.

----

## [0] VideoComposer: Compositional Video Synthesis with Motion Controllability

**Authors**: *Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html)

**Abstract**:

The pursuit of controllability as a higher standard of visual content creation has yielded remarkable progress in customizable image synthesis. However, achieving controllable video synthesis remains challenging due to the large variation of temporal dynamics and the requirement of cross-frame temporal consistency. Based on the paradigm of compositional generation, this work presents VideoComposer that allows users to flexibly compose a video with textual conditions, spatial conditions, and more importantly temporal conditions. Specifically, considering the characteristic of video data, we introduce the motion vector from compressed videos as an explicit control signal to provide guidance regarding temporal dynamics. In addition, we develop a Spatio-Temporal Condition encoder (STC-encoder) that serves as a unified interface to effectively incorporate the spatial and temporal relations of sequential inputs, with which the model could make better use of temporal conditions and hence achieve higher inter-frame consistency. Extensive experimental results suggest that VideoComposer is able to control the spatial and temporal patterns simultaneously within a synthesized video in various forms, such as text description, sketch sequence, reference video, or even simply hand-crafted motions. The code and models are publicly available athttps://videocomposer.github.io.

----

## [0] Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline RL

**Authors**: *Peng Cheng, Xianyuan Zhan, Zhi-Hao Wu, Wenjia Zhang, Youfang Lin, Shoucheng Song, Han Wang, Li Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/181a027913d36bc0a8857c0da661d621-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/181a027913d36bc0a8857c0da661d621-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) offers an appealing approach to real-world tasks by learning policies from pre-collected datasets without interacting with the environment. However, the performance of existing offline RL algorithms heavily depends on the scale and state-action space coverage of datasets. Real-world data collection is often expensive and uncontrollable, leading to small and narrowly covered datasets and posing significant challenges for practical deployments of offline RL. In this paper, we provide a new insight that leveraging the fundamental symmetry of system dynamics can substantially enhance offline RL performance under small datasets. Specifically, we propose a Time-reversal symmetry (T-symmetry) enforced Dynamics Model (TDM), which establishes consistency between a pair of forward and reverse latent dynamics. TDM provides both well-behaved representations for small datasets and a new reliability measure for OOD samples based on compliance with the T-symmetry. These can be readily used to construct a new offline RL algorithm (TSRL) with less conservative policy constraints and a reliable latent space data augmentation procedure. Based on extensive experiments, we find TSRL achieves great performance on small benchmark datasets with as few as 1% of the original samples, which significantly outperforms the recent offline RL algorithms in terms of data efficiency and generalizability. Code is available at:https://github.com/pcheng2/TSRL

----

## [0] Initialization-Dependent Sample Complexity of Linear Predictors and Neural Networks

**Authors**: *Roey Magen, Ohad Shamir*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18210aa6209b9adfc97b8c17c3741d95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/18210aa6209b9adfc97b8c17c3741d95-Abstract-Conference.html)

**Abstract**:

We provide several new results on the sample complexity of vector-valued linear predictors (parameterized by a matrix), and more generally neural networks. Focusing on size-independent bounds, where only the Frobenius norm distance of the parameters from some fixed reference matrix $W_0$ is controlled, we show that the sample complexity behavior can be surprisingly different than what we may expect considering the well-studied setting of scalar-valued linear predictors. This also leads to new sample complexity bounds for feed-forward neural networks, tackling some open questions in the literature, and establishing a new convex linear prediction problem that is provably learnable without uniform convergence.

----

## [0] Incentivizing Honesty among Competitors in Collaborative Learning and Optimization

**Authors**: *Florian E. Dorner, Nikola Konstantinov, Georgi Pashaliev, Martin T. Vechev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/182b39a4458fb4a9a8d6871a6671ff3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/182b39a4458fb4a9a8d6871a6671ff3e-Abstract-Conference.html)

**Abstract**:

Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entityâ€™s data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the effectiveness of our incentive scheme on a standard non-convex federated learning benchmark. Our work shows that explicitly modeling the incentives and actions of dishonest clients, rather than assuming them malicious, can enable strong robustness guarantees for collaborative learning.

----

## [0] SNAP: Self-Supervised Neural Maps for Visual Positioning and Semantic Understanding

**Authors**: *Paul-Edouard Sarlin, Eduard Trulls, Marc Pollefeys, Jan Hosang, Simon Lynen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/182c433412b33c14e32a7c4fc2c3e290-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/182c433412b33c14e32a7c4fc2c3e290-Abstract-Conference.html)

**Abstract**:

Semantic 2D maps are commonly used by humans and machines for navigation purposes, whether it's walking or driving. However, these maps have limitations: they lack detail, often contain inaccuracies, and are difficult to create and maintain, especially in an automated fashion. Can we use  raw imagery to automatically create better maps that can be easily interpreted by both humans and machines? We introduce SNAP, a deep network that learns rich 2D neural maps from ground-level and overhead images. We train our model to align neural maps estimated from different inputs, supervised only with camera poses over tens of millions of StreetView images. SNAP can resolve the location of challenging image queries beyond the reach of traditional methods, outperforming the state of the art in localization by a large margin. Moreover, our neural maps encode not only geometry and appearance but also high-level semantics, discovered without explicit supervision. This enables effective pre-training for data-efficient semantic scene understanding, with the potential to unlock cost-efficient creation of more detailed maps.

----

## [0] Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research

**Authors**: *Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, John D. Co-Reyes, Rishabh Agarwal, Rebecca Roelofs, Yao Lu, Nico Montali, Paul Mougin, Zoey Yang, Brandyn White, Aleksandra Faust, Rowan McAllister, Dragomir Anguelov, Benjamin Sapp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Simulation is an essential tool to develop and benchmark autonomous vehicle planning software in a safe and cost-effective manner. However, realistic simulation requires accurate modeling of multi-agent interactive behaviors to be trustworthy, behaviors which can be highly nuanced and complex. To address these challenges, we introduce Waymax, a new data-driven simulator for autonomous driving in multi-agent scenes, designed for large-scale simulation and testing.  Waymax uses publicly-released, real-world driving data (e.g., the Waymo Open Motion Dataset) to initialize or play back a diverse set of multi-agent simulated scenarios.   It runs entirely on hardware accelerators such as TPUs/GPUs and supports in-graph simulation for training, making it suitable for modern large-scale, distributed machine learning workflows. To support online training and evaluation, Waymax includes several learned and hard-coded behavior models that allow for realistic interaction within simulation. To supplement Waymax, we benchmark a suite of popular imitation and reinforcement learning algorithms with ablation studies on different design decisions, where we highlight the effectiveness of routes as guidance for planning agents and the ability of RL to overfit against simulated agents.

----

## [0] Equal Opportunity of Coverage in Fair Regression

**Authors**: *Fangxin Wang, Lu Cheng, Ruocheng Guo, Kay Liu, Philip S. Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1849b94ed817ae7043a6b6934ef410c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1849b94ed817ae7043a6b6934ef410c1-Abstract-Conference.html)

**Abstract**:

We study fair machine learning (ML) under predictive uncertainty to enable reliable and trustworthy decision-making. The seminal work of 'equalized coverage' proposed an uncertainty-aware fairness notion. However, it does not guarantee equal coverage rates across more fine-grained groups (e.g., low-income females) conditioning on the true label and is biased in the assessment of uncertainty. To tackle these limitations, we propose a new uncertainty-aware fairness -- Equal Opportunity of Coverage (EOC) -- that aims to achieve two properties: (1) coverage rates for different groups with similar outcomes are close, and (2) the coverage rate for the entire population remains at a predetermined level. Further, the prediction intervals should be narrow to be informative. We propose Binned Fair Quantile Regression (BFQR), a distribution-free post-processing method to improve EOC with reasonable width for any trained ML models. It first calibrates a hold-out set to bound deviation from EOC, then leverages conformal prediction to maintain EOC on a test set, meanwhile optimizing prediction interval width. Experimental results demonstrate the effectiveness of our method in improving EOC.

----

## [0] Nonparametric Teaching for Multiple Learners

**Authors**: *Chen Zhang, Xiaofeng Cao, Weiyang Liu, Ivor W. Tsang, James T. Kwok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/184a03a3ad07e8897c62461c02634b02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/184a03a3ad07e8897c62461c02634b02-Abstract-Conference.html)

**Abstract**:

We study the problem of teaching multiple learners simultaneously in the nonparametric iterative teaching setting, where the teacher iteratively provides examples to the learner for accelerating the acquisition of a target concept. This problem is motivated by the gap between current single-learner teaching setting and the real-world scenario of human instruction where a teacher typically imparts knowledge to multiple students. Under the new problem formulation, we introduce a novel framework -- Multi-learner Nonparametric Teaching (MINT). In MINT, the teacher aims to instruct multiple learners, with each learner focusing on learning a scalar-valued target model. To achieve this, we frame the problem as teaching a vector-valued target model and extend the target model space from a scalar-valued reproducing kernel Hilbert space used in single-learner scenarios to a vector-valued space. Furthermore, we demonstrate that MINT offers significant teaching speed-up over repeated single-learner teaching, particularly when the multiple learners can communicate with each other. Lastly, we conduct extensive experiments to validate the practicality and efficiency of MINT.

----

## [0] EvoPrompting: Language Models for Code-Level Neural Architecture Search

**Authors**: *Angelica Chen, David Dohan, David R. So*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html)

**Abstract**:

Given the recent impressive accomplishments of language models (LMs) for code generation, we explore the use of LMs as general adaptive mutation and crossover operators for an evolutionary neural architecture search (NAS) algorithm.While NAS still proves too difficult a task for LMs to succeed at solely through prompting, we find that the combination of evolutionary prompt engineering with soft prompt-tuning, a method we term EvoPrompting, consistently finds diverse and high performing models. We first demonstrate that EvoPrompting is effective on the computationally efficient MNIST-1D dataset, where EvoPrompting produces convolutional architecture variants that outperform both those designed by human experts and naive few-shot prompting in terms of accuracy and model size. We then apply our method to searching for graph neural networks on the CLRS Algorithmic Reasoning Benchmark, where EvoPrompting is able to design novel architectures that outperform current state-of-the-art models on 21 out of 30 algorithmic reasoning tasks while maintaining similar model size. EvoPrompting is successful at designing accurate and efficient neural network architectures across a variety of machine learning tasks, while also being general enough for easy adaptation to other tasks beyond neural network design.

----

## [0] Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction

**Authors**: *Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1857d2e8f51ed219ca0c2663239b38e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1857d2e8f51ed219ca0c2663239b38e5-Abstract-Conference.html)

**Abstract**:

Reconstructing 3D clothed human avatars from single images is a challenging task, especially when encountering complex poses and loose clothing. Current methods exhibit limitations in performance, largely attributable to their dependence on insufficient 2D image features and inconsistent query methods. Owing to this, we present the Global-correlated 3D-decoupling Transformer for clothed Avatar reconstruction (GTA), a novel transformer-based architecture that reconstructs clothed human avatars from monocular images. Our approach leverages transformer architectures by utilizing a Vision Transformer model as an encoder for capturing global-correlated image features. Subsequently, our innovative 3D-decoupling decoder employs cross-attention to decouple tri-plane features, using learnable embeddings as queries for cross-plane generation. To effectively enhance feature fusion with the tri-plane 3D feature and human body prior, we propose a hybrid prior fusion strategy combining spatial and prior-enhanced queries, leveraging the benefits of spatial localization and human body prior knowledge. Comprehensive experiments on CAPE and THuman2.0 datasets illustrate that our method outperforms state-of-the-art approaches in both geometry and texture reconstruction, exhibiting high robustness to challenging poses and loose clothing, and producing higher-resolution textures.  Codes are available at https://github.com/River-Zhang/GTA.

----

## [0] TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models

**Authors**: *Pum Jun Kim, Yoojin Jang, Jisu Kim, Jaejun Yoo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html)

**Abstract**:

We propose a robust and reliable evaluation metric for generative models called Topological Precision and Recall (TopP&R, pronounced “topper”), which systematically estimates supports by retaining only topologically and statistically significant features with a certain level of confidence. Existing metrics, such as Inception Score (IS), Frechet Inception Distance (FID), and various Precision and Recall (P&R) variants, rely heavily on support estimates derived from sample features. However, the reliability of these estimates has been overlooked, even though the quality of the evaluation hinges entirely on their accuracy. In this paper, we demonstrate that current methods not only fail to accurately assess sample quality when support estimation is unreliable, but also yield inconsistent results. In contrast, TopP&R reliably evaluates the sample quality and ensures statistical consistency in its results. Our theoretical and experimental findings reveal that TopP&R provides a robust evaluation, accurately capturing the true trend of change in samples, even in the presence of outliers and non-independent and identically distributed (Non-IID) perturbations where other methods result in inaccurate support estimations. To our knowledge, TopP&R is the first evaluation metric specifically focused on the robust estimation of supports, offering statistical consistency under noise conditions.

----

## [0] A Unified Detection Framework for Inference-Stage Backdoor Defenses

**Authors**: *Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, Jie Ding*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html)

**Abstract**:

Backdoor attacks involve inserting poisoned samples during training, resulting in a model containing a hidden backdoor that can trigger specific behaviors without impacting performance on normal samples. These attacks are challenging to detect, as the backdoored model appears normal until activated by the backdoor trigger, rendering them particularly stealthy. In this study, we devise a unified inference-stage detection framework to defend against backdoor attacks. We first rigorously formulate the inference-stage backdoor detection problem, encompassing various existing methods, and discuss several challenges and limitations. We then propose a framework with provable guarantees on the false positive rate or the probability of misclassifying a clean sample.  Further, we derive the most powerful detection rule to maximize the detection power, namely the rate of accurately identifying a backdoor sample, given a false positive rate under classical learning scenarios. Based on the theoretically optimal detection rule, we suggest a practical and effective approach for real-world applications based on the latent representations of backdoored deep nets. We extensively evaluate our method on 14 different backdoor attacks using Computer Vision (CV) and Natural Language Processing (NLP) benchmark datasets. The experimental findings align with our theoretical results. We significantly surpass the state-of-the-art methods, e.g., up to 300\% improvement on the detection power as evaluated by AUCROC, over the state-of-the-art defense against advanced adaptive backdoor attacks.

----

## [0] Non-Stationary Bandits with Auto-Regressive Temporal Dependency

**Authors**: *Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html)

**Abstract**:

Traditional multi-armed bandit (MAB) frameworks, predominantly examined under stochastic or adversarial settings, often overlook the temporal dynamics inherent in many real-world applications such as recommendation systems and online advertising. This paper introduces a novel non-stationary MAB framework that captures the temporal structure of these real-world dynamics through an auto-regressive (AR) reward structure. We propose an algorithm that integrates two key mechanisms: (i) an alternation mechanism adept at leveraging temporal dependencies to dynamically balance exploration and exploitation, and (ii) a restarting mechanism designed to discard out-of-date information. Our algorithm achieves a regret upper bound that nearly matches the lower bound, with regret measured against a robust dynamic benchmark. Finally, via a real-world case study on tourism demand prediction, we demonstrate both the efficacy of our algorithm and the broader applicability of our techniques to more complex, rapidly evolving time series.

----

## [0] Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces

**Authors**: *Martin Ryner, Jan Kronqvist, Johan Karlsson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/188409d2ad91db4fb13644d024d99074-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/188409d2ad91db4fb13644d024d99074-Abstract-Conference.html)

**Abstract**:

This paper presents a framework for computing the Gromov-Wasserstein problem between two sets of points in low dimensional spaces, where the discrepancy is the squared Euclidean norm.The Gromov-Wasserstein problem is a generalization of the optimal transport problem that finds the assignment between two sets preserving pairwise distances as much as possible. This can be used to quantify the similarity between two formations or shapes, a common problem in AI and machine learning.The problem can be formulated as a Quadratic Assignment Problem (QAP), which is in general computationally intractable even for small problems. Our framework addresses this challenge by reformulating the QAP as an optimization problem with a low-dimensional domain, leveraging the fact that the problem can be expressed as a concave quadratic optimization problem with low rank. The method scales well with the number of points, and it can be used to find the global solution for large-scale problems with thousands of points.We compare the computational complexity of our approach with state-of-the-art methods on synthetic problems and apply it to a near-symmetrical problem which is of particular interest in computational biology.

----

## [0] Combinatorial Optimization with Policy Adaptation using Latent Space Search

**Authors**: *Félix Chalumeau, Shikha Surana, Clément Bonnet, Nathan Grinsztajn, Arnu Pretorius, Alexandre Laterre, Tom Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html)

**Abstract**:

Combinatorial Optimization underpins many real-world applications and yet, designing performant algorithms to solve these complex, typically NP-hard, problems remains a significant research challenge. Reinforcement Learning (RL) provides a versatile framework for designing heuristics across a broad spectrum of problem domains. However, despite notable progress, RL has not yet supplanted industrial solvers as the go-to solution. Current approaches emphasize pre-training heuristics that construct solutions, but often rely on search procedures with limited variance, such as stochastically sampling numerous solutions from a single policy, or employing computationally expensive fine-tuning of the policy on individual problem instances. Building on the intuition that performant search at inference time should be anticipated during pre-training, we propose COMPASS, a novel RL approach that parameterizes a distribution of diverse and specialized policies conditioned on a continuous latent space. We evaluate COMPASS across three canonical problems - Travelling Salesman, Capacitated Vehicle Routing, and Job-Shop Scheduling - and demonstrate that our search strategy (i) outperforms state-of-the-art approaches in 9 out of 11 standard benchmarking tasks and (ii) generalizes better, surpassing all other approaches on a set of 18 procedurally transformed instance distributions.

----

## [0] SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking

**Authors**: *Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, Lester Mackey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Subseasonal forecasting of the weather two to six weeks in advance is critical for resource allocation and climate adaptation but poses many challenges for the forecasting community. At this forecast horizon, physics-based dynamical models have limited skill, and the targets for prediction depend in a complex manner on both local weather variables and global climate variables. Recently, machine learning methods have shown promise in advancing the state of the art but  only at the cost of complex data curation, integrating expert knowledge with aggregation across multiple relevant data sources, file formats, and temporal and spatial resolutions.To streamline this process and accelerate future development, we introduce SubseasonalClimateUSA, a curated dataset for training and benchmarking subseasonal forecasting models in the United States. We use this dataset to benchmark a diverse suite of models, including operational dynamical models, classical meteorological baselines, and ten state-of-the-art machine learning and deep learning-based methods from the literature. Overall, our benchmarks suggest simple and effective ways to extend the accuracy of current operational models. SubseasonalClimateUSA is regularly updated and accessible via the https://github.com/microsoft/subseasonal_data/ Python package.

----

## [0] RenderMe-360: A Large Digital Asset Library and Benchmarks Towards High-fidelity Head Avatars

**Authors**: *Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, Ziwei Liu, Chen Change Loy, Chen Qian, Wayne Wu, Dahua Lin, Kwan-Yee Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1909ac72220bf5016b6c93f08b66cf36-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1909ac72220bf5016b6c93f08b66cf36-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthesizing high-fidelity head avatars is a central problem for computer vision and graphics. While head avatar synthesis algorithms have advanced rapidly, the best ones still face great obstacles in real-world scenarios. One of the vital causes is the inadequate datasets  -- 1) current public datasets can only support researchers to explore high-fidelity head avatars in one or two task directions; 2) these datasets usually contain digital head assets with limited data volume, and narrow distribution over different attributes, such as expressions, ages, and accessories. In this paper, we present RenderMe-360, a comprehensive 4D human head dataset to drive advance in head avatar algorithms across different scenarios. It contains massive data assets, with 243+ million complete head frames and over 800k video sequences from 500 different identities captured by multi-view cameras at 30 FPS. It is a large-scale digital library for head avatars with three key attributes: 1) High Fidelity: all subjects are captured in 360 degrees via 60 synchronized, high-resolution 2K cameras. 2) High Diversity: The collected subjects vary from different ages, eras, ethnicities, and cultures, providing abundant materials with distinctive styles in appearance and geometry. Moreover, each subject is asked to perform various dynamic motions, such as expressions and head rotations, which further extend the richness of assets. 3) Rich Annotations: the dataset provides annotations with different granularities: cameras' parameters, background matting, scan, 2D/3D facial landmarks, FLAME fitting, and text description.    Based on the dataset, we build a comprehensive benchmark for head avatar research, with 16 state-of-the-art methods performed on five main tasks: novel view synthesis, novel expression synthesis, hair rendering, hair editing, and talking head generation. Our experiments uncover the strengths and flaws of state-of-the-art methods. RenderMe-360 opens the door for future exploration in modern head avatars. All of the data, code, and models will be publicly available at https://renderme-360.github.io/.

----

## [0] Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation

**Authors**: *Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, Zhen Li, Monica Cheng, Rahul Goutam, Haiyang Zhang, Karthik Subbian, Suhang Wang, Yizhou Sun, Jiliang Tang, Bing Yin, Xianfeng Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/193df57a2366d032fb18dcac0698d09a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/193df57a2366d032fb18dcac0698d09a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus,  accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences.To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish.Remarkably, the dataset can help us enhance personalization and understanding of user preferences, which can benefit various existing tasks as well as enable new tasks. To test the potential of the dataset, we introduce three tasks in this work:(1) next-product recommendation, (2) next-product recommendation with domain shifts, and (3) next-product title generation.With the above tasks, we benchmark a range of algorithms on our proposed dataset, drawing new insights for further research and practice. In addition, based on the proposed dataset and tasks, we hosted a competition in the KDD CUP 2023 https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge and have attracted thousands of users and submissions. The winning solutions and the associated workshop can be accessed at our website~https://kddcup23.github.io/.

----

## [0] Adversarial Resilience in Sequential Prediction via Abstention

**Authors**: *Surbhi Goel, Steve Hanneke, Shay Moran, Abhishek Shetty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1967f962c7c2083618236d80eeb9d1ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1967f962c7c2083618236d80eeb9d1ac-Abstract-Conference.html)

**Abstract**:

We study the problem of sequential prediction in the stochastic setting with an adversary that is allowed to inject clean-label adversarial (or out-of-distribution) examples. Algorithms designed to handle purely stochastic data tend to fail in the presence of such adversarial examples, often leading to erroneous predictions. This is undesirable in many high-stakes applications such as medical recommendations, where abstaining from predictions on adversarial examples is preferable to misclassification. On the other hand, assuming fully adversarial data leads to very pessimistic bounds that are often vacuous in practice.     To move away from these pessimistic guarantees, we propose a new model of sequential prediction that sits between the purely stochastic and fully adversarial settings by allowing the learner to abstain from making a prediction at no cost on adversarial examples, thereby asking the learner to make predictions with certainty. Assuming access to the marginal distribution on the non-adversarial examples, we design a learner whose error scales with the VC dimension (mirroring the stochastic setting) of the hypothesis class, as opposed to the Littlestone dimension which characterizes the fully adversarial setting. Furthermore, we design learners for VC dimension~1 classes and the class of axis-aligned rectangles, which work even in the absence of access to the marginal distribution. Our key technical contribution is a novel measure for quantifying uncertainty for learning VC classes, which may be of independent interest.

----

## [0] Simplicity Bias in 1-Hidden Layer Neural Networks

**Authors**: *Depen Morwani, Jatin Batra, Prateek Jain, Praneeth Netrapalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html)

**Abstract**:

Recent works have demonstrated that neural networks exhibit extreme *simplicity bias* (SB). That is,  they learn *only the simplest* features  to solve a task at hand, even in the presence of other, more robust but more complex features. Due to the lack of a general and rigorous definition of *features*, these works showcase SB on *semi-synthetic* datasets such as Color-MNIST , MNIST-CIFAR where defining features is relatively easier. In this work, we rigorously define as well as thoroughly establish SB for *one hidden layer* neural networks in the infinite width regime. More concretely, (i) we define SB as the network essentially being a function of a low dimensional projection of the inputs (ii) theoretically, we show that when the data is linearly separable, the network primarily depends on only the linearly separable ($1$-dimensional) subspace even in the presence of an arbitrarily large number of other, more complex features which could have led to a significantly more robust classifier,  (iii) empirically, we show that models trained on *real* datasets such as Imagenet and Waterbirds-Landbirds indeed depend on a low dimensional projection of the inputs, thereby demonstrating SB on these datasets, iv) finally, we present a natural ensemble approach that encourages diversity in  models by training successive models on features not used by earlier models, and demonstrate that it yields models that are significantly more robust to Gaussian noise.

----

## [0] AVOIDDS: Aircraft Vision-based Intruder Detection Dataset and Simulator

**Authors**: *Elysia Q. Smyers, Sydney M. Katz, Anthony Corso, Mykel J. Kochenderfer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19a260641ebaf68d412f427e591bb74a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/19a260641ebaf68d412f427e591bb74a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Designing robust machine learning systems remains an open problem, and there is a need for benchmark problems that cover both environmental changes and evaluation on a downstream task. In this work, we introduce AVOIDDS, a realistic object detection benchmark for the vision-based aircraft detect-and-avoid problem. We provide a labeled dataset consisting of 72,000 photorealistic images of intruder aircraft with various lighting conditions, weather conditions, relative geometries, and geographic locations.  We also provide an interface that evaluates trained models on slices of this dataset to identify changes in performance with respect to changing environmental conditions. Finally, we implement a fully-integrated, closed-loop simulator of the vision-based detect-and-avoid problem to evaluate trained models with respect to the downstream collision avoidance task. This benchmark will enable further research in the design of robust machine learning systems for use in safety-critical applications. The AVOIDDS dataset and code are publicly available at https://purl.stanford.edu/hj293cv5980 and https://github.com/sisl/VisionBasedAircraftDAA, respectively.

----

## [0] Temporally Disentangled Representation Learning under Unknown Nonstationarity

**Authors**: *Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html)

**Abstract**:

In unsupervised causal representation learning for sequential data with time-delayed latent causal influences, strong identifiability results for the disentanglement of causally-related latent variables have been established in stationary settings by leveraging temporal structure.However, in nonstationary setting, existing work only partially addressed the problem by either utilizing observed auxiliary variables (e.g., class labels and/or domain indexes) as side information or assuming simplified latent causal dynamics. Both constrain the method to a limited range of scenarios.In this study, we further explored the Markov Assumption under time-delayed causally related process in nonstationary setting and showed that under mild conditions, the independent latent components can be recovered from their nonlinear mixture up to a permutation and a component-wise transformation, without the observation of auxiliary variables. We then introduce NCTRL, a principled estimation framework, to reconstruct time-delayed latent causal variables and identify their relations from measured sequential data only.Empirical evaluations demonstrated the reliable identification of time-delayed latent causal influences, with our methodology substantially outperforming existing baselines that fail to exploit the nonstationarity adequately and then, consequently, cannot distinguish distribution shifts.

----

## [0] Accelerated Quasi-Newton Proximal Extragradient: Faster Rate for Smooth Convex Optimization

**Authors**: *Ruichen Jiang, Aryan Mokhtari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an accelerated quasi-Newton proximal extragradient method for solving unconstrained smooth convex optimization problems. With access only to the gradients of the objective, we prove that our method can achieve a convergence rate of $\mathcal{O}\bigl(\min\\{\frac{1}{k^2}, \frac{\sqrt{d\log k}}{k^{2.5}}\\}\bigr)$, where $d$ is the problem dimension and $k$ is the number of iterations. In particular, in the regime where $k = \mathcal{O}(d)$, our method matches the _optimal rate_ of $\mathcal{O}(\frac{1}{k^2})$ by Nesterov's accelerated gradient (NAG). Moreover, in the the regime where $k = \Omega(d \log d)$, it outperforms NAG and converges at a _faster rate_ of $\mathcal{O}\bigl(\frac{\sqrt{d\log k}}{k^{2.5}}\bigr)$. To the best of our knowledge, this result is the first to demonstrate a provable gain for a quasi-Newton-type method over NAG in the convex setting.  To achieve such results, we build our method on a recent variant of the Monteiro-Svaiter acceleration framework and adopt an online learning perspective to update the Hessian approximation matrices, in which we relate the convergence rate of our method to the dynamic regret of a specific online convex optimization problem in the space of matrices.

----

## [0] Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference

**Authors**: *Tao Lei, Junwen Bai, Siddhartha Brahma, Joshua Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Y. Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19d7204af519eae9993f7f72377a0ec0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19d7204af519eae9993f7f72377a0ec0-Abstract-Conference.html)

**Abstract**:

We propose Conditional Adapter (CoDA), a parameter-efficient transfer learning method that also improves inference efficiency. CoDA generalizes beyond standard adapter approaches to enable a new way of balancing speed and accuracy using conditional computation.Starting with an existing dense pretrained model, CoDA adds sparse activation together with a small number of new parameters and a light-weight training phase.Our experiments demonstrate that the CoDA approach provides an unexpectedly efficient way to transfer knowledge.Across a variety of language, vision, and speech tasks, CoDA achieves a 2x to 8x inference speed-up compared to the state-of-the-art Adapter approaches with moderate to no accuracy loss and the same parameter efficiency.

----

## [0] Time-Independent Information-Theoretic Generalization Bounds for SGLD

**Authors**: *Futoshi Futami, Masahiro Fujisawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19dbb86f771ddbf9986cf0c9b1c61c17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19dbb86f771ddbf9986cf0c9b1c61c17-Abstract-Conference.html)

**Abstract**:

We provide novel information-theoretic generalization bounds for stochastic gradient Langevin dynamics (SGLD) under the assumptions of smoothness and dissipativity, which are widely used in sampling and non-convex optimization studies.Our bounds are time-independent and decay to zero as the sample size increases, regardless of the number of iterations and whether the step size is fixed.Unlike previous studies, we derive the generalization error bounds by focusing on the time evolution of the Kullback--Leibler divergence, which is related to the stability of datasets and is the upper bound of the mutual information between output parameters and an input dataset.Additionally, we establish the first information-theoretic generalization bound when the training and test loss are the same by showing that a loss function of SGLD is sub-exponential.This bound is also time-independent and removes the problematic step size dependence in existing work, leading to an improved excess risk bound by combining our analysis with the existing non-convex optimization error bounds.

----

## [0] Topology-Aware Uncertainty for Image Segmentation

**Authors**: *Saumya Gupta, Yikai Zhang, Xiaoling Hu, Prateek Prasanna, Chao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html)

**Abstract**:

Segmentation of curvilinear structures such as vasculature and road networks is challenging due to relatively weak signals and complex geometry/topology. To facilitate and accelerate large scale annotation, one has to adopt semi-automatic approaches such as proofreading by experts. In this work, we focus on uncertainty estimation for such tasks, so that highly uncertain, and thus error-prone structures can be identified for human annotators to verify. Unlike most existing works, which provide pixel-wise uncertainty maps, we stipulate it is crucial to estimate uncertainty in the units of topological structures, e.g., small pieces of connections and branches. To achieve this, we leverage tools from topological data analysis, specifically discrete Morse theory (DMT), to first capture the structures, and then reason about their uncertainties. To model the uncertainty, we (1) propose a joint prediction model that estimates the uncertainty of a structure while taking the neighboring structures into consideration (inter-structural uncertainty); (2) propose a novel Probabilistic DMT to model the inherent uncertainty within each structure (intra-structural uncertainty) by sampling its representations via a perturb-and-walk scheme. On various 2D and 3D datasets, our method produces better structure-wise uncertainty maps compared to existing works. Code available at: https://github.com/Saumya-Gupta-26/struct-uncertainty

----

## [0] Multiplication-Free Transformer Training via Piecewise Affine Operations

**Authors**: *Atli Kosson, Martin Jaggi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19df21cd4931bd0caaa4d8480e9a59cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19df21cd4931bd0caaa4d8480e9a59cd-Abstract-Conference.html)

**Abstract**:

Multiplications are responsible for most of the computational cost involved in neural network training and inference. Recent research has thus looked for ways to reduce the cost associated with them. Inspired by Mogami 2020, we replace multiplication with a cheap piecewise affine approximation that is achieved by adding the bit representation of the floating point numbers together as integers. We show that transformers can be trained with the resulting modified matrix multiplications on both vision and language tasks with little to no performance impact, and without changes to the training hyperparameters. We further replace all non-linearities in the networks making them fully and jointly piecewise affine in both inputs and weights. Finally, we show that we can eliminate all multiplications in the entire training process, including operations in the forward pass, backward pass and optimizer update, demonstrating the first successful training of modern neural network architectures in a fully multiplication-free fashion.

----

## [0] A Unified Framework for Uniform Signal Recovery in Nonlinear Generative Compressed Sensing

**Authors**: *Junren Chen, Jonathan Scarlett, Michael Ng, Zhaoqiang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a04df6a405210aab4986994b873db9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a04df6a405210aab4986994b873db9b-Abstract-Conference.html)

**Abstract**:

In generative compressed sensing (GCS), we want to recover a signal $\mathbf{x^*}\in\mathbb{R}^n$ from $m$ measurements ($m\ll n$) using a generative prior $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$, where $G$ is typically an $L$-Lipschitz continuous generative model and $\mathbb{B}_2^k(r)$ represents the radius-$r$ $\ell_2$-ball in $\mathbb{R}^k$. Under nonlinear measurements, most prior results are non-uniform, i.e., they hold with high probability for a fixed $\mathbf{x^*}$ rather than for all $\mathbf{x^*}$ simultaneously. In this paper, we build a unified framework to derive uniform recovery guarantees for nonlinear GCS where the observation model is nonlinear and possibly discontinuous or unknown. Our framework accommodates GCS with 1-bit/uniformly quantized observations and single index model as canonical examples. Specifically, using a single   realization of the sensing ensemble and generalized Lasso,   all $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$ can be recovered up to an $\ell_2$-error at most $\epsilon$ using roughly $\tilde{O}({k}/{\epsilon^2})$ samples, with omitted logarithmic factors typically being dominated by $\log L$.  Notably, this almost coincides with existing non-uniform guarantees up to logarithmic factors, hence the uniformity costs very little.    As part of our technical contributions, we introduce Lipschitz approximation to handle discontinuous observation models.    We also develop a concentration inequality  that produces tighter bound for product process whose index sets have low metric entropy.  Experimental results are presented to corroborate our theory.

----

## [0] Tempo Adaptation in Non-stationary Reinforcement Learning

**Authors**: *Hyunin Lee, Yuhao Ding, Jongmin Lee, Ming Jin, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a0672689a693e0764f93f900488b3d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a0672689a693e0764f93f900488b3d9-Abstract-Conference.html)

**Abstract**:

We first raise and tackle a ``time synchronization'' issue between the agent and the environment in non-stationary reinforcement learning (RL), a crucial factor hindering its real-world applications. In reality, environmental changes occur over wall-clock time ($t$) rather than episode progress ($k$), where wall-clock time signifies the actual elapsed time within the fixed duration $t \in [0, T]$. In existing works, at episode $k$, the agent rolls a trajectory and trains a policy before transitioning to episode $k+1$. In the context of the time-desynchronized environment, however, the agent at time $t_{k}$ allocates $\Delta t$ for trajectory generation and training, subsequently moves to the next episode at $t_{k+1}=t_{k}+\Delta t$. Despite a fixed total number of episodes ($K$), the agent accumulates different trajectories influenced by the choice of interaction times ($t_1,t_2,...,t_K$), significantly impacting the suboptimality gap of the policy. We propose a Proactively Synchronizing Tempo ($\texttt{ProST}$) framework that computes a suboptimal sequence {$t_1,t_2,...,t_K$} (= { $t_{1:K}$}) by minimizing an upper bound on its performance measure, i.e., the dynamic regret. Our main contribution is that we show that a suboptimal {$t_{1:K}$} trades-off between the policy training time (agent tempo) and how fast the environment changes (environment tempo). Theoretically, this work develops a suboptimal {$t_{1:K}$} as a function of the degree of the environment's non-stationarity while also achieving a sublinear dynamic regret. Our experimental evaluation on various high-dimensional non-stationary environments shows that the $\texttt{ProST}$ framework achieves a higher online return at suboptimal {$t_{1:K}$} than the existing methods.

----

## [0] Unsupervised Semantic Correspondence Using Stable Diffusion

**Authors**: *Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models are now capable of generating images that are often indistinguishable from real images. To generate such images, these models must understand the semantics of the objects they are asked to generate. In this work we show that, without any training, one can leverage this semantic knowledge within diffusion models to find semantic correspondences â€“ locations in multiple images that have the same semantic meaning. Specifically, given an image, we optimize the prompt embeddings of these models for maximum attention on the regions of interest. These optimized embeddings capture semantic information about the location, which can then be transferred to another image. By doing so we obtain results on par with the strongly supervised state of the art on the PF-Willow dataset and significantly outperform (20.9% relative for the SPair-71k dataset) any existing weakly- or unsupervised method on PF-Willow, CUB-200 and SPair-71k datasets.

----

## [0] Efficient Subgame Refinement for Extensive-form Games

**Authors**: *Zhenxing Ge, Zheng Xu, Tianyu Ding, Wenbin Li, Yang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a2b4aba905a16733ff199888ac8eec4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a2b4aba905a16733ff199888ac8eec4-Abstract-Conference.html)

**Abstract**:

Subgame solving is an essential technique in addressing large imperfect information games, with various approaches developed to enhance the performance of refined strategies in the abstraction of the target subgame. However, directly applying existing subgame solving techniques may be difficult, due to the intricate nature and substantial size of many real-world games. To overcome this issue, recent subgame solving methods allow for subgame solving on limited knowledge order subgames, increasing their applicability in large games; yet this may still face obstacles due to extensive information set sizes. To address this challenge, we propose a generative subgame solving (GS2) framework, which utilizes a generation function to identify a subset of the earliest-reached nodes, reducing the size of the subgame. Our method is supported by a theoretical analysis and employs a diversity-based generation function to enhance safety. Experiments conducted on medium-sized games as well as the challenging large game of GuanDan demonstrate a significant improvement over the blueprint.

----

## [0] NeRF-IBVS: Visual Servo Based on NeRF for Visual Localization and Navigation

**Authors**: *Yuanze Wang, Yichao Yan, Dianxi Shi, Wenhan Zhu, Jianqiang Xia, Jeff Tan, Songchang Jin, Ke Gao, Xiaobo Li, Xiaokang Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a57081f257da7b440b8eda72a0b12d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a57081f257da7b440b8eda72a0b12d4-Abstract-Conference.html)

**Abstract**:

Visual localization is a fundamental task in computer vision and robotics. Training existing visual localization methods requires a large number of posed images to generalize to novel views, while state-of-the-art methods generally require dense ground truth 3D labels for supervision. However, acquiring a  large number of posed images and dense 3D labels in the real world is challenging and costly. In this paper, we present a novel visual localization method that achieves accurate localization while using only a few posed images compared to other localization methods. To achieve this, we first use a few posed images with coarse pseudo-3D labels provided by NeRF to train a coordinate regression network. Then a coarse pose is estimated from the regression network with PNP. Finally, we use the image-based visual servo (IBVS) with the scene prior provided by NeRF for pose optimization. Furthermore, our method can provide effective navigation prior, which enable navigation based on IBVS without using custom markers and depth sensor. Extensive experiments on 7-Scenes  and 12-Scenes datasets demonstrate that our method outperforms state-of-the-art methods under the same setting, with only 5\% to 25\% training data.  Furthermore, our framework can be naturally extended to the visual navigation task based on IBVS, and its effectiveness is verified in simulation experiments.

----

## [0] How Does Adaptive Optimization Impact Local Neural Network Geometry?

**Authors**: *Kaiqi Jiang, Dhruv Malik, Yuanzhi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a5e6d0441a8e1eda9a50717b0870f94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a5e6d0441a8e1eda9a50717b0870f94-Abstract-Conference.html)

**Abstract**:

Adaptive optimization methods are well known to achieve superior convergence relative to vanilla gradient methods. The traditional viewpoint in optimization, particularly in convex optimization, explains this improved performance by arguing that, unlike vanilla gradient schemes, adaptive algorithms mimic the behavior of a second-order method by adapting to the *global* geometry of the loss function. We argue that in the context of neural network optimization, this traditional viewpoint is insufficient. Instead, we advocate for a *local* trajectory analysis. For iterate trajectories produced by running a generic optimization algorithm OPT, we introduce $R^{\text{OPT}}\_{\text{med}}$, a statistic that is analogous to the condition number of the loss Hessian evaluated at the iterates. Through extensive experiments on language models where adaptive algorithms converge faster than vanilla gradient methods like SGD, we show that adaptive methods such as Adam bias the trajectories towards regions where $R^{\text{Adam}}_{\text{med}}$ is small, where one might expect faster optimization. By contrast, SGD (with momentum) biases the trajectories towards regions where $R^{\text{SGD}}\_{\text{med}}$ is comparatively large. We complement these empirical observations with a theoretical result that provably demonstrates this phenomenon in the simplified setting of a two-layer linear network. We view our findings as evidence for the need of a new explanation of the success of adaptive methods, one that is different than the conventional wisdom.

----

## [0] Are Diffusion Models Vision-And-Language Reasoners?

**Authors**: *Benno Krojer, Elinor Poole-Dayan, Vikram Voleti, Chris Pal, Siva Reddy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a675d804f50509b8e21d0d3ca709d03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a675d804f50509b8e21d0d3ca709d03-Abstract-Conference.html)

**Abstract**:

Text-conditioned image generation models have recently shown immense qualitative success using denoising diffusion processes. However, unlike discriminative vision-and-language models, it is a non-trivial task to subject these diffusion-based generative models to automatic fine-grained quantitative evaluation of high-level phenomena such as compositionality.Towards this goal, we perform two innovations. First, we transform diffusion-based models (in our case, Stable Diffusion) for any image-text matching (ITM) task using a novel method called DiffusionITM.Second, we introduce the Generative-Discriminative Evaluation Benchmark (GDBench) benchmark with 7 complex vision-and-language tasks, bias evaluation and detailed analysis.We find that Stable Diffusion + DiffusionITM is competitive on many tasks and outperforms CLIP on compositional tasks like like CLEVR and Winoground.We further boost its compositional performance with a transfer setup by fine-tuning on MS-COCO while retaining generative capabilities. We also measure the stereotypical bias in diffusion models, and find that Stable Diffusion 2.1 is, for the most part, less biased than Stable Diffusion 1.5.Overall, our results point in an exciting direction bringing discriminative and generative model evaluation closer. We will release code and benchmark setup soon.

----

## [0] ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

**Authors**: *Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a87980b9853e84dfb295855b425c262-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a87980b9853e84dfb295855b425c262-Abstract-Conference.html)

**Abstract**:

Score distillation sampling (SDS) has shown great promise in text-to-3D generation by distilling pretrained large-scale text-to-image diffusion models, but suffers from over-saturation, over-smoothing, and low-diversity problems. In this work, we propose to model the 3D parameter as a random variable instead of a constant as in SDS and present *variational score distillation* (VSD), a principled particle-based variational framework to explain and address the aforementioned issues in text-to-3D generation. We show that SDS is a special case of VSD and leads to poor samples with both small and large CFG weights. In comparison, VSD works well with various CFG weights as ancestral sampling from diffusion models and simultaneously improves the diversity and sample quality with a common CFG weight (i.e., 7.5). We further present various improvements in the design space for text-to-3D such as distillation time schedule and density initialization, which are orthogonal to the distillation algorithm yet not well explored. Our overall approach, dubbed *ProlificDreamer*, can generate high rendering resolution (i.e., 512$\times$512) and high-fidelity NeRF with rich structure and complex effects (e.g., smoke and drops). Further, initialized from NeRF, meshes fine-tuned by VSD are meticulously detailed and photo-realistic.

----

## [0] SAMoSSA: Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise

**Authors**: *Abdullah Alomar, Munther A. Dahleh, Sean Mann, Devavrat Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a8d295871250443f9747d239925b89d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a8d295871250443f9747d239925b89d-Abstract-Conference.html)

**Abstract**:

The well-established practice of time series analysis involves estimating deterministic, non-stationary trend and seasonality components followed by learning the residual stochastic, stationary components. Recently, it has been shown that one can learn the deterministic non-stationary components accurately using  multivariate Singular Spectrum Analysis (mSSA) in the absence of a correlated stationary component; meanwhile, in the absence of deterministic non-stationary components, the Autoregressive (AR) stationary component can also be learnt readily, e.g. via Ordinary Least Squares (OLS). However, a theoretical underpinning of multi-stage learning algorithms involving both deterministic and stationary components has been absent in the literature despite its pervasiveness. We resolve this open question by establishing desirable theoretical guarantees for a natural two-stage algorithm, where mSSA is first applied to estimate the non-stationary components despite the presence of a correlated stationary AR component, which is subsequently learned from the residual time series. We provide a finite-sample forecasting consistency bound for the proposed algorithm, SAMoSSA, which is data-driven and thus requires minimal parameter tuning. To establish theoretical guarantees, we overcome three hurdles: (i) we characterize the spectra of Page matrices of stable AR processes, thus extending the analysis of mSSA; (ii) we extend the analysis of AR process identification in the presence of arbitrary bounded perturbations; (iii) we characterize the out-of-sample or forecasting error, as opposed to solely considering model identification. Through representative empirical studies, we validate the superior performance of SAMoSSA compared to existing baselines. Notably, SAMoSSA's ability to account for AR noise structure yields improvements ranging from 5% to 37% across various benchmark datasets.

----

## [0] Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection

**Authors**: *Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, Ruimin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1abc87c67cc400a67b869358e627fe37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1abc87c67cc400a67b869358e627fe37-Abstract-Conference.html)

**Abstract**:

Unsupervised image Anomaly Detection (UAD) aims to learn robust and discriminative representations of normal samples. While separate solutions per class endow expensive computation and limited generalizability, this paper focuses on building a unified framework for multiple classes. Under such a challenging setting, popular reconstruction-based networks with continuous latent representation assumption always suffer from the "identical shortcut" issue, where both normal and abnormal samples can be well recovered and difficult to distinguish. To address this pivotal issue, we propose a hierarchical vector quantized prototype-oriented Transformer under a probabilistic framework. First, instead of learning the continuous representations, we preserve the typical normal patterns as discrete iconic prototypes, and confirm the importance of Vector Quantization in preventing the model from falling into the shortcut. The vector quantized iconic prototypes are integrated into the Transformer for reconstruction, such that the abnormal data point is flipped to a normal data point. Second, we investigate an exquisite hierarchical framework to relieve the codebook collapse issue and replenish frail normal patterns.  Third, a prototype-oriented optimal transport method is proposed to better regulate the prototypes and hierarchically evaluate the abnormal score. By evaluating on MVTec-AD and VisA datasets, our model surpasses the state-of-the-art alternatives and possesses good interpretability. The code is available at https://github.com/RuiyingLu/HVQ-Trans.

----

## [0] MCUFormer: Deploying Vision Tranformers on Microcontrollers with Limited Memory

**Authors**: *Yinan Liang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ae4999aefb509d75d8608e07280922c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ae4999aefb509d75d8608e07280922c-Abstract-Conference.html)

**Abstract**:

Due to the high price and heavy energy consumption of GPUs, deploying deep models on IoT devices such as microcontrollers makes significant contributions for ecological AI. Conventional methods successfully enable convolutional neural network inference of high resolution images on microcontrollers, while the framework for vision transformers that achieve the state-of-the-art performance in many vision applications still remains unexplored. In this paper, we propose a hardware-algorithm co-optimizations method called MCUFormer to deploy vision transformers on microcontrollers with extremely limited memory, where we jointly design transformer architecture and construct the inference operator library to fit the memory resource constraint. More specifically, we generalize the one-shot network architecture search (NAS) to discover the optimal architecture with highest task performance given the memory budget from the microcontrollers, where we enlarge the existing search space of vision transformers by considering the low-rank decomposition dimensions and patch resolution for memory reduction. For the construction of the inference operator library of vision transformers, we schedule the memory buffer during inference through operator integration, patch embedding decomposition, and token overwriting, allowing the memory buffer to be fully utilized to adapt to the forward pass of the vision transformer. Experimental results demonstrate that our MCUFormer achieves 73.62\% top-1 accuracy on ImageNet for image classification with 320KB memory on STM32F746 microcontroller. Code is available at https://github.com/liangyn22/MCUFormer.

----

## [0] Towards Accelerated Model Training via Bayesian Data Selection

**Authors**: *Zhijie Deng, Peng Cui, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1af3e0bf5905e33789979f666c31192d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1af3e0bf5905e33789979f666c31192d-Abstract-Conference.html)

**Abstract**:

Mislabeled, duplicated, or biased data in real-world scenarios can lead to prolonged training and even hinder model convergence. Traditional solutions prioritizing easy or hard samples lack the flexibility to handle such a variety simultaneously. Recent work has proposed a more reasonable data selection principle by examining the data's impact on the model's generalization loss. However, its practical adoption relies on less principled approximations and additional holdout data. This work solves these problems by leveraging a lightweight Bayesian treatment and incorporating off-the-shelf zero-shot predictors built on large-scale pre-trained models. The resulting algorithm is efficient and easy to implement. We perform extensive empirical studies on challenging benchmarks with considerable data noise and imbalance in the online batch selection scenario, and observe superior training efficiency over competitive baselines. Notably, on the challenging WebVision benchmark, our method can achieve similar predictive performance with significantly fewer training iterations than leading data selection methods.

----

## [0] CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels

**Authors**: *Wanxing Chang, Ye Shi, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b0da24d136f46bfaee78e8da907127e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b0da24d136f46bfaee78e8da907127e-Abstract-Conference.html)

**Abstract**:

Learning with noisy labels (LNL) poses a significant challenge in training a well-generalized model while avoiding overfitting to corrupted labels.Recent advances have achieved impressive performance by identifying clean labels and correcting corrupted labels for training.However, the current approaches rely heavily on the modelâ€™s predictions and evaluate each sample independently without considering either the global or local structure of the sample distribution.These limitations typically result in a suboptimal solution for the identification and correction processes, which eventually leads to models overfitting to incorrect labels.In this paper, we propose a novel optimal transport (OT) formulation, called Curriculum and Structure-aware Optimal Transport (CSOT). CSOT concurrently considers the inter- and intra-distribution structure of the samples to construct a robust denoising and relabeling allocator.During the training process, the allocator incrementally assigns reliable labels to a fraction of the samples with the highest confidence. These labels have both global discriminability and local coherence.Notably, CSOT is a new OT formulation with a nonconvex objective function and curriculum constraints, so it is not directly compatible with classical OT solvers. Here, we develop a lightspeed computational method that involves a scaling iteration within a generalized conditional gradient framework to solve CSOT efficiently.Extensive experiments demonstrate the superiority of our method over the current state-of-the-arts in LNL.

----

## [0] In-Context Learning Unlocked for Diffusion Models

**Authors**: *Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang (Atlas) Wang, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html)

**Abstract**:

We present Prompt Diffusion, a framework for enabling in-context learning in diffusion-based generative models. Given a pair of task-specific example images, such as depth from/to image and scribble from/to image, and a text guidance, our model automatically understands the underlying task and performs the same task on a new query image following the text guidance. To achieve this, we propose a vision-language prompt that can model a wide range of vision-language tasks and a diffusion model that takes it as input. The diffusion model is trained jointly on six different tasks using these prompts. The resulting Prompt Diffusion model becomes the first diffusion-based vision-language foundation model capable of in-context learning. It demonstrates high-quality in-context generation for the trained tasks and effectively generalizes to new, unseen vision tasks using their respective prompts. Our model also shows compelling text-guided image editing results. Our framework aims to facilitate research into in-context learning for computer vision. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Prompt-Diffusion.

----

## [0] Object-Centric Slot Diffusion

**Authors**: *Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html)

**Abstract**:

The recent success of transformer-based image generative models in object-centric learning highlights the importance of powerful image generators for handling complex scenes. However, despite the high expressiveness of diffusion models in image generation, their integration into object-centric learning remains largely unexplored in this domain. In this paper, we explore the feasibility and potential of integrating diffusion models into object-centric learning and investigate the pros and cons of this approach. We introduce Latent Slot Diffusion (LSD), a novel model that serves dual purposes: it is the first object-centric learning model to replace conventional slot decoders with a latent diffusion model conditioned on object slots, and it is also the first unsupervised compositional conditional diffusion model that operates without the need for supervised annotations like text. Through experiments on various object-centric tasks, including the first application of the FFHQ dataset in this field, we demonstrate that LSD significantly outperforms state-of-the-art transformer-based decoders, particularly in more complex scenes, and exhibits superior unsupervised compositional generation quality. In addition, we conduct a preliminary investigation into the integration of pre-trained diffusion models in LSD and demonstrate its effectiveness in real-world image segmentation and generation. Project page is available at https://latentslotdiffusion.github.io

----

## [0] NAS-X: Neural Adaptive Smoothing via Twisting

**Authors**: *Dieterich Lawson, Michael Li, Scott W. Linderman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3d005a2cb0e71e698e0b13ac657473-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3d005a2cb0e71e698e0b13ac657473-Abstract-Conference.html)

**Abstract**:

Sequential latent variable models (SLVMs) are essential tools in statistics and machine learning, with applications ranging from healthcare to neuroscience. As their flexibility increases, analytic inference and model learning can become challenging, necessitating approximate methods. Here we introduce neural adaptive smoothing via twisting (NAS-X), a method that extends reweighted wake-sleep (RWS) to the sequential setting by using smoothing sequential Monte Carlo (SMC) to estimate intractable posterior expectations. Combining RWS and smoothing SMC allows NAS-X to provide low-bias and low-variance gradient estimates, and fit both discrete and continuous latent variable models. We illustrate the theoretical advantages of NAS-X over previous methods and explore these advantages empirically in a variety of tasks, including a challenging application to mechanistic models of neuronal dynamics. These experiments show that NAS-X substantially outperforms previous VI- and RWS-based methods in inference and model learning, achieving lower parameter error and tighter likelihood bounds.

----

## [0] Reflexion: language agents with verbal reinforcement learning

**Authors**: *Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose \emph{Reflexion}, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91\% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance. We release all code, demos, and datasets at \url{https://github.com/noahshinn024/reflexion}.

----

## [0] Demographic Parity Constrained Minimax Optimal Regression under Linear Model

**Authors**: *Kazuto Fukuchi, Jun Sakuma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b4acad19cc425a7352a71d4e4468393-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b4acad19cc425a7352a71d4e4468393-Abstract-Conference.html)

**Abstract**:

We explore the minimax optimal error associated with a demographic parity-constrained regression problem within the context of a linear model. Our proposed model encompasses a broader range of discriminatory bias sources compared to the model presented by Chzhen and Schreuder. Our analysis reveals that the minimax optimal error for the demographic parity-constrained regression problem under our model is characterized by $\Theta(\frac{dM}{n})$, where $n$ denotes the sample size, $d$ represents the dimensionality, and $M$ signifies the number of demographic groups arising from sensitive attributes. Moreover, we demonstrate that the minimax error increases in conjunction with a larger bias present in the model.

----

## [0] GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

**Authors**: *Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html)

**Abstract**:

Worldwide Geo-localization aims to pinpoint the precise location of images taken anywhere on Earth. This task has considerable challenges due to the immense variation in geographic landscapes. The image-to-image retrieval-based approaches fail to solve this problem on a global scale as it is not feasible to construct a large gallery of images covering the entire world. Instead, existing approaches divide the globe into discrete geographic cells, transforming the problem into a classification task. However, their performance is limited by the predefined classes and often results in inaccurate localizations when an image's location significantly deviates from its class center. To overcome these limitations, we propose GeoCLIP, a novel CLIP-inspired Image-to-GPS retrieval approach that enforces alignment between the image and its corresponding GPS locations. GeoCLIP's location encoder models the Earth as a continuous function by employing positional encoding through random Fourier features and constructing a hierarchical representation that captures information at varying resolutions to yield a semantically rich high-dimensional feature suitable to use even beyond geo-localization. To the best of our knowledge, this is the first work employing GPS encoding for geo-localization. We demonstrate the efficacy of our method via extensive experiments and ablations on benchmark datasets. We achieve competitive performance with just 20% of training data, highlighting its effectiveness even in limited-data settings. Furthermore, we qualitatively demonstrate geo-localization using a text query by leveraging the CLIP backbone of our image encoder. The project webpage is available at: https://vicentevivan.github.io/GeoCLIP

----

## [0] RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks

**Authors**: *Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b80fe066fdbceb3a2960117bac33917-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b80fe066fdbceb3a2960117bac33917-Abstract-Conference.html)

**Abstract**:

Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel defense including detection and aggregation, named RECESS, to serve as a “vaccine” for FL against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Further, RECESS adopts a newly proposed trust scoring based mechanism to robustly aggregate gradients. Rather than previous methods of scoring in each iteration, RECESS takes into account the correlation of clients’ performance over multiple iterations to estimate the trust score, bringing in a significant increase in detection fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings including white/black-box, cross-silo/device FL, etc. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.

----

## [0] Minimum norm interpolation by perceptra: Explicit regularization and implicit bias

**Authors**: *Jiyoung Park, Ian Pelakh, Stephan Wojtowytsch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b8612e11c75456c90963fd408d75c4d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b8612e11c75456c90963fd408d75c4d-Abstract-Conference.html)

**Abstract**:

We investigate how shallow ReLU networks interpolate between known regions. Our analysis shows that empirical risk minimizers converge to a minimum norm interpolant as the number of data points and parameters tends to infinity when a weight decay regularizer is penalized with a coefficient which vanishes at a precise rate as the network width and the number of data points grow. With and without explicit regularization, we numerically study the implicit bias of common optimization algorithms towards known minimum norm interpolants.

----

## [0] Spectral Co-Distillation for Personalized Federated Learning

**Authors**: *Zihan Chen, Howard H. Yang, Tony Q. S. Quek, Kai Fong Ernest Chong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b86cf4b15cd83b6520d851eb7298228-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b86cf4b15cd83b6520d851eb7298228-Abstract-Conference.html)

**Abstract**:

Personalized federated learning (PFL) has been widely investigated to address the challenge of data heterogeneity, especially when a single generic model is inadequate in satisfying the diverse performance requirements of local clients simultaneously. Existing PFL methods are inherently based on the idea that the relations between the generic global and personalized local models are captured by the similarity of model weights. Such a similarity is primarily based on either partitioning the model architecture into generic versus personalized components or modeling client relationships via model weights. To better capture similar (yet distinct) generic versus personalized model representations, we propose $\textit{spectral distillation}$, a novel distillation method based on model spectrum information. Building upon spectral distillation, we also introduce a co-distillation framework that establishes a two-way bridge between generic and personalized model training. Moreover, to utilize the local idle time in conventional PFL, we propose a wait-free local training protocol. Through extensive experiments on multiple datasets over diverse heterogeneous data settings, we demonstrate the outperformance and efficacy of our proposed spectral co-distillation method, as well as our wait-free training protocol.

----

## [0] DVSOD: RGB-D Video Salient Object Detection

**Authors**: *Jingjing Li, Wei Ji, Size Wang, Wenbo Li, Li Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b88e65f737256d437e56764d39ba06d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b88e65f737256d437e56764d39ba06d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Salient object detection (SOD) aims to identify standout elements in a scene, with recent advancements primarily focused on integrating depth data (RGB-D) or temporal data from videos to enhance SOD in complex scenes. However, the unison of two types of crucial information remains largely underexplored due to data constraints. To bridge this gap, we in this work introduce the DViSal dataset, fueling further research in the emerging field of RGB-D video salient object detection (DVSOD). Our dataset features 237 diverse RGB-D videos alongside comprehensive annotations, including object and instance-level markings, as well as bounding boxes and scribbles. These resources enable a broad scope for potential research directions. We also conduct benchmarking experiments using various SOD models, affirming the efficacy of multimodal video input for salient object detection. Lastly, we highlight some intriguing findings and promising future research avenues. To foster growth in this field, our dataset and benchmark results are publicly accessible at: https://dvsod.github.io/.

----

## [0] Gradient Informed Proximal Policy Optimization

**Authors**: *Sanghyun Son, Laura Yu Zheng, Ryan Sullivan, Yi-Ling Qiao, Ming C. Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1bd8cfc0e4c53869b7f1d0ed4b1e78e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1bd8cfc0e4c53869b7f1d0ed4b1e78e1-Abstract-Conference.html)

**Abstract**:

We introduce a novel policy learning method that integrates analytical gradients from differentiable environments with the Proximal Policy Optimization (PPO) algorithm. To incorporate analytical gradients into the PPO framework, we introduce the concept of an α-policy that stands as a locally superior policy. By adaptively modifying the α value, we can effectively manage the influence of analytical policy gradients during learning. To this end, we suggest metrics for assessing the variance and bias of analytical gradients, reducing dependence on these gradients when high variance or bias is detected. Our proposed approach outperforms baseline algorithms in various scenarios, such as function optimization, physics simulations, and traffic control environments. Our code can be found online: https://github.com/SonSang/gippo.

----

## [0] SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model

**Authors**: *Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao, Liangpei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1be3843e534ee06d3a70c7f62b983b31-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1be3843e534ee06d3a70c7f62b983b31-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The success of the Segment Anything Model (SAM) demonstrates the significance of data-centric machine learning. However, due to the difficulties and high costs associated with annotating Remote Sensing (RS) images, a large amount of valuable RS data remains unlabeled, particularly at the pixel level. In this study, we leverage SAM and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS totally possesses 105,090 images and 1,668,241 instances, surpassing existing high-resolution RS segmentation datasets in size by several orders of magnitude. It provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects.  Moreover, preliminary experiments highlight the importance of conducting segmentation pre-training with SAMRS to address task discrepancies and alleviate the limitations posed by limited training data during fine-tuning. The code and dataset will be available at https://github.com/ViTAE-Transformer/SAMRS

----

## [0] Blockwise Parallel Transformers for Large Context Models

**Authors**: *Hao Liu, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1bfd87d2d92f0556819467dc08034f76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1bfd87d2d92f0556819467dc08034f76-Abstract-Conference.html)

**Abstract**:

Transformers have emerged as the cornerstone of state-of-the-art natural language processing models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands posed by the self-attention mechanism and the large feedforward network in Transformers limit their ability to handle long sequences, thereby creating challenges for tasks involving multiple long sequences or long-term dependencies. We present a distinct approach, Blockwise Parallel Transformer (BPT), that leverages blockwise computation of self-attention and feedforward network fusion to minimize memory costs. By processing longer input sequences while maintaining memory efficiency, BPT enables training sequences 32 times longer than vanilla Transformers and up to 4 times longer than previous memory-efficient methods. Extensive experiments on language modeling and reinforcement learning tasks demonstrate the effectiveness of BPT in reducing memory requirements and improving performance.

----

## [0] Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization

**Authors**: *Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html)

**Abstract**:

Neural combinatorial optimization (NCO) is a promising learning-based approach for solving challenging combinatorial optimization problems without specialized algorithm design by experts. However, most constructive NCO methods cannot solve problems with large-scale instance sizes, which significantly diminishes their usefulness for real-world applications. In this work, we propose a novel Light Encoder and Heavy Decoder (LEHD) model with a strong generalization ability to address this critical issue. The LEHD model can learn to dynamically capture the relationships between all available nodes of varying sizes, which is beneficial for model generalization to problems of various scales. Moreover, we develop a data-efficient training scheme and a flexible solution construction mechanism for the proposed LEHD model. By training on small-scale problem instances, the LEHD model can generate nearly optimal solutions for the Travelling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) with up to 1000 nodes, and also generalizes well to solve real-world TSPLib and CVRPLib problems. These results confirm our proposed LEHD model can significantly improve the state-of-the-art performance for constructive NCO.

----

## [0] Topological Obstructions and How to Avoid Them

**Authors**: *Babak Esmaeili, Robin Walters, Heiko Zimmermann, Jan-Willem van de Meent*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c12ccfc7720f6b680edea17300bfc2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c12ccfc7720f6b680edea17300bfc2b-Abstract-Conference.html)

**Abstract**:

Incorporating geometric inductive biases into models can aid interpretability and generalization, but encoding to a specific geometric structure can be challenging due to the imposed topological constraints. In this paper, we theoretically and empirically characterize obstructions to training encoders with geometric latent spaces. We show that local optima can arise due to singularities (e.g. self-intersection) or due to an incorrect degree or winding number. We then discuss how normalizing flows can potentially circumvent these obstructions by defining multimodal variational distributions. Inspired by this observation, we propose a new flow-based model that maps data points to multimodal distributions over geometric spaces and empirically evaluate our model on 2 domains. We observe improved stability during training and a higher chance of converging to a homeomorphic encoder.

----

## [0] The Double-Edged Sword of Implicit Bias: Generalization vs Robustness in ReLU Networks

**Authors**: *Spencer Frei, Gal Vardi, Peter L. Bartlett, Nati Srebro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c26c389d60ec419fd24b5fee5b35796-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c26c389d60ec419fd24b5fee5b35796-Abstract-Conference.html)

**Abstract**:

In this work, we study the implications of the implicit bias of gradient flow on generalization and adversarial robustness in ReLU networks.  We focus on a setting where the data consists of clusters and the correlations between cluster means are small, and show that in two-layer ReLU networks gradient flow is biased towards solutions that generalize well, but are vulnerable to adversarial examples. Our results hold even in cases where the network is highly overparameterized.  Despite the potential for harmful overfitting in such settings, we prove that the implicit bias of gradient flow prevents it.  However, the implicit bias also leads to non-robust solutions (susceptible to small adversarial $\ell_2$-perturbations), even though robust networks that fit the data exist.

----

## [0] PromptRestorer: A Prompting Image Restoration Method with Degradation Perception

**Authors**: *Cong Wang, Jinshan Pan, Wei Wang, Jiangxin Dong, Mengzhu Wang, Yakun Ju, Junyang Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html)

**Abstract**:

We show that raw degradation features can effectively guide deep restoration models, providing accurate degradation priors to facilitate better restoration. While networks that do not consider them for restoration forget gradually degradation during the learning process, model capacity is severely hindered. To address this, we propose a Prompting image Restorer, termed as PromptRestorer. Specifically, PromptRestorer contains two branches: a restoration branch and a prompting branch. The former is used to restore images, while the latter perceives degradation priors to prompt the restoration branch with reliable perceived content to guide the restoration process for better recovery. To better perceive the degradation which is extracted by a pre-trained model from given degradation observations, we propose a prompting degradation perception modulator, which adequately considers the characters of the self-attention mechanism and pixel-wise modulation, to better perceive the degradation priors from global and local perspectives. To control the propagation of the perceived content for the restoration branch, we propose gated degradation perception propagation, enabling the restoration branch to adaptively learn more useful features for better recovery. Extensive experimental results show that our PromptRestorer achieves state-of-the-art results on 4 image restoration tasks, including image deraining, deblurring, dehazing, and desnowing.

----

## [0] Beyond MLE: Convex Learning for Text Generation

**Authors**: *Chenze Shao, Zhengrui Ma, Min Zhang, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html)

**Abstract**:

Maximum likelihood estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution that best explain the observed data. In the context of text generation, MLE is often used to train generative language models, which can then be used to generate new text. However, we argue that MLE is not always necessary and optimal, especially for closed-ended text generation tasks like machine translation. In these tasks, the goal of model is to generate the most appropriate response, which does not necessarily require it to estimate the entire data distribution with MLE. To this end, we propose a novel class of training objectives based on convex functions, which enables text generation models to focus on highly probable outputs without having to estimate the entire data distribution. We investigate the theoretical properties of the optimal predicted distribution when applying convex functions to the loss, demonstrating that convex functions can sharpen the optimal distribution, thereby enabling the model to better capture outputs with high probabilities. Experiments on various text generation tasks and models show the effectiveness of our approach. It enables autoregressive models to bridge the gap between greedy and beam search, and facilitates the learning of non-autoregressive models with a maximum improvement of 9+ BLEU points. Moreover, our approach also exhibits significant impact on large language models (LLMs), substantially enhancing their generative capability on various tasks. Source code is available at \url{https://github.com/ictnlp/Convex-Learning}.

----

## [0] Bandit Task Assignment with Unknown Processing Time

**Authors**: *Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, Ken-ichi Kawarabayashi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c5ee7343f396954377c2c16dda33a96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c5ee7343f396954377c2c16dda33a96-Abstract-Conference.html)

**Abstract**:

This study considers a novel problem setting, referred to as \textit{bandit task assignment}, that incorporates the processing time of each task in the bandit setting. In this problem setting, a player sequentially chooses a set of tasks to start so that the set of processing tasks satisfies a given combinatorial constraint. The reward and processing time for each task follow unknown distributions, values of which are revealed only after the task has been completed. The problem generalizes the stochastic combinatorial semi-bandit problem and the budget-constrained bandit problem. For this problem setting, we propose an algorithm based on upper confidence bounds~(UCB) combined with a phased-update approach. The proposed algorithm admits a gap-dependent regret upper bound of $O(MN(1/\Delta){\log T})$ and a gap-free regret upper bound of $\tilde{O}( \sqrt{MNT} )$, where $N$ is the number of the tasks, $M$ is the maximum number of tasks run at the same time, $T$ is the time horizon, and $\Delta$ is the gap between expected per-round rewards of the optimal and best suboptimal sets of tasks. These regret bounds nearly match lower bounds.

----

## [0] Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text

**Authors**: *Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, Yejin Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input.This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., ``What do image A and image B have in common?''To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text.To date, however, large-scale data of this form have not been publicly available.We release Multimodal C4, an augmentation of the popular text-only C4 corpus with images interleaved.We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives.Multimodal C4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88\%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80\%). After filtering NSFW images, ads, etc., the resulting corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.

----

## [0] Towards Self-Interpretable Graph-Level Anomaly Detection

**Authors**: *Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c6f06863df46de009a7a41b41c95cad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c6f06863df46de009a7a41b41c95cad-Abstract-Conference.html)

**Abstract**:

Graph-level anomaly detection (GLAD) aims to identify graphs that exhibit notable dissimilarity compared to the majority in a collection. However, current works primarily focus on evaluating graph-level abnormality while failing to provide meaningful explanations for the predictions, which largely limits their reliability and application scope. In this paper, we investigate a new challenging problem, explainable GLAD, where the learning objective is to predict the abnormality of each graph sample with corresponding explanations, i.e., the vital subgraph that leads to the predictions. To address this challenging problem, we propose a Self-Interpretable Graph aNomaly dETection model (SIGNET for short) that detects anomalous graphs as well as generates informative explanations simultaneously. Specifically, we first introduce the multi-view subgraph information bottleneck (MSIB) framework, serving as the design basis of our self-interpretable GLAD approach. This way SIGNET is able to not only measure the abnormality of each graph based on cross-view mutual information but also provide informative graph rationales by extracting bottleneck subgraphs from the input graph and its dual hypergraph in a self-supervised way. Extensive experiments on 16 datasets demonstrate the anomaly detection capability and self-interpretability of SIGNET.

----

## [0] AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neuron Activity

**Authors**: *Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, Eli Shlizerman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c70ba3591d0694a535089e1c25888d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c70ba3591d0694a535089e1c25888d7-Abstract-Conference.html)

**Abstract**:

Latent Variable Models (LVMs) propose to model the dynamics of neural populations by capturing low-dimensional structures that represent features involved in neural activity. Recent LVMs are based on deep learning methodology where a deep neural network is trained to reconstruct the same neural activity given as input and as a result to build the latent representation. Without taking past or future activity into account such a task is non-causal. In contrast, the task of forecasting neural activity based on given input extends the reconstruction task. LVMs that are trained on such a task could potentially capture temporal causality constraints within its latent representation. Forecasting has received less attention than reconstruction due to recording challenges such as limited neural measurements and trials. In this work, we address modeling neural population dynamics via the forecasting task and improve forecasting performance by including a prior, which consists of pairwise neural unit interaction as a multivariate dynamic system. Our proposed model---Additive, Multiplicative, and Adaptive Graph Neural Network (AMAG)---leverages additive and multiplicative message-passing operations analogous to the interactions in neuronal systems and adaptively learns the interaction among neural units to forecast their future activity. We demonstrate the advantage of AMAG compared to non-GNN based methods on synthetic data and multiple modalities of neural recordings (field potentials from penetrating electrodes or surface-level micro-electrocorticography) from four rhesus macaques. Our results show the ability of AMAG to recover ground truth spatial interactions and yield estimation for future dynamics of the neural population.

----

## [0] PackQViT: Faster Sub-8-bit Vision Transformers via Full and Packed Quantization on the Mobile

**Authors**: *Peiyan Dong, Lei Lu, Chao Wu, Cheng Lyu, Geng Yuan, Hao Tang, Yanzhi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html)

**Abstract**:

While Vision Transformers (ViTs) have undoubtedly made impressive strides in computer vision (CV), their intricate network structures necessitate substantial computation and memory resources. A decision-making process for CV tasks typically entails performing computations with low latency, which is a tricky problem for ViT models.Model quantization is a widely-used technique to optimize the hardware efficiency of deep neural networks.Full quantization under Sub-8-bit precision, in particular, is a promising solution to reduce inference latency significantly. Unfortunately, current commodity hardware, such as CPUs and GPUs, still struggles to efficiently execute these sub-8-bit quantized networks, as their SIMD instructions only support a granularity of 8 bits or wider.Also, there is a scarcity of literature that presents a full quantization paradigm for ViTs.In this paper, we propose an activation-aware fully sub-8-bit quantization-aware training (QAT) framework called PackQViT for efficient yet accurate ViT acceleration on mobile devices to facilitate real-time AI-powered decision-making.Specifically, in revisiting data activation within the ViT dataflow, two characteristics are relevant to quantization strategy and precision: the long-tailed distribution and systematic channel-wise outliers.In response, we employ either log2 quantization or clipping to address the long-tailed distribution and incorporate outlier-aware training for residual link quantization to regulate the various channel-wise outliers more consistently.Notably, due to the systematic fixed pattern, outlier-aware training approach can predict the channel indices and regularized scales of outliers in advance, thus avoiding the runtime data-adaptive selection during inference.Furthermore, we employ Int-$2^{n}$-Softmax, Int-LayerNorm, and Integer GELU to enable integer-only computation flow. Finally, we develop a SIMD-based 4-bit packed multiplier to achieve end-to-end ViT acceleration on mobile phones.Compared to prior studies on ViT quantization using 8-bit precision, PackQViT surpasses other works by an improved accuracy ranging from 0.4\% to 17.9\% for various widely used ViTs on ImageNet dataset; under 4-bit precision, PackQViT demonstrates 0.4%$\sim$2.8% higher accuracy. Compared to the baseline multiplier, our implementations on the Realme GT Android smartphone with Snapdragon 870 SoC CPU achieve 2.6x$\sim$3.7x speedup under 8-bit scenario and 3.8x$\sim$5.9x speedup under 4-bit which ensures practical real-time performance.

----

## [0] Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman

**Authors**: *Jiarui Feng, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cac8326ce3fbe79171db9754211530c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cac8326ce3fbe79171db9754211530c-Abstract-Conference.html)

**Abstract**:

Message passing neural networks (MPNNs) have emerged as the most popular framework of graph neural networks (GNNs) in recent years. However, their expressive power is limited by the 1-dimensional Weisfeiler-Lehman (1-WL) test. Some works are inspired by $k$-WL/FWL (Folklore WL) and design the corresponding neural versions. Despite the high expressive power, there are serious limitations in this line of research. In particular, (1) $k$-WL/FWL requires at least $O(n^k)$ space complexity, which is impractical for large graphs even when $k=3$; (2) The design space of $k$-WL/FWL is rigid, with the only adjustable hyper-parameter being $k$. To tackle the first limitation, we propose an extension, $(k, t)$-FWL. We theoretically prove that even if we fix the space complexity to $O(n^k)$ (for any $k \geq 2$) in $(k, t)$-FWL, we can construct an expressiveness hierarchy up to solving the graph isomorphism problem. To tackle the second problem, we propose $k$-FWL+, which considers any equivariant set as neighbors instead of all nodes, thereby greatly expanding the design space of $k$-FWL. Combining these two modifications results in a flexible and powerful framework $(k, t)$-FWL+. We demonstrate $(k, t)$-FWL+ can implement most existing models with matching expressiveness. We then introduce an instance of $(k,t)$-FWL+ called Neighborhood$^2$-FWL (N$^2$-FWL), which is practically and theoretically sound. We prove that N$^2$-FWL is no less powerful than 3-WL, and can encode many substructures while only requiring $O(n^2)$ space. Finally, we design its neural version named **N$^2$-GNN** and evaluate its performance on various tasks. N$^2$-GNN achieves record-breaking results on ZINC-Subset (**0.059**), outperforming previous SOTA results by 10.6\%. Moreover, N$^2$-GNN achieves new SOTA results on the BREC dataset (**71.8\%**) among all existing high-expressive GNN methods.

----

## [0] Off-Policy Evaluation for Human Feedback

**Authors**: *Qitong Gao, Ge Gao, Juncheng Dong, Vahid Tarokh, Min Chi, Miroslav Pajic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cb57fcf7ff3f6d37eebae5becc9ea6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cb57fcf7ff3f6d37eebae5becc9ea6d-Abstract-Conference.html)

**Abstract**:

Off-policy evaluation (OPE) is important for closing the gap between offline training and evaluation of reinforcement learning (RL), by estimating performance and/or rank of target (evaluation) policies using offline trajectories only. It can improve the safety and efficiency of data collection and policy testing procedures in situations where online deployments are expensive, such as healthcare. However, existing OPE methods fall short in estimating human feedback (HF) signals, as HF may be conditioned over multiple underlying factors and are only sparsely available; as opposed to the agent-defined environmental rewards (used in policy optimization), which are usually determined over parametric functions or distributions. Consequently, the nature of HF signals makes extrapolating accurate OPE estimations to be challenging. To resolve this, we introduce an OPE for HF (OPEHF) framework that revives existing OPE methods in order to accurately evaluate the HF signals. Specifically, we develop an immediate human reward (IHR) reconstruction approach, regularized by environmental knowledge distilled in a latent space that captures the underlying dynamics of state transitions as well as issuing HF signals. Our approach has been tested over two real-world experiments, adaptive in-vivo neurostimulation and intelligent tutoring, and a simulation environment (visual Q&A). Results show that our approach significantly improves the performance toward estimating HF signals accurately, compared to directly applying (variants of) existing OPE methods.

----

## [0] Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion

**Authors**: *Yash Bhalgat, Iro Laina, João F. Henriques, Andrea Vedaldi, Andrew Zisserman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html)

**Abstract**:

Instance segmentation in 3D is a challenging task due to the lack of large-scale annotated datasets. In this paper, we show that this task can be addressed effectively by leveraging instead 2D pre-trained models for instance segmentation. We propose a novel approach to lift 2D segments to 3D and fuse them by means of a neural field representation, which encourages multi-view consistency across frames. The core of our approach is a slow-fast clustering objective function, which is scalable and well-suited for scenes with a large number of objects. Unlike previous approaches, our method does not require an upper bound on the number of objects or object tracking across frames. To demonstrate the scalability of the slow-fast clustering, we create a new semi-realistic dataset called the Messy Rooms dataset, which features scenes with up to 500 objects per scene. Our approach outperforms the state-of-the-art on challenging scenes from the ScanNet, Hypersim, and Replica datasets, as well as on our newly created Messy Rooms dataset, demonstrating the effectiveness and scalability of our slow-fast clustering method.

----

## [0] GALOPA: Graph Transport Learning with Optimal Plan Alignment

**Authors**: *Yejiang Wang, Yuhai Zhao, Daniel Zhengkui Wang, Ling Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d35af80e775e342f4cd3792e4405837-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d35af80e775e342f4cd3792e4405837-Abstract-Conference.html)

**Abstract**:

Self-supervised learning on graph aims to learn graph representations in an unsupervised manner. While graph contrastive learning (GCL - relying on graph augmentation for creating perturbation views of anchor graphs and maximizing/minimizing similarity for positive/negative pairs) is a popular self-supervised method, it faces challenges in finding label-invariant augmented graphs and determining the exact extent of similarity between sample pairs to be achieved. In this work, we propose an alternative self-supervised solution that (i) goes beyond the label invariance assumption without distinguishing between positive/negative samples, (ii) can calibrate the encoder for preserving not only the structural information inside the graph, but the matching information between different graphs, (iii) learns isometric embeddings that preserve the distance between graphs, a by-product of our objective. Motivated by optimal transport theory, this scheme relays on an observation that the optimal transport plans between node representations at the output space, which measure the matching probability between two distributions, should be consistent to the plans between the corresponding graphs at the input space. The experimental findings include: (i) The plan alignment strategy significantly outperforms the counterpart using the transport distance; (ii) The proposed model shows superior performance using only node attributes as calibration signals, without relying on edge information; (iii) Our model maintains robust results even under high perturbation rates; (iv) Extensive experiments on various benchmarks validate the effectiveness of the proposed method.

----



[Go to the previous page](NIPS-2023-list300.md)

[Go to the next page](NIPS-2023-list302.md)

[Go to the catalog section](README.md)